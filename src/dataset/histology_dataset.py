from collections import defaultdict
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import logging
import argparse
from scipy.spatial.distance import cdist
from typing import Dict, List, Optional, Tuple

import sys

from src.utils.dataset_utils import extract_xy_from_tile_stem
from src.core.registry_utils import register_dataset
from src.paths.paths import get_project_paths
from src.utils.dataset_utils import group_indices_by_slide

@register_dataset(('gsdb', 'image', 'none', False))
class HistologyDataset(Dataset):
    """Histology Image Dataset for tissue and inflammation classification.

    This Dataset loads histological images from pre-processed tiles and supports
    both inflammation detection and tissue type classification tasks. The dataset
    provides functionality to work with various splits (train/val/test) and
    can handle multi-scanner setups with domain adaptation requirements.

    Features:
        - Supports both inflammation and tissue type classification
        - Handles different stains (HE, PAS, MG) with option to filter
        - Provides detailed metadata with each sample for hierarchical analysis
        - Validates tissue content of tiles for quality assurance
    """

    def __init__(self, split: str, transform=None, task: str = 'inflammation',
                 he_only: bool = True, scanner1_only: bool = False, paths=None,
                 relevance_mode: str = 'diversity', label_mode: str = 'binary',
                 split_seed: int = 42, split_phase: Optional[str] = None,
                 max_white_fraction: float = 0.5):
        """Loads and prepares the tile-level histology dataset for the given split."""
        self.split = split
        self.max_white_fraction = float(max_white_fraction)
        self.transform = transform
        self.task = task
        self.relevance_mode = relevance_mode
        self.label_mode = label_mode
        self.split_seed = int(split_seed)
        self.split_phase = str(split_phase).strip().lower() if split_phase else None
        self.split_target_col = 'inflammation_status' if self.task == 'inflammation' else 'tissue_type'


        if paths is None:
            paths = get_project_paths()

        self.base_path = paths["BASE_DIR"]
        self.tile_path = paths["PROCESSED_DIR"] / 'tiles'

        logging.info(f"Dataset base path: {self.base_path}")
        logging.info(f"Tile path: {self.tile_path}")

        split_path = self._resolve_split_path(paths, split, he_only, self.split_phase)

        logging.info(f"Loading split file: {split_path}")
        self.split_df = pd.read_csv(split_path)


        if task == 'inflammation':
            metadata_path = paths["METRICS_DIR"] / 'inflammation_status.csv'
            self.metadata_df = pd.read_csv(metadata_path)
        else:
            metadata_path = paths["METRICS_DIR"] / 'tissue_types.csv'
            self.metadata_df = pd.read_csv(metadata_path)


        self.tiles_df = pd.read_csv(paths["METRICS_DIR"] / 'tiles_per_particle.csv')


        if he_only:
            self.metadata_df = self.metadata_df[self.metadata_df['stain'] == 'HE']
            self.tiles_df = self.tiles_df[self.tiles_df['stain'] == 'HE']

        if scanner1_only:
            self.metadata_df = self.metadata_df[self.metadata_df['scanner_id'] == 1]
            self.tiles_df = self.tiles_df[self.tiles_df['scanner_id'] == 1]


        split_slides = set(self.split_df['slide_name'])
        self.metadata_df = self.metadata_df[self.metadata_df['slide_name'].isin(split_slides)]
        self.tiles_df = self.tiles_df[self.tiles_df['slide_name'].isin(split_slides)]


        self.samples = self._load_samples()

        self._compute_relevance_scores()


        self._log_statistics()

    def _resolve_split_path(self, paths, split: str, he_only: bool, split_phase: Optional[str]) -> Path:
        splits_root = Path(paths["SPLITS_DIR"])
        seed_dir = f"seed_{self.split_seed}"
        phase_dir = split_phase

        if split.endswith('.csv'):
            split_file = split
        else:
            split_file = f"{split}_HE.csv" if he_only else f"{split}_all_stains.csv"

        candidates = []
        if phase_dir:
            candidates.extend([
                splits_root / "gsdb" / seed_dir / phase_dir / split_file,
                splits_root / seed_dir / phase_dir / split_file,
            ])
        candidates.extend([
            splits_root / "gsdb" / seed_dir / split_file,
            splits_root / seed_dir / split_file,
            splits_root / split_file,
        ])

        for candidate in candidates:
            if candidate.exists():
                return candidate

        nested = list(splits_root.rglob(split_file))
        if nested:
            return nested[0]

        raise FileNotFoundError(
            f"Split file not found for '{split_file}'. Checked: {[str(c) for c in candidates]}"
        )

    def _extract_particle_id(self, filepath: str) -> str:
        try:
            parts = str(filepath).split('particle_')
            if len(parts) > 1:

                particle_id = parts[1].split('_')[0]
                return particle_id
        except Exception as e:
            logging.error(f"Error extracting particle_id from {filepath}: {e}")
        return None

    def _load_samples(self):
        samples = []

        if self.task == 'inflammation':

            valid_slides = self.metadata_df[self.metadata_df['inflammation_status'].isin(['inflamed', 'noninflamed'])]
            logging.info(f"Found {len(valid_slides)} valid slides for inflammation evaluation")


            for _, slide_row in tqdm(valid_slides.iterrows(), desc=f"Loading {self.task} {self.split} dataset"):
                slide_name = slide_row['slide_name']
                inflammation_status = slide_row['inflammation_status']


                slide_particles = self.tiles_df[self.tiles_df['slide_name'] == slide_name]

                for _, particle_row in slide_particles.iterrows():
                    if particle_row['tiles_per_particle'] > 0:

                        tile_pattern = f"{slide_name}*particle_{particle_row['particle_id']}*.png"
                        matching_files = list(self.tile_path.glob(tile_pattern))

                        for tile_path in matching_files:
                            if self._is_valid_tile(tile_path):
                                label = 1 if inflammation_status == 'inflamed' else 0
                                x,y = extract_xy_from_tile_stem(tile_path.stem)
                                samples.append({
                                    'path': tile_path,
                                    'label': label,
                                    'slide_name': slide_name,
                                    'metadata': {
                                        'task': self.task,
                                        'split': self.split,
                                        'inflammation_status': inflammation_status,
                                        'particle_id': particle_row['particle_id'],
                                        'x': x,
                                        'y': y,
                                    }
                                })


        else:

            valid_particles = self.metadata_df[self.metadata_df['tissue_type'].isin(['corpus', 'antrum'])]

            for _, row in tqdm(valid_particles.iterrows(), desc=f"Loading {self.task} {self.split} dataset"):

                particle_tiles = self.tiles_df[
                    (self.tiles_df['slide_name'] == row['slide_name']) &
                    (self.tiles_df['particle_id'] == row['particle_id'])
                ]

                if particle_tiles.empty:
                    logging.warning(f"No tile information found for particle {row['particle_id']} in slide {row['slide_name']}")
                    continue


                tile_pattern = f"{row['slide_name']}*particle_{row['particle_id']}*.png"
                matching_files = list(self.tile_path.glob(tile_pattern))

                if not matching_files:
                    logging.warning(f"No tile files found for particle {row['particle_id']} in slide {row['slide_name']}")
                    continue


                for tile_path in matching_files:
                    if self._is_valid_tile(tile_path):
                        label = 1 if row['tissue_type'] == 'corpus' else 0
                        samples.append({
                            'path': tile_path,
                            'label': label,
                            'slide_name': row['slide_name'],
                            'metadata': {
                                'task': self.task,
                                'split': self.split,
                                'tissue_type': row['tissue_type'],
                                'particle_id': row['particle_id'],
                                'cluster_id': row['cluster_id'],
                            }
                        })

        if len(samples) == 0:
            logging.warning(f"No valid samples found for {self.task} task in {self.split} split!")
            logging.warning("Please check:")
            logging.warning("1. Tile files exist in the expected location")
            logging.warning("2. Metadata has matching entries for the split")
            logging.warning("3. Particles have the expected tissue types")
            logging.warning("4. Tiles can be matched to particles")

        else:
            unique_slides = len(set(s['slide_name'] for s in samples))
            unique_particles = len(set(s['metadata']['particle_id'] for s in samples))
            logging.info(f"Loaded {len(samples)} tiles from {unique_particles} particles across {unique_slides} slides")

        return samples

    def _is_valid_tile(self, image_path: Path) -> bool:
        try:
            with Image.open(image_path) as image:
                img_array = np.array(image.convert("L"))
                white_pixels = np.sum(img_array > 240) / img_array.size
                return white_pixels < self.max_white_fraction
        except Exception as e:
            logging.warning(f"Error validating tile {image_path}: {e}")
            return False

    def _log_statistics(self):
        total = len(self.samples)
        if total == 0:
            logging.warning(f"No samples found for {self.task} task in {self.split} split!")
            return


        labels = [sample['label'] for sample in self.samples]
        pos_count = sum(labels)
        neg_count = total - pos_count
        unique_wsis = len({sample['slide_name'] for sample in self.samples})


        logging.info(f"\n{self.split} Dataset Statistics ({self.task} task):")
        logging.info(f"Total samples: {total}")
        logging.info(f"Unique Slides: {unique_wsis}")


        if self.task == 'inflammation':
            logging.info(f"Inflamed: {pos_count} ({pos_count/total*100:.1f}%)")
            logging.info(f"Non-inflamed: {neg_count} ({neg_count/total*100:.1f}%)")


            inf_status_ = pd.Series([s['metadata']['inflammation_status'] for s in self.samples]).value_counts()
            logging.info("\nInflammation status distribution:")
            for inf_status, count in inf_status_.items():
                logging.info(f"{inf_status}: {count} ({count/total*100:.1f}%)")
        else:
            logging.info(f"Corpus: {pos_count} ({pos_count/total*100:.1f}%)")
            logging.info(f"Antrum: {neg_count} ({neg_count/total*100:.1f}%)")


            tissue_types = pd.Series([s['metadata']['tissue_type'] for s in self.samples]).value_counts()
            logging.info("\nTissue type distribution:")
            for tissue_type, count in tissue_types.items():
                logging.info(f"{tissue_type}: {count} ({count/total*100:.1f}%)")

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        try:
            with Image.open(sample['path']) as image:
                image = image.convert('RGB')
                if self.transform:
                    image = self.transform(image)
                else:

                    image = transforms.ToTensor()(image)


                metadata = sample['metadata'].copy()
                metadata['slide_name'] = sample['slide_name']

                return image, torch.tensor(sample['label'], dtype=torch.float), metadata

        except Exception as e:
            logging.error(f"Error loading image {sample['path']}: {e}")

            logging.warning(f"Returning placeholder for corrupted image {sample['path']}")
            metadata = sample['metadata'].copy()
            metadata['slide_name'] = sample['slide_name']
            metadata['corrupted'] = True
            return torch.zeros((3, 224, 224)), torch.tensor(sample['label'], dtype=torch.float), metadata

    def _compute_tile_signature(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as img:
            gray = np.array(img.convert("L"))


        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx)


        hist_ori, _ = np.histogram(
            orientation,
            bins=8,
            range=(-np.pi, np.pi),
            weights=magnitude,
            density=True
        )


        mean_grad = magnitude.mean()
        std_grad = magnitude.std()

        signature = np.concatenate([
            hist_ori,
            np.array([mean_grad, std_grad])
        ])

        return signature.astype(np.float32)

    def _compute_relevance_scores(self):
        logging.info(f"Computing relevance scores (mode: {self.relevance_mode})...")

        if self.relevance_mode == 'uniform':

            for sample in self.samples:
                sample["metadata"]["relevance"] = 0.5
            return


        particle_groups = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            pid = sample["metadata"]["particle_id"]
            particle_groups[pid].append(idx)

        if self.relevance_mode == 'diversity':
            self._compute_diversity_relevance(particle_groups)
        elif self.relevance_mode == 'quality':
            self._compute_quality_relevance(particle_groups)
        else:
            raise ValueError(f"Unknown relevance_mode: {self.relevance_mode}")

    def _compute_diversity_relevance(self, particle_groups):
        signatures = {}
        for idx, sample in enumerate(self.samples):
            signatures[idx] = self._compute_tile_signature(sample["path"])

        for pid, indices in particle_groups.items():
            if len(indices) < 3:

                for idx in indices:
                    self.samples[idx]["metadata"]["relevance"] = 0.8
                continue

            S = np.stack([signatures[idx] for idx in indices])


            pairwise_dists = cdist(S, S, metric="euclidean")



            avg_dists = pairwise_dists.mean(axis=1)


            if avg_dists.max() > avg_dists.min():
                relevance = (avg_dists - avg_dists.min()) / (avg_dists.max() - avg_dists.min())
            else:
                relevance = np.ones_like(avg_dists) * 0.5


            relevance = 0.2 + 0.8 * relevance

            for idx, r in zip(indices, relevance):
                self.samples[idx]["metadata"]["relevance"] = float(r)

    def _compute_quality_relevance(self, particle_groups):
        for pid, indices in particle_groups.items():
            for idx in indices:
                img = cv2.imread(str(self.samples[idx]["path"]))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


                tissue_score = 1.0 - (img_gray > 200).mean()


                variance_score = np.std(img_gray) / 128.0
                variance_score = min(variance_score, 1.0)


                edges = cv2.Canny(img_gray, 50, 150)
                edge_score = edges.mean() / 255.0


                relevance = (
                    0.5 * tissue_score +
                    0.3 * variance_score +
                    0.2 * edge_score
                )


                relevance = np.clip(relevance, 0.0, 1.0)

                self.samples[idx]["metadata"]["relevance"] = float(relevance)



@register_dataset(('gsdb', 'embedding', 'pov-gigapath', False))
class EmbeddedHistologyDataset(HistologyDataset):
    def __init__(self, split='train', transform=None, task: str = 'inflammation',
                 he_only: bool = False, scanner1_only: bool = False, paths=None,
                 relevance_mode: str = 'diversity', label_mode: str = 'binary',
                 split_seed: int = 42, split_phase: Optional[str] = None):
        """Loads the histology dataset for the split and attaches precomputed tile embeddings."""
        super().__init__(split=split, transform=transform, task=task,
                         he_only=he_only, scanner1_only=scanner1_only, paths=paths,
                         relevance_mode=relevance_mode, label_mode=label_mode,
                         split_seed=split_seed, split_phase=split_phase)
        self.embedding_root = paths["EMBEDDING_DIR"]
        logging.info(f"Attaching embeddings from {self.embedding_root}")
        self._attach_embeddings()

    def __getitem__(self, idx):
        """Returns the precomputed embedding, label, and metadata for a given sample index."""
        sample = self.samples[idx]

        emb_path = sample.get("embedding_path", None)
        if emb_path is None:
            raise RuntimeError(f"No embedding for sample {sample['path']}")

        embedding = np.load(emb_path, allow_pickle=False)
        embedding = torch.from_numpy(embedding).float()

        label = torch.tensor(sample["label"], dtype=torch.float)

        metadata = sample["metadata"].copy()
        metadata["slide_name"] = sample["slide_name"]

        return embedding, label, metadata

    def _attach_embeddings(self):
        missing = 0
        for sample in self.samples:
            slide = sample["slide_name"]
            tile_stem = sample["path"].stem

            emb_path = (
                self.embedding_root
                / f"{tile_stem}_embedding.npy"
            )

            if not emb_path.exists():
                sample["embedding_path"] = None
                missing += 1
            else:
                sample["embedding_path"] = emb_path

        if missing > 0:
            logging.warning(f"{missing} samples without embeddings")


@register_dataset(('gsdb', 'multimodal', 'pov-gigapath', False))
class EmbeddedImageHistologyDataset(HistologyDataset):
    def __init__(self, split='train', transform=None, task: str = 'inflammation',
                 he_only: bool = True, scanner1_only: bool = False, paths=None,
                 label_mode: str = 'binary', split_seed: int = 42, split_phase: Optional[str] = None):
        """Loads the histology dataset for the split and attaches precomputed tile embeddings."""
        super().__init__(split=split, transform=transform, task=task,
                         he_only=he_only, scanner1_only=scanner1_only, paths=paths,
                         label_mode=label_mode, split_seed=split_seed, split_phase=split_phase)
        self.embedding_root = paths["EMBEDDING_DIR"]
        self._attach_embeddings()

    def __getitem__(self, idx):
        """Returns the image, precomputed embedding, label, and metadata for a given sample index."""
        sample = self.samples[idx]

        emb_path = sample.get("embedding_path", None)
        if emb_path is None:
            raise RuntimeError(f"No embedding for sample {sample['path']}")

        image = None
        try:
            with Image.open(sample['path']) as image:
                image = image.convert('RGB')
                if self.transform:
                    image = self.transform(image)
        except Exception as e:
            logging.error(f"Error loading image {sample['path']}: {e}")
            logging.warning(f"Returning placeholder for corrupted image {sample['path']}")
            image = torch.zeros((3, 224, 224))

        embedding = np.load(emb_path, allow_pickle=False)
        embedding = torch.from_numpy(embedding).float()

        label = torch.tensor(sample["label"], dtype=torch.float)

        metadata = sample["metadata"].copy()
        metadata["slide_name"] = sample["slide_name"]

        return image, embedding, label, metadata

    def _attach_embeddings(self):
        missing = 0
        for sample in self.samples:
            slide = sample["slide_name"]
            tile_stem = sample["path"].stem

            emb_path = (
                self.embedding_root
                / f"{tile_stem}_embedding.npy"
            )

            if not emb_path.exists():
                sample["embedding_path"] = None
                missing += 1
            else:
                sample["embedding_path"] = emb_path

        if missing > 0:
            logging.warning(f"{missing} samples without embeddings")

@register_dataset(('gsdb', 'multimodal', 'pov-gigapath', True))
class MILDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', transform=None, task: str = 'inflammation',
                 he_only: bool = True, scanner1_only: bool = False, paths=None,
                 label_mode: str = 'binary', split_seed: int = 42, split_phase: Optional[str] = None):
        """Builds a slide-level MIL dataset of image+embedding tile bags from EmbeddedImageHistologyDataset."""
        self.base_dataset = EmbeddedImageHistologyDataset(
            split=split,
            transform=transform,
            task=task,
            he_only=he_only,
            scanner1_only=scanner1_only,
            paths=paths,
            label_mode=label_mode,
            split_seed=split_seed,
            split_phase=split_phase,
        )
        self.base = self.base_dataset
        self.bags = group_indices_by_slide(self.base_dataset, task=task)
        self.max_tiles = 100
        self.use_embeddings = True

    def __len__(self):
        """Returns the number of slide-level bags in the dataset."""
        return len(self.bags)

    def __getitem__(self, idx):
        """Returns the stacked tile images (and embeddings) and bag label for a given slide index."""
        indices = self.bags[idx]

        if (self.max_tiles is not None) and (len(indices) > self.max_tiles):
            indices = np.random.choice(indices, self.max_tiles, replace=False)

        images = []
        embeddings = [] if self.use_embeddings else None
        bag_label = None

        for i in indices:
            sample = self.base[i]

            if self.use_embeddings:
                img, emb, label, meta = sample
                embeddings.append(emb)
            else:
                img, label, meta = sample

            images.append(img)

            if bag_label is None:
                bag_label = (
                    float(label.item())
                    if torch.is_tensor(label)
                    else float(label)
                )

        bag_images = torch.stack(images, 0)

        if self.use_embeddings:
            bag_embeddings = torch.stack(embeddings, 0)
            return bag_images, bag_embeddings, bag_label

        return bag_images, bag_label



@register_dataset(('gsdb', 'image', 'none', True))
class MILDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', transform=None, task: str = 'inflammation',
                 he_only: bool = True, scanner1_only: bool = False, paths=None,
                 label_mode: str = 'binary', split_seed: int = 42, split_phase: Optional[str] = None,
                 max_white_fraction: float = 0.5):
        """Builds a slide-level MIL dataset of image-only tile bags from HistologyDataset."""
        self.base_dataset = HistologyDataset(
            split=split,
            transform=transform,
            task=task,
            he_only=he_only,
            scanner1_only=scanner1_only,
            paths=paths,
            label_mode=label_mode,
            split_seed=split_seed,
            split_phase=split_phase,
            max_white_fraction=max_white_fraction,
        )
        self.base = self.base_dataset
        self.bags = group_indices_by_slide(self.base_dataset, task=task)
        self.max_tiles = 60
        self.use_embeddings = False
        self.split = split

    def __len__(self):
        """Returns the number of slide-level bags in the dataset."""
        return len(self.bags)

    def __getitem__(self, idx):
        """Returns the stacked tile images and bag label for a given slide index."""
        indices = self.bags[idx]

        if (self.max_tiles is not None) and (len(indices) > self.max_tiles):
            if self.split == 'train':
                # Keep the original behavior here: a fresh random subsample each epoch acts as a
                # mild augmentation (more of the slide gets seen across training, just never all
                # at once). Only eval's reproducibility was the actual problem -- see below.
                indices = np.random.choice(indices, self.max_tiles, replace=False)
            else:
                # Was np.random.choice on the unseeded global RNG for every split, train included:
                # two evaluation runs of the exact same checkpoint on the exact same slide could
                # pick a different random tile subset and get a different slide-level prediction,
                # purely from this sampling noise -- not from any model difference. Bag/slide idx
                # is a stable per-slide identity within a given split (see self.bags =
                # group_indices_by_slide(...) in __init__), so seeding a local RNG off it makes
                # the subsample -- and therefore eval metrics -- reproducible run to run, while
                # still varying which tiles are seen across different slides.
                indices = np.random.RandomState(idx).choice(indices, self.max_tiles, replace=False)

        images = []
        embeddings = [] if self.use_embeddings else None
        bag_label = None

        for i in indices:
            sample = self.base[i]

            if self.use_embeddings:
                img, emb, label, meta = sample
                embeddings.append(emb)
            else:
                img, label, meta = sample

            images.append(img)

            if bag_label is None:
                bag_label = (
                    float(label.item())
                    if torch.is_tensor(label)
                    else float(label)
                )

        bag_images = torch.stack(images, 0)

        if self.use_embeddings:
            bag_embeddings = torch.stack(embeddings, 0)
            return bag_images, bag_embeddings, bag_label

        return bag_images, bag_label
