from collections import defaultdict
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset
from tqdm import tqdm

from src.paths.paths import get_project_paths
from src.utils.dataset_utils import extract_xy_from_tile_stem
from src.core.registry_utils import register_dataset


CAMELYON_MULTICLASS_STAGE_MAP = {
    "negative": 0,
    "itc": 1,
    "micro": 2,
    "macro": 3,
}

@register_dataset(('camelyon', 'image', 'none', False))
class CamelyonHistologyDataset(Dataset):
    """CAMELYON image dataset with split-compatible API to HistologyDataset."""

    def __init__(
        self,
        split: str,
        transform=None,
        task: str = "lymph",
        he_only: bool = True,
        scanner1_only: bool = False,
        paths=None,
        relevance_mode: str = "diversity",
        label_mode: str = "binary",
        split_seed: int = 42,
        split_phase: Optional[str] = None,
        normal_tumor_max_ratio: Optional[float] = 1.0,
    ):
        """Loads and prepares the CAMELYON tile-level dataset for the given split."""
        self.split = split
        self.transform = transform
        self.task = task
        self.relevance_mode = relevance_mode
        self.label_mode = str(label_mode).strip().lower()
        self.split_seed = int(split_seed)
        self.split_phase = str(split_phase).strip().lower() if split_phase else None
        self.normal_tumor_max_ratio = normal_tumor_max_ratio
        self.split_target_col = "tumor_status" if self.label_mode == "binary" else "stage"

        if self.label_mode not in {"binary", "multiclass"}:
            raise ValueError("label_mode must be one of: binary, multiclass")

        self.class_to_index: Dict[str, int] = {}
        self.index_to_class: Dict[int, str] = {}
        if self.label_mode == "multiclass":
            self.class_to_index = dict(CAMELYON_MULTICLASS_STAGE_MAP)
            self.index_to_class = {v: k for k, v in self.class_to_index.items()}

        if paths is None:
            paths = get_project_paths()

        self.tile_path = paths["PROCESSED_DIR"] / "camelyon"
        self.base_path = paths["BASE_DIR"]
        self.splits_root = Path(paths["SPLITS_DIR"])
        self.embedding_root = paths["EMBEDDING_DIR"]

        if he_only:
            logging.info("[CAMELYON] he_only=True is ignored (CAMELYON has no stain split).")
        if scanner1_only:
            logging.info("[CAMELYON] scanner1_only=True is ignored unless scanner is encoded in split generation.")
        if task not in {"inflammation", "lymph"}:
            logging.warning("[CAMELYON] task='%s' requested. CAMELYON loader currently maps stage labels for lymph/inflammation semantics.", task)

        split_path = self._resolve_split_path(split)
        logging.info("[CAMELYON] Loading split file: %s", split_path)

        self.split_df = pd.read_csv(split_path)
        if "annotation_used" in self.split_df.columns:
            self.split_df['annotation_used'] = self.split_df['annotation_used'].map({'no': False, 'yes': True})
        self._validate_split_df()

        self.samples = self._load_samples()
        self._compute_relevance_scores()
        self._log_statistics()

    def _resolve_split_path(self, split: str) -> Path:
        seed_dir = f"seed_{self.split_seed}"
        phase_dir = self.split_phase

        if split.endswith(".csv"):
            split_file = split
        else:
            split_file = f"{split}.csv"

        candidates = []
        if phase_dir:
            candidates.extend([
                self.splits_root / "camelyon" / seed_dir / phase_dir / split_file,
                self.splits_root / seed_dir / phase_dir / split_file,
            ])
        candidates.extend([
            self.splits_root / "camelyon" / seed_dir / split_file,
            self.splits_root / seed_dir / split_file,
            self.splits_root / split_file,
        ])

        for candidate in candidates:
            if candidate.exists():
                return candidate

        nested = list(self.splits_root.rglob(split_file))
        if nested:
            return nested[0]

        raise FileNotFoundError(
            f"Split file not found for '{split_file}'. Checked: {[str(c) for c in candidates]}"
        )

    def _validate_split_df(self):
        required = {"slide_name"}
        missing = required - set(self.split_df.columns)
        if missing:
            raise ValueError(f"CAMELYON split file missing columns: {sorted(missing)}")

        if self.label_mode == "binary":
            if "tumor_status" not in self.split_df.columns:
                raise ValueError(
                    "label_mode='binary' requires the 'tumor_status' column in the split CSV. "
                    "Other label columns (stage, inflammation_status) are not used for binary classification."
                )
        elif self.label_mode == "multiclass":
            if "stage" not in self.split_df.columns:
                raise ValueError(
                    "label_mode='multiclass' requires the 'stage' column in the split CSV. "
                    "Other label columns (tumor_status, inflammation_status) are not used for multiclass classification."
                )

    def _encode_label(self, value: str) -> int:
        if self.label_mode == "binary":
            norm = str(value).strip().lower()
            if norm == "normal":
                return 0
            if norm == "tumor":
                return 1
            raise ValueError(
                f"Unknown tumor_status value for binary mode: '{value}'. "
                "Expected 'normal' or 'tumor'."
            )

        stage_norm = self._normalize_stage_label(value)
        if stage_norm not in self.class_to_index:
            raise ValueError(
                "Unknown CAMELYON stage for multiclass mode: "
                f"'{value}'. Supported: {sorted(self.class_to_index.keys())}"
            )
        return self.class_to_index[stage_norm]

    @staticmethod
    def _normalize_stage_label(stage: str) -> str:
        s = str(stage).strip().lower()
        if "negative" in s or s in {"pn0", "n0", "0"}:
            return "negative"
        if "itc" in s:
            return "itc"
        if "micro" in s:
            return "micro"
        if "macro" in s:
            return "macro"
        return s

    def _find_tiles_for_slide(self, directory_name, slide_name: str) -> List[Path]:
        def _belongs_to_slide(tile_stem: str, target_slide: str) -> bool:
            if tile_stem == target_slide:
                return True
            if not tile_stem.startswith(target_slide):
                return False
            return tile_stem[len(target_slide)] in {"_", "-", "."}

        pattern = f"{directory_name}/{slide_name}*.png"
        files = [p for p in self.tile_path.glob(pattern) if _belongs_to_slide(p.stem, slide_name)]
        if files:
            return sorted(files)

        nested_files = [p for p in self.tile_path.rglob(pattern) if _belongs_to_slide(p.stem, slide_name)]
        return sorted(nested_files)

    @staticmethod
    def _annotation_used_for_row(row: pd.Series) -> bool:
        raw_value = row.get("annotation_used", False)
        if pd.isna(raw_value):
            return False
        if isinstance(raw_value, str):
            return raw_value.strip().lower() == "yes"
        return bool(raw_value)

    def _load_samples(self) -> List[Dict]:
        samples: List[Dict] = []

        rows = self.split_df.drop_duplicates(subset=["slide_name"]).copy()

        for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"Loading CAMELYON {self.split} dataset"):
            slide_name = str(row["slide_name"])
            if self.label_mode == "binary":
                label_value = str(row["tumor_status"])
            else:
                label_value = str(row["stage"])
            label = self._encode_label(label_value)

            if self.label_mode == "binary":
                normal_tile_label = 0
                tumor_tile_label = 1
            else:
                normal_tile_label = CAMELYON_MULTICLASS_STAGE_MAP["negative"]
                tumor_tile_label = label

            matching_files_normal = self._find_tiles_for_slide("normal", slide_name)
            matching_files_tumor = self._find_tiles_for_slide("tumor", slide_name)

            annotation_used = self._annotation_used_for_row(row)
            if (
                self.split == "train"
                and self.normal_tumor_max_ratio is not None
                and annotation_used
                and matching_files_tumor
                and len(matching_files_normal) > int(self.normal_tumor_max_ratio * len(matching_files_tumor))
            ):
                cap = int(self.normal_tumor_max_ratio * len(matching_files_tumor))
                rng = random.Random(self.split_seed)
                matching_files_normal = rng.sample(matching_files_normal, cap)
                matching_files_normal = sorted(matching_files_normal)
                logging.debug(
                    "[CAMELYON] Slide %s: capped normal tiles to %d (ratio %.1f × %d tumor tiles)",
                    slide_name, cap, self.normal_tumor_max_ratio, len(matching_files_tumor),
                )

            if not matching_files_normal and not matching_files_tumor:
                logging.warning("[CAMELYON] No tile files found for slide %s", slide_name)
                continue

            for tile_path in matching_files_normal:
                if not self._is_valid_tile(tile_path):
                    continue

                try:
                    x, y = extract_xy_from_tile_stem(tile_path.stem)
                except Exception:
                    x, y = 0, 0

                samples.append(
                    {
                        "path": tile_path,
                        "label": normal_tile_label,
                        "slide_name": slide_name,
                        "metadata": {
                            "task": self.task,
                            "split": self.split,
                            "label_value": label_value,
                            "particle_id": 0,
                            "x": x,
                            "y": y,
                        },
                    }
                )

            for tile_path in matching_files_tumor:
                if not self._is_valid_tile(tile_path):
                    continue

                try:
                    x, y = extract_xy_from_tile_stem(tile_path.stem)
                except Exception:
                    x, y = 0, 0

                samples.append(
                    {
                        "path": tile_path,
                        "label": tumor_tile_label,
                        "slide_name": slide_name,
                        "metadata": {
                            "task": self.task,
                            "split": self.split,
                            "label_value": label_value,
                            "particle_id": 0,
                            "x": x,
                            "y": y,
                        },
                    }
                )

        if len(samples) == 0:
            logging.warning("[CAMELYON] No valid samples found for split '%s'", self.split)
            logging.warning("Please check split CSV and tile extraction output path")
        else:
            unique_slides = len({s["slide_name"] for s in samples})
            logging.info("[CAMELYON] Loaded %d tiles across %d slides", len(samples), unique_slides)

        return samples

    def _is_valid_tile(self, image_path: Path) -> bool:
        try:
            with Image.open(image_path) as image:
                img_array = np.array(image.convert("L"))
                white_pixels = np.sum(img_array > 240) / img_array.size
                return white_pixels < 0.5
        except Exception as e:
            logging.warning("Error validating tile %s: %s", image_path, e)
            return False

    def _log_statistics(self):
        total = len(self.samples)
        if total == 0:
            logging.warning("No samples found for %s task in %s split", self.task, self.split)
            return

        labels = [sample["label"] for sample in self.samples]
        unique_wsis = len({sample["slide_name"] for sample in self.samples})

        logging.info("\n%s CAMELYON Dataset Statistics (%s task):", self.split, self.task)
        logging.info("Total samples: %d", total)
        logging.info("Unique Slides: %d", unique_wsis)
        if self.label_mode == "binary":
            pos_count = int(sum(labels))
            neg_count = total - pos_count
            logging.info("Positive: %d (%.1f%%)", pos_count, pos_count / total * 100)
            logging.info("Negative: %d (%.1f%%)", neg_count, neg_count / total * 100)
        else:
            label_dist = pd.Series(labels).value_counts().sort_index()
            logging.info("Multiclass label distribution:")
            for class_idx, count in label_dist.items():
                class_name = self.index_to_class.get(int(class_idx), str(class_idx))
                logging.info("class %s (%s): %d (%.1f%%)", class_idx, class_name, count, count / total * 100)

        label_val_dist = pd.Series([s["metadata"]["label_value"] for s in self.samples]).value_counts()
        logging.info("\nLabel value distribution:")
        for val, count in label_val_dist.items():
            logging.info("%s: %d (%.1f%%)", val, count, count / total * 100)

    def __len__(self) -> int:
        """Returns the number of tile samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Loads and returns the image, label, and metadata for a given sample index."""
        sample = self.samples[idx]
        try:
            with Image.open(sample["path"]) as image:
                image = image.convert("RGB")
                if self.transform:
                    image = self.transform(image)
                else:
                    image = transforms.ToTensor()(image)

            metadata = sample["metadata"].copy()
            metadata["slide_name"] = sample["slide_name"]
            label_dtype = torch.long if self.label_mode == "multiclass" else torch.float
            return image, torch.tensor(sample["label"], dtype=label_dtype), metadata
        except Exception as e:
            logging.error("Error loading image %s: %s", sample["path"], e)
            metadata = sample["metadata"].copy()
            metadata["slide_name"] = sample["slide_name"]
            metadata["corrupted"] = True
            label_dtype = torch.long if self.label_mode == "multiclass" else torch.float
            return torch.zeros((3, 224, 224)), torch.tensor(sample["label"], dtype=label_dtype), metadata

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
            density=True,
        )

        mean_grad = magnitude.mean()
        std_grad = magnitude.std()

        signature = np.concatenate([hist_ori, np.array([mean_grad, std_grad])])
        return signature.astype(np.float32)

    def _compute_relevance_scores(self):
        logging.info("Computing relevance scores (mode: %s)...", self.relevance_mode)

        if self.relevance_mode == "uniform":
            for sample in self.samples:
                sample["metadata"]["relevance"] = 0.5
            return

        slide_groups = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            slide_groups[sample["slide_name"]].append(idx)

        if self.relevance_mode == "diversity":
            self._compute_diversity_relevance(slide_groups)
        elif self.relevance_mode == "quality":
            self._compute_quality_relevance(slide_groups)
        else:
            raise ValueError(f"Unknown relevance_mode: {self.relevance_mode}")

    def _compute_diversity_relevance(self, slide_groups):
        signatures = {}
        for idx, sample in enumerate(self.samples):
            signatures[idx] = self._compute_tile_signature(sample["path"])

        for _, indices in slide_groups.items():
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

    def _compute_quality_relevance(self, slide_groups):
        for _, indices in slide_groups.items():
            for idx in indices:
                img = cv2.imread(str(self.samples[idx]["path"]))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                tissue_score = 1.0 - (img_gray > 200).mean()
                variance_score = np.std(img_gray) / 128.0
                variance_score = min(variance_score, 1.0)
                edges = cv2.Canny(img_gray, 50, 150)
                edge_score = edges.mean() / 255.0

                relevance = 0.5 * tissue_score + 0.3 * variance_score + 0.2 * edge_score
                relevance = np.clip(relevance, 0.0, 1.0)
                self.samples[idx]["metadata"]["relevance"] = float(relevance)