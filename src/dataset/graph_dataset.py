from pathlib import Path
import sys
import zlib
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, get_worker_info
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph




project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.paths.paths import get_project_paths
from src.core.registry_utils import register_dataset
from src.dataset.histology_dataset import EmbeddedHistologyDataset
from src.dataset.region_resplit import resplit_large_regions, merge_small_regions
from src.dataset.graph_augment import apply_graph_augmentations
from src.utils.dataset_utils import extract_xy_from_tile_stem
from src.utils.logging import logging


@register_dataset(('gsdb', 'graph', 'pov-gigapath', True))
class WSIGraphDataset(Dataset):
    """
    Graph-based WSI Dataset. Crafted from normal tile structure of histology dataset with related feature embeddings (from tiles)

    One sample = one WSI = one graph
    Nodes      = tiles
    Node feats = tile embeddings
    Edges      = kNN or Radius Neighbors in physical (x,y) space
    Label      = slide-level label
    """

    def __init__(
        self,
        split='train',
        transform=None,
        task: str = 'inflammation',
        dataset: str = 'gsdb',
        split_seed: int = 42,
        split_phase: Optional[str] = None,
        he_only: bool = False,
        scanner1_only:bool = False,
        paths = None,
        k: int = 8,
        radius: float = None,
        normalize_pos: bool = True,
        use_relevance_filter: bool = False,
        relevance_threshold: float = 0.4,
        max_nodes_per_graph: int = None,
        add_relevance_feature: bool = True,
        edge_weighting: str = 'relevance',
        distance_metric: str = 'gaussian',
        embedding_type: str = 'pov-gigapath',
        pipeline_type: str = None,
        label_mode: str = 'binary',
        cache_in_memory: bool = False,
        resplit_large_regions: bool = False,
        resplit_target_patches_per_region: int = 250,
        resplit_k_region: int = 5,
    ):
        """Builds a per-slide graph dataset from an underlying embedded tile dataset."""
        self.base_dataset = EmbeddedHistologyDataset(
            split=split,
            transform=transform,
            task=task,
            he_only=he_only,
            scanner1_only=scanner1_only,
            paths=paths,
            relevance_mode='diversity',
            split_phase=split_phase,
        )
        self.base = self.base_dataset
        self.k = k
        self.radius = radius
        self.normalize_pos = normalize_pos

        self.use_relevance_filter = use_relevance_filter
        self.relevance_threshold = relevance_threshold
        self.max_nodes_per_graph = max_nodes_per_graph
        self.add_relevance_feature = add_relevance_feature
        self.edge_weighting = edge_weighting
        self.distance_metric = distance_metric
        self.label_mode = str(label_mode).strip().lower()

        self.cache_in_memory = bool(cache_in_memory)
        self._graph_cache: Dict[int, "Data"] = {}

        self.slide_groups = self._group_by_slide()

    def __len__(self):
        """Returns the number of slide-level graphs in the dataset."""
        return len(self.slide_groups)

    def __getitem__(self, idx):
        """Builds and returns the graph for the slide at the given index."""
        if self.cache_in_memory and idx in self._graph_cache:
            return self._graph_cache[idx]

        indices = self.slide_groups[idx]

        node_features = []
        positions = []
        relevance_scores = []
        label = None

        for i in indices:
            emb, y, meta = self.base[i]


            if self.use_relevance_filter:
                if 'relevance' in meta and meta["relevance"] < self.relevance_threshold:
                    continue

            node_features.append(emb)
            positions.append([meta["x"], meta["y"]])
            relevance_scores.append(meta.get("relevance", 0.5))

            if label is None:
                label = y

        if len(node_features) == 0:
            raise ValueError(f"No tiles found for slide index {idx}")


        if self.max_nodes_per_graph is not None and len(node_features) > self.max_nodes_per_graph:

            sorted_indices = np.argsort(relevance_scores)[::-1][:self.max_nodes_per_graph]
            node_features = [node_features[i] for i in sorted_indices]
            positions = [positions[i] for i in sorted_indices]
            relevance_scores = [relevance_scores[i] for i in sorted_indices]

        x = torch.stack(node_features)
        pos = torch.tensor(positions, dtype=torch.float)
        rel = torch.tensor(relevance_scores, dtype=torch.float).unsqueeze(1)


        if self.add_relevance_feature:
            x = torch.cat([x, rel], dim=1)

        if self.normalize_pos:
            pos = self._normalize_positions(pos)

        if self.radius is not None:
            edge_index = self._build_radius_graph(pos)
        else:
            edge_index = self._build_knn_graph(pos)


        edge_attr = None
        if self.edge_weighting is not None and edge_index.size(1) > 0:
            edge_attr = self._compute_edge_weights(edge_index, pos, x, rel)

        data = Data(
            x=x,
            edge_index=edge_index,
            pos=pos,
            y=torch.tensor([label], dtype=torch.float),
            relevance=rel,
            edge_attr=edge_attr
        )
        if self.cache_in_memory:
            self._graph_cache[idx] = data
        return data





    def _group_by_slide(self):
        slide_to_indices = {}

        for idx in range(len(self.base)):
            _, _, meta = self.base[idx]
            slide = meta["slide_name"]

            if slide not in slide_to_indices:
                slide_to_indices[slide] = []
            slide_to_indices[slide].append(idx)

        return list(slide_to_indices.values())

    def _normalize_positions(self, pos: torch.Tensor) -> torch.Tensor:
        pos = pos - pos.mean(dim=0, keepdim=True)
        pos = pos / (pos.std(dim=0, keepdim=True) + 1e-6)
        return pos

    def _compute_edge_weights(self, edge_index: torch.Tensor, pos: torch.Tensor,
                             x: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]

        if self.edge_weighting == 'distance':

            pos_src = pos[src]
            pos_dst = pos[dst]
            distances = torch.norm(pos_src - pos_dst, dim=1)

            if self.distance_metric == 'gaussian':

                sigma = distances.std() + 1e-6
                weights = torch.exp(-(distances ** 2) / (2 * sigma ** 2))
            else:

                weights = 1.0 / (1.0 + distances)

        elif self.edge_weighting == 'relevance':

            rel_src = rel[src].squeeze()
            rel_dst = rel[dst].squeeze()

            weights = torch.sqrt(rel_src * rel_dst)

        elif self.edge_weighting == 'similarity':


            if self.add_relevance_feature:
                x_clean = x[:, :-1]
            else:
                x_clean = x

            x_src = x_clean[src]
            x_dst = x_clean[dst]


            x_src_norm = torch.nn.functional.normalize(x_src, p=2, dim=1)
            x_dst_norm = torch.nn.functional.normalize(x_dst, p=2, dim=1)
            weights = (x_src_norm * x_dst_norm).sum(dim=1)


            weights = (weights + 1.0) / 2.0

        else:
            raise ValueError(f"Unknown edge_weighting: {self.edge_weighting}")

        return weights.unsqueeze(1)

    def _build_knn_graph(self, pos: torch.Tensor) -> torch.Tensor:
        pos_np = pos.cpu().numpy()
        n_nodes = pos_np.shape[0]

        if n_nodes <= 1:
            return torch.empty((2, 0), dtype=torch.long)

        k = min(self.k + 1, n_nodes)

        nbrs = NearestNeighbors(
            n_neighbors=k,
            algorithm="ball_tree"
        ).fit(pos_np)

        _, neighbors = nbrs.kneighbors(pos_np)

        edges = []
        for i, neigh in enumerate(neighbors):
            for j in neigh[1:]:
                edges.append([i, j])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def _build_radius_graph(self, pos: torch.Tensor) -> torch.Tensor:
        pos_np = pos.cpu().numpy()
        n_nodes = pos_np.shape[0]

        if n_nodes <= 1:
            return torch.empty((2, 0), dtype=torch.long)

        adj = radius_neighbors_graph(pos_np, radius=self.radius, mode='connectivity', include_self=False)
        rows, cols = adj.nonzero()

        edges = []
        for r, c in zip(rows, cols):
            edges.append([r, c])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index


@register_dataset(('gsdb', "graph_cached", "pov-gigapath", True))
@register_dataset(('gsdb', "graph_cached", "pov_gigapath", True))
@register_dataset(('gsdb', "graph_cached", "uni2", True))
@register_dataset(('gsdb', "graph_cached", "resnet50", True))
@register_dataset(('camelyon', "graph_cached", "pov-gigapath", True))
@register_dataset(('camelyon', "graph_cached", "uni2", True))
@register_dataset(('camelyon', "graph_cached", "resnet50", True))
class WSICachedGraphDataset(Dataset):
    """
    Loads one precomputed graph (.pt) per WSI from disk.

    Graph files are resolved from paths["GRAPH_DIR"], while slide names and labels
    are taken from the requested CSV split.
    """

    def __init__(
        self,
        split: str = "train",
        transform=None,
        task: str = "inflammation",
        he_only: bool = True,
        scanner1_only: bool = False,
        paths: Optional[Dict[str, Path]] = None,
        dataset: str = 'gsdb',
        split_seed: int = 42,
        split_phase: Optional[str] = None,
        k: Optional[int] = None,
        radius: Optional[float] = None,
        normalize_pos: bool = True,
        embedding_type: str = 'uni2',
        pipeline_type: str = 'hierarchical',
        label_mode: str = 'binary',
        graph_target_dir: Optional[Path] = None,
        strict_graph_name_match: bool = True,
        cache_in_memory: bool = False,
        resplit_large_regions: bool = False,
        resplit_target_patches_per_region: int = 250,
        resplit_k_region: int = 5,
        resplit_min_patches_per_region: int = 0,
        train_augmentations: Optional[List[Dict[str, Any]]] = None,
        label_field: Optional[str] = None,
        exclude_label_field_values: Optional[List[str]] = None,
    ):
        """Loads and resolves cached per-slide graph records for the given split."""
        self.split = split
        self.transform = transform
        self.task = task
        self.dataset = dataset
        self.label_mode = str(label_mode).strip().lower()
        self.exclude_label_field_values = {
            str(v).strip().lower() for v in (exclude_label_field_values or [])
        }
        if self.label_mode not in {'binary', 'multiclass'}:
            raise ValueError("label_mode must be one of: binary, multiclass")
        self.label_field = str(label_field).strip() if label_field else None
        self._extra_label_field_names_cache: Optional[set] = None

        self.split_target_col = self._resolve_target_column()
        self.split_seed = int(split_seed)
        self.split_phase = str(split_phase).strip().lower() if split_phase else None
        self.pipeline_type = str(pipeline_type).strip() if pipeline_type is not None else None
        self.graph_target_dir = Path(graph_target_dir).expanduser().resolve() if graph_target_dir is not None else None
        self.strict_graph_name_match = bool(strict_graph_name_match)
        self.resplit_large_regions = bool(resplit_large_regions)
        self.resplit_target_patches_per_region = int(resplit_target_patches_per_region)
        self.resplit_k_region = int(resplit_k_region)
        self.resplit_min_patches_per_region = int(resplit_min_patches_per_region)
        self.cache_in_memory = bool(cache_in_memory)
        self.train_augmentations = list(train_augmentations or [])
        self._aug_call_counts: Dict[str, int] = {}
        if self.train_augmentations and self.cache_in_memory and split == "train":
            logging.warning(
                "train_augmentations is set together with cache_in_memory=True: __getitem__ "
                "returns the cached (already-augmented) graph on every access after the first, "
                "so augmentation only actually varies on epoch 1. Disable cache_in_memory for "
                "this dataset if per-epoch-fresh augmentation is required."
            )
        self._graph_cache: Dict[int, "Data"] = {}

        if paths is None:
            paths = get_project_paths()

        self.prefix = f"{self.pipeline_type}_{embedding_type}" if self.pipeline_type else None

        self.paths = paths
        graph_root = Path(paths["GRAPH_DIR"])
        if self.graph_target_dir is not None:
            if not self.graph_target_dir.exists() or not self.graph_target_dir.is_dir():
                raise FileNotFoundError(f"graph_target_dir does not exist or is not a directory: {self.graph_target_dir}")
            self.graph_dirs = [self.graph_target_dir]
        else:
            self.graph_dirs = self._resolve_graph_directories(graph_root, self.dataset, embedding_type, self.pipeline_type)
        self.graph_dir = self.graph_dirs[0] if self.graph_dirs else graph_root / self.dataset

        self.split_path = self._resolve_split_path(Path(paths["SPLITS_DIR"]), split, he_only)
        print(f"[GRAPH-CACHE][{self.split}] split csv: {self.split_path}")
        print(f"[GRAPH-CACHE][{self.split}] graph search directories ({len(self.graph_dirs)}):")
        preview_dirs = self.graph_dirs[:5]
        for graph_dir in preview_dirs:
            print(f"[GRAPH-CACHE][{self.split}]   - {graph_dir}")
        if len(self.graph_dirs) > len(preview_dirs):
            print(
                f"[GRAPH-CACHE][{self.split}]   ... and {len(self.graph_dirs) - len(preview_dirs)} more directories"
            )

        split_df = pd.read_csv(self.split_path)
        
        if scanner1_only and "scanner_id" in split_df.columns:
            split_df = split_df[split_df["scanner_id"] == 1]
        self.split_df = split_df

        logging.debug(f"Graph split preview for split='{split}': {self.split_df.head().to_dict(orient='records')}")

        self.records = self._build_records(self.split_df)

        logging.debug(f"Resolved graph records for split='{split}': {self.records}")

        self.num_missing_graphs = int(getattr(self, "num_missing_graphs", 0))
        self.num_resolved_graphs = len(self.records)
        self.num_expected_graphs = int(getattr(self, "num_expected_graphs", self.num_missing_graphs + self.num_resolved_graphs))
        self.num_missing_positive = int(getattr(self, "num_missing_positive", 0))
        self.num_missing_negative = int(getattr(self, "num_missing_negative", 0))
        self.num_resolved_positive = int(getattr(self, "num_resolved_positive", 0))
        self.num_resolved_negative = int(getattr(self, "num_resolved_negative", 0))

        if len(self.records) == 0:
            raise RuntimeError(f"No graph samples resolved for split '{split}' from {self.split_path}")

        logging.info(
            "Loaded WSICachedGraphDataset split=%s with %d graphs from %s",
            split,
            len(self.records),
            ", ".join(str(p) for p in self.graph_dirs[:3]),
        )

        if self.num_missing_graphs > 0:
            missing_pct = (100.0 * self.num_missing_graphs / self.num_expected_graphs) if self.num_expected_graphs > 0 else 0.0
            print(
                f"[GRAPH-CACHE][{self.split}] overview: expected={self.num_expected_graphs} | "
                f"resolved={self.num_resolved_graphs} | missing={self.num_missing_graphs} ({missing_pct:.1f}%)"
            )
            if self.label_mode == "binary":
                print(
                    f"[GRAPH-CACHE][{self.split}] class overview (resolved): "
                    f"positive={self.num_resolved_positive}, negative={self.num_resolved_negative}"
                )
                print(
                    f"[GRAPH-CACHE][{self.split}] class overview (missing): "
                    f"positive={self.num_missing_positive}, negative={self.num_missing_negative}"
                )
            print(
                f"[GRAPH-CACHE][{self.split}] missing graphs: {self.num_missing_graphs} | "
                f"resolved: {self.num_resolved_graphs}"
            )

    def __len__(self) -> int:
        """Returns the number of resolved graph records in the dataset."""
        return len(self.records)

    def __getitem__(self, idx: int):
        """Loads, processes, and returns the cached graph for the sample at the given index."""
        if self.cache_in_memory and idx in self._graph_cache:
            return self._graph_cache[idx]

        record = self.records[idx]

        map_location = torch.device("cpu")

        try:
            data = torch.load(record["graph_path"], map_location=map_location, weights_only=False)
        except TypeError:
            data = torch.load(record["graph_path"], map_location=map_location)

        self._normalize_graph_meta(data)

        if hasattr(data, "keys"):
            for key in list(data.keys()):
                if key == "meta":
                    continue
                if isinstance(data[key], dict):
                    del data[key]


        if hasattr(data, 'x') and torch.is_tensor(data.x):
            label_device = data.x.device
        elif hasattr(data, 'node_types') and 'patch' in data.node_types and torch.is_tensor(data['patch'].x):
            label_device = data['patch'].x.device
        else:
            label_device = map_location
        if self.label_field:
            attr = f"{self.label_field}_y"
            if not (hasattr(data, "node_types") and "slide" in data.node_types and hasattr(data["slide"], attr)):
                raise ValueError(
                    f"label_field='{self.label_field}' requested but graph for slide "
                    f"'{record.get('slide_name', 'unknown')}' has no data['slide'].{attr} -- "
                    "rebuild graphs with a label_fields-enabled graph-build config (see "
                    "configs/graph_build/*_labels.yaml)."
                )
            slide_label = float(data["slide"][attr].view(-1)[0].item())
        else:
            slide_label = float(record["label"])
        label_tensor = torch.tensor([slide_label], dtype=torch.float32, device=label_device)

        self._validate_loaded_graph(data=data, slide_name=str(record.get("slide_name", "unknown")))

        if self.resplit_min_patches_per_region > 0:
            merge_small_regions(
                data,
                min_patches_per_region=self.resplit_min_patches_per_region,
            )

        if self.resplit_large_regions:
            resplit_large_regions(
                data,
                target_patches_per_region=self.resplit_target_patches_per_region,
                k_region=self.resplit_k_region,
            )

        if self.split == "train" and self.train_augmentations:
            aug_rng = self._get_aug_rng(str(record.get("slide_name") or record["graph_path"]))
            apply_graph_augmentations(
                data,
                self.train_augmentations,
                rng=aug_rng,
                second_graph_provider=lambda: self._load_random_graph_for_mixup(aug_rng),
            )
            if hasattr(data, "_mixup_lambda"):
                lam = data._mixup_lambda
                label_b = data._mixup_label_b
                slide_label = lam * slide_label + (1.0 - lam) * label_b
                label_tensor = torch.tensor([slide_label], dtype=torch.float32, device=label_device)
                del data._mixup_lambda
                del data._mixup_label_b

        self._ensure_uniform_annotation_keys(data=data, device=label_device)
        if not self.label_field:
            self._attach_hierarchical_supervision(data=data, slide_label=slide_label, device=label_device)

        data.y = label_tensor
        if hasattr(data, "node_types") and "slide" in data.node_types:
            data["slide"].y = label_tensor.clone()
        data.slide_name = record["slide_name"]

        if self.cache_in_memory:
            self._graph_cache[idx] = data

        return data

    def _get_aug_rng(self, record_key: str) -> np.random.Generator:
        visit_count = self._aug_call_counts.get(record_key, 0)
        self._aug_call_counts[record_key] = visit_count + 1
        seed = (zlib.crc32(f"{self.split_seed}:{record_key}".encode("utf-8")) + visit_count * 1_000_003) & 0xFFFFFFFF
        return np.random.default_rng(seed)

    def _load_random_graph_for_mixup(self, rng: np.random.Generator):
        if len(self.records) <= 1:
            return None
        other_idx = int(rng.integers(0, len(self.records)))
        record_b = self.records[other_idx]
        try:
            try:
                data_b = torch.load(record_b["graph_path"], map_location=torch.device("cpu"), weights_only=False)
            except TypeError:
                data_b = torch.load(record_b["graph_path"], map_location=torch.device("cpu"))
        except Exception as exc:
            logging.warning("graph_mixup: failed to load second graph '%s' (%s) -- skipping this mixup call.", record_b.get("slide_name", "?"), exc)
            return None

        if self.label_field:
            attr = f"{self.label_field}_y"
            if not (hasattr(data_b, "node_types") and "slide" in data_b.node_types and hasattr(data_b["slide"], attr)):
                logging.warning(
                    "graph_mixup: second graph '%s' has no data['slide'].%s -- skipping this mixup call.",
                    record_b.get("slide_name", "?"), attr,
                )
                return None
            label_b = float(data_b["slide"][attr].view(-1)[0].item())
            return data_b, label_b

        return data_b, float(record_b["label"])

    def _normalize_graph_meta(self, data) -> None:
        raw_meta = getattr(data, "meta", None)
        if not isinstance(raw_meta, dict):
            raw_meta = {}

        def _as_str_list(value) -> list[str]:
            return [str(v) for v in value] if isinstance(value, (list, tuple)) else []

        normalized = {
            "annotation_label_vocab": _as_str_list(raw_meta.get("annotation_label_vocab", [])),
            "annotation_labels_enabled": bool(raw_meta.get("annotation_labels_enabled", False)),
        }

        if self._extra_label_field_names_cache is None:
            extra_field_names: set[str] = set()
            extra_field_names.update(_as_str_list(raw_meta.get("extra_label_fields", [])))
            extra_field_names.update(_as_str_list(raw_meta.get("label_fields", [])))
            self._extra_label_field_names_cache = extra_field_names
        extra_field_names = self._extra_label_field_names_cache
        for field_name in extra_field_names:
            normalized[f"{field_name}_label_vocab"] = _as_str_list(raw_meta.get(f"{field_name}_label_vocab", []))

        data.meta = normalized

    def _ensure_uniform_annotation_keys(self, data, device: torch.device) -> None:
        if not hasattr(data, "node_types"):
            return

        for node_type in ("patch", "region"):
            if node_type not in data.node_types:
                continue

            store = data[node_type]
            num_nodes = int(getattr(store, "num_nodes", 0) or (store.x.size(0) if hasattr(store, "x") else 0))
            if num_nodes <= 0:
                continue

            raw = getattr(store, "annotation_y", None)
            if torch.is_tensor(raw):
                values = raw.view(-1).to(device=device, dtype=torch.long)
                if values.numel() == num_nodes:
                    store.annotation_y = values
                    continue

            store.annotation_y = torch.full((num_nodes,), -1, dtype=torch.long, device=device)

        if "region" in data.node_types:
            region_store = data["region"]
            num_regions = int(getattr(region_store, "num_nodes", 0) or (region_store.x.size(0) if hasattr(region_store, "x") else 0))
            if num_regions > 0:
                raw_ambiguous = getattr(region_store, "label_ambiguous", None)
                if torch.is_tensor(raw_ambiguous) and raw_ambiguous.numel() == num_regions:
                    region_store.label_ambiguous = raw_ambiguous.view(-1).to(device=device, dtype=torch.bool)
                else:
                    region_store.label_ambiguous = torch.zeros((num_regions,), dtype=torch.bool, device=device)

    def _validate_loaded_graph(self, data, slide_name: str) -> None:
        if not hasattr(data, "node_types"):
            if not hasattr(data, "x") or not hasattr(data, "edge_index"):
                raise ValueError(
                    f"Loaded cached graph for slide '{slide_name}' has neither HeteroData schema nor flat Data tensors."
                )
            return

        required_node_types = {"patch", "region", "slide"}
        node_types = set(getattr(data, "node_types", []))
        missing_node_types = required_node_types - node_types
        if missing_node_types:
            raise ValueError(
                f"Cached graph for slide '{slide_name}' is missing required node types: {sorted(missing_node_types)}"
            )

        required_edges = {
            ("patch", "in", "region"),
            ("region", "part_of", "slide"),
        }
        edge_types = set(getattr(data, "edge_types", []))
        missing_edges = required_edges - edge_types
        if missing_edges:
            raise ValueError(
                f"Cached graph for slide '{slide_name}' is missing required contextual edge types: {sorted(missing_edges)}"
            )

    def _attach_hierarchical_supervision(self, data, slide_label: float, device: torch.device) -> None:
        if not hasattr(data, "node_types"):
            return
        if "patch" not in data.node_types or "region" not in data.node_types:
            return

        slide_label_id = int(round(slide_label))

        patch_store = data["patch"]
        region_store = data["region"]

        n_patches = int(getattr(patch_store, "num_nodes", 0) or (patch_store.x.size(0) if hasattr(patch_store, "x") else 0))
        n_regions = int(getattr(region_store, "num_nodes", 0) or (region_store.x.size(0) if hasattr(region_store, "x") else 0))
        if n_patches <= 0 or n_regions <= 0:
            return

        annotation_vocab = []
        if hasattr(data, "meta") and isinstance(data.meta, dict):
            annotation_vocab = data.meta.get("annotation_label_vocab") or []

        patch_annotation = self._extract_annotation_ids(patch_store, size=n_patches, device=device)
        region_annotation = self._extract_annotation_ids(region_store, size=n_regions, device=device)

        patch_ann_binary = self._map_annotation_ids_to_binary(
            annotation_ids=patch_annotation,
            annotation_vocab=annotation_vocab,
            device=device,
        )
        region_ann_binary = self._map_annotation_ids_to_binary(
            annotation_ids=region_annotation,
            annotation_vocab=annotation_vocab,
            device=device,
        )

        region_supervision = torch.full((n_regions,), slide_label_id, dtype=torch.long, device=device)
        region_has_label = region_ann_binary >= 0
        region_supervision[region_has_label] = region_ann_binary[region_has_label]

        patch_supervision = patch_ann_binary.clone()
        patch_missing = patch_supervision < 0

        patch_to_region = data[("patch", "in", "region")].edge_index
        if torch.is_tensor(patch_to_region) and patch_to_region.numel() > 0:
            src_patch = patch_to_region[0].long().to(device)
            dst_region = patch_to_region[1].long().to(device)
            valid = (
                (src_patch >= 0)
                & (src_patch < n_patches)
                & (dst_region >= 0)
                & (dst_region < n_regions)
            )
            src_patch = src_patch[valid]
            dst_region = dst_region[valid]
            if src_patch.numel() > 0:
                region_fallback = region_supervision[dst_region]
                unresolved = patch_supervision[src_patch] < 0
                patch_supervision[src_patch[unresolved]] = region_fallback[unresolved]

        patch_supervision[patch_supervision < 0] = slide_label_id

        patch_store.y = patch_supervision
        region_store.y = region_supervision

        if hasattr(data, "meta") and isinstance(data.meta, dict):
            data.meta["supervision_fallback"] = {
                "policy": "patch->region->slide",
                "n_patches": int(n_patches),
                "n_regions": int(n_regions),
                "patch_direct_annotation": int((patch_ann_binary >= 0).sum().item()),
                "region_direct_annotation": int((region_ann_binary >= 0).sum().item()),
                "patch_final_from_slide": int((patch_supervision == slide_label_id).sum().item()),
            }

    def _extract_annotation_ids(self, node_store, size: int, device: torch.device) -> torch.Tensor:
        raw = getattr(node_store, "annotation_y", None)
        if torch.is_tensor(raw):
            values = raw.view(-1).to(device=device, dtype=torch.long)
            if values.numel() == size:
                return values
        return torch.full((size,), -1, dtype=torch.long, device=device)

    def _map_annotation_ids_to_binary(
        self,
        annotation_ids: torch.Tensor,
        annotation_vocab: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        mapped = torch.full_like(annotation_ids, -1, device=device)
        if annotation_ids.numel() == 0:
            return mapped

        non_negative = annotation_ids[annotation_ids >= 0]
        if non_negative.numel() == 0:
            return mapped

        unique_ids = sorted(set(int(x) for x in non_negative.tolist()))

        id_to_binary: Dict[int, int] = {}
        if annotation_vocab:
            for ann_id in unique_ids:
                if ann_id < 0 or ann_id >= len(annotation_vocab):
                    continue
                label_text = str(annotation_vocab[ann_id]).strip().lower()
                if self._is_tumor_like_annotation(label_text):
                    id_to_binary[ann_id] = 1
                elif self._is_normal_like_annotation(label_text):
                    id_to_binary[ann_id] = 0

        for ann_id, binary in id_to_binary.items():
            mapped[annotation_ids == ann_id] = int(binary)

        return mapped

    @staticmethod
    def _is_tumor_like_annotation(label_text: str) -> bool:
        tokens = ("tumor", "metast", "macro", "micro", "itc", "positive", "status=inflamed")
        return any(token in label_text for token in tokens)

    @staticmethod
    def _is_normal_like_annotation(label_text: str) -> bool:
        tokens = ("normal", "negative", "benign", "healthy", "background", "non_tumor", "status=noninflamed")
        return any(token in label_text for token in tokens)

    def _resolve_split_path(self, splits_dir: Path, split: str, he_only: bool) -> Path:
        seed_dir = f"seed_{self.split_seed}"
        dataset_root = splits_dir / self.dataset
        phase_dir = self.split_phase

        if split.endswith(".csv"):
            candidates = []
            if phase_dir:
                candidates.extend([
                    dataset_root / seed_dir / phase_dir / split,
                    splits_dir / seed_dir / phase_dir / split,
                ])
            candidates.extend([
                dataset_root / seed_dir / split,
                splits_dir / seed_dir / split,
                dataset_root / split,
                splits_dir / split,
            ])
            for direct in candidates:
                if direct.exists():
                    return direct

            nested = list(dataset_root.rglob(split)) if dataset_root.exists() else []
            if not nested:
                nested = list(splits_dir.rglob(split))
            if nested:
                return nested[0]
            raise FileNotFoundError(f"Split CSV not found: {split}")

        if self.dataset == 'camelyon':
            candidates = [f"{split}.csv"]
        else:
            if he_only:
                candidates = [f"{split}_HE.csv"]
            else:
                candidates = [f"{split}_all_stains.csv"]

        for candidate in candidates:
            preferred = dataset_root / seed_dir / candidate
            legacy = splits_dir / seed_dir / candidate
            phase_preferred = dataset_root / seed_dir / phase_dir / candidate if phase_dir else None
            phase_legacy = splits_dir / seed_dir / phase_dir / candidate if phase_dir else None
            if phase_preferred is not None and phase_preferred.exists():
                return phase_preferred
            if phase_legacy is not None and phase_legacy.exists():
                return phase_legacy
            if preferred.exists():
                return preferred
            if legacy.exists():
                return legacy

            nested = list(dataset_root.rglob(candidate)) if dataset_root.exists() else []
            if not nested:
                nested = list(splits_dir.rglob(candidate))
            if nested:
                return nested[0]

        raise FileNotFoundError(
            f"No split CSV found for split='{split}' in {splits_dir}. "
            f"Tried: {candidates}"
        )

    def _build_records(self, split_df: pd.DataFrame) -> List[Dict]:
        records: List[Dict] = []
        missing_graphs: List[str] = []
        skipped_invalid_task_rows: List[str] = []
        conflicting_slide_labels: List[str] = []
        unknown_label_field_slides: List[str] = []
        excluded_label_field_slides: List[str] = []
        missing_positive = 0
        missing_negative = 0
        resolved_positive = 0
        resolved_negative = 0

        slide_to_label: Dict[str, float] = {}

        for _, row in split_df.iterrows():
            raw_slide_name = self._extract_slide_name(row)
            if raw_slide_name is None:
                skipped_invalid_task_rows.append("unknown")
                continue

            try:
                label = self._extract_label(row)
            except ValueError:
                skipped_invalid_task_rows.append(raw_slide_name)
                continue

            slide_name = raw_slide_name

            if slide_name in slide_to_label and slide_to_label[slide_name] != label:
                conflicting_slide_labels.append(raw_slide_name)
                continue

            slide_to_label[slide_name] = label

        if self.label_field:
            print(
                f"[GRAPH-CACHE][{self.split}] checking label_field='{self.label_field}' on "
                f"{len(slide_to_label)} candidate graph(s) (loads each file once to exclude "
                "'unknown'-labeled slides) -- this can take a while with no output in between..."
            )
        label_field_checked = 0

        for slide_name, label in slide_to_label.items():
            candidate_names = [slide_name]
            if self.prefix:
                candidate_names.insert(0, f"{slide_name}_{self.prefix}")

            graph_path = None
            resolved_candidate_name = None
            for candidate_name in candidate_names:
                graph_path = self._resolve_graph_path(candidate_name)
                if graph_path is not None:
                    resolved_candidate_name = candidate_name
                    break

            if graph_path is None:
                missing_graphs.append(slide_name)
                if self.label_mode == "binary":
                    if float(label) > 0.5:
                        missing_positive += 1
                    else:
                        missing_negative += 1
                continue

            if self.label_field:
                is_unknown = self._resolve_label_field_is_unknown(graph_path)
                label_field_checked += 1
                if label_field_checked % 25 == 0 or label_field_checked == len(slide_to_label):
                    print(
                        f"[GRAPH-CACHE][{self.split}] label_field check: "
                        f"{label_field_checked}/{len(slide_to_label)}"
                    )
                if is_unknown:
                    unknown_label_field_slides.append(slide_name)
                    continue
                if self._resolve_label_field_is_excluded(graph_path):
                    excluded_label_field_slides.append(slide_name)
                    continue

            if self.label_mode == "binary":
                if float(label) > 0.5:
                    resolved_positive += 1
                else:
                    resolved_negative += 1

            records.append(
                {
                    "slide_name": slide_name,
                    "resolved_graph_key": resolved_candidate_name,
                    "label": label,
                    "graph_path": graph_path,
                }
            )

        self.num_expected_graphs = len(slide_to_label)
        self.num_missing_graphs = len(missing_graphs)
        self.num_missing_positive = missing_positive
        self.num_missing_negative = missing_negative
        self.num_resolved_positive = resolved_positive
        self.num_resolved_negative = resolved_negative
        if missing_graphs:
            preview_count = min(25, len(missing_graphs))
            for slide_name in missing_graphs[:preview_count]:
                print(f"[GRAPH-CACHE][{self.split}] missing graph for slide: {slide_name}")
            if len(missing_graphs) > preview_count:
                print(
                    f"[GRAPH-CACHE][{self.split}] ... and {len(missing_graphs) - preview_count} more missing graphs"
                )

            logging.warning(
                "Missing graph files for %d slides (showing first 10): %s",
                len(missing_graphs),
                missing_graphs[:10],
            )

        if skipped_invalid_task_rows:
            logging.info(
                "Skipped %d rows for task '%s' due to invalid or missing target labels (showing first 10 slides): %s",
                len(skipped_invalid_task_rows),
                self.task,
                skipped_invalid_task_rows[:10],
            )

        if conflicting_slide_labels:
            logging.warning(
                "Found %d slides with conflicting labels for task '%s' (showing first 10): %s",
                len(conflicting_slide_labels),
                self.task,
                conflicting_slide_labels[:10],
            )

        self.num_unknown_label_field_dropped = len(unknown_label_field_slides)
        if unknown_label_field_slides:
            logging.warning(
                "Excluded %d slide(s) from split '%s' whose label_field='%s' resolved to the "
                "graph-build pipeline's 'unknown' catch-all class (no matching annotation/CSV-join "
                "value at graph-build time) -- training/evaluating on an undefined ground truth "
                "would just inject label noise (showing first 10): %s",
                len(unknown_label_field_slides),
                self.split,
                self.label_field,
                unknown_label_field_slides[:10],
            )

        self.num_excluded_label_field_dropped = len(excluded_label_field_slides)
        if excluded_label_field_slides:
            logging.info(
                "Excluded %d slide(s) from split '%s' whose label_field='%s' resolved to one of "
                "exclude_label_field_values=%s (opt-in filter, e.g. dropping 'negative' to train "
                "a tumor-positive-only subtype head; showing first 10): %s",
                len(excluded_label_field_slides),
                self.split,
                self.label_field,
                sorted(self.exclude_label_field_values),
                excluded_label_field_slides[:10],
            )

        return records

    def _resolve_label_field_is_unknown(self, graph_path: Path) -> bool:
        """True if this graph's label_field slide target resolved to the 'unknown' vocab class.

        Only ever called when self.label_field is set (CAMELYON subtype-style multiclass runs) --
        GSDB and every binary run never pass label_field, so this check (and the exclusion it
        drives in _build_records) never engages for them. 'unknown' here means the graph-build
        pipeline's CSV-join/annotation lookup found no matching value for this slide (see
        resolve_label_fields in src/graph_creation_new/helper/labels.py) -- it is a data-quality
        marker, not a real class, so slides landing on it are excluded from the dataset entirely
        rather than trained/evaluated against a made-up label.
        """
        try:
            try:
                data = torch.load(graph_path, map_location="cpu", weights_only=False)
            except TypeError:
                data = torch.load(graph_path, map_location="cpu")
        except Exception:
            return False

        self._normalize_graph_meta(data)
        attr = f"{self.label_field}_y"
        if not (hasattr(data, "node_types") and "slide" in data.node_types and hasattr(data["slide"], attr)):
            return False

        vocab = data.meta.get(f"{self.label_field}_label_vocab") if hasattr(data, "meta") else None
        if not isinstance(vocab, list) or "unknown" not in vocab:
            return False

        resolved_id = int(getattr(data["slide"], attr).view(-1)[0].item())
        return resolved_id == vocab.index("unknown")

    def _resolve_label_field_is_excluded(self, graph_path: Path) -> bool:
        """True if this graph's label_field slide target resolved to one of
        self.exclude_label_field_values (e.g. dropping 'negative' slides entirely to train a
        tumor-positive-only subtype head -- CAMELYON's itc/micro/macro-vs-negative split).

        Only engages when self.exclude_label_field_values is non-empty (opt-in, default empty
        set) AND self.label_field is set -- every existing caller (GSDB, every binary run, every
        camelyon subtype run that doesn't pass this) leaves it empty, so this is a no-op for them.
        Mirrors _resolve_label_field_is_unknown's graph-loading/vocab-lookup exactly, just
        matching a caller-supplied set of class names instead of the fixed 'unknown' marker.
        """
        if not self.exclude_label_field_values:
            return False
        try:
            try:
                data = torch.load(graph_path, map_location="cpu", weights_only=False)
            except TypeError:
                data = torch.load(graph_path, map_location="cpu")
        except Exception:
            return False

        self._normalize_graph_meta(data)
        attr = f"{self.label_field}_y"
        if not (hasattr(data, "node_types") and "slide" in data.node_types and hasattr(data["slide"], attr)):
            return False

        vocab = data.meta.get(f"{self.label_field}_label_vocab") if hasattr(data, "meta") else None
        if not isinstance(vocab, list):
            return False

        resolved_id = int(getattr(data["slide"], attr).view(-1)[0].item())
        if resolved_id < 0 or resolved_id >= len(vocab):
            return False
        return str(vocab[resolved_id]).strip().lower() in self.exclude_label_field_values

    def _resolve_target_column(self) -> str:
        dataset_name = str(self.dataset).strip().lower()

        if dataset_name == "gsdb":
            return "inflammation_status" if self.label_mode == "binary" else "tissue_type"

        if dataset_name == "camelyon":
            return "tumor_status" if self.label_mode == "binary" else "stage"

        return "inflammation_status" if self.task == "inflammation" else "label"

    @staticmethod
    def _normalize_camelyon_stage(stage: str) -> Optional[str]:
        s = str(stage).strip().lower()

        if not s:
            return None
        if "negative" in s or s in {"pn0", "n0", "0"}:
            return "negative"
        if s in {"itc", "pn0(i+)", "n0(i+)", "1"}:
            return "itc"
        if "micro" in s or s in {"pn1mi", "n1mi", "2"}:
            return "micro"
        if "macro" in s or s in {"pn1", "pn2", "n1", "n2", "3"}:
            return "macro"

        return None

    def _extract_slide_name(self, row: pd.Series) -> Optional[str]:
        for column in ("slide_name", "wsi_name", "image_name"):
            value = row.get(column, None)
            if isinstance(value, str):
                value = value.strip()
                if value:
                    return value

        return None

    def _resolve_graph_path(self, slide_name: str) -> Optional[Path]:
        for graph_dir in self.graph_dirs:
            exact = graph_dir / f"{slide_name}.pt"
            if exact.exists():
                return exact

            exact_matches = list(graph_dir.rglob(f"{slide_name}.pt"))
            if exact_matches:
                return exact_matches[0]

            suffixed_matches = list(graph_dir.rglob(f"{slide_name}_*.pt"))
            if suffixed_matches:
                return suffixed_matches[0]

            if not self.strict_graph_name_match:
                contains_matches = list(graph_dir.rglob(f"*{slide_name}*.pt"))
                if contains_matches:
                    return contains_matches[0]

        return None

    @staticmethod
    def _dedupe_existing_dirs(candidates: List[Path]) -> List[Path]:
        deduped: List[Path] = []
        seen = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if not resolved.exists() or not resolved.is_dir():
                continue
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(resolved)
        return deduped

    def _resolve_graph_directories(
        self,
        graph_root: Path,
        dataset_name: str,
        embedding_type: str,
        pipeline_type: Optional[str],
    ) -> List[Path]:
        dataset_root = graph_root / dataset_name
        search_roots: List[Path] = [dataset_root, graph_root]
        candidates: List[Path] = []

        pipeline_token = pipeline_type.strip() if isinstance(pipeline_type, str) and pipeline_type.strip() else None
        legacy_prefix = f"{pipeline_token}_{embedding_type}" if pipeline_token else None

        if legacy_prefix:
            candidates.extend([
                dataset_root / legacy_prefix,
                graph_root / legacy_prefix,
            ])

        if pipeline_token:
            if pipeline_token.startswith("pl-") or "__" in pipeline_token:
                if pipeline_token.endswith(f"_{embedding_type}"):
                    candidates.extend([dataset_root / pipeline_token, graph_root / pipeline_token])
                else:
                    candidates.extend([
                        dataset_root / f"{pipeline_token}_{embedding_type}",
                        graph_root / f"{pipeline_token}_{embedding_type}",
                        dataset_root / pipeline_token,
                        graph_root / pipeline_token,
                    ])

        for root in search_roots:
            if not root.exists() or not root.is_dir():
                continue
            for child in sorted(root.iterdir(), key=lambda p: p.name):
                if not child.is_dir():
                    continue
                if not child.name.endswith(f"_{embedding_type}"):
                    continue
                if pipeline_token and pipeline_token not in child.name and not child.name.startswith(f"pl-{pipeline_token}"):
                    continue
                candidates.append(child)

        resolved = self._dedupe_existing_dirs(candidates)
        if resolved:
            return resolved

        fallback = self._dedupe_existing_dirs([dataset_root, graph_root])
        return fallback if fallback else [dataset_root]

    def _extract_label(self, row: pd.Series) -> float:
        if self.label_field:
            return 0.0

        dataset_name = str(self.dataset).strip().lower()

        if dataset_name == "gsdb" and self.label_mode == "binary":
            status = row.get("inflammation_status", None)
            if isinstance(status, str):
                status = status.strip().lower()
                if status == "inflamed":
                    return 1.0
                if status == "noninflamed":
                    return 0.0
            raise ValueError(
                f"Could not infer inflammation label for slide '{row.get('slide_name', 'unknown')}'. "
                "Expected inflammation_status to be one of: inflamed, noninflamed"
            )

        if dataset_name == "gsdb" and self.label_mode == "multiclass":
            tissue_type = row.get("tissue_type", None)
            if isinstance(tissue_type, str):
                tissue_type = tissue_type.strip().lower()
                if tissue_type == "corpus":
                    return 1.0
                if tissue_type == "antrum":
                    return 0.0

            label = self._extract_numeric_label(row)
            if label is not None:
                return label

            raise ValueError(
                f"Could not infer tissue label for slide '{row.get('slide_name', 'unknown')}'. "
                "Expected one of: tissue_type, label, y"
            )

        if dataset_name == "camelyon" and self.label_mode == "binary":
            tumor_status = row.get("tumor_status", None)
            if isinstance(tumor_status, str):
                tumor_status = tumor_status.strip().lower()
                if tumor_status == "normal":
                    return 0.0
                if tumor_status == "tumor":
                    return 1.0

            label = self._extract_numeric_label(row)
            if label is not None:
                return label

            raise ValueError(
                f"Could not infer CAMELYON binary label for slide '{row.get('slide_name', 'unknown')}'. "
                "Expected one of: tumor_status, label, y"
            )

        if dataset_name == "camelyon" and self.label_mode == "multiclass":
            stage_value = row.get("stage", None)
            if pd.notna(stage_value):
                stage_norm = self._normalize_camelyon_stage(str(stage_value))
                stage_to_index = {
                    "negative": 0.0,
                    "itc": 1.0,
                    "micro": 2.0,
                    "macro": 3.0,
                }
                if stage_norm in stage_to_index:
                    return stage_to_index[stage_norm]

            label = self._extract_numeric_label(row)
            if label is not None:
                return label

            raise ValueError(
                f"Could not infer CAMELYON multiclass label for slide '{row.get('slide_name', 'unknown')}'. "
                "Expected one of: stage, label, y"
            )

        label = self._extract_numeric_label(row)
        if label is not None:
            return label

        raise ValueError(
            f"Could not infer label for dataset '{self.dataset}', task '{self.task}', mode '{self.label_mode}' and slide '{row.get('slide_name', 'unknown')}'. "
            "Provide label column in split CSV."
        )

    def _extract_numeric_label(self, row: pd.Series) -> Optional[float]:
        for column in ("label", "y"):
            value = row.get(column, None)
            if pd.isna(value):
                continue

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue

            if np.isfinite(numeric_value):
                return numeric_value

        return None
