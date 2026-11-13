from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.paths.path_resolver import PathResolver
from src.graph_creation_new.enums import (
    AnnotationType,
    DatasetType,
    MaskType,
    NodeCreationMode,
    EdgeConstructionMode,
)


@dataclass
class BuildConfig:
    patch_level: int = 0
    patch_size: int = 256
    stride: int = 256
    target_patch_size_um: Optional[float] = None
    auto_detect_target_patch_size_um: bool = False

    thumb_max_size: int = 2000

    tissue_frac_min: float = 0.30
    sat_min: int = 15
    val_max: int = 245
    mask_enable_color_normalization: bool = True
    mask_color_norm_strength: float = 0.85
    mask_use_adaptive_hsv: bool = True
    mask_sat_percentile: float = 35.0
    mask_val_percentile: float = 99.0

    morph_kernel: int = 7
    min_component_area: int = 300
    mask_drop_border_components: bool = False
    mask_border_margin_px: int = 0
    mask_require_core_connection_for_border_components: bool = False
    mask_core_margin_px: int = 0
    mask_keep_largest_components: Optional[int] = None
    region_construction_mode: str = "annotation_driven"
    geometric_max_patches_per_region: Optional[int] = None

    k_patch: int = 12
    k_region: int = 6
    patch_radius_px: Optional[float] = None

    pipeline: str = "dynamical"
    feature_batch_size: int = 64
    feature_prep_workers: int = 1
    cv2_num_threads: Optional[int] = None
    torch_num_threads: Optional[int] = None
    # Threads for numpy's BLAS/LAPACK (OpenBLAS here). Macenko (eigh/percentile/lstsq) runs inside the
    # feature_prep_workers thread pool; concurrent calls into a multi-threaded pthreads OpenBLAS are a
    # suspected source of rare, non-reproducible single-patch differences between otherwise identical builds.
    blas_num_threads: Optional[int] = None
    # Run every feature batch at least twice (fresh host->GPU upload each time) and accept it only once two
    # passes agree bit for bit; also checksum the model weights around every pass and restore them from a host
    # copy if they changed. Guards against silent GPU memory/compute errors (the RTX A5000s run without ECC):
    # two otherwise identical builds each had exactly one corrupted patch embedding (2026-10-08). ~2x GPU time.
    feature_redundancy_check: bool = False
    feature_image_size: Optional[int] = None
    feature_use_cuda_if_available: bool = True
    feature_imagenet_normalize: bool = False
    feature_transforms: List[Dict[str, Any]] = field(default_factory=list)
    transform_seed: int = 42

@dataclass
class CreationConfig:
    annotation_type: AnnotationType = AnnotationType.GEOJSON
    mask_type: MaskType = MaskType.SLIDE
    use_annotations: bool = True
    use_precomputed_mask: bool = True
    use_cell_segmentation: bool = False
    node_creation_mode: NodeCreationMode = NodeCreationMode.PATCH_ENCODER
    cell_segmentation_level: Optional[int] = None
    cell_use_gpu_if_available: bool = True
    cellpose_model_type: str = "cyto3"
    cellpose_diameter: Optional[float] = None
    cellpose_channels: Tuple[int, int] = (0, 0)
    cellpose_flow_threshold: Optional[float] = None
    cellpose_cellprob_threshold: Optional[float] = None
    cell_min_area_px: int = 16
    cell_segmentation_tile_size: Optional[int] = 1024
    cell_segmentation_tile_overlap: int = 128
    cell_segmentation_border_margin: Optional[int] = None
    cell_k: Optional[int] = None
    patch_patch_edge_mode: EdgeConstructionMode = EdgeConstructionMode.KNN
    region_region_edge_mode: EdgeConstructionMode = EdgeConstructionMode.KNN
    cell_cell_edge_mode: EdgeConstructionMode = EdgeConstructionMode.KNN
    patch_patch_distance_radius_px: Optional[float] = None
    region_region_distance_radius_px: Optional[float] = None
    cell_cell_distance_radius_px: Optional[float] = None
    filter_graph_with_annotations: bool = False
    use_annotation_labels: bool = False
    annotation_label_source: str = "auto"
    annotation_classification_field: Optional[str] = None
    extra_annotation_classification_fields: List[str] = field(default_factory=list)
    label_fields: List[Dict[str, Any]] = field(default_factory=list)

    mask_path_override: Optional[Path] = None
    annotations_dir: Optional[Path] = None
    masks_dir: Optional[Path] = None
    build_slides_without_annotations: bool = False
    build_slides_with_mismatched_annotations: bool = False
    annotation_mismatch_match_rate_floor: float = 0.0
    annotation_mismatch_swap_factor: float = 5.0
    annotation_qa_sample_size: Optional[int] = None

    def __post_init__(self) -> None:
        """Normalize and validate all config fields after dataclass construction."""
        if not isinstance(self.annotation_type, AnnotationType):
            raw_annotation = str(self.annotation_type).strip()
            if "." in raw_annotation:
                raw_annotation = raw_annotation.split(".")[-1]
            normalized_annotation = raw_annotation.lower()
            by_value_annotation = {item.value.lower(): item for item in AnnotationType}
            by_name_annotation = {item.name.lower(): item for item in AnnotationType}
            resolved_annotation = by_value_annotation.get(normalized_annotation) or by_name_annotation.get(normalized_annotation)
            if resolved_annotation is None:
                allowed = ", ".join(item.value for item in AnnotationType)
                raise ValueError(f"annotation_type must be one of: {allowed}")
            self.annotation_type = resolved_annotation

        if not isinstance(self.mask_type, MaskType):
            raw_mask = str(self.mask_type).strip()
            if "." in raw_mask:
                raw_mask = raw_mask.split(".")[-1]
            normalized_mask = raw_mask.lower()
            by_value_mask = {item.value.lower(): item for item in MaskType}
            by_name_mask = {item.name.lower(): item for item in MaskType}
            resolved_mask = by_value_mask.get(normalized_mask) or by_name_mask.get(normalized_mask)
            if resolved_mask is None:
                allowed = ", ".join(item.value for item in MaskType)
                raise ValueError(f"mask_type must be one of: {allowed}")
            self.mask_type = resolved_mask

        normalized_source = str(self.annotation_label_source).strip().lower()
        allowed_sources = {"auto", "classification", "group", "name"}
        if normalized_source not in allowed_sources:
            allowed = ", ".join(sorted(allowed_sources))
            raise ValueError(f"annotation_label_source must be one of: {allowed}")
        self.annotation_label_source = normalized_source

        if not isinstance(self.node_creation_mode, NodeCreationMode):
            raw_mode = str(self.node_creation_mode).strip()

            if "." in raw_mode:
                raw_mode = raw_mode.split(".")[-1]

            normalized_mode = raw_mode.strip().lower()
            by_value = {mode.value.lower(): mode for mode in NodeCreationMode}
            by_name = {mode.name.lower(): mode for mode in NodeCreationMode}

            resolved = by_value.get(normalized_mode) or by_name.get(normalized_mode)
            if resolved is None:
                allowed = ", ".join(mode.value for mode in NodeCreationMode)
                raise ValueError(f"node_creation_mode must be one of: {allowed}")
            self.node_creation_mode = resolved

        def _normalize_edge_mode(raw_mode: Any, field_name: str) -> EdgeConstructionMode:
            if isinstance(raw_mode, EdgeConstructionMode):
                return raw_mode

            text = str(raw_mode).strip()
            if "." in text:
                text = text.split(".")[-1]
            normalized = text.lower()
            by_value = {mode.value.lower(): mode for mode in EdgeConstructionMode}
            by_name = {mode.name.lower(): mode for mode in EdgeConstructionMode}
            resolved_mode = by_value.get(normalized) or by_name.get(normalized)
            if resolved_mode is None:
                allowed = ", ".join(mode.value for mode in EdgeConstructionMode)
                raise ValueError(f"{field_name} must be one of: {allowed}")
            return resolved_mode

        self.patch_patch_edge_mode = _normalize_edge_mode(self.patch_patch_edge_mode, "patch_patch_edge_mode")
        self.region_region_edge_mode = _normalize_edge_mode(self.region_region_edge_mode, "region_region_edge_mode")
        self.cell_cell_edge_mode = _normalize_edge_mode(self.cell_cell_edge_mode, "cell_cell_edge_mode")

        if self.use_cell_segmentation and self.node_creation_mode == NodeCreationMode.PATCH_ENCODER:
            self.node_creation_mode = NodeCreationMode.CELL_SEGMENTATION

        self.use_cell_segmentation = self.node_creation_mode == NodeCreationMode.CELL_SEGMENTATION

        if len(self.cellpose_channels) != 2:
            raise ValueError("cellpose_channels must contain exactly two integers, e.g. (0, 0)")

        if self.cell_segmentation_level is not None and int(self.cell_segmentation_level) < 0:
            raise ValueError("cell_segmentation_level must be >= 0")

        if int(self.cell_min_area_px) < 1:
            raise ValueError("cell_min_area_px must be >= 1")

        if self.patch_patch_distance_radius_px is not None and float(self.patch_patch_distance_radius_px) <= 0:
            raise ValueError("patch_patch_distance_radius_px must be > 0 or None")

        if self.region_region_distance_radius_px is not None and float(self.region_region_distance_radius_px) <= 0:
            raise ValueError("region_region_distance_radius_px must be > 0 or None")

        if self.cell_cell_distance_radius_px is not None and float(self.cell_cell_distance_radius_px) <= 0:
            raise ValueError("cell_cell_distance_radius_px must be > 0 or None")

        if self.cell_segmentation_tile_size is not None and int(self.cell_segmentation_tile_size) < 64:
            raise ValueError("cell_segmentation_tile_size must be >= 64 or None")

        if int(self.cell_segmentation_tile_overlap) < 0:
            raise ValueError("cell_segmentation_tile_overlap must be >= 0")

        if self.cell_segmentation_border_margin is not None and int(self.cell_segmentation_border_margin) < 0:
            raise ValueError("cell_segmentation_border_margin must be >= 0 or None")

        if self.cell_segmentation_tile_size is not None and int(self.cell_segmentation_tile_overlap) >= int(self.cell_segmentation_tile_size):
            raise ValueError("cell_segmentation_tile_overlap must be smaller than cell_segmentation_tile_size")

        if (self.filter_graph_with_annotations or self.use_annotation_labels) and not self.use_annotations:
            raise ValueError(
                "Annotation-based filtering or labeling requires use_annotations=True"
            )

        if self.annotations_dir is not None:
            self.annotations_dir = Path(self.annotations_dir)

        if self.masks_dir is not None:
            self.masks_dir = Path(self.masks_dir)

        if self.mask_path_override is not None:
            self.mask_path_override = Path(self.mask_path_override)

        _valid_label_node_types = {"slide", "region", "patch"}
        _valid_label_sources = {"annotation_primary", "annotation_extra", "csv_join"}
        _valid_label_collapse = {None, "positive_negative"}
        seen_label_names: set = set()
        for entry in self.label_fields:
            if not isinstance(entry, dict):
                raise ValueError(f"label_fields entries must be dicts, got: {entry!r}")

            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"label_fields entry missing a non-empty 'name': {entry!r}")
            if name in seen_label_names:
                raise ValueError(f"label_fields has a duplicate name: {name!r}")
            seen_label_names.add(name)

            node_types = entry.get("node_types")
            if not isinstance(node_types, (list, tuple)) or not node_types:
                raise ValueError(f"label_fields entry '{name}' must set a non-empty 'node_types' list")
            if not set(node_types) <= _valid_label_node_types:
                allowed = ", ".join(sorted(_valid_label_node_types))
                raise ValueError(f"label_fields entry '{name}' node_types must be a subset of: {allowed}")

            source = entry.get("source")
            if source not in _valid_label_sources:
                allowed = ", ".join(sorted(_valid_label_sources))
                raise ValueError(f"label_fields entry '{name}' source must be one of: {allowed}")

            collapse = entry.get("collapse")
            if collapse not in _valid_label_collapse:
                raise ValueError(f"label_fields entry '{name}' collapse must be one of: null, 'positive_negative'")

            if source == "annotation_extra" and not entry.get("extra_field"):
                raise ValueError(f"label_fields entry '{name}' has source='annotation_extra' but no 'extra_field' set")

            if source == "csv_join":
                required = ("csv_path", "csv_key_column", "csv_value_column", "join_on")
                missing = [key for key in required if not entry.get(key)]
                if missing:
                    raise ValueError(f"label_fields entry '{name}' has source='csv_join' but is missing: {', '.join(missing)}")
                if entry.get("join_on") not in {"slide_stem", "patient_zip"}:
                    raise ValueError(f"label_fields entry '{name}' join_on must be one of: slide_stem, patient_zip")
            elif "slide" in node_types and entry.get("slide_aggregation", "majority_mixed") != "majority_mixed":
                raise ValueError(f"label_fields entry '{name}' slide_aggregation must be 'majority_mixed'")

@dataclass
class BuildContext:
    path_resolver: PathResolver
    dataset: DatasetType
    slide_path: str
    slide: object
    cfg: BuildConfig
    creation_cfg: CreationConfig
    annotations: object = None
    mask: object = None
    mask_downsized: object = None
    thumb_rgb: Optional[np.ndarray] = None
    mask_thumb: Optional[np.ndarray] = None
    mask_source_path: Optional[Path] = None
    mask_source_kind: Optional[str] = None
    patch_xy: Optional[np.ndarray] = None
    patch_frac: Optional[np.ndarray] = None
    patch_topleft: Optional[np.ndarray] = None
    region_labels: Optional[np.ndarray] = None
    region_xy: Optional[np.ndarray] = None
    region_annotation_labels: Optional[List[Optional[str]]] = None
    region_diagnostics: Optional[Dict[str, Any]] = None
    p2p_src: Optional[np.ndarray] = None
    p2p_dst: Optional[np.ndarray] = None
    p2p_dist: Optional[np.ndarray] = None
    r2r_src: Optional[np.ndarray] = None
    r2r_dst: Optional[np.ndarray] = None
    r2r_dist: Optional[np.ndarray] = None
    extra_relations: Optional[Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None
    node_features_cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    slide_mpp_um_x: Optional[float] = None
    slide_mpp_um_y: Optional[float] = None
    patch_level_mpp_um_x: Optional[float] = None
    patch_level_mpp_um_y: Optional[float] = None
    mpp_source: Optional[str] = None
    node_feature_model: Optional[Any] = None
    tissue_regions: Optional[List[Dict[str, Any]]] = None
    graph_data: Optional[Any] = None