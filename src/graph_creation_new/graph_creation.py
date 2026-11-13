from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Callable, Dict, Optional, Tuple, Union
import logging
import traceback

import numpy as np

from torch_geometric.data import HeteroData

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.graph_creation_new.config import BuildConfig, BuildContext, CreationConfig
from src.graph_creation_new.base import StepPipeline, GraphBuildPipeline
from src.paths.path_resolver import PathResolver
from src.paths.paths import get_project_paths
from src.graph_creation_new.enums import MaskType, AnnotationType, DatasetType, SlideType

from src.graph_creation_new.helper.annotations import load_asap_annotation, load_geojson_annotations, assess_annotation_quality
from src.graph_creation_new.error import AnnotationQualitySkipError
from src.graph_creation_new.helper.thumbnail import safe_thumbnail
from src.graph_creation_new.helper.mask import load_binary_mask, load_mask_from_path, generate_mask_from_thumbnail, refine_mask_with_config
from src.graph_creation_new.helper.slide import prepare_slide, generate_preview
from src.graph_creation_new.helper.subgraph import create_subgraph
from src.graph_creation_new.helper.patches import enumerate_patches
from src.graph_creation_new.helper.regions import build_regions
from src.graph_creation_new.visualization import debug_plot_region_cells


def _import_openslide():
    try:
        import openslide
    except Exception as exc:
        raise RuntimeError(
            "openslide-python konnte nicht importiert werden. Bitte OpenSlide + openslide-python installieren."
        ) from exc
    return openslide


def _candidate_mask_paths(context: BuildContext) -> Tuple[Path, ...]:
    slide_path = Path(context.slide_path)
    slide_stem = slide_path.stem
    dataset = context.dataset.value
    cfg = context.creation_cfg

    candidates = []

    if cfg.mask_path_override is not None:
        candidates.append(Path(cfg.mask_path_override))

    explicit_dir = cfg.masks_dir
    base_raw = context.path_resolver._get("RAW_DIR")

    dataset_aliases = [dataset]
    for alias in (dataset.replace("17", ""), dataset.replace("_17", ""), "camelyon"):
        alias = alias.strip()
        if alias and alias not in dataset_aliases:
            dataset_aliases.append(alias)

    search_dirs = []
    if explicit_dir is not None:
        search_dirs.append(Path(explicit_dir))
    for alias in dataset_aliases:
        search_dirs.append(base_raw / alias / "masks")
    search_dirs.append(base_raw / "masks")

    file_names = [
        f"{slide_stem}_mask.tif",
        f"{slide_stem}_mask.tiff",
        f"{slide_stem}_mask.png",
        f"{slide_stem}_mask.jpg",
        f"{slide_stem}_mask.jpeg",
    ]

    for search_dir in search_dirs:
        for file_name in file_names:
            candidates.append(search_dir / file_name)

        if search_dir.exists() and search_dir.is_dir():
            for pattern in (
                f"{slide_stem}_mask.*",
            ):
                for match in search_dir.glob(pattern):
                    if match.is_file():
                        candidates.append(match)

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(resolved)

    return tuple(unique_candidates)


def _discover_mask_path(context: BuildContext) -> Optional[Path]:
    for candidate in _candidate_mask_paths(context):
        if candidate.exists():
            return candidate
    return None


class DynamicalPipeline(StepPipeline):
    name = "dynamical"

    @property
    def steps(self) -> Tuple[Callable[[BuildContext], None], ...]:
        """Return the ordered sequence of build steps for this pipeline."""
        return (
            self._store_thumbnail,
            self._load_annoations,
            self._load_mask,
            self._prepare_slide,
            self._craft_patch,
            self._check_annotation_quality,
            self._craft_regions,
            self._create_subgraph
        )

    def _store_thumbnail(self, context: BuildContext) -> None:
        context.thumb_rgb = safe_thumbnail(context.slide, context.cfg.thumb_max_size)

    def _load_annoations(self, context: BuildContext) -> None:
        cfg = context.creation_cfg

        if not cfg.use_annotations:
            return

        annotation_path = context.path_resolver.annotation_path(
            Path(context.slide_path),
            context.dataset.value,
            cfg.annotation_type,
            annotation_dir=cfg.annotations_dir,
        )

        logging.info("Loading annotations for %s from %s", context.slide_path, annotation_path)

        match cfg.annotation_type:
            case AnnotationType.ASAP:
                context.annotations = load_asap_annotation(annotation_path)
            case AnnotationType.GEOJSON:
                context.annotations = load_geojson_annotations(annotation_path)
            case _:
                logging.warning(
                    f"Annotation creation not supported for type: {cfg.annotation_type}"
                )

        ann_count = len(context.annotations) if context.annotations is not None else 0
        if ann_count > 0:
            logging.info("Loaded %d annotations for %s", ann_count, context.slide_path)
        else:
            logging.info("No annotations available for %s", context.slide_path)

    def _load_mask(self, context: BuildContext) -> None:
        cfg = context.creation_cfg

        preferred_mask_path = None
        if cfg.use_precomputed_mask:
            preferred_mask_path = _discover_mask_path(context)
            if preferred_mask_path is None:
                preferred_mask_path = context.path_resolver.mask_path(
                    Path(context.slide_path),
                    context.dataset.value,
                    None,
                    cfg.mask_type,
                    mask_dir=cfg.masks_dir,
                )

        mask = None
        downsample = None

        if preferred_mask_path is not None:
            mask, downsample = load_mask_from_path(preferred_mask_path, slide_dimensions=context.slide.dimensions)
            if mask is not None and downsample is not None:
                logging.info("Loaded precomputed mask for %s from %s", context.slide_path, preferred_mask_path)
                context.mask_source_path = preferred_mask_path
                context.mask_source_kind = "precomputed"
            else:
                logging.warning(
                    "Mask path was found but unusable for %s: %s. Falling back to generated mask.",
                    context.slide_path,
                    preferred_mask_path,
                )

        if mask is None or downsample is None:
            logging.warning(
                "No usable precomputed mask for %s -> generating mask from WSI thumbnail",
                context.slide_path,
            )
            mask, downsample = generate_mask_from_thumbnail(context)
            if mask is not None and downsample is not None:
                context.mask_source_path = None
                context.mask_source_kind = "generated"

        if mask is not None:
            mask = refine_mask_with_config(mask.astype(np.uint8), context.cfg).astype(bool)

        context.mask = mask if mask is not None else None
        context.mask_downsized = downsample if downsample is not None else None

        if context.mask_source_kind is None and context.mask is not None:
            context.mask_source_kind = "unknown"

        if context.mask is None or context.mask_downsized is None:
            match cfg.mask_type:
                case MaskType.IMAGE:
                    print(f"Image mask not supported for type: {cfg.mask_type.value}")
                case _:
                    logging.warning(
                        f"Mask creation not supported for type: {cfg.mask_type.value}"
                    )

        if context.mask is None or context.mask_downsized is None:
            logging.warning("Mask still unavailable after fallback -> graph building may fail for %s", context.slide_path)

    def _prepare_slide(self, context: BuildContext) -> None:
        prepare_slide(context)
        generate_preview(context)

    def _craft_patch(self, context: BuildContext) -> None:
        enumerate_patches(context=context)

    def _check_annotation_quality(self, context: BuildContext) -> None:
        cfg = context.creation_cfg
        if not cfg.use_annotations:
            return

        report = assess_annotation_quality(
            context.annotations,
            context.patch_xy,
            downsample=float(context.slide.level_downsamples[int(context.cfg.patch_level)]),
            sample_size=cfg.annotation_qa_sample_size,
            match_rate_floor=cfg.annotation_mismatch_match_rate_floor,
            swap_factor=cfg.annotation_mismatch_swap_factor,
        )
        reason = report["reason"]

        skip = (
            (reason == "no_annotations" and not cfg.build_slides_without_annotations)
            or (reason in ("mismatched_xy_swap", "mismatched_unknown") and not cfg.build_slides_with_mismatched_annotations)
        )
        if not skip:
            if reason != "ok":
                logging.info(
                    "Annotation QA for %s: reason=%s (building anyway -- policy flag allows it) "
                    "match_rate_normal=%.4f match_rate_swapped=%.4f",
                    context.slide_path, reason, report["match_rate_normal"], report["match_rate_swapped"],
                )
            return

        slide_name = Path(context.slide_path).stem
        logging.warning(
            "Annotation QA skip for %s: reason=%s n_annotations=%d match_rate_normal=%.4f "
            "match_rate_swapped=%.4f -- set CreationConfig.build_slides_without_annotations/"
            "build_slides_with_mismatched_annotations to build it anyway.",
            slide_name, reason, report["n_annotations"], report["match_rate_normal"], report["match_rate_swapped"],
        )
        raise AnnotationQualitySkipError(
            reason=reason,
            slide_name=slide_name,
            n_annotations=report["n_annotations"],
            n_patches=report["n_patches"],
            match_rate_normal=report["match_rate_normal"],
            match_rate_swapped=report["match_rate_swapped"],
        )

    def _craft_regions(self, context: BuildContext) -> None:
        build_regions(context=context)

    def _create_subgraph(self, context: BuildContext) -> None:
        create_subgraph(context=context)

    def _create_embedding(self, context: BuildContext) -> None:
        pass

    def _create_edge(self, context: BuildContext) -> None:
        pass

    def finalize(self, context: BuildContext) -> HeteroData:
        """Return the built graph, or an empty HeteroData if none was produced."""
        if context.graph_data is None:
            return HeteroData()
        return context.graph_data

class TumorFocusPipeline(DynamicalPipeline):
    """Annotation-filtered, fine-grained variant of DynamicalPipeline.

    Identical step sequence to DynamicalPipeline. The behaviour difference
    (patch filtering to annotation polygons, per-patch/region annotation labels,
    finer patch_size/stride) is driven entirely by CreationConfig and BuildConfig
    fields set in tumor_focus_pipeline.yaml.  A separate named class is required
    so that cfg.pipeline='tumor_focus' resolves correctly and the output filename
    carries the pipeline identifier.
    """

    name = "tumor_focus"


PIPELINES: Dict[str, GraphBuildPipeline] = {
    DynamicalPipeline.name: DynamicalPipeline(),
    TumorFocusPipeline.name: TumorFocusPipeline(),
}


def register_pipeline(name: str, pipeline: GraphBuildPipeline) -> None:
    """Register a pipeline instance under the given name."""
    PIPELINES[name] = pipeline


def available_pipelines() -> Tuple[str, ...]:
    """Return the names of all registered pipelines, sorted."""
    return tuple(sorted(PIPELINES.keys()))


def _resolve_pipeline(cfg: BuildConfig) -> GraphBuildPipeline:
    if cfg.pipeline not in PIPELINES:
        available = ", ".join(sorted(PIPELINES.keys()))
        raise ValueError(f"Unknown graph pipeline '{cfg.pipeline}'. Available: {available}")
    return PIPELINES[cfg.pipeline]


def build_graph_from_slide(
    slide: Union[str, Path],
    cfg: BuildConfig,
    node_feature_model: Optional[Any] = None,
    creation_cfg: Optional[CreationConfig] = None,
    dataset: DatasetType = DatasetType.CAMELYON17,
) -> HeteroData:
    """Build a graph for a single slide using the pipeline resolved from cfg.pipeline."""
    openslide = _import_openslide()
    slide_path = str(slide)
    logging.info(f"Building graph for slide: {slide_path}")
    handle = openslide.OpenSlide(slide_path)

    try:
        context = BuildContext(
            path_resolver=PathResolver(get_project_paths()),
            dataset=dataset,
            slide_path=slide_path,
            slide=handle,
            cfg=cfg,
            node_feature_model=node_feature_model,
            creation_cfg=creation_cfg if creation_cfg is not None else CreationConfig())
        pipeline = _resolve_pipeline(cfg)
        return pipeline.build(context)
    finally:
        handle.close()


def build_graph_from_mrxs(
    wsi_path: Union[str, Path],
    cfg: BuildConfig,
    node_feature_model: Optional[Any] = None,
    dataset: DatasetType = DatasetType.CAMELYON17,
) -> HeteroData:
    """Build a graph for a single MRXS slide."""
    return build_graph_from_slide(wsi_path, cfg, node_feature_model=node_feature_model, dataset=dataset)


def run_pipeline_with_visualization(
    slide: Union[str, Path],
    pipeline_name: str,
    out_dir: Union[str, Path],
    node_feature_model: Optional[Any] = None,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    viz_cfg: Optional[Any] = None,
    creation_cfg: Optional[CreationConfig] = None,
    dataset: DatasetType = DatasetType.CAMELYON17,
) -> Dict[str, Any]:
    """Build a graph for a slide and generate all validation plots for it."""
    cfg_kwargs: Dict[str, Any] = {"pipeline": pipeline_name}
    if cfg_overrides:
        cfg_kwargs.update(cfg_overrides)

    cfg = BuildConfig(**cfg_kwargs)
    data = build_graph_from_slide(
        slide=slide,
        cfg=cfg,
        node_feature_model=node_feature_model,
        creation_cfg=creation_cfg,
        dataset=dataset,
    )

    from src.graph_creation_new.visualization import VisualizationConfig, visualize_graph_structure

    resolved_viz_cfg = viz_cfg if viz_cfg is not None else VisualizationConfig()
    output_dir = Path(out_dir)
    summary = visualize_graph_structure(
        data=data,
        out_dir=output_dir,
        slide_path=slide,
        build_cfg=cfg,
        viz_cfg=resolved_viz_cfg,
    )

    return {
        "data": data,
        "summary": summary,
        "out_dir": str(output_dir),
        "overview_path": str(output_dir / "overview.png"),
        "hierarchical_graph_path": str(output_dir / "hierarchical_graph.png"),
    }


def run_pipeline_for_slides(
    slides: list[Union[str, Path]],
    pipeline_name: str,
    out_root: Union[str, Path],
    node_feature_model: Optional[Any] = None,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    viz_cfg: Optional[Any] = None,
    creation_cfg: Optional[CreationConfig] = None,
    dataset: DatasetType = DatasetType.CAMELYON17,
    *,
    continue_on_error: bool = True,
) -> list[Dict[str, Any]]:
    """Build and visualize graphs for multiple slide paths, each into its own output folder."""
    output_root = Path(out_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[Dict[str, Any]] = []
    total = int(len(slides))

    for index, slide in enumerate(slides, start=1):
        slide_path = Path(str(slide)).expanduser()
        if not slide_path.is_absolute():
            slide_path = (Path.cwd() / slide_path).resolve()
        else:
            slide_path = slide_path.resolve()

        slide_stem = slide_path.stem or f"slide_{index:03d}"
        slide_out_dir = output_root / f"{index:03d}_{slide_stem}"
        slide_out_dir.mkdir(parents=True, exist_ok=True)

        logging.info("[%d/%d] Running graph pipeline for slide: %s", index, total, slide_path)

        try:
            payload = run_pipeline_with_visualization(
                slide=slide_path,
                pipeline_name=pipeline_name,
                out_dir=slide_out_dir,
                node_feature_model=node_feature_model,
                cfg_overrides=cfg_overrides,
                viz_cfg=viz_cfg,
                creation_cfg=creation_cfg,
                dataset=dataset,
            )
            summary = payload.get("summary") or {}
            results.append(
                {
                    "slide": str(slide_path),
                    "status": "ok",
                    "out_dir": str(slide_out_dir),
                    "num_patches": int(summary.get("num_patches", 0)),
                    "num_regions": int(summary.get("num_regions", 0)),
                    "num_patch_edges": int(summary.get("num_patch_edges", 0)),
                    "num_region_edges": int(summary.get("num_region_edges", 0)),
                    "overview_path": payload.get("overview_path"),
                    "hierarchical_graph_path": payload.get("hierarchical_graph_path"),
                }
            )
        except Exception as exc:
            tb = traceback.format_exc()
            logging.exception("Failed to process slide %s", slide_path)
            results.append(
                {
                    "slide": str(slide_path),
                    "status": "error",
                    "out_dir": str(slide_out_dir),
                    "error": str(exc),
                    "traceback": tb,
                }
            )
            if not continue_on_error:
                raise

    status_values = np.asarray([entry.get("status", "") for entry in results], dtype=object)
    n_ok = int(np.sum(status_values == "ok"))
    n_err = int(np.sum(status_values == "error"))
    logging.info(
        "Batch graph pipeline finished: ok=%d error=%d total=%d out_root=%s",
        n_ok,
        n_err,
        total,
        output_root,
    )

    return results


__all__ = [
    "BuildConfig",
    "BuildContext",
    "GraphBuildPipeline",
    "StepPipeline",
    "DynamicalPipeline",
    "register_pipeline",
    "available_pipelines",
    "build_graph_from_slide",
    "debug_plot_region_cells",
    "build_graph_from_mrxs",
    "run_pipeline_with_visualization",
    "run_pipeline_for_slides",
]
