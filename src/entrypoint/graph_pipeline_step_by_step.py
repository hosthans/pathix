from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import yaml

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.graph_creation_new.config import BuildConfig, BuildContext, CreationConfig
from src.graph_creation_new.enums import DatasetType
from src.graph_creation_new.graph_creation import DynamicalPipeline
from src.graph_creation_new.helper import mask as mask_module
from src.graph_creation_new.helper.annotations import (
    annotation_contains_point,
    annotation_label,
    build_annotation_label_vocab,
    load_asap_annotation,
    load_geojson_annotations,
)
from src.paths.path_resolver import PathResolver
from src.paths.paths import get_project_paths
from src.utils.logging import tee_stdout_to_file


AVAILABLE_PIPELINE_CONFIGS = [
    "simple_pipeline.yaml",
    "context_aware_pipeline.yaml",
    "context_aware_robust_pipeline.yaml",
    "context_aware_sensitivity_pipeline.yaml",
    "tumor_focus_pipeline.yaml",
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run graph_creation_new pipeline step-by-step and export all notebook-style plots."
    )
    parser.add_argument("--slide-path", type=Path, required=True, help="Path to WSI slide file")
    parser.add_argument(
        "--pipeline-config",
        type=str,
        default="context_aware_pipeline.yaml",
        help="Pipeline config name in configs/graph_build or absolute YAML path",
    )
    parser.add_argument("--annotation-file", type=Path, default=None, help="Optional annotation file (.xml/.json/.geojson)")
    parser.add_argument("--mask-file", type=Path, default=None, help="Optional mask file override")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for plots/tables")
    parser.add_argument(
        "--annotation-label-source",
        type=str,
        default=None,
        help="Optional override for creation config annotation_label_source",
    )
    parser.add_argument("--show-plots", action="store_true", help="Show plots interactively in addition to saving")
    parser.add_argument("--skip-extras", action="store_true", help="Skip extra sections (Zusatz 1-3)")
    parser.add_argument("--auto-close-slide", action="store_true", help="Close openslide handle at end")
    return parser.parse_args()


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_with_extends(config_path: Path, visited: set[Path] | None = None) -> dict:
    if visited is None:
        visited = set()

    resolved = config_path.expanduser().resolve()
    if resolved in visited:
        chain = " -> ".join(str(p) for p in list(visited) + [resolved])
        raise ValueError(f"Circular config extends detected: {chain}")

    visited.add(resolved)
    with resolved.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a mapping at root: {resolved}")

    extends_value = raw.get("extends")
    merged = {}
    if extends_value is not None:
        extends_list = extends_value if isinstance(extends_value, list) else [extends_value]
        for item in extends_list:
            parent_path = Path(str(item))
            if not parent_path.is_absolute():
                parent_path = resolved.parent / parent_path
            parent_cfg = _load_yaml_with_extends(parent_path, visited=visited)
            merged = _deep_merge_dicts(merged, parent_cfg)

    current_cfg = {k: v for k, v in raw.items() if k != "extends"}
    merged = _deep_merge_dicts(merged, current_cfg)
    visited.remove(resolved)
    return merged


def _open_slide(path: Path):
    import openslide  # type: ignore

    return openslide.OpenSlide(str(path))


def _ensure_slide_open(context: BuildContext, slide_path: Path):
    try:
        _ = context.slide.level_count
        return context.slide
    except Exception:
        context.slide = _open_slide(slide_path)
        return context.slide


def _level_xy_to_thumb(xy_level: np.ndarray, context: BuildContext, slide_path: Path) -> np.ndarray:
    if xy_level.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    slide = _ensure_slide_open(context, slide_path)
    downsample = float(slide.level_downsamples[int(context.cfg.patch_level)])
    xy_l0 = xy_level.astype(np.float32) * downsample
    thumb_h, thumb_w = context.thumb_rgb.shape[:2]
    full_w, full_h = slide.dimensions
    scale_x = thumb_w / float(full_w)
    scale_y = thumb_h / float(full_h)
    out = np.zeros_like(xy_l0, dtype=np.float32)
    out[:, 0] = xy_l0[:, 0] * scale_x
    out[:, 1] = xy_l0[:, 1] * scale_y
    return out


def _mask_to_thumb(mask: np.ndarray, context: BuildContext) -> np.ndarray:
    thumb_h, thumb_w = context.thumb_rgb.shape[:2]
    return cv2.resize(mask.astype(np.uint8), (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST).astype(bool)


def _save_fig(fig, out_dir: Path, name: str, show_plots: bool) -> None:
    out_path = out_dir / name
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"saved: {out_path}")
    if show_plots:
        plt.show()
    plt.close(fig)


def _show_big(title: str, img: np.ndarray, out_dir: Path, name: str, show_plots: bool, cmap=None) -> None:
    fig = plt.figure(figsize=(11, 9))
    if cmap is None:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap)
    plt.title(title, fontsize=14)
    plt.axis("off")
    _save_fig(fig, out_dir, name, show_plots)


def _load_annotations_into_context(
    context: BuildContext,
    pipeline: DynamicalPipeline,
    annotation_file: Path | None,
) -> list[dict]:
    if annotation_file is None:
        pipeline.steps[1](context)
        return context.annotations or []

    annotation_file = Path(annotation_file).expanduser().resolve()
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    suffix = annotation_file.suffix.lower()
    if suffix == ".xml":
        annotations = load_asap_annotation(annotation_file)
    elif suffix in {".json", ".geojson"}:
        annotations = load_geojson_annotations(annotation_file)
    else:
        raise ValueError(f"Unsupported annotation file suffix: {suffix}")

    context.annotations = annotations
    return annotations


def _annotation_palette(labels: list[str]) -> dict[str, tuple[float, float, float]]:
    cmap = plt.get_cmap("tab20")
    palette = {}
    for idx, label in enumerate(labels):
        palette[label] = cmap(idx % 20)[:3]
    return palette


def _region_annotation_labels(context: BuildContext, slide_path: Path, source: str = "group") -> tuple[np.ndarray, list[str]]:
    if context.region_xy is None:
        return np.zeros((0,), dtype=np.int64), []
    annotations = context.annotations or []
    if not annotations:
        return np.full((context.region_xy.shape[0],), -1, dtype=np.int64), []

    slide = _ensure_slide_open(context, slide_path)
    vocab = build_annotation_label_vocab(annotations, source=source)
    label_to_id = {label: idx for idx, label in enumerate(vocab)}
    region_ids = np.full((context.region_xy.shape[0],), -1, dtype=np.int64)

    region_xy_l0 = context.region_xy.astype(np.float32) * float(slide.level_downsamples[int(context.cfg.patch_level)])
    for region_idx, center in enumerate(region_xy_l0):
        matched = None
        for ann in annotations:
            if annotation_contains_point(ann, (float(center[0]), float(center[1]))):
                matched = annotation_label(ann, source=source)
                if matched is not None:
                    break
        if matched is not None and matched in label_to_id:
            region_ids[region_idx] = int(label_to_id[matched])

    return region_ids, vocab


def _resolve_graph_build_config(project_root: Path, pipeline_config: str) -> Path:
    token = Path(str(pipeline_config))
    if token.is_absolute():
        path = token.resolve()
    else:
        path = (project_root / "configs" / "graph_build" / token).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")
    return path


def _save_df(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    path = out_dir / name
    df.to_csv(path, index=False)
    print(f"saved: {path}")


def _run(args: argparse.Namespace) -> None:
    """Run the graph creation pipeline step-by-step, saving plots and tables for each stage."""
    slide_path = Path(args.slide_path).expanduser().resolve()
    if not slide_path.exists():
        raise FileNotFoundError(f"Slide not found: {slide_path}")

    annotation_file = Path(args.annotation_file).expanduser().resolve() if args.annotation_file is not None else None
    mask_file = Path(args.mask_file).expanduser().resolve() if args.mask_file is not None else None

    if annotation_file is not None and not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    if mask_file is not None and not mask_file.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_file}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir is not None else (project_root / "results" / "graph_pipeline_step_by_step" / slide_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    # This script uses plain print() throughout rather than `logging` -- tee_stdout_to_file
    # redirects stdout/stderr so those prints also land in a log file colocated with the numbered
    # per-stage artifacts it writes into out_dir, instead of only ever reaching the console.
    tee_stdout_to_file(out_dir, operation="graph_pipeline_step_by_step")

    print("PROJECT_ROOT =", project_root)
    print("SLIDE_PATH =", slide_path)
    print("PIPELINE_CONFIG =", args.pipeline_config)
    print("AVAILABLE_PIPELINE_CONFIGS =", AVAILABLE_PIPELINE_CONFIGS)
    print("ANNOTATION_FILE =", annotation_file)
    print("MASK_FILE =", mask_file)
    print("OUT_DIR =", out_dir)

    graph_build_config_path = _resolve_graph_build_config(project_root, args.pipeline_config)
    graph_build_yaml = _load_yaml_with_extends(graph_build_config_path)
    graph_build_payload = (graph_build_yaml.get("graph_build") or graph_build_yaml) if isinstance(graph_build_yaml, dict) else {}
    build_cfg_dict = dict((graph_build_payload.get("build") or {}))
    creation_cfg_dict = dict((graph_build_payload.get("creation") or {}))

    if mask_file is not None:
        creation_cfg_dict["mask_path_override"] = str(mask_file)
        creation_cfg_dict["use_precomputed_mask"] = True
    if args.annotation_label_source is not None:
        creation_cfg_dict["annotation_label_source"] = str(args.annotation_label_source)

    build_cfg = BuildConfig(**build_cfg_dict)
    creation_cfg = CreationConfig(**creation_cfg_dict)

    _save_df(pd.DataFrame([asdict(build_cfg)]), out_dir, "build_config.csv")
    _save_df(pd.DataFrame([asdict(creation_cfg)]), out_dir, "creation_config.csv")

    slide_handle = _open_slide(slide_path)
    context = BuildContext(
        path_resolver=PathResolver(get_project_paths()),
        dataset=DatasetType.CAMELYON17,
        slide_path=str(slide_path),
        slide=slide_handle,
        cfg=build_cfg,
        creation_cfg=creation_cfg,
    )
    pipeline = DynamicalPipeline()

    steps_df = pd.DataFrame({"step_idx": np.arange(1, len(pipeline.steps) + 1), "step": [fn.__name__ for fn in pipeline.steps]})
    _save_df(steps_df, out_dir, "pipeline_steps.csv")

    pipeline.steps[0](context)
    _show_big("Step 1 - Thumbnail", context.thumb_rgb, out_dir, "step1_thumbnail.png", args.show_plots)

    annotations = _load_annotations_into_context(context, pipeline, annotation_file)
    print("Loaded annotations:", len(annotations))
    print("Annotation source:", annotation_file if annotation_file is not None else "Pipeline resolver")

    slide = _ensure_slide_open(context, slide_path)
    thumb = context.thumb_rgb.copy()
    thumb_h, thumb_w = thumb.shape[:2]
    full_w, full_h = slide.dimensions
    sx = thumb_w / float(full_w)
    sy = thumb_h / float(full_h)

    annotation_labels = []
    for ann in annotations:
        label = annotation_label(ann, source=context.creation_cfg.annotation_label_source)
        if label is not None:
            annotation_labels.append(label)
    annotation_labels = sorted(set(annotation_labels))
    palette = _annotation_palette(annotation_labels)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(thumb)
    for ann in annotations[:400]:
        coords = ann.get("coordinates") or []
        if len(coords) < 3:
            continue
        poly = np.asarray(coords, dtype=np.float32)
        x = poly[:, 0] * sx
        y = poly[:, 1] * sy
        label = annotation_label(ann, source=context.creation_cfg.annotation_label_source)
        color = palette.get(label, (1.0, 0.0, 0.0))
        ax.plot(x, y, linewidth=1.4, color=color)

    ax.set_title(f"Step 2 - Annotationen (n={len(annotations)})", fontsize=14)
    ax.axis("off")
    _save_fig(fig, out_dir, "step2_annotations.png", args.show_plots)

    if annotations:
        ann_df = pd.DataFrame(
            [
                {
                    "label": annotation_label(a, source=context.creation_cfg.annotation_label_source),
                    "group": a.get("group"),
                    "name": a.get("name"),
                    "num_points": len(a.get("coordinates") or []),
                }
                for a in annotations[:200]
            ]
        )
        _save_df(ann_df, out_dir, "step2_annotations_overview.csv")

    pipeline.steps[2](context)
    if context.mask is None:
        raise RuntimeError("Maske wurde nicht erzeugt/geladen.")

    print("Mask source kind:", getattr(context, "mask_source_kind", None))
    print("Mask source path:", getattr(context, "mask_source_path", None))

    mask_thumb = _mask_to_thumb(np.asarray(context.mask, dtype=bool), context)
    overlay = context.thumb_rgb.copy()
    overlay[~mask_thumb] = (overlay[~mask_thumb] * 0.15).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(mask_thumb, cmap="gray")
    ax.set_title("Step 3 - Geladene Maske (Basisbild)")
    ax.axis("off")
    _save_fig(fig, out_dir, "step3_mask_loaded_base.png", args.show_plots)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(mask_thumb, cmap="gray")
    axes[0].set_title("Step 3 - Tissue Maske")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("Step 3 - Mask Overlay")
    axes[1].axis("off")
    _save_fig(fig, out_dir, "step3_mask_overlay.png", args.show_plots)

    stored_mask = None
    stored_mask_path = getattr(context, "mask_source_path", None)
    if stored_mask_path is not None and Path(stored_mask_path).exists():
        stored_mask, _ = mask_module.load_mask_from_path(Path(stored_mask_path), slide_dimensions=context.slide.dimensions)
        if stored_mask is not None:
            stored_mask = mask_module.refine_mask_with_config(stored_mask.astype(np.uint8), context.cfg).astype(bool)

    computed_mask, _ = mask_module.generate_mask_from_thumbnail(context)
    if computed_mask is not None:
        computed_mask = mask_module.refine_mask_with_config(computed_mask.astype(np.uint8), context.cfg).astype(bool)

    used_mask = np.asarray(context.mask, dtype=bool) if context.mask is not None else None

    if stored_mask is not None:
        stored_thumb = _mask_to_thumb(stored_mask, context)
        computed_thumb = _mask_to_thumb(computed_mask, context) if computed_mask is not None else None
        used_thumb = _mask_to_thumb(used_mask, context) if used_mask is not None else None

        fig, axes = plt.subplots(1, 4, figsize=(22, 8))
        axes[0].imshow(stored_thumb, cmap="gray")
        axes[0].set_title("Gespeicherte Maske")
        axes[0].axis("off")

        if computed_thumb is not None:
            axes[1].imshow(computed_thumb, cmap="gray")
        axes[1].set_title("Berechnete Maske")
        axes[1].axis("off")

        if used_thumb is not None:
            axes[2].imshow(used_thumb, cmap="gray")
        axes[2].set_title("In der Pipeline verwendet")
        axes[2].axis("off")

        diff = np.zeros_like(stored_thumb, dtype=np.uint8)
        if computed_thumb is not None:
            diff = np.logical_xor(stored_thumb, computed_thumb).astype(np.uint8)
        axes[3].imshow(diff, cmap="magma")
        axes[3].set_title("Differenz gespeichert vs berechnet")
        axes[3].axis("off")
        _save_fig(fig, out_dir, "step3b_mask_compare.png", args.show_plots)

        if computed_thumb is not None:
            compare_df = pd.DataFrame(
                [
                    {
                        "stored_mask_fraction_thumb": float(np.mean(stored_thumb)),
                        "computed_mask_fraction_thumb": float(np.mean(computed_thumb)),
                        "overlap_fraction_thumb": float(np.mean(np.logical_and(stored_thumb, computed_thumb))),
                        "difference_fraction_thumb": float(np.mean(np.logical_xor(stored_thumb, computed_thumb))),
                        "used_equals_stored": bool(np.array_equal(used_thumb, stored_thumb)) if used_thumb is not None else False,
                        "used_equals_computed": bool(np.array_equal(used_thumb, computed_thumb)) if used_thumb is not None else False,
                    }
                ]
            )
            _save_df(compare_df, out_dir, "step3b_mask_compare_stats.csv")

    pipeline.steps[3](context)
    regions = context.tissue_regions or []
    print("Extracted tissue regions:", len(regions))

    thumb = context.thumb_rgb.copy()
    thumb_h, thumb_w = thumb.shape[:2]
    full_w, full_h = context.slide.dimensions
    sx = thumb_w / float(full_w)
    sy = thumb_h / float(full_h)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(thumb)
    for r in regions[:200]:
        x0, y0, x1, y1 = r["bbox_level0_xyxy"]
        rx = x0 * sx
        ry = y0 * sy
        rw = max(1.0, (x1 - x0 + 1) * sx)
        rh = max(1.0, (y1 - y0 + 1) * sy)
        rect = Rectangle((rx, ry), rw, rh, fill=False, linewidth=1.2)
        ax.add_patch(rect)
    ax.set_title(f"Step 4 - Tissue Regionen (n={len(regions)})", fontsize=14)
    ax.axis("off")
    _save_fig(fig, out_dir, "step4_tissue_regions.png", args.show_plots)

    pipeline.steps[4](context)
    if context.patch_xy is None or context.patch_xy.size == 0:
        raise RuntimeError("Keine Patches erzeugt.")

    patch_xy_thumb = _level_xy_to_thumb(context.patch_xy, context, slide_path)
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(context.thumb_rgb)
    ax.scatter(patch_xy_thumb[:, 0], patch_xy_thumb[:, 1], s=2, alpha=0.65)
    ax.set_title(f"Step 5 - Patches (n={patch_xy_thumb.shape[0]})", fontsize=14)
    ax.axis("off")
    _save_fig(fig, out_dir, "step5_patches.png", args.show_plots)

    pipeline.steps[5](context)
    if context.region_labels is None or context.region_xy is None:
        raise RuntimeError("Region-Clustering fehlt.")

    patch_xy_thumb = _level_xy_to_thumb(context.patch_xy, context, slide_path)
    region_xy_thumb = _level_xy_to_thumb(context.region_xy, context, slide_path)
    cluster_ids = context.region_labels.astype(np.int64)
    region_annotation_ids, region_vocab = _region_annotation_labels(context, slide_path, source=context.creation_cfg.annotation_label_source)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(context.thumb_rgb)
    ax.scatter(patch_xy_thumb[:, 0], patch_xy_thumb[:, 1], c=cluster_ids, s=3, cmap="tab20", alpha=0.75)
    ax.scatter(region_xy_thumb[:, 0], region_xy_thumb[:, 1], c="black", s=45, marker="x")

    for region_idx, (x, y) in enumerate(region_xy_thumb):
        label_text = f"R{region_idx}"
        ann_id = int(region_annotation_ids[region_idx]) if region_annotation_ids.size > region_idx else -1
        if ann_id >= 0 and ann_id < len(region_vocab):
            short_label = region_vocab[ann_id].replace("group:", "").replace("name:", "").replace("classification:", "")
            label_text += f"\\n{short_label[:18]}"
        ax.text(x + 3, y + 3, label_text, fontsize=7, color="black", bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=0.6))

    ax.set_title(f"Step 6 - Clustering: Patches -> Regionen (k={region_xy_thumb.shape[0]})", fontsize=14)
    ax.axis("off")
    _save_fig(fig, out_dir, "step6_clustering.png", args.show_plots)

    cluster_sizes = pd.Series(cluster_ids).value_counts().sort_index()
    cluster_df = pd.DataFrame(
        {
            "region_id": np.arange(region_xy_thumb.shape[0], dtype=int),
            "num_patches": [int(cluster_sizes.get(idx, 0)) for idx in range(region_xy_thumb.shape[0])],
            "annotation_label": [
                region_vocab[ann_id] if ann_id >= 0 and ann_id < len(region_vocab) else None
                for ann_id in region_annotation_ids.tolist()
            ],
        }
    )
    _save_df(cluster_df, out_dir, "step6_cluster_sizes.csv")

    fig = plt.figure(figsize=(10, 4))
    plt.bar(cluster_df["region_id"].astype(str), cluster_df["num_patches"])
    plt.xticks(rotation=90)
    plt.title("Cluster-Groesse je Region")
    plt.ylabel("Patches")
    plt.tight_layout()
    _save_fig(fig, out_dir, "step6_cluster_sizes.png", args.show_plots)

    region_ids = np.arange(int(context.region_xy.shape[0]), dtype=int)
    region_sizes = pd.Series(context.region_labels.astype(np.int64)).value_counts().sort_index()
    region_labels_from_context = getattr(context, "region_annotation_labels", None)

    if region_labels_from_context is None or len(region_labels_from_context) != len(region_ids):
        ann_ids, ann_vocab = _region_annotation_labels(context, slide_path, source=context.creation_cfg.annotation_label_source)
        region_labels_from_context = [ann_vocab[idx] if idx >= 0 and idx < len(ann_vocab) else None for idx in ann_ids.tolist()]

    region_control_df = pd.DataFrame(
        {
            "region_id": region_ids,
            "annotation_label": region_labels_from_context,
            "num_patches": [int(region_sizes.get(rid, 0)) for rid in region_ids],
        }
    )
    region_control_df["is_positive_region"] = region_control_df["annotation_label"].fillna("").str.contains("tumor", case=False)
    region_control_df["label_source"] = "context.region_annotation_labels"
    _save_df(region_control_df, out_dir, "step6b_region_control_table.csv")

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_df = region_control_df.sort_values("num_patches", ascending=False).reset_index(drop=True)
    colors = ["#e63946" if flag else "#457b9d" for flag in plot_df["is_positive_region"].tolist()]
    ax.bar(plot_df["region_id"].astype(str), plot_df["num_patches"], color=colors)
    ax.set_title("Regiongroessen (rot = Tumor-Region)")
    ax.set_xlabel("region_id")
    ax.set_ylabel("num_patches")
    plt.xticks(rotation=90)
    plt.tight_layout()
    _save_fig(fig, out_dir, "step6b_region_control_table_plot.png", args.show_plots)

    pipeline.steps[6](context)
    data = context.graph_data
    if data is None:
        raise RuntimeError("Graph data wurde nicht erzeugt.")

    annotation_vocab = data.meta.get("annotation_label_vocab", []) if hasattr(data, "meta") else []
    if hasattr(data["region"], "annotation_y"):
        region_ann = data["region"].annotation_y.cpu().numpy()
    else:
        region_ann = np.full((int(data["region"].num_nodes),), -1, dtype=np.int64)

    summary_df = pd.DataFrame(
        [
            {
                "patch_nodes": int(data["patch"].num_nodes) if "patch" in data.node_types else 0,
                "region_nodes": int(data["region"].num_nodes) if "region" in data.node_types else 0,
                "slide_nodes": int(data["slide"].num_nodes) if "slide" in data.node_types else 0,
                "patch_edges": int(data["patch", "near", "patch"].edge_index.shape[1]) if ("patch", "near", "patch") in data.edge_types else 0,
                "region_edges": int(data["region", "near", "region"].edge_index.shape[1]) if ("region", "near", "region") in data.edge_types else 0,
                "annotation_vocab_size": int(len(annotation_vocab)),
            }
        ]
    )
    _save_df(summary_df, out_dir, "step7_graph_summary.csv")

    region_label_df = pd.DataFrame(
        {
            "region_id": np.arange(region_ann.shape[0], dtype=int),
            "annotation_y": region_ann,
            "annotation_label": [
                annotation_vocab[idx] if idx >= 0 and idx < len(annotation_vocab) else None for idx in region_ann.tolist()
            ],
        }
    )
    _save_df(region_label_df, out_dir, "step7_region_labels.csv")

    patch_xy_thumb = _level_xy_to_thumb(context.patch_xy, context, slide_path)
    region_xy_thumb = _level_xy_to_thumb(context.region_xy, context, slide_path)

    p2p = data["patch", "near", "patch"].edge_index.cpu().numpy() if ("patch", "near", "patch") in data.edge_types else np.zeros((2, 0), dtype=np.int64)
    r2r = data["region", "near", "region"].edge_index.cpu().numpy() if ("region", "near", "region") in data.edge_types else np.zeros((2, 0), dtype=np.int64)
    p2r = data["patch", "in", "region"].edge_index.cpu().numpy() if ("patch", "in", "region") in data.edge_types else np.zeros((2, 0), dtype=np.int64)

    rng = np.random.default_rng(42)

    def _sample_edges(edge_index: np.ndarray, max_edges: int) -> np.ndarray:
        m = edge_index.shape[1]
        if m <= max_edges:
            return edge_index
        idx = rng.choice(m, size=max_edges, replace=False)
        return edge_index[:, idx]

    p2p_s = _sample_edges(p2p, 2500)
    r2r_s = _sample_edges(r2r, 800)
    p2r_s = _sample_edges(p2r, 2500)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(context.thumb_rgb)

    for s, t in p2p_s.T:
        x0, y0 = patch_xy_thumb[int(s)]
        x1, y1 = patch_xy_thumb[int(t)]
        ax.plot([x0, x1], [y0, y1], linewidth=0.15, alpha=0.18)

    for s, t in p2r_s.T:
        x0, y0 = patch_xy_thumb[int(s)]
        x1, y1 = region_xy_thumb[int(t)]
        ax.plot([x0, x1], [y0, y1], linewidth=0.2, alpha=0.22)

    for s, t in r2r_s.T:
        x0, y0 = region_xy_thumb[int(s)]
        x1, y1 = region_xy_thumb[int(t)]
        ax.plot([x0, x1], [y0, y1], linewidth=0.7, alpha=0.7)

    annotations = context.annotations or []
    if annotations:
        slide = _ensure_slide_open(context, slide_path)
        thumb_h, thumb_w = context.thumb_rgb.shape[:2]
        full_w, full_h = slide.dimensions
        sx = thumb_w / float(full_w)
        sy = thumb_h / float(full_h)

        ann_labels = []
        for ann in annotations:
            lbl = annotation_label(ann, source=context.creation_cfg.annotation_label_source)
            if lbl is not None:
                ann_labels.append(lbl)
        ann_labels = sorted(set(ann_labels))
        ann_palette = _annotation_palette(ann_labels)

        for ann in annotations[:400]:
            coords = ann.get("coordinates") or []
            if len(coords) < 3:
                continue
            poly = np.asarray(coords, dtype=np.float32)
            x = poly[:, 0] * sx
            y = poly[:, 1] * sy
            lbl = annotation_label(ann, source=context.creation_cfg.annotation_label_source)
            color = ann_palette.get(lbl, (1.0, 0.0, 0.0))
            ax.plot(x, y, linewidth=1.1, color=color, alpha=0.85)

    ax.scatter(patch_xy_thumb[:, 0], patch_xy_thumb[:, 1], s=1.8, alpha=0.55)
    ax.scatter(region_xy_thumb[:, 0], region_xy_thumb[:, 1], s=28, c="black", marker="x")
    ax.set_title("Step 7 - Finaler Graph Overlay inkl. Annotationen (gesampelte Kanten)", fontsize=14)
    ax.axis("off")
    _save_fig(fig, out_dir, "step7_final_graph_overlay_with_annotations.png", args.show_plots)

    if not args.skip_extras:
        if context.patch_xy is None or context.patch_xy.size == 0:
            raise RuntimeError("Keine Patches vorhanden.")

        patch_xy_thumb = _level_xy_to_thumb(context.patch_xy, context, slide_path)
        patch_xy_l0 = context.patch_xy.astype(np.float32) * float(context.slide.level_downsamples[int(context.cfg.patch_level)])
        annotations = context.annotations or []

        patch_annotation_text = []
        for center in patch_xy_l0:
            matched = None
            for ann in annotations:
                if annotation_contains_point(ann, (float(center[0]), float(center[1]))):
                    matched = annotation_label(ann, source=context.creation_cfg.annotation_label_source)
                    if matched is not None:
                        break
            patch_annotation_text.append(matched if matched is not None else "unlabeled")

        patch_annotation_text = np.asarray(patch_annotation_text, dtype=object)
        unique_patch_labels = sorted(set(patch_annotation_text.tolist()))
        palette = _annotation_palette([label for label in unique_patch_labels if label != "unlabeled"])

        patch_colors = []
        for label in patch_annotation_text.tolist():
            if label == "unlabeled":
                patch_colors.append((0.6, 0.6, 0.6))
            else:
                patch_colors.append(palette.get(label, (1.0, 0.0, 0.0)))

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(context.thumb_rgb)
        ax.scatter(patch_xy_thumb[:, 0], patch_xy_thumb[:, 1], s=2.2, c=patch_colors, alpha=0.8)
        ax.set_title("Patch-Zentren farbig nach Annotation-Label", fontsize=14)
        ax.axis("off")
        _save_fig(fig, out_dir, "extra_patch_annotation_coverage.png", args.show_plots)

        patch_label_counts = pd.Series(patch_annotation_text).value_counts(dropna=False).rename_axis("annotation_label").to_frame("num_patches")
        patch_label_counts["fraction_pct"] = 100.0 * patch_label_counts["num_patches"] / float(len(patch_annotation_text))
        _save_df(patch_label_counts.reset_index(), out_dir, "extra_patch_annotation_coverage_stats.csv")

        if context.region_labels is None or context.region_xy is None:
            raise RuntimeError("Regionen fehlen.")

        patch_xy_thumb = _level_xy_to_thumb(context.patch_xy, context, slide_path)
        region_xy_thumb = _level_xy_to_thumb(context.region_xy, context, slide_path)
        cluster_ids = context.region_labels.astype(np.int64)

        region_summary_rows = []
        for region_id in range(int(context.region_xy.shape[0])):
            member_idx = np.where(cluster_ids == region_id)[0]
            member_labels = patch_annotation_text[member_idx] if member_idx.size > 0 else np.asarray([], dtype=object)
            label_counts = pd.Series(member_labels).value_counts() if member_labels.size > 0 else pd.Series(dtype=int)
            top_label = label_counts.index[0] if not label_counts.empty else None
            top_count = int(label_counts.iloc[0]) if not label_counts.empty else 0
            region_summary_rows.append(
                {
                    "region_id": region_id,
                    "num_patches": int(member_idx.size),
                    "majority_label": top_label,
                    "majority_count": top_count,
                    "majority_fraction_pct": (100.0 * top_count / float(member_idx.size)) if member_idx.size > 0 else 0.0,
                }
            )

        region_summary_df = pd.DataFrame(region_summary_rows)
        _save_df(region_summary_df, out_dir, "extra_region_majority_stats.csv")

        rng_local = np.random.default_rng(7)
        sample_idx = np.arange(patch_xy_thumb.shape[0])
        if sample_idx.size > 3000:
            sample_idx = rng_local.choice(sample_idx, size=3000, replace=False)

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(context.thumb_rgb)
        for idx in sample_idx.tolist():
            x0, y0 = patch_xy_thumb[idx]
            region_id = int(cluster_ids[idx])
            x1, y1 = region_xy_thumb[region_id]
            ax.plot([x0, x1], [y0, y1], linewidth=0.18, alpha=0.16, color="#264653")
        ax.scatter(patch_xy_thumb[:, 0], patch_xy_thumb[:, 1], s=1.2, c="yellow", alpha=0.35)
        ax.scatter(region_xy_thumb[:, 0], region_xy_thumb[:, 1], s=50, c="red", marker="x")
        ax.set_title("Patch -> Region Zuordnung (Teilmenge der Linien)", fontsize=14)
        ax.axis("off")
        _save_fig(fig, out_dir, "extra_patch_to_region_assignments.png", args.show_plots)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].bar(region_summary_df["region_id"].astype(str), region_summary_df["num_patches"], color="#457b9d")
        axes[0].set_title("Patches pro Region")
        axes[0].set_ylabel("num_patches")

        axes[1].bar(region_summary_df["region_id"].astype(str), region_summary_df["majority_fraction_pct"], color="#e76f51")
        axes[1].set_title("Mehrheitsanteil des Region-Labels")
        axes[1].set_ylabel("percent")
        axes[1].set_ylim(0, 100)
        plt.tight_layout()
        _save_fig(fig, out_dir, "extra_region_majority_stats.png", args.show_plots)

        if context.mask is None:
            pipeline.steps[2](context)

        annotations = context.annotations or []
        mask_thumb = _mask_to_thumb(np.asarray(context.mask, dtype=bool), context)

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        thumb_a = context.thumb_rgb.copy()
        thumb_a[~mask_thumb] = (thumb_a[~mask_thumb] * 0.15).astype(np.uint8)
        axes[0].imshow(thumb_a)
        axes[0].set_title("A) Tissue-Maske Overlay")
        axes[0].axis("off")

        thumb_b = context.thumb_rgb.copy()
        thumb_b[~mask_thumb] = (thumb_b[~mask_thumb] * 0.15).astype(np.uint8)
        axes[1].imshow(thumb_b)

        if annotations:
            slide = _ensure_slide_open(context, slide_path)
            thumb_h, thumb_w = context.thumb_rgb.shape[:2]
            full_w, full_h = slide.dimensions
            sx = thumb_w / float(full_w)
            sy = thumb_h / float(full_h)

            ann_labels = []
            for ann in annotations:
                lbl = annotation_label(ann, source=context.creation_cfg.annotation_label_source)
                if lbl is not None:
                    ann_labels.append(lbl)
            ann_labels = sorted(set(ann_labels))
            ann_palette = _annotation_palette(ann_labels)

            for ann in annotations[:600]:
                coords = ann.get("coordinates") or []
                if len(coords) < 3:
                    continue
                poly = np.asarray(coords, dtype=np.float32)
                x = poly[:, 0] * sx
                y = poly[:, 1] * sy
                lbl = annotation_label(ann, source=context.creation_cfg.annotation_label_source)
                color = ann_palette.get(lbl, (1.0, 0.0, 0.0))
                axes[1].fill(x, y, color=color, alpha=0.14)
                axes[1].plot(x, y, linewidth=1.3, color=color, alpha=0.95)

        axes[1].set_title(f"B) Tissue-Maske + Annotationen (n={len(annotations)})")
        axes[1].axis("off")
        plt.tight_layout()
        _save_fig(fig, out_dir, "extra_mask_and_annotations_only.png", args.show_plots)

    if args.auto_close_slide:
        try:
            context.slide.close()
            print("Slide handle closed.")
        except Exception as exc:
            print("Close skipped:", exc)

    summary_payload = {
        "slide_path": str(slide_path),
        "annotation_file": str(annotation_file) if annotation_file is not None else None,
        "mask_file": str(mask_file) if mask_file is not None else None,
        "pipeline_config": str(args.pipeline_config),
        "graph_build_config_path": str(graph_build_config_path),
        "out_dir": str(out_dir),
        "show_plots": bool(args.show_plots),
        "skip_extras": bool(args.skip_extras),
        "available_pipeline_configs": AVAILABLE_PIPELINE_CONFIGS,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print("Alle Step-by-Step Outputs in:", out_dir)


def main() -> None:
    """Parse arguments and run the step-by-step graph pipeline."""
    args = parse_args()
    _run(args)


if __name__ == "__main__":
    main()
