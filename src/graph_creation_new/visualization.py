from __future__ import annotations

import json
import os
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib as mpl
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import HeteroData

from src.graph_creation_new.config import BuildConfig, BuildContext, CreationConfig
from src.graph_creation_new.enums import AnnotationType, DatasetType, MaskType
from src.graph_creation_new.helper.annotations import load_asap_annotation, load_geojson_annotations
from src.graph_creation_new.helper.thumbnail import safe_thumbnail
from src.graph_creation_new.helper.mask import load_binary_mask, generate_mask_from_thumbnail
from src.graph_creation_new.helper.slide import prepare_slide, generate_preview
from src.paths.path_resolver import PathResolver
from src.paths.paths import get_project_paths


@dataclass
class VisualizationConfig:
    thumbnail_max_size: int = 2000
    patch_point_size: float = 2.0
    region_point_size: float = 40.0
    hierarchy_patch_point_size: float = 6.0
    hierarchy_region_point_size: float = 90.0
    hierarchy_slide_point_size: float = 240.0
    patch_edge_max: int = 5000
    region_edge_max: int = 2000
    hierarchy_edge_max: int = 8000
    patch_edge_lw: float = 0.25
    region_edge_lw: float = 0.6
    hierarchy_edge_lw: float = 0.35
    edge_color_patch_near: str = "#ffd166"
    edge_color_region_near: str = "#06d6a0"
    edge_color_patch_in_region: str = "#118ab2"
    edge_color_region_part_of: str = "#ef476f"
    edge_alpha_patch_near: float = 0.25
    edge_alpha_region_near: float = 0.7
    edge_alpha_hierarchy: float = 0.45
    edge_insight_bins: int = 50
    publication_mode: bool = True
    publication_dpi: int = 350
    publication_export_pdf: bool = False
    publication_export_svg: bool = False
    publication_font_family: str = "DejaVu Sans"
    publication_font_size: float = 11.0
    publication_title_size: float = 13.0
    publication_label_size: float = 10.0
    publication_legend_size: float = 9.0
    publication_axes_linewidth: float = 0.8
    seed: int = 0


def _import_openslide_optional():
    try:
        import openslide
    except Exception:
        return None
    return openslide


def plot_region_cell_debug(
    context: BuildContext,
    region_id: int,
    segmentation_level: Optional[int] = None,
    max_edges_to_draw: int = 3000,
    figsize: Tuple[float, float] = (8.0, 8.0),
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an overlay plot for one region with segmented cell nodes and their kNN edges."""
    from src.graph_creation_new.helper.cells import build_region_cell_debug_payload

    payload = build_region_cell_debug_payload(
        context=context,
        region_id=region_id,
        segmentation_level=segmentation_level,
    )

    image_rgb = payload["image_rgb"]
    local_xy = payload["local_xy"]
    edge_src = payload["edge_src"]
    edge_dst = payload["edge_dst"]
    n_cells = int(local_xy.shape[0])
    n_edges = int(edge_src.shape[0])

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image_rgb)

    if n_edges > 0 and n_cells > 0:
        draw_count = min(int(max_edges_to_draw), n_edges)
        for src, dst in zip(edge_src[:draw_count], edge_dst[:draw_count]):
            p0 = local_xy[int(src)]
            p1 = local_xy[int(dst)]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="yellow", linewidth=0.35, alpha=0.45)

    if n_cells > 0:
        ax.scatter(
            local_xy[:, 0],
            local_xy[:, 1],
            s=10,
            c="lime",
            edgecolors="black",
            linewidths=0.3,
            alpha=0.95,
        )

    ax.set_axis_off()
    ax.set_title(
        (
            f"Region {payload['region_id']} | cells={n_cells} | edges={n_edges} | "
            f"k={payload['cell_k']} | level={payload['segmentation_level']}"
        ),
        fontsize=10,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    payload["figure"] = fig
    payload["axes"] = ax
    payload["num_cells"] = n_cells
    payload["num_edges"] = n_edges
    return payload


def plot_region_cells_detailed(
    context: BuildContext,
    region_id: int,
    segmentation_level: Optional[int] = None,
    figsize: Tuple[float, float] = (16.0, 16.0),
    save_path: Optional[str] = None,
    show_cell_labels: bool = True,
    show_cell_area: bool = True,
) -> Dict[str, Any]:
    """Create a detailed visualization of a region with all cells marked, labeled, and indexed."""
    from src.graph_creation_new.helper.cells import build_region_cell_debug_payload

    payload = build_region_cell_debug_payload(
        context=context,
        region_id=region_id,
        segmentation_level=segmentation_level,
    )

    image_rgb = payload["image_rgb"]
    local_xy = payload["local_xy"]
    cell_area = payload["cell_area"]
    edge_src = payload["edge_src"]
    edge_dst = payload["edge_dst"]
    n_cells = int(local_xy.shape[0])
    n_edges = int(edge_src.shape[0])

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image_rgb)

    if n_edges > 0 and n_cells > 0:
        for src, dst in zip(edge_src, edge_dst):
            p0 = local_xy[int(src)]
            p1 = local_xy[int(dst)]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="cyan", linewidth=0.25, alpha=0.25)

    if n_cells > 0:
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, max(1, n_cells))))
        
        for cell_idx in range(n_cells):
            x, y = local_xy[cell_idx]
            area = cell_area[cell_idx] if cell_area.size > 0 else 0.0
            
            circle = plt.Circle(
                (x, y),
                radius=8.0,
                color="lime",
                fill=False,
                linewidth=1.2,
                alpha=0.8,
            )
            ax.add_patch(circle)
            
            ax.plot(x, y, "o", markersize=6, color="red", alpha=0.9)
            
            if show_cell_labels:
                ax.text(
                    x + 5,
                    y - 5,
                    f"{cell_idx}",
                    fontsize=7,
                    color="white",
                    weight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, edgecolor="none"),
                    zorder=100,
                )
            
            if show_cell_area and area > 0:
                ax.text(
                    x + 5,
                    y + 12,
                    f"A:{int(area)}",
                    fontsize=6,
                    color="yellow",
                    weight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.5, edgecolor="none"),
                    zorder=100,
                )

    ax.set_axis_off()
    ax.set_title(
        (
            f"Region {payload['region_id']} - Detailed Cell Annotation | "
            f"cells={n_cells} | edges={n_edges} | "
            f"k={payload['cell_k']} | level={payload['segmentation_level']} | "
            f"tiles={payload['cell_segmentation_tile_size']}px (overlap={payload['cell_segmentation_tile_overlap']}px)"
        ),
        fontsize=12,
        weight="bold",
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logging.info(f"Saved detailed cell visualization to: {save_path}")

    payload["figure"] = fig
    payload["axes"] = ax
    payload["num_cells"] = n_cells
    payload["num_edges"] = n_edges
    return payload


def debug_plot_region_cells(
    slide: Union[str, Path],
    cfg: BuildConfig,
    region_id: int,
    creation_cfg: Optional[CreationConfig] = None,
    segmentation_level: Optional[int] = None,
    max_edges_to_draw: int = 3000,
    figsize: Tuple[float, float] = (8.0, 8.0),
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the slide context and return a debug payload for one region's segmented cells."""
    openslide = _import_openslide_optional()
    if openslide is None:
        raise RuntimeError("openslide-python konnte nicht importiert werden. Bitte OpenSlide + openslide-python installieren.")

    slide_path = str(slide)
    handle = openslide.OpenSlide(slide_path)

    try:
        resolved_creation_cfg = creation_cfg if creation_cfg is not None else CreationConfig()
        context = BuildContext(
            path_resolver=PathResolver(get_project_paths()),
            dataset=DatasetType.CAMELYON17,
            slide_path=slide_path,
            slide=handle,
            cfg=cfg,
            node_feature_model=None,
            creation_cfg=resolved_creation_cfg,
        )

        context.thumb_rgb = safe_thumbnail(context.slide, context.cfg.thumb_max_size)

        if resolved_creation_cfg.use_annotations:
            annotation_path = context.path_resolver.annotation_path(
                Path(context.slide_path),
                context.dataset.value,
                resolved_creation_cfg.annotation_type,
            )
            match resolved_creation_cfg.annotation_type:
                case AnnotationType.ASAP:
                    context.annotations = load_asap_annotation(annotation_path)
                case AnnotationType.GEOJSON:
                    context.annotations = load_geojson_annotations(annotation_path)
                case _:
                    logging.warning(
                        "Annotation creation not supported for type: %s",
                        resolved_creation_cfg.annotation_type,
                    )

        if not resolved_creation_cfg.use_precomputed_mask:
            mask, downsample = generate_mask_from_thumbnail(context)
            context.mask = mask if mask is not None else None
            context.mask_downsized = downsample if downsample is not None else None
        else:
            mask_path = context.path_resolver.mask_path(
                Path(context.slide_path),
                context.dataset.value,
                None,
                resolved_creation_cfg.mask_type,
            )
            match resolved_creation_cfg.mask_type:
                case MaskType.SLIDE:
                    mask, downsample = load_binary_mask(mask_path)
                    if mask is None or downsample is None:
                        logging.warning(
                            "Precomputed mask missing for %s -> generating mask from thumbnail instead",
                            context.slide_path,
                        )
                        mask, downsample = generate_mask_from_thumbnail(context)
                    context.mask = mask if mask is not None else None
                    context.mask_downsized = downsample if downsample is not None else None
                case MaskType.IMAGE:
                    raise ValueError("Image mask not supported for region cell debug plot")
                case _:
                    raise ValueError(f"Mask creation not supported for type: {resolved_creation_cfg.mask_type.value}")

        prepare_slide(context)
        generate_preview(context)

        return plot_region_cell_debug(
            context=context,
            region_id=region_id,
            segmentation_level=segmentation_level,
            max_edges_to_draw=max_edges_to_draw,
            figsize=figsize,
            save_path=save_path,
        )
    finally:
        handle.close()


def debug_plot_region_cells_detailed(
    slide: Union[str, Path],
    cfg: BuildConfig,
    region_id: int,
    creation_cfg: Optional[CreationConfig] = None,
    segmentation_level: Optional[int] = None,
    figsize: Tuple[float, float] = (16.0, 16.0),
    save_path: Optional[str] = None,
    show_cell_labels: bool = True,
    show_cell_area: bool = True,
) -> Dict[str, Any]:
    """Build the slide context and return a detailed cell-annotation visualization for one region."""
    openslide = _import_openslide_optional()
    if openslide is None:
        raise RuntimeError("openslide-python konnte nicht importiert werden. Bitte OpenSlide + openslide-python installieren.")

    slide_path = str(slide)
    handle = openslide.OpenSlide(slide_path)

    try:
        resolved_creation_cfg = creation_cfg if creation_cfg is not None else CreationConfig()
        context = BuildContext(
            path_resolver=PathResolver(get_project_paths()),
            dataset=DatasetType.CAMELYON17,
            slide_path=slide_path,
            slide=handle,
            cfg=cfg,
            node_feature_model=None,
            creation_cfg=resolved_creation_cfg,
        )

        context.thumb_rgb = safe_thumbnail(context.slide, context.cfg.thumb_max_size)

        if resolved_creation_cfg.use_annotations:
            annotation_path = context.path_resolver.annotation_path(
                Path(context.slide_path),
                context.dataset.value,
                resolved_creation_cfg.annotation_type,
            )
            match resolved_creation_cfg.annotation_type:
                case AnnotationType.ASAP:
                    context.annotations = load_asap_annotation(annotation_path)
                case AnnotationType.GEOJSON:
                    context.annotations = load_geojson_annotations(annotation_path)
                case _:
                    logging.warning(
                        "Annotation creation not supported for type: %s",
                        resolved_creation_cfg.annotation_type,
                    )

        if not resolved_creation_cfg.use_precomputed_mask:
            mask, downsample = generate_mask_from_thumbnail(context)
            context.mask = mask if mask is not None else None
            context.mask_downsized = downsample if downsample is not None else None
        else:
            mask_path = context.path_resolver.mask_path(
                Path(context.slide_path),
                context.dataset.value,
                None,
                resolved_creation_cfg.mask_type,
            )
            match resolved_creation_cfg.mask_type:
                case MaskType.SLIDE:
                    mask, downsample = load_binary_mask(mask_path)
                    if mask is None or downsample is None:
                        logging.warning(
                            "Precomputed mask missing for %s -> generating mask from thumbnail instead",
                            context.slide_path,
                        )
                        mask, downsample = generate_mask_from_thumbnail(context)
                    context.mask = mask if mask is not None else None
                    context.mask_downsized = downsample if downsample is not None else None
                case MaskType.IMAGE:
                    raise ValueError("Image mask not supported for region cell debug plot")
                case _:
                    raise ValueError(f"Mask creation not supported for type: {resolved_creation_cfg.mask_type.value}")

        prepare_slide(context)
        generate_preview(context)

        return plot_region_cells_detailed(
            context=context,
            region_id=region_id,
            segmentation_level=segmentation_level,
            figsize=figsize,
            save_path=save_path,
            show_cell_labels=show_cell_labels,
            show_cell_area=show_cell_area,
        )
    finally:
        handle.close()


def _ensure_dir(path: Union[str, Path]) -> None:
    os.makedirs(str(path), exist_ok=True)


def _apply_publication_style(viz_cfg: VisualizationConfig) -> None:
    if not viz_cfg.publication_mode:
        return

    mpl.rcParams.update({
        "font.family": viz_cfg.publication_font_family,
        "font.size": viz_cfg.publication_font_size,
        "axes.titlesize": viz_cfg.publication_title_size,
        "axes.labelsize": viz_cfg.publication_label_size,
        "legend.fontsize": viz_cfg.publication_legend_size,
        "axes.linewidth": viz_cfg.publication_axes_linewidth,
        "xtick.labelsize": max(8.0, viz_cfg.publication_label_size - 1.0),
        "ytick.labelsize": max(8.0, viz_cfg.publication_label_size - 1.0),
        "savefig.dpi": viz_cfg.publication_dpi,
        "figure.dpi": max(120, min(200, int(viz_cfg.publication_dpi // 2))),
    })


def _save_figure(fig: plt.Figure, path: Union[str, Path], viz_cfg: VisualizationConfig, pad_inches: float = 0.08) -> None:
    out_path = Path(path)
    fig.savefig(str(out_path), dpi=viz_cfg.publication_dpi, bbox_inches="tight", pad_inches=pad_inches)

    if viz_cfg.publication_mode and viz_cfg.publication_export_pdf:
        fig.savefig(str(out_path.with_suffix(".pdf")), bbox_inches="tight", pad_inches=pad_inches)
    if viz_cfg.publication_mode and viz_cfg.publication_export_svg:
        fig.savefig(str(out_path.with_suffix(".svg")), bbox_inches="tight", pad_inches=pad_inches)


def _save_img(path: Union[str, Path], arr: np.ndarray, viz_cfg: VisualizationConfig) -> None:
    plt.figure(figsize=(10, 10))
    if arr.ndim == 2:
        plt.imshow(arr, cmap="gray")
    else:
        plt.imshow(arr)
    plt.axis("off")
    plt.tight_layout()
    _save_figure(plt.gcf(), path, viz_cfg, pad_inches=0.0)
    plt.close()


def _safe_thumbnail(slide, max_size: int) -> np.ndarray:
    width0, height0 = slide.dimensions
    scale = max(width0, height0) / float(max_size)
    if scale < 1.0:
        scale = 1.0
    width_t, height_t = int(round(width0 / scale)), int(round(height0 / scale))
    width_t = max(64, width_t)
    height_t = max(64, height_t)

    thumb = slide.get_thumbnail((width_t, height_t))
    thumb_rgb = np.asarray(thumb)
    if thumb_rgb.ndim == 2:
        thumb_rgb = np.stack([thumb_rgb, thumb_rgb, thumb_rgb], axis=-1)
    if thumb_rgb.shape[-1] == 4:
        thumb_rgb = thumb_rgb[..., :3]
    return thumb_rgb.astype(np.uint8)


def _coords_level_to_thumb(
    slide,
    cfg: BuildConfig,
    thumb_shape: Tuple[int, int],
    xy_level: np.ndarray,
) -> np.ndarray:
    downsample = float(slide.level_downsamples[cfg.patch_level])
    width0, height0 = slide.dimensions
    thumb_h, thumb_w = thumb_shape
    scale_x = thumb_w / float(width0)
    scale_y = thumb_h / float(height0)

    xy0 = xy_level * downsample
    mapped = np.empty_like(xy0, dtype=np.float32)
    mapped[:, 0] = xy0[:, 0] * scale_x
    mapped[:, 1] = xy0[:, 1] * scale_y
    return mapped


def _make_canvas_from_points(patch_xy: np.ndarray, region_xy: np.ndarray, size: int = 2048) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_xy = np.concatenate([patch_xy, region_xy], axis=0)
    min_xy = all_xy.min(axis=0)
    max_xy = all_xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)

    pad = 40.0
    usable = float(size) - 2.0 * pad
    scale = min(usable / span[0], usable / span[1])

    patch_m = (patch_xy - min_xy) * scale + pad
    region_m = (region_xy - min_xy) * scale + pad

    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    return canvas, patch_m.astype(np.float32), region_m.astype(np.float32)

def _to_numpy_edge_index(edge) -> np.ndarray:
    import torch
    if hasattr(edge, 'edge_index'):
        edge = edge.edge_index
    if isinstance(edge, torch.Tensor):
        return edge.cpu().numpy()
    if isinstance(edge, np.ndarray):
        return edge
    raise TypeError(f"Cannot extract edge_index from object of type {type(edge)}")


def _plot_points(
    background: np.ndarray,
    pts_xy: np.ndarray,
    out_path: Union[str, Path],
    title: str,
    size: float,
    color: str,
    viz_cfg: VisualizationConfig,
) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(background)
    plt.scatter(pts_xy[:, 0], pts_xy[:, 1], s=size, c=color, linewidths=0)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    _save_figure(plt.gcf(), out_path, viz_cfg, pad_inches=0.0)
    plt.close()


def _plot_edges(
    background: np.ndarray,
    pts_xy: np.ndarray,
    edge_index,
    out_path: Union[str, Path],
    title: str,
    max_edges: int,
    lw: float,
    color: str,
    seed: int,
    viz_cfg: VisualizationConfig,
) -> None:
    edge_index = _to_numpy_edge_index(edge_index)
    rng = np.random.default_rng(seed)
    n_edges = edge_index.shape[1]
    if n_edges > max_edges:
        idx = rng.choice(n_edges, size=max_edges, replace=False)
        edges = edge_index[:, idx]
    else:
        edges = edge_index

    plt.figure(figsize=(10, 10))
    plt.imshow(background)
    for src, dst in edges.T:
        x1, y1 = pts_xy[int(src)]
        x2, y2 = pts_xy[int(dst)]
        plt.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=0.6)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    _save_figure(plt.gcf(), out_path, viz_cfg, pad_inches=0.0)
    plt.close()


def _region_color_map(count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 3), dtype=np.uint8)

    base = np.asarray([
        [0, 114, 178],
        [230, 159, 0],
        [86, 180, 233],
        [0, 158, 115],
        [240, 228, 66],
        [213, 94, 0],
        [204, 121, 167],
        [68, 68, 68],
    ], dtype=np.uint8)

    if count <= base.shape[0]:
        return base[:count]

    repeats = int(np.ceil(count / float(base.shape[0])))
    tiled = np.tile(base, (repeats, 1))
    return tiled[:count]


def _plot_regions_colored(
    background: np.ndarray,
    patch_xy: np.ndarray,
    region_labels: np.ndarray,
    region_xy: np.ndarray,
    out_path: Union[str, Path],
    title: str,
    patch_size: float,
    region_size: float,
    viz_cfg: VisualizationConfig,
) -> None:
    n_regions = int(region_xy.shape[0])
    colors = _region_color_map(n_regions)
    patch_colors = colors[region_labels]

    plt.figure(figsize=(10, 10))
    plt.imshow(background)
    plt.scatter(patch_xy[:, 0], patch_xy[:, 1], s=patch_size, c=patch_colors / 255.0, linewidths=0, alpha=0.8)
    plt.scatter(region_xy[:, 0], region_xy[:, 1], s=region_size, c="white", edgecolors="black", linewidths=0.7)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    _save_figure(plt.gcf(), out_path, viz_cfg, pad_inches=0.0)
    plt.close()


def _plot_hierarchy(
    background: np.ndarray,
    region_xy: np.ndarray,
    out_path: Union[str, Path],
    title: str,
    viz_cfg: VisualizationConfig,
) -> None:
    center = np.array([[background.shape[1] / 2.0, background.shape[0] / 2.0]], dtype=np.float32)
    plt.figure(figsize=(10, 10))
    plt.imshow(background)
    for x, y in region_xy:
        plt.plot([x, center[0, 0]], [y, center[0, 1]], color="white", linewidth=0.25, alpha=0.5)
    plt.scatter(region_xy[:, 0], region_xy[:, 1], s=40, c="white", edgecolors="black", linewidths=0.6)
    plt.scatter([center[0, 0]], [center[0, 1]], s=80, c="red")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    _save_figure(plt.gcf(), out_path, viz_cfg, pad_inches=0.0)
    plt.close()


def _sample_edges(edge_index: np.ndarray, max_edges: int, seed: int) -> np.ndarray:
    n_edges = edge_index.shape[1]
    if n_edges <= max_edges:
        return edge_index
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_edges, size=max_edges, replace=False)
    return edge_index[:, idx]


def _node_degree(edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    degrees = np.zeros((max(0, int(n_nodes)),), dtype=np.int64)
    if edge_index.size == 0 or n_nodes <= 0:
        return degrees
    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)
    np.add.at(degrees, src, 1)
    np.add.at(degrees, dst, 1)
    return degrees


def _edge_lengths(xy: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    if xy.size == 0 or edge_index.size == 0:
        return np.zeros((0,), dtype=np.float32)
    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)
    diff = xy[src] - xy[dst]
    return np.linalg.norm(diff, axis=1).astype(np.float32)


def _style_stat_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_hist_with_stats(
    ax: plt.Axes,
    values: np.ndarray,
    bins: int,
    color: str,
    title: str,
    xlabel: str,
) -> None:
    if values.size == 0:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        _style_stat_axis(ax)
        return

    values = values.astype(np.float32)
    ax.hist(values, bins=max(10, int(bins)), color=color, alpha=0.75, density=True)

    mean_v = float(np.mean(values))
    med_v = float(np.median(values))
    q1_v = float(np.quantile(values, 0.25))
    q3_v = float(np.quantile(values, 0.75))
    p95_v = float(np.quantile(values, 0.95))

    ax.axvline(med_v, color="#111111", linewidth=1.3, linestyle="-", label="median")
    ax.axvspan(q1_v, q3_v, color="#111111", alpha=0.12, label="IQR")
    ax.axvline(p95_v, color="#333333", linewidth=1.0, linestyle=":", label="p95")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    _style_stat_axis(ax)

    stats_text = (
        f"n={values.size}\n"
        f"mean={mean_v:.2f}\n"
        f"median={med_v:.2f}\n"
        f"IQR=[{q1_v:.2f}, {q3_v:.2f}]"
    )
    ax.text(
        0.985,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82, edgecolor="#cccccc"),
    )


def _plot_multiedge_overlay(
    background: np.ndarray,
    patch_xy: np.ndarray,
    region_xy: np.ndarray,
    p2p: np.ndarray,
    r2r: np.ndarray,
    p2r: np.ndarray,
    out_path: Union[str, Path],
    viz_cfg: VisualizationConfig,
    seed: int,
) -> None:
    p2p_s = _sample_edges(p2p, viz_cfg.patch_edge_max, seed)
    r2r_s = _sample_edges(r2r, viz_cfg.region_edge_max, seed)
    p2r_s = _sample_edges(p2r, viz_cfg.patch_edge_max, seed)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(background)
    ax.set_title("Multi-edge overlay by edge type")
    ax.axis("off")

    for src, dst in p2p_s.T:
        x1, y1 = patch_xy[int(src)]
        x2, y2 = patch_xy[int(dst)]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=viz_cfg.edge_color_patch_near,
            linewidth=max(0.1, viz_cfg.patch_edge_lw),
            alpha=viz_cfg.edge_alpha_patch_near,
        )

    for src, dst in r2r_s.T:
        x1, y1 = region_xy[int(src)]
        x2, y2 = region_xy[int(dst)]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=viz_cfg.edge_color_region_near,
            linewidth=max(0.2, viz_cfg.region_edge_lw),
            alpha=viz_cfg.edge_alpha_region_near,
        )

    for src, dst in p2r_s.T:
        x1, y1 = patch_xy[int(src)]
        x2, y2 = region_xy[int(dst)]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=viz_cfg.edge_color_patch_in_region,
            linewidth=max(0.1, viz_cfg.patch_edge_lw * 0.9),
            alpha=viz_cfg.edge_alpha_hierarchy,
        )

    ax.scatter(
        patch_xy[:, 0],
        patch_xy[:, 1],
        s=max(1.0, viz_cfg.patch_point_size * 0.8),
        c="#f8f9fa",
        linewidths=0,
        alpha=0.8,
    )
    ax.scatter(
        region_xy[:, 0],
        region_xy[:, 1],
        s=max(18.0, viz_cfg.region_point_size * 0.65),
        c="#ffffff",
        edgecolors="#111111",
        linewidths=0.7,
        alpha=1.0,
    )

    handles = [
        plt.Line2D([0], [0], color=viz_cfg.edge_color_patch_near, lw=2, label="patch->near->patch"),
        plt.Line2D([0], [0], color=viz_cfg.edge_color_region_near, lw=2, label="region->near->region"),
        plt.Line2D([0], [0], color=viz_cfg.edge_color_patch_in_region, lw=2, label="patch->in->region"),
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.9)

    overlay_text = (
        f"patch nodes={patch_xy.shape[0]} | region nodes={region_xy.shape[0]}\n"
        f"patch-near edges={p2p.shape[1]} | region-near edges={r2r.shape[1]} | patch-in-region edges={p2r.shape[1]}"
    )
    ax.text(
        0.02,
        0.03,
        overlay_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#cccccc"),
    )

    fig.tight_layout()
    _save_figure(fig, out_path, viz_cfg, pad_inches=0.05)
    plt.close(fig)


def _plot_graph_insights(
    patch_xy: np.ndarray,
    region_xy: np.ndarray,
    p2p: np.ndarray,
    r2r: np.ndarray,
    p2r: np.ndarray,
    out_path: Union[str, Path],
    viz_cfg: VisualizationConfig,
) -> None:
    patch_lengths = _edge_lengths(patch_xy, p2p)
    region_lengths = _edge_lengths(region_xy, r2r)

    n_patches = int(patch_xy.shape[0])
    n_regions = int(region_xy.shape[0])
    patch_degree = _node_degree(p2p, n_patches)
    region_degree = _node_degree(r2r, n_regions)

    patch_assignments = np.zeros((n_patches,), dtype=np.int64)
    if p2r.size > 0:
        np.add.at(patch_assignments, p2r[0].astype(np.int64), 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    _plot_hist_with_stats(
        ax=ax,
        values=patch_lengths,
        bins=int(viz_cfg.edge_insight_bins),
        color=viz_cfg.edge_color_patch_near,
        title="Patch edge length distribution",
        xlabel="distance",
    )

    ax = axes[0, 1]
    _plot_hist_with_stats(
        ax=ax,
        values=region_lengths,
        bins=max(10, int(viz_cfg.edge_insight_bins // 2)),
        color=viz_cfg.edge_color_region_near,
        title="Region edge length distribution",
        xlabel="distance",
    )

    ax = axes[1, 0]
    if patch_degree.size > 0:
        bins = np.arange(0, int(max(1, patch_degree.max())) + 2) - 0.5
        ax.hist(patch_degree, bins=bins, color="#6c757d", alpha=0.88, density=True)
        ax.axvline(float(np.median(patch_degree)), color="#111111", linewidth=1.3)
    ax.set_title("Patch node degree")
    ax.set_xlabel("degree")
    ax.set_ylabel("density")
    _style_stat_axis(ax)

    ax = axes[1, 1]
    if region_degree.size > 0:
        bins = np.arange(0, int(max(1, region_degree.max())) + 2) - 0.5
        ax.hist(region_degree, bins=bins, color="#495057", alpha=0.7, density=True, label="region kNN degree")
    if patch_assignments.size > 0:
        bins = np.arange(0, int(max(1, patch_assignments.max())) + 2) - 0.5
        ax.hist(
            patch_assignments,
            bins=bins,
            color=viz_cfg.edge_color_patch_in_region,
            alpha=0.5,
            density=True,
            label="patch->region assignment degree",
        )
    ax.set_title("Region degree and assignment diagnostics")
    ax.set_xlabel("degree")
    ax.set_ylabel("density")
    _style_stat_axis(ax)
    ax.legend(loc="upper right")

    diagnostics = (
        f"Patch degree median: {float(np.median(patch_degree)) if patch_degree.size else 0:.2f}\n"
        f"Region degree median: {float(np.median(region_degree)) if region_degree.size else 0:.2f}\n"
        f"Assignment median: {float(np.median(patch_assignments)) if patch_assignments.size else 0:.2f}"
    )
    ax.text(
        0.02,
        0.97,
        diagnostics,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )

    fig.tight_layout()
    _save_figure(fig, out_path, viz_cfg, pad_inches=0.08)
    plt.close(fig)


def _normalize_to_unit_interval(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    min_v = np.min(values)
    max_v = np.max(values)
    span = max(float(max_v - min_v), 1e-6)
    return ((values - min_v) / span).astype(np.float32)


def _hierarchy_layout(
    region_xy: np.ndarray,
    patch_xy: np.ndarray,
    region_labels: np.ndarray,
    canvas_width: int = 2200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    usable_w = float(canvas_width - 240)
    x_min = 120.0
    x_max = x_min + usable_w

    slide_band = (70.0, 360.0)
    region_band = (520.0, 940.0)
    patch_band = (1080.0, 1520.0)

    patch_x = _normalize_to_unit_interval(patch_xy[:, 0]) * usable_w + x_min
    patch_x = np.clip(patch_x, x_min + 8.0, x_max - 8.0)
    patch_y_norm = _normalize_to_unit_interval(patch_xy[:, 1])
    patch_y = patch_band[0] + 42.0 + patch_y_norm * max(1.0, (patch_band[1] - patch_band[0] - 84.0))

    region_x = _normalize_to_unit_interval(region_xy[:, 0]) * usable_w + x_min
    region_x = np.clip(region_x, x_min + 8.0, x_max - 8.0)
    region_y_norm = _normalize_to_unit_interval(region_xy[:, 1]) if region_xy.size > 0 else np.zeros((0,), dtype=np.float32)
    region_y = region_band[0] + 70.0 + region_y_norm * max(1.0, (region_band[1] - region_band[0] - 140.0))

    slide_xy = np.array([[canvas_width / 2.0, (slide_band[0] + slide_band[1]) / 2.0]], dtype=np.float32)

    region_palette = _region_color_map(max(1, int(region_xy.shape[0])))
    patch_colors = region_palette[region_labels] / 255.0

    return (
        np.stack([patch_x, patch_y], axis=1).astype(np.float32),
        np.stack([region_x, region_y], axis=1).astype(np.float32),
        slide_xy,
        patch_colors,
    )


def _plot_hierarchical_graph(
    data: HeteroData,
    out_path: Union[str, Path],
    viz_cfg: VisualizationConfig,
    seed: int,
) -> None:
    patch_xy = data["patch"].centroid.cpu().numpy()
    region_xy = data["region"].centroid.cpu().numpy()
    if patch_xy.size == 0 or region_xy.size == 0:
        raise ValueError("Cannot visualize hierarchical graph without patches and regions")

    p2r = _to_numpy_edge_index(data["patch", "in", "region"])
    p2p = _to_numpy_edge_index(data["patch", "near", "patch"])
    r2r = _to_numpy_edge_index(data["region", "near", "region"])

    n_patches = int(data["patch"].num_nodes)
    n_regions = int(data["region"].num_nodes)

    region_palette = _region_color_map(max(1, n_regions))
    patch_colors = region_palette[p2r[1]] / 255.0
    patch_xy_h, region_xy_h, slide_xy_h, patch_colors = _hierarchy_layout(
        region_xy=region_xy,
        patch_xy=patch_xy,
        region_labels=p2r[1],
        canvas_width=2200,
    )

    patch_edges = _sample_edges(p2p, viz_cfg.hierarchy_edge_max, seed)
    region_edges = _sample_edges(r2r, max(1, viz_cfg.hierarchy_edge_max // 2), seed)

    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], width_ratios=[1.2, 1.1, 0.9])

    ax = fig.add_subplot(gs[0, :])
    ax.set_title("Hierarchical graph layout", fontsize=16, weight="bold")
    ax.set_xlim(0, 2200)
    ax.set_ylim(0, 1600)
    ax.invert_yaxis()
    ax.axis("off")

    band_specs = [
        (70, 360, "slide", "#ececec"),
        (520, 940, "regions", "#f7f7f7"),
        (1080, 1520, "patches", "#fcfcfc"),
    ]
    for y0, y1, label, color in band_specs:
        ax.add_patch(plt.Rectangle((40, y0), 2120, y1 - y0, facecolor=color, edgecolor="#d0d0d0", linewidth=0.9, zorder=0))
        ax.text(
            65,
            y0 + 10,
            label.upper(),
            fontsize=11,
            weight="bold",
            color="#555555",
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="#d0d0d0"),
            clip_on=True,
        )

    slide_node = slide_xy_h[0]
    ax.scatter([slide_node[0]], [slide_node[1]], s=viz_cfg.hierarchy_slide_point_size, c="#c0392b", edgecolors="white", linewidths=1.1, zorder=5)
    ax.text(slide_node[0] + 14, slide_node[1] - 10, "slide", fontsize=11, weight="bold", color="#7f1d1d")

    for src, dst in region_edges.T:
        x1, y1 = region_xy_h[int(src)]
        x2, y2 = region_xy_h[int(dst)]
        ax.plot([x1, x2], [y1, y2], color=viz_cfg.edge_color_region_near, linewidth=max(0.18, viz_cfg.hierarchy_edge_lw * 0.9), alpha=0.28, zorder=1)

    for src, dst in patch_edges.T:
        x1, y1 = patch_xy_h[int(src)]
        x2, y2 = patch_xy_h[int(dst)]
        ax.plot([x1, x2], [y1, y2], color=viz_cfg.edge_color_patch_near, linewidth=max(0.12, viz_cfg.hierarchy_edge_lw * 0.55), alpha=0.12, zorder=1)

    for region_id, (x, y) in enumerate(region_xy_h):
        ax.scatter([x], [y], s=viz_cfg.hierarchy_region_point_size, c=[region_palette[region_id] / 255.0], edgecolors="#222222", linewidths=0.7, zorder=4)
        if n_regions <= 30 or region_id % max(1, n_regions // 20) == 0:
            ax.text(x + 7, y - 8, str(region_id), fontsize=8, color="#333333")
        ax.plot([slide_node[0], x], [slide_node[1] + 58, y - 24], color=viz_cfg.edge_color_region_part_of, linewidth=0.65, alpha=0.42, zorder=2)

    for patch_id, (x, y) in enumerate(patch_xy_h):
        ax.scatter([x], [y], s=viz_cfg.hierarchy_patch_point_size, c=[patch_colors[patch_id]], edgecolors="none", alpha=0.92, zorder=3)

    ax.text(
        64,
        1545,
        f"patches={n_patches} | regions={n_regions} | patch->region edges={int(p2r.shape[1])}",
        fontsize=10,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.82, edgecolor="#cccccc"),
    )

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("Region assignment in slide space", fontsize=12)
    ax2.axis("off")
    ax2.scatter(patch_xy[:, 0], patch_xy[:, 1], s=1.0, c=patch_colors, linewidths=0, alpha=0.82)
    ax2.scatter(region_xy[:, 0], region_xy[:, 1], s=28, c="white", edgecolors="#222222", linewidths=0.6)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title("Edge-type diagnostics", fontsize=12)
    ax3.axis("off")
    legend_handles = [
        plt.Line2D([0], [0], color=viz_cfg.edge_color_patch_near, lw=2, label="patch->near->patch"),
        plt.Line2D([0], [0], color=viz_cfg.edge_color_region_near, lw=2, label="region->near->region"),
        plt.Line2D([0], [0], color=viz_cfg.edge_color_patch_in_region, lw=2, label="patch->in->region"),
        plt.Line2D([0], [0], color=viz_cfg.edge_color_region_part_of, lw=2, label="region->part_of->slide"),
    ]
    ax3.legend(handles=legend_handles, loc="upper left", framealpha=0.95)

    edge_text = [
        f"patch edges: {int(p2p.shape[1])}",
        f"region edges: {int(r2r.shape[1])}",
        f"patch->region: {int(p2r.shape[1])}",
        f"subsampled patch edges: {int(patch_edges.shape[1])}",
        f"subsampled region edges: {int(region_edges.shape[1])}",
    ]
    ax3.text(
        0.02,
        0.58,
        "\n".join(edge_text),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
    )

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_title("Graph summary", fontsize=12)
    ax4.axis("off")
    summary = graph_structure_summary(data)
    meta = summary.get("meta", {}) if isinstance(summary.get("meta"), dict) else {}
    lines = [
        f"slide: {summary['node_counts'].get('slide', 0)}",
        f"region: {summary['node_counts'].get('region', 0)}",
        f"patch: {summary['node_counts'].get('patch', 0)}",
        "",
        f"coverage: {summary.get('patch_assignment_coverage', 0.0):.3f}",
        "",
        f"pipeline: {meta.get('pipeline', 'n/a')}",
        f"feature dim: {meta.get('feature_dim', 0)}",
    ]
    ax4.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
    )

    _save_figure(fig, out_path, viz_cfg, pad_inches=0.08)
    plt.close(fig)


def _plot_compact_overview(
    background: np.ndarray,
    patch_xy: np.ndarray,
    region_xy: np.ndarray,
    region_labels: np.ndarray,
    patch_edges,
    region_edges,
    out_path: Union[str, Path],
    summary: Dict[str, object],
    patch_edge_max: int,
    region_edge_max: int,
    seed: int,
    viz_cfg: VisualizationConfig,
) -> None:
    patch_edges = _to_numpy_edge_index(patch_edges)
    region_edges = _to_numpy_edge_index(region_edges)
    patch_edges_s = _sample_edges(patch_edges, min(int(patch_edge_max), 2500), seed)
    region_edges_s = _sample_edges(region_edges, min(int(region_edge_max), 1000), seed)

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.9], height_ratios=[1.0, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(background)
    ax.scatter(patch_xy[:, 0], patch_xy[:, 1], s=1.0, c="#8ecae6", linewidths=0, alpha=0.9)
    ax.set_title("Patch distribution")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(background)
    colors = _region_color_map(int(region_xy.shape[0]))
    ax.scatter(patch_xy[:, 0], patch_xy[:, 1], s=0.9, c=colors[region_labels] / 255.0, linewidths=0, alpha=0.8)
    ax.scatter(region_xy[:, 0], region_xy[:, 1], s=20, c="white", edgecolors="black", linewidths=0.55)
    ax.set_title("Region assignment")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 0:2])
    ax.imshow(background)
    for src, dst in patch_edges_s.T:
        x1, y1 = patch_xy[int(src)]
        x2, y2 = patch_xy[int(dst)]
        ax.plot([x1, x2], [y1, y2], color=viz_cfg.edge_color_patch_near, linewidth=0.18, alpha=0.22)
    for src, dst in region_edges_s.T:
        x1, y1 = region_xy[int(src)]
        x2, y2 = region_xy[int(dst)]
        ax.plot([x1, x2], [y1, y2], color=viz_cfg.edge_color_region_near, linewidth=0.65, alpha=0.75)
    ax.scatter(region_xy[:, 0], region_xy[:, 1], s=22, c="white", edgecolors="black", linewidths=0.6)
    ax.set_title("Edge overlay (subsampled)")
    ax.axis("off")

    ax = fig.add_subplot(gs[:, 2])
    ax.axis("off")
    node_counts = summary.get("node_counts", {})
    edge_counts = summary.get("edge_counts", {})
    meta = summary.get("meta", {}) if isinstance(summary.get("meta"), dict) else {}

    lines = [
        "Graph overview",
        "",
        f"Nodes: slide={node_counts.get('slide', 0)}, region={node_counts.get('region', 0)}, patch={node_counts.get('patch', 0)}",
        "",
        "Key edges:",
        f"- patch->near->patch: {edge_counts.get('patch->near->patch', 0)}",
        f"- region->near->region: {edge_counts.get('region->near->region', 0)}",
        f"- patch->in->region: {edge_counts.get('patch->in->region', 0)}",
        "",
        f"Feature model: {bool(meta.get('has_model_features', False))}",
        f"Feature dim: {int(meta.get('feature_dim', 0)) if meta.get('feature_dim') is not None else 0}",
        f"Pipeline: {meta.get('pipeline', 'n/a')}",
        "",
        f"Patch edges shown: {patch_edges_s.shape[1]}",
        f"Region edges shown: {region_edges_s.shape[1]}",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
    )

    _save_figure(fig, out_path, viz_cfg, pad_inches=0.1)
    plt.close(fig)


def _plot_compact_overview_cells(
    background: np.ndarray,
    cell_xy: np.ndarray,
    region_xy: np.ndarray,
    region_labels: np.ndarray,
    cell_edges,
    out_path: Union[str, Path],
    summary: Dict[str, object],
    cell_edge_max: int,
    seed: int,
    viz_cfg: VisualizationConfig,
) -> None:
    cell_edges = _to_numpy_edge_index(cell_edges)
    cell_edges_s = _sample_edges(cell_edges, min(int(cell_edge_max), 3500), seed)

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.9], height_ratios=[1.0, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(background)
    ax.scatter(cell_xy[:, 0], cell_xy[:, 1], s=2.2, c="#90be6d", linewidths=0, alpha=0.9)
    ax.set_title("Cell distribution")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(background)
    colors = _region_color_map(int(region_xy.shape[0]))
    if cell_xy.size > 0:
        ax.scatter(cell_xy[:, 0], cell_xy[:, 1], s=2.0, c=colors[region_labels] / 255.0, linewidths=0, alpha=0.8)
    ax.scatter(region_xy[:, 0], region_xy[:, 1], s=26, c="white", edgecolors="black", linewidths=0.6)
    ax.set_title("Region assignment for cells")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 0:2])
    ax.imshow(background)
    for src, dst in cell_edges_s.T:
        x1, y1 = cell_xy[int(src)]
        x2, y2 = cell_xy[int(dst)]
        ax.plot([x1, x2], [y1, y2], color="#f9c74f", linewidth=0.18, alpha=0.25)
    ax.scatter(region_xy[:, 0], region_xy[:, 1], s=24, c="white", edgecolors="black", linewidths=0.6)
    ax.set_title("Cell connectivity overlay (subsampled)")
    ax.axis("off")

    ax = fig.add_subplot(gs[:, 2])
    ax.axis("off")
    node_counts = summary.get("node_counts", {})
    edge_counts = summary.get("edge_counts", {})
    meta = summary.get("meta", {}) if isinstance(summary.get("meta"), dict) else {}

    lines = [
        "Graph overview (cell mode)",
        "",
        f"Nodes: slide={node_counts.get('slide', 0)}, region={node_counts.get('region', 0)}, cell={node_counts.get('cell', 0)}",
        "",
        "Key edges:",
        f"- cell->near->cell: {edge_counts.get('cell->near->cell', 0)}",
        f"- cell->in->region: {edge_counts.get('cell->in->region', 0)}",
        f"- region->has->cell: {edge_counts.get('region->has->cell', 0)}",
        "",
        f"Segmentation: {meta.get('segmentation_status', 'n/a')}",
        f"Cell feature dim: {meta.get('cell_feature_dim', 0)}",
        f"Cell edges shown: {cell_edges_s.shape[1]}",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
    )

    _save_figure(fig, out_path, viz_cfg, pad_inches=0.1)
    plt.close(fig)


def _plot_hierarchical_graph_cells(
    data: HeteroData,
    out_path: Union[str, Path],
    viz_cfg: VisualizationConfig,
    seed: int,
) -> None:
    cell_xy = data["cell"].centroid.cpu().numpy()
    region_xy = data["region"].centroid.cpu().numpy()
    if region_xy.size == 0:
        raise ValueError("Cannot visualize cell hierarchy without regions")

    c2r = _to_numpy_edge_index(data["cell", "in", "region"])
    c2c = _to_numpy_edge_index(data["cell", "near", "cell"])

    n_cells = int(data["cell"].num_nodes)
    n_regions = int(data["region"].num_nodes)

    region_palette = _region_color_map(max(1, n_regions))
    cell_colors = region_palette[c2r[1]] / 255.0 if c2r.size > 0 else np.full((n_cells, 3), 0.7, dtype=np.float32)

    usable_w = 2200.0 - 240.0
    x_min = 120.0
    x_max = x_min + usable_w
    slide_y = 210.0
    region_y0, region_y1 = 520.0, 940.0
    cell_y0, cell_y1 = 1080.0, 1520.0

    cell_x = _normalize_to_unit_interval(cell_xy[:, 0]) * usable_w + x_min if n_cells > 0 else np.zeros((0,), dtype=np.float32)
    cell_x = np.clip(cell_x, x_min + 8.0, x_max - 8.0) if n_cells > 0 else cell_x
    cell_y_norm = _normalize_to_unit_interval(cell_xy[:, 1]) if n_cells > 0 else np.zeros((0,), dtype=np.float32)
    cell_y = cell_y0 + 42.0 + cell_y_norm * max(1.0, (cell_y1 - cell_y0 - 84.0)) if n_cells > 0 else np.zeros((0,), dtype=np.float32)

    region_x = _normalize_to_unit_interval(region_xy[:, 0]) * usable_w + x_min
    region_x = np.clip(region_x, x_min + 8.0, x_max - 8.0)
    region_y_norm = _normalize_to_unit_interval(region_xy[:, 1])
    region_y = region_y0 + 70.0 + region_y_norm * max(1.0, (region_y1 - region_y0 - 140.0))

    cell_xy_h = np.stack([cell_x, cell_y], axis=1).astype(np.float32) if n_cells > 0 else np.zeros((0, 2), dtype=np.float32)
    region_xy_h = np.stack([region_x, region_y], axis=1).astype(np.float32)
    slide_xy_h = np.array([[1100.0, slide_y]], dtype=np.float32)

    cell_edges = _sample_edges(c2c, viz_cfg.hierarchy_edge_max, seed)

    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Hierarchical graph layout (cell mode)", fontsize=16, weight="bold")
    ax.set_xlim(0, 2200)
    ax.set_ylim(0, 1600)
    ax.invert_yaxis()
    ax.axis("off")

    band_specs = [
        (70, 360, "slide", "#ececec"),
        (520, 940, "regions", "#f7f7f7"),
        (1080, 1520, "cells", "#fcfcfc"),
    ]
    for y0, y1, label, color in band_specs:
        ax.add_patch(plt.Rectangle((40, y0), 2120, y1 - y0, facecolor=color, edgecolor="#d0d0d0", linewidth=0.9, zorder=0))
        ax.text(65, y0 + 10, label.upper(), fontsize=11, weight="bold", color="#555555", va="top", ha="left")

    slide_node = slide_xy_h[0]
    ax.scatter([slide_node[0]], [slide_node[1]], s=viz_cfg.hierarchy_slide_point_size, c="#c0392b", edgecolors="white", linewidths=1.1, zorder=5)

    for src, dst in cell_edges.T:
        x1, y1 = cell_xy_h[int(src)]
        x2, y2 = cell_xy_h[int(dst)]
        ax.plot([x1, x2], [y1, y2], color="#f9c74f", linewidth=max(0.12, viz_cfg.hierarchy_edge_lw * 0.55), alpha=0.12, zorder=1)

    for region_id, (x, y) in enumerate(region_xy_h):
        ax.scatter([x], [y], s=viz_cfg.hierarchy_region_point_size, c=[region_palette[region_id] / 255.0], edgecolors="#222222", linewidths=0.7, zorder=4)
        ax.plot([slide_node[0], x], [slide_node[1] + 58, y - 24], color=viz_cfg.edge_color_region_part_of, linewidth=0.65, alpha=0.42, zorder=2)

    if c2r.size > 0:
        for cell_idx, region_idx in zip(c2r[0].tolist(), c2r[1].tolist()):
            x1, y1 = cell_xy_h[int(cell_idx)]
            x2, y2 = region_xy_h[int(region_idx)]
            ax.plot([x1, x2], [y1 - 6, y2 + 6], color=viz_cfg.edge_color_patch_in_region, linewidth=0.25, alpha=0.18, zorder=1)

    if n_cells > 0:
        ax.scatter(cell_xy_h[:, 0], cell_xy_h[:, 1], s=max(2.0, viz_cfg.hierarchy_patch_point_size), c=cell_colors, edgecolors="none", alpha=0.9, zorder=3)

    ax.text(
        64,
        1545,
        f"cells={n_cells} | regions={n_regions} | cell->region edges={int(c2r.shape[1]) if c2r.size > 0 else 0}",
        fontsize=10,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.82, edgecolor="#cccccc"),
    )

    _save_figure(fig, out_path, viz_cfg, pad_inches=0.08)
    plt.close(fig)


def graph_structure_summary(data: HeteroData) -> Dict[str, object]:
    """Compute node/edge counts and coverage/meta stats summarizing a built graph."""
    summary: Dict[str, object] = {
        "node_counts": {node_type: int(data[node_type].num_nodes) for node_type in data.node_types},
        "edge_counts": {
            f"{edge_type[0]}->{edge_type[1]}->{edge_type[2]}": int(data[edge_type].edge_index.shape[1])
            for edge_type in data.edge_types
        },
    }

    if ("patch", "in", "region") in data.edge_types:
        p2r = data["patch", "in", "region"].edge_index.cpu().numpy()
        n_patches = int(data["patch"].num_nodes)
        valid_parent_links = int(np.unique(p2r[0]).size)
        summary["patch_assignment_coverage"] = float(valid_parent_links / max(1, n_patches))

    if ("cell", "in", "region") in data.edge_types:
        c2r = data["cell", "in", "region"].edge_index.cpu().numpy()
        n_cells = int(data["cell"].num_nodes)
        valid_parent_links = int(np.unique(c2r[0]).size)
        summary["cell_assignment_coverage"] = float(valid_parent_links / max(1, n_cells))

    if hasattr(data, "meta"):
        summary["meta"] = data.meta

    return summary


def visualize_graph_structure(
    data: HeteroData,
    out_dir: Union[str, Path] = "outputs/vis",
    slide_path: Optional[Union[str, Path]] = None,
    build_cfg: Optional[BuildConfig] = None,
    viz_cfg: Optional[VisualizationConfig] = None,
) -> Dict[str, object]:
    """Render structure-validation plots for a built graph and write them to out_dir."""
    if viz_cfg is None:
        viz_cfg = VisualizationConfig()

    _apply_publication_style(viz_cfg)

    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    has_patch_mode = (
        "patch" in data.node_types
        and ("patch", "near", "patch") in data.edge_types
        and ("patch", "in", "region") in data.edge_types
    )
    has_cell_mode = (
        "cell" in data.node_types
        and ("cell", "near", "cell") in data.edge_types
        and ("cell", "in", "region") in data.edge_types
    )

    if not has_patch_mode and not has_cell_mode:
        raise ValueError(
            "Graph visualization supports patch-mode or cell-mode graphs. "
            "Expected either patch or cell node/edge schema."
        )

    region_xy = data["region"].centroid.cpu().numpy()
    primary_xy = data["patch"].centroid.cpu().numpy() if has_patch_mode else data["cell"].centroid.cpu().numpy()

    background: np.ndarray
    patch_xy_vis: np.ndarray
    region_xy_vis: np.ndarray

    if slide_path is not None and build_cfg is not None:
        openslide = _import_openslide_optional()
        if openslide is None:
            raise RuntimeError("slide_path wurde gesetzt, aber openslide ist nicht verfügbar.")
        slide = openslide.OpenSlide(str(slide_path))
        try:
            thumb = _safe_thumbnail(slide, viz_cfg.thumbnail_max_size)
            primary_xy_vis = _coords_level_to_thumb(slide, build_cfg, thumb.shape[:2], primary_xy)
            region_xy_vis = _coords_level_to_thumb(slide, build_cfg, thumb.shape[:2], region_xy)
            background = thumb
            _save_img(out_dir / "thumbnail.png", thumb, viz_cfg)
        finally:
            slide.close()
    else:
        background, primary_xy_vis, region_xy_vis = _make_canvas_from_points(primary_xy, region_xy)
        _save_img(out_dir / "thumbnail.png", background, viz_cfg)

    summary = graph_structure_summary(data)

    if has_patch_mode:
        patch_xy_vis = primary_xy_vis

        _plot_points(
            background,
            patch_xy_vis,
            out_dir / "patches.png",
            title=f"Patches (N={int(data['patch'].num_nodes)})",
            size=viz_cfg.patch_point_size,
            color="#8ecae6",
            viz_cfg=viz_cfg,
        )

        p2r = _to_numpy_edge_index(data["patch", "in", "region"])
        region_labels = np.zeros((int(data["patch"].num_nodes),), dtype=np.int64)
        region_labels[p2r[0]] = p2r[1]

        _plot_regions_colored(
            background,
            patch_xy_vis,
            region_labels,
            region_xy_vis,
            out_dir / "regions.png",
            title=f"Regions (K={int(data['region'].num_nodes)}) + Patch assignment",
            patch_size=1.5,
            region_size=viz_cfg.region_point_size,
            viz_cfg=viz_cfg,
        )

        p2p = _to_numpy_edge_index(data["patch", "near", "patch"])
        _plot_edges(
            background,
            patch_xy_vis,
            p2p,
            out_dir / "patch_knn_edges.png",
            title="Patch kNN edges (subsampled)",
            max_edges=viz_cfg.patch_edge_max,
            lw=viz_cfg.patch_edge_lw,
            color=viz_cfg.edge_color_patch_near,
            seed=viz_cfg.seed,
            viz_cfg=viz_cfg,
        )

        r2r = _to_numpy_edge_index(data["region", "near", "region"])
        _plot_edges(
            background,
            region_xy_vis,
            r2r,
            out_dir / "region_knn_edges.png",
            title="Region kNN edges",
            max_edges=viz_cfg.region_edge_max,
            lw=viz_cfg.region_edge_lw,
            color=viz_cfg.edge_color_region_near,
            seed=viz_cfg.seed,
            viz_cfg=viz_cfg,
        )

        _plot_hierarchy(
            background,
            region_xy_vis,
            out_dir / "hierarchy_overlay.png",
            title="Hierarchy: region -> slide connections (visual)",
            viz_cfg=viz_cfg,
        )

        _plot_compact_overview(
            background=background,
            patch_xy=patch_xy_vis,
            region_xy=region_xy_vis,
            region_labels=region_labels,
            patch_edges=p2p,
            region_edges=r2r,
            out_path=out_dir / "overview.png",
            summary=summary,
            patch_edge_max=viz_cfg.patch_edge_max,
            region_edge_max=viz_cfg.region_edge_max,
            seed=viz_cfg.seed,
            viz_cfg=viz_cfg,
        )

        _plot_hierarchical_graph(
            data=data,
            out_path=out_dir / "hierarchical_graph.png",
            viz_cfg=viz_cfg,
            seed=viz_cfg.seed,
        )

        _plot_multiedge_overlay(
            background=background,
            patch_xy=patch_xy_vis,
            region_xy=region_xy_vis,
            p2p=p2p,
            r2r=r2r,
            p2r=p2r,
            out_path=out_dir / "edge_type_overlay.png",
            viz_cfg=viz_cfg,
            seed=viz_cfg.seed,
        )

        _plot_graph_insights(
            patch_xy=patch_xy_vis,
            region_xy=region_xy_vis,
            p2p=p2p,
            r2r=r2r,
            p2r=p2r,
            out_path=out_dir / "graph_insights.png",
            viz_cfg=viz_cfg,
        )
    else:
        cell_xy_vis = primary_xy_vis

        _plot_points(
            background,
            cell_xy_vis,
            out_dir / "cells.png",
            title=f"Cells (N={int(data['cell'].num_nodes)})",
            size=max(2.0, viz_cfg.patch_point_size * 1.4),
            color="#90be6d",
            viz_cfg=viz_cfg,
        )

        c2r = _to_numpy_edge_index(data["cell", "in", "region"])
        n_cells = int(data["cell"].num_nodes)
        if hasattr(data["cell"], "region_id"):
            region_labels = data["cell"].region_id.cpu().numpy().astype(np.int64)
        else:
            region_labels = np.zeros((n_cells,), dtype=np.int64)
            if c2r.size > 0:
                region_labels[c2r[0]] = c2r[1]

        _plot_regions_colored(
            background,
            cell_xy_vis,
            region_labels,
            region_xy_vis,
            out_dir / "regions.png",
            title=f"Regions (K={int(data['region'].num_nodes)}) + Cell assignment",
            patch_size=max(1.8, viz_cfg.patch_point_size * 0.9),
            region_size=viz_cfg.region_point_size,
            viz_cfg=viz_cfg,
        )

        c2c = _to_numpy_edge_index(data["cell", "near", "cell"])
        _plot_edges(
            background,
            cell_xy_vis,
            c2c,
            out_dir / "cell_knn_edges.png",
            title="Cell kNN edges (subsampled)",
            max_edges=viz_cfg.patch_edge_max,
            lw=max(0.18, viz_cfg.patch_edge_lw),
            color=viz_cfg.edge_color_patch_near,
            seed=viz_cfg.seed,
            viz_cfg=viz_cfg,
        )

        _plot_hierarchy(
            background,
            region_xy_vis,
            out_dir / "hierarchy_overlay.png",
            title="Hierarchy: region -> slide connections (visual)",
            viz_cfg=viz_cfg,
        )

        _plot_compact_overview_cells(
            background=background,
            cell_xy=cell_xy_vis,
            region_xy=region_xy_vis,
            region_labels=region_labels,
            cell_edges=c2c,
            out_path=out_dir / "overview.png",
            summary=summary,
            cell_edge_max=viz_cfg.patch_edge_max,
            seed=viz_cfg.seed,
            viz_cfg=viz_cfg,
        )

        _plot_hierarchical_graph_cells(
            data=data,
            out_path=out_dir / "hierarchical_graph.png",
            viz_cfg=viz_cfg,
            seed=viz_cfg.seed,
        )

    summary_payload = {
        "summary": summary,
        "build_config": asdict(build_cfg) if build_cfg is not None else None,
        "visualization_config": asdict(viz_cfg),
        "slide_path": str(slide_path) if slide_path is not None else None,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)

    with open(out_dir / "meta.txt", "w", encoding="utf-8") as handle:
        handle.write("Graph visualization summary\n")
        handle.write(json.dumps(summary, indent=2, ensure_ascii=False))

    return summary


__all__ = [
    "VisualizationConfig",
    "graph_structure_summary",
    "_plot_hierarchical_graph",
    "plot_region_cell_debug",
    "debug_plot_region_cells",
    "visualize_graph_structure",
]
