from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
import logging
import time
import importlib.metadata
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData
from skimage.measure import regionprops

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.graph_creation_new.config import BuildContext
from src.graph_creation_new.helper.edges import build_edges, build_edge_features
from src.graph_creation_new.helper.slide import iter_tissue_region_crops


class _CellposeNoiseFilter(logging.Filter):
    """Drop known repetitive Cellpose messages that flood long runs."""

    _DROP_SUBSTRINGS = (
        "channels deprecated in v4.0.1+",
        "No cell pixels found.",
        "no seeds found in get_masks_torch - no masks found.",
        "model_type argument is not used in v4.0.1+",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False for log records matching a known noisy Cellpose message."""
        msg = record.getMessage()
        return not any(token in msg for token in self._DROP_SUBSTRINGS)


def _region_centroids_at_patch_level(context: BuildContext) -> np.ndarray:
    regions = getattr(context, "tissue_regions", None) or []
    if not regions:
        return np.zeros((0, 2), dtype=np.float32)

    patch_downsample = float(context.slide.level_downsamples[int(context.cfg.patch_level)])
    centroids: List[np.ndarray] = []
    for region in regions:
        centroid_level0 = np.asarray(region.get("centroid_level0_xy", (0.0, 0.0)), dtype=np.float32)
        centroids.append((centroid_level0 / max(1e-6, patch_downsample)).astype(np.float32))

    return np.asarray(centroids, dtype=np.float32)


def _version_tuple(version_text: str) -> Tuple[int, int, int]:
    chunks = str(version_text).split(".")
    parsed: List[int] = []
    for chunk in chunks[:3]:
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parsed.append(int(digits) if digits else 0)
    while len(parsed) < 3:
        parsed.append(0)
    return (parsed[0], parsed[1], parsed[2])


def _supports_legacy_cellpose_api(version_text: str) -> bool:
    normalized = str(version_text or "").strip().lower()
    if normalized in {"", "0.0.0", "unknown", "none"}:
        return False
    return _version_tuple(normalized) < (4, 0, 1)


@lru_cache(maxsize=1)
def _cellpose_runtime() -> Tuple[Any, str]:
    try:
        import cellpose
        from cellpose import models
    except Exception as exc:
        raise RuntimeError(
            "cellpose konnte nicht importiert werden. Bitte `cellpose` installieren oder node_creation_mode auf patch_encoder setzen."
        ) from exc

    version = str(getattr(cellpose, "__version__", "")).strip()
    if not version:
        try:
            version = str(importlib.metadata.version("cellpose"))
        except Exception:
            version = "0.0.0"

    logging.getLogger().addFilter(_CellposeNoiseFilter())

    return models, version


def _build_cellpose_model(context: BuildContext):
    models, cellpose_version = _cellpose_runtime()
    supports_legacy_args = _supports_legacy_cellpose_api(cellpose_version)
    use_gpu = bool(context.creation_cfg.cell_use_gpu_if_available and torch.cuda.is_available())
    model_type = str(context.creation_cfg.cellpose_model_type).strip() or "cyto3"
    if supports_legacy_args:
        logging.info("Cellpose setup: version=%s model_type=%s gpu=%s", cellpose_version, model_type, use_gpu)
        return models.CellposeModel(gpu=use_gpu, model_type=model_type)

    logging.info("Cellpose setup: version=%s gpu=%s", cellpose_version, use_gpu)
    return models.CellposeModel(gpu=use_gpu)


def _cellpose_eval_kwargs(context: BuildContext) -> Dict[str, Any]:
    cfg = context.creation_cfg
    _, cellpose_version = _cellpose_runtime()
    supports_legacy_channels = _supports_legacy_cellpose_api(cellpose_version)

    kwargs: Dict[str, Any] = {}
    if supports_legacy_channels:
        kwargs["channels"] = [int(cfg.cellpose_channels[0]), int(cfg.cellpose_channels[1])]
    if cfg.cellpose_diameter is not None:
        kwargs["diameter"] = float(cfg.cellpose_diameter)
    if cfg.cellpose_flow_threshold is not None:
        kwargs["flow_threshold"] = float(cfg.cellpose_flow_threshold)
    if cfg.cellpose_cellprob_threshold is not None:
        kwargs["cellprob_threshold"] = float(cfg.cellpose_cellprob_threshold)
    return kwargs


def _segment_cells_in_region(
    image_rgb: np.ndarray,
    model,
    eval_kwargs: Dict[str, Any],
    min_area_px: int,
) -> Tuple[np.ndarray, np.ndarray]:
    masks, _, _ = model.eval(image_rgb, **eval_kwargs)
    props = regionprops(masks)

    if not props:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    centroids: List[Tuple[float, float]] = []
    areas: List[float] = []

    for p in props:
        area = float(p.area)
        if area < float(min_area_px):
            continue
        centroids.append((float(p.centroid[1]), float(p.centroid[0])))
        areas.append(area)

    if not centroids:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return np.asarray(centroids, dtype=np.float32), np.asarray(areas, dtype=np.float32)


def _tile_starts(length: int, tile_size: int, step: int) -> List[int]:
    if length <= tile_size:
        return [0]

    starts = list(range(0, max(1, length - tile_size + 1), max(1, step)))
    last = max(0, length - tile_size)
    if starts[-1] != last:
        starts.append(last)
    return starts


def _segment_cells_in_region_chunked(
    image_rgb: np.ndarray,
    model,
    eval_kwargs: Dict[str, Any],
    min_area_px: int,
    tile_size: Optional[int],
    tile_overlap: int,
    border_margin: Optional[int],
    tile_progress_callback: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = int(image_rgb.shape[0]), int(image_rgb.shape[1])
    if height <= 0 or width <= 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    if tile_size is None:
        return _segment_cells_in_region(image_rgb, model, eval_kwargs, min_area_px)

    tile = int(tile_size)
    if tile <= 0 or (width <= tile and height <= tile):
        return _segment_cells_in_region(image_rgb, model, eval_kwargs, min_area_px)

    overlap = max(0, int(tile_overlap))
    step = max(1, tile - overlap)
    margin = int(border_margin) if border_margin is not None else max(0, overlap // 2)

    all_centroids: List[Tuple[float, float]] = []
    all_areas: List[float] = []

    y_starts = _tile_starts(height, tile, step)
    x_starts = _tile_starts(width, tile, step)
    total_tiles = max(1, len(y_starts) * len(x_starts))
    processed_tiles = 0

    for y0 in y_starts:
        for x0 in x_starts:
            processed_tiles += 1
            if tile_progress_callback is not None:
                tile_progress_callback(processed_tiles, total_tiles)

            y1 = min(height, y0 + tile)
            x1 = min(width, x0 + tile)
            chunk = image_rgb[y0:y1, x0:x1]
            if chunk.size == 0:
                continue

            local_xy, local_area = _segment_cells_in_region(
                image_rgb=chunk,
                model=model,
                eval_kwargs=eval_kwargs,
                min_area_px=min_area_px,
            )
            if local_xy.size == 0:
                continue

            chunk_h, chunk_w = int(chunk.shape[0]), int(chunk.shape[1])
            x_min_keep = 0.0 if x0 == 0 else float(margin)
            y_min_keep = 0.0 if y0 == 0 else float(margin)
            x_max_keep = float(chunk_w) if x1 >= width else float(max(0, chunk_w - margin))
            y_max_keep = float(chunk_h) if y1 >= height else float(max(0, chunk_h - margin))

            keep_mask = (
                (local_xy[:, 0] >= x_min_keep)
                & (local_xy[:, 0] < x_max_keep)
                & (local_xy[:, 1] >= y_min_keep)
                & (local_xy[:, 1] < y_max_keep)
            )
            if not np.any(keep_mask):
                continue

            kept_xy = local_xy[keep_mask].copy()
            kept_xy[:, 0] += float(x0)
            kept_xy[:, 1] += float(y0)
            kept_area = local_area[keep_mask]

            all_centroids.extend((float(x), float(y)) for x, y in kept_xy)
            all_areas.extend(float(a) for a in kept_area)

    if not all_centroids:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return np.asarray(all_centroids, dtype=np.float32), np.asarray(all_areas, dtype=np.float32)


def _global_patch_level_xy(
    local_xy: np.ndarray,
    bbox_level0_xyxy: Tuple[int, int, int, int],
    region_level_downsample: float,
    patch_downsample: float,
) -> np.ndarray:
    x0, y0, _, _ = bbox_level0_xyxy
    level0_x = float(x0) + local_xy[:, 0] * float(region_level_downsample)
    level0_y = float(y0) + local_xy[:, 1] * float(region_level_downsample)
    global_xy = np.stack([level0_x, level0_y], axis=1) / max(1e-6, float(patch_downsample))
    return global_xy.astype(np.float32)


def _resolve_segmentation_level(context: BuildContext, segmentation_level: Optional[int]) -> int:
    level = int(
        segmentation_level
        if segmentation_level is not None
        else (
            context.creation_cfg.cell_segmentation_level
            if context.creation_cfg.cell_segmentation_level is not None
            else context.cfg.patch_level
        )
    )
    if level < 0 or level >= int(context.slide.level_count):
        raise ValueError(
            f"cell_segmentation_level={level} invalid for slide with level_count={int(context.slide.level_count)}"
        )
    return level


def build_region_cell_debug_payload(
    context: BuildContext,
    region_id: int,
    segmentation_level: Optional[int] = None,
) -> Dict[str, Any]:
    """Run cell segmentation for exactly one tissue region and return debug payload."""
    level = _resolve_segmentation_level(context, segmentation_level)

    selected_crop: Optional[Dict[str, Any]] = None
    for crop in iter_tissue_region_crops(
        context=context,
        level=level,
        apply_mask=True,
        min_area_level0_px=None,
    ) or []:
        if int(crop["region_id"]) == int(region_id):
            selected_crop = crop
            break

    if selected_crop is None:
        available_regions = [
            int(region.get("region_id", -1))
            for region in (getattr(context, "tissue_regions", None) or [])
        ]
        raise ValueError(
            f"Region {region_id} not found. Available region_ids: {available_regions[:50]}"
        )

    cell_model = _build_cellpose_model(context)
    eval_kwargs = _cellpose_eval_kwargs(context)
    min_area_px = int(context.creation_cfg.cell_min_area_px)
    tile_size = context.creation_cfg.cell_segmentation_tile_size
    tile_overlap = int(context.creation_cfg.cell_segmentation_tile_overlap)
    border_margin = context.creation_cfg.cell_segmentation_border_margin

    logging.info(
        "Cell segmentation tiling: tile_size=%s overlap=%d border_margin=%s",
        str(tile_size),
        tile_overlap,
        str(border_margin),
    )

    masked_rgb = np.asarray(selected_crop["masked_rgb"], dtype=np.uint8)
    local_xy, cell_area = _segment_cells_in_region_chunked(
        image_rgb=masked_rgb,
        model=cell_model,
        eval_kwargs=eval_kwargs,
        min_area_px=min_area_px,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        border_margin=border_margin,
        tile_progress_callback=None,
    )

    cell_k = int(context.creation_cfg.cell_k) if context.creation_cfg.cell_k is not None else int(context.cfg.k_patch)
    c2c_src, c2c_dst, c2c_dist = build_edges(
        xy=local_xy,
        mode=context.creation_cfg.cell_cell_edge_mode,
        k=cell_k,
        radius_px=context.creation_cfg.cell_cell_distance_radius_px,
        semantic_ids=np.zeros((int(local_xy.shape[0]),), dtype=np.int64),
        relation_name="cell_near_cell_debug",
    )

    patch_downsample = float(context.slide.level_downsamples[int(context.cfg.patch_level)])
    global_xy_patch_level = _global_patch_level_xy(
        local_xy=local_xy,
        bbox_level0_xyxy=tuple(selected_crop["bbox_level0_xyxy"]),
        region_level_downsample=float(selected_crop["downsample"]),
        patch_downsample=patch_downsample,
    )

    return {
        "region_id": int(region_id),
        "segmentation_level": int(level),
        "image_rgb": masked_rgb,
        "local_xy": local_xy,
        "global_xy_patch_level": global_xy_patch_level,
        "cell_area": cell_area,
        "edge_src": c2c_src,
        "edge_dst": c2c_dst,
        "edge_dist": c2c_dist,
        "cell_k": int(cell_k),
        "bbox_level0_xyxy": tuple(selected_crop["bbox_level0_xyxy"]),
        "cellpose_model_type": str(context.creation_cfg.cellpose_model_type),
        "cellpose_channels": tuple(int(x) for x in context.creation_cfg.cellpose_channels),
        "cellpose_diameter": context.creation_cfg.cellpose_diameter,
        "cellpose_flow_threshold": context.creation_cfg.cellpose_flow_threshold,
        "cellpose_cellprob_threshold": context.creation_cfg.cellpose_cellprob_threshold,
        "cell_min_area_px": int(min_area_px),
        "cell_segmentation_tile_size": tile_size,
        "cell_segmentation_tile_overlap": tile_overlap,
        "cell_segmentation_border_margin": border_margin,
    }


def create_cell_subgraph_structure(context: BuildContext) -> None:
    """Create a cell-level graph by running cell segmentation on tissue regions."""
    region_xy = _region_centroids_at_patch_level(context)
    n_regions = int(region_xy.shape[0])
    segmentation_level = _resolve_segmentation_level(context, segmentation_level=None)

    cell_model = _build_cellpose_model(context)
    eval_kwargs = _cellpose_eval_kwargs(context)
    patch_downsample = float(context.slide.level_downsamples[int(context.cfg.patch_level)])
    min_area_px = int(context.creation_cfg.cell_min_area_px)
    tile_size = context.creation_cfg.cell_segmentation_tile_size
    tile_overlap = int(context.creation_cfg.cell_segmentation_tile_overlap)
    border_margin = context.creation_cfg.cell_segmentation_border_margin

    all_cell_xy: List[np.ndarray] = []
    all_cell_area: List[np.ndarray] = []
    all_cell_region: List[np.ndarray] = []
    segmented_regions = 0
    slide_name = Path(str(context.slide_path)).name
    total_regions = max(1, n_regions)
    next_progress_pct = 10

    logging.info(
        "Cell segmentation progress [%s]: 0%% (%d/%d regions)",
        slide_name,
        0,
        total_regions,
    )

    for crop in iter_tissue_region_crops(
        context=context,
        level=segmentation_level,
        apply_mask=True,
        min_area_level0_px=None,
    ) or []:
        region_id = int(crop["region_id"])
        masked_rgb = np.asarray(crop["masked_rgb"], dtype=np.uint8)
        region_start = time.perf_counter()
        region_next_progress_pct = 25

        logging.info(
            "Cell segmentation region [%s | region=%d]: start",
            slide_name,
            region_id,
        )

        def _region_tile_progress(done_tiles: int, total_tiles_for_region: int) -> None:
            nonlocal region_next_progress_pct
            region_progress_pct = int((100.0 * float(done_tiles)) / float(max(1, total_tiles_for_region)))
            while region_progress_pct >= region_next_progress_pct and region_next_progress_pct <= 100:
                logging.info(
                    "Cell segmentation region [%s | region=%d]: %d%% (%d/%d tiles)",
                    slide_name,
                    region_id,
                    region_next_progress_pct,
                    done_tiles,
                    total_tiles_for_region,
                )
                region_next_progress_pct += 25

        local_xy, cell_area = _segment_cells_in_region_chunked(
            image_rgb=masked_rgb,
            model=cell_model,
            eval_kwargs=eval_kwargs,
            min_area_px=min_area_px,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            border_margin=border_margin,
            tile_progress_callback=_region_tile_progress,
        )
        if local_xy.size == 0:
            segmented_regions += 1
            region_elapsed = time.perf_counter() - region_start
            logging.info(
                "Cell segmentation region [%s | region=%d]: done, cells=0, elapsed=%.1fs",
                slide_name,
                region_id,
                region_elapsed,
            )
            progress_pct = int((100.0 * float(segmented_regions)) / float(total_regions))
            while progress_pct >= next_progress_pct and next_progress_pct <= 100:
                logging.info(
                    "Cell segmentation progress [%s]: %d%% (%d/%d regions)",
                    slide_name,
                    next_progress_pct,
                    segmented_regions,
                    total_regions,
                )
                next_progress_pct += 10
            continue

        global_xy = _global_patch_level_xy(
            local_xy=local_xy,
            bbox_level0_xyxy=tuple(crop["bbox_level0_xyxy"]),
            region_level_downsample=float(crop["downsample"]),
            patch_downsample=patch_downsample,
        )

        all_cell_xy.append(global_xy)
        all_cell_area.append(cell_area)
        all_cell_region.append(np.full((global_xy.shape[0],), region_id, dtype=np.int64))
        segmented_regions += 1
        region_elapsed = time.perf_counter() - region_start
        logging.info(
            "Cell segmentation region [%s | region=%d]: done, cells=%d, elapsed=%.1fs",
            slide_name,
            region_id,
            int(global_xy.shape[0]),
            region_elapsed,
        )
        progress_pct = int((100.0 * float(segmented_regions)) / float(total_regions))
        while progress_pct >= next_progress_pct and next_progress_pct <= 100:
            logging.info(
                "Cell segmentation progress [%s]: %d%% (%d/%d regions)",
                slide_name,
                next_progress_pct,
                segmented_regions,
                total_regions,
            )
            next_progress_pct += 10

    if next_progress_pct <= 100:
        logging.info(
            "Cell segmentation progress [%s]: 100%% (%d/%d regions)",
            slide_name,
            segmented_regions,
            total_regions,
        )

    if all_cell_xy:
        cell_xy = np.concatenate(all_cell_xy, axis=0).astype(np.float32)
        cell_area = np.concatenate(all_cell_area, axis=0).astype(np.float32)
        cell_region = np.concatenate(all_cell_region, axis=0).astype(np.int64)
    else:
        cell_xy = np.zeros((0, 2), dtype=np.float32)
        cell_area = np.zeros((0,), dtype=np.float32)
        cell_region = np.zeros((0,), dtype=np.int64)

    n_cells = int(cell_xy.shape[0])

    cell_k = int(context.creation_cfg.cell_k) if context.creation_cfg.cell_k is not None else int(context.cfg.k_patch)
    c2c_src, c2c_dst, c2c_dist = build_edges(
        xy=cell_xy,
        mode=context.creation_cfg.cell_cell_edge_mode,
        k=cell_k,
        radius_px=context.creation_cfg.cell_cell_distance_radius_px,
        semantic_ids=cell_region,
        relation_name="cell_near_cell",
    )

    data = HeteroData()
    data.meta = {}

    data["slide"].num_nodes = 1
    data["slide"].x = torch.zeros((1, 1), dtype=torch.float32)
    data["slide"].centroid = torch.tensor(
        [[context.slide.dimensions[0] / 2.0, context.slide.dimensions[1] / 2.0]],
        dtype=torch.float32,
    )

    data["region"].num_nodes = n_regions
    data["region"].x = torch.tensor(region_xy, dtype=torch.float32) if n_regions > 0 else torch.zeros((0, 2), dtype=torch.float32)
    data["region"].centroid = data["region"].x.clone()

    data["cell"].num_nodes = n_cells
    if n_cells > 0:
        cell_spatial = np.concatenate([cell_xy, cell_area[:, None]], axis=1)
        data["cell"].x = torch.tensor(cell_spatial, dtype=torch.float32)
        data["cell"].centroid = torch.tensor(cell_xy, dtype=torch.float32)
        data["cell"].area = torch.tensor(cell_area[:, None], dtype=torch.float32)
        data["cell"].region_id = torch.tensor(cell_region, dtype=torch.long)
    else:
        data["cell"].x = torch.zeros((0, 3), dtype=torch.float32)
        data["cell"].centroid = torch.zeros((0, 2), dtype=torch.float32)
        data["cell"].area = torch.zeros((0, 1), dtype=torch.float32)
        data["cell"].region_id = torch.zeros((0,), dtype=torch.long)

    empty_edges = torch.zeros((2, 0), dtype=torch.long)
    empty_attr = torch.zeros((0, 1), dtype=torch.float32)
    empty_feat = torch.zeros((0, 6), dtype=torch.float32)

    r2r_src, r2r_dst, r2r_dist = build_edges(
        xy=region_xy,
        mode=context.creation_cfg.region_region_edge_mode,
        k=min(int(context.cfg.k_region), max(1, n_regions - 1)),
        radius_px=context.creation_cfg.region_region_distance_radius_px,
        relation_name="region_near_region",
    )

    if r2r_src.size > 0:
        data["region", "near", "region"].edge_index = torch.tensor(np.stack([r2r_src, r2r_dst], axis=0), dtype=torch.long)
        data["region", "near", "region"].edge_attr = torch.tensor(r2r_dist[:, None], dtype=torch.float32)
        data["region", "near", "region"].edge_feat = torch.tensor(
            build_edge_features(
                src_xy=region_xy,
                dst_xy=region_xy,
                src_idx=r2r_src,
                dst_idx=r2r_dst,
                dist=r2r_dist,
                src_semantic_ids=np.arange(n_regions, dtype=np.int64),
                dst_semantic_ids=np.arange(n_regions, dtype=np.int64),
            ),
            dtype=torch.float32,
        )
    else:
        data["region", "near", "region"].edge_index = empty_edges.clone()
        data["region", "near", "region"].edge_attr = empty_attr.clone()
        data["region", "near", "region"].edge_feat = empty_feat.clone()

    if c2c_src.size > 0:
        data["cell", "near", "cell"].edge_index = torch.tensor(np.stack([c2c_src, c2c_dst], axis=0), dtype=torch.long)
        data["cell", "near", "cell"].edge_attr = torch.tensor(c2c_dist[:, None], dtype=torch.float32)
        data["cell", "near", "cell"].edge_feat = torch.tensor(
            build_edge_features(
                src_xy=cell_xy,
                dst_xy=cell_xy,
                src_idx=c2c_src,
                dst_idx=c2c_dst,
                dist=c2c_dist,
                src_semantic_ids=cell_region,
                dst_semantic_ids=cell_region,
            ),
            dtype=torch.float32,
        )
    else:
        data["cell", "near", "cell"].edge_index = empty_edges.clone()
        data["cell", "near", "cell"].edge_attr = empty_attr.clone()
        data["cell", "near", "cell"].edge_feat = empty_feat.clone()

    if n_cells > 0:
        c2r_src = np.arange(n_cells, dtype=np.int64)
        c2r_dst = cell_region.astype(np.int64)
        r2c_src = c2r_dst.copy()
        r2c_dst = c2r_src.copy()
        data["cell", "in", "region"].edge_index = torch.tensor(np.stack([c2r_src, c2r_dst], axis=0), dtype=torch.long)
        data["region", "has", "cell"].edge_index = torch.tensor(np.stack([r2c_src, r2c_dst], axis=0), dtype=torch.long)
        data["cell", "in", "region"].edge_feat = torch.tensor(
            build_edge_features(
                src_xy=cell_xy,
                dst_xy=region_xy,
                src_idx=c2r_src,
                dst_idx=c2r_dst,
                src_semantic_ids=cell_region,
                dst_semantic_ids=np.arange(n_regions, dtype=np.int64),
            ),
            dtype=torch.float32,
        )
        data["region", "has", "cell"].edge_feat = torch.tensor(
            build_edge_features(
                src_xy=region_xy,
                dst_xy=cell_xy,
                src_idx=r2c_src,
                dst_idx=r2c_dst,
                src_semantic_ids=np.arange(n_regions, dtype=np.int64),
                dst_semantic_ids=cell_region,
            ),
            dtype=torch.float32,
        )
    else:
        data["cell", "in", "region"].edge_index = empty_edges.clone()
        data["region", "has", "cell"].edge_index = empty_edges.clone()
        data["cell", "in", "region"].edge_feat = empty_feat.clone()
        data["region", "has", "cell"].edge_feat = empty_feat.clone()

    if n_regions > 0:
        r2s_src = torch.arange(n_regions, dtype=torch.long)
        r2s_dst = torch.zeros((n_regions,), dtype=torch.long)
        s2r_src = torch.zeros((n_regions,), dtype=torch.long)
        s2r_dst = torch.arange(n_regions, dtype=torch.long)
        data["region", "part_of", "slide"].edge_index = torch.stack([r2s_src, r2s_dst], dim=0)
        data["slide", "contains", "region"].edge_index = torch.stack([s2r_src, s2r_dst], dim=0)
        r2s_src_np = r2s_src.cpu().numpy().astype(np.int64)
        r2s_dst_np = r2s_dst.cpu().numpy().astype(np.int64)
        s2r_src_np = s2r_src.cpu().numpy().astype(np.int64)
        s2r_dst_np = s2r_dst.cpu().numpy().astype(np.int64)
        slide_xy = data["slide"].centroid.cpu().numpy()
        data["region", "part_of", "slide"].edge_feat = torch.tensor(
            build_edge_features(
                src_xy=region_xy,
                dst_xy=slide_xy,
                src_idx=r2s_src_np,
                dst_idx=r2s_dst_np,
                src_semantic_ids=np.arange(n_regions, dtype=np.int64),
                dst_semantic_ids=np.zeros((1,), dtype=np.int64),
            ),
            dtype=torch.float32,
        )
        data["slide", "contains", "region"].edge_feat = torch.tensor(
            build_edge_features(
                src_xy=slide_xy,
                dst_xy=region_xy,
                src_idx=s2r_src_np,
                dst_idx=s2r_dst_np,
                src_semantic_ids=np.zeros((1,), dtype=np.int64),
                dst_semantic_ids=np.arange(n_regions, dtype=np.int64),
            ),
            dtype=torch.float32,
        )
    else:
        data["region", "part_of", "slide"].edge_index = empty_edges.clone()
        data["slide", "contains", "region"].edge_index = empty_edges.clone()
        data["region", "part_of", "slide"].edge_feat = empty_feat.clone()
        data["slide", "contains", "region"].edge_feat = empty_feat.clone()

    data.meta.update({
        "wsi_path": context.slide_path,
        "config": asdict(context.cfg),
        "creation_config": asdict(context.creation_cfg),
        "pipeline": str(context.cfg.pipeline),
        "node_creation_mode": "cell_segmentation",
        "cell_structure_only": False,
        "num_regions": n_regions,
        "num_cells": n_cells,
        "num_cell_edges": int(c2c_src.shape[0]),
        "num_region_edges": int(r2r_src.shape[0]),
        "region_region_edge_mode": str(context.creation_cfg.region_region_edge_mode.value),
        "cell_cell_edge_mode": str(context.creation_cfg.cell_cell_edge_mode.value),
        "cell_cell_semantic_source": "region_id" if context.creation_cfg.cell_cell_edge_mode.value == "semantic_equality" else "none",
        "edge_feat_dim": 6,
        "segmentation_status": "cellpose",
        "cell_segmentation_level": int(segmentation_level),
        "cell_segmentation_tile_size": tile_size,
        "cell_segmentation_tile_overlap": tile_overlap,
        "cell_segmentation_border_margin": border_margin,
        "segmented_regions": int(segmented_regions),
        "cell_feature_dim": int(data["cell"].x.shape[1]),
    })

    context.graph_data = data
