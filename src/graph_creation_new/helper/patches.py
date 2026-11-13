from __future__ import annotations

from pathlib import Path
import sys
import logging
from typing import List, Sequence, Tuple

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.graph_creation_new.config import BuildContext
from src.graph_creation_new.error import MissingAnnotationPolygonsError
from src.graph_creation_new.error import NoTissuePatchesFoundError
from src.graph_creation_new.helper.slide import resolve_patch_size_px, resolve_stride_px
def _patch_fraction(mask_thumb: np.ndarray, x: int, y: int, window: int = 15) -> float:
    height, width = mask_thumb.shape
    radius = max(1, int(window) // 2)
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    patch = mask_thumb[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(patch.mean())


def _patch_fraction_for_patch(
    mask_thumb: np.ndarray,
    top_left_level0: Tuple[float, float],
    patch_size_level0: float,
    slide_size: Tuple[int, int],
) -> float:
    slide_w0, slide_h0 = slide_size
    thumb_h, thumb_w = mask_thumb.shape
    scale_x = thumb_w / float(slide_w0)
    scale_y = thumb_h / float(slide_h0)

    x0, y0 = top_left_level0
    x1 = x0 + patch_size_level0
    y1 = y0 + patch_size_level0

    tx0 = max(0, min(thumb_w, int(np.floor(x0 * scale_x))))
    ty0 = max(0, min(thumb_h, int(np.floor(y0 * scale_y))))
    tx1 = max(0, min(thumb_w, int(np.ceil(x1 * scale_x))))
    ty1 = max(0, min(thumb_h, int(np.ceil(y1 * scale_y))))

    if tx1 <= tx0 or ty1 <= ty0:
        return 0.0

    patch = mask_thumb[ty0:ty1, tx0:tx1]
    if patch.size == 0:
        return 0.0
    return float(patch.mean())


def _annotation_polygons(context: BuildContext) -> List[np.ndarray]:
    annotations = getattr(context, "annotations", None) or []
    polygons: List[np.ndarray] = []
    for annotation in annotations:
        coordinates = annotation.get("coordinates") or []
        if len(coordinates) < 3:
            continue
        polygons.append(np.asarray(coordinates, dtype=np.float32))
    return polygons


def _point_in_any_polygon(point_xy: Tuple[float, float], polygons: Sequence[np.ndarray]) -> bool:
    if not polygons:
        return True

    point = (float(point_xy[0]), float(point_xy[1]))
    for polygon in polygons:
        if polygon.shape[0] < 3:
            continue
        if cv2.pointPolygonTest(polygon, point, False) >= 0:
            return True
    return False


def _iter_candidate_top_lefts(context: BuildContext) -> Tuple[np.ndarray, np.ndarray]:
    slide = context.slide
    cfg = context.cfg

    patch_level = int(cfg.patch_level)
    if patch_level < 0 or patch_level >= int(slide.level_count):
        raise ValueError(f"patch_level={patch_level} invalid for slide with level_count={int(slide.level_count)}")

    level_w, level_h = slide.level_dimensions[patch_level]
    patch_size = resolve_patch_size_px(context)
    stride = resolve_stride_px(context, patch_size)
    downsample = float(slide.level_downsamples[patch_level])

    tissue_regions = getattr(context, "tissue_regions", None) or []
    if tissue_regions:
        boxes = [tuple(region["bbox_level0_xyxy"]) for region in tissue_regions]
    else:
        boxes = [(0, 0, int(slide.dimensions[0]) - 1, int(slide.dimensions[1]) - 1)]

    candidates: List[Tuple[int, int]] = []
    for x0, y0, x1, y1 in boxes:
        left = max(0, int(np.floor(x0 / downsample)))
        top = max(0, int(np.floor(y0 / downsample)))
        right = min(max(0, level_w - patch_size), int(np.floor(x1 / downsample)))
        bottom = min(max(0, level_h - patch_size), int(np.floor(y1 / downsample)))

        if right < left or bottom < top:
            continue

        for y in range(top, bottom + 1, stride):
            for x in range(left, right + 1, stride):
                if x < 0 or y < 0:
                    continue
                if x + patch_size > level_w or y + patch_size > level_h:
                    continue
                candidates.append((x, y))

    if not candidates:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0, 2), dtype=np.float32)

    top_left = np.asarray(candidates, dtype=np.int32)
    centroids = top_left.astype(np.float32) + (patch_size / 2.0)
    return top_left, centroids


def enumerate_patches(context: BuildContext) -> None:
    """Enumerate and filter candidate patches by tissue fraction and annotation membership, storing the result on the context."""
    if context.mask is None or context.mask_downsized is None:
        raise ValueError("Mask and mask downsample are required for patch enumeration")

    cfg = context.cfg
    slide = context.slide
    mask_thumb = np.asarray(context.mask, dtype=bool)
    thumb_rgb = np.asarray(context.thumb_rgb) if context.thumb_rgb is not None else None

    if thumb_rgb is None:
        raise ValueError("Thumbnail is required for patch enumeration")

    patch_level = int(cfg.patch_level)
    downsample = float(slide.level_downsamples[patch_level])
    patch_size = resolve_patch_size_px(context)
    annotation_polygons = _annotation_polygons(context) if context.creation_cfg.filter_graph_with_annotations else []
    slide_size = (int(slide.dimensions[0]), int(slide.dimensions[1]))
    patch_size_level0 = float(patch_size * downsample)

    top_lefts, centroids = _iter_candidate_top_lefts(context)
    kept_centroids: List[Tuple[float, float]] = []
    kept_fractions: List[float] = []
    kept_top_lefts: List[Tuple[int, int]] = []
    rejected_outside_slide = 0
    rejected_by_tissue = 0
    rejected_by_annotation = 0

    for top_left, centroid in zip(top_lefts.tolist(), centroids.tolist()):
        center_x, center_y = float(centroid[0]), float(centroid[1])
        center_x0 = center_x * downsample
        center_y0 = center_y * downsample

        top_left_level0 = (center_x0 - patch_size_level0 / 2.0, center_y0 - patch_size_level0 / 2.0)
        if top_left_level0[0] < 0 or top_left_level0[1] < 0:
            rejected_outside_slide += 1
            continue

        tissue_fraction = _patch_fraction_for_patch(mask_thumb, top_left_level0, patch_size_level0, slide_size)
        if tissue_fraction < float(cfg.tissue_frac_min):
            rejected_by_tissue += 1
            continue

        if annotation_polygons and not _point_in_any_polygon((center_x0, center_y0), annotation_polygons):
            rejected_by_annotation += 1
            continue

        kept_centroids.append((center_x, center_y))
        kept_fractions.append(tissue_fraction)
        kept_top_lefts.append((int(top_left[0]), int(top_left[1])))

    candidate_count = int(top_lefts.shape[0])
    kept_count = int(len(kept_centroids))
    logging.info(
        (
            "Patch enumeration for %s: candidates=%d kept=%d removed=%d "
            "removed_outside=%d removed_tissue=%d removed_annotation=%d "
            "annotation_filter=%s mask_source=%s tissue_frac_min=%.3f"
        ),
        Path(context.slide_path).name,
        candidate_count,
        kept_count,
        candidate_count - kept_count,
        rejected_outside_slide,
        rejected_by_tissue,
        rejected_by_annotation,
        bool(context.creation_cfg.filter_graph_with_annotations),
        str(getattr(context, "mask_source_kind", "unknown")),
        float(cfg.tissue_frac_min),
    )

    if context.creation_cfg.filter_graph_with_annotations and not annotation_polygons:
        raise MissingAnnotationPolygonsError(
            f"Annotation filtering requested but no annotation polygons are available for {context.slide_path}"
        )

    if not kept_centroids:
        msg = (
            "No patches found after filtering "
            f"(candidates={candidate_count}, "
            f"removed_outside={rejected_outside_slide}, "
            f"removed_tissue={rejected_by_tissue}, "
            f"removed_annotation={rejected_by_annotation}, "
            f"annotation_filter={bool(context.creation_cfg.filter_graph_with_annotations)}, "
            f"mask_source={getattr(context, 'mask_source_kind', 'unknown')}). "
            "Reduce thresholds or verify mask/annotation alignment."
        )
        raise NoTissuePatchesFoundError(msg)

    context.patch_xy = np.asarray(kept_centroids, dtype=np.float32)
    context.patch_frac = np.asarray(kept_fractions, dtype=np.float32)
    context.patch_topleft = np.asarray(kept_top_lefts, dtype=np.int32)
