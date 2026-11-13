import numpy as np
import logging
from typing import Optional, List, Dict, Any, Iterator
from PIL import Image
import cv2

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.graph_creation_new.config import BuildContext


def resolve_patch_size_px(context: BuildContext) -> int:
    """Compute the effective raw pixel patch size at cfg.patch_level for this slide."""
    cfg = context.cfg
    target_um = getattr(cfg, "target_patch_size_um", None)
    if target_um is None:
        return max(16, int(cfg.patch_size))

    mpp_x = None
    try:
        raw = context.slide.properties.get("openslide.mpp-x")
        mpp_x = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        mpp_x = None

    if not mpp_x or mpp_x <= 0:
        logging.warning(
            "target_patch_size_um=%.3f set but slide %s has no usable openslide.mpp-x "
            "property -- falling back to cfg.patch_size=%d unchanged.",
            float(target_um), getattr(context, "slide_path", "?"), int(cfg.patch_size),
        )
        return max(16, int(cfg.patch_size))

    level_downsample = float(context.slide.level_downsamples[int(cfg.patch_level)])
    mpp_x_at_level = mpp_x * level_downsample
    patch_size_px = int(round(float(target_um) / mpp_x_at_level))
    return max(16, patch_size_px)


def resolve_stride_px(context: BuildContext, effective_patch_size_px: int) -> int:
    """Compute the effective patch stride in pixels for this slide."""
    cfg = context.cfg
    if getattr(cfg, "target_patch_size_um", None) is None:
        return max(1, int(cfg.stride))
    ratio = float(effective_patch_size_px) / max(1, int(cfg.patch_size))
    return max(1, int(round(int(cfg.stride) * ratio)))


def _extract_tissue_regions(mask: np.ndarray, downsample: tuple[float, float]) -> List[Dict[str, Any]]:
    binary = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    scale_x, scale_y = downsample
    regions: List[Dict[str, Any]] = []

    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area_mask = int(stats[label_id, cv2.CC_STAT_AREA])

        if w <= 0 or h <= 0 or area_mask <= 0:
            continue

        x_min = int(round(x * scale_x))
        y_min = int(round(y * scale_y))
        x_max = int(round((x + w - 1) * scale_x))
        y_max = int(round((y + h - 1) * scale_y))

        cx_mask, cy_mask = centroids[label_id]
        cx_lvl0 = float(cx_mask * scale_x)
        cy_lvl0 = float(cy_mask * scale_y)

        area_lvl0 = float(area_mask * scale_x * scale_y)
        local_component_mask = (labels[y:y + h, x:x + w] == label_id)

        regions.append({
            "region_id": int(label_id - 1),
            "area_mask_px": area_mask,
            "area_level0_px": area_lvl0,
            "bbox_mask_xyxy": (x, y, x + w - 1, y + h - 1),
            "bbox_level0_xyxy": (x_min, y_min, x_max, y_max),
            "centroid_mask_xy": (float(cx_mask), float(cy_mask)),
            "centroid_level0_xy": (cx_lvl0, cy_lvl0),
            "mask_crop": local_component_mask,
        })

    regions.sort(key=lambda r: r["area_mask_px"], reverse=True)

    for new_id, region in enumerate(regions):
        region["region_id"] = int(new_id)

    return regions


def prepare_slide(context: BuildContext) -> None:
    """Compute the tissue bounding box and extract connected tissue regions from the mask, storing them on the context."""
    mask = context.mask
    downsample = context.mask_downsized

    if mask is None:
        logging.warning("No mask available -> skipping slide preparation")
        return

    mask = mask.astype(bool)

    if not mask.any():
        logging.warning("Empty mask -> skipping slide preparation")
        return

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    scale_x, scale_y = downsample

    x_min = int(x_min * scale_x)
    x_max = int(x_max * scale_x)
    y_min = int(y_min * scale_y)
    y_max = int(y_max * scale_y)

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    logging.info(
        f"Tissue ROI (level 0): x={x_min}:{x_max}, y={y_min}:{y_max}"
    )

    context.tissue_bbox = (x_min, y_min, x_max, y_max)
    context.tissue_region_size = (width, height)

    regions = _extract_tissue_regions(mask, (scale_x, scale_y))
    context.tissue_regions = regions
    logging.info("Extracted %d connected tissue regions", len(regions))

    if regions:
        largest = regions[0]
        logging.info(
            "Largest tissue region: area_mask_px=%d, bbox_level0=%s",
            largest["area_mask_px"],
            largest["bbox_level0_xyxy"],
        )

def generate_preview(context, target_max_size=1024):
    """Create and save a downscaled preview image of the slide with its tissue mask overlay."""
    slide = context.slide
    mask = context.mask

    if mask is None:
        raise ValueError("Mask required")

    slide_w, slide_h = slide.dimensions

    scale = target_max_size / max(slide_w, slide_h)

    new_w = int(slide_w * scale)
    new_h = int(slide_h * scale)

    level = slide.get_best_level_for_downsample(1 / scale)

    downsample = slide.level_downsamples[level]

    level_w, level_h = slide.level_dimensions[level]

    slide_img = slide.read_region(
        (0, 0),
        level,
        (level_w, level_h)
    ).convert("RGB")

    slide_img = slide_img.resize((new_w, new_h))

    mask_resized = cv2.resize(
        mask.astype(np.uint8),
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    slide_np = np.array(slide_img)

    overlay = slide_np.copy()
    overlay[~mask_resized] = [0, 0, 0]

    Image.fromarray(overlay).save("preview_masked_loaded.png")

    slide_img.save("preview_slide.png")

    context.preview_size = (new_w, new_h)

    regions = getattr(context, "tissue_regions", None) or []
    if not regions:
        return

    slide_stem = Path(context.slide_path).stem if getattr(context, "slide_path", None) else "slide"
    preview_root = Path("preview_tissue_regions") / slide_stem
    preview_root.mkdir(parents=True, exist_ok=True)

    scale_x = new_w / float(slide_w)
    scale_y = new_h / float(slide_h)

    overview = slide_np.copy()

    for region in regions:
        region_id = int(region["region_id"])
        x0, y0, x1, y1 = region["bbox_level0_xyxy"]

        px0 = max(0, min(new_w - 1, int(round(x0 * scale_x))))
        py0 = max(0, min(new_h - 1, int(round(y0 * scale_y))))
        px1 = max(0, min(new_w - 1, int(round(x1 * scale_x))))
        py1 = max(0, min(new_h - 1, int(round(y1 * scale_y))))

        if px1 <= px0 or py1 <= py0:
            continue

        crop = slide_np[py0:py1 + 1, px0:px1 + 1].copy()
        region_mask = np.asarray(region.get("mask_crop", None))
        if region_mask.size > 0:
            region_mask_resized = cv2.resize(
                region_mask.astype(np.uint8),
                (crop.shape[1], crop.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            crop[~region_mask_resized] = [0, 0, 0]

        out_path = preview_root / f"region_{region_id:03d}.png"
        Image.fromarray(crop).save(out_path)

        cv2.rectangle(overview, (px0, py0), (px1, py1), (255, 0, 0), 2)
        cv2.putText(
            overview,
            str(region_id),
            (px0, max(12, py0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    Image.fromarray(overview).save(preview_root / "regions_overview.png")
    logging.info("Saved %d tissue region previews to %s", len(regions), preview_root)


def iter_tissue_region_crops(
    context: BuildContext,
    level: int = 0,
    apply_mask: bool = True,
    min_area_level0_px: Optional[float] = None,
) -> Iterator[Dict[str, Any]]:
    """Iterate over connected tissue regions as real WSI crops."""
    regions = getattr(context, "tissue_regions", None) or []
    if not regions:
        return

    slide = context.slide
    if level < 0 or level >= int(slide.level_count):
        raise ValueError(f"Invalid level={level}. Available levels: 0..{int(slide.level_count) - 1}")

    downsample = float(slide.level_downsamples[level])

    for region in regions:
        area_level0 = float(region.get("area_level0_px", 0.0))
        if min_area_level0_px is not None and area_level0 < float(min_area_level0_px):
            continue

        region_id = int(region["region_id"])
        x0, y0, x1, y1 = region["bbox_level0_xyxy"]
        w0 = int(x1 - x0 + 1)
        h0 = int(y1 - y0 + 1)
        if w0 <= 0 or h0 <= 0:
            continue

        wl = max(1, int(round(w0 / downsample)))
        hl = max(1, int(round(h0 / downsample)))

        img = slide.read_region((int(x0), int(y0)), int(level), (wl, hl)).convert("RGB")
        rgb = np.asarray(img)

        region_mask = np.asarray(region.get("mask_crop", None))
        if region_mask.size > 0:
            mask_resized = cv2.resize(
                region_mask.astype(np.uint8),
                (wl, hl),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            mask_resized = np.ones((hl, wl), dtype=bool)

        masked_rgb = rgb.copy()
        if apply_mask:
            masked_rgb[~mask_resized] = [0, 0, 0]

        yield {
            "region_id": region_id,
            "bbox_level0_xyxy": (int(x0), int(y0), int(x1), int(y1)),
            "level": int(level),
            "downsample": downsample,
            "image_rgb": rgb,
            "mask": mask_resized,
            "masked_rgb": masked_rgb,
            "area_level0_px": area_level0,
        }


def save_tissue_region_crops(
    context: BuildContext,
    out_dir: Path | str,
    level: int = 0,
    apply_mask: bool = True,
    min_area_level0_px: Optional[float] = None,
) -> int:
    """Save connected tissue regions as images read directly from the WSI."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    count = 0
    for item in iter_tissue_region_crops(
        context,
        level=level,
        apply_mask=apply_mask,
        min_area_level0_px=min_area_level0_px,
    ):
        region_id = int(item["region_id"])
        out_file = out_path / f"region_{region_id:03d}_lvl{int(level)}.png"
        arr = item["masked_rgb"] if apply_mask else item["image_rgb"]
        Image.fromarray(arr).save(out_file)
        count += 1

    logging.info("Saved %d tissue region crops at level=%d to %s", count, level, out_path)
    return count