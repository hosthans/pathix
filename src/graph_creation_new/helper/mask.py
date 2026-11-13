from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import tifffile
import cv2
import logging

import sys
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.graph_creation_new.config import BuildContext


def _mask_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(mask.astype(bool).mean())


def _postprocess_mask(mask: np.ndarray, kernel_size: int, min_component_area: int) -> np.ndarray:
    kernel_size = max(3, int(kernel_size) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    keep = np.zeros_like(cleaned)
    min_area = max(1, int(min_component_area))
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            keep[labels == label_id] = 1
    return keep.astype(np.uint8)


def _drop_border_components(
    mask: np.ndarray,
    border_margin_px: int = 0,
    require_core_connection: bool = False,
    core_margin_px: int = 0,
) -> np.ndarray:
    binary = mask.astype(np.uint8)
    if binary.size == 0:
        return binary

    height, width = binary.shape
    margin = max(0, int(border_margin_px))
    core_margin = max(0, int(core_margin_px))
    core_x0 = min(max(core_margin, margin + 1), max(0, width - 1))
    core_y0 = min(max(core_margin, margin + 1), max(0, height - 1))
    core_x1 = max(core_x0, width - 1 - max(core_margin, margin + 1))
    core_y1 = max(core_y0, height - 1 - max(core_margin, margin + 1))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros_like(binary)

    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        x0 = x
        y0 = y
        x1 = x + w - 1
        y1 = y + h - 1

        touches_border = (
            x0 <= margin
            or y0 <= margin
            or x1 >= (width - 1 - margin)
            or y1 >= (height - 1 - margin)
        )

        if not touches_border:
            keep[labels == label_id] = 1
            continue

        if require_core_connection:
            overlaps_core = bool(
                np.any(labels[core_y0:core_y1 + 1, core_x0:core_x1 + 1] == label_id)
            )
            if overlaps_core:
                keep[labels == label_id] = 1

    return keep.astype(np.uint8)


def _keep_largest_components(mask: np.ndarray, max_components: int) -> np.ndarray:
    binary = mask.astype(np.uint8)
    if binary.size == 0:
        return binary

    k = int(max_components)
    if k <= 0:
        return np.zeros_like(binary, dtype=np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    areas = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area > 0:
            areas.append((label_id, area))

    if not areas:
        return np.zeros_like(binary, dtype=np.uint8)

    areas.sort(key=lambda item: item[1], reverse=True)
    keep_labels = {label_id for label_id, _ in areas[:k]}

    keep = np.zeros_like(binary)
    for label_id in keep_labels:
        keep[labels == label_id] = 1
    return keep.astype(np.uint8)


def refine_mask_with_config(mask: np.ndarray, cfg) -> np.ndarray:
    """Refine a binary tissue mask using morphology, border-dropping, and largest-component config."""
    refined = _postprocess_mask(mask, cfg.morph_kernel, cfg.min_component_area)

    if bool(getattr(cfg, "mask_drop_border_components", False)):
        refined = _drop_border_components(
            refined,
            border_margin_px=int(getattr(cfg, "mask_border_margin_px", 0)),
            require_core_connection=bool(
                getattr(cfg, "mask_require_core_connection_for_border_components", False)
            ),
            core_margin_px=int(getattr(cfg, "mask_core_margin_px", 0)),
        )

    keep_k = getattr(cfg, "mask_keep_largest_components", None)
    if keep_k is not None:
        refined = _keep_largest_components(refined, int(keep_k))

    return refined.astype(np.uint8)


def _choose_otsu_polarity(mask_dark: np.ndarray, mask_light: np.ndarray, mask_hsv: np.ndarray) -> tuple[np.ndarray, str]:
    overlap_dark = _mask_ratio((mask_dark.astype(bool) & mask_hsv.astype(bool)))
    overlap_light = _mask_ratio((mask_light.astype(bool) & mask_hsv.astype(bool)))

    if overlap_light > overlap_dark:
        return mask_light, "light"
    return mask_dark, "dark"


def _gray_world_color_normalize(rgb: np.ndarray) -> np.ndarray:
    float_img = rgb.astype(np.float32)
    means = float_img.reshape(-1, 3).mean(axis=0)
    eps = 1e-6
    target = float(np.mean(means))
    scales = target / np.maximum(means, eps)
    balanced = float_img * scales.reshape(1, 1, 3)
    balanced = np.clip(balanced, 0.0, 255.0)
    return balanced.astype(np.uint8)


def _blend_rgb(base: np.ndarray, adapted: np.ndarray, alpha: float) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    if a <= 0.0:
        return base
    if a >= 1.0:
        return adapted
    mixed = ((1.0 - a) * base.astype(np.float32)) + (a * adapted.astype(np.float32))
    return np.clip(mixed, 0.0, 255.0).astype(np.uint8)


def _effective_hsv_thresholds(hsv: np.ndarray, cfg) -> tuple[int, int]:
    sat = hsv[..., 1]
    val = hsv[..., 2]

    sat_min_default = int(getattr(cfg, "sat_min", 15))
    val_max_default = int(getattr(cfg, "val_max", 245))
    adaptive_enabled = bool(getattr(cfg, "mask_use_adaptive_hsv", True))

    if not adaptive_enabled:
        return sat_min_default, val_max_default

    sat_p = float(getattr(cfg, "mask_sat_percentile", 35.0))
    val_p = float(getattr(cfg, "mask_val_percentile", 99.0))
    sat_p = float(np.clip(sat_p, 0.0, 100.0))
    val_p = float(np.clip(val_p, 0.0, 100.0))

    sat_min_adaptive = int(np.percentile(sat, sat_p))
    val_max_adaptive = int(np.percentile(val, val_p))

    sat_min_eff = int(np.clip(max(sat_min_default, sat_min_adaptive), 5, 250))
    val_max_eff = int(np.clip(min(val_max_default, val_max_adaptive), 5, 255))
    return sat_min_eff, val_max_eff


def _lab_chroma(lab: np.ndarray) -> np.ndarray:
    a = lab[..., 1].astype(np.float32) - 128.0
    b = lab[..., 2].astype(np.float32) - 128.0
    return np.sqrt((a * a) + (b * b))


def _build_neutral_near_black_mask(
    rgb: np.ndarray,
    lab: np.ndarray,
    hsv: np.ndarray,
    sat_min_eff: int,
) -> np.ndarray:
    value = hsv[..., 2].astype(np.uint8)
    chroma = _lab_chroma(lab)
    rgb_max = rgb.max(axis=2).astype(np.uint8)

    chroma_min = max(5.0, 0.35 * float(sat_min_eff))
    black_rgb_max = max(6, min(18, int(np.percentile(rgb_max, 1.0)) + 8))
    black_value_min = max(8, min(28, int(np.percentile(value, 1.0)) + 10))

    return (
        (rgb_max <= max(black_rgb_max + 2, 14))
        & (value <= max(black_value_min + 4, 24))
        & (chroma < chroma_min)
    ).astype(np.uint8)


def _build_non_background_mask(
    rgb: np.ndarray,
    lab: np.ndarray,
    hsv: np.ndarray,
    sat_min_eff: int,
    val_max_eff: int,
) -> np.ndarray:
    lightness = lab[..., 0].astype(np.uint8)
    value = hsv[..., 2].astype(np.uint8)
    chroma = _lab_chroma(lab)
    rgb_max = rgb.max(axis=2).astype(np.uint8)
    neutral_near_black = _build_neutral_near_black_mask(rgb, lab, hsv, sat_min_eff).astype(bool)

    chroma_min = max(6.0, 0.35 * float(sat_min_eff))
    white_value_max = min(250, int(val_max_eff) + 20)
    white_lightness_max = min(252, max(215, int(np.percentile(lightness, 97.5))))

    not_white = (
        (value < white_value_max)
        & ((lightness < white_lightness_max) | (chroma > chroma_min))
    )

    not_black = (
        (rgb_max > max(6, min(18, int(np.percentile(rgb_max, 1.0)) + 8)))
        | ((value > max(8, min(28, int(np.percentile(value, 1.0)) + 10))) & (chroma > 4.0))
        | (chroma > (chroma_min + 2.0))
    )

    return (not_white & not_black & (~neutral_near_black)).astype(np.uint8)

def load_binary_mask(mask_path: Optional[Path]):
    """Load a binary tissue mask from a multi-level TIFF file at its lowest resolution level."""
    if mask_path is None or not Path(mask_path).exists():
        return None, None

    with tifffile.TiffFile(mask_path) as tif:
        series = tif.series[0]
        level = len(series.levels) - 1

        mask = series.levels[level].asarray()

        full_res_shape = series.levels[0].shape
        low_res_shape = mask.shape

        downsample_y = full_res_shape[0] / low_res_shape[0]
        downsample_x = full_res_shape[1] / low_res_shape[1]

    if mask.ndim == 3:
        mask = mask[..., 0]

    return (mask > 0), (downsample_x, downsample_y)


def load_mask_from_path(mask_path: Optional[Path], slide_dimensions: Optional[Tuple[int, int]] = None):
    """Load a binary tissue mask from a TIFF or image file, dispatching by file suffix."""
    if mask_path is None or not Path(mask_path).exists():
        return None, None

    mask_path = Path(mask_path)
    suffix = mask_path.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        mask, downsample = load_binary_mask(mask_path)
        if mask is None:
            return None, None
        return mask, downsample

    if suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
        image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return None, None
        if image.ndim == 3:
            image = image[..., 0]
        mask = (image > 0)
        if slide_dimensions is None:
            return mask, (1.0, 1.0)
        slide_w, slide_h = slide_dimensions
        mask_h, mask_w = mask.shape
        if mask_w == 0 or mask_h == 0:
            return None, None
        return mask, (float(slide_w) / float(mask_w), float(slide_h) / float(mask_h))

    logging.warning("Unsupported mask suffix '%s' for %s", suffix, mask_path)
    return None, None


def generate_mask_from_thumbnail(context: BuildContext):
    """Generate a binary tissue mask from the slide's thumbnail image."""
    thumb = context.thumb_rgb
    cfg = context.cfg

    if thumb is None:
        logging.warning("No thumbnail in context -> cannot generate tissue mask")
        return None, None

    if thumb.ndim != 3 or thumb.shape[-1] != 3:
        logging.warning("Unexpected thumbnail shape %s -> cannot generate tissue mask", thumb.shape)
        return None, None

    color_norm_enabled = bool(getattr(cfg, "mask_enable_color_normalization", True))
    color_norm_alpha = float(getattr(cfg, "mask_color_norm_strength", 0.85))
    adapted_thumb = _gray_world_color_normalize(thumb)
    work_thumb = _blend_rgb(thumb, adapted_thumb, color_norm_alpha) if color_norm_enabled else thumb

    lab = cv2.cvtColor(work_thumb, cv2.COLOR_RGB2LAB)
    lightness = lab[..., 0]
    _, otsu = cv2.threshold(lightness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_otsu_dark = (otsu == 0).astype(np.uint8)
    mask_otsu_light = (otsu > 0).astype(np.uint8)

    hsv = cv2.cvtColor(work_thumb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    sat_min_eff, val_max_eff = _effective_hsv_thresholds(hsv, cfg)
    mask_neutral_black = _build_neutral_near_black_mask(work_thumb, lab, hsv, sat_min_eff)
    mask_hsv = ((saturation > sat_min_eff) & (value < val_max_eff)).astype(np.uint8)
    mask_non_background = _build_non_background_mask(work_thumb, lab, hsv, sat_min_eff, val_max_eff)
    mask_reference = ((mask_hsv | mask_non_background) & (~mask_neutral_black.astype(bool))).astype(np.uint8)

    mask_otsu, otsu_mode = _choose_otsu_polarity(mask_otsu_dark, mask_otsu_light, mask_reference)
    mask = (mask_otsu & mask_reference).astype(np.uint8)

    clean = refine_mask_with_config(mask, cfg)

    used_fallback = False
    used_relaxed_hsv = False
    if not np.any(clean):
        sat_relaxed = max(5, int(sat_min_eff) // 2)
        val_relaxed = min(255, int(val_max_eff) + 20)
        mask_hsv_relaxed = ((saturation > sat_relaxed) & (value < val_relaxed)).astype(np.uint8)
        mask_relaxed = (mask_otsu & ((mask_hsv_relaxed | mask_non_background) & (~mask_neutral_black.astype(bool)))).astype(np.uint8)
        clean = _postprocess_mask(
            mask_relaxed,
            max(3, int(cfg.morph_kernel) - 2),
            max(32, int(cfg.min_component_area) // 2),
        )
        if bool(getattr(cfg, "mask_drop_border_components", False)):
            clean = _drop_border_components(
                clean,
                border_margin_px=int(getattr(cfg, "mask_border_margin_px", 0)),
                require_core_connection=bool(
                    getattr(cfg, "mask_require_core_connection_for_border_components", False)
                ),
                core_margin_px=int(getattr(cfg, "mask_core_margin_px", 0)),
            )
        keep_k = getattr(cfg, "mask_keep_largest_components", None)
        if keep_k is not None:
            clean = _keep_largest_components(clean, int(keep_k))
        used_relaxed_hsv = bool(np.any(clean))

    if not np.any(clean):
        logging.warning(
            "Generated tissue mask is empty after strict+relaxed filtering. Falling back to Otsu-only tissue mask."
        )
        clean = mask_otsu
        used_fallback = True

    mask_bool = clean.astype(bool)

    full_w, full_h = context.slide.dimensions
    low_h, low_w = mask_bool.shape

    if low_h == 0 or low_w == 0:
        logging.warning("Generated tissue mask has invalid shape %s", mask_bool.shape)
        return None, None

    slide_name = Path(context.slide_path).name if getattr(context, "slide_path", None) else "unknown-slide"
    logging.info(
        (
            "Tissue mask stats [%s] - "
            "color_norm=%s(alpha=%.2f), hsv_thr=(sat>%d,val<%d), "
            "otsu(mode=%s)=%.2f%%, hsv=%.2f%%, non_background=%.2f%%, combined=%.2f%%, final=%.2f%%, relaxed_hsv=%s, fallback=%s"
        ),
        slide_name,
        color_norm_enabled,
        color_norm_alpha,
        sat_min_eff,
        val_max_eff,
        otsu_mode,
        100.0 * _mask_ratio(mask_otsu),
        100.0 * _mask_ratio(mask_hsv),
        100.0 * _mask_ratio(mask_non_background),
        100.0 * _mask_ratio(mask),
        100.0 * _mask_ratio(mask_bool),
        used_relaxed_hsv,
        used_fallback,
    )

    return mask_bool, (full_w / low_w, full_h / low_h)