from __future__ import annotations

import functools
import json
import zlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from src.utils.stain_normalization import macenko_normalization, macenko_stain_normalize, macenko_stain_augment


def _seed_for_patch(seed: int, slide_key: str, patch_xy: Tuple[int, int]) -> int:
    x, y = int(patch_xy[0]), int(patch_xy[1])
    slide_hash = zlib.crc32(str(slide_key).encode("utf-8")) & 0xFFFFFFFF
    return (int(seed) * 1_000_003 + slide_hash * 97 + x * 9176 + y * 26699) % (2**32 - 1)


@functools.lru_cache(maxsize=16)
def _load_reference_stats(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def color_normalize(
    arr: np.ndarray,
    *,
    method: str = "macenko",
    target_stain_matrix: Any = None,
    target_max_concentrations: Any = None,
    target_means: Any = None,
    target_stds: Any = None,
    target_stats_path: str | None = None,
    macenko_beta: float = 0.15,
    macenko_angular_percentile: float = 1.0,
    macenko_rescale_concentration: bool = True,
    blend_strength: float = 1.0,
    **_ignored: Any,
) -> np.ndarray:
    """Normalize a patch's color/staining toward a fixed reference using Macenko or Reinhard-LAB transfer."""
    stats = None
    if target_stats_path is not None:
        stats = _load_reference_stats(str(target_stats_path))

    method = str(method).strip().lower()
    if method == "macenko":
        if target_stain_matrix is None:
            if stats is None or "stain_matrix" not in stats:
                raise ValueError(
                    "color_normalize(method='macenko') requires target_stain_matrix "
                    "or a target_stats_path JSON containing it "
                    "(see scripts/graph_creation/compute_color_reference_stats.py)"
                )
            target_stain_matrix = stats["stain_matrix"]
        if macenko_rescale_concentration and target_max_concentrations is None:
            if stats is None or "max_concentrations" not in stats:
                raise ValueError(
                    "color_normalize(method='macenko', macenko_rescale_concentration=True) "
                    "requires target_max_concentrations or a target_stats_path JSON containing "
                    "it (see scripts/graph_creation/compute_color_reference_stats.py), or set "
                    "macenko_rescale_concentration=False to skip intensity rescaling entirely"
                )
            target_max_concentrations = stats["max_concentrations"]
        normalized = macenko_stain_normalize(
            arr,
            target_stain_matrix=target_stain_matrix,
            target_max_concentrations=target_max_concentrations,
            beta=macenko_beta,
            angular_percentile=macenko_angular_percentile,
            rescale_concentration=macenko_rescale_concentration,
        )
    elif method == "reinhard_lab":
        if target_means is None or target_stds is None:
            if stats is None or "means" not in stats or "stds" not in stats:
                raise ValueError(
                    "color_normalize(method='reinhard_lab') requires target_means+target_stds "
                    "or a target_stats_path JSON containing both "
                    "(see scripts/graph_creation/compute_color_reference_stats.py)"
                )
            target_means = stats["means"]
            target_stds = stats["stds"]
        normalized = macenko_normalization(
            arr,
            target_means=np.asarray(target_means, dtype=np.float64),
            target_stds=np.asarray(target_stds, dtype=np.float64),
        )
    else:
        raise ValueError(f"color_normalize: unknown method '{method}', expected 'macenko' or 'reinhard_lab'")

    if blend_strength >= 1.0:
        return normalized
    if blend_strength <= 0.0:
        return arr
    blended = blend_strength * normalized.astype(np.float32) + (1.0 - blend_strength) * arr.astype(np.float32)
    return np.round(blended).clip(0, 255).astype(np.uint8)


def color_jitter(
    arr: np.ndarray,
    *,
    rng: np.random.Generator,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
    p: float = 1.0,
    **_ignored: Any,
) -> np.ndarray:
    """Apply stochastic HSV-space brightness/contrast/saturation/hue jitter to a patch."""
    if rng.random() >= p:
        return arr
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    if brightness > 0:
        hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + rng.uniform(-brightness, brightness)), 0, 255)
    if saturation > 0:
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + rng.uniform(-saturation, saturation)), 0, 255)
    if hue > 0:
        hsv[..., 0] = np.mod(hsv[..., 0] + rng.uniform(-hue, hue) * 179.0, 180)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if contrast > 0:
        factor = 1.0 + rng.uniform(-contrast, contrast)
        mean = out.mean(axis=(0, 1), keepdims=True)
        out = np.clip((out.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
    return out


def gaussian_noise(
    arr: np.ndarray,
    *,
    rng: np.random.Generator,
    std: float = 5.0,
    p: float = 1.0,
    **_ignored: Any,
) -> np.ndarray:
    """Additive per-pixel Gaussian noise (std in 0-255 pixel-value units) on the uint8 RGB array."""
    if rng.random() >= p:
        return arr
    noise = rng.normal(loc=0.0, scale=float(std), size=arr.shape)
    return np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def stain_augment(
    arr: np.ndarray,
    *,
    rng: np.random.Generator,
    sigma_alpha: float = 0.2,
    sigma_beta: float = 0.2,
    macenko_beta: float = 0.15,
    macenko_angular_percentile: float = 1.0,
    p: float = 1.0,
    **_ignored: Any,
) -> np.ndarray:
    """Apply randomized Macenko stain-concentration augmentation to a patch."""
    if rng.random() >= p:
        return arr
    return macenko_stain_augment(
        arr, rng, sigma_alpha=sigma_alpha, sigma_beta=sigma_beta,
        beta=macenko_beta, angular_percentile=macenko_angular_percentile,
    )


def gaussian_blur(
    arr: np.ndarray,
    *,
    rng: np.random.Generator,
    max_sigma: float = 1.5,
    p: float = 1.0,
    **_ignored: Any,
) -> np.ndarray:
    """Apply a random Gaussian blur to a patch, simulating scanner focus differences."""
    if rng.random() >= p:
        return arr
    sigma = float(rng.uniform(0.1, max_sigma))
    ksize = max(3, int(2 * round(3 * sigma) + 1))  # odd kernel covering ~3 sigma each side
    return cv2.GaussianBlur(arr, (ksize, ksize), sigmaX=sigma)


def jpeg_compression(
    arr: np.ndarray,
    *,
    rng: np.random.Generator,
    quality_min: int = 30,
    quality_max: int = 90,
    p: float = 1.0,
    **_ignored: Any,
) -> np.ndarray:
    """Apply a random JPEG compression round-trip to a patch, simulating scanner compression artifacts."""
    if rng.random() >= p:
        return arr
    quality = int(rng.integers(quality_min, quality_max + 1))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return arr
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)


_TRANSFORM_REGISTRY = {
    "color_normalize": color_normalize,
    "color_jitter": color_jitter,
    "gaussian_noise": gaussian_noise,
    "stain_augment": stain_augment,
    "gaussian_blur": gaussian_blur,
    "jpeg_compression": jpeg_compression,
}


def apply_patch_transforms(
    arr: np.ndarray,
    specs: List[Dict[str, Any]],
    *,
    seed: int,
    slide_key: str,
    patch_xy: Tuple[int, int],
) -> np.ndarray:
    """Apply a configured sequence of transforms to a single raw uint8 RGB patch."""
    if not specs:
        return arr

    rng = np.random.default_rng(_seed_for_patch(seed, slide_key, patch_xy))
    out = arr
    for spec in specs:
        if "name" not in spec:
            raise ValueError(f"Transform spec missing required 'name' key: {spec}")
        name = str(spec["name"])
        fn = _TRANSFORM_REGISTRY.get(name)
        if fn is None:
            raise ValueError(f"Unknown transform '{name}'. Available: {sorted(_TRANSFORM_REGISTRY)}")
        params = {k: v for k, v in spec.items() if k != "name"}
        out = fn(out, rng=rng, **params)
    return out
