import logging
import threading
import cv2
import numpy as np
from typing import Any, Dict, Optional, Tuple

def macenko_normalization(img: np.ndarray, target_means: Optional[np.ndarray] = None, target_stds: Optional[np.ndarray] = None) -> np.ndarray:
    """Normalize an image's LAB channel statistics to target means/stds (Reinhard-style color transfer)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    for channel, arr in enumerate([l, a, b]):
        mean = arr.mean() if target_means is None else target_means[channel]
        std = arr.std() if target_stds is None else target_stds[channel]
        arr = (arr - arr.mean()) / (arr.std() + 1e-8) * std + mean
        arr = np.clip(arr, 0, 255)
        if channel == 0:
            l = arr
        elif channel == 1:
            a = arr
        else:
            b = arr
    lab = cv2.merge([l.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def vahadane_normalization(img: np.ndarray) -> np.ndarray:
    """No-op placeholder; returns img unchanged."""
    return img


# ---------------------------------------------------------------------------
# Real Macenko (2009) stain-vector deconvolution + normalization.
#
# M. Macenko et al., "A method for normalizing histology slides for quantitative analysis",
# ISBI 2009. Unlike `macenko_normalization` above (plain LAB mean/std matching), this actually
# estimates the two dominant stain vectors (Hematoxylin/Eosin) per image via PCA on optical
# density, then re-expresses the image in terms of a *target* image's/reference's stain vectors
# and concentration scale -- robust to concentration/structure differences, not just a global
# color cast.
# ---------------------------------------------------------------------------

def rgb_to_optical_density(img: np.ndarray) -> np.ndarray:
    """uint8 RGB image (H,W,3) -> optical density, flattened to (H*W, 3)."""
    px = np.maximum(img.reshape(-1, 3).astype(np.float64), 1.0)
    return -np.log10(px / 255.0)


def optical_density_to_rgb(od: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Inverse of rgb_to_optical_density, reshaping back to `shape` (H,W,3)."""
    px = 255.0 * np.power(10.0, -od)
    return np.clip(px, 0, 255).reshape(shape).astype(np.uint8)


def _solve_concentrations(stain_matrix: np.ndarray, od: np.ndarray) -> np.ndarray:
    return (np.linalg.pinv(stain_matrix.T) @ od.T).T


def estimate_macenko_stain_matrix(
    od: np.ndarray,
    beta: float = 0.15,
    angular_percentile: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the two dominant stain vectors (Macenko et al. 2009) from optical-density pixel values."""
    od_tissue = od[~np.any(od < beta, axis=1)]
    if od_tissue.shape[0] < 10:
        raise ValueError(
            f"Not enough non-background pixels ({od_tissue.shape[0]}) to estimate Macenko "
            "stain vectors -- patch may be mostly background."
        )

    cov = np.cov(od_tissue.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalue order
    top2 = eigvecs[:, -2:]

    proj = od_tissue @ top2
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    min_angle = np.percentile(angles, angular_percentile)
    max_angle = np.percentile(angles, 100.0 - angular_percentile)

    v1 = top2 @ np.array([np.cos(min_angle), np.sin(min_angle)])
    v2 = top2 @ np.array([np.cos(max_angle), np.sin(max_angle)])

    stain_matrix = np.stack([v1, v2]) if v1[0] > v2[0] else np.stack([v2, v1])
    stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=1, keepdims=True)

    concentrations = _solve_concentrations(stain_matrix, od)
    concentrations = np.clip(concentrations, 0, None)
    max_concentrations = np.maximum(np.percentile(concentrations, 99, axis=0), 1e-6)

    return stain_matrix, max_concentrations


# Cascade of beta multipliers estimate_macenko_stain_matrix is retried at (in order, stopping
# at the first that succeeds) before macenko_stain_normalize/macenko_stain_augment give up and
# return the image unchanged. Root-caused 2026-08-12 comparing scanner1 vs scanner2 patches
# across all three GSDB stains (HE/PAS/MG): at the previous fixed beta=0.15, MG-stained slides
# fell back (i.e. were silently left un-normalized) at 2-95% of patches depending on the slide
# (one scanner-1 training slide, 73_52_1_MG, failed on 57/60 tissue_frac>=0.30 patches -- the
# color-normalization step meant to close the scanner domain gap was doing essentially nothing
# on that slide's data). Root cause: od[..., c] < beta must hold on ALL 3 channels for a pixel to
# count as "tissue" (see estimate_macenko_stain_matrix) -- a single fixed threshold implicitly
# tuned for H&E's characteristically strong absorbance rejects real-but-faint tissue in lightly
# counter-stained MG/PAS regions as background. On a representative failing MG patch (mean RGB
# ~(243,237,243), i.e. visually near-white but tissue_frac said >=30% real tissue by mask area),
# 0/30 sampled patches succeeded at beta=0.15 vs 27/30 at beta=0.05 -- lowering beta recovers
# real, sparse staining signal rather than admitting pure background (verified: even the
# rescued patches still had <10 tissue pixels at beta=0.15, not zero, so lowering the floor
# doesn't turn background into "tissue", it just stops discarding faint real tissue). Kept as a
# retry cascade rather than simply lowering the default beta everywhere, so well-stained
# patches (the vast majority, mainly HE) get byte-identical stain-vector estimates to before --
# only patches that would otherwise have skipped normalization entirely are affected.
_BETA_FALLBACK_MULTIPLIERS: Tuple[float, ...] = (1.0, 0.5, 0.25, 1.0 / 3.0)


class _MacenkoFallbackStats:
    """Thread-safe counters for how often the beta cascade above fired, aggregated instead of
    logged once per patch. Root-caused 2026-08-12: a real build (context_aware_domain_robust)
    showed this firing on a large fraction of patches per slide (not just MG/PAS -- any faint
    tissue, e.g. near a region edge, can trigger it) -- one INFO line per occurrence buried the
    actually-useful log lines under thousands of near-duplicate ones. reset() once per slide
    (src/graph_creation_new/helper/features.py's compute_node_features, before feature
    extraction starts), format_summary()/snapshot() once after -- see that call site for where
    the per-slide summary line gets logged and stored in region_diagnostics.

    Thread-safe (a plain lock, not e.g. thread-local) because BuildConfig.feature_prep_workers>1
    runs _read_and_transform_one_patch -- and therefore every call into this class -- across a
    real ThreadPoolExecutor for a single slide's patches; counts must merge across those threads,
    not stay siloed per-thread.
    """

    def __init__(self) -> None:
        """Initialize the fallback/give-up counters."""
        self._lock = threading.Lock()
        self._reset_locked()

    def _reset_locked(self) -> None:
        self._fallback_counts: Dict[Tuple[str, int], int] = {}
        self._gave_up_counts: Dict[str, int] = {}

    def reset(self) -> None:
        """Reset all counters to zero."""
        with self._lock:
            self._reset_locked()

    def record_fallback(self, context: str, multiplier_index: int) -> None:
        """Record one occurrence of the beta cascade falling back to a lower beta."""
        with self._lock:
            key = (context or "unknown", int(multiplier_index))
            self._fallback_counts[key] = self._fallback_counts.get(key, 0) + 1

    def record_gave_up(self, context: str) -> None:
        """Record one occurrence of the beta cascade exhausting all fallbacks."""
        with self._lock:
            key = context or "unknown"
            self._gave_up_counts[key] = self._gave_up_counts.get(key, 0) + 1

    def snapshot(self) -> Dict[str, Any]:
        """Return a dict snapshot of the current fallback/give-up counts."""
        with self._lock:
            by_context_and_multiplier = {
                f"{ctx}_{multiplier_index + 1}x": count
                for (ctx, multiplier_index), count in self._fallback_counts.items()
            }
            gave_up_by_context = dict(self._gave_up_counts)
        return {
            "total_fallback": sum(by_context_and_multiplier.values()),
            "total_gave_up_unchanged": sum(gave_up_by_context.values()),
            "by_context_and_multiplier": by_context_and_multiplier,
            "gave_up_by_context": gave_up_by_context,
        }

    def format_summary(self) -> Optional[str]:
        """Return a human-readable summary of fallback/give-up counts, or None if there were none."""
        s = self.snapshot()
        if s["total_fallback"] == 0 and s["total_gave_up_unchanged"] == 0:
            return None
        parts = ", ".join(f"{k}={v}" for k, v in sorted(s["by_context_and_multiplier"].items()))
        msg = f"{s['total_fallback']} patch(es) needed a lower Macenko beta ({parts})" if parts else "0 patches needed a lower Macenko beta"
        if s["total_gave_up_unchanged"]:
            gave_up_parts = ", ".join(f"{k}={v}" for k, v in sorted(s["gave_up_by_context"].items()))
            msg += f"; {s['total_gave_up_unchanged']} patch(es) still failed after all fallbacks and stayed unnormalized ({gave_up_parts})"
        return msg


macenko_fallback_stats = _MacenkoFallbackStats()


def _estimate_macenko_stain_matrix_with_fallback(
    od: np.ndarray,
    beta: float,
    angular_percentile: float,
    context: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    last_error: Optional[ValueError] = None
    for i, multiplier in enumerate(_BETA_FALLBACK_MULTIPLIERS):
        effective_beta = beta * multiplier
        try:
            result = estimate_macenko_stain_matrix(od, beta=effective_beta, angular_percentile=angular_percentile)
        except ValueError as exc:
            last_error = exc
            continue
        if i > 0:
            macenko_fallback_stats.record_fallback(context, i)
        return result
    assert last_error is not None
    raise last_error


def macenko_stain_normalize(
    img: np.ndarray,
    target_stain_matrix: np.ndarray,
    target_max_concentrations: Optional[np.ndarray],
    beta: float = 0.15,
    angular_percentile: float = 1.0,
    rescale_concentration: bool = True,
) -> np.ndarray:
    """Re-express `img` in terms of a target's stain vectors, and by default also its concentration scale."""
    shape = img.shape
    od = rgb_to_optical_density(img)

    try:
        source_stain_matrix, source_max_concentrations = _estimate_macenko_stain_matrix_with_fallback(
            od, beta=beta, angular_percentile=angular_percentile, context="macenko_stain_normalize"
        )
    except ValueError:
        macenko_fallback_stats.record_gave_up("macenko_stain_normalize")
        return img

    concentrations = _solve_concentrations(source_stain_matrix, od)
    concentrations = np.clip(concentrations, 0, None)

    target_stain_matrix = np.asarray(target_stain_matrix, dtype=np.float64)

    if rescale_concentration:
        if target_max_concentrations is None:
            raise ValueError("target_max_concentrations is required when rescale_concentration=True")
        target_max_concentrations = np.asarray(target_max_concentrations, dtype=np.float64)
        concentrations_rescaled = concentrations * (target_max_concentrations / source_max_concentrations)
    else:
        concentrations_rescaled = concentrations

    od_normalized = concentrations_rescaled @ target_stain_matrix
    return optical_density_to_rgb(od_normalized, shape)


def macenko_stain_augment(
    img: np.ndarray,
    rng: np.random.Generator,
    sigma_alpha: float = 0.2,
    sigma_beta: float = 0.2,
    beta: float = 0.15,
    angular_percentile: float = 1.0,
) -> np.ndarray:
    """Randomly perturb an image's own Hematoxylin/Eosin stain concentrations (Tellez et al. 2018 style augmentation)."""
    shape = img.shape
    od = rgb_to_optical_density(img)

    try:
        stain_matrix, max_concentrations = _estimate_macenko_stain_matrix_with_fallback(
            od, beta=beta, angular_percentile=angular_percentile, context="macenko_stain_augment"
        )
    except ValueError:
        macenko_fallback_stats.record_gave_up("macenko_stain_augment")
        return img

    concentrations = _solve_concentrations(stain_matrix, od)
    concentrations = np.clip(concentrations, 0, None)

    alpha = rng.uniform(1.0 - sigma_alpha, 1.0 + sigma_alpha, size=2)
    beta_shift = rng.uniform(-sigma_beta, sigma_beta, size=2) * max_concentrations
    concentrations_aug = np.clip(concentrations * alpha + beta_shift, 0, None)

    od_aug = concentrations_aug @ stain_matrix
    return optical_density_to_rgb(od_aug, shape)
