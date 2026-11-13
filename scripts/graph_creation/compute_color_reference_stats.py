"""Compute reference color/stain statistics from a sample of tissue patches, for use as the
`target_stats_path` of the `color_normalize` transform in
src/graph_creation_new/helper/transforms.py (BuildConfig.feature_transforms).

Writes BOTH:
  - `stain_matrix` (2x3) + `max_concentrations` (2,) -- real Macenko (2009) stain-vector
    reference, pooled across ALL sampled tissue pixels (more robust than per-patch estimation).
    This is what `color_normalize(method="macenko")` (the default) uses.
  - `means`/`stds` (LAB, 3 values each) -- for the older, cheaper `method="reinhard_lab"`.

Typical use: compute stats from one scanner's training slides, then apply color_normalize with
those stats to BOTH that scanner's slides AND a different scanner's slides at graph-build time,
so both land in a shared reference stain space before feature extraction -- this is what closes
a scanner/staining color-domain gap (see docs/pipeline.md, graph corpora). Normalizing
only one side does not help.

Usage:
    python scripts/graph_creation/compute_color_reference_stats.py \\
        --dataset gsdb --split-csv data/splits/gsdb/seed_42/train.csv --scanner-id 1 \\
        --patch-size 128 --patch-level 0 --num-slides 20 --patches-per-slide 40 \\
        --output configs/graph_build/reference_stats/gsdb_scanner1_lab.json --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.entrypoint.build_graph import _resolve_slide_path  # noqa: E402
from src.paths.paths import get_project_paths  # noqa: E402
from src.utils.stain_normalization import estimate_macenko_stain_matrix, rgb_to_optical_density  # noqa: E402


def _is_tissue_patch(arr_rgb: np.ndarray, sat_min: int = 15, val_max: int = 245) -> bool:
    """Cheap background/whitespace filter, same thresholds BuildConfig.sat_min/val_max default
    to for tissue-mask detection -- avoids polluting reference stats with blank slide area."""
    hsv = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2HSV)
    sat_ok = hsv[..., 1].mean() >= sat_min
    val_ok = hsv[..., 2].mean() <= val_max
    return bool(sat_ok and val_ok)


def _read_patch_rgb(slide, xy: tuple[int, int], level: int, patch_size: int) -> np.ndarray:
    downsample = float(slide.level_downsamples[level])
    img = slide.read_region((int(xy[0] * downsample), int(xy[1] * downsample)), level, (patch_size, patch_size))
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[-1] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr


def sample_slide_patches(slide_path: Path, level: int, patch_size: int, n_patches: int, rng: np.random.Generator) -> list[np.ndarray]:
    import openslide

    handle = openslide.OpenSlide(str(slide_path))
    try:
        level_w, level_h = handle.level_dimensions[level]
        max_x = max(1, level_w - patch_size)
        max_y = max(1, level_h - patch_size)

        patches: list[np.ndarray] = []
        attempts = 0
        max_attempts = n_patches * 20
        while len(patches) < n_patches and attempts < max_attempts:
            attempts += 1
            xy = (int(rng.integers(0, max_x)), int(rng.integers(0, max_y)))
            arr = _read_patch_rgb(handle, xy, level, patch_size)
            if _is_tissue_patch(arr):
                patches.append(arr)
        return patches
    finally:
        handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["gsdb", "camelyon"])
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--scanner-id", type=int, default=None, help="Filter split CSV to this scanner_id, if the column exists.")
    parser.add_argument("--stain", type=str, default=None, help="Filter split CSV to this stain value, if the column exists.")
    parser.add_argument("--patch-size", type=int, default=128, help="Should match the actual pipeline's build.patch_size (128 for both GSDB and Camelyon's current configs).")
    parser.add_argument("--patch-level", type=int, default=0)
    parser.add_argument("--num-slides", type=int, default=20)
    parser.add_argument("--patches-per-slide", type=int, default=40)
    parser.add_argument("--slide-dir", type=Path, default=None, help="Defaults to RAW_DIR/slides (gsdb) or RAW_DIR/camelyon/images (camelyon).")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--macenko-beta", type=float, default=0.15, help="OD threshold below which a pixel is treated as background and excluded from stain-vector estimation.")
    parser.add_argument("--macenko-angular-percentile", type=float, default=1.0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.split_csv)
    if args.scanner_id is not None and "scanner_id" in df.columns:
        df = df[df["scanner_id"] == args.scanner_id]
    if args.stain is not None and "stain" in df.columns:
        df = df[df["stain"] == args.stain]
    if df.empty:
        raise ValueError(f"No rows left in {args.split_csv} after filtering (scanner_id={args.scanner_id}, stain={args.stain})")

    slide_names = df["slide_name"].drop_duplicates().tolist()
    if len(slide_names) > args.num_slides:
        idx = rng.choice(len(slide_names), size=args.num_slides, replace=False)
        slide_names = [slide_names[i] for i in sorted(idx)]

    paths = get_project_paths()
    slide_dir = args.slide_dir
    if slide_dir is None:
        slide_dir = paths["RAW_DIR"] / "slides" if args.dataset == "gsdb" else paths["RAW_DIR"] / "camelyon" / "images"

    lab_pixels: list[np.ndarray] = []
    od_pixels: list[np.ndarray] = []
    n_slides_used = 0
    n_patches_total = 0
    for slide_name in slide_names:
        try:
            slide_path = _resolve_slide_path(slide_dir, str(slide_name))
        except FileNotFoundError as exc:
            print(f"[skip] {slide_name}: {exc}")
            continue

        patches = sample_slide_patches(slide_path, args.patch_level, args.patch_size, args.patches_per_slide, rng)
        if not patches:
            print(f"[skip] {slide_name}: no tissue patches found")
            continue
        n_slides_used += 1
        n_patches_total += len(patches)
        for arr in patches:
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float64)
            lab_pixels.append(lab.reshape(-1, 3))
            od_pixels.append(rgb_to_optical_density(arr))
        print(f"[ok] {slide_name}: {len(patches)} patches")

    if not lab_pixels:
        raise RuntimeError("No patches collected from any slide -- check --split-csv/--scanner-id/--slide-dir.")

    all_lab_pixels = np.concatenate(lab_pixels, axis=0)
    means = all_lab_pixels.mean(axis=0).tolist()
    stds = all_lab_pixels.std(axis=0).tolist()

    all_od_pixels = np.concatenate(od_pixels, axis=0)
    stain_matrix, max_concentrations = estimate_macenko_stain_matrix(
        all_od_pixels, beta=args.macenko_beta, angular_percentile=args.macenko_angular_percentile
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stain_matrix": stain_matrix.tolist(),
        "max_concentrations": max_concentrations.tolist(),
        "means": means,
        "stds": stds,
        "meta": {
            "dataset": args.dataset,
            "split_csv": str(args.split_csv),
            "scanner_id": args.scanner_id,
            "stain": args.stain,
            "num_slides_used": n_slides_used,
            "num_patches": n_patches_total,
            "num_pixels": int(all_lab_pixels.shape[0]),
            "patch_size": args.patch_size,
            "patch_level": args.patch_level,
            "seed": args.seed,
            "macenko_beta": args.macenko_beta,
            "macenko_angular_percentile": args.macenko_angular_percentile,
        },
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(
        f"Wrote {args.output}: stain_matrix={stain_matrix.tolist()} "
        f"max_concentrations={max_concentrations.tolist()} means={means} stds={stds} "
        f"(from {n_slides_used} slides, {n_patches_total} patches, {all_lab_pixels.shape[0]} pixels)"
    )


if __name__ == "__main__":
    main()
