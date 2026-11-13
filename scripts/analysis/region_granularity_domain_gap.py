"""Read-only diagnostic: does per-slide region-count/size-distribution alone still separate
GSDB scanner1 from scanner2, after the build_regions() stride-mismatch fix (2026-08-11) that
collapsed scanner2 regions to one-per-patch? Reuses this project's proven domain-classifier-AUC
methodology (previously run on region.x embeddings; here run on region SIZE statistics only, not
feature content at all) to decide whether region-granularity is still worth fixing before writing
any merge/split mechanism for it.

Use with:
  python -u scripts/analysis/region_granularity_domain_gap.py \
    --graph_dir data/processed/graphs/gsdb/context_aware_color_normalized \
    --scanner1_csv data/splits/gsdb/seed_42/train.csv \
    --scanner2_csv data/splits/gsdb/seed_42/test_scanner2_all_stains.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from src.dataset.region_resplit import merge_small_regions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Region-granularity domain-gap diagnostic")
    parser.add_argument("--graph_dir", type=str, required=True)
    parser.add_argument("--scanner1_csv", type=str, required=True)
    parser.add_argument("--scanner2_csv", type=str, required=True)
    parser.add_argument("--floors", type=int, nargs="+", default=[5, 8, 12])
    parser.add_argument(
        "--merge_floor",
        type=int,
        default=0,
        help="If >0, apply merge_small_regions(min_patches_per_region=merge_floor) in-memory to "
        "each graph before computing stats (Stage 2b validation of the merge mechanism). "
        "Read-only: never writes back to the cached .pt files.",
    )
    return parser.parse_args()


def load_split(csv_path: str) -> list[str]:
    with open(csv_path) as f:
        return [row["slide_name"] for row in csv.DictReader(f)]


def graph_path(graph_dir: Path, slide_name: str) -> Path | None:
    p = graph_dir / f"{slide_name}_dynamical_pov-gigapath.pt"
    return p if p.exists() else None


def per_slide_region_stats(
    graph_dir: Path, slide_names: list[str], floors: list[int], merge_floor: int = 0
) -> pd.DataFrame:
    rows = []
    for name in slide_names:
        p = graph_path(graph_dir, name)
        if p is None:
            continue
        d = torch.load(p, weights_only=False)
        if "patch" not in d.node_types or ("patch", "in", "region") not in d.edge_types:
            continue
        if merge_floor > 0:
            merge_small_regions(d, min_patches_per_region=merge_floor)
        p2r = d[("patch", "in", "region")].edge_index
        n_regions = int(d["region"].num_nodes)
        if n_regions == 0:
            continue
        counts = torch.bincount(p2r[1], minlength=n_regions).float().numpy()
        row = {
            "slide_name": name,
            "num_regions": n_regions,
            "mean_patches_per_region": float(counts.mean()),
            "median_patches_per_region": float(np.median(counts)),
            "p90_patches_per_region": float(np.percentile(counts, 90)),
            "min_patches_per_region": float(counts.min()),
            "std_patches_per_region": float(counts.std()),
        }
        for floor in floors:
            row[f"frac_regions_below_{floor}"] = float((counts < floor).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    graph_dir = Path(args.graph_dir)

    s1_slides = load_split(args.scanner1_csv)
    s2_slides = load_split(args.scanner2_csv)

    if args.merge_floor > 0:
        print(f"Applying merge_small_regions(min_patches_per_region={args.merge_floor}) in-memory before stats...")

    print(f"Loading scanner1 ({len(s1_slides)} slides listed)...")
    s1_df = per_slide_region_stats(graph_dir, s1_slides, args.floors, args.merge_floor)
    print(f"  {len(s1_df)}/{len(s1_slides)} resolved")

    print(f"Loading scanner2 ({len(s2_slides)} slides listed)...")
    s2_df = per_slide_region_stats(graph_dir, s2_slides, args.floors, args.merge_floor)
    print(f"  {len(s2_df)}/{len(s2_slides)} resolved")

    feature_cols = [c for c in s1_df.columns if c != "slide_name"]

    print("\n=== Per-slide region-size distribution: scanner1 vs scanner2 ===")
    summary = pd.DataFrame(
        {
            "scanner1_mean": s1_df[feature_cols].mean(),
            "scanner1_median": s1_df[feature_cols].median(),
            "scanner2_mean": s2_df[feature_cols].mean(),
            "scanner2_median": s2_df[feature_cols].median(),
        }
    )
    print(summary.round(3))

    X = pd.concat([s1_df[feature_cols], s2_df[feature_cols]], ignore_index=True).values
    y = np.concatenate([np.zeros(len(s1_df)), np.ones(len(s2_df))])

    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=2000)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = cross_val_predict(clf, Xs, y, cv=skf, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, preds)

    print(f"\n=== Domain classifier AUC using ONLY region-size-distribution features (5-fold CV) ===")
    print(f"AUC = {auc:.4f}  (0.5 = indistinguishable by region granularity alone, 1.0 = trivially separable)")
    print(f"Features used: {feature_cols}")


if __name__ == "__main__":
    main()
