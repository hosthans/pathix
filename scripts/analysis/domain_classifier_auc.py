"""Read-only diagnostic: can a simple classifier tell scanner1 from scanner2 slides using ONLY
their region-embedding content (region.x, post-encoder/post-color-transform)? Reused this session
on region-size-distribution statistics (scripts/analysis/region_granularity_domain_gap.py) and,
generalized here as a standalone script, for the Stage 3 color_normalize blend_strength ablation
-- run once per candidate graph corpus (e.g. one throwaway build per blend_strength value) to see
which color treatment narrows the scanner1/scanner2 embedding gap most, without paying for a full
rebuild or a full retrain.

Per slide: mean-pool every region node's `.x` (pov-gigapath embedding, post color-transform) into
one 1536-dim slide vector. Logistic regression + StratifiedKFold cross-validation on these vectors,
scanner label as target -- AUC 0.5 = indistinguishable, 1.0 = trivially separable (still leaking
scanner identity through embedding content, independent of any label signal).

Use with:
  python -u scripts/analysis/domain_classifier_auc.py \
    --graph_dir data/processed/graphs/gsdb/blend_ablation_TEST_blend10 \
    --scanner1_csv data/splits/gsdb/blend_ablation_TEST.csv \
    --scanner2_csv data/splits/gsdb/blend_ablation_TEST.csv \
    --scanner_column scanner_id
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Region-embedding domain-gap diagnostic")
    parser.add_argument("--graph_dir", type=str, required=True)
    parser.add_argument(
        "--scanner1_csv", type=str, required=True,
        help="CSV with a slide_name column. If it also has a scanner_id column matching "
        "--scanner_column, rows are filtered to scanner1; otherwise every listed slide is "
        "treated as scanner1.",
    )
    parser.add_argument("--scanner2_csv", type=str, required=True, help="Same as --scanner1_csv, for scanner2.")
    parser.add_argument("--scanner_column", type=str, default="scanner_id")
    parser.add_argument("--pipeline_suffix", type=str, default="_dynamical_pov-gigapath.pt")
    return parser.parse_args()


def load_slides(csv_path: str, scanner_column: str, scanner_id: int) -> list[str]:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if rows and scanner_column in rows[0]:
        rows = [r for r in rows if str(r[scanner_column]) == str(scanner_id)]
    return [r["slide_name"] for r in rows]


def slide_embeddings(graph_dir: Path, slide_names: list[str], suffix: str) -> tuple[np.ndarray, list[str]]:
    vecs, resolved = [], []
    for name in slide_names:
        p = graph_dir / f"{name}{suffix}"
        if not p.exists():
            continue
        d = torch.load(p, weights_only=False)
        if "region" not in d.node_types or not hasattr(d["region"], "x") or d["region"].x is None:
            continue
        x = d["region"].x
        if x.numel() == 0:
            continue
        vecs.append(x.mean(dim=0).numpy())
        resolved.append(name)
    return np.stack(vecs) if vecs else np.zeros((0, 0)), resolved


def main() -> None:
    args = parse_args()
    graph_dir = Path(args.graph_dir)

    s1_slides = load_slides(args.scanner1_csv, args.scanner_column, 1)
    s2_slides = load_slides(args.scanner2_csv, args.scanner_column, 2)

    print(f"scanner1: {len(s1_slides)} slides listed")
    print(f"scanner2: {len(s2_slides)} slides listed")

    X1, r1 = slide_embeddings(graph_dir, s1_slides, args.pipeline_suffix)
    X2, r2 = slide_embeddings(graph_dir, s2_slides, args.pipeline_suffix)
    print(f"scanner1: {len(r1)}/{len(s1_slides)} resolved in {graph_dir}")
    print(f"scanner2: {len(r2)}/{len(s2_slides)} resolved in {graph_dir}")

    if len(r1) < 3 or len(r2) < 3:
        print("Too few resolved slides on one side for a meaningful CV split -- aborting.")
        return

    X = np.concatenate([X1, X2], axis=0)
    y = np.concatenate([np.zeros(len(r1)), np.ones(len(r2))])

    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=2000)
    n_splits = max(2, min(5, min(len(r1), len(r2))))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    preds = cross_val_predict(clf, Xs, y, cv=skf, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, preds)

    print(f"\n=== Domain classifier AUC using mean-pooled region.x embeddings ({n_splits}-fold CV) ===")
    print(f"AUC = {auc:.4f}  (0.5 = indistinguishable, 1.0 = trivially separable)")
    print(f"Corpus: {graph_dir}")


if __name__ == "__main__":
    main()
