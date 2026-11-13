"""Create a CAMELYON17 split with an explicit WHOLE-CENTER train/val/test assignment -- the
convention used in most CAMELYON17 papers (e.g. centers 0,1,2 -> train, center 3 -> val,
center 4 -> test), as opposed to a ratio-based carve-out within a pooled set of centers.

GSDB splits are created with src/entrypoint/create_splits.py.

Three annotation-composition modes for train/val (test is always left at its natural,
unbalanced, all-annotation-statuses distribution):
  - natural:        keep all tumor slides (annotated and unannotated)
  - annotated_only: keep only annotation_used=yes tumor slides
  - majority:       cap unannotated tumor slides so annotated is >= --target-annotated-fraction
                     of each split's tumor slides

Train and val are additionally subsampled to an exact 1:1 tumor:normal ratio (normal slides drawn
with seed+2 for train, seed+3 for val). Patients are disjoint across splits by construction
(each patient belongs to exactly one center) and this is verified before writing.

Output: data/splits/camelyon/seed_<seed>/{train,val,test}.csv (with --phase <name>:
data/splits/camelyon/seed_<seed>/<name>/, read with --split-phase <name>); log under
results/logs/split_creation/.

Center rotation (every center is the test center once; val = the preceding center):
    python src/entrypoint/create_camelyon_splits.py --mode natural --seed 42 \\
        --train-centers 1,2,3 --val-center 4 --test-center 0 --phase test_c0
    (… test_c1 … test_c4; test_c4 is identical to seed_42 itself)

Usage (the split used in the experiments, data/splits/camelyon/seed_42):
    python src/entrypoint/create_camelyon_splits.py --mode natural --seed 42 \\
        --train-centers 0,1,2 --val-center 3 --test-center 4

Other modes:
    python src/entrypoint/create_camelyon_splits.py --mode annotated_only --seed 62
    python src/entrypoint/create_camelyon_splits.py --mode majority --seed 63 --target-annotated-fraction 0.65
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from src.paths.paths import get_project_paths, add_path_args
from src.utils.logging import setup_logging, quiet_external_loggers
from src.utils.split_utils import (
    PROCESSED_CSV,
    SPLIT_COLUMNS,
    STAGES_CSV,
    apply_composition,
    assert_patient_disjoint,
    balance_1to1,
    load_camelyon_slides,
)


def build_center_split(
    df: pd.DataFrame,
    mode: str,
    seed: int,
    train_centers,
    val_center: int,
    test_center: int,
    target_annotated_fraction: float = 0.65,
) -> Dict[str, pd.DataFrame]:
    """Whole-center split: compose train/val per `mode`, balance them 1:1, leave test untouched."""
    df_train_raw = df[df.center.isin(train_centers)].copy()
    df_val_raw = df[df.center == val_center].copy()
    df_test_raw = df[df.center == test_center].copy()

    df_train = apply_composition(df_train_raw, mode, target_annotated_fraction, seed)
    df_val = apply_composition(df_val_raw, mode, target_annotated_fraction, seed + 1)
    df_train = balance_1to1(df_train, seed + 2)
    df_val = balance_1to1(df_val, seed + 3)
    df_test = df_test_raw  # natural, untouched

    return {"train": df_train, "val": df_val, "test": df_test}


def _log_composition(splits: Dict[str, pd.DataFrame]) -> None:
    for split_name, split_df in splits.items():
        tumor = split_df[split_df.tumor_status == "tumor"]
        ann = int((tumor.annotation_used == "yes").sum())
        unann = int((tumor.annotation_used == "no").sum())
        normal = int((split_df.tumor_status == "normal").sum())
        logging.info(
            "%s: total=%d tumor=%d (annotated=%d, unannotated=%d, annotated_fraction=%.2f) normal=%d "
            "patients=%d centers=%s",
            split_name, len(split_df), len(tumor), ann, unann, ann / max(1, len(tumor)), normal,
            split_df.patient_id.nunique(), sorted(int(c) for c in split_df.center.unique()),
        )
        logging.info("%s stage distribution: %s", split_name, dict(split_df.stage.value_counts()))


def main():
    parser = argparse.ArgumentParser(
        description="Create a whole-center CAMELYON17 split (GSDB: src/entrypoint/create_splits.py)"
    )
    parser = add_path_args(parser)
    parser.add_argument("--mode", choices=["natural", "annotated_only", "majority"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-centers", type=str, default="0,1,2")
    parser.add_argument("--val-center", type=int, default=3)
    parser.add_argument("--test-center", type=int, default=4)
    parser.add_argument("--target-annotated-fraction", type=float, default=0.65,
                        help="Only used with --mode majority")
    parser.add_argument("--phase", type=str, default=None,
                        help="Write to data/splits/camelyon/seed_<seed>/<phase>/ instead of seed_<seed>/ "
                             "(read in training/evaluation with --split-phase <phase>), e.g. one folder per "
                             "test-center rotation")
    parser.add_argument("--stages-csv", type=Path, default=STAGES_CSV,
                        help="CAMELYON17 stage CSV (slide -> stage, center)")
    parser.add_argument("--processed-csv", type=Path, default=PROCESSED_CSV,
                        help="Processed-slides CSV with the annotation_used column")
    args = parser.parse_args()

    paths = get_project_paths(base_dir=args.base_dir)
    if args.data_dir:
        paths["DATA_DIR"] = args.data_dir
        paths["SPLITS_DIR"] = args.data_dir / "splits"
    if args.output_dir:
        paths["RESULTS_DIR"] = args.output_dir
        paths["LOGS_DIR"] = args.output_dir / "logs"

    # Own subfolder under LOGS_DIR, shared with create_splits.py (GSDB).
    setup_logging(paths["LOGS_DIR"] / "split_creation", operation="camelyon_split_creation", force_timestamp=True)
    quiet_external_loggers()

    train_centers = [int(c) for c in args.train_centers.split(",") if c.strip()]
    out_dir = paths["SPLITS_DIR"] / "camelyon" / f"seed_{args.seed}"
    if args.phase:
        out_dir = out_dir / args.phase
    start_time = datetime.now()

    try:
        logging.info("Starting CAMELYON17 split creation at %s", start_time)
        logging.info("Command: %s", " ".join(sys.argv))
        logging.info(
            "Mode: %s | seed=%d | train_centers=%s val_center=%d test_center=%d | target_annotated_fraction=%s",
            args.mode, args.seed, train_centers, args.val_center, args.test_center,
            args.target_annotated_fraction if args.mode == "majority" else "n/a",
        )
        logging.info("Stage CSV: %s | processed CSV: %s", args.stages_csv, args.processed_csv)
        logging.info("Split output directory: %s", out_dir)

        df = load_camelyon_slides(stages_csv=args.stages_csv, processed_csv=args.processed_csv)
        splits = build_center_split(
            df, args.mode, args.seed, train_centers, args.val_center, args.test_center,
            args.target_annotated_fraction,
        )

        assert_patient_disjoint({name: set(split_df.patient_id) for name, split_df in splits.items()})
        _log_composition(splits)

        out_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_df in splits.items():
            split_df[SPLIT_COLUMNS].to_csv(out_dir / f"{split_name}.csv", index=False)

        logging.info("Split creation completed successfully. Duration: %s", datetime.now() - start_time)
    except Exception as e:
        logging.error("Error during split creation: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
