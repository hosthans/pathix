"""Create the GSDB train/val/test splits (patient-level, scanner-1 pool; scanner 2 -> test_scanner2).

GSDB only. CAMELYON17 splits are created with src/entrypoint/create_camelyon_splits.py.

Output: data/splits/gsdb/seed_<seed>/{train,val,test,test_scanner2}{,_HE,_all_stains}.csv and,
with --kfold-k K, kfold_k<K>_fold<i>/{train,val}*.csv.

Usage:
    python src/entrypoint/create_splits.py --seed 42 --kfold-k 5
"""
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Set
import argparse
from sklearn.model_selection import train_test_split, GroupKFold

import sys
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from src.paths.paths import get_project_paths, add_path_args
from src.utils.logging import setup_logging, quiet_external_loggers


class DataSplitter:
    def __init__(self, paths=None, random_seed: int = 42):
        """Initialize DataSplitter with project paths and configuration."""
        if paths is None:
            paths = get_project_paths()

        self.base_dir = paths["BASE_DIR"]
        self.seed = random_seed
        self.dataset_name = "gsdb"

        self.data_dir = paths["DATA_DIR"]
        self.raw_dir = paths["RAW_DIR"]
        self.processed_dir = paths["PROCESSED_DIR"]
        self.splits_dir = paths["SPLITS_DIR"]
        self.dataset_splits_dir = self.splits_dir / self.dataset_name
        # Own subfolder under LOGS_DIR (not flat) -- every split_creation run's log lands in the
        # same place instead of accumulating as loose files alongside every other operation's logs.
        self.logs_dir = paths["LOGS_DIR"] / "split_creation"
        self.figures_dir = paths["FIGURES_DIR"]
        self.tables_dir = paths["TABLES_DIR"]

        for dir_path in [self.splits_dir, self.dataset_splits_dir, self.logs_dir, self.figures_dir, self.tables_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

    def _setup_logging(self):
        setup_logging(self.logs_dir, operation="split_creation", force_timestamp=True)
        quiet_external_loggers()

    def _positive_mask_for_split(self, split_df: pd.DataFrame) -> pd.Series:
        if "inflammation_status" in split_df.columns:
            return split_df["inflammation_status"].astype(str).str.strip().str.lower() != "noninflamed"
        raise ValueError("Cannot apply --only-positives: 'inflammation_status' is not present in split data")

    def filter_only_positive_splits(
        self,
        splits: Dict[str, pd.DataFrame],
        required_nonempty_splits: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Filter splits to positive samples only."""
        result: Dict[str, pd.DataFrame] = {}
        required = required_nonempty_splits or ["train", "val", "test"]

        logging.info("Applying --only-positives filtering to split outputs")

        for split_name, split_df in splits.items():
            if split_df.empty:
                result[split_name] = split_df.copy()
                continue

            positive_mask = self._positive_mask_for_split(split_df)
            before_count = len(split_df)
            after_df = split_df.loc[positive_mask].copy()
            after_count = len(after_df)

            logging.info(
                "only-positives filter [%s]: before=%d after=%d removed=%d",
                split_name,
                before_count,
                after_count,
                before_count - after_count,
            )

            result[split_name] = after_df

        for split_name in required:
            split_df = result.get(split_name)
            if split_df is None:
                continue
            if split_df.empty:
                raise ValueError(
                    f"--only-positives produced an empty '{split_name}' split. Disable --only-positives."
                )

        return result

    def create_slide_info_df(self) -> pd.DataFrame:
        """Create DataFrame with slide information from annotations."""
        annotations_dir = self.raw_dir / "annotations"
        slides_data = []

        for root, _, files in os.walk(annotations_dir):
            for file in files:
                if not file.endswith('.json'):
                    continue

                try:
                    file_path = Path(root) / file
                    with open(file_path) as f:
                        data = json.load(f)

                        if not data:
                            continue

                        base_name = file.replace('_annotations.json', '')
                        parts = base_name.split('_')

                        slide_id = parts[0]
                        patient_id = parts[1]
                        scanner_id = int(parts[2])
                        stain = parts[3]

                        inflammation_status = data[0]['properties']['classification']['inflammation_status']

                        slides_data.append({
                            'slide_name': f"{slide_id}_{patient_id}_{scanner_id}_{stain}",
                            'slide_id': slide_id,
                            'patient_id': patient_id,
                            'scanner_id': scanner_id,
                            'stain': stain,
                            'inflammation_status': inflammation_status
                        })

                except Exception as e:
                    logging.error(f"Error processing {file}: {str(e)}")
                    continue

        df_slides = pd.DataFrame(slides_data)
        return df_slides

    def create_split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create train/val/test splits based on patient_id."""
        np.random.seed(self.seed)
        random.seed(self.seed)
        logging.info(f"Set random seed to {self.seed} for reproducible data splitting")

        logging.info(f"\nInitial dataset statistics:")
        logging.info(f"Total patients before exclusion: {df['patient_id'].nunique()}")
        logging.info(f"Total slides before exclusion: {len(df)}")

        excluded_patients = {'4', '9', '14', '18', '24', '29', '34', '39', '44', '49',
                            '54', '59', '64', '69', '74', '75', '79', '83', '180', '197',
                             '213', '217'}
        logging.info(f"\nExcluding {len(excluded_patients)} patients:")
        logging.info(f"Excluded patient IDs: {sorted(excluded_patients)}")

        df_filtered = df[~df['patient_id'].isin(excluded_patients)]
        logging.info(f"\nDataset statistics after exclusion:")
        logging.info(f"Remaining patients: {df_filtered['patient_id'].nunique()}")
        logging.info(f"Remaining slides: {len(df_filtered)}")
        logging.info(f"Removed slides: {len(df) - len(df_filtered)}")

        df_scanner1 = df_filtered[df_filtered['scanner_id'] == 1]
        df_scanner2 = df_filtered[df_filtered['scanner_id'] == 2]

        logging.info(f"\nScanner distribution:")
        logging.info(f"Scanner 1 slides: {len(df_scanner1)}")
        logging.info(f"Scanner 1 patients: {df_scanner1['patient_id'].nunique()}")
        logging.info(f"Scanner 2 slides: {len(df_scanner2)}")
        logging.info(f"Scanner 2 patients: {df_scanner2['patient_id'].nunique()}")

        unique_patients = np.sort(df_scanner1['patient_id'].unique())

        train_patients, temp_patients = train_test_split(
            unique_patients,
            test_size=0.4,
            random_state=self.seed + 2
        )

        val_patients, test_patients = train_test_split(
            temp_patients,
            test_size=0.5,
            random_state=self.seed + 2
        )

        logging.info(f"\nSplit sizes (Scanner 1 patients):")
        logging.info(f"Training: {len(train_patients)} patients ({len(train_patients)/len(unique_patients):.1%})")
        logging.info(f"Validation: {len(val_patients)} patients ({len(val_patients)/len(unique_patients):.1%})")
        logging.info(f"Test: {len(test_patients)} patients ({len(test_patients)/len(unique_patients):.1%})")

        splits = {
            'train': df_scanner1[df_scanner1['patient_id'].isin(train_patients)],
            'val': df_scanner1[df_scanner1['patient_id'].isin(val_patients)],
            'test': df_scanner1[df_scanner1['patient_id'].isin(test_patients)],
            'test_scanner2': df_scanner2
        }

        for split_name, split_df in splits.items():
            logging.info(f"\n{split_name.upper()} split:")
            logging.info(f"Slides: {len(split_df)}")
            logging.info(f"Patients: {split_df['patient_id'].nunique()}")
            logging.info("Inflammation distribution:")
            logging.info(split_df['inflammation_status'].value_counts())

        self._create_split_variants(splits)
        self._validate_splits(splits)

        return splits

    def create_gsdb_kfold_splits(self, splits: Dict[str, pd.DataFrame], k: int) -> None:
        """Generate k patient-grouped, complementary CV folds over the train+val scanner1 pool."""
        pool_df = pd.concat([splits['train'], splits['val']], ignore_index=True)
        groups = pool_df['patient_id'].to_numpy()

        gkf = GroupKFold(n_splits=int(k))
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(pool_df, groups=groups)):
            train_df = pool_df.iloc[train_idx]
            val_df = pool_df.iloc[val_idx]

            train_patients = set(train_df['patient_id'].unique())
            val_patients = set(val_df['patient_id'].unique())
            overlap = train_patients & val_patients
            if overlap:
                raise ValueError(
                    f"kfold fold {fold_idx}: train/val patient overlap despite GroupKFold: {sorted(overlap)}"
                )

            logging.info(
                "kfold fold %d/%d: train=%d slides/%d patients, val=%d slides/%d patients",
                fold_idx, k, len(train_df), len(train_patients), len(val_df), len(val_patients),
            )
            logging.info("  train inflammation distribution: %s", dict(train_df['inflammation_status'].value_counts()))
            logging.info("  val inflammation distribution: %s", dict(val_df['inflammation_status'].value_counts()))

            fold_dir = self.dataset_splits_dir / f'seed_{self.seed}' / f'kfold_k{k}_fold{fold_idx}'
            self._write_split_variants_to_dir({'train': train_df, 'val': val_df}, fold_dir)

        logging.info("Wrote %d-fold CV splits under %s/seed_%s/kfold_k%d_fold*", k, self.dataset_splits_dir, self.seed, k)

    def _create_split_variants(self, splits: Dict[str, pd.DataFrame]):
        self._write_split_variants_to_dir(splits, self.dataset_splits_dir / f'seed_{self.seed}')

    def _write_split_variants_to_dir(self, splits: Dict[str, pd.DataFrame], output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_df in splits.items():
            split_df.to_csv(output_dir / f'{split_name}.csv', index=False)

            if 'stain' in split_df.columns:
                he_only = split_df[split_df['stain'] == 'HE']
                he_only.to_csv(output_dir / f'{split_name}_HE.csv', index=False)
                split_df.to_csv(output_dir / f'{split_name}_all_stains.csv', index=False)

    def _validate_splits(self, splits: Dict[str, pd.DataFrame], enforce_patient_disjoint: bool = True):
        logging.info("\nValidating splits...")

        if enforce_patient_disjoint:
            used_patients: Set[str] = set()
            for name, split_df in splits.items():
                if name != 'test_scanner2':
                    current_patients = set(split_df['patient_id'])
                    overlap = current_patients & used_patients
                    assert not overlap, f"Found patient_id overlap in {name}: {overlap}"
                    used_patients.update(current_patients)
        else:
            logging.info("Patient overlap validation disabled")

        if 'test_scanner2' in splits and any('scanner_id' in df.columns for df in splits.values()):
            for name, split_df in splits.items():
                if 'scanner_id' not in split_df.columns:
                    continue
                if name == 'test_scanner2':
                    assert all(split_df['scanner_id'] == 2), \
                        "Scanner 2 test set contains non-scanner-2 slides"
                else:
                    assert all(split_df['scanner_id'] == 1), \
                        f"{name} split contains non-scanner-1 slides"

        self._log_split_statistics(splits)

    def _log_split_statistics(self, splits: Dict[str, pd.DataFrame]):
        for name, split_df in splits.items():
            logging.info(f"\n{name} split statistics:")
            logging.info(f"Total samples: {len(split_df)}")
            logging.info(f"Unique patient_ids: {split_df['patient_id'].nunique()}")

            if 'stain' in split_df.columns:
                logging.info("\nStain distribution:")
                logging.info(split_df['stain'].value_counts())

            if 'inflammation_status' in split_df.columns:
                logging.info("\nInflammation distribution:")
                logging.info(split_df['inflammation_status'].value_counts())

            if 'stain' in split_df.columns and 'inflammation_status' in split_df.columns:
                he_only = split_df[split_df['stain'] == 'HE']
                logging.info(f"\nHE-only samples: {len(he_only)}")
                logging.info("HE inflammation distribution:")
                logging.info(he_only['inflammation_status'].value_counts())


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Create GSDB dataset splits (CAMELYON17: src/entrypoint/create_camelyon_splits.py)"
    )
    parser = add_path_args(parser)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsdb"],
        default="gsdb",
        help="Kept for backward compatibility; only gsdb is supported "
             "(CAMELYON17: src/entrypoint/create_camelyon_splits.py)",
    )
    parser.add_argument(
        "--kfold-k",
        type=int,
        default=0,
        help=(
            "If >0, also generate this many patient-grouped CV folds over the "
            "train+val scanner1 pool (test/test_scanner2 stay fixed, shared across all folds). "
            "Written to data/splits/gsdb/seed_<seed>/kfold_k<K>_fold<i>/{train,val}*.csv -- use "
            "with --split-phase kfold_k<K>_fold<i> on train_model.py to train on one fold."
        ),
    )
    parser.add_argument(
        "--only-positives",
        action="store_true",
        default=False,
        help="Keep only inflamed slides (inflammation_status != noninflamed) in the exported split CSVs.",
    )
    parser.add_argument(
        "--allow-patient-overlap",
        action="store_false",
        dest="enforce_patient_disjoint",
        default=True,
        help="Skip the patient-disjointness check when re-validating --only-positives splits (not recommended).",
    )
    args = parser.parse_args()

    paths = get_project_paths(base_dir=args.base_dir)

    if args.data_dir:
        paths["DATA_DIR"] = args.data_dir
        paths["RAW_DIR"] = args.data_dir / "raw"
        paths["PROCESSED_DIR"] = args.data_dir / "processed"
        paths["SPLITS_DIR"] = args.data_dir / "splits"

    if args.output_dir:
        paths["RESULTS_DIR"] = args.output_dir
        paths["LOGS_DIR"] = args.output_dir / "logs"
        paths["FIGURES_DIR"] = args.output_dir / "figures"
        paths["TABLES_DIR"] = args.output_dir / "tables"

    start_time = datetime.now()

    try:
        splitter = DataSplitter(paths=paths, random_seed=args.seed)

        logging.info(f"Starting split creation at {start_time}")
        logging.info(f"Using base directory: {paths['BASE_DIR']}")
        logging.info("Dataset mode: gsdb")
        logging.info("Effective input mode: annotations")
        logging.info(f"Split output directory: {splitter.dataset_splits_dir / f'seed_{args.seed}'}")

        df = splitter.create_slide_info_df()
        splits = splitter.create_split(df)
        if args.kfold_k and int(args.kfold_k) > 0:
            splitter.create_gsdb_kfold_splits(splits, k=int(args.kfold_k))
        if args.only_positives:
            splits = splitter.filter_only_positive_splits(splits)
            splitter._validate_splits(splits, enforce_patient_disjoint=args.enforce_patient_disjoint)
            splitter._create_split_variants(splits)

        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"Split creation completed successfully. Duration: {duration}")

    except Exception as e:
        logging.error(f"Error during split creation: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
