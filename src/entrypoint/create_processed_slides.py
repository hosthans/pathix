from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

import sys

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.paths.paths import add_path_args, get_project_paths
from src.utils.logging import setup_logging, quiet_external_loggers


def _parse_extensions(value: str) -> Tuple[str, ...]:
    raw = [item.strip() for item in value.split(",") if item.strip()]
    normalized: List[str] = []
    for item in raw:
        normalized.append(item if item.startswith(".") else f".{item}")
    return tuple(normalized)


def _resolve_stage_csv(args_input_csv: Optional[Path], paths: Dict[str, Path]) -> Path:
    if args_input_csv is not None:
        return args_input_csv

    candidates = [
        paths["RAW_DIR"] / "camelyon17" / "stages.csv",
        paths["RAW_DIR"] / "camelyon17" / "stage_labels.csv",
        paths["RAW_DIR"] / "camelyon" / "stages.csv",
        paths["RAW_DIR"] / "camelyon" / "stage_labels.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _resolve_annotations_dir(args_annotations_dir: Optional[Path], paths: Dict[str, Path]) -> Path:
    if args_annotations_dir is not None:
        return args_annotations_dir

    candidates = [
        paths["RAW_DIR"] / "camelyon17" / "annotations",
        paths["RAW_DIR"] / "camelyon" / "annotations",
        paths["RAW_DIR"] / "annotations",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _find_annotation_file(slide_id: str, annotations_dir: Path, exts: Tuple[str, ...]) -> Optional[Path]:
    direct_candidates: List[Path] = []
    for ext in exts:
        direct_candidates.append(annotations_dir / f"{slide_id}{ext}")
        direct_candidates.append(annotations_dir / f"{slide_id}_annotations{ext}")

    direct_match = _first_existing(direct_candidates)
    if direct_match is not None:
        return direct_match

    recursive_candidates: List[Path] = []
    for ext in exts:
        recursive_candidates.extend(sorted(annotations_dir.rglob(f"{slide_id}{ext}")))
        recursive_candidates.extend(sorted(annotations_dir.rglob(f"{slide_id}_annotations{ext}")))

    return _first_existing(recursive_candidates)


def _derive_slide_id(value: str) -> str:
    return Path(str(value)).stem.strip()


def build_processed_slides(
    stage_csv: Path,
    slide_col: str,
    stage_col: str,
    center_col: str,
    annotations_dir: Path,
    annotation_extensions: Tuple[str, ...],
) -> pd.DataFrame:
    """Build the processed_slides DataFrame with annotation_used flags."""
    if not stage_csv.exists() or not stage_csv.is_file():
        raise FileNotFoundError(f"Stage CSV not found: {stage_csv}")

    if not annotations_dir.exists() or not annotations_dir.is_dir():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    df = pd.read_csv(stage_csv)
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    required = [slide_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    rows: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        slide_raw = str(row.get(slide_col, "")).strip()
        if not slide_raw:
            continue

        slide_id = _derive_slide_id(slide_raw)
        annotation_path = _find_annotation_file(slide_id, annotations_dir, annotation_extensions)
        annotation_used = "yes" if annotation_path is not None else "no"

        out_row: Dict[str, str] = {
            "slide_id": slide_id,
            "annotation_used": annotation_used,
            "annotation_path": str(annotation_path) if annotation_path is not None else "",
            "annotation_ext": annotation_path.suffix.lower() if annotation_path is not None else "",
            "slide_file": slide_raw,
        }

        if stage_col in df.columns:
            out_row["stage"] = str(row.get(stage_col, "")).strip()
        if center_col in df.columns:
            out_row["center"] = str(row.get(center_col, "")).strip()

        rows.append(out_row)

    if not rows:
        raise ValueError("No slide rows found after reading input CSV")

    out_df = pd.DataFrame(rows)
    before = len(out_df)
    out_df = out_df.drop_duplicates(subset=["slide_id"], keep="first").reset_index(drop=True)
    dropped = before - len(out_df)
    if dropped > 0:
        logging.warning("Dropped %d duplicate slide_id rows while building processed CSV", dropped)

    return out_df


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create processed_slides.csv with annotation_used for CAMELYON split generation"
    )
    parser = add_path_args(parser)
    parser.add_argument("--input-csv", type=Path, default=None, help="Path to stage CSV (e.g. stages.csv)")
    parser.add_argument("--slide-col", type=str, default="patient", help="Slide/file column name in stage CSV")
    parser.add_argument("--stage-col", type=str, default="stage", help="Optional stage column name")
    parser.add_argument("--center-col", type=str, default="center", help="Optional center column name")
    parser.add_argument("--annotations-dir", type=Path, default=None, help="Directory containing annotation files")
    parser.add_argument(
        "--annotation-extensions",
        type=str,
        default=".xml,.json,.geojson",
        help="Comma-separated annotation extensions to consider",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output path for processed_slides.csv (default: data/processed/camelyon/processed_slides.csv)",
    )
    return parser.parse_args()


def main() -> None:
    """Build and save processed_slides.csv from a stage CSV and annotations directory."""
    args = parse_args()

    paths = get_project_paths(base_dir=args.base_dir)

    if args.data_dir:
        paths["DATA_DIR"] = args.data_dir
        paths["RAW_DIR"] = args.data_dir / "raw"
        paths["PROCESSED_DIR"] = args.data_dir / "processed"

    if args.output_dir:
        paths["RESULTS_DIR"] = args.output_dir
        paths["LOGS_DIR"] = args.output_dir / "logs"

    # Own subfolder under LOGS_DIR (not flat) -- see create_splits.py's identical comment.
    logs_dir = paths["LOGS_DIR"] / "create_processed_slides"
    logs_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(logs_dir, operation="create_processed_slides", force_timestamp=True)
    quiet_external_loggers()

    stage_csv = _resolve_stage_csv(args.input_csv, paths)
    annotations_dir = _resolve_annotations_dir(args.annotations_dir, paths)
    annotation_extensions = _parse_extensions(args.annotation_extensions)

    output_csv = args.output_csv
    if output_csv is None:
        output_csv = paths["PROCESSED_DIR"] / "camelyon" / "processed_slides.csv"

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Using stage CSV: %s", stage_csv)
    logging.info("Using annotations dir: %s", annotations_dir)
    logging.info("Using annotation extensions: %s", ", ".join(annotation_extensions))

    out_df = build_processed_slides(
        stage_csv=stage_csv,
        slide_col=args.slide_col,
        stage_col=args.stage_col,
        center_col=args.center_col,
        annotations_dir=annotations_dir,
        annotation_extensions=annotation_extensions,
    )

    out_df = out_df.sort_values(["slide_id"]).reset_index(drop=True)
    out_df.to_csv(output_csv, index=False)

    annotated_count = int((out_df["annotation_used"].astype(str).str.lower() == "yes").sum())
    logging.info("Processed slides saved: %s", output_csv)
    logging.info("Total slides: %d", len(out_df))
    logging.info("Annotated slides: %d", annotated_count)
    logging.info("Unannotated slides: %d", len(out_df) - annotated_count)


if __name__ == "__main__":
    main()
