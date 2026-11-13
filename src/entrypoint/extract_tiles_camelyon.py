import argparse
import csv
import logging
import re
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np
import openslide


project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.preprocessing.Preprocessor import PreprocessConfig, TilePreprocessing
from src.paths.paths import get_project_paths
from src.utils.logging import setup_logging, quiet_external_loggers


SUPPORTED_EXTENSIONS = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu")


CAMELYON_PRESET = {
    "patch_size": 256,
    "step_size": 192,
    "patch_level": 2,
    "seg": True,
    "patch": True,
    "save_mask": True,
    "recursive": True,
    "seg_params": {
        "seg_level": -1,
        "sthresh": 8,
        "sthresh_up": 255,
        "mthresh": 7,
        "close": 4,
        "use_otsu": True,
        "keep_ids": "none",
        "exclude_ids": "none",
    },
    "filter_params": {"a_t": 100, "a_h": 16, "max_n_holes": 8},
    "patch_params": {
        "use_padding": True,
        "contour_fn": "four_pt",
        "filter_black_white": True,
        "white_thresh": 225,
        "white_pct": 0.25,
        "black_thresh": 15,
        "black_pct": 0.10,
    },
}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="CAMELYON Tile-Extraktion via Preprocessing-Pipeline")
    parser.add_argument("--base-dir", type=Path, default=None, help="Projekt-Basisverzeichnis")
    parser.add_argument("--source", type=Path, default=None, help="Optionaler Override für CAMELYON-WSIs")
    parser.add_argument("--annotations", type=Path, default=None, help="Optionaler Override für CAMELYON-XML-Annotationen")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optionaler Override für CAMELYON Stage-Labels CSV (Spalten: patient, stage)",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=None,
        help="Optionaler Override für Basis-Ausgabeordner (enthält 'tiles' und 'logs')",
    )
 
    parser.add_argument("--patch_size", type=int, default=None, help="Optionaler Override für Patchgröße")
    parser.add_argument("--step_size", type=int, default=None, help="Optionaler Override für Schrittweite")
    parser.add_argument("--patch_level", type=int, default=None, help="Optionaler Override für Patch-Level")
    parser.add_argument(
        "--tumor_threshold",
        type=float,
        default=0.10,
        help="Tumor-Anteil [0,1], ab dem ein Tile als Tumor gelabelt wird",
    )
    parser.add_argument("--no_masks", action="store_true", help="Maskenexport deaktivieren")
    parser.add_argument("--no_auto_skip", default=True, action="store_false", help="Existierende Slides erneut verarbeiten")
    parser.add_argument("--non_recursive", action="store_true", help="Source nicht rekursiv durchsuchen")
    return parser


def _discover_slides(source: str, recursive: bool) -> list[Path]:
    source_path = Path(source)
    if source_path.is_file():
        return [source_path] if source_path.suffix.lower() in SUPPORTED_EXTENSIONS else []

    iterator = source_path.rglob("*") if recursive else source_path.iterdir()
    return [entry for entry in sorted(iterator) if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS]


def _discover_annotations(annotation_root: str, recursive: bool) -> dict[str, Path]:
    root = Path(annotation_root)
    iterator = root.rglob("*.xml") if recursive else root.glob("*.xml")
    return {xml_file.stem: xml_file for xml_file in sorted(iterator) if xml_file.is_file()}


def _resolve_labels_csv_path(args_labels_csv: Path | None, paths: dict[str, Path]) -> Path:
    if args_labels_csv is not None:
        return args_labels_csv

    candidates = [
        paths["RAW_DIR"] / "camelyon" / "stage_labels.csv",
        paths["RAW_DIR"] / "camelyon17" / "stage_labels.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _load_ground_truth_labels(labels_csv: Path) -> dict[str, str]:
    if not labels_csv.exists():
        raise FileNotFoundError(f"Stage-Labels CSV nicht gefunden: {labels_csv}")

    labels_by_key: dict[str, str] = {}
    with open(labels_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Leere CSV ohne Header: {labels_csv}")

        fields = {name.strip().lower(): name for name in reader.fieldnames}
        patient_col = fields.get("patient")
        stage_col = fields.get("stage")
        if patient_col is None or stage_col is None:
            raise ValueError(
                f"CSV muss Spalten 'patient' und 'stage' enthalten: {labels_csv}"
            )

        for row in reader:
            patient_raw = str(row.get(patient_col, "")).strip()
            stage_raw = str(row.get(stage_col, "")).strip()
            if not patient_raw:
                continue

            patient_path = Path(patient_raw)
            labels_by_key[patient_path.name.lower()] = stage_raw
            labels_by_key[patient_path.stem.lower()] = stage_raw

    return labels_by_key


def _get_ground_truth_stage(slide_path: Path, labels_by_key: dict[str, str]) -> str:
    return labels_by_key.get(slide_path.name.lower(), labels_by_key.get(slide_path.stem.lower(), ""))


def _is_tumor_stage(stage: str) -> bool | None:
    stage_norm = stage.strip().lower()
    if stage_norm in {"", "nan", "none", "null", "na", "n/a"}:
        return None

    if stage_norm in {"negative", "normal", "pn0"}:
        return False

    if stage_norm.startswith("pn0") and "i+" not in stage_norm:
        return False

    return True


def _binary_ground_truth_label(stage: str) -> str:
    is_tumor = _is_tumor_stage(stage)
    if is_tumor is True:
        return "tumor"
    if is_tumor is False:
        return "normal"
    return "unknown"


def _load_tumor_polygons(xml_path: Path) -> list[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    polygons: list[dict] = []

    for annotation in root.findall(".//Annotation"):
        label = annotation.attrib.get("PartOfGroup", "").strip().lower()
        if "tumor" not in label:
            continue

        coords = []
        for coord in annotation.findall(".//Coordinate"):
            coords.append((float(coord.attrib["X"]), float(coord.attrib["Y"])))

        if len(coords) < 3:
            continue

        points = np.array(coords, dtype=np.float32)
        min_x = float(np.min(points[:, 0]))
        min_y = float(np.min(points[:, 1]))
        max_x = float(np.max(points[:, 0]))
        max_y = float(np.max(points[:, 1]))
        polygons.append(
            {
                "points": points,
                "bbox": (min_x, min_y, max_x, max_y),
            }
        )

    return polygons


def _bbox_intersects(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def _tumor_ratio_for_patch(
    x0: int,
    y0: int,
    window_width: int,
    window_height: int,
    polygons: list[dict],
) -> float:
    if window_width <= 0 or window_height <= 0:
        return 0.0

    patch_box = (float(x0), float(y0), float(x0 + window_width), float(y0 + window_height))
    local_mask = np.zeros((window_height, window_width), dtype=np.uint8)
    used = False

    for poly in polygons:
        if not _bbox_intersects(poly["bbox"], patch_box):
            continue

        shifted = np.round(poly["points"] - np.array([x0, y0], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(local_mask, [shifted], 1)
        used = True

    if not used:
        return 0.0

    return float(np.mean(local_mask))


def _collect_slide_tiles(raw_patch_dir: Path, slide_id: str) -> list[tuple[Path, int, int]]:
    pattern = re.compile(rf"^{re.escape(slide_id)}_x(\d+)_y(\d+)\.png$")
    tiles = []
    for tile_path in raw_patch_dir.glob(f"{slide_id}_x*_y*.png"):
        match = pattern.match(tile_path.name)
        if match is None:
            continue
        tiles.append((tile_path, int(match.group(1)), int(match.group(2))))
    return tiles


def _route_tiles_by_tumor(
    *,
    slide_path: Path,
    slide_id: str,
    xml_path: Path,
    raw_patch_dir: Path,
    tumor_dir: Path,
    normal_dir: Path,
    patch_size: int,
    patch_level: int,
    tumor_threshold: float,
) -> tuple[int, int]:
    polygons = _load_tumor_polygons(xml_path)
    tiles = _collect_slide_tiles(raw_patch_dir, slide_id)

    if len(tiles) == 0:
        return 0, 0

    slide = openslide.OpenSlide(str(slide_path))
    try:
        safe_patch_level = max(0, min(int(patch_level), slide.level_count - 1))
        downsample_x = max(1, int(float(slide.level_downsamples[safe_patch_level])))
        downsample_y = downsample_x

        level0_w, level0_h = slide.level_dimensions[0]
        level_w, level_h = slide.level_dimensions[safe_patch_level]
        if level_w > 0 and level_h > 0:
            downsample_x = max(1, int(level0_w / float(level_w)))
            downsample_y = max(1, int(level0_h / float(level_h)))
    finally:
        slide.close()

    window_width = max(1, int(patch_size * downsample_x))
    window_height = max(1, int(patch_size * downsample_y))

    tumor_count = 0
    normal_count = 0
    for tile_path, x, y in tiles:
        tumor_ratio = _tumor_ratio_for_patch(x, y, window_width, window_height, polygons)
        target_dir = tumor_dir if tumor_ratio >= tumor_threshold else normal_dir
        target_path = target_dir / tile_path.name

        if target_path.exists():
            tile_path.unlink(missing_ok=True)
            continue

        shutil.move(str(tile_path), str(target_path))
        if target_dir == tumor_dir:
            tumor_count += 1
        else:
            normal_count += 1

    return tumor_count, normal_count


def _route_tiles_all_normal(
    *,
    slide_id: str,
    raw_patch_dir: Path,
    normal_dir: Path,
) -> tuple[int, int]:
    tiles = _collect_slide_tiles(raw_patch_dir, slide_id)
    if len(tiles) == 0:
        return 0, 0

    normal_count = 0
    for tile_path, _, _ in tiles:
        target_path = normal_dir / tile_path.name
        if target_path.exists():
            tile_path.unlink(missing_ok=True)
            continue

        shutil.move(str(tile_path), str(target_path))
        normal_count += 1

    return 0, normal_count


def _route_tiles_all_tumor(
    *,
    slide_id: str,
    raw_patch_dir: Path,
    tumor_dir: Path,
) -> tuple[int, int]:
    tiles = _collect_slide_tiles(raw_patch_dir, slide_id)
    if len(tiles) == 0:
        return 0, 0

    tumor_count = 0
    for tile_path, _, _ in tiles:
        target_path = tumor_dir / tile_path.name
        if target_path.exists():
            tile_path.unlink(missing_ok=True)
            continue

        shutil.move(str(tile_path), str(target_path))
        tumor_count += 1

    return tumor_count, 0


def main() -> None:
    """Extract and label CAMELYON tumor/normal tiles for all discovered slides."""
    args = build_parser().parse_args()

    if args.tumor_threshold < 0.0 or args.tumor_threshold > 1.0:
        raise ValueError("--tumor_threshold muss zwischen 0 und 1 liegen.")

    paths = get_project_paths(base_dir=args.base_dir)
    if args.output_dir:
        paths["RESULTS_DIR"] = args.output_dir
        paths["LOGS_DIR"] = args.output_dir / "logs"

    source_path = args.source if args.source else paths["RAW_DIR"] / "camelyon17" / "images"
    annotation_root = args.annotations if args.annotations else paths["RAW_DIR"] / "camelyon17" / "annotations"
    labels_csv = _resolve_labels_csv_path(args.labels_csv, paths)
    labels_by_key = _load_ground_truth_labels(labels_csv)

    preset = CAMELYON_PRESET
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    labeled_root = paths["PROCESSED_DIR"] / "camelyon"
    tumor_dir = labeled_root / "tumor"
    normal_dir = labeled_root / "normal"
    # Own subfolder under LOGS_DIR (not flat) -- see create_splits.py's identical comment; also
    # keeps the pipeline-stats CSV below grouped with its run's log.
    log_dir: Path = paths["LOGS_DIR"] / "tile_extraction_camelyon"
    preprocess_save_dir = paths["RESULTS_DIR"] / "camelyon_preprocessing"
    mask_save_dir = preprocess_save_dir / "masks"

    for d in (tumor_dir, normal_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_file = setup_logging(log_dir, operation="tile_extraction_camelyon", force_timestamp=True)
    quiet_external_loggers()
    logger = logging.getLogger(__name__)

    logger.info("Project paths:")
    for k, v in paths.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"Source:          {source_path}")
    logger.info(f"Annotations:     {annotation_root}")
    logger.info(f"Labels CSV:      {labels_csv}")
    logger.info(f"Tiles output:    {labeled_root}")
    logger.info(f"Log directory:   {log_dir}")
    logger.info(f"Tumor threshold: {args.tumor_threshold}")

    recursive = False if args.non_recursive else bool(preset["recursive"])
    slides = _discover_slides(str(source_path), recursive=recursive)
    annotations_by_stem = _discover_annotations(str(annotation_root), recursive=recursive)
    slides_with_annotation = sum(1 for slide in slides if slide.stem in annotations_by_stem)
    slides_without_annotation = len(slides) - slides_with_annotation
    logger.info(
        f"Slides gefunden: {len(slides)}, davon mit Annotation: {slides_with_annotation}, "
        f"ohne: {slides_without_annotation}"
    )

    if len(slides) == 0:
        logger.warning("Keine unterstützten Slides gefunden. Abbruch.")
        return

    summary_path = labeled_root / "tile_extraction_camelyon_stats.csv"
    _SUMMARY_FIELDS = [
        "slide_id",
        "status",
        "annotation_used",
        "ground_truth_label",
        "label_class",
        "routing_strategy",
        "overall_label",
        "seg_level",
        "seg_time_s",
        "patch_time_s",
        "patches_saved",
        "patches_skipped_filter",
        "tumor_tiles",
        "normal_tiles",
        "mask_path",
        "xml_path",
        "patch_dir",
        "processed_at",
        "error",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as _f:
        csv.DictWriter(_f, fieldnames=_SUMMARY_FIELDS).writeheader()
    logger.info(f"Summary CSV: {summary_path}")

    _pipeline_log = log_dir / f"tile_extraction_pipeline_{run_timestamp}.csv"

    pipeline = TilePreprocessing()
    total_tumor = 0
    total_normal = 0

    with tempfile.TemporaryDirectory(prefix="camelyon_tiles_") as raw_patch_tmp:
        raw_patch_dir = Path(raw_patch_tmp)
        logger.info(f"Temporäres Patch-Staging: {raw_patch_dir}")

        for idx, slide_path in enumerate(slides, start=1):
            slide_id = slide_path.stem
            xml_path = annotations_by_stem.get(slide_id)
            has_annotation = xml_path is not None
            label_class = _get_ground_truth_stage(slide_path, labels_by_key)
            ground_truth_label = _binary_ground_truth_label(label_class)
            logger.info(f"[{idx}/{len(slides)}] {slide_path.name}")

            has_tumor = any(tumor_dir.glob(f"{slide_id}_x*_y*.png"))
            has_normal = any(normal_dir.glob(f"{slide_id}_x*_y*.png"))
            if args.no_auto_skip and (has_tumor or has_normal):
                logger.info(f"{slide_id} bereits gelabelt, übersprungen")
                row = {
                    "slide_id": slide_id,
                    "status": "skipped_existing",
                    "annotation_used": "yes" if has_annotation else "no",
                    "ground_truth_label": ground_truth_label,
                    "label_class": label_class,
                    "routing_strategy": "skipped_existing",
                    "overall_label": "tumor" if has_tumor else "normal",
                    "tumor_tiles": sum(1 for _ in tumor_dir.glob(f"{slide_id}_x*_y*.png")),
                    "normal_tiles": sum(1 for _ in normal_dir.glob(f"{slide_id}_x*_y*.png")),
                    "xml_path": str(xml_path) if has_annotation else "",
                    "patch_dir": str(labeled_root),
                    "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                with open(summary_path, "a", newline="", encoding="utf-8") as _f:
                    csv.DictWriter(_f, fieldnames=_SUMMARY_FIELDS, extrasaction="ignore").writerow(row)
                continue

            config = PreprocessConfig(
                source=str(slide_path),
                save_dir=str(preprocess_save_dir),
                patch_save_dir=str(raw_patch_dir),
                mask_save_dir=str(mask_save_dir),
                patch_size=int(args.patch_size if args.patch_size is not None else preset["patch_size"]),
                step_size=int(args.step_size if args.step_size is not None else preset["step_size"]),
                patch_level=int(args.patch_level if args.patch_level is not None else preset["patch_level"]),
                seg=bool(preset["seg"]),
                patch=bool(preset["patch"]),
                save_mask=False if args.no_masks else bool(preset["save_mask"]),
                auto_skip=False,
                recursive=False,
                seg_params=dict(preset["seg_params"]),
                filter_params=dict(preset["filter_params"]),
                patch_params=dict(preset["patch_params"]),
            )

            pipeline.run(config)

            pipeline_row: dict = {}
            if _pipeline_log.exists():
                with open(_pipeline_log, newline="", encoding="utf-8") as _f:
                    _rows = list(csv.DictReader(_f))
                    if _rows:
                        pipeline_row = _rows[-1]

            if has_annotation:
                routing_strategy = "annotation_polygon_routing"
                tumor_count, normal_count = _route_tiles_by_tumor(
                    slide_path=slide_path,
                    slide_id=slide_id,
                    xml_path=xml_path,
                    raw_patch_dir=raw_patch_dir,
                    tumor_dir=tumor_dir,
                    normal_dir=normal_dir,
                    patch_size=config.patch_size,
                    patch_level=config.patch_level,
                    tumor_threshold=float(args.tumor_threshold),
                )
            else:
                is_tumor = _is_tumor_stage(label_class)
                if is_tumor is True:
                    routing_strategy = "no_annotation_gt_tumor_all_tumor"
                    tumor_count, normal_count = _route_tiles_all_tumor(
                        slide_id=slide_id,
                        raw_patch_dir=raw_patch_dir,
                        tumor_dir=tumor_dir,
                    )
                else:
                    routing_strategy = "no_annotation_gt_normal_all_normal"
                    if is_tumor is None:
                        routing_strategy = "no_annotation_gt_missing_fallback_all_normal"
                        logger.warning(
                            "%s: Kein gültiges Ground-Truth-Label gefunden, route alle Tiles als normal.",
                            slide_id,
                        )
                    tumor_count, normal_count = _route_tiles_all_normal(
                        slide_id=slide_id,
                        raw_patch_dir=raw_patch_dir,
                        normal_dir=normal_dir,
                    )

            total_tumor += tumor_count
            total_normal += normal_count
            overall_label = "tumor" if tumor_count > 0 else "normal"
            logger.info(
                f"{slide_id}: tumor_tiles={tumor_count}, normal_tiles={normal_count}, "
                f"overall_label={overall_label}, "
                f"seg_time={pipeline_row.get('seg_time_s', '?')}s, "
                f"patch_time={pipeline_row.get('patch_time_s', '?')}s"
            )

            row = {
                "slide_id": pipeline_row.get("slide_id", slide_id),
                "status": pipeline_row.get("status", "processed"),
                "annotation_used": "yes" if has_annotation else "no",
                "ground_truth_label": ground_truth_label,
                "label_class": label_class,
                "routing_strategy": routing_strategy,
                "overall_label": overall_label,
                "seg_level": pipeline_row.get("seg_level", ""),
                "seg_time_s": pipeline_row.get("seg_time_s", ""),
                "patch_time_s": pipeline_row.get("patch_time_s", ""),
                "patches_saved": pipeline_row.get("patches_saved", ""),
                "patches_skipped_filter": pipeline_row.get("patches_skipped_filter", ""),
                "tumor_tiles": tumor_count,
                "normal_tiles": normal_count,
                "mask_path": pipeline_row.get("mask_path", ""),
                "xml_path": str(xml_path) if has_annotation else "",
                "patch_dir": str(labeled_root),
                "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "error": pipeline_row.get("error", ""),
            }
            with open(summary_path, "a", newline="", encoding="utf-8") as _f:
                csv.DictWriter(_f, fieldnames=_SUMMARY_FIELDS, extrasaction="ignore").writerow(row)

    logger.info("Fertig.")
    logger.info(f"Slides verarbeitet: {len(slides)}")
    logger.info(f"Gesamt Tumor-Tiles: {total_tumor}")
    logger.info(f"Gesamt Normal-Tiles: {total_normal}")
    logger.info(f"Ausgabeordner: {labeled_root}")
    logger.info(f"Summary CSV: {summary_path}")
    logger.info(f"Log-Datei: {log_file}")


if __name__ == "__main__":
    main()
