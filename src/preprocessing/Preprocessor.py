from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import csv
import logging
import os
import time

import cv2
import numpy as np
import openslide

import sys

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from src.preprocessing.WholeSlideImage import WholeSlideImage
except ImportError:
    print("Error Importing WholeSlideImage class")


SUPPORTED_EXTENSIONS = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu")


DEFAULT_SEG_PARAMS = {
    "seg_level": -1,
    "sthresh": 8,
    "sthresh_up": 255,
    "mthresh": 7,
    "close": 4,
    "use_otsu": False,
    "keep_ids": "none",
    "exclude_ids": "none",
}

DEFAULT_FILTER_PARAMS = {"a_t": 100, "a_h": 16, "max_n_holes": 8}

DEFAULT_PATCH_PARAMS = {
    "use_padding": True,
    "contour_fn": "four_pt",
    "filter_black_white": True,
    "white_thresh": 220,
    "white_pct": 0.20,
    "black_thresh": 20,
    "black_pct": 0.05,
}


@dataclass
class PreprocessConfig:
    source: str
    save_dir: str
    patch_size: int = 256
    step_size: int = 256
    patch_level: int = 0
    seg: bool = False
    patch: bool = False
    save_mask: bool = True
    auto_skip: bool = True
    recursive: bool = False
    patch_save_dir: str | None = None
    mask_save_dir: str | None = None
    seg_params: dict = field(
        default_factory=lambda: dict(DEFAULT_SEG_PARAMS)
    )
    filter_params: dict = field(default_factory=lambda: dict(DEFAULT_FILTER_PARAMS))
    patch_params: dict = field(
        default_factory=lambda: dict(DEFAULT_PATCH_PARAMS)
    )


class PreprocessPipeline(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the pipeline's identifying name."""
        raise NotImplementedError

    @abstractmethod
    def run(self, config: PreprocessConfig):
        """Run the pipeline for the given config."""
        raise NotImplementedError


def _parse_id_list(raw_value) -> list[int]:
    as_str = str(raw_value).strip()
    if as_str in {"", "none", "None"}:
        return []
    return [int(item.strip()) for item in as_str.split(",") if item.strip()]


def _discover_slides(source: str, recursive: bool = False) -> list[str]:
    source_path = Path(source)
    if source_path.is_file():
        return [str(source_path)] if source_path.suffix.lower() in SUPPORTED_EXTENSIONS else []

    slides = []
    if recursive:
        iterator = source_path.rglob("*")
    else:
        iterator = source_path.iterdir()

    for entry in sorted(iterator):
        if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS:
            slides.append(str(entry))
    return slides


def _resolve_level(wsi_object: WholeSlideImage, preferred_level: int) -> int:
    if preferred_level >= 0:
        return min(preferred_level, len(wsi_object.level_dim) - 1)
    return min(2, len(wsi_object.level_dim) - 1)


def _build_tissue_mask(wsi_object: WholeSlideImage, seg_level: int) -> np.ndarray:
    width, height = wsi_object.level_dim[seg_level]
    mask = np.zeros((height, width), dtype=np.uint8)
    if not wsi_object.contours_tissue:
        return mask

    scale = np.array(wsi_object.level_downsamples[seg_level], dtype=np.float32)
    contours_scaled = [np.round(cont / scale).astype(np.int32) for cont in wsi_object.contours_tissue]
    holes_scaled = [
        [np.round(hole / scale).astype(np.int32) for hole in contour_holes]
        for contour_holes in (wsi_object.holes_tissue or [])
    ]

    cv2.drawContours(mask, contours_scaled, contourIdx=-1, color=255, thickness=-1)
    for contour_holes in holes_scaled:
        if contour_holes:
            cv2.drawContours(mask, contour_holes, contourIdx=-1, color=0, thickness=-1)
    return mask


def segment(wsi_object: WholeSlideImage, seg_params: dict, filter_params: dict):
    """Segment tissue on a slide and return the updated WSI object, seg level, and elapsed time."""
    start_time = time.time()

    current_seg_level = _resolve_level(wsi_object, int(seg_params.get("seg_level", -1)))
    current_keep_ids = _parse_id_list(seg_params.get("keep_ids", "none"))
    current_exclude_ids = _parse_id_list(seg_params.get("exclude_ids", "none"))

    wsi_object.segment_tissue(
        seg_level=current_seg_level,
        sthresh=int(seg_params.get("sthresh", 8)),
        sthresh_up=int(seg_params.get("sthresh_up", 255)),
        mthresh=int(seg_params.get("mthresh", 7)),
        close=int(seg_params.get("close", 4)),
        use_otsu=bool(seg_params.get("use_otsu", False)),
        filter_params=filter_params,
        ref_patch_size=int(seg_params.get("ref_patch_size", 512)),
        keep_ids=current_keep_ids,
        exclude_ids=current_exclude_ids,
    )

    seg_time_elapsed = time.time() - start_time
    return wsi_object, current_seg_level, seg_time_elapsed


def _collect_patch_coords(
    wsi_object: WholeSlideImage,
    patch_level: int,
    patch_size: int,
    step_size: int,
    contour_fn: str,
    use_padding: bool,
) -> np.ndarray:
    if not wsi_object.contours_tissue:
        return np.empty((0, 2), dtype=np.int64)

    all_coords: list[np.ndarray] = []
    for idx, contour in enumerate(wsi_object.contours_tissue):
        contour_holes = wsi_object.holes_tissue[idx] if wsi_object.holes_tissue else []
        asset_dict, _ = wsi_object.process_contour(
            cont=contour,
            contour_holes=contour_holes,
            patch_level=patch_level,
            save_path="",
            patch_size=patch_size,
            step_size=step_size,
            contour_fn=contour_fn,
            use_padding=use_padding,
        )
        coords = asset_dict.get("coords")
        if coords is not None and len(coords) > 0:
            all_coords.append(coords)

    if not all_coords:
        return np.empty((0, 2), dtype=np.int64)

    return np.unique(np.vstack(all_coords), axis=0)


def _is_black_patch(patch: np.ndarray, rgb_thresh: int, percentage: float) -> bool:
    num_pixels = patch.shape[0] * patch.shape[1]
    dark_pixels = np.all(patch < rgb_thresh, axis=2).sum()
    return dark_pixels > num_pixels * percentage


def _is_white_patch(patch: np.ndarray, rgb_thresh: int, percentage: float) -> bool:
    num_pixels = patch.shape[0] * patch.shape[1]
    bright_pixels = np.all(patch > rgb_thresh, axis=2).sum()
    return bright_pixels > num_pixels * percentage


def _extract_and_filter_patch(
    args: tuple,
) -> tuple | None:
    x, y, slide_path, patch_level, patch_size, filter_black_white, white_thresh, white_pct, black_thresh, black_pct, slide_id = args
    
    try:
        wsi = openslide.open_slide(str(slide_path))
        patch = wsi.read_region((int(x), int(y)), patch_level, (patch_size, patch_size)).convert("RGB")
        wsi.close()
        
        patch_np = np.array(patch)
        
        if filter_black_white:
            if _is_black_patch(patch_np, black_thresh, black_pct) or _is_white_patch(patch_np, white_thresh, white_pct):
                return None
        
        return (patch, x, y, slide_id)
    except Exception as e:
        print(f"Error extracting patch at ({x}, {y}): {e}")
        return None


def _disable_worker_logging() -> None:
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def patching(
    wsi_object: WholeSlideImage,
    patch_save_dir: str,
    slide_id: str,
    patch_level: int,
    patch_size: int,
    step_size: int,
    patch_params: dict,
):
    """Extract, filter, and save patches for a slide, returning save stats."""
    start_time = time.time()

    coords = _collect_patch_coords(
        wsi_object=wsi_object,
        patch_level=patch_level,
        patch_size=patch_size,
        step_size=step_size,
        contour_fn=str(patch_params.get("contour_fn", "four_pt")),
        use_padding=bool(patch_params.get("use_padding", True)),
    )

    patch_dir = Path(patch_save_dir)
    patch_dir.mkdir(parents=True, exist_ok=True)

    filter_black_white = bool(patch_params.get("filter_black_white", True))
    white_thresh = int(patch_params.get("white_thresh", 220))
    white_pct = float(patch_params.get("white_pct", 0.20))
    black_thresh = int(patch_params.get("black_thresh", 20))
    black_pct = float(patch_params.get("black_pct", 0.05))

    num_workers = max(1, min(4, __import__('multiprocessing').cpu_count() - 1))
    pool = __import__('multiprocessing').Pool(num_workers, initializer=_disable_worker_logging)
    
    task_args = [
        (x, y, wsi_object.path, patch_level, patch_size, 
         filter_black_white, white_thresh, white_pct, black_thresh, black_pct, slide_id)
        for x, y in coords
    ]
    
    results = pool.map(_extract_and_filter_patch, task_args)
    pool.close()
    pool.join()

    saved = 0
    skipped = 0
    for result in results:
        if result is not None:
            patch, x, y, sid = result
            patch.save(patch_dir / f"{sid}_x{int(x)}_y{int(y)}.png")
            saved += 1
        else:
            skipped += 1

    patch_time_elapsed = time.time() - start_time
    return str(patch_dir), saved, skipped, patch_time_elapsed


class TilePreprocessing(PreprocessPipeline):
    name = "tile_preprocessing"

    def _resolve_dirs(self, config: PreprocessConfig) -> tuple[str, str]:
        patch_save_dir = config.patch_save_dir or config.save_dir
        mask_save_dir = config.mask_save_dir or os.path.join(config.save_dir, "masks")
        return patch_save_dir, mask_save_dir

    def run(self, config: PreprocessConfig):
        """Run segmentation, masking, and patching for all slides found in the source."""
        patch_save_dir, mask_save_dir = self._resolve_dirs(config)

        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(patch_save_dir, exist_ok=True)
        os.makedirs(mask_save_dir, exist_ok=True)

        slides = _discover_slides(config.source, recursive=config.recursive)
        total = len(slides)
        if total == 0:
            print(f"Keine unterstützten WSI-Dateien in {config.source} gefunden.")
            return -1.0, -1.0

        seg_times = 0.0
        patch_times = 0.0
        processed = 0
        run_rows = []

        for idx, full_path in enumerate(slides):
            slide = os.path.basename(full_path)
            slide_id, _ = os.path.splitext(slide)

            print(f"\nprogress: {idx / total:.2f}, {idx + 1}/{total}")
            print(f"processing {slide}")

            existing_patch_glob = f"{slide_id}_x*_y*.png"
            has_existing_patches = any(Path(patch_save_dir).glob(existing_patch_glob))
            if config.auto_skip and config.patch and has_existing_patches:
                print(f"{slide_id} already exists in destination location, skipped")
                run_rows.append(
                    {
                        "slide_id": slide_id,
                        "status": "skipped_existing",
                        "seg_level": "",
                        "seg_time_s": "",
                        "patch_time_s": "",
                        "patches_saved": "",
                        "patches_skipped_filter": "",
                        "mask_path": "",
                        "patch_dir": str(Path(patch_save_dir)),
                        "error": "",
                    }
                )
                continue

            wsi_object = WholeSlideImage(Path(full_path))
            mask_path = ""
            patch_dir = ""
            patches_saved = 0
            patches_skipped = 0
            try:
                need_segmentation = config.seg or config.save_mask or config.patch
                current_seg_level = _resolve_level(wsi_object, int(config.seg_params.get("seg_level", -1)))
                seg_time_elapsed = -1.0

                if need_segmentation:
                    wsi_object, current_seg_level, seg_time_elapsed = segment(
                        wsi_object=wsi_object,
                        seg_params=config.seg_params,
                        filter_params=config.filter_params,
                    )

                if config.save_mask:
                    mask = _build_tissue_mask(wsi_object, current_seg_level)
                    mask_path = os.path.join(mask_save_dir, f"{slide_id}.png")
                    cv2.imwrite(mask_path, mask)

                patch_time_elapsed = -1.0
                if config.patch:
                    patch_dir, patches_saved, patches_skipped, patch_time_elapsed = patching(
                        wsi_object=wsi_object,
                        patch_save_dir=patch_save_dir,
                        slide_id=slide_id,
                        patch_level=config.patch_level,
                        patch_size=config.patch_size,
                        step_size=config.step_size,
                        patch_params=config.patch_params,
                    )
                    print(f"Saved {patches_saved} patches to {patch_dir} (filtered: {patches_skipped})")

                print(f"segmentation took {seg_time_elapsed} seconds")
                print(f"patching took {patch_time_elapsed} seconds")

                if seg_time_elapsed >= 0:
                    seg_times += seg_time_elapsed
                if patch_time_elapsed >= 0:
                    patch_times += patch_time_elapsed
                processed += 1
                run_rows.append(
                    {
                        "slide_id": slide_id,
                        "status": "processed",
                        "seg_level": current_seg_level,
                        "seg_time_s": seg_time_elapsed,
                        "patch_time_s": patch_time_elapsed,
                        "patches_saved": patches_saved,
                        "patches_skipped_filter": patches_skipped,
                        "mask_path": mask_path,
                        "patch_dir": patch_dir,
                        "error": "",
                    }
                )

            except Exception as exc:
                run_rows.append(
                    {
                        "slide_id": slide_id,
                        "status": "failed",
                        "seg_level": "",
                        "seg_time_s": "",
                        "patch_time_s": "",
                        "patches_saved": "",
                        "patches_skipped_filter": "",
                        "mask_path": mask_path,
                        "patch_dir": patch_dir,
                        "error": str(exc),
                    }
                )
                print(f"Fehler bei {slide_id}: {exc}")

            finally:
                wsi_object.close()

        divisor = max(processed, 1)
        seg_times /= divisor
        patch_times /= divisor

        log_path = os.path.join(config.save_dir, "process_log.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "slide_id",
                    "status",
                    "seg_level",
                    "seg_time_s",
                    "patch_time_s",
                    "patches_saved",
                    "patches_skipped_filter",
                    "mask_path",
                    "patch_dir",
                    "error",
                ],
            )
            writer.writeheader()
            writer.writerows(run_rows)

        print(f"average segmentation time in s per slide: {seg_times}")
        print(f"average patching time in s per slide: {patch_times}")
        print(f"process log written to: {log_path}")
        return seg_times, patch_times