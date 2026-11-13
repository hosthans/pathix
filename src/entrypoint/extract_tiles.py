import os
import json
import csv
import openslide
from datetime import datetime
import time
from PIL import Image
from shapely.geometry import Polygon, Point
import math
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import cv2
import tifffile
import zarr
from tqdm import tqdm
import argparse
import sys

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.paths.paths import get_project_paths, add_path_args
from src.utils.logging import setup_logging, quiet_external_loggers

@dataclass
class ExtractionConfig:
    tile_size: int = 256
    downsample: int = 10
    overlap: int = 64
    num_workers: int = 32
    batch_size: int = 512
    prefix_parent_dir_in_wsi_name: bool = False
    overwrite_existing: bool = False
    min_tissue_fraction: float = 0.02
    max_black_fraction: float = 0.95
    max_white_fraction: float = 0.98
    min_intensity_std: float = 5.0
    mask_mode: str = "standard"


@dataclass
class SlideExtractionStats:
    slide_name: str
    slide_width: int
    slide_height: int
    slide_pixels: int
    overview_downsample: float
    mask_width: int
    mask_height: int
    mask_pixels: int
    mask_tissue_pixels: int
    mask_tissue_fraction: float
    extraction_size_px: int
    step_size_px: int
    mask_mode: str = "standard"
    generated_tile_coords: int = 0
    in_bounds_tile_coords: int = 0
    mask_filtered_tile_coords: int = 0
    polygon_filtered_tile_coords: int = 0
    kept_tile_coords: int = 0
    saved_tile_files: int = 0
    estimated_covered_slide_pixels: int = 0
    estimated_covered_slide_fraction: float = 0.0
    estimated_covered_tissue_pixels: int = 0
    estimated_covered_tissue_fraction: float = 0.0
    preview_slide_path: str = ""
    preview_mask_path: str = ""

    def to_row(self) -> Dict[str, object]:
        """Convert this stats record into a flat dict suitable for a CSV row."""
        return {
            "slide_name": self.slide_name,
            "slide_width": self.slide_width,
            "slide_height": self.slide_height,
            "slide_pixels": self.slide_pixels,
            "overview_downsample": round(self.overview_downsample, 6),
            "mask_width": self.mask_width,
            "mask_height": self.mask_height,
            "mask_pixels": self.mask_pixels,
            "mask_tissue_pixels": self.mask_tissue_pixels,
            "mask_tissue_fraction": round(self.mask_tissue_fraction, 6),
            "extraction_size_px": self.extraction_size_px,
            "step_size_px": self.step_size_px,
            "mask_mode": self.mask_mode,
            "generated_tile_coords": self.generated_tile_coords,
            "in_bounds_tile_coords": self.in_bounds_tile_coords,
            "mask_filtered_tile_coords": self.mask_filtered_tile_coords,
            "polygon_filtered_tile_coords": self.polygon_filtered_tile_coords,
            "kept_tile_coords": self.kept_tile_coords,
            "saved_tile_files": self.saved_tile_files,
            "estimated_covered_slide_pixels": self.estimated_covered_slide_pixels,
            "estimated_covered_slide_fraction": round(self.estimated_covered_slide_fraction, 6),
            "estimated_covered_tissue_pixels": self.estimated_covered_tissue_pixels,
            "estimated_covered_tissue_fraction": round(self.estimated_covered_tissue_fraction, 6),
            "preview_slide_path": self.preview_slide_path,
            "preview_mask_path": self.preview_mask_path,
        }


def normalize_to_uint8(image_array: np.ndarray) -> np.ndarray:
    """Convert an image array of arbitrary dtype to uint8."""
    if image_array.dtype == np.uint8:
        return image_array

    if np.issubdtype(image_array.dtype, np.floating):
        max_value = float(np.nanmax(image_array)) if image_array.size else 0.0
        if max_value <= 1.0:
            image_array = image_array * 255.0
        return np.clip(image_array, 0, 255).astype(np.uint8)

    if image_array.dtype == np.uint16:
        return (image_array / 257).astype(np.uint8)

    max_dtype_value = np.iinfo(image_array.dtype).max if np.issubdtype(image_array.dtype, np.integer) else 255
    if max_dtype_value <= 255:
        return image_array.astype(np.uint8)

    scaled = image_array.astype(np.float32) * (255.0 / max_dtype_value)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def to_rgb_uint8(image_array: np.ndarray) -> np.ndarray:
    """Convert an image array to a 3-channel RGB uint8 array."""
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.ndim == 3 and image_array.shape[-1] == 1:
        image_array = np.repeat(image_array, 3, axis=-1)
    elif image_array.ndim == 3 and image_array.shape[-1] > 3:
        image_array = image_array[..., :3]

    return normalize_to_uint8(image_array)


def has_sufficient_tissue(
    tile_array: np.ndarray,
    min_tissue_fraction: float,
    max_black_fraction: float,
    max_white_fraction: float,
    min_intensity_std: float,
) -> bool:
    """Check whether a tile contains enough non-background tissue to keep."""
    if tile_array.size == 0:
        return False

    rgb_tile = to_rgb_uint8(tile_array)
    gray_tile = rgb_tile.mean(axis=2)

    black_fraction = float(np.mean(gray_tile <= 8))
    if black_fraction > max_black_fraction:
        return False

    white_fraction = float(np.mean(gray_tile >= 245))
    if white_fraction > max_white_fraction:
        return False

    non_background_fraction = float(np.mean((gray_tile > 8) & (gray_tile < 245)))
    if non_background_fraction < min_tissue_fraction:
        return False

    if float(np.std(gray_tile)) < min_intensity_std:
        return False

    return True


class SlideReader:
    def __init__(self, slide_path: Path):
        """Initialize the reader for the given slide path."""
        self.slide_path = Path(slide_path)
        self.backend = None
        self._openslide = None
        self._tiff = None
        self._tiff_levels = []
        self._tiff_level_shapes = []
        self._tiff_level_downsamples = []
        self.dimensions = (0, 0)

    def __enter__(self):
        """Open the underlying slide backend and return self."""
        suffix = self.slide_path.suffix.lower()
        if suffix in {'.tif', '.tiff'}:
            self._open_tiff()
        else:
            self._open_openslide()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the underlying slide backend."""
        self.close()

    def _open_openslide(self):
        self.backend = 'openslide'
        self._openslide = openslide.OpenSlide(str(self.slide_path))
        self.dimensions = self._openslide.dimensions

    def _open_tiff(self):
        self.backend = 'tifffile'
        self._tiff = tifffile.TiffFile(str(self.slide_path))
        series = self._tiff.series[0]
        self._tiff_levels = list(series.levels)

        base_shape = self._tiff_levels[0].shape
        base_height, base_width = int(base_shape[0]), int(base_shape[1])
        self.dimensions = (base_width, base_height)

        for level in self._tiff_levels:
            level_shape = level.shape
            level_height, level_width = int(level_shape[0]), int(level_shape[1])
            self._tiff_level_shapes.append((level_width, level_height))
            self._tiff_level_downsamples.append(base_width / level_width)

    def close(self):
        """Close any open slide handles."""
        if self._openslide is not None:
            self._openslide.close()
            self._openslide = None
        if self._tiff is not None:
            self._tiff.close()
            self._tiff = None

    def _select_level_index(self, requested_downsample: int, available_downsamples: List[float]) -> int:
        eligible_levels = [
            index for index, level_downsample in enumerate(available_downsamples)
            if level_downsample <= requested_downsample
        ]
        if eligible_levels:
            return eligible_levels[-1]

        closest_level = min(
            range(len(available_downsamples)),
            key=lambda index: abs(available_downsamples[index] - requested_downsample)
        )
        return closest_level

    def _read_region_openslide(self, x: int, y: int, downsample: int, tile_size: int) -> np.ndarray:
        level_index = self._select_level_index(downsample, list(self._openslide.level_downsamples))
        level_downsample = float(self._openslide.level_downsamples[level_index])
        region_size = max(1, int(round(tile_size * downsample / level_downsample)))

        region = self._openslide.read_region((x, y), level_index, (region_size, region_size))
        try:
            region = region.convert('RGB')
            if region.size != (tile_size, tile_size):
                region = region.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
            return np.asarray(region, dtype=np.uint8, order='C')
        finally:
            if hasattr(region, 'close'):
                region.close()

    def _read_region_tiff(self, x: int, y: int, downsample: int, tile_size: int) -> np.ndarray:
        level_index = self._select_level_index(downsample, self._tiff_level_downsamples)
        level_downsample = self._tiff_level_downsamples[level_index]
        region_size = max(1, int(round(tile_size * downsample / level_downsample)))

        level_width, level_height = self._tiff_level_shapes[level_index]
        level_x = int(round(x / level_downsample))
        level_y = int(round(y / level_downsample))
        level_x = max(0, min(level_x, max(0, level_width - region_size)))
        level_y = max(0, min(level_y, max(0, level_height - region_size)))

        store = self._tiff_levels[level_index].aszarr()
        try:
            array = zarr.open(store, mode='r')
            region_array = np.asarray(array[level_y:level_y + region_size, level_x:level_x + region_size])
        finally:
            if hasattr(store, 'close'):
                store.close()

        region_array = to_rgb_uint8(region_array)
        tile = Image.fromarray(region_array, mode='RGB')
        try:
            if tile.size != (tile_size, tile_size):
                tile = tile.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
            return np.asarray(tile, dtype=np.uint8, order='C')
        finally:
            if hasattr(tile, 'close'):
                tile.close()

    def read_tile(self, x: int, y: int, downsample: int, tile_size: int) -> np.ndarray:
        """Read a single tile at the given coordinates and downsample."""
        if self.backend == 'tifffile':
            return self._read_region_tiff(x, y, downsample, tile_size)
        return self._read_region_openslide(x, y, downsample, tile_size)

    def read_overview(self) -> Tuple[np.ndarray, float]:
        """Read the lowest-resolution overview image and its downsample factor."""
        if self.backend == 'tifffile':
            level_index = len(self._tiff_levels) - 1
            overview = self._tiff_levels[level_index].asarray()
            return to_rgb_uint8(overview), float(self._tiff_level_downsamples[level_index])

        level_index = self._openslide.level_count - 1
        level_dimensions = self._openslide.level_dimensions[level_index]
        overview = self._openslide.read_region((0, 0), level_index, level_dimensions)
        try:
            overview = overview.convert('RGB')
            return np.asarray(overview, dtype=np.uint8, order='C'), float(self._openslide.level_downsamples[level_index])
        finally:
            if hasattr(overview, 'close'):
                overview.close()


def process_tile_batch_no_annotation(params: Tuple) -> List[Optional[Tuple]]:
    """Process a batch of tiles from a single WSI without polygon filtering."""
    (
        wsi_path,
        tile_coords,
        downsample,
        tile_size,
        min_tissue_fraction,
        max_black_fraction,
        max_white_fraction,
        min_intensity_std,
    ) = params
    results = []
    wsi = None

    if not Path(wsi_path).exists():
        logging.error(f"WSI file does not exist: {wsi_path}")
        return []

    try:
        with SlideReader(Path(wsi_path)) as wsi:
            for orig_x, orig_y in tile_coords:
                try:
                    tile_array = wsi.read_tile(orig_x, orig_y, downsample, tile_size)
                    if has_sufficient_tissue(
                        tile_array,
                        min_tissue_fraction,
                        max_black_fraction,
                        max_white_fraction,
                        min_intensity_std,
                    ):
                        results.append((orig_x, orig_y, tile_array))
                except Exception as e:
                    logging.error(f"Error processing tile at ({orig_x}, {orig_y}): {str(e)}")
                    continue
    except Exception as e:
        logging.error(f"Error in process_tile_batch_no_annotation: {str(e)}")
        return []

    return results

def process_tile_batch(params: Tuple) -> List[Optional[Tuple]]:
    """Process a batch of tiles from a single WSI using read_region"""
    (
        wsi_path,
        tile_coords,
        polygon,
        downsample,
        tile_size,
        min_tissue_fraction,
        max_black_fraction,
        max_white_fraction,
        min_intensity_std,
    ) = params
    results = []
    wsi = None
    
    if not Path(wsi_path).exists():
        logging.error(f"WSI file does not exist: {wsi_path}")
        return []
    
    try:
        with SlideReader(Path(wsi_path)) as wsi:
            for orig_x, orig_y in tile_coords:
                try:
                    ds_x = orig_x // downsample
                    ds_y = orig_y // downsample
                    
                    tile_center = Point(ds_x + tile_size / 2, ds_y + tile_size / 2)
                    
                    if polygon.contains(tile_center):
                        tile_array = wsi.read_tile(orig_x, orig_y, downsample, tile_size)
                        if has_sufficient_tissue(
                            tile_array,
                            min_tissue_fraction,
                            max_black_fraction,
                            max_white_fraction,
                            min_intensity_std,
                        ):
                            results.append((orig_x, orig_y, tile_array))
                        
                except Exception as e:
                    logging.error(f"Error processing tile at ({orig_x}, {orig_y}): {str(e)}")
                    continue
                
    except Exception as e:
        logging.error(f"Error in process_tile_batch: {str(e)}")
        return []
            
    return results

class WSITileExtractor:
    def __init__(
        self,
        config: ExtractionConfig,
        paths: Dict[str, Path],
        stats_csv_path: Optional[Path] = None,
        preview_dir: Optional[Path] = None,
    ):
        """Initialize the extractor, its logger, and its stats/preview output paths."""
        self.config = config
        self.paths = paths
        self.stats_csv_path = stats_csv_path
        self.preview_dir = preview_dir
        self.setup_logging()
        logging.info(f"Using PIL version {Image.__version__}")
        logging.info(f"Configuration: tile_size={config.tile_size}, "
                    f"downsample={config.downsample}, "
                    f"overlap={config.overlap}, "
                    f"num_workers={config.num_workers}, "
                    f"batch_size={config.batch_size}")
        if self.stats_csv_path:
            self._ensure_stats_csv_exists()
        if self.preview_dir:
            self.preview_dir.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Attach a module-level logger to this instance."""
        self.logger = logging.getLogger(__name__)

    def _ensure_stats_csv_exists(self):
        if self.stats_csv_path is None or self.stats_csv_path.exists():
            return

        with open(self.stats_csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(SlideExtractionStats.__dataclass_fields__.keys()))
            writer.writeheader()

    def append_stats_row(self, stats: SlideExtractionStats) -> None:
        """Append a slide's stats as a row to the stats CSV."""
        if self.stats_csv_path is None:
            return

        with open(self.stats_csv_path, "a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(SlideExtractionStats.__dataclass_fields__.keys()))
            writer.writerow(stats.to_row())

    def compute_tile_coverage_stats(
        self,
        kept_coords: List[Tuple[int, int]],
        tissue_mask: np.ndarray,
        mask_shape: Tuple[int, int],
        overview_downsample: float,
        extraction_size: int,
        slide_pixels: int,
    ) -> Tuple[int, float, int, float]:
        """Estimate slide- and tissue-pixel coverage from the kept tile coordinates."""
        if not kept_coords:
            return 0, 0.0, 0, 0.0

        mask_height, mask_width = mask_shape
        coverage_mask = np.zeros((mask_height, mask_width), dtype=bool)
        mask_window_size = max(1, int(math.ceil(extraction_size / overview_downsample)))

        for x, y in kept_coords:
            mask_x = int(x / overview_downsample)
            mask_y = int(y / overview_downsample)
            mask_x_end = min(mask_width, mask_x + mask_window_size)
            mask_y_end = min(mask_height, mask_y + mask_window_size)
            if mask_x >= mask_width or mask_y >= mask_height or mask_x_end <= mask_x or mask_y_end <= mask_y:
                continue
            coverage_mask[mask_y:mask_y_end, mask_x:mask_x_end] = True

        coverage_fraction = float(np.mean(coverage_mask))
        covered_pixels = int(round(coverage_fraction * slide_pixels))
        covered_tissue_mask = coverage_mask & tissue_mask
        tissue_fraction = 0.0
        covered_tissue_pixels = 0
        tissue_pixel_count = int(np.count_nonzero(tissue_mask))
        if tissue_pixel_count > 0:
            tissue_fraction = float(np.count_nonzero(covered_tissue_mask) / tissue_pixel_count)
            tissue_pixels_estimate = int(round((tissue_pixel_count / tissue_mask.size) * slide_pixels))
            covered_tissue_pixels = int(round(tissue_fraction * tissue_pixels_estimate))

        return covered_pixels, coverage_fraction, covered_tissue_pixels, tissue_fraction

    def log_stats(self, stats: SlideExtractionStats) -> None:
        """Log a summary line (and preview paths) for a slide's extraction stats."""
        self.logger.info(
            f"Stats for {stats.slide_name}: slide={stats.slide_width}x{stats.slide_height} ({stats.slide_pixels} px), "
            f"mask={stats.mask_width}x{stats.mask_height} ({stats.mask_pixels} px), "
            f"mask_mode={stats.mask_mode}, "
            f"mask_tissue={stats.mask_tissue_pixels} px ({stats.mask_tissue_fraction * 100:.2f}%), "
            f"coords generated={stats.generated_tile_coords}, in_bounds={stats.in_bounds_tile_coords}, "
            f"mask_filtered={stats.mask_filtered_tile_coords}, polygon_filtered={stats.polygon_filtered_tile_coords}, "
            f"kept={stats.kept_tile_coords}, saved={stats.saved_tile_files}, "
            f"tile_coverage={stats.estimated_covered_slide_pixels} px ({stats.estimated_covered_slide_fraction * 100:.2f}% of slide), "
            f"tissue_coverage={stats.estimated_covered_tissue_pixels} px ({stats.estimated_covered_tissue_fraction * 100:.2f}% of tissue)"
        )
        if stats.preview_slide_path or stats.preview_mask_path:
            self.logger.info(
                f"Previews for {stats.slide_name}: slide={stats.preview_slide_path}, mask={stats.preview_mask_path}"
            )

    def save_slide_previews(
        self,
        wsi_path: Path,
        slide_name: str,
        tissue_mask: np.ndarray,
    ) -> Tuple[str, str]:
        """Save an overview preview and a tissue-mask overlay preview for a slide."""
        if self.preview_dir is None:
            return "", ""

        with SlideReader(wsi_path) as wsi:
            overview_rgb, _ = wsi.read_overview()

        slide_preview_path = self.preview_dir / f"{slide_name}_overview.png"
        mask_preview_path = self.preview_dir / f"{slide_name}_mask_overlay.png"

        Image.fromarray(overview_rgb, mode='RGB').save(slide_preview_path, "PNG")

        overlay_rgb = overview_rgb.copy()
        mask_bool = tissue_mask.astype(bool)
        if np.any(mask_bool):
            red_overlay = np.zeros_like(overlay_rgb)
            red_overlay[:, :, 0] = 255
            alpha = 0.35
            overlay_rgb[mask_bool] = (
                (1.0 - alpha) * overlay_rgb[mask_bool] + alpha * red_overlay[mask_bool]
            ).astype(np.uint8)

        Image.fromarray(overlay_rgb, mode='RGB').save(mask_preview_path, "PNG")
        return str(slide_preview_path), str(mask_preview_path)

    def build_tissue_mask(self, wsi_path: Path) -> Tuple[np.ndarray, float]:
        """Build a tissue mask from the slide's overview image."""
        with SlideReader(wsi_path) as wsi:
            overview_rgb, overview_downsample = wsi.read_overview()

        gray_overview = cv2.cvtColor(overview_rgb, cv2.COLOR_RGB2GRAY)
        hsv_overview = cv2.cvtColor(overview_rgb, cv2.COLOR_RGB2HSV)
        saturation = hsv_overview[:, :, 1]

        non_black_mask = gray_overview > 8
        tissue_mask = np.zeros_like(gray_overview, dtype=np.uint8)
        standard_mask = non_black_mask.astype(np.uint8)
        if np.any(non_black_mask):
            gray_non_black = gray_overview[non_black_mask].reshape(-1, 1)
            otsu_threshold, _ = cv2.threshold(gray_non_black, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            intensity_threshold = min(float(otsu_threshold), 235.0)
            tissue_mask = (
                non_black_mask & ((gray_overview < intensity_threshold) | (saturation > 20))
            ).astype(np.uint8)
            kernel = np.ones((3, 3), dtype=np.uint8)
            tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
            tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
            standard_mask = cv2.morphologyEx(standard_mask, cv2.MORPH_OPEN, kernel)
            standard_mask = cv2.morphologyEx(standard_mask, cv2.MORPH_CLOSE, kernel)

        if self.config.mask_mode == "standard":
            return standard_mask.astype(bool), overview_downsample

        return tissue_mask.astype(bool), overview_downsample

    def filter_tile_coords_with_tissue_mask(
        self,
        tile_coords: List[Tuple[int, int]],
        tissue_mask: np.ndarray,
        overview_downsample: float,
        extraction_size: int,
    ) -> List[Tuple[int, int]]:
        """Filter tile coordinates to those overlapping enough tissue in the mask."""
        if tissue_mask.size == 0:
            return []

        mask_height, mask_width = tissue_mask.shape
        min_mask_tissue_fraction = max(0.01, min(self.config.min_tissue_fraction, 0.10))
        filtered_coords = []

        for x, y in tile_coords:
            mask_x = int(x / overview_downsample)
            mask_y = int(y / overview_downsample)
            mask_window_size = max(1, int(math.ceil(extraction_size / overview_downsample)))

            mask_x_end = min(mask_width, mask_x + mask_window_size)
            mask_y_end = min(mask_height, mask_y + mask_window_size)
            if mask_x >= mask_width or mask_y >= mask_height or mask_x_end <= mask_x or mask_y_end <= mask_y:
                continue

            tissue_fraction = float(np.mean(tissue_mask[mask_y:mask_y_end, mask_x:mask_x_end]))
            if tissue_fraction >= min_mask_tissue_fraction:
                filtered_coords.append((x, y))

        return filtered_coords

    def get_effective_wsi_name(self, wsi_path: Path) -> str:
        """Build the filename-safe WSI identifier used for output tile names."""
        wsi_name = wsi_path.stem
        if self.config.prefix_parent_dir_in_wsi_name:
            return f"{wsi_path.parent.name}_{wsi_name}"
        return wsi_name

    @staticmethod
    def load_annotations(json_file: Path) -> Optional[List[dict]]:
        """Load a list of annotation dicts from a JSON file."""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Annotation data must be a list")
            return data
        except Exception as e:
            logging.error(f"Error loading {json_file}: {str(e)}")
            return None

    def is_processed(self, output_dir: Path, wsi_name: str, particle_id: str) -> bool:
        """Check if tiles for this WSI and particle_id combination have already been processed"""
        pattern = f"{wsi_name}_particle_{particle_id}_tile_*"
        existing_files = list(output_dir.glob(pattern))
        return len(existing_files) > 0

    def extract_tiles(self, wsi_path: Path, annotation_data: List[dict], output_dir: Path) -> SlideExtractionStats:
        """Extract and save tiles within each annotation polygon for a slide."""
        if not wsi_path.exists():
            self.logger.error(f"WSI file not found: {wsi_path}")
            raise FileNotFoundError(f"WSI file not found: {wsi_path}")
            
        try:
            wsi_name = self.get_effective_wsi_name(wsi_path)
            
            with SlideReader(wsi_path) as wsi:
                dimensions = wsi.dimensions
                self.logger.info(f"Processing {wsi_path.stem} from {wsi_path.parent}, dimensions: {dimensions}")

            slide_width, slide_height = dimensions
            slide_pixels = slide_width * slide_height
            extraction_size = self.config.tile_size * self.config.downsample
            max_x_start = slide_width - extraction_size
            max_y_start = slide_height - extraction_size
            tissue_mask, overview_downsample = self.build_tissue_mask(wsi_path)
            step_size = (self.config.tile_size - self.config.overlap) * self.config.downsample
            stats = SlideExtractionStats(
                slide_name=wsi_name,
                slide_width=slide_width,
                slide_height=slide_height,
                slide_pixels=slide_pixels,
                overview_downsample=overview_downsample,
                mask_width=int(tissue_mask.shape[1]),
                mask_height=int(tissue_mask.shape[0]),
                mask_pixels=int(tissue_mask.size),
                mask_tissue_pixels=int(np.count_nonzero(tissue_mask)),
                mask_tissue_fraction=float(np.mean(tissue_mask)) if tissue_mask.size else 0.0,
                extraction_size_px=extraction_size,
                step_size_px=step_size,
                mask_mode=self.config.mask_mode,
            )
            stats.preview_slide_path, stats.preview_mask_path = self.save_slide_previews(
                wsi_path,
                wsi_name,
                tissue_mask,
            )
            kept_coords: List[Tuple[int, int]] = []

            if max_x_start < 0 or max_y_start < 0:
                self.logger.warning(
                    f"Skipping {wsi_path.stem}: extraction window {extraction_size}x{extraction_size} "
                    f"is larger than slide dimensions {slide_width}x{slide_height}"
                )
                self.log_stats(stats)
                self.append_stats_row(stats)
                return stats

            for idx, annotation in enumerate(annotation_data):
                try:
                    particle_id = annotation['id']

                    if not self.config.overwrite_existing and self.is_processed(output_dir, wsi_name, particle_id):
                        self.logger.info(f"Skipping {wsi_name} particle {particle_id} - already processed")
                        continue

                    coords = annotation['geometry']['coordinates'][0]
                    
                    original_polygon = Polygon(coords)
                    
                    polygon = Polygon([
                        (x // self.config.downsample, y // self.config.downsample) 
                        for x, y in original_polygon.exterior.coords
                    ])

                    min_x, min_y, max_x, max_y = original_polygon.bounds

                    tile_coords = []
                    
                    current_x = int(min_x)
                    while current_x < max_x:
                        current_y = int(min_y)
                        while current_y < max_y:
                            tile_coords.append((current_x, current_y))
                            current_y += step_size
                        current_x += step_size
                    stats.generated_tile_coords += len(tile_coords)

                    valid_tile_coords = [
                        (x, y)
                        for x, y in tile_coords
                        if 0 <= x <= max_x_start and 0 <= y <= max_y_start
                    ]
                    stats.in_bounds_tile_coords += len(valid_tile_coords)
                    valid_tile_coords = self.filter_tile_coords_with_tissue_mask(
                        valid_tile_coords,
                        tissue_mask,
                        overview_downsample,
                        extraction_size,
                    )
                    stats.mask_filtered_tile_coords += len(valid_tile_coords)

                    if not valid_tile_coords:
                        self.logger.info(
                            f"No valid in-bounds tiles for {wsi_name} particle {particle_id}"
                        )
                        continue

                    coord_batches = [
                        valid_tile_coords[i:i + self.config.batch_size] 
                        for i in range(0, len(valid_tile_coords), self.config.batch_size)
                    ]

                    with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                        try:
                            futures = []
                            for batch in coord_batches:
                                future = executor.submit(
                                    process_tile_batch,
                                    (
                                        wsi_path,
                                        batch,
                                        polygon,
                                        self.config.downsample,
                                        self.config.tile_size,
                                        self.config.min_tissue_fraction,
                                        self.config.max_black_fraction,
                                        self.config.max_white_fraction,
                                        self.config.min_intensity_std,
                                    )
                                )
                                futures.append(future)

                            from threading import Lock
                            output_lock = Lock()
                            
                            for future in tqdm(as_completed(futures), 
                                            total=len(futures),
                                            desc=f"Processing {wsi_name} particle {idx}"):
                                try:
                                    results = future.result()
                                    if results:
                                        stats.polygon_filtered_tile_coords += len(results)
                                        for orig_x, orig_y, tile_array in results:
                                            try:
                                                tile = Image.fromarray(tile_array, mode='RGB')
                                                
                                                original_size = self.config.tile_size * self.config.downsample
                                                
                                                filename = (
                                                    f"{wsi_name}_particle_{particle_id}_"
                                                    f"tile_{orig_x}_{orig_y}_"
                                                    f"size_{original_size}x{original_size}.png"
                                                )
                                                
                                                with output_lock:
                                                    output_path = output_dir / filename
                                                    if self.config.overwrite_existing or not output_path.exists():
                                                        tile.save(output_path, "PNG", quality=100)
                                                        stats.saved_tile_files += 1
                                                    kept_coords.append((orig_x, orig_y))
                                                    stats.kept_tile_coords += 1
                                                
                                            except Exception as e:
                                                self.logger.error(f"Error saving tile at ({orig_x}, {orig_y}): {str(e)}")
                                                continue
                                            finally:
                                                if hasattr(tile, 'close'):
                                                    tile.close()
                                except Exception as e:
                                    self.logger.error(f"Error processing batch result: {str(e)}")
                                    continue
                                    
                        except KeyboardInterrupt:
                            self.logger.warning("Caught KeyboardInterrupt, cleaning up...")
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise
                            
                except Exception as e:
                    self.logger.error(f"Error processing particle {idx}: {str(e)}")
                    continue

        except KeyboardInterrupt:
            self.logger.warning("Processing interrupted by user")
            raise
        except Exception as e:
            self.logger.error(f"Error processing WSI {wsi_path}: {str(e)}")
            raise

        (
            stats.estimated_covered_slide_pixels,
            stats.estimated_covered_slide_fraction,
            stats.estimated_covered_tissue_pixels,
            stats.estimated_covered_tissue_fraction,
        ) = self.compute_tile_coverage_stats(
            kept_coords,
            tissue_mask,
            tissue_mask.shape,
            overview_downsample,
            extraction_size,
            slide_pixels,
        )
        self.log_stats(stats)
        self.append_stats_row(stats)
        return stats

    def extract_tiles_without_annotations(self, wsi_path: Path, output_dir: Path) -> SlideExtractionStats:
        """Extract tiles from the full slide area without using annotations."""
        if not wsi_path.exists():
            self.logger.error(f"WSI file not found: {wsi_path}")
            raise FileNotFoundError(f"WSI file not found: {wsi_path}")

        try:
            wsi_name = self.get_effective_wsi_name(wsi_path)
            with SlideReader(wsi_path) as wsi:
                width, height = wsi.dimensions
                self.logger.info(
                    f"Processing full slide for {wsi_name} from {wsi_path.parent}, dimensions: {(width, height)}"
                )

            extraction_size = self.config.tile_size * self.config.downsample
            max_x_start = width - extraction_size
            max_y_start = height - extraction_size
            tissue_mask, overview_downsample = self.build_tissue_mask(wsi_path)
            slide_pixels = width * height
            step_size = (self.config.tile_size - self.config.overlap) * self.config.downsample
            stats = SlideExtractionStats(
                slide_name=wsi_name,
                slide_width=width,
                slide_height=height,
                slide_pixels=slide_pixels,
                overview_downsample=overview_downsample,
                mask_width=int(tissue_mask.shape[1]),
                mask_height=int(tissue_mask.shape[0]),
                mask_pixels=int(tissue_mask.size),
                mask_tissue_pixels=int(np.count_nonzero(tissue_mask)),
                mask_tissue_fraction=float(np.mean(tissue_mask)) if tissue_mask.size else 0.0,
                extraction_size_px=extraction_size,
                step_size_px=step_size,
                mask_mode=self.config.mask_mode,
            )
            stats.preview_slide_path, stats.preview_mask_path = self.save_slide_previews(
                wsi_path,
                wsi_name,
                tissue_mask,
            )
            kept_coords: List[Tuple[int, int]] = []

            if max_x_start < 0 or max_y_start < 0:
                self.logger.warning(
                    f"Skipping {wsi_name}: extraction window {extraction_size}x{extraction_size} "
                    f"is larger than slide dimensions {width}x{height}"
                )
                self.log_stats(stats)
                self.append_stats_row(stats)
                return stats

            tile_coords = []

            current_x = 0
            while current_x <= max_x_start:
                current_y = 0
                while current_y <= max_y_start:
                    tile_coords.append((current_x, current_y))
                    current_y += step_size
                current_x += step_size
            stats.generated_tile_coords = len(tile_coords)

            tile_coords = self.filter_tile_coords_with_tissue_mask(
                tile_coords,
                tissue_mask,
                overview_downsample,
                extraction_size,
            )
            stats.in_bounds_tile_coords = len(tile_coords)
            stats.mask_filtered_tile_coords = len(tile_coords)

            coord_batches = [
                tile_coords[i:i + self.config.batch_size]
                for i in range(0, len(tile_coords), self.config.batch_size)
            ]

            with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = []
                for batch in coord_batches:
                    future = executor.submit(
                        process_tile_batch_no_annotation,
                        (
                            wsi_path,
                            batch,
                            self.config.downsample,
                            self.config.tile_size,
                            self.config.min_tissue_fraction,
                            self.config.max_black_fraction,
                            self.config.max_white_fraction,
                            self.config.min_intensity_std,
                        )
                    )
                    futures.append(future)

                from threading import Lock
                output_lock = Lock()
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {wsi_name} full slide"):
                    try:
                        results = future.result()
                        if results:
                            stats.kept_tile_coords += len(results)
                            for orig_x, orig_y, tile_array in results:
                                tile = None
                                try:
                                    tile = Image.fromarray(tile_array, mode='RGB')
                                    original_size = self.config.tile_size * self.config.downsample
                                    filename = (
                                        f"{wsi_name}_tile_{orig_x}_{orig_y}_"
                                        f"size_{original_size}x{original_size}.png"
                                    )
                                    with output_lock:
                                        output_path = output_dir / filename
                                        if self.config.overwrite_existing or not output_path.exists():
                                            tile.save(output_path, "PNG", quality=100)
                                            stats.saved_tile_files += 1
                                        kept_coords.append((orig_x, orig_y))
                                except Exception as e:
                                    self.logger.error(f"Error saving tile at ({orig_x}, {orig_y}): {str(e)}")
                                finally:
                                    if tile is not None and hasattr(tile, 'close'):
                                        tile.close()
                    except Exception as e:
                        self.logger.error(f"Error processing batch result: {str(e)}")

        except KeyboardInterrupt:
            self.logger.warning("Processing interrupted by user")
            raise
        except Exception as e:
            self.logger.error(f"Error processing WSI {wsi_path}: {str(e)}")
            raise

        (
            stats.estimated_covered_slide_pixels,
            stats.estimated_covered_slide_fraction,
            stats.estimated_covered_tissue_pixels,
            stats.estimated_covered_tissue_fraction,
        ) = self.compute_tile_coverage_stats(
            kept_coords,
            tissue_mask,
            tissue_mask.shape,
            overview_downsample,
            extraction_size,
            slide_pixels,
        )
        self.log_stats(stats)
        self.append_stats_row(stats)
        return stats

def main():
    """Extract tiles from whole slide images across a directory and save extraction stats."""
    parser = argparse.ArgumentParser(description="Extract tiles from whole slide images")
    parser = add_path_args(parser)
    
    parser.add_argument("--tile-size", type=int, default=256, help="Size of extracted tiles")
    parser.add_argument("--downsample", type=int, default=10, help="Downsample factor")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap between tiles")
    parser.add_argument("--num-workers", type=int, default=32, help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for processing")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing tiles instead of skipping already processed slides")
    parser.add_argument(
        "--mask-mode",
        type=str,
        choices=["standard", "tissue"],
        default="standard",
        help="Mask mode for coordinate filtering and previews: standard excludes only black background, tissue keeps only tissue",
    )
    parser.add_argument("--min-tissue-fraction", type=float, default=0.02, help="Minimum fraction of non-background tissue pixels required to keep a tile")
    parser.add_argument("--max-black-fraction", type=float, default=0.95, help="Maximum allowed fraction of near-black pixels in a saved tile")
    parser.add_argument("--max-white-fraction", type=float, default=0.98, help="Maximum allowed fraction of near-white pixels in a saved tile")
    parser.add_argument("--min-intensity-std", type=float, default=5.0, help="Minimum grayscale standard deviation required to keep a tile")
    parser.add_argument(
        "--slides-dir",
        type=Path,
        default=None,
        help="Directory containing slide files. Defaults to <raw_dir>/slides"
    )
    parser.add_argument(
        "--scan-mode",
        type=str,
        choices=["recursive", "flat"],
        default="recursive",
        help="How to scan --slides-dir: recursive for all subfolders, flat for only this folder"
    )
    parser.add_argument(
        "--ds-name",
        type=str,
        default=None,
        help="Optional dataset name. If set, tiles are stored under data/processed/<ds_name>/tiles"
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Extract tiles on a full-slide grid without using annotation files"
    )
    
    args = parser.parse_args()
    
    paths = get_project_paths(base_dir=args.base_dir)
    
    if args.data_dir:
        paths["DATA_DIR"] = args.data_dir
        paths["RAW_DIR"] = args.data_dir / "raw"
        paths["PROCESSED_DIR"] = args.data_dir / "processed" 
    
    if args.output_dir:
        paths["RESULTS_DIR"] = args.output_dir
        paths["LOGS_DIR"] = args.output_dir / "logs"
    
    start_time = time.time()
    
    config = ExtractionConfig(
        tile_size=args.tile_size,
        downsample=args.downsample,
        overlap=args.overlap,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        prefix_parent_dir_in_wsi_name=(args.scan_mode == "recursive"),
        overwrite_existing=args.overwrite,
        mask_mode=args.mask_mode,
        min_tissue_fraction=args.min_tissue_fraction,
        max_black_fraction=args.max_black_fraction,
        max_white_fraction=args.max_white_fraction,
        min_intensity_std=args.min_intensity_std,
    )
    
    wsi_dir = args.slides_dir if args.slides_dir else (paths["RAW_DIR"] / 'slides')
    annotation_dir = paths["RAW_DIR"] / 'annotations'
    processed_base_dir = paths["PROCESSED_DIR"]
    if args.ds_name and args.ds_name.strip():
        processed_base_dir = processed_base_dir / args.ds_name.strip()
    output_dir = processed_base_dir / 'tiles'
    # Own subfolder under LOGS_DIR (not flat) -- see create_splits.py's identical comment; also
    # keeps the stats CSV/preview dir below grouped with their run's log instead of loose in
    # LOGS_DIR alongside every other operation's files.
    log_dir = paths["LOGS_DIR"] / "tile_extraction"

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_csv_path = log_dir / f'tile_extraction_stats_{run_timestamp}.csv'
    preview_dir = log_dir / f'tile_extraction_previews_{run_timestamp}'

    setup_logging(log_dir, operation="tile_extraction", force_timestamp=True)
    quiet_external_loggers()
    
    logging.info("Project paths:")
    for path_name, path_value in paths.items():
        logging.info(f"  {path_name}: {path_value}")
    
    extractor = WSITileExtractor(
        config,
        paths,
        stats_csv_path=stats_csv_path,
        preview_dir=preview_dir,
    )
    logging.info(f"Tile extraction stats CSV: {stats_csv_path}")
    logging.info(f"Tile extraction preview directory: {preview_dir}")
    
    if not wsi_dir.exists():
        raise FileNotFoundError(f"Slides directory does not exist: {wsi_dir}")

    if args.scan_mode == "recursive":
        supported_patterns = ['**/*.mrxs', '**/*.tif', '**/*.tiff']
    else:
        supported_patterns = ['*.mrxs', '*.tif', '*.tiff']

    wsi_files = []
    for pattern in supported_patterns:
        wsi_files.extend(wsi_dir.glob(pattern))

    wsi_files = sorted(set(wsi_files))
    total_wsis = len(wsi_files)
    logging.info(f"Found {total_wsis} WSI files to process")
    
    for wsi_idx, wsi_path in enumerate(tqdm(wsi_files, desc="Processing WSIs"), 1):
        if args.no_annotations:
            extractor.extract_tiles_without_annotations(wsi_path, output_dir)
            logging.info(f"Completed WSI {wsi_idx}/{total_wsis}: {wsi_path.stem}")
            continue

        annotation_path = annotation_dir / f"{wsi_path.stem}_annotations.json"
        
        if not annotation_path.exists():
            logging.warning(f"No annotation file found for {wsi_path.stem}")
            continue
            
        annotation_data = extractor.load_annotations(annotation_path)
        if not annotation_data:
            continue
            
        extractor.extract_tiles(wsi_path, annotation_data, output_dir)
        
        logging.info(f"Completed WSI {wsi_idx}/{total_wsis}: {wsi_path.stem}")
    
    execution_time = time.time() - start_time
    logging.info(f"Total execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()