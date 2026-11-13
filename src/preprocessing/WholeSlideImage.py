import openslide
from pathlib import Path
import numpy as np
import cv2
import time
import os
import math
import multiprocessing as mp
import sys

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


from src.preprocessing.util_classes import isInContourV3_Easy, isInContourV3_Hard, isInContourV2, isInContourV1, \
    Contour_Checking_fn


class WholeSlideImage:
    def __init__(self, path: Path):
        """Open the slide at the given path and initialize its tissue/tumor contour state."""
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")

        self.path = path
        self.wsi = openslide.open_slide(str(path))

        self.level_downsamples = self._assert_level_downsamples()
        self.level_dim = self.wsi.level_dimensions

        self.contours_tissue = None
        self.contours_tumor = None
        self.holes_tissue = None

    def segment_tissue(self, seg_level=0, sthresh=20, sthresh_up=255, mthresh=7, close=0, use_otsu=False,
                      filter_params={'a_t': 100}, ref_patch_size=512, exclude_ids=[], keep_ids=[]):
        """Segment tissue contours via HSV -> median blur -> binary threshold."""

        def _filter_contours(_contours, _hierarchy, _filter_params):
            filtered = []

            hierarchy_1 = np.flatnonzero(_hierarchy[:, 1] == -1)
            all_holes = []

            for cont_idx in hierarchy_1:
                cont = _contours[cont_idx]
                holes = np.flatnonzero(_hierarchy[:, 1] == cont_idx)
                a = cv2.contourArea(cont)
                hole_areas = [cv2.contourArea(_contours[hole_idx]) for hole_idx in holes]
                a = a - np.array(hole_areas).sum()
                if a == 0: continue
                if tuple((_filter_params['a_t'],)) < tuple((a,)):
                    filtered.append(cont_idx)
                    all_holes.append(holes)

            foreground_contours = [_contours[cont_idx] for cont_idx in filtered]

            hole_contours = []

            for hole_ids in all_holes:
                unfiltered_holes = [_contours[idx] for idx in hole_ids]
                unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                unfilered_holes = unfilered_holes[:_filter_params['max_n_holes']]
                filtered_holes = []

                for hole in unfilered_holes:
                    if cv2.contourArea(hole) > _filter_params['a_h']:
                        filtered_holes.append(hole)

                hole_contours.append(filtered_holes)

            return foreground_contours, hole_contours

        img = np.array(self.wsi.read_region((0, 0), seg_level, self.level_dim[seg_level]))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)

        if use_otsu:
            _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        else:
            _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

        scale = self.level_downsamples[seg_level]
        scaled_ref_patch_area = int(ref_patch_size ** 2 / (scale[0] * scale[1]))
        filter_params = filter_params.copy()
        filter_params.setdefault('a_h', 16)
        filter_params.setdefault('max_n_holes', 8)
        filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area
        filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area

        contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None or len(contours) == 0:
            self.contours_tissue = []
            self.holes_tissue = []
            return

        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        if filter_params:
            foreground_contours, hole_contours = _filter_contours(
                contours,
                hierarchy,
                filter_params,
            )
        else:
            foreground_contours, hole_contours = contours, []

        self.contours_tissue = self.scaleContourDim(foreground_contours, scale)
        self.holes_tissue = self.scaleHolesDim(hole_contours, scale)

        if len(keep_ids) > 0:
            contour_ids = set(keep_ids) - set(exclude_ids)
        else:
            contour_ids = set(np.arange(len(self.contours_tissue))) - set(exclude_ids)

        self.contours_tissue = [self.contours_tissue[i] for i in contour_ids]
        self.holes_tissue = [self.holes_tissue[i] for i in contour_ids]

    def _assert_level_downsamples(self):
        level_downsamples = []
        dim_0 = self.wsi.level_dimensions[0]

        for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
            estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (
            downsample, downsample) else level_downsamples.append((downsample, downsample))

        return level_downsamples

    def process_contours(self, save_path, patch_level=0, patch_size=256, step_size=256, **kwargs):
        """Process all tissue contours into patch coordinates and return the hdf5 file."""
        save_path_hdf5 = os.path.join(save_path, str(self.name) + '.h5')
        print("Creating patches for: ", self.name, "...", )
        elapsed = time.time()
        n_contours = len(self.contours_tissue)
        print("Total number of contours to process: ", n_contours)
        fp_chunk_size = math.ceil(n_contours * 0.05)
        init = True
        for idx, cont in enumerate(self.contours_tissue):
            if (idx + 1) % fp_chunk_size == fp_chunk_size:
                print('Processing contour {}/{}'.format(idx, n_contours))

            asset_dict, attr_dict = self.process_contour(cont, self.holes_tissue[idx], patch_level, save_path,
                                                         patch_size, step_size, **kwargs)

        return self.hdf5_file

    def process_contour(self, cont, contour_holes, patch_level, save_path, patch_size=256, step_size=256,
                        contour_fn='four_pt', use_padding=True, top_left=None, bot_right=None):
        """Generate patch coordinates within a single tissue contour, respecting its holes."""
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (
        0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])

        patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
        ref_patch_size = (patch_size * patch_downsample[0], patch_size * patch_downsample[1])

        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y + h
            stop_x = start_x + w
        else:
            stop_y = min(start_y + h, img_h - ref_patch_size[1] + 1)
            stop_x = min(start_x + w, img_w - ref_patch_size[0] + 1)

        print("Bounding Box:", start_x, start_y, w, h)
        print("Contour Area:", cv2.contourArea(cont))

        if bot_right is not None:
            stop_y = min(bot_right[1], stop_y)
            stop_x = min(bot_right[0], stop_x)
        if top_left is not None:
            start_y = max(top_left[1], start_y)
            start_x = max(top_left[0], start_x)

        if bot_right is not None or top_left is not None:
            w, h = stop_x - start_x, stop_y - start_y
            if w <= 0 or h <= 0:
                print("Contour is not in specified ROI, skip")
                return {}, {}
            else:
                print("Adjusted Bounding Box:", start_x, start_y, w, h)

        if isinstance(contour_fn, str):
            if contour_fn == 'four_pt':
                cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'four_pt_hard':
                cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'center':
                cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size[0])
            elif contour_fn == 'basic':
                cont_check_fn = isInContourV1(contour=cont)
            else:
                raise NotImplementedError
        else:
            assert isinstance(contour_fn, Contour_Checking_fn)
            cont_check_fn = contour_fn

        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]

        x_range = np.arange(start_x, stop_x, step=step_size_x)
        y_range = np.arange(start_y, stop_y, step=step_size_y)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

        num_workers = mp.cpu_count()
        if num_workers > 4:
            num_workers = 4
        pool = mp.Pool(num_workers)

        iterable = [(coord, contour_holes, ref_patch_size[0], cont_check_fn) for coord in coord_candidates]
        results = pool.starmap(WholeSlideImage.process_coord_candidate, iterable)
        pool.close()
        pool.join()
        results = np.array([result for result in results if result is not None])

        print('Extracted {} coordinates'.format(len(results)))

        if len(results) > 0:
            asset_dict = {'coords': results}

            attr = {'patch_size': patch_size,
                    'patch_level': patch_level,
                    'downsample': self.level_downsamples[patch_level],
                    'downsampled_level_dim': tuple(np.array(self.level_dim[patch_level])),
                    'level_dim': self.level_dim[patch_level],
                    'name': self.name,
                    'save_path': save_path}

            attr_dict = {'coords': attr}
            return asset_dict, attr_dict

        else:
            return {}, {}

    @staticmethod
    def scaleContourDim(contours, scale):
        """Scale a list of contours by the given factor."""
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        """Scale a list of per-contour hole lists by the given factor."""
        return [[np.array(hole * scale, dtype='int32') for hole in holes] for holes in contours]

    @staticmethod
    def process_coord_candidate(coord, contour_holes, ref_patch_size, cont_check_fn):
        """Return the coordinate if it lies within the contour (outside its holes), else None."""
        if WholeSlideImage.isInContours(cont_check_fn, coord, contour_holes, ref_patch_size):
            return coord
        else:
            return None

    @staticmethod
    def isInHoles(holes, pt, patch_size):
        """Return whether a patch's center point falls inside any of the given holes."""
        center = (pt[0] + patch_size // 2, pt[1] + patch_size // 2)
        for hole in holes:
            if cv2.pointPolygonTest(hole, tuple(np.array(center).astype(float)), False) > 0:
                return 1
        return 0

    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, patch_size=256):
        """Return whether a point is inside the contour and outside any of its holes."""
        if cont_check_fn(pt):
            if holes is not None and len(holes) > 0:
                return not WholeSlideImage.isInHoles(holes, pt, patch_size)
            return 1
        return 0

    @property
    def slide(self):
        """Return the underlying openslide object."""
        return self.wsi

    @property
    def name(self) -> str:
        """Return the slide's file name."""
        return self.path.name

    @property
    def dimensions(self):
        """Return the slide's dimensions at level 0."""
        return self.wsi.dimensions

    def close(self):
        """Close the underlying slide handle."""
        self.wsi.close()