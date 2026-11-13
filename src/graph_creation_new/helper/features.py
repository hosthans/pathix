from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import logging
from pathlib import Path
import sys
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.graph_creation_new.config import BuildContext
from src.graph_creation_new.helper.transforms import apply_patch_transforms
from src.graph_creation_new.helper.slide import resolve_patch_size_px
from src.utils.stain_normalization import macenko_fallback_stats


def _log_extraction_progress_snapshot(slide_name: str, patches_done: int, patches_total: int) -> None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            mem_avail_kib = mem_total_kib = 0
            for line in fh:
                if line.startswith("MemAvailable:"):
                    mem_avail_kib = int(line.split()[1])
                elif line.startswith("MemTotal:"):
                    mem_total_kib = int(line.split()[1])
    except Exception:
        mem_avail_kib = mem_total_kib = 0

    pressure_some_avg10 = None
    try:
        with open("/proc/pressure/memory", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("some "):
                    for token in line.split():
                        if token.startswith("avg10="):
                            pressure_some_avg10 = float(token.split("=", 1)[1])
                            break
                    break
    except Exception:
        pressure_some_avg10 = None

    gpu_free_gib = gpu_total_gib = None
    if torch.cuda.is_available():
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            gpu_free_gib = free_b / (1024.0 ** 3)
            gpu_total_gib = total_b / (1024.0 ** 3)
        except Exception:
            pass

    logging.info(
        "[%s] feature extraction progress %d/%d -- host mem_avail=%.2fGiB/%.2fGiB, "
        "memory.pressure(some,avg10)=%s%%, gpu_free=%s/%sGiB",
        slide_name, patches_done, patches_total,
        mem_avail_kib / (1024.0 ** 2), mem_total_kib / (1024.0 ** 2),
        f"{pressure_some_avg10:.1f}" if pressure_some_avg10 is not None else "n/a",
        f"{gpu_free_gib:.2f}" if gpu_free_gib is not None else "n/a",
        f"{gpu_total_gib:.2f}" if gpu_total_gib is not None else "n/a",
    )


def _as_feature_matrix(values: Union[np.ndarray, torch.Tensor], expected_batch: int) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)

    if array.ndim == 1:
        if expected_batch != 1:
            raise ValueError(f"Feature output shape {array.shape} does not match batch={expected_batch}")
        array = array[None, :]
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)

    if array.shape[0] != expected_batch:
        raise ValueError(f"Feature output batch {array.shape[0]} does not match expected {expected_batch}")

    return array.astype(np.float32)


def _read_and_transform_one_patch(
    context: BuildContext,
    x: int,
    y: int,
    patch_size: int,
    downsample: float,
    target_size: int,
) -> np.ndarray:
    img = context.slide.read_region(
        (int(x * downsample), int(y * downsample)),
        context.cfg.patch_level,
        (patch_size, patch_size),
    )
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[-1] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    if context.cfg.feature_transforms:
        arr = apply_patch_transforms(
            arr,
            context.cfg.feature_transforms,
            seed=context.cfg.transform_seed,
            slide_key=str(context.slide_path),
            patch_xy=(int(x), int(y)),
        )
    if target_size != patch_size:
        arr = cv2.resize(arr, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return arr.astype(np.float32) / 255.0


def _to_patch_tensor(context: BuildContext, top_left_xy: np.ndarray) -> torch.Tensor:
    patch_size = resolve_patch_size_px(context)
    downsample = float(context.slide.level_downsamples[context.cfg.patch_level])
    target_size = int(context.cfg.feature_image_size) if context.cfg.feature_image_size is not None else patch_size

    n_workers = max(1, int(getattr(context.cfg, "feature_prep_workers", 1)))
    coords = [(int(x), int(y)) for x, y in top_left_xy]
    if n_workers > 1 and len(coords) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            patch_images = list(
                executor.map(
                    lambda xy: _read_and_transform_one_patch(
                        context, xy[0], xy[1], patch_size, downsample, target_size
                    ),
                    coords,
                )
            )
    else:
        patch_images = [
            _read_and_transform_one_patch(context, x, y, patch_size, downsample, target_size)
            for x, y in coords
        ]

    batch = np.stack(patch_images, axis=0)
    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()

    if context.cfg.feature_imagenet_normalize:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[None, :, None, None]
        tensor = (tensor - mean) / std

    return tensor


_INT_VIEW_BY_ITEMSIZE = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _weight_fingerprint(model: torch.nn.Module) -> list:
    """Per-tensor integer sum over the raw bits of every parameter/buffer.

    Any single bit flip changes the sum of its tensor (by +-2^k, also under int64 wraparound), so this
    detects silent corruption of the weights in GPU memory. One device->host sync for the whole model.
    """
    sums = []
    with torch.no_grad():
        for tensor in model.state_dict().values():
            if not torch.is_tensor(tensor) or tensor.numel() == 0:
                continue
            bits = tensor.detach().contiguous().view(_INT_VIEW_BY_ITEMSIZE[tensor.element_size()])
            sums.append(bits.sum(dtype=torch.int64))
    return torch.stack(sums).cpu().tolist() if sums else []


class _RedundantInference:
    """Bit-exact feature inference on hardware that can silently corrupt a result (non-ECC GPU memory).

    Every batch is computed until two passes agree bit for bit (at most MAX_PASSES), each pass with a fresh
    upload from host memory (ECC) so a corrupted input copy on the GPU is caught too. The weights are
    checksummed before and after every pass; if they changed they are restored from a host copy taken when
    the guard was created (right after the model was loaded) and the pass is discarded.
    """

    MAX_PASSES = 5

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.reference_fingerprint = _weight_fingerprint(model)
        self.host_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

    @classmethod
    def for_model(cls, model: torch.nn.Module) -> "_RedundantInference":
        """One guard per model object (the worker reuses the model across slides; the host copy is large)."""
        guard = model.__dict__.get("_redundant_inference_guard")
        if guard is None:
            guard = cls(model)
            model.__dict__["_redundant_inference_guard"] = guard
        return guard

    def _weights_intact(self) -> bool:
        return _weight_fingerprint(self.model) == self.reference_fingerprint

    def _restore_weights(self, where: str) -> None:
        logging.error("Feature model weights changed on the GPU (%s) -- restoring them from the host copy.", where)
        self.model.load_state_dict(self.host_state)
        if not self._weights_intact():
            raise RuntimeError("Feature model weights still differ from the reference after restoring them.")

    def infer(self, cpu_tensor: torch.Tensor, run_once, where: str) -> np.ndarray:
        """Return run_once(cpu_tensor) as soon as two passes agree bit for bit."""
        passes = []
        for _ in range(self.MAX_PASSES):
            if not self._weights_intact():
                self._restore_weights(f"{where}, before pass")
            features = run_once(cpu_tensor)
            if not self._weights_intact():
                self._restore_weights(f"{where}, during pass")
                continue
            bits = features.view(np.uint32)
            for previous in passes:
                if np.array_equal(previous.view(np.uint32), bits):
                    if len(passes) > 1:
                        logging.warning(
                            "Redundant feature inference %s: passes disagreed, accepted the result after %d passes.",
                            where, len(passes) + 1,
                        )
                    return features
            if passes:
                differing_rows = np.nonzero((passes[0].view(np.uint32) != bits).any(axis=1))[0]
                logging.warning(
                    "Redundant feature inference %s: pass %d differs from pass 1 in %d row(s) (first: %s) -- "
                    "running another pass.",
                    where, len(passes) + 1, differing_rows.size, differing_rows[:10].tolist(),
                )
            passes.append(features)
        raise RuntimeError(
            f"Redundant feature inference {where}: no two of {self.MAX_PASSES} passes agreed bit for bit -- "
            "the feature model is not deterministic on this setup."
        )


def _infer_patch_features_with_model(context: BuildContext, top_left_xy: np.ndarray) -> np.ndarray:
    model = context.node_feature_model
    if model is None:
        raise ValueError("node_feature_model is None")

    n_patches = int(top_left_xy.shape[0])
    batch_size = max(1, int(context.cfg.feature_batch_size))
    device = torch.device("cuda" if context.cfg.feature_use_cuda_if_available and torch.cuda.is_available() else "cpu")

    is_torch_module = isinstance(model, torch.nn.Module)
    previous_training_mode = None
    if is_torch_module:
        previous_training_mode = model.training
        model.eval()
        model.to(device)

    total_batches = max(1, (n_patches + batch_size - 1) // batch_size)
    batch_starts = range(0, n_patches, batch_size)
    feature_desc = f"features:{Path(context.slide_path).name}"

    all_features: Optional[np.ndarray] = None
    guard = (
        _RedundantInference.for_model(model)
        if is_torch_module and bool(getattr(context.cfg, "feature_redundancy_check", False))
        else None
    )

    def _prepare_batch(start: int) -> torch.Tensor:
        stop = min(start + batch_size, n_patches)
        return _to_patch_tensor(context, top_left_xy[start:stop])

    def _run_model_once(cpu_tensor: torch.Tensor) -> np.ndarray:
        tensor = cpu_tensor.to(device, non_blocking=True)
        if is_torch_module:
            raw = model(tensor)
        elif callable(model):
            raw = model(tensor)
        else:
            raise ValueError("node_feature_model must be a torch.nn.Module or callable")

        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        return _as_feature_matrix(raw, expected_batch=tensor.shape[0])

    try:
        with torch.no_grad():
            with ThreadPoolExecutor(max_workers=1) as prefetch_executor:
                pending = prefetch_executor.submit(_prepare_batch, batch_starts[0]) if n_patches > 0 else None
                slide_name = Path(context.slide_path).name
                last_progress_log_ts = time.monotonic()
                for i, start in enumerate(tqdm(batch_starts, total=total_batches, desc=feature_desc, unit="batch", leave=False)):
                    cpu_tensor = pending.result()
                    next_start = start + batch_size
                    if next_start < n_patches:
                        pending = prefetch_executor.submit(_prepare_batch, next_start)

                    if guard is None:
                        batch_features = _run_model_once(cpu_tensor)
                    else:
                        batch_features = guard.infer(
                            cpu_tensor, _run_model_once, where=f"[{slide_name} batch {i + 1}/{total_batches}]"
                        )
                    if all_features is None:
                        all_features = np.empty((n_patches, batch_features.shape[1]), dtype=np.float32)
                    all_features[start:start + batch_features.shape[0]] = batch_features

                    del cpu_tensor, batch_features

                    if (i + 1) % 500 == 0:
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                    now = time.monotonic()
                    if now - last_progress_log_ts >= 60.0:
                        _log_extraction_progress_snapshot(slide_name, min(next_start, n_patches), n_patches)
                        last_progress_log_ts = now
    finally:
        if is_torch_module and previous_training_mode is not None:
            model.train(previous_training_mode)

    if all_features is None:
        return np.empty((0, 0), dtype=np.float32)
    return all_features


def _aggregate_region_features(patch_features: np.ndarray, region_labels: np.ndarray, n_regions: int) -> np.ndarray:
    dim = int(patch_features.shape[1])
    region_features = np.zeros((n_regions, dim), dtype=np.float32)
    counts = np.zeros((n_regions,), dtype=np.int64)

    for idx, region_id in enumerate(region_labels):
        region_features[int(region_id)] += patch_features[idx]
        counts[int(region_id)] += 1

    nonzero = counts > 0
    region_features[nonzero] /= counts[nonzero, None].astype(np.float32)
    return region_features


def compute_node_features(context: BuildContext) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute patch-, region-, and slide-level node features for the current build context."""
    if context.node_features_cache is not None:
        return context.node_features_cache

    if context.node_feature_model is None:
        return None

    if context.patch_topleft is None or context.region_labels is None or context.region_xy is None:
        raise ValueError("Patch and region data must exist before computing node features")

    # Reset before, one summary line + region_diagnostics entry after instead of one INFO line
    # per patch (see stain_normalization.py's _MacenkoFallbackStats docstring for why the
    # per-patch version was suppressed 2026-08-12 -- it was drowning out the rest of the build
    # log on slides with a lot of faint tissue).
    macenko_fallback_stats.reset()
    patch_features = _infer_patch_features_with_model(context, context.patch_topleft)
    fallback_summary_text = macenko_fallback_stats.format_summary()
    if fallback_summary_text is not None:
        logging.info("Macenko beta-fallback for %s: %s", Path(context.slide_path).name, fallback_summary_text)
    if isinstance(getattr(context, "region_diagnostics", None), dict):
        context.region_diagnostics["macenko_fallback"] = macenko_fallback_stats.snapshot()

    region_features = _aggregate_region_features(
        patch_features,
        context.region_labels,
        n_regions=int(context.region_xy.shape[0]),
    )
    slide_features = region_features.mean(axis=0, keepdims=True).astype(np.float32)
    context.node_features_cache = (patch_features, region_features, slide_features)
    return context.node_features_cache
