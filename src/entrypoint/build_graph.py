import os
import sys


def _autoselect_least_busy_gpu(default: str = '0') -> str:
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10, check=True,
        )
        candidates = []
        for line in result.stdout.strip().splitlines():
            idx_str, free_str = (part.strip() for part in line.split(','))
            candidates.append((int(idx_str), int(free_str)))
        if not candidates:
            return default
        best_idx, best_free = max(candidates, key=lambda pair: pair[1])
        sys.stderr.write(
            f"[GPU auto-select] Picking GPU {best_idx} ({best_free} MiB free) "
            f"out of {candidates} -- set CUDA_VISIBLE_DEVICES explicitly to override.\n"
        )
        return str(best_idx)
    except Exception as exc:
        sys.stderr.write(f"[GPU auto-select] Falling back to GPU {default}: {exc}\n")
        return default


os.environ.setdefault('CUDA_VISIBLE_DEVICES', _autoselect_least_busy_gpu())

import argparse
import atexit
import csv
import faulthandler
import gc
import json
import logging
import multiprocessing as mp
import queue
import random
import resource
import signal
import shutil
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import openslide
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision
import yaml
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.graph_creation_new.enums import DatasetType, EdgeConstructionMode, NodeCreationMode
from src.graph_creation_new.error import AnnotationQualitySkipError, MissingAnnotationPolygonsError, NoTissuePatchesFoundError
from src.graph_creation_new.graph_creation import BuildConfig, CreationConfig, build_graph_from_slide
from src.graph_creation_new.visualization import graph_structure_summary
from src.paths.paths import get_project_paths
from src.utils.logging import setup_logging, quiet_external_loggers


_FAULT_LOG_FH: Optional[Any] = None
_SHUTDOWN_REQUESTED: bool = False
_SHUTDOWN_SIGNAL_COUNT: int = 0



def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.parent / f"{path.name}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)
        fh.write("\n")
    os.replace(tmp_path, path)


def _log_shutdown_signal(signum: int, _frame: Optional[Any]) -> None:
    global _SHUTDOWN_REQUESTED
    global _SHUTDOWN_SIGNAL_COUNT
    _SHUTDOWN_REQUESTED = True
    _SHUTDOWN_SIGNAL_COUNT += 1
    signal_name = signal.Signals(signum).name if signum in signal.Signals.__members__.values() else str(signum)
    logging.warning(
        "Received signal %s (%s) — stopping after current slide. (count=%d)",
        signum,
        signal_name,
        _SHUTDOWN_SIGNAL_COUNT,
    )
    try:
        faulthandler.dump_traceback(file=_FAULT_LOG_FH if _FAULT_LOG_FH is not None else sys.stderr, all_threads=True)
    except Exception:
        logging.exception("Failed to dump traceback for signal %s", signum)

    if _SHUTDOWN_SIGNAL_COUNT >= 3:
        logging.error("Received repeated interrupt signals. Forcing immediate exit.")
        os._exit(130)


def _configure_runtime_diagnostics(output_dir: Path, operation_name: str) -> Path:
    global _FAULT_LOG_FH
    crash_log = output_dir / f"{operation_name}_crash.log"
    _FAULT_LOG_FH = open(crash_log, "a", encoding="utf-8")
    faulthandler.enable(file=_FAULT_LOG_FH, all_threads=True)
    try:
        faulthandler.register(signal.SIGUSR1, file=_FAULT_LOG_FH, all_threads=True)
    except Exception:
        logging.debug("Could not register faulthandler for SIGUSR1.")

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _log_shutdown_signal)
        except Exception:
            logging.debug("Could not attach signal handler for %s", sig)

    def _cleanup_fault_log() -> None:
        global _FAULT_LOG_FH
        try:
            if _FAULT_LOG_FH is not None:
                _FAULT_LOG_FH.flush()
                _FAULT_LOG_FH.close()
        except Exception:
            pass
        _FAULT_LOG_FH = None

    atexit.register(_cleanup_fault_log)
    logging.info("Runtime diagnostics enabled. Crash log: %s", crash_log)
    return crash_log


def _cleanup_runtime_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _bytes_to_gib(value: float) -> float:
    return float(value) / (1024.0 ** 3)


def _read_meminfo_kib() -> Dict[str, int]:
    fields = {
        "MemTotal": 0,
        "MemAvailable": 0,
        "SwapTotal": 0,
        "SwapFree": 0,
    }
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 2:
                    continue
                key = parts[0].rstrip(":")
                if key in fields:
                    fields[key] = int(parts[1])
    except Exception:
        return fields
    return fields


def _log_runtime_memory_snapshot(context: str) -> None:
    mem = _read_meminfo_kib()
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        rss_gib = float(ru.ru_maxrss) / (1024.0 ** 2)
    except Exception:
        rss_gib = -1.0

    logging.info(
        "[%s] Host memory: mem_avail=%.2fGiB mem_total=%.2fGiB swap_free=%.2fGiB swap_total=%.2fGiB rss_max=%.2fGiB",
        context,
        float(mem.get("MemAvailable", 0)) / (1024.0 ** 2),
        float(mem.get("MemTotal", 0)) / (1024.0 ** 2),
        float(mem.get("SwapFree", 0)) / (1024.0 ** 2),
        float(mem.get("SwapTotal", 0)) / (1024.0 ** 2),
        rss_gib,
    )

    if torch.cuda.is_available():
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            allocated_b = torch.cuda.memory_allocated()
            reserved_b = torch.cuda.memory_reserved()
            logging.info(
                "[%s] GPU memory: free=%.2fGiB total=%.2fGiB allocated=%.2fGiB reserved=%.2fGiB",
                context,
                _bytes_to_gib(free_b),
                _bytes_to_gib(total_b),
                _bytes_to_gib(allocated_b),
                _bytes_to_gib(reserved_b),
            )
        except Exception:
            logging.exception("[%s] Could not query GPU memory snapshot", context)


def _save_graph_atomic(graph: Any, output_path: Path) -> None:
    temp_path = output_path.parent / f"{output_path.name}.tmp"
    try:
        torch.save(graph, temp_path)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                logging.warning("Could not remove temporary graph file: %s", temp_path)


def _build_graph_with_retry(
    wsi_path: Path,
    cfg: BuildConfig,
    node_feature_model: Optional[Any],
    creation_cfg: Optional[CreationConfig],
    dataset_enum: DatasetType,
    max_attempts: int = 2,
) -> Any:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return build_graph_from_slide(
                slide=wsi_path,
                cfg=cfg,
                node_feature_model=node_feature_model,
                creation_cfg=creation_cfg,
                dataset=dataset_enum,
            )
        except MissingAnnotationPolygonsError:
            raise
        except RuntimeError as exc:
            last_exc = exc
            message = str(exc).lower()
            is_cuda_oom = "out of memory" in message and "cuda" in message
            if is_cuda_oom and attempt < max_attempts:
                logging.warning(
                    "CUDA OOM while processing %s (attempt %d/%d). Cleaning cache and retrying once.",
                    wsi_path,
                    attempt,
                    max_attempts,
                )
                _cleanup_runtime_memory()
                time.sleep(1.0)
                continue
            raise
        except Exception as exc:
            last_exc = exc
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Graph build failed without explicit exception")


def _model_cfg_overrides(model_name: str) -> Dict[str, Any]:
    if model_name == "uni2":
        return {
            "feature_image_size": 224,
            "feature_imagenet_normalize": True,
            "feature_batch_size": 16,
            "feature_use_cuda_if_available": True,
        }
    if model_name == "prov_gigapath":
        return {
            "feature_image_size": 224,
            "feature_imagenet_normalize": True,
            "feature_batch_size": 2,
            "feature_use_cuda_if_available": True,
        }
    if model_name == "resnet50":
        return {
            "feature_image_size": 224,
            "feature_imagenet_normalize": True,
            "feature_batch_size": 32,
            "feature_use_cuda_if_available": True,
        }
    raise ValueError(f"Unsupported model: {model_name}")


def _effective_build_cfg(build_cfg: BuildConfig, explicit_build_fields: Set[str], model_name: str) -> BuildConfig:
    cfg_dict = asdict(build_cfg)
    for key, value in _model_cfg_overrides(model_name).items():
        if key not in explicit_build_fields:
            cfg_dict[key] = value
    return BuildConfig(**cfg_dict)


def _resolve_target_patch_size_um(
    build_cfg: BuildConfig,
    csv_path: Path,
    raw_path: Path,
    slide_name_column: str,
) -> BuildConfig:
    if build_cfg.target_patch_size_um is not None or not build_cfg.auto_detect_target_patch_size_um:
        return build_cfg
    df = pd.read_csv(csv_path)
    all_names = df[slide_name_column].astype(str).tolist()
    return _resolve_target_patch_size_um_from_names(build_cfg, all_names, raw_path)


def _resolve_target_patch_size_um_from_names(
    build_cfg: BuildConfig,
    all_names: List[str],
    raw_path: Path,
) -> BuildConfig:
    if build_cfg.target_patch_size_um is not None:
        return build_cfg
    if not build_cfg.auto_detect_target_patch_size_um:
        return build_cfg

    for slide_name in all_names:
        try:
            wsi_path = _resolve_slide_path(raw_path, slide_name)
            slide = openslide.OpenSlide(str(wsi_path))
            try:
                raw_mpp = slide.properties.get("openslide.mpp-x")
                mpp_x = float(raw_mpp) if raw_mpp is not None else None
            finally:
                slide.close()
        except Exception as exc:
            logging.warning("Could not read openslide.mpp-x for slide '%s' (trying next): %s", slide_name, exc)
            continue

        if mpp_x and mpp_x > 0:
            computed_target_um = float(build_cfg.patch_size) * mpp_x
            logging.info(
                "Auto-detected target_patch_size_um=%.4f from the first usable slide '%s' "
                "(patch_size=%d * openslide.mpp-x=%.5f). Every slide processed in this run -- "
                "including this one -- is scaled to match this physical size.",
                computed_target_um, slide_name, int(build_cfg.patch_size), mpp_x,
            )
            return replace(build_cfg, target_patch_size_um=computed_target_um)

        logging.warning("Slide '%s' has no usable openslide.mpp-x -- trying next slide.", slide_name)

    logging.warning(
        "auto_detect_target_patch_size_um is set but none of %d candidate slides had a usable "
        "openslide.mpp-x -- skipping; leaving target_patch_size_um unset (cfg.patch_size used as-is).",
        len(all_names),
    )
    return build_cfg


def _save_first_graph_artifacts(
    graph_obj: Any,
    graph_slide_path: Path,
    graph_slide_name: str,
    output_dir: Path,
    cfg: BuildConfig,
    csv_stem: str,
) -> bool:
    run_prefix = f"first_graph__{csv_stem}__{graph_slide_name}"
    return _save_graph_artifacts(
        graph_obj=graph_obj,
        graph_slide_path=graph_slide_path,
        graph_slide_name=graph_slide_name,
        output_dir=output_dir,
        cfg=cfg,
        run_prefix=run_prefix,
    )


def _save_graph_artifacts(
    graph_obj: Any,
    graph_slide_path: Path,
    graph_slide_name: str,
    output_dir: Path,
    cfg: BuildConfig,
    run_prefix: str,
) -> bool:
    from src.graph_creation_new.visualization import visualize_graph_structure

    try:
        visualize_graph_structure(
            data=graph_obj,
            out_dir=output_dir,
            slide_path=graph_slide_path,
            build_cfg=cfg,
        )
        artifact_names = [
            "thumbnail.png",
            "patches.png",
            "cells.png",
            "regions.png",
            "patch_knn_edges.png",
            "cell_knn_edges.png",
            "region_knn_edges.png",
            "hierarchy_overlay.png",
            "overview.png",
            "hierarchical_graph.png",
            "edge_type_overlay.png",
            "graph_insights.png",
            "summary.json",
            "meta.txt",
        ]
        copied = 0
        for artifact_name in artifact_names:
            source = output_dir / artifact_name
            if not source.exists():
                continue
            target = output_dir / f"{run_prefix}__{artifact_name}"
            shutil.copy2(source, target)
            copied += 1

        if copied <= 0:
            logging.warning(
                "No graph artifacts found to copy for slide %s (prefix=%s).",
                graph_slide_name,
                run_prefix,
            )
            return False

        logging.info("Graph visualization saved in %s", output_dir)
        logging.info("Saved %d first-graph validation artifacts with prefix '%s'", copied, run_prefix)
        return True
    except Exception as exc:
        logging.error("Graph visualization failed: %s", exc)
        return False


def _process_single_slide(
    slide_name: str,
    raw_path: Path,
    output_dir: Path,
    cfg: BuildConfig,
    embedding_tag: str,
    overwrite: bool,
    creation_cfg: Optional[CreationConfig],
    dataset_enum: DatasetType,
    node_feature_model: Optional[Any],
    config_path: Path,
    config_output_dir_name: str,
    graph_retry_attempts: int,
    visualize_each_graph: bool = False,
    csv_stem: str = "run",
) -> Dict[str, Any]:
    start_time = time.perf_counter()
    cache_name_candidates = _cache_name_candidates(slide_name)
    candidate_output_paths = [
        output_dir / f"{name}_{cfg.pipeline}_{embedding_tag}.pt"
        for name in cache_name_candidates
    ]
    existing_output_path = next((path for path in candidate_output_paths if path.exists()), None)

    if existing_output_path is not None and not overwrite:
        elapsed = time.perf_counter() - start_time
        return {
            "status": "skipped_existing",
            "slide_name": slide_name,
            "elapsed": elapsed,
            "output_path": str(existing_output_path),
        }

    try:
        wsi_path = _resolve_slide_path(raw_path, slide_name)
    except FileNotFoundError as exc:
        elapsed = time.perf_counter() - start_time
        return {
            "status": "missing_slide",
            "slide_name": slide_name,
            "elapsed": elapsed,
            "error": str(exc),
        }

    canonical_slide_name = wsi_path.stem
    output_path = output_dir / f"{canonical_slide_name}_{cfg.pipeline}_{embedding_tag}.pt"
    if output_path.exists() and not overwrite:
        elapsed = time.perf_counter() - start_time
        return {
            "status": "skipped_existing",
            "slide_name": slide_name,
            "elapsed": elapsed,
            "output_path": str(output_path),
        }

    try:
        _log_runtime_memory_snapshot(f"{slide_name}:pre_build")
        logging.info(
            (
                "Processing slide=%s with mask/patch params: "
                "tissue_frac_min=%.3f sat_min=%d val_max=%d morph_kernel=%d "
                "min_component_area=%d drop_border=%s border_margin=%d keep_largest=%s"
            ),
            slide_name,
            float(cfg.tissue_frac_min),
            int(cfg.sat_min),
            int(cfg.val_max),
            int(cfg.morph_kernel),
            int(cfg.min_component_area),
            bool(getattr(cfg, "mask_drop_border_components", False)),
            int(getattr(cfg, "mask_border_margin_px", 0)),
            str(getattr(cfg, "mask_keep_largest_components", None)),
        )

        build_start = time.perf_counter()
        graph = _build_graph_with_retry(
            wsi_path=wsi_path,
            cfg=cfg,
            node_feature_model=node_feature_model,
            creation_cfg=creation_cfg,
            dataset_enum=dataset_enum,
            max_attempts=graph_retry_attempts,
        )
        build_elapsed = time.perf_counter() - build_start
        logging.info("Slide %s graph construction finished in %.2fs", slide_name, build_elapsed)
        _log_runtime_memory_snapshot(f"{slide_name}:post_build")
    except MissingAnnotationPolygonsError as exc:
        _log_runtime_memory_snapshot(f"{slide_name}:skip_no_annotations")
        elapsed = time.perf_counter() - start_time
        return {
            "status": "skipped_no_annotations",
            "slide_name": slide_name,
            "elapsed": elapsed,
            "error": str(exc),
        }
    except NoTissuePatchesFoundError as exc:
        _log_runtime_memory_snapshot(f"{slide_name}:skip_no_patches")
        elapsed = time.perf_counter() - start_time
        return {
            "status": "skipped_no_patches",
            "slide_name": slide_name,
            "elapsed": elapsed,
            "error": str(exc),
        }
    except AnnotationQualitySkipError as exc:
        _log_runtime_memory_snapshot(f"{slide_name}:skip_annotation_qa")
        elapsed = time.perf_counter() - start_time
        return {
            "status": "skipped_annotation_qa",
            "slide_name": slide_name,
            "elapsed": elapsed,
            "error": str(exc),
            "annotation_qa": {
                "reason": exc.reason,
                "n_annotations": exc.n_annotations,
                "n_patches": exc.n_patches,
                "match_rate_normal": exc.match_rate_normal,
                "match_rate_swapped": exc.match_rate_swapped,
            },
        }
    except Exception as exc:
        _log_runtime_memory_snapshot(f"{slide_name}:build_failed")
        elapsed = time.perf_counter() - start_time
        return {
            "status": "build_failed",
            "slide_name": slide_name,
            "elapsed": elapsed,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    if not hasattr(graph, "meta") or getattr(graph, "meta") is None:
        graph.meta = {}
    if not isinstance(graph.meta, dict):
        graph.meta = {"meta_raw": str(graph.meta)}

    graph.meta["graph_config_path"] = str(config_path)
    graph.meta["graph_config_name"] = str(config_path.name)
    graph.meta["graph_config_stem"] = str(config_path.stem)
    graph.meta["graph_output_dir_name"] = str(config_output_dir_name)
    graph.meta["graph_output_dir"] = str(output_dir)

    _log_runtime_memory_snapshot(f"{slide_name}:pre_save")
    save_start = time.perf_counter()
    _save_graph_atomic(graph, output_path)
    save_elapsed = time.perf_counter() - save_start
    try:
        file_size_gib = _bytes_to_gib(output_path.stat().st_size)
    except Exception:
        file_size_gib = -1.0
    logging.info(
        "Slide %s graph saved in %.2fs -> %s (size=%.2fGiB)",
        slide_name,
        save_elapsed,
        output_path,
        file_size_gib,
    )
    _log_runtime_memory_snapshot(f"{slide_name}:post_save")

    if visualize_each_graph:
        try:
            viz_start = time.perf_counter()
            run_prefix = f"graph__{csv_stem}__{slide_name}"
            _save_graph_artifacts(
                graph_obj=graph,
                graph_slide_path=wsi_path,
                graph_slide_name=slide_name,
                output_dir=output_dir,
                cfg=cfg,
                run_prefix=run_prefix,
            )
            viz_elapsed = time.perf_counter() - viz_start
            logging.info("Slide %s visualization finished in %.2fs", slide_name, viz_elapsed)
            _log_runtime_memory_snapshot(f"{slide_name}:post_visualize")
        except Exception as exc:
            logging.warning("Could not visualize built graph for %s: %s", slide_name, exc)

    elapsed = time.perf_counter() - start_time
    return {
        "status": "processed",
        "slide_name": slide_name,
        "elapsed": elapsed,
        "output_path": str(output_path),
        "resolved_slide_path": str(wsi_path),
    }


_BLAS_THREAD_LIMITER = None  # keeps the threadpoolctl limit object alive for the whole process


def _pin_thread_counts(build_cfg: BuildConfig) -> None:
    """Apply cv2/torch/BLAS thread counts from the config to the CURRENT process.

    These settings are per process, so the spawned graph worker must call this itself -- setting them only
    in main() never reaches the process that actually reads, normalizes and embeds the patches.
    """
    global _BLAS_THREAD_LIMITER
    if build_cfg.cv2_num_threads is not None:
        cv2.setNumThreads(int(build_cfg.cv2_num_threads))
        logging.info("cv2.setNumThreads(%d)", int(build_cfg.cv2_num_threads))
    if build_cfg.torch_num_threads is not None:
        torch.set_num_threads(int(build_cfg.torch_num_threads))
        logging.info("torch.set_num_threads(%d)", int(build_cfg.torch_num_threads))
    if build_cfg.blas_num_threads is not None:
        try:
            from threadpoolctl import threadpool_info, threadpool_limits
        except ImportError:
            logging.warning(
                "blas_num_threads=%d set but threadpoolctl is not installed -- BLAS threads NOT limited "
                "(pip install threadpoolctl, or export OPENBLAS_NUM_THREADS before starting the build).",
                int(build_cfg.blas_num_threads),
            )
            return
        _BLAS_THREAD_LIMITER = threadpool_limits(limits=int(build_cfg.blas_num_threads), user_api="blas")
        libs = [f"{i.get('internal_api')}={i.get('num_threads')}" for i in threadpool_info() if i.get("user_api") == "blas"]
        logging.info("BLAS threads limited to %d (%s)", int(build_cfg.blas_num_threads), ", ".join(libs) or "no BLAS lib found")


def _graph_build_worker_loop(
    task_queue: Any,
    result_queue: Any,
    raw_path_str: str,
    output_dir_str: str,
    model_name: str,
    build_cfg_payload: Dict[str, Any],
    creation_cfg_payload: Optional[Dict[str, Any]],
    dataset_name: str,
    overwrite: bool,
    config_path_str: str,
    config_output_dir_name: str,
    graph_retry_attempts: int,
    visualize_each_graph: bool,
    csv_stem: str,
    seed: int,
    log_file_str: Optional[str] = None,
) -> None:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        if log_file_str:
            try:
                file_handler = logging.FileHandler(log_file_str)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
            except Exception:
                logging.exception(
                    "Worker could not attach a FileHandler at %s -- worker logs will only "
                    "reach the console/tmux pane, not the log file.", log_file_str,
                )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)

    raw_path = Path(raw_path_str)
    output_dir = Path(output_dir_str)
    creation_cfg = CreationConfig(**creation_cfg_payload) if creation_cfg_payload is not None else None
    cfg = BuildConfig(**build_cfg_payload)
    _pin_thread_counts(cfg)
    config_path = Path(config_path_str)
    embedding_tag = _embedding_tag_for_model(model_name)
    dataset_enum = DatasetType.GSDB if str(dataset_name).lower() == "gsdb" else DatasetType.CAMELYON17

    if creation_cfg is not None and creation_cfg.node_creation_mode == NodeCreationMode.CELL_SEGMENTATION:
        logging.info(
            "Worker: cell-segmentation mode detected, skipping node feature model loading for '%s'.",
            model_name,
        )
        node_feature_model = None
    else:
        node_feature_model, _ = _resolve_node_feature_model(model_name)
        logging.info("Worker: feature model '%s' loaded once and reused across slides.", model_name)

    while True:
        task = task_queue.get()
        if task is None:
            break

        slide_name = str(task["slide_name"]).strip()

        result = _process_single_slide(
            slide_name=slide_name,
            raw_path=raw_path,
            output_dir=output_dir,
            cfg=cfg,
            embedding_tag=embedding_tag,
            overwrite=overwrite,
            creation_cfg=creation_cfg,
            dataset_enum=dataset_enum,
            node_feature_model=node_feature_model,
            config_path=config_path,
            config_output_dir_name=config_output_dir_name,
            graph_retry_attempts=graph_retry_attempts,
            visualize_each_graph=visualize_each_graph,
            csv_stem=csv_stem,
        )
        result_queue.put(result)
        _cleanup_runtime_memory()


def _quiet_external_loggers() -> None:
    quiet_external_loggers()


def _float_token(value: Optional[float]) -> str:
    if value is None:
        return "na"
    if float(value).is_integer():
        return str(int(value))
    return str(float(value)).replace(".", "p")


def _edge_mode_token(prefix: str, mode: EdgeConstructionMode, radius_px: Optional[float]) -> str:
    if mode == EdgeConstructionMode.DISTANCE:
        return f"{prefix}-dist{_float_token(radius_px)}"
    if mode == EdgeConstructionMode.SEMANTIC_EQUALITY:
        return f"{prefix}-sem"
    return f"{prefix}-knn"


def _graph_run_signature(cfg: BuildConfig, creation_cfg: CreationConfig) -> str:
    node_token = "nodes-cell" if creation_cfg.node_creation_mode == NodeCreationMode.CELL_SEGMENTATION else "nodes-patch"
    edge_tokens = [
        _edge_mode_token("pp", creation_cfg.patch_patch_edge_mode, creation_cfg.patch_patch_distance_radius_px),
        _edge_mode_token("rr", creation_cfg.region_region_edge_mode, creation_cfg.region_region_distance_radius_px),
        _edge_mode_token("cc", creation_cfg.cell_cell_edge_mode, creation_cfg.cell_cell_distance_radius_px),
    ]
    patch_token = f"patch-l{int(cfg.patch_level)}s{int(cfg.patch_size)}st{int(cfg.stride)}"
    k_token = f"k-p{int(cfg.k_patch)}r{int(cfg.k_region)}"

    if creation_cfg.node_creation_mode == NodeCreationMode.CELL_SEGMENTATION:
        seg_level = "auto" if creation_cfg.cell_segmentation_level is None else str(int(creation_cfg.cell_segmentation_level))
        cell_token = f"cell-l{seg_level}t{_float_token(float(creation_cfg.cell_segmentation_tile_size) if creation_cfg.cell_segmentation_tile_size is not None else None)}o{int(creation_cfg.cell_segmentation_tile_overlap)}"
    else:
        cell_token = "cell-off"

    return "__".join([
        f"pl-{cfg.pipeline}",
        node_token,
        *edge_tokens,
        patch_token,
        k_token,
        cell_token,
    ])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the graph build entrypoint."""
    parser = argparse.ArgumentParser(description="Build graphs from a YAML configuration file")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file for graph generation",
    )
    return parser.parse_args()


class CudaAmpFeatureModel(nn.Module):
    def __init__(self, base_model: nn.Module) -> None:
        """Wrap a base model to run in CUDA half-precision with autocast."""
        super().__init__()
        self.base_model = base_model.eval().cuda().half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the wrapped model on the input under CUDA autocast."""
        x = x.to(device="cuda", dtype=torch.float16, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = self.base_model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.float()


def _embedding_tag_for_model(model_name: str) -> str:
    if model_name == "prov_gigapath":
        return "pov-gigapath"
    return model_name


def _config_output_dir_name(config_path: Path) -> str:
    stem = config_path.stem.strip()
    return stem or "graph_build_config"


def _resolve_slide_path(raw_path: Path, slide_name: str) -> Path:
    supported_exts = {".mrxs", ".tif", ".tiff"}
    slide_path = Path(slide_name)
    ext = slide_path.suffix.lower()

    base_variants = [slide_name]
    if ext and ext not in supported_exts:
        base_variants.append(slide_path.stem)

    seen = set()
    base_variants = [b for b in base_variants if not (b in seen or seen.add(b))]

    candidates = []
    for base in base_variants:
        base_ext = Path(base).suffix.lower()
        if base_ext in supported_exts:
            candidates.append(raw_path / base)
        else:
            candidates.extend([
                raw_path / f"{base}.mrxs",
                raw_path / f"{base}.tif",
                raw_path / f"{base}.tiff",
            ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for base in base_variants:
        recursive = list(raw_path.rglob(f"{base}.*"))
        recursive = [p for p in recursive if p.suffix.lower() in supported_exts]
        if recursive:
            return recursive[0]

    raise FileNotFoundError(f"No slide file found for '{slide_name}' in {raw_path}")


def _cache_name_candidates(slide_name: str) -> list[str]:
    token = Path(str(slide_name).strip()).name
    if token == "":
        return []

    candidates = [token]
    stem = Path(token).stem
    if stem and stem != token:
        candidates.append(stem)

    seen = set()
    return [c for c in candidates if not (c in seen or seen.add(c))]


def _resolve_node_feature_model(model_name: str) -> Tuple[Optional[Any], Dict[str, Any]]:
    if model_name == "uni2":
        logging.info("Selected model: uni2")
        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs).eval()
        return model, _model_cfg_overrides(model_name)

    if model_name == "prov_gigapath":
        logging.info("Selected model: prov_gigapath")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA ist fuer prov_gigapath erforderlich (wie im Notebook-Workflow).")

        base_model = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath",
            pretrained=True,
            num_classes=0,
        ).eval()
        model = CudaAmpFeatureModel(base_model)
        torch.cuda.empty_cache()
        return model, _model_cfg_overrides(model_name)

    if model_name == "resnet50":
        logging.info("Selected model: resnet50 (ImageNet-pretrained)")
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        model = resnet.eval()
        return model, _model_cfg_overrides(model_name)

    raise ValueError(f"Unsupported model: {model_name}")


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_with_extends(config_path: Path, visited: Optional[Set[Path]] = None) -> Dict[str, Any]:
    if visited is None:
        visited = set()

    resolved = config_path.expanduser().resolve()
    if resolved in visited:
        chain = " -> ".join(str(p) for p in list(visited) + [resolved])
        raise ValueError(f"Circular config extends detected: {chain}")

    visited.add(resolved)
    with open(resolved, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a mapping at root: {resolved}")

    extends_value = raw.get("extends")
    merged: Dict[str, Any] = {}

    if extends_value is not None:
        extends_list = extends_value if isinstance(extends_value, list) else [extends_value]
        for item in extends_list:
            parent_path = Path(str(item))
            if not parent_path.is_absolute():
                parent_path = resolved.parent / parent_path
            parent_cfg = _load_yaml_with_extends(parent_path, visited=visited)
            merged = _deep_merge_dicts(merged, parent_cfg)

    current_cfg = {k: v for k, v in raw.items() if k != "extends"}
    merged = _deep_merge_dicts(merged, current_cfg)
    visited.remove(resolved)
    return merged


def _load_config(config_path: Path) -> Tuple[Dict[str, Any], BuildConfig, CreationConfig, Set[str]]:
    raw = _load_yaml_with_extends(config_path)

    payload = raw.get("graph_build", raw)
    if not isinstance(payload, dict):
        raise ValueError("Config section 'graph_build' must be a mapping")

    build_cfg_ref = payload.get("build_config")
    pipeline_payload: Dict[str, Any] = {}
    if build_cfg_ref is not None:
        build_cfg_path = Path(str(build_cfg_ref))
        if not build_cfg_path.is_absolute():
            build_cfg_path = config_path.parent / build_cfg_path
        build_raw = _load_yaml_with_extends(build_cfg_path)
        pipeline_payload = build_raw.get("graph_build", build_raw)
        if not isinstance(pipeline_payload, dict):
            raise ValueError(f"Build config must contain a mapping at root or in 'graph_build': {build_cfg_path}")

    merged_payload = _deep_merge_dicts(pipeline_payload, payload)

    runtime_defaults: Dict[str, Any] = {
        "dataset": "gsdb",
        "model": None,
        "build_config": None,
        "csv": None,
        "csv_path": None,
        "slide_dir": None,
        "slide_name_column": "slide_name",
        "seed": 42,
        "overwrite": False,
        "visualize_each_graph": False,
        "quiet_external_loggers": True,
        "worker_start_method": "spawn",
        "worker_queue_maxsize": 1,
        "worker_result_poll_sec": 20,
        "worker_max_restarts_per_slide": 2,
        "graph_retry_attempts": 2,
    }

    runtime_cfg = dict(runtime_defaults)
    for key in runtime_defaults:
        if key in merged_payload:
            runtime_cfg[key] = merged_payload[key]

    if runtime_cfg["model"] is None:
        raise ValueError(
            "Config must define 'model' in root or graph_build section. "
            "If you pass a pipeline-only config from configs/graph_build, use a runtime config from configs/graph_runtime instead."
        )

    dataset = str(runtime_cfg["dataset"]).lower()
    if dataset not in {"gsdb", "camelyon"}:
        raise ValueError("dataset must be one of: gsdb, camelyon")
    runtime_cfg["dataset"] = dataset

    if runtime_cfg["csv"] is not None and runtime_cfg["csv_path"] is not None:
        raise ValueError("Use either csv or csv_path in config, not both")

    runtime_cfg["worker_start_method"] = str(runtime_cfg["worker_start_method"]).lower()
    if runtime_cfg["worker_start_method"] not in {"spawn", "fork", "forkserver"}:
        raise ValueError("worker_start_method must be one of: spawn, fork, forkserver")

    runtime_cfg["worker_queue_maxsize"] = int(runtime_cfg["worker_queue_maxsize"])
    runtime_cfg["worker_result_poll_sec"] = int(runtime_cfg["worker_result_poll_sec"])
    runtime_cfg["worker_max_restarts_per_slide"] = int(runtime_cfg["worker_max_restarts_per_slide"])
    runtime_cfg["graph_retry_attempts"] = int(runtime_cfg["graph_retry_attempts"])
    runtime_cfg["visualize_each_graph"] = bool(runtime_cfg["visualize_each_graph"])

    if runtime_cfg["worker_queue_maxsize"] < 1:
        raise ValueError("worker_queue_maxsize must be >= 1")
    if runtime_cfg["worker_result_poll_sec"] < 1:
        raise ValueError("worker_result_poll_sec must be >= 1")
    if runtime_cfg["worker_max_restarts_per_slide"] < 0:
        raise ValueError("worker_max_restarts_per_slide must be >= 0")
    if runtime_cfg["graph_retry_attempts"] < 1:
        raise ValueError("graph_retry_attempts must be >= 1")

    build_section = merged_payload.get("build", {})
    if not isinstance(build_section, dict):
        raise ValueError("build section must be a mapping")

    creation_section = merged_payload.get("creation", {})
    if not isinstance(creation_section, dict):
        raise ValueError("creation section must be a mapping")

    build_cfg = BuildConfig(**build_section)
    creation_cfg = CreationConfig(**creation_section)
    explicit_build_fields = set(build_section.keys())

    return runtime_cfg, build_cfg, creation_cfg, explicit_build_fields


def build_graphs_from_csv(
    csv_path: Path,
    raw_path: Path,
    output_dir: Path,
    config_path: Path,
    config_output_dir_name: str,
    model_name: str,
    build_cfg: BuildConfig,
    explicit_build_fields: Set[str],
    overwrite: bool = False,
    seed: int = 42,
    creation_cfg: Optional[CreationConfig] = None,
    dataset_name: str = "gsdb",
    slide_name_column: str = "slide_name",
    max_worker_restarts_per_slide: int = 2,
    worker_result_poll_sec: int = 20,
    graph_retry_attempts: int = 2,
    worker_queue_maxsize: int = 1,
    worker_start_method: str = "spawn",
    visualize_each_graph: bool = False,
    log_file: Optional[Path] = None,
) -> None:
    """Build and save graphs for every slide listed in a CSV, using a worker process."""
    df = pd.read_csv(csv_path)
    if slide_name_column not in df.columns:
        available_columns = ", ".join(df.columns.astype(str).tolist())
        raise KeyError(
            f"CSV '{csv_path}' does not contain slide name column '{slide_name_column}'. "
            f"Available columns: {available_columns}"
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # The parent process itself never runs GPU inference (that happens exclusively in the
    # spawned worker(s), which apply the same flags independently -- see
    # _graph_build_worker_loop), but setting these here too keeps the parent's own torch state
    # consistent and covers any future parent-side model usage.

    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _effective_build_cfg(build_cfg, explicit_build_fields, model_name)

    embedding_tag = _embedding_tag_for_model(model_name)

    rows = df.to_dict("records")
    processed = 0
    skipped = 0
    missing_slides = 0
    existing_graphs = 0
    build_failures = 0
    no_patch_skips = 0
    annotation_qa_skips = 0
    invalid_slide_rows = 0
    progress_desc = f"{csv_path.stem}:{model_name}"
    total_worker_crashes = 0
    run_state_path = output_dir / f"{csv_path.stem}__{model_name}__run_state.json"
    heartbeat_path = output_dir / f"{csv_path.stem}__{model_name}__heartbeat.json"
    run_started_at = datetime.now().isoformat(timespec="seconds")

    def _persist_state(last_event: str, slide_name: Optional[str] = None, error: Optional[str] = None) -> None:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_started_at": run_started_at,
            "csv_path": str(csv_path),
            "model": model_name,
            "dataset": str(dataset_name),
            "output_dir": str(output_dir),
            "processed": processed,
            "skipped": skipped,
            "existing_graphs": existing_graphs,
            "missing_slides": missing_slides,
            "build_failures": build_failures,
            "no_patch_skips": no_patch_skips,
            "annotation_qa_skips": annotation_qa_skips,
            "invalid_slide_rows": invalid_slide_rows,
            "worker_crashes": total_worker_crashes,
            "last_event": last_event,
            "last_slide": slide_name,
        }
        if error is not None:
            payload["last_error"] = error
        _write_json_atomic(run_state_path, payload)

    def _write_heartbeat(status: str, slide_name: Optional[str] = None) -> None:
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "status": status,
            "csv_path": str(csv_path),
            "slide": slide_name,
            "processed": processed,
            "skipped": skipped,
            "invalid_slide_rows": invalid_slide_rows,
        }
        _write_json_atomic(heartbeat_path, payload)

    # One persistent audit-trail CSV PER OUTPUT DIRECTORY (not per run/csv_path, unlike
    # run_state/heartbeat above) -- appended to incrementally, one row per skipped slide, as
    # soon as the skip happens, rather than collected in memory and written once at the end.
    # Root-caused the need for this 2026-08-12 (AnnotationQualitySkipError/
    # assess_annotation_quality): a full-corpus build already crashed once (see
    # build_graph_dynamical_prov_gigapath_patch_encoder_crash.log in this exact output_dir) --
    # an in-memory list written only at successful completion would have lost the entire
    # annotation-QA audit trail on that same kind of crash. Same directory across multiple CSVs
    # processed against this output_dir (e.g. separate train/val/test runs), so it accumulates
    # into one place rather than one file per run.
    annotation_qa_csv_path = output_dir / "annotation_qa_skipped.csv"
    _ANNOTATION_QA_CSV_FIELDS = [
        "timestamp", "slide_name", "reason", "n_annotations", "n_patches",
        "match_rate_normal", "match_rate_swapped", "source_csv",
    ]

    def _append_annotation_qa_row(slide_name: str, qa: Dict[str, Any]) -> None:
        write_header = not annotation_qa_csv_path.exists()
        with open(annotation_qa_csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_ANNOTATION_QA_CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "slide_name": slide_name,
                "reason": qa.get("reason"),
                "n_annotations": qa.get("n_annotations"),
                "n_patches": qa.get("n_patches"),
                "match_rate_normal": qa.get("match_rate_normal"),
                "match_rate_swapped": qa.get("match_rate_swapped"),
                "source_csv": csv_path.stem,
            })

    ctx = mp.get_context(worker_start_method)

    def _start_worker() -> Tuple[mp.Process, Any, Any]:
        task_queue = ctx.Queue(maxsize=worker_queue_maxsize)
        result_queue = ctx.Queue(maxsize=worker_queue_maxsize)
        worker = ctx.Process(
            target=_graph_build_worker_loop,
            args=(
                task_queue,
                result_queue,
                str(raw_path),
                str(output_dir),
                model_name,
                asdict(cfg),
                asdict(creation_cfg) if creation_cfg is not None else None,
                str(dataset_name),
                overwrite,
                str(config_path),
                str(config_output_dir_name),
                graph_retry_attempts,
                visualize_each_graph,
                csv_path.stem,
                seed,
                str(log_file) if log_file is not None else None,
            ),
            daemon=False,
        )
        worker.start()
        logging.info("Started graph worker (pid=%s)", worker.pid)
        return worker, task_queue, result_queue

    def _stop_worker(worker: Optional[mp.Process], task_queue: Optional[Any]) -> None:
        if worker is None:
            return
        try:
            if task_queue is not None and worker.is_alive():
                task_queue.put(None)
        except Exception:
            pass
        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)

    first_graph_visualized = False
    _persist_state(last_event="started")
    _write_heartbeat(status="running")
    worker: Optional[mp.Process] = None
    task_queue: Optional[Any] = None
    result_queue: Optional[Any] = None

    try:
        with tqdm(rows, total=len(rows), desc=progress_desc, unit="slide") as slide_bar:
            for row in slide_bar:
                raw_slide_name = row.get(slide_name_column, None)
                if pd.isna(raw_slide_name) or str(raw_slide_name).strip() == "":
                    invalid_slide_rows += 1
                    skipped += 1
                    logging.warning(
                        "Skipping CSV row with missing/empty slide name in column '%s'.",
                        slide_name_column,
                    )
                    slide_bar.set_postfix({"ok": processed, "skip": skipped, "invalid": invalid_slide_rows})
                    _persist_state(last_event="invalid_slide_name", error=f"missing value in column '{slide_name_column}'")
                    continue

                slide_name = str(raw_slide_name).strip()
                _write_heartbeat(status="processing", slide_name=slide_name)
                _persist_state(last_event="slide_started", slide_name=slide_name)

                cache_name_candidates = _cache_name_candidates(slide_name)
                candidate_output_paths = [
                    output_dir / f"{name}_{cfg.pipeline}_{embedding_tag}.pt"
                    for name in cache_name_candidates
                ]
                existing_output_path = next((path for path in candidate_output_paths if path.exists()), None)
                if existing_output_path is not None and not overwrite:
                    existing_graphs += 1
                    skipped += 1
                    try:
                        existing_size_gib = _bytes_to_gib(existing_output_path.stat().st_size)
                        logging.info(
                            "Skipping existing graph for %s -> %s (size=%.2fGiB)",
                            slide_name,
                            existing_output_path,
                            existing_size_gib,
                        )
                    except Exception:
                        logging.info("Skipping existing graph for %s -> %s", slide_name, existing_output_path)
                    slide_bar.set_postfix({"ok": processed, "skip": skipped})
                    _persist_state(last_event="skipped_existing", slide_name=slide_name)
                    continue

                restart_attempt = 0
                result: Optional[Dict[str, Any]] = None

                if worker is None:
                    worker, task_queue, result_queue = _start_worker()

                while result is None:
                    if _SHUTDOWN_REQUESTED:
                        logging.warning("Shutdown requested — stopping graph build.")
                        _stop_worker(worker, task_queue)
                        return

                    if not worker.is_alive():
                        total_worker_crashes += 1
                        _persist_state(last_event="worker_crash", slide_name=slide_name, error="worker not alive before dispatch")
                        _stop_worker(worker, task_queue)
                        worker, task_queue, result_queue = _start_worker()

                    if task_queue is None or result_queue is None:
                        worker, task_queue, result_queue = _start_worker()

                    task_queue.put({"slide_name": slide_name})

                    while True:
                        if _SHUTDOWN_REQUESTED:
                            logging.warning("Shutdown requested — stopping graph build.")
                            _stop_worker(worker, task_queue)
                            return

                        try:
                            result = result_queue.get(timeout=worker_result_poll_sec)
                            break
                        except queue.Empty:
                            if worker.is_alive():
                                _write_heartbeat(status="processing", slide_name=slide_name)
                                continue

                            total_worker_crashes += 1
                            restart_attempt += 1
                            _persist_state(
                                last_event="worker_crash",
                                slide_name=slide_name,
                                error=f"worker died while processing slide (attempt={restart_attempt})",
                            )
                            logging.error(
                                "Worker crashed while processing %s (attempt %d/%d).",
                                slide_name,
                                restart_attempt,
                                max_worker_restarts_per_slide,
                            )
                            _stop_worker(worker, task_queue)

                            if restart_attempt > max_worker_restarts_per_slide:
                                skipped += 1
                                build_failures += 1
                                result = {
                                    "status": "build_failed",
                                    "slide_name": slide_name,
                                    "elapsed": 0.0,
                                    "error": "worker_crash_limit_exceeded",
                                }
                                break

                            worker, task_queue, result_queue = _start_worker()
                            break

                    if result is not None:
                        break

                status = str(result.get("status"))
                elapsed = float(result.get("elapsed", 0.0))

                if status == "processed":
                    processed += 1
                    if not first_graph_visualized and not visualize_each_graph:
                        try:
                            output_path = Path(str(result["output_path"]))
                            graph_obj = torch.load(output_path, map_location="cpu", weights_only=False)
                            graph_slide_path = Path(str(result["resolved_slide_path"]))
                            if _save_first_graph_artifacts(
                                graph_obj=graph_obj,
                                graph_slide_path=graph_slide_path,
                                graph_slide_name=slide_name,
                                output_dir=output_dir,
                                cfg=cfg,
                                csv_stem=csv_path.stem,
                            ):
                                first_graph_visualized = True
                        except Exception as exc:
                            logging.warning("Could not visualize first built graph for %s: %s", slide_name, exc)

                    slide_bar.set_postfix({"ok": processed, "skip": skipped, "last_s": f"{elapsed:.1f}"})
                    _persist_state(last_event="slide_completed", slide_name=slide_name)

                elif status == "skipped_existing":
                    skipped += 1
                    existing_graphs += 1
                    slide_bar.set_postfix({"ok": processed, "skip": skipped, "last_s": f"{elapsed:.1f}"})
                    _persist_state(last_event="skipped_existing", slide_name=slide_name)

                elif status == "missing_slide":
                    skipped += 1
                    missing_slides += 1
                    logging.warning("Skipping slide %s because slide file is missing: %s", slide_name, result.get("error"))
                    slide_bar.set_postfix({"ok": processed, "skip": skipped, "missing": missing_slides, "last_s": f"{elapsed:.1f}"})
                    _persist_state(last_event="missing_slide", slide_name=slide_name, error=str(result.get("error")))

                elif status == "skipped_no_annotations":
                    skipped += 1
                    logging.info("Skipping slide %s because no annotation polygons are available.", slide_name)
                    slide_bar.set_postfix({"ok": processed, "skip": skipped, "last_s": f"{elapsed:.1f}"})
                    _persist_state(last_event="skipped_no_annotations", slide_name=slide_name, error=str(result.get("error")))

                elif status == "skipped_no_patches":
                    skipped += 1
                    no_patch_skips += 1
                    logging.warning(
                        "Skipping slide %s because no tissue patches survived filtering: %s",
                        slide_name,
                        result.get("error"),
                    )
                    slide_bar.set_postfix({"ok": processed, "skip": skipped, "no_patch": no_patch_skips, "last_s": f"{elapsed:.1f}"})
                    _persist_state(last_event="skipped_no_patches", slide_name=slide_name, error=str(result.get("error")))

                elif status == "skipped_annotation_qa":
                    skipped += 1
                    annotation_qa_skips += 1
                    qa = result.get("annotation_qa") or {}
                    logging.warning(
                        "Skipping slide %s due to annotation QA policy (reason=%s): %s",
                        slide_name, qa.get("reason"), result.get("error"),
                    )
                    _append_annotation_qa_row(slide_name, qa)
                    slide_bar.set_postfix({"ok": processed, "skip": skipped, "ann_qa": annotation_qa_skips, "last_s": f"{elapsed:.1f}"})
                    _persist_state(last_event="skipped_annotation_qa", slide_name=slide_name, error=str(result.get("error")))

                else:
                    skipped += 1
                    build_failures += 1
                    logging.warning("Skipping slide %s due to graph build error: %s", slide_name, result.get("error"))
                    if result.get("traceback"):
                        logging.debug("Graph build traceback for %s:\n%s", slide_name, result.get("traceback"))
                    slide_bar.set_postfix({"ok": processed, "skip": skipped, "last_s": f"{elapsed:.1f}"})
                    _persist_state(last_event="build_failed", slide_name=slide_name, error=str(result.get("error")))

                _write_heartbeat(status="running", slide_name=slide_name)
    finally:
        _stop_worker(worker, task_queue)

    logging.info(
        "Graph build summary for %s: total=%d, processed=%d, skipped=%d, invalid_rows=%d, existing=%d, missing_slides=%d, no_patch_skips=%d, annotation_qa_skips=%d, build_failures=%d, worker_crashes=%d",
        csv_path,
        len(rows),
        processed,
        skipped,
        invalid_slide_rows,
        existing_graphs,
        missing_slides,
        no_patch_skips,
        annotation_qa_skips,
        build_failures,
        total_worker_crashes,
    )
    if annotation_qa_skips:
        logging.info(
            "%d slide(s) skipped by annotation QA policy this run -- see %s for the full "
            "audit trail (reason/match rates per slide). Set CreationConfig."
            "build_slides_without_annotations / build_slides_with_mismatched_annotations to "
            "build them anyway.",
            annotation_qa_skips, annotation_qa_csv_path,
        )
    _persist_state(last_event="completed")
    _write_heartbeat(status="completed")


def _write_graph_statistics_csv(output_dir: Path) -> None:
    csv_output = output_dir / "statistics.csv"
    
    graph_files = sorted(output_dir.glob("*.pt"))
    if not graph_files:
        logging.info("No graph files found in %s, skipping statistics generation", output_dir)
        return
    
    logging.info("Generating statistics for %d graphs", len(graph_files))
    rows = []
    failed_count = 0
    
    for graph_path in graph_files:
        slide_name = graph_path.stem
        try:
            graph_data = torch.load(graph_path, map_location="cpu", weights_only=False)
            summary = graph_structure_summary(graph_data)
            
            row = {"slide_name": slide_name}
            
            node_counts = summary.get("node_counts", {})
            for node_type in ["patch", "region", "cell", "slide"]:
                row[f"n_{node_type}"] = node_counts.get(node_type, 0)
            
            edge_counts = summary.get("edge_counts", {})
            for edge_type, count in edge_counts.items():
                clean_name = edge_type.replace("->", "_").replace(">", "")
                row[f"e_{clean_name}"] = count
            
            row["patch_coverage"] = summary.get("patch_assignment_coverage")
            row["cell_coverage"] = summary.get("cell_assignment_coverage")
            
            if hasattr(graph_data, "y"):
                if torch.is_tensor(graph_data.y):
                    row["label"] = int(round(graph_data.y.item())) if graph_data.y.numel() == 1 else None
            
            rows.append(row)
        except Exception as e:
            logging.warning("Failed to extract stats from %s: %s", slide_name, e)
            failed_count += 1
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_output, index=False)
        logging.info("Graph statistics saved to %s (%d graphs, %d failed)", csv_output, len(rows), failed_count)
    else:
        logging.warning("No graph statistics collected, skipping CSV write")


def main() -> None:
    """Entrypoint: load config and build graphs for the configured dataset/CSV(s)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    runtime_cfg, build_cfg, creation_cfg, explicit_build_fields = _load_config(config_path)

    # Process-global, must happen once before any patch processing -- see BuildConfig's
    # cv2_num_threads/torch_num_threads docstring for why these are worth pinning explicitly
    # once feature_prep_workers>1 (avoids cv2/torch's own internal thread fanout stacking with
    # our thread pool, especially relevant if running more than one build process at a time).
    # The spawned graph worker applies the same settings to itself (see _pin_thread_counts).
    _pin_thread_counts(build_cfg)

    if bool(runtime_cfg.get("quiet_external_loggers", True)):
        _quiet_external_loggers()

    paths = get_project_paths()
    output_dir_name = _config_output_dir_name(config_path)
    output_dir = paths["GRAPH_DIR"] / str(runtime_cfg["dataset"]) / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    operation_name = f"build_graph_{build_cfg.pipeline}_{runtime_cfg['model']}_{creation_cfg.node_creation_mode.value}"
    log_file = setup_logging(output_dir, operation=operation_name)
    _configure_runtime_diagnostics(output_dir=output_dir, operation_name=operation_name)
    logging.info("Output directory (config-based): %s", output_dir)

    slide_dir = Path(str(runtime_cfg["slide_dir"])) if runtime_cfg["slide_dir"] else paths["RAW_DIR"] / "slides"
    logging.info("Using slide directory: %s", slide_dir)
    if creation_cfg.annotations_dir:
        logging.info("Using custom annotations directory: %s", creation_cfg.annotations_dir)
    if creation_cfg.masks_dir:
        logging.info("Using custom masks directory: %s", creation_cfg.masks_dir)

    direct_csv_path = Path(str(runtime_cfg["csv_path"])).expanduser().resolve() if runtime_cfg["csv_path"] else None
    target_csv = str(runtime_cfg["csv"]) if runtime_cfg["csv"] else None

    if direct_csv_path is not None:
        if not direct_csv_path.exists() or not direct_csv_path.is_file():
            raise FileNotFoundError(f"Direct CSV path not found: {direct_csv_path}")

        logging.info("Using direct CSV path mode: %s", direct_csv_path)
        build_cfg = _resolve_target_patch_size_um(
            build_cfg, csv_path=direct_csv_path, raw_path=slide_dir,
            slide_name_column=str(runtime_cfg["slide_name_column"]),
        )
        build_graphs_from_csv(
            csv_path=direct_csv_path,
            raw_path=slide_dir,
            output_dir=output_dir,
            config_path=config_path,
            config_output_dir_name=output_dir_name,
            model_name=str(runtime_cfg["model"]),
            build_cfg=build_cfg,
            explicit_build_fields=explicit_build_fields,
            overwrite=bool(runtime_cfg["overwrite"]),
            seed=int(runtime_cfg["seed"]),
            creation_cfg=creation_cfg,
            dataset_name=str(runtime_cfg["dataset"]),
            slide_name_column=str(runtime_cfg["slide_name_column"]),
            max_worker_restarts_per_slide=int(runtime_cfg["worker_max_restarts_per_slide"]),
            worker_result_poll_sec=int(runtime_cfg["worker_result_poll_sec"]),
            graph_retry_attempts=int(runtime_cfg["graph_retry_attempts"]),
            worker_queue_maxsize=int(runtime_cfg["worker_queue_maxsize"]),
            worker_start_method=str(runtime_cfg["worker_start_method"]),
            visualize_each_graph=bool(runtime_cfg["visualize_each_graph"]),
            log_file=log_file,
        )
        return

    splits_root = Path(paths["SPLITS_DIR"])
    preferred_seed_path = splits_root / str(runtime_cfg["dataset"]) / f"seed_{int(runtime_cfg['seed'])}"
    legacy_seed_path = splits_root / f"seed_{int(runtime_cfg['seed'])}"
    if preferred_seed_path.exists():
        seed_path = preferred_seed_path
    elif runtime_cfg["dataset"] == "gsdb" and legacy_seed_path.exists():
        seed_path = legacy_seed_path
    else:
        raise FileNotFoundError(
            f"Seed split directory not found for dataset='{runtime_cfg['dataset']}', seed={runtime_cfg['seed']}. "
            f"Checked: {preferred_seed_path}"
        )

    list_of_csvs = sorted([x for x in os.listdir(seed_path) if x.lower().endswith('.csv')])

    if target_csv in list_of_csvs:
        csv_path = seed_path / target_csv
        build_cfg = _resolve_target_patch_size_um(
            build_cfg, csv_path=csv_path, raw_path=slide_dir,
            slide_name_column=str(runtime_cfg["slide_name_column"]),
        )
        build_graphs_from_csv(
            csv_path=csv_path,
            raw_path=slide_dir,
            output_dir=output_dir,
            config_path=config_path,
            config_output_dir_name=output_dir_name,
            model_name=str(runtime_cfg["model"]),
            build_cfg=build_cfg,
            explicit_build_fields=explicit_build_fields,
            overwrite=bool(runtime_cfg["overwrite"]),
            seed=int(runtime_cfg["seed"]),
            creation_cfg=creation_cfg,
            dataset_name=str(runtime_cfg["dataset"]),
            slide_name_column=str(runtime_cfg["slide_name_column"]),
            max_worker_restarts_per_slide=int(runtime_cfg["worker_max_restarts_per_slide"]),
            worker_result_poll_sec=int(runtime_cfg["worker_result_poll_sec"]),
            graph_retry_attempts=int(runtime_cfg["graph_retry_attempts"]),
            worker_queue_maxsize=int(runtime_cfg["worker_queue_maxsize"]),
            worker_start_method=str(runtime_cfg["worker_start_method"]),
            visualize_each_graph=bool(runtime_cfg["visualize_each_graph"]),
            log_file=log_file,
        )

    elif target_csv is None:
        logging.info("No specific CSV provided in config, processing all CSVs in splits directory")
        # Resolved ONCE from the union of every CSV about to be processed, not per-CSV inside
        # the loop below -- see _resolve_target_patch_size_um's docstring for why doing this
        # per-CSV would be wrong (a single-scanner/center CSV would auto-detect its own
        # resolution as "the" reference instead of the corpus-wide one).
        slide_name_column = str(runtime_cfg["slide_name_column"])
        all_slide_names: List[str] = []
        for csv_file in list_of_csvs:
            all_slide_names.extend(pd.read_csv(seed_path / csv_file)[slide_name_column].astype(str).tolist())
        build_cfg = _resolve_target_patch_size_um_from_names(build_cfg, all_slide_names, slide_dir)

        for csv_file in list_of_csvs:
            csv_path = seed_path / csv_file
            build_graphs_from_csv(
                csv_path=csv_path,
                raw_path=slide_dir,
                output_dir=output_dir,
                config_path=config_path,
                config_output_dir_name=output_dir_name,
                model_name=str(runtime_cfg["model"]),
                build_cfg=build_cfg,
                explicit_build_fields=explicit_build_fields,
                overwrite=bool(runtime_cfg["overwrite"]),
                seed=int(runtime_cfg["seed"]),
                creation_cfg=creation_cfg,
                dataset_name=str(runtime_cfg["dataset"]),
                slide_name_column=str(runtime_cfg["slide_name_column"]),
                max_worker_restarts_per_slide=int(runtime_cfg["worker_max_restarts_per_slide"]),
                worker_result_poll_sec=int(runtime_cfg["worker_result_poll_sec"]),
                graph_retry_attempts=int(runtime_cfg["graph_retry_attempts"]),
                worker_queue_maxsize=int(runtime_cfg["worker_queue_maxsize"]),
                worker_start_method=str(runtime_cfg["worker_start_method"]),
                visualize_each_graph=bool(runtime_cfg["visualize_each_graph"]),
                log_file=log_file,
            )
    else:
        raise FileNotFoundError(f"CSV '{target_csv}' not found in split directory: {seed_path}")

    logging.info("Generating statistics summary for dataset: %s", runtime_cfg["dataset"])
    _write_graph_statistics_csv(output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("build_graph.py terminated with an unhandled exception")
        raise
