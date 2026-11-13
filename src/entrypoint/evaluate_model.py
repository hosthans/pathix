import argparse
from pathlib import Path
import logging
from datetime import datetime
import random
import sys
import warnings
import numpy as np

# Cosmetic only -- see train_model.py's matching filter for the full rationale (torch_geometric's
# scatter(reduce='max'/'min') fallback warning, fires on every forward pass without the optional
# 'torch-scatter' package installed).
warnings.filterwarnings("ignore", message=".*torch-scatter.*")

import os


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


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ.setdefault('CUDA_VISIBLE_DEVICES', _autoselect_least_busy_gpu())
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.paths.paths import add_path_args, get_project_paths
from src.paths.pathconfig import PathConfigurator, prefix_arg
from src.utils.logging import setup_logging, quiet_external_loggers
from src.core.validation import Validator

from src.utils.console_colors import ColorFormatter

from src.core.holder import Holder
from src.core.meta_processor import MetaProcessor, MetaData
from src.core.factories import get_validation_threshold
from src.utils.graph_patch_utils import resolve_negative_label_source
from src.pipelines.gsdb_parity_eval import is_gsdb_parity_eval, run_gsdb_parity_evaluation


def _list_graph_dir_candidates(paths, dataset: str, embedding_type: str, pipeline_name: str = None):
    graph_root = Path(paths["GRAPH_DIR"])
    dataset_root = graph_root / str(dataset)
    candidates = []

    search_roots = [dataset_root, graph_root]
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        for child in sorted(root.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            if embedding_type and not child.name.endswith(f"_{embedding_type}"):
                continue
            if pipeline_name:
                if pipeline_name not in child.name and not child.name.startswith(f"pl-{pipeline_name}"):
                    continue
            candidates.append(child)

    seen = set()
    deduped = []
    for candidate in candidates:
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train histology classifier")
    parser.add_argument(
        "--test_split",
        required=False,
        help="Evaluation split"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Optional list of evaluation splits. If set, overrides --test_split.",
    )
    parser.add_argument(
        "--split-phase",
        dest="split_phase",
        type=str,
        default=None,
        help=(
            "Select splits from data/splits/<dataset>/seed_<seed>/<phase>/ when set (e.g. "
            "'kfold_k5_fold0'), falling back to the shared seed-level files for any split not "
            "written under this phase's subdirectory -- mirrors train_model.py's --split-phase."
        ),
    )
    parser.add_argument(
        "--seed", 
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode for reproducibility (may reduce performance)"
    )
    parser.add_argument(
        "--meta_file",
        type=str,
        default=None,
        help="Optional meta YAML file name or path to use directly instead of interactive selection",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=None,
        help=(
            "Evaluate a checkpoint directly by path, bypassing the meta-file system entirely "
            "(useful while a training run hasn't finished/written its meta yet). Requires "
            "--task/--dataset/--model_type/--architecture/--input_type/--embedding_type/"
            "--config_name (same flags as train_model.py) to describe how to build the model "
            "and resolve the config -- mutually exclusive with --meta_file."
        ),
    )
    parser.add_argument("--task", default=None, choices=['inflammation', 'tissue', 'lymph'], help="Classification task (only used with --checkpoint_path)")
    parser.add_argument("--dataset", type=str, choices=['gsdb', 'camelyon'], default=None, help="Dataset family (only used with --checkpoint_path)")
    parser.add_argument("--model_type", type=str, choices=['cnn', 'gnn', 'multimodal', 'mil'], default=None, help="Model type (only used with --checkpoint_path)")
    parser.add_argument("--architecture", type=str, default=None, help="Model architecture (only used with --checkpoint_path)")
    parser.add_argument("--feature_extractor", choices=['resnet18', 'densenet121', 'densenet169', 'convnext_large', 'swin_v2_b', 'gigapath', 'none'], default='none', help="Feature extractor backbone (only used with --checkpoint_path)")
    parser.add_argument("--input_type", type=str, choices=['image', 'embeddings', 'graph', 'multimodal', 'mil'], default=None, help="Input data type (only used with --checkpoint_path)")
    parser.add_argument("--embedding_type", type=str, choices=['none', 'pov-gigapath', 'uni2', 'resnet50'], default='none', help="Embedding type (only used with --checkpoint_path)")
    parser.add_argument("--pipeline_type", type=str, default=None, help="Graph cache pipeline identifier (only used with --checkpoint_path)")
    parser.add_argument("--pipeline", type=str, default=None, help="Alias for --pipeline_type (only used with --checkpoint_path)")
    parser.add_argument("--mil", action="store_true", help="Use Multiple Instance Learning (only used with --checkpoint_path)")
    parser.add_argument("--cached", action="store_true", help="Use cached graph data (only used with --checkpoint_path)")
    parser.add_argument("--graph_target_dir", type=Path, default=None, help="Directory containing cached .pt graphs (only used with --checkpoint_path)")
    parser.add_argument("--label_mode", type=str, choices=['binary', 'multiclass'], default=None, help="Label mode override (only used with --checkpoint_path)")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes override (only used with --checkpoint_path)")
    parser.add_argument(
        "--label_field",
        type=str,
        default=None,
        help=(
            "Graph-stored slide label field override (e.g. 'subtype', 'stage') -- see "
            "train_model.py's --label_field. Normally restored automatically from the meta file "
            "(Holder._init_metadata() saves it); only needed with --checkpoint_path, or to "
            "override what a meta file recorded."
        ),
    )
    parser.add_argument("--config_name", type=str, default=None, help="Config file name inside configs/ (only used with --checkpoint_path)")
    parser.add_argument(
        "--eval_prefix",
        type=prefix_arg,
        default=None,
        help=(
            "Optional prefix for this evaluation's output folder (<eval_prefix>_<usual name>). "
            "Default: the prefix the run was trained with (stored in its meta file), if any."
        ),
    )
    seg_group = parser.add_mutually_exclusive_group()
    seg_group.add_argument("--segmentation", dest="segmentation", action="store_true", help="Force segmentation evaluation mode")
    seg_group.add_argument("--no-segmentation", dest="segmentation", action="store_false", help="Force standard evaluation mode")
    parser.set_defaults(segmentation=None)
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help=(
            "Skip generating and saving diagnostic plots/overlays (histograms, ROC/PR curves, "
            "slide overlays) during evaluation. Metrics, thresholds, and predictions CSVs are "
            "still written -- only the plot images are skipped. Useful for quick evaluation runs "
            "where the overlay rendering (one image per test slide) dominates the runtime."
        ),
    )

    parser = add_path_args(parser)
    
    args = parser.parse_args()
    return args

def set_deterministic(seed=42):
    """Seed all RNGs and enable deterministic algorithm settings."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # warn_only: deterministic kernels are used where they exist; ops without one warn instead of aborting the run
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_meta_from_argument(meta_processor: MetaProcessor, meta_file: str | None) -> tuple[MetaData, str]:
    if meta_file:
        candidate_path = Path(meta_file)
        if candidate_path.is_file():
            return MetaData.from_yaml(candidate_path), str(candidate_path)

        candidate_path = meta_processor.training_meta_dir / meta_file
        if candidate_path.is_file():
            return MetaData.from_yaml(candidate_path), str(candidate_path)

        raise FileNotFoundError(f"Meta file not found: {meta_file}")

    meta_files = sorted(meta_processor.get_metas())
    if not meta_files:
        raise FileNotFoundError(f"No checkpoint files found in {meta_files}")

    if len(meta_files) == 1:
        selected_file = meta_files[0]
        return meta_processor.load_meta(selected_file), selected_file

    sys.stdout.write("Available checkpoint files:\n")
    for idx, f in enumerate(meta_files):
        sys.stdout.write(f"  [{idx}] {f}\n")
    while True:
        try:
            selection = int(input(f"Select checkpoint file [0-{len(meta_files)-1}]: "))
            if 0 <= selection < len(meta_files):
                selected_file = meta_files[selection]
                return meta_processor.load_meta(selected_file), selected_file
            sys.stdout.write("Invalid selection. Try again.\n")
        except Exception:
            sys.stdout.write("Invalid input. Enter a number.\n")

def _build_meta_from_checkpoint_args(args: argparse.Namespace) -> MetaData:
    required = ["task", "dataset", "model_type", "architecture", "input_type", "embedding_type"]
    missing = [name for name in required if getattr(args, name, None) is None]
    if missing:
        raise ValueError(
            f"--checkpoint_path requires the following flags to also be set: {missing} "
            "(same as train_model.py)."
        )

    meta_dict = {
        "task": args.task,
        "dataset": args.dataset,
        "model_type": args.model_type,
        "architecture": args.architecture,
        "input_type": args.input_type,
        "embedding_type": args.embedding_type,
        "feature_extractor": args.feature_extractor,
        "mil": bool(args.mil),
        "cached": bool(args.cached),
        "segmentation": bool(args.segmentation) if args.segmentation is not None else False,
        "graph_target_dir": str(args.graph_target_dir) if args.graph_target_dir else None,
        "pipeline": args.pipeline or args.pipeline_type,
        "checkpoint_path": str(args.checkpoint_path),
        "config_name": args.config_name,
        "seed": args.seed,
        "label_mode": args.label_mode,
        "num_classes": args.num_classes,
        "label_field": args.label_field,
    }
    return MetaData(meta_dict)


def main():
    """Evaluate a trained model checkpoint on one or more test splits."""
    args = parse_args()

    # Split names may be given with or without ".csv" (e.g. test_scanner2 / test_scanner2.csv);
    # normalize to the bare name so output folders stay "split_test_scanner2".
    def _strip_csv(name):
        return name[:-4] if name and str(name).endswith('.csv') else name
    args.test_split = _strip_csv(args.test_split)
    if args.splits:
        args.splits = [_strip_csv(s) for s in args.splits]

    set_deterministic(seed=args.seed)

    paths = get_project_paths(base_dir=args.base_dir)

    if args.checkpoint_path is not None:
        if args.meta_file:
            raise ValueError("--checkpoint_path and --meta_file are mutually exclusive.")
        meta = _build_meta_from_checkpoint_args(args)
        selected_file = f"<direct checkpoint, no meta file: {args.checkpoint_path}>"
    else:
        meta_processor = MetaProcessor(paths["META_DIR"])
        meta, selected_file = _load_meta_from_argument(meta_processor, args.meta_file)
    sys.stdout.write(f"Meta file selected: {selected_file}\n")

    if hasattr(meta, 'meta_dict'):
        for k, v in meta.meta_dict.items():
            setattr(args, k, v)

    if args.segmentation is None:
        args.segmentation = bool(getattr(meta, 'meta_dict', {}).get('segmentation', False))

    if bool(getattr(args, 'segmentation', False)):
        model_type = str(getattr(args, 'model_type', '')).lower()
        input_type = str(getattr(args, 'input_type', '')).lower()
        if model_type != 'gnn' or input_type != 'graph':
            raise ValueError("--segmentation requires --model_type gnn and --input_type graph")

    if getattr(args, 'dataset', None) is None:
        args.dataset = 'camelyon' if getattr(args, 'task', None) == 'lymph' else 'gsdb'

    if getattr(args, 'input_type', None) == 'graph' and bool(getattr(args, 'cached', False)):
        resolved_pipeline = (
            getattr(args, 'pipeline', None)
            or getattr(args, 'pipeline_type', None)
            or getattr(meta, 'pipeline', None)
        )
        if not resolved_pipeline:
            logging.warning(
                "Selected meta has no pipeline value. This is expected for older runs. "
                "Graph resolution will use embedding-based directory fallback."
            )

        candidate_dirs = _list_graph_dir_candidates(
            paths=paths,
            dataset=getattr(args, 'dataset', 'gsdb'),
            embedding_type=getattr(args, 'embedding_type', None),
            pipeline_name=resolved_pipeline,
        )
        logging.info(
            "Graph directory candidates for evaluation (%d, showing up to 10): %s",
            len(candidate_dirs),
            [str(p) for p in candidate_dirs[:10]],
        )

        if getattr(args, 'graph_target_dir', None):
            logging.info("Using graph_target_dir from args/meta: %s", args.graph_target_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Config name in the folder name: without it, evaluations of different configs on the same
    # task/split/architecture are indistinguishable at a glance (e.g. every camelyon subtype or
    # GSDB inflammation ablation config landed in "<task>_test_tissue_region_net_<timestamp>").
    # Accepts a bare name, a filename with .yaml, or a full path (--config_name / meta both occur).
    cfg_token = Path(str(getattr(args, 'config_name', None) or '')).stem
    if cfg_token.lower() in ('', 'none'):
        cfg_token = ''
    else:
        # A single path component is capped at 255 bytes; descriptive config names (see
        # the E3 augmentation-ablation configs) can be long, so cap the token
        # and add a short hash so two long names sharing a prefix can't collide.
        max_cfg_len = 120
        if len(cfg_token.encode('utf-8')) > max_cfg_len:
            import hashlib
            digest = hashlib.sha1(cfg_token.encode('utf-8')).hexdigest()[:8]
            cfg_token = f"{cfg_token[:max_cfg_len - 9]}_{digest}"
    dir_parts = [str(meta.task)]
    if cfg_token:
        dir_parts.append(f"cfg-{cfg_token}")
    # Marks the *training* setup only -- the evaluated split is always the full CSV below.
    if bool(getattr(meta, 'meta_dict', {}).get('he_only', False)):
        dir_parts.append("trainheonly")
    # Split phase (e.g. a CAMELYON17 center-rotation fold): restored from the meta above, so the evaluated
    # split always matches the training run; shown in the folder name as in the training/log names.
    split_phase = getattr(args, 'split_phase', None)
    if split_phase and str(split_phase).strip().lower() != "none":
        dir_parts.append(str(split_phase).strip())
        logging.info("Split phase (from meta / --split-phase): %s", split_phase)
    dir_parts += [str(args.test_split), str(meta.architecture), timestamp]
    dir_name = PathConfigurator.apply_prefix("_".join(dir_parts), PathConfigurator.get_eval_prefix(args))
    output_dir = paths["RESULTS_DIR"] / "test_results" / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir, operation="evaluation")
    quiet_external_loggers()

    holder = Holder(args=args, paths=paths)
    holder.meta_data = meta

    if isinstance(getattr(meta, 'model_config', None), dict) and meta.model_config:
        holder.config = meta.model_config
        holder.arch_config = holder.config.get('architecture', holder.arch_config)

    checkpoint_path = Path(meta.checkpoint_path) if getattr(meta, 'checkpoint_path', None) else holder.checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint referenced by meta not found: {checkpoint_path}")

    holder.load_model_from_checkpoint(checkpoint_path=checkpoint_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    logging.info(f"Model:   {type(holder.model).__name__}")

    if is_gsdb_parity_eval(holder.config, holder.meta_data.model_type, holder.meta_data.input_type, holder.meta_data.mil):
        run_gsdb_parity_evaluation(
            model=holder.model,
            task=holder._resolve_runtime_task(holder.meta_data.task),
            paths=paths,
            config=holder.config,
            output_dir=output_dir,
            splits=args.splits or ([args.test_split] if args.test_split else None),
            seed=args.seed,
            split_phase=getattr(args, 'split_phase', None),
        )
        logging.info("Evaluation completed successfully!")
        return

    eval_splits = [str(s) for s in (args.splits if args.splits else [args.test_split or 'test'])]

    # Val first (only needed for the thresholds), then every test split once, after the thresholds
    # are computed -- no dataset is built twice.
    holder.load_val_data()

    validation_dir = output_dir / "validation_thresholds"
    validation_dir.mkdir(parents=True, exist_ok=True)
    val_method = get_validation_threshold(
        input_type=holder.meta_data.input_type,
        embedding_type=holder.meta_data.embedding_type,
        mil=holder.meta_data.mil,
    )
    val_kwargs = {
        "model": holder.model,
        "val_dataset": holder.val_dataset,
        "task": holder._resolve_runtime_task(holder.meta_data.task),
        "output_dir": validation_dir,
    }
    if str(holder.meta_data.input_type) == 'graph':
        val_kwargs["negative_label_source"] = resolve_negative_label_source(holder.config)
    validation_thresholds = val_method(**val_kwargs)

    for split in eval_splits:
        args.test_split = str(split)
        holder.args.test_split = str(split)
        holder.load_test_data(test_split=str(split))
        split_output_dir = output_dir / f"split_{split}"
        split_output_dir.mkdir(parents=True, exist_ok=True)
        holder.test(output_dir=split_output_dir, validation_thresholds=validation_thresholds)

    logging.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()