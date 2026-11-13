import argparse
from pathlib import Path
import logging
import random
import sys
import warnings
import numpy as np

# Cosmetic only -- torch_geometric's scatter(reduce='max'/'min') falls back to a slower pure-
# PyTorch path without the optional 'torch-scatter' package, and warns on every single call
# (region_logit_max/patch_to_region aggregation in tissue_region_extraction/model.py,
# use_patch_evidence in robust_slide_classifier.py both hit this every forward pass -- floods
# stdout over a training run). Not installing torch-scatter here on purpose: it needs a wheel
# built against this exact torch+CUDA version and isn't a hard requirement, just a speed-up for
# scatter ops that aren't this pipeline's bottleneck. Filtered by message text (not blanket
# UserWarning) so unrelated warnings still surface normally.
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
from src.paths.pathconfig import prefix_arg
from src.utils.logging import setup_logging
from src.core.meta_processor import MetaProcessor, MetaData

from src.utils.console_colors import ColorFormatter

from src.core.holder import Holder
from src.core.validation import Validator
from src.core.factories import get_validation_threshold
from src.utils.graph_patch_utils import resolve_negative_label_source
from src.pipelines.gsdb_parity_eval import is_gsdb_parity_eval, run_gsdb_parity_evaluation


def _load_meta_from_argument(meta_processor: MetaProcessor, meta_file: str | None) -> tuple[MetaData, str | None]:
    if not meta_file:
        return None, None

    candidate_path = Path(meta_file)
    if candidate_path.is_file():
        return MetaData.from_yaml(candidate_path), str(candidate_path)

    candidate_path = meta_processor.training_meta_dir / meta_file
    if candidate_path.is_file():
        return MetaData.from_yaml(candidate_path), str(candidate_path)

    raise FileNotFoundError(f"Meta file not found: {meta_file}")


def _collect_cli_overrides(argv: list[str]) -> set[str]:
    option_to_dest = {
        '--task': 'task',
        '--dataset': 'dataset',
        '--model_type': 'model_type',
        '--architecture': 'architecture',
        '--feature_extractor': 'feature_extractor',
        '--input_type': 'input_type',
        '--embedding_type': 'embedding_type',
        '--pipeline_type': 'pipeline_type',
        '--pipeline': 'pipeline',
        '--graph_target_dir': 'graph_target_dir',
        '--batch-size': 'batch_size',
        '--epochs': 'epochs',
        '--learning-rate': 'learning_rate',
        '--weight_decay': 'weight_decay',
        '--meta_file': 'meta_file',
        '--config_name': 'config_name',
        '--seed': 'seed',
        '--prefix': 'prefix',
        '--eval_prefix': 'eval_prefix',
        '--label_mode': 'label_mode',
        '--num_classes': 'num_classes',
        '--label_field': 'label_field',
        '--mil': 'mil',
        '--cached': 'cached',
        '--split-phase': 'split_phase',
        '--scanner1_only': 'scanner1_only',
        '--he_only': 'he_only',
        '--deterministic': 'deterministic',
    }
    provided = set()
    for token in argv:
        if not token.startswith('--'):
            continue
        key = token.split('=', 1)[0]
        dest = option_to_dest.get(key)
        if dest:
            provided.add(dest)
    return provided

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train histology classifier")
    parser.add_argument(
        "--task",
        default=None,
        choices=['inflammation', 'tissue', 'lymph'],
        help="Classification task"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['gsdb', 'camelyon'],
        default='gsdb',
        help="Dataset family"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=['cnn', 'gnn', 'multimodal', 'mil'],
        default=None,
        help="Type of model to train"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        help="Model architecture to use",
        default=None
    )
    parser.add_argument(
        "--feature_extractor",
        choices=['resnet18', 'densenet121', 'densenet169', 'convnext_large', 'swin_v2_b', 'gigapath'],
        default='none',
        help="Determines the feature extractor backbone for Embedding models"
    )
    parser.add_argument(
        "--input_type",
        type=str,
        choices=['image', 'embeddings', 'graph', 'multimodal', 'mil'],
        default=None,
        help="Type of input data"
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        choices=['none', 'pov-gigapath', 'uni2', 'resnet50'],
        default='none',
        help="Use embeddings in addition to images"
    )
    parser.add_argument(
        "--pipeline_type",
        type=str,
        default=None,
        help="Graph cache pipeline identifier (legacy names like 'hierarchical' or run-signature names like 'pl-dynamical__...')"
    )
    parser.add_argument(
        "--mil",
        action="store_true",
        help="Use Multiple Instance Learning"
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Flag to use cached graph data if available"
    )
    parser.add_argument(
        "--split-phase",
        type=str,
        default=None,
        help=(
            "Select train/val/test splits from data/splits/<dataset>/seed_<seed>/<phase>/ when "
            "set -- e.g. 'pretrain'/'finetune' (camelyon phase exports) or 'kfold_k<K>_fold<i>' "
            "(gsdb CV folds, see create_splits.py --kfold-k). Not choices-restricted: "
            "graph_dataset.py's _resolve_split_path falls back to the shared seed-level files "
            "for any split not written under this phase's subdirectory (e.g. test/test_scanner2 "
            "for CV folds, which are deliberately not duplicated per fold)."
        ),
    )
    parser.add_argument(
        "--segmentation",
        action="store_true",
        help="Enable region-level graph segmentation training focused on tumor region detection",
    )
    parser.add_argument(
        "--scanner1_only",
        action="store_true",
        help="Flag to train only on scanner 1 data (for domain generalization experiments)"
    )
    parser.add_argument(
        "--he_only",
        action="store_true",
        help="Flag to train only on HE stained data (for domain generalization experiments)"
    )
    parser.add_argument(
        "--label_mode",
        type=str,
        choices=['binary', 'multiclass'],
        default=None,
        help="Optional override for label setup (otherwise loaded from dataset config)"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Optional override for number of classes (otherwise loaded from dataset config)"
    )
    parser.add_argument(
        "--label_field",
        type=str,
        default=None,
        help=(
            "Optional graph-stored label field name (e.g. 'status', 'subtype', 'stage', "
            "'tissue_type') to use as the slide target instead of the split CSV's default "
            "tumor_status/stage/inflammation_status/tissue_type column. Requires graphs built "
            "with a CreationConfig.label_fields-enabled pipeline config (see "
            "configs/graph_build/*_labels.yaml) that produced data['slide'][f'{label_field}_y']. "
            "Combine with --label_mode multiclass --num_classes <vocab size>. Omit for the "
            "existing CSV-driven binary/multiclass training path (unaffected either way)."
        ),
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default=None,
        help="Alias for --pipeline_type. Accepts legacy names and run-signature identifiers.",
    )
    parser.add_argument(
        "--graph_target_dir",
        type=Path,
        default=None,
        help="Optional directory containing cached .pt graphs. If set, this directory is always used.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size from config"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs to train"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override learning rate from config"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        help="Override weight decay from config"
    )
    parser.add_argument(
        "--meta_file",
        type=str,
        default=None,
        help="Optional meta YAML file name or path to resume/fine-tune from an existing run",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Optional config file name or path inside configs/ to use for this run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--prefix",
        type=prefix_arg,
        default=None,
        help=(
            "Optional run prefix: meta file, checkpoints, training history, logs and evaluation "
            "outputs are saved as <prefix>_<usual name>. Letters, digits, '.', '_' and '-' are kept; "
            "other characters become '-'."
        ),
    )
    parser.add_argument(
        "--eval_prefix",
        type=prefix_arg,
        default=None,
        help="Optional separate prefix for the post-training evaluation outputs (default: --prefix).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode for reproducibility (may reduce performance)"
    )
    parser.add_argument(
        "--strict_deterministic",
        action="store_true",
        help=(
            "Implies --deterministic, but operations without a deterministic CUDA kernel raise an error "
            "instead of only logging a warning (use to check that a run is bitwise reproducible)."
        ),
    )
    parser = add_path_args(parser)

    args = parser.parse_args()
    args._cli_overrides = _collect_cli_overrides(sys.argv[1:])
    if args.strict_deterministic:
        args.deterministic = True  # strict implies --deterministic (e.g. graph loaders then use num_workers=0)

    if args.meta_file is None:
        missing = [name for name in ('task', 'model_type', 'architecture', 'input_type') if getattr(args, name, None) is None]
        if missing:
            parser.error(f"Missing required arguments without --meta_file: {', '.join(missing)}")

    if args.dataset == 'gsdb' and args.task is not None and args.task not in {'inflammation', 'tissue'}:
        parser.error("For --dataset gsdb, --task must be one of: inflammation, tissue")
    if args.dataset == 'camelyon' and args.task is not None and args.task != 'lymph':
        parser.error("For --dataset camelyon, --task must be: lymph")

    if args.label_mode is not None and args.num_classes is not None:
        if args.label_mode == 'multiclass' and args.num_classes <= 1:
            parser.error("--label_mode multiclass requires --num_classes > 1")
        if args.label_mode == 'binary' and args.num_classes != 1:
            logging.warning("Binary mode with num_classes != 1 requested. This may lead to unexpected behavior.")

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

DEFAULT_FINETUNE_FREEZE_LAYERS = [
    "input_proj", "input_norm", "region_convs", "region_norms",
    "patch_in", "patch_norm", "patch_convs",
]


def _apply_meta_for_finetuning(args, meta_processor: MetaProcessor) -> None:
    meta, selected_meta_file = _load_meta_from_argument(meta_processor, getattr(args, 'meta_file', None))
    args.fine_tune = False
    if meta is None:
        return

    logging.info(f"Loaded meta file for fine-tuning/training start: {selected_meta_file}")

    cli_override_keys = getattr(args, '_cli_overrides', set())
    cli_override_values = {
        key: getattr(args, key)
        for key in cli_override_keys
        if hasattr(args, key)
    }

    if hasattr(meta, 'meta_dict'):
        for key, value in meta.meta_dict.items():
            setattr(args, key, value)

    for key, value in cli_override_values.items():
        setattr(args, key, value)

    if getattr(args, 'config_name', None) is None:
        args.config_name = getattr(meta, 'config_name', None) or getattr(meta, 'model_config_name', None)

    args.model_config = {}

    checkpoint_path = getattr(meta, 'checkpoint_path', None)
    if checkpoint_path:
        fine_tuning_cfg = {
            'enabled': True,
            'pretrained_checkpoint': str(checkpoint_path),
            'strict_load': True,
            'freeze_layers': list(DEFAULT_FINETUNE_FREEZE_LAYERS),
        }
        meta_model_config = getattr(meta, 'model_config', None)
        if isinstance(meta_model_config, dict):
            candidate_ft = meta_model_config.get('fine_tuning', {})
            if isinstance(candidate_ft, dict):
                for key in ('strict_load', 'freeze_layers', 'trainable_layers',
                            'freeze_all', 'freeze_backbone', 'freeze_feature_extractor'):
                    value = candidate_ft.get(key)
                    if value not in (None, [], False):
                        fine_tuning_cfg[key] = value

        fine_tuning_cfg['enabled'] = True
        args.model_config['fine_tuning'] = fine_tuning_cfg
        args.fine_tune = True
        logging.info(
            "Fine-tuning activated from checkpoint %s | strict_load=%s | frozen layers: %s",
            checkpoint_path, fine_tuning_cfg.get('strict_load'), fine_tuning_cfg.get('freeze_layers'),
        )


def _evaluate_holder_on_test(holder: Holder) -> dict | None:
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint_path = holder.checkpoint_path
        if not checkpoint_path.is_file():
            logging.warning("No checkpoint found at %s; skipping post-training evaluation.", checkpoint_path)
            return None

        holder.load_model_from_checkpoint(checkpoint_path=checkpoint_path, device=device)
        if is_gsdb_parity_eval(holder.config, holder.meta_data.model_type, holder.meta_data.input_type, holder.meta_data.mil):
            return run_gsdb_parity_evaluation(
                model=holder.model,
                task=holder._resolve_runtime_task(holder.meta_data.task),
                paths=holder.paths,
                config=holder.config,
                output_dir=holder.evaluation_path / "gsdb_parity",
                seed=getattr(holder.args, 'seed', 42),
                split_phase=getattr(holder.args, 'split_phase', None),
            )

        holder.args.test_split = 'test'
        holder.load_val_data()  # val first: only needed for the thresholds

        eval_dir = holder.evaluation_path
        eval_dir.mkdir(parents=True, exist_ok=True)
        validation_dir = eval_dir / "validation_thresholds"
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

        holder.load_test_data(test_split='test')  # test after the thresholds are computed
        split_output_dir = eval_dir / "split_test"
        split_output_dir.mkdir(parents=True, exist_ok=True)
        return holder.test(output_dir=split_output_dir, validation_thresholds=validation_thresholds)
    except Exception:
        logging.exception("Post-training evaluation on the test split failed.")
        return None


def _run_training(args, paths, meta_processor: MetaProcessor) -> tuple[Holder, dict | None]:
    validator = Validator(args=args, paths=paths)
    try:
        validator.validate_args()
    except ValueError as e:
        logging.error(f"Argument validation error: {e}")
        sys.exit(1)

    holder = Holder(args=args, paths=paths)
    holder.load_model()
    holder.load_data()
    holder.train()
    logging.info("Training completed successfully!")
    logging.info(f"Storing metadata in: {paths['META_DIR']}")
    try:
        meta_processor.save_meta(holder.meta_data)
    except Exception:
        # The meta file is bookkeeping (config/hyperparameter snapshot for later --meta_file
        # reuse) -- it must never be allowed to abort the evaluation below and discard hours of
        # training for want of a small YAML write. Log and keep going; the checkpoint (already on
        # disk from holder.train()) and this evaluation are what actually matter.
        logging.exception(
            "Failed to save meta data -- continuing to evaluation anyway (checkpoint is already "
            "saved; --checkpoint_path can be used to evaluate/resume without a meta file)."
        )

    eval_metrics = _evaluate_holder_on_test(holder)
    return holder, eval_metrics


def main():
    """Train a single model on one seed's split, then evaluate it on that split's test set."""
    args = parse_args()

    if args.deterministic or args.strict_deterministic:
        set_deterministic(seed=args.seed)

    paths = get_project_paths(base_dir=args.base_dir)
    print(paths)

    meta_processor = MetaProcessor(meta_path=paths['META_DIR'])

    _apply_meta_for_finetuning(args, meta_processor)
    _run_training(args, paths, meta_processor)


if __name__ == "__main__":
    main()