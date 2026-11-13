import torch
import platform
import logging
import gc
import contextlib
from datetime import datetime
from typing import List, Optional

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch_geometric.loader import DataLoader as GeoDataLoader
import torchvision.transforms as T
from pathlib import Path
import logging
from datetime import datetime
import yaml
import numpy as np
from PIL import Image
import json
import sys
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.dataset.graph_augment import apply_graph_augmentations
from src.utils.visualization_utils import plot_training_history
from src.paths import paths
from src.paths.paths import add_path_args, get_project_paths
from src.utils.dataset_utils import mil_collate, mil_collate_no_embeddings
import src.utils.training_utils as training_utils
import src.utils.metrics_utils as metrics_utils
from src.utils.graph_patch_utils import resolve_negative_label_source
from src.core.registry_utils import register_training_pipeline
from src.paths.pathconfig import PathConfigurator

def log_training_settings(args, config, optimizer_name, momentum):
    """Logs the training run's environment, args, and config to the log."""
    logging.info("\n========== TRAINING SETTINGS ==========")
    logging.info(f"Timestamp: {datetime.now().isoformat()}")
    logging.info(f"Host: {platform.node()} | Platform: {platform.platform()}")
    logging.info(f"CUDA available: {torch.cuda.is_available()} | CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device: {torch.cuda.get_device_name(0)} | Total memory: {torch.cuda.get_device_properties(0).total_memory // (1024**2)} MB")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"Args: {vars(args)}")
    logging.info(f"Runtime seed (args.seed): {getattr(args, 'seed', None)}")
    if isinstance(config, dict) and 'seed' in config:
        logging.info(f"Config seed (config.seed): {config.get('seed')}")
    logging.info(f"Optimizer: {optimizer_name} | Momentum: {momentum}")
    logging.info(f"Config (flattened):")
    def flatten(d, parent_key='', sep='.'):
        """Flattens a nested dict into a single-level dict with dotted keys."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    flat_config = flatten(config)
    for k, v in flat_config.items():
        logging.info(f"  {k}: {v}")
    logging.info("=======================================\n")


def _is_multiclass_setup(args) -> bool:
    label_mode = str(getattr(args, 'label_mode', 'binary')).strip().lower()
    num_classes = int(getattr(args, 'num_classes', 1) or 1)
    return label_mode == 'multiclass' or num_classes > 1


def _format_binary_metric_block(metrics: dict, monitor_metric_name: Optional[str] = None) -> str:
    def _tag(*keys: str) -> str:
        return "  <-- MONITOR METRIC" if monitor_metric_name in keys else ""

    block = (
        f"  Loss: {metrics.get('loss', 0.0):.4f}{_tag('loss', 'region_loss')}\n"
        f"  Accuracy: {metrics.get('accuracy', 0.0):.2f}%\n"
        f"  Sensitivity: {metrics.get('sensitivity', 0.0):.2f}%\n"
        f"  Specificity: {metrics.get('specificity', 0.0):.2f}%\n"
        f"  F1: {metrics.get('f1', 0.0):.2f}%{_tag('f1', 'region_f1', 'region_f1_best_threshold')}"
    )
    if 'region_f2' in metrics or 'f2' in metrics:
        block += f"\n  F2: {metrics.get('region_f2', metrics.get('f2', 0.0)):.2f}%{_tag('region_f2')}"
    block += f"\n  AUC: {metrics.get('auc', 0.0):.2f}%"
    if 'slide_auc' in metrics:
        block += (
            f"\n  --- Slide-level (the actual diagnostic target) ---\n"
            f"  Slide Accuracy: {metrics.get('slide_accuracy', 0.0):.2f}%\n"
            f"  Slide Sensitivity: {metrics.get('slide_sensitivity', 0.0):.2f}%\n"
            f"  Slide Specificity: {metrics.get('slide_specificity', 0.0):.2f}%\n"
            f"  Slide F1: {metrics.get('slide_f1', 0.0):.2f}%{_tag('slide_f1')}"
        )
        if 'slide_f2' in metrics:
            block += f"\n  Slide F2: {metrics.get('slide_f2', 0.0):.2f}%{_tag('slide_f2')}"
        block += f"\n  Slide AUC: {metrics.get('slide_auc', 0.0):.2f}%"
    return block


def _format_multiclass_metric_block(metrics: dict) -> str:
    return (
        f"  Loss: {metrics.get('loss', 0.0):.4f}\n"
        f"  Accuracy: {metrics.get('accuracy', 0.0):.2f}%\n"
        f"  Balanced Accuracy: {metrics.get('balanced_accuracy', metrics.get('specificity', 0.0)):.2f}%\n"
        f"  F1 (Macro): {metrics.get('f1', 0.0):.2f}%\n"
        f"  Precision (Macro): {metrics.get('precision', 0.0):.2f}%\n"
        f"  Recall (Macro): {metrics.get('sensitivity', 0.0):.2f}%\n"
        f"  F1 (Weighted): {metrics.get('f1_weighted', 0.0):.2f}%\n"
        f"  AUC OVO (Macro): {metrics.get('auc', 0.0):.2f}%\n"
        f"  MCC: {metrics.get('mcc', 0.0):.2f}%\n"
        f"  Kappa: {metrics.get('kappa', 0.0):.2f}%\n"
        f"  Kappa (Quadratic): {metrics.get('kappa_quadratic', 0.0):.2f}%\n"
        f"  Top-2 Accuracy: {metrics.get('top2_accuracy', 0.0):.2f}%"
    )


def _log_epoch_metrics(
    epoch: int,
    total_epochs: int,
    train_metrics: dict,
    val_metrics: dict,
    multiclass: bool,
    monitor_metric_name: Optional[str] = None,
) -> None:
    if multiclass:
        train_block = _format_multiclass_metric_block(train_metrics)
        val_block = _format_multiclass_metric_block(val_metrics)
    else:
        train_block = _format_binary_metric_block(train_metrics, monitor_metric_name)
        val_block = _format_binary_metric_block(val_metrics, monitor_metric_name)
    logging.info(
        f"Epoch {epoch+1}/{total_epochs}\n"
        f"Train Metrics:\n"
        f"{train_block}\n"
        f"Validation Metrics:\n"
        f"{val_block}"
    )


def _extract_label_scalar(value) -> float:
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("Empty tensor label")
        return float(value.reshape(-1)[0].detach().cpu().item())
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            raise ValueError("Empty sequence label")
        return _extract_label_scalar(value[0])
    return float(value)


def _infer_binary_graph_labels(dataset) -> list:
    labels = []

    if hasattr(dataset, 'records') and isinstance(getattr(dataset, 'records'), list):
        for record in dataset.records:
            if isinstance(record, dict) and 'label' in record:
                try:
                    labels.append(_extract_label_scalar(record['label']))
                except (TypeError, ValueError):
                    continue
        return labels

    if hasattr(dataset, 'slide_groups') and hasattr(dataset, 'base'):
        for indices in getattr(dataset, 'slide_groups', []):
            if not indices:
                continue
            try:
                _, label, _ = dataset.base[indices[0]]
                labels.append(_extract_label_scalar(label))
            except (TypeError, ValueError, KeyError, IndexError):
                continue
        return labels

    logging.warning(
        "Could not infer binary graph labels for automatic class balancing. "
        "Proceeding with configured pos_weight only."
    )
    return labels


def _resolve_monitor_metric(config: dict, default_metric: str = 'loss') -> tuple[str, str, float]:
    early_cfg = config.get('early_stopping', {}) if isinstance(config.get('early_stopping', {}), dict) else {}
    metric_name = str(
        early_cfg.get('monitor_metric', early_cfg.get('monitor', config.get('monitor_metric', default_metric)))
    ).strip().lower()
    if metric_name.startswith('val_'):
        metric_name = metric_name[4:]

    maximize_metrics = {
        'auc', 'f1', 'accuracy', 'sensitivity', 'specificity', 'precision',
        'balanced_accuracy', 'f1_weighted', 'top2_accuracy', 'mcc', 'kappa', 'kappa_quadratic'
    }
    mode = 'max' if metric_name in maximize_metrics else 'min'

    min_delta = float(early_cfg.get('min_delta', 0.0))
    return metric_name, mode, min_delta


def _metric_value(metrics: dict, metric_name: str) -> float:
    if metric_name in metrics and metrics[metric_name] is not None:
        return float(metrics[metric_name])

    fallback_key = f"val_{metric_name}"
    if fallback_key in metrics and metrics[fallback_key] is not None:
        return float(metrics[fallback_key])

    if metric_name == 'loss':
        return float(metrics.get('loss', float('inf')))

    logging.warning("Requested monitor metric '%s' missing or None. Falling back to validation loss.", metric_name)
    return float(metrics.get('loss', float('inf')))


def _is_positive_region_label(label_text: str) -> bool:
    text = str(label_text).strip().lower()
    return any(token in text for token in ("tumor", "metast", "macro", "micro", "itc", "positive", "status=inflamed"))


def _is_normal_region_label(label_text: str) -> bool:
    text = str(label_text).strip().lower()
    return any(token in text for token in ("normal", "negative", "benign", "healthy", "background", "non_tumor", "status=noninflamed"))


_extract_annotation_vocab_from_graph_batch = training_utils.extract_annotation_vocab_from_graph_batch


def _annotation_enabled_per_graph_in_batch(data, num_graphs: int, device: torch.device) -> torch.Tensor:
    meta_obj = getattr(data, 'meta', None)
    flags = [False] * num_graphs

    def _as_scalar_bool(value) -> bool:
        if torch.is_tensor(value):
            return bool(value.reshape(-1)[0].item())
        return bool(value)

    if isinstance(meta_obj, dict):
        raw = meta_obj.get('annotation_labels_enabled', False)
        if torch.is_tensor(raw) and raw.numel() == num_graphs:
            flags = [bool(v) for v in raw.tolist()]
        elif isinstance(raw, (list, tuple)) and len(raw) == num_graphs:
            flags = [bool(v) for v in raw]
        else:
            flags = [_as_scalar_bool(raw)] * num_graphs
    elif isinstance(meta_obj, list):
        for i, meta_entry in enumerate(meta_obj):
            if i >= num_graphs:
                break
            if isinstance(meta_entry, dict):
                flags[i] = _as_scalar_bool(meta_entry.get('annotation_labels_enabled', False))

    return torch.tensor(flags, dtype=torch.bool, device=device)


def _map_region_annotation_ids_to_binary(ann_ids: torch.Tensor, vocab: list[str]) -> torch.Tensor:
    mapped = torch.full_like(ann_ids, fill_value=-1)
    if ann_ids.numel() == 0:
        return mapped

    unique_ids = sorted(set(int(v) for v in ann_ids[ann_ids >= 0].tolist()))
    if not unique_ids:
        return mapped

    if vocab:
        id_to_binary = {}
        for ann_id in unique_ids:
            if ann_id < 0 or ann_id >= len(vocab):
                continue
            label_text = str(vocab[ann_id]).strip().lower()
            if _is_positive_region_label(label_text):
                id_to_binary[ann_id] = 1
            elif _is_normal_region_label(label_text):
                id_to_binary[ann_id] = 0
        for ann_id, binary_label in id_to_binary.items():
            mapped[ann_ids == ann_id] = int(binary_label)

    return mapped


def _align_node_tensor_to_logits(
    tensor: torch.Tensor | None,
    n_logits: int,
    node_idx: torch.Tensor | None,
) -> torch.Tensor | None:
    if not torch.is_tensor(tensor):
        return None
    if tensor.size(0) == n_logits:
        return tensor
    if torch.is_tensor(node_idx) and node_idx.numel() == n_logits:
        idx = node_idx.to(tensor.device).long()
        if idx.numel() == 0 or int(idx.max().item()) < tensor.size(0):
            return tensor.index_select(0, idx)
    return None


def _smooth_slide_labels(slide_labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0.0:
        return slide_labels
    eps = float(min(max(smoothing, 0.0), 1.0))
    return slide_labels * (1.0 - eps) + 0.5 * eps


def _apply_mil_pseudo_labels(
    mapped: torch.Tensor,
    fallback_mask_all: torch.Tensor,
    region_logits: torch.Tensor,
    still_unlabeled: torch.Tensor,
    region_batch: torch.Tensor | None,
    top_fraction: float,
) -> None:
    if not torch.any(still_unlabeled):
        return

    detached_scores = region_logits.detach()
    if region_batch is not None:
        graph_ids = torch.unique(region_batch[still_unlabeled])
    else:
        region_batch = torch.zeros_like(mapped)
        graph_ids = torch.zeros(1, dtype=torch.long, device=mapped.device)

    for gid in graph_ids.tolist():
        idx = torch.where(still_unlabeled & (region_batch == gid))[0]
        if idx.numel() < 2:
            continue
        k = max(1, min(int(round(idx.numel() * top_fraction)), idx.numel() - 1))
        top_idx = idx[torch.argsort(detached_scores[idx], descending=True, stable=True)[:k]]
        mapped[idx] = 0
        mapped[top_idx] = 1
        fallback_mask_all[idx] = True


def _resolve_node_binary_targets(
    model_output,
    data,
    device: torch.device,
    slide_labels: torch.Tensor | None = None,
    unlabeled_region_policy: str = 'normal',
    node_type: str = 'region',
    pseudo_label_top_fraction: float = 0.0,
    negative_label_source: str = 'slide_derived',
):
    negative_label_source = str(negative_label_source or 'slide_derived').strip().lower()
    confident_negative = negative_label_source == 'slide_derived'
    if not isinstance(model_output, dict):
        return None, None, None, None

    region_logits = model_output.get(f'{node_type}_logits', None)
    if not torch.is_tensor(region_logits):
        region_probs = model_output.get(f'{node_type}_probs', None)
        if torch.is_tensor(region_probs):
            region_probs = region_probs.float().clamp(min=1e-6, max=1.0 - 1e-6)
            region_logits = torch.log(region_probs / (1.0 - region_probs))
        else:
            return None, None, None, None

    try:
        region_store = data[node_type]
    except Exception:
        return None, None, None, None

    region_logits = region_logits.view(-1).to(device).float()
    n_regions = int(region_logits.numel())
    node_idx = model_output.get(f'{node_type}_node_idx', None)

    mapped = torch.full((n_regions,), fill_value=-1, dtype=torch.long, device=device)
    annotation_y = _align_node_tensor_to_logits(getattr(region_store, 'annotation_y', None), n_regions, node_idx)
    ann_ids = None
    if torch.is_tensor(annotation_y):
        ann_ids = annotation_y.view(-1).to(device).long()
        if ann_ids.numel() != n_regions:
            return None, None, None, None
        vocab = _extract_annotation_vocab_from_graph_batch(data)
        mapped = _map_region_annotation_ids_to_binary(ann_ids, vocab)

    def _region_slide_binary_labels() -> torch.Tensor | None:
        if slide_labels is None or slide_labels.numel() <= 0:
            return None
        slide_binary = (slide_labels.view(-1).to(device).float() > 0.5).long()
        region_batch = _align_node_tensor_to_logits(getattr(region_store, 'batch', None), n_regions, node_idx)
        if torch.is_tensor(region_batch) and region_batch.numel() == n_regions:
            region_batch = region_batch.view(-1).to(device).long()
            valid_batch = (region_batch >= 0) & (region_batch < slide_binary.numel())
            per_region = torch.full((n_regions,), fill_value=-1, dtype=torch.long, device=device)
            if torch.any(valid_batch):
                per_region[valid_batch] = slide_binary[region_batch[valid_batch]]
            return per_region
        return torch.full((n_regions,), int(slide_binary[0].item()), dtype=torch.long, device=device)

    fallback_mask_all = torch.zeros_like(mapped, dtype=torch.bool)

    if ann_ids is not None:
        unresolved_ann_mask = (ann_ids >= 0) & (mapped < 0)
        if torch.any(unresolved_ann_mask):
            per_region_slide = _region_slide_binary_labels()
            if per_region_slide is not None:
                assign_mask = unresolved_ann_mask & (per_region_slide >= 0)
                if torch.any(assign_mask):
                    mapped[assign_mask] = per_region_slide[assign_mask]
                    negative_derived_is_fallback = (per_region_slide == 0) if not confident_negative else torch.zeros_like(per_region_slide, dtype=torch.bool)
                    fallback_mask_all |= assign_mask & ((per_region_slide == 1) | negative_derived_is_fallback)

    policy = str(unlabeled_region_policy or 'normal').strip().lower()
    unlabeled_mask = mapped < 0

    if torch.any(unlabeled_mask):
        if policy in {'normal', 'negative'}:
            mapped[unlabeled_mask] = 0
            fallback_mask_all |= unlabeled_mask
        elif policy in {'slide_aware', 'positive_ignore_negative_normal'} and slide_labels is not None and slide_labels.numel() > 0:
            slide_binary = (slide_labels.view(-1).to(device).float() > 0.5).long()
            annotation_enabled = _annotation_enabled_per_graph_in_batch(data, slide_binary.numel(), device)
            region_batch = _align_node_tensor_to_logits(getattr(region_store, 'batch', None), n_regions, node_idx)
            if torch.is_tensor(region_batch) and region_batch.numel() == n_regions:
                region_batch = region_batch.view(-1).to(device).long()
                valid_batch = (region_batch >= 0) & (region_batch < slide_binary.numel())
                negative_slide_region = valid_batch & (slide_binary[region_batch] == 0)
                annotated_slide_region = valid_batch & annotation_enabled[region_batch]
                confidently_negative_region = negative_slide_region | annotated_slide_region
                assign_mask = unlabeled_mask & confidently_negative_region
                if torch.any(assign_mask):
                    mapped[assign_mask] = 0
                    if not confident_negative:
                        fallback_mask_all |= assign_mask
            else:
                if int(slide_binary[0].item()) == 0 or bool(annotation_enabled[0].item()):
                    mapped[unlabeled_mask] = 0
                    if not confident_negative:
                        fallback_mask_all |= unlabeled_mask

            if pseudo_label_top_fraction > 0.0:
                region_batch_for_pl = (
                    region_batch if torch.is_tensor(region_batch) and region_batch.numel() == n_regions else None
                )
                _apply_mil_pseudo_labels(
                    mapped=mapped,
                    fallback_mask_all=fallback_mask_all,
                    region_logits=region_logits,
                    still_unlabeled=(mapped < 0),
                    region_batch=region_batch_for_pl,
                    top_fraction=pseudo_label_top_fraction,
                )
        elif policy == 'slide' and slide_labels is not None and slide_labels.numel() > 0:
            slide_binary = (slide_labels.view(-1).to(device).float() > 0.5).long()
            region_batch = _align_node_tensor_to_logits(getattr(region_store, 'batch', None), n_regions, node_idx)
            if torch.is_tensor(region_batch) and region_batch.numel() == n_regions:
                region_batch = region_batch.view(-1).to(device).long()
                valid_batch = (region_batch >= 0) & (region_batch < slide_binary.numel())
                assign_mask = unlabeled_mask & valid_batch
                if torch.any(assign_mask):
                    mapped[assign_mask] = slide_binary[region_batch[assign_mask]]
                    fallback_mask_all |= assign_mask
            else:
                mapped[unlabeled_mask] = slide_binary[0]
                fallback_mask_all |= unlabeled_mask

    if node_type == 'region':
        label_ambiguous = _align_node_tensor_to_logits(getattr(region_store, 'label_ambiguous', None), n_regions, node_idx)
        if torch.is_tensor(label_ambiguous) and label_ambiguous.numel() == n_regions:
            ambiguous_mask = label_ambiguous.view(-1).to(device).bool() & (mapped >= 0)
            fallback_mask_all = fallback_mask_all | ambiguous_mask

    return region_logits, mapped, fallback_mask_all, node_idx


def _extract_region_supervision_tensors(
    model_output,
    data,
    device: torch.device,
    slide_labels: torch.Tensor | None = None,
    unlabeled_region_policy: str = 'normal',
    node_type: str = 'region',
    pseudo_label_top_fraction: float = 0.0,
    negative_label_source: str = 'slide_derived',
):
    region_logits, mapped, fallback_mask_all, _node_idx = _resolve_node_binary_targets(
        model_output,
        data,
        device,
        slide_labels=slide_labels,
        unlabeled_region_policy=unlabeled_region_policy,
        node_type=node_type,
        pseudo_label_top_fraction=pseudo_label_top_fraction,
        negative_label_source=negative_label_source,
    )
    if region_logits is None:
        return None, None, 0, 0, None

    labeled_mask = mapped >= 0
    if not torch.any(labeled_mask):
        return None, None, 0, 0, None

    targets = mapped[labeled_mask].float()
    logits = region_logits[labeled_mask]
    fallback_mask_labeled = fallback_mask_all[labeled_mask]

    return logits, targets, int(labeled_mask.sum().item()), int(fallback_mask_labeled.sum().item()), fallback_mask_labeled


def _extract_patch_supervision_tensors_with_region_fallback(
    model_output,
    data,
    device: torch.device,
    slide_labels: torch.Tensor | None = None,
    unlabeled_region_policy: str = 'slide_aware',
    pseudo_label_top_fraction: float = 0.0,
    negative_label_source: str = 'slide_derived',
):
    if not isinstance(model_output, dict):
        return None, None, 0, 0, None

    patch_logits = model_output.get('patch_logits', None)
    if not torch.is_tensor(patch_logits):
        return None, None, 0, 0, None

    try:
        patch_store = data['patch']
        region_store = data['region']
    except Exception:
        return None, None, 0, 0, None

    if ('patch', 'in', 'region') not in data.edge_types:
        return None, None, 0, 0, None

    patch_x = getattr(patch_store, 'x', None)
    if not torch.is_tensor(patch_x) or patch_x.size(0) <= 0:
        return None, None, 0, 0, None

    patch_logits = patch_logits.view(-1).to(device).float()
    n_patches_out = int(patch_logits.numel())
    patch_node_idx = model_output.get('patch_node_idx', None)

    own_mapped = torch.full((n_patches_out,), fill_value=-1, dtype=torch.long, device=device)
    own_annotation_y = _align_node_tensor_to_logits(getattr(patch_store, 'annotation_y', None), n_patches_out, patch_node_idx)
    if torch.is_tensor(own_annotation_y):
        ann_ids = own_annotation_y.view(-1).to(device).long()
        if ann_ids.numel() == n_patches_out:
            vocab = _extract_annotation_vocab_from_graph_batch(data)
            own_mapped = _map_region_annotation_ids_to_binary(ann_ids, vocab)

    _region_logits_full, region_mapped_full, region_fallback_full, _region_node_idx = _resolve_node_binary_targets(
        model_output=model_output,
        data=data,
        device=device,
        slide_labels=slide_labels,
        unlabeled_region_policy=unlabeled_region_policy,
        node_type='region',
        pseudo_label_top_fraction=pseudo_label_top_fraction,
        negative_label_source=negative_label_source,
    )
    if region_mapped_full is None:
        n_region_nodes = int(getattr(region_store, 'x').size(0)) if torch.is_tensor(getattr(region_store, 'x', None)) else 0
        region_mapped_full = torch.full((n_region_nodes,), fill_value=-1, dtype=torch.long, device=device)
        region_fallback_full = torch.zeros((n_region_nodes,), dtype=torch.bool, device=device)

    n_patches_orig = int(patch_x.size(0))
    p2r_edge_index = data[('patch', 'in', 'region')].edge_index.long().to(device)
    patch_to_region_orig = torch.full((n_patches_orig,), fill_value=-1, dtype=torch.long, device=device)
    p_src, p_dst = p2r_edge_index[0], p2r_edge_index[1]
    valid_edge = (p_src >= 0) & (p_src < n_patches_orig) & (p_dst >= 0) & (p_dst < region_mapped_full.numel())
    patch_to_region_orig[p_src[valid_edge]] = p_dst[valid_edge]

    if torch.is_tensor(patch_node_idx) and patch_node_idx.numel() == n_patches_out:
        orig_idx = patch_node_idx.to(device).long()
    else:
        orig_idx = torch.arange(n_patches_out, device=device)
    valid_orig = (orig_idx >= 0) & (orig_idx < n_patches_orig)

    region_for_patch = torch.full((n_patches_out,), fill_value=-1, dtype=torch.long, device=device)
    region_for_patch[valid_orig] = patch_to_region_orig[orig_idx[valid_orig]]
    has_region = region_for_patch >= 0

    region_inherited = torch.full((n_patches_out,), fill_value=-1, dtype=torch.long, device=device)
    inherited_fallback = torch.zeros((n_patches_out,), dtype=torch.bool, device=device)
    if torch.any(has_region):
        region_inherited[has_region] = region_mapped_full[region_for_patch[has_region]]
        inherited_fallback[has_region] = region_fallback_full[region_for_patch[has_region]]

    mapped = own_mapped.clone()
    fallback_mask_all = torch.zeros((n_patches_out,), dtype=torch.bool, device=device)
    needs_fallback = (mapped < 0) & (region_inherited >= 0)
    mapped[needs_fallback] = region_inherited[needs_fallback]
    fallback_mask_all[needs_fallback] = inherited_fallback[needs_fallback]

    labeled_mask = mapped >= 0
    if not torch.any(labeled_mask):
        return None, None, 0, 0, None

    targets = mapped[labeled_mask].float()
    logits = patch_logits[labeled_mask]
    fallback_mask_labeled = fallback_mask_all[labeled_mask]

    return logits, targets, int(labeled_mask.sum().item()), int(fallback_mask_labeled.sum().item()), fallback_mask_labeled


def _apply_fallback_subsampling(
    logits: torch.Tensor,
    targets: torch.Tensor,
    fallback_mask: torch.Tensor,
    max_fallback_to_explicit_ratio: float,
    prefer_explicit_only_when_available: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.numel() == 0:
        return logits, targets, fallback_mask

    if max_fallback_to_explicit_ratio <= 0:
        return logits, targets, fallback_mask

    explicit_mask = ~fallback_mask
    explicit_count = int(explicit_mask.sum().item())
    fallback_count = int(fallback_mask.sum().item())

    if prefer_explicit_only_when_available and explicit_count > 0 and fallback_count > 0:
        return logits[explicit_mask], targets[explicit_mask], fallback_mask[explicit_mask]

    if fallback_count <= 0:
        return logits, targets, fallback_mask

    if explicit_count <= 0:
        return logits, targets, fallback_mask

    max_fallback_keep = int(round(explicit_count * float(max_fallback_to_explicit_ratio)))
    if fallback_count <= max_fallback_keep:
        return logits, targets, fallback_mask

    fallback_indices = torch.where(fallback_mask)[0]
    perm = torch.randperm(fallback_indices.numel(), device=logits.device)
    keep_fallback_indices = fallback_indices[perm[:max(1, max_fallback_keep)]]

    keep_mask = explicit_mask.clone()
    keep_mask[keep_fallback_indices] = True

    return logits[keep_mask], targets[keep_mask], fallback_mask[keep_mask]


def _f_beta_from_precision_recall(precision_pct: float, recall_pct: float, beta: float) -> float:
    p = float(precision_pct) / 100.0
    r = float(recall_pct) / 100.0
    beta2 = float(beta) ** 2
    denom = beta2 * p + r
    if denom <= 0.0:
        return 0.0
    return 100.0 * (1.0 + beta2) * p * r / denom


def _binary_metric_block(labels: list[float], raw_scores: list[float]) -> dict:
    empty_block = {
        'accuracy': 0.0,
        'sensitivity': 0.0,
        'specificity': 0.0,
        'precision': 0.0,
        'f1': 0.0,
        'f2': 0.0,
        'balanced_accuracy': 0.0,
        'auc': 50.0,
        'f1_best_threshold': 0.0,
        'best_threshold': 0.5,
    }
    if not labels:
        return empty_block

    scores = np.asarray(raw_scores, dtype=float)
    labels_np = np.asarray(labels, dtype=float)

    finite_mask = np.isfinite(scores) & np.isfinite(labels_np)
    if not np.all(finite_mask):
        dropped = int((~finite_mask).sum())
        logging.warning("_binary_metric_block: dropping %d non-finite label/score entries.", dropped)
        scores = scores[finite_mask]
        labels_np = labels_np[finite_mask]
        labels = labels_np.tolist()
        raw_scores = scores.tolist()

    if not labels:
        return empty_block

    preds = (scores > 0.0).astype(int).tolist()
    metrics = metrics_utils.calculate_basic_metrics(labels=labels, preds=preds, raw_preds=raw_scores)
    metrics['f2'] = _f_beta_from_precision_recall(
        metrics.get('precision', 0.0), metrics.get('sensitivity', 0.0), beta=2.0
    )
    metrics['balanced_accuracy'] = (
        float(metrics.get('sensitivity', 0.0)) + float(metrics.get('specificity', 0.0))
    ) / 2.0

    best_thr = 0.5
    best_f1 = float(metrics.get('f1', 0.0)) / 100.0

    if np.unique(labels_np).size > 1:
        probs = 1.0 / (1.0 + np.exp(-scores))
        for thr in np.linspace(0.1, 0.9, 17):
            pred_thr = (probs >= float(thr)).astype(int)
            tp = int(np.sum((labels_np == 1.0) & (pred_thr == 1)))
            fp = int(np.sum((labels_np == 0.0) & (pred_thr == 1)))
            fn = int(np.sum((labels_np == 1.0) & (pred_thr == 0)))
            f1_thr = float((2 * tp) / max(1, (2 * tp + fp + fn)))
            if f1_thr > best_f1:
                best_f1 = f1_thr
                best_thr = float(thr)

    metrics['f1_best_threshold'] = 100.0 * best_f1
    metrics['best_threshold'] = best_thr
    return metrics


def _region_slide_consistency_loss(
    model_output,
    data,
    slide_labels: torch.Tensor,
    device: torch.device,
    node_type: str = 'region',
) -> torch.Tensor:
    if not isinstance(model_output, dict):
        return torch.zeros((), device=device)

    region_logits = model_output.get(f'{node_type}_logits', None)
    if not torch.is_tensor(region_logits):
        region_probs = model_output.get(f'{node_type}_probs', None)
        if torch.is_tensor(region_probs):
            region_probs = region_probs.float().clamp(min=1e-6, max=1.0 - 1e-6)
            region_logits = torch.log(region_probs / (1.0 - region_probs))
        else:
            return torch.zeros((), device=device)

    try:
        region_store = data[node_type]
    except Exception:
        return torch.zeros((), device=device)

    region_logits = region_logits.view(-1).to(device).float()
    if region_logits.numel() == 0 or slide_labels.numel() == 0:
        return torch.zeros((), device=device)

    slide_targets = training_utils._sanitize_binary_labels(slide_labels.view(-1).to(device).float())
    node_idx = model_output.get(f'{node_type}_node_idx', None)
    region_batch = _align_node_tensor_to_logits(getattr(region_store, 'batch', None), int(region_logits.numel()), node_idx)

    if torch.is_tensor(region_batch) and region_batch.numel() == region_logits.numel():
        region_batch = region_batch.view(-1).to(device).long()
        max_logits_per_slide = []
        max_targets_per_slide = []
        for slide_idx in range(slide_targets.numel()):
            mask = region_batch == slide_idx
            if torch.any(mask):
                max_logits_per_slide.append(region_logits[mask].max())
                max_targets_per_slide.append(slide_targets[slide_idx])
        if not max_logits_per_slide:
            return torch.zeros((), device=device)
        slide_logits_from_regions = torch.stack(max_logits_per_slide)
        slide_targets_from_regions = torch.stack(max_targets_per_slide)
    else:
        slide_logits_from_regions = region_logits.max().reshape(1)
        slide_targets_from_regions = slide_targets[:1]

    return nn.BCEWithLogitsLoss()(slide_logits_from_regions, slide_targets_from_regions)


def _compute_patch_fallback_loss(
    model_output,
    data,
    device: torch.device,
    slide_labels: torch.Tensor,
    unlabeled_region_policy: str,
    pseudo_label_top_fraction: float,
    negative_label_source: str,
    max_fallback_to_explicit_ratio: float,
    prefer_explicit_only_when_available: bool,
    fallback_region_weight: float,
    loss_dtype: torch.dtype,
):
    patch_logits, patch_targets, num_labeled, _fallback_count, fallback_mask = (
        _extract_patch_supervision_tensors_with_region_fallback(
            model_output,
            data,
            device,
            slide_labels=slide_labels,
            unlabeled_region_policy=unlabeled_region_policy,
            pseudo_label_top_fraction=pseudo_label_top_fraction,
            negative_label_source=negative_label_source,
        )
    )
    if num_labeled <= 0:
        return torch.zeros((), dtype=loss_dtype, device=device), False, None, None, None

    patch_logits, patch_targets, fallback_mask = _apply_fallback_subsampling(
        patch_logits,
        patch_targets,
        fallback_mask,
        max_fallback_to_explicit_ratio=max_fallback_to_explicit_ratio,
        prefer_explicit_only_when_available=prefer_explicit_only_when_available,
    )

    pos = float((patch_targets >= 0.5).sum().item())
    neg = float((patch_targets < 0.5).sum().item())
    if pos > 0.0 and neg > 0.0:
        patch_pos_weight = torch.tensor(min(12.0, max(1.0, neg / pos)), device=device)
    else:
        patch_pos_weight = torch.tensor(1.0, device=device)

    per_sample = nn.BCEWithLogitsLoss(pos_weight=patch_pos_weight, reduction='none')(
        patch_logits,
        patch_targets,
    )
    sample_weights = torch.ones_like(per_sample)
    if torch.any(fallback_mask):
        sample_weights = torch.where(
            fallback_mask,
            torch.full_like(sample_weights, float(fallback_region_weight)),
            sample_weights,
        )
    patch_loss = (per_sample * sample_weights).mean()
    return patch_loss, True, patch_logits, patch_targets, fallback_mask


def _is_cuda_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _safe_graph_identity(data) -> str:
    parts = []
    slide_name = getattr(data, 'slide_name', None)
    if isinstance(slide_name, (list, tuple)) and slide_name:
        slide_name = slide_name[0]
    if slide_name is not None:
        parts.append(f"slide={slide_name}")
    try:
        if hasattr(data, 'node_types') and 'patch' in data.node_types:
            n_patch = int(getattr(data['patch'], 'x').size(0))
            parts.append(f"patch_nodes={n_patch}")
        if hasattr(data, 'node_types') and 'region' in data.node_types:
            n_region = int(getattr(data['region'], 'x').size(0))
            parts.append(f"region_nodes={n_region}")
    except Exception:
        pass
    return ", ".join(parts) if parts else "unknown"


def _resolve_augmentation_consistency_perturbation(
    data,
    specs: list,
    rng: np.random.Generator,
    p: float,
):
    if not specs or rng.random() >= p:
        return None
    data_perturbed = data.clone()
    apply_graph_augmentations(data_perturbed, specs, rng=rng)
    return data_perturbed


def _augmentation_consistency_loss_from_perturbed(
    model: nn.Module,
    data_perturbed,
    slide_logits: torch.Tensor,
    slide_h: Optional[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if data_perturbed is None:
        zero = torch.zeros((), dtype=slide_logits.dtype, device=device)
        return zero, zero

    model_output_perturbed = model(data_perturbed)
    slide_logits_perturbed, _aux_unused = training_utils._unpack_graph_model_output(model_output_perturbed, device)
    slide_logits_perturbed = training_utils._sanitize_logits_for_loss(slide_logits_perturbed, multiclass=False)

    output_loss = ((torch.sigmoid(slide_logits) - torch.sigmoid(slide_logits_perturbed)) ** 2).mean()

    feature_loss = torch.zeros((), dtype=slide_logits.dtype, device=device)
    slide_h_perturbed = model_output_perturbed.get('slide_h') if isinstance(model_output_perturbed, dict) else None
    if slide_h is not None and slide_h_perturbed is not None:
        feature_loss = (1.0 - nn.functional.cosine_similarity(slide_h, slide_h_perturbed, dim=-1)).mean()

    return output_loss, feature_loss


def _compute_segmentation_step_losses(
    model: nn.Module,
    data,
    slide_labels: torch.Tensor,
    mixup_data,
    mixup_lambda,
    slide_criterion: nn.Module,
    region_loss_weight: float,
    slide_loss_weight: float,
    aux_loss_weight: float,
    fallback_region_weight: float,
    max_fallback_to_explicit_ratio: float,
    prefer_explicit_only_when_available: bool,
    region_slide_consistency_weight: float,
    unlabeled_region_policy: str,
    node_type: str,
    pseudo_label_top_fraction: float,
    negative_label_source: str,
    patch_loss_weight: float,
    augmentation_consistency_weight: float,
    augmentation_consistency_perturbed,
    feature_consistency_weight: float,
    slide_label_smoothing: float,
    device: torch.device,
    use_amp: bool,
) -> dict:
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16) if (use_amp and device.type == 'cuda') else contextlib.nullcontext()
    with ctx:
        model_output = model(data, mixup_data=mixup_data, mixup_lambda=mixup_lambda)
        slide_logits, aux_loss = training_utils._unpack_graph_model_output(model_output, device)
        slide_logits = training_utils._sanitize_logits_for_loss(slide_logits, multiclass=False)
        slide_h = model_output.get('slide_h') if isinstance(model_output, dict) else None
        slide_loss = slide_criterion(slide_logits, _smooth_slide_labels(slide_labels, slide_label_smoothing))
        consistency_loss = _region_slide_consistency_loss(
            model_output=model_output,
            data=data,
            slide_labels=slide_labels,
            device=device,
            node_type=node_type,
        )

        region_logits, region_targets, num_labeled, fallback_count, fallback_mask = _extract_region_supervision_tensors(
            model_output,
            data,
            device,
            slide_labels=slide_labels,
            unlabeled_region_policy=unlabeled_region_policy,
            node_type=node_type,
            pseudo_label_top_fraction=pseudo_label_top_fraction,
            negative_label_source=negative_label_source,
        )
        has_region_supervision = num_labeled > 0
        if has_region_supervision:
            region_logits, region_targets, fallback_mask = _apply_fallback_subsampling(
                region_logits,
                region_targets,
                fallback_mask,
                max_fallback_to_explicit_ratio=max_fallback_to_explicit_ratio,
                prefer_explicit_only_when_available=prefer_explicit_only_when_available,
            )

            pos = float((region_targets >= 0.5).sum().item())
            neg = float((region_targets < 0.5).sum().item())
            if pos > 0.0 and neg > 0.0:
                region_pos_weight = torch.tensor(min(12.0, max(1.0, neg / pos)), device=device)
            else:
                region_pos_weight = torch.tensor(1.0, device=device)

            per_sample = nn.BCEWithLogitsLoss(pos_weight=region_pos_weight, reduction='none')(
                region_logits,
                region_targets,
            )
            sample_weights = torch.ones_like(per_sample)
            if torch.any(fallback_mask):
                sample_weights = torch.where(
                    fallback_mask,
                    torch.full_like(sample_weights, float(fallback_region_weight)),
                    sample_weights,
                )
            region_loss = (per_sample * sample_weights).mean()
        else:
            region_loss = torch.zeros((), dtype=slide_loss.dtype, device=device)

        if patch_loss_weight > 0.0:
            patch_loss, has_patch_supervision, patch_logits_kept, patch_targets_kept, patch_fallback_mask_kept = (
                _compute_patch_fallback_loss(
                    model_output=model_output,
                    data=data,
                    device=device,
                    slide_labels=slide_labels,
                    unlabeled_region_policy=unlabeled_region_policy,
                    pseudo_label_top_fraction=pseudo_label_top_fraction,
                    negative_label_source=negative_label_source,
                    max_fallback_to_explicit_ratio=max_fallback_to_explicit_ratio,
                    prefer_explicit_only_when_available=prefer_explicit_only_when_available,
                    fallback_region_weight=fallback_region_weight,
                    loss_dtype=slide_loss.dtype,
                )
            )
        else:
            patch_loss = torch.zeros((), dtype=slide_loss.dtype, device=device)
            has_patch_supervision, patch_logits_kept, patch_targets_kept, patch_fallback_mask_kept = False, None, None, None

        augmentation_consistency_loss, feature_consistency_loss = _augmentation_consistency_loss_from_perturbed(
            model=model,
            data_perturbed=augmentation_consistency_perturbed,
            slide_logits=slide_logits,
            slide_h=slide_h,
            device=device,
        )

        total_step_loss = (
            float(region_loss_weight) * region_loss
            + float(slide_loss_weight) * slide_loss
            + float(aux_loss_weight) * aux_loss
            + float(region_slide_consistency_weight) * consistency_loss
            + float(patch_loss_weight) * patch_loss
            + float(augmentation_consistency_weight) * augmentation_consistency_loss
            + float(feature_consistency_weight) * feature_consistency_loss
        )

    return {
        'total': total_step_loss,
        'region_loss': region_loss,
        'slide_loss': slide_loss,
        'aux_loss': aux_loss,
        'consistency_loss': consistency_loss,
        'patch_loss': patch_loss,
        'augmentation_consistency_loss': augmentation_consistency_loss,
        'feature_consistency_loss': feature_consistency_loss,
        'slide_logits': slide_logits,
        'has_region_supervision': has_region_supervision,
        'region_logits': region_logits,
        'region_targets': region_targets,
        'fallback_mask': fallback_mask,
        'has_patch_supervision': has_patch_supervision,
        'patch_logits_kept': patch_logits_kept,
        'patch_targets_kept': patch_targets_kept,
        'patch_fallback_mask_kept': patch_fallback_mask_kept,
        'region_cls_correct': model_output.get('region_cls_correct') if isinstance(model_output, dict) else None,
        'region_cls_total': model_output.get('region_cls_total') if isinstance(model_output, dict) else None,
        'patch_cls_correct': model_output.get('patch_cls_correct') if isinstance(model_output, dict) else None,
        'patch_cls_total': model_output.get('patch_cls_total') if isinstance(model_output, dict) else None,
        'region_cls_canonical': model_output.get('region_cls_canonical') if isinstance(model_output, dict) else None,
        'region_cls_logits': model_output.get('region_cls_logits') if isinstance(model_output, dict) else None,
        'patch_cls_canonical': model_output.get('patch_cls_canonical') if isinstance(model_output, dict) else None,
        'patch_cls_logits': model_output.get('patch_cls_logits') if isinstance(model_output, dict) else None,
    }


def _compute_cls_kappa_macro_f1(
    all_region_cls_true: list, all_region_cls_pred: list,
    all_patch_cls_true: list, all_patch_cls_pred: list,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    region_cls_kappa = region_cls_macro_f1 = None
    patch_cls_kappa = patch_cls_macro_f1 = None
    if all_region_cls_true or all_patch_cls_true:
        from sklearn.metrics import cohen_kappa_score, f1_score as _f1_score
        if all_region_cls_true:
            region_cls_kappa = float(cohen_kappa_score(all_region_cls_true, all_region_cls_pred))
            region_cls_macro_f1 = float(_f1_score(all_region_cls_true, all_region_cls_pred, average='macro', zero_division=0))
        if all_patch_cls_true:
            patch_cls_kappa = float(cohen_kappa_score(all_patch_cls_true, all_patch_cls_pred))
            patch_cls_macro_f1 = float(_f1_score(all_patch_cls_true, all_patch_cls_pred, average='macro', zero_division=0))
    return region_cls_kappa, region_cls_macro_f1, patch_cls_kappa, patch_cls_macro_f1


def _accumulate_cls_correct_total(
    source: dict, prefix: str, running_correct: int, running_total: int,
) -> tuple[int, int]:
    correct = source.get(f'{prefix}_cls_correct')
    total = source.get(f'{prefix}_cls_total')
    if torch.is_tensor(correct) and torch.is_tensor(total):
        running_correct += int(correct.item())
        running_total += int(total.item())
    return running_correct, running_total


def _accumulate_cls_true_pred(
    source: dict, prefix: str, true_list: list, pred_list: list,
) -> None:
    canonical = source.get(f'{prefix}_cls_canonical')
    logits = source.get(f'{prefix}_cls_logits')
    if not torch.is_tensor(canonical) or not torch.is_tensor(logits):
        return
    if canonical.numel() != logits.size(0):
        return
    valid = canonical != -100
    if not torch.any(valid):
        return
    preds = logits.detach().argmax(dim=-1)
    true_list.extend(canonical[valid].detach().cpu().tolist())
    pred_list.extend(preds[valid].detach().cpu().tolist())


def _log_multiclass_classification_report(noun: str, class_names: list, y_true: list, y_pred: list) -> None:
    if not y_true:
        return
    from src.pipelines.evaluate import _compute_multiclass_classification_metrics

    metrics = _compute_multiclass_classification_metrics(np.array(y_true), np.array(y_pred), None, class_names)

    name_width = max(len(str(n)) for n in class_names + ["class"])
    header = f"  {'class':<{name_width}}  {'precision':>9}  {'recall':>9}  {'f1':>9}  {'specificity':>11}  {'support':>7}"
    lines = [f"Val {noun}_cls classification report (n={metrics['num_samples']}):", header]
    for name in class_names:
        row = metrics['per_class'][name]
        spec = row['specificity']
        lines.append(
            f"  {name:<{name_width}}  {row['precision']:>9.3f}  {row['recall']:>9.3f}  {row['f1']:>9.3f}  "
            f"{(spec if spec is not None else float('nan')):>11.3f}  {row['support']:>7d}"
        )
    lines.append(
        f"  overall: accuracy={metrics['accuracy']:.3f} balanced_accuracy={metrics['balanced_accuracy']:.3f} "
        f"kappa={metrics['cohen_kappa']:.3f} mcc={metrics['mcc']:.3f} macro_f1={metrics['macro_f1']:.3f} "
        f"weighted_f1={metrics['weighted_f1']:.3f}"
    )

    cm = metrics['confusion_matrix']
    col_width = max(name_width, max((len(str(v)) for row in cm for v in row), default=1))
    lines.append(f"  confusion matrix (rows=true, cols=pred):")
    lines.append("  " + " " * name_width + "  " + "  ".join(f"{n:>{col_width}}" for n in class_names))
    for name, row in zip(class_names, cm):
        lines.append(f"  {name:<{name_width}}  " + "  ".join(f"{v:>{col_width}d}" for v in row))

    logging.info("\n".join(lines))


def _train_epoch_graph_segmentation(
    model: nn.Module,
    train_loader: GeoDataLoader,
    optimizer: torch.optim.Optimizer,
    slide_criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    region_loss_weight: float,
    slide_loss_weight: float,
    aux_loss_weight: float,
    fallback_region_weight: float,
    max_fallback_to_explicit_ratio: float,
    prefer_explicit_only_when_available: bool,
    region_slide_consistency_weight: float,
    unlabeled_region_policy: str,
    use_amp: bool = True,
    gradient_clip_value: float = None,
    node_type: str = 'region',
    pseudo_label_top_fraction: float = 0.0,
    negative_label_source: str = 'slide_derived',
    patch_loss_weight: float = 0.0,
    augmentation_consistency_weight: float = 0.0,
    augmentation_consistency_specs: Optional[list] = None,
    augmentation_consistency_p: float = 1.0,
    feature_consistency_weight: float = 0.0,
    manifold_mixup_p: float = 0.0,
    manifold_mixup_alpha: float = 0.4,
    consistency_rng: Optional[np.random.Generator] = None,
    slide_label_smoothing: float = 0.0,
    grad_accum_steps: int = 1,
    sam: Optional["training_utils.GraphSAM"] = None,
):
    """Runs one training epoch over the graph segmentation train loader."""
    model.train()
    if consistency_rng is None:
        consistency_rng = np.random.default_rng()
    augmentation_consistency_specs = augmentation_consistency_specs or []
    accum_steps = max(1, int(grad_accum_steps))
    if sam is not None and accum_steps != 1:
        raise ValueError(
            "SAM (sam_enabled) requires grad_accum_steps=1 -- ascending mid-accumulation-window "
            "would mean the ascent direction reflects only one microbatch, not the full "
            "windowed loss landscape SAM's rationale depends on. Set grad_accum_steps=1 "
            "whenever sam_enabled=True (train_graph_segmentation_model should already enforce "
            "this at config-read time; reaching this error means that guard was bypassed)."
        )
    if sam is not None and use_amp:
        raise ValueError(
            "SAM (sam_enabled) requires use_amp=False -- GradScaler's single scale/step/update "
            "cycle isn't designed for SAM's two backward passes per step. "
            "train_graph_segmentation_model should already force mixed_precision off whenever "
            "sam_enabled=True; reaching this error means that guard was bypassed."
        )
    total_loss = 0.0
    total_region_loss = 0.0
    total_slide_loss = 0.0
    total_aux_loss = 0.0
    total_consistency_loss = 0.0
    total_patch_loss = 0.0
    total_augmentation_consistency_loss = 0.0
    total_feature_consistency_loss = 0.0
    total_region_cls_correct = 0
    total_region_cls_total = 0
    total_patch_cls_correct = 0
    total_patch_cls_total = 0
    all_region_cls_true: list[int] = []
    all_region_cls_pred: list[int] = []
    all_patch_cls_true: list[int] = []
    all_patch_cls_pred: list[int] = []
    processed_batches = 0
    skipped_unlabeled_batches = 0
    total_fallback_labels = 0
    total_explicit_labels = 0
    total_patch_fallback_labels = 0
    total_patch_explicit_labels = 0

    all_region_labels: list[float] = []
    all_region_logits: list[float] = []
    all_region_labels_explicit: list[float] = []
    all_region_logits_explicit: list[float] = []
    all_slide_labels: list[float] = []
    all_slide_logits: list[float] = []
    all_patch_labels: list[float] = []
    all_patch_logits: list[float] = []
    all_patch_labels_explicit: list[float] = []
    all_patch_logits_explicit: list[float] = []

    optimizer.zero_grad(set_to_none=True)
    total_steps = len(train_loader)

    for step_idx, data in enumerate(tqdm(train_loader, desc="Training (Graph Segmentation)", leave=False), start=1):
        try:
            data = data.to(device)

            if hasattr(data, 'y') and data.y is not None:
                slide_labels = data.y.view(-1).to(device).float()
            else:
                slide_labels = data['slide'].y.view(-1).to(device).float()
            slide_labels = training_utils._sanitize_binary_labels(slide_labels)

            # Manifold Mixup (see model.py's _apply_manifold_mixup docstring) -- off by default
            # (manifold_mixup_p=0.0). Reuses train_loader.dataset's lightweight second-graph
            # loader (the same one graph_augment.py's graph_mixup uses via second_graph_provider)
            # rather than a full __getitem__ call, since only raw region.x is needed here, not
            # the full resplit/augmentation/supervision pipeline. slide_labels is blended the
            # same way graph_mixup already blends data.y -- safe now that calculate_basic_metrics
            # rounds soft labels back to binary before confusion_matrix/roc_auc_score.
            mixup_data = None
            mixup_lambda = None
            if manifold_mixup_p > 0.0 and consistency_rng.random() < manifold_mixup_p:
                second_graph_provider = getattr(train_loader.dataset, "_load_random_graph_for_mixup", None)
                if second_graph_provider is not None:
                    sampled = second_graph_provider()
                    if sampled is not None:
                        mixup_data_raw, mixup_label_b = sampled
                        mixup_data = mixup_data_raw.to(device)
                        mixup_lambda = (
                            float(consistency_rng.beta(manifold_mixup_alpha, manifold_mixup_alpha))
                            if manifold_mixup_alpha > 0
                            else 1.0
                        )
                        slide_labels = mixup_lambda * slide_labels + (1.0 - mixup_lambda) * float(mixup_label_b)

            augmentation_consistency_perturbed = _resolve_augmentation_consistency_perturbation(
                data, augmentation_consistency_specs, consistency_rng, augmentation_consistency_p,
            )
            loss_kwargs = dict(
                model=model,
                data=data,
                slide_labels=slide_labels,
                mixup_data=mixup_data,
                mixup_lambda=mixup_lambda,
                slide_criterion=slide_criterion,
                region_loss_weight=region_loss_weight,
                slide_loss_weight=slide_loss_weight,
                aux_loss_weight=aux_loss_weight,
                fallback_region_weight=fallback_region_weight,
                max_fallback_to_explicit_ratio=max_fallback_to_explicit_ratio,
                prefer_explicit_only_when_available=prefer_explicit_only_when_available,
                region_slide_consistency_weight=region_slide_consistency_weight,
                unlabeled_region_policy=unlabeled_region_policy,
                node_type=node_type,
                pseudo_label_top_fraction=pseudo_label_top_fraction,
                negative_label_source=negative_label_source,
                patch_loss_weight=patch_loss_weight,
                augmentation_consistency_weight=augmentation_consistency_weight,
                augmentation_consistency_perturbed=augmentation_consistency_perturbed,
                feature_consistency_weight=feature_consistency_weight,
                slide_label_smoothing=slide_label_smoothing,
                device=device,
            )

            if sam is None:
                loss_dict = _compute_segmentation_step_losses(**loss_kwargs, use_amp=use_amp)
                total_step_loss = loss_dict['total']
                if not torch.isfinite(total_step_loss):
                    continue

                # step_idx == total_steps flushes a trailing partial accumulation window at
                # epoch end (mirrors train_epoch_graph's identical pattern, training_utils.py).
                # Known accepted edge case, same as that reference: if the very last
                # microbatch(es) of an epoch are skipped (non-finite loss above, or OOM below),
                # any gradient already accumulated earlier in that final window is never
                # stepped -- bounded to at most accum_steps-1 slides' worth, once per epoch,
                # only on an already-rare failure path.
                should_step = (step_idx % accum_steps == 0) or (step_idx == total_steps)

                if use_amp and device.type == 'cuda':
                    scaler.scale(total_step_loss / accum_steps).backward()
                    if should_step:
                        scaler.unscale_(optimizer)
                        if training_utils._has_non_finite_gradients(model):
                            logging.warning(
                                "Skipping segmentation optimizer step at batch %d due to non-finite gradients.",
                                step_idx,
                            )
                            optimizer.zero_grad(set_to_none=True)
                            scaler.update()
                        else:
                            if gradient_clip_value is not None:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                else:
                    (total_step_loss / accum_steps).backward()
                    if should_step:
                        if training_utils._has_non_finite_gradients(model):
                            logging.warning(
                                "Skipping segmentation optimizer step at batch %d due to non-finite gradients.",
                                step_idx,
                            )
                            optimizer.zero_grad(set_to_none=True)
                        else:
                            if gradient_clip_value is not None:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
            else:
                # SAM path -- __init__-time guards above already enforce accum_steps==1 and
                # use_amp==False, so this iteration IS the whole "step", no accumulation-boundary
                # bookkeeping needed. Every abort branch below `continue`s rather than falling
                # through to metrics accumulation (stricter than the non-SAM non-finite-gradient
                # case above, which still counts that microbatch) -- deliberate: SAM's aborts are
                # a newer, less battle-tested path, and picking which of two disagreeing passes'
                # logits would even be "representative" for logging is its own can of worms not
                # worth opening for what should be a rare failure mode.
                loss_dict = _compute_segmentation_step_losses(**loss_kwargs, use_amp=False)
                total_step_loss = loss_dict['total']
                if not torch.isfinite(total_step_loss):
                    continue

                total_step_loss.backward()
                if gradient_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)

                if not sam.grads_finite():
                    logging.warning(
                        "Skipping SAM step at batch %d due to non-finite gradients (pass 1, ascent direction).",
                        step_idx,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                sam.first_step(zero_grad=True)

                loss_dict = _compute_segmentation_step_losses(**loss_kwargs, use_amp=False)
                total_step_loss = loss_dict['total']
                if not torch.isfinite(total_step_loss):
                    logging.warning(
                        "Aborting SAM step at batch %d: non-finite loss at ascended weights "
                        "(pass 2) -- restoring pre-ascent weights, no optimizer step taken.",
                        step_idx,
                    )
                    sam.restore_only()
                    optimizer.zero_grad(set_to_none=True)
                    continue

                total_step_loss.backward()
                if gradient_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
                if not sam.grads_finite():
                    logging.warning(
                        "Aborting SAM step at batch %d: non-finite gradients at ascended weights "
                        "(pass 2) -- restoring pre-ascent weights, no optimizer step taken.",
                        step_idx,
                    )
                    sam.restore_only()
                    optimizer.zero_grad(set_to_none=True)
                    continue

                sam.second_step(zero_grad=True)

            region_loss = loss_dict['region_loss']
            slide_loss = loss_dict['slide_loss']
            aux_loss = loss_dict['aux_loss']
            consistency_loss = loss_dict['consistency_loss']
            patch_loss = loss_dict['patch_loss']
            augmentation_consistency_loss = loss_dict['augmentation_consistency_loss']
            feature_consistency_loss = loss_dict['feature_consistency_loss']
            slide_logits = loss_dict['slide_logits']
            has_region_supervision = loss_dict['has_region_supervision']
            region_logits = loss_dict['region_logits']
            region_targets = loss_dict['region_targets']
            fallback_mask = loss_dict['fallback_mask']
            has_patch_supervision = loss_dict['has_patch_supervision']
            patch_logits_kept = loss_dict['patch_logits_kept']
            patch_targets_kept = loss_dict['patch_targets_kept']
            patch_fallback_mask_kept = loss_dict['patch_fallback_mask_kept']
            if not has_region_supervision:
                skipped_unlabeled_batches += 1
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not _is_cuda_oom_error(exc):
                raise
            logging.warning(
                "CUDA OOM during training step (%s) -- skipping this graph instead of "
                "crashing the whole run. If this recurs often, lower "
                "max_patch_nodes_for_full_forward/max_patch_edges_for_full_forward in the "
                "config, or check `nvidia-smi` -- this machine is shared and free memory "
                "varies with other users' jobs.",
                _safe_graph_identity(data),
            )
            if sam is not None:
                # If pass 1's first_step() already perturbed the weights before this OOM hit
                # (e.g. during pass 2's forward/backward), restore_only() undoes it -- otherwise
                # the model is left permanently offset by rho and every later step trains from a
                # corrupted point. A no-op if first_step() never ran this iteration (empty
                # snapshot).
                sam.restore_only()
            # Deliberately NOT zeroing grads here: with grad_accum_steps > 1, earlier
            # microbatches in the current accumulation window may have already contributed
            # real, valid gradients -- wiping them on an unrelated OOM from a later microbatch
            # would silently discard that work. The window just proceeds with one fewer
            # contributing microbatch. No-op when grad_accum_steps=1 (grads are already zeroed
            # from the prior step there).
            gc.collect()
            torch.cuda.empty_cache()
            continue

        processed_batches += 1
        total_loss += float(total_step_loss.detach().item())
        total_region_loss += float(region_loss.detach().item())
        total_slide_loss += float(slide_loss.detach().item())
        total_aux_loss += float(aux_loss.detach().item()) if torch.is_tensor(aux_loss) else float(aux_loss)
        total_consistency_loss += float(consistency_loss.detach().item())
        total_patch_loss += float(patch_loss.detach().item())
        total_augmentation_consistency_loss += float(augmentation_consistency_loss.detach().item())
        total_feature_consistency_loss += float(feature_consistency_loss.detach().item())
        total_region_cls_correct, total_region_cls_total = _accumulate_cls_correct_total(
            loss_dict, 'region', total_region_cls_correct, total_region_cls_total
        )
        total_patch_cls_correct, total_patch_cls_total = _accumulate_cls_correct_total(
            loss_dict, 'patch', total_patch_cls_correct, total_patch_cls_total
        )
        _accumulate_cls_true_pred(loss_dict, 'region', all_region_cls_true, all_region_cls_pred)
        _accumulate_cls_true_pred(loss_dict, 'patch', all_patch_cls_true, all_patch_cls_pred)

        all_slide_labels.extend(slide_labels.detach().cpu().numpy().astype(float).tolist())
        all_slide_logits.extend(slide_logits.detach().cpu().numpy().astype(float).tolist())

        if has_region_supervision:
            fallback_count_kept = int(fallback_mask.sum().item())
            explicit_count_kept = int((~fallback_mask).sum().item())
            total_fallback_labels += fallback_count_kept
            total_explicit_labels += explicit_count_kept

            all_region_labels.extend(region_targets.detach().cpu().numpy().astype(float).tolist())
            all_region_logits.extend(region_logits.detach().cpu().numpy().astype(float).tolist())
            if explicit_count_kept > 0:
                explicit_mask = ~fallback_mask
                all_region_labels_explicit.extend(region_targets[explicit_mask].detach().cpu().numpy().astype(float).tolist())
                all_region_logits_explicit.extend(region_logits[explicit_mask].detach().cpu().numpy().astype(float).tolist())

        if has_patch_supervision:
            patch_fallback_count_kept = int(patch_fallback_mask_kept.sum().item())
            patch_explicit_count_kept = int((~patch_fallback_mask_kept).sum().item())
            total_patch_fallback_labels += patch_fallback_count_kept
            total_patch_explicit_labels += patch_explicit_count_kept

            all_patch_labels.extend(patch_targets_kept.detach().cpu().numpy().astype(float).tolist())
            all_patch_logits.extend(patch_logits_kept.detach().cpu().numpy().astype(float).tolist())
            if patch_explicit_count_kept > 0:
                patch_explicit_mask = ~patch_fallback_mask_kept
                all_patch_labels_explicit.extend(patch_targets_kept[patch_explicit_mask].detach().cpu().numpy().astype(float).tolist())
                all_patch_logits_explicit.extend(patch_logits_kept[patch_explicit_mask].detach().cpu().numpy().astype(float).tolist())

    region_metrics = _binary_metric_block(all_region_labels_explicit, all_region_logits_explicit)
    region_metrics_explicit = region_metrics
    slide_metrics = _binary_metric_block(all_slide_labels, all_slide_logits)
    patch_metrics = _binary_metric_block(all_patch_labels_explicit, all_patch_logits_explicit)
    patch_metrics_explicit = patch_metrics

    region_labels_np = np.asarray(all_region_labels, dtype=float) if all_region_labels else np.asarray([], dtype=float)
    region_logits_np = np.asarray(all_region_logits, dtype=float) if all_region_logits else np.asarray([], dtype=float)
    region_pos_count = int((region_labels_np >= 0.5).sum()) if region_labels_np.size > 0 else 0
    region_neg_count = int((region_labels_np < 0.5).sum()) if region_labels_np.size > 0 else 0
    region_pred_pos_count = int((region_logits_np > 0.0).sum()) if region_logits_np.size > 0 else 0
    region_pred_neg_count = int((region_logits_np <= 0.0).sum()) if region_logits_np.size > 0 else 0

    explicit_labels_np = np.asarray(all_region_labels_explicit, dtype=float) if all_region_labels_explicit else np.asarray([], dtype=float)
    explicit_logits_np = np.asarray(all_region_logits_explicit, dtype=float) if all_region_logits_explicit else np.asarray([], dtype=float)
    explicit_pos_count = int((explicit_labels_np >= 0.5).sum()) if explicit_labels_np.size > 0 else 0
    explicit_neg_count = int((explicit_labels_np < 0.5).sum()) if explicit_labels_np.size > 0 else 0
    explicit_pred_pos_count = int((explicit_logits_np > 0.0).sum()) if explicit_logits_np.size > 0 else 0
    explicit_pred_neg_count = int((explicit_logits_np <= 0.0).sum()) if explicit_logits_np.size > 0 else 0

    patch_labels_np = np.asarray(all_patch_labels, dtype=float) if all_patch_labels else np.asarray([], dtype=float)
    patch_logits_np = np.asarray(all_patch_logits, dtype=float) if all_patch_logits else np.asarray([], dtype=float)
    patch_pos_count = int((patch_labels_np >= 0.5).sum()) if patch_labels_np.size > 0 else 0
    patch_neg_count = int((patch_labels_np < 0.5).sum()) if patch_labels_np.size > 0 else 0
    patch_pred_pos_count = int((patch_logits_np > 0.0).sum()) if patch_logits_np.size > 0 else 0
    patch_pred_neg_count = int((patch_logits_np <= 0.0).sum()) if patch_logits_np.size > 0 else 0

    if processed_batches <= 0:
        raise ValueError(
            f"No graph batches were available for segmentation training."
        )

    region_cls_kappa, region_cls_macro_f1, patch_cls_kappa, patch_cls_macro_f1 = _compute_cls_kappa_macro_f1(
        all_region_cls_true, all_region_cls_pred, all_patch_cls_true, all_patch_cls_pred
    )

    return {
        'loss': total_loss / processed_batches,
        'region_loss': total_region_loss / processed_batches,
        'slide_loss': total_slide_loss / processed_batches,
        'aux_loss': total_aux_loss / processed_batches,
        'consistency_loss': total_consistency_loss / processed_batches,
        'patch_loss': total_patch_loss / processed_batches,
        'augmentation_consistency_loss': total_augmentation_consistency_loss / processed_batches,
        'feature_consistency_loss': total_feature_consistency_loss / processed_batches,
        'accuracy': region_metrics.get('accuracy', 0.0),
        'sensitivity': region_metrics.get('sensitivity', 0.0),
        'specificity': region_metrics.get('specificity', 0.0),
        'precision': region_metrics.get('precision', 0.0),
        'f1': region_metrics.get('f1', 0.0),
        'auc': region_metrics.get('auc', 50.0),
        'region_accuracy': region_metrics.get('accuracy', 0.0),
        'region_f1': region_metrics.get('f1', 0.0),
        'region_f2': region_metrics.get('f2', 0.0),
        'region_f1_best_threshold': region_metrics.get('f1_best_threshold', region_metrics.get('f1', 0.0)),
        'region_best_threshold': region_metrics.get('best_threshold', 0.5),
        'region_auc': region_metrics.get('auc', 50.0),
        'region_f1_explicit': region_metrics_explicit.get('f1', 0.0),
        'region_f2_explicit': region_metrics_explicit.get('f2', 0.0),
        'region_f1_best_threshold_explicit': region_metrics_explicit.get('f1_best_threshold', region_metrics_explicit.get('f1', 0.0)),
        'region_auc_explicit': region_metrics_explicit.get('auc', 50.0),
        'slide_accuracy': slide_metrics.get('accuracy', 0.0),
        'slide_sensitivity': slide_metrics.get('sensitivity', 0.0),
        'slide_specificity': slide_metrics.get('specificity', 0.0),
        'slide_f1': slide_metrics.get('f1', 0.0),
        'slide_f2': slide_metrics.get('f2', 0.0),
        'slide_balanced_accuracy': slide_metrics.get('balanced_accuracy', 0.0),
        'slide_auc': slide_metrics.get('auc', 50.0),
        'patch_f1': patch_metrics.get('f1', 0.0),
        'patch_f2': patch_metrics.get('f2', 0.0),
        'patch_f1_best_threshold': patch_metrics.get('f1_best_threshold', patch_metrics.get('f1', 0.0)),
        'patch_best_threshold': patch_metrics.get('best_threshold', 0.5),
        'patch_auc': patch_metrics.get('auc', 50.0),
        'patch_f1_explicit': patch_metrics_explicit.get('f1', 0.0),
        'patch_f2_explicit': patch_metrics_explicit.get('f2', 0.0),
        'patch_f1_best_threshold_explicit': patch_metrics_explicit.get('f1_best_threshold', patch_metrics_explicit.get('f1', 0.0)),
        'patch_auc_explicit': patch_metrics_explicit.get('auc', 50.0),
        'processed_batches': int(processed_batches),
        'skipped_unlabeled_batches': int(skipped_unlabeled_batches),
        'labeled_regions': int(len(all_region_labels)),
        'explicit_regions': int(total_explicit_labels),
        'fallback_regions': int(total_fallback_labels),
        'region_pos_count': region_pos_count,
        'region_neg_count': region_neg_count,
        'region_pred_pos_count': region_pred_pos_count,
        'region_pred_neg_count': region_pred_neg_count,
        'explicit_pos_count': explicit_pos_count,
        'explicit_neg_count': explicit_neg_count,
        'explicit_pred_pos_count': explicit_pred_pos_count,
        'explicit_pred_neg_count': explicit_pred_neg_count,
        'labeled_patches': int(len(all_patch_labels)),
        'explicit_patches': int(total_patch_explicit_labels),
        'fallback_patches': int(total_patch_fallback_labels),
        'patch_pos_count': patch_pos_count,
        'patch_neg_count': patch_neg_count,
        'patch_pred_pos_count': patch_pred_pos_count,
        'patch_pred_neg_count': patch_pred_neg_count,
        # Accuracy for TissueRegionGraphModel's optional region_cls_target_field/
        # patch_cls_target_field multiclass head (e.g. GSDB tissue_type) -- None whenever that
        # head isn't enabled (region_cls_class_names/patch_cls_class_names unset in the model
        # config) or nothing was supervised this epoch, NOT 0.0, so callers/logging can tell
        # "not applicable" apart from "trained to 0% accuracy".
        'region_cls_accuracy': (total_region_cls_correct / total_region_cls_total) if total_region_cls_total > 0 else None,
        'region_cls_labeled_count': int(total_region_cls_total),
        'region_cls_kappa': region_cls_kappa,
        'region_cls_macro_f1': region_cls_macro_f1,
        'patch_cls_accuracy': (total_patch_cls_correct / total_patch_cls_total) if total_patch_cls_total > 0 else None,
        'patch_cls_labeled_count': int(total_patch_cls_total),
        'patch_cls_kappa': patch_cls_kappa,
        'patch_cls_macro_f1': patch_cls_macro_f1,
    }


@torch.no_grad()
def _validate_epoch_graph_segmentation(
    model: nn.Module,
    val_loader: GeoDataLoader,
    slide_criterion: nn.Module,
    device: torch.device,
    region_loss_weight: float,
    slide_loss_weight: float,
    aux_loss_weight: float,
    fallback_region_weight: float,
    max_fallback_to_explicit_ratio: float,
    prefer_explicit_only_when_available: bool,
    region_slide_consistency_weight: float,
    unlabeled_region_policy: str,
    node_type: str = 'region',
    negative_label_source: str = 'slide_derived',
    patch_loss_weight: float = 0.0,
):
    """Runs one validation epoch over the graph segmentation val loader."""
    model.eval()

    total_loss = 0.0
    total_region_loss = 0.0
    total_slide_loss = 0.0
    total_aux_loss = 0.0
    total_consistency_loss = 0.0
    total_patch_loss = 0.0
    total_region_cls_correct = 0
    total_region_cls_total = 0
    total_patch_cls_correct = 0
    total_patch_cls_total = 0
    all_region_cls_true: list[int] = []
    all_region_cls_pred: list[int] = []
    all_patch_cls_true: list[int] = []
    all_patch_cls_pred: list[int] = []
    processed_batches = 0
    skipped_unlabeled_batches = 0
    total_fallback_labels = 0
    total_explicit_labels = 0
    total_patch_fallback_labels = 0
    total_patch_explicit_labels = 0

    all_region_labels: list[float] = []
    all_region_logits: list[float] = []
    all_region_labels_explicit: list[float] = []
    all_region_logits_explicit: list[float] = []
    all_slide_labels: list[float] = []
    all_slide_logits: list[float] = []
    all_patch_labels: list[float] = []
    all_patch_logits: list[float] = []
    all_patch_labels_explicit: list[float] = []
    all_patch_logits_explicit: list[float] = []

    for data in tqdm(val_loader, desc="Validating (Graph Segmentation)", leave=False):
        try:
            data = data.to(device)

            if hasattr(data, 'y') and data.y is not None:
                slide_labels = data.y.view(-1).to(device).float()
            else:
                slide_labels = data['slide'].y.view(-1).to(device).float()
            slide_labels = training_utils._sanitize_binary_labels(slide_labels)

            model_output = model(data)
            slide_logits, aux_loss = training_utils._unpack_graph_model_output(model_output, device)
            slide_logits = training_utils._sanitize_logits_for_loss(slide_logits, multiclass=False)
            slide_loss = slide_criterion(slide_logits, slide_labels)
            consistency_loss = _region_slide_consistency_loss(
                model_output=model_output,
                data=data,
                slide_labels=slide_labels,
                device=device,
                node_type=node_type,
            )

            region_logits, region_targets, num_labeled, fallback_count, fallback_mask = _extract_region_supervision_tensors(
                model_output,
                data,
                device,
                slide_labels=slide_labels,
                unlabeled_region_policy=unlabeled_region_policy,
                node_type=node_type,
                negative_label_source=negative_label_source,
            )
            has_region_supervision = num_labeled > 0
            if has_region_supervision:
                region_logits, region_targets, fallback_mask = _apply_fallback_subsampling(
                    region_logits,
                    region_targets,
                    fallback_mask,
                    max_fallback_to_explicit_ratio=max_fallback_to_explicit_ratio,
                    prefer_explicit_only_when_available=prefer_explicit_only_when_available,
                )

                pos = float((region_targets >= 0.5).sum().item())
                neg = float((region_targets < 0.5).sum().item())
                if pos > 0.0 and neg > 0.0:
                    region_pos_weight = torch.tensor(min(12.0, max(1.0, neg / pos)), device=device)
                else:
                    region_pos_weight = torch.tensor(1.0, device=device)

                per_sample = nn.BCEWithLogitsLoss(pos_weight=region_pos_weight, reduction='none')(
                    region_logits,
                    region_targets,
                )
                sample_weights = torch.ones_like(per_sample)
                if torch.any(fallback_mask):
                    sample_weights = torch.where(
                        fallback_mask,
                        torch.full_like(sample_weights, float(fallback_region_weight)),
                        sample_weights,
                    )
                region_loss = (per_sample * sample_weights).mean()
            else:
                skipped_unlabeled_batches += 1
                region_loss = torch.zeros((), dtype=slide_loss.dtype, device=device)

            if patch_loss_weight > 0.0:
                patch_loss, has_patch_supervision, patch_logits_kept, patch_targets_kept, patch_fallback_mask_kept = (
                    _compute_patch_fallback_loss(
                        model_output=model_output,
                        data=data,
                        device=device,
                        slide_labels=slide_labels,
                        unlabeled_region_policy=unlabeled_region_policy,
                        pseudo_label_top_fraction=0.0,
                        negative_label_source=negative_label_source,
                        max_fallback_to_explicit_ratio=max_fallback_to_explicit_ratio,
                        prefer_explicit_only_when_available=prefer_explicit_only_when_available,
                        fallback_region_weight=fallback_region_weight,
                        loss_dtype=slide_loss.dtype,
                    )
                )
            else:
                patch_loss = torch.zeros((), dtype=slide_loss.dtype, device=device)
                has_patch_supervision, patch_logits_kept, patch_targets_kept, patch_fallback_mask_kept = False, None, None, None

            total_step_loss = (
                float(region_loss_weight) * region_loss
                + float(slide_loss_weight) * slide_loss
                + float(aux_loss_weight) * aux_loss
                + float(region_slide_consistency_weight) * consistency_loss
                + float(patch_loss_weight) * patch_loss
            )

            if not torch.isfinite(total_step_loss):
                continue
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not _is_cuda_oom_error(exc):
                raise
            logging.warning(
                "CUDA OOM during validation step (%s) -- skipping this graph instead of "
                "crashing the whole run.",
                _safe_graph_identity(data),
            )
            gc.collect()
            torch.cuda.empty_cache()
            continue

        processed_batches += 1
        total_loss += float(total_step_loss.detach().item())
        total_region_loss += float(region_loss.detach().item())
        total_slide_loss += float(slide_loss.detach().item())
        total_aux_loss += float(aux_loss.detach().item()) if torch.is_tensor(aux_loss) else float(aux_loss)
        total_consistency_loss += float(consistency_loss.detach().item())
        total_patch_loss += float(patch_loss.detach().item())
        total_region_cls_correct, total_region_cls_total = _accumulate_cls_correct_total(
            model_output if isinstance(model_output, dict) else {}, 'region', total_region_cls_correct, total_region_cls_total
        )
        total_patch_cls_correct, total_patch_cls_total = _accumulate_cls_correct_total(
            model_output if isinstance(model_output, dict) else {}, 'patch', total_patch_cls_correct, total_patch_cls_total
        )
        _accumulate_cls_true_pred(model_output if isinstance(model_output, dict) else {}, 'region', all_region_cls_true, all_region_cls_pred)
        _accumulate_cls_true_pred(model_output if isinstance(model_output, dict) else {}, 'patch', all_patch_cls_true, all_patch_cls_pred)

        all_slide_labels.extend(slide_labels.detach().cpu().numpy().astype(float).tolist())
        all_slide_logits.extend(slide_logits.detach().cpu().numpy().astype(float).tolist())

        if has_region_supervision:
            fallback_count_kept = int(fallback_mask.sum().item())
            explicit_count_kept = int((~fallback_mask).sum().item())
            total_fallback_labels += fallback_count_kept
            total_explicit_labels += explicit_count_kept

            all_region_labels.extend(region_targets.detach().cpu().numpy().astype(float).tolist())
            all_region_logits.extend(region_logits.detach().cpu().numpy().astype(float).tolist())
            if explicit_count_kept > 0:
                explicit_mask = ~fallback_mask
                all_region_labels_explicit.extend(region_targets[explicit_mask].detach().cpu().numpy().astype(float).tolist())
                all_region_logits_explicit.extend(region_logits[explicit_mask].detach().cpu().numpy().astype(float).tolist())

        if has_patch_supervision:
            patch_fallback_count_kept = int(patch_fallback_mask_kept.sum().item())
            patch_explicit_count_kept = int((~patch_fallback_mask_kept).sum().item())
            total_patch_fallback_labels += patch_fallback_count_kept
            total_patch_explicit_labels += patch_explicit_count_kept

            all_patch_labels.extend(patch_targets_kept.detach().cpu().numpy().astype(float).tolist())
            all_patch_logits.extend(patch_logits_kept.detach().cpu().numpy().astype(float).tolist())
            if patch_explicit_count_kept > 0:
                patch_explicit_mask = ~patch_fallback_mask_kept
                all_patch_labels_explicit.extend(patch_targets_kept[patch_explicit_mask].detach().cpu().numpy().astype(float).tolist())
                all_patch_logits_explicit.extend(patch_logits_kept[patch_explicit_mask].detach().cpu().numpy().astype(float).tolist())

    if processed_batches <= 0:
        raise ValueError("No graph batches were available for segmentation validation.")

    # See _compute_cls_kappa_macro_f1 for the full rationale (Cohen's Kappa/macro-F1 from the
    # epoch's full (true_class, pred_class) pairs, used for segmentation.monitor_metric:
    # region_cls_kappa/patch_cls_kappa checkpoint selection).
    region_cls_kappa, region_cls_macro_f1, patch_cls_kappa, patch_cls_macro_f1 = _compute_cls_kappa_macro_f1(
        all_region_cls_true, all_region_cls_pred, all_patch_cls_true, all_patch_cls_pred
    )

    region_metrics = _binary_metric_block(all_region_labels_explicit, all_region_logits_explicit)
    region_metrics_explicit = region_metrics
    slide_metrics = _binary_metric_block(all_slide_labels, all_slide_logits)
    patch_metrics = _binary_metric_block(all_patch_labels_explicit, all_patch_logits_explicit)
    patch_metrics_explicit = patch_metrics

    region_labels_np = np.asarray(all_region_labels, dtype=float) if all_region_labels else np.asarray([], dtype=float)
    region_logits_np = np.asarray(all_region_logits, dtype=float) if all_region_logits else np.asarray([], dtype=float)
    region_pos_count = int((region_labels_np >= 0.5).sum()) if region_labels_np.size > 0 else 0
    region_neg_count = int((region_labels_np < 0.5).sum()) if region_labels_np.size > 0 else 0
    region_pred_pos_count = int((region_logits_np > 0.0).sum()) if region_logits_np.size > 0 else 0
    region_pred_neg_count = int((region_logits_np <= 0.0).sum()) if region_logits_np.size > 0 else 0

    explicit_labels_np = np.asarray(all_region_labels_explicit, dtype=float) if all_region_labels_explicit else np.asarray([], dtype=float)
    explicit_logits_np = np.asarray(all_region_logits_explicit, dtype=float) if all_region_logits_explicit else np.asarray([], dtype=float)
    explicit_pos_count = int((explicit_labels_np >= 0.5).sum()) if explicit_labels_np.size > 0 else 0
    explicit_neg_count = int((explicit_labels_np < 0.5).sum()) if explicit_labels_np.size > 0 else 0
    explicit_pred_pos_count = int((explicit_logits_np > 0.0).sum()) if explicit_logits_np.size > 0 else 0
    explicit_pred_neg_count = int((explicit_logits_np <= 0.0).sum()) if explicit_logits_np.size > 0 else 0

    patch_labels_np = np.asarray(all_patch_labels, dtype=float) if all_patch_labels else np.asarray([], dtype=float)
    patch_logits_np = np.asarray(all_patch_logits, dtype=float) if all_patch_logits else np.asarray([], dtype=float)
    patch_pos_count = int((patch_labels_np >= 0.5).sum()) if patch_labels_np.size > 0 else 0
    patch_neg_count = int((patch_labels_np < 0.5).sum()) if patch_labels_np.size > 0 else 0
    patch_pred_pos_count = int((patch_logits_np > 0.0).sum()) if patch_logits_np.size > 0 else 0
    patch_pred_neg_count = int((patch_logits_np <= 0.0).sum()) if patch_logits_np.size > 0 else 0

    return {
        'loss': total_loss / processed_batches,
        'region_loss': total_region_loss / processed_batches,
        'slide_loss': total_slide_loss / processed_batches,
        'aux_loss': total_aux_loss / processed_batches,
        'consistency_loss': total_consistency_loss / processed_batches,
        'patch_loss': total_patch_loss / processed_batches,
        'accuracy': region_metrics.get('accuracy', 0.0),
        'sensitivity': region_metrics.get('sensitivity', 0.0),
        'specificity': region_metrics.get('specificity', 0.0),
        'precision': region_metrics.get('precision', 0.0),
        'f1': region_metrics.get('f1', 0.0),
        'auc': region_metrics.get('auc', 50.0),
        'region_accuracy': region_metrics.get('accuracy', 0.0),
        'region_f1': region_metrics.get('f1', 0.0),
        'region_f2': region_metrics.get('f2', 0.0),
        'region_f1_best_threshold': region_metrics.get('f1_best_threshold', region_metrics.get('f1', 0.0)),
        'region_best_threshold': region_metrics.get('best_threshold', 0.5),
        'region_auc': region_metrics.get('auc', 50.0),
        'region_f1_explicit': region_metrics_explicit.get('f1', 0.0),
        'region_f2_explicit': region_metrics_explicit.get('f2', 0.0),
        'region_f1_best_threshold_explicit': region_metrics_explicit.get('f1_best_threshold', region_metrics_explicit.get('f1', 0.0)),
        'region_auc_explicit': region_metrics_explicit.get('auc', 50.0),
        'slide_accuracy': slide_metrics.get('accuracy', 0.0),
        'slide_sensitivity': slide_metrics.get('sensitivity', 0.0),
        'slide_specificity': slide_metrics.get('specificity', 0.0),
        'slide_f1': slide_metrics.get('f1', 0.0),
        'slide_f2': slide_metrics.get('f2', 0.0),
        'slide_balanced_accuracy': slide_metrics.get('balanced_accuracy', 0.0),
        'slide_auc': slide_metrics.get('auc', 50.0),
        'patch_f1': patch_metrics.get('f1', 0.0),
        'patch_f2': patch_metrics.get('f2', 0.0),
        'patch_f1_best_threshold': patch_metrics.get('f1_best_threshold', patch_metrics.get('f1', 0.0)),
        'patch_best_threshold': patch_metrics.get('best_threshold', 0.5),
        'patch_auc': patch_metrics.get('auc', 50.0),
        'patch_f1_explicit': patch_metrics_explicit.get('f1', 0.0),
        'patch_f2_explicit': patch_metrics_explicit.get('f2', 0.0),
        'patch_f1_best_threshold_explicit': patch_metrics_explicit.get('f1_best_threshold', patch_metrics_explicit.get('f1', 0.0)),
        'patch_auc_explicit': patch_metrics_explicit.get('auc', 50.0),
        'processed_batches': int(processed_batches),
        'skipped_unlabeled_batches': int(skipped_unlabeled_batches),
        'labeled_regions': int(len(all_region_labels)),
        'explicit_regions': int(total_explicit_labels),
        'fallback_regions': int(total_fallback_labels),
        'region_pos_count': region_pos_count,
        'region_neg_count': region_neg_count,
        'region_pred_pos_count': region_pred_pos_count,
        'region_pred_neg_count': region_pred_neg_count,
        'explicit_pos_count': explicit_pos_count,
        'explicit_neg_count': explicit_neg_count,
        'explicit_pred_pos_count': explicit_pred_pos_count,
        'explicit_pred_neg_count': explicit_pred_neg_count,
        'labeled_patches': int(len(all_patch_labels)),
        'explicit_patches': int(total_patch_explicit_labels),
        'fallback_patches': int(total_patch_fallback_labels),
        'patch_pos_count': patch_pos_count,
        'patch_neg_count': patch_neg_count,
        'patch_pred_pos_count': patch_pred_pos_count,
        'patch_pred_neg_count': patch_pred_neg_count,
        'region_cls_accuracy': (total_region_cls_correct / total_region_cls_total) if total_region_cls_total > 0 else None,
        'region_cls_labeled_count': int(total_region_cls_total),
        'region_cls_kappa': region_cls_kappa,
        'region_cls_macro_f1': region_cls_macro_f1,
        'patch_cls_accuracy': (total_patch_cls_correct / total_patch_cls_total) if total_patch_cls_total > 0 else None,
        'patch_cls_labeled_count': int(total_patch_cls_total),
        'patch_cls_kappa': patch_cls_kappa,
        'patch_cls_macro_f1': patch_cls_macro_f1,
        # Raw (true_class, pred_class) pairs for the epoch -- unlike the scalar accuracy/kappa/
        # macro_f1 above, these let a caller build a FULL classification report (per-class
        # precision/recall/f1/support + confusion matrix), not just aggregate scores. Only
        # returned from the VALIDATION epoch (not the train one) since that's what
        # train_graph_segmentation_model's multiclass-only report logs -- train-epoch pairs would
        # just be a noisier, less meaningful version of the same thing.
        'region_cls_true': all_region_cls_true,
        'region_cls_pred': all_region_cls_pred,
        'patch_cls_true': all_patch_cls_true,
        'patch_cls_pred': all_patch_cls_pred,
    }


def _resolve_consistency_settings(segmentation_cfg: dict):
    """Read the consistency settings and make `augmentation_consistency_weight <= 0` switch the whole
    mechanism off: no perturbed pass and no feature-consistency term. The perturbed pass is created
    whenever the spec list is non-empty and the feature term rides on that pass, so a config with weight 0
    but specs and `feature_consistency_weight > 0` would otherwise still regularize."""
    weight = float(segmentation_cfg.get('augmentation_consistency_weight', 0.0))
    specs = segmentation_cfg.get('augmentation_consistency_specs', []) or []
    p = float(segmentation_cfg.get('augmentation_consistency_p', 1.0))
    feature_weight = float(segmentation_cfg.get('feature_consistency_weight', 0.0))
    if weight <= 0.0:
        if specs or feature_weight > 0.0:
            logging.info(
                "augmentation_consistency_weight <= 0: consistency regularization is off (no perturbed pass, "
                "feature_consistency_weight=%.3f ignored).", feature_weight,
            )
        specs, feature_weight = [], 0.0
    return weight, specs, p, feature_weight


def train_graph_segmentation_model(args, model, train_dataset, val_dataset, config, optimizer_name, momentum, paths):
    """Trains a graph model for tumor-region segmentation."""
    log_training_settings(args, config, optimizer_name, momentum)

    common_config = config.get('common', {}) if isinstance(config.get('common', {}), dict) else {}

    def _cfg(name, default):
        return common_config.get(name, config.get(name, default))

    segmentation_cfg = config.get('segmentation', {}) if isinstance(config.get('segmentation', {}), dict) else {}

    segmentation_target = str(segmentation_cfg.get('target', 'region')).strip().lower()
    if segmentation_target not in {'region', 'patch'}:
        raise ValueError(f"segmentation.target must be 'region' or 'patch', got {segmentation_target!r}")
    logging.info("Graph segmentation training target: %s", segmentation_target)

    gradient_clip_value = _cfg('gradient_clipping', None)
    use_mixed_precision = bool(_cfg('mixed_precision', False))
    save_frequency = int(_cfg('save_frequency', 5))
    # Was read from config but never wired into this training loop (bug, found 2026-08-13):
    # every optimizer.step() fired per single graph regardless of this value, i.e. batch_size=1
    # was the REAL effective batch size the whole time despite grad_accum_steps=4 in the config
    # implying 4 slides' gradients should be averaged first. Now wired -- see
    # _train_epoch_graph_segmentation's accum_steps/should_step logic (mirrors
    # training_utils.train_epoch_graph's identical pattern, used by region_slide_net).
    grad_accum_steps = int(_cfg('grad_accum_steps', 1))
    if grad_accum_steps > 1:
        logging.info(
            "Gradient accumulation enabled for segmentation training: accum_steps=%d "
            "(effective batch size ~%d slides per optimizer step).",
            grad_accum_steps, grad_accum_steps,
        )

    # Sharpness-Aware Minimization (GraphSAM in training_utils.py) -- off by default. Seeks
    # flat loss-surface minima, documented in the DG literature to help out-of-domain
    # generalization from a single training domain, no domain labels needed. Two forced
    # consequences when enabled, not user-configurable overrides: (1) mixed_precision is
    # disabled regardless of the config's own mixed_precision value -- GradScaler's single
    # scale/step/update cycle isn't designed for SAM's two backward passes per step, and
    # correctness was chosen over speed rather than attempting an unvalidated double-cycle
    # workaround; (2) grad_accum_steps is forced to 1 -- ascending mid-accumulation-window would
    # mean the ascent direction reflects only one microbatch, not the full windowed loss
    # landscape SAM's rationale depends on. Both are logged loudly since they silently override
    # other config values.
    sam_enabled = bool(segmentation_cfg.get('sam_enabled', False))
    sam_rho = float(segmentation_cfg.get('sam_rho', 0.05))
    if sam_enabled:
        if use_mixed_precision:
            logging.warning(
                "sam_enabled=True: forcing mixed_precision off (was True in config) -- "
                "GradScaler is not designed for SAM's two-backward-pass-per-step structure."
            )
            use_mixed_precision = False
        if grad_accum_steps != 1:
            logging.warning(
                "sam_enabled=True: forcing grad_accum_steps=1 (was %d in config) -- SAM's "
                "ascent direction must reflect the full accumulation window, which an "
                "unmodified accumulation loop does not guarantee mid-window.",
                grad_accum_steps,
            )
            grad_accum_steps = 1
        logging.info(
            "Sharpness-Aware Minimization (SAM) enabled for segmentation training: rho=%.4f "
            "(roughly ~2-3x per-step compute: closure evaluated twice per optimizer step, on "
            "top of augmentation_consistency's own extra forward when it fires).",
            sam_rho,
        )

    region_loss_weight = float(segmentation_cfg.get('region_loss_weight', 1.0))
    slide_loss_weight = float(segmentation_cfg.get('slide_loss_weight', 0.08))
    aux_loss_weight = float(segmentation_cfg.get('aux_loss_weight', 1.0))
    fallback_region_weight = float(segmentation_cfg.get('fallback_region_weight', 0.20))
    max_fallback_to_explicit_ratio = float(segmentation_cfg.get('max_fallback_to_explicit_ratio', 0.75))
    prefer_explicit_only_when_available = bool(segmentation_cfg.get('prefer_explicit_only_when_available', False))
    region_slide_consistency_weight = float(segmentation_cfg.get('region_slide_consistency_weight', 0.30))
    slide_label_smoothing = float(segmentation_cfg.get('slide_label_smoothing', 0.0))
    patch_loss_weight = float(segmentation_cfg.get('patch_loss_weight', 0.0))
    if patch_loss_weight > 0.0:
        logging.info(
            "Patch-level loss enabled: patch_loss_weight=%.3f (patches without their own "
            "annotation fall back to their parent region's resolved label).",
            patch_loss_weight,
        )

    # True for configs like model_config_tissue_region_tissue_type_only.yaml / _merged_other.yaml
    # -- region_cls/patch_cls is the ONLY thing actually training (every binary segmentation loss
    # term zeroed). Per-epoch logging below is split on this: the binary segmentation metric wall
    # (Loss/Accuracy/Sensitivity/.../Dice/IoU-style blocks -- meaningless when nothing computed
    # them from a real gradient) is skipped entirely, and a proper multiclass classification
    # report (confusion matrix + full per-class table, see _log_multiclass_classification_report)
    # takes its place as the primary per-epoch output instead of being three lines buried under
    # an unrelated wall of numbers.
    is_multiclass_only_run = (
        (getattr(model, 'region_cls_enabled', False) or getattr(model, 'patch_cls_enabled', False))
        and region_loss_weight <= 0.0
        and slide_loss_weight <= 0.0
        and patch_loss_weight <= 0.0
    )
    # Consistency regularization for OOD robustness (see _compute_augmentation_consistency_loss
    # and docs/experiments/cross_domain_experiments_details.md, E3): off by default (weight 0.0 / empty specs) --
    # current behavior unchanged. When on, doubles per-step forward-pass compute (train only --
    # never applied during validation, so early-stopping/checkpoint-selection stays judged on
    # real, unperturbed predictions).
    (augmentation_consistency_weight, augmentation_consistency_specs,
     augmentation_consistency_p, feature_consistency_weight) = _resolve_consistency_settings(segmentation_cfg)
    if augmentation_consistency_weight > 0.0 and augmentation_consistency_specs:
        logging.info(
            "Augmentation consistency regularization enabled: weight=%.3f p=%.2f specs=%s "
            "(doubles per-step forward-pass compute when it fires).",
            augmentation_consistency_weight, augmentation_consistency_p, augmentation_consistency_specs,
        )
    # Feature-space consistency (see _augmentation_consistency_loss_from_perturbed in this file
    # and model.py's "slide_h" output key): off by default (weight 0.0) -- current behavior
    # unchanged. Rides the SAME clean/perturbed forward pair augmentation_consistency_weight
    # already produces (no extra compute of its own beyond a cosine_similarity call) -- only
    # meaningful when augmentation_consistency_weight > 0 and its specs are non-empty, since
    # that's what determines whether a perturbed pass happens at all this step.
    if feature_consistency_weight > 0.0:
        logging.info(
            "Feature-space consistency regularization enabled: weight=%.3f (constrains slide_h "
            "itself to be stable under perturbation, not just the final prediction).",
            feature_consistency_weight,
        )
    # Manifold Mixup (see model.py's _apply_manifold_mixup docstring): off by default
    # (manifold_mixup_p=0.0) -- current behavior unchanged. When on, fires with probability
    # manifold_mixup_p per training step, loading one extra graph from disk via the dataset's
    # lightweight second-graph loader (same one graph_mixup uses) -- train only, never applied
    # during validation.
    manifold_mixup_p = float(segmentation_cfg.get('manifold_mixup_p', 0.0))
    manifold_mixup_alpha = float(segmentation_cfg.get('manifold_mixup_alpha', 0.4))
    if manifold_mixup_p > 0.0:
        logging.info(
            "Manifold Mixup enabled: p=%.2f alpha=%.2f (blends region-branch hidden states + "
            "slide label with a randomly resampled second training graph).",
            manifold_mixup_p, manifold_mixup_alpha,
        )
    unlabeled_region_policy = str(segmentation_cfg.get('unlabeled_region_policy', 'slide_aware')).strip().lower()
    negative_label_source = resolve_negative_label_source(config)
    monitor_metric_name = str(segmentation_cfg.get('monitor_metric', 'region_f1_best_threshold')).strip().lower()
    monitor_mode = 'min' if monitor_metric_name in {'loss', 'region_loss', 'slide_loss'} else 'max'

    pseudo_label_top_fraction = float(segmentation_cfg.get('pseudo_label_top_fraction', 0.0))
    pseudo_label_warmup_epochs = int(segmentation_cfg.get('pseudo_label_warmup_epochs', 10))
    if pseudo_label_top_fraction > 0.0:
        logging.info(
            "MIL pseudo-labeling enabled for unannotated tumor-positive slides: top_fraction=%.3f, "
            "warmup_epochs=%d (inactive until then, falls back to current slide_aware behavior)",
            pseudo_label_top_fraction,
            pseudo_label_warmup_epochs,
        )

    configured_device = str(_cfg('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    device = torch.device(configured_device if torch.cuda.is_available() or configured_device == 'cpu' else 'cpu')
    if int(config.get('batch_size', 1)) > 1:
        model = training_utils.handle_multi_gpu_model(model)
    model = model.to(device)

    # Inverse-frequency class weights for TissueRegionGraphModel's optional region_cls_target_
    # field/patch_cls_target_field multiclass head (e.g. GSDB tissue_type -- see
    # model_config_tissue_region_tissue_type_only.yaml). Off by default (None on the model =
    # unweighted CrossEntropyLoss, current behavior unchanged) whenever the head isn't enabled.
    # Must happen here, not in build_model()/__init__: the training dataset (needed to compute a
    # meaningful weighting) isn't available at model-construction time (Holder.load_model() runs
    # before load_data()). Computed ONCE per training run (a full one-time scan of train_dataset,
    # see infer_node_multiclass_weights_from_dataset's docstring for the cost), not per epoch.
    if getattr(model, 'region_cls_enabled', False):
        region_cls_weights = training_utils.infer_node_multiclass_weights_from_dataset(
            train_dataset, 'region', model.region_cls_target_field, model.region_cls_class_names,
        )
        if region_cls_weights is not None:
            model.region_cls_weight_tensor = region_cls_weights.to(device)
            logging.info(
                "region_cls (%s) inverse-frequency class weights: %s",
                model.region_cls_target_field,
                {name: round(float(w), 3) for name, w in zip(model.region_cls_class_names, region_cls_weights.tolist())},
            )
        else:
            logging.warning(
                "region_cls_enabled but could not infer class weights from train_dataset -- "
                "falling back to unweighted CrossEntropyLoss for region_cls_loss."
            )
    if getattr(model, 'patch_cls_enabled', False):
        patch_cls_weights = training_utils.infer_node_multiclass_weights_from_dataset(
            train_dataset, 'patch', model.patch_cls_target_field, model.patch_cls_class_names,
        )
        if patch_cls_weights is not None:
            model.patch_cls_weight_tensor = patch_cls_weights.to(device)
            logging.info(
                "patch_cls (%s) inverse-frequency class weights: %s",
                model.patch_cls_target_field,
                {name: round(float(w), 3) for name, w in zip(model.patch_cls_class_names, patch_cls_weights.tolist())},
            )
        else:
            logging.warning(
                "patch_cls_enabled but could not infer class weights from train_dataset -- "
                "falling back to unweighted CrossEntropyLoss for patch_cls_loss."
            )

    # Graph-level oversampling for region_cls/patch_cls's rare classes -- see
    # infer_graph_oversampling_weights's docstring. Per-node class weighting above cannot fix a
    # graph coming up entirely without a rare class at batch_size=1 (one graph = one gradient
    # step); this instead makes WeightedRandomSampler visit graphs that DO contain rare-class
    # content more often, so more gradient steps per epoch actually carry a signal for them.
    # None (both when no region_cls/patch_cls head is enabled, and when every graph ends up with
    # an identical weight) falls back to the original shuffle=True below -- zero effect on any
    # run that doesn't use these heads.
    graph_sampler_specs: List[tuple] = []
    if getattr(model, 'region_cls_enabled', False):
        graph_sampler_specs.append(('region', model.region_cls_target_field, model.region_cls_class_names))
    if getattr(model, 'patch_cls_enabled', False):
        graph_sampler_specs.append(('patch', model.patch_cls_target_field, model.patch_cls_class_names))

    graph_sampling_weights = None
    if graph_sampler_specs:
        graph_sampling_weights = training_utils.infer_graph_oversampling_weights(train_dataset, graph_sampler_specs)
        if graph_sampling_weights is not None:
            n_boosted = sum(1 for w in graph_sampling_weights if w > 1.0 + 1e-9)
            logging.info(
                "Graph-level oversampling for region_cls/patch_cls: %d/%d training graphs "
                "boosted (per-graph weight range %.2f-%.2f) via WeightedRandomSampler.",
                n_boosted, len(graph_sampling_weights),
                min(graph_sampling_weights), max(graph_sampling_weights),
            )

    effective_pos_weight = float(config.get('pos_weight', 1.0))
    labels = _infer_binary_graph_labels(train_dataset)
    if labels:
        labels_np = np.asarray(labels, dtype=float)
        positives = int((labels_np > 0.5).sum())
        negatives = int((labels_np <= 0.5).sum())
        if positives > 0 and negatives > 0:
            effective_pos_weight = float(negatives / positives)

    training_components = training_utils.configure_training_components(
        model=model,
        learning_rate=config['optimizer']['learning_rate'],
        weight_decay=config['optimizer'].get('weight_decay', 0.0001),
        pos_weight=effective_pos_weight,
        optimizer_name=optimizer_name,
        momentum=momentum,
        device=device,
        num_classes=1,
        label_mode='binary',
        class_weights=None,
    )

    optimizer = training_components['optimizer']
    slide_criterion = training_components['criterion']
    scaler = training_components['scaler']

    sam = training_utils.GraphSAM(optimizer, rho=sam_rho) if sam_enabled else None

    early_cfg = config.get('early_stopping', {}) if isinstance(config.get('early_stopping', {}), dict) else {}
    early_stopping = None
    if bool(early_cfg.get('enabled', True)):
        patience = int(early_cfg.get('patience', 8))
        min_delta = float(early_cfg.get('min_delta', 0.0))
        early_stopping = training_utils.EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            mode=monitor_mode,
        )
        logging.info(
            "Segmentation early stopping enabled: monitor=%s mode=%s patience=%d min_delta=%.6f",
            monitor_metric_name,
            monitor_mode,
            patience,
            min_delta,
        )

    graph_num_workers = int(_cfg('num_workers', 0))
    if graph_num_workers > 0 and bool(getattr(args, 'deterministic', False)):
        logging.info(
            "--deterministic set: forcing num_workers=0 for GeoDataLoader. Forked DataLoader "
            "worker processes each get their own copy of the dataset's augmentation RNG state "
            "at fork time, which is a separate source of run-to-run non-reproducibility beyond "
            "just seeding -- running everything in the main process sidesteps that entirely. "
            "GPU compute per step dominates wall-clock time here anyway (see the OOM/throughput "
            "notes below), so this costs little in practice."
        )
        graph_num_workers = 0
    if graph_num_workers > 0:
        try:
            torch.multiprocessing.set_sharing_strategy('file_system')
            logging.info(
                "Set torch multiprocessing sharing strategy to 'file_system' "
                "(avoids ancdata/fd-transfer failures with large cached graph tensors)."
            )
        except RuntimeError as exc:
            logging.warning(
                "Could not set multiprocessing sharing strategy to 'file_system' (%s); "
                "falling back to num_workers=0.", exc,
            )
            graph_num_workers = 0

    graph_pin_memory = bool(_cfg('pin_memory', False) and torch.cuda.is_available())
    if graph_sampling_weights is not None:
        train_loader = GeoDataLoader(
            train_dataset,
            batch_size=int(config['batch_size']),
            sampler=WeightedRandomSampler(
                torch.as_tensor(graph_sampling_weights, dtype=torch.double),
                num_samples=len(train_dataset),
                replacement=True,
                # Explicit generator instead of PyTorch's ambient global default RNG: makes the
                # sampled order self-contained and independent of how much else has already
                # consumed the global RNG stream before this point in the run (that stream's
                # state, not just its initial seed, would otherwise leak into sampling order).
                generator=torch.Generator().manual_seed(int(args.seed)),
            ),
            num_workers=graph_num_workers,
            pin_memory=graph_pin_memory,
            persistent_workers=graph_num_workers > 0,
            worker_init_fn=training_utils.dataloader_worker_init_fn,
        )
    else:
        train_loader = GeoDataLoader(
            train_dataset,
            batch_size=int(config['batch_size']),
            shuffle=True,
            num_workers=graph_num_workers,
            pin_memory=graph_pin_memory,
            persistent_workers=graph_num_workers > 0,
            worker_init_fn=training_utils.dataloader_worker_init_fn,
        )
    val_loader = GeoDataLoader(
        val_dataset,
        batch_size=max(1, int(config.get('val_batch_size', config['batch_size']))),
        shuffle=False,
        num_workers=graph_num_workers,
        pin_memory=graph_pin_memory,
        persistent_workers=graph_num_workers > 0,
        worker_init_fn=training_utils.dataloader_worker_init_fn,
    )

    scheduler = None
    if config.get('scheduler', {}).get('enabled', False):
        if config['scheduler'].get('name', 'CosineAnnealingLR') == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['scheduler'].get('T_max', 50)
            )
        elif config['scheduler'].get('name') == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min' if monitor_mode == 'min' else 'max',
                factor=config['scheduler'].get('factor', 0.1),
                patience=config['scheduler'].get('patience', 5)
            )

    history = {
        'train_loss': [],
        'train_accuracy': [],
        'train_f1': [],
        'train_auc': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_auc': [],
        'train_slide_accuracy': [],
        'val_slide_accuracy': [],
        'train_slide_sensitivity': [],
        'val_slide_sensitivity': [],
        'train_slide_specificity': [],
        'val_slide_specificity': [],
        'train_slide_f1': [],
        'val_slide_f1': [],
        'train_slide_f2': [],
        'val_slide_f2': [],
        'train_slide_auc': [],
        'val_slide_auc': [],
        'train_region_f1': [],
        'val_region_f1': [],
        'train_region_f2': [],
        'val_region_f2': [],
        'train_region_f1_explicit': [],
        'val_region_f1_explicit': [],
        'train_region_f1_best_threshold': [],
        'val_region_f1_best_threshold': [],
        'train_region_f1_best_threshold_explicit': [],
        'val_region_f1_best_threshold_explicit': [],
        'train_region_auc': [],
        'val_region_auc': [],
        'train_region_auc_explicit': [],
        'val_region_auc_explicit': [],
        'val_region_best_threshold': [],
        'train_patch_f1': [],
        'val_patch_f1': [],
        'train_patch_f2': [],
        'val_patch_f2': [],
        'train_patch_f1_explicit': [],
        'val_patch_f1_explicit': [],
        'train_patch_auc': [],
        'val_patch_auc': [],
        'train_patch_auc_explicit': [],
        'val_patch_auc_explicit': [],
        'train_region_cls_accuracy': [],
        'val_region_cls_accuracy': [],
        'train_patch_cls_accuracy': [],
        'val_patch_cls_accuracy': [],
        'train_region_cls_kappa': [],
        'val_region_cls_kappa': [],
        'train_region_cls_macro_f1': [],
        'val_region_cls_macro_f1': [],
        'train_patch_cls_kappa': [],
        'val_patch_cls_kappa': [],
        'train_patch_cls_macro_f1': [],
        'val_patch_cls_macro_f1': [],
        'metadata': {
            'mode': 'graph_segmentation',
            'segmentation_target': segmentation_target,
            'random_seed': args.seed,
            'deterministic': args.deterministic,
            'model_name': args.architecture,
            'timestamp': datetime.now().isoformat(),
            'monitor_metric': monitor_metric_name,
            'monitor_mode': monitor_mode,
            'region_loss_weight': region_loss_weight,
            'slide_loss_weight': slide_loss_weight,
            'aux_loss_weight': aux_loss_weight,
            'fallback_region_weight': fallback_region_weight,
            'max_fallback_to_explicit_ratio': max_fallback_to_explicit_ratio,
            'prefer_explicit_only_when_available': prefer_explicit_only_when_available,
            'region_slide_consistency_weight': region_slide_consistency_weight,
            'unlabeled_region_policy': unlabeled_region_policy,
            'negative_label_source': negative_label_source,
            'pseudo_label_top_fraction': pseudo_label_top_fraction,
            'pseudo_label_warmup_epochs': pseudo_label_warmup_epochs,
            'patch_loss_weight': patch_loss_weight,
            'augmentation_consistency_weight': augmentation_consistency_weight,
            'augmentation_consistency_specs': augmentation_consistency_specs,
            'augmentation_consistency_p': augmentation_consistency_p,
        }
    }

    best_monitor_score = float('inf') if monitor_mode == 'min' else float('-inf')
    pseudo_labeling_active_logged = False

    # Seeded once, outside the epoch loop, from args.seed -- so augmentation_consistency's
    # firing pattern and manifold mixup's sampling are reproducible run-to-run (previously
    # np.random.default_rng() with no seed, recreated fresh every epoch, pulled from OS entropy
    # every time -- neither run-to-run nor epoch-to-epoch reproducible despite --seed).
    consistency_rng = np.random.default_rng(int(getattr(args, 'seed', 42)))

    for epoch in range(int(config['epochs'])):
        effective_pseudo_label_top_fraction = (
            pseudo_label_top_fraction if epoch >= pseudo_label_warmup_epochs else 0.0
        )
        if effective_pseudo_label_top_fraction > 0.0 and not pseudo_labeling_active_logged:
            logging.info(
                "MIL pseudo-labeling now active from epoch %d onward (warmup complete).", epoch + 1
            )
            pseudo_labeling_active_logged = True

        train_metrics = _train_epoch_graph_segmentation(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            slide_criterion=slide_criterion,
            scaler=scaler,
            device=device,
            region_loss_weight=region_loss_weight,
            slide_loss_weight=slide_loss_weight,
            aux_loss_weight=aux_loss_weight,
            fallback_region_weight=fallback_region_weight,
            max_fallback_to_explicit_ratio=max_fallback_to_explicit_ratio,
            prefer_explicit_only_when_available=prefer_explicit_only_when_available,
            region_slide_consistency_weight=region_slide_consistency_weight,
            unlabeled_region_policy=unlabeled_region_policy,
            use_amp=use_mixed_precision,
            gradient_clip_value=gradient_clip_value,
            node_type=segmentation_target,
            pseudo_label_top_fraction=effective_pseudo_label_top_fraction,
            negative_label_source=negative_label_source,
            patch_loss_weight=patch_loss_weight,
            augmentation_consistency_weight=augmentation_consistency_weight,
            augmentation_consistency_specs=augmentation_consistency_specs,
            augmentation_consistency_p=augmentation_consistency_p,
            feature_consistency_weight=feature_consistency_weight,
            manifold_mixup_p=manifold_mixup_p,
            manifold_mixup_alpha=manifold_mixup_alpha,
            consistency_rng=consistency_rng,
            slide_label_smoothing=slide_label_smoothing,
            grad_accum_steps=grad_accum_steps,
            sam=sam,
        )

        val_metrics = _validate_epoch_graph_segmentation(
            model=model,
            val_loader=val_loader,
            slide_criterion=slide_criterion,
            device=device,
            region_loss_weight=region_loss_weight,
            slide_loss_weight=slide_loss_weight,
            aux_loss_weight=aux_loss_weight,
            fallback_region_weight=fallback_region_weight,
            max_fallback_to_explicit_ratio=max_fallback_to_explicit_ratio,
            prefer_explicit_only_when_available=prefer_explicit_only_when_available,
            region_slide_consistency_weight=region_slide_consistency_weight,
            unlabeled_region_policy=unlabeled_region_policy,
            node_type=segmentation_target,
            negative_label_source=negative_label_source,
            patch_loss_weight=patch_loss_weight,
        )

        current_monitor_value = _metric_value(val_metrics, monitor_metric_name)

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_monitor_value)
            else:
                scheduler.step()

        history['train_loss'].append(train_metrics.get('loss', 0.0))
        history['train_accuracy'].append(train_metrics.get('accuracy', 0.0))
        history['train_f1'].append(train_metrics.get('f1', 0.0))
        history['train_auc'].append(train_metrics.get('auc', 50.0))
        history['val_loss'].append(val_metrics.get('loss', 0.0))
        history['val_accuracy'].append(val_metrics.get('accuracy', 0.0))
        history['val_f1'].append(val_metrics.get('f1', 0.0))
        history['val_auc'].append(val_metrics.get('auc', 50.0))
        history['train_slide_accuracy'].append(train_metrics.get('slide_accuracy', 0.0))
        history['val_slide_accuracy'].append(val_metrics.get('slide_accuracy', 0.0))
        history['train_slide_sensitivity'].append(train_metrics.get('slide_sensitivity', 0.0))
        history['val_slide_sensitivity'].append(val_metrics.get('slide_sensitivity', 0.0))
        history['train_slide_specificity'].append(train_metrics.get('slide_specificity', 0.0))
        history['val_slide_specificity'].append(val_metrics.get('slide_specificity', 0.0))
        history['train_slide_f1'].append(train_metrics.get('slide_f1', 0.0))
        history['val_slide_f1'].append(val_metrics.get('slide_f1', 0.0))
        history['train_slide_f2'].append(train_metrics.get('slide_f2', 0.0))
        history['val_slide_f2'].append(val_metrics.get('slide_f2', 0.0))
        history['train_slide_auc'].append(train_metrics.get('slide_auc', 50.0))
        history['val_slide_auc'].append(val_metrics.get('slide_auc', 50.0))
        history['train_region_f1'].append(train_metrics.get('region_f1', 0.0))
        history['val_region_f1'].append(val_metrics.get('region_f1', 0.0))
        history['train_region_f2'].append(train_metrics.get('region_f2', 0.0))
        history['val_region_f2'].append(val_metrics.get('region_f2', 0.0))
        history['train_region_f1_explicit'].append(train_metrics.get('region_f1_explicit', 0.0))
        history['val_region_f1_explicit'].append(val_metrics.get('region_f1_explicit', 0.0))
        history['train_region_f1_best_threshold'].append(train_metrics.get('region_f1_best_threshold', 0.0))
        history['val_region_f1_best_threshold'].append(val_metrics.get('region_f1_best_threshold', 0.0))
        history['train_region_f1_best_threshold_explicit'].append(train_metrics.get('region_f1_best_threshold_explicit', 0.0))
        history['val_region_f1_best_threshold_explicit'].append(val_metrics.get('region_f1_best_threshold_explicit', 0.0))
        history['train_region_auc'].append(train_metrics.get('region_auc', 50.0))
        history['val_region_auc'].append(val_metrics.get('region_auc', 50.0))
        history['train_region_auc_explicit'].append(train_metrics.get('region_auc_explicit', 50.0))
        history['val_region_auc_explicit'].append(val_metrics.get('region_auc_explicit', 50.0))
        history['val_region_best_threshold'].append(val_metrics.get('region_best_threshold', 0.5))
        history['train_patch_f1'].append(train_metrics.get('patch_f1', 0.0))
        history['val_patch_f1'].append(val_metrics.get('patch_f1', 0.0))
        history['train_patch_f2'].append(train_metrics.get('patch_f2', 0.0))
        history['val_patch_f2'].append(val_metrics.get('patch_f2', 0.0))
        history['train_patch_f1_explicit'].append(train_metrics.get('patch_f1_explicit', 0.0))
        history['val_patch_f1_explicit'].append(val_metrics.get('patch_f1_explicit', 0.0))
        history['train_patch_auc'].append(train_metrics.get('patch_auc', 50.0))
        history['val_patch_auc'].append(val_metrics.get('patch_auc', 50.0))
        history['train_patch_auc_explicit'].append(train_metrics.get('patch_auc_explicit', 50.0))
        history['val_patch_auc_explicit'].append(val_metrics.get('patch_auc_explicit', 50.0))
        history['train_region_cls_accuracy'].append(train_metrics.get('region_cls_accuracy'))
        history['val_region_cls_accuracy'].append(val_metrics.get('region_cls_accuracy'))
        history['train_patch_cls_accuracy'].append(train_metrics.get('patch_cls_accuracy'))
        history['val_patch_cls_accuracy'].append(val_metrics.get('patch_cls_accuracy'))
        history['train_region_cls_kappa'].append(train_metrics.get('region_cls_kappa'))
        history['val_region_cls_kappa'].append(val_metrics.get('region_cls_kappa'))
        history['train_region_cls_macro_f1'].append(train_metrics.get('region_cls_macro_f1'))
        history['val_region_cls_macro_f1'].append(val_metrics.get('region_cls_macro_f1'))
        history['train_patch_cls_kappa'].append(train_metrics.get('patch_cls_kappa'))
        history['val_patch_cls_kappa'].append(val_metrics.get('patch_cls_kappa'))
        history['train_patch_cls_macro_f1'].append(train_metrics.get('patch_cls_macro_f1'))
        history['val_patch_cls_macro_f1'].append(val_metrics.get('patch_cls_macro_f1'))

        if is_multiclass_only_run:
            logging.info(
                "Epoch %d/%d | train_loss=%.4f val_loss=%.4f",
                epoch + 1, int(config['epochs']), train_metrics.get('loss', 0.0), val_metrics.get('loss', 0.0),
            )
        else:
            _log_epoch_metrics(
                epoch=epoch,
                total_epochs=int(config['epochs']),
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                multiclass=False,
                monitor_metric_name=monitor_metric_name,
            )

            logging.info(
                "F2 (beta=2, recall weighted 4x precision -- missed positives cost more than false "
                "alarms)%s | slide: train=%.2f val=%.2f | patch(explicit): train=%.2f val=%.2f | "
                "region(explicit): train=%.2f val=%.2f",
                "  <-- MONITOR METRIC IS slide_f2" if monitor_metric_name == 'slide_f2' else "",
                train_metrics.get('slide_f2', 0.0),
                val_metrics.get('slide_f2', 0.0),
                train_metrics.get('patch_f2_explicit', 0.0),
                val_metrics.get('patch_f2_explicit', 0.0),
                train_metrics.get('region_f2_explicit', 0.0),
                val_metrics.get('region_f2_explicit', 0.0),
            )

            logging.info(
                "Segmentation extras | val_region_f1_ORACLE(swept-thr, NOT deployable)=%.2f (explicit=%.2f) thr=%.2f | "
                "train_region_auc=%.2f val_region_auc=%.2f "
                "(explicit train/val=%.2f/%.2f) | labeled_regions(train/val)=%d/%d | explicit_regions(train/val)=%d/%d | "
                "fallback_regions(train/val)=%d/%d",
                val_metrics.get('region_f1_best_threshold', 0.0),
                val_metrics.get('region_f1_best_threshold_explicit', 0.0),
                val_metrics.get('region_best_threshold', 0.5),
                train_metrics.get('region_auc', 50.0),
                val_metrics.get('region_auc', 50.0),
                train_metrics.get('region_auc_explicit', 50.0),
                val_metrics.get('region_auc_explicit', 50.0),
                int(train_metrics.get('labeled_regions', 0)),
                int(val_metrics.get('labeled_regions', 0)),
                int(train_metrics.get('explicit_regions', 0)),
                int(val_metrics.get('explicit_regions', 0)),
                int(train_metrics.get('fallback_regions', 0)),
                int(val_metrics.get('fallback_regions', 0)),
            )

            logging.info(
                "Region label/pred distribution | train labels pos/neg=%d/%d preds pos/neg=%d/%d | "
                "val labels pos/neg=%d/%d preds pos/neg=%d/%d",
                int(train_metrics.get('region_pos_count', 0)),
                int(train_metrics.get('region_neg_count', 0)),
                int(train_metrics.get('region_pred_pos_count', 0)),
                int(train_metrics.get('region_pred_neg_count', 0)),
                int(val_metrics.get('region_pos_count', 0)),
                int(val_metrics.get('region_neg_count', 0)),
                int(val_metrics.get('region_pred_pos_count', 0)),
                int(val_metrics.get('region_pred_neg_count', 0)),
            )

        # TissueRegionGraphModel's optional region_cls_target_field/patch_cls_target_field
        # multiclass head (e.g. GSDB tissue_type, see model_config_tissue_region_tissue_type_cls
        # .yaml) -- only logged when actually enabled (val labeled_count > 0), so existing
        # binary-only runs see nothing new here. This is the ONLY place this task's own
        # performance is visible -- it contributes to aux_loss/total loss (and therefore trains)
        # regardless of this logging, but was otherwise invisible (no dedicated metric existed
        # before this).
        if val_metrics.get('region_cls_labeled_count', 0) > 0 or val_metrics.get('patch_cls_labeled_count', 0) > 0:
            def _fmt_cls_acc(value: Optional[float]) -> str:
                return f"{value * 100:.2f}%" if value is not None else "n/a"

            def _fmt_cls_score(value: Optional[float]) -> str:
                return f"{value:.4f}" if value is not None else "n/a"

            logging.info(
                "Region/patch cls (multiclass label_fields) accuracy | "
                "region: train=%s val=%s (labeled train/val=%d/%d) | "
                "patch: train=%s val=%s (labeled train/val=%d/%d)",
                _fmt_cls_acc(train_metrics.get('region_cls_accuracy')),
                _fmt_cls_acc(val_metrics.get('region_cls_accuracy')),
                int(train_metrics.get('region_cls_labeled_count', 0)),
                int(val_metrics.get('region_cls_labeled_count', 0)),
                _fmt_cls_acc(train_metrics.get('patch_cls_accuracy')),
                _fmt_cls_acc(val_metrics.get('patch_cls_accuracy')),
                int(train_metrics.get('patch_cls_labeled_count', 0)),
                int(val_metrics.get('patch_cls_labeled_count', 0)),
            )
            # Cohen's Kappa/macro-F1 -- what segmentation.monitor_metric: region_cls_kappa/
            # patch_cls_kappa (or *_macro_f1) actually selects checkpoints on, see
            # model_config_tissue_region_tissue_type_only.yaml. Accuracy alone (logged above) can
            # look deceptively high on an imbalanced taxonomy by riding the majority classes;
            # kappa/macro-F1 don't.
            logging.info(
                "Region/patch cls kappa/macro-F1%s | region: kappa train=%s val=%s | f1 train=%s val=%s | "
                "patch: kappa train=%s val=%s | f1 train=%s val=%s",
                "  <-- MONITOR METRIC" if monitor_metric_name in {'region_cls_kappa', 'patch_cls_kappa', 'region_cls_macro_f1', 'patch_cls_macro_f1'} else "",
                _fmt_cls_score(train_metrics.get('region_cls_kappa')),
                _fmt_cls_score(val_metrics.get('region_cls_kappa')),
                _fmt_cls_score(train_metrics.get('region_cls_macro_f1')),
                _fmt_cls_score(val_metrics.get('region_cls_macro_f1')),
                _fmt_cls_score(train_metrics.get('patch_cls_kappa')),
                _fmt_cls_score(val_metrics.get('patch_cls_kappa')),
                _fmt_cls_score(train_metrics.get('patch_cls_macro_f1')),
                _fmt_cls_score(val_metrics.get('patch_cls_macro_f1')),
            )

            # Full classification report (confusion matrix + per-class precision/recall/f1/
            # support + aggregate scores) -- the primary per-epoch output for a multiclass-only
            # run, in place of the binary segmentation metric wall skipped above. Kept out of the
            # regular (joint binary+multiclass) path's logging: there, region_cls/patch_cls is a
            # minor additive term (see model_config_tissue_region_tissue_type_cls.yaml) and the
            # slide/region/patch binary metrics above remain the primary signal, so a full report
            # every epoch would just be extra noise on top of what's already meaningful there.
            if is_multiclass_only_run:
                if getattr(model, 'region_cls_enabled', False):
                    _log_multiclass_classification_report(
                        'region', model.region_cls_class_names,
                        val_metrics.get('region_cls_true', []), val_metrics.get('region_cls_pred', []),
                    )
                if getattr(model, 'patch_cls_enabled', False):
                    _log_multiclass_classification_report(
                        'patch', model.patch_cls_class_names,
                        val_metrics.get('patch_cls_true', []), val_metrics.get('patch_cls_pred', []),
                    )

        if not is_multiclass_only_run:
            logging.info(
                "Explicit-only distribution | train labels pos/neg=%d/%d preds pos/neg=%d/%d | "
                "val labels pos/neg=%d/%d preds pos/neg=%d/%d",
                int(train_metrics.get('explicit_pos_count', 0)),
                int(train_metrics.get('explicit_neg_count', 0)),
                int(train_metrics.get('explicit_pred_pos_count', 0)),
                int(train_metrics.get('explicit_pred_neg_count', 0)),
                int(val_metrics.get('explicit_pos_count', 0)),
                int(val_metrics.get('explicit_neg_count', 0)),
                int(val_metrics.get('explicit_pred_pos_count', 0)),
                int(val_metrics.get('explicit_pred_neg_count', 0)),
            )

            logging.info(
                "Sanity explicit supervision | train used explicit positives=%d negatives=%d total=%d | "
                "val used explicit positives=%d negatives=%d total=%d",
                int(train_metrics.get('explicit_pos_count', 0)),
                int(train_metrics.get('explicit_neg_count', 0)),
                int(train_metrics.get('explicit_pos_count', 0)) + int(train_metrics.get('explicit_neg_count', 0)),
                int(val_metrics.get('explicit_pos_count', 0)),
                int(val_metrics.get('explicit_neg_count', 0)),
                int(val_metrics.get('explicit_pos_count', 0)) + int(val_metrics.get('explicit_neg_count', 0)),
            )

        if patch_loss_weight > 0.0:
            logging.info(
                "Patch loss | train=%.4f val=%.4f | train_patch_f1=%.2f (explicit=%.2f) "
                "val_patch_f1=%.2f (explicit=%.2f) | train_patch_auc=%.2f val_patch_auc=%.2f "
                "(explicit train/val=%.2f/%.2f) | labeled_patches(train/val)=%d/%d | "
                "explicit_patches(train/val)=%d/%d | fallback_patches(train/val)=%d/%d "
                "(fallback = inherited from parent region, not the patch's own annotation)",
                train_metrics.get('patch_loss', 0.0),
                val_metrics.get('patch_loss', 0.0),
                train_metrics.get('patch_f1', 0.0),
                train_metrics.get('patch_f1_explicit', 0.0),
                val_metrics.get('patch_f1', 0.0),
                val_metrics.get('patch_f1_explicit', 0.0),
                train_metrics.get('patch_auc', 50.0),
                val_metrics.get('patch_auc', 50.0),
                train_metrics.get('patch_auc_explicit', 50.0),
                val_metrics.get('patch_auc_explicit', 50.0),
                int(train_metrics.get('labeled_patches', 0)),
                int(val_metrics.get('labeled_patches', 0)),
                int(train_metrics.get('explicit_patches', 0)),
                int(val_metrics.get('explicit_patches', 0)),
                int(train_metrics.get('fallback_patches', 0)),
                int(val_metrics.get('fallback_patches', 0)),
            )
            logging.info(
                "Patch label/pred distribution | train labels pos/neg=%d/%d preds pos/neg=%d/%d | "
                "val labels pos/neg=%d/%d preds pos/neg=%d/%d",
                int(train_metrics.get('patch_pos_count', 0)),
                int(train_metrics.get('patch_neg_count', 0)),
                int(train_metrics.get('patch_pred_pos_count', 0)),
                int(train_metrics.get('patch_pred_neg_count', 0)),
                int(val_metrics.get('patch_pos_count', 0)),
                int(val_metrics.get('patch_neg_count', 0)),
                int(val_metrics.get('patch_pred_pos_count', 0)),
                int(val_metrics.get('patch_pred_neg_count', 0)),
            )

        if augmentation_consistency_weight > 0.0:
            logging.info(
                "Augmentation consistency loss (train only, never validation) | train=%.4f "
                "| weight=%.3f (contribution to total_step_loss = weight * this value)",
                train_metrics.get('augmentation_consistency_loss', 0.0),
                augmentation_consistency_weight,
            )

        if not is_multiclass_only_run and int(val_metrics.get('region_pos_count', 0)) <= 0:
            logging.warning(
                "Validation contains no positive region labels in this epoch (region_pos_count=0). "
                "Sensitivity/F1/F2 and region threshold selection are not informative on val."
            )

        improved = (
            current_monitor_value <= best_monitor_score
            if monitor_mode == 'min'
            else current_monitor_value >= best_monitor_score
        )
        if improved:
            best_monitor_score = current_monitor_value
            model_path = PathConfigurator.get_checkpoint_path(args, paths, config)
            training_utils.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
                filepath=str(model_path),
                seed=args.seed,
                deterministic=args.deterministic,
            )
            logging.info(
                "Saved best segmentation model with %s=%.4f to %s",
                monitor_metric_name,
                current_monitor_value,
                model_path,
            )

        if early_stopping and early_stopping(current_monitor_value):
            logging.info("Early stopping triggered at epoch %d", epoch + 1)
            break

        if (epoch + 1) % save_frequency == 0:
            periodic_path = PathConfigurator.get_periodic_checkpoint_path(args, paths, config, epoch + 1, tag="seg")
            training_utils.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
                filepath=str(periodic_path),
                seed=args.seed,
                deterministic=args.deterministic,
            )
            logging.info("Saved periodic segmentation checkpoint to %s", periodic_path)

    history_path = PathConfigurator.get_training_history_path(args, paths, config, tag="segmentation")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    try:
        path = PathConfigurator.get_training_path(args, paths)
        plot_training_history(history, save_dir=str(path))
        logging.info("Training plots saved to %s/", path)
    except Exception as e:
        logging.warning("Could not save training plots: %s", e)
    logging.info("Segmentation training complete. Training history saved to %s", history_path)


class ColorNormalizeTransform:
    """Tile-level colour normalisation identical to the graph build's `color_normalize` feature
    transform (src/graph_creation_new/helper/transforms.py) -- same function, same parameters,
    applied to the raw uint8 RGB tile before any resize/crop, exactly as the graph pipeline does per
    patch before embedding. Enabled via a `color_normalization:` block in the model config (used by
    the E1 model-family configs); absent by default. Module-level class so DataLoader workers can
    pickle it."""

    def __init__(self, **params):
        params = dict(params)
        stats_path = params.get('target_stats_path')
        if stats_path is not None and not Path(stats_path).is_absolute():
            project_root = Path(__file__).resolve().parent.parent.parent
            params['target_stats_path'] = str(project_root / stats_path)
        self.params = params

    def __call__(self, image):
        from src.graph_creation_new.helper.transforms import color_normalize
        arr = np.asarray(image.convert('RGB'), dtype=np.uint8)
        out = color_normalize(arr, **self.params)
        return Image.fromarray(np.clip(np.asarray(out), 0, 255).astype(np.uint8))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.params})"


def get_transforms(task: str, is_training: bool = False, config: dict = None) -> T.Compose:
    """Builds the image transform pipeline for a task from config."""
    if config is None:

        return _get_default_transforms(task, is_training)

    transforms_list = []

    # Optional (E1 model-family configs): Macenko normalisation as in the graph build, for train and
    # eval alike. Absent in every other config -> behaviour unchanged.
    color_normalization_cfg = config.get('color_normalization')
    if isinstance(color_normalization_cfg, dict):
        transforms_list.append(ColorNormalizeTransform(**color_normalization_cfg))


    common_config = config.get('common', {})
    norm_config = common_config.get('transforms', {}).get('normalization', {})
    norm_mean = norm_config.get('mean', [0.485, 0.456, 0.406])
    norm_std = norm_config.get('std', [0.229, 0.224, 0.225])

    if is_training:

        transform_config = config.get('transforms', {}).get('training', {})

        # Optional deterministic geometry for training (E1 configs: no augmentation). Same keys and
        # semantics as the validation block (Resize(resize) -> CenterCrop(center_crop)); absent in
        # every other config, so their training pipelines are unchanged.
        if 'resize' in transform_config:
            transforms_list.append(T.Resize(transform_config['resize']))
        if 'center_crop' in transform_config:
            transforms_list.append(T.CenterCrop(transform_config['center_crop']))

        if 'random_resized_crop' in transform_config:
            rrc_config = transform_config['random_resized_crop']
            size = rrc_config.get('size', 224)
            scale = tuple(rrc_config.get('scale', [0.8, 1.0]))
            transforms_list.append(T.RandomResizedCrop(size, scale=scale))


        if transform_config.get('random_horizontal_flip', False):
            transforms_list.append(T.RandomHorizontalFlip())

        if transform_config.get('random_vertical_flip', False):
            transforms_list.append(T.RandomVerticalFlip())


        if 'color_jitter' in transform_config:
            jitter_config = transform_config['color_jitter']
            brightness = jitter_config.get('brightness', 0)
            contrast = jitter_config.get('contrast', 0)
            saturation = jitter_config.get('saturation', 0)
            hue = jitter_config.get('hue', 0)
            transforms_list.append(T.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue
            ))


        if transform_config.get('random_rotation', {}).get('enabled', False):
            degrees = transform_config.get('random_rotation', {}).get('degrees', 180)
            transforms_list.append(T.RandomRotation(degrees))


        if transform_config.get('random_blur', {}).get('enabled', False):
            kernel_size = transform_config.get('random_blur', {}).get('kernel_size', 3)
            transforms_list.append(T.GaussianBlur(kernel_size=kernel_size))


        if transform_config.get('random_affine', {}).get('enabled', False):
            degrees = transform_config.get('random_affine', {}).get('degrees', 15)
            translate = tuple(transform_config.get('random_affine', {}).get('translate', [0.1, 0.1]))
            transforms_list.append(T.RandomAffine(degrees=degrees, translate=translate))
    else:

        val_config = common_config.get('transforms', {}).get('validation', {})
        resize = val_config.get('resize', 256)
        center_crop = val_config.get('center_crop', 224)

        transforms_list.extend([
            T.Resize(resize),
            T.CenterCrop(center_crop)
        ])


    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])

    return T.Compose(transforms_list)

def _get_default_transforms(task: str, is_training: bool = False) -> T.Compose:
    transforms_list = []

    if is_training:

        transforms_list.extend([
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2)
        ])


        if task == 'inflammation':
            transforms_list.extend([
                T.RandomRotation(180),
                T.GaussianBlur(kernel_size=3),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1))
            ])
        else:
            transforms_list.append(T.RandomRotation(180))
    else:
        transforms_list.extend([
            T.Resize(256),
            T.CenterCrop(224)
        ])


    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return T.Compose(transforms_list)

@register_training_pipeline(('inflammation', 'cnn', 'image'))
@register_training_pipeline(('tissue', 'cnn', 'image'))
def train_model_image(args, model, train_dataset, val_dataset, config, optimizer_name, momentum, paths):
    """Main training function with standardized approach and config-based transforms."""


    log_training_settings(args, config, optimizer_name, momentum)

    common_config = config.get('common', {}) if isinstance(config.get('common', {}), dict) else {}

    def _cfg(name, default):
        return common_config.get(name, config.get(name, default))

    gradient_clip_value = _cfg('gradient_clipping', None)
    use_mixed_precision = bool(_cfg('mixed_precision', False))
    save_frequency = int(_cfg('save_frequency', 5))
    grad_accum_steps = int(_cfg('grad_accum_steps', 1))
    num_workers = int(_cfg('num_workers', 4))
    pin_memory = torch.cuda.is_available()

    cudnn_allow_tf32 = _cfg('cudnn_allow_tf32', None)
    if cudnn_allow_tf32 is not None and torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32 = bool(cudnn_allow_tf32)
        logging.info("cudnn.allow_tf32 set to %s from config", bool(cudnn_allow_tf32))


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = training_utils.handle_multi_gpu_model(model)
    model = model.to(device)
    fine_tuning_config = config.get('fine_tuning', {})
    if isinstance(fine_tuning_config, dict):
        model = training_utils.apply_fine_tuning_config(model, fine_tuning_config)

    class_weights = None
    if str(getattr(args, 'label_mode', 'binary')).lower() == 'multiclass' or int(getattr(args, 'num_classes', 1)) > 1:
        class_weights = training_utils.infer_multiclass_weights_from_dataset(
            train_dataset,
            int(getattr(args, 'num_classes', 1)),
        )

    effective_pos_weight = float(config.get('pos_weight', 1.0))
    is_binary = str(getattr(args, 'label_mode', 'binary')).lower() == 'binary' and int(getattr(args, 'num_classes', 1)) == 1
    auto_pos_weight = bool(config.get('auto_pos_weight', True))
    if is_binary and not auto_pos_weight:
        logging.info("Using configured static pos_weight=%.4f (auto_pos_weight disabled).", effective_pos_weight)
    if is_binary and auto_pos_weight:
        inferred_labels = []
        if hasattr(train_dataset, 'records') and isinstance(getattr(train_dataset, 'records'), list):
            inferred_labels = [r.get('label') for r in train_dataset.records if isinstance(r, dict) and 'label' in r]
        elif hasattr(train_dataset, 'samples') and isinstance(getattr(train_dataset, 'samples'), list):
            inferred_labels = [s.get('label') for s in train_dataset.samples if isinstance(s, dict) and 'label' in s]

        cleaned_labels = []
        for val in inferred_labels:
            if val is None:
                continue
            try:
                cleaned_labels.append(float(val))
            except (TypeError, ValueError):
                continue

        if cleaned_labels:
            labels_np = np.asarray(cleaned_labels, dtype=float)
            positives = int((labels_np > 0.5).sum())
            negatives = int((labels_np <= 0.5).sum())
            if positives > 0 and negatives > 0:
                auto_pos_weight = negatives / positives
                effective_pos_weight = float(auto_pos_weight)
                logging.info(
                    "Auto-calibrated binary pos_weight from train split: %.4f "
                    "(negatives=%d, positives=%d)",
                    effective_pos_weight,
                    negatives,
                    positives,
                )
            else:
                logging.warning(
                    "Could not auto-calibrate pos_weight due to degenerate class counts "
                    "(negatives=%d, positives=%d). Using configured value %.4f.",
                    negatives,
                    positives,
                    effective_pos_weight,
                )


    training_components = training_utils.configure_training_components(
        model=model,
        learning_rate=config['optimizer']['learning_rate'],
        weight_decay=config['optimizer'].get('weight_decay', 0.0001),
        pos_weight=effective_pos_weight,
        optimizer_name=optimizer_name,
        momentum=momentum,
        device=device,
        num_classes=getattr(args, 'num_classes', 1),
        label_mode=getattr(args, 'label_mode', 'binary'),
        class_weights=class_weights,
    )

    optimizer = training_components['optimizer']
    criterion = training_components['criterion']
    scaler = training_components['scaler']


    early_stopping = None
    if config.get('early_stopping', {}).get('enabled', False):
        patience = config['early_stopping'].get('patience', 10)
        early_stopping = training_utils.EarlyStopping(patience=patience, mode='min')
        logging.info(f"Early stopping enabled with patience {patience}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=training_utils.dataloader_worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'] * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=training_utils.dataloader_worker_init_fn,
    )


    scheduler = None
    if config.get('scheduler', {}).get('enabled', False):
        if config['scheduler'].get('name', 'CosineAnnealingLR') == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['scheduler'].get('T_max', 50)
            )
        elif config['scheduler'].get('name') == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config['scheduler'].get('factor', 0.1),
                patience=config['scheduler'].get('patience', 5)
            )


    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_auc': [],
        'val_sensitivity': [],
        'val_specificity': [],
        'val_precision': [],
        'val_balanced_accuracy': [],
        'val_mcc': [],
        'val_kappa': [],
        'learning_rate': [],

        'metadata': {
            'random_seed': args.seed,
            'deterministic': args.deterministic,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'model_name': args.architecture,
            'timestamp': datetime.now().isoformat()
        }
    }


    best_val_loss = float('inf')
    for epoch in range(config['epochs']):

        train_metrics = training_utils.train_epoch_img(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            use_amp=use_mixed_precision,
            deterministic=args.deterministic,
            gradient_clip_value=gradient_clip_value
        )


        val_metrics = training_utils.validate_img(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )


        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()


        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_sensitivity'].append(val_metrics.get('sensitivity', 0.0))
        history['val_specificity'].append(val_metrics.get('specificity', 0.0))
        history['val_precision'].append(val_metrics.get('precision', 0.0))
        history['val_balanced_accuracy'].append(val_metrics.get('balanced_accuracy', 0.0))
        history['val_mcc'].append(val_metrics.get('mcc', 0.0))
        history['val_kappa'].append(val_metrics.get('kappa', 0.0))


        _log_epoch_metrics(
            epoch=epoch,
            total_epochs=config['epochs'],
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            multiclass=_is_multiclass_setup(args),
        )


        if early_stopping:
            if early_stopping(val_metrics['loss']):
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break


        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            model_path = PathConfigurator.get_checkpoint_path(args, paths, config)
            training_utils.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
                filepath=str(model_path),
                seed=args.seed,
                deterministic=args.deterministic
            )
            logging.info(f"Saved best model with validation loss: {val_metrics['loss']:.4f} to {model_path}")
            logging.info(f"Model saved with seed={args.seed}, deterministic={args.deterministic}")

        if (epoch + 1) % save_frequency == 0:
            periodic_path = PathConfigurator.get_periodic_checkpoint_path(args, paths, config, epoch + 1)
            training_utils.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
                filepath=str(periodic_path),
                seed=args.seed,
                deterministic=args.deterministic
            )
            logging.info(f"Saved periodic checkpoint to {periodic_path}")

    history_path = PathConfigurator.get_training_history_path(args, paths, config)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    try:
        path = PathConfigurator.get_training_path(args, paths)
        plot_training_history(history, save_dir=str(path))
        logging.info(f"Training plots saved to {path}/")
    except Exception as e:
        logging.warning(f"Could not save training plots: {e}")

    logging.info(f"Training complete. Training history saved to {history_path}")


@register_training_pipeline(('inflammation', 'cnn', 'multimodal'))
def train_model_image_embedding(args, model, train_dataset, val_dataset, config, optimizer_name, momentum, paths):
    """Main training function with standardized approach and config-based transforms."""


    log_training_settings(args, config, optimizer_name, momentum)

    common_config = config.get('common', {})
    gradient_clip_value = common_config.get('gradient_clipping', None)
    use_mixed_precision = common_config.get('mixed_precision', False)
    save_frequency = common_config.get('save_frequency', 5)
    num_workers = int(common_config.get('num_workers', 4))
    pin_memory = torch.cuda.is_available()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = training_utils.handle_multi_gpu_model(model)
    model = model.to(device)
    fine_tuning_config = config.get('fine_tuning', {})
    if isinstance(fine_tuning_config, dict):
        model = training_utils.apply_fine_tuning_config(model, fine_tuning_config)

    class_weights = None
    if str(getattr(args, 'label_mode', 'binary')).lower() == 'multiclass' or int(getattr(args, 'num_classes', 1)) > 1:
        class_weights = training_utils.infer_multiclass_weights_from_dataset(
            train_dataset,
            int(getattr(args, 'num_classes', 1)),
        )

    effective_pos_weight = float(config.get('pos_weight', 1.0))
    auto_pos_weight = bool(config.get('auto_pos_weight', True))
    train_sampler = None
    is_binary = str(getattr(args, 'label_mode', 'binary')).lower() == 'binary' and int(getattr(args, 'num_classes', 1)) == 1
    if is_binary and auto_pos_weight and hasattr(train_dataset, 'records') and isinstance(getattr(train_dataset, 'records'), list):
        labels = []
        for record in train_dataset.records:
            if isinstance(record, dict) and 'label' in record:
                try:
                    labels.append(float(record['label']))
                except (TypeError, ValueError):
                    continue

        if labels:
            labels_np = np.asarray(labels, dtype=float)
            positives = int((labels_np > 0.5).sum())
            negatives = int((labels_np <= 0.5).sum())
            total = positives + negatives
            if positives > 0 and negatives > 0:
                inferred_pos_weight = negatives / positives
                effective_pos_weight = inferred_pos_weight
                logging.info(
                    "Graph dataset effective label distribution: total=%d, positives=%d, negatives=%d, pos_weight=%.4f",
                    total,
                    positives,
                    negatives,
                    effective_pos_weight,
                )

                use_balanced_sampler = bool(common_config.get('balanced_sampling', True))
                if use_balanced_sampler:
                    pos_w = 1.0 / positives
                    neg_w = 1.0 / negatives
                    sample_weights = [pos_w if float(v) > 0.5 else neg_w for v in labels_np]
                    train_sampler = WeightedRandomSampler(
                        weights=torch.as_tensor(sample_weights, dtype=torch.double),
                        num_samples=len(sample_weights),
                        replacement=True,
                        generator=torch.Generator().manual_seed(int(args.seed)),
                    )
                    logging.info("Enabled WeightedRandomSampler for binary graph training.")
            else:
                logging.warning(
                    "Binary graph labels are degenerate (positives=%d, negatives=%d). "
                    "Using configured pos_weight %.4f without balanced sampling.",
                    positives,
                    negatives,
                    effective_pos_weight,
                )


    training_components = training_utils.configure_training_components(
        model=model,
        learning_rate=config['optimizer']['learning_rate'],
        weight_decay=config['optimizer'].get('weight_decay', 0.0001),
        pos_weight=effective_pos_weight,
        optimizer_name=optimizer_name,
        momentum=momentum,
        device=device,
        num_classes=getattr(args, 'num_classes', 1),
        label_mode=getattr(args, 'label_mode', 'binary'),
        class_weights=class_weights,
    )

    optimizer = training_components['optimizer']
    criterion = training_components['criterion']
    scaler = training_components['scaler']


    early_stopping = None
    if config.get('early_stopping', {}).get('enabled', False):
        patience = config['early_stopping'].get('patience', 10)
        early_stopping = training_utils.EarlyStopping(patience=patience, mode='min')
        logging.info(f"Early stopping enabled with patience {patience}")


    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=training_utils.dataloader_worker_init_fn,
    )

    batch = next(iter(train_loader))

    images, embeddings, labels, metadata = batch

    logging.info(
        f"Sanity batch shapes - images: {images.shape}, embeddings: {embeddings.shape}, labels: {labels.shape}, metadata_type: {type(metadata)}"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'] * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=training_utils.dataloader_worker_init_fn,
    )



    scheduler = None
    if config.get('scheduler', {}).get('enabled', False):
        if config['scheduler'].get('name', 'CosineAnnealingLR') == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['scheduler'].get('T_max', 50)
            )
        elif config['scheduler'].get('name') == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config['scheduler'].get('factor', 0.1),
                patience=config['scheduler'].get('patience', 5)
            )

    model_name = args.architecture


    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_auc': [],
        'val_sensitivity': [],
        'val_specificity': [],
        'val_precision': [],
        'val_balanced_accuracy': [],
        'val_mcc': [],
        'val_kappa': [],
        'learning_rate': [],

        'metadata': {
            'random_seed': args.seed,
            'deterministic': args.deterministic,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'model_name': args.architecture,
            'timestamp': datetime.now().isoformat()
        }
    }


    best_val_loss = float('inf')
    for epoch in range(config['epochs']):

        train_metrics = training_utils.train_epoch_img_emb(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            use_amp=use_mixed_precision,
            deterministic=args.deterministic,
            gradient_clip_value=gradient_clip_value
        )


        val_metrics = training_utils.validate_img_emb(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )

        history['learning_rate'].append(optimizer.param_groups[0]['lr'])




        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_sensitivity'].append(val_metrics.get('sensitivity', 0.0))
        history['val_specificity'].append(val_metrics.get('specificity', 0.0))
        history['val_precision'].append(val_metrics.get('precision', 0.0))
        history['val_balanced_accuracy'].append(val_metrics.get('balanced_accuracy', 0.0))
        history['val_mcc'].append(val_metrics.get('mcc', 0.0))
        history['val_kappa'].append(val_metrics.get('kappa', 0.0))


        _log_epoch_metrics(
            epoch=epoch,
            total_epochs=config['epochs'],
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            multiclass=_is_multiclass_setup(args),
        )


        if early_stopping:
            if early_stopping(val_metrics['loss']):
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break


        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            model_path = PathConfigurator.get_checkpoint_path(args, paths, config)
            training_utils.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
                filepath=str(model_path),
                seed=args.seed,
                deterministic=args.deterministic
            )
            logging.info(f"Saved best model with validation loss: {val_metrics['loss']:.4f} to {model_path}")
            logging.info(f"Model saved with seed={args.seed}, deterministic={args.deterministic}")

        if (epoch + 1) % save_frequency == 0:
            periodic_path = PathConfigurator.get_periodic_checkpoint_path(args, paths, config, epoch + 1)
            training_utils.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
                filepath=str(periodic_path),
                seed=args.seed,
                deterministic=args.deterministic
            )
            logging.info(f"Saved periodic checkpoint to {periodic_path}")

    history_path = PathConfigurator.get_training_history_path(args, paths, config)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    try:
        path = PathConfigurator.get_training_path(args, paths)
        plot_training_history(history, save_dir=str(path))
        logging.info(f"Training plots saved to {path}/")
    except Exception as e:
        logging.warning(f"Could not save training plots: {e}")

    logging.info(f"Training complete. Training history saved to {history_path}")

@register_training_pipeline(('inflammation', 'mil', 'multimodal'))
def train_mil_model_multimodal(args, model, train_dataset, val_dataset, config, optimizer_name, momentum, paths):
    """Main training function with standardized approach and config-based transforms."""


    log_training_settings(args, config, optimizer_name, momentum)

    common_config = config.get('common', {}) if isinstance(config.get('common', {}), dict) else {}

    # load_config_new flattens 'common' into the top level, so config['common'] is empty -- read the
    # flattened keys as fallback (same pattern as train_model_image / train_graph_segmentation_model).
    # Previously gradient_clipping/mixed_precision from the config were silently ignored here.
    def _cfg(name, default):
        return common_config.get(name, config.get(name, default))

    gradient_clip_value = _cfg('gradient_clipping', None)
    use_mixed_precision = bool(_cfg('mixed_precision', False))
    save_frequency = int(_cfg('save_frequency', 5))


    fine_tuning_config = config.get('fine_tuning', {})
    unfreeze_epoch = fine_tuning_config.get('unfreeze_backbone_epoch', None)
    fine_tune_lr = fine_tuning_config.get('fine_tune_lr', None)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = training_utils.handle_multi_gpu_model(model)
    model = model.to(device)
    if isinstance(fine_tuning_config, dict):
        model = training_utils.apply_fine_tuning_config(model, fine_tuning_config)

    class_weights = None
    if str(getattr(args, 'label_mode', 'binary')).lower() == 'multiclass' or int(getattr(args, 'num_classes', 1)) > 1:
        class_weights = training_utils.infer_multiclass_weights_from_dataset(
            train_dataset,
            int(getattr(args, 'num_classes', 1)),
        )

    effective_pos_weight = float(config.get('pos_weight', 1.0))
    auto_pos_weight = bool(config.get('auto_pos_weight', True))
    train_sampler = None
    is_binary = str(getattr(args, 'label_mode', 'binary')).lower() == 'binary' and int(getattr(args, 'num_classes', 1)) == 1
    if is_binary and auto_pos_weight and hasattr(train_dataset, 'records') and isinstance(getattr(train_dataset, 'records'), list):
        labels = []
        for record in train_dataset.records:
            if isinstance(record, dict) and 'label' in record:
                try:
                    labels.append(float(record['label']))
                except (TypeError, ValueError):
                    continue

        if labels:
            labels_np = np.asarray(labels, dtype=float)
            positives = int((labels_np > 0.5).sum())
            negatives = int((labels_np <= 0.5).sum())
            total = positives + negatives
            if positives > 0 and negatives > 0:
                inferred_pos_weight = negatives / positives
                effective_pos_weight = inferred_pos_weight
                logging.info(
                    "Graph dataset effective label distribution: total=%d, positives=%d, negatives=%d, pos_weight=%.4f",
                    total,
                    positives,
                    negatives,
                    effective_pos_weight,
                )

                use_balanced_sampler = bool(common_config.get('balanced_sampling', True))
                if use_balanced_sampler:
                    pos_w = 1.0 / positives
                    neg_w = 1.0 / negatives
                    sample_weights = [pos_w if float(v) > 0.5 else neg_w for v in labels_np]
                    train_sampler = WeightedRandomSampler(
                        weights=torch.as_tensor(sample_weights, dtype=torch.double),
                        num_samples=len(sample_weights),
                        replacement=True,
                        generator=torch.Generator().manual_seed(int(args.seed)),
                    )
                    logging.info("Enabled WeightedRandomSampler for binary graph training.")
            else:
                logging.warning(
                    "Binary graph labels are degenerate (positives=%d, negatives=%d). "
                    "Using configured pos_weight %.4f without balanced sampling.",
                    positives,
                    negatives,
                    effective_pos_weight,
                )
    elif is_binary and not auto_pos_weight:
        logging.info(
            "Using configured static pos_weight=%.4f (auto_pos_weight disabled).",
            effective_pos_weight,
        )


    training_components = training_utils.configure_training_components(
        model=model,
        learning_rate=config['optimizer']['learning_rate'],
        weight_decay=config['optimizer'].get('weight_decay', 0.0001),
        pos_weight=effective_pos_weight,
        optimizer_name=optimizer_name,
        momentum=momentum,
        device=device,
        num_classes=getattr(args, 'num_classes', 1),
        label_mode=getattr(args, 'label_mode', 'binary'),
        class_weights=class_weights,
    )

    optimizer = training_components['optimizer']
    criterion = training_components['criterion']
    scaler = training_components['scaler']


    early_stopping = None
    if config.get('early_stopping', {}).get('enabled', False):
        patience = config['early_stopping'].get('patience', 10)
        early_stopping = training_utils.EarlyStopping(patience=patience, mode='min')
        logging.info(f"Early stopping enabled with patience {patience}")

    batch_size = config['batch_size']
    val_batch_size = max(1, batch_size // 2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        collate_fn=mil_collate,
        persistent_workers=False,
        worker_init_fn=training_utils.dataloader_worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=mil_collate,
        pin_memory=False,
        persistent_workers=False,
        worker_init_fn=training_utils.dataloader_worker_init_fn,
    )


    batch_size_reduced = False



    scheduler = None
    if config.get('scheduler', {}).get('enabled', False):
        if config['scheduler'].get('name', 'CosineAnnealingLR') == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['scheduler'].get('T_max', 50)
            )
        elif config['scheduler'].get('name') == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config['scheduler'].get('factor', 0.1),
                patience=config['scheduler'].get('patience', 5)
            )

    model_name = args.architecture


    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_auc': [],
        'val_sensitivity': [],
        'val_specificity': [],
        'val_precision': [],
        'val_balanced_accuracy': [],
        'val_mcc': [],
        'val_kappa': [],
        'learning_rate': [],

        'metadata': {
            'random_seed': args.seed,
            'deterministic': args.deterministic,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'model_name': args.architecture,
            'timestamp': datetime.now().isoformat()
        }
    }


    best_val_loss = float('inf')
    for epoch in range(config['epochs']):

        if unfreeze_epoch is not None and epoch == unfreeze_epoch and fine_tune_lr is not None:
            logging.info(f"Epoch {epoch+1}: Unfreezing backbone for fine-tuning")
            optimizer = training_utils.unfreeze_model_backbone(model, fine_tune_lr, optimizer)

            if not batch_size_reduced:
                batch_size = max(1, batch_size // 2)
                val_batch_size = max(1, val_batch_size // 2)
                logging.info(f"Reducing batch_size after unfreeze: train={batch_size}, val={val_batch_size}")
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=2,
                    pin_memory=False,
                    collate_fn=mil_collate,
                    persistent_workers=False,
                    worker_init_fn=training_utils.dataloader_worker_init_fn,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=val_batch_size,
                    shuffle=True,
                    num_workers=2,
                    collate_fn=mil_collate,
                    pin_memory=False,
                    persistent_workers=False,
                    worker_init_fn=training_utils.dataloader_worker_init_fn,
                )
                batch_size_reduced = True


        train_metrics = training_utils.train_epoch_mil(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            use_amp=use_mixed_precision,
            deterministic=args.deterministic,
            gradient_clip_value=gradient_clip_value
        )


        val_metrics = training_utils.validate_mil(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )

        history['learning_rate'].append(optimizer.param_groups[0]['lr'])






        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics.get('accuracy', 0.0))
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics.get('accuracy', 0.0))
        history['val_f1'].append(val_metrics.get('f1', 0.0))
        history['val_auc'].append(val_metrics.get('auc', 0.0))
        history['val_sensitivity'].append(val_metrics.get('sensitivity', 0.0))
        history['val_specificity'].append(val_metrics.get('specificity', 0.0))
        history['val_precision'].append(val_metrics.get('precision', 0.0))
        history['val_balanced_accuracy'].append(val_metrics.get('balanced_accuracy', 0.0))
        history['val_mcc'].append(val_metrics.get('mcc', 0.0))
        history['val_kappa'].append(val_metrics.get('kappa', 0.0))



        _log_epoch_metrics(
            epoch=epoch,
            total_epochs=config['epochs'],
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            multiclass=_is_multiclass_setup(args),
        )


        if early_stopping:
            if early_stopping(val_metrics['loss']):
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break


        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            model_path = PathConfigurator.get_checkpoint_path(args, paths, config)
            training_utils.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
                filepath=str(model_path),
                seed=args.seed,
                deterministic=args.deterministic
            )
            logging.info(f"Saved best model with validation loss: {val_metrics['loss']:.4f} to {model_path}")
            logging.info(f"Model saved with seed={args.seed}, deterministic={args.deterministic}")

        if (epoch + 1) % save_frequency == 0:
            periodic_path = PathConfigurator.get_periodic_checkpoint_path(args, paths, config, epoch + 1)
            training_utils.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
                filepath=str(periodic_path),
                seed=args.seed,
                deterministic=args.deterministic
            )
            logging.info(f"Saved periodic checkpoint to {periodic_path}")


    history_path = PathConfigurator.get_training_history_path(args, paths, config)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    try:
        path = PathConfigurator.get_training_path(args, paths)
        plot_training_history(history, save_dir=str(path))
        logging.info(f"Training plots saved to {path}/")
    except Exception as e:
        logging.warning(f"Could not save training plots: {e}")

    logging.info(f"Training complete. Training history saved to {history_path}")



@register_training_pipeline(('inflammation', 'gnn', 'graph'))
def train_graph_model(args, model, train_dataset, val_dataset, config, optimizer_name, momentum, paths):
    """Main training function with standardized approach and config-based transforms."""


    log_training_settings(args, config, optimizer_name, momentum)
    common_config = config.get('common', {}) if isinstance(config.get('common', {}), dict) else {}

    def _cfg(name, default):
        return common_config.get(name, config.get(name, default))

    gradient_clip_value = _cfg('gradient_clipping', None)
    use_mixed_precision = bool(_cfg('mixed_precision', False))
    save_frequency = int(_cfg('save_frequency', 5))
    grad_accum_steps = int(_cfg('grad_accum_steps', 1))


    fine_tuning_config = config.get('fine_tuning', {})
    unfreeze_epoch = fine_tuning_config.get('unfreeze_backbone_epoch', None)
    fine_tune_lr = fine_tuning_config.get('fine_tune_lr', None)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = training_utils.handle_multi_gpu_model(model)
    model = model.to(device)
    if isinstance(fine_tuning_config, dict):
        model = training_utils.apply_fine_tuning_config(model, fine_tuning_config)

    class_weights = None
    multiclass_sample_weights = None
    if str(getattr(args, 'label_mode', 'binary')).lower() == 'multiclass' or int(getattr(args, 'num_classes', 1)) > 1:
        class_weights, multiclass_sample_weights = training_utils.infer_multiclass_class_and_sample_weights_from_dataset(
            train_dataset,
            int(getattr(args, 'num_classes', 1)),
        )

    effective_pos_weight = float(config.get('pos_weight', 1.0))
    auto_pos_weight = bool(config.get('auto_pos_weight', True))
    train_sampler = None
    is_binary = str(getattr(args, 'label_mode', 'binary')).lower() == 'binary' and int(getattr(args, 'num_classes', 1)) == 1
    if is_binary and auto_pos_weight:
        labels = _infer_binary_graph_labels(train_dataset)
        if labels:
            labels_np = np.asarray(labels, dtype=float)
            positives = int((labels_np > 0.5).sum())
            negatives = int((labels_np <= 0.5).sum())
            total = positives + negatives
            if positives > 0 and negatives > 0:
                effective_pos_weight = negatives / positives
                logging.info(
                    "Graph dataset effective label distribution: total=%d, positives=%d, negatives=%d, pos_weight=%.4f",
                    total,
                    positives,
                    negatives,
                    effective_pos_weight,
                )

                use_balanced_sampler = bool(_cfg('balanced_sampling', True))
                if use_balanced_sampler:
                    pos_w = 1.0 / positives
                    neg_w = 1.0 / negatives
                    sample_weights = [pos_w if float(v) > 0.5 else neg_w for v in labels_np]
                    train_sampler = WeightedRandomSampler(
                        weights=torch.as_tensor(sample_weights, dtype=torch.double),
                        num_samples=len(sample_weights),
                        replacement=True,
                        generator=torch.Generator().manual_seed(int(args.seed)),
                    )
                    logging.info("Enabled WeightedRandomSampler for graph training.")
            else:
                logging.warning(
                    "Binary graph labels are degenerate (positives=%d, negatives=%d). "
                    "Using configured pos_weight %.4f without balanced sampling.",
                    positives,
                    negatives,
                    effective_pos_weight,
                )
        else:
            logging.warning(
                "Could not infer binary labels for automatic graph class balancing. "
                "Using configured static pos_weight=%.4f.",
                effective_pos_weight,
            )
    elif is_binary:
        logging.info(
            "Automatic class balancing disabled. Using configured static pos_weight=%.4f.",
            effective_pos_weight,
        )
    elif str(getattr(args, 'label_mode', 'binary')).lower() == 'multiclass' and bool(_cfg('balanced_sampling', True)):
        # Class-weighted CrossEntropyLoss (class_weights above) rescales gradients per class but
        # every sample is still drawn with equal probability -- a ~8%-of-slides class (e.g.
        # CAMELYON subtype itc) is still seen far less often per epoch than the majority class.
        # Oversample by the same inverse-frequency weights (computed alongside class_weights
        # above, from the same dataset scan) so rare classes are actually seen more, not just
        # weighted more when they are.
        if multiclass_sample_weights:
            train_sampler = WeightedRandomSampler(
                weights=torch.as_tensor(multiclass_sample_weights, dtype=torch.double),
                num_samples=len(multiclass_sample_weights),
                replacement=True,
                generator=torch.Generator().manual_seed(int(args.seed)),
            )
            logging.info("Enabled WeightedRandomSampler for multiclass graph training.")
            if bool(_cfg('avoid_double_balance_correction', False)):
                # Both corrections above come from the identical inverse-frequency counts (see
                # infer_multiclass_class_and_sample_weights_from_dataset): the sampler already
                # equalizes how often each class is *seen* per epoch, so also reweighting the
                # loss on top of that compounds the correction for a class that's already
                # oversampled. Opt-in and off by default so no existing run's behavior changes --
                # e.g. CAMELYON subtype's itc/micro classes showed a negative -> itc/micro
                # false-positive pattern in the confusion matrix consistent with over-correction;
                # this flag lets a run test sampler-only balancing (unweighted CE) against that.
                class_weights = None
                logging.info(
                    "avoid_double_balance_correction=True: using unweighted CrossEntropyLoss "
                    "since WeightedRandomSampler already balances class exposure."
                )
        else:
            logging.warning(
                "Could not infer multiclass labels for automatic graph balanced sampling; "
                "falling back to class-weighted loss only."
            )


    training_components = training_utils.configure_training_components(
        model=model,
        learning_rate=config['optimizer']['learning_rate'],
        weight_decay=config['optimizer'].get('weight_decay', 0.0001),
        pos_weight=effective_pos_weight,
        optimizer_name=optimizer_name,
        momentum=momentum,
        device=device,
        num_classes=getattr(args, 'num_classes', 1),
        label_mode=getattr(args, 'label_mode', 'binary'),
        class_weights=class_weights,
        loss_type=str(_cfg('loss_type', 'ce')),
        focal_gamma=float(_cfg('focal_gamma', 2.0)),
        focal_use_alpha=bool(_cfg('focal_use_alpha', True)),
    )

    optimizer = training_components['optimizer']
    criterion = training_components['criterion']
    scaler = training_components['scaler']


    monitor_metric_name, monitor_mode, monitor_min_delta = _resolve_monitor_metric(config, default_metric='loss')

    # Raw class ids -> ordinal severity rank, only used for kappa_quadratic (see
    # training_utils._compute_epoch_metrics). Optional: label fields whose raw ids already
    # increase with severity don't need this, so it defaults to None (no remap).
    ordinal_class_order = config.get('ordinal_class_order')
    if ordinal_class_order is not None:
        ordinal_class_order = [int(x) for x in ordinal_class_order]
        expected_num_classes = int(getattr(args, 'num_classes', 1) or 1)
        if len(ordinal_class_order) != expected_num_classes:
            raise ValueError(
                f"ordinal_class_order has {len(ordinal_class_order)} entries but num_classes="
                f"{expected_num_classes}"
            )

    early_stopping = None
    if config.get('early_stopping', {}).get('enabled', False):
        patience = config['early_stopping'].get('patience', 10)
        early_stopping = training_utils.EarlyStopping(
            patience=patience,
            min_delta=monitor_min_delta,
            mode=monitor_mode,
        )
        logging.info(
            "Early stopping enabled with patience=%d, monitor=%s, mode=%s, min_delta=%.6f",
            patience,
            monitor_metric_name,
            monitor_mode,
            monitor_min_delta,
        )

    resolved_label_mode = str(getattr(args, 'label_mode', 'binary')).strip().lower()
    resolved_num_classes = int(getattr(args, 'num_classes', 1) or 1)
    balanced_sampling_requested = bool(_cfg('balanced_sampling', True)) if (is_binary or resolved_label_mode == 'multiclass') else False
    logging.info(
        "GRAPH TRAINING STARTUP SUMMARY | "
        "dataset=%s | task=%s | architecture=%s | "
        "label_mode=%s | num_classes=%d | "
        "is_binary=%s | auto_pos_weight=%s | effective_pos_weight=%.4f | "
        "balanced_sampling_requested=%s | balanced_sampler_enabled=%s | "
        "monitor_metric=%s | monitor_mode=%s | monitor_min_delta=%.6f",
        getattr(args, 'dataset', 'unknown'),
        getattr(args, 'task', 'unknown'),
        getattr(args, 'architecture', 'unknown'),
        resolved_label_mode,
        resolved_num_classes,
        is_binary,
        auto_pos_weight,
        float(effective_pos_weight),
        balanced_sampling_requested,
        train_sampler is not None,
        monitor_metric_name,
        monitor_mode,
        monitor_min_delta,
    )

    graph_num_workers = int(_cfg('num_workers', 0))
    if getattr(args, 'cached', False) and graph_num_workers > 0:
        logging.warning(
            "Cached graph training detected: forcing num_workers=0 for GeoDataLoader "
            "to avoid multiprocessing ancdata/file-descriptor transfer failures."
        )
        graph_num_workers = 0

    if graph_num_workers > 0:
        try:
            torch.multiprocessing.set_sharing_strategy('file_system')
            logging.info("Set torch multiprocessing sharing strategy to 'file_system'.")
        except RuntimeError as exc:
            logging.warning(f"Could not set multiprocessing sharing strategy: {exc}")

    graph_pin_memory = bool(_cfg('pin_memory', False) and torch.cuda.is_available())

    train_loader = GeoDataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=graph_num_workers,
        pin_memory=graph_pin_memory,
        persistent_workers=graph_num_workers > 0,
        worker_init_fn=training_utils.dataloader_worker_init_fn,
    )

    val_batch_size = int(config.get('val_batch_size', 1 if getattr(args, 'cached', False) and args.input_type == 'graph' else config['batch_size']))

    val_loader = GeoDataLoader(
        val_dataset,
        batch_size=max(1, val_batch_size),
        shuffle=False,
        num_workers=graph_num_workers,
        pin_memory=graph_pin_memory,
        persistent_workers=graph_num_workers > 0,
        worker_init_fn=training_utils.dataloader_worker_init_fn,
    )



    scheduler = None
    if config.get('scheduler', {}).get('enabled', False):
        if config['scheduler'].get('name', 'CosineAnnealingLR') == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['scheduler'].get('T_max', 50)
            )
        elif config['scheduler'].get('name') == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config['scheduler'].get('mode', monitor_mode),
                factor=config['scheduler'].get('factor', 0.1),
                patience=config['scheduler'].get('patience', 5)
            )

    model_name = args.architecture


    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_auc': [],
        'val_sensitivity': [],
        'val_specificity': [],
        'val_precision': [],
        'val_balanced_accuracy': [],
        'val_mcc': [],
        'val_kappa': [],
        'val_kappa_quadratic': [],
        'learning_rate': [],

        'metadata': {
            'random_seed': args.seed,
            'deterministic': args.deterministic,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'model_name': args.architecture,
            'timestamp': datetime.now().isoformat()
        }
    }


    # Optional consistency regularization (see training_utils._compute_graph_consistency_loss).
    # Off by default (weight 0.0) for every architecture that doesn't set these -- opt-in via
    # the model config. Seeded once here, outside the epoch loop, from args.seed so its firing
    # pattern is reproducible run-to-run (the same fix as _train_epoch_graph_segmentation's
    # consistency_rng -- see that function's comment for why an unseeded per-epoch RNG silently
    # broke reproducibility there).
    consistency_weight = float(config.get('consistency_weight', 0.0))
    consistency_specs = config.get('consistency_specs', []) or []
    consistency_p = float(config.get('consistency_p', 1.0))
    consistency_rng = np.random.default_rng(int(getattr(args, 'seed', 42)))
    if consistency_weight > 0.0 and consistency_specs:
        logging.info(
            "Consistency regularization enabled: weight=%.3f p=%.2f specs=%s "
            "(doubles per-step forward-pass compute when it fires).",
            consistency_weight, consistency_p, consistency_specs,
        )

    # Domain-adversarial training (see GradientReversalLayer's docstring in
    # src/tissue_region_extraction/model.py). Off by default. Shares consistency_p/consistency_rng
    # above as its firing gate -- when it fires alongside consistency regularization, both losses
    # come from the SAME augmented forward pass (see
    # training_utils._compute_graph_consistency_and_domain_losses), so enabling this does not add
    # a second extra forward pass on top of consistency regularization already being on.
    domain_adversarial_weight = float(config.get('domain_adversarial_weight', 0.0))
    domain_adversarial_specs = config.get('domain_adversarial_specs', []) or []
    if domain_adversarial_weight > 0.0 and domain_adversarial_specs:
        logging.info(
            "Domain-adversarial regularization enabled: weight=%.3f specs=%s",
            domain_adversarial_weight, domain_adversarial_specs,
        )

    slide_label_smoothing = float(config.get('slide_label_smoothing', 0.0))
    if slide_label_smoothing > 0.0:
        logging.info("Slide label smoothing enabled: eps=%.3f", slide_label_smoothing)

    best_monitor_score = float('inf') if monitor_mode == 'min' else float('-inf')
    best_model_metric_value = float('inf') if monitor_mode == 'min' else float('-inf')
    for epoch in range(config['epochs']):

        if unfreeze_epoch is not None and epoch == unfreeze_epoch and fine_tune_lr is not None:
            logging.info(f"Epoch {epoch+1}: Unfreezing backbone for fine-tuning")
            optimizer = training_utils.unfreeze_model_backbone(model, fine_tune_lr, optimizer)


        train_metrics = training_utils.train_epoch_graph(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            use_amp=use_mixed_precision,
            deterministic=args.deterministic,
            gradient_clip_value=gradient_clip_value,
            gradient_accumulation_steps=grad_accum_steps,
            consistency_weight=consistency_weight,
            consistency_specs=consistency_specs,
            consistency_p=consistency_p,
            consistency_rng=consistency_rng,
            domain_adversarial_weight=domain_adversarial_weight,
            domain_adversarial_specs=domain_adversarial_specs,
            slide_label_smoothing=slide_label_smoothing,
            ordinal_class_order=ordinal_class_order,
        )


        val_metrics = training_utils.validate_graph(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            ordinal_class_order=ordinal_class_order,
        )

        current_monitor_value = _metric_value(val_metrics, monitor_metric_name)

        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_monitor_value)
            else:
                scheduler.step()






        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics.get('accuracy', 0.0))
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics.get('accuracy', 0.0))
        history['val_f1'].append(val_metrics.get('f1', 0.0))
        history['val_auc'].append(val_metrics.get('auc', 0.0))
        history['val_sensitivity'].append(val_metrics.get('sensitivity', 0.0))
        history['val_specificity'].append(val_metrics.get('specificity', 0.0))
        history['val_precision'].append(val_metrics.get('precision', 0.0))
        history['val_balanced_accuracy'].append(val_metrics.get('balanced_accuracy', 0.0))
        history['val_mcc'].append(val_metrics.get('mcc', 0.0))
        history['val_kappa'].append(val_metrics.get('kappa', 0.0))
        history['val_kappa_quadratic'].append(val_metrics.get('kappa_quadratic', 0.0))



        _log_epoch_metrics(
            epoch=epoch,
            total_epochs=config['epochs'],
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            multiclass=_is_multiclass_setup(args),
        )


        if early_stopping:
            if early_stopping(current_monitor_value):
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break


        improved = (current_monitor_value <= best_monitor_score) if monitor_mode == 'min' else (current_monitor_value >= best_monitor_score)
        if improved:
            best_monitor_score = current_monitor_value
            best_model_metric_value = current_monitor_value
            model_path = PathConfigurator.get_checkpoint_path(args, paths, config)
            training_utils.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
                filepath=str(model_path),
                seed=args.seed,
                deterministic=args.deterministic
            )
            logging.info(
                "Saved best model with validation %s=%.4f to %s",
                monitor_metric_name,
                current_monitor_value,
                model_path,
            )
            logging.info(f"Model saved with seed={args.seed}, deterministic={args.deterministic}")

        if (epoch + 1) % save_frequency == 0:
            periodic_path = PathConfigurator.get_periodic_checkpoint_path(args, paths, config, epoch + 1)
            training_utils.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
                filepath=str(periodic_path),
                seed=args.seed,
                deterministic=args.deterministic
            )
            logging.info(f"Saved periodic checkpoint to {periodic_path}")


    history_path = PathConfigurator.get_training_history_path(args, paths, config)
    history['metadata'].update({
        'monitor_metric': monitor_metric_name,
        'monitor_mode': monitor_mode,
        'best_monitor_value': best_model_metric_value,
        'effective_pos_weight': float(effective_pos_weight),
        'balanced_sampler_enabled': train_sampler is not None,
    })
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    try:
        path = PathConfigurator.get_training_path(args, paths)
        plot_training_history(history, save_dir=str(path))
        logging.info(f"Training plots saved to {path}/")
    except Exception as e:
        logging.warning(f"Could not save training plots: {e}")

    logging.info(f"Training complete. Training history saved to {history_path}")

@register_training_pipeline(('inflammation', 'mil', 'image'))
def train_mil_model_image(args, model, train_dataset, val_dataset, config, optimizer_name, momentum, paths):
    """Main training function with standardized approach and config-based transforms."""


    common_config = config.get('common', {}) if isinstance(config.get('common', {}), dict) else {}

    # load_config_new flattens 'common' into the top level, so config['common'] is empty -- read the
    # flattened keys as fallback (same pattern as train_model_image / train_graph_segmentation_model).
    # Previously gradient_clipping/mixed_precision from the config were silently ignored here.
    def _cfg(name, default):
        return common_config.get(name, config.get(name, default))

    gradient_clip_value = _cfg('gradient_clipping', None)
    use_mixed_precision = bool(_cfg('mixed_precision', False))
    save_frequency = int(_cfg('save_frequency', 5))


    fine_tuning_config = config.get('fine_tuning', {})
    unfreeze_epoch = fine_tuning_config.get('unfreeze_backbone_epoch', None)
    fine_tune_lr = fine_tuning_config.get('fine_tune_lr', None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = training_utils.handle_multi_gpu_model(model)
    model = model.to(device)
    if isinstance(fine_tuning_config, dict):
        model = training_utils.apply_fine_tuning_config(model, fine_tuning_config)

    class_weights = None
    if str(getattr(args, 'label_mode', 'binary')).lower() == 'multiclass' or int(getattr(args, 'num_classes', 1)) > 1:
        class_weights = training_utils.infer_multiclass_weights_from_dataset(
            train_dataset,
            int(getattr(args, 'num_classes', 1)),
        )

    training_components = training_utils.configure_training_components(
        model=model,
        learning_rate=config['optimizer']['learning_rate'],
        weight_decay=config['optimizer'].get('weight_decay', 0.0001),
        pos_weight=config.get('pos_weight', 1.0),
        optimizer_name=optimizer_name,
        momentum=momentum,
        device=device,
        num_classes=getattr(args, 'num_classes', 1),
        label_mode=getattr(args, 'label_mode', 'binary'),
        class_weights=class_weights,
    )

    optimizer = training_components['optimizer']
    criterion = training_components['criterion']
    scaler = training_components['scaler']

    early_stopping = None
    if config.get('early_stopping', {}).get('enabled', False):
        patience = config['early_stopping'].get('patience', 10)
        early_stopping = training_utils.EarlyStopping(patience=patience, mode='min')
        logging.info(f"Early stopping enabled with patience {patience}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=mil_collate_no_embeddings,
        worker_init_fn=training_utils.dataloader_worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, config['batch_size']//2),
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=mil_collate_no_embeddings,
        worker_init_fn=training_utils.dataloader_worker_init_fn,
    )

    model_name = args.architecture

    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_auc': [],
        'val_sensitivity': [],
        'val_specificity': [],
        'val_precision': [],
        'val_balanced_accuracy': [],
        'val_mcc': [],
        'val_kappa': [],
        'learning_rate': [],
        'metadata': {
            'random_seed': args.seed,
            'deterministic': args.deterministic,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'model_name': args.architecture,
            'timestamp': datetime.now().isoformat()
        }
    }

    best_val_loss = float('inf')
    for epoch in range(config['epochs']):

        if unfreeze_epoch is not None and epoch == unfreeze_epoch and fine_tune_lr is not None:
            logging.info(f"Epoch {epoch+1}: Unfreezing backbone for fine-tuning")
            optimizer = training_utils.unfreeze_model_backbone(model, fine_tune_lr, optimizer)

        train_metrics = training_utils.train_epoch_mil_no_embeddings(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            use_amp=use_mixed_precision,
            deterministic=args.deterministic,
            gradient_clip_value=gradient_clip_value
        )

        val_metrics = training_utils.validate_mil_no_embeddings(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )

        history['learning_rate'].append(optimizer.param_groups[0]['lr'])






        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics.get('accuracy', 0.0))
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics.get('accuracy', 0.0))
        history['val_f1'].append(val_metrics.get('f1', 0.0))
        history['val_auc'].append(val_metrics.get('auc', 0.0))
        history['val_sensitivity'].append(val_metrics.get('sensitivity', 0.0))
        history['val_specificity'].append(val_metrics.get('specificity', 0.0))
        history['val_precision'].append(val_metrics.get('precision', 0.0))
        history['val_balanced_accuracy'].append(val_metrics.get('balanced_accuracy', 0.0))
        history['val_mcc'].append(val_metrics.get('mcc', 0.0))
        history['val_kappa'].append(val_metrics.get('kappa', 0.0))

        _log_epoch_metrics(
            epoch=epoch,
            total_epochs=config['epochs'],
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            multiclass=_is_multiclass_setup(args),
        )

        if early_stopping and early_stopping(val_metrics['loss']):
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            model_path = PathConfigurator.get_checkpoint_path(args, paths, config)
            training_utils.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
                filepath=str(model_path),
                seed=args.seed,
                deterministic=args.deterministic
            )
            logging.info(f"Saved best model with validation loss: {val_metrics['loss']:.4f} to {model_path}")


        if (epoch + 1) % save_frequency == 0:
            periodic_path = PathConfigurator.get_periodic_checkpoint_path(args, paths, config, epoch + 1)
            training_utils.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
                filepath=str(periodic_path),
                seed=args.seed,
                deterministic=args.deterministic
            )
            logging.info(f"Saved periodic checkpoint to {periodic_path}")

    history_path = PathConfigurator.get_training_history_path(args, paths, config)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    try:
        path = PathConfigurator.get_training_path(args, paths)
        plot_training_history(history, save_dir=str(path))
        logging.info(f"Training plots saved to {path}/")
    except Exception as e:
        logging.warning(f"Could not save training plots: {e}")

    logging.info(f"Training complete. Training history saved to {history_path}")


train_mil_model = train_mil_model_image