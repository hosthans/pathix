import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, get_worker_info
import logging
from typing import Dict, Any, Optional, List, Iterable, Tuple
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef,
    cohen_kappa_score,
    top_k_accuracy_score,
)
import random
import os
from datetime import datetime

import sys
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import src.utils.metrics_utils as metrics_utils


def _is_multiclass_criterion(criterion: nn.Module) -> bool:
    # MulticlassFocalLoss (defined further down in this module -- fine to reference here since
    # this name is only resolved when the function is actually called, by which point the whole
    # module has finished loading) is multiclass just like CrossEntropyLoss; missing it here made
    # every focal-loss training epoch get treated as binary -- wrong prediction extraction
    # (flattened logits instead of argmax) that crashed confusion_matrix() with a 4x sample-count
    # mismatch (num_classes=4) as soon as the first epoch's metrics were computed.
    return isinstance(criterion, (nn.CrossEntropyLoss, MulticlassFocalLoss))


def _flatten_binary_outputs_and_labels(outputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = outputs.reshape(-1)
    labels = labels.reshape(-1)
    if outputs.numel() != labels.numel():
        raise ValueError(
            f"Binary output/label size mismatch: outputs={outputs.shape}, labels={labels.shape}"
        )
    return outputs, labels


def _tensor_1d_list(tensor: torch.Tensor) -> List[float]:
    return np.asarray(tensor.detach().cpu()).reshape(-1).tolist()


def _normalize_pattern_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _matches_parameter_name(name: str, patterns: List[str]) -> bool:
    for pattern in patterns:
        if not pattern:
            continue
        if name == pattern or name.startswith(pattern) or pattern in name:
            return True
    return False


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _sanitize_binary_labels(labels: torch.Tensor) -> torch.Tensor:
    labels = torch.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
    return labels.clamp(min=0.0, max=1.0)


def _sanitize_logits_for_loss(logits: torch.Tensor, multiclass: bool) -> torch.Tensor:
    clamp = 50.0 if multiclass else 30.0
    logits = torch.nan_to_num(logits, nan=0.0, posinf=clamp, neginf=-clamp)
    return logits.clamp(min=-clamp, max=clamp)


def _smooth_binary_labels(labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0.0:
        return labels
    eps = float(min(max(smoothing, 0.0), 1.0))
    return labels * (1.0 - eps) + 0.5 * eps


def _has_non_finite_gradients(model: nn.Module) -> bool:
    for param in model.parameters():
        grad = getattr(param, "grad", None)
        if grad is not None and not torch.all(torch.isfinite(grad)):
            return True
    return False

def _reset_or_pop_wsigraph_fallback_count(model: nn.Module) -> int:
    actual_model = _unwrap_model(model)
    pop_fn = getattr(actual_model, "pop_region_only_fallback_count", None)
    if callable(pop_fn):
        try:
            return int(pop_fn())
        except Exception:
            return 0
    return 0


def apply_fine_tuning_config(model: nn.Module, fine_tuning_config: Optional[Dict[str, Any]] = None) -> nn.Module:
    """Apply config-driven freezing/unfreezing rules to a model's parameters."""
    config = fine_tuning_config or {}
    if not isinstance(config, dict):
        return model

    actual_model = _unwrap_model(model)

    freeze_all = bool(config.get("freeze_all", False))
    freeze_backbone = bool(config.get("freeze_backbone", False))
    freeze_feature_extractor = bool(config.get("freeze_feature_extractor", False))

    freeze_patterns = []
    freeze_patterns.extend(_normalize_pattern_list(config.get("freeze_layers")))
    freeze_patterns.extend(_normalize_pattern_list(config.get("freeze_modules")))
    freeze_patterns.extend(_normalize_pattern_list(config.get("freeze_patterns")))

    trainable_patterns = []
    trainable_patterns.extend(_normalize_pattern_list(config.get("trainable_layers")))
    trainable_patterns.extend(_normalize_pattern_list(config.get("trainable_modules")))
    trainable_patterns.extend(_normalize_pattern_list(config.get("trainable_patterns")))

    if freeze_all:
        for param in actual_model.parameters():
            param.requires_grad = False

    if freeze_backbone and hasattr(actual_model, "backbone"):
        for param in actual_model.backbone.parameters():
            param.requires_grad = False

    if freeze_feature_extractor and hasattr(actual_model, "feature_extractor"):
        for param in actual_model.feature_extractor.parameters():
            param.requires_grad = False

    if freeze_patterns:
        for name, param in actual_model.named_parameters():
            if _matches_parameter_name(name, freeze_patterns):
                param.requires_grad = False

    if trainable_patterns:
        for name, param in actual_model.named_parameters():
            if _matches_parameter_name(name, trainable_patterns):
                param.requires_grad = True

    initialized_params = [
        param for param in actual_model.parameters()
        if not isinstance(param, nn.parameter.UninitializedParameter)
    ]
    trainable_count = sum(param.numel() for param in initialized_params if param.requires_grad)
    total_count = sum(param.numel() for param in initialized_params)
    logging.info(
        "Applied fine-tuning config: trainable_params=%d/%d | freeze_all=%s | freeze_backbone=%s | freeze_feature_extractor=%s | freeze_patterns=%s | trainable_patterns=%s",
        trainable_count,
        total_count,
        freeze_all,
        freeze_backbone,
        freeze_feature_extractor,
        freeze_patterns,
        trainable_patterns,
    )

    return model


def _trainable_parameters(model: nn.Module):
    return [param for param in model.parameters() if param.requires_grad]


def _unpack_graph_model_output(model_output: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.is_tensor(model_output):
        return model_output, torch.zeros((), device=device)

    if isinstance(model_output, dict):
        if "logits" not in model_output or not torch.is_tensor(model_output["logits"]):
            raise ValueError("Graph model output dict must contain tensor key 'logits'.")

        logits = model_output["logits"]
        aux_terms: List[torch.Tensor] = []

        for key in ("aux_loss", "regularization_loss"):
            value = model_output.get(key, None)
            if torch.is_tensor(value):
                aux_terms.append(value)

        losses_dict = model_output.get("losses", None)
        if isinstance(losses_dict, dict):
            for value in losses_dict.values():
                if torch.is_tensor(value):
                    aux_terms.append(value)

        if aux_terms:
            aux_loss = torch.stack([term.reshape(()) for term in aux_terms]).sum()
        else:
            aux_loss = torch.zeros((), device=device)
        return logits, aux_loss

    raise TypeError(
        "Unsupported graph model output type. Expected Tensor or dict with key 'logits'."
    )


def _compute_graph_consistency_and_domain_losses(
    model: nn.Module,
    data: Any,
    clean_output: Any,
    clean_logits: torch.Tensor,
    device: torch.device,
    multiclass: bool,
    consistency_specs: List[Dict[str, Any]],
    domain_adversarial_specs: List[Dict[str, Any]],
    rng: "np.random.Generator",
) -> tuple[torch.Tensor, torch.Tensor]:
    from src.dataset.graph_augment import apply_graph_augmentations

    combined_specs = list(consistency_specs) + list(domain_adversarial_specs)

    try:
        data_aug = data.clone()
        apply_graph_augmentations(data_aug, combined_specs, rng)
        aug_output = model(data_aug)
    except torch.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        logging.warning(
            "OOM computing the augmented (consistency/domain-adversarial) forward pass -- "
            "skipping this step's consistency/domain loss, keeping the main step's loss/backward."
        )
        return torch.zeros((), device=device), torch.zeros((), device=device)

    aug_logits, _ = _unpack_graph_model_output(aug_output, device)
    aug_logits = _sanitize_logits_for_loss(aug_logits, multiclass=multiclass)

    consistency_loss = torch.zeros((), device=device)
    if consistency_specs:
        if multiclass:
            p_clean = torch.softmax(clean_logits, dim=-1)
            p_aug = torch.softmax(aug_logits, dim=-1)
        else:
            p_clean = torch.sigmoid(clean_logits)
            p_aug = torch.sigmoid(aug_logits)
        consistency_loss = torch.nn.functional.mse_loss(p_clean, p_aug)

    domain_loss = torch.zeros((), device=device)
    if domain_adversarial_specs:
        clean_domain_logits = clean_output.get("domain_logits") if isinstance(clean_output, dict) else None
        aug_domain_logits = aug_output.get("domain_logits") if isinstance(aug_output, dict) else None
        if torch.is_tensor(clean_domain_logits) and torch.is_tensor(aug_domain_logits):
            domain_logits = torch.cat([clean_domain_logits, aug_domain_logits], dim=0)
            domain_labels = torch.cat(
                [
                    torch.zeros_like(clean_domain_logits),
                    torch.ones_like(aug_domain_logits),
                ],
                dim=0,
            )
            domain_loss = torch.nn.functional.binary_cross_entropy_with_logits(domain_logits, domain_labels)

    return consistency_loss, domain_loss


def _compute_epoch_metrics(
    labels: List,
    preds: List,
    raw_preds: Optional[List],
    multiclass: bool,
    ordinal_class_order: Optional[List[int]] = None,
) -> Dict[str, Any]:
    if multiclass:
        y_true = np.asarray(labels).astype(int)
        y_pred = np.asarray(preds).astype(int)
        if y_true.size == 0:
            return {
                "accuracy": 0.0,
                "sensitivity": 0.0,
                "specificity": 0.0,
                "precision": 0.0,
                "f1": 0.0,
                "auc": 0.0,
                "balanced_accuracy": 0.0,
                "precision_weighted": 0.0,
                "recall_weighted": 0.0,
                "f1_weighted": 0.0,
                "mcc": 0.0,
                "kappa": 0.0,
                "kappa_quadratic": 0.0,
                "top2_accuracy": 0.0,
            }

        if raw_preds is not None and len(raw_preds) > 0:
            y_score_shape = np.asarray(raw_preds).shape
            if len(y_score_shape) == 2 and y_score_shape[1] > 1:
                class_labels = np.arange(y_score_shape[1])
            else:
                class_labels = np.unique(np.concatenate([y_true, y_pred]))
        else:
            class_labels = np.unique(np.concatenate([y_true, y_pred]))

        accuracy = 100.0 * accuracy_score(y_true, y_pred)
        balanced_acc = 100.0 * balanced_accuracy_score(y_true, y_pred)
        precision = 100.0 * precision_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)
        recall_macro = 100.0 * recall_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)
        f1 = 100.0 * f1_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)
        precision_weighted = 100.0 * precision_score(y_true, y_pred, labels=class_labels, average="weighted", zero_division=0)
        recall_weighted = 100.0 * recall_score(y_true, y_pred, labels=class_labels, average="weighted", zero_division=0)
        f1_weighted = 100.0 * f1_score(y_true, y_pred, labels=class_labels, average="weighted", zero_division=0)
        mcc = 100.0 * matthews_corrcoef(y_true, y_pred)
        kappa = 100.0 * cohen_kappa_score(y_true, y_pred)
        # Quadratic-weighted kappa: penalizes confusions between classes further apart on an
        # ordinal severity scale more than adjacent-class confusions -- matches the official
        # CAMELYON17 pN-stage evaluation protocol. This only makes sense if the raw class ids
        # already increase with severity; ordinal_class_order remaps them (e.g. the camelyon
        # `subtype` label field's ids come from alphabetically sorting itc/macro/micro/negative,
        # NOT clinical order) before the distance calc. Unweighted `kappa` above needs no such
        # remap since it treats every confusion identically regardless of label order.
        if ordinal_class_order is not None:
            order = np.asarray(ordinal_class_order)
            y_true_ord = order[y_true]
            y_pred_ord = order[np.clip(y_pred, 0, len(order) - 1)]
            # sklearn's quadratic weight matrix is built from `labels`' *position*, not its
            # value, so this must be sorted rank order (0..n-1) -- an unsorted list here would
            # silently re-scramble the adjacency the remap above just fixed.
            kappa_labels = np.sort(order[class_labels])
        else:
            y_true_ord, y_pred_ord = y_true, y_pred
            kappa_labels = class_labels
        # Pass `labels` explicitly so the confusion/weight matrix always spans the full class
        # range: sklearn otherwise infers it from the classes observed in just this epoch, which
        # would silently rescale the quadratic distance penalty on any epoch missing a class.
        kappa_quadratic = 100.0 * cohen_kappa_score(
            y_true_ord, y_pred_ord, labels=kappa_labels, weights='quadratic'
        )

        auc = 0.0
        top2_accuracy = 0.0
        if raw_preds is not None and len(raw_preds) > 0:
            y_score = np.asarray(raw_preds)
            if y_score.ndim == 1:
                y_score = y_score.reshape(-1, 1)

            row_sums = y_score.sum(axis=1) if y_score.ndim == 2 else None
            looks_like_prob = (
                y_score.ndim == 2
                and np.all(y_score >= 0.0)
                and np.all(y_score <= 1.0)
                and row_sums is not None
                and np.allclose(row_sums, 1.0, atol=1e-3)
            )
            if not looks_like_prob:
                shifted = y_score - np.max(y_score, axis=1, keepdims=True)
                exp_scores = np.exp(shifted)
                y_score = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            try:
                auc = 100.0 * roc_auc_score(
                    y_true,
                    y_score,
                    multi_class="ovo",
                    average="macro",
                    labels=np.arange(y_score.shape[1]),
                )
            except ValueError:
                auc = 0.0

            try:
                n_classes = y_score.shape[1]
                if n_classes > 2:
                    top2_accuracy = 100.0 * top_k_accuracy_score(
                        y_true,
                        y_score,
                        k=min(2, n_classes),
                        labels=np.arange(n_classes),
                    )
            except ValueError:
                top2_accuracy = 0.0

        return {
            "accuracy": accuracy,
            "sensitivity": recall_macro,
            "specificity": balanced_acc,
            "precision": precision,
            "f1": f1,
            "auc": auc,
            "balanced_accuracy": balanced_acc,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "mcc": mcc,
            "kappa": kappa,
            "kappa_quadratic": kappa_quadratic,
            "top2_accuracy": top2_accuracy,
        }

    return metrics_utils.calculate_basic_metrics(labels, preds, raw_preds)


def extract_annotation_vocab_from_graph_batch(data) -> list[str]:
    """Read data.meta['annotation_label_vocab'] off a (possibly PyG-batched) graph."""
    meta_obj = getattr(data, 'meta', None)
    if isinstance(meta_obj, dict):
        raw_vocab = meta_obj.get('annotation_label_vocab')
        if isinstance(raw_vocab, list):
            return [str(v) for v in raw_vocab]

    if isinstance(meta_obj, list):
        for meta_entry in meta_obj:
            if isinstance(meta_entry, dict):
                raw_vocab = meta_entry.get('annotation_label_vocab')
                if isinstance(raw_vocab, list):
                    return [str(v) for v in raw_vocab]

    return []


def _inverse_frequency_weights(counts: np.ndarray, num_classes: int) -> np.ndarray:
    counts = counts.copy()
    counts[counts <= 0.0] = 1.0
    weights = counts.sum() / (float(num_classes) * counts)
    return weights / weights.mean()


def _extract_multiclass_label_values(dataset: Any) -> List[int]:
    """Scan common dataset layouts for one integer class label per sample."""
    label_sources: List[Iterable] = []
    if hasattr(dataset, "samples") and isinstance(getattr(dataset, "samples"), list):
        try:
            label_sources.append([s.get("label") for s in dataset.samples if isinstance(s, dict) and "label" in s])
        except Exception:
            pass

    base = getattr(dataset, "base_dataset", None)
    if base is None:
        base = getattr(dataset, "base", None)
    if base is not None and hasattr(base, "samples") and isinstance(getattr(base, "samples"), list):
        try:
            label_sources.append([s.get("label") for s in base.samples if isinstance(s, dict) and "label" in s])
        except Exception:
            pass

    label_values: List[int] = []
    for src in label_sources:
        if src:
            try:
                label_values = [int(x) for x in src if x is not None]
                if label_values:
                    break
            except Exception:
                continue

    if not label_values:
        # WSICachedGraphDataset/WSIGraphDataset (src/dataset/graph_dataset.py) expose neither
        # `.samples` nor `.base*.samples` -- their per-slide records live in `.records`, and worse,
        # when `label_field` is set (every multiclass graph run: subtype, tissue_type, ...)
        # `.records[i]["label"]` is a dummy 0.0 placeholder, not the real class -- the actual label
        # only exists inside the cached graph file itself (`data["slide"][f"{label_field}_y"]`,
        # read by __getitem__ into `data.y`). So for every graph/gnn multiclass run this function
        # fell through both checks above and silently returned None here, meaning
        # configure_training_components built a plain unweighted CrossEntropyLoss regardless of
        # how skewed the label distribution actually was. Fall back to scanning dataset[idx].y --
        # the one label representation every such dataset actually populates correctly -- mirroring
        # infer_node_multiclass_weights_from_dataset's identical scan-the-whole-dataset approach
        # right below (same one-time, before-epoch-1 cost, already accepted there).
        if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
            try:
                n = len(dataset)
            except Exception:
                n = 0
            scanned: List[int] = []
            for idx in range(n):
                try:
                    y = dataset[idx].y
                    scanned.append(int(y.view(-1)[0].item()) if torch.is_tensor(y) else int(y))
                except Exception:
                    continue
            if scanned:
                label_values = scanned

    return label_values


def infer_multiclass_weights_from_dataset(dataset: Any, num_classes: int) -> Optional[np.ndarray]:
    """Infer inverse-frequency class weights from common dataset layouts, or None if labels can't be extracted."""
    class_weights, _ = infer_multiclass_class_and_sample_weights_from_dataset(dataset, num_classes)
    return class_weights


def infer_multiclass_class_and_sample_weights_from_dataset(
    dataset: Any, num_classes: int
) -> tuple[Optional[np.ndarray], Optional[List[float]]]:
    """Inverse-frequency class weights (for the loss) and per-sample weights (for a sampler).

    Both are derived from one scan of the dataset -- for a cached graph dataset, `dataset[idx]`
    means a `torch.load` of that slide's graph file, so computing them from two separate scans
    (as calling infer_multiclass_weights_from_dataset then a per-sample equivalent would) doubles
    a potentially expensive one-time disk-I/O pass for no reason.

    Class weights rescale each sample's gradient contribution; sample weights (for
    WeightedRandomSampler, replacement=True) change how often each sample is drawn per epoch --
    loss reweighting alone doesn't fix a rare class still being *seen* far less often.
    """
    if int(num_classes) <= 1:
        return None, None

    label_values = _extract_multiclass_label_values(dataset)
    if not label_values:
        return None, None

    labels_arr = np.asarray(label_values, dtype=int)
    counts = np.bincount(labels_arr, minlength=int(num_classes)).astype(float)
    class_weights = _inverse_frequency_weights(counts, int(num_classes)).astype(np.float32)
    sample_weights = class_weights[labels_arr].astype(np.float64).tolist()
    return class_weights, sample_weights


def infer_node_multiclass_weights_from_dataset(
    dataset: Any,
    node_type: str,
    target_field: str,
    class_names: List[str],
) -> Optional["torch.Tensor"]:
    """Infer inverse-frequency class weights for a region/patch-level multiclass head by scanning the dataset."""
    from src.tissue_region_extraction.model import TissueRegionGraphModel  # local import: avoids a
    # training_utils <-> tissue_region_extraction.model import cycle at module load time (model.py
    # already imports from src.utils.graph_patch_utils, not this module, but training_utils is
    # imported very early by src/pipelines/train.py -- keeping this import local/lazy sidesteps
    # ever having to reason about load order between the two modules).

    if len(class_names) <= 1:
        return None

    counts = np.zeros((len(class_names),), dtype=np.float64)
    n_scanned = 0
    for idx in range(len(dataset)):
        try:
            data = dataset[idx]
        except Exception:
            continue

        if not hasattr(data, "node_types") or node_type not in data.node_types:
            continue

        target_y = getattr(data[node_type], f"{target_field}_y", None)
        if not torch.is_tensor(target_y):
            continue

        vocab: List[str] = []
        if hasattr(data, "meta") and isinstance(data.meta, dict):
            raw_vocab = data.meta.get(f"{target_field}_label_vocab")
            if isinstance(raw_vocab, list):
                vocab = [str(v) for v in raw_vocab]

        n_scanned += 1
        for local_id in target_y.tolist():
            if local_id < 0 or local_id >= len(vocab):
                continue
            mapped = TissueRegionGraphModel._canonical_class_id(vocab[local_id], class_names)
            if mapped is not None:
                counts[mapped] += 1

    if n_scanned == 0 or counts.sum() <= 0:
        return None

    weights = _inverse_frequency_weights(counts, len(class_names))
    return torch.tensor(weights, dtype=torch.float32)


def infer_graph_oversampling_weights(
    dataset: Any,
    specs: List[Tuple[str, str, List[str]]],
) -> Optional[List[float]]:
    """Compute per-graph WeightedRandomSampler weights that boost graphs containing rare-class nodes."""
    from src.tissue_region_extraction.model import TissueRegionGraphModel  # local import: see
    # infer_node_multiclass_weights_from_dataset's identical import for the cycle-avoidance
    # rationale -- applies the same way here.

    active_specs = [(nt, tf, cn) for nt, tf, cn in specs if tf and cn]
    if not active_specs:
        return None

    n = len(dataset)
    counts = [np.zeros((len(cn),), dtype=np.float64) for _, _, cn in active_specs]
    graph_present: List[List[set]] = [[set() for _ in active_specs] for _ in range(n)]
    n_scanned = 0

    for idx in range(n):
        try:
            data = dataset[idx]
        except Exception:
            continue
        if not hasattr(data, "node_types"):
            continue
        found_any = False
        for spec_i, (node_type, target_field, class_names) in enumerate(active_specs):
            if node_type not in data.node_types:
                continue
            target_y = getattr(data[node_type], f"{target_field}_y", None)
            if not torch.is_tensor(target_y):
                continue
            vocab: List[str] = []
            if hasattr(data, "meta") and isinstance(data.meta, dict):
                raw_vocab = data.meta.get(f"{target_field}_label_vocab")
                if isinstance(raw_vocab, list):
                    vocab = [str(v) for v in raw_vocab]
            for local_id in target_y.tolist():
                if local_id < 0 or local_id >= len(vocab):
                    continue
                mapped = TissueRegionGraphModel._canonical_class_id(vocab[local_id], class_names)
                if mapped is not None:
                    counts[spec_i][mapped] += 1
                    graph_present[idx][spec_i].add(mapped)
                    found_any = True
        if found_any:
            n_scanned += 1

    if n_scanned == 0:
        return None

    class_weights: List[Optional[np.ndarray]] = []
    for spec_i, (_, _, class_names) in enumerate(active_specs):
        c = counts[spec_i].copy()
        if c.sum() <= 0:
            class_weights.append(None)
            continue
        c[c <= 0.0] = 1.0
        w = c.sum() / (float(len(class_names)) * c)
        w = w / w.mean()
        class_weights.append(w)

    graph_weights = [1.0] * n
    for idx in range(n):
        best = 1.0
        for spec_i in range(len(active_specs)):
            w = class_weights[spec_i]
            if w is None:
                continue
            for cls_id in graph_present[idx][spec_i]:
                best = max(best, float(w[cls_id]))
        graph_weights[idx] = best

    if max(graph_weights) - min(graph_weights) < 1e-9:
        return None

    return graph_weights


def set_all_seeds(seed: int = 42, deterministic: bool = False, strict: bool = False) -> None:
    """Set all random seeds (Python, NumPy, PyTorch, CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        logging.info("TF32 disabled for numerical stability")

    if deterministic or strict:
        # strict (--strict_deterministic): ops without a deterministic CUDA kernel raise instead of warning.
        torch.use_deterministic_algorithms(True, warn_only=not strict)
        logging.info(
            "torch.use_deterministic_algorithms(True, warn_only=%s) enabled -- ops without a "
            "deterministic CUDA kernel will %s (search logs for 'does not have a "
            "deterministic implementation').",
            not strict, "raise" if strict else "log a warning rather than silently varying run-to-run",
        )

    logging.info(f"All random seeds set to {seed} for reproducibility")
    logging.info("Note: Deterministic mode is enabled, which may slightly reduce performance")


def dataloader_worker_init_fn(worker_id: int) -> None:
    """Reseed numpy/random (and any dataset `_aug_rng`) inside a DataLoader worker process from its own torch seed."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    worker_info = get_worker_info()
    if worker_info is None:
        return
    dataset = worker_info.dataset
    for _ in range(4):  # unwrap Subset/wrapper datasets that hold `.dataset`
        if hasattr(dataset, "_aug_rng"):
            dataset._aug_rng = np.random.default_rng(worker_seed)
            break
        inner = getattr(dataset, "dataset", None)
        if inner is None:
            break
        dataset = inner


def train_epoch_img(model: nn.Module,
               train_loader: DataLoader,
               optimizer: torch.optim.Optimizer,
               criterion: nn.Module,
               scaler: torch.amp.GradScaler,
               device: torch.device,
               use_amp: bool = True,
               deterministic: bool = True,
               gradient_clip_value: float = None) -> Dict[str, Any]:
    """Train the image classification model for one epoch."""
    model.train()
    total_loss = 0.0
    all_labels = []
    all_raw_preds = []



    if deterministic and torch.cuda.is_available():
        if not torch.backends.cudnn.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logging.debug("Set CUDNN to deterministic mode for training")

    multiclass = _is_multiclass_criterion(criterion)

    for inputs, labels, batch_metadata in tqdm(train_loader, desc="Training", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if multiclass:
            labels = labels.long()
        else:
            labels = labels.float()

        optimizer.zero_grad(set_to_none=True)


        if use_amp and device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                outputs = model(inputs)
                if not multiclass:
                    outputs, labels = _flatten_binary_outputs_and_labels(outputs, labels)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            if gradient_clip_value is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
            scaler.step(optimizer)
            scaler.update()
        else:

            outputs = model(inputs)
            if not multiclass:
                outputs, labels = _flatten_binary_outputs_and_labels(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            if gradient_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
            optimizer.step()


        with torch.no_grad():
            all_labels.extend(_tensor_1d_list(labels))
            all_raw_preds.extend(outputs.detach().cpu().numpy().tolist())

        total_loss += loss.item()
        del inputs, labels, outputs, loss, batch_metadata
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if all_raw_preds:
        raw_pred_array = np.asarray(all_raw_preds)
        if multiclass:
            all_preds = np.argmax(raw_pred_array, axis=1).tolist()
        else:
            all_preds = (raw_pred_array > 0).astype(int).reshape(-1).tolist()
    else:
        all_preds = []


    metrics = _compute_epoch_metrics(
        all_labels,
        all_preds,
        all_raw_preds,
        multiclass=multiclass,
    )


    metrics['loss'] = total_loss / len(train_loader)


    logging.debug(f"Train metrics - Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.2f}%, " +
                 f"Sensitivity: {metrics['sensitivity']:.2f}%, Specificity: {metrics['specificity']:.2f}%, " +
                 f"F1: {metrics['f1']:.2f}%, AUC: {metrics.get('auc', 0):.2f}%")

    return metrics

def validate_img(model: nn.Module,
            val_loader: DataLoader,
            criterion: nn.Module,
            device: torch.device) -> Dict[str, Any]:
    """Validate the image classification model on val_loader."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_raw_preds = []
    metadata = []


    if len(val_loader) == 0:
        logging.warning("Validation dataloader is empty!")
        metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0,
            'precision': 0.0,
            'f1': 0.0,
            'auc': 50.0
        }
        return metrics

    multiclass = _is_multiclass_criterion(criterion)

    with torch.no_grad():
        for inputs, labels, batch_metadata in tqdm(val_loader, desc="Validating", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            if multiclass:
                labels = labels.long()
            else:
                labels = labels.float()

            outputs = model(inputs)
            if not multiclass:
                outputs, labels = _flatten_binary_outputs_and_labels(outputs, labels)
            loss = criterion(outputs, labels)

            if multiclass:
                preds = torch.argmax(outputs, dim=1)
            else:
                preds = (outputs > 0).float()


            all_preds.extend(_tensor_1d_list(preds))
            all_labels.extend(_tensor_1d_list(labels))
            if multiclass:
                all_raw_preds.extend(outputs.detach().cpu().numpy().tolist())
            else:
                all_raw_preds.extend(_tensor_1d_list(outputs))


            for i in range(len(labels)):
                meta_dict = {k: v[i] if isinstance(v, list) else v for k, v in batch_metadata.items()}
                metadata.append(meta_dict)

            total_loss += loss.item()


    metrics = _compute_epoch_metrics(
        all_labels,
        all_preds,
        all_raw_preds,
        multiclass=multiclass,
    )


    metrics['loss'] = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0


    logging.debug(f"Validation metrics - Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.2f}%, " +
                 f"Sensitivity: {metrics['sensitivity']:.2f}%, Specificity: {metrics['specificity']:.2f}%, " +
                 f"F1: {metrics['f1']:.2f}%, AUC: {metrics.get('auc', 0):.2f}%")

    return metrics


def train_epoch_img_emb(model: nn.Module,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                scaler: torch.amp.GradScaler,
                device: torch.device,
                use_amp: bool = True,
                deterministic: bool = True,
                gradient_clip_value: float = None) -> Dict[str, Any]:
    """Train the image+embedding classification model for one epoch."""
    model.train()
    total_loss = 0.0
    all_labels, all_raw_preds = [], []


    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    multiclass = _is_multiclass_criterion(criterion)

    for images, embeddings, labels, batch_metadata in tqdm(
        train_loader, desc="Training", leave=False
    ):
        images = images.to(device, non_blocking=True)
        embeddings = embeddings.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if multiclass:
            labels = labels.long()
        else:
            labels = labels.float()

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images, embeddings)
                if not multiclass:
                    outputs, labels = _flatten_binary_outputs_and_labels(outputs, labels)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            if gradient_clip_value is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, embeddings)
            if not multiclass:
                outputs, labels = _flatten_binary_outputs_and_labels(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            if gradient_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
            optimizer.step()


        with torch.no_grad():
            all_labels.extend(_tensor_1d_list(labels))
            all_raw_preds.extend(outputs.detach().cpu().numpy().tolist())

        total_loss += loss.item()
        del images, embeddings, labels, outputs, loss, batch_metadata
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    raw_pred_array = np.asarray(all_raw_preds)
    if multiclass:
        all_preds = np.argmax(raw_pred_array, axis=1).tolist()
    else:
        all_preds = (raw_pred_array > 0).astype(int).reshape(-1).tolist()


    metrics = _compute_epoch_metrics(
        all_labels,
        all_preds,
        all_raw_preds,
        multiclass=multiclass,
    )

    metrics["loss"] = total_loss / len(train_loader)

    logging.debug(
        f"Train | Loss {metrics['loss']:.4f} | "
        f"Acc {metrics['accuracy']:.2f}% | "
        f"Sens {metrics['sensitivity']:.2f}% | "
        f"Spec {metrics['specificity']:.2f}% | "
        f"F1 {metrics['f1']:.2f}% | "
        f"AUC {metrics.get('auc', 0):.2f}%"
    )

    return metrics

def validate_img_emb(model: nn.Module,
             val_loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Dict[str, Any]:
    """Validate the image+embedding classification model on val_loader."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_raw_preds = [], [], []
    metadata_all = []

    if len(val_loader) == 0:
        logging.warning("Validation dataloader is empty!")
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "auc": 50.0
        }

    multiclass = _is_multiclass_criterion(criterion)

    with torch.no_grad():
        for images, embeddings, labels, batch_metadata in tqdm(
            val_loader, desc="Validating", leave=False
        ):
            images = images.to(device)
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            if multiclass:
                labels = labels.long()
            else:
                labels = labels.float()

            outputs = model(images, embeddings)
            if not multiclass:
                outputs, labels = _flatten_binary_outputs_and_labels(outputs, labels)
            loss = criterion(outputs, labels)

            if multiclass:
                preds = torch.argmax(outputs, dim=1)
            else:
                preds = (outputs > 0).float()

            all_preds.extend(_tensor_1d_list(preds))
            all_labels.extend(_tensor_1d_list(labels))
            if multiclass:
                all_raw_preds.extend(outputs.detach().cpu().numpy().tolist())
            else:
                all_raw_preds.extend(_tensor_1d_list(outputs))

            metadata_all.append(batch_metadata)

            total_loss += loss.item()

    metrics = _compute_epoch_metrics(
        all_labels,
        all_preds,
        all_raw_preds,
        multiclass=multiclass,
    )

    metrics["loss"] = total_loss / len(val_loader)

    logging.debug(
        f"Val | Loss {metrics['loss']:.4f} | "
        f"Acc {metrics['accuracy']:.2f}% | "
        f"Sens {metrics['sensitivity']:.2f}% | "
        f"Spec {metrics['specificity']:.2f}% | "
        f"F1 {metrics['f1']:.2f}% | "
        f"AUC {metrics.get('auc', 0):.2f}%"
    )

    return metrics


def train_epoch_mil(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool = True,
    deterministic: bool = True,
    gradient_clip_value: float = None
):
    """Train the MIL (with embeddings) model for one epoch."""
    model.train()
    total_loss = 0.0

    all_labels, all_raw_preds = [], []

    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    multiclass = _is_multiclass_criterion(criterion)

    for images_flat, bag_idx, labels, embeds_flat in tqdm(
        train_loader, desc="Training (MIL)", leave=False
    ):
        images_flat = images_flat.to(device, non_blocking=True)
        bag_idx = bag_idx.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if multiclass:
            labels = labels.long()
        else:
            labels = labels.float()
        embeds_flat = embeds_flat.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)


        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits, _ = model(images_flat, bag_idx, embeds_flat)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            if gradient_clip_value is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(images_flat, bag_idx, embeds_flat)
            loss = criterion(logits, labels)
            loss.backward()
            if gradient_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
            optimizer.step()


        with torch.no_grad():
            all_labels.extend(labels.cpu().numpy())
            all_raw_preds.extend(logits.detach().cpu().numpy())

        total_loss += loss.item()
        del images_flat, bag_idx, labels, embeds_flat, logits, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    raw_pred_array = np.asarray(all_raw_preds)
    if multiclass:
        all_preds = np.argmax(raw_pred_array, axis=1).tolist()
    else:
        all_preds = (raw_pred_array > 0).astype(int).reshape(-1).tolist()

    metrics = _compute_epoch_metrics(
        all_labels,
        all_preds,
        all_raw_preds,
        multiclass=multiclass,
    )

    metrics["loss"] = total_loss / len(train_loader)

    torch.cuda.empty_cache()

    return metrics


def validate_mil(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
):
    """Validate the MIL (with embeddings) model on val_loader."""
    model.eval()
    total_loss = 0.0

    all_preds, all_labels, all_raw_preds = [], [], []

    if len(val_loader) == 0:
        logging.warning("Validation dataloader is empty!")
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "auc": 50.0
        }

    multiclass = _is_multiclass_criterion(criterion)

    with torch.no_grad():
        for images_flat, bag_idx, labels, embeds_flat in tqdm(
            val_loader, desc="Validating (MIL)", leave=False
        ):
            images_flat = images_flat.to(device)
            bag_idx = bag_idx.to(device)
            labels = labels.to(device)
            if multiclass:
                labels = labels.long()
            else:
                labels = labels.float()
            embeds_flat = embeds_flat.to(device)

            logits, _ = model(images_flat, bag_idx, embeds_flat)
            loss = criterion(logits, labels)

            if multiclass:
                preds = torch.argmax(logits, dim=1)
            else:
                preds = (logits > 0).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_raw_preds.extend(logits.detach().cpu().numpy())

            total_loss += loss.item()

    metrics = _compute_epoch_metrics(
        all_labels,
        all_preds,
        all_raw_preds,
        multiclass=multiclass,
    )

    metrics["loss"] = total_loss / len(val_loader)

    torch.cuda.empty_cache()

    return metrics


def train_epoch_mil_no_embeddings(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool = True,
    deterministic: bool = True,
    gradient_clip_value: float = None
):
    """Train the MIL (no embeddings) model for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels, all_raw_preds = [], [], []

    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    multiclass = _is_multiclass_criterion(criterion)

    for images_flat, bag_idx, labels in tqdm(
        train_loader, desc="Training (MIL)", leave=False
    ):
        images_flat = images_flat.to(device, non_blocking=True)
        bag_idx = bag_idx.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if multiclass:
            labels = labels.long()
        else:
            labels = labels.float()

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits, _ = model(images_flat, bag_idx)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            if gradient_clip_value is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(images_flat, bag_idx)
            loss = criterion(logits, labels)
            loss.backward()
            if gradient_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
            optimizer.step()

        with torch.no_grad():
            if multiclass:
                preds = torch.argmax(logits, dim=1)
            else:
                preds = (logits > 0).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_raw_preds.extend(logits.detach().cpu().numpy())

        total_loss += loss.item()
        del images_flat, bag_idx, labels, logits, preds, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    metrics = _compute_epoch_metrics(
        all_labels,
        all_preds,
        all_raw_preds,
        multiclass=multiclass,
    )
    metrics["loss"] = total_loss / len(train_loader)

    torch.cuda.empty_cache()
    return metrics

def validate_mil_no_embeddings(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
):
    """Validate the MIL (no embeddings) model on val_loader."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_raw_preds = [], [], []

    if len(val_loader) == 0:
        logging.warning("Validation dataloader is empty!")
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "auc": 50.0
        }

    multiclass = _is_multiclass_criterion(criterion)

    with torch.no_grad():
        for images_flat, bag_idx, labels in tqdm(
            val_loader, desc="Validating (MIL)", leave=False
        ):
            images_flat = images_flat.to(device)
            bag_idx = bag_idx.to(device)
            labels = labels.to(device)
            if multiclass:
                labels = labels.long()
            else:
                labels = labels.float()

            logits, _ = model(images_flat, bag_idx)
            loss = criterion(logits, labels)

            if multiclass:
                preds = torch.argmax(logits, dim=1)
            else:
                preds = (logits > 0).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_raw_preds.extend(logits.detach().cpu().numpy())

            total_loss += loss.item()

    metrics = _compute_epoch_metrics(
        all_labels,
        all_preds,
        all_raw_preds,
        multiclass=multiclass,
    )
    metrics["loss"] = total_loss / len(val_loader)

    torch.cuda.empty_cache()
    return metrics

def train_epoch_graph(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool = True,
    deterministic: bool = True,
    gradient_clip_value: float = None,
    gradient_accumulation_steps: int = 1,
    consistency_weight: float = 0.0,
    consistency_specs: Optional[List[Dict[str, Any]]] = None,
    consistency_p: float = 1.0,
    consistency_rng: Optional["np.random.Generator"] = None,
    domain_adversarial_weight: float = 0.0,
    domain_adversarial_specs: Optional[List[Dict[str, Any]]] = None,
    slide_label_smoothing: float = 0.0,
    ordinal_class_order: Optional[List[int]] = None,
):
    """Train the graph model for one epoch, with optional consistency/domain-adversarial losses."""
    model.train()
    total_loss = 0.0
    total_aux_loss = 0.0
    total_consistency_loss = 0.0
    total_domain_loss = 0.0
    processed_batches = 0
    skipped_non_finite_loss_batches = 0
    skipped_non_finite_grad_steps = 0

    all_labels, all_raw_preds = [], []

    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    multiclass = _is_multiclass_criterion(criterion)
    accum_steps = max(1, int(gradient_accumulation_steps))

    consistency_specs = consistency_specs or []
    domain_adversarial_specs = domain_adversarial_specs or []
    consistency_active = consistency_weight > 0.0 and bool(consistency_specs)
    domain_adversarial_active = domain_adversarial_weight > 0.0 and bool(domain_adversarial_specs)
    augment_and_compare_active = consistency_active or domain_adversarial_active
    if augment_and_compare_active and consistency_rng is None:
        consistency_rng = np.random.default_rng()

    _reset_or_pop_wsigraph_fallback_count(model)

    optimizer.zero_grad(set_to_none=True)

    for step_idx, data in enumerate(tqdm(train_loader, desc="Training (Graph)", leave=False), start=1):
        data = data.to(device)


        if hasattr(data, 'y') and data.y is not None:
            labels = data.y.view(-1).to(device)
        else:

            labels = data['slide'].y.view(-1).to(device)
        if multiclass:
            labels = labels.long()
        else:
            labels = _sanitize_binary_labels(labels.float())

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                model_output = model(data)
                logits, aux_loss = _unpack_graph_model_output(model_output, device)
                logits = _sanitize_logits_for_loss(logits, multiclass=multiclass)
                loss_labels = labels if multiclass else _smooth_binary_labels(labels, slide_label_smoothing)
                loss = criterion(logits, loss_labels)

                consistency_loss = torch.zeros((), device=device)
                domain_loss = torch.zeros((), device=device)
                if augment_and_compare_active and consistency_rng.random() < consistency_p:
                    consistency_loss, domain_loss = _compute_graph_consistency_and_domain_losses(
                        model, data, model_output, logits, device, multiclass,
                        consistency_specs, domain_adversarial_specs, consistency_rng,
                    )

                total_step_loss = (
                    loss + aux_loss
                    + float(consistency_weight) * consistency_loss
                    + float(domain_adversarial_weight) * domain_loss
                )

            if not (torch.isfinite(loss) and torch.isfinite(aux_loss) and torch.isfinite(consistency_loss) and torch.isfinite(domain_loss) and torch.isfinite(total_step_loss)):
                skipped_non_finite_loss_batches += 1
                logging.warning(
                    "Skipping graph training batch %d due to non-finite loss values (loss=%s, aux_loss=%s).",
                    step_idx,
                    str(loss.detach().item()) if torch.is_tensor(loss) else str(loss),
                    str(aux_loss.detach().item()) if torch.is_tensor(aux_loss) else str(aux_loss),
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(total_step_loss / accum_steps).backward()

            should_step = (step_idx % accum_steps == 0) or (step_idx == len(train_loader))
            if should_step:
                scaler.unscale_(optimizer)
                if _has_non_finite_gradients(model):
                    skipped_non_finite_grad_steps += 1
                    logging.warning(
                        "Skipping graph optimizer step at batch %d due to non-finite gradients.",
                        step_idx,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
                if gradient_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            model_output = model(data)
            logits, aux_loss = _unpack_graph_model_output(model_output, device)
            logits = _sanitize_logits_for_loss(logits, multiclass=multiclass)
            loss_labels = labels if multiclass else _smooth_binary_labels(labels, slide_label_smoothing)
            loss = criterion(logits, loss_labels)

            consistency_loss = torch.zeros((), device=device)
            domain_loss = torch.zeros((), device=device)
            if augment_and_compare_active and consistency_rng.random() < consistency_p:
                consistency_loss, domain_loss = _compute_graph_consistency_and_domain_losses(
                    model, data, model_output, logits, device, multiclass,
                    consistency_specs, domain_adversarial_specs, consistency_rng,
                )

            total_step_loss = (
                loss + aux_loss
                + float(consistency_weight) * consistency_loss
                + float(domain_adversarial_weight) * domain_loss
            )

            if not (torch.isfinite(loss) and torch.isfinite(aux_loss) and torch.isfinite(consistency_loss) and torch.isfinite(domain_loss) and torch.isfinite(total_step_loss)):
                skipped_non_finite_loss_batches += 1
                logging.warning(
                    "Skipping graph training batch %d due to non-finite loss values (loss=%s, aux_loss=%s).",
                    step_idx,
                    str(loss.detach().item()) if torch.is_tensor(loss) else str(loss),
                    str(aux_loss.detach().item()) if torch.is_tensor(aux_loss) else str(aux_loss),
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            (total_step_loss / accum_steps).backward()

            should_step = (step_idx % accum_steps == 0) or (step_idx == len(train_loader))
            if should_step:
                if _has_non_finite_gradients(model):
                    skipped_non_finite_grad_steps += 1
                    logging.warning(
                        "Skipping graph optimizer step at batch %d due to non-finite gradients.",
                        step_idx,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                if gradient_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            all_labels.extend(labels.cpu().numpy())
            all_raw_preds.extend(logits.detach().cpu().numpy())

        total_loss += total_step_loss.detach().item()
        total_aux_loss += aux_loss.detach().item()
        total_consistency_loss += consistency_loss.detach().item()
        total_domain_loss += domain_loss.detach().item()
        processed_batches += 1
        del data, labels, logits, loss, aux_loss, consistency_loss, domain_loss, total_step_loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    raw_pred_array = np.asarray(all_raw_preds)
    if multiclass:
        all_preds = np.argmax(raw_pred_array, axis=1).tolist()
    else:
        all_preds = (raw_pred_array > 0).astype(int).reshape(-1).tolist()

    if processed_batches > 0:
        metrics = _compute_epoch_metrics(
            all_labels,
            all_preds,
            all_raw_preds,
            multiclass=multiclass,
            ordinal_class_order=ordinal_class_order,
        )
    else:
        metrics = {
            "accuracy": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "auc": 50.0,
        }

    metrics["loss"] = total_loss / max(1, processed_batches)
    metrics["region_only_fallbacks"] = _reset_or_pop_wsigraph_fallback_count(model)
    metrics["aux_loss"] = total_aux_loss / max(1, processed_batches)
    metrics["consistency_loss"] = total_consistency_loss / max(1, processed_batches)
    metrics["domain_loss"] = total_domain_loss / max(1, processed_batches)
    metrics["processed_batches"] = int(processed_batches)
    metrics["skipped_non_finite_loss_batches"] = int(skipped_non_finite_loss_batches)
    metrics["skipped_non_finite_grad_steps"] = int(skipped_non_finite_grad_steps)

    torch.cuda.empty_cache()
    return metrics


def validate_graph(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    ordinal_class_order: Optional[List[int]] = None,
):
    """Validate the graph model on val_loader."""
    model.eval()
    total_loss = 0.0
    total_aux_loss = 0.0
    processed_batches = 0
    skipped_non_finite_batches = 0

    all_preds, all_labels, all_raw_preds = [], [], []

    if len(val_loader) == 0:
        import logging
        logging.warning("Validation dataloader is empty!")
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "auc": 50.0
        }

    multiclass = _is_multiclass_criterion(criterion)

    _reset_or_pop_wsigraph_fallback_count(model)

    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validating (Graph)", leave=False):
            data = data.to(device)


            if hasattr(data, 'y') and data.y is not None:
                labels = data.y.view(-1).to(device)
            else:
                labels = data['slide'].y.view(-1).to(device)
            if multiclass:
                labels = labels.long()
            else:
                labels = _sanitize_binary_labels(labels.float())

            model_output = model(data)
            logits, aux_loss = _unpack_graph_model_output(model_output, device)
            logits = _sanitize_logits_for_loss(logits, multiclass=multiclass)
            loss = criterion(logits, labels)
            total_step_loss = loss + aux_loss

            if not (torch.isfinite(loss) and torch.isfinite(aux_loss) and torch.isfinite(total_step_loss)):
                skipped_non_finite_batches += 1
                logging.warning(
                    "Skipping graph validation batch due to non-finite loss values (loss=%s, aux_loss=%s).",
                    str(loss.detach().item()) if torch.is_tensor(loss) else str(loss),
                    str(aux_loss.detach().item()) if torch.is_tensor(aux_loss) else str(aux_loss),
                )
                continue

            if multiclass:
                preds = torch.argmax(logits, dim=1)
            else:
                preds = (logits > 0).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_raw_preds.extend(logits.detach().cpu().numpy())

            total_loss += total_step_loss.detach().item()
            total_aux_loss += aux_loss.detach().item()
            processed_batches += 1

            del data, labels, model_output, logits, aux_loss, loss, total_step_loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    if processed_batches > 0:
        metrics = _compute_epoch_metrics(
            all_labels,
            all_preds,
            all_raw_preds,
            multiclass=multiclass,
            ordinal_class_order=ordinal_class_order,
        )
    else:
        metrics = {
            "accuracy": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "auc": 50.0,
        }

    metrics["loss"] = total_loss / max(1, processed_batches)
    metrics["region_only_fallbacks"] = _reset_or_pop_wsigraph_fallback_count(model)
    metrics["aux_loss"] = total_aux_loss / max(1, processed_batches)
    metrics["processed_batches"] = int(processed_batches)
    metrics["skipped_non_finite_loss_batches"] = int(skipped_non_finite_batches)

    torch.cuda.empty_cache()
    return metrics



def handle_multi_gpu_model(model: nn.Module) -> nn.Module:
    """Wrap model in DataParallel if multiple GPUs are available."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs")
        return nn.DataParallel(model)
    return model

class MulticlassFocalLoss(nn.Module):
    """Multiclass focal loss (Lin et al., 2017, RetinaNet). Class-weighted CrossEntropyLoss
    rescales each class's per-example contribution, but doesn't stop the sheer *volume* of easy,
    already-confident majority-class examples (e.g. CAMELYON subtype's 'negative'/'macro') from
    outweighing a handful of correctly-weighted rare/hard examples (e.g. 'itc') in total gradient
    -- observed as itc recall getting *worse*, not better, across training runs that otherwise
    improved (model becomes more confident/conservative and stops committing to the rare class).
    The (1 - p_t)^gamma term suppresses loss from examples the model already classifies
    confidently, regardless of class, keeping gradient budget on hard examples throughout
    training. `alpha` (typically the same inverse-frequency weights used for plain CE) still
    applies on top, as in the original paper.
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # nn.CrossEntropyLoss/F.nll_loss silently accept any integer target dtype (this codebase's
        # graph label tensors come through as e.g. int8/uint8 to save cache memory); Tensor.gather
        # is strict and requires int64, so cast explicitly instead of erroring on the first batch.
        targets = targets.long()
        log_probs = torch.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = target_log_probs.exp()
        focal_term = (1.0 - target_probs).clamp(min=0.0).pow(self.gamma)
        loss = -focal_term * target_log_probs
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[targets]
        return loss.mean()


def configure_training_components(model: nn.Module,
                                 learning_rate: float,
                                 weight_decay: float,
                                 pos_weight: float = 1.0,
                                 optimizer_name: str = "AdamW",
                                 momentum: float = 0.9,
                                 device: torch.device = None,
                                 num_classes: int = 1,
                                 label_mode: str = "binary",
                                 class_weights: Optional[np.ndarray] = None,
                                 loss_type: str = "ce",
                                 focal_gamma: float = 2.0,
                                 focal_use_alpha: bool = True) -> Dict[str, Any]:
    """Build the optimizer, loss criterion, and AMP scaler for training."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)
    if isinstance(weight_decay, str):
        weight_decay = float(weight_decay)
    if isinstance(momentum, str):
        momentum = float(momentum)
    if isinstance(pos_weight, str):
        pos_weight = float(pos_weight)

    trainable_params = _trainable_parameters(model)
    if not trainable_params:
        raise ValueError(
            "No trainable parameters found. Check your fine_tuning freeze settings before building the optimizer."
        )


    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        logging.info(f"Using SGD optimizer with learning_rate={learning_rate}, momentum={momentum}, weight_decay={weight_decay}")
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        logging.info(f"Using AdamW optimizer with learning_rate={learning_rate}, weight_decay={weight_decay}")

    resolved_mode = str(label_mode).strip().lower()
    if resolved_mode not in {"binary", "multiclass"}:
        raise ValueError("label_mode must be one of: binary, multiclass")

    if resolved_mode == "multiclass" or int(num_classes) > 1:
        class_weights_tensor = None
        if class_weights is not None:
            class_weights_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)

        if str(loss_type).strip().lower() == "focal":
            # focal_use_alpha=False drops the static inverse-frequency class_weights and relies
            # only on the (1-p_t)^gamma dynamic hard-example term -- stacking alpha on top of
            # gamma double/triple-counts the same rare-class correction that balanced_sampling's
            # WeightedRandomSampler already applies (same pathology as
            # avoid_double_balance_correction, just recurring with focal loss instead of CE; see
            # that config's comment for the CE-side ablation that showed compounding corrections
            # hurt rather than helped here).
            focal_alpha = class_weights_tensor if focal_use_alpha else None
            criterion = MulticlassFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            logging.info(
                "Using multiclass focal loss (gamma=%.2f, focal_use_alpha=%s) for multiclass "
                "training (num_classes=%d, alpha=%s)",
                float(focal_gamma), focal_use_alpha, num_classes,
                focal_alpha.detach().cpu().tolist() if focal_alpha is not None else None,
            )
        elif class_weights_tensor is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            logging.info(
                "Using weighted CrossEntropyLoss for multiclass training "
                f"(num_classes={num_classes}, weights={class_weights_tensor.detach().cpu().tolist()})"
            )
        else:
            criterion = nn.CrossEntropyLoss()
            logging.info(f"Using CrossEntropyLoss for multiclass training (num_classes={num_classes})")
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device)
        )
        logging.info("Using BCEWithLogitsLoss for binary training")

    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    return {
        'optimizer': optimizer,
        'criterion': criterion,
        'scaler': scaler
    }

def save_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   metrics: Dict[str, Any],
                   config: Dict[str, Any],
                   filepath: str,
                   seed: int = None,
                   deterministic: bool = False) -> None:
    """Save a model checkpoint with metrics, config, and reproducibility information."""

    required_metrics = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']
    for metric in required_metrics:
        if metric not in metrics:
            logging.warning(f"Metric '{metric}' missing from checkpoint metrics")


    reproducibility_info = {
        'random_seed': seed,
        'deterministic_mode': deterministic,
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'cudnn_deterministic': torch.backends.cudnn.deterministic if torch.cuda.is_available() else None,
        'cudnn_benchmark': torch.backends.cudnn.benchmark if torch.cuda.is_available() else None,
        'timestamp': datetime.now().isoformat()
    }

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
        'reproducibility': reproducibility_info
    }

    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved to {filepath}")


    logging.info(f"Checkpoint metrics - Acc: {metrics.get('accuracy', 0):.2f}%, " +
               f"Sens: {metrics.get('sensitivity', 0):.2f}%, " +
               f"Spec: {metrics.get('specificity', 0):.2f}%, " +
               f"F1: {metrics.get('f1', 0):.2f}%, " +
               f"AUC: {metrics.get('auc', 0):.2f}%")

def unfreeze_model_backbone(model: nn.Module, fine_tune_lr: float, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """Unfreeze the model backbone and build a new optimizer with separate backbone/head learning rates."""

    current_lr = optimizer.param_groups[0]['lr']


    for param in model.parameters():
        param.requires_grad = True


    if isinstance(model, nn.DataParallel):
        actual_model = model.module
    else:
        actual_model = model



    param_groups = []


    if hasattr(actual_model, 'backbone'):

        param_groups.append({
            'params': actual_model.backbone.parameters(),
            'lr': fine_tune_lr
        })

        classifier_params = [p for n, p in actual_model.named_parameters() if 'backbone' not in n]
        if classifier_params:
            param_groups.append({
                'params': classifier_params,
                'lr': current_lr
            })
    elif hasattr(actual_model, 'feature_extractor'):

        param_groups.append({
            'params': actual_model.feature_extractor.parameters(),
            'lr': fine_tune_lr
        })

        other_params = [p for n, p in actual_model.named_parameters() if 'feature_extractor' not in n]
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': current_lr
            })
    else:

        logging.warning("Model structure not recognized for fine-tuning, using fine_tune_lr for all parameters")
        param_groups.append({
            'params': actual_model.parameters(),
            'lr': fine_tune_lr
        })


    optimizer_class = type(optimizer)
    if optimizer_class == torch.optim.SGD:
        new_optimizer = torch.optim.SGD(param_groups, momentum=optimizer.param_groups[0].get('momentum', 0.9))
    elif optimizer_class == torch.optim.AdamW:
        new_optimizer = torch.optim.AdamW(param_groups, weight_decay=optimizer.param_groups[0].get('weight_decay', 0.0))
    else:
        new_optimizer = torch.optim.AdamW(param_groups)

    logging.info(f"Unfroze backbone with fine_tune_lr={fine_tune_lr}, head_lr={current_lr}")
    return new_optimizer


class GraphSAM:
    """Sharpness-Aware Minimization (Foret et al. 2020) wrapping an existing optimizer, opt-in
    for tissue_region_net's segmentation training loop (see _train_epoch_graph_segmentation /
    train_graph_segmentation_model in src/pipelines/train.py). Seeks flat loss-surface minima --
    documented in the domain-generalization literature (e.g. Cha et al. 2021 "SWAD") to improve
    out-of-domain generalization even from a single training domain, no domain labels needed,
    which is exactly this project's constraint (scanner1-only training, scanner2 unseen).

    Standard two-pass form per optimizer step:
      1. first_step(): compute gradient at current weights (caller's own .backward() already
         ran), ascend weights toward the local worst case (`w += rho * grad/||grad||`), snapshot
         the original weights so they can be restored.
      2. Caller recomputes loss/gradient at the PERTURBED weights (a full second forward+backward
         through the same loss function -- see _compute_segmentation_step_losses's docstring for
         why both passes must use the exact same resolved randomness).
      3. second_step(): restore the original (pre-ascent) weights, then take the real optimizer
         step using the gradient computed at the perturbed point.

    Iterates `base_optimizer.param_groups` directly (not `model.parameters()`), so it
    automatically respects whatever the existing fine_tuning/freeze config already produced --
    frozen parameters have no `.grad` and are skipped for free, no separate freeze-awareness
    needed here.

    Deliberately scoped OUT: AMP/GradScaler support (this project's segmentation training forces
    use_amp=False whenever SAM is enabled -- GradScaler's single scale/step/update cycle isn't
    designed for SAM's two backward passes per step, and the correctness-over-speed tradeoff was
    an explicit choice, not an oversight -- see train_graph_segmentation_model) and gradient
    accumulation (SAM requires grad_accum_steps=1 -- ascending mid-accumulation-window would
    mean the ascent direction reflects only a fraction of the effective batch, not the full
    windowed loss landscape SAM's rationale depends on; also enforced by the caller).
    """

    def __init__(self, base_optimizer: torch.optim.Optimizer, rho: float = 0.05, eps: float = 1e-12):
        """Wrap base_optimizer with Sharpness-Aware Minimization state."""
        self.base_optimizer = base_optimizer
        self.rho = float(rho)
        self.eps = float(eps)
        # List of (param, pre-ascent snapshot), built fresh by first_step() and consumed by
        # second_step()/restore_only(). Deliberately NOT keyed off _params()'s live grad-based
        # filter at restore time: first_step(zero_grad=True) clears every param's .grad, so a
        # LATER call to _params() would find nothing to iterate and silently restore zero
        # parameters -- caught by this class's own unit test. Storing the actual param objects
        # (not just their ids) up front sidesteps that entirely.
        self._snapshot: List[tuple] = []

    def _params(self):
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad and p.grad is not None:
                    yield p

    def grads_finite(self) -> bool:
        """Return whether all tracked parameters' gradients are finite."""
        return all(torch.isfinite(p.grad).all() for p in self._params())

    def first_step(self, zero_grad: bool = True) -> None:
        """Ascend toward the local worst-case direction and snapshot the original weights."""
        params = list(self._params())
        grad_norm = torch.norm(torch.stack([p.grad.norm(2) for p in params]), 2)
        scale = self.rho / (grad_norm + self.eps)
        self._snapshot = [(p, p.data.clone()) for p in params]
        for p, _ in self._snapshot:
            p.data.add_(p.grad * scale)
        if zero_grad:
            self.base_optimizer.zero_grad(set_to_none=True)

    def second_step(self, zero_grad: bool = True) -> None:
        """Restore pre-ascent weights, then take the real optimizer step at the perturbed gradient."""
        for p, old in self._snapshot:
            p.data.copy_(old)
        self._snapshot = []
        self.base_optimizer.step()
        if zero_grad:
            self.base_optimizer.zero_grad(set_to_none=True)

    def restore_only(self) -> None:
        """Undo an ascent step without taking the real optimizer step."""
        for p, old in self._snapshot:
            p.data.copy_(old)
        self._snapshot = []


class EarlyStopping:
    """Early stopping handler to prevent overfitting.

    Monitors validation metrics and stops training if no improvement
    is seen for a specified number of epochs.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """Initialize the early stopping handler."""
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
        self.min_validation_loss = float('inf')

        logging.info(f"Early stopping initialized: patience={patience}, min_delta={min_delta}, mode={mode}")

    def __call__(self, current_score: float) -> bool:
        """Update state with the current score and return whether training should stop."""
        if self.mode == 'min':
            score = -1.0 * current_score
        else:
            score = current_score

        if self.best_score == float('inf') or self.best_score == float('-inf'):

            self.best_score = score
            return False

        delta = score - self.best_score

        if delta > self.min_delta:

            self.best_score = score
            self.counter = 0


            if self.mode == 'min':
                self.min_validation_loss = current_score

            return False


        self.counter += 1
        if self.counter >= self.patience:
            logging.info(f"Early stopping triggered after {self.counter} epochs without improvement")
            self.early_stop = True
            return True

        return False

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.early_stop = False
        self.min_validation_loss = float('inf')