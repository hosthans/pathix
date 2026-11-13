from typing import Dict, List, Optional, Tuple
import gc
import math
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, matthews_corrcoef, brier_score_loss, roc_curve, precision_recall_curve
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as T
from pathlib import Path
import logging
from datetime import datetime
import yaml
import numpy as np
import json
import sys
from tqdm import tqdm
import pandas as pd


project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import src.utils.metrics_utils as metrics_utils
from src.utils.dataset_utils import mil_collate
from src.utils.graph_patch_utils import apply_slide_derived_negative_labels
import src.utils.training_utils as training_utils
from src.paths import paths
from src.paths.paths import add_path_args, get_project_paths
from src.utils.dataset_utils import mil_collate
import src.utils.training_utils as training_utils
from src.core.registry_utils import register_evaluation_pipeline
from src.utils.plot_style import apply_ieee_style

# Applied once at import time so every figure drawn in this module (existing
# plots and the newer plot_classification_diagnostics calls alike) shares the
# same IEEE-style rcParams (serif font, colorblind-safe palette, 300 DPI).
apply_ieee_style()


def _is_multiclass_logits(outputs: torch.Tensor) -> bool:
    return isinstance(outputs, torch.Tensor) and outputs.dim() > 1 and outputs.size(-1) > 1


def _unwrap_graph_logits(outputs):
    if isinstance(outputs, dict):
        logits = outputs.get('logits', None)
        if not torch.is_tensor(logits):
            raise ValueError("Graph model output dict must contain tensor key 'logits'.")
        return logits
    return outputs


def _to_python_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return tuple(value.detach().cpu().reshape(-1).tolist())
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.reshape(-1)[0].item()
        return tuple(value.reshape(-1).tolist())
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _to_python_scalar(value[0])
        return tuple(_to_python_scalar(v) for v in value)
    return value


def _is_positive_region_label(label_text: str) -> bool:
    text = str(label_text).strip().lower()
    return any(token in text for token in ("tumor", "metast", "macro", "micro", "itc", "positive", "status=inflamed"))


def _is_normal_region_label(label_text: str) -> bool:
    text = str(label_text).strip().lower()
    return any(token in text for token in ("normal", "negative", "benign", "healthy", "background", "non_tumor", "status=noninflamed"))


_extract_annotation_vocab_from_graph_batch = training_utils.extract_annotation_vocab_from_graph_batch


def _extract_graph_meta_entries_from_graph_batch(data) -> list[dict]:
    meta_obj = getattr(data, 'meta', None)
    if isinstance(meta_obj, dict):
        return [meta_obj]
    if isinstance(meta_obj, list):
        return [entry for entry in meta_obj if isinstance(entry, dict)]
    return []


def _extract_meta_vocab_list(data, key: str) -> list[str]:
    for meta_entry in _extract_graph_meta_entries_from_graph_batch(data):
        raw = meta_entry.get(key)
        if not isinstance(raw, list) or not raw:
            continue
        if isinstance(raw[0], list):
            return [str(v) for v in raw[0]]
        return [str(v) for v in raw]
    return []


def _resolve_slide_names(data) -> list[str]:
    slide_name_attr = getattr(data, 'slide_name', None)
    if isinstance(slide_name_attr, (list, tuple)):
        return [str(v) for v in slide_name_attr]
    if slide_name_attr is None:
        return ["graph_0"]
    return [str(slide_name_attr)]


def _batch_slide_name(batch_slide_names: list[str], i: int, global_idx: int) -> str:
    """Real slide name of graph i in the batch; falls back to the positional graph_<idx> name."""
    if len(batch_slide_names) > i:
        return batch_slide_names[i]
    return f"graph_{global_idx}"


def _compute_per_stain_accuracy(
    df: pd.DataFrame,
    slide_groups: Optional[pd.DataFrame],
    threshold: float,
) -> Dict[str, Dict]:
    """How well each stain is classified: correct/total slides per stain at the overall threshold.

    Deliberately a simple overview (no per-stain AUC/threshold sweep) -- the stain groups are small
    (e.g. 4-6 MG/PAS slides in GSDB test), so it only indicates which stains are harder, it is not a
    separate evaluation. Returns {stain: {n, correct, accuracy, fp, fn}}.
    """
    if slide_groups is None or 'stain' not in slide_groups.columns or 'slide_name' not in df.columns:
        return {}
    lookup = slide_groups[['slide_name', 'stain']].drop_duplicates('slide_name')
    joined = df.merge(lookup, on='slide_name', how='left')
    unmatched = int(joined['stain'].isna().sum())
    if unmatched:
        logging.warning(
            "Per-stain overview: %d/%d predictions could not be matched to the split CSV by "
            "slide_name -- left out of the per-stain breakdown.", unmatched, len(joined),
        )
    joined = joined.dropna(subset=['stain'])
    pred = (joined['prob'].to_numpy(dtype=float) > float(threshold)).astype(int)
    label = joined['label'].to_numpy(dtype=int)
    results: Dict[str, Dict] = {}
    for stain in sorted(joined['stain'].astype(str).unique()):
        mask = (joined['stain'].astype(str) == stain).to_numpy()
        n = int(mask.sum())
        correct = int((pred[mask] == label[mask]).sum())
        results[stain] = {
            'n': n,
            'correct': correct,
            'accuracy': correct / n if n else float('nan'),
            'fp': int(((pred[mask] == 1) & (label[mask] == 0)).sum()),
            'fn': int(((pred[mask] == 0) & (label[mask] == 1)).sum()),
        }
    return results


def _plot_per_stain_accuracy(per_stain: Dict[str, Dict], output_path: Path, title: str) -> None:
    """Horizontal bar per stain: share of correctly classified slides, labeled correct/n."""
    stains = list(per_stain)
    acc = [per_stain[s]['accuracy'] for s in stains]
    fig, ax = plt.subplots(figsize=(6, 0.6 * len(stains) + 1.2))
    y = np.arange(len(stains))
    ax.barh(y, [1.0] * len(stains), height=0.5, color='#f0efec')  # 100% track
    ax.barh(y, acc, height=0.5, color='#2a78d6')
    for i, s in enumerate(stains):
        m = per_stain[s]
        ax.text(1.02, i, f"{m['correct']}/{m['n']}  ({m['accuracy']:.0%})", va='center', fontsize=9, color='#0b0b0b')
    ax.set_yticks(y, stains)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.grid(False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel('Correctly classified slides', color='#52514e')
    ax.set_title(title, fontsize=10, color='#0b0b0b')
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False, colors='#52514e')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _annotation_enabled_per_graph_in_batch(data, num_graphs: int) -> torch.Tensor:
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

    return torch.tensor(flags, dtype=torch.bool)


def _infer_centroid_scale_to_level0(meta_entry: dict) -> float:
    if not isinstance(meta_entry, dict):
        return 1.0

    cfg = meta_entry.get('config', {})
    if not isinstance(cfg, dict):
        cfg = {}

    patch_level = cfg.get('patch_level', None)
    level_downsamples = meta_entry.get('level_downsamples', None)
    if isinstance(level_downsamples, (list, tuple)) and isinstance(patch_level, (int, np.integer)):
        idx = int(patch_level)
        if 0 <= idx < len(level_downsamples):
            try:
                scale = float(level_downsamples[idx])
                if np.isfinite(scale) and scale > 0:
                    return scale
            except Exception:
                pass

    return 1.0


def _map_region_annotation_ids_to_binary(ann_ids: torch.Tensor, vocab: list[str]) -> torch.Tensor:
    mapped = torch.full_like(ann_ids, fill_value=-1)
    if ann_ids.numel() == 0:
        return mapped

    unique_ids = sorted(set(int(v) for v in ann_ids[ann_ids >= 0].tolist()))
    if not unique_ids:
        return mapped

    if vocab:
        id_to_binary: Dict[int, int] = {}
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


def _extract_region_eval_rows(
    model_out,
    data,
    row_offset: int = 0,
    negative_label_source: str = "annotation_only",
) -> tuple[list[dict], int]:
    if not isinstance(model_out, dict):
        return [], row_offset

    region_probs = model_out.get('region_probs', None)
    if not torch.is_tensor(region_probs):
        return [], row_offset

    try:
        region_store = data['region']
    except Exception:
        return [], row_offset

    ann = getattr(region_store, 'annotation_y', None)
    if not torch.is_tensor(ann):
        return [], row_offset

    probs = region_probs.detach().view(-1).cpu().float()
    ann_ids = ann.detach().view(-1).cpu().long()
    if probs.numel() != ann_ids.numel():
        logging.warning(
            "Skipping region evaluation rows due to length mismatch: probs=%d labels=%d",
            int(probs.numel()),
            int(ann_ids.numel()),
        )
        return [], row_offset

    region_batch = getattr(region_store, 'batch', None)
    if torch.is_tensor(region_batch) and region_batch.numel() == ann_ids.numel():
        region_batch_cpu = region_batch.detach().view(-1).cpu().long()
        region_batch_arr = region_batch_cpu.numpy()
    else:
        region_batch_cpu = None
        region_batch_arr = np.zeros((ann_ids.numel(),), dtype=np.int64)

    meta_entries = _extract_graph_meta_entries_from_graph_batch(data)

    vocab = _extract_annotation_vocab_from_graph_batch(data)
    mapped = _map_region_annotation_ids_to_binary(ann_ids, vocab)

    slide_labels = getattr(data, 'y', None)
    if torch.is_tensor(slide_labels):
        slide_binary = (slide_labels.detach().view(-1).cpu().float() > 0.5).long()
        annotation_enabled = _annotation_enabled_per_graph_in_batch(data, int(slide_binary.numel()))
        if annotation_enabled.numel() == slide_binary.numel():
            mapped = apply_slide_derived_negative_labels(
                mapped=mapped,
                node_batch=region_batch_cpu,
                slide_binary=slide_binary,
                annotation_enabled=annotation_enabled,
                negative_label_source=negative_label_source,
            )

    label_ambiguous = getattr(region_store, 'label_ambiguous', None)
    if torch.is_tensor(label_ambiguous) and label_ambiguous.numel() == ann_ids.numel():
        ambiguous_mask = label_ambiguous.detach().view(-1).cpu().bool()
        mapped = mapped.clone()
        mapped[ambiguous_mask] = -1

    labeled_mask = mapped >= 0
    if not torch.any(labeled_mask):
        return [], row_offset

    centroids = getattr(region_store, 'centroid', None)
    has_centroids = torch.is_tensor(centroids) and centroids.dim() == 2 and centroids.size(0) == ann_ids.numel()

    slide_names = _resolve_slide_names(data)

    idxs = torch.where(labeled_mask)[0].cpu().numpy()
    rows: list[dict] = []
    dropped_non_finite = 0
    for local_idx in idxs.tolist():
        prob = float(probs[local_idx].item())
        if not math.isfinite(prob):
            dropped_non_finite += 1
            continue
        label = int(mapped[local_idx].item())
        graph_id = int(region_batch_arr[local_idx]) if region_batch_arr is not None else 0
        slide_name = slide_names[min(max(graph_id, 0), len(slide_names) - 1)] if slide_names else f"graph_{graph_id}"
        meta_entry = meta_entries[min(max(graph_id, 0), len(meta_entries) - 1)] if meta_entries else {}
        centroid_scale_to_level0 = _infer_centroid_scale_to_level0(meta_entry)
        centroid_x = None
        centroid_y = None
        if has_centroids:
            centroid_x = float(centroids[local_idx, 0].detach().cpu().item())
            if int(centroids.size(1)) >= 2:
                centroid_y = float(centroids[local_idx, 1].detach().cpu().item())
        rows.append(
            {
                "region_eval_idx": int(row_offset),
                "graph_id_in_batch": graph_id,
                "slide_name": slide_name,
                "region_prob": prob,
                "region_pred": int(prob >= 0.5),
                "region_label": label,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "centroid_scale_to_level0": float(centroid_scale_to_level0),
                "region_local_idx": int(local_idx),
            }
        )
        row_offset += 1

    if dropped_non_finite > 0:
        logging.warning(
            "Dropped %d region row(s) with non-finite region_prob before evaluation metrics.",
            dropped_non_finite,
        )

    return rows, row_offset


def _extract_patch_eval_rows(
    model_out,
    data,
    row_offset: int = 0,
    negative_label_source: str = "annotation_only",
) -> tuple[list[dict], int]:
    if not isinstance(model_out, dict):
        return [], row_offset

    patch_probs = model_out.get('patch_probs', None)
    if not torch.is_tensor(patch_probs):
        patch_logits = model_out.get('patch_logits', None)
        if torch.is_tensor(patch_logits):
            patch_probs = torch.sigmoid(patch_logits)
        else:
            return [], row_offset

    try:
        patch_store = data['patch']
    except Exception:
        return [], row_offset

    ann = getattr(patch_store, 'annotation_y', None)
    if not torch.is_tensor(ann):
        return [], row_offset

    probs = patch_probs.detach().view(-1).cpu().float()
    full_ann_ids = ann.detach().view(-1).cpu().long()

    patch_node_idx = model_out.get('patch_node_idx', None)
    if torch.is_tensor(patch_node_idx):
        sel_idx = patch_node_idx.detach().view(-1).cpu().long()
        if sel_idx.numel() != probs.numel():
            logging.warning(
                "Skipping patch evaluation rows due to length mismatch: probs=%d patch_node_idx=%d",
                int(probs.numel()),
                int(sel_idx.numel()),
            )
            return [], row_offset
    else:
        if probs.numel() != full_ann_ids.numel():
            logging.warning(
                "Skipping patch evaluation rows due to length mismatch: probs=%d labels=%d",
                int(probs.numel()),
                int(full_ann_ids.numel()),
            )
            return [], row_offset
        sel_idx = torch.arange(full_ann_ids.numel(), dtype=torch.long)

    ann_ids = full_ann_ids[sel_idx]

    region_local_idx_full = None
    try:
        p2r_edge_index = data['patch', 'in', 'region'].edge_index
    except Exception:
        p2r_edge_index = None
    if torch.is_tensor(p2r_edge_index) and p2r_edge_index.dim() == 2 and p2r_edge_index.size(0) == 2:
        p2r_cpu = p2r_edge_index.detach().cpu().long()
        num_patch_nodes = int(full_ann_ids.numel())
        valid_edge = (p2r_cpu[0] >= 0) & (p2r_cpu[0] < num_patch_nodes)
        region_local_idx_full = torch.full((num_patch_nodes,), -1, dtype=torch.long)
        region_local_idx_full[p2r_cpu[0][valid_edge]] = p2r_cpu[1][valid_edge]
    region_local_idx_sel = region_local_idx_full[sel_idx] if region_local_idx_full is not None else None

    patch_batch = getattr(patch_store, 'batch', None)
    if torch.is_tensor(patch_batch) and patch_batch.numel() == full_ann_ids.numel():
        patch_batch_cpu = patch_batch.detach().view(-1).cpu().long()[sel_idx]
        patch_batch_arr = patch_batch_cpu.numpy()
    else:
        patch_batch_cpu = None
        patch_batch_arr = np.zeros((ann_ids.numel(),), dtype=np.int64)

    meta_entries = _extract_graph_meta_entries_from_graph_batch(data)

    vocab = _extract_annotation_vocab_from_graph_batch(data)
    mapped = _map_region_annotation_ids_to_binary(ann_ids, vocab)

    slide_labels = getattr(data, 'y', None)
    if torch.is_tensor(slide_labels):
        slide_binary = (slide_labels.detach().view(-1).cpu().float() > 0.5).long()
        annotation_enabled = _annotation_enabled_per_graph_in_batch(data, int(slide_binary.numel()))
        if annotation_enabled.numel() == slide_binary.numel():
            mapped = apply_slide_derived_negative_labels(
                mapped=mapped,
                node_batch=patch_batch_cpu,
                slide_binary=slide_binary,
                annotation_enabled=annotation_enabled,
                negative_label_source=negative_label_source,
            )

    labeled_mask = mapped >= 0
    if not torch.any(labeled_mask):
        return [], row_offset

    centroids_full = getattr(patch_store, 'centroid', None)
    has_centroids = torch.is_tensor(centroids_full) and centroids_full.dim() == 2 and centroids_full.size(0) == full_ann_ids.numel()
    centroids = centroids_full[sel_idx] if has_centroids else None

    slide_names = _resolve_slide_names(data)

    idxs = torch.where(labeled_mask)[0].cpu().numpy()
    rows: list[dict] = []
    dropped_non_finite = 0
    for local_idx in idxs.tolist():
        prob = float(probs[local_idx].item())
        if not math.isfinite(prob):
            dropped_non_finite += 1
            continue
        label = int(mapped[local_idx].item())
        graph_id = int(patch_batch_arr[local_idx]) if patch_batch_arr is not None else 0
        slide_name = slide_names[min(max(graph_id, 0), len(slide_names) - 1)] if slide_names else f"graph_{graph_id}"
        meta_entry = meta_entries[min(max(graph_id, 0), len(meta_entries) - 1)] if meta_entries else {}
        centroid_scale_to_level0 = _infer_centroid_scale_to_level0(meta_entry)

        patch_halfsize_level0 = None
        cfg = meta_entry.get('config', {}) if isinstance(meta_entry, dict) else {}
        if isinstance(cfg, dict):
            patch_size = cfg.get('patch_size', None)
            if isinstance(patch_size, (int, float, np.integer, np.floating)) and np.isfinite(patch_size) and patch_size > 0:
                patch_halfsize_level0 = 0.5 * float(patch_size) * float(centroid_scale_to_level0)

        centroid_x = None
        centroid_y = None
        if has_centroids:
            centroid_x = float(centroids[local_idx, 0].detach().cpu().item())
            if int(centroids.size(1)) >= 2:
                centroid_y = float(centroids[local_idx, 1].detach().cpu().item())
        region_local_idx = int(region_local_idx_sel[local_idx].item()) if region_local_idx_sel is not None else -1
        rows.append(
            {
                "patch_eval_idx": int(row_offset),
                "graph_id_in_batch": graph_id,
                "slide_name": slide_name,
                "patch_prob": prob,
                "patch_pred": int(prob >= 0.5),
                "patch_label": label,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "centroid_scale_to_level0": float(centroid_scale_to_level0),
                "patch_halfsize_level0": patch_halfsize_level0,
                "region_local_idx": region_local_idx,
            }
        )
        row_offset += 1

    if dropped_non_finite > 0:
        logging.warning(
            "Dropped %d patch row(s) with non-finite patch_prob before evaluation metrics.",
            dropped_non_finite,
        )

    return rows, row_offset


def _extract_region_cls_eval_rows(
    model: nn.Module,
    model_out,
    data,
    row_offset: int = 0,
) -> tuple[list[dict], int]:
    if not isinstance(model_out, dict):
        return [], row_offset

    target_field = getattr(model, 'region_cls_target_field', None)
    class_names = getattr(model, 'region_cls_class_names', None) or []
    if not target_field or not class_names:
        return [], row_offset

    logits = model_out.get('region_cls_logits', None)
    if not torch.is_tensor(logits):
        return [], row_offset

    try:
        region_store = data['region']
    except Exception:
        return [], row_offset

    target_y = getattr(region_store, f"{target_field}_y", None)
    remap_fn = getattr(model, '_remap_cls_targets', None)
    vocab_fn = getattr(model, '_extract_meta_list', None)
    if not torch.is_tensor(target_y) or remap_fn is None or vocab_fn is None:
        return [], row_offset

    vocab = [str(v) for v in vocab_fn(data, f"{target_field}_label_vocab")]
    canonical = remap_fn(target_y, vocab, class_names, logits.device)
    if canonical is None or canonical.numel() != logits.size(0):
        return [], row_offset

    canonical_cpu = canonical.detach().view(-1).cpu().long()
    labeled_mask = canonical_cpu != -100
    if not torch.any(labeled_mask):
        return [], row_offset

    probs = torch.softmax(logits.detach().float(), dim=-1).cpu()
    preds = probs.argmax(dim=-1)

    region_batch = getattr(region_store, 'batch', None)
    if torch.is_tensor(region_batch) and region_batch.numel() == canonical_cpu.numel():
        region_batch_arr = region_batch.detach().view(-1).cpu().long().numpy()
    else:
        region_batch_arr = np.zeros((canonical_cpu.numel(),), dtype=np.int64)

    meta_entries = _extract_graph_meta_entries_from_graph_batch(data)
    centroids = getattr(region_store, 'centroid', None)
    has_centroids = torch.is_tensor(centroids) and centroids.dim() == 2 and centroids.size(0) == canonical_cpu.numel()

    slide_names = _resolve_slide_names(data)
    centroids_cpu = centroids.detach().cpu() if has_centroids else None

    idxs = torch.where(labeled_mask)[0].numpy()
    rows: list[dict] = []
    for local_idx in idxs.tolist():
        pred_class = int(preds[local_idx].item())
        true_class = int(canonical_cpu[local_idx].item())
        graph_id = int(region_batch_arr[local_idx])
        slide_name = slide_names[min(max(graph_id, 0), len(slide_names) - 1)] if slide_names else f"graph_{graph_id}"
        meta_entry = meta_entries[min(max(graph_id, 0), len(meta_entries) - 1)] if meta_entries else {}
        centroid_scale_to_level0 = _infer_centroid_scale_to_level0(meta_entry)
        centroid_x = None
        centroid_y = None
        if has_centroids:
            centroid_x = float(centroids_cpu[local_idx, 0].item())
            if int(centroids_cpu.size(1)) >= 2:
                centroid_y = float(centroids_cpu[local_idx, 1].item())
        row = {
            "region_eval_idx": int(row_offset),
            "graph_id_in_batch": graph_id,
            "slide_name": slide_name,
            "target_field": target_field,
            "true_class": true_class,
            "true_class_name": class_names[true_class],
            "pred_class": pred_class,
            "pred_class_name": class_names[pred_class],
            "correct": int(pred_class == true_class),
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "centroid_scale_to_level0": float(centroid_scale_to_level0),
            "region_local_idx": int(local_idx),
        }
        for class_idx, name in enumerate(class_names):
            row[f"prob_{name}"] = float(probs[local_idx, class_idx].item())
        rows.append(row)
        row_offset += 1

    return rows, row_offset


def _extract_patch_cls_eval_rows(
    model: nn.Module,
    model_out,
    data,
    row_offset: int = 0,
) -> tuple[list[dict], int]:
    if not isinstance(model_out, dict):
        return [], row_offset

    target_field = getattr(model, 'patch_cls_target_field', None)
    class_names = getattr(model, 'patch_cls_class_names', None) or []
    if not target_field or not class_names:
        return [], row_offset

    logits = model_out.get('patch_cls_logits', None)
    if not torch.is_tensor(logits):
        return [], row_offset

    try:
        patch_store = data['patch']
    except Exception:
        return [], row_offset

    full_target_y = getattr(patch_store, f"{target_field}_y", None)
    remap_fn = getattr(model, '_remap_cls_targets', None)
    vocab_fn = getattr(model, '_extract_meta_list', None)
    if not torch.is_tensor(full_target_y) or remap_fn is None or vocab_fn is None:
        return [], row_offset

    patch_node_idx = model_out.get('patch_node_idx', None)
    if torch.is_tensor(patch_node_idx):
        sel_idx = patch_node_idx.detach().view(-1).cpu().long()
        if sel_idx.numel() != logits.size(0):
            logging.warning(
                "Skipping patch_cls evaluation rows due to length mismatch: logits=%d patch_node_idx=%d",
                int(logits.size(0)), int(sel_idx.numel()),
            )
            return [], row_offset
    else:
        if full_target_y.numel() != logits.size(0):
            logging.warning(
                "Skipping patch_cls evaluation rows due to length mismatch: logits=%d labels=%d",
                int(logits.size(0)), int(full_target_y.numel()),
            )
            return [], row_offset
        sel_idx = torch.arange(full_target_y.numel(), dtype=torch.long)

    target_y = full_target_y[sel_idx]
    vocab = [str(v) for v in vocab_fn(data, f"{target_field}_label_vocab")]
    canonical = remap_fn(target_y, vocab, class_names, logits.device)
    if canonical is None or canonical.numel() != logits.size(0):
        return [], row_offset

    canonical_cpu = canonical.detach().view(-1).cpu().long()
    labeled_mask = canonical_cpu != -100
    if not torch.any(labeled_mask):
        return [], row_offset

    region_local_idx_full = None
    try:
        p2r_edge_index = data['patch', 'in', 'region'].edge_index
    except Exception:
        p2r_edge_index = None
    if torch.is_tensor(p2r_edge_index) and p2r_edge_index.dim() == 2 and p2r_edge_index.size(0) == 2:
        p2r_cpu = p2r_edge_index.detach().cpu().long()
        num_patch_nodes = int(full_target_y.numel())
        valid_edge = (p2r_cpu[0] >= 0) & (p2r_cpu[0] < num_patch_nodes)
        region_local_idx_full = torch.full((num_patch_nodes,), -1, dtype=torch.long)
        region_local_idx_full[p2r_cpu[0][valid_edge]] = p2r_cpu[1][valid_edge]
    region_local_idx_sel = region_local_idx_full[sel_idx] if region_local_idx_full is not None else None

    probs = torch.softmax(logits.detach().float(), dim=-1).cpu()
    preds = probs.argmax(dim=-1)

    full_patch_batch = getattr(patch_store, 'batch', None)
    if torch.is_tensor(full_patch_batch) and full_patch_batch.numel() == full_target_y.numel():
        patch_batch_arr = full_patch_batch.detach().view(-1).cpu().long()[sel_idx].numpy()
    else:
        patch_batch_arr = np.zeros((canonical_cpu.numel(),), dtype=np.int64)

    meta_entries = _extract_graph_meta_entries_from_graph_batch(data)
    centroids_full = getattr(patch_store, 'centroid', None)
    has_centroids = torch.is_tensor(centroids_full) and centroids_full.dim() == 2 and centroids_full.size(0) == full_target_y.numel()
    centroids = centroids_full[sel_idx] if has_centroids else None
    centroids_cpu = centroids.detach().cpu() if has_centroids else None

    slide_names = _resolve_slide_names(data)

    idxs = torch.where(labeled_mask)[0].numpy()
    rows: list[dict] = []
    for local_idx in idxs.tolist():
        pred_class = int(preds[local_idx].item())
        true_class = int(canonical_cpu[local_idx].item())
        graph_id = int(patch_batch_arr[local_idx])
        slide_name = slide_names[min(max(graph_id, 0), len(slide_names) - 1)] if slide_names else f"graph_{graph_id}"
        meta_entry = meta_entries[min(max(graph_id, 0), len(meta_entries) - 1)] if meta_entries else {}
        centroid_scale_to_level0 = _infer_centroid_scale_to_level0(meta_entry)

        patch_halfsize_level0 = None
        cfg = meta_entry.get('config', {}) if isinstance(meta_entry, dict) else {}
        if isinstance(cfg, dict):
            patch_size = cfg.get('patch_size', None)
            if isinstance(patch_size, (int, float, np.integer, np.floating)) and np.isfinite(patch_size) and patch_size > 0:
                patch_halfsize_level0 = 0.5 * float(patch_size) * float(centroid_scale_to_level0)

        centroid_x = None
        centroid_y = None
        if has_centroids:
            centroid_x = float(centroids_cpu[local_idx, 0].item())
            if int(centroids_cpu.size(1)) >= 2:
                centroid_y = float(centroids_cpu[local_idx, 1].item())
        region_local_idx = int(region_local_idx_sel[local_idx].item()) if region_local_idx_sel is not None else -1
        row = {
            "patch_eval_idx": int(row_offset),
            "graph_id_in_batch": graph_id,
            "slide_name": slide_name,
            "target_field": target_field,
            "true_class": true_class,
            "true_class_name": class_names[true_class],
            "pred_class": pred_class,
            "pred_class_name": class_names[pred_class],
            "correct": int(pred_class == true_class),
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "centroid_scale_to_level0": float(centroid_scale_to_level0),
            "patch_halfsize_level0": patch_halfsize_level0,
            "region_local_idx": region_local_idx,
        }
        for class_idx, name in enumerate(class_names):
            row[f"prob_{name}"] = float(probs[local_idx, class_idx].item())
        rows.append(row)
        row_offset += 1

    return rows, row_offset


def _compute_multiclass_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    class_names: list,
) -> Dict:
    from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score

    labels = list(range(len(class_names)))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = int(y_true.shape[0])

    accuracy = float((y_true == y_pred).mean()) if n else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    kappa = float(cohen_kappa_score(y_true, y_pred, labels=labels)) if n else 0.0
    kappa_quadratic = float(cohen_kappa_score(y_true, y_pred, labels=labels, weights='quadratic')) if n else 0.0
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred)) if n else 0.0
    if n and len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        try:
            mcc = float(matthews_corrcoef(y_true, y_pred))
        except Exception:
            mcc = 0.0
    else:
        mcc = 0.0

    per_class_precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    per_class: Dict[str, Dict] = {}
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else None
        recall_i = float(per_class_recall[i])
        balanced_acc_i = float((recall_i + specificity) / 2.0) if specificity is not None else None
        per_class[name] = {
            "precision": float(per_class_precision[i]),
            "recall": recall_i,
            "specificity": specificity,
            "balanced_accuracy": balanced_acc_i,
            "f1": float(per_class_f1[i]),
            "support": tp + fn,
        }

    auc_ovo_macro = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        auc_y_true, auc_y_prob = y_true, np.asarray(y_prob, dtype=float)
        finite_row_mask = np.all(np.isfinite(auc_y_prob), axis=1)
        if not np.all(finite_row_mask):
            dropped = int((~finite_row_mask).sum())
            logging.warning(
                "_compute_multiclass_classification_metrics: dropping %d non-finite probability row(s) "
                "before AUROC.", dropped,
            )
            auc_y_true = auc_y_true[finite_row_mask]
            auc_y_prob = auc_y_prob[finite_row_mask]
        if auc_y_true.size and len(np.unique(auc_y_true)) > 1:
            try:
                auc_ovo_macro = float(
                    roc_auc_score(auc_y_true, auc_y_prob, multi_class='ovo', average='macro', labels=labels)
                )
            except Exception as exc:
                logging.warning("Could not calculate multiclass AUROC: %s", exc)

    return {
        "available": True,
        "num_samples": n,
        "class_names": list(class_names),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "cohen_kappa": kappa,
        "cohen_kappa_quadratic_weighted": kappa_quadratic,
        "mcc": mcc,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)),
        "auc_ovo_macro": auc_ovo_macro,
    }


def _compute_node_multiclass_metrics(df: pd.DataFrame, class_names: list, noun: str = 'regions') -> Dict:
    if df.empty:
        return {"available": False, "reason": f"no_labeled_{noun}"}

    y_true = df['true_class'].astype(int).values
    y_pred = df['pred_class'].astype(int).values
    prob_cols = [f"prob_{name}" for name in class_names]
    y_prob = df[prob_cols].values if all(c in df.columns for c in prob_cols) else None

    metrics = _compute_multiclass_classification_metrics(y_true, y_pred, y_prob, class_names)
    metrics[f"num_labeled_{noun}"] = metrics.pop("num_samples")
    return metrics


def _drop_non_finite_agg_rows(agg_df: pd.DataFrame, prob_col: str = 'prob', name_col: str = 'slide_name') -> pd.DataFrame:
    """Exclude rows whose aggregated probability is NaN/Inf before AUC/confusion-matrix use.

    Mirrors evaluate_model_graph's fp16-autocast-overflow defense: a non-finite `prob_col` value
    used to compare False against any threshold (`NaN > threshold`), so these rows were silently
    counted as "predicted negative" regardless of their true label, and fed straight into
    roc_auc_score, which raises on NaN -- caught by the callers below and previously papered over
    with a fake AUC=0.5 fallback. Excluding them here (once, right after the per-slide/particle
    aggregation) instead of at each metric call site keeps every downstream consumer (thresholding,
    AUC, confusion matrix, the optional PR-curve plot) consistent and honest about what was
    actually evaluated.
    """
    if prob_col not in agg_df.columns:
        return agg_df
    finite_mask = np.isfinite(agg_df[prob_col].to_numpy(dtype=float))
    num_non_finite = int((~finite_mask).sum())
    if num_non_finite > 0:
        excluded_names = agg_df.loc[~finite_mask, name_col].tolist() if name_col in agg_df.columns else []
        logging.warning(
            "%d/%d aggregated '%s' rows are non-finite (NaN/Inf) -- excluding them from "
            "AUC/confusion-matrix metrics instead of silently treating them as \"predicted "
            "negative\". Affected %s: %s",
            num_non_finite, len(agg_df), prob_col, name_col, excluded_names,
        )
    return agg_df.loc[finite_mask]


_WILSON_Z_95 = 1.959963984540054  # two-sided 95% normal quantile


def _wilson_confidence_interval(successes: int, total: int, z: float = _WILSON_Z_95) -> tuple:
    if total <= 0:
        return (0.0, 0.0)
    phat = float(successes) / float(total)
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2 * total)) / denom
    margin = (z * math.sqrt(phat * (1.0 - phat) / total + (z * z) / (4 * total * total))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom <= 0.0:
        return 0.0
    return (1.0 + beta2) * precision * recall / denom


def _compute_region_binary_metrics(region_df: pd.DataFrame, label_col: str = 'region_label', prob_col: str = 'region_prob', noun: str = 'regions') -> Dict:
    if region_df.empty:
        return {"available": False, "reason": f"no_labeled_{noun}"}

    y_true = region_df[label_col].astype(int).values
    y_prob = region_df[prob_col].astype(float).values

    finite_mask = np.isfinite(y_prob)
    if not np.all(finite_mask):
        dropped = int((~finite_mask).sum())
        logging.warning("_compute_region_binary_metrics: dropping %d non-finite %s.", dropped, noun)
        y_true = y_true[finite_mask]
        y_prob = y_prob[finite_mask]
        if y_true.size == 0:
            return {"available": False, "reason": f"no_labeled_{noun}"}

    def _metrics_for_threshold(threshold: float) -> Dict:
        y_pred = (y_prob >= float(threshold)).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        dice = float(f1_score(y_true, y_pred, zero_division=0))

        specificity = None
        if (tn + fp) > 0:
            specificity = float(tn / (tn + fp))

        balanced_accuracy = None
        if specificity is not None:
            balanced_accuracy = float((recall + specificity) / 2.0)

        iou = float(tp / max(1, tp + fp + fn))
        f2 = _f_beta(precision, recall, beta=2.0)

        if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
            mcc = 0.0
        else:
            try:
                mcc = float(matthews_corrcoef(y_true, y_pred))
            except Exception:
                mcc = 0.0

        sensitivity_ci = _wilson_confidence_interval(int(tp), int(tp + fn))
        specificity_ci = _wilson_confidence_interval(int(tn), int(tn + fp))

        return {
            "threshold": float(threshold),
            "accuracy": float((tp + tn) / max(1, tp + tn + fp + fn)),
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "sensitivity": recall,
            "sensitivity_ci95": sensitivity_ci,
            "specificity": specificity,
            "specificity_ci95": specificity_ci,
            "f1": dice,
            "f2": f2,
            "dice": dice,
            "iou": iou,
            "mcc": mcc,
            "label_positive_rate": float(np.mean(y_true)),
            "pred_positive_rate": float(np.mean(y_pred)),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }

    metrics_at_0_5 = _metrics_for_threshold(0.5)

    auc = None
    ap = None
    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = None
        try:
            ap = float(average_precision_score(y_true, y_prob))
        except Exception:
            ap = None

    try:
        brier = float(brier_score_loss(y_true, y_prob))
    except Exception:
        brier = None

    num_bins = 10
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for b in range(num_bins):
        left = float(bin_edges[b])
        right = float(bin_edges[b + 1])
        if b == num_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc_bin = float(np.mean(y_true[mask]))
        ece += abs(acc_bin - conf) * (float(np.sum(mask)) / float(len(y_true)))

    best_threshold = 0.5
    best_f1 = float(metrics_at_0_5.get("f1", 0.0))
    best_balanced_acc_threshold = 0.5
    best_balanced_acc = float(metrics_at_0_5.get("balanced_accuracy") or 0.0)
    for thr in np.linspace(0.05, 0.95, 19):
        cur_metrics = _metrics_for_threshold(float(thr))
        cur_f1 = float(cur_metrics.get("f1", 0.0))
        cur_balanced_acc = float(cur_metrics.get("balanced_accuracy") or 0.0)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_threshold = float(thr)
        if cur_balanced_acc > best_balanced_acc:
            best_balanced_acc = cur_balanced_acc
            best_balanced_acc_threshold = float(thr)

    return {
        "available": True,
        f"num_labeled_{noun}": int(y_true.size),
        f"positive_{noun}": int((y_true == 1).sum()),
        f"negative_{noun}": int((y_true == 0).sum()),
        "threshold_0_5": {
            **metrics_at_0_5,
            "brier": brier,
            "ece": float(ece),
        },
        "ranking": {
            "auc": auc,
            "average_precision": ap,
        },
        "oracle_test_best_f1": {
            "threshold": float(best_threshold),
            "f1": float(best_f1),
        },
        "oracle_test_best_balanced_accuracy": {
            "threshold": float(best_balanced_acc_threshold),
            "balanced_accuracy": float(best_balanced_acc),
        },
    }


def _compute_slide_oracle_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    def _metrics_for_threshold(threshold: float) -> Dict:
        y_pred = (y_prob >= float(threshold)).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        return {
            "threshold": float(threshold),
            "accuracy": float((tp + tn) / max(1, tp + tn + fp + fn)),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": float((sensitivity + specificity) / 2.0),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }

    candidate_thresholds = sorted(set(np.linspace(0.05, 0.95, 19).tolist()) | set(np.unique(y_prob).tolist()))
    all_metrics = [_metrics_for_threshold(t) for t in candidate_thresholds]

    return {
        "oracle_test_best_f1": max(all_metrics, key=lambda m: m["f1"]),
        "oracle_test_best_balanced_accuracy": max(all_metrics, key=lambda m: m["balanced_accuracy"]),
    }


def _compute_slide_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    base_threshold: float,
    adaptive_cfg: Optional[Dict] = None,
    target_rate: Optional[float] = None,
) -> Dict:
    """Slide-level binary metrics, shared by evaluate_model_graph and evaluate_model_img_mil so a
    MIL run and a graph run on the same split are computed identically and their slide-level
    numbers are directly comparable. Previously evaluate_model_img_mil had its own separate,
    narrower inline copy of this logic (no F2, no Wilson CIs, no adaptive-threshold recalibration,
    no oracle-threshold sweep) -- this is that same logic (moved here verbatim from
    evaluate_model_graph, which now also calls this), extended so both callers get identical
    output. target_rate is the validation label-positive rate the adaptive threshold quantile-
    matches against; pass None (skips recalibration, not just "disabled") when the caller's
    validation_thresholds don't carry that value -- see evaluate_model_img_mil's call site.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    finite_mask = np.isfinite(y_prob)
    num_non_finite = int((~finite_mask).sum())
    y_true = y_true[finite_mask]
    y_prob = y_prob[finite_mask]

    threshold = float(base_threshold)
    slide_preds = (y_prob > threshold).astype(int)

    try:
        slide_auc = float(roc_auc_score(y_true, y_prob))
    except Exception as e:
        logging.warning(f"Could not calculate slide AUC: {str(e)}")
        slide_auc = float('nan')

    adaptive_cfg = adaptive_cfg or {}
    # Off unless explicitly enabled: recalibrating a threshold on the evaluated split's own score distribution
    # (and AUC) uses the test set. Strict protocol = thresholds come from the validation split only.
    adaptive_enabled = adaptive_cfg.get('enabled', False)
    adaptive_min_auc = float(adaptive_cfg.get('min_auc', 0.60))
    adaptive_collapse_min_rate = float(adaptive_cfg.get('collapse_min_rate', 0.15))
    adaptive_min_samples = int(adaptive_cfg.get('min_samples', 20))

    pred_pos_rate = float(slide_preds.mean()) if len(slide_preds) > 0 else 0.0
    is_collapsed = pred_pos_rate <= adaptive_collapse_min_rate or pred_pos_rate >= (1.0 - adaptive_collapse_min_rate)

    # Threshold-dependent metrics at the plain validation threshold, before any adaptive
    # recalibration -- the primary (unbiased) numbers; the recalibrated ones below use the test-score
    # distribution and test AUC and are only a supplementary analysis.
    validation_threshold = float(threshold)
    _vtn, _vfp, _vfn, _vtp = confusion_matrix(y_true, slide_preds, labels=[0, 1]).ravel()
    _vsens = _vtp / (_vtp + _vfn) if (_vtp + _vfn) > 0 else 0.0
    _vspec = _vtn / (_vtn + _vfp) if (_vtn + _vfp) > 0 else 0.0
    _vprec = _vtp / (_vtp + _vfp) if (_vtp + _vfp) > 0 else 0.0
    at_validation_threshold = {
        'threshold': validation_threshold,
        'tp': int(_vtp), 'tn': int(_vtn), 'fp': int(_vfp), 'fn': int(_vfn),
        'accuracy': float((_vtp + _vtn) / (_vtp + _vtn + _vfp + _vfn)) if (_vtp + _vtn + _vfp + _vfn) > 0 else 0.0,
        'sensitivity': float(_vsens),
        'specificity': float(_vspec),
        'precision': float(_vprec),
        'f1': float(2 * _vtp / (2 * _vtp + _vfp + _vfn)) if (_vtp + _vfp + _vfn) > 0 else 0.0,
        'f2': float(_f_beta(precision=_vprec, recall=_vsens, beta=2.0)),
        'balanced_acc': float((_vsens + _vspec) / 2),
        'sensitivity_ci95': _wilson_confidence_interval(int(_vtp), int(_vtp + _vfn)),
        'specificity_ci95': _wilson_confidence_interval(int(_vtn), int(_vtn + _vfp)),
    }

    if (
        adaptive_enabled and target_rate is not None and 0.0 < float(target_rate) < 1.0
        and len(y_true) >= adaptive_min_samples and is_collapsed and slide_auc >= adaptive_min_auc
    ):
        adaptive_threshold = float(np.quantile(y_prob, 1.0 - float(target_rate)))
        adaptive_threshold = float(np.clip(adaptive_threshold, 1e-4, 1.0 - 1e-4))
        if abs(adaptive_threshold - threshold) > 1e-9:
            logging.warning(
                "Applying adaptive threshold due to calibration shift: "
                "pred_pos_rate=%.3f at thr=%.3f, auc=%.3f -> new_thr=%.3f (target_rate=%.3f)",
                pred_pos_rate, threshold, slide_auc, adaptive_threshold, float(target_rate),
            )
            threshold = adaptive_threshold
            slide_preds = (y_prob > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, slide_preds, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    f2 = _f_beta(precision=(tp / (tp + fp) if (tp + fp) > 0 else 0.0), recall=sensitivity, beta=2.0)
    sensitivity_ci = _wilson_confidence_interval(int(tp), int(tp + fn))
    specificity_ci = _wilson_confidence_interval(int(tn), int(tn + fp))

    metrics = {
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'total': int(tp + tn + fp + fn),
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity), 'sensitivity_ci95': sensitivity_ci,
        'specificity': float(specificity), 'specificity_ci95': specificity_ci,
        'f1': float(f1),
        'f2': float(f2),
        'balanced_acc': float((sensitivity + specificity) / 2),
        'auc': float(slide_auc),
        'threshold': float(threshold),
        'excluded_non_finite': num_non_finite,
        # Added keys only (existing values unchanged): whether the reported threshold metrics above
        # were adaptively recalibrated, and the same metrics at the plain validation threshold.
        'adaptive_applied': bool(abs(float(threshold) - validation_threshold) > 1e-9),
        'at_validation_threshold': at_validation_threshold,
    }
    if len(y_true) > 0:
        oracle_thresholds = _compute_slide_oracle_thresholds(y_true, y_prob)
        metrics['oracle_test_best_f1'] = oracle_thresholds['oracle_test_best_f1']
        metrics['oracle_test_best_balanced_accuracy'] = oracle_thresholds['oracle_test_best_balanced_accuracy']
    return metrics


def _region_metrics_for_existing_threshold(region_df: pd.DataFrame, threshold: float, label_col: str = 'region_label', prob_col: str = 'region_prob') -> Dict:
    if region_df.empty:
        return {}

    y_true = region_df[label_col].astype(int).values
    y_prob = region_df[prob_col].astype(float).values
    y_pred = (y_prob >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    dice = float(f1_score(y_true, y_pred, zero_division=0))
    iou = float(tp / max(1, tp + fp + fn))

    specificity = None
    if (tn + fp) > 0:
        specificity = float(tn / (tn + fp))
    balanced_accuracy = None
    if specificity is not None:
        balanced_accuracy = float((recall + specificity) / 2.0)

    return {
        "threshold": float(threshold),
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def _plot_test_region_predictions(region_df: pd.DataFrame, threshold: float, output_dir: Path, model_name: str, split: str) -> None:
    if region_df.empty:
        return

    y_true = region_df['region_label'].astype(int).values
    y_prob = region_df['region_prob'].astype(float).values
    y_pred = (y_prob >= float(threshold)).astype(int)

    plt.figure(figsize=(10, 6))
    plt.hist(y_prob[y_true == 1], bins=40, alpha=0.6, label='GT tumor', color='tab:red')
    plt.hist(y_prob[y_true == 0], bins=40, alpha=0.6, label='GT normal', color='tab:blue')
    plt.axvline(float(threshold), color='black', linestyle='--', linewidth=2, label=f'Threshold={float(threshold):.2f}')
    plt.xlabel('Region probability')
    plt.ylabel('Count')
    plt.title(f'Region probabilities vs GT ({split})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_{split}_region_prob_vs_gt.png", dpi=220)
    plt.close()

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    plt.figure(figsize=(8, 5))
    labels = ['TP', 'FN', 'FP', 'TN']
    values = [tp, fn, fp, tn]
    colors = ['tab:green', 'tab:orange', 'tab:red', 'tab:blue']
    plt.bar(labels, values, color=colors)
    plt.title(f'Region confusion counts ({split}, thr={float(threshold):.2f})')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_{split}_region_confusion_counts.png", dpi=220)
    plt.close()

    if {'centroid_x', 'centroid_y'}.issubset(region_df.columns):
        coord_df = region_df.dropna(subset=['centroid_x', 'centroid_y'])
        if not coord_df.empty:
            max_points = 12000
            if len(coord_df) > max_points:
                coord_df = coord_df.sample(n=max_points, random_state=42)

            plt.figure(figsize=(9, 7))
            norm_mask = coord_df['region_label'].astype(int).values == 0
            tum_mask = ~norm_mask
            plt.scatter(
                coord_df.loc[norm_mask, 'centroid_x'].values,
                coord_df.loc[norm_mask, 'centroid_y'].values,
                c=coord_df.loc[norm_mask, 'region_prob'].values,
                cmap='Blues',
                s=6,
                alpha=0.55,
                vmin=0.0,
                vmax=1.0,
                label='GT normal',
            )
            plt.scatter(
                coord_df.loc[tum_mask, 'centroid_x'].values,
                coord_df.loc[tum_mask, 'centroid_y'].values,
                c=coord_df.loc[tum_mask, 'region_prob'].values,
                cmap='Reds',
                s=8,
                alpha=0.7,
                vmin=0.0,
                vmax=1.0,
                label='GT tumor',
            )
            plt.gca().invert_yaxis()
            plt.xlabel('Centroid X')
            plt.ylabel('Centroid Y')
            plt.title(f'Region map: predicted score vs GT ({split})')
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(output_dir / f"{model_name}_{split}_region_map_pred_vs_gt.png", dpi=220)
            plt.close()


def _plot_region_diagnostics(region_df: pd.DataFrame, threshold: float, output_dir: Path, model_name: str, split: str) -> None:
    if region_df.empty:
        return

    y_true = region_df['region_label'].astype(int).values
    y_prob = region_df['region_prob'].astype(float).values
    y_pred = (y_prob >= float(threshold)).astype(int)

    has_both_classes = len(np.unique(y_true)) > 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    if has_both_classes:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = float(roc_auc_score(y_true, y_prob))
        axes[0].plot(fpr, tpr, color='tab:blue', linewidth=2, label=f'AUC={auc_val:.3f}')
        axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
        axes[0].set_title('Region ROC')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].legend(loc='lower right')
        axes[0].grid(alpha=0.25)
    else:
        axes[0].text(0.5, 0.5, 'ROC not available\n(single class in labels)', ha='center', va='center')
        axes[0].set_axis_off()

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    ap_val = float(average_precision_score(y_true, y_prob)) if has_both_classes else float(np.mean(y_true))
    baseline = float(np.mean(y_true))
    axes[1].plot(recall_curve, precision_curve, color='tab:red', linewidth=2, label=f'AP={ap_val:.3f}')
    axes[1].axhline(baseline, linestyle='--', color='gray', linewidth=1, label=f'Baseline={baseline:.3f}')
    axes[1].set_title('Region Precision-Recall')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend(loc='lower left')
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_{split}_region_roc_pr_curves.png")
    plt.close(fig)

    thresholds = np.linspace(0.05, 0.95, 19)
    f1_vals = []
    precision_vals = []
    recall_vals = []
    specificity_vals = []
    balacc_vals = []
    for thr in thresholds:
        cur_pred = (y_prob >= float(thr)).astype(int)
        cur_tn, cur_fp, cur_fn, cur_tp = confusion_matrix(y_true, cur_pred, labels=[0, 1]).ravel()
        cur_precision = float(precision_score(y_true, cur_pred, zero_division=0))
        cur_recall = float(cur_tp / max(1, cur_tp + cur_fn))
        cur_specificity = float(cur_tn / max(1, cur_tn + cur_fp))
        cur_f1 = float(f1_score(y_true, cur_pred, zero_division=0))
        cur_balacc = float((cur_recall + cur_specificity) / 2.0)
        f1_vals.append(cur_f1)
        precision_vals.append(cur_precision)
        recall_vals.append(cur_recall)
        specificity_vals.append(cur_specificity)
        balacc_vals.append(cur_balacc)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    ax.plot(thresholds, f1_vals, label='F1', linewidth=2)
    ax.plot(thresholds, balacc_vals, label='Balanced Accuracy', linewidth=2)
    ax.plot(thresholds, precision_vals, label='Precision', alpha=0.8)
    ax.plot(thresholds, recall_vals, label='Recall', alpha=0.8)
    ax.plot(thresholds, specificity_vals, label='Specificity', alpha=0.8)
    ax.axvline(float(threshold), linestyle='--', color='black', linewidth=1.5, label=f'Used thr={float(threshold):.2f}')
    ax.set_title('Region Threshold Sweep')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Metric value')
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc='best', ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_{split}_region_threshold_sweep.png")
    plt.close(fig)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=180)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred normal', 'Pred tumor'])
    ax.set_yticklabels(['GT normal', 'GT tumor'])
    ax.set_title(f'Region Confusion Matrix (thr={float(threshold):.2f})')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha='center', va='center', color='black', fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_{split}_region_confusion_matrix.png")
    plt.close(fig)

    bin_edges = np.linspace(0.0, 1.0, 11)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    confs = []
    accs = []
    counts = []
    for b in range(10):
        left = float(bin_edges[b])
        right = float(bin_edges[b + 1])
        if b == 9:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        counts.append(int(np.sum(mask)))
        if np.any(mask):
            confs.append(float(np.mean(y_prob[mask])))
            accs.append(float(np.mean(y_true[mask])))
        else:
            confs.append(np.nan)
            accs.append(np.nan)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=180)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    valid = np.isfinite(np.asarray(confs)) & np.isfinite(np.asarray(accs))
    if np.any(valid):
        ax.plot(np.asarray(confs)[valid], np.asarray(accs)[valid], marker='o', linewidth=2, color='tab:purple')
        for i, c in enumerate(np.asarray(bin_centers)[valid]):
            cnt = int(np.asarray(counts)[valid][i])
            if cnt > 0:
                ax.text(float(np.asarray(confs)[valid][i]), float(np.asarray(accs)[valid][i]), str(cnt), fontsize=8)
    ax.set_title('Region Reliability Diagram')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed tumor frequency')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_{split}_region_calibration.png")
    plt.close(fig)

    if 'slide_name' in region_df.columns:
        slide_stats = region_df.groupby('slide_name').agg(
            gt_tumor_fraction=('region_label', 'mean'),
            pred_tumor_fraction=('region_prob', lambda s: float(np.mean(np.asarray(s) >= float(threshold)))),
            pred_mean_prob=('region_prob', 'mean'),
            n_regions=('region_label', 'count'),
        ).reset_index()
        if not slide_stats.empty:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
            size = np.clip(np.sqrt(slide_stats['n_regions'].to_numpy(dtype=np.float32)) * 4.0, 20.0, 180.0)
            sc = ax.scatter(
                slide_stats['gt_tumor_fraction'].to_numpy(dtype=np.float32),
                slide_stats['pred_tumor_fraction'].to_numpy(dtype=np.float32),
                c=slide_stats['pred_mean_prob'].to_numpy(dtype=np.float32),
                s=size,
                cmap='viridis',
                alpha=0.85,
                edgecolors='black',
                linewidths=0.3,
            )
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
            ax.set_title('Slide Tumor Burden Agreement')
            ax.set_xlabel('GT tumor region fraction per slide')
            ax.set_ylabel('Predicted tumor region fraction per slide')
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.25)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Mean region probability')
            fig.tight_layout()
            fig.savefig(output_dir / f"{model_name}_{split}_region_slide_burden_agreement.png")
            plt.close(fig)


def _load_split_slide_file_lookup(split: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    splits_root = project_root / "data" / "splits"
    if not splits_root.exists():
        return mapping

    for csv_path in sorted(splits_root.glob(f"**/{str(split)}.csv")):
        try:
            split_df = pd.read_csv(csv_path)
        except Exception:
            continue

        if 'slide_name' not in split_df.columns:
            continue

        source_col = None
        for candidate in ('slide_file', 'wsi_path', 'slide_path', 'image_path'):
            if candidate in split_df.columns:
                source_col = candidate
                break

        if source_col is None:
            continue

        for _, row in split_df[['slide_name', source_col]].dropna().iterrows():
            slide_name = str(row['slide_name']).strip()
            slide_ref = str(row[source_col]).strip()
            if slide_name and slide_ref:
                mapping[slide_name] = slide_ref

    return mapping


def _resolve_wsi_path_for_slide(
    slide_name: str,
    split_lookup: Dict[str, str],
    filename_index: Dict[str, Path],
) -> Optional[Path]:
    raw_root = project_root / "data" / "raw"
    candidate_paths: list[Path] = []

    ref = split_lookup.get(str(slide_name))
    if ref:
        ref_path = Path(ref)
        if ref_path.is_absolute():
            candidate_paths.append(ref_path)
        else:
            candidate_paths.extend([
                project_root / ref_path,
                raw_root / ref_path,
                raw_root / "camelyon" / ref_path,
                raw_root / "camelyon" / "images" / ref_path,
            ])

    for ext in ('.tif', '.tiff', '.svs', '.ndpi', '.mrxs'):
        candidate_paths.append(raw_root / "camelyon" / "images" / f"{slide_name}{ext}")
        candidate_paths.append(raw_root / "images" / f"{slide_name}{ext}")
        candidate_paths.append(raw_root / "slides" / f"{slide_name}{ext}")

    for path in candidate_paths:
        if path.is_file():
            return path

    for ext in ('.tif', '.tiff', '.svs', '.ndpi', '.mrxs'):
        key = f"{slide_name}{ext}".lower()
        if key in filename_index:
            return filename_index[key]

    return None


def _build_wsi_filename_index() -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    roots = [
        project_root / "data" / "raw" / "camelyon" / "images",
        project_root / "data" / "raw" / "images",
        project_root / "data" / "raw" / "slides",
    ]

    allowed_ext = {'.tif', '.tiff', '.svs', '.ndpi', '.mrxs'}
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_file() and child.suffix.lower() in allowed_ext:
                index[child.name.lower()] = child
    return index


def _resolve_annotation_path_for_slide(slide_name: str) -> Optional[Path]:
    raw_root = project_root / "data" / "raw"
    candidates = [
        raw_root / "annotations" / f"{slide_name}_annotations.json",
        raw_root / "annotations" / f"{slide_name}.xml",
        raw_root / "annotations_small" / f"{slide_name}_annotations.json",
        raw_root / "other_annotations" / f"{slide_name}_annotations.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_slide_positive_annotation_polygons_level0(slide_name: str) -> List[np.ndarray]:
    path = _resolve_annotation_path_for_slide(str(slide_name))
    if path is None:
        return []
    try:
        from src.graph_creation_new.helper.annotations import (
            load_asap_annotation,
            load_geojson_annotations,
            annotation_label,
            is_positive_label_text,
        )
        annotations = load_asap_annotation(path) if path.suffix.lower() == '.xml' else load_geojson_annotations(path)
    except Exception as exc:
        logging.warning("Failed to load annotations for slide '%s' from %s: %s", slide_name, path, exc)
        return []

    polygons: List[np.ndarray] = []
    for annotation in annotations:
        label_text = annotation_label(annotation) or ""
        if not is_positive_label_text(label_text):
            continue
        coordinates = annotation.get("coordinates") or []
        if len(coordinates) >= 3:
            polygons.append(np.asarray(coordinates, dtype=np.float64))
    return polygons


def _current_process_rss_mb() -> Optional[float]:
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
    except Exception:
        return None
    return None


def _rasterize_squares_mask(
    cx: np.ndarray, cy: np.ndarray, halfsize: np.ndarray, thumb_h: int, thumb_w: int,
) -> np.ndarray:
    mask = np.zeros((thumb_h, thumb_w), dtype=np.uint8)
    if cx.size == 0:
        return mask

    half_i = np.maximum(1.0, np.round(halfsize))
    x0 = np.clip(np.round(cx - half_i), 0, thumb_w).astype(np.int64)
    x1 = np.clip(np.round(cx + half_i) + 1, 0, thumb_w).astype(np.int64)
    y0 = np.clip(np.round(cy - half_i), 0, thumb_h).astype(np.int64)
    y1 = np.clip(np.round(cy + half_i) + 1, 0, thumb_h).astype(np.int64)
    valid = (x1 > x0) & (y1 > y0)
    if not np.any(valid):
        return mask
    x0, x1, y0, y1 = x0[valid], x1[valid], y0[valid], y1[valid]

    diff = np.zeros((thumb_h + 1, thumb_w + 1), dtype=np.int32)
    np.add.at(diff, (y0, x0), 1)
    np.add.at(diff, (y0, x1), -1)
    np.add.at(diff, (y1, x0), -1)
    np.add.at(diff, (y1, x1), 1)
    stamped = np.cumsum(np.cumsum(diff, axis=0), axis=1)[:thumb_h, :thumb_w]
    return (stamped > 0).astype(np.uint8)


def _mask_to_closed_contours(
    mask: np.ndarray, close_kernel_px: int = 1, max_contours: int = 4000,
) -> List[Tuple[np.ndarray, bool]]:
    import cv2

    if close_kernel_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_px * 2 + 1, close_kernel_px * 2 + 1))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]

    entries = [
        (i, contour, int(h[3]) != -1, int(h[3]))
        for i, (contour, h) in enumerate(zip(contours, hierarchy))
        if len(contour) >= 3
    ]
    outer_indices = {i for i, _, is_hole, _ in entries if not is_hole}
    if len(outer_indices) > max_contours:
        areas = {i: cv2.contourArea(c) for i, c, is_hole, _ in entries if not is_hole}
        keep_outer = set(sorted(areas, key=areas.get, reverse=True)[:max_contours])
        logging.warning(
            "Predicted positive mask decomposed into %d separate areas; drawing only the %d "
            "largest (skipping %d small/fragmented ones) to bound rendering cost/memory.",
            len(outer_indices), len(keep_outer), len(outer_indices) - len(keep_outer),
        )
        outer_indices = keep_outer

    out: List[Tuple[np.ndarray, bool]] = []
    for i, contour, is_hole, parent in entries:
        if (parent not in outer_indices) if is_hole else (i not in outer_indices):
            continue
        simplified = cv2.approxPolyDP(contour, epsilon=1.5, closed=True)
        if len(simplified) < 3:
            simplified = contour
        out.append((simplified.reshape(-1, 2).astype(np.float64), is_hole))
    return out


def _draw_closed_polyline(ax, pts_xy: np.ndarray, **kwargs) -> None:
    closed = np.vstack([pts_xy, pts_xy[:1]])
    ax.plot(closed[:, 0], closed[:, 1], **kwargs)


def _expand_region_df_to_patch_footprints(region_df: pd.DataFrame, patch_df: pd.DataFrame) -> pd.DataFrame:
    required_region_cols = {'slide_name', 'region_local_idx', 'region_prob', 'region_label'}
    required_patch_cols = {'slide_name', 'region_local_idx', 'centroid_x', 'centroid_y', 'patch_halfsize_level0', 'centroid_scale_to_level0'}
    if not required_region_cols.issubset(region_df.columns) or not required_patch_cols.issubset(patch_df.columns):
        return pd.DataFrame()

    patch_side = patch_df.loc[patch_df['region_local_idx'] >= 0, list(required_patch_cols)]
    region_side = region_df.loc[region_df['region_local_idx'] >= 0, list(required_region_cols)]
    if patch_side.empty or region_side.empty:
        return pd.DataFrame()

    return patch_side.merge(region_side, on=['slide_name', 'region_local_idx'], how='inner')


def _expand_region_cls_df_to_patch_footprints(region_cls_df: pd.DataFrame, patch_cls_df: pd.DataFrame) -> pd.DataFrame:
    required_region_cols = {'slide_name', 'region_local_idx', 'pred_class_name', 'true_class_name'}
    required_patch_cols = {'slide_name', 'region_local_idx', 'centroid_x', 'centroid_y', 'patch_halfsize_level0', 'centroid_scale_to_level0'}
    if not required_region_cols.issubset(region_cls_df.columns) or not required_patch_cols.issubset(patch_cls_df.columns):
        return pd.DataFrame()

    patch_side = patch_cls_df.loc[patch_cls_df['region_local_idx'] >= 0, list(required_patch_cols)]
    region_side = region_cls_df.loc[region_cls_df['region_local_idx'] >= 0, list(required_region_cols)]
    if patch_side.empty or region_side.empty:
        return pd.DataFrame()

    return patch_side.merge(region_side, on=['slide_name', 'region_local_idx'], how='inner')


def _save_slide_tumor_overlays(
    df: pd.DataFrame,
    threshold: float,
    output_dir: Path,
    model_name: str,
    split: str,
    prob_col: str,
    dir_name: str,
    title_noun: str,
    halfsize_level0_col: str = 'patch_halfsize_level0',
    thumbnail_max_size: int = 2000,
    class_name: str = 'positive',
) -> None:
    if df.empty:
        return

    required_cols = {'slide_name', prob_col, 'centroid_x', 'centroid_y', halfsize_level0_col}
    missing = required_cols - set(df.columns)
    if missing:
        logging.info("Skipping %s overlays: required columns missing (%s)", title_noun, sorted(missing))
        return

    coord_df = df.dropna(subset=['centroid_x', 'centroid_y'])
    if coord_df.empty:
        logging.info("Skipping %s overlays: no centroid coordinates available.", title_noun)
        return

    try:
        import openslide
    except Exception:
        logging.warning("OpenSlide not available. Skipping slide %s overlays.", class_name)
        return

    split_lookup = _load_split_slide_file_lookup(split=str(split))
    filename_index = _build_wsi_filename_index()

    overlays_dir = output_dir / f"{model_name}_{split}_{dir_name}"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    n_slides_seen = 0
    n_total_slides = coord_df['slide_name'].nunique()
    for slide_name, slide_df in coord_df.groupby('slide_name'):
        n_slides_seen += 1
        try:
            rendered += _render_one_slide_tumor_overlay(
                slide_name=slide_name,
                slide_df=slide_df,
                threshold=threshold,
                prob_col=prob_col,
                halfsize_level0_col=halfsize_level0_col,
                title_noun=title_noun,
                class_name=class_name,
                thumbnail_max_size=thumbnail_max_size,
                overlays_dir=overlays_dir,
                split_lookup=split_lookup,
                filename_index=filename_index,
            )
        except Exception as exc:
            logging.warning("Failed to render %s %s overlay for slide '%s': %s", class_name, title_noun, str(slide_name), str(exc))
        finally:
            if n_slides_seen % 20 == 0:
                gc.collect()
                rss_mb = _current_process_rss_mb()
                if rss_mb is not None:
                    logging.info("[%s %s overlays] slide %d/%d, RSS=%.0f MB", class_name, title_noun, n_slides_seen, n_total_slides, rss_mb)

    logging.info("Saved %s %s overlays to %s (rendered=%d)", class_name, title_noun, overlays_dir, int(rendered))


def _open_slide_and_project_centroids(
    slide_name,
    slide_df: pd.DataFrame,
    halfsize_level0_col: str,
    thumbnail_max_size: int,
    split_lookup: Dict[str, str],
    filename_index: Dict[str, Path],
):
    import openslide

    wsi_path = _resolve_wsi_path_for_slide(
        slide_name=str(slide_name),
        split_lookup=split_lookup,
        filename_index=filename_index,
    )
    if wsi_path is None:
        logging.warning("Could not resolve WSI file for slide '%s'. Skipping overlay.", str(slide_name))
        return None

    try:
        slide = openslide.OpenSlide(str(wsi_path))
    except Exception as exc:
        logging.warning("Failed to open WSI '%s': %s", str(wsi_path), str(exc))
        return None

    try:
        width_0, height_0 = slide.dimensions
        max_dim = float(max(1, max(int(width_0), int(height_0))))
        thumb_scale = min(1.0, max(1e-6, float(thumbnail_max_size) / max_dim))
        thumb_w = max(64, int(round(float(width_0) * thumb_scale)))
        thumb_h = max(64, int(round(float(height_0) * thumb_scale)))
        thumb = np.asarray(slide.get_thumbnail((thumb_w, thumb_h)).convert('RGB'))
        thumb_h, thumb_w = int(thumb.shape[0]), int(thumb.shape[1])
    except Exception as exc:
        logging.warning("Failed to build thumbnail for '%s': %s", str(wsi_path), str(exc))
        return None
    finally:
        slide.close()

    x = slide_df['centroid_x'].to_numpy(dtype=np.float64)
    y = slide_df['centroid_y'].to_numpy(dtype=np.float64)
    halfsize = slide_df[halfsize_level0_col].to_numpy(dtype=np.float64)
    if x.size == 0:
        return None

    scales = slide_df['centroid_scale_to_level0'].to_numpy(dtype=np.float64) if 'centroid_scale_to_level0' in slide_df.columns else np.ones_like(x)
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
    x_l0 = x * scales
    y_l0 = y * scales
    plausible_scaled = bool(np.isfinite(x_l0).all() and np.median(x_l0) <= width_0 * 1.2 and np.median(y_l0) <= height_0 * 1.2)
    plausible_raw = bool(np.median(x) <= width_0 * 1.2 and np.median(y) <= height_0 * 1.2)
    if plausible_scaled or not plausible_raw:
        x_use, y_use = x_l0, y_l0
    else:
        x_use, y_use = x, y

    valid_half = np.isfinite(halfsize) & (halfsize > 0)
    if not np.any(valid_half):
        if x_use.size >= 2:
            # Nearest-neighbor distance per point, computed via a KD-tree rather than the full
            # dense pairwise distance matrix: for a slide with n patches (n can run into the tens
            # of thousands for the full patch-level overlay pass, vs. only region-member patches
            # for the region-level pass) the dense dx/dy/dist matrices are O(n^2) memory -- three
            # n x n float64 arrays -- which is what OOM-killed this process during the patch-level
            # overlay stage. cKDTree gives the exact same nearest-other-point distances in O(n log
            # n) time and O(n) memory.
            from scipy.spatial import cKDTree
            points = np.column_stack([x_use, y_use])
            tree = cKDTree(points)
            dist, _ = tree.query(points, k=2)
            nn = dist[:, 1]
            nn = nn[np.isfinite(nn) & (nn > 0)]
            inferred_halfsize = float(np.median(nn) / 2.0) if nn.size > 0 else float(max(width_0, height_0)) / 200.0
        else:
            inferred_halfsize = float(max(width_0, height_0)) / 200.0
        halfsize = np.full_like(x_use, inferred_halfsize)
    elif not np.all(valid_half):
        halfsize = np.where(valid_half, halfsize, float(np.median(halfsize[valid_half])))

    px = x_use * thumb_scale
    py = y_use * thumb_scale
    half_px = np.clip(halfsize * thumb_scale, 1.0, float(max(thumb_w, thumb_h)))

    return thumb, thumb_h, thumb_w, px, py, half_px, thumb_scale


def _render_one_slide_tumor_overlay(
    slide_name,
    slide_df: pd.DataFrame,
    threshold: float,
    prob_col: str,
    halfsize_level0_col: str,
    title_noun: str,
    class_name: str,
    thumbnail_max_size: int,
    overlays_dir: Path,
    split_lookup: Dict[str, str],
    filename_index: Dict[str, Path],
) -> int:
    projected = _open_slide_and_project_centroids(
        slide_name=slide_name,
        slide_df=slide_df,
        halfsize_level0_col=halfsize_level0_col,
        thumbnail_max_size=thumbnail_max_size,
        split_lookup=split_lookup,
        filename_index=filename_index,
    )
    if projected is None:
        return 0
    thumb, thumb_h, thumb_w, px, py, half_px, thumb_scale = projected

    probs = slide_df[prob_col].to_numpy(dtype=np.float64)
    preds = probs >= float(threshold)

    pred_mask = (
        _rasterize_squares_mask(px[preds], py[preds], half_px[preds], thumb_h, thumb_w)
        if np.any(preds) else np.zeros((thumb_h, thumb_w), dtype=np.uint8)
    )

    gt_polygons_level0 = _load_slide_positive_annotation_polygons_level0(str(slide_name))

    fig_w = max(7.0, float(thumb_w) / 220.0)
    fig_h = max(7.0, float(thumb_h) / 220.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    try:
        ax.imshow(thumb)

        drew_gt = False
        for polygon_level0 in gt_polygons_level0:
            polygon_thumb = polygon_level0 * thumb_scale
            _draw_closed_polyline(ax, polygon_thumb, color='#ef4444', linewidth=2.2, alpha=0.95, zorder=3)
            drew_gt = True

        n_pred_regions = 0
        if np.any(pred_mask > 0):
            for pts_xy, is_hole in _mask_to_closed_contours(pred_mask):
                if not is_hole:
                    n_pred_regions += 1
                _draw_closed_polyline(
                    ax, pts_xy,
                    color='#facc15' if is_hole else '#22c55e',
                    linewidth=1.6 if is_hole else 2.6,
                    linestyle='--' if is_hole else '-',
                    alpha=0.95,
                    zorder=4,
                )

        legend_handles = []
        if drew_gt:
            legend_handles.append(plt.Line2D([0], [0], color='#ef4444', lw=2.2, label=f'Ground-truth {class_name} annotation'))
        if n_pred_regions > 0:
            legend_handles.append(plt.Line2D([0], [0], color='#22c55e', lw=2.6, label=f'Predicted {class_name} {title_noun}'))
        if legend_handles:
            ax.legend(handles=legend_handles, loc='lower right', fontsize=8, framealpha=0.85)

        ax.set_title(
            f"{slide_name} | predicted {class_name} {title_noun}: {n_pred_regions} contiguous area(s) "
            f"({int(np.sum(preds))}/{int(len(preds))} {title_noun} >= thr) | thr={float(threshold):.2f}",
            fontsize=9,
        )
        ax.set_xlim(0, thumb_w)
        ax.set_ylim(thumb_h, 0)
        ax.axis('off')
        fig.tight_layout(pad=0.2)

        safe_name = str(slide_name).replace('/', '_').replace('\\', '_').replace(' ', '_')
        out_path = overlays_dir / f"{safe_name}.jpg"
        fig.savefig(out_path, dpi=180, format='jpg', pil_kwargs={"quality": 90, "optimize": True})
    finally:
        plt.close(fig)

    return 1


# Okabe-Ito colorblind-safe categorical palette (same family used by src/graph_creation_new/
# visualization.py's _region_color_map for build-time debug plots -- kept as its own copy here
# rather than importing that build-time module, consistent with how this file already mirrors
# rather than cross-imports small helpers between build/train/eval stages). Tiled for >8 classes.
_MULTICLASS_PALETTE = [
    "#e69f00", "#56b4e9", "#009e73", "#f0e442",
    "#0072b2", "#d55e00", "#cc79a7", "#999999",
]


def _class_color_map(class_names: List[str]) -> Dict[str, str]:
    return {name: _MULTICLASS_PALETTE[i % len(_MULTICLASS_PALETTE)] for i, name in enumerate(class_names)}


def _save_slide_multiclass_overlays(
    df: pd.DataFrame,
    class_names: List[str],
    output_dir: Path,
    model_name: str,
    split: str,
    dir_name: str,
    title_noun: str,
    target_field: str,
    halfsize_level0_col: str = 'patch_halfsize_level0',
    thumbnail_max_size: int = 2000,
) -> None:
    if df.empty or not class_names:
        return

    required_cols = {'slide_name', 'pred_class_name', 'true_class_name', 'centroid_x', 'centroid_y', halfsize_level0_col}
    missing = required_cols - set(df.columns)
    if missing:
        logging.info("Skipping %s %s overlays: required columns missing (%s)", target_field, title_noun, sorted(missing))
        return

    coord_df = df.dropna(subset=['centroid_x', 'centroid_y'])
    if coord_df.empty:
        logging.info("Skipping %s %s overlays: no centroid coordinates available.", target_field, title_noun)
        return

    try:
        import openslide
    except Exception:
        logging.warning("OpenSlide not available. Skipping %s %s overlays.", target_field, title_noun)
        return

    split_lookup = _load_split_slide_file_lookup(split=str(split))
    filename_index = _build_wsi_filename_index()
    overlays_dir = output_dir / f"{model_name}_{split}_{dir_name}"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    color_map = _class_color_map(class_names)

    rendered = 0
    n_slides_seen = 0
    n_total_slides = coord_df['slide_name'].nunique()
    for slide_name, slide_df in coord_df.groupby('slide_name'):
        n_slides_seen += 1
        try:
            rendered += _render_one_slide_multiclass_overlay(
                slide_name=slide_name,
                slide_df=slide_df,
                class_names=class_names,
                color_map=color_map,
                halfsize_level0_col=halfsize_level0_col,
                title_noun=title_noun,
                target_field=target_field,
                thumbnail_max_size=thumbnail_max_size,
                overlays_dir=overlays_dir,
                split_lookup=split_lookup,
                filename_index=filename_index,
            )
        except Exception as exc:
            logging.warning("Failed to render %s %s overlay for slide '%s': %s", target_field, title_noun, str(slide_name), str(exc))
        finally:
            if n_slides_seen % 20 == 0:
                gc.collect()
                rss_mb = _current_process_rss_mb()
                if rss_mb is not None:
                    logging.info("[%s %s overlays] slide %d/%d, RSS=%.0f MB", target_field, title_noun, n_slides_seen, n_total_slides, rss_mb)

    logging.info("Saved %s %s overlays to %s (rendered=%d)", target_field, title_noun, overlays_dir, int(rendered))


def _render_one_slide_multiclass_overlay(
    slide_name,
    slide_df: pd.DataFrame,
    class_names: List[str],
    color_map: Dict[str, str],
    halfsize_level0_col: str,
    title_noun: str,
    target_field: str,
    thumbnail_max_size: int,
    overlays_dir: Path,
    split_lookup: Dict[str, str],
    filename_index: Dict[str, Path],
) -> int:
    projected = _open_slide_and_project_centroids(
        slide_name=slide_name,
        slide_df=slide_df,
        halfsize_level0_col=halfsize_level0_col,
        thumbnail_max_size=thumbnail_max_size,
        split_lookup=split_lookup,
        filename_index=filename_index,
    )
    if projected is None:
        return 0
    thumb, thumb_h, thumb_w, px, py, half_px, _thumb_scale = projected

    pred_class_name = slide_df['pred_class_name'].to_numpy()
    true_class_name = slide_df['true_class_name'].to_numpy()

    fig_w = max(7.0, float(thumb_w) / 220.0)
    fig_h = max(7.0, float(thumb_h) / 220.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    try:
        ax.imshow(thumb)

        mismatch = pred_class_name != true_class_name
        classes_present = [c for c in class_names if np.any(pred_class_name == c) or np.any(true_class_name == c)]
        for class_name in classes_present:
            color = color_map.get(class_name, '#999999')

            gt_sel = (true_class_name == class_name) & mismatch
            if np.any(gt_sel):
                gt_mask = _rasterize_squares_mask(px[gt_sel], py[gt_sel], half_px[gt_sel], thumb_h, thumb_w)
                for pts_xy, is_hole in _mask_to_closed_contours(gt_mask):
                    _draw_closed_polyline(ax, pts_xy, color=color, linewidth=1.4, linestyle=':' if is_hole else '--', alpha=0.85, zorder=3)

            pred_sel = pred_class_name == class_name
            if np.any(pred_sel):
                pred_mask = _rasterize_squares_mask(px[pred_sel], py[pred_sel], half_px[pred_sel], thumb_h, thumb_w)
                for pts_xy, is_hole in _mask_to_closed_contours(pred_mask):
                    _draw_closed_polyline(ax, pts_xy, color=color, linewidth=2.2, linestyle=':' if is_hole else '-', alpha=0.95, zorder=4)

        legend_handles = [plt.Line2D([0], [0], color=color_map.get(c, '#999999'), lw=2.2, label=c) for c in classes_present]
        legend_handles.append(plt.Line2D([0], [0], color='black', lw=1.4, linestyle='--', label='true class where misclassified (dashed)'))
        legend_handles.append(plt.Line2D([0], [0], color='black', lw=2.2, linestyle='-', label='predicted (solid)'))
        ax.legend(handles=legend_handles, loc='lower right', fontsize=7, framealpha=0.85)

        n_correct = int(np.sum(pred_class_name == true_class_name))
        ax.set_title(
            f"{slide_name} | {target_field} predictions ({title_noun}) | {n_correct}/{len(pred_class_name)} correct",
            fontsize=9,
        )
        ax.set_xlim(0, thumb_w)
        ax.set_ylim(thumb_h, 0)
        ax.axis('off')
        fig.tight_layout(pad=0.2)

        safe_name = str(slide_name).replace('/', '_').replace('\\', '_').replace(' ', '_')
        out_path = overlays_dir / f"{safe_name}.jpg"
        fig.savefig(out_path, dpi=180, format='jpg', pil_kwargs={"quality": 90, "optimize": True})
    finally:
        plt.close(fig)

    return 1


def _select_region_overlay_threshold(validation_thresholds: Optional[Dict], region_metrics: Dict, key: str = 'region') -> tuple[float, str]:
    if isinstance(validation_thresholds, dict):
        node_block = validation_thresholds.get(key, {})
        if isinstance(node_block, dict) and node_block.get('threshold') is not None:
            return float(node_block['threshold']), f'validation_{key}_threshold'

    # No fallback to the test-set oracle threshold: without a validation threshold the overlay uses 0.5.
    return 0.5, 'default_0.5'


def _normalize_prediction_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    for col in ['slide_name', 'particle_id', 'inflammation_status', 'inflammation_type']:
        if col in df.columns:
            df[col] = df[col].map(_to_python_scalar)

    if 'slide_name' in df.columns:
        df['slide_name'] = df['slide_name'].astype(str)
    if 'particle_id' in df.columns:
        df['particle_id'] = df['particle_id'].astype(str)

    return df


def _safe_unique_particles(df: pd.DataFrame) -> int:
    if not {'slide_name', 'particle_id'}.issubset(df.columns):
        return 0
    try:
        return int(df.groupby(['slide_name', 'particle_id']).ngroups)
    except Exception as exc:
        logging.warning(f"Could not compute unique particle groups directly ({exc}). Falling back to string-normalized keys.")
        fallback = df[['slide_name', 'particle_id']].astype(str)
        return int(fallback.drop_duplicates().shape[0])


def _plot_multiclass_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    class_names: list,
    metrics: Dict,
    output_dir: Path,
    model_name: str,
    split: str,
) -> None:
    """Publication-style multiclass diagnostics: confusion matrix (counts + row-normalized),
    per-class precision/recall/F1 bars, and one-vs-rest ROC/PR curves. Uses the module-level
    apply_ieee_style() rcParams (serif fonts, Okabe-Ito colorblind palette, 300 DPI) like every
    other plot in this file, so figures are consistent enough to drop straight into a thesis."""
    n = len(class_names)
    if n == 0:
        return

    # --- Confusion matrix: raw counts + row-normalized (recall) side by side ---
    cm = np.array(metrics['confusion_matrix'], dtype=np.int64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    panel_w = max(3.6, 0.75 * n + 1.6)
    fig, axes = plt.subplots(1, 2, figsize=(2 * panel_w, max(3.2, 0.75 * n + 1.6)))
    for ax, mat, title, is_int in (
        (axes[0], cm, 'Confusion Matrix (counts)', True),
        (axes[1], cm_norm, 'Confusion Matrix (row-normalized)', False),
    ):
        im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=(mat.max() if mat.max() > 0 else 1))
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title(title)
        thresh = (mat.max() / 2.0) if mat.max() > 0 else 0.5
        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                text = f"{int(val)}" if is_int else f"{float(val):.2f}"
                ax.text(j, i, text, ha='center', va='center',
                        color='white' if val > thresh else 'black', fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_{split}_multiclass_confusion_matrix.png")
    plt.close(fig)

    # --- Per-class precision / recall / F1 bars, annotated with support ---
    per_class = metrics['per_class']
    names = [c for c in class_names if c in per_class]
    precisions = [per_class[c]['precision'] for c in names]
    recalls = [per_class[c]['recall'] for c in names]
    f1s = [per_class[c]['f1'] for c in names]
    supports = [per_class[c]['support'] for c in names]

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(4.5, 1.1 * len(names) + 1.8), 4.2))
    ax.bar(x - width, precisions, width, label='Precision')
    ax.bar(x, recalls, width, label='Recall')
    ax.bar(x + width, f1s, width, label='F1')
    ax.axhline(float(metrics['macro_f1']), linestyle='--', color='gray', linewidth=1,
               label=f"Macro F1={metrics['macro_f1']:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={s})" for c, s in zip(names, supports)])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Precision / Recall / F1')
    ax.legend(loc='best', ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_{split}_multiclass_per_class_metrics.png")
    plt.close(fig)

    # --- One-vs-rest ROC and Precision-Recall curves (needs class probabilities) ---
    if y_prob is None or y_prob.shape[1] != n:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    any_roc = False
    for i, cname in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        any_roc = True
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        auc_val = float(roc_auc_score(y_bin, y_prob[:, i]))
        axes[0].plot(fpr, tpr, label=f"{cname} (AUC={auc_val:.3f})")
    if any_roc:
        axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('One-vs-Rest ROC')
        axes[0].legend(loc='lower right', fontsize=6.5)
    else:
        axes[0].text(0.5, 0.5, 'ROC not available\n(degenerate class)', ha='center', va='center')
        axes[0].set_axis_off()

    any_pr = False
    for i, cname in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0:
            continue
        any_pr = True
        prec, rec, _ = precision_recall_curve(y_bin, y_prob[:, i])
        ap_val = float(average_precision_score(y_bin, y_prob[:, i]))
        baseline = float(y_bin.mean())
        axes[1].plot(rec, prec, label=f"{cname} (AP={ap_val:.3f}, base={baseline:.2f})")
    if any_pr:
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('One-vs-Rest Precision-Recall')
        axes[1].legend(loc='best', fontsize=6.5)
    else:
        axes[1].text(0.5, 0.5, 'PR not available\n(no positives)', ha='center', va='center')
        axes[1].set_axis_off()

    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_{split}_multiclass_roc_pr_curves.png")
    plt.close(fig)


def _return_multiclass_metrics(
    df: pd.DataFrame,
    validation_thresholds: Optional[Dict] = None,
    output_dir: Optional[Path] = None,
    model_name: str = "model",
    split: str = "test",
    save_predictions: bool = True,
    class_names: Optional[list] = None,
    save_plots: bool = True,
) -> Dict:
    validation_thresholds = validation_thresholds or {}
    y_true = df['label'].astype(int).values
    y_pred = df['pred_class'].astype(int).values
    prob_cols = sorted(
        [c for c in df.columns if c.startswith('prob_class_')],
        key=lambda name: int(name.split('_')[-1])
    )
    y_prob = df[prob_cols].values if prob_cols else None
    num_classes = int(len(prob_cols)) if prob_cols else int(df['pred_class'].nunique())
    resolved_class_names = (
        list(class_names) if class_names and len(class_names) == num_classes
        else [f"class_{i}" for i in range(num_classes)]
    )

    metrics = _compute_multiclass_classification_metrics(y_true, y_pred, y_prob, resolved_class_names)
    metrics['mode'] = 'multiclass'
    metrics['num_classes'] = num_classes
    metrics['validation'] = validation_thresholds
    metrics['multiclass_metrics'] = {
        'accuracy': metrics['accuracy'],
        'f1_macro': metrics['macro_f1'],
        'auc_ovo_macro': metrics.get('auc_ovo_macro') or 0.0,
    }

    logging.info("Multiclass evaluation detected: threshold-based binary metrics are skipped.")
    logging.info(
        "Multiclass accuracy=%.4f | balanced_accuracy=%.4f | Cohen's kappa=%.4f | MCC=%.4f | F1-macro=%.4f | AUROC-OVO-macro=%s",
        metrics['accuracy'], metrics['balanced_accuracy'], metrics['cohen_kappa'], metrics['mcc'], metrics['macro_f1'],
        f"{metrics['auc_ovo_macro']:.4f}" if metrics.get('auc_ovo_macro') is not None else "n/a",
    )
    for name, m in metrics['per_class'].items():
        logging.info(
            "  %s: precision=%.3f recall=%.3f specificity=%s balanced_acc=%s f1=%.3f support=%d",
            name, m['precision'], m['recall'],
            f"{m['specificity']:.3f}" if m['specificity'] is not None else "n/a",
            f"{m['balanced_accuracy']:.3f}" if m['balanced_accuracy'] is not None else "n/a",
            m['f1'], m['support'],
        )

    if output_dir and save_predictions:
        df.to_csv(output_dir / f"{model_name}_{split}_predictions.csv", index=False)
    if output_dir:
        with open(output_dir / f"{model_name}_{split}_multiclass_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    if output_dir and save_plots:
        _plot_multiclass_diagnostics(
            y_true=y_true, y_pred=y_pred, y_prob=y_prob, class_names=resolved_class_names,
            metrics=metrics, output_dir=output_dir, model_name=model_name, split=split,
        )

    return metrics


def _diagnostic_threshold(thresholds: Optional[Dict], *keys: str) -> float:
    """Validation threshold for a diagnostic figure: the first of `keys` present in `thresholds`, else 0.5."""
    for key in keys:
        entry = (thresholds or {}).get(key)
        if isinstance(entry, dict) and 'threshold' in entry:
            return float(entry['threshold'])
    return 0.5


def _mil_bag_slide_names(data_loader, task: str) -> Optional[List[str]]:
    """Slide name per bag for an unshuffled inflammation MILDataset loader, else None (-> 'bag_<idx>').

    MILDataset.bags groups tiles by slide_name for 'inflammation' (one bag per slide), and the eval
    loader uses shuffle=False, so bag i of the iteration is dataset.bags[i]. For 'tissue' a bag is a
    (slide, particle) pair, so slide names would not be unique per bag -- keep the bag index there.
    """
    from torch.utils.data import SequentialSampler

    dataset = getattr(data_loader, "dataset", None)
    base = getattr(dataset, "base", None)
    if (task != "inflammation" or not isinstance(getattr(data_loader, "sampler", None), SequentialSampler)
            or not hasattr(dataset, "bags") or not hasattr(base, "samples")):
        return None
    return [base.samples[bag[0]]["slide_name"] for bag in dataset.bags]


@register_evaluation_pipeline(('inflammation', 'mil', 'image'))
def evaluate_model_img_mil(
    model: nn.Module,
    data_loader: DataLoader,
    task: str,
    split: str,
    device: torch.device,
    output_dir: Optional[Path] = None,
    validation_thresholds: Optional[Dict] = None,
    save_predictions: bool = True
) -> Dict:
    """Evaluate an image-based MIL model at appropriate hierarchical levels."""
    model.eval()
    predictions = []

    if hasattr(model, 'model_name'):
        model_name = model.model_name
    else:
        model_name = model.__class__.__name__

    logging.info(f"Evaluating model: {model_name}")

    global_idx = 0
    multiclass_detected = False
    bag_slide_names = _mil_bag_slide_names(data_loader, task)

    with torch.no_grad():
        for images_flat, bag_idx, labels in tqdm(data_loader, desc=f"Evaluating {split}"):
            images_flat = images_flat.to(device)
            bag_idx = bag_idx.to(device)
            labels = labels.to(device).float()


            num_bags = labels.shape[0]
            for b in range(num_bags):
                mask = (bag_idx == b)
                bag_images = images_flat[mask]
                bag_label = labels[b]
                bag_bag_idx = bag_idx[mask]


                outputs, _ = model(bag_images, bag_bag_idx*0)
                if outputs.ndim > 0 and outputs.numel() > 1:
                    multiclass_detected = True
                    logits = outputs.view(1, -1)
                    probs = torch.softmax(logits, dim=1)
                    pred_class = int(torch.argmax(probs, dim=1).item())
                    label_val = int(bag_label.squeeze().item())

                    pred_dict = {
                        "raw_pred": float(logits[0, pred_class].item()),
                        "prob": float(probs[0, pred_class].item()),
                        "pred_class": pred_class,
                        "label": label_val,
                        "slide_name": bag_slide_names[global_idx] if bag_slide_names else f"bag_{global_idx}",
                        "particle_id": 0,
                    }
                    for class_idx in range(logits.size(1)):
                        pred_dict[f'prob_class_{class_idx}'] = float(probs[0, class_idx].item())

                    if task == "inflammation":
                        pred_dict["inflammation_type"] = "unknown"

                    predictions.append(pred_dict)
                    global_idx += 1
                    continue

                probs = torch.sigmoid(outputs)

                if outputs.ndim == 0:
                    raw_pred = outputs.item()
                    prob = probs.item()
                else:
                    raw_pred = outputs.squeeze().item()
                    prob = probs.squeeze().item()

                label_val = bag_label.squeeze().item()

                pred_dict = {
                    "raw_pred": float(raw_pred),
                    "prob": float(prob),
                    "label": int(label_val),
                    "slide_name": bag_slide_names[global_idx] if bag_slide_names else f"bag_{global_idx}",
                    "particle_id": 0,
                }

                if task == "inflammation":
                    pred_dict["inflammation_type"] = "unknown"

                predictions.append(pred_dict)
                global_idx += 1

    df = pd.DataFrame(predictions)
    df = _normalize_prediction_df(df)

    if multiclass_detected and 'pred_class' in df.columns:
        return _return_multiclass_metrics(
            df=df,
            validation_thresholds=validation_thresholds,
            output_dir=output_dir,
            model_name=model_name,
            split=split,
            save_predictions=save_predictions,
        )

    logging.info(f"\nEvaluation Summary:")
    logging.info(f"Total samples (bags): {len(df)}")


    pos_count = (df['label'] == 1).sum()
    neg_count = (df['label'] == 0).sum()
    logging.info(f"\nClass Distribution:")
    logging.info(f"Positive class: {pos_count} ({pos_count/len(df):.1%})")
    logging.info(f"Negative class: {neg_count} ({neg_count/len(df):.1%})")


    logging.info("\nPrediction Statistics:")
    logging.info(f"Raw logits mean: {df['raw_pred'].mean():.3f}")
    logging.info(f"Raw logits std: {df['raw_pred'].std():.3f}")
    logging.info(f"Raw logits min: {df['raw_pred'].min():.3f}")
    logging.info(f"Raw logits max: {df['raw_pred'].max():.3f}")



    if validation_thresholds is None:
        logging.warning(
            "No validation thresholds provided. Falling back to default thresholds/aggregation for evaluation only."
        )
        validation_thresholds = {}


    logging.info("\nUsing validation-optimized thresholds:")
    optimal_thresholds_dict = {k: v for k, v in validation_thresholds.items()
                             if k not in ['validation_tile_auroc', 'validation_slide_auroc',
                                        'validation_particle_auroc', 'optimal_aggregation',
                                        'adaptive_threshold']}
    for level, level_metrics in optimal_thresholds_dict.items():
        if isinstance(level_metrics, dict) and 'threshold' in level_metrics:
            logging.info(f"{level.capitalize()} level: {level_metrics['threshold']:.3f}")
            sensitivity = level_metrics.get('sensitivity', None)
            specificity = level_metrics.get('specificity', None)
            f1 = level_metrics.get('f1', None)
            if sensitivity is not None:
                logging.info(f"  Sensitivity: {float(sensitivity):.2f}")
            else:
                logging.info("  Sensitivity: n/a")
            if specificity is not None:
                logging.info(f"  Specificity: {float(specificity):.2f}")
            else:
                logging.info("  Specificity: n/a")
            if f1 is not None:
                logging.info(f"  F1 Score: {float(f1):.2f}")
            else:
                logging.info("  F1 Score: n/a")
        else:
            logging.info(f"{level.capitalize()} level: Invalid threshold format: {level_metrics}")


    for auc_key in ['validation_tile_auroc', 'validation_slide_auroc', 'validation_particle_auroc']:
        if auc_key in validation_thresholds:
            logging.info(f"{auc_key.replace('_', ' ').capitalize()}: {validation_thresholds[auc_key]:.4f}")


    _skip_legacy_agg = task == 'inflammation'
    if _skip_legacy_agg:
        agg_results = {
            'strategy': 'slide_score',
            'description': ('Slide score and decision threshold come from the validation set '
                            '(slide-level G-mean threshold); no test-set aggregation or threshold search.'),
        }
        if 'prob' not in df.columns:
            if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
                df['prob'] = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
            else:
                df['prob'] = df['raw_pred']
        slide_df = df.groupby('slide_name').agg({'prob': 'mean', 'label': 'first'}).reset_index()
    elif "optimal_aggregation" in validation_thresholds:
        logging.info("\nUsing validation-optimized aggregation strategy:")
        agg_results = validation_thresholds["optimal_aggregation"]

        if 'best_metrics' not in agg_results or not isinstance(agg_results['best_metrics'], dict):

            agg_results['best_metrics'] = {'threshold': 0.5}

        val_tp = agg_results['best_metrics'].get('tp', 0)
        val_tn = agg_results['best_metrics'].get('tn', 0)
        val_fp = agg_results['best_metrics'].get('fp', 0)
        val_fn = agg_results['best_metrics'].get('fn', 0)

        agg_results['validation_confusion_matrix'] = {
            'tp': val_tp,
            'tn': val_tn,
            'fp': val_fp,
            'fn': val_fn,
            'total': val_tp + val_tn + val_fp + val_fn
        }
        logging.info("\nValidation Confusion Matrix:")
        logging.info(f"  True Positives: {val_tp}")
        logging.info(f"  True Negatives: {val_tn}")
        logging.info(f"  False Positives: {val_fp}")
        logging.info(f"  False Negatives: {val_fn}")
        logging.info(f"  Total Samples: {val_tp + val_tn + val_fp + val_fn}")

        best_strategy = agg_results.get('best_strategy', 'mean')
        best_threshold = agg_results['best_metrics'].get('threshold', 0.5)

        if best_strategy == 'mean':
            agg_func = np.mean
        elif best_strategy == 'median':
            agg_func = np.median
        elif best_strategy == 'top_k_mean_10':
            agg_func = lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*0.1)):]) if len(x) > 0 else 0
        elif best_strategy == 'top_k_mean_20':
            agg_func = lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*0.2)):]) if len(x) > 0 else 0
        elif best_strategy == 'top_k_mean_30':
            agg_func = lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*0.3)):]) if len(x) > 0 else 0
        else:
            logging.warning(f"Unknown aggregation strategy: {best_strategy}, defaulting to mean")
            agg_func = np.mean


        if 'prob' not in df.columns:
            if 'raw_pred' in df.columns:
                if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
                    logging.info("Converting raw logits to probabilities with sigmoid...")
                    df['prob'] = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
                else:
                    logging.info("Raw predictions are already in probability range [0,1]")
                    df['prob'] = df['raw_pred']
            else:
                logging.warning("No raw_pred column found in dataframe")
                df['prob'] = 0.5

        logging.info("Skipping any test-set-based prediction inversion to preserve unbiased evaluation.")

        if task == 'inflammation':
            slide_df = df.groupby('slide_name').agg({
                'prob': agg_func,
                'label': 'first'
            }).reset_index()
            slide_df = _drop_non_finite_agg_rows(slide_df, prob_col='prob', name_col='slide_name')
            if 'prob' in slide_df.columns:
                slide_preds = (slide_df['prob'] > best_threshold).astype(int)
                logging.info(f"Applied threshold {best_threshold:.4f} to slide-level probabilities")
            else:
                logging.warning("No 'prob' column found in slide_df, using 'raw_pred' instead")
                slide_preds = (slide_df['raw_pred'] > best_threshold).astype(int)
            try:
                from sklearn.metrics import roc_auc_score
                optimal_agg_auc = roc_auc_score(slide_df['label'].values, slide_df['prob'].values)
                logging.info(f"Optimal aggregation AUC: {optimal_agg_auc:.4f}")
            except Exception as e:
                logging.warning(f"Could not calculate AUC for optimal aggregation: {str(e)}")
                optimal_agg_auc = float('nan')
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(slide_df['label'].values, slide_preds, labels=[0, 1]).ravel()
            test_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            test_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            test_f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 0
            agg_results['test_confusion_matrix'] = {
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'total': int(tp + tn + fp + fn),
                'accuracy': float(test_accuracy),
                'sensitivity': float(test_sensitivity),
                'specificity': float(test_specificity),
                'f1': float(test_f1),
                'balanced_acc': float((test_sensitivity + test_specificity) / 2),
                'auc': float(optimal_agg_auc)
            }

            agg_results['best_metrics'] = {
                'threshold': float(best_threshold),
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'accuracy': float(test_accuracy),
                'sensitivity': float(test_sensitivity),
                'specificity': float(test_specificity),
                'f1': float(test_f1),
                'balanced_acc': float((test_sensitivity + test_specificity) / 2),
                'auc': float(optimal_agg_auc)
            }
            logging.info("\nTest Confusion Matrix:")
            logging.info(f"  True Positives: {tp}")
            logging.info(f"  True Negatives: {tn}")
            logging.info(f"  False Positives: {fp}")
            logging.info(f"  False Negatives: {fn}")
            logging.info(f"  Total Samples: {tp + tn + fp + fn}")
            logging.info(f"  Accuracy: {test_accuracy:.4f}")
            logging.info(f"  F1 Score: {test_f1:.4f}")
            logging.info(f"  AUC: {optimal_agg_auc:.4f}")

        elif task == 'tissue':
            particle_df = df.groupby(['slide_name', 'particle_id']).agg({
                'prob': agg_func,
                'label': 'first'
            }).reset_index()
            particle_df = _drop_non_finite_agg_rows(particle_df, prob_col='prob', name_col='slide_name')
            if 'prob' in particle_df.columns:
                particle_preds = (particle_df['prob'] > best_threshold).astype(int)
                logging.info(f"Applied threshold {best_threshold:.4f} to particle-level probabilities")
            else:
                logging.warning("No 'prob' column found in particle_df, using 'raw_pred' instead")
                particle_preds = (particle_df['raw_pred'] > best_threshold).astype(int)
            try:
                from sklearn.metrics import roc_auc_score
                optimal_agg_auc = roc_auc_score(particle_df['label'].values, particle_df['prob'].values)
                logging.info(f"Optimal aggregation AUC: {optimal_agg_auc:.4f}")
            except Exception as e:
                logging.warning(f"Could not calculate AUC for optimal aggregation: {str(e)}")
                optimal_agg_auc = float('nan')
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(particle_df['label'].values, particle_preds, labels=[0, 1]).ravel()
            test_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            test_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            test_f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 0
            agg_results['test_confusion_matrix'] = {
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'total': int(tp + tn + fp + fn),
                'accuracy': float(test_accuracy),
                'sensitivity': float(test_sensitivity),
                'specificity': float(test_specificity),
                'f1': float(test_f1),
                'balanced_acc': float((test_sensitivity + test_specificity) / 2),
                'auc': float(optimal_agg_auc)
            }

            agg_results['best_metrics'] = {
                'threshold': float(best_threshold),
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'accuracy': float(test_accuracy),
                'sensitivity': float(test_sensitivity),
                'specificity': float(test_specificity),
                'f1': float(test_f1),
                'balanced_acc': float((test_sensitivity + test_specificity) / 2),
                'auc': float(optimal_agg_auc)
            }
            logging.info("\nTest Confusion Matrix:")
            logging.info(f"  True Positives: {tp}")
            logging.info(f"  True Negatives: {tn}")
            logging.info(f"  False Positives: {fp}")
            logging.info(f"  False Negatives: {fn}")
            logging.info(f"  Total Samples: {tp + tn + fp + fn}")
            logging.info(f"  Accuracy: {test_accuracy:.4f}")
            logging.info(f"  F1 Score: {test_f1:.4f}")
            logging.info(f"  AUC: {optimal_agg_auc:.4f}")
    else:
        logging.info("\nNo pre-computed aggregation strategy found in validation thresholds.")
        agg_results = validation_thresholds.get("optimal_aggregation", {})
        if not agg_results or 'best_metrics' not in agg_results or not isinstance(agg_results['best_metrics'], dict):
            logging.warning("Using default mean aggregation with threshold of 0.5")
            agg_results = {
                "best_strategy": "mean",
                "best_metrics": {"threshold": 0.5}
            }
            logging.info("This default strategy was NOT optimized - results may be suboptimal.")


    best_metrics = agg_results.get('best_metrics', {})
    if not _skip_legacy_agg:
        best_strategy = agg_results.get('best_strategy', 'mean')
        logging.info(f"  Strategy: {best_strategy}")
        logging.info(f"  Threshold: {best_metrics.get('threshold', 0.5):.3f}")
        logging.info(f"  Sensitivity: {best_metrics.get('sensitivity', 0):.2f}")
        logging.info(f"  Specificity: {best_metrics.get('specificity', 0):.2f}")
        logging.info(f"  F1 Score: {best_metrics.get('f1', 0):.2f}")


    if output_dir:
        probs = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()

        plt.figure(figsize=(16, 7), dpi=300)

        plt.subplot(1, 2, 1)
        plt.hist(df['raw_pred'][df['label']==1], bins=50, alpha=0.7, label='Positive class', color='#1f77b4', edgecolor='black', linewidth=0.7)
        plt.hist(df['raw_pred'][df['label']==0], bins=50, alpha=0.7, label='Negative class', color='#ff7f0e', edgecolor='black', linewidth=0.7)
        plt.axvline(x=0.0, color='r', linestyle='--', linewidth=2, label='Default threshold (logit=0)')
        if task == 'inflammation' and 'slide' in optimal_thresholds_dict:
            prob_threshold = optimal_thresholds_dict['slide']['threshold']
            logit_threshold = np.log(prob_threshold/(1-prob_threshold)) if 0 < prob_threshold < 1 else (10 if prob_threshold >= 1 else -10)
            plt.axvline(x=logit_threshold, color='g', linestyle='--', linewidth=2,
                      label=f'Optimal threshold (logit={logit_threshold:.2f})')
        elif task == 'tissue' and 'particle' in optimal_thresholds_dict:
            prob_threshold = optimal_thresholds_dict['particle']['threshold']
            logit_threshold = np.log(prob_threshold/(1-prob_threshold)) if 0 < prob_threshold < 1 else (10 if prob_threshold >= 1 else -10)
            plt.axvline(x=logit_threshold, color='g', linestyle='--', linewidth=2,
                      label=f'Optimal threshold (logit={logit_threshold:.2f})')
        plt.title('Raw Logits Distribution by Class', fontsize=18, fontweight='bold')
        plt.xlabel('Logit Value', fontsize=16, fontweight='bold')
        plt.ylabel('Count', fontsize=16, fontweight='bold')
        plt.legend(fontsize=13, frameon=True)
        plt.grid(True, alpha=0.4, linestyle='--')

        plt.subplot(1, 2, 2)
        plt.hist(probs[df['label']==1], bins=50, alpha=0.7, label='Positive class', color='#1f77b4', edgecolor='black', linewidth=0.7)
        plt.hist(probs[df['label']==0], bins=50, alpha=0.7, label='Negative class', color='#ff7f0e', edgecolor='black', linewidth=0.7)
        plt.axvline(x=0.5, color='r', linestyle='--', linewidth=2, label='Default threshold (prob=0.5)')
        if task == 'inflammation' and 'slide' in optimal_thresholds_dict:
            plt.axvline(x=optimal_thresholds_dict['slide']['threshold'], color='g', linestyle='--', linewidth=2,
                      label=f'Optimal threshold (prob={optimal_thresholds_dict["slide"]["threshold"]:.2f})')
        elif task == 'tissue' and 'particle' in optimal_thresholds_dict:
            plt.axvline(x=optimal_thresholds_dict['particle']['threshold'], color='g', linestyle='--', linewidth=2,
                      label=f'Optimal threshold (prob={optimal_thresholds_dict["particle"]["threshold"]:.2f})')
        if 'threshold' in best_metrics:
            plt.axvline(x=best_metrics['threshold'], color='purple', linestyle='-.', linewidth=2,
                      label=f'Best agg. threshold (prob={best_metrics["threshold"]:.2f})')
        plt.title('Probability Distribution by Class', fontsize=18, fontweight='bold')
        plt.xlabel('Probability', fontsize=16, fontweight='bold')
        plt.ylabel('Count', fontsize=16, fontweight='bold')
        plt.legend(fontsize=13, frameon=True)
        plt.grid(True, alpha=0.4, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_dir / f'pred_distribution_{split}.png', dpi=300, bbox_inches='tight')
        plt.close()

        try:
            from src.utils.visualization_utils import plot_classification_diagnostics
            diagnostic_stats = {}
            if 'slide_df' in locals():
                diagnostic_stats['slide'] = plot_classification_diagnostics(
                    slide_df['label'].values, slide_df['prob'].values, output_dir,
                    prefix=f"{model_name}_{split}_slide", level_name="Slide-level",
                    threshold=_diagnostic_threshold(validation_thresholds, 'slide_mean_prob', 'slide'),
                )
            elif 'particle_df' in locals():
                diagnostic_stats['particle'] = plot_classification_diagnostics(
                    particle_df['label'].values, particle_df['prob'].values, output_dir,
                    prefix=f"{model_name}_{split}_particle", level_name="Particle-level",
                    threshold=_diagnostic_threshold(validation_thresholds, 'particle'),
                )
            agg_results['test_diagnostic_plots'] = diagnostic_stats
        except Exception as exc:
            logging.warning(f"Could not generate classification diagnostic plots for split '{split}': {exc}")


    if task == 'inflammation' and 'slide' in optimal_thresholds_dict:
        selected_threshold = optimal_thresholds_dict['slide']['threshold']
        if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
            threshold_for_metrics = np.log(selected_threshold/(1-selected_threshold)) if 0 < selected_threshold < 1 else 0.0
            logging.info(f"Predictions are logits - converting probability threshold {selected_threshold:.4f} to logit threshold: {threshold_for_metrics:.4f}")
        else:
            threshold_for_metrics = selected_threshold
            logging.info(f"Predictions are probabilities - using probability threshold directly: {threshold_for_metrics:.4f}")
    elif task == 'tissue' and 'particle' in optimal_thresholds_dict:
        selected_threshold = optimal_thresholds_dict['particle']['threshold']
        if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
            threshold_for_metrics = np.log(selected_threshold/(1-selected_threshold)) if 0 < selected_threshold < 1 else 0.0
            logging.info(f"Predictions are logits - converting probability threshold {selected_threshold:.4f} to logit threshold: {threshold_for_metrics:.4f}")
        else:
            threshold_for_metrics = selected_threshold
            logging.info(f"Predictions are probabilities - using probability threshold directly: {threshold_for_metrics:.4f}")
    else:
        if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
            threshold_for_metrics = 0.0
        else:
            threshold_for_metrics = 0.5


    metrics = metrics_utils.calculate_hierarchical_metrics(
        df=df,
        split=split,
        model_name=model_name,
        threshold=threshold_for_metrics
    )


    combined_metrics = {
        **metrics,
        'optimal_thresholds': optimal_thresholds_dict,
        'optimal_aggregation': agg_results,
    }

    for auc_key in ['validation_tile_auroc', 'validation_slide_auroc', 'validation_particle_auroc']:
        if auc_key in validation_thresholds:
            combined_metrics[auc_key] = validation_thresholds[auc_key]

    if task == 'inflammation' and 'prob' in df.columns:
        # calculate_hierarchical_metrics above (metrics_utils.calculate_metrics) gives slide_accuracy/
        # sensitivity/specificity/f1/auroc at the (non-recalibrated) validation-optimized threshold,
        # with no F2, no Wilson CIs, no oracle-threshold sweep, and no adaptive-threshold
        # recalibration. evaluate_model_graph's slide-level binary report has all of those, computed
        # from ONE single call to _compute_slide_binary_metrics so every reported number (accuracy,
        # sensitivity, F1, F2, ...) uses the same, possibly-recalibrated threshold. To make a MIL run
        # and a graph run directly comparable at the slide level, the slide_* keys below are fully
        # replaced (not just extended) by that same single call, on the same per-slide df (each MIL
        # bag already equals exactly one slide) -- so slide_accuracy and slide_f2 can never disagree
        # about which threshold produced them, exactly as in the graph pipeline.
        #
        # adaptive_cfg/target_rate: calculate_validation_thresholds_mil_image (src/pipelines/
        # validation.py) now writes the same "adaptive_threshold" config block and the same
        # validation_pred_positive_rate/validation_label_positive_rate fields under 'slide' as
        # calculate_validation_thresholds_graph does, so this recalibrates identically to the graph
        # pipeline. Falls back to no recalibration (target_rate stays None) for an older
        # validation_thresholds.json predating this field.
        mil_adaptive_cfg = validation_thresholds.get('adaptive_threshold', {}) if validation_thresholds else {}
        mil_target_rate = optimal_thresholds_dict.get('slide', {}).get('validation_pred_positive_rate', None)
        if mil_target_rate is None:
            mil_target_rate = optimal_thresholds_dict.get('slide', {}).get('validation_label_positive_rate', None)

        slide_binary_metrics = _compute_slide_binary_metrics(
            y_true=df['label'].values,
            y_prob=df['prob'].values,
            base_threshold=float(optimal_thresholds_dict.get('slide', {}).get('threshold', 0.5)),
            adaptive_cfg=mil_adaptive_cfg,
            target_rate=mil_target_rate,
        )
        combined_metrics['slide_accuracy'] = slide_binary_metrics['accuracy']
        combined_metrics['slide_sensitivity'] = slide_binary_metrics['sensitivity']
        combined_metrics['slide_specificity'] = slide_binary_metrics['specificity']
        combined_metrics['slide_precision'] = (
            slide_binary_metrics['tp'] / (slide_binary_metrics['tp'] + slide_binary_metrics['fp'])
            if (slide_binary_metrics['tp'] + slide_binary_metrics['fp']) > 0 else 0.0
        )
        combined_metrics['slide_f1'] = slide_binary_metrics['f1']
        combined_metrics['slide_auroc'] = slide_binary_metrics['auc']
        combined_metrics['slide_f2'] = slide_binary_metrics['f2']
        combined_metrics['slide_balanced_acc'] = slide_binary_metrics['balanced_acc']
        combined_metrics['slide_threshold'] = slide_binary_metrics['threshold']
        combined_metrics['slide_confusion_matrix'] = {
            'tp': slide_binary_metrics['tp'], 'tn': slide_binary_metrics['tn'],
            'fp': slide_binary_metrics['fp'], 'fn': slide_binary_metrics['fn'],
            'total': slide_binary_metrics['total'],
        }
        combined_metrics['slide_sensitivity_ci95'] = slide_binary_metrics['sensitivity_ci95']
        combined_metrics['slide_specificity_ci95'] = slide_binary_metrics['specificity_ci95']
        combined_metrics['slide_oracle_test_best_f1'] = slide_binary_metrics['oracle_test_best_f1']
        combined_metrics['slide_oracle_test_best_balanced_accuracy'] = slide_binary_metrics['oracle_test_best_balanced_accuracy']
        combined_metrics['slide_adaptive_applied'] = slide_binary_metrics['adaptive_applied']
        combined_metrics['slide_at_validation_threshold'] = slide_binary_metrics['at_validation_threshold']
        logging.info(
            "\nSlide-level binary metrics (unified with evaluate_model_graph's method; threshold=%.4f%s):",
            slide_binary_metrics['threshold'],
            " [adaptively recalibrated]" if abs(slide_binary_metrics['threshold'] - float(optimal_thresholds_dict.get('slide', {}).get('threshold', 0.5))) > 1e-9 else "",
        )
        logging.info(
            "  Accuracy: %.4f | Sensitivity: %.4f (95%% CI %.4f-%.4f) | Specificity: %.4f (95%% CI %.4f-%.4f)",
            slide_binary_metrics['accuracy'],
            slide_binary_metrics['sensitivity'], slide_binary_metrics['sensitivity_ci95'][0], slide_binary_metrics['sensitivity_ci95'][1],
            slide_binary_metrics['specificity'], slide_binary_metrics['specificity_ci95'][0], slide_binary_metrics['specificity_ci95'][1],
        )
        logging.info(
            "  F1: %.4f | F2 (recall weighted 4x precision): %.4f | Balanced Accuracy: %.4f | AUC: %.4f",
            slide_binary_metrics['f1'], slide_binary_metrics['f2'], slide_binary_metrics['balanced_acc'], slide_binary_metrics['auc'],
        )

        # Persist like evaluate_model_img (<model>_<split>_predictions.csv / _slide_metrics.json) so
        # MIL test results can be compared from files instead of the log. One bag == one slide here.
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            if save_predictions:
                df.to_csv(output_dir / f"{model_name}_{split}_predictions.csv", index=False)
                df[['slide_name', 'prob', 'label']].to_csv(
                    output_dir / f"{model_name}_{split}_slide_predictions.csv", index=False)
            slide_metrics_out = {
                k: v for k, v in combined_metrics.items()
                if k.startswith('slide_') or k.startswith('validation_')
            }
            with open(output_dir / f"{model_name}_{split}_slide_metrics.json", "w", encoding="utf-8") as f:
                json.dump(slide_metrics_out, f, indent=2, default=float)

    return combined_metrics


@register_evaluation_pipeline(('inflammation', 'cnn', 'image'))
@register_evaluation_pipeline(('tissue', 'cnn', 'image'))
def evaluate_model_img(
    model: nn.Module,
    data_loader: DataLoader,
    task: str,
    split: str,
    device: torch.device,
    output_dir: Optional[Path] = None,
    validation_thresholds: Optional[Dict] = None,
    save_predictions: bool = True
) -> Dict:
    """Evaluate an image-based model at appropriate hierarchical levels."""
    model.eval()
    predictions = []


    if hasattr(model, 'model_name'):
        model_name = model.model_name
    else:

        model_name = model.__class__.__name__

    logging.info(f"Evaluating model: {model_name}")

    multiclass_detected = False


    with torch.no_grad():
        for inputs, labels, metadata_batch in tqdm(data_loader, desc=f"Evaluating {split}"):
            inputs = inputs.to(device)
            outputs = model(inputs)

            if _is_multiclass_logits(outputs):
                multiclass_detected = True
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                if isinstance(labels, torch.Tensor) and labels.dim() == 0:
                    labels = labels.unsqueeze(0)

                for i in range(outputs.size(0)):
                    metadata = {
                        k: _to_python_scalar(v[i] if isinstance(v, (list, torch.Tensor)) else v)
                        for k, v in metadata_batch.items()
                    }
                    pred_class = int(preds[i].item())
                    pred_dict = {
                        'raw_pred': float(outputs[i, pred_class].detach().cpu().item()),
                        'prob': float(probs[i, pred_class].detach().cpu().item()),
                        'pred_class': pred_class,
                        'label': int(labels[i].item()),
                        'particle_id': metadata['particle_id'],
                        'slide_name': metadata['slide_name'],
                    }
                    for class_idx in range(outputs.size(1)):
                        pred_dict[f'prob_class_{class_idx}'] = float(probs[i, class_idx].detach().cpu().item())

                    if task == 'inflammation':
                        pred_dict['inflammation_type'] = metadata.get('inflammation_status', 'unknown')

                    predictions.append(pred_dict)
                continue

            if outputs.dim() > 1:
                outputs = outputs.squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)

            if isinstance(labels, torch.Tensor) and labels.dim() == 0:
                labels = labels.unsqueeze(0)


            is_logits = False
            if outputs.numel() > 0:
                min_val = outputs.min().item()
                max_val = outputs.max().item()
                is_logits = min_val < 0 or max_val > 1

                if is_logits:

                    probs = torch.sigmoid(outputs)
                else:

                    probs = outputs
            else:
                probs = outputs


            for i in range(len(outputs)):
                metadata = {
                    k: _to_python_scalar(v[i] if isinstance(v, (list, torch.Tensor)) else v)
                    for k, v in metadata_batch.items()
                }

                pred_dict = {
                    'raw_pred': outputs[i].cpu().item(),
                    'prob': probs[i].cpu().item(),
                    'label': labels[i].item(),
                    'particle_id': metadata['particle_id'],
                    'slide_name': metadata['slide_name'],
                }


                if task == 'inflammation':

                    pred_dict['inflammation_type'] = metadata.get('inflammation_status', 'unknown')

                predictions.append(pred_dict)


    df = pd.DataFrame(predictions)
    df = _normalize_prediction_df(df)
    df = _normalize_prediction_df(df)

    if multiclass_detected and 'pred_class' in df.columns:
        return _return_multiclass_metrics(
            df=df,
            validation_thresholds=validation_thresholds,
            output_dir=output_dir,
            model_name=model_name,
            split=split,
            save_predictions=save_predictions,
        )


    logging.info(f"\nEvaluation Summary:")
    logging.info(f"Total samples: {len(df)}")
    logging.info(f"Unique slides: {df['slide_name'].nunique()}")
    logging.info(f"Unique particles: {_safe_unique_particles(df)}")


    pos_count = (df['label'] == 1).sum()
    neg_count = (df['label'] == 0).sum()
    logging.info(f"\nClass Distribution:")
    logging.info(f"Positive class: {pos_count} ({pos_count/len(df):.1%})")
    logging.info(f"Negative class: {neg_count} ({neg_count/len(df):.1%})")


    logging.info("\nPrediction Statistics:")
    logging.info(f"Raw logits mean: {df['raw_pred'].mean():.3f}")
    logging.info(f"Raw logits std: {df['raw_pred'].std():.3f}")
    logging.info(f"Raw logits min: {df['raw_pred'].min():.3f}")
    logging.info(f"Raw logits max: {df['raw_pred'].max():.3f}")



    if validation_thresholds is None:
        logging.warning(
            "No validation thresholds provided. Falling back to default thresholds/aggregation for evaluation only."
        )
        validation_thresholds = {}


    logging.info("\nUsing validation-optimized thresholds:")
    optimal_thresholds_dict = {k: v for k, v in validation_thresholds.items()
                             if k not in ['validation_tile_auroc', 'validation_slide_auroc',
                                        'validation_particle_auroc', 'optimal_aggregation',
                                        'adaptive_threshold', 'validation_slide_auroc_mean_logit']}
    for level, level_metrics in optimal_thresholds_dict.items():
        if isinstance(level_metrics, dict) and 'threshold' in level_metrics:
            logging.info(f"{level.capitalize()} level: {level_metrics['threshold']:.3f}")
            sensitivity = level_metrics.get('sensitivity', None)
            specificity = level_metrics.get('specificity', None)
            f1 = level_metrics.get('f1', None)
            logging.info(f"  Sensitivity: {float(sensitivity):.2f}" if sensitivity is not None else "  Sensitivity: n/a")
            logging.info(f"  Specificity: {float(specificity):.2f}" if specificity is not None else "  Specificity: n/a")
            logging.info(f"  F1 Score: {float(f1):.2f}" if f1 is not None else "  F1 Score: n/a")
        else:
            logging.info(f"{level.capitalize()} level: Invalid threshold format: {level_metrics}")


    for auc_key in ['validation_tile_auroc', 'validation_slide_auroc', 'validation_particle_auroc']:
        if auc_key in validation_thresholds:
            logging.info(f"{auc_key.replace('_', ' ').capitalize()}: {validation_thresholds[auc_key]:.4f}")


    _skip_legacy_agg = task == 'inflammation' and 'slide_mean_prob' in validation_thresholds
    if _skip_legacy_agg:
        agg_results = {
            'strategy': 'slide_score',
            'description': ('Slide score and decision threshold come from the validation set '
                            '(slide-level G-mean threshold); no test-set aggregation or threshold search.'),
        }
        if 'prob' not in df.columns:
            if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
                df['prob'] = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
            else:
                df['prob'] = df['raw_pred']
        slide_df = df.groupby('slide_name').agg({'prob': 'mean', 'label': 'first'}).reset_index()
    elif "optimal_aggregation" in validation_thresholds:
        logging.info("\nUsing validation-optimized aggregation strategy:")
        agg_results = validation_thresholds["optimal_aggregation"]
        if 'best_metrics' not in agg_results or not isinstance(agg_results['best_metrics'], dict):
            agg_results['best_metrics'] = {'threshold': 0.5}
        if 'best_strategy' not in agg_results:
            agg_results['best_strategy'] = 'mean'


        val_tp = agg_results['best_metrics'].get('tp', 0)
        val_tn = agg_results['best_metrics'].get('tn', 0)
        val_fp = agg_results['best_metrics'].get('fp', 0)
        val_fn = agg_results['best_metrics'].get('fn', 0)


        agg_results['validation_confusion_matrix'] = {
            'tp': val_tp,
            'tn': val_tn,
            'fp': val_fp,
            'fn': val_fn,
            'total': val_tp + val_tn + val_fp + val_fn
        }

        logging.info("\nValidation Confusion Matrix:")
        logging.info(f"  True Positives: {val_tp}")
        logging.info(f"  True Negatives: {val_tn}")
        logging.info(f"  False Positives: {val_fp}")
        logging.info(f"  False Negatives: {val_fn}")
        logging.info(f"  Total Samples: {val_tp + val_tn + val_fp + val_fn}")


        best_strategy = agg_results['best_strategy']
        best_threshold = agg_results['best_metrics']['threshold']


        if best_strategy == 'mean':
            agg_func = np.mean
        elif best_strategy == 'median':
            agg_func = np.median
        elif best_strategy == 'top_k_mean_10':
            agg_func = lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*0.1)):]) if len(x) > 0 else 0
        elif best_strategy == 'top_k_mean_20':
            agg_func = lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*0.2)):]) if len(x) > 0 else 0
        elif best_strategy == 'top_k_mean_30':
            agg_func = lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*0.3)):]) if len(x) > 0 else 0
        else:
            logging.warning(f"Unknown aggregation strategy: {best_strategy}, defaulting to mean")
            agg_func = np.mean


        if 'prob' not in df.columns:
            if 'raw_pred' in df.columns:
                if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
                    logging.info("Converting raw logits to probabilities with sigmoid...")
                    df['prob'] = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
                else:
                    logging.info("Raw predictions are already in probability range [0,1]")
                    df['prob'] = df['raw_pred']
            else:
                logging.warning("No raw_pred column found in dataframe")
                df['prob'] = 0.5

        logging.info("Skipping any test-set-based prediction inversion to preserve unbiased evaluation.")

        if task == 'inflammation':
            slide_df = df.groupby('slide_name').agg({
                'prob': agg_func,
                'label': 'first'
            }).reset_index()
            slide_df = _drop_non_finite_agg_rows(slide_df, prob_col='prob', name_col='slide_name')

            if 'prob' in slide_df.columns:
                slide_preds = (slide_df['prob'] > best_threshold).astype(int)
                logging.info(f"Applied threshold {best_threshold:.4f} to slide-level probabilities")
            else:
                logging.warning("No 'prob' column found in slide_df, using 'raw_pred' instead")
                slide_preds = (slide_df['raw_pred'] > best_threshold).astype(int)

            try:
                from sklearn.metrics import roc_auc_score
                optimal_agg_auc = roc_auc_score(slide_df['label'].values, slide_df['prob'].values)
                logging.info(f"Optimal aggregation AUC: {optimal_agg_auc:.4f}")
            except Exception as e:
                logging.warning(f"Could not calculate AUC for optimal aggregation: {str(e)}")
                optimal_agg_auc = float('nan')

            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(slide_df['label'].values, slide_preds, labels=[0, 1]).ravel()

            test_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            test_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            test_f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 0

            agg_results['test_confusion_matrix'] = {
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'total': int(tp + tn + fp + fn),
                'accuracy': float(test_accuracy),
                'sensitivity': float(test_sensitivity),
                'specificity': float(test_specificity),
                'f1': float(test_f1),
                'balanced_acc': float((test_sensitivity + test_specificity) / 2),
                'auc': float(optimal_agg_auc)
            }

            agg_results['best_metrics']['tp'] = int(tp)
            agg_results['best_metrics']['tn'] = int(tn)
            agg_results['best_metrics']['fp'] = int(fp)
            agg_results['best_metrics']['fn'] = int(fn)
            agg_results['best_metrics']['accuracy'] = float(test_accuracy)
            agg_results['best_metrics']['sensitivity'] = float(test_sensitivity)
            agg_results['best_metrics']['specificity'] = float(test_specificity)
            agg_results['best_metrics']['f1'] = float(test_f1)
            agg_results['best_metrics']['balanced_acc'] = float((test_sensitivity + test_specificity) / 2)
            agg_results['best_metrics']['auc'] = float(optimal_agg_auc)

            logging.info("\nTest Confusion Matrix:")
            logging.info(f"  True Positives: {tp}")
            logging.info(f"  True Negatives: {tn}")
            logging.info(f"  False Positives: {fp}")
            logging.info(f"  False Negatives: {fn}")
            logging.info(f"  Total Samples: {tp + tn + fp + fn}")
            logging.info(f"  Accuracy: {test_accuracy:.4f}")
            logging.info(f"  F1 Score: {test_f1:.4f}")
            logging.info(f"  AUC: {optimal_agg_auc:.4f}")

        elif task == 'tissue':
            particle_df = df.groupby(['slide_name', 'particle_id']).agg({
                'prob': agg_func,
                'label': 'first'
            }).reset_index()
            particle_df = _drop_non_finite_agg_rows(particle_df, prob_col='prob', name_col='slide_name')

            if 'prob' in particle_df.columns:
                particle_preds = (particle_df['prob'] > best_threshold).astype(int)
                logging.info(f"Applied threshold {best_threshold:.4f} to particle-level probabilities")
            else:
                logging.warning("No 'prob' column found in particle_df, using 'raw_pred' instead")
                particle_preds = (particle_df['raw_pred'] > best_threshold).astype(int)

            try:
                from sklearn.metrics import roc_auc_score
                optimal_agg_auc = roc_auc_score(particle_df['label'].values, particle_df['prob'].values)
                logging.info(f"Optimal aggregation AUC: {optimal_agg_auc:.4f}")
            except Exception as e:
                logging.warning(f"Could not calculate AUC for optimal aggregation: {str(e)}")
                optimal_agg_auc = float('nan')

            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(particle_df['label'].values, particle_preds, labels=[0, 1]).ravel()

            test_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            test_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            test_f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 0

            agg_results['test_confusion_matrix'] = {
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'total': int(tp + tn + fp + fn),
                'accuracy': float(test_accuracy),
                'sensitivity': float(test_sensitivity),
                'specificity': float(test_specificity),
                'f1': float(test_f1),
                'balanced_acc': float((test_sensitivity + test_specificity) / 2),
                'auc': float(optimal_agg_auc)
            }

            agg_results['best_metrics']['tp'] = int(tp)
            agg_results['best_metrics']['tn'] = int(tn)
            agg_results['best_metrics']['fp'] = int(fp)
            agg_results['best_metrics']['fn'] = int(fn)
            agg_results['best_metrics']['accuracy'] = float(test_accuracy)
            agg_results['best_metrics']['sensitivity'] = float(test_sensitivity)
            agg_results['best_metrics']['specificity'] = float(test_specificity)
            agg_results['best_metrics']['f1'] = float(test_f1)
            agg_results['best_metrics']['balanced_acc'] = float((test_sensitivity + test_specificity) / 2)
            agg_results['best_metrics']['auc'] = float(optimal_agg_auc)

            logging.info("\nTest Confusion Matrix:")
            logging.info(f"  True Positives: {tp}")
            logging.info(f"  True Negatives: {tn}")
            logging.info(f"  False Positives: {fp}")
            logging.info(f"  False Negatives: {fn}")
            logging.info(f"  Total Samples: {tp + tn + fp + fn}")
            logging.info(f"  Accuracy: {test_accuracy:.4f}")
            logging.info(f"  F1 Score: {test_f1:.4f}")
            logging.info(f"  AUC: {optimal_agg_auc:.4f}")
    else:
        logging.info("\nNo pre-computed aggregation strategy found in validation thresholds.")
        agg_results = validation_thresholds.get("optimal_aggregation", {})
        if not agg_results:
            logging.warning("Using default mean aggregation with threshold of 0.5")
            agg_results = {
                "best_strategy": "mean",
                "best_metrics": {"threshold": 0.5}
            }
            logging.info("This default strategy was NOT optimized - results may be suboptimal.")


    best_metrics = agg_results.get('best_metrics', {})
    if not _skip_legacy_agg:
        best_strategy = agg_results.get('best_strategy', 'mean')
        logging.info(f"  Strategy: {best_strategy}")
        logging.info(f"  Threshold: {best_metrics.get('threshold', 0.5):.3f}")
        logging.info(f"  Sensitivity: {best_metrics.get('sensitivity', 0):.2f}")
        logging.info(f"  Specificity: {best_metrics.get('specificity', 0):.2f}")
        logging.info(f"  F1 Score: {best_metrics.get('f1', 0):.2f}")


    if output_dir:
        probs = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()

        plt.figure(figsize=(16, 7))

        plt.subplot(1, 2, 1)
        plt.hist(df['raw_pred'][df['label']==1], bins=50, alpha=0.6, label='Positive class', color='blue')
        plt.hist(df['raw_pred'][df['label']==0], bins=50, alpha=0.6, label='Negative class', color='orange')
        plt.axvline(x=0.0, color='r', linestyle='--', label='Default threshold (logit=0)')

        if task == 'inflammation' and 'tile' in optimal_thresholds_dict:
            prob_threshold = optimal_thresholds_dict['tile']['threshold']
            logit_threshold = np.log(prob_threshold/(1-prob_threshold)) if 0 < prob_threshold < 1 else (10 if prob_threshold >= 1 else -10)
            plt.axvline(x=logit_threshold, color='g', linestyle='--',
                      label=f'Validation tile threshold (logit={logit_threshold:.2f})')
        elif task == 'tissue' and 'particle' in optimal_thresholds_dict:
            prob_threshold = optimal_thresholds_dict['particle']['threshold']
            logit_threshold = np.log(prob_threshold/(1-prob_threshold)) if 0 < prob_threshold < 1 else (10 if prob_threshold >= 1 else -10)
            plt.axvline(x=logit_threshold, color='g', linestyle='--',
                      label=f'Optimal threshold (logit={logit_threshold:.2f})')

        plt.title('Raw Logits Distribution by Class')
        plt.xlabel('Logit Value')
        plt.ylabel('Count')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(probs[df['label']==1], bins=50, alpha=0.6, label='Positive class', color='blue')
        plt.hist(probs[df['label']==0], bins=50, alpha=0.6, label='Negative class', color='orange')
        plt.axvline(x=0.5, color='r', linestyle='--', label='Default threshold (prob=0.5)')

        if task == 'inflammation' and 'tile' in optimal_thresholds_dict:
            plt.axvline(x=optimal_thresholds_dict['tile']['threshold'], color='g', linestyle='--',
                      label=f'Validation tile threshold (prob={optimal_thresholds_dict["tile"]["threshold"]:.2f})')
        elif task == 'tissue' and 'particle' in optimal_thresholds_dict:
            plt.axvline(x=optimal_thresholds_dict['particle']['threshold'], color='g', linestyle='--',
                      label=f'Optimal threshold (prob={optimal_thresholds_dict["particle"]["threshold"]:.2f})')

        if 'threshold' in best_metrics:
            plt.axvline(x=best_metrics['threshold'], color='purple', linestyle='-.',
                      label=f'Best agg. threshold (prob={best_metrics["threshold"]:.2f})')

        plt.title('Probability Distribution by Class')
        plt.xlabel('Probability')
        plt.ylabel('Count')
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_dir / f'pred_distribution_{split}.png', dpi=300)
        plt.close()

        try:
            from src.utils.visualization_utils import plot_classification_diagnostics
            diagnostic_stats = {
                'tile': plot_classification_diagnostics(
                    df['label'].values, probs, output_dir,
                    prefix=f"{model_name}_{split}_tile", level_name="Tile-level",
                    threshold=_diagnostic_threshold(validation_thresholds, 'tile'),
                )
            }
            if 'slide_df' in locals():
                diagnostic_stats['slide'] = plot_classification_diagnostics(
                    slide_df['label'].values, slide_df['prob'].values, output_dir,
                    prefix=f"{model_name}_{split}_slide", level_name="Slide-level",
                    threshold=_diagnostic_threshold(validation_thresholds, 'slide_mean_prob', 'slide'),
                )
            elif 'particle_df' in locals():
                diagnostic_stats['particle'] = plot_classification_diagnostics(
                    particle_df['label'].values, particle_df['prob'].values, output_dir,
                    prefix=f"{model_name}_{split}_particle", level_name="Particle-level",
                    threshold=_diagnostic_threshold(validation_thresholds, 'particle'),
                )
            agg_results['test_diagnostic_plots'] = diagnostic_stats
        except Exception as exc:
            logging.warning(f"Could not generate classification diagnostic plots for split '{split}': {exc}")


    if task == 'inflammation' and 'slide' in optimal_thresholds_dict:
        selected_threshold = optimal_thresholds_dict['slide']['threshold']
        if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
            threshold_for_metrics = np.log(selected_threshold/(1-selected_threshold)) if 0 < selected_threshold < 1 else 0.0
            logging.info(f"Predictions are logits - converting probability threshold {selected_threshold:.4f} to logit threshold: {threshold_for_metrics:.4f}")
        else:
            threshold_for_metrics = selected_threshold
            logging.info(f"Predictions are probabilities - using probability threshold directly: {threshold_for_metrics:.4f}")
    elif task == 'tissue' and 'particle' in optimal_thresholds_dict:
        selected_threshold = optimal_thresholds_dict['particle']['threshold']
        if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
            threshold_for_metrics = np.log(selected_threshold/(1-selected_threshold)) if 0 < selected_threshold < 1 else 0.0
            logging.info(f"Predictions are logits - converting probability threshold {selected_threshold:.4f} to logit threshold: {threshold_for_metrics:.4f}")
        else:
            threshold_for_metrics = selected_threshold
            logging.info(f"Predictions are probabilities - using probability threshold directly: {threshold_for_metrics:.4f}")
    else:
        if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
            threshold_for_metrics = 0.0
        else:
            threshold_for_metrics = 0.5


    metrics = metrics_utils.calculate_hierarchical_metrics(
        df=df,
        split=split,
        model_name=model_name,
        threshold=threshold_for_metrics
    )


    combined_metrics = {
        **metrics,
        'optimal_thresholds': optimal_thresholds_dict,
        'optimal_aggregation': agg_results,
    }

    for auc_key in ['validation_tile_auroc', 'validation_slide_auroc', 'validation_particle_auroc']:
        if auc_key in validation_thresholds:
            combined_metrics[auc_key] = validation_thresholds[auc_key]

    if task == 'inflammation' and 'prob' in df.columns and 'slide_mean_prob' in validation_thresholds:
        # Shared slide-level protocol (Experiment E1), identical to evaluate_model_img_mil /
        # evaluate_model_graph: one score per slide (here: mean tile probability), validation
        # threshold + adaptive recalibration from calculate_validation_thresholds_image, all slide_*
        # numbers (AUC, F1, F2, CIs, ...) from ONE _compute_slide_binary_metrics call. The slide_*
        # keys of calculate_hierarchical_metrics above are replaced, exactly as in the MIL pipeline;
        # tile_* metrics and the aggregation-strategy report stay as they are.
        mean_prob_cfg = validation_thresholds['slide_mean_prob']
        slide_mp = df.groupby('slide_name').agg(prob=('prob', 'mean'), label=('label', 'first')).reset_index()
        mp_target_rate = mean_prob_cfg.get('validation_pred_positive_rate', None)
        if mp_target_rate is None:
            mp_target_rate = mean_prob_cfg.get('validation_label_positive_rate', None)
        slide_binary_metrics = _compute_slide_binary_metrics(
            y_true=slide_mp['label'].values.astype(int),
            y_prob=slide_mp['prob'].values,
            base_threshold=float(mean_prob_cfg.get('threshold', 0.5)),
            adaptive_cfg=validation_thresholds.get('adaptive_threshold', {}),
            target_rate=mp_target_rate,
        )
        combined_metrics['slide_score'] = 'mean_tile_probability'
        combined_metrics['slide_accuracy'] = slide_binary_metrics['accuracy']
        combined_metrics['slide_sensitivity'] = slide_binary_metrics['sensitivity']
        combined_metrics['slide_specificity'] = slide_binary_metrics['specificity']
        combined_metrics['slide_precision'] = (
            slide_binary_metrics['tp'] / (slide_binary_metrics['tp'] + slide_binary_metrics['fp'])
            if (slide_binary_metrics['tp'] + slide_binary_metrics['fp']) > 0 else 0.0
        )
        combined_metrics['slide_f1'] = slide_binary_metrics['f1']
        combined_metrics['slide_auroc'] = slide_binary_metrics['auc']
        combined_metrics['slide_f2'] = slide_binary_metrics['f2']
        combined_metrics['slide_balanced_acc'] = slide_binary_metrics['balanced_acc']
        combined_metrics['slide_threshold'] = slide_binary_metrics['threshold']
        combined_metrics['slide_confusion_matrix'] = {
            'tp': slide_binary_metrics['tp'], 'tn': slide_binary_metrics['tn'],
            'fp': slide_binary_metrics['fp'], 'fn': slide_binary_metrics['fn'],
            'total': slide_binary_metrics['total'],
        }
        combined_metrics['slide_sensitivity_ci95'] = slide_binary_metrics['sensitivity_ci95']
        combined_metrics['slide_specificity_ci95'] = slide_binary_metrics['specificity_ci95']
        combined_metrics['slide_oracle_test_best_f1'] = slide_binary_metrics['oracle_test_best_f1']
        combined_metrics['slide_oracle_test_best_balanced_accuracy'] = slide_binary_metrics['oracle_test_best_balanced_accuracy']
        combined_metrics['slide_adaptive_applied'] = slide_binary_metrics['adaptive_applied']
        combined_metrics['slide_at_validation_threshold'] = slide_binary_metrics['at_validation_threshold']
        logging.info(
            "\nSlide-level binary metrics (shared MIL/graph protocol, slide score = mean tile probability; threshold=%.4f%s):",
            slide_binary_metrics['threshold'],
            " [adaptively recalibrated]" if abs(slide_binary_metrics['threshold'] - float(mean_prob_cfg.get('threshold', 0.5))) > 1e-9 else "",
        )
        logging.info(
            "  Accuracy: %.4f | Sensitivity: %.4f (95%% CI %.4f-%.4f) | Specificity: %.4f (95%% CI %.4f-%.4f)",
            slide_binary_metrics['accuracy'],
            slide_binary_metrics['sensitivity'], slide_binary_metrics['sensitivity_ci95'][0], slide_binary_metrics['sensitivity_ci95'][1],
            slide_binary_metrics['specificity'], slide_binary_metrics['specificity_ci95'][0], slide_binary_metrics['specificity_ci95'][1],
        )
        logging.info(
            "  F1: %.4f | F2 (recall weighted 4x precision): %.4f | Balanced Accuracy: %.4f | AUC: %.4f",
            slide_binary_metrics['f1'], slide_binary_metrics['f2'], slide_binary_metrics['balanced_acc'], slide_binary_metrics['auc'],
        )

        # Persist like evaluate_model_graph (<model>_<split>_predictions.csv / _slide_metrics.json) so
        # E1 numbers can be read from files instead of the log.
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            if save_predictions:
                df.to_csv(output_dir / f"{model_name}_{split}_predictions.csv", index=False)
                slide_mp.to_csv(output_dir / f"{model_name}_{split}_slide_predictions.csv", index=False)
            slide_metrics_out = {
                k: v for k, v in combined_metrics.items()
                if k.startswith('slide_') or k.startswith('validation_') or k.startswith('tile_')
            }
            with open(output_dir / f"{model_name}_{split}_slide_metrics.json", "w", encoding="utf-8") as f:
                json.dump(slide_metrics_out, f, indent=2, default=float)

    return combined_metrics


@register_evaluation_pipeline(('inflammation', 'cnn', 'multimodal'))
def evaluate_model_img_emb(
    model: nn.Module,
    data_loader: DataLoader,
    task: str,
    split: str,
    device: torch.device,
    output_dir: Optional[Path] = None,
    validation_thresholds: Optional[Dict] = None,
    save_predictions: bool = True
) -> Dict:
    """Evaluate an image+embedding model at appropriate hierarchical levels."""
    model.eval()
    predictions = []


    model_name = getattr(model, 'model_name', model.__class__.__name__)
    logging.info(f"Evaluating model: {model_name}")

    multiclass_detected = False

    with torch.no_grad():
        for images, embeddings, labels, metadata_batch in tqdm(data_loader, desc=f"Evaluating {split}"):
            images = images.to(device)
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            outputs = model(images, embeddings)

            if _is_multiclass_logits(outputs):
                multiclass_detected = True
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)

                batch_size = outputs.size(0)
                for i in range(batch_size):
                    metadata = {k: v[i] if isinstance(v, list) or isinstance(v, torch.Tensor) else v for k, v in metadata_batch.items()}
                    pred_class = int(preds[i].item())
                    pred_dict = {
                        'raw_pred': float(outputs[i, pred_class].detach().cpu().item()),
                        'prob': float(probs[i, pred_class].detach().cpu().item()),
                        'pred_class': pred_class,
                        'label': int(labels[i].item()),
                    }
                    for class_idx in range(outputs.size(1)):
                        pred_dict[f'prob_class_{class_idx}'] = float(probs[i, class_idx].detach().cpu().item())

                    for key in ['particle_id', 'slide_name', 'inflammation_status']:
                        if key in metadata:
                            pred_dict[key] = metadata[key]

                    if task == 'inflammation':
                        pred_dict['inflammation_type'] = metadata.get('inflammation_status', 'unknown')

                    predictions.append(pred_dict)
                continue

            outputs = outputs.squeeze()


            is_logits = False
            if outputs.numel() > 0:
                min_val = outputs.min().item()
                max_val = outputs.max().item()
                is_logits = min_val < 0 or max_val > 1
                probs = torch.sigmoid(outputs) if is_logits else outputs
            else:
                probs = outputs

            batch_size = outputs.shape[0] if outputs.dim() > 0 else 1
            for i in range(batch_size):
                metadata = {k: v[i] if isinstance(v, list) or isinstance(v, torch.Tensor) else v for k, v in metadata_batch.items()}

                pred_dict = {
                    'raw_pred': outputs[i].cpu().item() if batch_size > 1 else outputs.cpu().item(),
                    'prob': probs[i].cpu().item() if batch_size > 1 else probs.cpu().item(),
                    'label': labels[i].item() if batch_size > 1 else labels.item(),
                }


                for key in ['particle_id', 'slide_name', 'inflammation_status']:
                    if key in metadata:
                        pred_dict[key] = metadata[key]

                if task == 'inflammation':
                    pred_dict['inflammation_type'] = metadata.get('inflammation_status', 'unknown')

                predictions.append(pred_dict)

    df = pd.DataFrame(predictions)
    df = _normalize_prediction_df(df)

    if multiclass_detected and 'pred_class' in df.columns:
        return _return_multiclass_metrics(
            df=df,
            validation_thresholds=validation_thresholds,
            output_dir=output_dir,
            model_name=model_name,
            split=split,
            save_predictions=save_predictions,
        )


    logging.info(f"\nEvaluation Summary:")
    logging.info(f"Total samples: {len(df)}")
    logging.info(f"Unique slides: {df['slide_name'].nunique() if 'slide_name' in df else 'N/A'}")
    if 'slide_name' in df and 'particle_id' in df:
        logging.info(f"Unique particles: {_safe_unique_particles(df)}")

    pos_count = (df['label'] == 1).sum()
    neg_count = (df['label'] == 0).sum()
    logging.info(f"\nClass Distribution:")
    logging.info(f"Positive class: {pos_count} ({pos_count/len(df):.1%})")
    logging.info(f"Negative class: {neg_count} ({neg_count/len(df):.1%})")

    logging.info("\nPrediction Statistics:")
    logging.info(f"Raw logits mean: {df['raw_pred'].mean():.3f}")
    logging.info(f"Raw logits std: {df['raw_pred'].std():.3f}")
    logging.info(f"Raw logits min: {df['raw_pred'].min():.3f}")
    logging.info(f"Raw logits max: {df['raw_pred'].max():.3f}")



    if validation_thresholds is None:
        logging.warning(
            "No validation thresholds provided. Falling back to default thresholds/aggregation for evaluation only."
        )
        validation_thresholds = {}

    optimal_thresholds_dict = {k: v for k, v in validation_thresholds.items()
                             if k not in ['validation_tile_auroc', 'validation_slide_auroc',
                                         'validation_particle_auroc', 'optimal_aggregation']}


    logging.info("\nUsing validation-optimized thresholds:")
    for level, level_metrics in optimal_thresholds_dict.items():
        if isinstance(level_metrics, dict) and 'threshold' in level_metrics:
            logging.info(f"{level.capitalize()} level: {level_metrics['threshold']:.3f}")
            logging.info(f"  Sensitivity: {level_metrics.get('sensitivity', 0):.2f}")
            logging.info(f"  Specificity: {level_metrics.get('specificity', 0):.2f}")
            logging.info(f"  F1 Score: {level_metrics.get('f1', 0):.2f}")
        else:
            logging.info(f"{level.capitalize()} level: Invalid threshold format: {level_metrics}")


    for auc_key in ['validation_tile_auroc', 'validation_slide_auroc', 'validation_particle_auroc']:
        if auc_key in validation_thresholds:
            logging.info(f"{auc_key.replace('_', ' ').capitalize()}: {validation_thresholds[auc_key]:.4f}")

    agg_results = validation_thresholds.get("optimal_aggregation", {})
    if agg_results:
        if 'best_metrics' not in agg_results or not isinstance(agg_results['best_metrics'], dict):
            agg_results['best_metrics'] = {'threshold': 0.5}
        if 'best_strategy' not in agg_results:
            agg_results['best_strategy'] = 'mean'


        val_tp = agg_results['best_metrics'].get('tp', 0)
        val_tn = agg_results['best_metrics'].get('tn', 0)
        val_fp = agg_results['best_metrics'].get('fp', 0)
        val_fn = agg_results['best_metrics'].get('fn', 0)
        agg_results['validation_confusion_matrix'] = {
            'tp': val_tp,
            'tn': val_tn,
            'fp': val_fp,
            'fn': val_fn,
            'total': val_tp + val_tn + val_fp + val_fn
        }
        logging.info("\nValidation Confusion Matrix:")
        logging.info(f"  True Positives: {val_tp}")
        logging.info(f"  True Negatives: {val_tn}")
        logging.info(f"  False Positives: {val_fp}")
        logging.info(f"  False Negatives: {val_fn}")
        logging.info(f"  Total Samples: {val_tp + val_tn + val_fp + val_fn}")

        best_strategy = agg_results.get('best_strategy', 'mean')
        best_threshold = agg_results['best_metrics'].get('threshold', 0.5)


        if best_strategy == 'mean':
            agg_func = np.mean
        elif best_strategy == 'median':
            agg_func = np.median
        elif best_strategy.startswith('top_k_mean'):
            k_percent = int(best_strategy.split('_')[-1])
            agg_func = lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*k_percent/100)):]) if len(x) > 0 else 0
        else:
            logging.warning(f"Unknown aggregation strategy: {best_strategy}, defaulting to mean")
            agg_func = np.mean


        if 'prob' not in df.columns:
            if 'raw_pred' in df.columns:
                if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
                    df['prob'] = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
                else:
                    df['prob'] = df['raw_pred']
            else:
                df['prob'] = 0.5

        logging.info("Skipping any test-set-based prediction inversion to preserve unbiased evaluation.")

        if task == 'inflammation' and 'slide_name' in df.columns:
            slide_df = df.groupby('slide_name').agg({'prob': agg_func, 'label': 'first'}).reset_index()
            slide_df = _drop_non_finite_agg_rows(slide_df, prob_col='prob', name_col='slide_name')
            slide_preds = (slide_df['prob'] > best_threshold).astype(int)
            from sklearn.metrics import roc_auc_score, confusion_matrix

            try:
                optimal_agg_auc = roc_auc_score(slide_df['label'], slide_df['prob'])
            except Exception:
                optimal_agg_auc = float('nan')

            tn, fp, fn, tp = confusion_matrix(slide_df['label'], slide_preds, labels=[0, 1]).ravel()
            test_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            test_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            test_f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 0

            agg_results['test_confusion_matrix'] = {
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'total': int(tp + tn + fp + fn),
                'accuracy': float(test_accuracy),
                'sensitivity': float(test_sensitivity),
                'specificity': float(test_specificity),
                'f1': float(test_f1),
                'balanced_acc': float((test_sensitivity + test_specificity) / 2),
                'auc': float(optimal_agg_auc)
            }

            for key, value in agg_results['test_confusion_matrix'].items():
                agg_results['best_metrics'][key] = value

            logging.info("\nTest Confusion Matrix:")
            logging.info(f"  True Positives: {tp}")
            logging.info(f"  True Negatives: {tn}")
            logging.info(f"  False Positives: {fp}")
            logging.info(f"  False Negatives: {fn}")
            logging.info(f"  Total Samples: {tp + tn + fp + fn}")
            logging.info(f"  Accuracy: {test_accuracy:.4f}")
            logging.info(f"  F1 Score: {test_f1:.4f}")
            logging.info(f"  AUC: {optimal_agg_auc:.4f}")

        elif task == 'tissue' and {'slide_name', 'particle_id'}.issubset(df.columns):
            particle_df = df.groupby(['slide_name', 'particle_id']).agg({'prob': agg_func, 'label': 'first'}).reset_index()
            particle_df = _drop_non_finite_agg_rows(particle_df, prob_col='prob', name_col='slide_name')
            particle_preds = (particle_df['prob'] > best_threshold).astype(int)
            from sklearn.metrics import roc_auc_score, confusion_matrix

            try:
                optimal_agg_auc = roc_auc_score(particle_df['label'], particle_df['prob'])
            except Exception:
                optimal_agg_auc = float('nan')

            tn, fp, fn, tp = confusion_matrix(particle_df['label'], particle_preds, labels=[0, 1]).ravel()
            test_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            test_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            test_f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 0

            agg_results['test_confusion_matrix'] = {
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'total': int(tp + tn + fp + fn),
                'accuracy': float(test_accuracy),
                'sensitivity': float(test_sensitivity),
                'specificity': float(test_specificity),
                'f1': float(test_f1),
                'balanced_acc': float((test_sensitivity + test_specificity) / 2),
                'auc': float(optimal_agg_auc)
            }

            for key, value in agg_results['test_confusion_matrix'].items():
                agg_results['best_metrics'][key] = value

            logging.info("\nTest Confusion Matrix:")
            logging.info(f"  True Positives: {tp}")
            logging.info(f"  True Negatives: {tn}")
            logging.info(f"  False Positives: {fp}")
            logging.info(f"  False Negatives: {fn}")
            logging.info(f"  Total Samples: {tp + tn + fp + fn}")
            logging.info(f"  Accuracy: {test_accuracy:.4f}")
            logging.info(f"  F1 Score: {test_f1:.4f}")
            logging.info(f"  AUC: {optimal_agg_auc:.4f}")
    else:
        logging.info("\nNo pre-computed aggregation strategy found in validation thresholds.")
        agg_results = {"best_strategy": "mean", "best_metrics": {"threshold": 0.5}}
        logging.info("Default aggregation strategy 'mean' with threshold 0.5 applied.")


    best_strategy = agg_results.get('best_strategy', 'mean')
    best_metrics = agg_results.get('best_metrics', {})
    logging.info(f"Aggregation strategy used: {best_strategy}")
    logging.info(f"Threshold used: {best_metrics.get('threshold', 0.5):.3f}")
    logging.info(f"Sensitivity: {best_metrics.get('sensitivity', 0):.2f}")
    logging.info(f"Specificity: {best_metrics.get('specificity', 0):.2f}")
    logging.info(f"F1 Score: {best_metrics.get('f1', 0):.2f}")


    if output_dir:
        probs = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()

        plt.figure(figsize=(16, 7))
        plt.subplot(1, 2, 1)
        plt.hist(df['raw_pred'][df['label']==1], bins=50, alpha=0.6, label='Positive class', color='blue')
        plt.hist(df['raw_pred'][df['label']==0], bins=50, alpha=0.6, label='Negative class', color='orange')
        plt.axvline(x=0.0, color='r', linestyle='--', label='Default threshold (logit=0)')

        if task == 'inflammation' and 'tile' in optimal_thresholds_dict:
            prob_threshold = optimal_thresholds_dict['tile']['threshold']
            logit_threshold = np.log(prob_threshold/(1-prob_threshold)) if 0 < prob_threshold < 1 else (10 if prob_threshold >= 1 else -10)
            plt.axvline(x=logit_threshold, color='g', linestyle='--', label=f'Validation tile threshold (logit={logit_threshold:.2f})')
        elif task == 'tissue' and 'particle' in optimal_thresholds_dict:
            prob_threshold = optimal_thresholds_dict['particle']['threshold']
            logit_threshold = np.log(prob_threshold/(1-prob_threshold)) if 0 < prob_threshold < 1 else (10 if prob_threshold >= 1 else -10)
            plt.axvline(x=logit_threshold, color='g', linestyle='--', label=f'Optimal threshold (logit={logit_threshold:.2f})')

        plt.title('Raw Logits Distribution by Class')
        plt.xlabel('Logit Value')
        plt.ylabel('Count')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(probs[df['label']==1], bins=50, alpha=0.6, label='Positive class', color='blue')
        plt.hist(probs[df['label']==0], bins=50, alpha=0.6, label='Negative class', color='orange')
        plt.axvline(x=0.5, color='r', linestyle='--', label='Default threshold (prob=0.5)')

        if task == 'inflammation' and 'tile' in optimal_thresholds_dict:
            plt.axvline(x=optimal_thresholds_dict['tile']['threshold'], color='g', linestyle='--', label=f'Validation tile threshold (prob={optimal_thresholds_dict["tile"]["threshold"]:.2f})')
        elif task == 'tissue' and 'particle' in optimal_thresholds_dict:
            plt.axvline(x=optimal_thresholds_dict['particle']['threshold'], color='g', linestyle='--', label=f'Optimal threshold (prob={optimal_thresholds_dict["particle"]["threshold"]:.2f})')

        if 'threshold' in best_metrics:
            plt.axvline(x=best_metrics['threshold'], color='purple', linestyle='-.', label=f'Best agg. threshold (prob={best_metrics["threshold"]:.2f})')

        plt.title('Probability Distribution by Class')
        plt.xlabel('Probability')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f'pred_distribution_{split}.png', dpi=300)
        plt.close()

        try:
            from src.utils.visualization_utils import plot_classification_diagnostics
            diagnostic_stats = {
                'tile': plot_classification_diagnostics(
                    df['label'].values, probs, output_dir,
                    prefix=f"{model_name}_{split}_tile", level_name="Tile-level",
                    threshold=_diagnostic_threshold(validation_thresholds, 'tile'),
                )
            }
            if 'slide_df' in locals():
                diagnostic_stats['slide'] = plot_classification_diagnostics(
                    slide_df['label'].values, slide_df['prob'].values, output_dir,
                    prefix=f"{model_name}_{split}_slide", level_name="Slide-level",
                    threshold=_diagnostic_threshold(validation_thresholds, 'slide_mean_prob', 'slide'),
                )
            elif 'particle_df' in locals():
                diagnostic_stats['particle'] = plot_classification_diagnostics(
                    particle_df['label'].values, particle_df['prob'].values, output_dir,
                    prefix=f"{model_name}_{split}_particle", level_name="Particle-level",
                    threshold=_diagnostic_threshold(validation_thresholds, 'particle'),
                )
            agg_results['test_diagnostic_plots'] = diagnostic_stats
        except Exception as exc:
            logging.warning(f"Could not generate classification diagnostic plots for split '{split}': {exc}")


    if task == 'inflammation' and 'slide' in optimal_thresholds_dict:
        selected_threshold = optimal_thresholds_dict['slide']['threshold']
        if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
            threshold_for_metrics = np.log(selected_threshold/(1-selected_threshold)) if 0 < selected_threshold < 1 else 0.0
        else:
            threshold_for_metrics = selected_threshold
    elif task == 'tissue' and 'particle' in optimal_thresholds_dict:
        selected_threshold = optimal_thresholds_dict['particle']['threshold']
        if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
            threshold_for_metrics = np.log(selected_threshold/(1-selected_threshold)) if 0 < selected_threshold < 1 else 0.0
        else:
            threshold_for_metrics = selected_threshold
    else:
        threshold_for_metrics = 0.0 if (df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1) else 0.5

    metrics = metrics_utils.calculate_hierarchical_metrics(
        df=df,
        split=split,
        model_name=model_name,
        threshold=threshold_for_metrics
    )

    combined_metrics = {
        **metrics,
        'optimal_thresholds': optimal_thresholds_dict,
        'optimal_aggregation': agg_results,
    }


    for auc_key in ['validation_tile_auroc', 'validation_slide_auroc', 'validation_particle_auroc']:
        if auc_key in validation_thresholds:
            combined_metrics[auc_key] = validation_thresholds[auc_key]

    return combined_metrics


@register_evaluation_pipeline(('inflammation', 'mil', 'multimodal'))
def evaluate_model_mil_multimodal(
    model: nn.Module,
    data_loader: DataLoader,
    task: str,
    split: str,
    device: torch.device,
    output_dir: Optional[Path] = None,
    validation_thresholds: Optional[Dict] = None,
    save_predictions: bool = True,
    save_plots: bool = True,
) -> Dict:
    """Evaluate a multimodal (image + embedding) MIL model."""
    model.eval()
    predictions = []

    model_name = getattr(model, 'model_name', model.__class__.__name__)
    logging.info(f"Evaluating model: {model_name}")

    global_idx = 0
    multiclass_detected = False

    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc=f"Evaluating {split}"):

            if len(batch_data) == 4:
                images_flat, bag_idx, labels, embeds_flat = batch_data
                images_flat = images_flat.to(device)
                bag_idx = bag_idx.to(device)
                labels = labels.to(device)
                embeds_flat = embeds_flat.to(device)


                outputs, _ = model(images_flat, bag_idx, embeds_flat)
                if _is_multiclass_logits(outputs):
                    multiclass_detected = True
                    probs = torch.softmax(outputs, dim=1)
                    pred_classes = torch.argmax(outputs, dim=1)

                    batch_size = labels.shape[0]
                    for i in range(batch_size):
                        pred_class = int(pred_classes[i].item())
                        pred_dict = {
                            'raw_pred': float(outputs[i, pred_class].detach().cpu().item()),
                            'prob': float(probs[i, pred_class].detach().cpu().item()),
                            'pred_class': pred_class,
                            'label': int(labels[i].item()),
                            'slide_name': f'bag_{global_idx}',
                            'particle_id': 0,
                        }
                        for class_idx in range(outputs.size(1)):
                            pred_dict[f'prob_class_{class_idx}'] = float(probs[i, class_idx].detach().cpu().item())
                        if task == 'inflammation':
                            pred_dict['inflammation_type'] = 'unknown'
                        predictions.append(pred_dict)
                        global_idx += 1
                    continue

                probs = torch.sigmoid(outputs)


                batch_size = labels.shape[0]
                for i in range(batch_size):
                    raw_pred = outputs[i].item() if outputs.ndim > 0 else outputs.item()
                    prob = probs[i].item() if probs.ndim > 0 else probs.item()
                    label_val = labels[i].item() if labels.ndim > 0 else labels.item()

                    pred_dict = {
                        'raw_pred': float(raw_pred),
                        'prob': float(prob),
                        'label': int(label_val),
                        'slide_name': f'bag_{global_idx}',
                        'particle_id': 0,
                    }
                    if task == 'inflammation':
                        pred_dict['inflammation_type'] = 'unknown'
                    predictions.append(pred_dict)
                    global_idx += 1
            else:
                raise ValueError(f"Unexpected batch format: {len(batch_data)} elements (expected 4)")

    df = pd.DataFrame(predictions)
    df = _normalize_prediction_df(df)

    if multiclass_detected and 'pred_class' in df.columns:
        return _return_multiclass_metrics(
            df=df,
            validation_thresholds=validation_thresholds,
            output_dir=output_dir,
            model_name=model_name,
            split=split,
            save_predictions=save_predictions,
            save_plots=save_plots,
        )

    logging.info(f"\nEvaluation Summary:")
    logging.info(f"Total samples: {len(df)}")
    logging.info(f"Unique slides: {df['slide_name'].nunique() if 'slide_name' in df else 'N/A'}")
    if 'slide_name' in df and 'particle_id' in df:
        logging.info(f"Unique particles: {_safe_unique_particles(df)}")

    pos_count = (df['label'] == 1).sum()
    neg_count = (df['label'] == 0).sum()
    logging.info(f"\nClass Distribution:")
    logging.info(f"Positive class: {pos_count} ({pos_count/len(df):.1%})")
    logging.info(f"Negative class: {neg_count} ({neg_count/len(df):.1%})")

    logging.info("\nPrediction Statistics:")
    logging.info(f"Raw logits mean: {df['raw_pred'].mean():.3f}")
    logging.info(f"Raw logits std: {df['raw_pred'].std():.3f}")
    logging.info(f"Raw logits min: {df['raw_pred'].min():.3f}")
    logging.info(f"Raw logits max: {df['raw_pred'].max():.3f}")



    if validation_thresholds is None:
        logging.warning(
            "No validation thresholds provided. Falling back to default thresholds/aggregation for evaluation only."
        )
        validation_thresholds = {}


    optimal_thresholds_dict = {k: v for k, v in validation_thresholds.items()
                             if k not in ['validation_tile_auroc', 'validation_slide_auroc',
                                        'validation_particle_auroc', 'optimal_aggregation']}


    logging.info("\nUsing validation-optimized thresholds:")
    for level, level_metrics in optimal_thresholds_dict.items():
        if isinstance(level_metrics, dict) and 'threshold' in level_metrics:
            logging.info(f"{level.capitalize()} level: {level_metrics['threshold']:.3f}")
            logging.info(f"  Sensitivity: {level_metrics.get('sensitivity', 0):.2f}")
            logging.info(f"  Specificity: {level_metrics.get('specificity', 0):.2f}")
            logging.info(f"  F1 Score: {level_metrics.get('f1', 0):.2f}")


    for auc_key in ['validation_tile_auroc', 'validation_slide_auroc', 'validation_particle_auroc']:
        if auc_key in validation_thresholds:
            logging.info(f"{auc_key.replace('_', ' ').capitalize()}: {validation_thresholds[auc_key]:.4f}")


    if 'prob' not in df.columns:
        if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
            logging.info("Converting raw logits to probabilities with sigmoid...")
            df['prob'] = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
        else:
            logging.info("Raw predictions are already in probability range [0,1]")
            df['prob'] = df['raw_pred']

    logging.info("Skipping any test-set-based prediction inversion to preserve unbiased evaluation.")


    agg_results = validation_thresholds.get("optimal_aggregation", {})
    if agg_results:
        if 'best_metrics' not in agg_results or not isinstance(agg_results['best_metrics'], dict):
            agg_results['best_metrics'] = {'threshold': 0.5}
        if 'best_strategy' not in agg_results:
            agg_results['best_strategy'] = 'mean'

        val_tp = agg_results['best_metrics'].get('tp', 0)
        val_tn = agg_results['best_metrics'].get('tn', 0)
        val_fp = agg_results['best_metrics'].get('fp', 0)
        val_fn = agg_results['best_metrics'].get('fn', 0)

        agg_results['validation_confusion_matrix'] = {
            'tp': val_tp,
            'tn': val_tn,
            'fp': val_fp,
            'fn': val_fn,
            'total': val_tp + val_tn + val_fp + val_fn
        }

        logging.info("\nValidation Confusion Matrix:")
        logging.info(f"  True Positives: {val_tp}")
        logging.info(f"  True Negatives: {val_tn}")
        logging.info(f"  False Positives: {val_fp}")
        logging.info(f"  False Negatives: {val_fn}")
        logging.info(f"  Total Samples: {val_tp + val_tn + val_fp + val_fn}")

        best_strategy = agg_results.get('best_strategy', 'mean')
        best_threshold = agg_results['best_metrics'].get('threshold', 0.5)


        if best_strategy == 'mean':
            agg_func = np.mean
        elif best_strategy == 'median':
            agg_func = np.median
        elif best_strategy.startswith('top_k_mean'):
            k_percent = int(best_strategy.split('_')[-1])
            agg_func = lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*k_percent/100)):]) if len(x) > 0 else 0
        else:
            logging.warning(f"Unknown aggregation strategy: {best_strategy}, defaulting to mean")
            agg_func = np.mean


        if task == 'inflammation' and 'slide_name' in df.columns:
            slide_df = df.groupby('slide_name').agg({'prob': agg_func, 'label': 'first'}).reset_index()
            slide_df = _drop_non_finite_agg_rows(slide_df, prob_col='prob', name_col='slide_name')
            slide_preds = (slide_df['prob'] > best_threshold).astype(int)

            try:
                optimal_agg_auc = roc_auc_score(slide_df['label'], slide_df['prob'])
            except Exception:
                optimal_agg_auc = float('nan')

            tn, fp, fn, tp = confusion_matrix(slide_df['label'], slide_preds, labels=[0, 1]).ravel()
            test_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            test_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            test_f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 0

            agg_results['test_confusion_matrix'] = {
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'total': int(tp + tn + fp + fn),
                'accuracy': float(test_accuracy),
                'sensitivity': float(test_sensitivity),
                'specificity': float(test_specificity),
                'f1': float(test_f1),
                'balanced_acc': float((test_sensitivity + test_specificity) / 2),
                'auc': float(optimal_agg_auc)
            }

            for key, value in agg_results['test_confusion_matrix'].items():
                agg_results['best_metrics'][key] = value

            logging.info("\nTest Confusion Matrix (using optimal aggregation):")
            logging.info(f"  True Positives: {tp}")
            logging.info(f"  True Negatives: {tn}")
            logging.info(f"  False Positives: {fp}")
            logging.info(f"  False Negatives: {fn}")
            logging.info(f"  Total Samples: {tp + tn + fp + fn}")
            logging.info(f"  Accuracy: {test_accuracy:.4f}")
            logging.info(f"  F1 Score: {test_f1:.4f}")
            logging.info(f"  AUC: {optimal_agg_auc:.4f}")

        elif task == 'tissue' and {'slide_name', 'particle_id'}.issubset(df.columns):
            particle_df = df.groupby(['slide_name', 'particle_id']).agg({'prob': agg_func, 'label': 'first'}).reset_index()
            particle_df = _drop_non_finite_agg_rows(particle_df, prob_col='prob', name_col='slide_name')
            particle_preds = (particle_df['prob'] > best_threshold).astype(int)

            try:
                optimal_agg_auc = roc_auc_score(particle_df['label'], particle_df['prob'])
            except Exception:
                optimal_agg_auc = float('nan')

            tn, fp, fn, tp = confusion_matrix(particle_df['label'], particle_preds, labels=[0, 1]).ravel()
            test_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            test_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            test_f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 0

            agg_results['test_confusion_matrix'] = {
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'total': int(tp + tn + fp + fn),
                'accuracy': float(test_accuracy),
                'sensitivity': float(test_sensitivity),
                'specificity': float(test_specificity),
                'f1': float(test_f1),
                'balanced_acc': float((test_sensitivity + test_specificity) / 2),
                'auc': float(optimal_agg_auc)
            }

            for key, value in agg_results['test_confusion_matrix'].items():
                agg_results['best_metrics'][key] = value

            logging.info("\nTest Confusion Matrix (using optimal aggregation):")
            logging.info(f"  True Positives: {tp}")
            logging.info(f"  True Negatives: {tn}")
            logging.info(f"  False Positives: {fp}")
            logging.info(f"  False Negatives: {fn}")
            logging.info(f"  Total Samples: {tp + tn + fp + fn}")
            logging.info(f"  Accuracy: {test_accuracy:.4f}")
            logging.info(f"  F1 Score: {test_f1:.4f}")
            logging.info(f"  AUC: {optimal_agg_auc:.4f}")


    if output_dir and save_plots:
        if not region_df.empty and str(split).startswith('test'):
            region_threshold = 0.5
            if isinstance(validation_thresholds, dict):
                region_threshold = float((validation_thresholds.get('region') or {}).get('threshold', 0.5))
            _plot_test_region_predictions(
                region_df=region_df,
                threshold=region_threshold,
                output_dir=output_dir,
                model_name=model_name,
                split=split,
            )

        probs = df['prob'].values

        plt.figure(figsize=(16, 7))


        plt.subplot(1, 2, 1)
        plt.hist(df['raw_pred'][df['label']==1], bins=50, alpha=0.6, label='Positive class', color='blue')
        plt.hist(df['raw_pred'][df['label']==0], bins=50, alpha=0.6, label='Negative class', color='orange')
        plt.axvline(x=0.0, color='r', linestyle='--', label='Default threshold (logit=0)')

        if task == 'inflammation' and 'slide' in optimal_thresholds_dict:
            prob_threshold = optimal_thresholds_dict['slide']['threshold']
            logit_threshold = np.log(prob_threshold/(1-prob_threshold)) if 0 < prob_threshold < 1 else (10 if prob_threshold >= 1 else -10)
            plt.axvline(x=logit_threshold, color='g', linestyle='--', label=f'Optimal threshold (logit={logit_threshold:.2f})')
        elif task == 'tissue' and 'particle' in optimal_thresholds_dict:
            prob_threshold = optimal_thresholds_dict['particle']['threshold']
            logit_threshold = np.log(prob_threshold/(1-prob_threshold)) if 0 < prob_threshold < 1 else (10 if prob_threshold >= 1 else -10)
            plt.axvline(x=logit_threshold, color='g', linestyle='--', label=f'Optimal threshold (logit={logit_threshold:.2f})')

        plt.title('Raw Logits Distribution by Class')
        plt.xlabel('Logit Value')
        plt.ylabel('Count')
        plt.legend()


        plt.subplot(1, 2, 2)
        plt.hist(probs[df['label']==1], bins=50, alpha=0.6, label='Positive class', color='blue')
        plt.hist(probs[df['label']==0], bins=50, alpha=0.6, label='Negative class', color='orange')
        plt.axvline(x=0.5, color='r', linestyle='--', label='Default threshold (prob=0.5)')

        if task == 'inflammation' and 'slide' in optimal_thresholds_dict:
            plt.axvline(x=optimal_thresholds_dict['slide']['threshold'], color='g', linestyle='--',
                       label=f'Optimal threshold (prob={optimal_thresholds_dict["slide"]["threshold"]:.2f})')
        elif task == 'tissue' and 'particle' in optimal_thresholds_dict:
            plt.axvline(x=optimal_thresholds_dict['particle']['threshold'], color='g', linestyle='--',
                       label=f'Optimal threshold (prob={optimal_thresholds_dict["particle"]["threshold"]:.2f})')

        plt.title('Probability Distribution by Class')
        plt.xlabel('Probability')
        plt.ylabel('Count')
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_dir / f'pred_distribution_{split}.png', dpi=300)
        plt.close()

        try:
            from src.utils.visualization_utils import plot_classification_diagnostics
            diagnostic_stats = {}
            if 'slide_df' in locals():
                diagnostic_stats['slide'] = plot_classification_diagnostics(
                    slide_df['label'].values, slide_df['prob'].values, output_dir,
                    prefix=f"{model_name}_{split}_slide", level_name="Slide-level",
                    threshold=_diagnostic_threshold(validation_thresholds, 'slide_mean_prob', 'slide'),
                )
            elif 'particle_df' in locals():
                diagnostic_stats['particle'] = plot_classification_diagnostics(
                    particle_df['label'].values, particle_df['prob'].values, output_dir,
                    prefix=f"{model_name}_{split}_particle", level_name="Particle-level",
                    threshold=_diagnostic_threshold(validation_thresholds, 'particle'),
                )
            agg_results['test_diagnostic_plots'] = diagnostic_stats
        except Exception as exc:
            logging.warning(f"Could not generate classification diagnostic plots for split '{split}': {exc}")

    if output_dir and save_predictions:
        df.to_csv(output_dir / f"{model_name}_{split}_predictions.csv", index=False)

    if task == 'inflammation' and 'slide' in optimal_thresholds_dict:
        selected_threshold = optimal_thresholds_dict['slide']['threshold']
        if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
            threshold_for_metrics = np.log(selected_threshold/(1-selected_threshold)) if 0 < selected_threshold < 1 else 0.0
        else:
            threshold_for_metrics = selected_threshold
    elif task == 'tissue' and 'particle' in optimal_thresholds_dict:
        selected_threshold = optimal_thresholds_dict['particle']['threshold']
        if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
            threshold_for_metrics = np.log(selected_threshold/(1-selected_threshold)) if 0 < selected_threshold < 1 else 0.0
        else:
            threshold_for_metrics = selected_threshold
    else:
        threshold_for_metrics = 0.0 if (df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1) else 0.5

    metrics = metrics_utils.calculate_hierarchical_metrics(
        df=df,
        split=split,
        model_name=model_name,
        threshold=threshold_for_metrics
    )


    combined_metrics = {
        **metrics,
        **validation_thresholds,
        'optimal_thresholds': optimal_thresholds_dict,
    }

    if agg_results:
        combined_metrics['optimal_aggregation'] = agg_results


    for auc_key in ['validation_tile_auroc', 'validation_slide_auroc', 'validation_particle_auroc']:
        if auc_key in validation_thresholds:
            combined_metrics[auc_key] = validation_thresholds[auc_key]

    return combined_metrics


def _monitor_tag(monitor_metric: Optional[str], *keywords: str) -> str:
    if not monitor_metric:
        return ""
    normalized = str(monitor_metric).strip().lower()
    if any(f"{kw}_cls" in normalized for kw in keywords):
        return ""
    return "  <-- MONITOR METRIC (training checkpoint selection)" if all(kw in normalized for kw in keywords) else ""


@register_evaluation_pipeline(('inflammation', 'gnn', 'graph'))
def evaluate_model_graph(
    model: nn.Module,
    data_loader: DataLoader,
    task: str,
    split: str,
    device: torch.device,
    output_dir: Optional[Path] = None,
    validation_thresholds: Optional[Dict] = None,
    save_predictions: bool = True,
    save_plots: bool = True,
    negative_label_source: str = "annotation_only",
    monitor_metric: Optional[str] = None,
    label_field: Optional[str] = None,
    is_multiclass_only_run: bool = False,
    use_mixed_precision: bool = True,
    class_decision_bias: Optional[Dict[str, float]] = None,
    exclude_label_field_values: Optional[List[str]] = None,
    slide_groups: Optional[pd.DataFrame] = None,
) -> Dict:
    """Evaluate a graph model at appropriate hierarchical levels, mirroring the image-based evaluators.

    slide_groups: optional per-slide metadata (slide_name + stain, typically the test split CSV) --
    when given, a per-stain overview of correctly classified slides is logged/saved/plotted, at the
    same threshold as the overall slide metrics.

    class_decision_bias: optional {class_name: multiplier} applied to softmax probabilities
    (unbiased probabilities are still what's logged/saved) before argmax, for the multiclass
    slide-level prediction only. Post-hoc, eval-time only -- no retraining needed. Lets you trade
    precision for recall on a specific class without touching the loss/training run at all: e.g.
    CAMELYON subtype's itc consistently comes out precision~1.0/recall very low (the model only
    commits to itc when extremely confident), so {"itc": 1.5} tips borderline cases toward itc
    that argmax on raw probabilities would otherwise assign to whichever class narrowly edged it
    out (usually negative).
    """
    model.eval()
    predictions = []
    test_confusion_matrix = {}

    model_name = getattr(model, 'model_name', model.__class__.__name__)
    logging.info(f"Evaluating model: {model_name}")

    global_idx = 0
    region_row_offset = 0
    patch_row_offset = 0
    region_cls_row_offset = 0
    patch_cls_row_offset = 0
    multiclass_detected = False
    # Real class names for a multiclass SLIDE target (e.g. CAMELYON subtype), captured once from
    # the first graph's own data.meta[f"{label_field}_label_vocab"] -- see _return_multiclass_
    # metrics' class_names parameter. Stays [] (falls back to generic class_0/class_1/... names)
    # when label_field wasn't passed in or the graph has no matching vocab.
    slide_cls_class_names: list = []
    region_eval_rows: list[dict] = []
    patch_eval_rows: list[dict] = []
    # TissueRegionGraphModel's optional multiclass region_cls_target_field/patch_cls_target_field
    # head (e.g. GSDB tissue_type) -- see _extract_region_cls_eval_rows/_extract_patch_cls_eval_
    # rows. Stay empty (all downstream metrics/logging/file-writing become no-ops) for any
    # checkpoint where this head isn't enabled -- every existing binary-only run is unaffected.
    region_cls_eval_rows: list[dict] = []
    patch_cls_eval_rows: list[dict] = []

    with torch.no_grad():
        for data_cpu in tqdm(data_loader, desc=f"Evaluating {split}"):
            # use_mixed_precision mirrors the run's own config['mixed_precision'] (see Holder.test
            # threading it through) instead of always autocasting on CUDA regardless of that
            # setting -- previously eval used fp16 autocast even for runs trained with
            # mixed_precision: false, causing avoidable fp16-overflow retries below (GATv2Conv
            # attention exponentials + large pov-gigapath feature magnitudes) on exactly the
            # numerically borderline graphs.
            used_autocast = False
            try:
                data = data_cpu.to(device)
                if device.type == 'cuda' and use_mixed_precision:
                    used_autocast = True
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        model_out = model(data)
                else:
                    model_out = model(data)
            except torch.OutOfMemoryError:
                logging.warning(
                    "CUDA OOM evaluating one graph on %s (likely transient memory pressure from "
                    "another process on this shared GPU) -- clearing cache and retrying this "
                    "graph on CPU only; will resume using %s for subsequent graphs.",
                    device, device,
                )
                gc.collect()
                torch.cuda.empty_cache()
                model_was_on_gpu = next(model.parameters()).is_cuda
                if model_was_on_gpu:
                    model = model.cpu()
                data = data_cpu.cpu()
                model_out = model(data)
                if model_was_on_gpu:
                    model = model.to(device)
                    torch.cuda.empty_cache()
                # This CPU recovery path already ran without autocast (full fp32), so it isn't a
                # candidate for the fp16-autocast-overflow retry below -- and `data`/`model` no
                # longer agree on device by this point (model was just moved back to `device`
                # above) so re-invoking model(data) here would raise a device-mismatch error.
                used_autocast = False
            outputs = _unwrap_graph_logits(model_out)

            if used_autocast and not torch.isfinite(outputs).all():
                # fp16 autocast overflow (GATv2Conv attention exponentials + large-magnitude
                # pov-gigapath features occasionally blow past fp16's ~65504 max) yields NaN/Inf
                # logits for a small fraction of graphs. train.py already treats this as a known,
                # accepted fp16 failure mode and skips the step (see its `not torch.isfinite
                # (total_step_loss): continue` guards); at eval time there is no loss to skip, so
                # instead recompute this same batch in fp32 (no autocast) to recover a real,
                # finite prediction rather than silently emitting NaN into slide/region/patch
                # predictions (which previously got miscounted as "predicted negative" downstream
                # and poisoned roc_auc_score into a fake AUC=0.5 fallback).
                logging.warning(
                    "Non-finite logits from fp16 autocast forward on %s -- retrying this batch "
                    "in fp32 (no autocast) to recover finite predictions; will resume using "
                    "fp16 autocast for subsequent batches.",
                    device,
                )
                model_out = model(data)
                outputs = _unwrap_graph_logits(model_out)
                if not torch.isfinite(outputs).all():
                    logging.warning(
                        "Still non-finite after fp32 retry (likely a genuinely degenerate graph, "
                        "not an autocast artifact) -- affected slide(s) will be excluded from "
                        "slide-level AUC/confusion-matrix metrics below, with an explicit count "
                        "logged rather than silently treated as a negative prediction."
                    )

            batch_region_rows, region_row_offset = _extract_region_eval_rows(
                model_out,
                data,
                row_offset=region_row_offset,
                negative_label_source=negative_label_source,
            )
            if batch_region_rows:
                region_eval_rows.extend(batch_region_rows)

            batch_patch_rows, patch_row_offset = _extract_patch_eval_rows(
                model_out,
                data,
                row_offset=patch_row_offset,
                negative_label_source=negative_label_source,
            )
            if batch_patch_rows:
                patch_eval_rows.extend(batch_patch_rows)

            batch_region_cls_rows, region_cls_row_offset = _extract_region_cls_eval_rows(
                model, model_out, data, row_offset=region_cls_row_offset,
            )
            if batch_region_cls_rows:
                region_cls_eval_rows.extend(batch_region_cls_rows)

            batch_patch_cls_rows, patch_cls_row_offset = _extract_patch_cls_eval_rows(
                model, model_out, data, row_offset=patch_cls_row_offset,
            )
            if batch_patch_cls_rows:
                patch_cls_eval_rows.extend(batch_patch_cls_rows)

            if _is_multiclass_logits(outputs):
                multiclass_detected = True
                if not slide_cls_class_names and label_field:
                    # Drop the graph-build pipeline's trailing "unknown" catch-all (see
                    # resolve_label_fields, src/graph_creation_new/helper/labels.py): any slide
                    # resolving to it is already excluded from the dataset entirely (see
                    # WSICachedGraphDataset._resolve_label_field_is_unknown), so num_classes/the
                    # model's actual output width no longer has a slot for it either -- without
                    # this filter, len(slide_cls_class_names) (still 5, straight from the graph's
                    # own vocab) would mismatch num_classes (4) and _return_multiclass_metrics
                    # would silently fall back to generic "class_0".."class_3" names instead of
                    # itc/macro/micro/negative. Also drop exclude_label_field_values (same set the
                    # dataset itself was filtered on, e.g. {"negative"} for a tumor-positive-only
                    # 3-way itc/macro/micro head) for the same reason -- those classes have no
                    # training examples and no output slot either once excluded at dataset level.
                    excluded_names = {v.lower() for v in (exclude_label_field_values or [])} | {"unknown"}
                    slide_cls_class_names = [
                        name for name in _extract_meta_vocab_list(data, f"{label_field}_label_vocab")
                        if name.lower() not in excluded_names
                    ]
                probs = torch.softmax(outputs, dim=1)
                if class_decision_bias and slide_cls_class_names:
                    bias_vec = torch.tensor(
                        [float(class_decision_bias.get(name, 1.0)) for name in slide_cls_class_names],
                        device=probs.device, dtype=probs.dtype,
                    )
                    pred_classes = torch.argmax(probs * bias_vec, dim=1)
                else:
                    pred_classes = torch.argmax(outputs, dim=1)
                labels = data.y.view(-1).detach().cpu().numpy().flatten()
                batch_slide_names = _resolve_slide_names(data) if getattr(data, 'slide_name', None) is not None else []

                for i in range(outputs.size(0)):
                    pred_class = int(pred_classes[i].item())
                    pred_dict = {
                        "raw_pred": float(outputs[i, pred_class].detach().cpu().item()),
                        "prob": float(probs[i, pred_class].detach().cpu().item()),
                        "pred_class": pred_class,
                        "label": int(labels[i]),
                        "slide_name": _batch_slide_name(batch_slide_names, i, global_idx),
                        "particle_id": 0,
                    }
                    for class_idx in range(outputs.size(1)):
                        pred_dict[f'prob_class_{class_idx}'] = float(probs[i, class_idx].detach().cpu().item())
                    if task == "inflammation":
                        pred_dict["inflammation_type"] = "unknown"
                    predictions.append(pred_dict)
                    global_idx += 1
                continue

            probs = torch.sigmoid(outputs)


            if outputs.ndim == 0 or outputs.numel() == 1:

                raw_preds = [outputs.item()]
                probs_list = [probs.item()]
                labels = [data.y.item()]
            else:

                raw_preds = outputs.squeeze().cpu().numpy().flatten()
                probs_list = probs.squeeze().cpu().numpy().flatten()
                labels = data.y.squeeze().cpu().numpy().flatten()
            batch_slide_names = _resolve_slide_names(data) if getattr(data, 'slide_name', None) is not None else []

            for i in range(len(raw_preds)):
                pred_dict = {
                    "raw_pred": float(raw_preds[i]),
                    "prob": float(probs_list[i]),
                    "label": int(labels[i]),
                    "slide_name": _batch_slide_name(batch_slide_names, i, global_idx),
                    "particle_id": 0,
                }
                if task == "inflammation":
                    pred_dict["inflammation_type"] = "unknown"
                predictions.append(pred_dict)
                global_idx += 1

    df = pd.DataFrame(predictions)
    df = _normalize_prediction_df(df)
    region_df = pd.DataFrame(region_eval_rows)
    region_metrics = _compute_region_binary_metrics(region_df)
    # Hoisted above the is_multiclass_only_run guard below -- patch_df/patch_metrics are read
    # later (combined_metrics, the output_dir CSV/JSON saves) regardless of whether the binary
    # F1/F2/AUC logging itself gets printed.
    patch_df = pd.DataFrame(patch_eval_rows)
    patch_metrics = _compute_region_binary_metrics(patch_df, label_col='patch_label', prob_col='patch_prob', noun='patches')

    # Root-caused 2026-08-22 (repeat systemd-oomd kill during the patch-level overlay stage,
    # see _save_slide_tumor_overlays/_render_one_slide_tumor_overlay's memory hardening):
    # region_eval_rows/patch_eval_rows are plain Python list-of-dict accumulators built ACROSS
    # THE WHOLE test-set evaluation loop above -- for a large test split (patch_eval_rows can run
    # into the millions of rows across all slides) that's a genuinely large amount of per-object
    # Python overhead, and everything below this point (metrics, CSV/JSON writes, and especially
    # the per-slide overlay rendering loop, the heaviest stage) only ever needs the DataFrames,
    # never the original rows. Left alive as ordinary locals of this (very long-running) function,
    # they would otherwise sit in memory, doing nothing, for the rest of its execution -- on top
    # of whatever the overlay rendering loop itself needs. Freed explicitly rather than waiting
    # for these locals to merely fall out of scope, since that only happens when the whole
    # function returns, well after the overlay stage that has already OOM-killed this process
    # twice.
    del region_eval_rows, patch_eval_rows
    gc.collect()

    if not is_multiclass_only_run:
        if region_metrics.get("available", False):
            logging.info("Region-level evaluation available on labeled regions: n=%d", region_metrics["num_labeled_regions"])
            thr_metrics = region_metrics.get("threshold_0_5", {})
            f1_val = thr_metrics.get("f1", None)
            f2_val = thr_metrics.get("f2", None)
            prec_val = thr_metrics.get("precision", None)
            rec_val = thr_metrics.get("recall", None)
            rec_ci = thr_metrics.get("sensitivity_ci95", None)
            spec_val = thr_metrics.get("specificity", None)
            spec_ci = thr_metrics.get("specificity_ci95", None)
            logging.info(
                "Region metrics @0.5: F1=%s%s | F2=%s%s | Precision=%s | Recall=%s %s | Specificity=%s %s",
                f"{float(f1_val):.4f}" if f1_val is not None else "n/a",
                _monitor_tag(monitor_metric, 'region', 'f1'),
                f"{float(f2_val):.4f}" if f2_val is not None else "n/a",
                _monitor_tag(monitor_metric, 'region', 'f2'),
                f"{float(prec_val):.4f}" if prec_val is not None else "n/a",
                f"{float(rec_val):.4f}" if rec_val is not None else "n/a",
                f"(95% CI {rec_ci[0]:.4f}-{rec_ci[1]:.4f})" if rec_ci is not None else "",
                f"{float(spec_val):.4f}" if spec_val is not None else "n/a",
                f"(95% CI {spec_ci[0]:.4f}-{spec_ci[1]:.4f})" if spec_ci is not None else "",
            )
            bal_acc_val = thr_metrics.get("balanced_accuracy", None)
            mcc_val = thr_metrics.get("mcc", None)
            logging.info(
                "Region diagnostics @0.5: BalancedAcc=%s | MCC=%s | Brier=%s | ECE=%s",
                f"{float(bal_acc_val):.4f}" if bal_acc_val is not None else "n/a",
                f"{float(mcc_val):.4f}" if mcc_val is not None else "n/a",
                str(thr_metrics.get("brier")),
                str(thr_metrics.get("ece")),
            )
            ranking = region_metrics.get("ranking", {})
            logging.info(
                "Region ranking: AUC=%s | AP=%s",
                str(ranking.get("auc")),
                str(ranking.get("average_precision")),
            )
        else:
            logging.info("Region-level evaluation unavailable: %s", region_metrics.get("reason", "unknown"))

        if patch_metrics.get("available", False):
            logging.info("Patch-level evaluation available on labeled patches: n=%d", patch_metrics["num_labeled_patches"])
            patch_thr_metrics = patch_metrics.get("threshold_0_5", {})
            patch_f1_val = patch_thr_metrics.get("f1", None)
            patch_f2_val = patch_thr_metrics.get("f2", None)
            patch_prec_val = patch_thr_metrics.get("precision", None)
            patch_rec_val = patch_thr_metrics.get("recall", None)
            patch_rec_ci = patch_thr_metrics.get("sensitivity_ci95", None)
            patch_spec_val = patch_thr_metrics.get("specificity", None)
            patch_spec_ci = patch_thr_metrics.get("specificity_ci95", None)
            logging.info(
                "Patch metrics @0.5: F1=%s%s | F2=%s%s | Precision=%s | Recall (%% of real tumor patches detected)=%s %s | Specificity=%s %s",
                f"{float(patch_f1_val):.4f}" if patch_f1_val is not None else "n/a",
                _monitor_tag(monitor_metric, 'patch', 'f1'),
                f"{float(patch_f2_val):.4f}" if patch_f2_val is not None else "n/a",
                _monitor_tag(monitor_metric, 'patch', 'f2'),
                f"{float(patch_prec_val):.4f}" if patch_prec_val is not None else "n/a",
                f"{float(patch_rec_val):.4f}" if patch_rec_val is not None else "n/a",
                f"(95% CI {patch_rec_ci[0]:.4f}-{patch_rec_ci[1]:.4f})" if patch_rec_ci is not None else "",
                f"{float(patch_spec_val):.4f}" if patch_spec_val is not None else "n/a",
                f"(95% CI {patch_spec_ci[0]:.4f}-{patch_spec_ci[1]:.4f})" if patch_spec_ci is not None else "",
            )
            patch_ranking = patch_metrics.get("ranking", {})
            logging.info(
                "Patch ranking: AUC=%s | AP=%s",
                str(patch_ranking.get("auc")),
                str(patch_ranking.get("average_precision")),
            )
        else:
            logging.info("Patch-level evaluation unavailable: %s", patch_metrics.get("reason", "unknown"))
    else:
        logging.info(
            "Skipping region/patch-level binary evaluation (F1/F2/AUC @0.5): this checkpoint's "
            "region_head/patch_head were never optimized during training (region_cls/patch_cls-only "
            "run) -- see the region/patch multiclass report above instead."
        )

    if output_dir and save_predictions and not region_df.empty:
        region_df.to_csv(output_dir / f"{model_name}_{split}_region_predictions.csv", index=False)

    if output_dir and save_predictions and not patch_df.empty:
        patch_df.to_csv(output_dir / f"{model_name}_{split}_patch_predictions.csv", index=False)

    if output_dir:
        with open(output_dir / f"{model_name}_{split}_region_metrics.json", "w", encoding="utf-8") as f:
            json.dump(region_metrics, f, indent=2)

    if output_dir:
        with open(output_dir / f"{model_name}_{split}_patch_metrics.json", "w", encoding="utf-8") as f:
            json.dump(patch_metrics, f, indent=2)

    # TissueRegionGraphModel's optional multiclass region_cls_target_field/patch_cls_target_field
    # head (e.g. GSDB tissue_type) -- see _extract_region_cls_eval_rows/_compute_node_multiclass_
    # metrics. region_cls_class_names/patch_cls_class_names are [] (model attribute default) for
    # any checkpoint where this head isn't enabled, so _compute_node_multiclass_metrics never
    # runs and everything below is a no-op for every existing binary-only run.
    region_cls_class_names = getattr(model, 'region_cls_class_names', None) or []
    patch_cls_class_names = getattr(model, 'patch_cls_class_names', None) or []
    region_cls_target_field = getattr(model, 'region_cls_target_field', None)
    patch_cls_target_field = getattr(model, 'patch_cls_target_field', None)

    region_cls_df = pd.DataFrame(region_cls_eval_rows)
    region_cls_metrics = (
        _compute_node_multiclass_metrics(region_cls_df, region_cls_class_names, noun='regions')
        if region_cls_class_names else {"available": False, "reason": "region_cls_not_enabled"}
    )
    patch_cls_df = pd.DataFrame(patch_cls_eval_rows)
    patch_cls_metrics = (
        _compute_node_multiclass_metrics(patch_cls_df, patch_cls_class_names, noun='patches')
        if patch_cls_class_names else {"available": False, "reason": "patch_cls_not_enabled"}
    )

    def _log_node_cls_metrics(noun: str, target_field: Optional[str], metrics: Dict) -> None:
        if not metrics.get("available", False):
            return
        noun_singular = noun[:-1] if noun.endswith('s') else noun
        normalized_monitor = str(monitor_metric).strip().lower() if monitor_metric else ""
        _TAG = "  <-- MONITOR METRIC (training checkpoint selection)"
        accuracy_tag = _TAG if normalized_monitor == f"{noun_singular}_cls_accuracy" else ""
        kappa_tag = _TAG if normalized_monitor == f"{noun_singular}_cls_kappa" else ""
        macro_f1_tag = _TAG if normalized_monitor == f"{noun_singular}_cls_macro_f1" else ""
        logging.info(
            "%s multiclass (%s) evaluation: n=%d | accuracy=%.4f%s | balanced_accuracy=%.4f | "
            "cohen_kappa=%.4f%s | mcc=%.4f | macro_f1=%.4f%s | weighted_f1=%.4f",
            noun.capitalize(), target_field, metrics[f"num_labeled_{noun}"],
            metrics["accuracy"], accuracy_tag, metrics["balanced_accuracy"], metrics["cohen_kappa"], kappa_tag,
            metrics["mcc"], metrics["macro_f1"], macro_f1_tag, metrics["weighted_f1"],
        )
        for name, m in metrics["per_class"].items():
            logging.info(
                "  %s: precision=%.3f recall=%.3f specificity=%s balanced_acc=%s f1=%.3f support=%d",
                name, m["precision"], m["recall"],
                f"{m['specificity']:.3f}" if m["specificity"] is not None else "n/a",
                f"{m['balanced_accuracy']:.3f}" if m["balanced_accuracy"] is not None else "n/a",
                m["f1"], m["support"],
            )

    _log_node_cls_metrics('regions', region_cls_target_field, region_cls_metrics)
    _log_node_cls_metrics('patches', patch_cls_target_field, patch_cls_metrics)

    if output_dir and save_predictions and not region_cls_df.empty:
        region_cls_df.to_csv(output_dir / f"{model_name}_{split}_region_{region_cls_target_field}_predictions.csv", index=False)
    if output_dir and save_predictions and not patch_cls_df.empty:
        patch_cls_df.to_csv(output_dir / f"{model_name}_{split}_patch_{patch_cls_target_field}_predictions.csv", index=False)
    if output_dir and region_cls_metrics.get("available", False):
        with open(output_dir / f"{model_name}_{split}_region_{region_cls_target_field}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(region_cls_metrics, f, indent=2)
    if output_dir and patch_cls_metrics.get("available", False):
        with open(output_dir / f"{model_name}_{split}_patch_{patch_cls_target_field}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(patch_cls_metrics, f, indent=2)

    # Slide-thumbnail overlays for the multiclass region_cls/patch_cls heads (e.g. GSDB
    # tissue_type) -- the multiclass analogue of the binary region_head/patch_head overlays
    # above. Gated on the cls DataFrames themselves being non-empty, NOT on is_multiclass_only_run
    # (that flag only means "no binary head was trained"; a checkpoint can have BOTH binary and
    # multiclass heads active at once -- see evaluate_model_graph's docstring -- in which case
    # both overlay sets are wanted side by side).
    if output_dir and save_plots and str(split).startswith('test') and not region_cls_df.empty and region_cls_class_names:
        region_cls_patch_footprints = _expand_region_cls_df_to_patch_footprints(region_cls_df, patch_cls_df)
        if not region_cls_patch_footprints.empty:
            _save_slide_multiclass_overlays(
                df=region_cls_patch_footprints,
                class_names=region_cls_class_names,
                output_dir=output_dir,
                model_name=model_name,
                split=split,
                dir_name=f'original_slide_{region_cls_target_field}_region_overlays',
                title_noun='regions',
                target_field=region_cls_target_field,
            )
        else:
            logging.info(
                "Skipping region-level %s overlays: could not join region predictions to their "
                "member patches' footprints (patch<->region graph edge or patch_cls_df "
                "unavailable for this run).", region_cls_target_field,
            )
        del region_cls_patch_footprints
        gc.collect()

    if output_dir and save_plots and str(split).startswith('test') and not patch_cls_df.empty and patch_cls_class_names:
        _save_slide_multiclass_overlays(
            df=patch_cls_df,
            class_names=patch_cls_class_names,
            output_dir=output_dir,
            model_name=model_name,
            split=split,
            dir_name=f'original_slide_{patch_cls_target_field}_patch_overlays',
            title_noun='patches',
            target_field=patch_cls_target_field,
        )

    if multiclass_detected and 'pred_class' in df.columns:
        multiclass_metrics = _return_multiclass_metrics(
            df=df,
            validation_thresholds=validation_thresholds,
            output_dir=output_dir,
            model_name=model_name,
            split=split,
            save_predictions=save_predictions,
            save_plots=save_plots,
            class_names=slide_cls_class_names or None,
        )
        multiclass_metrics['region_metrics'] = region_metrics
        multiclass_metrics['patch_metrics'] = patch_metrics
        multiclass_metrics['region_cls_metrics'] = region_cls_metrics
        multiclass_metrics['patch_cls_metrics'] = patch_cls_metrics
        return multiclass_metrics


    # Hoisted above the is_multiclass_only_run guard below -- optimal_thresholds_dict and
    # df['prob'] are both read later (combined_metrics, the output_dir plotting section) 
    # regardless of whether the binary summary itself gets printed/computed.
    if validation_thresholds is None:
        logging.warning(
            "No validation thresholds provided. Falling back to default thresholds/aggregation for evaluation only."
        )
        validation_thresholds = {}
    optimal_thresholds_dict = {k: v for k, v in validation_thresholds.items()
                             if k not in ['validation_tile_auroc', 'validation_slide_auroc',
                                        'validation_particle_auroc', 'optimal_aggregation',
                                        'adaptive_threshold']}
    if 'prob' not in df.columns:
        if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
            logging.info("Converting raw logits to probabilities with sigmoid...")
            df['prob'] = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
        else:
            logging.info("Raw predictions are already in probability range [0,1]")
            df['prob'] = df['raw_pred']

    if not is_multiclass_only_run:
        logging.info(f"\nEvaluation Summary:")
        logging.info(f"Total samples (graphs/slides): {len(df)}")


        pos_count = (df['label'] == 1).sum()
        neg_count = (df['label'] == 0).sum()
        logging.info(f"\nClass Distribution:")
        logging.info(f"Positive class: {pos_count} ({pos_count/len(df):.1%})")
        logging.info(f"Negative class: {neg_count} ({neg_count/len(df):.1%})")


        logging.info("\nPrediction Statistics:")
        logging.info(f"Raw logits mean: {df['raw_pred'].mean():.3f}")
        logging.info(f"Raw logits std: {df['raw_pred'].std():.3f}")
        logging.info(f"Raw logits min: {df['raw_pred'].min():.3f}")
        logging.info(f"Raw logits max: {df['raw_pred'].max():.3f}")





        logging.info("\nUsing validation-optimized thresholds:")
        for level, level_metrics in optimal_thresholds_dict.items():
            if isinstance(level_metrics, dict) and 'threshold' in level_metrics:
                logging.info(f"{level.capitalize()} level: {level_metrics['threshold']:.3f}")
                sensitivity = level_metrics.get('sensitivity', None)
                specificity = level_metrics.get('specificity', None)
                f1 = level_metrics.get('f1', None)
                logging.info(f"  Sensitivity: {float(sensitivity):.2f}" if sensitivity is not None else "  Sensitivity: n/a")
                logging.info(f"  Specificity: {float(specificity):.2f}" if specificity is not None else "  Specificity: n/a")
                logging.info(f"  F1 Score: {float(f1):.2f}" if f1 is not None else "  F1 Score: n/a")


        for auc_key in ['validation_tile_auroc', 'validation_slide_auroc', 'validation_particle_auroc']:
            if auc_key in validation_thresholds:
                logging.info(f"{auc_key.replace('_', ' ').capitalize()}: {validation_thresholds[auc_key]:.4f}")



        logging.info("Skipping any test-set-based prediction inversion to preserve unbiased evaluation.")



        if task == 'inflammation':
            # Defense in depth against the fp32-retry above (which fixes the common fp16-autocast
            # overflow case): if a graph's prediction is STILL non-finite (a genuinely degenerate
            # graph, not an autocast artifact), exclude it here rather than silently feeding NaN
            # into roc_auc_score (which used to raise, caught below, and masked as a fake
            # AUC=0.5) or into `df['prob'] > threshold` (NaN compares False in numpy, so these
            # slides were previously miscounted as "predicted negative" in the confusion matrix
            # regardless of their true label).
            finite_mask = np.isfinite(df['prob'].to_numpy(dtype=float))
            num_non_finite = int((~finite_mask).sum())
            if num_non_finite > 0:
                excluded_names = (
                    df.loc[~finite_mask, 'slide_name'].tolist() if 'slide_name' in df.columns else []
                )
                logging.warning(
                    "%d/%d slide-level predictions are non-finite (NaN/Inf) even after the fp32 "
                    "retry above -- excluding them from slide-level AUC/confusion-matrix metrics "
                    "below instead of silently treating them as \"predicted negative\". "
                    "Affected slide_name(s): %s",
                    num_non_finite, len(df), excluded_names,
                )
            metrics_df = df.loc[finite_mask]

            threshold = optimal_thresholds_dict.get('slide', {}).get('threshold', 0.5)

            adaptive_cfg = validation_thresholds.get('adaptive_threshold', {}) if validation_thresholds else {}
            # TRIED 2026-08-13 and reverted: firing this on deviation-from-target_rate (not just
            # near-total collapse) for region_slide_net's test_scanner2 run (pred_pos_rate=0.809 vs
            # validation target_rate=0.487) moved Specificity 0.40 -> 0.77 and BalancedAcc
            # 0.658 -> 0.688, but at the cost of Sensitivity 0.915 -> 0.610 and F1 0.824 -> 0.706 --
            # a clear net loss given this project's own stated priority (slide_f2 as the training
            # monitor metric elsewhere: a missed positive costs more than a false alarm). The
            # deeper issue: target_rate here is scanner1 validation's label rate (48.7%), but
            # test_scanner2's TRUE label rate is 66.3% -- scanner1 and scanner2 genuinely differ in
            # prevalence, not just in measurement/domain shift, so quantile-matching to validation's
            # rate corrects toward the wrong prior. Left as the original near-total-collapse-only
            # check; a real fix would need a defensible target_rate for scanner2 specifically, not
            # scanner1's.
            target_rate = optimal_thresholds_dict.get('slide', {}).get('validation_pred_positive_rate', None)
            if target_rate is None:
                target_rate = optimal_thresholds_dict.get('slide', {}).get('validation_label_positive_rate', None)

            # Moved into the shared _compute_slide_binary_metrics helper (used identically by
            # evaluate_model_img_mil) so MIL and graph slide-level numbers are directly comparable --
            # see that function's docstring.
            test_confusion_matrix = _compute_slide_binary_metrics(
                y_true=metrics_df['label'].values,
                y_prob=metrics_df['prob'].values,
                base_threshold=threshold,
                adaptive_cfg=adaptive_cfg,
                target_rate=target_rate,
            )
            # metrics_df is already finite-filtered above (num_non_finite counted against the
            # original, unfiltered df) -- the helper's own finite-mask is a no-op here, so restore
            # the outer, pre-filter count rather than the helper's (always-0-in-this-caller) value.
            test_confusion_matrix['excluded_non_finite'] = num_non_finite
            slide_auc = test_confusion_matrix['auc']
            threshold = test_confusion_matrix['threshold']
            tp, tn, fp, fn = test_confusion_matrix['tp'], test_confusion_matrix['tn'], test_confusion_matrix['fp'], test_confusion_matrix['fn']
            test_accuracy = test_confusion_matrix['accuracy']
            test_sensitivity = test_confusion_matrix['sensitivity']
            test_specificity = test_confusion_matrix['specificity']
            test_f1 = test_confusion_matrix['f1']
            test_f2 = test_confusion_matrix['f2']
            sensitivity_ci = test_confusion_matrix['sensitivity_ci95']
            specificity_ci = test_confusion_matrix['specificity_ci95']
            logging.info(f"Slide-level AUC: {slide_auc:.4f}")

            logging.info("\nTest Confusion Matrix (Slide-level):")
            logging.info(f"  True Positives: {tp}")
            logging.info(f"  True Negatives: {tn}")
            logging.info(f"  False Positives: {fp}")
            logging.info(f"  False Negatives: {fn}")
            logging.info(f"  Total Samples: {tp + tn + fp + fn}" + (
                f" (+{num_non_finite} excluded as non-finite, {len(df)} total in split)"
                if num_non_finite > 0 else ""
            ))
            logging.info(f"  Accuracy: {test_accuracy:.4f}")
            logging.info(f"  Sensitivity: {test_sensitivity:.4f} (95% CI {sensitivity_ci[0]:.4f}-{sensitivity_ci[1]:.4f}, n_positive={tp + fn})")
            logging.info(f"  Specificity: {test_specificity:.4f} (95% CI {specificity_ci[0]:.4f}-{specificity_ci[1]:.4f}, n_negative={tn + fp})")
            logging.info(f"  F1 Score: {test_f1:.4f}{_monitor_tag(monitor_metric, 'slide', 'f1')}")
            logging.info(f"  F2 Score (recall weighted 4x precision): {test_f2:.4f}{_monitor_tag(monitor_metric, 'slide', 'f2')}")
            logging.info(f"  Balanced Accuracy: {(test_sensitivity + test_specificity) / 2:.4f}")
            logging.info(f"  AUC: {slide_auc:.4f}")

            if 'oracle_test_best_f1' in test_confusion_matrix:
                logging.info(
                    "\nSlide-level oracle thresholds (diagnostic only, swept directly on this test set — "
                    "NOT used for the metrics above, which use the validation-optimized threshold):"
                )
                best_f1 = test_confusion_matrix['oracle_test_best_f1']
                logging.info(
                    "  Best F1 @ threshold=%.3f: F1=%.4f | Sensitivity=%.4f | Specificity=%.4f | BalancedAcc=%.4f",
                    best_f1['threshold'], best_f1['f1'], best_f1['sensitivity'], best_f1['specificity'], best_f1['balanced_accuracy'],
                )
                best_bal_acc = test_confusion_matrix['oracle_test_best_balanced_accuracy']
                logging.info(
                    "  Best Balanced Accuracy @ threshold=%.3f: BalancedAcc=%.4f | Sensitivity=%.4f | Specificity=%.4f | F1=%.4f",
                    best_bal_acc['threshold'], best_bal_acc['balanced_accuracy'], best_bal_acc['sensitivity'], best_bal_acc['specificity'], best_bal_acc['f1'],
                )

            per_stain = _compute_per_stain_accuracy(metrics_df, slide_groups, threshold)
            if per_stain:
                test_confusion_matrix['per_stain'] = per_stain
                logging.info(f"\nCorrectly classified slides per stain (threshold={threshold:.3f}):")
                for stain, m in per_stain.items():
                    logging.info(
                        "  %-4s %3d/%-3d (%5.1f%%) | FP=%d | FN=%d",
                        stain, m['correct'], m['n'], 100 * m['accuracy'], m['fp'], m['fn'],
                    )
                if output_dir:
                    pd.DataFrame(
                        [{'stain': k, **v} for k, v in per_stain.items()]
                    ).to_csv(output_dir / f"{model_name}_{split}_per_stain_accuracy.csv", index=False)
                    if save_plots:
                        _plot_per_stain_accuracy(
                            per_stain,
                            output_dir / f"{model_name}_{split}_per_stain_accuracy.png",
                            title=f"{split}: correctly classified slides per stain",
                        )
    else:
        logging.info(
            "Skipping slide-level binary evaluation summary (AUC/confusion matrix/thresholds): "
            "this checkpoint's slide head was never optimized during training (region_cls/patch_cls-only "
            "run, see is_multiclass_only_run) -- see the region/patch multiclass report above instead."
        )


    if output_dir and save_plots:
        if is_multiclass_only_run:
            logging.info(
                "Skipping region/patch binary-head overlay images (tumor region/patch overlays, "
                "diagnostics plots): this checkpoint's region_head/patch_head were never optimized "
                "during training (region_cls/patch_cls-only run)."
            )
        if not region_df.empty and str(split).startswith('test') and not is_multiclass_only_run:
            region_threshold, threshold_source = _select_region_overlay_threshold(
                validation_thresholds=validation_thresholds,
                region_metrics=region_metrics,
            )
            used_thr_metrics = _region_metrics_for_existing_threshold(region_df, region_threshold)
            if used_thr_metrics:
                region_metrics['used_threshold_metrics'] = used_thr_metrics
            logging.info(
                "Using region overlay threshold %.4f (source=%s)",
                float(region_threshold),
                str(threshold_source),
            )
            if used_thr_metrics:
                logging.info(
                    "Region fit @used threshold %.3f: Dice=%.4f | IoU=%.4f | Precision=%.4f | Recall=%.4f | Specificity=%s",
                    float(used_thr_metrics.get('threshold', region_threshold)),
                    float(used_thr_metrics.get('dice', 0.0)),
                    float(used_thr_metrics.get('iou', 0.0)),
                    float(used_thr_metrics.get('precision', 0.0)),
                    float(used_thr_metrics.get('recall', 0.0)),
                    str(used_thr_metrics.get('specificity')),
                )
            _plot_test_region_predictions(
                region_df=region_df,
                threshold=region_threshold,
                output_dir=output_dir,
                model_name=model_name,
                split=split,
            )
            _plot_region_diagnostics(
                region_df=region_df,
                threshold=region_threshold,
                output_dir=output_dir,
                model_name=model_name,
                split=split,
            )
            region_patch_footprints = _expand_region_df_to_patch_footprints(region_df, patch_df)
            if not region_patch_footprints.empty:
                _save_slide_tumor_overlays(
                    df=region_patch_footprints,
                    threshold=region_threshold,
                    output_dir=output_dir,
                    model_name=model_name,
                    split=split,
                    prob_col='region_prob',
                    dir_name='original_slide_tumor_overlays',
                    title_noun='regions',
                )
            else:
                logging.info(
                    "Skipping region-level slide overlays: could not join region predictions to "
                    "their member patches' footprints (patch<->region graph edge or patch_df "
                    "unavailable for this run)."
                )
            # region_patch_footprints is a near-full-size duplicate of patch_df's columns (one
            # row per member patch of every resolved region) -- freed here rather than left alive
            # as a local for the rest of this function, since the patch-level overlay block right
            # below is the heaviest remaining stage (every scored patch in the test set, not just
            # region-joined ones) and doesn't need it at all. See the del region_eval_rows/
            # patch_eval_rows comment above for the same rationale applied earlier in this
            # function.
            del region_patch_footprints
            gc.collect()

        # region_df itself is dead from here on (only patch_df/df are read below) -- freed for
        # the same reason as region_patch_footprints/region_eval_rows/patch_eval_rows above,
        # right before the patch-level overlay block, which is the heaviest remaining stage.
        del region_df
        gc.collect()

        if not patch_df.empty and str(split).startswith('test') and not is_multiclass_only_run:
            patch_threshold, patch_threshold_source = _select_region_overlay_threshold(
                validation_thresholds=validation_thresholds,
                region_metrics=patch_metrics,
                key='patch',
            )
            used_patch_thr_metrics = _region_metrics_for_existing_threshold(
                patch_df, patch_threshold, label_col='patch_label', prob_col='patch_prob',
            )
            if used_patch_thr_metrics:
                patch_metrics['used_threshold_metrics'] = used_patch_thr_metrics
            logging.info(
                "Using patch overlay threshold %.4f (source=%s)",
                float(patch_threshold),
                str(patch_threshold_source),
            )
            if used_patch_thr_metrics:
                logging.info(
                    "Patch fit @used threshold %.3f: Dice=%.4f | IoU=%.4f | Precision=%.4f | Recall=%.4f | Specificity=%s",
                    float(used_patch_thr_metrics.get('threshold', patch_threshold)),
                    float(used_patch_thr_metrics.get('dice', 0.0)),
                    float(used_patch_thr_metrics.get('iou', 0.0)),
                    float(used_patch_thr_metrics.get('precision', 0.0)),
                    float(used_patch_thr_metrics.get('recall', 0.0)),
                    str(used_patch_thr_metrics.get('specificity')),
                )
            _save_slide_tumor_overlays(
                df=patch_df,
                threshold=patch_threshold,
                output_dir=output_dir,
                model_name=model_name,
                split=split,
                prob_col='patch_prob',
                dir_name='original_slide_tumor_patch_overlays',
                title_noun='patches',
                halfsize_level0_col='patch_halfsize_level0',
            )

        # patch_df is dead from here on (nothing below reads it, only df) -- freed for the same
        # reason as region_df just above.
        del patch_df
        gc.collect()

        probs = df['prob'].values

        plt.figure(figsize=(16, 7))


        plt.subplot(1, 2, 1)
        plt.hist(df['raw_pred'][df['label']==1], bins=50, alpha=0.6, label='Positive class', color='blue')
        plt.hist(df['raw_pred'][df['label']==0], bins=50, alpha=0.6, label='Negative class', color='orange')
        plt.axvline(x=0.0, color='r', linestyle='--', label='Default threshold (logit=0)')

        if 'slide' in optimal_thresholds_dict:
            prob_threshold = optimal_thresholds_dict['slide']['threshold']
            logit_threshold = np.log(prob_threshold/(1-prob_threshold)) if 0 < prob_threshold < 1 else (10 if prob_threshold >= 1 else -10)
            plt.axvline(x=logit_threshold, color='g', linestyle='--',
                        label=f'Optimal threshold (logit={logit_threshold:.2f})')

        plt.title('Raw Logits Distribution by Class')
        plt.xlabel('Logit Value')
        plt.ylabel('Count')
        plt.legend()


        plt.subplot(1, 2, 2)
        plt.hist(probs[df['label']==1], bins=50, alpha=0.6, label='Positive class', color='blue')
        plt.hist(probs[df['label']==0], bins=50, alpha=0.6, label='Negative class', color='orange')
        plt.axvline(x=0.5, color='r', linestyle='--', label='Default threshold (prob=0.5)')

        if 'slide' in optimal_thresholds_dict:
            plt.axvline(x=optimal_thresholds_dict['slide']['threshold'], color='g', linestyle='--',
                        label=f'Optimal threshold ({optimal_thresholds_dict["slide"]["threshold"]:.2f})')

        plt.title('Predicted Probabilities by Class')
        plt.xlabel('Probability')
        plt.ylabel('Count')
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name}_{split}_logits_prob_histograms.png")
        plt.close()

    if output_dir and save_predictions:
        df.to_csv(output_dir / f"{model_name}_{split}_predictions.csv", index=False)


    combined_metrics = {
        **validation_thresholds,
        'optimal_thresholds': optimal_thresholds_dict,
        'test_confusion_matrix': test_confusion_matrix,
        'region_metrics': region_metrics,
        'patch_metrics': patch_metrics,
        'region_cls_metrics': region_cls_metrics,
        'patch_cls_metrics': patch_cls_metrics,
    }


    for auc_key in ['validation_tile_auroc', 'validation_slide_auroc', 'validation_particle_auroc']:
        if auc_key in validation_thresholds:
            combined_metrics[auc_key] = validation_thresholds[auc_key]

    if output_dir:
        with open(output_dir / f"{model_name}_{split}_slide_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_confusion_matrix, f, indent=2)

    return combined_metrics


def plot_precision_recall_curve(df: pd.DataFrame, task: str, output_dir: Path) -> None:
    """Plot precision-recall curves for different hierarchical levels."""
    from sklearn.metrics import precision_recall_curve, average_precision_score


    output_dir.mkdir(parents=True, exist_ok=True)


    if 'raw_pred' in df.columns and (df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1):
        probs = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
    else:
        probs = df['raw_pred'].values


    tile_precision, tile_recall, _ = precision_recall_curve(df['label'], probs)
    tile_ap = average_precision_score(df['label'], probs)


    if task == 'inflammation':
        slide_df = df.groupby('slide_name').agg({
            'raw_pred': 'mean',
            'label': 'first'
        }).reset_index()

        if 'raw_pred' in slide_df.columns and (slide_df['raw_pred'].min() < 0 or slide_df['raw_pred'].max() > 1):
            slide_probs = torch.sigmoid(torch.tensor(slide_df['raw_pred'].values)).numpy()
        else:
            slide_probs = slide_df['raw_pred'].values

        slide_precision, slide_recall, _ = precision_recall_curve(slide_df['label'], slide_probs)
        slide_ap = average_precision_score(slide_df['label'], slide_probs)


        plt.figure(figsize=(10, 8))
        plt.plot(tile_recall, tile_precision, label=f'Tile-level (AP = {tile_ap:.3f})')
        plt.plot(slide_recall, slide_precision, label=f'Slide-level (AP = {slide_ap:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for Different Hierarchical Levels')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'precision_recall_curves.png', dpi=300)
        plt.close()

    else:
        particle_df = df.groupby(['slide_name', 'particle_id']).agg({
            'raw_pred': 'mean',
            'label': 'first'
        }).reset_index()

        if 'raw_pred' in particle_df.columns and (particle_df['raw_pred'].min() < 0 or particle_df['raw_pred'].max() > 1):
            particle_probs = torch.sigmoid(torch.tensor(particle_df['raw_pred'].values)).numpy()
        else:
            particle_probs = particle_df['raw_pred'].values

        particle_precision, particle_recall, _ = precision_recall_curve(particle_df['label'], particle_probs)
        particle_ap = average_precision_score(particle_df['label'], particle_probs)


        plt.figure(figsize=(10, 8))
        plt.plot(tile_recall, tile_precision, label=f'Tile-level (AP = {tile_ap:.3f})')
        plt.plot(particle_recall, particle_precision, label=f'Particle-level (AP = {particle_ap:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for Different Hierarchical Levels')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'precision_recall_curves.png', dpi=300)
        plt.close()
