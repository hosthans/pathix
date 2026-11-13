from typing import Dict, List, Tuple, Optional, Any, Union
import gc
from torch.utils.data import DataLoader
import pandas as pd
import logging
from pathlib import Path
import json
import torch
import sys
import torch.nn as nn
import numpy as np
from tqdm import tqdm


project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from src.utils.metrics_utils import optimize_aggregation_strategy, optimize_hierarchical_thresholds
from src.core.registry_utils import register_validation_threshold
from src.utils.graph_patch_utils import apply_slide_derived_negative_labels
from src.utils.training_utils import dataloader_worker_init_fn


def _unwrap_graph_logits(outputs):
    if isinstance(outputs, dict):
        logits = outputs.get('logits', None)
        if not torch.is_tensor(logits):
            raise ValueError("Graph model output dict must contain tensor key 'logits'.")
        return logits
    return outputs


def _is_positive_region_label(label_text: str) -> bool:
    text = str(label_text).strip().lower()
    return any(token in text for token in ("tumor", "metast", "macro", "micro", "itc", "positive", "status=inflamed"))


def _is_normal_region_label(label_text: str) -> bool:
    text = str(label_text).strip().lower()
    return any(token in text for token in ("normal", "negative", "benign", "healthy", "background", "non_tumor", "status=noninflamed"))


def _extract_annotation_vocab_from_graph_batch(data) -> list[str]:
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


def _extract_graph_meta_entries_from_graph_batch(data) -> list[dict]:
    meta_obj = getattr(data, 'meta', None)
    if isinstance(meta_obj, dict):
        return [meta_obj]
    if isinstance(meta_obj, list):
        return [entry for entry in meta_obj if isinstance(entry, dict)]
    return []


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

@register_validation_threshold(('image', 'none', False))
def calculate_validation_thresholds_image(model: Optional[nn.Module] = None, val_dataset = None,
                             task: str = None, output_dir: Optional[Path] = None) -> Dict:
    """Calculate optimal validation thresholds for an image-level model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=dataloader_worker_init_fn,
    )


    model.eval()
    predictions = []
    multiclass_detected = False
    detected_num_classes = 1

    with torch.no_grad():
        for inputs, labels, metadata_batch in tqdm(val_loader, desc="Collecting validation predictions"):
            inputs = inputs.to(device)
            outputs = model(inputs)

            is_multiclass_batch = outputs.dim() > 1 and outputs.size(-1) > 1
            if is_multiclass_batch:
                multiclass_detected = True
                detected_num_classes = max(detected_num_classes, int(outputs.size(-1)))

                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                if isinstance(labels, torch.Tensor) and labels.dim() == 0:
                    labels = labels.unsqueeze(0)

                for i in range(outputs.size(0)):
                    metadata = {}
                    for k, v in metadata_batch.items():
                        if isinstance(v, list):
                            metadata[k] = v[i]
                        elif isinstance(v, torch.Tensor):
                            metadata[k] = v[i].item() if v.dim() == 0 else v[i]
                        else:
                            metadata[k] = v

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
                        pred_dict['inflammation_status'] = metadata.get('inflammation_status', 'unknown')

                    predictions.append(pred_dict)
                continue

            if outputs.dim() > 1:
                outputs = outputs.squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)

            if isinstance(labels, torch.Tensor) and labels.dim() == 0:
                labels = labels.unsqueeze(0)

            for i in range(len(outputs)):
                metadata = {}
                for k, v in metadata_batch.items():
                    if isinstance(v, list):
                        metadata[k] = v[i]
                    elif isinstance(v, torch.Tensor):
                        metadata[k] = v[i].item() if v.dim() == 0 else v[i]
                    else:
                        metadata[k] = v

                pred_dict = {
                    'raw_pred': outputs[i].detach().cpu().item(),
                    'label': labels[i].item(),
                    'particle_id': metadata['particle_id'],
                    'slide_name': metadata['slide_name'],
                }

                if task == 'inflammation':
                    pred_dict['inflammation_status'] = metadata.get('inflammation_status', 'unknown')

                predictions.append(pred_dict)


    val_df = pd.DataFrame(predictions)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    if multiclass_detected:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        y_true = val_df['label'].astype(int).values
        y_pred = val_df['pred_class'].astype(int).values
        prob_cols = sorted(
            [c for c in val_df.columns if c.startswith('prob_class_')],
            key=lambda name: int(name.split('_')[-1])
        )
        y_score = val_df[prob_cols].values if prob_cols else None

        accuracy = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
        f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0)) if len(y_true) > 0 else 0.0
        auc_ovo_macro = 0.0
        if y_score is not None and len(np.unique(y_true)) > 1:
            try:
                auc_ovo_macro = float(
                    roc_auc_score(
                        y_true,
                        y_score,
                        multi_class='ovo',
                        average='macro',
                        labels=np.arange(y_score.shape[1]),
                    )
                )
            except Exception as exc:
                logging.warning(f"Could not calculate multiclass validation AUROC: {exc}")

        validation_optimized = {
            'mode': 'multiclass',
            'num_classes': int(max(detected_num_classes, len(prob_cols) if prob_cols else 1)),
            'optimal_aggregation': {
                'strategy': 'argmax',
                'description': 'Multiclass mode uses argmax over class probabilities; no binary threshold optimization.'
            },
            'validation_multiclass': {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'auc_ovo_macro': auc_ovo_macro,
            }
        }

        if output_dir:
            thresholds_file = output_dir / "validation_thresholds.json"
            with open(thresholds_file, 'w') as f:
                json.dump(validation_optimized, f, indent=2)
            logging.info(f"Validation multiclass metrics saved to: {thresholds_file}")
        return validation_optimized


    optimal_thresholds = optimize_hierarchical_thresholds(
        df=val_df,
        task=task,
        output_dir=output_dir
    )

    optimal_aggregation = optimize_aggregation_strategy(
        df=val_df,
        task=task,
        output_dir=output_dir
    )

    validation_auc_metrics = {}


    if 'raw_pred' in val_df.columns:

        if val_df['raw_pred'].min() < 0 or val_df['raw_pred'].max() > 1:
            tile_probs = torch.sigmoid(torch.tensor(val_df['raw_pred'].values)).numpy()
        else:
            tile_probs = val_df['raw_pred'].values


        from sklearn.metrics import roc_auc_score
        try:
            validation_auc_metrics['validation_tile_auroc'] = roc_auc_score(val_df['label'].values, tile_probs)
        except Exception as e:
            logging.warning(f"Could not calculate validation tile-level AUC: {str(e)}")
            validation_auc_metrics['validation_tile_auroc'] = 0.5


    if task == 'inflammation':

        slide_df = val_df.groupby('slide_name').agg({
            'raw_pred': 'mean',
            'label': 'first'
        }).reset_index()

        if 'raw_pred' in slide_df.columns:

            if slide_df['raw_pred'].min() < 0 or slide_df['raw_pred'].max() > 1:
                slide_probs = torch.sigmoid(torch.tensor(slide_df['raw_pred'].values)).numpy()
            else:
                slide_probs = slide_df['raw_pred'].values


            try:
                validation_auc_metrics['validation_slide_auroc'] = roc_auc_score(slide_df['label'].values, slide_probs)
            except Exception as e:
                logging.warning(f"Could not calculate validation slide-level AUC: {str(e)}")
                validation_auc_metrics['validation_slide_auroc'] = 0.5
    else:

        particle_df = val_df.groupby(['slide_name', 'particle_id']).agg({
            'raw_pred': 'mean',
            'label': 'first'
        }).reset_index()

        if 'raw_pred' in particle_df.columns:

            if particle_df['raw_pred'].min() < 0 or particle_df['raw_pred'].max() > 1:
                particle_probs = torch.sigmoid(torch.tensor(particle_df['raw_pred'].values)).numpy()
            else:
                particle_probs = particle_df['raw_pred'].values


            try:
                validation_auc_metrics['validation_particle_auroc'] = roc_auc_score(particle_df['label'].values, particle_probs)
            except Exception as e:
                logging.warning(f"Could not calculate validation particle-level AUC: {str(e)}")
                validation_auc_metrics['validation_particle_auroc'] = 0.5

    validation_optimized = {
        **optimal_thresholds,
        "optimal_aggregation": optimal_aggregation,
        **validation_auc_metrics
    }

    if task == 'inflammation' and 'raw_pred' in val_df.columns:
        # Shared slide-level protocol (Experiment E1): the same one-score-per-slide procedure as
        # calculate_validation_thresholds_mil_image / _graph, so a patch-CNN's slide numbers are
        # computed exactly like a MIL or graph run's. Slide score = mean tile probability; its
        # G-mean threshold and positive rates feed _compute_slide_binary_metrics at test time
        # (evaluate_model_img). validation_slide_auroc is re-based on this same slide score so Val
        # and Test AUC use one definition; the previous mean-of-logits value is kept alongside.
        from src.utils.metrics_utils import find_optimal_threshold
        if val_df['raw_pred'].min() < 0 or val_df['raw_pred'].max() > 1:
            tile_probs_mp = torch.sigmoid(torch.tensor(val_df['raw_pred'].values)).numpy()
        else:
            tile_probs_mp = val_df['raw_pred'].values
        slide_mp = pd.DataFrame({
            'slide_name': val_df['slide_name'].values,
            'prob': tile_probs_mp,
            'label': val_df['label'].values,
        }).groupby('slide_name').agg(prob=('prob', 'mean'), label=('label', 'first')).reset_index()
        slide_mp_labels = slide_mp['label'].values.astype(int)
        slide_mp_probs = slide_mp['prob'].values
        mp_threshold, mp_sensitivity, mp_specificity = find_optimal_threshold(slide_mp_labels, slide_mp_probs)
        try:
            mp_auc = float(roc_auc_score(slide_mp_labels, slide_mp_probs))
        except Exception:
            mp_auc = 0.5
        validation_optimized['slide_mean_prob'] = {
            'threshold': float(mp_threshold),
            'sensitivity': float(mp_sensitivity),
            'specificity': float(mp_specificity),
            'auroc': mp_auc,
            'validation_pred_positive_rate': float((slide_mp_probs > mp_threshold).mean()),
            'validation_label_positive_rate': float(slide_mp_labels.mean()),
        }
        # Same block as calculate_validation_thresholds_mil_image / _graph.
        validation_optimized['adaptive_threshold'] = {
            "enabled": False,  # strict protocol: the test set never influences a threshold
            "min_auc": 0.60,
            "collapse_min_rate": 0.15,
            "min_samples": 20,
        }
        if 'validation_slide_auroc' in validation_optimized:
            validation_optimized['validation_slide_auroc_mean_logit'] = validation_optimized['validation_slide_auroc']
        validation_optimized['validation_slide_auroc'] = mp_auc


    if output_dir:
        thresholds_file = output_dir / "validation_thresholds.json"
        with open(thresholds_file, 'w') as f:
            json.dump(validation_optimized, f, indent=2)

        logging.info(f"Validation-optimized thresholds and aggregation saved to: {thresholds_file}")

    return validation_optimized


@register_validation_threshold(('image', 'none', True))
def calculate_validation_thresholds_mil_image(
    model: Optional[nn.Module] = None,
    val_dataset=None,
    task: str = None,
    output_dir: Optional[Path] = None
) -> Dict:
    """Calculate optimal validation thresholds for a MIL image-level model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=dataloader_worker_init_fn,
    )

    model.eval()
    predictions = []
    multiclass_detected = False
    detected_num_classes = 1

    global_idx = 0

    with torch.no_grad():
        for bag_images, bag_labels in tqdm(val_loader, desc="Collecting validation predictions"):
            bag_images = bag_images.to(device)
            bag_images = bag_images.squeeze(0)

            num_tiles = bag_images.size(0)
            bag_idx = torch.zeros(num_tiles, dtype=torch.long, device=device)

            outputs, _ = model(bag_images, bag_idx)

            if outputs.ndim > 0 and outputs.numel() > 1:
                multiclass_detected = True
                logits = outputs.view(1, -1)
                probs = torch.softmax(logits, dim=1)
                pred_class = int(torch.argmax(probs, dim=1).item())
                label_val = int(bag_labels.squeeze().item())

                pred_dict = {
                    "raw_pred": float(logits[0, pred_class].item()),
                    "prob": float(probs[0, pred_class].item()),
                    "pred_class": pred_class,
                    "label": label_val,
                    "slide_name": f"bag_{global_idx}",
                    "particle_id": 0,
                }
                detected_num_classes = max(detected_num_classes, int(logits.size(1)))
                for class_idx in range(logits.size(1)):
                    pred_dict[f'prob_class_{class_idx}'] = float(probs[0, class_idx].item())

                if task == "inflammation":
                    pred_dict["inflammation_status"] = "unknown"

                predictions.append(pred_dict)
                global_idx += 1
                continue


            if outputs.ndim == 0:
                output_val = outputs.item()
            else:
                output_val = outputs.squeeze().item()

            label_val = bag_labels.squeeze().item()

            pred_dict = {
                "raw_pred": float(output_val),
                "label": int(label_val),
                "slide_name": f"bag_{global_idx}",
                "particle_id": 0,
            }

            if task == "inflammation":
                pred_dict["inflammation_status"] = "unknown"

            predictions.append(pred_dict)
            global_idx += 1

    val_df = pd.DataFrame(predictions)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    if multiclass_detected:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        y_true = val_df['label'].astype(int).values
        y_pred = val_df['pred_class'].astype(int).values
        prob_cols = sorted(
            [c for c in val_df.columns if c.startswith('prob_class_')],
            key=lambda name: int(name.split('_')[-1])
        )
        y_score = val_df[prob_cols].values if prob_cols else None

        accuracy = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
        f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0)) if len(y_true) > 0 else 0.0
        auc_ovo_macro = 0.0
        if y_score is not None and len(np.unique(y_true)) > 1:
            try:
                auc_ovo_macro = float(
                    roc_auc_score(
                        y_true,
                        y_score,
                        multi_class='ovo',
                        average='macro',
                        labels=np.arange(y_score.shape[1]),
                    )
                )
            except Exception as exc:
                logging.warning(f"Could not calculate multiclass validation AUROC (MIL image): {exc}")

        validation_optimized = {
            'mode': 'multiclass',
            'num_classes': int(max(detected_num_classes, len(prob_cols) if prob_cols else 1)),
            'optimal_aggregation': {
                'strategy': 'mil_argmax',
                'description': 'Multiclass MIL mode uses argmax over class probabilities; no binary threshold optimization.'
            },
            'validation_multiclass': {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'auc_ovo_macro': auc_ovo_macro,
            }
        }

        if output_dir:
            thresholds_file = output_dir / "validation_thresholds.json"
            with open(thresholds_file, "w") as f:
                json.dump(validation_optimized, f, indent=2)
            logging.info(f"Validation multiclass metrics saved to: {thresholds_file}")
        return validation_optimized



    from sklearn.metrics import roc_auc_score


    if val_df['raw_pred'].min() < 0 or val_df['raw_pred'].max() > 1:
        slide_probs = torch.sigmoid(torch.tensor(val_df['raw_pred'].values)).numpy()
    else:
        slide_probs = val_df['raw_pred'].values


    from src.utils.metrics_utils import find_optimal_threshold
    from sklearn.metrics import confusion_matrix, f1_score

    opt_threshold, sensitivity, specificity = find_optimal_threshold(
        val_df['label'].values, slide_probs
    )


    slide_preds = (slide_probs > opt_threshold).astype(int)
    validation_pred_positive_rate = float(slide_preds.mean())
    validation_label_positive_rate = float(val_df['label'].values.mean())
    tn, fp, fn, tp = confusion_matrix(val_df['label'].values, slide_preds, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = f1_score(val_df['label'].values, slide_preds, zero_division=0)

    optimal_thresholds = {
        'slide': {
            'threshold': float(opt_threshold),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'accuracy': float(accuracy),
            'f1': float(f1),
            'validation_pred_positive_rate': validation_pred_positive_rate,
            'validation_label_positive_rate': validation_label_positive_rate,
        }
    }


    optimal_aggregation = {
        'strategy': 'mil_attention',
        'description': 'Aggregation handled by MIL model attention mechanism'
    }

    validation_auc_metrics = {}

    from sklearn.metrics import roc_auc_score


    if val_df['raw_pred'].min() < 0 or val_df['raw_pred'].max() > 1:
        probs = torch.sigmoid(torch.tensor(val_df['raw_pred'].values)).numpy()
    else:
        probs = val_df['raw_pred'].values

    try:
        validation_auc_metrics['validation_tile_auroc'] = roc_auc_score(
            val_df['label'].values, probs
        )
    except Exception:
        validation_auc_metrics['validation_tile_auroc'] = 0.5


    try:
        validation_auc_metrics['validation_slide_auroc'] = roc_auc_score(
            val_df['label'].values, probs
        )
    except Exception:
        validation_auc_metrics['validation_slide_auroc'] = 0.5

    validation_optimized = {
        **optimal_thresholds,
        "optimal_aggregation": optimal_aggregation,
        # Same config/semantics as calculate_validation_thresholds_graph's adaptive_threshold block
        # -- see that function's inline comment for why collapse_min_rate=0.15 -- so MIL and graph
        # runs get identical adaptive-recalibration behavior at evaluation time (both consumed by
        # _compute_slide_binary_metrics in src/pipelines/evaluate.py).
        "adaptive_threshold": {
            "enabled": False,  # strict protocol: the test set never influences a threshold
            "min_auc": 0.60,
            "collapse_min_rate": 0.15,
            "min_samples": 20,
        },
        **validation_auc_metrics
    }

    if output_dir:
        thresholds_file = output_dir / "validation_thresholds.json"
        with open(thresholds_file, "w") as f:
            json.dump(validation_optimized, f, indent=2)

        logging.info(f"Validation-optimized thresholds saved to: {thresholds_file}")

    return validation_optimized


@register_validation_threshold(('multimodal', 'pov-gigapath', False))
def calculate_validation_thresholds_img_emb(
    model: nn.Module,
    val_dataset = None,
    task: str = "inflammation",
    output_dir: Optional[Path] = None
) -> Dict:
    """Calculate optimal validation thresholds for a multimodal image-embedding model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=dataloader_worker_init_fn,
    )

    predictions = []
    multiclass_detected = False
    detected_num_classes = 1
    with torch.no_grad():
        for images, embeddings, labels, metadata in tqdm(val_loader, desc="Validierungsprediktionen sammeln"):
            images = images.to(device)
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            output = model(images, embeddings)

            if output.dim() > 1 and output.size(-1) > 1:
                multiclass_detected = True
                detected_num_classes = max(detected_num_classes, int(output.size(-1)))
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(probs, dim=1)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)

                for i in range(output.size(0)):
                    pred_class = int(preds[i].item())
                    pred_dict = {
                        "raw_pred": float(output[i, pred_class].detach().cpu().item()),
                        "prob": float(probs[i, pred_class].detach().cpu().item()),
                        "pred_class": pred_class,
                        "label": int(labels[i].item()),
                    }
                    for class_idx in range(output.size(1)):
                        pred_dict[f'prob_class_{class_idx}'] = float(probs[i, class_idx].detach().cpu().item())
                    predictions.append(pred_dict)
                continue

            output = output.squeeze().cpu().item()
            label = labels.squeeze().cpu().item()

            pred_dict = {
                "raw_pred": float(output),
                "label": int(label),
            }
            predictions.append(pred_dict)

    df = pd.DataFrame(predictions)

    if multiclass_detected:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        y_true = df['label'].astype(int).values
        y_pred = df['pred_class'].astype(int).values
        prob_cols = sorted(
            [c for c in df.columns if c.startswith('prob_class_')],
            key=lambda name: int(name.split('_')[-1])
        )
        y_score = df[prob_cols].values if prob_cols else None

        accuracy = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
        f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0)) if len(y_true) > 0 else 0.0
        auc_ovo_macro = 0.0
        if y_score is not None and len(np.unique(y_true)) > 1:
            try:
                auc_ovo_macro = float(
                    roc_auc_score(
                        y_true,
                        y_score,
                        multi_class='ovo',
                        average='macro',
                        labels=np.arange(y_score.shape[1]),
                    )
                )
            except Exception as exc:
                logging.warning(f"Could not calculate multiclass validation AUROC (img_emb): {exc}")

        thresholds = {
            'mode': 'multiclass',
            'num_classes': int(max(detected_num_classes, len(prob_cols) if prob_cols else 1)),
            'optimal_aggregation': {
                'strategy': 'argmax',
                'description': 'Multiclass multimodal mode uses argmax over class probabilities; no binary threshold optimization.'
            },
            'validation_multiclass': {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'auc_ovo_macro': auc_ovo_macro,
            }
        }

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "validation_thresholds.json", "w") as f:
                json.dump(thresholds, f, indent=2)

        return thresholds


    optimal_threshold = 0.5


    thresholds = {
        "optimal_threshold": optimal_threshold,
        "validation_auroc": 0.0
    }

    try:
        from sklearn.metrics import roc_auc_score
        probs = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
        thresholds["validation_auroc"] = roc_auc_score(df['label'].values, probs)
    except Exception:
        thresholds["validation_auroc"] = 0.5

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "validation_thresholds.json", "w") as f:
            json.dump(thresholds, f, indent=2)

    return thresholds


@register_validation_threshold(('multimodal', 'pov-gigapath', True))
def calculate_validation_thresholds_mil_multimodal(
    model: nn.Module,
    val_dataset = None,
    task: str = "inflammation",
    output_dir: Optional[Path] = None
) -> Dict:
    """Calculate optimal validation thresholds for a MIL multimodal model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=dataloader_worker_init_fn,
    )

    predictions = []
    global_idx = 0
    multiclass_detected = False
    detected_num_classes = 1

    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Collecting validation predictions"):

            if len(batch_data) == 3:
                bag_images, bag_embeddings, bag_labels = batch_data
                bag_images = bag_images.to(device)
                bag_embeddings = bag_embeddings.to(device)
                bag_labels = torch.tensor([bag_labels], device=device).float()


                bag_images = bag_images.squeeze(0)
                bag_embeddings = bag_embeddings.squeeze(0)


                num_tiles = bag_images.size(0)
                bag_idx = torch.zeros(num_tiles, dtype=torch.long, device=device)
            else:
                raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")


            outputs, _ = model(bag_images, bag_idx, bag_embeddings)

            if outputs.ndim > 0 and outputs.numel() > 1:
                multiclass_detected = True
                logits = outputs.view(1, -1) if outputs.dim() == 1 else outputs
                probs = torch.softmax(logits, dim=1)
                pred_class = int(torch.argmax(probs, dim=1).item())
                label_val = int(bag_labels.squeeze().item())

                pred_dict = {
                    "raw_pred": float(logits[0, pred_class].item()),
                    "prob": float(probs[0, pred_class].item()),
                    "pred_class": pred_class,
                    "label": label_val,
                    "slide_name": f"bag_{global_idx}",
                    "particle_id": 0,
                }
                detected_num_classes = max(detected_num_classes, int(logits.size(1)))
                for class_idx in range(logits.size(1)):
                    pred_dict[f'prob_class_{class_idx}'] = float(probs[0, class_idx].item())

                if task == "inflammation":
                    pred_dict["inflammation_status"] = "unknown"

                predictions.append(pred_dict)
                global_idx += 1
                continue


            if outputs.ndim == 0:
                output_val = outputs.item()
            else:
                output_val = outputs.squeeze().item()

            label_val = bag_labels.squeeze().item()

            pred_dict = {
                "raw_pred": float(output_val),
                "label": int(label_val),
                "slide_name": f"bag_{global_idx}",
                "particle_id": 0,
            }

            if task == "inflammation":
                pred_dict["inflammation_status"] = "unknown"

            predictions.append(pred_dict)
            global_idx += 1

    val_df = pd.DataFrame(predictions)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    if multiclass_detected:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        y_true = val_df['label'].astype(int).values
        y_pred = val_df['pred_class'].astype(int).values
        prob_cols = sorted(
            [c for c in val_df.columns if c.startswith('prob_class_')],
            key=lambda name: int(name.split('_')[-1])
        )
        y_score = val_df[prob_cols].values if prob_cols else None

        accuracy = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
        f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0)) if len(y_true) > 0 else 0.0
        auc_ovo_macro = 0.0
        if y_score is not None and len(np.unique(y_true)) > 1:
            try:
                auc_ovo_macro = float(
                    roc_auc_score(
                        y_true,
                        y_score,
                        multi_class='ovo',
                        average='macro',
                        labels=np.arange(y_score.shape[1]),
                    )
                )
            except Exception as exc:
                logging.warning(f"Could not calculate multiclass validation AUROC (mil_multimodal): {exc}")

        validation_optimized = {
            'mode': 'multiclass',
            'num_classes': int(max(detected_num_classes, len(prob_cols) if prob_cols else 1)),
            'optimal_aggregation': {
                'strategy': 'mil_argmax',
                'description': 'Multiclass MIL multimodal mode uses argmax over class probabilities; no binary threshold optimization.'
            },
            'validation_multiclass': {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'auc_ovo_macro': auc_ovo_macro,
            }
        }

        if output_dir:
            thresholds_file = output_dir / "validation_thresholds.json"
            with open(thresholds_file, 'w') as f:
                json.dump(validation_optimized, f, indent=2)

        return validation_optimized


    from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
    from src.utils.metrics_utils import find_optimal_threshold


    if val_df['raw_pred'].min() < 0 or val_df['raw_pred'].max() > 1:
        probs = torch.sigmoid(torch.tensor(val_df['raw_pred'].values)).numpy()
    else:
        probs = val_df['raw_pred'].values

    optimal_thresholds = {}
    validation_auc_metrics = {}


    try:
        validation_auc_metrics['validation_tile_auroc'] = roc_auc_score(val_df['label'].values, probs)
    except Exception as e:
        logging.warning(f"Could not calculate validation tile-level AUC: {str(e)}")
        validation_auc_metrics['validation_tile_auroc'] = 0.5

    if task == 'inflammation':

        slide_df = val_df.groupby('slide_name').agg({
            'raw_pred': 'mean',
            'label': 'first'
        }).reset_index()


        if slide_df['raw_pred'].min() < 0 or slide_df['raw_pred'].max() > 1:
            slide_probs = torch.sigmoid(torch.tensor(slide_df['raw_pred'].values)).numpy()
        else:
            slide_probs = slide_df['raw_pred'].values


        opt_threshold, sensitivity, specificity = find_optimal_threshold(
            slide_df['label'].values, slide_probs
        )


        slide_preds = (slide_probs > opt_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(slide_df['label'].values, slide_preds, labels=[0, 1]).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = f1_score(slide_df['label'].values, slide_preds, zero_division=0)

        optimal_thresholds['slide'] = {
            'threshold': float(opt_threshold),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'accuracy': float(accuracy),
            'f1': float(f1)
        }


        try:
            validation_auc_metrics['validation_slide_auroc'] = roc_auc_score(slide_df['label'].values, slide_probs)
        except Exception as e:
            logging.warning(f"Could not calculate validation slide-level AUC: {str(e)}")
            validation_auc_metrics['validation_slide_auroc'] = 0.5

    else:

        particle_df = val_df.groupby(['slide_name', 'particle_id']).agg({
            'raw_pred': 'mean',
            'label': 'first'
        }).reset_index()


        if particle_df['raw_pred'].min() < 0 or particle_df['raw_pred'].max() > 1:
            particle_probs = torch.sigmoid(torch.tensor(particle_df['raw_pred'].values)).numpy()
        else:
            particle_probs = particle_df['raw_pred'].values


        opt_threshold, sensitivity, specificity = find_optimal_threshold(
            particle_df['label'].values, particle_probs
        )


        particle_preds = (particle_probs > opt_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(particle_df['label'].values, particle_preds, labels=[0, 1]).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = f1_score(particle_df['label'].values, particle_preds, zero_division=0)

        optimal_thresholds['particle'] = {
            'threshold': float(opt_threshold),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'accuracy': float(accuracy),
            'f1': float(f1)
        }


        try:
            validation_auc_metrics['validation_particle_auroc'] = roc_auc_score(particle_df['label'].values, particle_probs)
        except Exception as e:
            logging.warning(f"Could not calculate validation particle-level AUC: {str(e)}")
            validation_auc_metrics['validation_particle_auroc'] = 0.5


    from src.utils.metrics_utils import optimize_aggregation_strategy

    optimal_aggregation = optimize_aggregation_strategy(
        df=val_df,
        task=task,
        output_dir=output_dir
    )

    validation_optimized = {
        **optimal_thresholds,
        "optimal_aggregation": optimal_aggregation,
        **validation_auc_metrics
    }


    if output_dir:
        thresholds_file = output_dir / "validation_thresholds.json"
        with open(thresholds_file, 'w') as f:
            json.dump(validation_optimized, f, indent=2)

        logging.info(f"Validation-optimized thresholds saved to: {thresholds_file}")

    return validation_optimized


@register_validation_threshold(('graph', 'pov-gigapath', True))
@register_validation_threshold(('graph', 'pov_gigapath', True))
@register_validation_threshold(('graph', 'uni2', True))
@register_validation_threshold(('graph', 'resnet50', True))
def calculate_validation_thresholds_graph(
    model: nn.Module,
    val_dataset = None,
    task: str = "inflammation",
    output_dir: Optional[Path] = None,
    negative_label_source: str = "annotation_only",
) -> Dict:
    """Calculate optimal validation thresholds for a graph-based model."""
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
    from src.utils.metrics_utils import find_optimal_threshold

    original_device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device('cpu')
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    multiclass_detected = False
    detected_num_classes = 1

    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    predictions = []
    region_probs_list: list[float] = []
    region_labels_list: list[int] = []
    patch_probs_list: list[float] = []
    patch_labels_list: list[int] = []
    try:
        with torch.inference_mode():
            for data in tqdm(val_loader, desc="Collecting validation predictions"):
                data = data.to(device)
                try:
                    model_out = model(data)
                    outputs = _unwrap_graph_logits(model_out)
                except torch.OutOfMemoryError:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    data = data.cpu()
                    model_out = model(data)
                    outputs = _unwrap_graph_logits(model_out)

                label_tensor = data.y.detach().view(-1)
                if outputs.dim() > 1 and outputs.size(-1) > 1:
                    multiclass_detected = True
                    detected_num_classes = max(detected_num_classes, int(outputs.size(-1)))
                    probs = torch.softmax(outputs, dim=1)
                    pred_classes = torch.argmax(probs, dim=1)
                    output_values = outputs.detach().cpu()
                    prob_values = probs.detach().cpu()
                    label_values = label_tensor.cpu().tolist()
                else:
                    output_values = outputs.detach().view(-1).cpu().tolist()
                    label_values = label_tensor.cpu().tolist()

                slide_name_attr = getattr(data, 'slide_name', None)
                if isinstance(slide_name_attr, (list, tuple)):
                    slide_names = [str(name) for name in slide_name_attr]
                elif slide_name_attr is None:
                    slide_names = [f'graph_{len(predictions) + i}' for i in range(len(label_values))]
                else:
                    slide_names = [str(slide_name_attr)] * len(label_values)

                if len(slide_names) != len(label_values):
                    slide_names = [f'graph_{len(predictions) + i}' for i in range(len(label_values))]

                if multiclass_detected and isinstance(output_values, torch.Tensor):
                    for idx, (label, slide_name) in enumerate(zip(label_values, slide_names)):
                        pred_class = int(pred_classes[idx].item())
                        pred_dict = {
                            'raw_pred': float(output_values[idx, pred_class].item()),
                            'prob': float(prob_values[idx, pred_class].item()),
                            'pred_class': pred_class,
                            'label': int(label),
                            'slide_name': slide_name,
                            'particle_id': 0,
                        }
                        for class_idx in range(output_values.size(1)):
                            pred_dict[f'prob_class_{class_idx}'] = float(prob_values[idx, class_idx].item())

                        if task == 'inflammation':
                            pred_dict['inflammation_status'] = 'unknown'

                        predictions.append(pred_dict)
                else:
                    for raw_pred, label, slide_name in zip(output_values, label_values, slide_names):
                        pred_dict = {
                            'raw_pred': float(raw_pred),
                            'label': int(label),
                            'slide_name': slide_name,
                            'particle_id': 0,
                        }

                        if task == 'inflammation':
                            pred_dict['inflammation_status'] = 'unknown'

                        predictions.append(pred_dict)

                region_store = None
                try:
                    region_store = data['region']
                except Exception:
                    region_store = None

                ann = getattr(region_store, 'annotation_y', None) if region_store is not None else None
                region_probs = None
                if isinstance(model_out, dict):
                    candidate_region_probs = model_out.get('region_probs')
                    if torch.is_tensor(candidate_region_probs):
                        region_probs = candidate_region_probs.detach().view(-1).cpu().float()

                if torch.is_tensor(ann) and torch.is_tensor(region_probs):
                    ann_ids = ann.detach().view(-1).cpu().long()
                    if ann_ids.numel() == region_probs.numel():
                        vocab = _extract_annotation_vocab_from_graph_batch(data)
                        mapped = _map_region_annotation_ids_to_binary(ann_ids, vocab)

                        region_batch = getattr(region_store, 'batch', None)
                        region_batch_cpu = (
                            region_batch.detach().view(-1).cpu().long()
                            if torch.is_tensor(region_batch) and region_batch.numel() == ann_ids.numel()
                            else None
                        )
                        slide_labels = getattr(data, 'y', None)
                        if torch.is_tensor(slide_labels):
                            slide_binary = (slide_labels.detach().view(-1).cpu().float() > 0.5).long()
                            meta_entries = _extract_graph_meta_entries_from_graph_batch(data)
                            annotation_enabled = torch.tensor(
                                [bool(m.get('annotation_labels_enabled', False)) for m in (meta_entries or [{}])],
                                dtype=torch.bool,
                            )
                            if annotation_enabled.numel() == slide_binary.numel():
                                mapped = apply_slide_derived_negative_labels(
                                    mapped=mapped,
                                    node_batch=region_batch_cpu,
                                    slide_binary=slide_binary,
                                    annotation_enabled=annotation_enabled,
                                    negative_label_source=negative_label_source,
                                )

                        labeled_mask = mapped >= 0
                        if torch.any(labeled_mask):
                            selected_probs = region_probs[labeled_mask].tolist()
                            selected_labels = mapped[labeled_mask].tolist()
                            region_probs_list.extend(float(v) for v in selected_probs)
                            region_labels_list.extend(int(v) for v in selected_labels)

                patch_store = None
                try:
                    patch_store = data['patch']
                except Exception:
                    patch_store = None

                patch_ann_full = getattr(patch_store, 'annotation_y', None) if patch_store is not None else None
                patch_probs = None
                if isinstance(model_out, dict):
                    candidate_patch_probs = model_out.get('patch_probs')
                    if torch.is_tensor(candidate_patch_probs):
                        patch_probs = candidate_patch_probs.detach().view(-1).cpu().float()

                if torch.is_tensor(patch_ann_full) and torch.is_tensor(patch_probs):
                    patch_ann_full = patch_ann_full.detach().view(-1).cpu().long()
                    n_patch_logits = int(patch_probs.numel())

                    if patch_ann_full.numel() == n_patch_logits:
                        patch_ann_ids = patch_ann_full
                    else:
                        patch_node_idx = model_out.get('patch_node_idx') if isinstance(model_out, dict) else None
                        patch_ann_ids = None
                        if torch.is_tensor(patch_node_idx) and patch_node_idx.numel() == n_patch_logits:
                            idx = patch_node_idx.detach().cpu().long()
                            if idx.numel() == 0 or int(idx.max().item()) < patch_ann_full.numel():
                                patch_ann_ids = patch_ann_full.index_select(0, idx)

                    if patch_ann_ids is not None and patch_ann_ids.numel() == n_patch_logits:
                        patch_vocab = _extract_annotation_vocab_from_graph_batch(data)
                        mapped_patch = _map_region_annotation_ids_to_binary(patch_ann_ids, patch_vocab)

                        patch_batch = getattr(patch_store, 'batch', None)
                        if torch.is_tensor(patch_batch) and patch_batch.numel() == patch_ann_full.numel():
                            patch_batch_cpu = patch_batch.detach().view(-1).cpu().long()
                            if patch_ann_ids is not patch_ann_full:
                                patch_batch_cpu = patch_batch_cpu.index_select(0, idx)
                        else:
                            patch_batch_cpu = None

                        slide_labels = getattr(data, 'y', None)
                        if torch.is_tensor(slide_labels):
                            slide_binary = (slide_labels.detach().view(-1).cpu().float() > 0.5).long()
                            meta_entries = _extract_graph_meta_entries_from_graph_batch(data)
                            annotation_enabled = torch.tensor(
                                [bool(m.get('annotation_labels_enabled', False)) for m in (meta_entries or [{}])],
                                dtype=torch.bool,
                            )
                            if annotation_enabled.numel() == slide_binary.numel():
                                mapped_patch = apply_slide_derived_negative_labels(
                                    mapped=mapped_patch,
                                    node_batch=patch_batch_cpu,
                                    slide_binary=slide_binary,
                                    annotation_enabled=annotation_enabled,
                                    negative_label_source=negative_label_source,
                                )

                        labeled_patch_mask = mapped_patch >= 0
                        if torch.any(labeled_patch_mask):
                            selected_patch_probs = patch_probs[labeled_patch_mask].tolist()
                            selected_patch_labels = mapped_patch[labeled_patch_mask].tolist()
                            patch_probs_list.extend(float(v) for v in selected_patch_probs)
                            patch_labels_list.extend(int(v) for v in selected_patch_labels)

                del data, outputs, label_tensor, slide_name_attr
                if 'model_out' in locals():
                    del model_out
                if 'probs' in locals():
                    del probs
                if 'pred_classes' in locals():
                    del pred_classes
                if 'output_values' in locals():
                    del output_values
                if 'prob_values' in locals():
                    del prob_values
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        model.to(original_device)

    val_df = pd.DataFrame(predictions)

    if val_df.empty:
        raise ValueError("No validation predictions collected for graph dataset")

    labels_np = val_df['label'].astype(int).values
    unique_labels = sorted(pd.Series(labels_np).unique().tolist())
    logging.info(
        "Graph validation class distribution: %s",
        val_df['label'].value_counts().to_dict(),
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    if multiclass_detected:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        y_true = val_df['label'].astype(int).values
        y_pred = val_df['pred_class'].astype(int).values
        prob_cols = sorted(
            [c for c in val_df.columns if c.startswith('prob_class_')],
            key=lambda name: int(name.split('_')[-1])
        )
        y_score = val_df[prob_cols].values if prob_cols else None

        accuracy = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
        f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0)) if len(y_true) > 0 else 0.0
        auc_ovo_macro = 0.0
        if y_score is not None and len(np.unique(y_true)) > 1:
            try:
                auc_ovo_macro = float(
                    roc_auc_score(
                        y_true,
                        y_score,
                        multi_class='ovo',
                        average='macro',
                        labels=np.arange(y_score.shape[1]),
                    )
                )
            except Exception as exc:
                logging.warning(f"Could not calculate multiclass validation AUROC (graph): {exc}")

        validation_optimized = {
            'mode': 'multiclass',
            'num_classes': int(max(detected_num_classes, len(prob_cols) if prob_cols else 1)),
            'optimal_aggregation': {
                'strategy': 'graph_argmax',
                'description': 'Multiclass graph mode uses argmax over class probabilities; no binary threshold optimization.'
            },
            'validation_multiclass': {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'auc_ovo_macro': auc_ovo_macro,
            }
        }

        if output_dir:
            thresholds_file = output_dir / "validation_thresholds.json"
            with open(thresholds_file, 'w') as f:
                json.dump(validation_optimized, f, indent=2)
            logging.info(f"Validation multiclass metrics saved to: {thresholds_file}")
        return validation_optimized

    if val_df['raw_pred'].min() < 0 or val_df['raw_pred'].max() > 1:
        slide_probs = torch.sigmoid(torch.tensor(val_df['raw_pred'].values)).numpy()
    else:
        slide_probs = val_df['raw_pred'].values

    opt_threshold, sensitivity, specificity = find_optimal_threshold(
        labels_np, slide_probs
    )

    slide_preds = (slide_probs > opt_threshold).astype(int)
    validation_pred_positive_rate = float(slide_preds.mean())
    validation_label_positive_rate = float(labels_np.mean())
    tn, fp, fn, tp = confusion_matrix(labels_np, slide_preds, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = f1_score(labels_np, slide_preds, zero_division=0)

    optimal_thresholds = {
        'slide': {
            'threshold': float(opt_threshold),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'accuracy': float(accuracy),
            'f1': float(f1),
            'validation_pred_positive_rate': validation_pred_positive_rate,
            'validation_label_positive_rate': validation_label_positive_rate,
        }
    }

    optimal_aggregation = {
        'strategy': 'graph_pooling',
        'description': 'Aggregation handled by graph neural network pooling layers'
    }

    validation_auc_metrics = {}

    if len(unique_labels) <= 1:
        logging.warning(
            "Only one class present in graph validation labels (%s). Setting validation_slide_auroc=0.5",
            unique_labels,
        )
        validation_auc_metrics['validation_slide_auroc'] = 0.5
    else:
        try:
            validation_auc_metrics['validation_slide_auroc'] = roc_auc_score(labels_np, slide_probs)
        except Exception as e:
            logging.warning(f"Could not calculate validation slide-level AUC: {str(e)}")
            validation_auc_metrics['validation_slide_auroc'] = 0.5

    validation_optimized = {
        **optimal_thresholds,
        "optimal_aggregation": optimal_aggregation,
        "adaptive_threshold": {
            "enabled": False,  # strict protocol: the test set never influences a threshold
            "min_auc": 0.60,
            # 0.01 (i.e. only pred_pos_rate<=1% or >=99%) missed severe-but-not-total
            # calibration collapse on GSDB test_scanner2 -- e.g. 95.5%/96.6% positive-prediction
            # rate at the validation-optimized threshold (2026-08-11, post region-structure-fix
            # evaluation) left the badly-shifted validation threshold in place instead of
            # rethresholding, producing Specificity 0.07-0.10 / Balanced Accuracy ~0.52-0.54 even
            # though the same predictions' oracle-threshold sweep showed Balanced Accuracy
            # 0.67-0.72 was available. 0.15 (trigger at >=85%/<=15%) still comfortably clears
            # test_scanner2's true label positive rate (~66%, the "expected" imbalance a model
            # predicting close to the real class balance would show) while catching this kind of
            # near-total collapse.
            "collapse_min_rate": 0.15,
            "min_samples": 20,
        },
        **validation_auc_metrics
    }

    if len(region_probs_list) > 0 and len(region_probs_list) == len(region_labels_list):
        y_region_prob = np.asarray(region_probs_list, dtype=np.float32)
        y_region_true = np.asarray(region_labels_list, dtype=np.int64)
        region_best_thr = 0.5
        region_best_f1 = -1.0

        from sklearn.metrics import f1_score

        for thr in np.linspace(0.05, 0.95, 91):
            y_region_pred = (y_region_prob >= float(thr)).astype(np.int64)
            cur_f1 = float(f1_score(y_region_true, y_region_pred, zero_division=0))
            if cur_f1 > region_best_f1:
                region_best_f1 = cur_f1
                region_best_thr = float(thr)

        region_pos_rate = float(np.mean(y_region_true)) if y_region_true.size > 0 else 0.0
        # Sensitivity/specificity at the F1-optimal threshold -- previously only 'f1' was
        # reported here (unlike the slide-level block above, which reports all four), so
        # evaluate.py's "Using validation-optimized thresholds" summary always printed "n/a"
        # for Region/Patch Sensitivity/Specificity regardless of what the data actually showed.
        region_sensitivity = None
        region_specificity = None
        if len(np.unique(y_region_true)) > 1:
            y_region_pred_best = (y_region_prob >= region_best_thr).astype(np.int64)
            tn_r, fp_r, fn_r, tp_r = confusion_matrix(y_region_true, y_region_pred_best, labels=[0, 1]).ravel()
            region_sensitivity = float(tp_r / (tp_r + fn_r)) if (tp_r + fn_r) > 0 else None
            region_specificity = float(tn_r / (tn_r + fp_r)) if (tn_r + fp_r) > 0 else None
        validation_optimized['region'] = {
            'threshold': float(region_best_thr),
            'f1': float(max(0.0, region_best_f1)),
            'sensitivity': region_sensitivity,
            'specificity': region_specificity,
            'num_labeled_regions': int(y_region_true.size),
            'validation_label_positive_rate': region_pos_rate,
        }

    if len(patch_probs_list) > 0 and len(patch_probs_list) == len(patch_labels_list):
        y_patch_prob = np.asarray(patch_probs_list, dtype=np.float32)
        y_patch_true = np.asarray(patch_labels_list, dtype=np.int64)
        patch_best_thr = 0.5
        patch_best_f1 = -1.0

        from sklearn.metrics import f1_score

        for thr in np.linspace(0.05, 0.95, 91):
            y_patch_pred = (y_patch_prob >= float(thr)).astype(np.int64)
            cur_f1 = float(f1_score(y_patch_true, y_patch_pred, zero_division=0))
            if cur_f1 > patch_best_f1:
                patch_best_f1 = cur_f1
                patch_best_thr = float(thr)

        patch_pos_rate = float(np.mean(y_patch_true)) if y_patch_true.size > 0 else 0.0
        # See the region block above for why sensitivity/specificity are now computed here too.
        patch_sensitivity = None
        patch_specificity = None
        if len(np.unique(y_patch_true)) > 1:
            y_patch_pred_best = (y_patch_prob >= patch_best_thr).astype(np.int64)
            tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_patch_true, y_patch_pred_best, labels=[0, 1]).ravel()
            patch_sensitivity = float(tp_p / (tp_p + fn_p)) if (tp_p + fn_p) > 0 else None
            patch_specificity = float(tn_p / (tn_p + fp_p)) if (tn_p + fp_p) > 0 else None
        validation_optimized['patch'] = {
            'threshold': float(patch_best_thr),
            'f1': float(max(0.0, patch_best_f1)),
            'sensitivity': patch_sensitivity,
            'specificity': patch_specificity,
            'num_labeled_patches': int(y_patch_true.size),
            'validation_label_positive_rate': patch_pos_rate,
        }


    if output_dir:
        thresholds_file = output_dir / "validation_thresholds.json"
        with open(thresholds_file, 'w') as f:
            json.dump(validation_optimized, f, indent=2)

        logging.info(f"Validation-optimized thresholds and aggregation saved to: {thresholds_file}")

    return validation_optimized



calculate_validation_thresholds = calculate_validation_thresholds_image
