"""
Simplified metrics utility functions for histology classification evaluation.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve
)
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
from torch.utils.data import DataLoader
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime
import torch
import sys
import torch.nn as nn
from tqdm import tqdm

from src.dataset.histology_dataset import HistologyDataset
from src.models.model_loader import load_model
from src.utils.model_utils import get_transforms

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def calculate_metrics(y_true, y_pred, y_scores=None, prefix=''):
    """Calculate binary classification metrics (accuracy, sensitivity, specificity, precision, F1, and optionally AUROC)."""
    metrics = {}


    if len(y_true) == 0 or len(y_pred) == 0:
        metrics[f'{prefix}accuracy'] = 0
        metrics[f'{prefix}sensitivity'] = 0
        metrics[f'{prefix}specificity'] = 0
        metrics[f'{prefix}precision'] = 0
        metrics[f'{prefix}f1'] = 0
        if y_scores is not None:
            metrics[f'{prefix}auroc'] = 0.5
        return metrics


    unique_classes = np.unique(y_true)
    if len(unique_classes) <= 1:

        if unique_classes[0] == 1:
            tp = np.sum(y_pred == 1)
            fn = np.sum(y_pred == 0)
            fp = 0
            tn = 0
        else:
            tp = 0
            fn = 0
            fp = np.sum(y_pred == 1)
            tn = np.sum(y_pred == 0)


        metrics[f'{prefix}accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        metrics[f'{prefix}sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics[f'{prefix}specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics[f'{prefix}precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics[f'{prefix}f1'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        if y_scores is not None:
            metrics[f'{prefix}auroc'] = 0.5

        return metrics


    try:

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()


        metrics[f'{prefix}accuracy'] = accuracy_score(y_true, y_pred)
        metrics[f'{prefix}sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics[f'{prefix}specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics[f'{prefix}precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics[f'{prefix}f1'] = f1_score(y_true, y_pred, zero_division=0)


        if y_scores is not None:
            try:
                metrics[f'{prefix}auroc'] = roc_auc_score(y_true, y_scores)
            except ValueError as e:
                logging.warning(f"Could not calculate AUROC: {str(e)}")
                metrics[f'{prefix}auroc'] = 0.5
    except Exception as e:
        logging.warning(f"Error calculating metrics: {str(e)}")

        metrics[f'{prefix}accuracy'] = 0
        metrics[f'{prefix}sensitivity'] = 0
        metrics[f'{prefix}specificity'] = 0
        metrics[f'{prefix}precision'] = 0
        metrics[f'{prefix}f1'] = 0
        if y_scores is not None:
            metrics[f'{prefix}auroc'] = 0.5

    return metrics

def find_optimal_threshold(y_true: List, y_prob: List) -> Tuple[float, float, float]:
    """Find the optimal classification threshold using the geometric mean of sensitivity and specificity."""

    logging.info(f"Finding optimal threshold for {len(y_true)} samples")
    logging.info(f"Class distribution: {np.bincount(np.array(y_true, dtype=int))}")
    logging.info(f"Predictions range: min={min(y_prob):.4f}, max={max(y_prob):.4f}")


    if len(y_true) == 0 or len(np.unique(y_true)) <= 1:
        logging.warning("find_optimal_threshold: Empty input or only one class present. Using default threshold of 0.5")
        return 0.5, 0.0, 0.0


    if max(y_prob) == min(y_prob):
        logging.warning(f"All predictions have the same value: {max(y_prob):.4f}. Using default threshold of 0.5")
        return 0.5, 0.0, 0.0


    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)


    if np.isnan(y_prob_np).any() or np.isinf(y_prob_np).any():
        logging.warning("NaN or Inf values detected in predictions. Cleaning data.")

        y_prob_np = np.nan_to_num(y_prob_np, nan=0.5, posinf=1.0, neginf=0.0)


    try:
        auc_score = roc_auc_score(y_true_np, y_prob_np)
        logging.info(f"ROC AUC: {auc_score:.4f}")

        pos_probs = y_prob_np[y_true_np == 1]
        neg_probs = y_prob_np[y_true_np == 0]
        max_neg = float(np.max(neg_probs))
        min_pos = float(np.min(pos_probs))

        if max_neg < min_pos:
            optimal_threshold = float(np.clip((max_neg + min_pos) / 2.0, 0.02, 0.98))
            predictions = (y_prob_np > optimal_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true_np, predictions, labels=[0, 1]).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            logging.info(
                f"Validation set perfectly separable (max_neg={max_neg:.4f} < min_pos={min_pos:.4f}); "
                f"using max-margin threshold: {optimal_threshold:.4f}"
            )
            return optimal_threshold, sensitivity, specificity

        fpr, tpr, thresholds = roc_curve(y_true_np, y_prob_np)

        n_pos = int(np.sum(y_true_np == 1))
        n_neg = int(np.sum(y_true_np == 0))
        tpr = np.array([
            np.sum((y_prob_np > thr) & (y_true_np == 1)) / n_pos if n_pos > 0 else 0.0
            for thr in thresholds
        ])
        fpr = np.array([
            np.sum((y_prob_np > thr) & (y_true_np == 0)) / n_neg if n_neg > 0 else 0.0
            for thr in thresholds
        ])
        gmeans = np.sqrt(tpr * (1-fpr))

        if len(gmeans) > 0:
            max_gmean = np.max(gmeans)
            tied_indices = np.where(np.isclose(gmeans, max_gmean))[0]
            ix = tied_indices[len(tied_indices) // 2]
            optimal_threshold = thresholds[ix]


            predictions = (y_prob_np > optimal_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true_np, predictions, labels=[0, 1]).ravel()


            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0


            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

            logging.info(f"Found optimal threshold: {optimal_threshold:.4f}")
            logging.info(f"Metrics at this threshold: Accuracy={accuracy:.4f}, Sens={sensitivity:.4f}, Spec={specificity:.4f}, F1={f1:.4f}")
            logging.info(f"Confusion matrix: [TP={tp}, TN={tn}, FP={fp}, FN={fn}]")
        else:

            logging.warning("Empty gmeans array in find_optimal_threshold. Using default threshold of 0.5")
            optimal_threshold = 0.5
            sensitivity = 0.0
            specificity = 0.0

    except Exception as e:
        logging.error(f"Error in ROC curve calculation: {str(e)}")
        logging.warning("Defaulting to threshold of 0.5")
        return 0.5, 0.0, 0.0


    if optimal_threshold < 0.02:
        logging.warning(f"Threshold too low ({optimal_threshold:.4f}), adjusting to 0.02")
        optimal_threshold = 0.02

        predictions = (y_prob_np > optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_np, predictions, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    elif optimal_threshold > 0.98:
        logging.warning(f"Threshold too high ({optimal_threshold:.4f}), adjusting to 0.98")
        optimal_threshold = 0.98

        predictions = (y_prob_np > optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_np, predictions, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return optimal_threshold, sensitivity, specificity

def calculate_basic_metrics(
    labels: List,
    preds: List,
    raw_preds: Optional[List] = None
) -> Dict:
    """Calculate basic classification metrics (accuracy, sensitivity, specificity, precision, F1, and optionally AUC)."""
    metrics = {}


    labels = np.array(labels)
    # Ground truth must be strictly binary {0,1} for confusion_matrix/roc_auc_score below.
    # Soft/continuous labels can reach here (e.g. graph_mixup's lambda-blended slide label,
    # e.g. 0.62 instead of 0 or 1) and sklearn raises "mix of continuous and binary target"
    # rather than silently handling it. Round to the nearest class instead of crashing --
    # already-binary labels (0.0/1.0) round to themselves, so this is a no-op without mixup.
    labels = np.rint(labels).astype(int)
    preds = np.array(preds)


    if len(labels) == 0 or len(preds) == 0:
        logging.warning("Empty labels or predictions array")
        metrics['accuracy'] = 0
        metrics['sensitivity'] = 0
        metrics['specificity'] = 0
        metrics['precision'] = 0
        metrics['f1'] = 0
        if raw_preds is not None:
            metrics['auc'] = 50
        return metrics


    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:

        if unique_labels[0] == 1:

            tp = np.sum(preds == 1)
            fn = np.sum(preds == 0)
            fp = 0
            tn = 0
        else:

            tn = np.sum(preds == 0)
            fp = np.sum(preds == 1)
            tp = 0
            fn = 0
    else:

        cm = confusion_matrix(labels, preds)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:

            logging.warning(f"Unexpected confusion matrix shape: {cm.shape}")
            metrics['accuracy'] = np.mean(labels == preds) * 100
            metrics['sensitivity'] = 0
            metrics['specificity'] = 0
            metrics['precision'] = 0
            metrics['f1'] = 0
            if raw_preds is not None:
                metrics['auc'] = 50
            return metrics


    total = tp + tn + fp + fn
    metrics['accuracy'] = 100 * (tp + tn) / total if total > 0 else 0


    if tp + fn > 0:
        metrics['sensitivity'] = 100 * tp / (tp + fn)
    else:
        metrics['sensitivity'] = 0


    if tn + fp > 0:
        metrics['specificity'] = 100 * tn / (tn + fp)
    else:
        metrics['specificity'] = 0


    if tp + fp > 0:
        metrics['precision'] = 100 * tp / (tp + fp)
    else:
        metrics['precision'] = 0


    if tp == 0 and (fp == 0 or fn == 0):

        metrics['f1'] = 0
    else:

        denominator = 2 * tp + fp + fn
        metrics['f1'] = 100 * (2 * tp) / denominator if denominator > 0 else 0


    if raw_preds is not None:
        scores = np.asarray(raw_preds).reshape(-1)
        label_arr = np.asarray(labels).reshape(-1)

        n = min(scores.size, label_arr.size)
        scores = scores[:n]
        label_arr = label_arr[:n]

        finite_mask = np.isfinite(scores) & np.isfinite(label_arr)
        if not np.all(finite_mask):
            dropped = int((~finite_mask).sum())
            logging.warning(
                "AUC: dropping %d non-finite prediction/label entries before roc_auc_score.",
                dropped,
            )

        scores = scores[finite_mask]
        label_arr = label_arr[finite_mask]

        if scores.size == 0 or np.unique(label_arr).size < 2:
            metrics['auc'] = 50
        else:
            try:
                metrics['auc'] = 100 * roc_auc_score(label_arr, scores)
            except ValueError as e:
                logging.warning(f"Could not calculate AUC: {str(e)}")
                metrics['auc'] = 50

    return metrics

def calculate_hierarchical_metrics(
    df: pd.DataFrame,
    split: str,
    model_name: str,
    threshold: float = 0.0
) -> Dict:
    """Calculate tile-level metrics plus slide- or particle-level metrics, depending on the task."""
    metrics = {}


    is_inflammation_task = 'inflammation_status' in df.columns or any('inflammation' in col for col in df.columns)


    is_logits = df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1
    logging.info(f"Raw predictions range: min={df['raw_pred'].min():.4f}, max={df['raw_pred'].max():.4f}")
    logging.info(f"Predictions appear to be {'logits' if is_logits else 'probabilities'}")
    logging.info(f"Using threshold: {threshold:.4f} in {'logit' if is_logits else 'probability'} space")


    tile_preds = (df['raw_pred'] > threshold).astype(int)


    total_pos = tile_preds.sum()
    total_neg = len(tile_preds) - total_pos
    logging.info(f"Binary predictions: {total_pos} positive, {total_neg} negative out of {len(tile_preds)} total")



    if is_logits:
        scores = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
        logging.info(f"Converted logits to probabilities for AUC calculation")
    else:
        scores = df['raw_pred'].values

    tile_metrics = calculate_metrics(
        y_true=df['label'],
        y_pred=tile_preds,
        y_scores=scores,
        prefix='tile_'
    )
    metrics.update(tile_metrics)


    if is_inflammation_task:
        slide_df = df.groupby('slide_name').agg({
            'raw_pred': 'mean',
            'label': 'first'
        })


        is_slide_logits = slide_df['raw_pred'].min() < 0 or slide_df['raw_pred'].max() > 1
        logging.info(f"Slide-level raw predictions range: min={slide_df['raw_pred'].min():.4f}, max={slide_df['raw_pred'].max():.4f}")

        slide_preds = (slide_df['raw_pred'] > threshold).astype(int)


        slide_pos = slide_preds.sum()
        slide_neg = len(slide_preds) - slide_pos
        logging.info(f"Slide-level binary predictions: {slide_pos} positive, {slide_neg} negative out of {len(slide_preds)} total")


        if is_slide_logits:
            slide_scores = torch.sigmoid(torch.tensor(slide_df['raw_pred'].values)).numpy()
            logging.info(f"Converted slide-level logits to probabilities for AUC calculation")
        else:
            slide_scores = slide_df['raw_pred'].values

        slide_metrics = calculate_metrics(
            y_true=slide_df['label'],
            y_pred=slide_preds,
            y_scores=slide_scores,
            prefix='slide_'
        )
        metrics.update(slide_metrics)


    else:
        particle_df = df.groupby(['slide_name', 'particle_id']).agg({
            'raw_pred': 'mean',
            'label': 'first'
        })

        particle_preds = (particle_df['raw_pred'] > threshold).astype(int)

        if particle_df['raw_pred'].min() < 0:
            particle_scores = torch.sigmoid(torch.tensor(particle_df['raw_pred'].values)).numpy()
        else:
            particle_scores = particle_df['raw_pred'].values

        particle_metrics = calculate_metrics(
            y_true=particle_df['label'],
            y_pred=particle_preds,
            y_scores=particle_scores,
            prefix='particle_'
        )
        metrics.update(particle_metrics)


    metrics.update({
        'split': split,
        'model_name': model_name,
        'threshold': threshold,
        'total_samples': len(df),
        'task': 'inflammation' if is_inflammation_task else 'tissue',
        'predictions_df': df
    })

    return metrics

def optimize_hierarchical_thresholds(df: pd.DataFrame, task: str = 'inflammation', output_dir: Optional[Path] = None) -> Dict:
    """Find optimal classification thresholds for each hierarchical level, using validation data only."""
    results = {}


    if 'raw_pred' in df.columns:

        if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
            tile_probs = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
        else:
            tile_probs = df['raw_pred'].values


        opt_threshold, sensitivity, specificity = find_optimal_threshold(
            df['label'].values, tile_probs
        )


        tile_preds = (tile_probs > opt_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(df['label'].values, tile_preds, labels=[0, 1]).ravel()
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        f1 = f1_score(df['label'].values, tile_preds, zero_division=0)

        results['tile'] = {
            'threshold': float(opt_threshold),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'accuracy': float(accuracy),
            'f1': float(f1)
        }


        if output_dir:
            visualize_threshold_optimization(
                df['label'].values,
                tile_probs,
                opt_threshold,
                'tile',
                output_dir
            )


    if task == 'inflammation':

        slide_df = df.groupby('slide_name').agg({
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
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        f1 = f1_score(slide_df['label'].values, slide_preds, zero_division=0)

        results['slide'] = {
            'threshold': float(opt_threshold),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'accuracy': float(accuracy),
            'f1': float(f1)
        }


        if output_dir:
            visualize_threshold_optimization(
                slide_df['label'].values,
                slide_probs,
                opt_threshold,
                'slide',
                output_dir
            )

    else:

        particle_df = df.groupby(['slide_name', 'particle_id']).agg({
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
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        f1 = f1_score(particle_df['label'].values, particle_preds, zero_division=0)

        results['particle'] = {
            'threshold': float(opt_threshold),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'accuracy': float(accuracy),
            'f1': float(f1)
        }


        if output_dir:
            visualize_threshold_optimization(
                particle_df['label'].values,
                particle_probs,
                opt_threshold,
                'particle',
                output_dir
            )

    return results

def optimize_aggregation_strategy(df: pd.DataFrame, task: str = 'inflammation',
                                output_dir: Optional[Path] = None) -> Dict:
    """Find the optimal tile-aggregation strategy for hierarchical predictions, using validation data only."""
    import torch



    if 'prob' not in df.columns:
        if 'raw_pred' in df.columns:
            if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
                logging.info("Converting raw logits to probabilities with sigmoid in optimize_aggregation_strategy")
                df['prob'] = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
            else:
                logging.info("Raw predictions are already in probability range [0,1]")
                df['prob'] = df['raw_pred'].copy()
        else:
            logging.warning("No raw_pred column found in dataframe for optimize_aggregation_strategy")
            df['prob'] = 0.5


    aggregation_strategies = {
        'mean': lambda x: np.mean(x),
        'median': lambda x: np.median(x),
        'top_k_mean_10': lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*0.1)):]) if len(x) > 0 else 0,
        'top_k_mean_20': lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*0.2)):]) if len(x) > 0 else 0,
        'top_k_mean_30': lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*0.3)):]) if len(x) > 0 else 0,
    }


    results = {}


    if task == 'inflammation':

        for strategy_name, agg_func in aggregation_strategies.items():

            slide_agg_df = df.groupby('slide_name').agg({
                'prob': agg_func,
                'label': 'first'
            }).reset_index()


            opt_threshold, sensitivity, specificity = find_optimal_threshold(
                slide_agg_df['label'].values, slide_agg_df['prob'].values
            )


            slide_preds = (slide_agg_df['prob'] > opt_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(slide_agg_df['label'].values, slide_preds, labels=[0, 1]).ravel()


            total = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total > 0 else 0


            f1 = f1_score(slide_agg_df['label'].values, slide_preds, zero_division=0)


            results[strategy_name] = {
                'threshold': float(opt_threshold),
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'accuracy': float(accuracy),
                'f1': float(f1),
                'balanced_acc': float((sensitivity + specificity) / 2),
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            }

    else:

        for strategy_name, agg_func in aggregation_strategies.items():

            particle_agg_df = df.groupby(['slide_name', 'particle_id']).agg({
                'prob': agg_func,
                'label': 'first'
            }).reset_index()


            opt_threshold, sensitivity, specificity = find_optimal_threshold(
                particle_agg_df['label'].values, particle_agg_df['prob'].values
            )


            particle_preds = (particle_agg_df['prob'] > opt_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(particle_agg_df['label'].values, particle_preds, labels=[0, 1]).ravel()


            total = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total > 0 else 0


            f1 = f1_score(particle_agg_df['label'].values, particle_preds, zero_division=0)


            results[strategy_name] = {
                'threshold': float(opt_threshold),
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'accuracy': float(accuracy),
                'f1': float(f1),
                'balanced_acc': float((sensitivity + specificity) / 2),
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            }



    best_strategy = max(results.items(), key=lambda x: x[1]['balanced_acc'])


    logging.info("\nAggregation Strategy Comparison:")
    logging.info("--------------------------------")

    sorted_strategies = sorted(results.items(), key=lambda x: x[1]['balanced_acc'], reverse=True)

    for strategy_name, metrics in sorted_strategies:
        logging.info(f"{strategy_name}:")
        logging.info(f"  F1: {metrics['f1']:.4f}, Sens: {metrics['sensitivity']:.4f}, " +
                   f"Spec: {metrics['specificity']:.4f}, Acc: {metrics['accuracy']:.4f}")


    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        level = 'slide' if task == 'inflammation' else 'particle'


        plt.figure(figsize=(14, 8))
        strategies = list(results.keys())


        metric_names = ['F1 Score', 'Sensitivity', 'Specificity', 'Balanced Acc', 'Accuracy']
        metric_keys = ['f1', 'sensitivity', 'specificity', 'balanced_acc', 'accuracy']


        strategies_to_plot = {}
        for strategy in strategies:
            strategies_to_plot[strategy] = [results[strategy][key] for key in metric_keys]


        x = np.arange(len(metric_names))
        width = 0.15
        if len(strategies) > 5:
            width = 0.12
        elif len(strategies) < 4:
            width = 0.2

        multiplier = 0


        best_strategy_name = best_strategy[0]
        colors = {}
        for strategy in strategies:
            if strategy == best_strategy_name:
                colors[strategy] = 'darkred'
            else:
                colors[strategy] = None

        for strategy, values in strategies_to_plot.items():
            offset = width * multiplier
            plt.bar(x + offset, values, width, label=strategy,
                    color=colors[strategy], alpha=0.75 if strategy == best_strategy_name else 0.65)
            multiplier += 1

        plt.ylabel('Score')
        plt.title(f'Aggregation Strategies by Metric for {level.capitalize()} Level')
        plt.xticks(x + width*(len(strategies)-1)/2, metric_names, rotation=30, ha='right')


        if len(strategies) > 1:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    ncol=min(5, len(strategies)),
                    title=f"★ Best Strategy: {best_strategy_name} ★")
        else:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))


        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()


        plt.savefig(output_dir / f'{level}_aggregation_comparison.png', bbox_inches='tight')
        plt.close()

    return {
        'best_strategy': best_strategy[0],
        'best_metrics': best_strategy[1],
        'all_results': results
    }

def visualize_threshold_optimization(y_true: List, y_prob: List, optimal_threshold: float,
                                   level: str, output_dir: Path) -> None:
    """Plot accuracy/sensitivity/specificity/F1/gmean against threshold and mark the optimal threshold."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    plt.figure(figsize=(12, 8))


    thresholds = np.linspace(0.1, 0.9, 100)
    metrics = {
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'f1': [],
        'gmean': []
    }


    for threshold in thresholds:
        preds = (y_prob > threshold).astype(int)


        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()


        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0


        if tp + fp + fn == 0:
            f1 = 0
        else:
            f1 = 2 * tp / (2 * tp + fp + fn)


        gmean = np.sqrt(sensitivity * specificity)


        metrics['accuracy'].append(accuracy)
        metrics['sensitivity'].append(sensitivity)
        metrics['specificity'].append(specificity)
        metrics['f1'].append(f1)
        metrics['gmean'].append(gmean)


    for metric_name, values in metrics.items():
        plt.plot(thresholds, values, label=metric_name.capitalize())


    plt.axvline(x=optimal_threshold, color='r', linestyle='--',
               label=f'Optimal ({optimal_threshold:.3f})')


    plt.title(f'Metrics vs Threshold for {level.capitalize()} Level')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)


    plt.savefig(output_dir / f'{level}_threshold_optimization.png')
    plt.close()

def save_metrics(metrics: Dict, output_dir: Path) -> None:
    """Save evaluation metrics to a JSON file and create a human-readable summary."""

    metrics_copy = {}


    for key, value in metrics.items():
        if hasattr(value, 'to_dict'):
            if hasattr(value, 'to_records'):
                metrics_copy[key] = value.to_dict(orient='records')
            else:
                metrics_copy[key] = value.to_dict()
        elif isinstance(value, dict):

            metrics_copy[key] = {}
            for sub_key, sub_value in value.items():
                if hasattr(sub_value, 'to_dict'):
                    if hasattr(sub_value, 'to_records'):
                        metrics_copy[key][sub_key] = sub_value.to_dict(orient='records')
                    else:
                        metrics_copy[key][sub_key] = sub_value.to_dict()
                else:
                    metrics_copy[key][sub_key] = sub_value
        else:
            metrics_copy[key] = value


    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if 'optimal_aggregation' in metrics and 'test_confusion_matrix' in metrics['optimal_aggregation']:
        optimal_agg_auc = metrics['optimal_aggregation']['test_confusion_matrix'].get('auc', 0)


        if metrics.get('task') == 'inflammation' and 'slide_auroc' in metrics_copy:
            metrics_copy['slide_auroc'] = optimal_agg_auc
            logging.info(f"Updated slide_auroc in statistics.json to match optimal aggregation: {optimal_agg_auc:.4f}")


        elif metrics.get('task') == 'tissue' and 'particle_auroc' in metrics_copy:
            metrics_copy['particle_auroc'] = optimal_agg_auc
            logging.info(f"Updated particle_auroc in statistics.json to match optimal aggregation: {optimal_agg_auc:.4f}")



    stats_path = output_dir / "statistics.json"

    with open(stats_path, 'w') as f:
        json.dump(metrics_copy, f, indent=2)

    logging.info(f"Metrics saved to: {stats_path}")


    if 'predictions_df' in metrics:
        predictions_path = output_dir / "predictions.csv"
        metrics['predictions_df'].to_csv(predictions_path, index=False)
        logging.info(f"Predictions saved to: {predictions_path}")


    summary_path = output_dir / "metrics_summary.txt"
    create_summary_file(metrics, summary_path)

def plot_roc_curves(df: pd.DataFrame, task: str, output_dir: Path,
                   optimal_aggregation: Optional[Dict] = None,
                   metrics: Optional[Dict] = None):
    """Plot ROC curves using consistent AUC values from the metrics dictionary."""
    from sklearn.metrics import roc_curve, auc


    output_dir.mkdir(parents=True, exist_ok=True)


    use_precalculated = False
    if metrics and 'tile_auroc' in metrics:
        use_precalculated = True
        logging.info(f"Will use pre-calculated metrics: tile_auroc={metrics['tile_auroc']:.4f}")
    else:
        logging.warning("No pre-calculated metrics found, will calculate AUC values")


    agg_strategy = None
    if optimal_aggregation and 'best_strategy' in optimal_aggregation:
        agg_strategy = optimal_aggregation['best_strategy']


    if agg_strategy == 'median':
        agg_func = np.median
    elif agg_strategy == 'top_k_mean_10':
        agg_func = lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*0.1)):]) if len(x) > 0 else 0
    elif agg_strategy == 'top_k_mean_20':
        agg_func = lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*0.2)):]) if len(x) > 0 else 0
    elif agg_strategy == 'top_k_mean_30':
        agg_func = lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*0.3)):]) if len(x) > 0 else 0
    else:
        agg_func = np.mean


    if 'prob' not in df.columns:
        if 'raw_pred' in df.columns and (df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1):
            df['prob'] = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
        else:
            df['prob'] = df['raw_pred'].values


    tile_fpr, tile_tpr, _ = roc_curve(df['label'], df['prob'])


    if use_precalculated:
        tile_auc = metrics['tile_auroc']
        logging.info(f"Using pre-calculated tile AUC: {tile_auc:.4f}")
    else:
        tile_auc = auc(tile_fpr, tile_tpr)
        logging.info(f"Calculated tile AUC: {tile_auc:.4f} for ROC curve")


    plt.figure(figsize=(10, 8))


    plt.plot(tile_fpr, tile_tpr,
             label=f'Tile-level (AUC = {tile_auc:.3f})',
             linewidth=2)


    if task == 'inflammation':

        slide_df = df.groupby('slide_name').agg({
            'prob': agg_func,
            'label': 'first'
        }).reset_index()

        slide_fpr, slide_tpr, _ = roc_curve(slide_df['label'], slide_df['prob'])





        if use_precalculated:
            if (optimal_aggregation and
                'test_confusion_matrix' in optimal_aggregation and
                'auc' in optimal_aggregation['test_confusion_matrix']):
                slide_auc = optimal_aggregation['test_confusion_matrix']['auc']
                logging.info(f"Using optimal aggregation AUC: {slide_auc:.4f}")
            elif 'slide_auroc' in metrics:
                slide_auc = metrics['slide_auroc']
                logging.info(f"Using pre-calculated slide AUC: {slide_auc:.4f}")
            else:
                slide_auc = auc(slide_fpr, slide_tpr)
                logging.info(f"Calculated slide AUC: {slide_auc:.4f} for ROC curve")
        else:
            slide_auc = auc(slide_fpr, slide_tpr)
            logging.info(f"Calculated slide AUC: {slide_auc:.4f} for ROC curve")

        strategy_label = f" using {agg_strategy}" if agg_strategy else ""
        plt.plot(slide_fpr, slide_tpr,
                 label=f'Slide-level{strategy_label} (AUC = {slide_auc:.3f})',
                 linewidth=2)
    else:

        particle_df = df.groupby(['slide_name', 'particle_id']).agg({
            'prob': agg_func,
            'label': 'first'
        }).reset_index()

        particle_fpr, particle_tpr, _ = roc_curve(particle_df['label'], particle_df['prob'])





        if use_precalculated:
            if (optimal_aggregation and
                'test_confusion_matrix' in optimal_aggregation and
                'auc' in optimal_aggregation['test_confusion_matrix']):
                particle_auc = optimal_aggregation['test_confusion_matrix']['auc']
                logging.info(f"Using optimal aggregation AUC: {particle_auc:.4f}")
            elif 'particle_auroc' in metrics:
                particle_auc = metrics['particle_auroc']
                logging.info(f"Using pre-calculated particle AUC: {particle_auc:.4f}")
            else:
                particle_auc = auc(particle_fpr, particle_tpr)
                logging.info(f"Calculated particle AUC: {particle_auc:.4f} for ROC curve")
        else:
            particle_auc = auc(particle_fpr, particle_tpr)
            logging.info(f"Calculated particle AUC: {particle_auc:.4f} for ROC curve")

        strategy_label = f" using {agg_strategy}" if agg_strategy else ""
        plt.plot(particle_fpr, particle_tpr,
                 label=f'Particle-level{strategy_label} (AUC = {particle_auc:.3f})',
                 linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves for {task.capitalize()} Classification (Test Data)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)


    try:
        save_path = output_dir / f'{task}_test_roc_curves.png'
        plt.savefig(save_path, dpi=300)
        logging.info(f"Successfully saved test ROC curve visualization to {save_path}")
        plt.close()
    except Exception as e:
        logging.error(f"Failed to save ROC visualization: {str(e)}")
        plt.close()

def validate_metrics_consistency(metrics: Dict) -> Dict:
    """Check and correct mathematical consistency between sensitivity, specificity, and accuracy."""

    validated_metrics = metrics.copy()


    for level in ['tile', 'slide', 'particle']:

        sensitivity_key = f'{level}_sensitivity'
        specificity_key = f'{level}_specificity'
        accuracy_key = f'{level}_accuracy'

        if sensitivity_key in metrics and specificity_key in metrics and accuracy_key in metrics:
            sens = metrics[sensitivity_key]
            spec = metrics[specificity_key]
            acc = metrics[accuracy_key]


            if abs(sens - 1.0) < 1e-6 and abs(spec - 1.0) < 1e-6:

                if abs(acc - 1.0) > 1e-6:
                    logging.warning(f"Inconsistency detected: {level} level has perfect "
                                   f"sensitivity ({sens:.6f}) and specificity ({spec:.6f}), "
                                   f"but accuracy is {acc:.6f}")


                    validated_metrics[accuracy_key] = 1.0


                    validated_metrics[f'{level}_metrics_corrected'] = True
                    validated_metrics[f'{level}_original_accuracy'] = acc


    validated_metrics['validation_timestamp'] = datetime.now().isoformat()
    validated_metrics['metrics_validated'] = True

    return validated_metrics

def create_summary_file(metrics, output_path):
    """Write a human-readable summary file with validation and test performance metrics."""
    with open(output_path, 'w') as f:

        task = metrics.get('task', 'N/A')
        f.write("Histology Classification Metrics Summary\n")
        f.write("=====================================\n")
        f.write(f"Task: {task}\n")
        f.write(f"Split: {metrics.get('split', 'N/A')}\n")
        f.write(f"Model: {metrics.get('model_name', 'N/A')}\n\n")


        is_inflammation = task == 'inflammation'
        higher_level = "Slide" if is_inflammation else "Particle"
        auc_key = "slide_auroc" if is_inflammation else "particle_auroc"
        validation_auc_key = "validation_slide_auroc" if is_inflammation else "validation_particle_auroc"


        if 'optimal_aggregation' in metrics:
            agg = metrics['optimal_aggregation']
            f.write("OPTIMAL AGGREGATION STRATEGY\n")
            f.write("----------------------------\n")
            f.write(f"Strategy: {agg.get('best_strategy', 'N/A')}\n")
            f.write(f"Threshold: {agg.get('best_metrics', {}).get('threshold', 0):.3f}\n\n")


        if 'optimal_thresholds' in metrics:
            tile_opt = metrics['optimal_thresholds'].get('tile', {})
            higher_level_opt = metrics['optimal_thresholds'].get('slide' if is_inflammation else 'particle', {})
            f.write("VALIDATION PERFORMANCE\n")
            f.write("-------------------------\n")
            f.write(f"Tile-Level Accuracy: {tile_opt.get('accuracy', 0):.2%}\n")
            f.write(f"Tile-Level Sensitivity: {tile_opt.get('sensitivity', 0):.2%}\n")
            f.write(f"Tile-Level Specificity: {tile_opt.get('specificity', 0):.2%}\n")
            f.write(f"Tile-Level F1 Score: {tile_opt.get('f1', 0):.2%}\n")
            f.write(f"Tile-Level AUC: {metrics.get('validation_tile_auroc', 0):.2%}\n")
            f.write(f"{higher_level}-Level Accuracy: {higher_level_opt.get('accuracy', 0):.2%}\n")
            f.write(f"{higher_level}-Level Sensitivity: {higher_level_opt.get('sensitivity', 0):.2%}\n")
            f.write(f"{higher_level}-Level Specificity: {higher_level_opt.get('specificity', 0):.2%}\n")
            f.write(f"{higher_level}-Level F1 Score: {higher_level_opt.get('f1', 0):.2%}\n")
            f.write(f"{higher_level}-Level AUC: {metrics.get(validation_auc_key, 0):.2%}\n")
            f.write("\n")


        f.write("TEST PERFORMANCE\n")
        f.write("--------------------\n")


        f.write(f"Tile-Level Accuracy: {metrics.get('tile_accuracy', 0):.2%}\n")
        f.write(f"Tile-Level Sensitivity: {metrics.get('tile_sensitivity', 0):.2%}\n")
        f.write(f"Tile-Level Specificity: {metrics.get('tile_specificity', 0):.2%}\n")
        f.write(f"Tile-Level F1 Score: {metrics.get('tile_f1', 0):.2%}\n")
        f.write(f"Tile-Level AUC: {metrics.get('tile_auroc', 0):.2%}\n")


        if ('optimal_aggregation' in metrics and
            'test_confusion_matrix' in metrics['optimal_aggregation']):
            agg_cm = metrics['optimal_aggregation']['test_confusion_matrix']
            if all(key in agg_cm for key in ['tp', 'tn', 'fp', 'fn']):

                tp = agg_cm['tp']
                tn = agg_cm['tn']
                fp = agg_cm['fp']
                fn = agg_cm['fn']


                total = tp + tn + fp + fn
                accuracy = (tp + tn) / total if total > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                f1 = (2 * precision * sensitivity / (precision + sensitivity)
                      if (precision + sensitivity) > 0 else 0)


                if 'auc' in agg_cm:
                    auc = agg_cm['auc']
                else:

                    auc = metrics.get(auc_key, 0)


                f.write(f"{higher_level}-Level Accuracy: {accuracy:.2%}\n")
                f.write(f"{higher_level}-Level Sensitivity: {sensitivity:.2%}\n")
                f.write(f"{higher_level}-Level Specificity: {specificity:.2%}\n")
                f.write(f"{higher_level}-Level F1 Score: {f1:.2%}\n")
                f.write(f"{higher_level}-Level AUC: {auc:.2%}\n")
            else:

                f.write(f"{higher_level}-Level Accuracy: {metrics.get('slide_accuracy' if is_inflammation else 'particle_accuracy', 0):.2%}\n")
                f.write(f"{higher_level}-Level Sensitivity: {metrics.get('slide_sensitivity' if is_inflammation else 'particle_sensitivity', 0):.2%}\n")
                f.write(f"{higher_level}-Level Specificity: {metrics.get('slide_specificity' if is_inflammation else 'particle_specificity', 0):.2%}\n")
                f.write(f"{higher_level}-Level F1 Score: {metrics.get('slide_f1' if is_inflammation else 'particle_f1', 0):.2%}\n")
                f.write(f"{higher_level}-Level AUC: {metrics.get(auc_key, 0):.2%}\n")
        else:

            f.write(f"{higher_level}-Level Accuracy: {metrics.get('slide_accuracy' if is_inflammation else 'particle_accuracy', 0):.2%}\n")
            f.write(f"{higher_level}-Level Sensitivity: {metrics.get('slide_sensitivity' if is_inflammation else 'particle_sensitivity', 0):.2%}\n")
            f.write(f"{higher_level}-Level Specificity: {metrics.get('slide_specificity' if is_inflammation else 'particle_specificity', 0):.2%}\n")
            f.write(f"{higher_level}-Level F1 Score: {metrics.get('slide_f1' if is_inflammation else 'particle_f1', 0):.2%}\n")
            f.write(f"{higher_level}-Level AUC: {metrics.get(auc_key, 0):.2%}\n")