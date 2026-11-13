import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utils.plot_style import IEEE_COLORS, MARKERS, apply_ieee_style, figsize, new_figure, save_figure, style_axis


def _series(history: dict, key: str) -> Optional[list]:
    values = history.get(key)
    return values if values else None


def _plot_multi(ax, epochs: Sequence[int], entries: List[Tuple[str, list]]) -> None:
    for idx, (label, values) in enumerate(entries):
        color = IEEE_COLORS[idx % len(IEEE_COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        ax.plot(epochs, values, label=label, color=color, marker=marker, markersize=3)


def plot_training_history(history: dict, save_dir="plots") -> None:
    """Render the standard set of per-epoch training/validation diagnostic figures present in `history`."""
    apply_ieee_style()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n_epochs = len(history.get('train_loss') or [])
    if n_epochs == 0:
        logging.warning("plot_training_history: 'train_loss' is empty, skipping all plots.")
        return
    epochs = list(range(1, n_epochs + 1))

    # 1. Loss curve (train vs. validation).
    fig, ax = new_figure("single")
    entries = [('Training', history['train_loss'])]
    if _series(history, 'val_loss'):
        entries.append(('Validation', history['val_loss']))
    _plot_multi(ax, epochs, entries)
    style_axis(ax, 'Epoch', 'Loss', 'Training and Validation Loss')
    save_figure(fig, save_dir / "loss.png")

    # 2. Generalization gap: val_loss - train_loss. Positive and growing => overfitting.
    if _series(history, 'val_loss'):
        gap = [v - t for t, v in zip(history['train_loss'], history['val_loss'])]
        fig, ax = new_figure("single")
        ax.plot(epochs, gap, color=IEEE_COLORS[1], marker=MARKERS[0], markersize=3)
        ax.axhline(0.0, color='0.3', linewidth=0.8, linestyle=':')
        style_axis(ax, 'Epoch', 'Validation Loss - Training Loss', 'Generalization Gap')
        save_figure(fig, save_dir / "generalization_gap.png")

    # 3. Accuracy curve (train vs. validation).
    if _series(history, 'train_accuracy') or _series(history, 'val_accuracy'):
        fig, ax = new_figure("single")
        entries = []
        if _series(history, 'train_accuracy'):
            entries.append(('Training', history['train_accuracy']))
        if _series(history, 'val_accuracy'):
            entries.append(('Validation', history['val_accuracy']))
        _plot_multi(ax, epochs, entries)
        style_axis(ax, 'Epoch', 'Accuracy (%)', 'Classification Accuracy')
        save_figure(fig, save_dir / "accuracy.png")

    # 4. Validation diagnostic metrics: sensitivity/specificity/precision/F1/balanced accuracy.
    diag_keys = [
        ('val_sensitivity', 'Sensitivity (Recall)'),
        ('val_specificity', 'Specificity'),
        ('val_precision', 'Precision'),
        ('val_f1', 'F1 Score'),
        ('val_balanced_accuracy', 'Balanced Accuracy'),
    ]
    entries = [(label, history[key]) for key, label in diag_keys if _series(history, key)]
    if entries:
        fig, ax = new_figure("single")
        _plot_multi(ax, epochs, entries)
        style_axis(ax, 'Epoch', 'Score (%)', 'Validation Diagnostic Metrics')
        ax.legend(fontsize=7)
        save_figure(fig, save_dir / "classification_metrics.png")

    # 5. Ranking / chance-corrected agreement metrics: AUC-ROC, MCC, Cohen's kappa.
    agreement_keys = [
        ('val_auc', 'AUC-ROC'),
        ('val_mcc', 'Matthews Correlation Coefficient'),
        ('val_kappa', "Cohen's Kappa"),
    ]
    entries = [(label, history[key]) for key, label in agreement_keys if _series(history, key)]
    if entries:
        fig, ax = new_figure("single")
        _plot_multi(ax, epochs, entries)
        style_axis(ax, 'Epoch', 'Score (%)', 'Ranking and Chance-Corrected Agreement Metrics')
        save_figure(fig, save_dir / "auc_mcc_kappa.png")

    # 6. Learning-rate schedule.
    if _series(history, 'learning_rate'):
        fig, ax = new_figure("single")
        ax.plot(epochs, history['learning_rate'], color=IEEE_COLORS[2], marker=MARKERS[0], markersize=3)
        ax.set_yscale('log')
        style_axis(ax, 'Epoch', 'Learning Rate (log scale)', 'Learning Rate Schedule', legend=False)
        ax.grid(True, which='both')
        save_figure(fig, save_dir / "learning_rate.png")

    # 7. Hierarchical (slide/region/patch) metrics for MIL/segmentation-style histories.
    for level, level_label in (('slide', 'Slide-Level'), ('region', 'Region-Level'), ('patch', 'Patch-Level')):
        metric_specs = [
            (f'val_{level}_accuracy', 'Accuracy'),
            (f'val_{level}_sensitivity', 'Sensitivity'),
            (f'val_{level}_specificity', 'Specificity'),
            (f'val_{level}_f1', 'F1'),
            (f'val_{level}_f2', 'F2'),
            (f'val_{level}_auc', 'AUC'),
        ]
        entries = [(label, history[key]) for key, label in metric_specs if _series(history, key)]
        if entries:
            fig, ax = new_figure("single")
            _plot_multi(ax, epochs, entries)
            style_axis(ax, 'Epoch', '%', f'{level_label} Diagnostic Metrics')
            ax.legend(fontsize=7)
            save_figure(fig, save_dir / f"{level}_metrics.png")


_CALIBRATION_MIN_UNIQUE_PROBS = 2


def plot_classification_diagnostics(
    y_true,
    y_prob,
    output_dir,
    prefix: str,
    level_name: str = "Tile-level",
    n_calibration_bins: int = 10,
    also_pdf: bool = False,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Generate standard binary-classification diagnostic figures: ROC, PR, calibration, confusion matrix.

    `threshold` is the decision threshold of the confusion matrix (positive if prob > threshold, as in
    _compute_slide_binary_metrics). Pass the validation threshold so the figure matches the reported metrics.
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    apply_ieee_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true).reshape(-1).astype(float)
    y_prob = np.asarray(y_prob).reshape(-1).astype(float)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true, y_prob = y_true[mask], y_prob[mask]

    unique_labels = np.unique(y_true)
    if y_true.size == 0 or unique_labels.size != 2:
        logging.warning(
            "plot_classification_diagnostics(%s, %s): expected exactly 2 classes, "
            "got %d sample(s) with labels %s -- skipping ROC/PR/calibration/confusion-matrix plots.",
            prefix, level_name, y_true.size, unique_labels.tolist(),
        )
        return {}

    stats: Dict[str, float] = {}

    # ROC curve.
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = float(roc_auc_score(y_true, y_prob))
    stats['roc_auc'] = roc_auc
    fig, ax = new_figure("single")
    ax.plot(fpr, tpr, color=IEEE_COLORS[0], label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='0.5', linestyle=':', linewidth=1.0, label='Chance level')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    style_axis(ax, 'False Positive Rate', 'True Positive Rate', f'{level_name} ROC Curve')
    save_figure(fig, output_dir / f"{prefix}_roc_curve.png", also_pdf=also_pdf)

    # Precision-Recall curve.
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    average_precision = float(average_precision_score(y_true, y_prob))
    prevalence = float(np.mean(y_true))
    stats['average_precision'] = average_precision
    stats['prevalence'] = prevalence
    fig, ax = new_figure("single")
    ax.plot(recall, precision, color=IEEE_COLORS[1], label=f'PR (AP = {average_precision:.3f})')
    ax.axhline(prevalence, color='0.5', linestyle=':', linewidth=1.0,
               label=f'Chance level (prevalence = {prevalence:.3f})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    style_axis(ax, 'Recall', 'Precision', f'{level_name} Precision-Recall Curve')
    save_figure(fig, output_dir / f"{prefix}_pr_curve.png", also_pdf=also_pdf)

    # Calibration curve (reliability diagram) + Brier score.
    brier = float(brier_score_loss(y_true, y_prob))
    stats['brier_score'] = brier
    try:
        n_bins_eff = max(2, min(n_calibration_bins, len(np.unique(y_prob))))
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins_eff, strategy='uniform')
        fig, ax = new_figure("single")
        ax.plot([0, 1], [0, 1], color='0.5', linestyle=':', linewidth=1.0, label='Perfectly calibrated')
        ax.plot(mean_pred, frac_pos, color=IEEE_COLORS[2], marker='o', label=f'Model (Brier score = {brier:.3f})')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        style_axis(ax, 'Mean Predicted Probability', 'Observed Frequency',
                   f'{level_name} Calibration (Reliability Diagram)')
        save_figure(fig, output_dir / f"{prefix}_calibration_curve.png", also_pdf=also_pdf)
    except Exception as exc:
        logging.warning("Calibration curve failed for %s (%s): %s", prefix, level_name, exc)

    # Confusion matrix heatmap at the given decision threshold.
    y_pred = (y_prob > threshold).astype(int)
    cm = confusion_matrix(y_true.astype(int), y_pred, labels=[0, 1])
    row_sums = np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    cm_norm = cm.astype(float) / row_sums
    fig, ax = new_figure("single", aspect=0.9)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)", ha='center', va='center',
                    color='white' if cm_norm[i, j] > 0.5 else 'black', fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Negative', 'Positive'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'{level_name} Confusion Matrix (threshold = {threshold:.3f})')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Row-normalized fraction')
    save_figure(fig, output_dir / f"{prefix}_confusion_matrix.png", also_pdf=also_pdf)

    stats['n_samples'] = int(y_true.size)
    return stats
