"""
GSDB-parity evaluation for patch-level CNNs (resnet18 / gigapath, --input_type image).

Reproduces GSDB/src/models/evaluate.py exactly, so a run whose model config sets
`gsdb_parity_eval: true` yields the same numbers GSDB reports:

* evaluate_model() is copied verbatim from GSDB (incl. its test-set AUC<0.5 inversion check),
* calculate_validation_thresholds_gsdb() is GSDB's calculate_validation_thresholds() body with
  master_code's dataset passed in instead of GSDB's,
* all metric/threshold code comes from src.utils.gsdb_legacy_metrics (verbatim GSDB copy),
* data matches GSDB's evaluation inputs: HE-only splits (val_HE / test_HE / test_scanner2_HE),
  tiles with < 90% white pixels, and GSDB's *evaluation* transform
  (Resize(224, bicubic) -> CenterCrop(224) -- not the Resize(256) -> CenterCrop(224) used for
  validation during training), batch size 32, fp32.

Do not change the copied functions; master_code's maintained evaluation lives in evaluate.py.
"""
import copy
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import src.utils.gsdb_legacy_metrics as metrics_utils
from src.dataset.histology_dataset import HistologyDataset
from src.utils.training_utils import set_all_seeds
from src.pipelines.gsdb_parity_train import _prepare_gsdb_runtime

DEFAULT_GSDB_EVAL_SPLITS = ['test', 'test_scanner2']


def is_gsdb_parity_eval(config: Optional[dict], model_type, input_type, mil) -> bool:
    """True for patch-level CNN runs whose model config opts into GSDB-parity evaluation."""
    return (
        bool((config or {}).get('gsdb_parity_eval', False))
        and str(model_type) == 'cnn'
        and str(input_type) == 'image'
        and not bool(mil)
    )


def get_gsdb_eval_transforms() -> transforms.Compose:
    """GSDB's evaluation transform (GSDB/src/models/model_utils.get_transforms(is_training=False))."""
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                           std=(0.229, 0.224, 0.225))
    ])


def build_gsdb_eval_dataset(split: str, task: str, paths, config: dict,
                            split_seed: int = 42, split_phase: Optional[str] = None) -> HistologyDataset:
    """HE-only tile dataset for `split` (e.g. 'val', 'test', 'test_scanner2' -> *_HE.csv), as in GSDB."""
    split = str(split)
    if split.endswith('.csv'):
        split = split[:-4]
    if split.endswith('_HE'):
        split = split[:-3]
    return HistologyDataset(
        split=split,
        transform=get_gsdb_eval_transforms(),
        task=task,
        he_only=True,
        scanner1_only=False,
        paths=paths,
        relevance_mode='uniform',  # relevance is metadata only; skip the costly diversity pass
        label_mode='binary',
        split_seed=split_seed,
        split_phase=split_phase,
        max_white_fraction=float(config.get('max_white_fraction', 0.9)),
    )


def calculate_validation_thresholds_gsdb(model: nn.Module, val_dataset, task: str,
                                         output_dir: Optional[Path] = None) -> Dict:
    """GSDB's calculate_validation_thresholds() on a given validation dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Collect predictions on validation set
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for inputs, labels, metadata_batch in tqdm(val_loader, desc="Collecting validation predictions"):
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            # master_code (only deviation from GSDB): squeeze() turns a batch of size 1 into a 0-d
            # tensor, on which GSDB crashes (len() / BCE size mismatch). No-op for every other batch.
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            for i in range(len(outputs)):
                metadata = {k: v[i] if isinstance(v, list) else v for k, v in metadata_batch.items()}
                
                pred_dict = {
                    'raw_pred': outputs[i].cpu().item(),
                    'label': labels[i].item(),
                    'particle_id': metadata['particle_id'],
                    'slide_name': metadata['slide_name'],
                }
                
                if task == 'inflammation':
                    pred_dict['inflammation_status'] = metadata.get('inflammation_status', 'unknown')
                
                predictions.append(pred_dict)
    
    # Convert to DataFrame
    val_df = pd.DataFrame(predictions)
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate optimal thresholds on validation data
    optimal_thresholds = metrics_utils.optimize_hierarchical_thresholds(
        df=val_df,
        task=task,
        output_dir=output_dir
    )

    optimal_aggregation = metrics_utils.optimize_aggregation_strategy(
        df=val_df,
        task=task,
        output_dir=output_dir
    )
    
    validation_auc_metrics = {}
    
    # Calculate tile-level AUC
    if 'raw_pred' in val_df.columns:
        # Convert logits to probabilities if needed
        if val_df['raw_pred'].min() < 0 or val_df['raw_pred'].max() > 1:
            tile_probs = torch.sigmoid(torch.tensor(val_df['raw_pred'].values)).numpy()
        else:
            tile_probs = val_df['raw_pred'].values
            
        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        try:
            validation_auc_metrics['validation_tile_auroc'] = roc_auc_score(val_df['label'].values, tile_probs)
        except Exception as e:
            logging.warning(f"Could not calculate validation tile-level AUC: {str(e)}")
            validation_auc_metrics['validation_tile_auroc'] = 0.5
    
    # Calculate higher-level AUC based on task
    if task == 'inflammation':
        # Calculate slide-level AUC
        slide_df = val_df.groupby('slide_name').agg({
            'raw_pred': 'mean',
            'label': 'first'
        }).reset_index()
        
        if 'raw_pred' in slide_df.columns:
            # Convert logits to probabilities if needed
            if slide_df['raw_pred'].min() < 0 or slide_df['raw_pred'].max() > 1:
                slide_probs = torch.sigmoid(torch.tensor(slide_df['raw_pred'].values)).numpy()
            else:
                slide_probs = slide_df['raw_pred'].values
                
            # Calculate AUC
            try:
                validation_auc_metrics['validation_slide_auroc'] = roc_auc_score(slide_df['label'].values, slide_probs)
            except Exception as e:
                logging.warning(f"Could not calculate validation slide-level AUC: {str(e)}")
                validation_auc_metrics['validation_slide_auroc'] = 0.5
    else:  # tissue task
        # Calculate particle-level AUC
        particle_df = val_df.groupby(['slide_name', 'particle_id']).agg({
            'raw_pred': 'mean',
            'label': 'first'
        }).reset_index()
        
        if 'raw_pred' in particle_df.columns:
            # Convert logits to probabilities if needed
            if particle_df['raw_pred'].min() < 0 or particle_df['raw_pred'].max() > 1:
                particle_probs = torch.sigmoid(torch.tensor(particle_df['raw_pred'].values)).numpy()
            else:
                particle_probs = particle_df['raw_pred'].values
                
            # Calculate AUC
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

    # Save thresholds to file if output directory is provided
    if output_dir:
        thresholds_file = output_dir / "validation_thresholds.json"
        with open(thresholds_file, 'w') as f:
            json.dump(validation_optimized, f, indent=2)
            
        logging.info(f"Validation-optimized thresholds and aggregation saved to: {thresholds_file}")
    
    return validation_optimized


# --- verbatim copy of GSDB/src/models/evaluate.py::evaluate_model ---------------------------
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    task: str,
    split: str,
    device: torch.device,
    output_dir: Optional[Path] = None,
    validation_thresholds: Optional[Dict] = None, 
    save_predictions: bool = True
) -> Dict:
    """Evaluate model at appropriate hierarchical levels.
    
    Args:
        model: Neural network model to evaluate
        data_loader: DataLoader for evaluation data
        task: 'inflammation' or 'tissue'
        split: Dataset split name ('test', 'test_scanner2')
        device: Device to run evaluation on
        output_dir: Directory to save evaluation results
        validation_thresholds: Pre-computed thresholds from validation data
        
    Returns:
        Dict: Dictionary of evaluation metrics
    """
    model.eval()
    predictions = []
    
    # Get model name - use attribute if available
    if hasattr(model, 'model_name'):
        model_name = model.model_name
    else:
        # Default to model class name
        model_name = model.__class__.__name__
    
    logging.info(f"Evaluating model: {model_name}")
    
    # Collect predictions
    with torch.no_grad():
        for inputs, labels, metadata_batch in tqdm(data_loader, desc=f"Evaluating {split}"):
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()  # Get raw model outputs
            # master_code (only deviation from GSDB): squeeze() turns a batch of size 1 into a 0-d
            # tensor, on which GSDB crashes (len() / BCE size mismatch). No-op for every other batch.
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            # Check if outputs are logits (outside 0-1 range) or probabilities
            is_logits = False
            if outputs.numel() > 0:  # Check if tensor is not empty
                min_val = outputs.min().item()
                max_val = outputs.max().item()
                is_logits = min_val < 0 or max_val > 1
                
                if is_logits:
                    # Convert logits to probabilities
                    probs = torch.sigmoid(outputs)
                else:
                    # Already probabilities
                    probs = outputs
            else:
                probs = outputs  # Fallback
            
            # Process each sample in the batch
            for i in range(len(outputs)):
                metadata = {k: v[i] if isinstance(v, list) else v for k, v in metadata_batch.items()}
                
                pred_dict = {
                    'raw_pred': outputs[i].cpu().item(),  # Store raw model outputs
                    'prob': probs[i].cpu().item(),        # Store probabilities
                    'label': labels[i].item(),
                    'particle_id': metadata['particle_id'],
                    'slide_name': metadata['slide_name'],
                }
                
                # Add task field to support task detection
                if task == 'inflammation':
                    # Use inflammation_status from metadata, which is the correct field name in the dataset
                    pred_dict['inflammation_type'] = metadata.get('inflammation_status', 'unknown')
                
                predictions.append(pred_dict)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(predictions)
    
    # Log basic statistics
    logging.info(f"\nEvaluation Summary:")
    logging.info(f"Total samples: {len(df)}")
    logging.info(f"Unique slides: {df['slide_name'].nunique()}")
    logging.info(f"Unique particles: {len(df.groupby(['slide_name', 'particle_id']))}")
    
    # Calculate class distribution
    pos_count = (df['label'] == 1).sum()
    neg_count = (df['label'] == 0).sum()
    logging.info(f"\nClass Distribution:")
    logging.info(f"Positive class: {pos_count} ({pos_count/len(df):.1%})")
    logging.info(f"Negative class: {neg_count} ({neg_count/len(df):.1%})")
    
    # Log prediction statistics
    logging.info("\nPrediction Statistics:")
    logging.info(f"Raw logits mean: {df['raw_pred'].mean():.3f}")
    logging.info(f"Raw logits std: {df['raw_pred'].std():.3f}")
    logging.info(f"Raw logits min: {df['raw_pred'].min():.3f}")
    logging.info(f"Raw logits max: {df['raw_pred'].max():.3f}")
    
    # Always use validation thresholds or calculate them if not provided
    if validation_thresholds is None:
        logging.warning("No validation thresholds provided. Calculating thresholds from validation data...")
        # Calculate optimal thresholds on validation data - never on test data
        # master_code: never reached -- run_gsdb_parity_evaluation always passes thresholds.
        raise ValueError("GSDB-parity evaluation requires precomputed validation thresholds")
        logging.info("Validation-optimized thresholds calculated.")

    # Log the validation thresholds and AUCs separately
    logging.info("\nUsing validation-optimized thresholds:")
    optimal_thresholds_dict = {k: v for k, v in validation_thresholds.items() 
                             if k not in ['validation_tile_auroc', 'validation_slide_auroc', 
                                        'validation_particle_auroc', 'optimal_aggregation']}
    for level, level_metrics in optimal_thresholds_dict.items():
        if isinstance(level_metrics, dict) and 'threshold' in level_metrics:
            logging.info(f"{level.capitalize()} level: {level_metrics['threshold']:.3f}")
            logging.info(f"  Sensitivity: {level_metrics['sensitivity']:.2f}")
            logging.info(f"  Specificity: {level_metrics['specificity']:.2f}")
            logging.info(f"  F1 Score: {level_metrics['f1']:.2f}")
        else:
            logging.info(f"{level.capitalize()} level: Invalid threshold format: {level_metrics}")

    # Log validation AUCs separately
    for auc_key in ['validation_tile_auroc', 'validation_slide_auroc', 'validation_particle_auroc']:
        if auc_key in validation_thresholds:
            logging.info(f"{auc_key.replace('_', ' ').capitalize()}: {validation_thresholds[auc_key]:.4f}")

    # Use validation-optimized aggregation strategy if available
    if "optimal_aggregation" in validation_thresholds:
        logging.info("\nUsing validation-optimized aggregation strategy:")
        agg_results = validation_thresholds["optimal_aggregation"]
        
        # Extract validation confusion matrix first
        val_tp = agg_results['best_metrics'].get('tp', 0)
        val_tn = agg_results['best_metrics'].get('tn', 0)
        val_fp = agg_results['best_metrics'].get('fp', 0)
        val_fn = agg_results['best_metrics'].get('fn', 0)
        
        # Add new field to store validation confusion matrix
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
        
        # Apply the optimal strategy to the current test set
        best_strategy = agg_results['best_strategy']
        best_threshold = agg_results['best_metrics']['threshold']
        
        # Define aggregation function based on strategy name
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
    
        # Calculate probabilities from logits if needed and cache it
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
                df['prob'] = 0.5  # Default fallback
                
        # Check if predictions might be inverted (AUC < 0.5 suggests predictions are flipped)
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(df['label'], df['prob'])
            logging.info(f"Initial AUC score: {auc:.4f}")
            
            if auc < 0.5:
                inv_auc = roc_auc_score(df['label'], 1 - df['prob'])
                logging.warning(f"Low AUC detected: {auc:.4f}. Inverted AUC would be: {inv_auc:.4f}")
                
                if inv_auc > 0.6:  # Significantly better when inverted
                    logging.warning("Detected inverted predictions, auto-correcting...")
                    df['prob'] = 1 - df['prob']
                    if 'raw_pred' in df.columns:
                        if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
                            df['raw_pred'] = -df['raw_pred']
                        else:
                            df['raw_pred'] = 1 - df['raw_pred']
                    logging.info(f"Predictions inverted. New AUC: {inv_auc:.4f}")
        except Exception as e:
            logging.warning(f"Could not check for inverted predictions: {str(e)}")
        
        if task == 'inflammation':
            slide_df = df.groupby('slide_name').agg({
                'prob': agg_func,
                'label': 'first'
            }).reset_index()
            
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
                optimal_agg_auc = 0.5

            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(slide_df['label'].values, slide_preds).ravel()
            
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
                optimal_agg_auc = 0.5

            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(particle_df['label'].values, particle_preds).ravel()
            
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
    
    # Log aggregation results
    best_strategy = agg_results.get('best_strategy', 'mean')
    best_metrics = agg_results.get('best_metrics', {})
    logging.info(f"  Strategy: {best_strategy}")
    logging.info(f"  Threshold: {best_metrics.get('threshold', 0.5):.3f}")
    logging.info(f"  Sensitivity: {best_metrics.get('sensitivity', 0):.2f}")
    logging.info(f"  Specificity: {best_metrics.get('specificity', 0):.2f}")
    logging.info(f"  F1 Score: {best_metrics.get('f1', 0):.2f}")
    
    # Create enhanced distribution plots
    if output_dir:
        probs = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
        
        plt.figure(figsize=(16, 7))
        
        plt.subplot(1, 2, 1)
        plt.hist(df['raw_pred'][df['label']==1], bins=50, alpha=0.6, label='Positive class', color='blue')
        plt.hist(df['raw_pred'][df['label']==0], bins=50, alpha=0.6, label='Negative class', color='orange')
        plt.axvline(x=0.0, color='r', linestyle='--', label='Default threshold (logit=0)')
        
        if task == 'inflammation' and 'slide' in optimal_thresholds_dict:
            prob_threshold = optimal_thresholds_dict['slide']['threshold']
            logit_threshold = np.log(prob_threshold/(1-prob_threshold)) if 0 < prob_threshold < 1 else (10 if prob_threshold >= 1 else -10)
            plt.axvline(x=logit_threshold, color='g', linestyle='--', 
                      label=f'Optimal threshold (logit={logit_threshold:.2f})')
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
        
        if task == 'inflammation' and 'slide' in optimal_thresholds_dict:
            plt.axvline(x=optimal_thresholds_dict['slide']['threshold'], color='g', linestyle='--', 
                      label=f'Optimal threshold (prob={optimal_thresholds_dict["slide"]["threshold"]:.2f})')
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
    
    # Calculate metrics using validation-optimized thresholds
    if 'slide' in optimal_thresholds_dict:
        slide_threshold = optimal_thresholds_dict['slide']['threshold']
        if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
            threshold_for_metrics = np.log(slide_threshold/(1-slide_threshold)) if 0 < slide_threshold < 1 else 0.0
            logging.info(f"Predictions are logits - converting probability threshold {slide_threshold:.4f} to logit threshold: {threshold_for_metrics:.4f}")
        else:
            threshold_for_metrics = slide_threshold
            logging.info(f"Predictions are probabilities - using probability threshold directly: {threshold_for_metrics:.4f}")
    else:
        if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
            threshold_for_metrics = 0.0
        else:
            threshold_for_metrics = 0.5
    
    # Calculate metrics using specified threshold
    metrics = metrics_utils.calculate_hierarchical_metrics(
        df=df,
        split=split,
        model_name=model_name,
        threshold=threshold_for_metrics
    )
    
    # Combine results and elevate validation AUCs to top level
    combined_metrics = {
        **metrics,
        'optimal_thresholds': optimal_thresholds_dict,
        'optimal_aggregation': agg_results,
    }
    # Elevate validation AUCs to top level
    for auc_key in ['validation_tile_auroc', 'validation_slide_auroc', 'validation_particle_auroc']:
        if auc_key in validation_thresholds:
            combined_metrics[auc_key] = validation_thresholds[auc_key]
    
    return combined_metrics


def _gsdb_implementation_details(task: str, architecture: str) -> Dict:
    """statistics.json "implementation_details" exactly as GSDB's evaluate.py main() fills them.

    GSDB uses its CLI defaults (epochs=50, batch_size=32, rest None) and, only if its hard-coded
    server config exists, lr / pos_weight from there plus an architecture-based optimizer name.
    Purely informational -- not the values the model was trained with.
    """
    details = {"epochs": 50, "optimizer": None, "learning_rate": None, "batch_size": 32,
               "pos_class_weight": None}
    config_file = Path('/mnt/data/dliebel/2024_dliebel/configs/model_config.yaml')
    config = {}
    if config_file.exists():
        import yaml
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logging.warning(f"Failed to load config file: {e}")
    if config and task in config and architecture in config[task]:
        model_config = config[task][architecture]
        if 'optimizer' in model_config:
            details["learning_rate"] = model_config['optimizer'].get('learning_rate')
        details["pos_class_weight"] = model_config.get('pos_weight')
        details["optimizer"] = 'SGD' if architecture == 'resnet18' else 'AdamW'
    return details


def run_gsdb_parity_evaluation(model: nn.Module, task: str, paths, config: dict, output_dir: Path,
                               splits: Optional[Iterable[str]] = None, seed: int = 42,
                               split_phase: Optional[str] = None) -> Dict[str, Dict]:
    """Evaluate like GSDB's evaluate.py: thresholds from val_HE, then each split with them.

    Writes, per split, GSDB's outputs (statistics.json, predictions.csv, ROC curves, ...) to
    output_dir/split_<split>/ and the validation thresholds to output_dir/validation_thresholds/.
    """
    splits = list(splits) if splits else list(config.get('gsdb_parity_eval_splits', DEFAULT_GSDB_EVAL_SPLITS))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(seed)  # GSDB's evaluate.py main() seeds too (cudnn deterministic, no benchmark)
    _prepare_gsdb_runtime(config)  # no CUBLAS_WORKSPACE_CONFIG, cuDNN TF32 as in GSDB

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    logging.info("GSDB-parity evaluation: task=%s, splits=%s", task, splits)
    val_dataset = build_gsdb_eval_dataset('val', task, paths, config, split_seed=seed, split_phase=split_phase)
    validation_thresholds = calculate_validation_thresholds_gsdb(
        model=model,
        val_dataset=val_dataset,
        task=task,
        output_dir=output_dir / "validation_thresholds",
    )

    results = {}
    for split in splits:
        split_output_dir = output_dir / f"split_{split}"
        split_output_dir.mkdir(parents=True, exist_ok=True)

        dataset = build_gsdb_eval_dataset(split, task, paths, config, split_seed=seed, split_phase=split_phase)
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # GSDB recomputes thresholds per evaluate.py call and evaluate_model mutates them, so
        # every split gets its own untouched copy.
        metrics = evaluate_model(
            model=model,
            data_loader=dataloader,
            task=task,
            split=split,
            device=device,
            output_dir=split_output_dir,
            validation_thresholds=copy.deepcopy(validation_thresholds)
        )

        metrics["implementation_details"] = _gsdb_implementation_details(
            task, getattr(model, 'model_name', config.get('selected_model'))
        )

        metrics_utils.save_metrics(metrics, split_output_dir)

        if 'predictions_df' in metrics:
            try:
                metrics_utils.plot_roc_curves(
                    metrics['predictions_df'],
                    task,
                    split_output_dir,
                    metrics.get('optimal_aggregation', None),
                    metrics
                )
                logging.info("ROC curves generated with consistent metrics")
            except Exception as e:
                logging.warning(f"Failed to create ROC curves: {str(e)}")

        results[split] = metrics
        logging.info("GSDB-parity evaluation of split '%s' written to %s", split, split_output_dir)

    return results
