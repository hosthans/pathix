"""
GSDB-parity training for patch-level CNNs (resnet18 / gigapath, --input_type image).

Runs GSDB/src/models/train.py's training bit-for-bit inside master_code when the model config sets
`gsdb_parity_train: true`:

* train_model_gsdb_parity() is GSDB's train_model() copied verbatim, preceded by the
  set_all_seeds(seed) call GSDB's main() makes. Only paths (checkpoint/history), the dataset
  constructor and GigaPathClassifier's signature are adapted.
* get_transforms() is GSDB's, all loop/seed/checkpoint helpers come from
  src.utils.gsdb_legacy_training_utils (verbatim GSDB copy).
* Same RNG order as GSDB: seed -> build model -> build datasets (no RNG) -> DataLoaders. The
  Holder therefore does not build its own model/datasets for these runs (see Holder.load_model).
* master_code's import-time runtime tweaks that GSDB doesn't have are undone before CUDA is used
  (CUBLAS_WORKSPACE_CONFIG, PYTORCH_CUDA_ALLOC_CONF, cuDNN TF32 off).

Bit-identity additionally needs the same data (splits, tiles), hardware, CUDA/cuDNN and library
versions as the GSDB run -- and holds only as far as GSDB itself is reproducible run-to-run.
Do not change the copied functions.
"""
import copy
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader

import src.utils.gsdb_legacy_training_utils as training_utils
from src.dataset.histology_dataset import HistologyDataset
from src.models.base import HistologyClassifier
from src.models.gigapath import GigaPathClassifier
from src.paths.pathconfig import PathConfigurator


def is_gsdb_parity_train(config: Optional[dict], args) -> bool:
    """True for patch-level CNN runs whose model config opts into GSDB-parity training."""
    return (
        bool((config or {}).get('gsdb_parity_train', False))
        and str(getattr(args, 'dataset', 'gsdb')) == 'gsdb'
        and str(getattr(args, 'model_type', '')) == 'cnn'
        and str(getattr(args, 'input_type', '')) == 'image'
        and not bool(getattr(args, 'mil', False))
    )


def _prepare_gsdb_runtime(config: dict) -> None:
    """Undo master_code's process-wide CUDA tweaks that GSDB's train.py doesn't make.

    Must run before the first CUDA allocation / cuBLAS call of the process, which is why the Holder
    skips building its own model for GSDB-parity runs.
    """
    for var in ('CUBLAS_WORKSPACE_CONFIG', 'PYTORCH_CUDA_ALLOC_CONF'):
        if os.environ.pop(var, None) is not None:
            logging.info("GSDB parity: unset %s (GSDB runs without it)", var)
    # GSDB never calls torch.use_deterministic_algorithms (its --deterministic only sets the cuDNN
    # flags), but evaluate_model.py / train_model.py --deterministic enable it -- and without
    # CUBLAS_WORKSPACE_CONFIG cuBLAS then refuses to run.
    if torch.are_deterministic_algorithms_enabled():
        torch.use_deterministic_algorithms(False)
        logging.info("GSDB parity: disabled torch.use_deterministic_algorithms (GSDB doesn't use it)")
    if torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32 = bool(config.get('cudnn_allow_tf32', True))
        torch.backends.cuda.matmul.allow_tf32 = False  # PyTorch default, as in GSDB


def build_gsdb_train_dataset(split: str, transform, task: str, paths, config: dict, args) -> HistologyDataset:
    """GSDB's HistologyDataset(split, transform, task, paths): HE-only (*_HE.csv), < 90% white tiles."""
    return HistologyDataset(
        split=split,
        transform=transform,
        task=task,
        he_only=True,
        scanner1_only=False,
        paths=paths,
        relevance_mode='uniform',  # metadata only, consumes no RNG
        label_mode='binary',
        split_seed=getattr(args, 'seed', 42),
        split_phase=getattr(args, 'split_phase', None),
        max_white_fraction=float(config.get('max_white_fraction', 0.9)),
    )


# --- verbatim copy of GSDB/src/models/train.py::get_transforms / _get_default_transforms -------
def get_transforms(task: str, is_training: bool = False, config: dict = None) -> T.Compose:
    """Get data transformations for training or evaluation based on configuration.
    
    Args:
        task: 'inflammation' or 'tissue'
        is_training: Whether to include training augmentations
        config: Configuration dictionary containing transform settings
        
    Returns:
        Composed transforms
    """
    if config is None:
        # Fallback defaults if no config is provided
        return _get_default_transforms(task, is_training)
    
    transforms_list = []
    
    # Get common normalization params
    common_config = config.get('common', {})
    norm_config = common_config.get('transforms', {}).get('normalization', {})
    norm_mean = norm_config.get('mean', [0.485, 0.456, 0.406])
    norm_std = norm_config.get('std', [0.229, 0.224, 0.225])
    
    if is_training:
        # Get task and model specific transform settings
        transform_config = config.get('transforms', {}).get('training', {})
        
        # Random resized crop
        if 'random_resized_crop' in transform_config:
            rrc_config = transform_config['random_resized_crop']
            size = rrc_config.get('size', 224)
            scale = tuple(rrc_config.get('scale', [0.8, 1.0]))
            transforms_list.append(T.RandomResizedCrop(size, scale=scale))
        
        # Random flips
        if transform_config.get('random_horizontal_flip', False):
            transforms_list.append(T.RandomHorizontalFlip())
            
        if transform_config.get('random_vertical_flip', False):
            transforms_list.append(T.RandomVerticalFlip())
        
        # Color jitter
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
        
        # Random rotation
        if transform_config.get('random_rotation', {}).get('enabled', False):
            degrees = transform_config.get('random_rotation', {}).get('degrees', 180)
            transforms_list.append(T.RandomRotation(degrees))
        
        # Random blur
        if transform_config.get('random_blur', {}).get('enabled', False):
            kernel_size = transform_config.get('random_blur', {}).get('kernel_size', 3)
            transforms_list.append(T.GaussianBlur(kernel_size=kernel_size))
        
        # Random affine transform
        if transform_config.get('random_affine', {}).get('enabled', False):
            degrees = transform_config.get('random_affine', {}).get('degrees', 15)
            translate = tuple(transform_config.get('random_affine', {}).get('translate', [0.1, 0.1]))
            transforms_list.append(T.RandomAffine(degrees=degrees, translate=translate))
    else:
        # Validation transforms from common config
        val_config = common_config.get('transforms', {}).get('validation', {})
        resize = val_config.get('resize', 256)
        center_crop = val_config.get('center_crop', 224)
        
        transforms_list.extend([
            T.Resize(resize),
            T.CenterCrop(center_crop)
        ])
    
    # Common transforms
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    return T.Compose(transforms_list)

def _get_default_transforms(task: str, is_training: bool = False) -> T.Compose:
    """Fallback function for default transforms if no config is provided.
    
    Args:
        task: 'inflammation' or 'tissue'
        is_training: Whether to include training augmentations
        
    Returns:
        Composed transforms
    """
    transforms_list = []
    
    if is_training:
        # Basic augmentations for all tasks
        transforms_list.extend([
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2)
        ])
        
        # Default task-specific augmentations
        if task == 'inflammation':
            transforms_list.extend([
                T.RandomRotation(180),
                T.GaussianBlur(kernel_size=3),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1))
            ])
        else:  # tissue
            transforms_list.append(T.RandomRotation(180))
    else:
        transforms_list.extend([
            T.Resize(256),
            T.CenterCrop(224)
        ])
    
    # Common transforms
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return T.Compose(transforms_list)


# --- GSDB/src/models/train.py::train_model (verbatim except the marked adaptations) -----------
def train_model_gsdb_parity(args, model, train_dataset, val_dataset, config, optimizer_name, momentum, paths):
    """GSDB/src/models/train.py::train_model (+ the seeding its main() does right before it).

    `model`, `train_dataset`, `val_dataset`, `optimizer_name` and `momentum` from the Holder are
    ignored on purpose: GSDB builds model and datasets itself, right after seeding, and that exact
    order of RNG draws is what makes the run bit-identical.
    """
    _prepare_gsdb_runtime(config)
    # GSDB main(): training_utils.set_all_seeds(args.seed) directly before train_model(args, paths)
    training_utils.set_all_seeds(args.seed)
    
    # Load configuration with model selection
    # master_code: config comes from the Holder (same model_config.yaml values as GSDB's load_config)
    config = copy.deepcopy(config)
    
    # Extract architecture configuration
    arch_config = config['architecture']
    model_name = arch_config['name']
    
    # Log the selected model
    logging.info(f"Training {args.task} classifier using {model_name} architecture")
    
    # Override configuration with command line arguments if provided
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
        logging.info(f"Overriding batch size: {args.batch_size}")
        
    if args.learning_rate is not None:
        config['optimizer']['learning_rate'] = args.learning_rate
        logging.info(f"Overriding learning rate: {args.learning_rate}")
        
    if args.weight_decay is not None:
        config['optimizer']['weight_decay'] = args.weight_decay
        logging.info(f"Overriding weight decay: {args.weight_decay}")
        
    if args.epochs is not None:
        config['epochs'] = args.epochs
        logging.info(f"Overriding epochs: {args.epochs}")
    
    # Create model with proper configuration
    if model_name == 'gigapath':
        # Special case for GigaPath model
        model = GigaPathClassifier(
            model_name='gigapath',
            num_classes=1,
            dropout_rate=config['dropout_rate']
        )
    else:
        # Standard models from HistologyClassifier
        model = HistologyClassifier(
            model_name=model_name,
            num_classes=1,
            dropout_rate=config['dropout_rate']
        )
    
    # Setup device and multi-GPU support
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = training_utils.handle_multi_gpu_model(model)
    model = model.to(device)

    # Configure training components with optimizer name and momentum for SGD
    optimizer_name = config['optimizer'].get('name', 'AdamW')
    momentum = config['optimizer'].get('momentum', 0.9)
    
    training_components = training_utils.configure_training_components(
        model=model,
        learning_rate=config['optimizer']['learning_rate'],
        weight_decay=config['optimizer'].get('weight_decay', 0.0001),
        pos_weight=config.get('pos_weight', 1.0),
        optimizer_name=optimizer_name,
        momentum=momentum,
        device=device
    )
    
    optimizer = training_components['optimizer']
    criterion = training_components['criterion']
    scaler = training_components['scaler']
    
    # Setup early stopping if enabled
    early_stopping = None
    if config.get('early_stopping', {}).get('enabled', False):
        patience = config['early_stopping'].get('patience', 10)
        early_stopping = training_utils.EarlyStopping(patience=patience, mode='min')
        logging.info(f"Early stopping enabled with patience {patience}")
    
    # Create datasets and dataloaders with transforms from config
    train_transform = get_transforms(args.task, is_training=True, config=config)
    val_transform = get_transforms(args.task, is_training=False, config=config)
    
    train_dataset = build_gsdb_train_dataset('train', train_transform, args.task, paths, config, args)
    
    val_dataset = build_gsdb_train_dataset('val', val_transform, args.task, paths, config, args)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'] * 2,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # Configure scheduler
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
    
    # Training history with reproducibility information
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_auc': [],
        # Add reproducibility information
        'metadata': {
            'random_seed': args.seed,
            'deterministic': args.deterministic,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        # Training phase - pass deterministic flag from args
        train_metrics = training_utils.train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            use_amp=True,
            deterministic=args.deterministic 
        )
        
        # Validation phase
        val_metrics = training_utils.validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )
        
        # Update scheduler
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        
        # Log metrics
        logging.info(
            f"Epoch {epoch+1}/{config['epochs']}\n"
            f"Train Metrics:\n"
            f"  Loss: {train_metrics['loss']:.4f}\n"
            f"  Accuracy: {train_metrics['accuracy']:.2f}%\n"
            f"  Sensitivity: {train_metrics['sensitivity']:.2f}%\n"
            f"  Specificity: {train_metrics['specificity']:.2f}%\n"
            f"  F1: {train_metrics['f1']:.2f}%\n"
            f"  AUC: {train_metrics['auc']:.2f}%\n"
            f"Validation Metrics:\n"
            f"  Loss: {val_metrics['loss']:.4f}\n"
            f"  Accuracy: {val_metrics['accuracy']:.2f}%\n"
            f"  Sensitivity: {val_metrics['sensitivity']:.2f}%\n"
            f"  Specificity: {val_metrics['specificity']:.2f}%\n"
            f"  F1: {val_metrics['f1']:.2f}%\n"
            f"  AUC: {val_metrics['auc']:.2f}%"
        )
        
        # Check early stopping
        if early_stopping:
            if early_stopping(val_metrics['loss']):
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save best model
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
            
    # Save final training history
    history_path = PathConfigurator.get_training_history_path(args, paths, config)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
        
    logging.info(f"Training complete. Training history saved to {history_path}")
