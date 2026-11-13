import sys
from pathlib import Path
import logging

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


from src.core.registries import MODEL_REGISTRY, DATASET_REGISTRY, TRAINING_REGISTRY, EVALUATION_REGISTRY, VALIDATION_THRESHOLD_REGISTRY
from src.pipelines.train import get_transforms


import src.dataset
import src.models
import src.pipelines


def _resolve_runtime_task(task: str) -> str:
    return 'inflammation' if str(task).strip().lower() == 'lymph' else task


def build_evaluation_model(model_type, architecture, input_type, model_path, device, feature_extractor, config):
    """Instantiate a model for evaluation based on model_type."""
    key = (model_type, 'any' if model_type in ['multimodal', 'mil'] else architecture, input_type)

    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{key}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[key]




    if model_type in ['mil', 'multimodal'] and input_type in ['image', 'multimodal']:
        model_name = feature_extractor
    else:
        model_name = architecture

    model_params = {
        'model_name': model_name,
        'num_classes': config.get('num_classes', 1),
        'dropout_rate': getattr(config, 'dropout_rate', config.get('dropout_rate', 0.2)),
        'in_dim': config.get('in_dim', 1537),
    }


    if model_type == 'mil':

        model_params['instance_dim'] = config.get('instance_dim', 256)
        model_params['tile_embed_dim'] = config.get('tile_embed_dim', 1536)
        model_params['freeze_backbone'] = config.get('architecture', {}).get('feature_extraction', True)

    elif model_type == 'gnn':

        model_params['hidden_dim'] = config.get('hidden_dim', 256)
        model_params['num_layers'] = config.get('num_layers', 3 if architecture == 'wsigraph' else 4)
        if architecture == 'tissue_region_net':
            model_params['heads'] = config.get('heads', 4)
            model_params['edge_dropout_p'] = config.get('edge_dropout_p', 0.05)
            model_params['region_aux_weight'] = config.get('region_aux_weight', 0.20)
            model_params['region_aux_pos_weight_cap'] = config.get('region_aux_pos_weight_cap', 6.0)
            model_params['mixed_region_weight'] = config.get('mixed_region_weight', 0.20)
            model_params['patch_hidden_dim'] = config.get('patch_hidden_dim', None)
            model_params['patch_num_layers'] = config.get('patch_num_layers', 2)
            model_params['patch_aux_weight'] = config.get('patch_aux_weight', 0.30)
            model_params['patch_aux_pos_weight_cap'] = config.get('patch_aux_pos_weight_cap', 6.0)
            model_params['patch_edge_dropout_p'] = config.get('patch_edge_dropout_p', 0.05)
            model_params['max_patch_nodes_for_full_forward'] = config.get('max_patch_nodes_for_full_forward', 6000)
            model_params['max_patch_edges_for_full_forward'] = config.get('max_patch_edges_for_full_forward', 150000)
            model_params['graph_instance_norm'] = config.get('graph_instance_norm', False)
            model_params['patch_heads'] = config.get('patch_heads', None)
            model_params['use_edge_features'] = config.get('use_edge_features', True)
            model_params['edge_feat_dim'] = config.get('edge_feat_dim', 8)
            model_params['domain_adversarial'] = config.get('domain_adversarial', False)
            model_params['domain_adversarial_lambda'] = config.get('domain_adversarial_lambda', 1.0)
            model_params['region_cls_target_field'] = config.get('region_cls_target_field', None)
            model_params['region_cls_class_names'] = config.get('region_cls_class_names', None)
            model_params['region_cls_weight'] = config.get('region_cls_weight', 0.30)
            model_params['patch_cls_target_field'] = config.get('patch_cls_target_field', None)
            model_params['patch_cls_class_names'] = config.get('patch_cls_class_names', None)
            model_params['patch_cls_weight'] = config.get('patch_cls_weight', 0.30)
            model_params['use_extent_features'] = config.get('use_extent_features', False)
        if architecture == 'region_slide_net':
            model_params['heads'] = config.get('heads', 4)
            model_params['edge_dropout_p'] = config.get('edge_dropout_p', 0.1)
            model_params['graph_instance_norm'] = config.get('graph_instance_norm', True)
            model_params['use_patch_evidence'] = config.get('use_patch_evidence', True)
            model_params['patch_evidence_hidden'] = config.get('patch_evidence_hidden', 32)
            model_params['use_edge_features'] = config.get('use_edge_features', True)
            model_params['edge_feat_dim'] = config.get('edge_feat_dim', 8)
            model_params['domain_adversarial'] = config.get('domain_adversarial', False)
            model_params['domain_adversarial_lambda'] = config.get('domain_adversarial_lambda', 1.0)
        if architecture in ['wsigraph', 'wsigraph-pretraining']:
            model_params['max_patch_nodes_for_full_forward'] = config.get('max_patch_nodes_for_full_forward', 10000)
            model_params['max_patch_edges_for_full_forward'] = config.get('max_patch_edges_for_full_forward', 200000)
            model_params['pretraining_consistency_weight'] = config.get('pretraining_consistency_weight', 0.25)
            model_params['region_only_warn_every'] = config.get('region_only_warn_every', 25)
            model_params['region_aux_weight'] = config.get('region_aux_weight', 0.05)
            model_params['patch_aux_weight'] = config.get('patch_aux_weight', 0.02)
            model_params['max_patch_aux_nodes'] = config.get('max_patch_aux_nodes', 4096)
            model_params['micro_topk_ratio'] = config.get('micro_topk_ratio', 0.08)
            model_params['micro_logit_weight'] = config.get('micro_logit_weight', 0.35)
            model_params['edge_gate_strength'] = config.get('edge_gate_strength', 0.20)
            model_params['min_patches_per_region_in_large_graph'] = config.get('min_patches_per_region_in_large_graph', 1)
            model_params['patch_edge_dropout_p'] = config.get('patch_edge_dropout_p', 0.0)
            model_params['region_edge_dropout_p'] = config.get('region_edge_dropout_p', 0.0)
            model_params['sanitize_non_finite'] = config.get('sanitize_non_finite', True)
            model_params['logit_clamp_value'] = config.get('logit_clamp_value', 30.0)
        if architecture in ['graph_v2', 'hetero_wsi', 'hetero_wsi_v2', 'hetero_wsi_adaptive', 'hetero_wsi_dynamical']:
            model_params['heads'] = config.get('heads', 4)
        if architecture in ['hetero_wsi_v2', 'hetero_wsi_adaptive', 'hetero_wsi_dynamical']:
            model_params['drop_edge_p'] = config.get('drop_edge_p', 0.1)
        if architecture in ['hetero_wsi', 'hetero_wsi_v2', 'hetero_wsi_adaptive', 'hetero_wsi_dynamical']:
            model_params['use_edge_features'] = config.get('use_edge_features', True)
            model_params['learnable_edge_features'] = config.get('learnable_edge_features', True)
            model_params['use_hierarchical_edge_features'] = config.get('use_hierarchical_edge_features', True)
            model_params['edge_semantic_weight'] = config.get('edge_semantic_weight', 0.25)
        if architecture == 'hetero_wsi_dynamical':
            model_params['max_base_nodes_for_full_forward'] = config.get('max_base_nodes_for_full_forward', 20000)
            model_params['max_base_edges_for_full_forward'] = config.get('max_base_edges_for_full_forward', 400000)
            model_params['min_base_nodes_per_region_in_large_graph'] = config.get('min_base_nodes_per_region_in_large_graph', 1)

    elif model_type == 'multimodal' and input_type == 'multimodal':

        model_params['embedding_dim'] = config.get('embedding_dim', 1536)
        model_params['proj_dim'] = config.get('proj_dim', 256)

    logging.info(f"Building evaluation model with params: {model_params}")
    return model_cls(**model_params)

def build_model(args, config):
    """Instantiate a model based on args.model_type."""
    key = (args.model_type, 'any' if args.model_type in ['multimodal', 'mil'] else args.architecture, args.input_type)

    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{key}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[key]




    if args.model_type in ['mil', 'multimodal'] and args.input_type in ['image', 'multimodal']:
        model_name = args.feature_extractor
    else:
        model_name = args.architecture

    model_params = {
        'model_name': model_name,
        'num_classes': getattr(args, 'num_classes', 1),
        'dropout_rate': getattr(args, 'dropout_rate', config.get('dropout_rate', 0.2)),
        'in_dim': config.get('in_dim', 1537),
    }


    if args.model_type == 'mil':

        model_params['instance_dim'] = config.get('instance_dim', 256)
        model_params['tile_embed_dim'] = config.get('tile_embed_dim', 1536)
        model_params['freeze_backbone'] = config.get('architecture', {}).get('feature_extraction', True)

    elif args.model_type == 'gnn':

        model_params['hidden_dim'] = config.get('hidden_dim', 256)
        model_params['num_layers'] = config.get('num_layers', 3 if args.architecture == 'wsigraph' else 4)
        if args.architecture == 'tissue_region_net':
            model_params['heads'] = config.get('heads', 4)
            model_params['edge_dropout_p'] = config.get('edge_dropout_p', 0.05)
            model_params['region_aux_weight'] = config.get('region_aux_weight', 0.20)
            model_params['region_aux_pos_weight_cap'] = config.get('region_aux_pos_weight_cap', 6.0)
            model_params['mixed_region_weight'] = config.get('mixed_region_weight', 0.20)
            model_params['patch_hidden_dim'] = config.get('patch_hidden_dim', None)
            model_params['patch_num_layers'] = config.get('patch_num_layers', 2)
            model_params['patch_aux_weight'] = config.get('patch_aux_weight', 0.30)
            model_params['patch_aux_pos_weight_cap'] = config.get('patch_aux_pos_weight_cap', 6.0)
            model_params['patch_edge_dropout_p'] = config.get('patch_edge_dropout_p', 0.05)
            model_params['max_patch_nodes_for_full_forward'] = config.get('max_patch_nodes_for_full_forward', 6000)
            model_params['max_patch_edges_for_full_forward'] = config.get('max_patch_edges_for_full_forward', 150000)
            model_params['graph_instance_norm'] = config.get('graph_instance_norm', False)
            model_params['patch_heads'] = config.get('patch_heads', None)
            model_params['use_edge_features'] = config.get('use_edge_features', True)
            model_params['edge_feat_dim'] = config.get('edge_feat_dim', 8)
            model_params['domain_adversarial'] = config.get('domain_adversarial', False)
            model_params['domain_adversarial_lambda'] = config.get('domain_adversarial_lambda', 1.0)
            model_params['region_cls_target_field'] = config.get('region_cls_target_field', None)
            model_params['region_cls_class_names'] = config.get('region_cls_class_names', None)
            model_params['region_cls_weight'] = config.get('region_cls_weight', 0.30)
            model_params['patch_cls_target_field'] = config.get('patch_cls_target_field', None)
            model_params['patch_cls_class_names'] = config.get('patch_cls_class_names', None)
            model_params['patch_cls_weight'] = config.get('patch_cls_weight', 0.30)
            model_params['use_extent_features'] = config.get('use_extent_features', False)
        if args.architecture == 'region_slide_net':
            model_params['heads'] = config.get('heads', 4)
            model_params['edge_dropout_p'] = config.get('edge_dropout_p', 0.1)
            model_params['graph_instance_norm'] = config.get('graph_instance_norm', True)
            model_params['use_patch_evidence'] = config.get('use_patch_evidence', True)
            model_params['patch_evidence_hidden'] = config.get('patch_evidence_hidden', 32)
            model_params['use_edge_features'] = config.get('use_edge_features', True)
            model_params['edge_feat_dim'] = config.get('edge_feat_dim', 8)
            model_params['domain_adversarial'] = config.get('domain_adversarial', False)
            model_params['domain_adversarial_lambda'] = config.get('domain_adversarial_lambda', 1.0)
        if args.architecture in ['wsigraph', 'wsigraph-pretraining']:
            model_params['max_patch_nodes_for_full_forward'] = config.get('max_patch_nodes_for_full_forward', 10000)
            model_params['max_patch_edges_for_full_forward'] = config.get('max_patch_edges_for_full_forward', 200000)
            model_params['pretraining_consistency_weight'] = config.get('pretraining_consistency_weight', 0.25)
            model_params['region_only_warn_every'] = config.get('region_only_warn_every', 25)
            model_params['region_aux_weight'] = config.get('region_aux_weight', 0.05)
            model_params['patch_aux_weight'] = config.get('patch_aux_weight', 0.02)
            model_params['max_patch_aux_nodes'] = config.get('max_patch_aux_nodes', 4096)
            model_params['micro_topk_ratio'] = config.get('micro_topk_ratio', 0.08)
            model_params['micro_logit_weight'] = config.get('micro_logit_weight', 0.35)
            model_params['edge_gate_strength'] = config.get('edge_gate_strength', 0.20)
            model_params['min_patches_per_region_in_large_graph'] = config.get('min_patches_per_region_in_large_graph', 1)
            model_params['patch_edge_dropout_p'] = config.get('patch_edge_dropout_p', 0.0)
            model_params['region_edge_dropout_p'] = config.get('region_edge_dropout_p', 0.0)
            model_params['sanitize_non_finite'] = config.get('sanitize_non_finite', True)
            model_params['logit_clamp_value'] = config.get('logit_clamp_value', 30.0)
        if args.architecture in ['graph_v2', 'hetero_wsi', 'hetero_wsi_v2', 'hetero_wsi_adaptive', 'hetero_wsi_dynamical']:
            model_params['heads'] = config.get('heads', 4)
        if args.architecture in ['hetero_wsi_v2', 'hetero_wsi_adaptive', 'hetero_wsi_dynamical']:
            model_params['drop_edge_p'] = config.get('drop_edge_p', 0.1)
        if args.architecture in ['hetero_wsi', 'hetero_wsi_v2', 'hetero_wsi_adaptive', 'hetero_wsi_dynamical']:
            model_params['use_edge_features'] = config.get('use_edge_features', True)
            model_params['learnable_edge_features'] = config.get('learnable_edge_features', True)
            model_params['use_hierarchical_edge_features'] = config.get('use_hierarchical_edge_features', True)
            model_params['edge_semantic_weight'] = config.get('edge_semantic_weight', 0.25)
        if args.architecture == 'hetero_wsi_dynamical':
            model_params['max_base_nodes_for_full_forward'] = config.get('max_base_nodes_for_full_forward', 20000)
            model_params['max_base_edges_for_full_forward'] = config.get('max_base_edges_for_full_forward', 400000)
            model_params['min_base_nodes_per_region_in_large_graph'] = config.get('min_base_nodes_per_region_in_large_graph', 1)

    elif args.model_type == 'multimodal' and args.input_type == 'multimodal':

        model_params['embedding_dim'] = config.get('embedding_dim', 1536)
        model_params['proj_dim'] = config.get('proj_dim', 256)

    logging.info(f"Building model with params: {model_params}")
    return model_cls(**model_params)

def _eval_he_only(args, split) -> bool:
    """he_only for an evaluation dataset: the validation split follows the training run (a run
    trained with --he_only selects its thresholds on val_HE, not val_all_stains), every other split
    (test, test_scanner2, ...) is always loaded in full -- all stains -- as before.
    Runs trained without --he_only are unaffected (val stays val_all_stains)."""
    split_name = str(split)[:-4] if str(split).endswith('.csv') else str(split)
    return split_name == 'val' and bool(getattr(args, 'he_only', False))


def build_dataset_args(dataset, task, input_type, embedding_type, mil, cached, pipeline, config, split, label_mode, paths, args):
    """Instantiate a dataset based on (input_type, embedding_type)."""
    key = (dataset, f"{input_type}_cached" if cached else input_type, embedding_type, mil)
    if key not in DATASET_REGISTRY:
        raise ValueError(
            f"No dataset registered for {key}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    dataset_cls = DATASET_REGISTRY[key]

    logging.debug(f"Dataset config for split='{split}': {config}")


    resolved_task = _resolve_runtime_task(task)

    if input_type == 'graph':
        dataset_params = {
            'split': split,
            'transform': None,
            'task': task,
            'dataset': dataset,
            'label_mode': label_mode,
            'label_field': getattr(args, 'label_field', None),
            'exclude_label_field_values': config.get('exclude_label_field_values', None),
            # Always False here, unconditionally: build_dataset_args only ever builds val/test
            # datasets for EVALUATION (Holder.load_val_data / load_test_data -> val_dataset/test_dataset, used by
            # both train_model.py's post-training eval and the standalone evaluate_model.py
            # entrypoint). Evaluation must cover every stain regardless of whether training used
            # --he_only -- it previously read getattr(args, 'he_only', True), which silently
            # restricted every evaluate_model.py run (no --he_only flag exists there, so the
            # attribute is simply absent) to HE-only slides even when training used all stains,
            # e.g. GSDB test_scanner2 dropping from 90 to 30 slides. Deliberately not a
            # CLI-configurable parameter and not persisted to the training meta -- evaluation
            # always sees the full split. Exception: the val split (threshold selection) follows
            # the training run's he_only, see _eval_he_only.
            'he_only': _eval_he_only(args, split),
            'scanner1_only': getattr(args, 'scanner1_only', True if split in ['train', 'val'] else False),
            'paths': paths,
            'split_seed': getattr(args, 'seed', 42),
            'split_phase': getattr(args, 'split_phase', None),
            'k': config.get('k', None),
            'radius': config.get('radius', None),
            'normalize_pos': config.get('normalize_pos', True),
            'embedding_type': embedding_type,
            'pipeline_type': pipeline,
            'graph_target_dir': getattr(args, 'graph_target_dir', None),
            'strict_graph_name_match': config.get('strict_graph_name_match', True),
            'cache_in_memory': config.get('cache_in_memory', False),
            'resplit_large_regions': config.get('resplit_large_regions', False),
            'resplit_target_patches_per_region': config.get('resplit_target_patches_per_region', 250),
            'resplit_k_region': config.get('resplit_k_region', 5),
            'resplit_min_patches_per_region': config.get('resplit_min_patches_per_region', 0),
            'train_augmentations': config.get('train_augmentations', []),
        }
    else:
        dataset_params = {
            'split': split,
            'transform': get_transforms(resolved_task, True if split == 'train' else False, config),
            'task': task,
            # See the graph branch above: test splits always in full (all stains), only the val
            # split follows the training run's he_only (_eval_he_only).
            'he_only': _eval_he_only(args, split),
            'scanner1_only': getattr(args, 'scanner1_only', True if split in ['train', 'val'] else False),
            'paths': paths,
            'label_mode': label_mode,
            'split_seed': getattr(args, 'seed', 42),
            'split_phase': getattr(args, 'split_phase', None),
        }





    # Only CNN tile datasets declare this (GSDB-parity configs); other dataset classes don't accept it.
    if input_type != 'graph' and 'max_white_fraction' in config:
        dataset_params['max_white_fraction'] = config['max_white_fraction']

    return dataset_cls(**dataset_params)


def build_dataset(args, paths, config, split='train'):
    """Instantiate a dataset based on (input_type, embedding_type)."""
    key = (args.dataset, f"{args.input_type}_cached" if getattr(args, 'cached', True) else args.input_type, args.embedding_type, args.mil if hasattr(args, 'mil') else False)

    if key not in DATASET_REGISTRY:
        raise ValueError(
            f"No dataset registered for {key}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    dataset_cls = DATASET_REGISTRY[key]

    logging.debug(f"Dataset config for split='{split}': {config}")


    resolved_task = _resolve_runtime_task(args.task)

    pipeline_name = getattr(args, 'pipeline', None) or getattr(args, 'pipeline_type', None)

    if args.input_type == 'graph':
        dataset_params = {
            'split': split,
            'transform': None,
            'task': args.task,
            'dataset': getattr(args, 'dataset', 'gsdb'),
            'label_mode': getattr(args, 'label_mode', 'binary'),
            'label_field': getattr(args, 'label_field', None),
            'exclude_label_field_values': config.get('exclude_label_field_values', None),
            'he_only': getattr(args, 'he_only', True),
            'scanner1_only': getattr(args, 'scanner1_only', True if split in ['train', 'val'] else False),
            'paths': paths,
            'split_seed': getattr(args, 'seed', 42),
            'split_phase': getattr(args, 'split_phase', None),
            'k': config.get('k', None),
            'radius': config.get('radius', None),
            'normalize_pos': config.get('normalize_pos', True),
            'embedding_type': args.embedding_type,
            'pipeline_type': pipeline_name,
            'graph_target_dir': getattr(args, 'graph_target_dir', None),
            'strict_graph_name_match': config.get('strict_graph_name_match', True),
            'cache_in_memory': config.get('cache_in_memory', False),
            'resplit_large_regions': config.get('resplit_large_regions', False),
            'resplit_target_patches_per_region': config.get('resplit_target_patches_per_region', 250),
            'resplit_k_region': config.get('resplit_k_region', 5),
            'resplit_min_patches_per_region': config.get('resplit_min_patches_per_region', 0),
            'train_augmentations': config.get('train_augmentations', []),
        }
    else:
        dataset_params = {
            'split': split,
            'transform': get_transforms(resolved_task, True if split == 'train' else False, config),
            'task': args.task,
            'he_only': getattr(args, 'he_only', False),
            'scanner1_only': getattr(args, 'scanner1_only', True if split in ['train', 'val'] else False),
            'paths': paths,
            'label_mode': getattr(args, 'label_mode', 'binary'),
            'split_seed': getattr(args, 'seed', 42),
            'split_phase': getattr(args, 'split_phase', None),
        }





    # Only CNN tile datasets declare this (GSDB-parity configs); other dataset classes don't accept it.
    if args.input_type != 'graph' and 'max_white_fraction' in config:
        dataset_params['max_white_fraction'] = config['max_white_fraction']

    return dataset_cls(**dataset_params)


def get_training_pipeline(args):
    """Return the training pipeline class for (task, model_type, input_type)."""
    key = (_resolve_runtime_task(args.task), args.model_type, args.input_type)

    if key not in TRAINING_REGISTRY:
        raise ValueError(
            f"No training pipeline registered for {key}. "
            f"Available: {list(TRAINING_REGISTRY.keys())}"
        )

    pipeline_cls = TRAINING_REGISTRY[key]
    return pipeline_cls

def get_evaluation_pipeline(task, model_type, input_type):
    """Return the evaluation pipeline class for (task, model_type, input_type)."""
    key = (_resolve_runtime_task(task), model_type, input_type)

    if key not in EVALUATION_REGISTRY:
        raise ValueError(
            f"No evaluation pipeline registered for {key}. "
            f"Available: {list(EVALUATION_REGISTRY.keys())}"
        )

    pipeline_cls = EVALUATION_REGISTRY[key]
    return pipeline_cls

def get_validation_threshold(input_type, embedding_type, mil=False):
    """Return the validation threshold class for (input_type, embedding_type, mil)."""
    key = (input_type, embedding_type, mil)
    if key not in VALIDATION_THRESHOLD_REGISTRY:
        raise ValueError(
            f"No validation threshold registered for {key}. "
            f"Available: {list(VALIDATION_THRESHOLD_REGISTRY.keys())}"
        )

    threshold_cls = VALIDATION_THRESHOLD_REGISTRY[key]
    return threshold_cls