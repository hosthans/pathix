from datetime import datetime
import json
import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader




project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.paths.pathconfig import PathConfigurator
from src.core.factories import build_model, build_evaluation_model, build_dataset, build_dataset_args, get_training_pipeline, get_evaluation_pipeline, get_validation_threshold
from src.config.config_loader import load_config, load_config_new
from src.utils.training_utils import set_all_seeds
from src.utils.logging import setup_logging, quiet_external_loggers
from src.utils.model_utils import load_model, infer_checkpoint_in_dim
from src.utils import training_utils
from src.utils import metrics_utils
from src.utils.graph_patch_utils import resolve_negative_label_source
from src.core.meta_processor import MetaData
from src.utils.dataset_utils import mil_collate_no_embeddings, mil_collate
from src.pipelines.train import train_graph_segmentation_model
from src.pipelines.gsdb_parity_train import is_gsdb_parity_train, train_model_gsdb_parity



class Holder:
    """
    Zentrale Laufzeit-Hülle für:
    - Konfiguration
    - Modell
    - Daten
    """

    def __init__(self, args, paths):
        """Initialize the runtime holder from parsed args and project paths, then load config."""
        self.args = args
        self.paths = paths
        self.meta_data = MetaData()

        self.config = None
        self.arch_config = None
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self._eval_val_loaded = False

        set_all_seeds(
            getattr(args, 'seed', 42),
            deterministic=bool(getattr(args, 'deterministic', False)),
            strict=bool(getattr(args, 'strict_deterministic', False)),
        )

        self.load_config()

        self.logging_path = PathConfigurator.get_logging_path(self.args, self.paths, self.config)
        self.checkpoint_path = PathConfigurator.get_checkpoint_path(self.args, self.paths, self.config)
        self.evaluation_path = PathConfigurator.get_evaluation_path(self.args, self.paths, self.config)
        self.training_path = PathConfigurator.get_training_path(self.args, self.paths, self.config)

    def load_config(self):
        """Load and resolve the task/model config, reconciling it with the CLI args."""

        config_name = getattr(self.args, 'config_name', None)

        self.config = load_config_new(
            self.paths['CONFIG_DIR'],
            self.args.task,
            self.args.mil,
            self.args.architecture,
            dataset=getattr(self.args, 'dataset', 'gsdb'),
            config_name=config_name,
        )

        config_label_mode = str(self.config.get('label_mode', 'binary')).strip().lower()
        config_num_classes = int(self.config.get('num_classes', 1))

        requested_label_mode = getattr(self.args, 'label_mode', None)
        resolved_label_mode = str(requested_label_mode).strip().lower() if requested_label_mode is not None else config_label_mode
        if resolved_label_mode not in {'binary', 'multiclass'}:
            raise ValueError("label_mode must be one of: binary, multiclass")

        requested_num_classes = getattr(self.args, 'num_classes', None)
        if requested_num_classes is None:
            resolved_num_classes = 1 if resolved_label_mode == 'binary' else config_num_classes
        else:
            resolved_num_classes = int(requested_num_classes)

        if resolved_label_mode == 'binary':
            if resolved_num_classes != 1:
                logging.warning(
                    "Binary label_mode requested with num_classes=%s. Forcing num_classes=1.",
                    resolved_num_classes,
                )
            resolved_num_classes = 1
        elif resolved_num_classes <= 1:
            raise ValueError("Multiclass mode requires num_classes > 1 (from config or CLI)")

        self.args.label_mode = resolved_label_mode
        self.args.num_classes = resolved_num_classes

        self.config['label_mode'] = resolved_label_mode
        self.config['num_classes'] = resolved_num_classes

        # Same "CLI overrides, config provides the default" pattern as label_mode/num_classes
        # above -- lets a model config YAML declare label_field directly (e.g. CAMELYON's
        # "subtype"/"stage", GSDB's "status"/"tissue_type") so a training run only needs
        # --config_name, without also having to redundantly pass --label_field on the CLI. See
        # WSICachedGraphDataset's label_field docstring (src/dataset/graph_dataset.py) for what
        # this actually does at data-loading time.
        requested_label_field = getattr(self.args, 'label_field', None)
        resolved_label_field = requested_label_field if requested_label_field is not None else self.config.get('label_field', None)
        self.args.label_field = resolved_label_field
        self.config['label_field'] = resolved_label_field

        # A model config may pin he_only (e.g. GSDB-parity CNN configs); an explicit --he_only
        # on the CLI still wins. Resolved here, before any path/meta naming reads args.he_only.
        if 'he_only' not in getattr(self.args, '_cli_overrides', set()) and self.config.get('he_only') is not None:
            self.args.he_only = bool(self.config['he_only'])

        if hasattr(self.args, 'batch_size') and self.args.batch_size is not None:
            self.config['batch_size'] = self.args.batch_size

        meta_config = getattr(self.args, 'model_config', None)
        if isinstance(meta_config, dict) and meta_config:
            self.config.update(meta_config)

        self.arch_config = self.config['architecture']

    def load_model(self):
        """Build the model from config and optionally load a fine-tuning checkpoint."""
        if is_gsdb_parity_train(self.config, self.args):
            # GSDB builds the model inside training, right after seeding (RNG order + no CUDA use
            # before gsdb_parity_train resets the runtime); see src/pipelines/gsdb_parity_train.py.
            logging.info("GSDB-parity training: model is built inside train_model_gsdb_parity")
            self.model = None
            return self.model
        self.model = build_model(self.args, self.config)

        fine_tuning_config = self.config.get('fine_tuning', {})
        if not isinstance(fine_tuning_config, dict):
            fine_tuning_config = {}

        checkpoint_value = fine_tuning_config.get('pretrained_checkpoint')
        if checkpoint_value:
            checkpoint_path = Path(checkpoint_value)
            if not checkpoint_path.is_file():
                candidate_bases = [
                    self.paths.get('MODELS_DIR'),
                    self.paths.get('RESULTS_DIR'),
                    self.paths.get('CHECKPOINT_DIR'),
                    self.paths.get('TRAINING_DIR'),
                ]
                for base_dir in candidate_bases:
                    if base_dir is None:
                        continue
                    candidate_path = Path(base_dir) / checkpoint_value
                    if candidate_path.is_file():
                        checkpoint_path = candidate_path
                        break

            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Fine-tuning checkpoint not found: {checkpoint_value}")

            logging.info(f"Loading fine-tuning checkpoint from {checkpoint_path}")
            self.model = load_model(
                checkpoint_path,
                self.model,
                torch.device('cpu'),
                architecture=self.args.architecture,
                strict=bool(fine_tuning_config.get('strict_load', False)),
            )

        self.model = training_utils.apply_fine_tuning_config(self.model, fine_tuning_config)
        logging.info(f"Model loaded: {self.model.__class__.__name__}")
        return self.model

    def load_model_from_checkpoint(self, checkpoint_path: Path, device: torch.device):
        """Build the evaluation model and load its weights from a checkpoint."""
        eval_config = self.config
        if isinstance(self.config, dict):
            inferred_in_dim = infer_checkpoint_in_dim(checkpoint_path)
            declared_in_dim = self.config.get('in_dim')
            if inferred_in_dim is not None and inferred_in_dim != declared_in_dim:
                logging.info(
                    "Checkpoint's actual input dim (%d) differs from the model config's declared "
                    "in_dim (%s) -- building the eval model with the checkpoint's real shape "
                    "instead.",
                    inferred_in_dim, declared_in_dim,
                )
                eval_config = {**self.config, 'in_dim': inferred_in_dim}

        self.model = build_evaluation_model(self.meta_data.model_type, self.meta_data.architecture, self.meta_data.input_type, checkpoint_path, device, self.meta_data.feature_extractor, eval_config)
        strict_load = False
        fine_tuning_config = self.config.get('fine_tuning', {}) if isinstance(self.config, dict) else {}
        if isinstance(fine_tuning_config, dict):
            strict_load = bool(fine_tuning_config.get('strict_load', False))

        self.model = load_model(
            checkpoint_path,
            self.model,
            device,
            architecture=self.meta_data.architecture,
            strict=strict_load,
        )
        logging.info(f"Model loaded from checkpoint: {self.model.__class__.__name__}")
        return self.model

    def load_data(self):
        """Build the train and validation datasets."""
        if is_gsdb_parity_train(self.config, self.args):
            logging.info("GSDB-parity training: datasets are built inside train_model_gsdb_parity")
            return self.train_dataset, self.val_dataset
        self.train_dataset = build_dataset(self.args, self.paths, self.config, split='train')
        self.val_dataset = build_dataset(self.args, self.paths, self.config, split='val')
        return self.train_dataset, self.val_dataset

    def _resolve_eval_dataset_context(self):
        """Dataset name, label mode and pipeline name shared by every evaluation dataset."""
        dataset_name = getattr(self.args, 'dataset', 'gsdb')
        label_mode = getattr(self.meta_data, 'label_mode', None) or getattr(self.args, 'label_mode', 'binary')

        if getattr(self.args, 'graph_target_dir', None) is None:
            meta_graph_target_dir = getattr(self.meta_data, 'meta_dict', {}).get('graph_target_dir')
            if meta_graph_target_dir:
                self.args.graph_target_dir = meta_graph_target_dir

        pipeline_name = (
            getattr(self.meta_data, 'pipeline', None)
            or getattr(self.args, 'pipeline', None)
            or getattr(self.args, 'pipeline_type', None)
        )

        logging.info(
            "Evaluation dataset resolution: dataset=%s, pipeline=%s, graph_target_dir=%s",
            dataset_name,
            pipeline_name,
            getattr(self.args, 'graph_target_dir', None),
        )
        return dataset_name, label_mode, pipeline_name

    def _build_eval_dataset(self, split_file: str):
        dataset_name, label_mode, pipeline_name = self._resolve_eval_dataset_context()
        return build_dataset_args(dataset_name, self.meta_data.task, self.meta_data.input_type, self.meta_data.embedding_type, self.meta_data.mil, self.meta_data.cached, pipeline_name, self.config, split=split_file, label_mode=label_mode, paths=self.paths, args=self.args)

    def load_val_data(self):
        """Build the validation dataset used for threshold selection.

        Call it once per evaluation, before any test split: the thresholds are computed from it, and
        only afterwards are the test split(s) loaded (load_test_data builds the test dataset only).
        """
        self.val_dataset = self._build_eval_dataset('val')
        self._eval_val_loaded = True
        return self.val_dataset

    def load_test_data(self, test_split='test'):
        """Build the test dataset of one split (the validation dataset is built by load_val_data)."""
        # Accept both "test_scanner2" and "test_scanner2.csv" -- the split is always the exact CSV
        # named here (never the *_HE.csv variant), independent of the training run's --he_only.
        test_split_file = test_split if str(test_split).endswith('.csv') else f"{test_split}.csv"
        self.test_dataset = self._build_eval_dataset(test_split_file)

        if hasattr(self.test_dataset, 'graph_dirs'):
            graph_dirs = [str(p) for p in getattr(self.test_dataset, 'graph_dirs', [])[:5]]
            logging.info("Evaluation graph search dirs (up to 5): %s", graph_dirs)

        return self.test_dataset

    def _init_metadata(self):
        self.meta_data.task = self.args.task
        self.meta_data.dataset = getattr(self.args, 'dataset', 'gsdb')
        self.meta_data.model_type = self.args.model_type
        self.meta_data.architecture = self.args.architecture
        self.meta_data.input_type = self.args.input_type
        self.meta_data.embedding_type = self.args.embedding_type
        self.meta_data.cached = self.args.cached if hasattr(self.args, 'cached') else False
        self.meta_data.pipeline = (
            getattr(self.args, 'pipeline', None)
            or getattr(self.args, 'pipeline_type', None)
        )
        self.meta_data.meta_dict['graph_target_dir'] = (
            str(getattr(self.args, 'graph_target_dir', None))
            if getattr(self.args, 'graph_target_dir', None) is not None
            else None
        )
        self.meta_data.meta_dict['config_name'] = (
            getattr(self.args, 'config_name', None)
            or self.config.get('config_name')
        )
        self.meta_data.meta_dict['split_phase'] = getattr(self.args, 'split_phase', None)
        self.meta_data.meta_dict['segmentation'] = bool(getattr(self.args, 'segmentation', False))
        self.meta_data.meta_dict['cv_fold'] = getattr(self.args, 'cv_fold', None)
        self.meta_data.meta_dict['seed'] = getattr(self.args, 'seed', None)
        self.meta_data.meta_dict['prefix'] = PathConfigurator.get_prefix(self.args)
        self.meta_data.meta_dict['label_field'] = getattr(self.args, 'label_field', None)
        self.meta_data.meta_dict['num_classes'] = getattr(self.args, 'num_classes', None)
        self.meta_data.training_method = self.method.__class__.__name__ if self.method else None
        self.meta_data.feature_extractor = self.args.feature_extractor if hasattr(self.args, 'feature_extractor') else None
        self.meta_data.start_timestamp = datetime.now().isoformat()
        self.meta_data.mil = self.args.mil
        self.meta_data.he_only = bool(getattr(self.args, 'he_only', False))
        self.meta_data.scanner1_only = bool(getattr(self.args, 'scanner1_only', False))
        self.meta_data.label_mode = self.args.label_mode
        self.meta_data.model_config = self.config
        fine_tuning_cfg = self.config.get('fine_tuning', {}) if isinstance(self.config, dict) else {}
        self.meta_data.fine_tune = bool(isinstance(fine_tuning_cfg, dict) and fine_tuning_cfg.get('enabled', False))
        self.meta_data.checkpoint_path = PathConfigurator.get_checkpoint_path(self.args, self.paths, self.config)

    @staticmethod
    def _resolve_runtime_task(task: str) -> str:
        return 'inflammation' if str(task).strip().lower() == 'lymph' else task

    @staticmethod
    def _get_split_target_col(dataset):
        return getattr(dataset, 'split_target_col', None)


    def train(self):
        """Run the full training routine for the configured model and pipeline."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.training_path
        model_dir.mkdir(parents=True, exist_ok=True)
        self.logging_path.mkdir(parents=True, exist_ok=True)
        from src.core.meta_processor import MetaData
        meta = MetaData()
        meta.task = self.args.task
        meta.dataset = getattr(self.args, 'dataset', 'gsdb')
        meta.meta_dict['config_name'] = getattr(self.args, 'config_name', None) or self.config.get('config_name')
        meta.model_type = getattr(self.args, 'model_type', None)
        meta.architecture = getattr(self.args, 'architecture', None)
        meta.input_type = getattr(self.args, 'input_type', None)
        meta.embedding_type = getattr(self.args, 'embedding_type', None)
        meta.pipeline = getattr(self.args, 'pipeline', None) or getattr(self.args, 'pipeline_type', None)
        meta.feature_extractor = getattr(self.args, 'feature_extractor', None)
        meta.fine_tune = bool(getattr(self.args, 'fine_tune', False))
        meta.he_only = bool(getattr(self.args, 'he_only', False))
        meta.scanner1_only = bool(getattr(self.args, 'scanner1_only', False))
        meta.meta_dict['prefix'] = PathConfigurator.get_prefix(self.args)
        model_name = meta.get_consistent_model_name(with_timestamp=False)
        log_file = setup_logging(self.logging_path, operation=f"training_{model_name}")
        quiet_external_loggers()
        logging.info(f"Training logs initialized at {log_file}")

        try:
            from src.utils.split_logging import log_split_info
            if self.train_dataset is not None and hasattr(self.train_dataset, 'split_df'):
                log_split_info(
                    split_name='train',
                    split_path=getattr(self.train_dataset, 'split_df', None),
                    split_df=self.train_dataset.split_df,
                    output_dir=self.logging_path,
                    target_col=self._get_split_target_col(self.train_dataset)
                )
            if self.val_dataset is not None and hasattr(self.val_dataset, 'split_df'):
                log_split_info(
                    split_name='val',
                    split_path=getattr(self.val_dataset, 'split_df', None),
                    split_df=self.val_dataset.split_df,
                    output_dir=self.logging_path,
                    target_col=self._get_split_target_col(self.val_dataset)
                )
        except Exception as e:
            logging.warning(f"Split-Info-Logging (train/val) fehlgeschlagen: {e}")

        use_segmentation = bool(getattr(self.args, 'segmentation', False)) or bool(self.config.get('use_segmentation_training', False))
        self.args.segmentation = use_segmentation
        if use_segmentation:
            if str(getattr(self.args, 'model_type', '')).lower() != 'gnn' or str(getattr(self.args, 'input_type', '')).lower() != 'graph':
                raise ValueError("--segmentation requires --model_type gnn and --input_type graph")
            self.method = train_graph_segmentation_model
            logging.info("Using graph segmentation training routine (--segmentation or config's use_segmentation_training)")
        elif is_gsdb_parity_train(self.config, self.args):
            self.method = train_model_gsdb_parity
            logging.info("Using GSDB-parity training routine (config gsdb_parity_train)")
        else:
            self.method = get_training_pipeline(self.args)

        self._init_metadata()
        logging.info(f"MetaData initialized: {self.meta_data.__dict__}")

        logging.info(f"type of training method={type(self.method)}")
        logging.info(f"Resolved training config: {json.dumps(self.config, indent=2, default=str)}")


        logging.info(f"Starting training for {self.args.task} task")
        logging.info("Command line arguments:")
        for arg, value in vars(self.args).items():
            logging.info(f"  {arg}: {value}")


        logging.info("Project paths:")
        for path_name, path_value in self.paths.items():
            logging.info(f"  {path_name}: {path_value}")


        set_all_seeds(self.args.seed, deterministic=bool(getattr(self.args, 'deterministic', False)))

        logging.info(f"Starting training for {self.args.task} task")
        logging.info("Command line arguments:")
        for arg, value in vars(self.args).items():
            logging.info(f"  {arg}: {value}")


        logging.info("Project paths:")
        for path_name, path_value in self.paths.items():
            logging.info(f"  {path_name}: {path_value}")

        optimizer_config = self.config.get("optimizer", {})
        if optimizer_config == {}:
            logging.warning("No optimizer configuration found in config file. Using default values.")
            optimizer_config = {
                "name": "adamw",
                "momentum": 0.9
            }

        training_args = {
            "args": self.args,
            "model": self.model,
            "train_dataset": self.train_dataset,
            "val_dataset": self.val_dataset,
            "config": self.config,
            "optimizer_name": optimizer_config.get("name", "adamw"),
            "momentum": optimizer_config.get("momentum", 0.9),
            "paths": self.paths
        }

        self.method(**training_args)

    def test(self, output_dir: Path = None, validation_thresholds: dict | None = None):
        """Run the full evaluation routine for the configured model and pipeline."""
        seed = getattr(self.args, 'seed', None) or (self.config.get('seed') if self.config else None)

        if seed is not None:
            set_all_seeds(seed, deterministic=bool(getattr(self.args, 'deterministic', False)))
        self.method = get_evaluation_pipeline(self.meta_data.task, self.meta_data.model_type, self.meta_data.input_type)

        eval_output_dir = output_dir if output_dir is not None else self.evaluation_path
        os.makedirs(eval_output_dir, exist_ok=True)

        from src.core.meta_processor import MetaData
        meta = MetaData()
        meta.task = getattr(self.meta_data, 'task', None)
        meta.dataset = getattr(self.meta_data, 'dataset', None) or getattr(self.args, 'dataset', 'gsdb')
        meta.meta_dict['config_name'] = self.meta_data.meta_dict.get('config_name') if hasattr(self.meta_data, 'meta_dict') else None
        meta.model_type = getattr(self.meta_data, 'model_type', None)
        meta.architecture = getattr(self.meta_data, 'architecture', None)
        meta.input_type = getattr(self.meta_data, 'input_type', None)
        meta.embedding_type = getattr(self.meta_data, 'embedding_type', None)
        meta.pipeline = getattr(self.meta_data, 'pipeline', None)
        meta.feature_extractor = getattr(self.meta_data, 'feature_extractor', None)
        meta.label_mode = getattr(self.meta_data, 'label_mode', None)
        meta.fine_tune = bool(getattr(self.meta_data, 'fine_tune', False))
        meta.he_only = bool(getattr(self.meta_data, 'he_only', False))
        meta.scanner1_only = bool(getattr(self.meta_data, 'scanner1_only', False))
        meta.meta_dict['prefix'] = PathConfigurator.get_eval_prefix(self.args)
        model_name = meta.get_consistent_model_name(with_timestamp=False)
        setup_logging(eval_output_dir, operation=f"evaluation_{model_name}")
        quiet_external_loggers()

        try:
            from src.utils.split_logging import log_split_info
            if self.test_dataset is not None and hasattr(self.test_dataset, 'split_df'):
                log_split_info(
                    split_name='test',
                    split_path=getattr(self.test_dataset, 'split_df', None),
                    split_df=self.test_dataset.split_df,
                    output_dir=eval_output_dir,
                    target_col=self._get_split_target_col(self.test_dataset)
                )
            if self.val_dataset is not None and hasattr(self.val_dataset, 'split_df'):
                log_split_info(
                    split_name='val',
                    split_path=getattr(self.val_dataset, 'split_df', None),
                    split_df=self.val_dataset.split_df,
                    output_dir=eval_output_dir,
                    target_col=self._get_split_target_col(self.val_dataset)
                )
        except Exception as e:
            logging.warning(f"Split-Info-Logging (test/val) fehlgeschlagen: {e}")

        logging.info(f"Starting evaluation with method={type(self.method)}")
        logging.info("Command line arguments:")
        for arg, value in vars(self.args).items():
            logging.info(f"  {arg}: {value}")

        logging.info("Project paths:")
        for path_name, path_value in self.paths.items():
            logging.info(f"  {path_name}: {path_value}")

        if self.meta_data.input_type == 'graph':
            dataloader = GeoDataLoader(
                self.test_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                worker_init_fn=training_utils.dataloader_worker_init_fn,
            )
        else:
            if self.meta_data.mil:
                collate_fn = (
                    mil_collate_no_embeddings
                    if self.meta_data.input_type == "image"
                    else mil_collate
                )

                dataloader = DataLoader(
                    self.test_dataset,
                    batch_size=self.config['batch_size'],
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                    collate_fn=collate_fn,
                    worker_init_fn=training_utils.dataloader_worker_init_fn,
                )
            else:
                dataloader = DataLoader(
                    self.test_dataset,
                    batch_size=self.config['batch_size'],
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                    worker_init_fn=training_utils.dataloader_worker_init_fn,
                )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        validation_dir = eval_output_dir / "validation_thresholds"
        validation_dir.mkdir(exist_ok=True)

        is_graph_input = str(self.meta_data.input_type) == 'graph'
        negative_label_source = None
        if is_graph_input:
            negative_label_source = resolve_negative_label_source(self.config)

        if validation_thresholds is None:
            logging.info("No pre-computed thresholds found. Calculating using validation data only...")
            if not self._eval_val_loaded:
                self.load_val_data()
            val_method = get_validation_threshold(
                input_type=self.meta_data.input_type,
                embedding_type=self.meta_data.embedding_type,
                mil=self.meta_data.mil,
            )

            val_kwargs = {
                "model": self.model,
                "val_dataset": self.val_dataset,
                "task": self._resolve_runtime_task(self.meta_data.task),
                "output_dir": validation_dir,
            }
            if is_graph_input:
                val_kwargs["negative_label_source"] = negative_label_source

            validation_thresholds = val_method(**val_kwargs)

            logging.info(f"Validation thresholds calculated and saved to {validation_dir / 'validation_thresholds.json'}")

        evaluation_args = {
            "model": self.model,
            "data_loader": dataloader,
            "task": self._resolve_runtime_task(self.meta_data.task),
            "split": self.args.test_split,
            "device": device,
            "output_dir": eval_output_dir,
            "validation_thresholds": validation_thresholds,
        }
        is_mil_multimodal = str(self.meta_data.input_type) == 'multimodal' and bool(self.meta_data.mil)
        if is_graph_input or is_mil_multimodal:
            evaluation_args["save_plots"] = not bool(getattr(self.args, 'no_plots', False))
        if is_graph_input:
            evaluation_args["negative_label_source"] = negative_label_source
            # Mirror the run's own training-time mixed_precision setting (train.py's train_graph_
            # model defaults it to False the same way) instead of evaluate_model_graph always
            # autocasting fp16 on CUDA regardless of config -- see its use_mixed_precision docs.
            evaluation_args["use_mixed_precision"] = bool((self.config or {}).get('mixed_precision', False))
            # Post-hoc per-class prediction bias (see evaluate_model_graph's class_decision_bias
            # docs) -- e.g. {"itc": 1.5} to trade some precision for recall on a class the model
            # is otherwise too conservative about. Opt-in via config, no effect when absent.
            evaluation_args["class_decision_bias"] = (self.config or {}).get('class_decision_bias')
            # Same filter the dataset was built with (see WSICachedGraphDataset.exclude_label_field_
            # values) -- keeps class-name resolution in sync with what the model's output layer
            # actually has slots for (e.g. a tumor-positive-only 3-way itc/macro/micro head).
            evaluation_args["exclude_label_field_values"] = (self.config or {}).get('exclude_label_field_values')
            evaluation_args["monitor_metric"] = (self.config or {}).get('segmentation', {}).get('monitor_metric')
            evaluation_args["label_field"] = getattr(self.args, 'label_field', None)
            # Per-stain overview of correctly classified slides -- joined on slide_name.
            evaluation_args["slide_groups"] = getattr(self.test_dataset, 'split_df', None)
            segmentation_cfg = (self.config or {}).get('segmentation', {}) or {}
            evaluation_args["is_multiclass_only_run"] = (
                (getattr(self.model, 'region_cls_enabled', False) or getattr(self.model, 'patch_cls_enabled', False))
                and float(segmentation_cfg.get('region_loss_weight', 1.0)) <= 0.0
                and float(segmentation_cfg.get('slide_loss_weight', 0.08)) <= 0.0
                and float(segmentation_cfg.get('patch_loss_weight', 0.0)) <= 0.0
            )

        return self.method(**evaluation_args)