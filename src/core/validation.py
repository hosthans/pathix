
import sys
from pathlib import Path

import yaml

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config.config_loader import load_config_new
from src.utils import logging

class Validator:
    """
    Validator-Klasse zur Überprüfung der Konsistenz von Argumenten.
    """
    def __init__(self, args, paths):
        """Initialize the validator with parsed args and project paths, loading model configs."""
        self.args = args
        self.paths = paths
        self._load_config()

    def _load_config(self):
        mil_config_path = self.paths['CONFIG_DIR'] / 'model_config_mil.yaml'
        normal_config_path = self.paths['CONFIG_DIR'] / 'model_config.yaml'
        try:
            with open(mil_config_path, 'r') as f:
                self.mil_config = yaml.safe_load(f)
            try:
                with open(normal_config_path, 'r') as f:
                    self.normal_config = yaml.safe_load(f)
            except FileNotFoundError:
                logging.error(f"Config file not found: {normal_config_path}")
                raise
        except FileNotFoundError:
            logging.error(f"Config file not found: {mil_config_path}")
            raise

    def available_models(self, task: str = "inflammation", mil: bool = False):
        """
        Gibt eine Liste der verfügbaren Modelle zurück, basierend auf den geladenen Konfigurationen.
        """
        task_keys = [task]
        if task == "lymph":
            task_keys.append("inflammation")

        normal_models = set()
        mil_models = set()
        for task_key in task_keys:
            normal_models |= set(self.normal_config.get(task_key, {}).keys())
            mil_models |= set(self.mil_config.get(task_key, {}).keys())

        if getattr(self.args, 'model_type', None) == 'gnn':
            return normal_models | mil_models

        if mil:
            return mil_models
        return normal_models

    def validate_args(self):
        """
        Prüft, ob die Kombination der Argumente logisch konsistent ist.
        """

        if getattr(self.args, 'dataset', None) == 'gsdb' and self.args.task not in {'inflammation', 'tissue'}:
            raise ValueError("Invalid configuration: GSDB supports only tasks 'inflammation' and 'tissue'")

        if getattr(self.args, 'dataset', None) == 'camelyon' and self.args.task != 'lymph':
            raise ValueError("Invalid configuration: CAMELYON supports only task 'lymph'")

        if self.args.model_type == "gnn":
            allowed_architectures = self.available_models(task=self.args.task, mil=False)
        else:
            allowed_architectures = self.available_models(task=self.args.task, mil=self.args.mil)

        if self.args.architecture == "none" or self.args.architecture not in allowed_architectures:
            raise ValueError(
                f"Invalid configuration: Architecture '{self.args.architecture}' is not available for task '{self.args.task}' with model_type={self.args.model_type}"
            )


        if self.args.model_type == "cnn" and self.args.input_type != "image":
            raise ValueError(
                "Invalid configuration: CNN models require image input"
            )

        if self.args.model_type == "gnn" and self.args.input_type != "graph":
            raise ValueError(
                "Invalid configuration: GNN models require graph input"
            )

        if self.args.model_type == "multimodal" and self.args.input_type != "multimodal":
            raise ValueError(
                "Invalid configuration: Multimodal models require multimodal input"
            )

        if self.args.model_type == "multimodal" and self.args.embedding_type == "none":
            raise ValueError(
                "Invalid configuration: Multimodal models require embeddings"
            )

        if self.args.model_type in {"cnn", "mil"} and self.args.input_type == "graph":
            raise ValueError(
                "Invalid configuration: CNN/MIL models do not support graph input"
            )

        if self.args.model_type in {"cnn", "gnn"} and self.args.feature_extractor != "none":
            raise ValueError(
                "Invalid configuration: Feature extractors are only supported for multimodal models"
            )


        if self.args.model_type == "mil" and self.args.input_type not in ["image", 'multimodal', 'graph']:
            raise ValueError(
                "Invalid configuration: MIL expects image-level instances"
            )




        required = ["model_type", "input_type", "task", "architecture"]
        for field in required:
            if not hasattr(self.args, field):
                raise ValueError(f"Missing required argument: {field}")
