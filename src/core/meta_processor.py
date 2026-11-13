
import sys
from pathlib import Path
import os
import yaml
import logging

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.paths.pathconfig import PathConfigurator

class MetaData:
    def __init__(self, meta_dict=None):
        """Wrap a meta dictionary (or start a new empty one)."""
        self.meta_dict = meta_dict if meta_dict is not None else {}

    def get_consistent_model_name(self, with_timestamp: bool = False, timestamp: str = None) -> str:
        """Return the consistent meta/model filename for this metadata."""
        return PathConfigurator.get_meta_name(self.meta_dict, with_timestamp=with_timestamp, timestamp=timestamp)

    @property
    def task(self):
        """Get the task value."""
        return self.meta_dict.get("task")

    @task.setter
    def task(self, value):
        """Set the task value."""
        self.meta_dict["task"] = value

    @property
    def model_type(self):
        """Get the model_type value."""
        return self.meta_dict.get("model_type")

    @model_type.setter
    def model_type(self, value):
        """Set the model_type value."""
        self.meta_dict["model_type"] = value

    @property
    def architecture(self):
        """Get the architecture value."""
        return self.meta_dict.get("architecture")

    @architecture.setter
    def architecture(self, value):
        """Set the architecture value."""
        self.meta_dict["architecture"] = value

    @property
    def feature_extractor(self):
        """Get the feature_extractor value."""
        return self.meta_dict.get("feature_extractor")

    @feature_extractor.setter
    def feature_extractor(self, value):
        """Set the feature_extractor value."""
        self.meta_dict["feature_extractor"] = value


    @property
    def checkpoint_path(self):
        """Get the checkpoint_path value."""
        return self.meta_dict.get("checkpoint_path")

    @checkpoint_path.setter
    def checkpoint_path(self, value):
        """Set the checkpoint_path value."""
        self.meta_dict["checkpoint_path"] = value

    @property
    def training_method(self):
        """Get the training_method value."""
        return self.meta_dict.get("training_method")

    @training_method.setter
    def training_method(self, value):
        """Set the training_method value."""
        self.meta_dict["training_method"] = value

    @property
    def input_type(self):
        """Get the input_type value."""
        return self.meta_dict.get("input_type")

    @input_type.setter
    def input_type(self, value):
        """Set the input_type value."""
        self.meta_dict["input_type"] = value

    @property
    def embedding_type(self):
        """Get the embedding_type value."""
        return self.meta_dict.get("embedding_type")

    @embedding_type.setter
    def embedding_type(self, value):
        """Set the embedding_type value."""
        self.meta_dict["embedding_type"] = value

    @property
    def mil(self):
        """Get the mil value."""
        return self.meta_dict.get("mil", False)

    @mil.setter
    def mil(self, value):
        """Set the mil value."""
        self.meta_dict["mil"] = bool(value)

    @property
    def he_only(self):
        """Get the he_only value (training restricted to HE-stained slides)."""
        return self.meta_dict.get("he_only", False)

    @he_only.setter
    def he_only(self, value):
        """Set the he_only value."""
        self.meta_dict["he_only"] = bool(value)

    @property
    def scanner1_only(self):
        """Get the scanner1_only value (training restricted to scanner 1 slides)."""
        return self.meta_dict.get("scanner1_only", False)

    @scanner1_only.setter
    def scanner1_only(self, value):
        """Set the scanner1_only value."""
        self.meta_dict["scanner1_only"] = bool(value)

    @property
    def cached(self):
        """Get the cached value."""
        return self.meta_dict.get("cached", False)

    @cached.setter
    def cached(self, value):
        """Set the cached value."""
        self.meta_dict["cached"] = bool(value)

    @property
    def pipeline(self):
        """Get the pipeline value."""
        return self.meta_dict.get("pipeline")

    @pipeline.setter
    def pipeline(self, value):
        """Set the pipeline value."""
        self.meta_dict["pipeline"] = value

    @property
    def training_method(self):
        """Get the training_method value."""
        return self.meta_dict.get("training_method")

    @training_method.setter
    def training_method(self, value):
        """Set the training_method value."""
        self.meta_dict["training_method"] = value

    @property
    def label_mode(self):
        """Get the label_mode value."""
        return self.meta_dict.get("label_mode")
    
    @label_mode.setter
    def label_mode(self, value):
        """Set the label_mode value."""
        self.meta_dict["label_mode"] = value

    @property
    def best_metric_train(self):
        """Get the best_metric_train value."""
        return self.meta_dict.get("best_metric_train")

    @best_metric_train.setter
    def best_metric_train(self, value):
        """Set the best_metric_train value."""
        self.meta_dict["best_metric_train"] = value

    @property
    def best_metric_val(self):
        """Get the best_metric_val value."""
        return self.meta_dict.get("best_metric_val")

    @best_metric_val.setter
    def best_metric_val(self, value):
        """Set the best_metric_val value."""
        self.meta_dict["best_metric_val"] = value

    @property
    def start_timestamp(self):
        """Get the start_timestamp value."""
        return self.meta_dict.get("start_timestamp")

    @start_timestamp.setter
    def start_timestamp(self, value):
        """Set the start_timestamp value."""
        self.meta_dict["start_timestamp"] = value

    @property
    def end_timestamp(self):
        """Get the end_timestamp value."""
        return self.meta_dict.get("end_timestamp")

    @end_timestamp.setter
    def end_timestamp(self, value):
        """Set the end_timestamp value."""
        self.meta_dict["end_timestamp"] = value

    @property
    def model_config(self):
        """Get the model_config value."""
        return self.meta_dict.get("model_config", {})

    @model_config.setter
    def model_config(self, value):
        """Set the model_config value."""
        self.meta_dict["model_config"] = value

    @property
    def dataset(self):
        """Get the dataset value."""
        return self.meta_dict.get("dataset")

    @dataset.setter
    def dataset(self, value):
        """Set the dataset value."""
        self.meta_dict["dataset"] = value

    @property
    def fine_tune(self):
        """Get the fine_tune value."""
        return bool(self.meta_dict.get("fine_tune", False))

    @fine_tune.setter
    def fine_tune(self, value):
        """Set the fine_tune value."""
        self.meta_dict["fine_tune"] = bool(value)

    @property
    def time(self):
        """Get the time value."""
        return self.meta_dict.get("time") or self.start_timestamp

    @time.setter
    def time(self, value):
        """Set the time value."""
        self.meta_dict["time"] = value
        if "start_timestamp" not in self.meta_dict:
            self.meta_dict["start_timestamp"] = value

    @property
    def best_metrics_train(self):
        """Get the best_metrics_train value."""
        return self.meta_dict.get("best_metrics_train") or self.best_metric_train

    @best_metrics_train.setter
    def best_metrics_train(self, value):
        """Set the best_metrics_train value."""
        self.meta_dict["best_metrics_train"] = value
        if "best_metric_train" not in self.meta_dict:
            self.meta_dict["best_metric_train"] = value

    @property
    def best_metrics_val(self):
        """Get the best_metrics_val value."""
        return self.meta_dict.get("best_metrics_val") or self.best_metric_val

    @best_metrics_val.setter
    def best_metrics_val(self, value):
        """Set the best_metrics_val value."""
        self.meta_dict["best_metrics_val"] = value
        if "best_metric_val" not in self.meta_dict:
            self.meta_dict["best_metric_val"] = value

    def to_yaml(self, file_path: Path):
        """Write this metadata to a YAML file, converting Path values to strings."""

        def convert_paths(obj):
            """Recursively convert Path values in obj to strings."""
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            else:
                return obj
        meta_dict_str = convert_paths(self.meta_dict)
        with open(file_path, 'w') as f:
            yaml.safe_dump(meta_dict_str, f)

    @staticmethod
    def from_yaml(file_path: Path):
        """Load metadata from a YAML file."""
        with open(file_path, 'r') as f:
            meta_dict = yaml.safe_load(f)
        return MetaData(meta_dict or {})

class MetaProcessor:
    def __init__(self, meta_path: Path):
        """Set up the training-metas directory under the given meta path."""
        self.meta_path = meta_path
        self.training_meta_dir = self.meta_path / "training_metas"
        self.training_meta_dir.mkdir(parents=True, exist_ok=True)

    def load_meta(self, file_name: str):
        """Load a MetaData instance from a training-meta YAML file."""
        return MetaData.from_yaml(self.training_meta_dir / file_name)

    def save_meta(self, meta: MetaData):
        """Save a MetaData instance to a training-meta YAML file."""
        try:
            model_name = meta.get_consistent_model_name(with_timestamp=True, timestamp=meta.end_timestamp)
            file_path = self.training_meta_dir / f"{model_name}.yaml"
            meta.to_yaml(file_path)
        except Exception as e:
            # Previously re-raised as a fabricated MemoryError("...memory constraints...")
            # regardless of the actual cause -- misleading (the real cause seen in practice was
            # `OSError: [Errno 36] File name too long`, from a descriptive config_name pushing
            # get_meta_name's filename past the ext4 255-byte cap; see that function's own
            # truncation fix). Log the real exception and re-raise it as-is so whoever's debugging
            # this next isn't sent chasing a memory problem that was never real.
            logging.error(f"Error saving meta: {e}")
            raise

    def get_metas(self):
        """List the training-meta filenames on disk."""
        return os.listdir(self.training_meta_dir)






