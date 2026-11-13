import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.core.registries import MODEL_REGISTRY, DATASET_REGISTRY, TRAINING_REGISTRY, EVALUATION_REGISTRY, VALIDATION_THRESHOLD_REGISTRY


def register_model(key: tuple):
    """Decorator that registers a model class under the given key."""
    def wrapper(cls):
        """Register cls under key and return it unchanged."""
        if key in MODEL_REGISTRY:
            raise KeyError(f"Model '{key}' already registered")

        MODEL_REGISTRY[key] = cls
        return cls
    return wrapper

def register_training_pipeline(key: tuple):
    """Decorator that registers a training pipeline class under the given key."""
    def wrapper(cls):
        """Register cls under key and return it unchanged."""
        if key in TRAINING_REGISTRY:
            raise KeyError(f"Training pipeline '{key}' already registered")

        TRAINING_REGISTRY[key] = cls
        return cls
    return wrapper

def register_dataset(key: tuple):
    """Decorator that registers a dataset class under the given key."""
    def wrapper(cls):
        """Register cls under key and return it unchanged."""
        if key in DATASET_REGISTRY:
            raise KeyError(f"Dataset '{key}' already registered")

        DATASET_REGISTRY[key] = cls
        return cls
    return wrapper

def register_evaluation_pipeline(key: tuple):
    """Decorator that registers an evaluation pipeline class under the given key."""
    def wrapper(cls):
        """Register cls under key and return it unchanged."""
        if key in EVALUATION_REGISTRY:
            raise KeyError(f"Evaluation pipeline '{key}' already registered")

        EVALUATION_REGISTRY[key] = cls
        return cls
    return wrapper

def register_validation_threshold(key: tuple):
    """Decorator that registers a validation threshold class under the given key."""
    def wrapper(cls):
        """Register cls under key and return it unchanged."""
        if key in VALIDATION_THRESHOLD_REGISTRY:
            raise KeyError(f"Validation threshold '{key}' already registered")

        VALIDATION_THRESHOLD_REGISTRY[key] = cls
        return cls
    return wrapper