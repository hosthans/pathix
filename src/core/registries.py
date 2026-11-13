from typing import Dict, Tuple, Type

MODEL_REGISTRY: Dict[str, Type] = {}
DATASET_REGISTRY: Dict[Tuple[str, str, bool], Type] = {}
TRAINING_REGISTRY: Dict[Tuple[str, str, str], Type] = {}
EVALUATION_REGISTRY: Dict[Tuple[str, str, str], Type] = {}
VALIDATION_THRESHOLD_REGISTRY: Dict[Tuple[str, str, bool], Type] = {}