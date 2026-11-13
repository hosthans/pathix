"""Utilities for region-level tumor localization on graph-cached WSIs.

This module provides:
- A robust region-graph model.
- Constraint definitions for existing training routines.
- Inference and component extraction for unannotated positive slides.
"""

from .model import TissueRegionGraphModel
from .routines import (
    DEFAULT_CONSTRAINTS,
    TissueRegionTrainingConstraints,
    build_train_model_args,
    build_training_overrides,
    calibrate_threshold_f1,
    export_components_as_geojson,
    extract_tissue_region_components,
    infer_region_probabilities,
)

__all__ = [
    "TissueRegionGraphModel",
    "TissueRegionTrainingConstraints",
    "DEFAULT_CONSTRAINTS",
    "build_training_overrides",
    "build_train_model_args",
    "infer_region_probabilities",
    "extract_tissue_region_components",
    "export_components_as_geojson",
    "calibrate_threshold_f1",
]
