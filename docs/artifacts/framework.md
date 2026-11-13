# A1 — Modular framework for graph construction and unified evaluation

[Overview](README.md) · [A1 Framework](framework.md) · [A2 TRN](trn.md) · [A3 DG framework](dg_framework.md) · [Supporting artifacts](supporting_artifacts.md) · [References](references.md)

## Purpose

The framework turns whole-slide images (WSIs) into cached heterogeneous graphs. It then trains and
evaluates patch-based CNNs, MIL models and graph models on the same splits and with the same slide-level
protocol. Which components are combined is decided by configuration files, not by code changes.

## Stages

The codebase has five loosely coupled stages. They exchange data only through cached files and
configuration files.

```text
raw WSI ──► [1] graph construction ──► .pt graph (HeteroData) ──► [2] dataset ──► [3] model ──► [4] pipeline (train / evaluate)
              src/graph_creation_new/                              src/dataset/      src/models/      src/pipelines/
                                                                                     src/tissue_region_extraction/
[5] command-line entry points for every stage: src/entrypoint/
```

| Stage | Responsibility | Main files |
|---|---|---|
| 1 Graph construction | Tissue mask, patch extraction, pixel-level transforms, encoding, regions, edges, labels, `HeteroData` assembly, caching | `src/graph_creation_new/graph_creation.py`, `src/graph_creation_new/helper/*.py`, `src/graph_creation_new/config.py` |
| 2 Dataset | Loading cached graphs, region resplitting, training-time graph augmentation, hierarchical supervision | `src/dataset/graph_dataset.py` (`WSICachedGraphDataset`), `src/dataset/region_resplit.py`, `src/dataset/graph_augment.py` |
| 3 Model | Graph and baseline architectures | `src/tissue_region_extraction/model.py`, `src/models/` |
| 4 Pipeline | Training loops, losses, validation thresholds, evaluation and metrics | `src/pipelines/train.py`, `src/pipelines/validation.py`, `src/pipelines/evaluate.py`, `src/pipelines/gsdb_parity_*.py` |
| 5 Entry points | One CLI per step | `src/entrypoint/build_graph.py`, `train_model.py`, `evaluate_model.py`, `infer_tissue_regions.py`, `annotate_tissue_regions.py`, `create_splits.py`, `create_camelyon_splits.py` |

## Graph construction pipeline

A pipeline is a class with an ordered list of steps that all operate on a shared `BuildContext`
(`src/graph_creation_new/config.py`). Pipelines are registered by name (`register_pipeline`,
`available_pipelines` in `graph_creation.py`). The configuration selects one with `pipeline: <name>`.

The WSI files are read through OpenSlide. Steps of `DynamicalPipeline` (`name = "dynamical"`, the default
pipeline used for all thesis corpora), grouped into the five phases of the thesis:

| Phase | Step | Implementation | Thesis subsection |
|---|---|---|---|
| Setup | Thumbnail | `helper/thumbnail.py` `safe_thumbnail` | Thumbnail Generation |
| Setup | Annotations (ASAP XML or GeoJSON) parsed into labeled polygons | `helper/annotations.py` `load_asap_annotation`, `load_geojson_annotations` | Annotations |
| Setup | Tissue mask: precomputed, or generated from the thumbnail; refined in both cases | `helper/mask.py` `load_mask_from_path`, `generate_mask_from_thumbnail`, `refine_mask_with_config` | Tissue Mask Generation |
| Geometry | Patch geometry (patch size and stride, optionally in physical units) | `helper/slide.py` `prepare_slide`, `resolve_patch_size_px`, `resolve_stride_px` | Patch Extraction |
| Geometry | Patch enumeration (sliding window, tissue-fraction filter, optional annotation filter) | `helper/patches.py` `enumerate_patches` | Patch Extraction |
| Check | Annotation quality check (coordinate-system mismatch; may exclude a slide) | `helper/annotations.py` `assess_annotation_quality` | Annotations |
| Structure | Region construction: annotation-driven (`annotation_driven`, default) or geometry-driven (`geometric`) | `helper/regions.py` `build_regions` | Region Construction |
| Assembly | Pixel transforms → frozen encoder → region features (mean of patches) and slide feature (mean of regions) → edges → node labels → `HeteroData` | `helper/subgraph.py` `create_subgraph`; `helper/transforms.py`, `helper/features.py`, `helper/edges.py`, `helper/labels.py` | Feature Encoding; Edge Construction; Node Labeling; Hierarchical Graph Structure |

`TumorFocusPipeline` (`name = "tumor_focus"`) runs the same steps with different configuration values.

Feature inference in `helper/features.py` can use `_RedundantInference` (`feature_redundancy_check: true`).
It computes every batch again until two passes agree bit for bit.

## Registry-based composition

`src/core/registries.py` defines five registries:

| Registry | Key (examples) | Registered with |
|---|---|---|
| `MODEL_REGISTRY` | `(model_type, architecture, input_type)`, e.g. `("gnn", "tissue_region_net", "graph")`, `("mil", "any", "image")`, `("cnn", "resnet18", "image")` | `@register_model` (`src/core/registry_utils.py`) |
| `DATASET_REGISTRY` | `(dataset, input_type, embedding_type, cached)`, e.g. `("gsdb", "graph_cached", "pov-gigapath", True)` | `@register_dataset` |
| `TRAINING_REGISTRY` | `(task, model_type, input_type)` | training pipelines |
| `EVALUATION_REGISTRY` | `(task, model_type, input_type)`, e.g. `("inflammation", "cnn", "image")` | `@register_evaluation_pipeline` |
| `VALIDATION_THRESHOLD_REGISTRY` | `(input_type, embedding_type, mil)` | validation-threshold classes |

`src/core/holder.py` (`Holder`) loads the configuration and builds the model and datasets through
`src/core/factories.py`. It then selects the training routine:

1. `train_graph_segmentation_model` (`src/pipelines/train.py`) if `--segmentation` or
   `use_segmentation_training: true` is set. This routine is used for TRN.
2. `train_model_gsdb_parity` (`src/pipelines/gsdb_parity_train.py`) if `gsdb_parity_train` is set. This
   routine reproduces the patch-CNN of the GSDB reference implementation.
3. Otherwise the routine registered in `TRAINING_REGISTRY`.

The CAMELYON17 task `lymph` is resolved to the `inflammation` pipeline for training and evaluation.

## Unified evaluation protocol

- The checkpoint and the slide threshold are selected on the validation split only. The threshold is the
  G-mean point on the validation slide scores (`src/pipelines/validation.py`,
  `src/utils/metrics_utils.py` `find_optimal_threshold`).
- Each model family produces one score per slide: the mean tile probability for the patch-CNN, the bag
  probability for MIL, and the slide probability for TRN.
- `src/pipelines/evaluate.py` also computes region- and patch-level metrics and writes slide overlays for
  graph models.

## Configuration layers

| Layer | Folder | Content | Inheritance |
|---|---|---|---|
| Build | `configs/graph_build/` | Patch, mask, edge and region parameters (`build`, `creation`) | `extends:` (deep merge, cycle detection) |
| Runtime | `configs/graph_runtime/` | Dataset, encoder, paths, workers; points to a build config via `graph_build.build_config` | `extends:` |
| Flattened corpora | `configs/experiments/graph_corpora/` | Runtime + build written out in one file per corpus | none |
| Model | `configs/model_config*.yaml`, `configs/experiments/E*/` | Model, training, losses, augmentation, consistency per task | none |

## Builds on

- WSI access ([Goode et al., 2013](references.md#ref-goode-openslide-2013))
- patch-based slide pipeline (tissue segmentation, tiling, frozen encoder) ([Lu et al., 2020](references.md#ref-lu-data-2020))
- tissue mask: saturation and morphology ([Lu et al., 2020](references.md#ref-lu-data-2020))
- tissue mask: gray-world color normalization ([Buchsbaum, 1980](references.md#ref-buchsbaum-spatial-1980))
- tissue mask: Otsu thresholding ([Otsu, 1979](references.md#ref-otsu-threshold-1979))
- patch → region → slide hierarchy (cell-to-tissue graphs) ([Pati et al., 2022](references.md#ref-pati-hierarchical-2022))
- frozen encoders UNI2 and Prov-GigaPath ([Chen et al., 2024](references.md#ref-chen-towards-2024); [Xu et al., 2024](references.md#ref-xu-whole-slide-2024))
- graph models ([Fey and Lenssen, 2019](references.md#ref-fey-fast-2019))
- GSDB reference implementation (patch-CNN, splits) ([Liebel, 2025](references.md#ref-liebel-developing-2025))
