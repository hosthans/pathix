# A3 — Domain generalization framework

[Overview](README.md) · [A1 Framework](framework.md) · [A2 TRN](trn.md) · [A3 DG framework](dg_framework.md) · [Supporting artifacts](supporting_artifacts.md) · [References](references.md)

## Purpose

A set of independently configurable mechanisms that reduce the model's reliance on acquisition-specific
characteristics when no target-domain data are available during training. The mechanisms act at four
levels (pixel, feature, distribution, prediction) and are spread over graph construction, the dataset,
the model and the training loop.

## Components and locations

| Level | Mechanism | Applied when | Implementation | Configuration |
|---|---|---|---|---|
| Pixel | Macenko stain normalization | Graph build, once per patch | `helper/transforms.py` `color_normalize`; references from `scripts/graph_creation/compute_color_reference_stats.py` | `feature_transforms` entry `color_normalize` (`method: macenko`), `macenko_rescale_concentration`, reference file in `configs/graph_build/reference_stats/` |
| Pixel | Physical-aware scaling (patch size and stride in µm) | Graph build | `helper/slide.py` `resolve_patch_size_px`, `resolve_stride_px` | `target_patch_size_um`, `auto_detect_target_patch_size_um` |
| Pixel | Stain augmentation, Gaussian blur, JPEG compression | Graph build, once per patch, seeded per patch | `helper/transforms.py` `stain_augment`, `gaussian_blur`, `jpeg_compression`, `apply_patch_transforms` | `feature_transforms` entries, `transform_seed` |
| Feature | Graph instance normalization | Every forward pass | `model.py` `_graph_instance_normalize` | `graph_instance_norm` |
| Feature | Feature Gaussian noise, feature dropout, feature scale noise | Training, each time a graph is loaded | `src/dataset/graph_augment.py` | `train_augmentations` list |
| Feature | Graph Mixup | Training | `graph_augment.py` `graph_mixup` | `train_augmentations` entry `graph_mixup` (`alpha`, `p`) |
| – | Node dropping (patch nodes, annotated targets kept) | Consistency copy | `graph_augment.py` `node_drop` | `augmentation_consistency_specs` |
| Distribution | Graph MixStyle | Training | `graph_augment.py` `graph_mixstyle` | `train_augmentations` entry `graph_mixstyle` (`alpha`, `p`) |
| Prediction | Consistency regularization (perturbed copy with feature Gaussian noise, feature dropout and node dropping; MSE of slide probabilities + 1 − cosine of pooled slide embeddings) | Training, with probability `augmentation_consistency_p` | `train.py` `_resolve_augmentation_consistency_perturbation`, `_augmentation_consistency_loss_from_perturbed`, `_resolve_consistency_settings` | `augmentation_consistency_weight`, `augmentation_consistency_p`, `augmentation_consistency_specs`, `feature_consistency_weight` |

All training-time perturbations are registered by name in `_AUGMENTATION_REGISTRY`
(`src/dataset/graph_augment.py`). `apply_graph_augmentations` applies them in configured order. The
same functions serve two roles:

- **Augmentation:** `train_augmentations`, applied by `WSICachedGraphDataset` to the training graph.
- **Consistency perturbation:** `augmentation_consistency_specs`, applied only to a copy of the graph.

Pixel-level mechanisms are baked into the cached features of a corpus, so they define the corpus (S1).
Feature-, distribution- and prediction-level mechanisms are part of the model configuration.

## Builds on

- Macenko stain normalization ([Macenko et al., 2009](references.md#ref-macenko-method-2009))
- stain augmentation ([Tellez et al., 2018](references.md#ref-tellez-whole-slide-2018))
- graph instance normalization ([Cai et al., 2021](references.md#ref-cai-graphnorm-2021))
- feature Gaussian noise ([DeVries and Taylor, 2017](references.md#ref-devries-dataset-2017))
- feature dropout, feature scale noise ([Srivastava et al., 2014](references.md#ref-srivastava-dropout-2014))
- node dropping ([You et al., 2020](references.md#ref-you-graph-2020))
- Graph Mixup ([Zhang et al., 2018](references.md#ref-zhang-mixup-2018))
- Graph MixStyle ([Zhou et al., 2021](references.md#ref-zhou-domain-2021))
- consistency regularization ([Sajjadi et al., 2016](references.md#ref-sajjadi-regularization-2016); [Laine and Aila, 2017](references.md#ref-laine-temporal-2017))
