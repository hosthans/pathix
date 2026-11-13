# Supporting artifacts (S1–S6)

[Overview](README.md) · [A1 Framework](framework.md) · [A2 TRN](trn.md) · [A3 DG framework](dg_framework.md) · [Supporting artifacts](supporting_artifacts.md) · [References](references.md)

## S1 — Slide graph and graph corpora

One file per slide: `data/processed/graphs/<dataset>/<config name>/<slide>.pt`, a PyTorch Geometric
`HeteroData` assembled in `helper/subgraph.py` (`_assemble_hetero_data`).

| Element | Content |
|---|---|
| Node types | `slide` (1 node), `region`, `patch` |
| Node attributes | `x` (features; encoder embeddings or spatial features depending on `node_creation_mode`), `feat`, `spatial_x`, `centroid`; `patch.topleft`; `annotation_y` (patch, region); `region.label_ambiguous`; `<field>_y` for configured `label_fields` |
| Edge types | `("patch","near","patch")`, `("region","near","region")`, `("patch","in","region")`, `("region","has","patch")`, `("region","part_of","slide")`, `("slide","contains","region")` |
| Edge attributes | `edge_index`, `edge_attr` (distance), `edge_feat` (8 columns, `helper/edges.py` `build_edge_features`: distance, inverse distance, Δx, Δy, same-semantic flag, semantic-valid flag, cosine similarity, feature-valid flag) |
| `data.meta` | Annotation vocabularies and coverage, label-field vocabularies, `feature_dim` and build metadata |

All nodes share one feature space: patch features come from the frozen encoder, a region feature is the mean of
its patches, the slide feature is the mean of its regions. Encoders: UNI2 (1536), Prov-GigaPath (1536),
ResNet-50 ImageNet (2048).

The six corpora used in the thesis (G-aug, G-noaug, G-R50, C-aug, C-noaug, C-R50) and the settings in
which they differ are listed in [configs/experiments/graph_corpora/README.md](../../configs/experiments/graph_corpora/README.md).
Utilities: `src/entrypoint/summarize_graphs.py`, `scripts/analysis/compare_graph_dirs.py`,
`scripts/analysis/graph_content_fingerprint.py`, `scripts/graph_creation/patch_label_fields.py`.

## S2 — Data splits

GSDB: `src/entrypoint/create_splits.py` (patient-disjoint, scanner 2 held out). CAMELYON17:
`src/entrypoint/create_camelyon_splits.py` (whole centers). Output in `data/splits/`; details in
[pipeline.md §2](../pipeline.md#2-splits-seed_42).

## S3 — Configurations

`configs/`; the layers are described in [framework.md](framework.md#configuration-layers). Macenko reference
statistics: `configs/graph_build/reference_stats/`.

## S4 — Run outputs

Checkpoints, run metas and predictions under `results/`; paths in [pipeline.md §5](../pipeline.md#5-training-and-evaluation).

## S5 — Baseline models

| Model | Class | Registry key | File |
|---|---|---|---|
| Patch-CNN, ResNet-18 | `HistologyClassifier` | `("cnn", "resnet18", "image")` | `src/models/base.py` |
| Patch-CNN, Prov-GigaPath (frozen) + MLP head | `GigaPathClassifier` | `("cnn", "gigapath", "image")` | `src/models/gigapath.py` |
| Attention MIL | `AttentionMIL_NoEmbeddings`, `AttentionMIL` | `("mil", "any", "image")`, `("mil", "any", "multimodal")` | `src/models/multimodal.py` |

Further graph architectures are registered in `src/models/graph.py` and
`src/models/robust_slide_classifier.py`. The thesis experiments do not use them.

## S6 — Analysis

`scripts/analysis/` (see [scripts/README.md](../../scripts/README.md)) and `notebooks/analysis/`.
