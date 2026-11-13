# Artifacts

[Overview](README.md) · [A1 Framework](framework.md) · [A2 TRN](trn.md) · [A3 DG framework](dg_framework.md) · [Supporting artifacts](supporting_artifacts.md) · [References](references.md)

What each artifact of the thesis is, where it lives in the code and how the parts connect. How to run the code:
[pipeline.md](../pipeline.md); experiments E1–E4: [experiments/](../experiments/cross_domain_experiments.md).

The three contributions of the thesis are the primary artifacts:

| # | Artifact | Thesis section | Main code location |
|---|---|---|---|
| [A1](framework.md) | Modular framework for graph construction and unified evaluation | Overall Architecture; Pipeline for Graph Creation; Implementation Notes (appendix) | `src/graph_creation_new/`, `src/core/`, `src/pipelines/`, `src/entrypoint/` |
| [A2](trn.md) | Tissue Region Network (TRN) | Structure-Aware Graph Neural Architecture | `src/tissue_region_extraction/model.py` |
| [A3](dg_framework.md) | Domain generalization (DG) framework | Domain Generalization Framework | `src/graph_creation_new/helper/transforms.py`, `src/graph_creation_new/helper/slide.py`, `src/dataset/graph_augment.py`, `src/tissue_region_extraction/model.py`, `src/pipelines/train.py` |

Supporting artifacts that the primary ones produce or consume:

| # | Artifact | Location |
|---|---|---|
| [S1](supporting_artifacts.md) | Heterogeneous slide graph (`HeteroData`, one `.pt` per slide) and graph corpora | `data/processed/graphs/<dataset>/<config name>/` (gitignored), configs in `configs/experiments/graph_corpora/` |
| [S2](supporting_artifacts.md) | Data splits | `src/entrypoint/create_splits.py`, `src/entrypoint/create_camelyon_splits.py`, output in `data/splits/` |
| [S3](supporting_artifacts.md) | Configurations (graph build / runtime / model) | `configs/` |
| [S4](supporting_artifacts.md) | Training and evaluation outputs (checkpoints, metas, predictions, metrics) | `results/` (gitignored) |
| [S5](supporting_artifacts.md) | Baseline models (patch-CNN, attention MIL) | `src/models/base.py`, `src/models/gigapath.py`, `src/models/multimodal.py` |
| [S6](supporting_artifacts.md) | Analysis scripts and notebooks | `scripts/analysis/`, `notebooks/analysis/` |

Sources: [references.md](references.md).
