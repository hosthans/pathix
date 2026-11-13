# scripts

Helper scripts around the entrypoints in `src/entrypoint/`. Run everything from the repository root.
Training, evaluation and inference are called directly through the entrypoints (`train_model.py`,
`evaluate_model.py`, `infer_tissue_regions.py`, `annotate_tissue_regions.py`); the thesis experiments
(E1–E4) are listed in [docs/experiments/cross_domain_experiments.md](../docs/experiments/cross_domain_experiments.md).
Split creation: `src/entrypoint/create_splits.py` (GSDB) and `create_camelyon_splits.py` (CAMELYON17).

```text
scripts/
├── setup_data_dirs.sh                 # create the expected data/ folder layout (docs/pipeline.md)
├── graph_creation/
│   ├── compute_color_reference_stats.py   # Macenko reference stats (configs/graph_build/reference_stats/*.json)
│   └── patch_label_fields.py              # update label fields of already-built cached graphs
└── analysis/
    ├── compare_graph_dirs.py              # full content diff of two graph corpora (determinism checks)
    ├── graph_content_fingerprint.py       # one content hash per graph (compare machines/checkouts)
    ├── inspect_raw_structure.py           # compare data/raw trees between machines
    ├── domain_classifier_auc.py           # scanner-1 vs scanner-2 domain-classifier AUC on graph embeddings
    └── region_granularity_domain_gap.py   # domain gap of region-size statistics
```
