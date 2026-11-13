# Graph Runtime Configs

Runtime/environment settings live here.

Current setup:

- `simple_runtime.yaml` / `simple_runtime_gsdb.yaml` -- base runtime (dataset identity, paths,
  worker settings), one per dataset.
- `context_aware_runtime.yaml` / `context_aware_runtime_gsdb.yaml` -- extends the matching
  `simple_runtime*.yaml`, switches to the context-aware pipeline profile. This is the
  recommended default for both datasets.
- `context_aware_robust_runtime.yaml`, `context_aware_sensitivity_runtime.yaml`,
  `tumor_focus_runtime.yaml` -- CAMELYON-only variants, extend `context_aware_runtime.yaml` /
  `simple_runtime.yaml`.

Each file should contain only runtime settings (dataset/model/csv/worker/paths)
and reference one pipeline config via:

`graph_build.build_config`

Run (CAMELYON):

```bash
python src/entrypoint/build_graph.py --config configs/graph_runtime/context_aware_runtime.yaml
```

Run (GSDB):

```bash
python src/entrypoint/build_graph.py --config configs/graph_runtime/context_aware_runtime_gsdb.yaml
```

## GSDB: no single "build everything" CSV

Unlike CAMELYON, which builds from one master CSV (`stages.csv`) covering every slide, GSDB has
no combined list -- `data/splits/gsdb/seed_<seed>/` only holds pre-split slide lists per stain
(`train_HE.csv` / `train_all_stains.csv`, `val_HE.csv` / `val_all_stains.csv`,
`test_HE.csv` / `test_all_stains.csv`). `simple_runtime_gsdb.yaml`'s `graph_build.csv` picks
exactly one of these. To build the full GSDB dataset, run `build_graph.py` **three times**,
changing only `csv` between runs (e.g. `train_all_stains.csv` -> `val_all_stains.csv` ->
`test_all_stains.csv`); already-built `.pt` files are skipped unless `overwrite: true`, so
re-running with a different split is safe and cheap to repeat.

Note that the split CSVs only include slides with non-empty annotation JSON (see
`create_splits.py`'s `create_slide_info_df`, which skips `[]`-annotated slides) -- 22 of the 360
slides under `data/raw/slides` never made it into any split (the former all-slides config,
`context_aware_runtime_gsdb_all_slides.yaml`, was removed).
