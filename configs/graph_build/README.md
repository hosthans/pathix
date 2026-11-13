# Graph Build Configs

Pipeline-specific graph settings live here.

Current setup:

- `simple_pipeline.yaml` -- base profile. Defaults to `annotation_type: geojson` (GSDB's
  format); CAMELYON-specific pipelines override this to `asap`.
- `context_aware_pipeline.yaml` -- extends `simple_pipeline.yaml`, CAMELYON-oriented
  (`annotation_type: asap`, precomputed masks). Recommended default pipeline for CAMELYON.
- `context_aware_pipeline_gsdb.yaml` -- extends `context_aware_pipeline.yaml`, same structural
  settings but switched to GSDB's actual data format (`annotation_type: geojson`,
  `use_precomputed_mask: false` -- GSDB has no precomputed masks directory). Recommended
  default pipeline for GSDB.
- `context_aware_robust_pipeline.yaml`, `context_aware_sensitivity_pipeline.yaml`,
  `tumor_focus_pipeline.yaml` -- CAMELYON-only variants.

These files should only contain graph construction settings (`build` and `creation`).
Runtime/environment settings (dataset paths, CSV selection, worker settings) are defined in
`configs/graph_runtime` -- see that directory's README for which runtime config to pair with
which pipeline, and the GSDB per-split build workflow.
