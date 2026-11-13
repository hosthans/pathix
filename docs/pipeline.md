# Pipeline: setup → splits → graphs → train → evaluate

Only what is needed to understand and reproduce the experiments E1–E4
([experiments/cross_domain_experiments.md](experiments/cross_domain_experiments.md)). Run everything from the
repository root in the `gsdb` conda env.

## 1. Setup and data

```bash
conda env create -f environment.yml && conda activate gsdb
bash scripts/setup_data_dirs.sh            # creates the data/ skeleton (data/ is gitignored)
export HF_TOKEN=...                         # Prov-GigaPath (gated Hugging Face repo)
```

| Data | Location |
|---|---|
| GSDB slides / annotations | `data/raw/slides/` (`.mrxs`/`.tif`), `data/raw/annotations/<slide>_annotations.json` (flat) |
| GSDB tiles (patch-CNN, MIL) | `data/processed/tiles/` — 256 px, downsample 10, overlap 64, from the GSDB tile extractor |
| GSDB tile labels | `results/metrics/{inflammation_status,tiles_per_particle,tissue_types}.csv` (from the GSDB repo) |
| CAMELYON17 slides | `data/raw/camelyon/images/`, ASAP XML in `.../annotations/` |
| CAMELYON17 tables | `stages.csv`, `processed_slides.csv` (paths in `src/utils/split_utils.py`; `processed_slides.csv` via `src/entrypoint/create_processed_slides.py`) |

Outside the repo root set `GASTRIC_BASE_DIR=/abs/path/to/master_code`. GPU: least busy one unless
`CUDA_VISIBLE_DEVICES` is set.

## 2. Splits (`seed_42`)

```bash
python src/entrypoint/create_splits.py --seed 42 --kfold-k 5                         # GSDB
python src/entrypoint/create_camelyon_splits.py --mode natural --seed 42 \
  --train-centers 0,1,2 --val-center 3 --test-center 4                              # CAMELYON17
```

- **GSDB** (`data/splits/gsdb/seed_42/`): all scanner-2 slides → `test_scanner2` (S2, 90 slides / 30 patients:
  30 H&E + 30 MG + 30 PAS). Scanner-1 patients → train / val / test (S1) = 60 / 20 / 20 %
  (`train_test_split`, `random_state = seed + 2`, patient-disjoint, not label-stratified). Per split also
  `_HE.csv` and `_all_stains.csv`. `--kfold-k 5` adds patient-grouped CV folds (`kfold_k5_fold<i>/`, not used in
  E1–E4; train one with `train_model.py --split-phase kfold_k5_fold<i>`).
- **CAMELYON17** (`data/splits/camelyon/seed_42/`): whole centers 0,1,2 → train, 3 → val, 4 → test. Train/val keep
  all tumor slides (`--mode natural`) and are subsampled to 1:1 tumor:normal (normals drawn with seed + 2 / + 3);
  test (100 slides, 39 tumor) is untouched. No in-domain test set.
- **CAMELYON17 center rotation** (E4): `--phase test_c<t>` writes the split to `seed_42/test_c<t>/` instead of
  `seed_42/` (read with `--split-phase test_c<t>`). Test center t, val center t − 1 (mod 5), train = the other
  three; five folds, every center is the test center once. `test_c4` = the `seed_42` split. Without
  `--split-phase` training/evaluation keep reading `seed_42/*.csv` (the fold folders are never picked up).
  Commands and fold table: [experiments § Splits](experiments/cross_domain_experiments.md#splits).
- Both commands reproduce the current files byte-identically (verified 2026-10-19). Logs:
  `results/logs/split_creation/`. Test run without touching `data/splits/`: add `--data-dir <d> --output-dir <d>`
  (GSDB also needs `<d>/raw` → `data/raw`) and `diff -r` against `data/splits/<dataset>/seed_42`.
- The hard-coded GSDB patient exclusion list matches no `patient_id` ("Removed slides: 0"), so nobody is excluded.

## 3. Graph corpora

A WSI becomes one cached `HeteroData` graph (`.pt`): tissue mask → 256 px patches → frozen encoder embedding per
patch → regions (groups of patches) → slide node, with kNN / semantic edges. Implementation:
`src/graph_creation_new/`. One flattened config per corpus (no inheritance) in
`configs/experiments/graph_corpora/`; output folder = `data/processed/graphs/<dataset>/<config name>`
(= `--graph_target_dir`).

```bash
PYTHONHASHSEED=42 python -u src/entrypoint/build_graph.py --config configs/experiments/graph_corpora/<config>.yaml
```

| Corpus | Config | Encoder (dim) | Image augmentation | Used by |
|---|---|---|---|---|
| G-aug | `context_aware_gsdb_inflammation` | Prov-GigaPath (1536) | ✓ | E2, E3 |
| G-noaug | `context_aware_gsdb_inflammation_noaug` | Prov-GigaPath | ✗ | E1 graph, E3 |
| G-R50 | `context_aware_runtime_non_hist_gsdb_inflammation` | ResNet-50 ImageNet (2048) | ✓ | E2 |
| C-aug | `context_aware_camelyon_domain_robust_labels` | Prov-GigaPath | ✓ | E2, E4 |
| C-noaug | `context_aware_camelyon_labels_noaug` | Prov-GigaPath | ✗ | E4 |
| C-R50 | `context_aware_runtime_non_hist_camelyon_domain_robust_labels` | ResNet-50 | ✓ | E2 |

- **All corpora:** Macenko colour normalisation (references in `configs/graph_build/reference_stats/`, recompute
  with `scripts/graph_creation/compute_color_reference_stats.py`), patch 256 px / stride 256 / level 0, tissue
  fraction ≥ 0.08, region–region kNN (k = 5), `transform_seed: 42`.
- **Image augmentation** (aug corpora): stain augmentation (σ 0.2, p 0.5), Gaussian blur (σ ≤ 1.2, p 0.2), JPEG
  (quality 40–95, p 0.2), applied once per patch at build time. It is therefore baked into the cached features of
  **every** slide, including validation and test slides.
- **GSDB vs. CAMELYON17:** GSDB uses a physically calibrated patch size (same tissue area on both scanners; the
  first slide of `data/splits/gsdb/all_slides.csv` with a readable mpp defines it), geometric regions (connected
  tissue, bisected to ≤ 100 patches), `semantic_equality` patch edges, per-slide tissue masks, GeoJSON labels
  (`inflammation_status`), slides without annotation are not built. CAMELYON17 uses fixed 256 px, annotation-driven
  regions, patch kNN (k = 12), precomputed masks, ASAP labels (`status`, `subtype`, `stage`).
- **Bit-exact builds:** a rare (≈ 1 per 333-slide build), non-reproducible perturbation of single patch embeddings
  inside the GPU forward pass (non-ECC GPUs) made two builds of the same config differ. `feature_redundancy_check:
  true` (all configs) computes every feature batch until two passes agree bit for bit (≈ 2× build time). Keep
  `feature_batch_size: 1500`. Corpora built before the guard: rebuild, or move the affected `.pt` away and rerun
  (`overwrite: false` rebuilds only missing slides). Compare two corpora with
  `scripts/analysis/compare_graph_dirs.py <A> <B> --workers 8 --ignore-key blas_num_threads --ignore-key feature_redundancy_check`.
- `strict_graph_name_match: false` in the model configs: check `missing` = 0 in the dataset log. The GSDB graph of
  `282_248_2_MG` is missing in all corpora (graph models: S2 n = 89).

## 4. Models

| Family | Input | Model | Slide score |
|---|---|---|---|
| Patch-CNN | tiles | ResNet-18 (ImageNet, fine-tuned) or Prov-GigaPath (frozen) + classification head, trained on tiles with the slide label | mean tile probability |
| MIL | bag of 60 tiles per slide | attention MIL on ResNet-18 features | bag probability |
| Graph (`tissue_region_net`) | cached graph | region branch (GATv2 over region edges) + patch branch (SAGE over patch edges, gated fusion); heads for region, patch and slide (attention pooling + max region logit) | slide probability |

The Tissue-Region-Net (TRN) is trained with `--segmentation`: weighted sum of region, slide, auxiliary (region/patch BCE on
annotated nodes), region–slide consistency and patch losses. Unlabeled nodes: on negative slides → negative,
outside tumor polygons of annotated slides → negative, on positive slides without annotation → ignored (optionally
pseudo-labelled after a warm-up). Training-time graph augmentations (`train_augmentations`: feature Gaussian noise,
feature dropout, graph Mixup, feature scale noise, graph MixStyle) and the augmentation-consistency loss are set
in the model config (E3). Code: `src/tissue_region_extraction/model.py`, `src/pipelines/train.py`.

## 5. Training and evaluation

```bash
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py ... --seed 42 --deterministic --prefix <run> --config_name <config>
PYTHONHASHSEED=42 python src/entrypoint/evaluate_model.py --splits test test_scanner2 --meta_file results/metas/training_metas/<run>_<...>.yaml
```

- **Outputs:** checkpoint `results/models/best_model_<run>.pt`, history `results/models/training_history_<run>.json`
  (both without timestamp: a retrain overwrites them), meta `results/metas/training_metas/<run>_..._<timestamp>.yaml`,
  post-training evaluation on `test` in `results/evaluations/<run>_..._<timestamp>/`, `evaluate_model.py` output in
  `results/test_results/<run>_..._<timestamp>/split_<split>/` (`*_slide_metrics.json`, `*_slide_predictions.csv`).
- **Checkpoint and threshold** come from the validation split only (GSDB `val_HE`, CAMELYON17 center 3): patch-CNN /
  MIL = min. validation loss, graph = `monitor_metric` (`slide_f2`). Slide threshold = G-mean on the validation
  slide scores; no recalibration on any test set.
- **Determinism:** `--deterministic` seeds everything and uses deterministic kernels where available (ops without
  one only warn); fp16 runs are not bit-identical across retrains. Use the same GPU for compared runs.
- **Long runs** (protected from `systemd-oomd`, which kills tmux panes under memory pressure): once
  `loginctl enable-linger $(whoami)`, then
  `systemd-run --user --unit=<name> --working-directory="$(pwd)" -p ManagedOOMPreference=omit -- python -u ...`;
  follow with `journalctl --user -u <name> -f`, stop with `systemctl --user stop <name>`.

## 6. Analysis

`notebooks/analysis/`: `Training.ipynb` (training curves), `Testing.ipynb` (E1: ID vs. OOD), `Encoders.ipynb` (E2),
`Augmentation.ipynb` (E3). Each takes explicit evaluation paths in its configuration cell; ID and OOD of one run must
come from the same checkpoint (the notebooks warn if the validation thresholds differ).
