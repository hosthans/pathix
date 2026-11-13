# Cross-Scanner / Cross-Center Generalization Experiments (E1–E4)

How the experiments work: data and evaluation protocol, what differs between the compared runs, and known
limitations. Commands and result tables: [cross_domain_experiments.md](cross_domain_experiments.md); splits,
graph corpora, models and outputs: [../pipeline.md](../pipeline.md).

Model configs per experiment (files shared between experiments are symlinks to one real file):

```text
configs/experiments/
├── E1_model_family/            # model_config.yaml, model_config_mil.yaml, TRN-Plain (augablation_00_none_consistoff)
├── E2_feature_representation/  # pseudolabel(.yaml|_resnet50.yaml), camelyon_subtype_cls(.yaml|_resnet50.yaml)
├── E3_augmentation/
│   ├── graph_level/            # symlinks: TRN-Plain (-> E1), TRN-Ref = pseudolabel (-> E2); TRN-Plain + instance norm (_instnorm, own file)
│   └── training_level/         # augablation_00..05, augablation_loo_minus_*, consistablation_00..02
└── E4_transfer/                # symlink: pseudolabel (-> E2)
```

### Splits

All runs use `seed_42` (construction: [../pipeline.md §2](../pipeline.md#2-splits-seed_42)). GSDB: S1 = `test`
(scanner 1, 53 slides), S2 = `test_scanner2` (scanner 2, 90 slides, patient-disjoint). CAMELYON17: train centers
0,1,2, val center 3, test center 4 (E2).

**CAMELYON17 center rotation (E4).** Five folds `seed_42/test_c<t>/` (`create_camelyon_splits.py --phase
test_c<t>`, training/evaluation with `--split-phase test_c<t>`): each center is the test center once, the
preceding center (t − 1 mod 5) is the validation center, the other three are training centers. Same construction
as `seed_42` (whole centers, `--mode natural`, train/val 1:1 tumor:normal, test natural, patients disjoint);
`test_c4` is byte-identical to `seed_42`. Fold sizes and the command:
[cross_domain_experiments.md § Splits](cross_domain_experiments.md#splits).

| Fold | Train centers | Val | Test | Train / val / test slides (tumor) |
|---|---|---|---|---|
| `test_c0` | 1, 2, 3 | 4 | 0 | 214 (107) / 78 (39) / 100 (36) |
| `test_c1` | 2, 3, 4 | 0 | 1 | 208 (104) / 72 (36) / 100 (42) |
| `test_c2` | 0, 3, 4 | 1 | 2 | 230 (115) / 84 (42) / 100 (25) |
| `test_c3` | 0, 1, 4 | 2 | 3 | 234 (117) / 50 (25) / 100 (40) |
| `test_c4` | 0, 1, 2 | 3 | 4 | 206 (103) / 80 (40) / 100 (39) |

The validation set of `test_c3` is the smallest (center 2 has only 25 tumor slides → 50 val slides). The slide
counts are from the split files; slides without a graph (annotation QA) drop out at training time.

Shared: `--seed 42 --deterministic`; checkpoint/threshold selected on the in-domain validation
set only (scanner-1 for GSDB, train-centers for CAMELYON17).
**Gap** (generalization gap) = Test AUC (S1) − Test AUC (S2) for GSDB, i.e. the AUC lost under the
scanner shift on two test sets with the same stains; positive = worse on S2. CAMELYON17 has no in-domain
test set (whole-center split), so there Gap = Val AUC − Test AUC (center 4).
**All GSDB runs (E1–E3) train and validate on HE only** (`--he_only`; the patch-CNN config sets it) and
are tested on all stains (S1 `test`, S2 `test_scanner2`).

**Run prefix:** every training command in [cross_domain_experiments.md](cross_domain_experiments.md) carries `--prefix e<X>-<description>` (e.g. `--prefix e3-train-ladder1-gaussnoise`, X = experiment number). **Keep it, and keep it unique per run:** checkpoint names contain the config name but not the corpus, so without a prefix `e1-graph-plain-noaug` / `e3-graph-plain-aug`, `e2-gsdb-bin-gigapath` / `e3-graph-ref-noaug` and `e2-cam-bin-gigapath` / `e4-cam-bin-aug` / `e4-cam-bin-noaug` would overwrite each other's checkpoint. Keep it under ~50 characters (longest file names are already ~230 of 255 bytes). Meta file, checkpoints, training history, logs and the post-training evaluation are then saved as `<prefix>_<usual name>`. `--eval_prefix <name>` gives the evaluation its own prefix (training command: post-training evaluation; `evaluate_model.py`: that evaluation folder); default is the run's prefix, which is stored in its meta file. Allowed: letters, digits, `.`, `_`, `-` (anything else becomes `-`). Without `--prefix` all names stay exactly as before.

---

## E1 — Model family: Patch-CNN vs. MIL vs. Graph (GSDB, binary)

### Data protocol (all families)

| | Split files | Slides (GSDB `seed_42`) |
|---|---|---|
| Train | `train_HE.csv` | 125 (scanner 1, HE) |
| Val (checkpoint + threshold) | `val_HE.csv` | 42 (scanner 1, HE) |
| Test S1 | `test.csv` (= `test_all_stains.csv`) | 53 (scanner 1: 43 HE + 4 MG + 6 PAS) |
| Test S2 | `test_scanner2.csv` (= `test_scanner2_all_stains.csv`) | 90 (scanner 2: 30 HE + 30 MG + 30 PAS) |

Training and validation are HE-only, testing is on all stains.

### Evaluation protocol (all families)

| | Patch-CNN | MIL | Graph |
|---|---|---|---|
| Checkpoint selection (val_HE) | min. val loss (tile BCE) | min. val loss | `slide_f2` (config `monitor_metric`) |
| Slide score | mean tile probability | bag probability | slide probability |
| Slide threshold | G-mean on val_HE slide scores (`find_optimal_threshold`), no recalibration | same | same |
| Slide metrics (AUC, F1, **F2**, CIs, ...) | `_compute_slide_binary_metrics` | same function | same function |

### Commands

[cross_domain_experiments.md § E1](cross_domain_experiments.md#e1--patch-cnn-vs-mil-vs-graph-gsdb).

- Patch: no `--he_only` needed (the config sets `he_only: true`).
- MIL / graph: `--he_only` is required for this protocol.
- **How to test:** after each training, evaluate that run on `test` (scanner 1, all stains) and `test_scanner2` (scanner 2, all stains) with its own training meta (`evaluate_model.py --splits test test_scanner2 --meta_file <meta file starting with the run's prefix>`; training itself only evaluates `test`). Report the slide-level AUC and the threshold metrics at the validation threshold; the test sets are never used to adjust a threshold.

### Where the numbers are

Thresholds come from the validation split (`val_HE`, scanner 1) only. **Scanner 2 is used for prediction only:
no recalibration and no other adjustment on its scores or labels** (the adaptive recalibration is switched off
by default; `slide_adaptive_applied` / `adaptive_applied` is always `False`, so the top-level threshold metrics
`slide_f2`, `slide_accuracy`, ... resp. `f2`, `accuracy`, ... equal the values at the validation threshold
(`slide_at_validation_threshold` for patch-CNN / MIL, `at_validation_threshold` in the graph
`*_slide_metrics.json`)). AUC is threshold-free. Scanner separation relies on the split files
(`train_HE`, `val_HE` = scanner 1; `test_scanner2` = scanner 2), there is no runtime check. The reports still
contain `oracle_test_best_*` entries (best threshold swept on the test set): diagnostic only, never applied.

| Family | Val AUC | Test AUC / F2 (S1, S2) |
|---|---|---|
| Patch-CNN | `validation_thresholds/validation_thresholds.json` → `validation_slide_auroc` | `split_<split>/<model>_<split>_slide_metrics.json` → `slide_auroc`, `slide_f2` (+ log) |
| MIL | `validation_thresholds.json` → `validation_slide_auroc` | `<model>_<split>_slide_metrics.json` → `slide_auroc`, `slide_f2` (evaluations before 2026-10-16: log only) |
| Graph | `validation_thresholds.json` → `validation_slide_auroc` | `<model>_<split>_slide_metrics.json` (+ log) |

Patch-CNN (ResNet-18) vs. MIL: same tiles and backbone, differ mainly in the aggregation.
Patch-CNN (Prov-GigaPath) vs. Graph: same encoder, but **not** the same features -- the tiles
(256 px at downsample 10, `extract_tiles`) cover a 10× larger edge length at 10× lower resolution
than the graph patches (256 px at level 0, resized to 224), so this comparison confounds graph
modelling with input magnification.

### Input pipeline (all families, aligned)

E1 compares model families, not augmentation strategies (that is E3). All families therefore get the
same input pipeline: **Macenko colour normalisation to the scanner-1 reference and no augmentation**.

| | Patch-CNN | MIL | Graph (TRN-Plain) |
|---|---|---|---|
| Colour normalisation | Macenko → `gsdb_scanner1_lab.json`, `macenko_rescale_concentration: false` | same | same (graph build, `context_aware_pipeline_gsdb_inflammation_noaug`) |
| Implementation | `color_normalize` from `src/graph_creation_new/helper/transforms.py`, on the raw RGB tile before resize/crop | same | same function, on the raw patch before embedding |
| Training augmentation | none: `transforms.training` = `resize: 256`, `center_crop: 224` (identical to the validation transform) | none (same) | none (`noaug` corpus, `train_augmentations: []`, no consistency / mixup) |
| Tile filter | < 90 % white | < 90 % white | graph build tissue detection |

The Macenko transform was checked to be pixel-identical to the graph build's `color_normalize` with these
parameters.

**Still family-specific** (part of each method's own recipe, not aligned):

| | Patch-CNN (ResNet-18) | MIL | Graph (TRN-Plain) |
|---|---|---|---|
| Features / backbone | ImageNet ResNet-18, fully fine-tuned | ImageNet ResNet-18, frozen → unfrozen at epoch 10 | Prov-GigaPath embeddings (frozen) |
| Input scale | tiles 256 px at downsample 10 | same tiles | patches 256 px at level 0 (→ 224 px) |
| Regularisation | dropout 0 | dropout 0.3, 60 random tiles per bag | dropout 0.35, edge dropout 0.10, patch-edge dropout 0.6 |
| Per-slide feature normalisation | none | none | none (`graph_instance_norm: false`; its effect is measured in E3) |
| `pos_weight` | 1 (fixed) | 0.45 (fixed) | auto (neg/pos) |
| Precision | fp16 AMP | fp32 | fp16 AMP |
| Gradient clipping | no | 1.0 | 1.0 |
| Optimiser / schedule | SGD 1e-3, 25 ep., ES 15 (val loss) | AdamW 1e-4, 50 ep., ES 15 (val loss), cosine | AdamW 8e-5, 80 ep., ES 10 (`slide_f2`), cosine |

Note (MIL, fixed): `train_mil_model_image` / `train_mil_model_multimodal` used to read
`gradient_clipping` / `mixed_precision` / `save_frequency` from `config['common']`, which is empty after
the config is flattened -- so MIL trained **without** the configured gradient clipping (1.0). Fixed: they
now fall back to the flattened keys like the patch and graph pipelines. Effect: every MIL run from now on
clips gradients at 1.0 as configured (`mixed_precision: false` and `save_frequency: 5` resolve to the same
values as before). **MIL runs trained before this fix are not comparable** -- retrain MIL for E1.