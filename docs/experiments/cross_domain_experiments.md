# Experiments E1–E4

Run from the repo root, one command after another. `--prefix` must stay unique per run (checkpoint names do not
contain the corpus). Thresholds come from `val_HE` (scanner 1) only; scanner 2 is prediction only.
Gap = Test AUC S1 − Test AUC S2 (GSDB); CAMELYON17 has no in-domain test set, so there Gap = Val AUC − Test AUC. How the experiments work: [cross_domain_experiments_details.md](cross_domain_experiments_details.md); setup, splits, graph corpora, models and outputs: [../pipeline.md](../pipeline.md).
Determinism: all commands use `--deterministic` and `PYTHONHASHSEED=42`; nothing aborts (ops without a deterministic kernel only warn). Use the same GPU for every run (`CUDA_VISIBLE_DEVICES=<n>`, otherwise the freest GPU is picked). Limitations: [Validity notes](cross_domain_experiments_details.md#validity-notes-known-limitations-of-the-experiments).

## Splits
```bash
# GSDB -> data/splits/gsdb/seed_42/ (train/val/test/test_scanner2 + _HE/_all_stains, kfold_k5_fold0-4)
python src/entrypoint/create_splits.py --seed 42 --kfold-k 5

# CAMELYON17 -> data/splits/camelyon/seed_42/{train,val,test}.csv (whole centers 0,1,2 / 3 / 4)
python src/entrypoint/create_camelyon_splits.py --mode natural --seed 42 \
  --train-centers 0,1,2 --val-center 3 --test-center 4
```

CAMELYON17 center rotation (every center is the test center once, validation = the preceding center, train = the
other three) -> `data/splits/camelyon/seed_42/test_c<t>/`, used with `--split-phase test_c<t>` in training and
evaluation. `test_c4` is byte-identical to the `seed_42` split above (verified 2026-10-28).

```bash
for t in 0 1 2 3 4; do
  v=$(( (t + 4) % 5 )); tr=$(for c in 0 1 2 3 4; do [ $c -ne $t ] && [ $c -ne $v ] && printf "%s," $c; done); tr=${tr%,}
  python src/entrypoint/create_camelyon_splits.py --mode natural --seed 42 \
    --train-centers $tr --val-center $v --test-center $t --phase test_c$t
done
```

| Fold | Train centers | Val center | Test center | Train / val / test slides (tumor) |
|---|---|---|---|---|
| `test_c0` | 1, 2, 3 | 4 | 0 | 214 (107) / 78 (39) / 100 (36) |
| `test_c1` | 2, 3, 4 | 0 | 1 | 208 (104) / 72 (36) / 100 (42) |
| `test_c2` | 0, 3, 4 | 1 | 2 | 230 (115) / 84 (42) / 100 (25) |
| `test_c3` | 0, 1, 4 | 2 | 3 | 234 (117) / 50 (25) / 100 (40) |
| `test_c4` | 0, 1, 2 | 3 | 4 | 206 (103) / 80 (40) / 100 (39) |

Limitation: the CAMELYON17 Macenko reference (`configs/graph_build/reference_stats/camelyon_train_lab.json`) was
computed from 20 slides of the `seed_42` train split (centers 0, 1, 2), and all corpora are normalised to it. In
folds `test_c0`–`test_c2` the test center therefore contributed to the colour-normalisation target.

The non-phase commands overwrite the existing `seed_42` folder (files the scripts do not write are kept). Logs:
`results/logs/split_creation/`. To test without touching `data/splits/`, add `--data-dir <dir> --output-dir <dir>`
(GSDB additionally needs `<dir>/raw` -> `data/raw`).

## Graph creation

Corpora and their settings: [../pipeline.md §3](../pipeline.md#3-graph-corpora).

```bash
# Macenko reference stats (files already in configs/graph_build/reference_stats/; only to recompute)
PYTHONHASHSEED=42 python scripts/graph_creation/compute_color_reference_stats.py \
  --dataset gsdb --split-csv data/splits/gsdb/seed_42/train.csv --scanner-id 1 \
  --patch-size 128 --patch-level 0 --num-slides 20 --patches-per-slide 40 \
  --output configs/graph_build/reference_stats/gsdb_scanner1_lab.json --seed 42
PYTHONHASHSEED=42 python scripts/graph_creation/compute_color_reference_stats.py \
  --dataset camelyon --split-csv data/splits/camelyon/seed_42/train.csv \
  --patch-size 256 --patch-level 0 --num-slides 20 --patches-per-slide 40 \
  --output configs/graph_build/reference_stats/camelyon_train_lab.json --seed 42

# output: data/processed/graphs/<dataset>/<config name>
PYTHONHASHSEED=42 python -u src/entrypoint/build_graph.py \
  --config configs/experiments/graph_corpora/context_aware_gsdb_inflammation.yaml
PYTHONHASHSEED=42 python -u src/entrypoint/build_graph.py \
  --config configs/experiments/graph_corpora/context_aware_gsdb_inflammation_noaug.yaml
PYTHONHASHSEED=42 python -u src/entrypoint/build_graph.py \
  --config configs/experiments/graph_corpora/context_aware_runtime_non_hist_gsdb_inflammation.yaml
PYTHONHASHSEED=42 python -u src/entrypoint/build_graph.py \
  --config configs/experiments/graph_corpora/context_aware_camelyon_domain_robust_labels.yaml
PYTHONHASHSEED=42 python -u src/entrypoint/build_graph.py \
  --config configs/experiments/graph_corpora/context_aware_camelyon_labels_noaug.yaml
PYTHONHASHSEED=42 python -u src/entrypoint/build_graph.py \
  --config configs/experiments/graph_corpora/context_aware_runtime_non_hist_camelyon_domain_robust_labels.yaml
```

## E1 — Patch-CNN vs. MIL vs. Graph (GSDB)

All values in the result tables below are taken from the thesis (AUROC and F2 in %, gap in pp, single runs, seed 42).

```bash
# E1: Patch-CNN (ResNet-18, ImageNet, fine-tuned); the config sets he_only: true
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset gsdb --task inflammation --model_type cnn \
  --architecture resnet18 --input_type image --seed 42 --deterministic \
  --prefix e1-patch-resnet18 \
  --config_name configs/experiments/E1_model_family/model_config.yaml

# E1: Patch-CNN (Prov-GigaPath, frozen encoder; needs HF_TOKEN)
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset gsdb --task inflammation --model_type cnn \
  --architecture gigapath --input_type image --seed 42 --deterministic \
  --prefix e1-patch-gigapath \
  --config_name configs/experiments/E1_model_family/model_config.yaml

# E1: MIL (attention_mil, ResNet-18, 60-tile bags)
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset gsdb --task inflammation --model_type mil \
  --architecture attention_mil --feature_extractor resnet18 --input_type image --mil --he_only \
  --seed 42 --deterministic --prefix e1-mil-resnet18 \
  --config_name configs/experiments/E1_model_family/model_config_mil.yaml

# E1: MIL (attention_mil, Prov-GigaPath tile encoder = the frozen encoder of the patch-GigaPath; needs HF_TOKEN)
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset gsdb --task inflammation --model_type mil \
  --architecture attention_mil --feature_extractor gigapath --input_type image --mil --he_only \
  --seed 42 --deterministic --prefix e1-mil-gigapath \
  --config_name configs/experiments/E1_model_family/model_config_mil_gigapath.yaml

# E1: TRN-Plain on the non-augmented Prov-GigaPath corpus (no graph instance norm)
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset gsdb --task inflammation --model_type gnn \
  --architecture tissue_region_net --input_type graph --embedding_type pov-gigapath \
  --graph_target_dir data/processed/graphs/gsdb/context_aware_gsdb_inflammation_noaug --cached --mil \
  --he_only --seed 42 --deterministic \
  --prefix e1-graph-plain-noaug \
  --config_name configs/experiments/E1_model_family/model_config_tissue_region_gsdb_inflammation_augablation_00_none_consistoff.yaml
```

Test: `PYTHONHASHSEED=42 python src/entrypoint/evaluate_model.py --splits test test_scanner2 --meta_file <meta of the run>`

| Model | Val AUC | Test AUC S1 | Test AUC S2 | Gap (S1 − S2) | Test F2 (S2) |
|---|---|---|---|---|---|
| Patch-CNN ResNet-18 | 99.7 | 98.6 | 72.7 | 25.9 | 80.3 |
| Patch-CNN Prov-GigaPath | 100.0 | 100.0 | 80.4 | 19.6 | 87.6 |
| MIL ResNet-18 | 98.2 | 93.5 | 74.6 | 18.9 | 67.9 |
| MIL Prov-GigaPath | 96.8 | 95.8 | 80.1 | 15.8 | 86.0 |
| TRN-Plain | 97.9 | 96.9 | 82.6 | 14.3 | 54.3 |

## E2 — Prov-GigaPath vs. ResNet-50 (graph)

```bash
# E2 GSDB binary, TRN-Ref: Prov-GigaPath
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset gsdb --task inflammation --model_type gnn \
  --architecture tissue_region_net --input_type graph --embedding_type pov-gigapath \
  --graph_target_dir data/processed/graphs/gsdb/context_aware_gsdb_inflammation --cached --mil \
  --he_only --seed 42 --deterministic \
  --prefix e2-gsdb-bin-gigapath \
  --config_name configs/experiments/E2_feature_representation/model_config_tissue_region_segmentation_region_monitorloss_pseudolabel.yaml

# E2 GSDB binary, TRN-Ref: ResNet-50
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset gsdb --task inflammation --model_type gnn \
  --architecture tissue_region_net --input_type graph --embedding_type resnet50 \
  --graph_target_dir data/processed/graphs/gsdb/context_aware_runtime_non_hist_gsdb_inflammation \
  --cached --mil --he_only --seed 42 --deterministic \
  --prefix e2-gsdb-bin-resnet50 \
  --config_name configs/experiments/E2_feature_representation/model_config_tissue_region_segmentation_region_monitorloss_pseudolabel_resnet50.yaml

# E2 CAMELYON17 binary, TRN-Ref: Prov-GigaPath
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset camelyon --task lymph --model_type gnn \
  --architecture tissue_region_net --input_type graph --embedding_type pov-gigapath \
  --graph_target_dir data/processed/graphs/camelyon/context_aware_camelyon_domain_robust_labels \
  --cached --mil --seed 42 --deterministic \
  --prefix e2-cam-bin-gigapath \
  --config_name configs/experiments/E2_feature_representation/model_config_tissue_region_segmentation_region_monitorloss_pseudolabel.yaml

# E2 CAMELYON17 binary, TRN-Ref: ResNet-50
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset camelyon --task lymph --model_type gnn \
  --architecture tissue_region_net --input_type graph --embedding_type resnet50 \
  --graph_target_dir data/processed/graphs/camelyon/context_aware_runtime_non_hist_camelyon_domain_robust_labels \
  --cached --mil --seed 42 --deterministic \
  --prefix e2-cam-bin-resnet50 \
  --config_name configs/experiments/E2_feature_representation/model_config_tissue_region_segmentation_region_monitorloss_pseudolabel_resnet50.yaml

# E2 CAMELYON17 multiclass (subtype, 4 classes): Prov-GigaPath
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset camelyon --task lymph --model_type gnn \
  --architecture tissue_region_net --input_type graph --embedding_type pov-gigapath \
  --graph_target_dir data/processed/graphs/camelyon/context_aware_camelyon_domain_robust_labels \
  --cached --mil --seed 42 --deterministic \
  --prefix e2-cam-multi-gigapath \
  --config_name configs/experiments/E2_feature_representation/model_config_tissue_region_camelyon_subtype_cls.yaml

# E2 CAMELYON17 multiclass (subtype, 4 classes): ResNet-50
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset camelyon --task lymph --model_type gnn \
  --architecture tissue_region_net --input_type graph --embedding_type resnet50 \
  --graph_target_dir data/processed/graphs/camelyon/context_aware_runtime_non_hist_camelyon_domain_robust_labels \
  --cached --mil --seed 42 --deterministic \
  --prefix e2-cam-multi-resnet50 \
  --config_name configs/experiments/E2_feature_representation/model_config_tissue_region_camelyon_subtype_cls_resnet50.yaml
```

Test: GSDB `PYTHONHASHSEED=42 python src/entrypoint/evaluate_model.py --splits test test_scanner2 --meta_file <meta of the run>`; CAMELYON17 `... --splits test ...`

| Dataset / task | Embedding | Val AUC | Test AUC (S1 / center 4) | Test AUC S2 | Gap |
|---|---|---|---|---|---|
| GSDB binary | Prov-GigaPath | 100.0 | 95.8 | 89.7 | 6.1 |
| GSDB binary | ResNet-50 | 94.7 | 90.9 | 78.6 | 12.3 |
| CAMELYON17 binary | Prov-GigaPath | 93.8 | 95.3 | – | −1.5 |
| CAMELYON17 binary | ResNet-50 ¹ | 61.7 | 58.9 | – | 2.8 |

Gap: GSDB = S1 − S2; CAMELYON17 = Val − Test.
¹ Selected checkpoint is epoch 1 of 11, where the model called every validation slide tumor; the values reflect this checkpoint, not the encoder.

| CAMELYON17 multiclass | Val AUROC | Test AUROC | Test Macro-F1 | Test κ |
|---|---|---|---|---|
| Prov-GigaPath | 79.2 | 78.6 | 39.5 | 63.0 |
| ResNet-50 | n/r | n/r | n/r | n/r |

n/r: not reported in the thesis.

## E3 — Augmentation study (GSDB)

```bash
# E3 graph level: TRN-Plain on the augmented corpus  (Plain/G-noaug = E1, Ref/G-aug = E2)
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset gsdb --task inflammation --model_type gnn \
  --architecture tissue_region_net --input_type graph --embedding_type pov-gigapath \
  --graph_target_dir data/processed/graphs/gsdb/context_aware_gsdb_inflammation --cached --mil \
  --he_only --seed 42 --deterministic \
  --prefix e3-graph-plain-aug \
  --config_name configs/experiments/E3_augmentation/graph_level/model_config_tissue_region_gsdb_inflammation_augablation_00_none_consistoff.yaml

# E3 graph level: TRN-Plain + graph instance norm on the non-augmented corpus
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset gsdb --task inflammation --model_type gnn \
  --architecture tissue_region_net --input_type graph --embedding_type pov-gigapath \
  --graph_target_dir data/processed/graphs/gsdb/context_aware_gsdb_inflammation_noaug --cached --mil \
  --he_only --seed 42 --deterministic \
  --prefix e3-graph-plain-in-noaug \
  --config_name configs/experiments/E3_augmentation/graph_level/model_config_tissue_region_gsdb_inflammation_augablation_00_none_consistoff_instnorm.yaml

# E3 graph level: TRN-Plain + graph instance norm on the augmented corpus
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset gsdb --task inflammation --model_type gnn \
  --architecture tissue_region_net --input_type graph --embedding_type pov-gigapath \
  --graph_target_dir data/processed/graphs/gsdb/context_aware_gsdb_inflammation --cached --mil \
  --he_only --seed 42 --deterministic \
  --prefix e3-graph-plain-in-aug \
  --config_name configs/experiments/E3_augmentation/graph_level/model_config_tissue_region_gsdb_inflammation_augablation_00_none_consistoff_instnorm.yaml

# E3 graph level: TRN-Ref on the non-augmented corpus
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset gsdb --task inflammation --model_type gnn \
  --architecture tissue_region_net --input_type graph --embedding_type pov-gigapath \
  --graph_target_dir data/processed/graphs/gsdb/context_aware_gsdb_inflammation_noaug --cached --mil \
  --he_only --seed 42 --deterministic \
  --prefix e3-graph-ref-noaug \
  --config_name configs/experiments/E3_augmentation/graph_level/model_config_tissue_region_segmentation_region_monitorloss_pseudolabel.yaml

# not run again (identical configs): augablation_05_full, consistablation_02_full (= E2 TRN-Ref), augablation_loo_minus_graph_mixstyle (= augablation_04)

# E3 training level: augablation_00_none (augmented corpus)
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset gsdb --task inflammation --model_type gnn \
  --architecture tissue_region_net --input_type graph --embedding_type pov-gigapath \
  --graph_target_dir data/processed/graphs/gsdb/context_aware_gsdb_inflammation --cached --mil \
  --he_only --seed 42 --deterministic \
  --prefix e3-train-aug00-none \
  --config_name configs/experiments/E3_augmentation/training_level/model_config_tissue_region_gsdb_inflammation_augablation_00_none.yaml

# E3 training level: augablation_01_plus_feature_gaussian_noise (augmented corpus)
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset gsdb --task inflammation --model_type gnn \
  --architecture tissue_region_net --input_type graph --embedding_type pov-gigapath \
  --graph_target_dir data/processed/graphs/gsdb/context_aware_gsdb_inflammation --cached --mil \
  --he_only --seed 42 --deterministic \
  --prefix e3-train-ladder1-gaussnoise \
  --config_name configs/experiments/E3_augmentation/training_level/model_config_tissue_region_gsdb_inflammation_augablation_01_plus_feature_gaussian_noise.yaml

# E3 training level: augablation_02_plus_feature_dropout (augmented corpus)
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset gsdb --task inflammation --model_type gnn \
  --architecture tissue_region_net --input_type graph --embedding_type pov-gigapath \
  --graph_target_dir data/processed/graphs/gsdb/context_aware_gsdb_inflammation --cached --mil \
  --he_only --seed 42 --deterministic \
  --prefix e3-train-ladder2-dropout \
  --config_name configs/experiments/E3_augmentation/training_level/model_config_tissue_region_gsdb_inflammation_augablation_02_plus_feature_dropout.yaml

# E3 training level: augablation_03_plus_graph_mixup (augmented corpus)
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset gsdb --task inflammation --model_type gnn \
  --architecture tissue_region_net --input_type graph --embedding_type pov-gigapath \
  --graph_target_dir data/processed/graphs/gsdb/context_aware_gsdb_inflammation --cached --mil \
  --he_only --seed 42 --deterministic \
  --prefix e3-train-ladder3-mixup \
  --config_name configs/experiments/E3_augmentation/training_level/model_config_tissue_region_gsdb_inflammation_augablation_03_plus_graph_mixup.yaml

# E3 training level: augablation_04_plus_feature_scale_noise (augmented corpus)
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset gsdb --task inflammation --model_type gnn \
  --architecture tissue_region_net --input_type graph --embedding_type pov-gigapath \
  --graph_target_dir data/processed/graphs/gsdb/context_aware_gsdb_inflammation --cached --mil \
  --he_only --seed 42 --deterministic \
  --prefix e3-train-ladder4-scalenoise \
  --config_name configs/experiments/E3_augmentation/training_level/model_config_tissue_region_gsdb_inflammation_augablation_04_plus_feature_scale_noise.yaml

# E3 training level: augablation_loo_minus_feature_gaussian_noise (augmented corpus)
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset gsdb --task inflammation --model_type gnn \
  --architecture tissue_region_net --input_type graph --embedding_type pov-gigapath \
  --graph_target_dir data/processed/graphs/gsdb/context_aware_gsdb_inflammation --cached --mil \
  --he_only --seed 42 --deterministic \
  --prefix e3-train-loo-no-gaussnoise \
  --config_name configs/experiments/E3_augmentation/training_level/model_config_tissue_region_gsdb_inflammation_augablation_loo_minus_feature_gaussian_noise.yaml

# E3 training level: augablation_loo_minus_feature_dropout (augmented corpus)
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset gsdb --task inflammation --model_type gnn \
  --architecture tissue_region_net --input_type graph --embedding_type pov-gigapath \
  --graph_target_dir data/processed/graphs/gsdb/context_aware_gsdb_inflammation --cached --mil \
  --he_only --seed 42 --deterministic \
  --prefix e3-train-loo-no-dropout \
  --config_name configs/experiments/E3_augmentation/training_level/model_config_tissue_region_gsdb_inflammation_augablation_loo_minus_feature_dropout.yaml

# E3 training level: augablation_loo_minus_graph_mixup (augmented corpus)
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset gsdb --task inflammation --model_type gnn \
  --architecture tissue_region_net --input_type graph --embedding_type pov-gigapath \
  --graph_target_dir data/processed/graphs/gsdb/context_aware_gsdb_inflammation --cached --mil \
  --he_only --seed 42 --deterministic \
  --prefix e3-train-loo-no-mixup \
  --config_name configs/experiments/E3_augmentation/training_level/model_config_tissue_region_gsdb_inflammation_augablation_loo_minus_graph_mixup.yaml

# E3 training level: augablation_loo_minus_feature_scale_noise (augmented corpus)
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset gsdb --task inflammation --model_type gnn \
  --architecture tissue_region_net --input_type graph --embedding_type pov-gigapath \
  --graph_target_dir data/processed/graphs/gsdb/context_aware_gsdb_inflammation --cached --mil \
  --he_only --seed 42 --deterministic \
  --prefix e3-train-loo-no-scalenoise \
  --config_name configs/experiments/E3_augmentation/training_level/model_config_tissue_region_gsdb_inflammation_augablation_loo_minus_feature_scale_noise.yaml

# E3 training level: consistablation_00_off (augmented corpus)
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset gsdb --task inflammation --model_type gnn \
  --architecture tissue_region_net --input_type graph --embedding_type pov-gigapath \
  --graph_target_dir data/processed/graphs/gsdb/context_aware_gsdb_inflammation --cached --mil \
  --he_only --seed 42 --deterministic \
  --prefix e3-train-consist-off \
  --config_name configs/experiments/E3_augmentation/training_level/model_config_tissue_region_gsdb_inflammation_consistablation_00_off.yaml

# E3 training level: consistablation_01_half (augmented corpus)
PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset gsdb --task inflammation --model_type gnn \
  --architecture tissue_region_net --input_type graph --embedding_type pov-gigapath \
  --graph_target_dir data/processed/graphs/gsdb/context_aware_gsdb_inflammation --cached --mil \
  --he_only --seed 42 --deterministic \
  --prefix e3-train-consist-half \
  --config_name configs/experiments/E3_augmentation/training_level/model_config_tissue_region_gsdb_inflammation_consistablation_01_half.yaml
```

Test: as E1.

| Graph level | Corpus | Val AUC | Test AUC S1 | Test AUC S2 | Gap (S1 − S2) |
|---|---|---|---|---|---|
| TRN-Plain | noaug | 97.9 | 96.9 | 82.6 | 14.3 |
| TRN-Plain | aug | 98.2 | 95.4 | 81.9 | 13.6 |
| TRN-Plain + instance norm | noaug | 100.0 | 97.5 | 91.9 | 5.6 |
| TRN-Plain + instance norm | aug | 99.7 | 96.9 | 79.6 | 17.3 |
| TRN-Ref | noaug | 99.7 | 96.4 | 86.9 | 9.5 |
| TRN-Ref | aug | 100.0 | 95.8 | 89.7 | 6.1 |

| Training level | Val AUC | Test AUC S1 | Test AUC S2 | Gap (S1 − S2) |
|---|---|---|---|---|
| none | 99.5 | 98.2 | 79.4 | 18.8 |
| + Gaussian noise | 100.0 | 98.1 | 82.5 | 15.6 |
| + feature dropout | 98.7 | 97.2 | 82.7 | 14.5 |
| + graph Mixup | 99.5 | 97.3 | 85.7 | 11.6 |
| + feature scale noise | 100.0 | 97.2 | 85.6 | 11.6 |
| + graph MixStyle (= TRN-Ref) | 100.0 | 95.8 | 89.7 | 6.1 |
| − Gaussian noise | 99.7 | 97.1 | 84.7 | 12.3 |
| − feature dropout | 98.9 | 96.9 | 80.1 | 16.8 |
| − graph Mixup | 99.5 | 97.4 | 81.1 | 16.3 |
| − feature scale noise | 99.5 | 98.0 | 85.3 | 12.7 |
| − graph MixStyle (= + scale noise) | 100.0 | 97.2 | 85.6 | 11.6 |
| consistency 0 | 98.9 | 96.9 | 89.7 | 7.2 |
| consistency 0.225 | 100.0 | 96.0 | 86.9 | 9.1 |
| consistency 0.45 (= TRN-Ref) | 100.0 | 95.8 | 89.7 | 6.1 |

## E4 — CAMELYON17 transfer (binary)

E4 is trained over the five center-rotation splits (`test_c0`–`test_c4`, see [Splits](#splits); create them
first): every center is the test center once, both corpora (C-aug, C-noaug) per fold, 10 runs. Same config for
both corpora; only `--graph_target_dir` differs. `test_c4` is the `seed_42` split of E2, it is trained again
here so that all folds come from the same protocol.

```bash
# E4: TRN-Ref on the augmented (C-aug) and the non-augmented (C-noaug) CAMELYON17 corpus, one run per test center
for t in 0 1 2 3 4; do
  for corpus in aug noaug; do
    case $corpus in
      aug)   dir=context_aware_camelyon_domain_robust_labels ;;
      noaug) dir=context_aware_camelyon_labels_noaug ;;
    esac
    CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset camelyon --task lymph \
      --model_type gnn --architecture tissue_region_net --input_type graph --embedding_type pov-gigapath \
      --graph_target_dir data/processed/graphs/camelyon/$dir --cached --mil --seed 42 --deterministic \
      --split-phase test_c$t \
      --prefix e4-cam-bin-$corpus-tc$t \
      --config_name configs/experiments/E4_transfer/model_config_tissue_region_segmentation_region_monitorloss_pseudolabel.yaml
  done
done
```

Single run, e.g. C-noaug with center 0 as test center:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python -u src/entrypoint/train_model.py --dataset camelyon --task lymph \
  --model_type gnn --architecture tissue_region_net --input_type graph --embedding_type pov-gigapath \
  --graph_target_dir data/processed/graphs/camelyon/context_aware_camelyon_labels_noaug --cached --mil --seed 42 \
  --deterministic --split-phase test_c0 --prefix e4-cam-bin-noaug-tc0 \
  --config_name configs/experiments/E4_transfer/model_config_tissue_region_segmentation_region_monitorloss_pseudolabel.yaml
```

Test: training evaluates val (threshold) and test of the same fold automatically (`--split-phase` is passed
through). Manual re-evaluation:
`PYTHONHASHSEED=42 python src/entrypoint/evaluate_model.py --splits test --meta_file <meta of the run>`
(the split phase is restored from the meta file and takes precedence over a `--split-phase` on the command line).
Check in the run log: `grep "split csv" results/logs/<prefix>_*/*.log` must show `seed_42/test_c<t>/`.

The fold is part of every output name: log folder, checkpoints, training history and post-training evaluation end
in `…_test_c<t>[_<timestamp>]`, the meta file carries `split_phase-test_c<t>`, and a manual evaluation folder in
`results/test_results/` carries `test_c<t>`. Runs without `--split-phase` keep their previous names.

| Test center | Val center | Corpus | Val AUC | Test AUC | Gap (Val − Test) |
|---|---|---|---|---|---|
| 0 | 4 | aug | 96.2 | 90.9 | 5.3 |
| 0 | 4 | noaug | 96.0 | 91.6 | 4.5 |
| 1 | 0 | aug | 95.5 | 80.2 | 15.3 |
| 1 | 0 | noaug ² | 48.8 | 53.2 | −4.4 |
| 2 | 1 | aug | 85.1 | 94.3 | −9.2 |
| 2 | 1 | noaug | 87.5 | 94.1 | −6.6 |
| 3 | 2 | aug | 90.9 | 95.9 | −5.0 |
| 3 | 2 | noaug | 91.4 | 95.5 | −4.2 |
| 4 | 3 | aug | 93.8 | 95.3 | −1.5 |
| 4 | 3 | noaug | 95.1 | 97.1 | −2.0 |
| mean ± SD over folds | | aug | 92.3 ± 4.5 | 91.3 ± 6.5 | 1.0 ± 9.6 |
| mean ± SD over folds | | noaug | 83.8 ± 19.8 | 86.3 ± 18.6 | −2.5 ± 4.2 |

² Selected checkpoint is epoch 1 of 11, where the model called every slide tumor (sens. 100, spec. 0).
