# A2 — Tissue Region Network (TRN)

[Overview](README.md) · [A1 Framework](framework.md) · [A2 TRN](trn.md) · [A3 DG framework](dg_framework.md) · [Supporting artifacts](supporting_artifacts.md) · [References](references.md)

## Identity

| Item | Value |
|---|---|
| Class | `TissueRegionGraphModel` |
| File | `src/tissue_region_extraction/model.py` |
| Registry key | `("gnn", "tissue_region_net", "graph")` |
| Input | One cached `HeteroData` graph (S1) |
| Outputs | `slide_logits`, region logits, patch logits, auxiliary loss, optional multiclass logits |
| Training routine | `train_graph_segmentation_model` (`src/pipelines/train.py`) |
| Inference helpers | `src/tissue_region_extraction/routines.py` (region probabilities, connected positive components, GeoJSON export), `src/entrypoint/infer_tissue_regions.py`, `annotate_tissue_regions.py` |

## Forward path

```text
region.x ──► input_proj + LayerNorm + GELU ──► GATv2 ×num_layers (residual, LayerNorm) ──► h_r ─┐
                                                                                                  ├─► gated fusion ──► h̃_r ──┬─► region_head ──► region logits ──► max ─┐
patch.x  ──► patch_in + LayerNorm + GELU   ──► GATv2 ×patch_num_layers (residual)       ──► h_p ─┘   (scatter-mean over      │                                       │
                     │                                                                            ("patch","in","region"))  └─► attention pooling ──► slide_h ───────┴─► concat ─► slide_head ─► slide_logits
                     └──────────────────────────────────────────────────────────────────────► patch_head ──► patch logits (auxiliary loss)
```

| Component | Attribute(s) in code | Description |
|---|---|---|
| Input projection | `input_proj`, `input_norm`, `patch_in`, `patch_norm` | Linear → LayerNorm → GELU to `hidden_dim` / `patch_hidden_dim` |
| Region encoder | `region_convs`, `region_norms` | `GATv2Conv` stack with LayerNorm after each layer; uses edge features (`edge_dim = edge_feat_dim`) |
| Patch encoder | `patch_convs` | `GATv2Conv` stack without intermediate normalization; uses edge features |
| Gated cross-branch fusion | `patch_to_region_proj`, `patch_to_region_gate`, `region_fuse_norm` | `g = σ(MLP([h_r ‖ h_rᴾ]))`, `h̃_r = LayerNorm(h_r + g ⊙ h_rᴾ)` |
| Slide read-out | `pool` (`GlobalAttention`, GELU scorer), `slide_head` | Attention-pooled region representation concatenated with the maximum region logit (`use_extent_features: true` adds mean region/patch logits and positive fractions) |
| Region / patch heads | `region_head`, `patch_head` | One logit per node; binary auxiliary BCE with a capped positive weight |
| Optional multiclass heads | `region_cls_head`, `patch_cls_head` | Present only if `region_cls_target_field` / `patch_cls_target_field` and class names are set |
| Edge features | `_recompute_cosine_sim_edge_attr` | The cosine-similarity column is recomputed from the current node features |
| Large-graph guard | `max_patch_nodes_for_full_forward`, `max_patch_edges_for_full_forward` | Patch subgraph is subsampled above these limits |
| Optional, unused in the thesis | `GradientReversalLayer` (`domain_adversarial`), `_apply_manifold_mixup` | Domain-adversarial head and Manifold Mixup; weight / probability 0 in all thesis configs |

Configured sizes (thesis): hidden size 240 (region) and 160 (patch) on GSDB, 160 for both on CAMELYON17;
3 region layers; 5 patch layers on GSDB, 2 (binary) or 3 (multiclass) on CAMELYON17.

## Model configuration keys

These keys are read from the model config, for example
`configs/experiments/E2_feature_representation/model_config_tissue_region_segmentation_region_monitorloss_pseudolabel.yaml`:

- Architecture: `in_dim`, `hidden_dim`, `num_layers`, `heads`, `dropout_rate`, `edge_dropout_p`,
  `patch_hidden_dim`, `patch_num_layers`, `patch_heads`, `patch_edge_dropout_p`, `use_edge_features`,
  `edge_feat_dim`, `use_extent_features`.
- Auxiliary supervision: `region_aux_weight`, `region_aux_pos_weight_cap`, `mixed_region_weight`,
  `patch_aux_weight`, `patch_aux_pos_weight_cap`, `region_cls_*`, `patch_cls_*`.
- DG option inside the model: `graph_instance_norm` (see A3).
- Dataset-side: `resplit_large_regions`, `resplit_target_patches_per_region`, `resplit_k_region`,
  `resplit_min_patches_per_region`, `strict_graph_name_match`.

## Training objective

`train_graph_segmentation_model` combines the following terms. The weights are set in the `segmentation:`
block of the model config.

| Term | Config weight | Function |
|---|---|---|
| Slide BCE | `slide_loss_weight` | Slide label vs. `slide_logits` |
| Region BCE on resolved node labels | `region_loss_weight` | `_resolve_node_binary_targets`; fallback-labelled regions weighted with `fallback_region_weight` |
| Patch BCE on resolved node labels | `patch_loss_weight` | as above, patch level |
| Model-internal auxiliary loss | `aux_loss_weight` | returned by the model's forward pass |
| Region–slide consistency | `region_slide_consistency_weight` | `_region_slide_consistency_loss`: maximum region logit vs. slide label |
| Augmentation / feature consistency | `augmentation_consistency_weight`, `feature_consistency_weight` | see A3 |

Labels of unannotated nodes are resolved by the policy set in `unlabeled_region_policy` and
`negative_label_source` (`dataset_settings`). Optional pseudo-labels (`_apply_mil_pseudo_labels`) are
controlled by `pseudo_label_top_fraction` and `pseudo_label_warmup_epochs`. The share of fallback labels
is limited by `max_fallback_to_explicit_ratio` and `prefer_explicit_only_when_available`
(`_apply_fallback_subsampling`). The checkpoint is selected by `monitor_metric`.

## Builds on

- GATv2 graph encoders ([Brody et al., 2022](references.md#ref-brody-how-2022))
- representations derived from foundation models ([Chen et al., 2024](references.md#ref-chen-towards-2024))
- multi-level graph strategy ([Liang et al., 2023](references.md#ref-liang-interpretable-2023); [Pati et al., 2022](references.md#ref-pati-hierarchical-2022); [Hou et al., 2022](references.md#ref-hou-h2-mil-2022); [Chan et al., 2023](references.md#ref-chan-histopathology-2023))
- patch-to-region aggregation ([Chen et al., 2022](references.md#ref-chen-scaling-2022); [Guo et al., 2023](references.md#ref-guo-higt-2023))
- learned feature-wise gating (cross-branch fusion) ([Srivastava et al., 2015](references.md#ref-srivastava-highway-2015); [Ovalle et al., 2017](references.md#ref-ovalle-gated-2017))
- attention-based pooling ([Ilse et al., 2018](references.md#ref-ilse-attention-based-2018))
- attention + maximum-instance read-out (DSMIL) ([Li et al., 2021](references.md#ref-li-dual-stream-2021))
- patch-level auxiliary loss (as in CLAM) ([Lu et al., 2020](references.md#ref-lu-data-2020))
