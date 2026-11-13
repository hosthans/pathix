from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GlobalAttention
from torch_geometric.utils import scatter

from src.core.registry_utils import register_model
from src.graph_creation_new.helper.edges import (
    EDGE_FEAT_COL_COSINE_SIM,
    EDGE_FEAT_COL_FEAT_VALID,
    normalize_edge_feat_for_model,
)
from src.utils import graph_patch_utils


def _sanitize_edge_attr(
    edge_attr: Optional[torch.Tensor],
    edge_index: torch.Tensor,
    use_edge_features: bool,
    edge_feat_dim: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if (
        not use_edge_features
        or not torch.is_tensor(edge_attr)
        or edge_attr.dim() != 2
        or edge_attr.size(0) != edge_index.size(1)
        or edge_attr.size(1) != edge_feat_dim
    ):
        return None
    return normalize_edge_feat_for_model(edge_attr.float().to(device))


class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        """Pass the input through unchanged, stashing lambd for the backward pass."""
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Return the negated, lambd-scaled gradient."""
        return -ctx.lambd * grad_output, None


class GradientReversalLayer(nn.Module):
    """Identity on the forward pass, negates (and scales by `lambd`) the gradient on the
    backward pass -- the standard DANN (Ganin & Lempitsky 2015) trick for domain-adversarial
    training. A domain classifier reading from behind this layer learns normally (tries to tell
    domains apart), while everything upstream of it gets pushed AWAY from features that make the
    domain classifier's job easier, i.e. is trained to discard domain-identifying signal.

    Used here with a synthetic proxy domain (original graph vs. its own augmented copy, see
    train_epoch_graph's domain-adversarial loss) rather than real scanner1-vs-scanner2 labels,
    since no scanner2 data exists at training time at all (pure domain generalization, not
    adaptation) -- the augmentation is chosen to mimic the kind of shift (magnitude/scale)
    measured to actually differ between the two real scanners.
    """

    def __init__(self, lambd: float = 1.0) -> None:
        """Store the gradient-reversal scale factor."""
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the identity-forward/reversed-gradient-backward transform."""
        return _GradReverseFn.apply(x, self.lambd)


def _graph_instance_normalize(x: torch.Tensor, batch: Optional[torch.Tensor], eps: float = 1e-5) -> torch.Tensor:
    x = x.float()
    if batch is None or batch.numel() != x.size(0):
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + eps)

    batch = batch.long()
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    mean = scatter(x, batch, dim=0, dim_size=num_graphs, reduce='mean')
    mean_sq = scatter(x * x, batch, dim=0, dim_size=num_graphs, reduce='mean')
    var = (mean_sq - mean * mean).clamp(min=0.0)
    std = torch.sqrt(var + eps)
    return (x - mean[batch]) / std[batch]


def _recompute_cosine_sim_edge_attr(
    edge_attr: Optional[torch.Tensor],
    edge_index: torch.Tensor,
    x_for_cosine_sim: torch.Tensor,
) -> Optional[torch.Tensor]:
    if edge_attr is None or not torch.is_tensor(edge_attr) or edge_attr.dim() != 2:
        return edge_attr
    if edge_attr.size(1) <= max(EDGE_FEAT_COL_COSINE_SIM, EDGE_FEAT_COL_FEAT_VALID):
        return edge_attr
    if edge_index.numel() == 0:
        return edge_attr

    src_idx = edge_index[0].long()
    dst_idx = edge_index[1].long()
    x = x_for_cosine_sim.float()
    src_vec = x[src_idx]
    dst_vec = x[dst_idx]
    src_norm = src_vec.norm(dim=1)
    dst_norm = dst_vec.norm(dim=1)
    denom = (src_norm * dst_norm).clamp(min=1e-8)
    cosine_sim = (src_vec * dst_vec).sum(dim=1) / denom
    valid = (src_norm > 1e-8) & (dst_norm > 1e-8)
    cosine_sim = cosine_sim.clamp(-1.0, 1.0) * valid.float()

    edge_attr = edge_attr.clone()
    edge_attr[:, EDGE_FEAT_COL_COSINE_SIM] = cosine_sim.to(edge_attr.dtype)
    edge_attr[:, EDGE_FEAT_COL_FEAT_VALID] = valid.to(edge_attr.dtype)
    return edge_attr


@register_model(("gnn", "tissue_region_net", "graph"))
class TissueRegionGraphModel(nn.Module):
    """Region-aware graph model for slide-level classification + region localization.

    The model is trained by existing graph training routines (slide-level labels)
    and can optionally add auxiliary region supervision from annotation_y. It also
    processes patch nodes with their own GNN branch, trained with an auxiliary
    patch-level BCE loss (from patch.annotation_y within annotated tumor regions)
    and fused into the region embeddings before pooling/region-head — this is what
    lets the model pick up fine-grained tumor cues instead of relying solely on a
    coarse per-region embedding.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 1,
        dropout_rate: float = 0.2,
        in_dim: int = 1536,
        hidden_dim: int = 256,
        num_layers: int = 3,
        heads: int = 4,
        edge_dropout_p: float = 0.05,
        region_aux_weight: float = 0.20,
        region_aux_pos_weight_cap: float = 6.0,
        mixed_region_weight: float = 0.20,
        patch_hidden_dim: Optional[int] = None,
        patch_num_layers: int = 2,
        patch_aux_weight: float = 0.30,
        patch_aux_pos_weight_cap: float = 6.0,
        patch_edge_dropout_p: float = 0.05,
        max_patch_nodes_for_full_forward: int = 6000,
        max_patch_edges_for_full_forward: int = 150000,
        graph_instance_norm: bool = False,
        patch_heads: Optional[int] = None,
        use_edge_features: bool = True,
        edge_feat_dim: int = 8,
        domain_adversarial: bool = False,
        domain_adversarial_lambda: float = 1.0,
        region_cls_target_field: Optional[str] = None,
        region_cls_class_names: Optional[List[str]] = None,
        region_cls_weight: float = 0.30,
        patch_cls_target_field: Optional[str] = None,
        patch_cls_class_names: Optional[List[str]] = None,
        patch_cls_weight: float = 0.30,
        use_extent_features: bool = False,
    ) -> None:
        """Build the region/patch GNN branches and their auxiliary heads from config."""
        super().__init__()

        self.model_name = str(model_name)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = max(1, int(num_layers))
        self.dropout_rate = float(dropout_rate)
        self.edge_dropout_p = float(edge_dropout_p)
        # Off by default -- current behavior unchanged. When on, raw region/patch input
        # features are normalized per-slide (per graph, using only that graph's own nodes)
        # before input_proj/patch_in -- see _graph_instance_normalize's docstring for why this
        # targets domain shift specifically. Opt-in via the model config's
        # `graph_instance_norm: true` (src/core/factories.py wires it through).
        self.graph_instance_norm = bool(graph_instance_norm)
        self.region_aux_weight = float(region_aux_weight)
        self.region_aux_pos_weight_cap = max(1.0, float(region_aux_pos_weight_cap))
        # Per-sample weight (not the outer region_aux_weight loss-term weight above) applied to
        # region_aux_loss specifically for regions helper/regions.py's geometric mode flagged
        # MIXED_REGION_LABEL (region.label_ambiguous) -- see _node_aux_loss's sample_weight
        # docstring. Default 0.20 matches src/pipelines/train.py's fallback_region_weight default
        # (the SEPARATE, externally-computed region_loss's own down-weighting for the same kind
        # of region) purely for consistency between the two independent region-supervision paths
        # -- not read from the same config key, so keep them in sync manually if either changes.
        self.mixed_region_weight = max(0.0, min(1.0, float(mixed_region_weight)))
        self._declared_in_dim = int(in_dim)

        self.patch_hidden_dim = int(patch_hidden_dim) if patch_hidden_dim is not None else self.hidden_dim
        self.patch_num_layers = max(1, int(patch_num_layers))
        self.patch_aux_weight = float(patch_aux_weight)
        self.patch_aux_pos_weight_cap = max(1.0, float(patch_aux_pos_weight_cap))
        self.patch_edge_dropout_p = float(patch_edge_dropout_p)
        self.max_patch_nodes_for_full_forward = int(max_patch_nodes_for_full_forward)
        self.max_patch_edges_for_full_forward = int(max_patch_edges_for_full_forward)
        # Deterministic-seed disambiguator for subsample_large_patch_graph's `seed=` -- see
        # WSIGraphModel._subsample_large_patch_graph (src/models/graph.py) for the full rationale.
        self._patch_subsample_counter = 0
        # Consumes ("region","near","region") and ("patch","near","patch") edge_feat (see
        # src/graph_creation_new/helper/edges.py's build_edge_features -- distance/inverse-
        # distance/delta-xy/same-region flag, plus a cosine-similarity between the two
        # endpoints' own pov-gigapath embeddings as of 2026-08-09). Added 2026-08-11 alongside
        # the same wiring already done for src/models/robust_slide_classifier.py's
        # RegionSlideClassifier -- previously computed at graph-build time and stored, but never
        # read by this (the larger, region+patch) model either. patch_heads defaults to `heads`
        # (the region branch's head count) when not set, rather than forcing callers to specify
        # a second heads value for a branch that previously had none (SAGEConv has no heads
        # concept at all).
        self.use_edge_features = bool(use_edge_features)
        self.edge_feat_dim = int(edge_feat_dim)
        # Off by default -- current behavior unchanged (slide_head keeps its original input shape,
        # so existing checkpoints stay loadable). When on, slide_head also sees mean tumor-logit
        # and positive-fraction stats at region AND patch granularity, not just the single
        # region_logit_max scalar -- see forward()'s use_extent_features branch for the rationale.
        self.use_extent_features = bool(use_extent_features)
        resolved_patch_heads = int(patch_heads) if patch_heads is not None else int(heads)

        self.input_proj = nn.Linear(int(in_dim), self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)

        self.region_convs = nn.ModuleList()
        self.region_norms = nn.ModuleList()
        for _ in range(self.num_layers):
            self.region_convs.append(
                GATv2Conv(
                    in_channels=self.hidden_dim,
                    out_channels=max(1, self.hidden_dim // int(heads)),
                    heads=int(heads),
                    concat=True,
                    dropout=self.dropout_rate,
                    edge_dim=self.edge_feat_dim if self.use_edge_features else None,
                )
            )
            self.region_norms.append(nn.LayerNorm(self.hidden_dim))

        self.patch_in = nn.Linear(int(in_dim), self.patch_hidden_dim)
        self.patch_norm = nn.LayerNorm(self.patch_hidden_dim)
        # SAGEConv -> GATv2Conv (2026-08-11): SAGEConv has no edge-feature support at all in PyG,
        # so the patch branch could never consume patch-near-patch edge_feat regardless of what
        # was stored on the graph -- including patch_patch_edge_mode: semantic_equality's effect
        # on which patches even get connected (that topology change WAS already live for this
        # branch, just not the per-edge feature values). GATv2Conv matches the region branch's
        # existing choice/dropout/residual pattern above rather than introducing a third conv
        # type into this file.
        self.patch_convs = nn.ModuleList(
            [
                GATv2Conv(
                    in_channels=self.patch_hidden_dim,
                    out_channels=max(1, self.patch_hidden_dim // resolved_patch_heads),
                    heads=resolved_patch_heads,
                    concat=True,
                    dropout=self.dropout_rate,
                    edge_dim=self.edge_feat_dim if self.use_edge_features else None,
                )
                for _ in range(self.patch_num_layers)
            ]
        )
        self.patch_head = nn.Linear(self.patch_hidden_dim, 1)

        self.patch_to_region_proj = nn.Linear(self.patch_hidden_dim, self.hidden_dim)
        self.patch_to_region_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )
        self.region_fuse_norm = nn.LayerNorm(self.hidden_dim)

        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 1),
            )
        )

        slide_head_extra_dim = 5 if self.use_extent_features else 1
        self.slide_head = nn.Sequential(
            nn.Linear(self.hidden_dim + slide_head_extra_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        self.region_head = nn.Linear(self.hidden_dim, 1)

        # Optional, independent multiclass auxiliary classification head -- a DIFFERENT
        # supervision task on the same region/patch nodes (e.g. GSDB tissue_type: antrum/corpus/
        # intermediate/other) than the binary tumor/normal region_head/patch_head above, not a
        # multiclass replacement for it. Symmetric with region_aux_weight/patch_aux_weight's
        # existing pattern: off by default (target_field=None or class_names empty -> no head
        # built, no new params, no new compute, output dict unchanged) -- see
        # CreationConfig.label_fields (src/graph_creation_new/config.py) for how
        # data[node_type][f"{field}_y"] + data.meta[f"{field}_label_vocab"] get onto the graph in
        # the first place. class_names is a caller-declared CANONICAL, ordered class list -- {
        # field}_label_vocab is built PER-SLIDE from only that slide's own annotations (helper/
        # annotations.py's build_annotation_label_vocab), so its integer ids are NOT comparable
        # across graphs; _canonical_class_id below remaps each graph's local vocab strings onto
        # this shared class_names list at loss time instead of trusting raw ids directly.
        self.region_cls_target_field = str(region_cls_target_field).strip() if region_cls_target_field else None
        self.region_cls_class_names = list(region_cls_class_names) if region_cls_class_names else []
        self.region_cls_weight = float(region_cls_weight)
        self.region_cls_enabled = bool(self.region_cls_target_field) and bool(self.region_cls_class_names)
        if self.region_cls_enabled:
            self.region_cls_head = nn.Linear(self.hidden_dim, len(self.region_cls_class_names))
        # Optional per-class weight tensor for region_cls_loss's CrossEntropyLoss (e.g. inverse-
        # frequency weighting for an imbalanced taxonomy like GSDB tissue_type). None (default) =
        # unweighted, current behavior unchanged. Not a constructor arg: the dataset needed to
        # compute a meaningful weighting isn't available yet at model-construction time (Holder.
        # load_model() runs before load_data()) -- set post-construction instead, see
        # src/utils/training_utils.py's infer_node_multiclass_weights_from_dataset and
        # train_graph_segmentation_model's call site.
        self.region_cls_weight_tensor: Optional[torch.Tensor] = None

        self.patch_cls_target_field = str(patch_cls_target_field).strip() if patch_cls_target_field else None
        self.patch_cls_class_names = list(patch_cls_class_names) if patch_cls_class_names else []
        self.patch_cls_weight = float(patch_cls_weight)
        self.patch_cls_enabled = bool(self.patch_cls_target_field) and bool(self.patch_cls_class_names)
        if self.patch_cls_enabled:
            self.patch_cls_head = nn.Linear(self.patch_hidden_dim, len(self.patch_cls_class_names))
        self.patch_cls_weight_tensor: Optional[torch.Tensor] = None

        # Domain-adversarial auxiliary head, off by default -- see GradientReversalLayer's
        # docstring. Reads the pure GNN-pooled slide_h, before region_logit_max gets concatenated
        # onto it for slide_head, matching RegionSlideClassifier's equivalent placement.
        self.domain_adversarial = bool(domain_adversarial)
        if self.domain_adversarial:
            self.domain_grl = GradientReversalLayer(domain_adversarial_lambda)
            self.domain_classifier = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim // 2, 1),
            )

    @staticmethod
    def _is_tumor_label(label_text: str) -> bool:
        text = str(label_text).strip().lower()
        return any(token in text for token in ("tumor", "metast", "macro", "micro", "itc", "positive", "status=inflamed"))

    @staticmethod
    def _is_normal_label(label_text: str) -> bool:
        text = str(label_text).strip().lower()
        return any(token in text for token in ("normal", "negative", "benign", "healthy", "background", "non_tumor", "status=noninflamed"))

    def _map_annotation_to_binary(self, ann_ids: torch.Tensor, vocab: List[str]) -> torch.Tensor:
        mapped = torch.full_like(ann_ids, fill_value=-1)
        if ann_ids.numel() == 0:
            return mapped

        unique_ids = sorted(set(int(v) for v in ann_ids[ann_ids >= 0].tolist()))
        if not unique_ids:
            return mapped

        if vocab:
            id_to_binary: Dict[int, int] = {}
            for ann_id in unique_ids:
                if ann_id < 0 or ann_id >= len(vocab):
                    continue
                label_text = str(vocab[ann_id]).strip().lower()
                if self._is_tumor_label(label_text):
                    id_to_binary[ann_id] = 1
                elif self._is_normal_label(label_text):
                    id_to_binary[ann_id] = 0
            for ann_id, binary_label in id_to_binary.items():
                mapped[ann_ids == ann_id] = int(binary_label)
        elif set(unique_ids).issubset({0, 1}):
            mapped = ann_ids.clone()

        return mapped

    @staticmethod
    def _canonical_class_id(raw_label: str, class_names: List[str]) -> Optional[int]:
        text = str(raw_label).strip().lower()
        if "=" in text:
            text = text.rsplit("=", 1)[-1]
        for idx, name in enumerate(class_names):
            aliases = [alias.strip().lower() for alias in str(name).split("+")]
            if text in aliases:
                return idx
        return None

    def _remap_cls_targets(
        self,
        target_y: Optional[torch.Tensor],
        vocab: List[str],
        class_names: List[str],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not torch.is_tensor(target_y) or not class_names:
            return None

        local_ids = target_y.view(-1).to(device).long()
        ignore_index = -100
        canonical = torch.full_like(local_ids, fill_value=ignore_index)
        unique_local_ids = sorted(set(int(v) for v in local_ids[local_ids >= 0].tolist()))
        for local_id in unique_local_ids:
            if local_id < 0 or local_id >= len(vocab):
                continue
            mapped = self._canonical_class_id(vocab[local_id], class_names)
            if mapped is not None:
                canonical[local_ids == local_id] = mapped
        return canonical

    def _node_cls_loss(
        self,
        target_y: Optional[torch.Tensor],
        logits: torch.Tensor,
        vocab: List[str],
        class_names: List[str],
        class_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        canonical = self._remap_cls_targets(target_y, vocab, class_names, logits.device)
        if canonical is None or canonical.numel() != logits.size(0):
            return torch.zeros((), dtype=logits.dtype, device=logits.device)

        ignore_index = -100
        if torch.all(canonical == ignore_index):
            return torch.zeros((), dtype=logits.dtype, device=logits.device)

        weight = class_weight.to(device=logits.device, dtype=logits.dtype) if torch.is_tensor(class_weight) else None
        return F.cross_entropy(logits, canonical, ignore_index=ignore_index, weight=weight)

    def _node_cls_correct_total(
        self,
        target_y: Optional[torch.Tensor],
        logits: torch.Tensor,
        vocab: List[str],
        class_names: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        canonical = self._remap_cls_targets(target_y, vocab, class_names, logits.device)
        zero = torch.zeros((), dtype=torch.long, device=logits.device)
        if canonical is None or canonical.numel() != logits.size(0):
            return zero, zero

        valid = canonical != -100
        total = valid.sum()
        if int(total.item()) == 0:
            return zero, zero

        preds = logits.detach().argmax(dim=-1)
        correct = ((preds == canonical) & valid).sum()
        return correct, total

    def _node_aux_loss(
        self,
        annotation_y: Optional[torch.Tensor],
        logits: torch.Tensor,
        pos_weight_cap: float,
        vocab: List[str],
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(annotation_y):
            return torch.zeros((), dtype=logits.dtype, device=logits.device)

        ann = annotation_y.view(-1).to(logits.device).long()
        if ann.numel() != logits.numel():
            return torch.zeros((), dtype=logits.dtype, device=logits.device)

        target = self._map_annotation_to_binary(ann, vocab)
        labeled_mask = target >= 0
        if not torch.any(labeled_mask):
            return torch.zeros((), dtype=logits.dtype, device=logits.device)

        y = target[labeled_mask].float()
        z = logits[labeled_mask]

        pos = float((y >= 0.5).sum().item())
        neg = float((y < 0.5).sum().item())
        if pos <= 0.0:
            pos_weight = torch.tensor(1.0, device=logits.device)
        else:
            pos_weight = torch.tensor(
                min(float(pos_weight_cap), max(1.0, neg / pos)),
                device=logits.device,
            )

        per_sample = F.binary_cross_entropy_with_logits(z, y, pos_weight=pos_weight, reduction="none")
        if sample_weight is None:
            return per_sample.mean()

        w = sample_weight.view(-1).to(device=logits.device, dtype=per_sample.dtype)
        if w.numel() != ann.numel():
            return per_sample.mean()
        w = w[labeled_mask]
        denom = w.sum().clamp(min=1e-8)
        return (per_sample * w).sum() / denom

    def _ensure_linear_input(self, layer_name: str, feature_dim: int, out_dim: int, device: torch.device) -> None:
        layer = getattr(self, layer_name)
        if int(feature_dim) == int(layer.in_features):
            return

        setattr(self, layer_name, nn.Linear(int(feature_dim), out_dim).to(device))

    def _ensure_input_projection(self, feature_dim: int, device: torch.device) -> None:
        self._ensure_linear_input("input_proj", feature_dim, self.hidden_dim, device)

    def _ensure_patch_input_projection(self, feature_dim: int, device: torch.device) -> None:
        self._ensure_linear_input("patch_in", feature_dim, self.patch_hidden_dim, device)

    @staticmethod
    def _extract_meta_list(data, key: str) -> List[Any]:
        if not hasattr(data, "meta") or not isinstance(data.meta, dict):
            return []
        raw = data.meta.get(key)
        if not isinstance(raw, list):
            return []
        if raw and isinstance(raw[0], list):
            return raw[0]
        return raw

    def _annotation_vocab(self, data) -> List[str]:
        return [str(v) for v in self._extract_meta_list(data, "annotation_label_vocab")]

    def _run_patch_branch(self, data, region_h: torch.Tensor, vocab: List[str]):
        device = region_h.device
        zero_long = torch.zeros((), dtype=torch.long, device=device)
        no_patch_result = (
            region_h, None, torch.zeros((), dtype=region_h.dtype, device=device), None,
            None, torch.zeros((), dtype=region_h.dtype, device=device), zero_long, zero_long, None,
            None,
        )

        if not hasattr(data, "node_types") or "patch" not in data.node_types:
            return no_patch_result

        patch_store = data["patch"]
        patch_x_raw = getattr(patch_store, "x", None)
        if not torch.is_tensor(patch_x_raw) or patch_x_raw.dim() != 2 or patch_x_raw.size(0) == 0:
            return no_patch_result

        p2r_edge_index = None
        if ("patch", "in", "region") in data.edge_types:
            p2r_edge_index = data[("patch", "in", "region")].edge_index

        patch_edge_index = None
        patch_edge_attr = None
        if ("patch", "near", "patch") in data.edge_types:
            patch_edge_index = data[("patch", "near", "patch")].edge_index
            patch_edge_attr = getattr(data[("patch", "near", "patch")], "edge_feat", None)
        elif ("patch", "spatial", "patch") in data.edge_types:
            patch_edge_index = data[("patch", "spatial", "patch")].edge_index
            patch_edge_attr = getattr(data[("patch", "spatial", "patch")], "edge_feat", None)

        if p2r_edge_index is None or patch_edge_index is None:
            return no_patch_result

        patch_edge_index = patch_edge_index.long().to(device)
        p2r_edge_index = p2r_edge_index.long().to(device)
        if torch.is_tensor(patch_edge_attr):
            patch_edge_attr = patch_edge_attr.to(device)
        patch_x = patch_x_raw
        patch_annotation_y = getattr(patch_store, "annotation_y", None)
        patch_batch_for_norm = getattr(patch_store, "batch", None)
        selected_patch_idx = None

        if graph_patch_utils.should_use_region_only_forward(
            patch_store, patch_edge_index, self.max_patch_nodes_for_full_forward, self.max_patch_edges_for_full_forward
        ):
            self._patch_subsample_counter += 1
            seed = torch.initial_seed() + self._patch_subsample_counter * 1_000_003
            sampled = graph_patch_utils.subsample_large_patch_graph(
                patch_store,
                patch_edge_index,
                p2r_edge_index,
                target=self.max_patch_nodes_for_full_forward,
                patch_edge_feat=patch_edge_attr,
                seed=seed,
            )
            patch_x = sampled["patch_x"]
            patch_edge_index = sampled["patch_edge_index"]
            p2r_edge_index = sampled["p2r_edge_index"]
            patch_annotation_y = sampled["patch_annotation_y"]
            patch_batch_for_norm = sampled["patch_batch"]
            selected_patch_idx = sampled["selected_patch_idx"]
            patch_edge_attr = sampled["patch_edge_feat"]

        self._ensure_patch_input_projection(int(patch_x.size(1)), patch_x.device)

        patch_x_in = patch_x.float()
        if self.graph_instance_norm:
            patch_x_in = _graph_instance_normalize(patch_x_in, patch_batch_for_norm)

        patch_h = self.patch_in(patch_x_in)
        patch_h = self.patch_norm(patch_h)
        patch_h = F.gelu(patch_h)

        patch_edge_attr = _sanitize_edge_attr(
            patch_edge_attr, patch_edge_index, self.use_edge_features, self.edge_feat_dim, patch_h.device
        )
        patch_edge_attr = _recompute_cosine_sim_edge_attr(patch_edge_attr, patch_edge_index, patch_x_in)

        if self.patch_edge_dropout_p > 0.0 and self.training and patch_edge_index.numel() > 0:
            keep_mask = torch.rand(patch_edge_index.size(1), device=patch_h.device) >= self.patch_edge_dropout_p
            patch_edge_index = patch_edge_index[:, keep_mask]
            if patch_edge_attr is not None:
                patch_edge_attr = patch_edge_attr[keep_mask]

        for conv in self.patch_convs:
            patch_h_res = patch_h
            patch_h = conv(patch_h, patch_edge_index, edge_attr=patch_edge_attr)
            patch_h = F.gelu(patch_h)
            patch_h = F.dropout(patch_h, p=self.dropout_rate, training=self.training)
            patch_h = patch_h + patch_h_res

        p_src = p2r_edge_index[0].long()
        p_dst = p2r_edge_index[1].long()
        valid = (p_src >= 0) & (p_src < patch_h.size(0)) & (p_dst >= 0) & (p_dst < region_h.size(0))
        p_src = p_src[valid]
        p_dst = p_dst[valid]

        patch_to_region = graph_patch_utils.scatter_mean(
            patch_h[p_src] if p_src.numel() > 0 else patch_h.new_zeros((0, patch_h.size(1))),
            p_dst,
            n_out=region_h.size(0),
        )
        patch_to_region = self.patch_to_region_proj(patch_to_region)

        gate = self.patch_to_region_gate(torch.cat([region_h, patch_to_region], dim=1))
        fused_region_h = self.region_fuse_norm(region_h + gate * patch_to_region)

        patch_logits = self.patch_head(patch_h).squeeze(-1)
        patch_aux_loss = self._node_aux_loss(
            patch_annotation_y, patch_logits, self.patch_aux_pos_weight_cap, vocab
        )

        patch_cls_logits = None
        patch_cls_loss = torch.zeros((), dtype=patch_h.dtype, device=device)
        patch_cls_correct = zero_long
        patch_cls_total = zero_long
        patch_cls_canonical = None
        if self.patch_cls_enabled:
            patch_cls_logits = self.patch_cls_head(patch_h)
            patch_cls_target = getattr(patch_store, f"{self.patch_cls_target_field}_y", None)
            if selected_patch_idx is not None and torch.is_tensor(patch_cls_target):
                patch_cls_target = patch_cls_target[selected_patch_idx.to(patch_cls_target.device)]
            patch_cls_vocab = [str(v) for v in self._extract_meta_list(data, f"{self.patch_cls_target_field}_label_vocab")]
            patch_cls_loss = self._node_cls_loss(
                patch_cls_target, patch_cls_logits, patch_cls_vocab, self.patch_cls_class_names,
                class_weight=self.patch_cls_weight_tensor,
            )
            patch_cls_correct, patch_cls_total = self._node_cls_correct_total(
                patch_cls_target, patch_cls_logits, patch_cls_vocab, self.patch_cls_class_names
            )
            patch_cls_canonical = self._remap_cls_targets(
                patch_cls_target, patch_cls_vocab, self.patch_cls_class_names, patch_cls_logits.device
            )

        return (
            fused_region_h, patch_logits, patch_aux_loss, selected_patch_idx,
            patch_cls_logits, patch_cls_loss, patch_cls_correct, patch_cls_total, patch_cls_canonical,
            patch_batch_for_norm,
        )

    def _project_region_input(self, region_x: torch.Tensor, region_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._ensure_input_projection(int(region_x.size(1)), region_x.device)
        region_x_in = region_x.float()
        if self.graph_instance_norm:
            region_x_in = _graph_instance_normalize(region_x_in, region_batch)
        h = self.input_proj(region_x_in)
        h = self.input_norm(h)
        h = F.gelu(h)
        return h

    def _apply_manifold_mixup(self, h: torch.Tensor, mixup_data, lam: float) -> torch.Tensor:
        region_store_b = mixup_data["region"] if hasattr(mixup_data, "node_types") else None
        region_x_b = getattr(region_store_b, "x", None) if region_store_b is not None else None
        if not torch.is_tensor(region_x_b) or region_x_b.numel() == 0:
            return h
        region_batch_b = getattr(region_store_b, "batch", None)
        h_b = self._project_region_input(region_x_b.to(h.device), region_batch_b)
        if h_b.size(0) == 0:
            return h
        resample_idx = torch.randint(0, h_b.size(0), (h.size(0),), device=h.device)
        h_b_matched = h_b[resample_idx]
        return lam * h + (1.0 - lam) * h_b_matched

    def forward(
        self,
        x: torch.Tensor,
        mixup_data=None,
        mixup_lambda: Optional[float] = None,
    ) -> torch.Tensor:
        """Run the region/patch GNN branches and return slide/region/patch predictions and losses."""
        data = x
        if not hasattr(data, "node_types") or "region" not in data.node_types:
            raise ValueError("TissueRegionGraphModel expects HeteroData with 'region' node type.")

        region_store = data["region"]
        region_x = getattr(region_store, "x", None)
        if not torch.is_tensor(region_x) or region_x.dim() != 2:
            raise ValueError("Region node features (data['region'].x) are required.")

        region_batch_for_norm = getattr(region_store, "batch", None)
        h = self._project_region_input(region_x, region_batch_for_norm)

        # Off by default (mixup_data/mixup_lambda both None -- current behavior unchanged).
        # Opt-in via segmentation.manifold_mixup_p > 0 in the model config; see
        # src/pipelines/train.py's _train_epoch_graph_segmentation for where these get set,
        # train-only (never during validation/eval).
        if mixup_data is not None and mixup_lambda is not None:
            h = self._apply_manifold_mixup(h, mixup_data, float(mixup_lambda))

        edge_index = None
        edge_attr = None
        if hasattr(data, "edge_types") and ("region", "near", "region") in data.edge_types:
            edge_index = data[("region", "near", "region")].edge_index
            edge_attr = getattr(data[("region", "near", "region")], "edge_feat", None)
        elif hasattr(data, "edge_types") and ("region", "spatial", "region") in data.edge_types:
            edge_index = data[("region", "spatial", "region")].edge_index
            edge_attr = getattr(data[("region", "spatial", "region")], "edge_feat", None)

        if edge_index is None:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=h.device)
        else:
            edge_index = edge_index.long().to(h.device)

        edge_attr = _sanitize_edge_attr(edge_attr, edge_index, self.use_edge_features, self.edge_feat_dim, h.device)
        # Recomputed here (not reusing _project_region_input's internal region_x_in) so this
        # stays independent of manifold mixup's SECOND-graph call to that same helper above --
        # edge_attr must always reflect THIS graph's own region_x, mixed or not. Cheap
        # duplicate: same .float()/_graph_instance_normalize call _project_region_input already
        # made for h above, just also kept around for _recompute_cosine_sim_edge_attr.
        region_x_in = region_x.float()
        if self.graph_instance_norm:
            region_x_in = _graph_instance_normalize(region_x_in, region_batch_for_norm)
        edge_attr = _recompute_cosine_sim_edge_attr(edge_attr, edge_index, region_x_in.to(h.device))

        if self.edge_dropout_p > 0.0 and self.training and edge_index.numel() > 0:
            keep_mask = torch.rand(edge_index.size(1), device=h.device) >= self.edge_dropout_p
            edge_index = edge_index[:, keep_mask]
            if edge_attr is not None:
                edge_attr = edge_attr[keep_mask]

        for conv, norm in zip(self.region_convs, self.region_norms):
            h_res = h
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = norm(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
            h = h + h_res

        vocab = self._annotation_vocab(data)

        (
            h, patch_logits, patch_aux_loss, selected_patch_idx,
            patch_cls_logits, patch_cls_loss, patch_cls_correct, patch_cls_total, patch_cls_canonical,
            patch_batch,
        ) = self._run_patch_branch(data, h, vocab)

        region_logits = self.region_head(h).squeeze(-1)

        region_batch = getattr(region_store, "batch", None)
        if region_batch is None:
            region_batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        slide_h = self.pool(h, region_batch)

        domain_logits = None
        if self.domain_adversarial:
            domain_logits = self.domain_classifier(self.domain_grl(slide_h)).squeeze(-1)

        region_logit_max = scatter(region_logits, region_batch, dim=0, dim_size=slide_h.size(0), reduce='max')
        if self.use_extent_features:
            # itc/micro/macro are literally defined by tumor EXTENT (isolated cells vs. <2mm vs.
            # >2mm), but region_logit_max alone only says "is at least one region tumor-like" --
            # it can't distinguish a slide with one tumor region from one with twenty. Add how
            # MUCH of the slide looks tumor-positive: mean tumor-logit and positive-region/patch
            # fraction, at both the region and patch granularity. Opt-in (use_extent_features
            # config flag, default False) so this never changes slide_head's input shape --
            # and therefore never changes behavior or checkpoint compatibility -- for any existing
            # run (e.g. GSDB inflammation) that doesn't set it.
            region_logit_mean = scatter(region_logits, region_batch, dim=0, dim_size=slide_h.size(0), reduce='mean')
            region_positive_frac = scatter(
                (region_logits > 0).float(), region_batch, dim=0, dim_size=slide_h.size(0), reduce='mean'
            )
            if patch_logits is not None:
                patch_batch_resolved = patch_batch
                if patch_batch_resolved is None:
                    patch_batch_resolved = torch.zeros(patch_logits.size(0), dtype=torch.long, device=h.device)
                patch_logit_mean = scatter(
                    patch_logits, patch_batch_resolved, dim=0, dim_size=slide_h.size(0), reduce='mean'
                )
                patch_positive_frac = scatter(
                    (patch_logits > 0).float(), patch_batch_resolved, dim=0, dim_size=slide_h.size(0), reduce='mean'
                )
            else:
                patch_logit_mean = torch.zeros_like(region_logit_max)
                patch_positive_frac = torch.zeros_like(region_logit_max)
            extent_feats = torch.stack(
                [region_logit_max, region_logit_mean, region_positive_frac, patch_logit_mean, patch_positive_frac],
                dim=-1,
            )
        else:
            extent_feats = region_logit_max.unsqueeze(-1)
        slide_logits = self.slide_head(torch.cat([slide_h, extent_feats], dim=-1))

        if slide_logits.dim() == 2 and slide_logits.size(-1) == 1:
            slide_logits = slide_logits.squeeze(-1)

        region_sample_weight = None
        region_label_ambiguous = getattr(region_store, "label_ambiguous", None)
        if torch.is_tensor(region_label_ambiguous) and region_label_ambiguous.numel() == region_logits.numel():
            region_sample_weight = torch.where(
                region_label_ambiguous.to(device=region_logits.device).bool(),
                torch.tensor(self.mixed_region_weight, device=region_logits.device, dtype=torch.float32),
                torch.tensor(1.0, device=region_logits.device, dtype=torch.float32),
            )
        region_aux_loss = self._node_aux_loss(
            getattr(region_store, "annotation_y", None),
            region_logits,
            self.region_aux_pos_weight_cap,
            vocab,
            sample_weight=region_sample_weight,
        )

        region_cls_logits = None
        region_cls_loss = torch.zeros((), dtype=h.dtype, device=h.device)
        region_cls_correct = torch.zeros((), dtype=torch.long, device=h.device)
        region_cls_total = torch.zeros((), dtype=torch.long, device=h.device)
        region_cls_canonical = None
        if self.region_cls_enabled:
            region_cls_logits = self.region_cls_head(h)
            region_cls_vocab = [str(v) for v in self._extract_meta_list(data, f"{self.region_cls_target_field}_label_vocab")]
            region_cls_target = getattr(region_store, f"{self.region_cls_target_field}_y", None)
            region_cls_loss = self._node_cls_loss(
                region_cls_target, region_cls_logits, region_cls_vocab, self.region_cls_class_names,
                class_weight=self.region_cls_weight_tensor,
            )
            region_cls_correct, region_cls_total = self._node_cls_correct_total(
                region_cls_target, region_cls_logits, region_cls_vocab, self.region_cls_class_names,
            )
            # See patch_cls_canonical's comment in _run_patch_branch -- same rationale, used by
            # train.py for per-epoch Cohen's Kappa/macro-F1 tracking.
            region_cls_canonical = self._remap_cls_targets(
                region_cls_target, region_cls_vocab, self.region_cls_class_names, region_cls_logits.device
            )

        aux_loss = (
            (self.region_aux_weight * region_aux_loss)
            + (self.patch_aux_weight * patch_aux_loss)
            + (self.region_cls_weight * region_cls_loss)
            + (self.patch_cls_weight * patch_cls_loss)
        )

        out = {
            "logits": slide_logits,
            "aux_loss": aux_loss,
            "region_logits": region_logits,
            "region_probs": torch.sigmoid(region_logits),
            # Pooled slide-level representation BEFORE concatenation with region_logit_max and
            # BEFORE the final slide_head classifier -- the actual "learned feature" a
            # feature-space consistency loss should act on (see train.py's
            # feature_consistency_weight), as opposed to slide_logits, which already reflects a
            # specific classification decision, not the underlying representation.
            "slide_h": slide_h,
        }
        if domain_logits is not None:
            out["domain_logits"] = domain_logits
        if patch_logits is not None:
            out["patch_logits"] = patch_logits
            out["patch_probs"] = torch.sigmoid(patch_logits)
            if selected_patch_idx is not None:
                out["patch_node_idx"] = selected_patch_idx
        if region_cls_logits is not None:
            out["region_cls_logits"] = region_cls_logits
            out["region_cls_correct"] = region_cls_correct
            out["region_cls_total"] = region_cls_total
            out["region_cls_canonical"] = region_cls_canonical
        if patch_cls_logits is not None:
            out["patch_cls_logits"] = patch_cls_logits
            out["patch_cls_correct"] = patch_cls_correct
            out["patch_cls_total"] = patch_cls_total
            out["patch_cls_canonical"] = patch_cls_canonical
        return out

    @torch.no_grad()
    def predict_region_proba(self, data) -> torch.Tensor:
        """Return per-region tumor probabilities for a slide."""
        out = self.forward(data)
        return out["region_probs"]
