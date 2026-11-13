from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GlobalAttention
from torch_geometric.utils import scatter

from src.core.registry_utils import register_model
from src.tissue_region_extraction.model import (
    _graph_instance_normalize,
    _recompute_cosine_sim_edge_attr,
    _sanitize_edge_attr,
    GradientReversalLayer,
)


@register_model(("gnn", "region_slide_net", "graph"))
class RegionSlideClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 1,
        dropout_rate: float = 0.4,
        in_dim: int = 1536,
        hidden_dim: int = 96,
        num_layers: int = 2,
        heads: int = 4,
        edge_dropout_p: float = 0.1,
        graph_instance_norm: bool = True,
        use_patch_evidence: bool = True,
        patch_evidence_hidden: int = 32,
        use_edge_features: bool = True,
        edge_feat_dim: int = 8,
        domain_adversarial: bool = False,
        domain_adversarial_lambda: float = 1.0,
    ) -> None:
        """Initialize the region-graph slide classifier and its optional auxiliary heads."""
        super().__init__()

        self.model_name = str(model_name)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = max(1, int(num_layers))
        self.dropout_rate = float(dropout_rate)
        self.edge_dropout_p = float(edge_dropout_p)
        self.graph_instance_norm = bool(graph_instance_norm)

        self.use_patch_evidence = bool(use_patch_evidence)
        if self.use_patch_evidence:
            self.patch_evidence_scorer = nn.Sequential(
                nn.Linear(int(in_dim), int(patch_evidence_hidden)),
                nn.GELU(),
                nn.Linear(int(patch_evidence_hidden), 1),
            )

        self.use_edge_features = bool(use_edge_features)
        self.edge_feat_dim = int(edge_feat_dim)

        self.input_proj = nn.Linear(int(in_dim), self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=self.hidden_dim,
                    out_channels=max(1, self.hidden_dim // int(heads)),
                    heads=int(heads),
                    concat=True,
                    dropout=self.dropout_rate,
                    edge_dim=self.edge_feat_dim if self.use_edge_features else None,
                )
            )
            self.norms.append(nn.LayerNorm(self.hidden_dim))

        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 1),
            )
        )

        classifier_in_dim = self.hidden_dim + (1 if self.use_patch_evidence else 0)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, self.num_classes),
        )

        self.domain_adversarial = bool(domain_adversarial)
        if self.domain_adversarial:
            self.domain_grl = GradientReversalLayer(domain_adversarial_lambda)
            self.domain_classifier = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim // 2, 1),
            )

    def forward(self, data):
        """Compute slide-level logits (and optional domain logits) from a region-level graph."""
        if not hasattr(data, "node_types") or "region" not in data.node_types:
            raise ValueError("RegionSlideClassifier expects HeteroData with 'region' node type.")

        region_store = data["region"]
        region_x = getattr(region_store, "x", None)
        if not torch.is_tensor(region_x) or region_x.dim() != 2:
            raise ValueError("Region node features (data['region'].x) are required.")

        region_batch = getattr(region_store, "batch", None)

        x = region_x.float()
        if self.graph_instance_norm:
            x = _graph_instance_normalize(x, region_batch)

        h = self.input_proj(x)
        h = self.input_norm(h)
        h = F.gelu(h)

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
        edge_attr = _recompute_cosine_sim_edge_attr(edge_attr, edge_index, x)

        if self.edge_dropout_p > 0.0 and self.training and edge_index.numel() > 0:
            keep_mask = torch.rand(edge_index.size(1), device=h.device) >= self.edge_dropout_p
            edge_index = edge_index[:, keep_mask]
            if edge_attr is not None:
                edge_attr = edge_attr[keep_mask]

        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = norm(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
            h = h + h_res

        if region_batch is None:
            region_batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        slide_h = self.pool(h, region_batch)

        domain_logits = None
        if self.domain_adversarial:
            domain_logits = self.domain_classifier(self.domain_grl(slide_h)).squeeze(-1)

        if self.use_patch_evidence:
            slide_evidence = self._patch_evidence(data, h, region_batch, slide_h.size(0))
            slide_h = torch.cat([slide_h, slide_evidence.unsqueeze(-1)], dim=-1)

        logits = self.classifier(slide_h).squeeze(-1)

        if domain_logits is None:
            return logits
        return {"logits": logits, "domain_logits": domain_logits}

    def _patch_evidence(
        self,
        data,
        region_h: torch.Tensor,
        region_batch: torch.Tensor,
        num_slides: int,
    ) -> torch.Tensor:
        device = region_h.device
        zeros = torch.zeros(num_slides, device=device)

        if not hasattr(data, "node_types") or "patch" not in data.node_types:
            return zeros
        patch_store = data["patch"]
        patch_x = getattr(patch_store, "x", None)
        if not torch.is_tensor(patch_x) or patch_x.dim() != 2 or patch_x.size(0) == 0:
            return zeros

        p2r_edge_index = None
        if hasattr(data, "edge_types") and ("patch", "in", "region") in data.edge_types:
            p2r_edge_index = data[("patch", "in", "region")].edge_index
        if p2r_edge_index is None or p2r_edge_index.numel() == 0:
            return zeros

        patch_scores = self.patch_evidence_scorer(patch_x.float().to(device)).squeeze(-1)

        p_src = p2r_edge_index[0].long().to(device)
        p_dst = p2r_edge_index[1].long().to(device)
        valid = (p_src >= 0) & (p_src < patch_scores.size(0)) & (p_dst >= 0) & (p_dst < region_h.size(0))
        p_src = p_src[valid]
        p_dst = p_dst[valid]
        if p_src.numel() == 0:
            return zeros

        region_evidence = scatter(
            patch_scores[p_src], p_dst, dim=0, dim_size=region_h.size(0), reduce="max"
        )
        slide_evidence = scatter(
            region_evidence, region_batch, dim=0, dim_size=num_slides, reduce="max"
        )
        return slide_evidence
