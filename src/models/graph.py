import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional

from torch_geometric.nn import (
    SAGEConv,
    GATv2Conv,
    Set2Set,
    JumpingKnowledge,
    GlobalAttention,
    HeteroConv,
    global_mean_pool,
)
from torch_geometric.nn.aggr import AttentionalAggregation

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.core.registry_utils import register_model
from src.utils import graph_patch_utils

from torch_geometric.nn import GCNConv, global_mean_pool


def _format_graph_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 0:
        return logits.unsqueeze(0)
    if logits.dim() == 1:
        return logits
    if logits.dim() == 2 and logits.size(-1) == 1:
        return logits.squeeze(-1)
    return logits

@register_model(('gnn', 'simple_gcn', 'graph'))
class SimpleGCN(nn.Module):
    def __init__(
        self, 
        model_name: str,
        num_classes: int = 1,
        dropout_rate: float = 0.2,
        in_dim: int = None,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        """Initialize the SimpleGCN model."""
        super().__init__()
        self._in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.input_proj = nn.Linear(in_dim, hidden_dim) if in_dim is not None else None
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self._convs_built = False
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _safe_edge_weight(self, edge_store, edge_index, device: torch.device):
        edge_attr = getattr(edge_store, "edge_attr", None)
        if edge_attr is None or not torch.is_tensor(edge_attr):
            return None

        if edge_attr.dim() == 2 and edge_attr.size(1) >= 1:
            edge_weight = edge_attr[:, 0]
        elif edge_attr.dim() == 1:
            edge_weight = edge_attr
        else:
            return None

        edge_weight = edge_weight.to(device=device, dtype=torch.float32)
        if edge_weight.numel() != int(edge_index.size(1)):
            return None
        return edge_weight

    def _build_layers(self, device: torch.device):
        if self._convs_built:
            return
        n_layers = max(int(self.num_layers), 2)
        self.convs = nn.ModuleList([
            GCNConv(self.hidden_dim, self.hidden_dim).to(device) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim).to(device) for _ in range(n_layers)
        ])
        self._convs_built = True

    def forward(self, data):
        """Run the GCN forward pass over homogeneous or patch/cell graph input."""
        if hasattr(data, 'x') and data.x is not None:
            x = data.x
            edge_index = data.edge_index
            batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            edge_weight = self._safe_edge_weight(data, edge_index, x.device)
        elif hasattr(data, 'node_types') and ("patch" in data.node_types or "cell" in data.node_types):
            use_region = (
                "region" in data.node_types
                and getattr(data["region"], "x", None) is not None
                and ("region", "near", "region") in data.edge_types
            )

            if use_region:
                x = data["region"].x
                edge_store = data["region", "near", "region"]
                edge_index = edge_store.edge_index
                batch = getattr(data["region"], "batch", None)
            else:
                if "patch" in data.node_types and getattr(data["patch"], "x", None) is not None:
                    x = data["patch"].x
                    if ("patch", "near", "patch") in data.edge_types:
                        edge_store = data["patch", "near", "patch"]
                        edge_index = edge_store.edge_index
                    elif ("patch", "spatial", "patch") in data.edge_types:
                        edge_store = data["patch", "spatial", "patch"]
                        edge_index = edge_store.edge_index
                    else:
                        raise ValueError("No suitable patch-patch edge_index found in HeteroData.")
                    batch = getattr(data["patch"], "batch", None)
                elif "cell" in data.node_types and getattr(data["cell"], "x", None) is not None:
                    x = data["cell"].x
                    if ("cell", "near", "cell") in data.edge_types:
                        edge_store = data["cell", "near", "cell"]
                        edge_index = edge_store.edge_index
                    else:
                        raise ValueError("No suitable cell-cell edge_index found in HeteroData.")
                    batch = getattr(data["cell"], "batch", None)
                else:
                    raise ValueError("No suitable patch/cell node features found in HeteroData.")

            edge_weight = self._safe_edge_weight(edge_store, edge_index, x.device)

            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        else:
            raise ValueError("Input data must have either .x or a 'patch'/'cell' node type with .x.")

        if self.input_proj is None:
            self.input_proj = nn.Linear(x.shape[1], self.hidden_dim).to(x.device)
        elif x.shape[1] != self.input_proj.in_features:
            raise RuntimeError(
                f"Input feature dim {x.shape[1]} does not match model in_dim {self.input_proj.in_features}. "
                "Please check your pipeline and model config."
            )

        self._build_layers(x.device)

        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = x + x_res

        x = global_mean_pool(x, batch)
        out = self.classifier(x)
        return _format_graph_logits(out)

@register_model(('gnn', 'wsigraph', 'graph'))
@register_model(('gnn', 'wsigraph-pretraining', 'graph'))
class WSIGraphModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 1,
        dropout_rate: float = 0.2,
        in_dim: int = 1536,
        hidden_dim: int = 256,
        num_layers: int = 2,
        max_patch_nodes_for_full_forward: int = 10000,
        max_patch_edges_for_full_forward: int = 200000,
        pretraining_consistency_weight: float = 0.25,
        region_only_warn_every: int = 25,
        region_aux_weight: float = 0.08,
        patch_aux_weight: float = 0.04,
        max_patch_aux_nodes: int = 4096,
        micro_topk_ratio: float = 0.08,
        micro_logit_weight: float = 0.35,
        edge_gate_strength: float = 0.20,
        min_patches_per_region_in_large_graph: int = 1,
        patch_edge_dropout_p: float = 0.0,
        region_edge_dropout_p: float = 0.0,
        sanitize_non_finite: bool = True,
        logit_clamp_value: float = 30.0,
    ):
        """Initialize the WSIGraphModel patch/region hierarchical GNN."""
        super().__init__()

        self.model_name = str(model_name)
        self.is_pretraining = "pretraining" in self.model_name
        self.num_classes = int(num_classes)
        self.dropout_rate = float(dropout_rate)

        self.patch_in = nn.Linear(in_dim, hidden_dim)
        self.region_in = nn.Linear(in_dim, hidden_dim)

        depth = max(1, int(num_layers))
        self.patch_convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(depth)])
        self.region_convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(depth)])

        self.region_norm = nn.LayerNorm(hidden_dim)
        self.patch_to_region_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim // 2, self.num_classes),
        )

        edge_gate_hidden = max(16, hidden_dim // 4)
        self.patch_edge_gate = nn.Sequential(
            nn.Linear(2, edge_gate_hidden),
            nn.GELU(),
            nn.Linear(edge_gate_hidden, 1),
        )
        self.region_edge_gate = nn.Sequential(
            nn.Linear(2, edge_gate_hidden),
            nn.GELU(),
            nn.Linear(edge_gate_hidden, 1),
        )

        self.region_head = nn.Linear(hidden_dim, 1)
        self.patch_head = nn.Linear(hidden_dim, 1)
        self.region_aux_weight = float(region_aux_weight)
        self.patch_aux_weight = float(patch_aux_weight)
        self.max_patch_aux_nodes = int(max_patch_aux_nodes)
        self.max_patch_nodes_for_full_forward = int(max_patch_nodes_for_full_forward)
        self.max_patch_edges_for_full_forward = int(max_patch_edges_for_full_forward)
        self.pretraining_consistency_weight = float(pretraining_consistency_weight)
        self.region_only_warn_every = max(1, int(region_only_warn_every))
        self.micro_topk_ratio = float(max(0.01, min(1.0, micro_topk_ratio)))
        self.micro_logit_weight = float(micro_logit_weight)
        self.edge_gate_strength = float(edge_gate_strength)
        self.min_patches_per_region_in_large_graph = max(1, int(min_patches_per_region_in_large_graph))
        self.patch_edge_dropout_p = float(max(0.0, min(0.8, patch_edge_dropout_p)))
        self.region_edge_dropout_p = float(max(0.0, min(0.8, region_edge_dropout_p)))
        self.sanitize_non_finite = bool(sanitize_non_finite)
        self.logit_clamp_value = max(1.0, float(logit_clamp_value))
        self._region_only_counter = 0

        self._logger = logging.getLogger(__name__)

    def _sanitize_activation(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.sanitize_non_finite or not torch.is_tensor(tensor):
            return tensor
        if torch.all(torch.isfinite(tensor)):
            return tensor
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    def _sanitize_logits(self, tensor: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(tensor):
            return tensor
        clamp = self.logit_clamp_value
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=clamp, neginf=-clamp)
        return tensor.clamp(min=-clamp, max=clamp)

    def _maybe_dropout_edges(self, edge_index: torch.Tensor, p: float) -> torch.Tensor:
        if not torch.is_tensor(edge_index) or edge_index.numel() == 0:
            return edge_index
        if not self.training or p <= 0.0:
            return edge_index

        num_edges = int(edge_index.size(1))
        keep_prob = 1.0 - float(p)
        keep_mask = torch.rand((num_edges,), device=edge_index.device) < keep_prob
        if not torch.any(keep_mask):
            keep_mask[torch.randint(0, num_edges, (1,), device=edge_index.device)] = True
        return edge_index[:, keep_mask]

    def pop_region_only_fallback_count(self) -> int:
        """Return and reset region-only fallback counter for monitoring."""
        count = int(self._region_only_counter)
        self._region_only_counter = 0
        return count

    @staticmethod
    def _edge_index_for(data, relation_options):
        return graph_patch_utils.edge_index_for(data, relation_options)

    @staticmethod
    def _edge_store_and_index_for(data, relation_options):
        return graph_patch_utils.edge_store_and_index_for(data, relation_options)

    @staticmethod
    def _batch_for(store, x: torch.Tensor) -> torch.Tensor:
        return graph_patch_utils.batch_for(store, x)

    @staticmethod
    def _scatter_mean(values: torch.Tensor, index: torch.Tensor, n_out: int) -> torch.Tensor:
        return graph_patch_utils.scatter_mean(values, index, n_out)

    def _edge_features_for_gate(self, edge_store, edge_index: torch.Tensor, device: torch.device) -> torch.Tensor:
        edge_count = int(edge_index.size(1)) if torch.is_tensor(edge_index) else 0
        if edge_count <= 0:
            return torch.zeros((0, 2), dtype=torch.float32, device=device)

        edge_feat = getattr(edge_store, 'edge_feat', None)
        if torch.is_tensor(edge_feat):
            feat = edge_feat.to(device=device, dtype=torch.float32)
            if feat.dim() == 1:
                feat = feat.unsqueeze(-1)
            if feat.size(1) >= 5:
                return self._sanitize_activation(torch.stack([feat[:, 0], feat[:, 4]], dim=1))
            if feat.size(1) >= 2:
                return self._sanitize_activation(feat[:, :2])
            return self._sanitize_activation(torch.cat([feat[:, :1], torch.zeros((feat.size(0), 1), dtype=feat.dtype, device=device)], dim=1))

        edge_attr = getattr(edge_store, 'edge_attr', None)
        if torch.is_tensor(edge_attr):
            attr = edge_attr.to(device=device, dtype=torch.float32)
            if attr.dim() == 1:
                attr = attr.unsqueeze(-1)
            return self._sanitize_activation(torch.cat([attr[:, :1], torch.zeros((attr.size(0), 1), dtype=attr.dtype, device=device)], dim=1))

        return self._sanitize_activation(torch.zeros((edge_count, 2), dtype=torch.float32, device=device))

    def _edge_gate_from_store(
        self,
        edge_store,
        edge_index: torch.Tensor,
        num_nodes: int,
        device: torch.device,
        gate_net: nn.Module,
    ) -> torch.Tensor:
        if edge_store is None or edge_index is None or num_nodes <= 0:
            return torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)
        if not torch.is_tensor(edge_index) or edge_index.numel() == 0:
            return torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)

        edge_features = self._edge_features_for_gate(edge_store, edge_index, device)
        if edge_features.size(0) == 0:
            return torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)

        dst = edge_index[1].long()
        valid = (dst >= 0) & (dst < num_nodes)
        if not torch.any(valid):
            return torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)

        dst = dst[valid]
        edge_features = edge_features[valid]
        edge_context = self._scatter_mean(edge_features, dst, n_out=num_nodes)
        gate = torch.sigmoid(gate_net(edge_context))
        return self._sanitize_activation((2.0 * gate) - 1.0)

    def _infer_patch_batch(
        self,
        patch_store,
        region_batch: torch.Tensor,
        p2r_edge_index: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        return graph_patch_utils.infer_patch_batch(patch_store, region_batch, p2r_edge_index, device)

    def _topk_graph_score(self, values: torch.Tensor, batch: torch.Tensor, ratio: float) -> torch.Tensor:
        values = self._sanitize_activation(values)
        if values.numel() == 0:
            n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            return torch.zeros((n_graphs,), dtype=values.dtype, device=values.device)

        n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        scores = []
        for graph_idx in range(n_graphs):
            graph_mask = batch == graph_idx
            graph_vals = values[graph_mask]
            if graph_vals.numel() == 0:
                scores.append(values.new_zeros(()))
                continue
            k = max(1, int(round(float(graph_vals.numel()) * float(ratio))))
            k = min(k, int(graph_vals.numel()))
            topk_vals = torch.topk(graph_vals, k=k, largest=True).values
            scores.append(topk_vals.mean())
        return torch.stack(scores, dim=0)

    def _node_aux_bce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets is None or not torch.is_tensor(targets):
            return torch.zeros((), dtype=logits.dtype, device=logits.device)

        y = targets.view(-1).to(logits.device)
        if y.numel() != logits.numel():
            return torch.zeros((), dtype=logits.dtype, device=logits.device)

        valid = (y >= 0) & torch.isfinite(y) & torch.isfinite(logits)
        if not torch.any(valid):
            return torch.zeros((), dtype=logits.dtype, device=logits.device)

        y = y[valid].float()
        z = logits[valid]
        loss = F.binary_cross_entropy_with_logits(z, y)
        if not torch.isfinite(loss):
            return torch.zeros((), dtype=logits.dtype, device=logits.device)
        return loss

    def _region_only_forward(self, region_store, region_edge_index):
        region_h = self._sanitize_activation(F.relu(self.region_in(region_store.x)))

        region_gate = self._edge_gate_from_store(
            edge_store=region_store,
            edge_index=region_edge_index,
            num_nodes=region_h.size(0),
            device=region_h.device,
            gate_net=self.region_edge_gate,
        )

        for conv in self.region_convs:
            region_h = conv(region_h, region_edge_index)
            region_h = region_h * (1.0 + self.edge_gate_strength * region_gate)
            region_h = F.relu(region_h)
            region_h = F.dropout(region_h, p=self.dropout_rate, training=self.training)
            region_h = self._sanitize_activation(region_h)

        region_batch = self._batch_for(region_store, region_h)
        graph_emb = self.pool(region_h, region_batch)
        logits = self._sanitize_logits(_format_graph_logits(self.classifier(graph_emb)))
        region_logits = self._sanitize_logits(self.region_head(region_h).view(-1))

        if self.num_classes == 1 and self.micro_logit_weight != 0.0:
            micro_region = self._topk_graph_score(region_logits, region_batch, self.micro_topk_ratio)
            logits = self._sanitize_logits(logits + (self.micro_logit_weight * micro_region))

        aux_loss = torch.zeros((), dtype=graph_emb.dtype, device=graph_emb.device)
        if self.training:
            aux_loss = self.region_aux_weight * self._node_aux_bce(
                region_logits,
                getattr(region_store, 'annotation_y', None),
            )

        return self._sanitize_logits(logits), torch.nan_to_num(aux_loss, nan=0.0, posinf=0.0, neginf=0.0)

    def _patch_region_consistency_loss(
        self,
        patch_embeddings: torch.Tensor,
        region_embeddings: torch.Tensor,
        p2r_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_tensor(p2r_edge_index) or p2r_edge_index.numel() == 0:
            return torch.zeros((), dtype=region_embeddings.dtype, device=region_embeddings.device)

        p_src = p2r_edge_index[0].long()
        p_dst = p2r_edge_index[1].long()
        valid = (
            (p_src >= 0)
            & (p_src < patch_embeddings.size(0))
            & (p_dst >= 0)
            & (p_dst < region_embeddings.size(0))
        )
        p_src = p_src[valid]
        p_dst = p_dst[valid]
        if p_src.numel() == 0:
            return torch.zeros((), dtype=region_embeddings.dtype, device=region_embeddings.device)

        patch_embeddings = self._sanitize_activation(patch_embeddings)
        region_embeddings = self._sanitize_activation(region_embeddings)

        patch_to_region = self._scatter_mean(
            values=patch_embeddings[p_src],
            index=p_dst,
            n_out=region_embeddings.size(0),
        )

        counts = torch.zeros((region_embeddings.size(0),), dtype=region_embeddings.dtype, device=region_embeddings.device)
        counts.index_add_(0, p_dst, torch.ones((p_dst.numel(),), dtype=region_embeddings.dtype, device=region_embeddings.device))
        valid_regions = counts > 0
        if not torch.any(valid_regions):
            return torch.zeros((), dtype=region_embeddings.dtype, device=region_embeddings.device)

        region_norm = F.normalize(region_embeddings[valid_regions], dim=1)
        patch_norm = F.normalize(patch_to_region[valid_regions], dim=1)
        cosine = torch.sum(region_norm * patch_norm, dim=1)
        cosine = self._sanitize_activation(cosine)
        loss = (1.0 - cosine).mean()
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

    def _use_region_only_forward(self, patch_store, patch_edge_index):
        return graph_patch_utils.should_use_region_only_forward(
            patch_store,
            patch_edge_index,
            self.max_patch_nodes_for_full_forward,
            self.max_patch_edges_for_full_forward,
        )

    def _subsample_large_patch_graph(self, patch_store, patch_edge_index: torch.Tensor, p2r_edge_index: torch.Tensor):
        seed = torch.initial_seed() + self._region_only_counter * 1_000_003
        return graph_patch_utils.subsample_large_patch_graph(
            patch_store,
            patch_edge_index,
            p2r_edge_index,
            target=self.max_patch_nodes_for_full_forward,
            min_patches_per_region_in_large_graph=self.min_patches_per_region_in_large_graph,
            seed=seed,
        )

    def forward(self, data):
        """Run the WSIGraphModel forward pass over homogeneous or patch/region hetero graph input."""
        if hasattr(data, 'x') and data.x is not None and not hasattr(data, 'node_types'):
            x = self._sanitize_activation(F.relu(self.region_in(data.x)))
            edge_index = data.edge_index
            for conv in self.region_convs:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
                x = self._sanitize_activation(x)

            batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            pooled = self.pool(x, batch)
            logits = self.classifier(pooled)
            return self._sanitize_logits(_format_graph_logits(logits))

        if not hasattr(data, 'node_types') or 'patch' not in data.node_types or 'region' not in data.node_types:
            raise ValueError("WSIGraphModel expects HeteroData with at least 'patch' and 'region' nodes.")

        patch_store = data['patch']
        region_store = data['region']

        patch_edge_store, patch_edge_index = self._edge_store_and_index_for(data, [('patch', 'near', 'patch'), ('patch', 'spatial', 'patch')])
        region_edge_store, region_edge_index = self._edge_store_and_index_for(data, [('region', 'near', 'region')])
        p2r_edge_index = self._edge_index_for(data, [('patch', 'in', 'region')])

        if patch_edge_index is None:
            raise ValueError("Missing patch neighborhood edges ('patch','near'/'spatial','patch').")
        if region_edge_index is None:
            raise ValueError("Missing region neighborhood edges ('region','near','region').")
        if p2r_edge_index is None:
            raise ValueError("Missing patch->region assignment edges ('patch','in','region').")

        large_graph_mode = self._use_region_only_forward(patch_store, patch_edge_index)
        if large_graph_mode:
            self._region_only_counter += 1
            if self._region_only_counter <= 3 or (self._region_only_counter % self.region_only_warn_every == 0):
                self._logger.info(
                    "WSIGraphModel large-graph patch subsampling triggered %d time(s) | patch_nodes=%d patch_edges=%d",
                    self._region_only_counter,
                    int(getattr(patch_store, 'num_nodes', 0) or (patch_store.x.size(0) if hasattr(patch_store, 'x') else 0)),
                    int(patch_edge_index.size(1)) if torch.is_tensor(patch_edge_index) else 0,
                )
            sampled = self._subsample_large_patch_graph(patch_store, patch_edge_index, p2r_edge_index)
            patch_x = sampled['patch_x']
            patch_edge_index = sampled['patch_edge_index']
            p2r_edge_index = sampled['p2r_edge_index']
            patch_annotation_y = sampled['patch_annotation_y']
            patch_batch_attr = sampled['patch_batch']
            patch_edge_store = None
        else:
            patch_x = patch_store.x
            patch_annotation_y = getattr(patch_store, 'annotation_y', None)
            patch_batch_attr = getattr(patch_store, 'batch', None)

        patch_x = self._sanitize_activation(patch_x)
        patch_h = self._sanitize_activation(F.relu(self.patch_in(patch_x)))
        region_h = self._sanitize_activation(F.relu(self.region_in(region_store.x)))

        patch_gate = self._edge_gate_from_store(
            edge_store=patch_edge_store,
            edge_index=patch_edge_index,
            num_nodes=patch_h.size(0),
            device=patch_h.device,
            gate_net=self.patch_edge_gate,
        )
        region_gate = self._edge_gate_from_store(
            edge_store=region_edge_store,
            edge_index=region_edge_index,
            num_nodes=region_h.size(0),
            device=region_h.device,
            gate_net=self.region_edge_gate,
        )

        patch_edge_index_conv = self._maybe_dropout_edges(patch_edge_index, self.patch_edge_dropout_p)
        region_edge_index_conv = self._maybe_dropout_edges(region_edge_index, self.region_edge_dropout_p)

        for conv in self.patch_convs:
            patch_h = conv(patch_h, patch_edge_index_conv)
            patch_h = patch_h * (1.0 + self.edge_gate_strength * patch_gate)
            patch_h = F.relu(patch_h)
            patch_h = F.dropout(patch_h, p=self.dropout_rate, training=self.training)
            patch_h = self._sanitize_activation(patch_h)

        for conv in self.region_convs:
            region_h = conv(region_h, region_edge_index_conv)
            region_h = region_h * (1.0 + self.edge_gate_strength * region_gate)
            region_h = F.relu(region_h)
            region_h = F.dropout(region_h, p=self.dropout_rate, training=self.training)
            region_h = self._sanitize_activation(region_h)

        p_src = p2r_edge_index[0].long()
        p_dst = p2r_edge_index[1].long()
        valid = (
            (p_src >= 0)
            & (p_src < patch_h.size(0))
            & (p_dst >= 0)
            & (p_dst < region_h.size(0))
        )
        p_src = p_src[valid]
        p_dst = p_dst[valid]

        patch_to_region = self._scatter_mean(
            values=patch_h[p_src] if p_src.numel() > 0 else patch_h.new_zeros((0, patch_h.size(1))),
            index=p_dst if p_dst.numel() > 0 else p_dst,
            n_out=region_h.size(0),
        )
        patch_to_region = self._sanitize_activation(patch_to_region)

        gate = self.patch_to_region_gate(torch.cat([region_h, patch_to_region], dim=1))
        fused_region = self._sanitize_activation(self.region_norm(region_h + gate * patch_to_region))

        region_batch = self._batch_for(region_store, fused_region)
        if torch.is_tensor(patch_batch_attr):
            patch_batch = patch_batch_attr.view(-1)
        else:
            patch_batch = self._infer_patch_batch(patch_store, region_batch, p2r_edge_index, patch_h.device)
        region_logits = self._sanitize_logits(self.region_head(fused_region).view(-1))
        patch_logits_full = self._sanitize_logits(self.patch_head(patch_h).view(-1))

        graph_emb = self.pool(fused_region, region_batch)
        logits = self._sanitize_logits(_format_graph_logits(self.classifier(graph_emb)))

        if self.num_classes == 1 and self.micro_logit_weight != 0.0:
            micro_region = self._topk_graph_score(region_logits, region_batch, self.micro_topk_ratio)
            micro_patch = self._topk_graph_score(patch_logits_full, patch_batch, self.micro_topk_ratio)
            micro_score = 0.5 * (micro_region + micro_patch)
            logits = self._sanitize_logits(logits + (self.micro_logit_weight * micro_score))

        aux_loss = torch.zeros((), dtype=graph_emb.dtype, device=graph_emb.device)
        aux_loss = aux_loss + self.region_aux_weight * self._node_aux_bce(
            region_logits,
            getattr(region_store, 'annotation_y', None),
        )

        if patch_h.size(0) > self.max_patch_aux_nodes and self.training:
            perm = torch.randperm(patch_h.size(0), device=patch_h.device)[: self.max_patch_aux_nodes]
            patch_aux_feat = patch_h[perm]
            patch_aux_target = patch_annotation_y
            if torch.is_tensor(patch_aux_target):
                patch_aux_target = patch_aux_target.view(-1)[perm]
        else:
            patch_aux_feat = patch_h
            patch_aux_target = patch_annotation_y

        patch_logits = self._sanitize_logits(self.patch_head(patch_aux_feat).view(-1))
        aux_loss = aux_loss + self.patch_aux_weight * self._node_aux_bce(patch_logits, patch_aux_target)

        if self.is_pretraining and self.training and self.pretraining_consistency_weight > 0.0:
            consistency_loss = self._patch_region_consistency_loss(
                patch_embeddings=patch_h,
                region_embeddings=fused_region,
                p2r_edge_index=p2r_edge_index,
            )
            aux_loss = aux_loss + (self.pretraining_consistency_weight * consistency_loss)

        return {
            'logits': self._sanitize_logits(logits),
            'aux_loss': torch.nan_to_num(aux_loss, nan=0.0, posinf=0.0, neginf=0.0),
        }


@register_model(('gnn', 'graph_v2', 'graph'))
class WSIGraphModelV2(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 1,
        dropout_rate: float = 0.2,
        in_dim: int = 1537,
        hidden_dim: int = 256,
        num_layers: int = 4,
        heads: int = 4,
    ):
        """Initialize the WSIGraphModelV2 GATv2-based GNN."""
        super().__init__()

        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(
            GATv2Conv(
                in_dim,
                hidden_dim // heads,
                heads=heads,
                dropout=dropout_rate,
            )
        )
        self.norms.append(nn.LayerNorm(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=dropout_rate,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.jk = JumpingKnowledge(mode="cat")

        self.pool = Set2Set(
            in_channels=hidden_dim * num_layers,
            processing_steps=3,
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim * num_layers, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        """Run the WSIGraphModelV2 GATv2 forward pass."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        layer_outputs = []
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)

            if residual.shape == x.shape:
                x = x + residual

            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            layer_outputs.append(x)

        x = self.jk(layer_outputs)
        x = self.pool(x, batch)
        logits = self.classifier(x)
        return _format_graph_logits(logits)


@register_model(('gnn', 'hetero_wsi', 'graph'))
class WSIHeteroGraphModel(nn.Module):
    """
    Hierarchical Heterogeneous GNN for WSI classification.

    Architecture:
    1. Per-node-type input projection (LazyLinear -> handles any feature dim)
    2. N layers of HeteroConv:
       - patch<->patch and region<->region: GATv2Conv (attention over spatial neighbors,
         uses edge_attr distance weights)
       - patch<->region (bipartite): SAGEConv (aggregation across hierarchy levels)
       - region<->slide (bipartite): SAGEConv
    3. Residual connections + LayerNorm per layer per node type
    4. Hierarchical bottom-up pooling:
       - patch level: GlobalAttention (learned importance per patch)
       - region level: GlobalAttention (learned importance per region)
       - slide level: identity (single node per graph)
    5. Gated cross-level fusion: patch_emb, region_emb, slide_emb fused via
       a learnable gate instead of naive mean
    6. MLP classifier head
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 1,
        dropout_rate: float = 0.2,
        in_dim: int = 1537,
        hidden_dim: int = 256,
        num_layers: int = 3,
        heads: int = 4,
        use_edge_features: bool = True,
        learnable_edge_features: bool = False,
        use_hierarchical_edge_features: bool = False,
        edge_semantic_weight: float = 0.25,
    ):
        """Initialize the WSIHeteroGraphModel hierarchical heterogeneous GNN."""
        super().__init__()
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        self.node_types = ("patch", "region", "slide")
        self.in_dim = in_dim
        self.use_edge_features = bool(use_edge_features)
        self.learnable_edge_features = bool(learnable_edge_features)
        self.use_hierarchical_edge_features = bool(use_hierarchical_edge_features)
        self.edge_semantic_weight = float(edge_semantic_weight)


        self.input_proj = nn.ModuleDict({
            nt: nn.Sequential(nn.LazyLinear(hidden_dim), nn.LayerNorm(hidden_dim))
            for nt in self.node_types
        })


        self.convs = nn.ModuleList()
        self.norms = nn.ModuleDict()


        gat_out = hidden_dim // heads

        for layer_idx in range(num_layers):
            in_dim = hidden_dim
            conv = HeteroConv(
                {

                    ("patch", "near", "patch"): GATv2Conv(
                        (in_dim, in_dim), gat_out, heads=heads,
                        dropout=dropout_rate, add_self_loops=False, edge_dim=1
                    ),
                    ("region", "near", "region"): GATv2Conv(
                        (in_dim, in_dim), gat_out, heads=heads,
                        dropout=dropout_rate, add_self_loops=False, edge_dim=1
                    ),

                    ("patch", "in", "region"): SAGEConv((in_dim, in_dim), hidden_dim),
                    ("region", "has", "patch"): SAGEConv((in_dim, in_dim), hidden_dim),
                    ("region", "part_of", "slide"): SAGEConv((in_dim, in_dim), hidden_dim),
                    ("slide", "contains", "region"): SAGEConv((in_dim, in_dim), hidden_dim),
                },
                aggr="mean",
            )
            self.convs.append(conv)

            for nt in self.node_types:
                self.norms[f"{nt}_{layer_idx}"] = nn.LayerNorm(hidden_dim)


        gate_nn = lambda: nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1)
        )
        self.patch_pool = AttentionalAggregation(gate_nn())
        self.region_pool = AttentionalAggregation(gate_nn())




        self.fusion_gate = nn.Sequential(
            nn.Linear(3 * hidden_dim, 3),
        )
        self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)


        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 4, num_classes),
        )





    def _batch(self, data, node_type: str, x: torch.Tensor) -> torch.Tensor:
        store = data[node_type]
        if hasattr(store, "batch") and store.batch is not None:
            return store.batch
        return torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    def _edge_attr(self, data, edge_type: tuple):
        store = data[edge_type]
        if self.use_edge_features and hasattr(store, "edge_feat") and store.edge_feat is not None:
            feat = store.edge_feat
            if feat.dim() == 1:
                feat = feat.unsqueeze(-1)
            if feat.size(1) >= 1:
                return feat[:, :1]
        if hasattr(store, "edge_attr") and store.edge_attr is not None:
            return store.edge_attr
        return None





    def forward(self, data):
        """Run the WSIHeteroGraphModel hierarchical forward pass over patch/region/slide HeteroData."""
        if not hasattr(data, "x_dict") or not hasattr(data, "edge_index_dict"):
            raise ValueError("WSIHeteroGraphModel expects HeteroData as input")


        x_dict = {}
        for nt in self.node_types:
            if nt in data.node_types and hasattr(data[nt], "x"):
                x_dict[nt] = self.input_proj[nt](data[nt].x)


        edge_attr_dict = {}
        for edge_type in [("patch", "near", "patch"), ("region", "near", "region")]:
            ea = self._edge_attr(data, edge_type)
            if ea is not None:
                edge_attr_dict[edge_type] = ea


        for layer_idx, conv in enumerate(self.convs):

            new_x = conv(x_dict, data.edge_index_dict, edge_attr_dict=edge_attr_dict)

            for nt, new_feat in new_x.items():
                old_feat = x_dict.get(nt)
                new_feat = F.gelu(new_feat)
                new_feat = F.dropout(new_feat, p=self.dropout_rate, training=self.training)

                if old_feat is not None and old_feat.shape == new_feat.shape:
                    new_feat = new_feat + old_feat
                new_feat = self.norms[f"{nt}_{layer_idx}"](new_feat)
                x_dict[nt] = new_feat


        level_embeddings = []

        if "patch" in x_dict:
            batch_p = self._batch(data, "patch", x_dict["patch"])
            level_embeddings.append(self.patch_pool(x_dict["patch"], batch_p))
        else:
            level_embeddings.append(None)

        if "region" in x_dict:
            batch_r = self._batch(data, "region", x_dict["region"])
            level_embeddings.append(self.region_pool(x_dict["region"], batch_r))
        else:
            level_embeddings.append(None)

        if "slide" in x_dict:
            batch_s = self._batch(data, "slide", x_dict["slide"])
            level_embeddings.append(global_mean_pool(x_dict["slide"], batch_s))
        else:
            level_embeddings.append(None)


        ref = next(e for e in level_embeddings if e is not None)
        B, H = ref.shape
        level_embeddings = [
            e if e is not None else torch.zeros(B, H, device=ref.device)
            for e in level_embeddings
        ]


        concat = torch.cat(level_embeddings, dim=1)
        gates = torch.softmax(self.fusion_gate(concat), dim=1)
        stacked = torch.stack(level_embeddings, dim=1)
        fused = (gates.unsqueeze(-1) * stacked).sum(dim=1)
        fused = self.fusion_proj(fused)


        logits = self.classifier(fused)
        return _format_graph_logits(logits)






from torch_geometric.nn import JumpingKnowledge as _JK


@register_model(('gnn', 'hetero_wsi_v2', 'graph'))
class WSIHeteroGraphModelV2(nn.Module):
    """
    High-Performance Hierarchical Heterogeneous GNN for WSI classification.

    Improvements over WSIHeteroGraphModel:
    1. Coordinate-based Positional Encoding
       - Uses data["patch"].spatial_x = (x, y, tissue_frac) and
         data["region"].spatial_x = (x, y) (always available from graph pipeline)
       - Per-graph normalization to [0, 1], then MLP → hidden_dim added to node features
    2. Structured 3-phase message passing per layer
       - Phase A: Spatial  (GATv2, patch↔patch, region↔region) with normalized edge_attr
       - Phase B: Bottom-up (SAGEConv, patch→region, region→slide)
       - Phase C: Top-down  (SAGEConv, slide→region, region→patch)
       - Residual + LayerNorm after every phase per node type
    3. JumpingKnowledge ("cat") across all GNN layers for patches and regions
       - Captures multi-scale feature representations
    4. Deeper AttentionalAggregation pooling with LayerNorm in gate_nn
    5. Cross-Attention Fusion (MultiheadAttention, learnable CLS query)
       - Replaces the scalar 3-weight gate → much more expressive
    6. Learnable edge-distance normalization (sigmoid with per-layer scale param)
    Cross-Scanner Robustness (added):
    7. Per-graph feature whitening before input projection
       - Z-score normalization per graph removes scanner-induced distribution
         shifts in CNN backbone output (color, contrast, stain differences)
    8. Relative edge features (edge_dim 1 → 3)
       - (Δx_norm, Δy_norm, dist_norm) computed from centroid coordinates,
         normalized by each graph's bounding box → resolution/magnification
         invariant. Absolute pixel distances are scanner-specific; relative
         topology is not.
    9. DropEdge during training (drop_edge_p)
       - Randomly mask edges to force robustness against topological
         variations and different scanner connectivity patterns.
    """


    _EDGE_DIM: int = 3
    _HIER_EDGE_DIM: int = 1

    def __init__(
        self,
        model_name: str,
        num_classes: int = 1,
        dropout_rate: float = 0.2,
        drop_edge_p: float = 0.1,
        in_dim: int = 1537,
        hidden_dim: int = 256,
        num_layers: int = 3,
        heads: int = 4,
        use_edge_features: bool = True,
        learnable_edge_features: bool = True,
        use_hierarchical_edge_features: bool = True,
        edge_semantic_weight: float = 0.25,
    ):
        """Initialize the WSIHeteroGraphModelV2 multi-phase heterogeneous GNN."""
        super().__init__()
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        self.drop_edge_p = drop_edge_p
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.node_types = ("patch", "region", "slide")
        self.use_edge_features = bool(use_edge_features)
        self.learnable_edge_features = bool(learnable_edge_features)
        self.use_hierarchical_edge_features = bool(use_hierarchical_edge_features)
        self.edge_semantic_weight = float(edge_semantic_weight)




        self.input_proj = nn.ModuleDict({
            nt: nn.Sequential(nn.LazyLinear(hidden_dim), nn.LayerNorm(hidden_dim))
            for nt in self.node_types
        })







        self.patch_pe = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.region_pe = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )




        gat_out = hidden_dim // heads

        self.spatial_convs = nn.ModuleList()
        self.bottom_up_convs = nn.ModuleList()
        self.top_down_convs = nn.ModuleList()
        self.norms = nn.ModuleDict()


        self.edge_scale = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(num_layers)
        ])

        if self.learnable_edge_features:
            self.near_edge_encoder = nn.Sequential(
                nn.Linear(6, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, self._EDGE_DIM),
            )
            self.hier_edge_encoder = nn.Sequential(
                nn.Linear(6, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, self._HIER_EDGE_DIM),
            )
        else:
            self.near_edge_encoder = None
            self.hier_edge_encoder = None

        for layer_idx in range(num_layers):


            self.spatial_convs.append(HeteroConv(
                {
                    ("patch", "near", "patch"): GATv2Conv(
                        (hidden_dim, hidden_dim), gat_out, heads=heads,
                        dropout=dropout_rate, add_self_loops=False, edge_dim=self._EDGE_DIM,
                    ),
                    ("region", "near", "region"): GATv2Conv(
                        (hidden_dim, hidden_dim), gat_out, heads=heads,
                        dropout=dropout_rate, add_self_loops=False, edge_dim=self._EDGE_DIM,
                    ),
                    ("patch", "similar", "patch"): GATv2Conv(
                        (hidden_dim, hidden_dim), gat_out, heads=heads,
                        dropout=dropout_rate, add_self_loops=False, edge_dim=1,
                    ),
                    ("region", "similar", "region"): GATv2Conv(
                        (hidden_dim, hidden_dim), gat_out, heads=heads,
                        dropout=dropout_rate, add_self_loops=False, edge_dim=1,
                    ),
                    ("patch", "long", "patch"): GATv2Conv(
                        (hidden_dim, hidden_dim), gat_out, heads=heads,
                        dropout=dropout_rate, add_self_loops=False, edge_dim=1,
                    ),
                    ("region", "long", "region"): GATv2Conv(
                        (hidden_dim, hidden_dim), gat_out, heads=heads,
                        dropout=dropout_rate, add_self_loops=False, edge_dim=1,
                    ),
                },
                aggr="mean",
            ))


            if self.use_hierarchical_edge_features:
                self.bottom_up_convs.append(HeteroConv(
                    {
                        ("patch", "in", "region"): GATv2Conv(
                            (hidden_dim, hidden_dim), gat_out, heads=heads,
                            dropout=dropout_rate, add_self_loops=False, edge_dim=self._HIER_EDGE_DIM,
                        ),
                        ("region", "part_of", "slide"): GATv2Conv(
                            (hidden_dim, hidden_dim), gat_out, heads=heads,
                            dropout=dropout_rate, add_self_loops=False, edge_dim=self._HIER_EDGE_DIM,
                        ),
                    },
                    aggr="mean",
                ))
            else:
                self.bottom_up_convs.append(HeteroConv(
                    {
                        ("patch", "in", "region"):      SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                        ("region", "part_of", "slide"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                    },
                    aggr="mean",
                ))


            if self.use_hierarchical_edge_features:
                self.top_down_convs.append(HeteroConv(
                    {
                        ("slide", "contains", "region"): GATv2Conv(
                            (hidden_dim, hidden_dim), gat_out, heads=heads,
                            dropout=dropout_rate, add_self_loops=False, edge_dim=self._HIER_EDGE_DIM,
                        ),
                        ("region", "has", "patch"): GATv2Conv(
                            (hidden_dim, hidden_dim), gat_out, heads=heads,
                            dropout=dropout_rate, add_self_loops=False, edge_dim=self._HIER_EDGE_DIM,
                        ),
                    },
                    aggr="mean",
                ))
            else:
                self.top_down_convs.append(HeteroConv(
                    {
                        ("slide", "contains", "region"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                        ("region", "has", "patch"):      SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                    },
                    aggr="mean",
                ))


            for nt in self.node_types:
                for phase in ("spa", "bu", "td"):
                    self.norms[f"{nt}_{layer_idx}_{phase}"] = nn.LayerNorm(hidden_dim)




        self.patch_jk = _JK(mode="cat", channels=hidden_dim, num_layers=num_layers)
        self.region_jk = _JK(mode="cat", channels=hidden_dim, num_layers=num_layers)
        self.patch_jk_proj = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.region_jk_proj = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )




        def _gate_nn():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1),
            )

        self.patch_pool = AttentionalAggregation(_gate_nn())
        self.region_pool = AttentionalAggregation(_gate_nn())




        self.fusion_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )




        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 4, num_classes),
        )





    def _batch_vec(self, data, node_type: str, x: torch.Tensor) -> torch.Tensor:
        store = data[node_type]
        if hasattr(store, "batch") and store.batch is not None:
            return store.batch
        return torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    def _whiten_features(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1
        device = x.device
        D = x.size(1)
        idx = batch.unsqueeze(1).expand(-1, D)


        count = torch.bincount(batch, minlength=num_graphs).float().clamp(min=1.0)
        mean = torch.zeros(num_graphs, D, device=device)
        mean.scatter_add_(0, idx, x)
        mean = mean / count.unsqueeze(1)


        diff = x - mean[batch]
        var = torch.zeros(num_graphs, D, device=device)
        var.scatter_add_(0, idx, diff ** 2)
        std = (var / count.unsqueeze(1) + 1e-6).sqrt()

        return diff / std[batch]

    def _relative_edge_features(
        self,
        edge_index: torch.Tensor,
        centroid: torch.Tensor,
        raw_dist: torch.Tensor,
        batch: torch.Tensor,
        raw_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = centroid.device
        src, dst = edge_index[0], edge_index[1]
        num_graphs = int(batch.max().item()) + 1


        delta = centroid[dst] - centroid[src]


        max_xy = torch.zeros(num_graphs, 2, device=device)
        min_xy = torch.full((num_graphs, 2), float("inf"), device=device)
        max_xy.scatter_reduce_(0, batch.unsqueeze(1).expand(-1, 2), centroid, reduce="amax", include_self=True)
        min_xy.scatter_reduce_(0, batch.unsqueeze(1).expand(-1, 2), centroid, reduce="amin", include_self=True)
        span = (max_xy - min_xy).clamp(min=1.0)
        delta_norm = delta / span[batch[src]]


        d = raw_dist.squeeze(-1) if raw_dist.dim() > 1 else raw_dist
        same_sem = torch.zeros_like(d)
        sem_valid = torch.zeros_like(d)
        inv_d = 1.0 / d.clamp(min=1e-6)

        if self.use_edge_features and raw_feat is not None:
            feat = raw_feat.float()
            if feat.dim() == 1:
                feat = feat.unsqueeze(-1)
            if feat.size(1) >= 6:
                d = feat[:, 0]
                inv_d = feat[:, 1]
                feat_dx = feat[:, 2]
                feat_dy = feat[:, 3]
                same_sem = feat[:, 4]
                sem_valid = feat[:, 5]
                delta_norm = torch.stack([feat_dx, feat_dy], dim=1) / span[batch[src]]
        max_d = torch.zeros(num_graphs, device=device)
        max_d.scatter_reduce_(0, batch[src], d, reduce="amax", include_self=True)
        dist_norm = (d / max_d[batch[src]].clamp(min=1.0)).unsqueeze(-1)

        inv_max = torch.zeros(num_graphs, device=device)
        inv_max.scatter_reduce_(0, batch[src], inv_d, reduce="amax", include_self=True)
        inv_norm = (inv_d / inv_max[batch[src]].clamp(min=1e-6)).unsqueeze(-1)

        rich = torch.cat(
            [
                dist_norm,
                inv_norm,
                delta_norm,
                same_sem.unsqueeze(-1),
                sem_valid.unsqueeze(-1),
            ],
            dim=1,
        )

        if self.learnable_edge_features and self.near_edge_encoder is not None:
            return self.near_edge_encoder(rich)

        semantic_boost = self.edge_semantic_weight * same_sem.unsqueeze(-1) * sem_valid.unsqueeze(-1)
        dist_out = (dist_norm + semantic_boost).clamp(min=0.0, max=1.0)
        return torch.cat([delta_norm, dist_out], dim=1)

    def _hierarchical_edge_features(self, store, edge_count: int, device: torch.device) -> torch.Tensor:
        raw_feat = getattr(store, "edge_feat", None)
        if self.use_edge_features and torch.is_tensor(raw_feat):
            feat = raw_feat.to(device).float()
            if feat.dim() == 1:
                feat = feat.unsqueeze(-1)
            if feat.size(1) >= 6:
                base = feat[:, :6]
            else:
                pad = torch.zeros((feat.size(0), 6 - feat.size(1)), dtype=feat.dtype, device=device)
                base = torch.cat([feat, pad], dim=1)
        else:
            raw_attr = getattr(store, "edge_attr", None)
            if torch.is_tensor(raw_attr):
                attr = raw_attr.to(device).float()
                if attr.dim() == 1:
                    attr = attr.unsqueeze(-1)
                if attr.size(1) > 1:
                    attr = attr[:, :1]
                inv = 1.0 / attr.clamp(min=1e-6)
                zeros = torch.zeros((attr.size(0), 4), dtype=attr.dtype, device=device)
                base = torch.cat([attr, inv, zeros], dim=1)
            else:
                base = torch.zeros((edge_count, 6), dtype=torch.float32, device=device)

        if self.learnable_edge_features and self.hier_edge_encoder is not None:
            return self.hier_edge_encoder(base)

        return base[:, :1]

    def _drop_edges(
        self,
        edge_index_dict: dict,
        edge_attr_dict: dict,
    ) -> tuple[dict, dict]:
        if not self.training or self.drop_edge_p <= 0.0:
            return edge_index_dict, edge_attr_dict

        new_ei: dict = {}
        new_ea: dict = {}
        hier_edges = {
            ("patch", "in", "region"),
            ("region", "part_of", "slide"),
            ("slide", "contains", "region"),
            ("region", "has", "patch"),
        }
        for et, ei in edge_index_dict.items():
            E = ei.size(1)
            if E == 0:
                new_ei[et] = ei
                new_ea[et] = edge_attr_dict.get(et)
                continue
            if self.use_hierarchical_edge_features and et in hier_edges:
                new_ei[et] = ei
                new_ea[et] = edge_attr_dict.get(et)
                continue
            keep = torch.rand(E, device=ei.device) > self.drop_edge_p
            new_ei[et] = ei[:, keep]
            if et in edge_attr_dict and edge_attr_dict[et] is not None:
                new_ea[et] = edge_attr_dict[et][keep]
        return new_ei, new_ea

    def _normalize_spatial(
        self, coords: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        device = coords.device
        has_extra = coords.size(1) > 2
        xy = coords[:, :2]


        num_graphs = int(batch.max().item()) + 1
        max_xy = torch.zeros(num_graphs, 2, device=device)
        max_xy.scatter_reduce_(
            0,
            batch.unsqueeze(1).expand(-1, 2),
            xy,
            reduce="amax",
            include_self=True,
        )
        max_xy = max_xy.clamp(min=1.0)
        norm_xy = xy / max_xy[batch]

        if has_extra:
            return torch.cat([norm_xy, coords[:, 2:]], dim=1)
        return norm_xy

    def _run_phase(
        self,
        conv,
        x_dict: dict,
        edge_index_dict: dict,
        edge_attr_dict: dict | None = None,
    ) -> dict:
        try:
            if edge_attr_dict:
                new_x = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            else:
                new_x = conv(x_dict, edge_index_dict)
        except Exception:
            new_x = {}
        return new_x

    def _merge(
        self,
        x_dict: dict,
        new_x: dict,
        layer_idx: int,
        phase: str,
    ) -> None:
        for nt, feat in new_x.items():
            old = x_dict.get(nt)
            feat = F.gelu(feat)
            feat = F.dropout(feat, p=self.dropout_rate, training=self.training)
            if old is not None and old.shape == feat.shape:
                feat = feat + old
            key = f"{nt}_{layer_idx}_{phase}"
            if key in self.norms:
                feat = self.norms[key](feat)
            x_dict[nt] = feat





    def forward(self, data):
        """Run the WSIHeteroGraphModelV2 multi-phase forward pass over patch/region/slide HeteroData."""
        if not hasattr(data, "x_dict") or not hasattr(data, "edge_index_dict"):
            raise ValueError("WSIHeteroGraphModelV2 expects HeteroData as input")

        device = next(
            (data[nt].x for nt in self.node_types if nt in data.node_types and hasattr(data[nt], "x")),
            None,
        )
        if device is None:
            raise ValueError("No node features found in HeteroData")
        device = device.device




        x_dict: dict = {}
        for nt in self.node_types:
            if nt in data.node_types and hasattr(data[nt], "x"):
                feat = data[nt].x.to(device)
                if feat.size(0) > 1:
                    b = self._batch_vec(data, nt, feat)
                    feat = self._whiten_features(feat, b)
                x_dict[nt] = self.input_proj[nt](feat)


        if "patch" in x_dict:
            spatial = getattr(data["patch"], "spatial_x", None)
            if spatial is not None:
                batch_p = self._batch_vec(data, "patch", x_dict["patch"])
                sp_norm = self._normalize_spatial(spatial.to(device), batch_p)
                x_dict["patch"] = x_dict["patch"] + self.patch_pe(sp_norm)

        if "region" in x_dict:
            spatial = getattr(data["region"], "spatial_x", None)
            if spatial is not None:
                batch_r = self._batch_vec(data, "region", x_dict["region"])
                sp_norm = self._normalize_spatial(spatial.to(device), batch_r)
                x_dict["region"] = x_dict["region"] + self.region_pe(sp_norm)





        rel_edge_attr: dict = {}
        centroid_map = {
            ("patch", "near", "patch"): ("patch", data["patch"].centroid if hasattr(data["patch"], "centroid") else None),
            ("region", "near", "region"): ("region", data["region"].centroid if hasattr(data["region"], "centroid") else None),
        }
        for edge_type, (nt, centroid) in centroid_map.items():
            if edge_type not in data.edge_index_dict:
                continue
            store = data[edge_type]
            raw_dist = getattr(store, "edge_attr", None)
            raw_feat = getattr(store, "edge_feat", None)
            if raw_dist is None or centroid is None:
                continue
            ei = data.edge_index_dict[edge_type]
            b = self._batch_vec(data, nt, x_dict.get(nt, centroid))
            rel_edge_attr[edge_type] = self._relative_edge_features(
                ei.to(device),
                centroid.to(device),
                raw_dist.to(device),
                b.to(device),
                raw_feat=raw_feat.to(device) if raw_feat is not None else None,
            )

        scalar_relations = [
            ("patch", "similar", "patch"),
            ("region", "similar", "region"),
            ("patch", "long", "patch"),
            ("region", "long", "region"),
        ]
        for edge_type in scalar_relations:
            if edge_type not in data.edge_index_dict:
                continue
            store = data[edge_type]
            raw_attr = getattr(store, "edge_attr", None)
            if raw_attr is None:
                edge_count = int(data.edge_index_dict[edge_type].size(1))
                rel_edge_attr[edge_type] = torch.ones((edge_count, 1), dtype=torch.float32, device=device)
                continue

            attr = raw_attr.to(device)
            if attr.dim() == 1:
                attr = attr.unsqueeze(-1)
            if attr.size(1) > 1:
                attr = attr[:, :1]
            rel_edge_attr[edge_type] = attr.float().clamp(min=0.0, max=1.0)


        patch_layer_outs: list = []
        region_layer_outs: list = []

        for layer_idx in range(self.num_layers):

            cur_ei_dict, cur_ea_dict = self._drop_edges(data.edge_index_dict, rel_edge_attr)



            scale = self.edge_scale[layer_idx]
            scaled_edge_attr = {}
            for et, ea in cur_ea_dict.items():

                scaled = ea.clone()
                if scaled.dim() == 1:
                    scaled = scaled.unsqueeze(-1)

                if scaled.size(1) >= 3:
                    scaled[:, 2:3] = torch.sigmoid(ea[:, 2:3] * scale)
                else:
                    scaled = torch.sigmoid(ea * scale)
                scaled_edge_attr[et] = scaled


            new_spa = self._run_phase(
                self.spatial_convs[layer_idx],
                x_dict, cur_ei_dict,
                edge_attr_dict=scaled_edge_attr,
            )
            self._merge(x_dict, new_spa, layer_idx, "spa")

            hier_edge_attr = {}
            if self.use_hierarchical_edge_features:
                hier_edges = [
                    ("patch", "in", "region"),
                    ("region", "part_of", "slide"),
                    ("slide", "contains", "region"),
                    ("region", "has", "patch"),
                ]
                for et in hier_edges:
                    if et in cur_ei_dict and et in data.edge_types:
                        store = data[et]
                        edge_count = int(cur_ei_dict[et].size(1))
                        full_feat = self._hierarchical_edge_features(store, edge_count, device)
                        if full_feat.size(0) == edge_count:
                            hier_edge_attr[et] = full_feat
                        elif full_feat.size(0) > edge_count:
                            hier_edge_attr[et] = full_feat[:edge_count]


            new_bu = self._run_phase(
                self.bottom_up_convs[layer_idx],
                x_dict, cur_ei_dict,
                edge_attr_dict=hier_edge_attr if self.use_hierarchical_edge_features else None,
            )
            self._merge(x_dict, new_bu, layer_idx, "bu")


            new_td = self._run_phase(
                self.top_down_convs[layer_idx],
                x_dict, cur_ei_dict,
                edge_attr_dict=hier_edge_attr if self.use_hierarchical_edge_features else None,
            )
            self._merge(x_dict, new_td, layer_idx, "td")

            if "patch" in x_dict:
                patch_layer_outs.append(x_dict["patch"])
            if "region" in x_dict:
                region_layer_outs.append(x_dict["region"])


        if patch_layer_outs:
            x_dict["patch"] = self.patch_jk_proj(self.patch_jk(patch_layer_outs))
        if region_layer_outs:
            x_dict["region"] = self.region_jk_proj(self.region_jk(region_layer_outs))


        level_embeddings: list = []

        if "patch" in x_dict:
            batch_p = self._batch_vec(data, "patch", x_dict["patch"])
            level_embeddings.append(self.patch_pool(x_dict["patch"], batch_p))
        else:
            level_embeddings.append(None)

        if "region" in x_dict:
            batch_r = self._batch_vec(data, "region", x_dict["region"])
            level_embeddings.append(self.region_pool(x_dict["region"], batch_r))
        else:
            level_embeddings.append(None)

        if "slide" in x_dict:
            batch_s = self._batch_vec(data, "slide", x_dict["slide"])
            level_embeddings.append(global_mean_pool(x_dict["slide"], batch_s))
        else:
            level_embeddings.append(None)


        ref = next(e for e in level_embeddings if e is not None)
        B, H = ref.shape
        level_embeddings = [
            e if e is not None else torch.zeros(B, H, device=device)
            for e in level_embeddings
        ]



        kv = torch.stack(level_embeddings, dim=1)
        q = self.fusion_query.expand(B, -1, -1)
        fused, _ = self.fusion_attn(q, kv, kv)
        fused = fused.squeeze(1)
        fused = self.fusion_norm(fused)
        fused = self.fusion_proj(fused)


        logits = self.classifier(fused)
        return _format_graph_logits(logits)


@register_model(('gnn', 'hetero_wsi_adaptive', 'graph'))
class WSIHeteroGraphModelAdaptive(nn.Module):
    """
    Optimized model for adaptive_multirel graphs.

    Focus:
    - Relation-aware spatial propagation (near/similar/long separated)
    - Learnable per-layer relation fusion
    - Robustness primitives from v2 (whitening, relative edge handling, DropEdge)
    """

    _EDGE_DIM_NEAR: int = 3
    _EDGE_DIM_HIER: int = 1

    def __init__(
        self,
        model_name: str,
        num_classes: int = 1,
        dropout_rate: float = 0.15,
        drop_edge_p: float = 0.10,
        in_dim: int = 1537,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 2,
        use_edge_features: bool = True,
        learnable_edge_features: bool = True,
        use_hierarchical_edge_features: bool = True,
        edge_semantic_weight: float = 0.25,
    ):
        """Initialize the WSIHeteroGraphModelAdaptive relation-aware heterogeneous GNN."""
        super().__init__()
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        self.drop_edge_p = drop_edge_p
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.node_types = ("patch", "region", "slide")
        self.use_edge_features = bool(use_edge_features)
        self.learnable_edge_features = bool(learnable_edge_features)
        self.use_hierarchical_edge_features = bool(use_hierarchical_edge_features)
        self.edge_semantic_weight = float(edge_semantic_weight)

        self.input_proj = nn.ModuleDict({
            nt: nn.Sequential(
                nn.LazyLinear(hidden_dim),
                nn.LayerNorm(hidden_dim)
            ) for nt in self.node_types
        })

        self.patch_pe = nn.Sequential(
            nn.Linear(3, max(8, hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(8, hidden_dim // 2), hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.region_pe = nn.Sequential(
            nn.Linear(2, max(8, hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(8, hidden_dim // 2), hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        gat_out = max(8, hidden_dim // heads)

        self.spatial_near_convs = nn.ModuleList()
        self.spatial_similar_convs = nn.ModuleList()
        self.spatial_long_convs = nn.ModuleList()
        self.bottom_up_convs = nn.ModuleList()
        self.top_down_convs = nn.ModuleList()
        self.norms = nn.ModuleDict()

        self.edge_scale = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(num_layers)
        ])
        self.relation_gate = nn.ModuleList([
            nn.ModuleDict({
                nt: nn.Sequential(
                    nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
                    nn.GELU(),
                    nn.Linear(max(8, hidden_dim // 2), 3)
                ) for nt in self.node_types
            })
            for _ in range(num_layers)
        ])

        if self.learnable_edge_features:
            self.edge_encoder = nn.ModuleDict({
                "near": nn.Sequential(
                    nn.Linear(6, max(8, hidden_dim // 2)),
                    nn.GELU(),
                    nn.Linear(max(8, hidden_dim // 2), 3),
                ),
                "scalar": nn.Sequential(
                    nn.Linear(1, max(8, hidden_dim // 2)),
                    nn.GELU(),
                    nn.Linear(max(8, hidden_dim // 2), 1),
                ),
                "hier": nn.Sequential(
                    nn.Linear(6, max(8, hidden_dim // 2)),
                    nn.GELU(),
                    nn.Linear(max(8, hidden_dim // 2), self._EDGE_DIM_HIER),
                ),
            })
        else:
            self.edge_encoder = None

        for layer_idx in range(num_layers):
            self.spatial_near_convs.append(HeteroConv(
                {
                    ("patch", "near", "patch"): GATv2Conv(
                        (hidden_dim, hidden_dim), gat_out, heads=heads,
                        dropout=dropout_rate, add_self_loops=False, edge_dim=self._EDGE_DIM_NEAR,
                    ),
                    ("region", "near", "region"): GATv2Conv(
                        (hidden_dim, hidden_dim), gat_out, heads=heads,
                        dropout=dropout_rate, add_self_loops=False, edge_dim=self._EDGE_DIM_NEAR,
                    ),
                },
                aggr="mean",
            ))

            self.spatial_similar_convs.append(HeteroConv(
                {
                    ("patch", "similar", "patch"): GATv2Conv(
                        (hidden_dim, hidden_dim), gat_out, heads=heads,
                        dropout=dropout_rate, add_self_loops=False, edge_dim=1,
                    ),
                    ("region", "similar", "region"): GATv2Conv(
                        (hidden_dim, hidden_dim), gat_out, heads=heads,
                        dropout=dropout_rate, add_self_loops=False, edge_dim=1,
                    ),
                },
                aggr="mean",
            ))

            self.spatial_long_convs.append(HeteroConv(
                {
                    ("patch", "long", "patch"): GATv2Conv(
                        (hidden_dim, hidden_dim), gat_out, heads=heads,
                        dropout=dropout_rate, add_self_loops=False, edge_dim=1,
                    ),
                    ("region", "long", "region"): GATv2Conv(
                        (hidden_dim, hidden_dim), gat_out, heads=heads,
                        dropout=dropout_rate, add_self_loops=False, edge_dim=1,
                    ),
                },
                aggr="mean",
            ))

            if self.use_hierarchical_edge_features:
                self.bottom_up_convs.append(HeteroConv(
                    {
                        ("patch", "in", "region"): GATv2Conv(
                            (hidden_dim, hidden_dim), gat_out, heads=heads,
                            dropout=dropout_rate, add_self_loops=False, edge_dim=self._EDGE_DIM_HIER,
                        ),
                        ("region", "part_of", "slide"): GATv2Conv(
                            (hidden_dim, hidden_dim), gat_out, heads=heads,
                            dropout=dropout_rate, add_self_loops=False, edge_dim=self._EDGE_DIM_HIER,
                        ),
                    },
                    aggr="mean",
                ))
            else:
                self.bottom_up_convs.append(HeteroConv(
                    {
                        ("patch", "in", "region"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                        ("region", "part_of", "slide"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                    },
                    aggr="mean",
                ))

            if self.use_hierarchical_edge_features:
                self.top_down_convs.append(HeteroConv(
                    {
                        ("slide", "contains", "region"): GATv2Conv(
                            (hidden_dim, hidden_dim), gat_out, heads=heads,
                            dropout=dropout_rate, add_self_loops=False, edge_dim=self._EDGE_DIM_HIER,
                        ),
                        ("region", "has", "patch"): GATv2Conv(
                            (hidden_dim, hidden_dim), gat_out, heads=heads,
                            dropout=dropout_rate, add_self_loops=False, edge_dim=self._EDGE_DIM_HIER,
                        ),
                    },
                    aggr="mean",
                ))
            else:
                self.top_down_convs.append(HeteroConv(
                    {
                        ("slide", "contains", "region"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                        ("region", "has", "patch"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                    },
                    aggr="mean",
                ))

            for nt in self.node_types:
                for phase in ("spa", "bu", "td"):
                    self.norms[f"{nt}_{layer_idx}_{phase}"] = nn.LayerNorm(hidden_dim)

        self.patch_jk = _JK(mode="lstm", channels=hidden_dim, num_layers=num_layers)
        self.region_jk = _JK(mode="lstm", channels=hidden_dim, num_layers=num_layers)
        self.patch_jk_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.region_jk_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        def _gate_nn():
            return nn.Sequential(
                nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
                nn.GELU(),
                nn.LayerNorm(max(8, hidden_dim // 2)),
                nn.Linear(max(8, hidden_dim // 2), 1),
            )

        self.patch_pool = AttentionalAggregation(_gate_nn())
        self.region_pool = AttentionalAggregation(_gate_nn())

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(max(8, hidden_dim // 2), num_classes),
        )

    def _batch_vec(self, data, node_type: str, x: torch.Tensor) -> torch.Tensor:
        store = data[node_type]
        if hasattr(store, "batch") and store.batch is not None:
            return store.batch
        return torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    def _whiten_features(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1
        device = x.device
        D = x.size(1)
        idx = batch.unsqueeze(1).expand(-1, D)

        count = torch.bincount(batch, minlength=num_graphs).float().clamp(min=1.0)
        mean = torch.zeros(num_graphs, D, device=device)
        mean.scatter_add_(0, idx, x)
        mean = mean / count.unsqueeze(1)

        diff = x - mean[batch]
        var = torch.zeros(num_graphs, D, device=device)
        var.scatter_add_(0, idx, diff ** 2)
        std = (var / count.unsqueeze(1) + 1e-6).sqrt()
        return diff / std[batch]

    def _relative_edge_features(
        self,
        edge_index: torch.Tensor,
        centroid: torch.Tensor,
        raw_dist: torch.Tensor,
        batch: torch.Tensor,
        raw_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = centroid.device
        src, dst = edge_index[0], edge_index[1]
        num_graphs = int(batch.max().item()) + 1

        delta = centroid[dst] - centroid[src]

        max_xy = torch.zeros(num_graphs, 2, device=device)
        min_xy = torch.full((num_graphs, 2), float("inf"), device=device)
        max_xy.scatter_reduce_(0, batch.unsqueeze(1).expand(-1, 2), centroid, reduce="amax", include_self=True)
        min_xy.scatter_reduce_(0, batch.unsqueeze(1).expand(-1, 2), centroid, reduce="amin", include_self=True)
        span = (max_xy - min_xy).clamp(min=1.0)
        delta_norm = delta / span[batch[src]]

        d = raw_dist.squeeze(-1) if raw_dist.dim() > 1 else raw_dist
        same_sem = torch.zeros_like(d)
        sem_valid = torch.zeros_like(d)
        inv_d = 1.0 / d.clamp(min=1e-6)

        if self.use_edge_features and raw_feat is not None:
            feat = raw_feat.float()
            if feat.dim() == 1:
                feat = feat.unsqueeze(-1)
            if feat.size(1) >= 6:
                d = feat[:, 0]
                inv_d = feat[:, 1]
                feat_dx = feat[:, 2]
                feat_dy = feat[:, 3]
                same_sem = feat[:, 4]
                sem_valid = feat[:, 5]
                delta_norm = torch.stack([feat_dx, feat_dy], dim=1) / span[batch[src]]
        max_d = torch.zeros(num_graphs, device=device)
        max_d.scatter_reduce_(0, batch[src], d, reduce="amax", include_self=True)
        dist_norm = (d / max_d[batch[src]].clamp(min=1.0)).unsqueeze(-1)

        inv_max = torch.zeros(num_graphs, device=device)
        inv_max.scatter_reduce_(0, batch[src], inv_d, reduce="amax", include_self=True)
        inv_norm = (inv_d / inv_max[batch[src]].clamp(min=1e-6)).unsqueeze(-1)

        rich = torch.cat(
            [
                dist_norm,
                inv_norm,
                delta_norm,
                same_sem.unsqueeze(-1),
                sem_valid.unsqueeze(-1),
            ],
            dim=1,
        )

        if self.learnable_edge_features and self.edge_encoder is not None:
            return self.edge_encoder["near"](rich)

        semantic_boost = self.edge_semantic_weight * same_sem.unsqueeze(-1) * sem_valid.unsqueeze(-1)
        dist_out = (dist_norm + semantic_boost).clamp(min=0.0, max=1.0)
        return torch.cat([delta_norm, dist_out], dim=1)

    def _hierarchical_edge_features(self, store, edge_count: int, device: torch.device) -> torch.Tensor:
        raw_feat = getattr(store, "edge_feat", None)
        if self.use_edge_features and torch.is_tensor(raw_feat):
            feat = raw_feat.to(device).float()
            if feat.dim() == 1:
                feat = feat.unsqueeze(-1)
            if feat.size(1) >= 6:
                base = feat[:, :6]
            else:
                pad = torch.zeros((feat.size(0), 6 - feat.size(1)), dtype=feat.dtype, device=device)
                base = torch.cat([feat, pad], dim=1)
        else:
            raw_attr = getattr(store, "edge_attr", None)
            if torch.is_tensor(raw_attr):
                attr = raw_attr.to(device).float()
                if attr.dim() == 1:
                    attr = attr.unsqueeze(-1)
                if attr.size(1) > 1:
                    attr = attr[:, :1]
                inv = 1.0 / attr.clamp(min=1e-6)
                zeros = torch.zeros((attr.size(0), 4), dtype=attr.dtype, device=device)
                base = torch.cat([attr, inv, zeros], dim=1)
            else:
                base = torch.zeros((edge_count, 6), dtype=torch.float32, device=device)

        if self.learnable_edge_features and self.edge_encoder is not None:
            return self.edge_encoder["hier"](base)

        return base[:, :1]

    def _drop_edges(self, edge_index_dict: dict, edge_attr_dict: dict) -> tuple[dict, dict]:
        if not self.training or self.drop_edge_p <= 0.0:
            return edge_index_dict, edge_attr_dict

        new_ei: dict = {}
        new_ea: dict = {}
        for et, ei in edge_index_dict.items():
            E = ei.size(1)
            if E == 0:
                new_ei[et] = ei
                new_ea[et] = edge_attr_dict.get(et)
                continue
            hierarchical_edges = [
                ("patch", "in", "region"),
                ("region", "part_of", "slide"),
                ("slide", "contains", "region"),
                ("region", "has", "patch"),
            ]

            drop_p = 0.0 if (self.use_hierarchical_edge_features and et in hierarchical_edges) else (0.05 if et in hierarchical_edges else self.drop_edge_p)
            keep = torch.rand(E, device=ei.device) > drop_p
            new_ei[et] = ei[:, keep]
            if et in edge_attr_dict and edge_attr_dict[et] is not None:
                new_ea[et] = edge_attr_dict[et][keep]
        return new_ei, new_ea

    def _normalize_spatial(self, coords: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        device = coords.device
        has_extra = coords.size(1) > 2
        xy = coords[:, :2]

        num_graphs = int(batch.max().item()) + 1
        max_xy = torch.zeros(num_graphs, 2, device=device)
        max_xy.scatter_reduce_(
            0,
            batch.unsqueeze(1).expand(-1, 2),
            xy,
            reduce="amax",
            include_self=True,
        )
        max_xy = max_xy.clamp(min=1.0)
        norm_xy = xy / max_xy[batch]

        if has_extra:
            return torch.cat([norm_xy, coords[:, 2:]], dim=1)
        return norm_xy

    def _run_phase(self, conv, x_dict: dict, edge_index_dict: dict, edge_attr_dict: dict | None = None) -> dict:
        try:
            if edge_attr_dict:
                new_x = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            else:
                new_x = conv(x_dict, edge_index_dict)
        except Exception:
            new_x = {}
        return new_x

    def _merge(self, x_dict: dict, new_x: dict, layer_idx: int, phase: str) -> None:
        for nt, feat in new_x.items():
            if not torch.is_tensor(feat):
                continue
            old = x_dict.get(nt)
            feat = F.gelu(feat)
            feat = F.dropout(feat, p=self.dropout_rate, training=self.training)
            if old is not None and old.shape == feat.shape:
                feat = feat + old
            key = f"{nt}_{layer_idx}_{phase}"
            if key in self.norms:
                feat = self.norms[key](feat)
            x_dict[nt] = feat

    def _combine_spatial_outputs(self, layer_idx: int, near_x: dict, similar_x: dict, long_x: dict, x_dict: dict) -> dict:
        out: dict = {}
        for nt in self.node_types:
            if nt not in x_dict:
                continue
            if not torch.is_tensor(x_dict[nt]):
                continue
            gate_logits = self.relation_gate[layer_idx][nt](x_dict[nt])
            weights = torch.softmax(gate_logits, dim=-1)

            combined = 0
            if nt in near_x and torch.is_tensor(near_x[nt]):
                combined = combined + weights[:, 0:1] * near_x[nt]
            if nt in similar_x and torch.is_tensor(similar_x[nt]):
                combined = combined + weights[:, 1:2] * similar_x[nt]
            if nt in long_x and torch.is_tensor(long_x[nt]):
                combined = combined + weights[:, 2:3] * long_x[nt]

            out[nt] = combined

        return out

    def forward(self, data):
        """Run the WSIHeteroGraphModelAdaptive relation-aware forward pass over patch/region/slide HeteroData."""
        if not hasattr(data, "x_dict") or not hasattr(data, "edge_index_dict"):
            raise ValueError("WSIHeteroGraphModelAdaptive expects HeteroData as input")

        device = None
        for nt in self.node_types:
            if nt in data.node_types and hasattr(data[nt], "x"):
                device = data[nt].x.device
                break
        if device is None:
            raise ValueError("No node features found in HeteroData")

        x_dict: dict = {}
        for nt in self.node_types:
            if nt in data.node_types and hasattr(data[nt], "x"):
                feat = data[nt].x.to(device)
                if not torch.is_tensor(feat):
                    raise TypeError(f"Feature for node type {nt} is not a tensor!")
                if feat.size(0) > 1:
                    b = self._batch_vec(data, nt, feat)
                    feat = 0.7 * feat + 0.3 * self._whiten_features(feat, b)
                x_dict[nt] = self.input_proj[nt](feat)

        if "patch" in x_dict:
            spatial = getattr(data["patch"], "spatial_x", None)
            if spatial is not None:
                batch_p = self._batch_vec(data, "patch", x_dict["patch"])
                sp_norm = self._normalize_spatial(spatial.to(device), batch_p)
                x_dict["patch"] = x_dict["patch"] + self.patch_pe(sp_norm)

        if "region" in x_dict:
            spatial = getattr(data["region"], "spatial_x", None)
            if spatial is not None:
                batch_r = self._batch_vec(data, "region", x_dict["region"])
                sp_norm = self._normalize_spatial(spatial.to(device), batch_r)
                x_dict["region"] = x_dict["region"] + self.region_pe(sp_norm)

        rel_edge_attr: dict = {}
        near_rel = {
            ("patch", "near", "patch"): ("patch", data["patch"].centroid if hasattr(data["patch"], "centroid") else None),
            ("region", "near", "region"): ("region", data["region"].centroid if hasattr(data["region"], "centroid") else None),
        }
        for edge_type, (nt, centroid) in near_rel.items():
            if edge_type not in data.edge_index_dict:
                continue
            store = data[edge_type]
            raw_dist = getattr(store, "edge_attr", None)
            raw_feat = getattr(store, "edge_feat", None)
            if raw_dist is None or centroid is None:
                continue
            ei = data.edge_index_dict[edge_type]
            b = self._batch_vec(data, nt, x_dict.get(nt, centroid))
            if not (torch.is_tensor(ei) and torch.is_tensor(centroid) and torch.is_tensor(raw_dist) and torch.is_tensor(b)):
                raise TypeError(f"Edge features for {edge_type} are not all tensors!")
            rel = self._relative_edge_features(
                ei.to(device),
                centroid.to(device),
                raw_dist.to(device),
                b.to(device),
                raw_feat=raw_feat.to(device) if torch.is_tensor(raw_feat) else None,
            )
            if self.learnable_edge_features and self.edge_encoder is not None:
                rel_edge_attr[edge_type] = rel
            else:
                rel_edge_attr[edge_type] = rel

        scalar_relations = [
            ("patch", "similar", "patch"),
            ("region", "similar", "region"),
            ("patch", "long", "patch"),
            ("region", "long", "region"),
        ]
        for edge_type in scalar_relations:
            if edge_type not in data.edge_index_dict:
                continue
            store = data[edge_type]
            raw_attr = getattr(store, "edge_attr", None)
            if raw_attr is None:
                edge_count = int(data.edge_index_dict[edge_type].size(1))
                rel_edge_attr[edge_type] = torch.ones((edge_count, 1), dtype=torch.float32, device=device)
                continue

            attr = raw_attr.to(device)
            if attr.dim() == 1:
                attr = attr.unsqueeze(-1)
            if attr.size(1) > 1:
                attr = attr[:, :1]
            if not torch.is_tensor(attr):
                raise TypeError(f"Edge attribute for {edge_type} is not a tensor!")
            if self.learnable_edge_features and self.edge_encoder is not None:
                rel_edge_attr[edge_type] = self.edge_encoder["scalar"](attr.float())
            else:
                rel_edge_attr[edge_type] = attr.float().clamp(min=0.0, max=1.0)

        patch_layer_outs: list = []
        region_layer_outs: list = []

        for layer_idx in range(self.num_layers):
            cur_ei_dict, cur_ea_dict = self._drop_edges(data.edge_index_dict, rel_edge_attr)

            scale = self.edge_scale[layer_idx]
            scaled_edge_attr = {}
            for et, ea in cur_ea_dict.items():
                scaled = ea.clone()
                if scaled.dim() == 1:
                    scaled = scaled.unsqueeze(-1)
                if scaled.size(1) >= 3:
                    scaled[:, 2:3] = torch.sigmoid(ea[:, 2:3] * scale)
                else:
                    scaled = torch.sigmoid(ea * scale)
                scaled_edge_attr[et] = scaled

            near_ei = {
                k: v for k, v in cur_ei_dict.items()
                if k in [("patch", "near", "patch"), ("region", "near", "region")]
            }
            near_ea = {k: scaled_edge_attr[k] for k in near_ei.keys() if k in scaled_edge_attr}

            similar_ei = {
                k: v for k, v in cur_ei_dict.items()
                if k in [("patch", "similar", "patch"), ("region", "similar", "region")]
            }
            similar_ea = {k: scaled_edge_attr[k] for k in similar_ei.keys() if k in scaled_edge_attr}

            long_ei = {
                k: v for k, v in cur_ei_dict.items()
                if k in [("patch", "long", "patch"), ("region", "long", "region")]
            }
            long_ea = {k: scaled_edge_attr[k] for k in long_ei.keys() if k in scaled_edge_attr}

            near_x = self._run_phase(self.spatial_near_convs[layer_idx], x_dict, near_ei, edge_attr_dict=near_ea)
            similar_x = self._run_phase(self.spatial_similar_convs[layer_idx], x_dict, similar_ei, edge_attr_dict=similar_ea)
            long_x = self._run_phase(self.spatial_long_convs[layer_idx], x_dict, long_ei, edge_attr_dict=long_ea)
            new_spa = self._combine_spatial_outputs(layer_idx, near_x, similar_x, long_x, x_dict)
            self._merge(x_dict, new_spa, layer_idx, "spa")

            hier_edge_attr = {}
            if self.use_hierarchical_edge_features:
                for et in [
                    ("patch", "in", "region"),
                    ("region", "part_of", "slide"),
                    ("slide", "contains", "region"),
                    ("region", "has", "patch"),
                ]:
                    if et in cur_ei_dict and et in data.edge_types:
                        store = data[et]
                        edge_count = int(cur_ei_dict[et].size(1))
                        full_feat = self._hierarchical_edge_features(store, edge_count, device)
                        if full_feat.size(0) == edge_count:
                            hier_edge_attr[et] = full_feat
                        elif full_feat.size(0) > edge_count:
                            hier_edge_attr[et] = full_feat[:edge_count]

            bu_ei = {
                k: v for k, v in cur_ei_dict.items()
                if k in [("patch", "in", "region"), ("region", "part_of", "slide")]
            }
            bu_ea = {k: hier_edge_attr[k] for k in bu_ei.keys() if k in hier_edge_attr}
            new_bu = self._run_phase(
                self.bottom_up_convs[layer_idx],
                x_dict,
                bu_ei,
                edge_attr_dict=bu_ea if self.use_hierarchical_edge_features else None,
            )
            self._merge(x_dict, new_bu, layer_idx, "bu")

            td_ei = {
                k: v for k, v in cur_ei_dict.items()
                if k in [("slide", "contains", "region"), ("region", "has", "patch")]
            }
            td_ea = {k: hier_edge_attr[k] for k in td_ei.keys() if k in hier_edge_attr}
            new_td = self._run_phase(
                self.top_down_convs[layer_idx],
                x_dict,
                td_ei,
                edge_attr_dict=td_ea if self.use_hierarchical_edge_features else None,
            )
            self._merge(x_dict, new_td, layer_idx, "td")

            if "patch" in x_dict:
                patch_layer_outs.append(x_dict["patch"])
            if "region" in x_dict:
                region_layer_outs.append(x_dict["region"])

        if patch_layer_outs:
            patch_jk_out = self.patch_jk(patch_layer_outs)
            x_dict["patch"] = patch_jk_out
        if region_layer_outs:
            region_jk_out = self.region_jk(region_layer_outs)
            x_dict["region"] = region_jk_out

        level_embeddings: list = []
        if "patch" in x_dict:
            batch_p = self._batch_vec(data, "patch", x_dict["patch"])
            level_embeddings.append(self.patch_pool(x_dict["patch"], batch_p))
        else:
            level_embeddings.append(None)

        if "region" in x_dict:
            batch_r = self._batch_vec(data, "region", x_dict["region"])
            level_embeddings.append(self.region_pool(x_dict["region"], batch_r))
        else:
            level_embeddings.append(None)

        if "slide" in x_dict:
            batch_s = self._batch_vec(data, "slide", x_dict["slide"])
            level_embeddings.append(global_mean_pool(x_dict["slide"], batch_s))
        else:
            level_embeddings.append(None)

        ref = next(e for e in level_embeddings if e is not None)
        B, H = ref.shape
        level_embeddings = [
            e if e is not None else torch.zeros(B, H, device=device)
            for e in level_embeddings
        ]

        kv = torch.stack(level_embeddings, dim=1)
        q = self.query_proj(torch.mean(kv, dim=1, keepdim=True))
        fused, _ = self.fusion_attn(q, kv, kv)
        fused = fused.squeeze(1)
        fused = self.fusion_norm(fused)
        fused = self.fusion_proj(fused)

        logits = self.classifier(fused)
        return _format_graph_logits(logits)


@register_model(('gnn', 'hetero_wsi_dynamical', 'graph'))
class WSIHeteroGraphModelDynamical(nn.Module):
    """Heterogeneous graph model for graphs produced by graph_creation_new.

    Supports both node creation modes:
    - patch encoder graphs: patch/region/slide
    - cell segmentation graphs: cell/region/slide
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 1,
        dropout_rate: float = 0.2,
        drop_edge_p: float = 0.1,
        in_dim: int = 1537,
        hidden_dim: int = 192,
        num_layers: int = 3,
        heads: int = 4,
        use_edge_features: bool = True,
        learnable_edge_features: bool = True,
        use_hierarchical_edge_features: bool = True,
        edge_semantic_weight: float = 0.25,
        max_base_nodes_for_full_forward: int = 20000,
        max_base_edges_for_full_forward: int = 400000,
        min_base_nodes_per_region_in_large_graph: int = 1,
    ):
        """Initialize the WSIHeteroGraphModelDynamical patch/cell hierarchical GNN."""
        super().__init__()
        self.model_name = model_name
        self.dropout_rate = float(dropout_rate)
        self.drop_edge_p = float(drop_edge_p)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.use_edge_features = bool(use_edge_features)
        self.learnable_edge_features = bool(learnable_edge_features)
        self.use_hierarchical_edge_features = bool(use_hierarchical_edge_features)
        self.edge_semantic_weight = float(edge_semantic_weight)
        self.max_base_nodes_for_full_forward = int(max_base_nodes_for_full_forward)
        self.max_base_edges_for_full_forward = int(max_base_edges_for_full_forward)
        self.min_base_nodes_per_region_in_large_graph = int(min_base_nodes_per_region_in_large_graph)
        self._large_graph_counter = 0
        self._logger = logging.getLogger(__name__)

        self.node_types = ("patch", "cell", "region", "slide")
        self.input_proj = nn.ModuleDict({
            nt: nn.Sequential(nn.LazyLinear(hidden_dim), nn.LayerNorm(hidden_dim))
            for nt in self.node_types
        })

        gat_out = max(1, hidden_dim // max(1, heads))
        self.near_edge_dim = 4
        self.hier_edge_dim = 2

        if self.learnable_edge_features:
            self.near_edge_encoder = nn.Sequential(
                nn.Linear(6, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, self.near_edge_dim),
            )
            self.hier_edge_encoder = nn.Sequential(
                nn.Linear(6, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, self.hier_edge_dim),
            )
        else:
            self.near_edge_encoder = None
            self.hier_edge_encoder = None

        self.patch_layers = nn.ModuleList()
        self.cell_layers = nn.ModuleList()
        self.norms = nn.ModuleDict()

        for layer_idx in range(self.num_layers):
            self.patch_layers.append(
                HeteroConv(
                    {
                        ("patch", "near", "patch"): GATv2Conv(
                            (hidden_dim, hidden_dim),
                            gat_out,
                            heads=heads,
                            dropout=dropout_rate,
                            add_self_loops=False,
                            edge_dim=self.near_edge_dim,
                        ),
                        ("region", "near", "region"): GATv2Conv(
                            (hidden_dim, hidden_dim),
                            gat_out,
                            heads=heads,
                            dropout=dropout_rate,
                            add_self_loops=False,
                            edge_dim=self.near_edge_dim,
                        ),
                        ("patch", "in", "region"): GATv2Conv(
                            (hidden_dim, hidden_dim),
                            gat_out,
                            heads=heads,
                            dropout=dropout_rate,
                            add_self_loops=False,
                            edge_dim=self.hier_edge_dim,
                        ),
                        ("region", "has", "patch"): GATv2Conv(
                            (hidden_dim, hidden_dim),
                            gat_out,
                            heads=heads,
                            dropout=dropout_rate,
                            add_self_loops=False,
                            edge_dim=self.hier_edge_dim,
                        ),
                        ("region", "part_of", "slide"): GATv2Conv(
                            (hidden_dim, hidden_dim),
                            gat_out,
                            heads=heads,
                            dropout=dropout_rate,
                            add_self_loops=False,
                            edge_dim=self.hier_edge_dim,
                        ),
                        ("slide", "contains", "region"): GATv2Conv(
                            (hidden_dim, hidden_dim),
                            gat_out,
                            heads=heads,
                            dropout=dropout_rate,
                            add_self_loops=False,
                            edge_dim=self.hier_edge_dim,
                        ),
                    },
                    aggr="mean",
                )
            )

            self.cell_layers.append(
                HeteroConv(
                    {
                        ("cell", "near", "cell"): GATv2Conv(
                            (hidden_dim, hidden_dim),
                            gat_out,
                            heads=heads,
                            dropout=dropout_rate,
                            add_self_loops=False,
                            edge_dim=self.near_edge_dim,
                        ),
                        ("region", "near", "region"): GATv2Conv(
                            (hidden_dim, hidden_dim),
                            gat_out,
                            heads=heads,
                            dropout=dropout_rate,
                            add_self_loops=False,
                            edge_dim=self.near_edge_dim,
                        ),
                        ("cell", "in", "region"): GATv2Conv(
                            (hidden_dim, hidden_dim),
                            gat_out,
                            heads=heads,
                            dropout=dropout_rate,
                            add_self_loops=False,
                            edge_dim=self.hier_edge_dim,
                        ),
                        ("region", "has", "cell"): GATv2Conv(
                            (hidden_dim, hidden_dim),
                            gat_out,
                            heads=heads,
                            dropout=dropout_rate,
                            add_self_loops=False,
                            edge_dim=self.hier_edge_dim,
                        ),
                        ("region", "part_of", "slide"): GATv2Conv(
                            (hidden_dim, hidden_dim),
                            gat_out,
                            heads=heads,
                            dropout=dropout_rate,
                            add_self_loops=False,
                            edge_dim=self.hier_edge_dim,
                        ),
                        ("slide", "contains", "region"): GATv2Conv(
                            (hidden_dim, hidden_dim),
                            gat_out,
                            heads=heads,
                            dropout=dropout_rate,
                            add_self_loops=False,
                            edge_dim=self.hier_edge_dim,
                        ),
                    },
                    aggr="mean",
                )
            )

            for nt in self.node_types:
                self.norms[f"{nt}_{layer_idx}"] = nn.LayerNorm(hidden_dim)

        gate_nn = lambda: nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.base_pool = AttentionalAggregation(gate_nn())
        self.region_pool = AttentionalAggregation(gate_nn())

        self.fusion_gate = nn.Linear(3 * hidden_dim, 3)
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _batch_vec(self, data, node_type: str, x: torch.Tensor) -> torch.Tensor:
        store = data[node_type]
        if hasattr(store, "batch") and store.batch is not None:
            return store.batch
        return torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    def _whiten(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if x.size(0) <= 1:
            return x
        num_graphs = int(batch.max().item()) + 1
        d = x.size(1)
        idx = batch.unsqueeze(1).expand(-1, d)

        count = torch.bincount(batch, minlength=num_graphs).float().clamp(min=1.0)
        mean = torch.zeros(num_graphs, d, device=x.device)
        mean.scatter_add_(0, idx, x)
        mean = mean / count.unsqueeze(1)

        diff = x - mean[batch]
        var = torch.zeros(num_graphs, d, device=x.device)
        var.scatter_add_(0, idx, diff * diff)
        std = (var / count.unsqueeze(1) + 1e-6).sqrt()
        return diff / std[batch]

    def _encode_edge(self, store, edge_count: int, device: torch.device, near: bool) -> torch.Tensor:
        raw_feat = getattr(store, "edge_feat", None)
        if self.use_edge_features and torch.is_tensor(raw_feat):
            feat = raw_feat.to(device).float()
            if feat.dim() == 1:
                feat = feat.unsqueeze(-1)
            if feat.size(1) >= 6:
                base = feat[:, :6]
            else:
                pad = torch.zeros((feat.size(0), 6 - feat.size(1)), device=device, dtype=feat.dtype)
                base = torch.cat([feat, pad], dim=1)
        else:
            raw_attr = getattr(store, "edge_attr", None)
            if torch.is_tensor(raw_attr):
                attr = raw_attr.to(device).float()
                if attr.dim() == 1:
                    attr = attr.unsqueeze(-1)
                attr = attr[:, :1]
                inv = 1.0 / attr.clamp(min=1e-6)
                zeros = torch.zeros((attr.size(0), 4), dtype=attr.dtype, device=device)
                base = torch.cat([attr, inv, zeros], dim=1)
            else:
                base = torch.zeros((edge_count, 6), dtype=torch.float32, device=device)

        if near:
            if self.learnable_edge_features and self.near_edge_encoder is not None:
                return self.near_edge_encoder(base)
            return base[:, : self.near_edge_dim]

        if self.learnable_edge_features and self.hier_edge_encoder is not None:
            return self.hier_edge_encoder(base)
        return base[:, : self.hier_edge_dim]

    def _drop_edges(self, edge_index_dict: dict, edge_attr_dict: dict, near_relations: set[tuple]) -> tuple[dict, dict]:
        if not self.training or self.drop_edge_p <= 0.0:
            return edge_index_dict, edge_attr_dict

        new_ei = {}
        new_ea = {}
        for et, ei in edge_index_dict.items():
            edge_count = int(ei.size(1))
            if edge_count == 0 or et not in near_relations:
                new_ei[et] = ei
                if et in edge_attr_dict:
                    new_ea[et] = edge_attr_dict[et]
                continue

            keep = torch.rand(edge_count, device=ei.device) > self.drop_edge_p
            new_ei[et] = ei[:, keep]
            if et in edge_attr_dict:
                new_ea[et] = edge_attr_dict[et][keep]
        return new_ei, new_ea

    def _maybe_subsample_base_graph(self, data, base_nt: str, device: torch.device) -> Optional[dict]:
        base_store = data[base_nt]
        near_et = (base_nt, "near", base_nt)
        in_region_et = (base_nt, "in", "region")
        has_base_et = ("region", "has", base_nt)

        near_edge_index = (
            data[near_et].edge_index.to(device) if near_et in data.edge_types
            else torch.zeros((2, 0), dtype=torch.long, device=device)
        )

        if not graph_patch_utils.should_use_region_only_forward(
            base_store, near_edge_index, self.max_base_nodes_for_full_forward, self.max_base_edges_for_full_forward,
        ):
            return None

        if in_region_et not in data.edge_types:
            return None

        p2r_edge_index = data[in_region_et].edge_index.to(device)

        self._large_graph_counter += 1
        if self._large_graph_counter <= 3 or (self._large_graph_counter % 25 == 0):
            self._logger.info(
                "%s large-graph %s subsampling triggered %d time(s) | %s_nodes=%d %s_edges=%d",
                self.__class__.__name__,
                base_nt,
                self._large_graph_counter,
                base_nt,
                int(base_store.x.size(0)),
                base_nt,
                int(near_edge_index.size(1)),
            )

        seed = torch.initial_seed() + self._large_graph_counter * 1_000_003
        sampled = graph_patch_utils.subsample_large_patch_graph(
            base_store,
            near_edge_index,
            p2r_edge_index,
            target=self.max_base_nodes_for_full_forward,
            min_patches_per_region_in_large_graph=self.min_base_nodes_per_region_in_large_graph,
            seed=seed,
        )

        num_base_nodes_orig = int(base_store.x.size(0))
        selected = sampled["selected_patch_idx"]
        mapping = torch.full((num_base_nodes_orig,), -1, dtype=torch.long, device=device)
        mapping[selected] = torch.arange(selected.numel(), device=device)

        has_edge_index_sub = None
        if has_base_et in data.edge_types:
            has_ei = data[has_base_et].edge_index.to(device)
            r_src = has_ei[0]
            b_dst = has_ei[1]
            valid = (b_dst >= 0) & (b_dst < num_base_nodes_orig) & (mapping[b_dst] >= 0)
            has_edge_index_sub = torch.stack([r_src[valid], mapping[b_dst[valid]]], dim=0)

        return {
            "x": sampled["patch_x"],
            "batch": sampled["patch_batch"],
            near_et: sampled["patch_edge_index"],
            in_region_et: sampled["p2r_edge_index"],
            has_base_et: has_edge_index_sub,
        }

    def forward(self, data):
        """Run the WSIHeteroGraphModelDynamical forward pass over patch/cell/region/slide HeteroData."""
        if not hasattr(data, "x_dict") or not hasattr(data, "edge_index_dict"):
            raise ValueError("WSIHeteroGraphModelDynamical expects HeteroData input.")

        base_nt = "cell" if "cell" in data.node_types else "patch"
        if base_nt not in data.node_types or "region" not in data.node_types:
            raise ValueError("Expected either patch/region or cell/region node types.")

        device = data[base_nt].x.device

        overrides = self._maybe_subsample_base_graph(data, base_nt, device)

        x_dict = {}
        batch_base_override = None
        for nt in self.node_types:
            if nt in data.node_types and hasattr(data[nt], "x") and torch.is_tensor(data[nt].x):
                if overrides is not None and nt == base_nt:
                    feat = overrides["x"].to(device)
                    batch = overrides["batch"].to(device) if overrides.get("batch") is not None else torch.zeros(feat.size(0), dtype=torch.long, device=device)
                    batch_base_override = batch
                else:
                    feat = data[nt].x.to(device)
                    batch = self._batch_vec(data, nt, feat)
                feat = self._whiten(feat, batch)
                x_dict[nt] = self.input_proj[nt](feat)

        if "slide" in x_dict and x_dict["slide"].size(0) == 1:
            x_dict["slide"] = x_dict["slide"] + 0.0

        near_relations = {(base_nt, "near", base_nt), ("region", "near", "region")}
        hierarchy_relations = {
            (base_nt, "in", "region"),
            ("region", "has", base_nt),
            ("region", "part_of", "slide"),
            ("slide", "contains", "region"),
        }
        required_relations = near_relations | hierarchy_relations

        conv_stack = self.cell_layers if base_nt == "cell" else self.patch_layers

        for layer_idx, conv in enumerate(conv_stack):
            edge_index_dict = {}
            edge_attr_dict = {}
            for et in required_relations:
                if overrides is not None and et in overrides:
                    ei = overrides[et]
                    if ei is None:
                        continue
                    ei = ei.to(device)
                    store = None
                elif et in data.edge_types:
                    store = data[et]
                    ei = store.edge_index.to(device)
                else:
                    continue
                edge_index_dict[et] = ei
                edge_attr_dict[et] = self._encode_edge(
                    store=store,
                    edge_count=int(ei.size(1)),
                    device=device,
                    near=et in near_relations,
                )

            edge_index_dict, edge_attr_dict = self._drop_edges(edge_index_dict, edge_attr_dict, near_relations)

            updates = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            for nt, new_feat in updates.items():
                old_feat = x_dict.get(nt)
                new_feat = F.gelu(new_feat)
                new_feat = F.dropout(new_feat, p=self.dropout_rate, training=self.training)
                if old_feat is not None and old_feat.shape == new_feat.shape:
                    new_feat = new_feat + old_feat
                x_dict[nt] = self.norms[f"{nt}_{layer_idx}"](new_feat)

        batch_base = batch_base_override if batch_base_override is not None else self._batch_vec(data, base_nt, x_dict[base_nt])
        base_emb = self.base_pool(x_dict[base_nt], batch_base)

        batch_region = self._batch_vec(data, "region", x_dict["region"])
        region_emb = self.region_pool(x_dict["region"], batch_region)

        if "slide" in x_dict:
            batch_slide = self._batch_vec(data, "slide", x_dict["slide"])
            slide_emb = global_mean_pool(x_dict["slide"], batch_slide)
        else:
            slide_emb = torch.zeros_like(base_emb)

        concat = torch.cat([base_emb, region_emb, slide_emb], dim=1)
        gates = torch.softmax(self.fusion_gate(concat), dim=1)
        stacked = torch.stack([base_emb, region_emb, slide_emb], dim=1)
        fused = (gates.unsqueeze(-1) * stacked).sum(dim=1)
        fused = self.fusion_proj(fused)

        logits = self.classifier(fused)
        return _format_graph_logits(logits)
