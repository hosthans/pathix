from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch


def resolve_negative_label_source(config: Optional[Dict[str, Any]]) -> str:
    """Resolve the effective negative_label_source from a merged model config, forcing "annotation_only" when label_mode isn't binary."""
    cfg = config or {}
    negative_label_source = str(cfg.get('negative_label_source', 'annotation_only'))
    label_mode = str(cfg.get('label_mode', 'binary')).strip().lower()
    if label_mode != 'binary':
        return 'annotation_only'
    return negative_label_source


def apply_slide_derived_negative_labels(
    mapped: torch.Tensor,
    node_batch: Optional[torch.Tensor],
    slide_binary: torch.Tensor,
    annotation_enabled: torch.Tensor,
    negative_label_source: str = "annotation_only",
) -> torch.Tensor:
    """Extend annotation-derived region/patch labels with confidently-negative labels when negative_label_source == "slide_derived"."""
    if str(negative_label_source).strip().lower() != "slide_derived":
        return mapped

    unlabeled = mapped < 0
    if not torch.any(unlabeled):
        return mapped

    num_nodes = int(mapped.numel())
    num_graphs = int(slide_binary.numel())
    if num_graphs == 0:
        return mapped

    slide_binary = slide_binary.to(device=mapped.device)
    annotation_enabled = annotation_enabled.to(device=mapped.device)

    if node_batch is not None and node_batch.numel() == num_nodes:
        batch = node_batch.to(device=mapped.device).long()
        valid = (batch >= 0) & (batch < num_graphs)
        confidently_negative = torch.zeros_like(unlabeled)
        clipped_batch = batch.clamp(min=0, max=num_graphs - 1)
        per_node_confident = (slide_binary[clipped_batch] == 0) | annotation_enabled[clipped_batch]
        confidently_negative[valid] = per_node_confident[valid]
    else:
        is_negative = bool((slide_binary[0] == 0).item()) or bool(annotation_enabled[0].item())
        confidently_negative = torch.full_like(unlabeled, fill_value=is_negative)

    assign = unlabeled & confidently_negative
    if torch.any(assign):
        mapped = mapped.clone()
        mapped[assign] = 0
    return mapped


def scatter_mean(values: torch.Tensor, index: torch.Tensor, n_out: int) -> torch.Tensor:
    """Compute the mean of values grouped by index into n_out groups."""
    if values.numel() == 0:
        return torch.zeros((n_out, values.size(1)), dtype=values.dtype, device=values.device)

    out = torch.zeros((n_out, values.size(1)), dtype=values.dtype, device=values.device)
    count = torch.zeros((n_out,), dtype=values.dtype, device=values.device)
    out.index_add_(0, index, values)
    ones = torch.ones((index.numel(),), dtype=values.dtype, device=values.device)
    count.index_add_(0, index, ones)
    count = count.clamp(min=1.0).unsqueeze(1)
    return out / count


def edge_index_for(data: Any, relation_options: Sequence[Tuple[str, str, str]]) -> Optional[torch.Tensor]:
    """Return the edge_index for the first relation in relation_options present in data."""
    for relation in relation_options:
        if relation in data.edge_types:
            return data[relation].edge_index
    return None


def edge_store_and_index_for(
    data: Any, relation_options: Sequence[Tuple[str, str, str]]
) -> Tuple[Optional[Any], Optional[torch.Tensor]]:
    """Return the edge store and edge_index for the first relation in relation_options present in data."""
    for relation in relation_options:
        if relation in data.edge_types:
            store = data[relation]
            return store, store.edge_index
    return None, None


def batch_for(store: Any, x: torch.Tensor) -> torch.Tensor:
    """Return store's batch tensor, or all-zeros if it has none."""
    batch = getattr(store, 'batch', None)
    if batch is None:
        return torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    return batch


def infer_patch_batch(
    patch_store: Any,
    region_batch: torch.Tensor,
    p2r_edge_index: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return the patch batch tensor, inferring it from patch-to-region edges if not already present."""
    patch_batch = getattr(patch_store, 'batch', None)
    if torch.is_tensor(patch_batch):
        return patch_batch

    num_patches = int(getattr(patch_store, 'num_nodes', 0) or (patch_store.x.size(0) if hasattr(patch_store, 'x') else 0))
    fallback = torch.zeros((num_patches,), dtype=torch.long, device=device)
    if not torch.is_tensor(p2r_edge_index) or p2r_edge_index.numel() == 0:
        return fallback

    p_src = p2r_edge_index[0].long()
    p_dst = p2r_edge_index[1].long()
    valid = (
        (p_src >= 0)
        & (p_src < num_patches)
        & (p_dst >= 0)
        & (p_dst < region_batch.size(0))
    )
    p_src = p_src[valid]
    p_dst = p_dst[valid]
    if p_src.numel() > 0:
        fallback[p_src] = region_batch[p_dst]
    return fallback


def should_use_region_only_forward(
    patch_store: Any,
    patch_edge_index: torch.Tensor,
    max_patch_nodes_for_full_forward: int,
    max_patch_edges_for_full_forward: int,
) -> bool:
    """Return whether the patch graph exceeds the configured node/edge thresholds for a full forward pass."""
    patch_nodes = int(getattr(patch_store, 'num_nodes', 0) or (patch_store.x.size(0) if hasattr(patch_store, 'x') else 0))
    patch_edges = int(patch_edge_index.size(1)) if torch.is_tensor(patch_edge_index) else 0
    return (
        patch_nodes > int(max_patch_nodes_for_full_forward)
        or patch_edges > int(max_patch_edges_for_full_forward)
    )


def subsample_large_patch_graph(
    patch_store: Any,
    patch_edge_index: torch.Tensor,
    p2r_edge_index: torch.Tensor,
    target: int,
    min_patches_per_region_in_large_graph: int = 1,
    patch_edge_feat: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Subsample a large patch graph down to `target` patches, keeping at least one representative patch per region."""
    patch_x_full = patch_store.x
    num_patches = int(patch_x_full.size(0))
    target = int(target)
    _gen = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFF) if seed is not None else None

    def _randperm(n: int, device: torch.device) -> torch.Tensor:
        if _gen is not None:
            return torch.randperm(n, generator=_gen).to(device)
        return torch.randperm(n, device=device)
    if num_patches <= target:
        full_idx = torch.arange(num_patches, device=patch_x_full.device)
        return {
            'patch_x': patch_x_full,
            'patch_edge_index': patch_edge_index,
            'p2r_edge_index': p2r_edge_index,
            'patch_annotation_y': getattr(patch_store, 'annotation_y', None),
            'patch_batch': getattr(patch_store, 'batch', None),
            'selected_patch_idx': full_idx,
            'patch_edge_feat': patch_edge_feat,
        }

    p_src = p2r_edge_index[0].long()
    p_dst = p2r_edge_index[1].long()
    valid = (p_src >= 0) & (p_src < num_patches)
    p_src = p_src[valid]
    p_dst = p_dst[valid]

    selected_parts = []

    patch_annotation_full = getattr(patch_store, 'annotation_y', None)
    if torch.is_tensor(patch_annotation_full) and patch_annotation_full.numel() == num_patches:
        annotated_idx = torch.where(patch_annotation_full.view(-1) >= 0)[0]
        if annotated_idx.numel() > 0:
            selected_parts.append(annotated_idx)

    if p_src.numel() > 0 and min_patches_per_region_in_large_graph > 0:
        order = torch.argsort(p_dst)
        sorted_dst = p_dst[order]
        sorted_src = p_src[order]
        keep = torch.ones_like(sorted_dst, dtype=torch.bool)
        keep[1:] = sorted_dst[1:] != sorted_dst[:-1]
        repr_patches = torch.unique(sorted_src[keep])
        if repr_patches.numel() > 0:
            selected_parts.append(repr_patches)

    if selected_parts:
        selected = torch.unique(torch.cat(selected_parts, dim=0))
    else:
        selected = torch.empty((0,), dtype=torch.long, device=patch_x_full.device)

    if selected.numel() >= target:
        if torch.is_tensor(patch_annotation_full) and patch_annotation_full.numel() == num_patches:
            annotated_mask = patch_annotation_full.view(-1)[selected] >= 0
        else:
            annotated_mask = torch.zeros_like(selected, dtype=torch.bool)
        must_keep = selected[annotated_mask]
        optional = selected[~annotated_mask]
        if must_keep.numel() >= target:
            perm = _randperm(must_keep.numel(), patch_x_full.device)[:target]
            selected = must_keep[perm]
        else:
            remaining = target - must_keep.numel()
            perm = _randperm(optional.numel(), patch_x_full.device)[:remaining]
            selected = torch.cat([must_keep, optional[perm]], dim=0)
    else:
        remaining = target - selected.numel()
        all_idx = torch.arange(num_patches, device=patch_x_full.device)
        pick_mask = torch.ones((num_patches,), dtype=torch.bool, device=patch_x_full.device)
        if selected.numel() > 0:
            pick_mask[selected] = False
        pool = all_idx[pick_mask]
        if pool.numel() > 0 and remaining > 0:
            perm = _randperm(pool.numel(), patch_x_full.device)[:remaining]
            selected = torch.cat([selected, pool[perm]], dim=0)

    selected = torch.unique(selected)
    if selected.numel() == 0:
        selected = _randperm(num_patches, patch_x_full.device)[: min(target, num_patches)]

    mapping = torch.full((num_patches,), -1, dtype=torch.long, device=patch_x_full.device)
    mapping[selected] = torch.arange(selected.numel(), device=patch_x_full.device)

    e_src = patch_edge_index[0].long()
    e_dst = patch_edge_index[1].long()
    e_valid = (
        (e_src >= 0)
        & (e_src < num_patches)
        & (e_dst >= 0)
        & (e_dst < num_patches)
        & (mapping[e_src] >= 0)
        & (mapping[e_dst] >= 0)
    )
    e_src = mapping[e_src[e_valid]]
    e_dst = mapping[e_dst[e_valid]]
    patch_edge_index_sub = torch.stack([e_src, e_dst], dim=0) if e_src.numel() > 0 else patch_edge_index.new_zeros((2, 0))

    patch_edge_feat_sub = None
    if torch.is_tensor(patch_edge_feat) and patch_edge_feat.size(0) == patch_edge_index.size(1):
        patch_edge_feat_sub = patch_edge_feat[e_valid]

    p_valid = (
        (p_src >= 0)
        & (p_src < num_patches)
        & (mapping[p_src] >= 0)
    )
    p_src_sub = mapping[p_src[p_valid]]
    p_dst_sub = p_dst[p_valid]
    p2r_sub = torch.stack([p_src_sub, p_dst_sub], dim=0) if p_src_sub.numel() > 0 else p2r_edge_index.new_zeros((2, 0))

    patch_annotation = getattr(patch_store, 'annotation_y', None)
    if torch.is_tensor(patch_annotation) and patch_annotation.numel() == num_patches:
        patch_annotation = patch_annotation.view(-1)[selected]
    else:
        patch_annotation = None

    patch_batch = getattr(patch_store, 'batch', None)
    if torch.is_tensor(patch_batch) and patch_batch.numel() == num_patches:
        patch_batch = patch_batch.view(-1)[selected]
    else:
        patch_batch = None

    return {
        'patch_x': patch_x_full[selected],
        'patch_edge_index': patch_edge_index_sub,
        'p2r_edge_index': p2r_sub,
        'patch_annotation_y': patch_annotation,
        'patch_batch': patch_batch,
        'selected_patch_idx': selected,
        'patch_edge_feat': patch_edge_feat_sub,
    }
