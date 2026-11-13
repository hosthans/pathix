from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def feature_gaussian_noise(
    data,
    *,
    rng: np.random.Generator,
    std: float = 0.05,
    node_types: Sequence[str] = ("patch", "region"),
    **_ignored: Any,
) -> None:
    """Adds Gaussian noise to node features in place, scaled relative to each node type's feature std."""
    if not hasattr(data, "node_types"):
        return
    for node_type in node_types:
        if node_type not in data.node_types:
            continue
        store = data[node_type]
        x = getattr(store, "x", None)
        if not torch.is_tensor(x) or x.numel() == 0:
            continue
        feature_std = float(x.detach().float().std().item()) or 1.0
        noise = rng.normal(loc=0.0, scale=std * feature_std, size=tuple(x.shape))
        store.x = x + torch.from_numpy(noise).to(dtype=x.dtype, device=x.device)


def feature_dropout(
    data,
    *,
    rng: np.random.Generator,
    p: float = 0.1,
    node_types: Sequence[str] = ("patch", "region"),
    **_ignored: Any,
) -> None:
    """Randomly zeroes individual feature dimensions per node, in place."""
    if not hasattr(data, "node_types"):
        return
    for node_type in node_types:
        if node_type not in data.node_types:
            continue
        store = data[node_type]
        x = getattr(store, "x", None)
        if not torch.is_tensor(x) or x.numel() == 0:
            continue
        keep = rng.random(size=tuple(x.shape)) >= p
        store.x = x * torch.from_numpy(keep).to(dtype=x.dtype, device=x.device)


def feature_scale_noise(
    data,
    *,
    rng: np.random.Generator,
    std: float = 0.1,
    node_types: Sequence[str] = ("patch", "region"),
    per_dimension: bool = False,
    **_ignored: Any,
) -> None:
    """Applies multiplicative Gaussian noise to node features in place."""
    if not hasattr(data, "node_types"):
        return
    for node_type in node_types:
        if node_type not in data.node_types:
            continue
        store = data[node_type]
        x = getattr(store, "x", None)
        if not torch.is_tensor(x) or x.numel() == 0:
            continue
        noise_shape = tuple(x.shape) if per_dimension else (x.shape[0], 1)
        scale = 1.0 + rng.normal(loc=0.0, scale=std, size=noise_shape)
        store.x = x * torch.from_numpy(scale).to(dtype=x.dtype, device=x.device)


def node_drop(
    data,
    *,
    rng: np.random.Generator,
    p: float = 0.1,
    node_type: str = "patch",
    protect_annotated: bool = True,
    min_remaining: int = 8,
    **_ignored: Any,
) -> None:
    """Randomly drops a fraction of patch nodes and every edge touching them, in place."""
    if node_type != "patch":
        raise ValueError(
            "node_drop currently only supports node_type='patch' -- dropping region/slide "
            "nodes would need explicit handling in the label-supervision fallback cascade."
        )
    if not hasattr(data, "node_types") or "patch" not in data.node_types:
        return

    patch_store = data["patch"]
    x = getattr(patch_store, "x", None)
    if not torch.is_tensor(x):
        return
    n_patches = int(x.size(0))
    if n_patches <= min_remaining:
        return

    drop_mask = rng.random(n_patches) < p
    if protect_annotated:
        annotation_y = getattr(patch_store, "annotation_y", None)
        if torch.is_tensor(annotation_y) and annotation_y.numel() == n_patches:
            has_annotation = (annotation_y.detach().cpu().numpy() >= 0)
            drop_mask &= ~has_annotation

    if not drop_mask.any():
        return
    n_keep = n_patches - int(drop_mask.sum())
    if n_keep < min_remaining:
        return

    keep_idx = torch.from_numpy(np.where(~drop_mask)[0]).to(device=x.device, dtype=torch.long)
    mapping = torch.full((n_patches,), -1, dtype=torch.long, device=x.device)
    mapping[keep_idx] = torch.arange(keep_idx.numel(), device=x.device)

    for attr in ("x", "feat", "annotation_y", "centroid", "topleft", "spatial_x", "batch", "y"):
        val = getattr(patch_store, attr, None)
        if torch.is_tensor(val) and val.size(0) == n_patches:
            setattr(patch_store, attr, val[keep_idx])
    patch_store.num_nodes = int(keep_idx.numel())

    patch_edge_key = None
    for candidate in (("patch", "near", "patch"), ("patch", "spatial", "patch")):
        if candidate in data.edge_types:
            patch_edge_key = candidate
            break
    if patch_edge_key is not None:
        ei = data[patch_edge_key].edge_index
        src, dst = ei[0].long(), ei[1].long()
        valid = (src < n_patches) & (dst < n_patches) & (mapping[src] >= 0) & (mapping[dst] >= 0)
        data[patch_edge_key].edge_index = (
            torch.stack([mapping[src[valid]], mapping[dst[valid]]], dim=0) if valid.any() else ei.new_zeros((2, 0))
        )

    if ("patch", "in", "region") in data.edge_types:
        ei = data[("patch", "in", "region")].edge_index
        src, dst = ei[0].long(), ei[1].long()
        valid = (src < n_patches) & (mapping[src] >= 0)
        new_src = mapping[src[valid]]
        new_dst = dst[valid]
        new_p2r = torch.stack([new_src, new_dst], dim=0) if valid.any() else ei.new_zeros((2, 0))
        data[("patch", "in", "region")].edge_index = new_p2r
        if ("region", "has", "patch") in data.edge_types:
            data[("region", "has", "patch")].edge_index = (
                torch.stack([new_dst, new_src], dim=0) if valid.any() else ei.new_zeros((2, 0))
            )


def graph_mixup(
    data,
    *,
    rng: np.random.Generator,
    second_graph_provider: Optional[Callable[[], Optional[Tuple[Any, float]]]] = None,
    alpha: float = 0.4,
    node_types: Sequence[str] = ("patch", "region"),
    p: float = 0.3,
    **_ignored: Any,
) -> None:
    """Blends this graph's node features and label with a second, randomly sampled graph (feature-space mixup), in place."""
    if second_graph_provider is None or rng.random() >= p:
        return
    sampled = second_graph_provider()
    if sampled is None:
        return
    graph_b, label_b = sampled

    lam = float(rng.beta(alpha, alpha)) if alpha > 0 else 1.0
    for node_type in node_types:
        if not hasattr(data, "node_types") or node_type not in data.node_types:
            continue
        if not hasattr(graph_b, "node_types") or node_type not in graph_b.node_types:
            continue
        x_a = getattr(data[node_type], "x", None)
        x_b = getattr(graph_b[node_type], "x", None)
        if not torch.is_tensor(x_a) or not torch.is_tensor(x_b) or x_a.numel() == 0 or x_b.numel() == 0:
            continue
        resample_idx = rng.integers(0, x_b.size(0), size=x_a.size(0))
        x_b_matched = x_b[torch.from_numpy(resample_idx).long()].to(dtype=x_a.dtype, device=x_a.device)
        data[node_type].x = lam * x_a + (1.0 - lam) * x_b_matched

    data._mixup_lambda = lam
    data._mixup_label_b = float(label_b)


def graph_mixstyle(
    data,
    *,
    rng: np.random.Generator,
    second_graph_provider: Optional[Callable[[], Optional[Tuple[Any, float]]]] = None,
    alpha: float = 0.1,
    p: float = 0.4,
    node_types: Sequence[str] = ("patch", "region"),
    eps: float = 1e-6,
    min_nodes: int = 2,
    **_ignored: Any,
) -> None:
    """Mixes this graph's per-feature-dimension statistics with a second, randomly sampled graph's statistics (MixStyle-style), in place."""
    if second_graph_provider is None or rng.random() >= p:
        return
    sampled = second_graph_provider()
    if sampled is None:
        return
    graph_b, _label_b = sampled

    lam = float(rng.beta(alpha, alpha)) if alpha > 0 else 1.0
    for node_type in node_types:
        if not hasattr(data, "node_types") or node_type not in data.node_types:
            continue
        if not hasattr(graph_b, "node_types") or node_type not in graph_b.node_types:
            continue
        x_a = getattr(data[node_type], "x", None)
        x_b = getattr(graph_b[node_type], "x", None)
        if not torch.is_tensor(x_a) or not torch.is_tensor(x_b):
            continue
        if x_a.size(0) < min_nodes or x_b.size(0) < min_nodes:
            continue

        mu_a = x_a.mean(dim=0, keepdim=True)
        sigma_a = x_a.std(dim=0, keepdim=True, unbiased=False)
        mu_b = x_b.mean(dim=0, keepdim=True).to(dtype=x_a.dtype, device=x_a.device)
        sigma_b = x_b.std(dim=0, keepdim=True, unbiased=False).to(dtype=x_a.dtype, device=x_a.device)

        x_norm = (x_a - mu_a) / (sigma_a + eps)
        mu_mix = lam * mu_a + (1.0 - lam) * mu_b
        sigma_mix = lam * sigma_a + (1.0 - lam) * sigma_b
        data[node_type].x = x_norm * sigma_mix + mu_mix

    # Deliberately no data._mixup_lambda / data._mixup_label_b -- labels are never touched here,
    # unlike graph_mixup.


_AUGMENTATION_REGISTRY = {
    "feature_gaussian_noise": feature_gaussian_noise,
    "feature_scale_noise": feature_scale_noise,
    "feature_dropout": feature_dropout,
    "node_drop": node_drop,
    "graph_mixup": graph_mixup,
    "graph_mixstyle": graph_mixstyle,
}


def apply_graph_augmentations(
    data,
    specs: List[Dict[str, Any]],
    rng: np.random.Generator,
    second_graph_provider: Optional[Callable[[], Optional[Tuple[Any, float]]]] = None,
) -> None:
    """Applies a sequence of configured augmentations to `data` in place, in order."""
    if not specs:
        return
    for spec in specs:
        if "name" not in spec:
            raise ValueError(f"Augmentation spec missing required 'name' key: {spec}")
        name = str(spec["name"])
        fn = _AUGMENTATION_REGISTRY.get(name)
        if fn is None:
            raise ValueError(f"Unknown augmentation '{name}'. Available: {sorted(_AUGMENTATION_REGISTRY)}")
        params = {k: v for k, v in spec.items() if k != "name"}
        fn(data, rng=rng, second_graph_provider=second_graph_provider, **params)
