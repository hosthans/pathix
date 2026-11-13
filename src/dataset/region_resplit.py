from typing import Iterator, List, Optional

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from src.graph_creation_new.helper.edges import build_edge_features


def resplit_large_regions(
    data,
    target_patches_per_region: int = 250,
    k_region: int = 5,
) -> None:
    """Splits any region with more than `target_patches_per_region` patches into smaller, spatially contiguous subregions, in place."""
    if not hasattr(data, "node_types") or "region" not in data.node_types or "patch" not in data.node_types:
        return

    region_store = data["region"]
    patch_store = data["patch"]

    n_patches = int(getattr(patch_store, "num_nodes", 0) or (patch_store.x.size(0) if hasattr(patch_store, "x") else 0))
    n_regions_old = int(getattr(region_store, "num_nodes", 0) or (region_store.x.size(0) if hasattr(region_store, "x") else 0))
    if n_patches == 0 or n_regions_old == 0:
        return

    if ("patch", "in", "region") not in data.edge_types:
        return
    p2r_edge_index = data[("patch", "in", "region")].edge_index
    if not torch.is_tensor(p2r_edge_index) or p2r_edge_index.numel() == 0:
        return

    patch_pos = getattr(patch_store, "centroid", None)
    if not torch.is_tensor(patch_pos):
        patch_pos = getattr(patch_store, "topleft", None)
    if not torch.is_tensor(patch_pos) or patch_pos.size(0) != n_patches:
        return

    patch_pos_np = patch_pos.detach().cpu().numpy().astype(np.float64)
    p_src = p2r_edge_index[0].long().cpu().numpy()
    p_dst = p2r_edge_index[1].long().cpu().numpy()

    old_member_patch_idx = [p_src[p_dst == r] for r in range(n_regions_old)]

    old_x = region_store.x
    old_feat = getattr(region_store, "feat", old_x)
    old_centroid = getattr(region_store, "centroid", None)
    old_annotation_y = getattr(region_store, "annotation_y", None)
    old_label_ambiguous = getattr(region_store, "label_ambiguous", None)
    has_old_label_ambiguous = torch.is_tensor(old_label_ambiguous) and old_label_ambiguous.numel() == n_regions_old
    patch_x = patch_store.x
    patch_feat = getattr(patch_store, "feat", patch_x)
    patch_annotation_y = getattr(patch_store, "annotation_y", None)
    has_patch_annotation = torch.is_tensor(patch_annotation_y) and patch_annotation_y.numel() == n_patches

    extra_fields = _extra_patch_region_y_fields(region_store, patch_store)
    extra_old_region_y = {}
    extra_patch_y = {}
    for field in extra_fields:
        old_t = _valid_y_tensor(region_store, field, n_regions_old)
        patch_t = _valid_y_tensor(patch_store, field, n_patches)
        if old_t is not None and patch_t is not None:
            extra_old_region_y[field] = old_t
            extra_patch_y[field] = patch_t
    extra_region_rows = {field: [] for field in extra_old_region_y}
    extra_unknown_ids = _extra_field_unknown_ids(data, list(extra_old_region_y))

    region_x_rows = []
    region_feat_rows = []
    region_centroid_rows = []
    region_annotation_rows = [] if old_annotation_y is not None else None
    region_ambiguous_rows: List[bool] = []
    new_p2r_src_parts = []
    new_p2r_dst_parts = []

    next_id = 0
    any_split = False
    for old_r, members in enumerate(old_member_patch_idx):
        if members.size == 0:
            continue

        if members.size <= target_patches_per_region:
            region_x_rows.append(old_x[old_r])
            region_feat_rows.append(old_feat[old_r])
            if old_centroid is not None:
                region_centroid_rows.append(old_centroid[old_r])
            else:
                region_centroid_rows.append(torch.as_tensor(patch_pos_np[members].mean(axis=0), dtype=torch.float32))
            if region_annotation_rows is not None:
                region_annotation_rows.append(old_annotation_y[old_r])
            region_ambiguous_rows.append(bool(old_label_ambiguous[old_r].item()) if has_old_label_ambiguous else False)
            for field, rows in extra_region_rows.items():
                rows.append(extra_old_region_y[field][old_r])
            new_p2r_src_parts.append(members)
            new_p2r_dst_parts.append(np.full(members.shape, next_id, dtype=np.int64))
            next_id += 1
            continue

        any_split = True
        for group in _recursive_bisect(patch_pos_np[members], target_patches_per_region):
            patch_ids = members[group]
            region_x_rows.append(patch_x[patch_ids].mean(dim=0))
            region_feat_rows.append(patch_feat[patch_ids].mean(dim=0))
            region_centroid_rows.append(torch.as_tensor(patch_pos_np[patch_ids].mean(axis=0), dtype=torch.float32))
            region_ambiguous_rows.append(_is_ambiguous_group(patch_annotation_y, patch_ids) if has_patch_annotation else False)
            if region_annotation_rows is not None:
                region_annotation_rows.append(_majority_label(patch_annotation_y, patch_ids) if has_patch_annotation else torch.tensor(-1, dtype=torch.long))
            for field, rows in extra_region_rows.items():
                rows.append(_majority_label(extra_patch_y[field], patch_ids, unknown_id=extra_unknown_ids.get(field)))
            new_p2r_src_parts.append(patch_ids)
            new_p2r_dst_parts.append(np.full(patch_ids.shape, next_id, dtype=np.int64))
            next_id += 1

    if not any_split:
        return

    n_regions_new = next_id
    region_x_new = torch.stack(region_x_rows)
    region_feat_new = torch.stack(region_feat_rows)
    region_centroid_new = torch.stack(region_centroid_rows)
    region_annotation_new = torch.stack(region_annotation_rows) if region_annotation_rows is not None else None
    region_ambiguous_new = torch.tensor(region_ambiguous_rows, dtype=torch.bool)

    new_p2r_src = np.concatenate(new_p2r_src_parts)
    new_p2r_dst = np.concatenate(new_p2r_dst_parts)
    p2r_edge_new = torch.tensor(np.stack([new_p2r_src, new_p2r_dst]), dtype=torch.long)
    r2p_edge_new = torch.tensor(np.stack([new_p2r_dst, new_p2r_src]), dtype=torch.long)

    k = max(1, min(int(k_region), n_regions_new - 1))
    if n_regions_new > 1:
        centroid_np = region_centroid_new.numpy()
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(centroid_np)
        _, neighbor_idx = nbrs.kneighbors(centroid_np)
        r2r_src = np.repeat(np.arange(n_regions_new), k)
        r2r_dst = neighbor_idx[:, 1:1 + k].reshape(-1)
        r2r_edge_new = torch.tensor(np.stack([r2r_src, r2r_dst]), dtype=torch.long)
    else:
        r2r_edge_new = torch.zeros((2, 0), dtype=torch.long)

    r2s_src = np.arange(n_regions_new, dtype=np.int64)
    r2s_dst = np.zeros((n_regions_new,), dtype=np.int64)
    r2s_edge_new = torch.tensor(np.stack([r2s_src, r2s_dst]), dtype=torch.long)
    s2r_edge_new = torch.tensor(np.stack([r2s_dst, r2s_src]), dtype=torch.long)

    region_store.num_nodes = n_regions_new
    region_store.x = region_x_new
    region_store.feat = region_feat_new
    region_store.centroid = region_centroid_new
    if hasattr(region_store, "spatial_x"):
        region_store.spatial_x = region_centroid_new.clone()
    if region_annotation_new is not None:
        region_store.annotation_y = region_annotation_new
    region_store.label_ambiguous = region_ambiguous_new
    for field, rows in extra_region_rows.items():
        setattr(region_store, f"{field}_y", torch.stack(rows))

    data[("patch", "in", "region")].edge_index = p2r_edge_new
    data[("region", "has", "patch")].edge_index = r2p_edge_new
    data[("region", "near", "region")].edge_index = r2r_edge_new
    data[("region", "part_of", "slide")].edge_index = r2s_edge_new
    data[("slide", "contains", "region")].edge_index = s2r_edge_new

    patch_region_labels_new = np.full((n_patches,), -1, dtype=np.int64)
    patch_region_labels_new[new_p2r_src] = new_p2r_dst
    region_feat_np = region_feat_new.detach().cpu().numpy()
    region_centroid_np = region_centroid_new.detach().cpu().numpy()
    region_semantic_ids = (
        region_annotation_new.detach().cpu().numpy().astype(np.int64)
        if region_annotation_new is not None
        else np.arange(n_regions_new, dtype=np.int64)
    )
    patch_feat_np = patch_feat.detach().cpu().numpy()

    slide_store = data["slide"] if "slide" in data.node_types else None
    slide_centroid_np = (
        slide_store.centroid.detach().cpu().numpy()
        if slide_store is not None and torch.is_tensor(getattr(slide_store, "centroid", None))
        else np.zeros((1, region_centroid_np.shape[1]), dtype=np.float64)
    )
    slide_feat_np = (
        slide_store.feat.detach().cpu().numpy()
        if slide_store is not None and torch.is_tensor(getattr(slide_store, "feat", None))
        else None
    )

    data[("patch", "in", "region")].edge_feat = torch.tensor(
        build_edge_features(
            src_xy=patch_pos_np, dst_xy=region_centroid_np,
            src_idx=new_p2r_src, dst_idx=new_p2r_dst,
            src_semantic_ids=patch_region_labels_new, dst_semantic_ids=np.arange(n_regions_new, dtype=np.int64),
            src_feat=patch_feat_np, dst_feat=region_feat_np,
        ),
        dtype=torch.float32,
    )
    data[("region", "has", "patch")].edge_feat = torch.tensor(
        build_edge_features(
            src_xy=region_centroid_np, dst_xy=patch_pos_np,
            src_idx=new_p2r_dst, dst_idx=new_p2r_src,
            src_semantic_ids=np.arange(n_regions_new, dtype=np.int64), dst_semantic_ids=patch_region_labels_new,
            src_feat=region_feat_np, dst_feat=patch_feat_np,
        ),
        dtype=torch.float32,
    )

    r2r_src_np = r2r_edge_new[0].numpy()
    r2r_dst_np = r2r_edge_new[1].numpy()
    r2r_dist = np.linalg.norm(region_centroid_np[r2r_src_np] - region_centroid_np[r2r_dst_np], axis=1) if r2r_src_np.size else np.zeros((0,), dtype=np.float64)
    data[("region", "near", "region")].edge_attr = torch.tensor(r2r_dist[:, None], dtype=torch.float32)
    data[("region", "near", "region")].edge_feat = torch.tensor(
        build_edge_features(
            src_xy=region_centroid_np, dst_xy=region_centroid_np,
            src_idx=r2r_src_np, dst_idx=r2r_dst_np, dist=r2r_dist,
            src_semantic_ids=region_semantic_ids, dst_semantic_ids=region_semantic_ids,
            src_feat=region_feat_np, dst_feat=region_feat_np,
        ),
        dtype=torch.float32,
    )

    data[("region", "part_of", "slide")].edge_feat = torch.tensor(
        build_edge_features(
            src_xy=region_centroid_np, dst_xy=slide_centroid_np,
            src_idx=r2s_src, dst_idx=r2s_dst,
            src_semantic_ids=np.arange(n_regions_new, dtype=np.int64), dst_semantic_ids=np.zeros((1,), dtype=np.int64),
            src_feat=region_feat_np, dst_feat=slide_feat_np,
        ),
        dtype=torch.float32,
    )
    data[("slide", "contains", "region")].edge_feat = torch.tensor(
        build_edge_features(
            src_xy=slide_centroid_np, dst_xy=region_centroid_np,
            src_idx=r2s_dst, dst_idx=r2s_src,
            src_semantic_ids=np.zeros((1,), dtype=np.int64), dst_semantic_ids=np.arange(n_regions_new, dtype=np.int64),
            src_feat=slide_feat_np, dst_feat=region_feat_np,
        ),
        dtype=torch.float32,
    )


def merge_small_regions(
    data,
    min_patches_per_region: int = 0,
) -> None:
    """Merges regions with fewer than `min_patches_per_region` patches into their nearest same-label neighbor, in place."""
    if min_patches_per_region <= 0:
        return
    if not hasattr(data, "node_types") or "region" not in data.node_types or "patch" not in data.node_types:
        return

    region_store = data["region"]
    patch_store = data["patch"]

    n_patches = int(getattr(patch_store, "num_nodes", 0) or (patch_store.x.size(0) if hasattr(patch_store, "x") else 0))
    n_regions_old = int(getattr(region_store, "num_nodes", 0) or (region_store.x.size(0) if hasattr(region_store, "x") else 0))
    if n_patches == 0 or n_regions_old <= 1:
        return

    if ("patch", "in", "region") not in data.edge_types:
        return
    p2r_edge_index = data[("patch", "in", "region")].edge_index
    if not torch.is_tensor(p2r_edge_index) or p2r_edge_index.numel() == 0:
        return

    patch_pos = getattr(patch_store, "centroid", None)
    if not torch.is_tensor(patch_pos):
        patch_pos = getattr(patch_store, "topleft", None)
    if not torch.is_tensor(patch_pos) or patch_pos.size(0) != n_patches:
        return

    patch_pos_np = patch_pos.detach().cpu().numpy().astype(np.float64)
    p_src = p2r_edge_index[0].long().cpu().numpy()
    p_dst = p2r_edge_index[1].long().cpu().numpy()

    old_annotation_y = getattr(region_store, "annotation_y", None)
    has_region_annotation = torch.is_tensor(old_annotation_y) and old_annotation_y.numel() == n_regions_old

    groups = []
    for r in range(n_regions_old):
        members = p_src[p_dst == r]
        if members.size == 0:
            continue
        label = int(old_annotation_y[r].item()) if has_region_annotation else -1
        groups.append({"members": members, "label": label})

    if len(groups) <= 1:
        return

    def _group_centroid(group: dict) -> np.ndarray:
        return patch_pos_np[group["members"]].mean(axis=0)

    merged_any = False
    for _iteration in range(24):
        sizes = np.array([g["members"].size for g in groups])
        small_idx = np.where(sizes < min_patches_per_region)[0]
        if small_idx.size == 0:
            break
        small_idx = small_idx[np.argsort(sizes[small_idx])]

        centroids = np.stack([_group_centroid(g) for g in groups])
        labels = np.array([g["label"] for g in groups])

        to_remove: set = set()
        did_merge_this_round = False
        for i in small_idx.tolist():
            if i in to_remove:
                continue
            lbl = labels[i]
            candidates = [j for j in range(len(groups)) if j != i and j not in to_remove and labels[j] == lbl]
            if not candidates:
                continue
            dists = np.linalg.norm(centroids[candidates] - centroids[i], axis=1)
            target = candidates[int(np.argmin(dists))]
            groups[target]["members"] = np.concatenate([groups[target]["members"], groups[i]["members"]])
            to_remove.add(i)
            did_merge_this_round = True
            merged_any = True

        if not did_merge_this_round:
            break
        groups = [g for idx, g in enumerate(groups) if idx not in to_remove]

    if not merged_any:
        return

    n_regions_new = len(groups)
    old_x = region_store.x
    old_feat = getattr(region_store, "feat", old_x)
    patch_x = patch_store.x
    patch_feat = getattr(patch_store, "feat", patch_x)
    patch_annotation_y = getattr(patch_store, "annotation_y", None)
    has_patch_annotation = torch.is_tensor(patch_annotation_y) and patch_annotation_y.numel() == n_patches

    extra_fields = _extra_patch_region_y_fields(region_store, patch_store)
    extra_patch_y = {}
    for field in extra_fields:
        patch_t = _valid_y_tensor(patch_store, field, n_patches)
        if patch_t is not None:
            extra_patch_y[field] = patch_t
    extra_region_rows = {field: [] for field in extra_patch_y}
    extra_unknown_ids = _extra_field_unknown_ids(data, list(extra_patch_y))

    region_x_rows = []
    region_feat_rows = []
    region_centroid_rows = []
    region_annotation_rows = [] if has_region_annotation else None
    region_ambiguous_rows: List[bool] = []
    new_p2r_src_parts = []
    new_p2r_dst_parts = []

    for new_r, group in enumerate(groups):
        patch_ids = group["members"]
        region_x_rows.append(patch_x[patch_ids].mean(dim=0))
        region_feat_rows.append(patch_feat[patch_ids].mean(dim=0))
        region_centroid_rows.append(torch.as_tensor(patch_pos_np[patch_ids].mean(axis=0), dtype=torch.float32))
        if region_annotation_rows is not None:
            region_annotation_rows.append(
                _majority_label(patch_annotation_y, patch_ids) if has_patch_annotation else torch.tensor(group["label"], dtype=torch.long)
            )
        region_ambiguous_rows.append(_is_ambiguous_group(patch_annotation_y, patch_ids) if has_patch_annotation else False)
        for field, rows in extra_region_rows.items():
            rows.append(_majority_label(extra_patch_y[field], patch_ids, unknown_id=extra_unknown_ids.get(field)))
        new_p2r_src_parts.append(patch_ids)
        new_p2r_dst_parts.append(np.full(patch_ids.shape, new_r, dtype=np.int64))

    region_x_new = torch.stack(region_x_rows)
    region_feat_new = torch.stack(region_feat_rows)
    region_centroid_new = torch.stack(region_centroid_rows)
    region_annotation_new = torch.stack(region_annotation_rows) if region_annotation_rows is not None else None
    region_ambiguous_new = torch.tensor(region_ambiguous_rows, dtype=torch.bool)

    new_p2r_src = np.concatenate(new_p2r_src_parts)
    new_p2r_dst = np.concatenate(new_p2r_dst_parts)
    p2r_edge_new = torch.tensor(np.stack([new_p2r_src, new_p2r_dst]), dtype=torch.long)
    r2p_edge_new = torch.tensor(np.stack([new_p2r_dst, new_p2r_src]), dtype=torch.long)

    k = max(1, min(5, n_regions_new - 1))
    if n_regions_new > 1:
        centroid_np = region_centroid_new.numpy()
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(centroid_np)
        _, neighbor_idx = nbrs.kneighbors(centroid_np)
        r2r_src = np.repeat(np.arange(n_regions_new), k)
        r2r_dst = neighbor_idx[:, 1:1 + k].reshape(-1)
        r2r_edge_new = torch.tensor(np.stack([r2r_src, r2r_dst]), dtype=torch.long)
    else:
        r2r_edge_new = torch.zeros((2, 0), dtype=torch.long)

    r2s_src = np.arange(n_regions_new, dtype=np.int64)
    r2s_dst = np.zeros((n_regions_new,), dtype=np.int64)
    r2s_edge_new = torch.tensor(np.stack([r2s_src, r2s_dst]), dtype=torch.long)
    s2r_edge_new = torch.tensor(np.stack([r2s_dst, r2s_src]), dtype=torch.long)

    region_store.num_nodes = n_regions_new
    region_store.x = region_x_new
    region_store.feat = region_feat_new
    region_store.centroid = region_centroid_new
    if hasattr(region_store, "spatial_x"):
        region_store.spatial_x = region_centroid_new.clone()
    if region_annotation_new is not None:
        region_store.annotation_y = region_annotation_new
    region_store.label_ambiguous = region_ambiguous_new
    for field, rows in extra_region_rows.items():
        setattr(region_store, f"{field}_y", torch.stack(rows))

    data[("patch", "in", "region")].edge_index = p2r_edge_new
    data[("region", "has", "patch")].edge_index = r2p_edge_new
    data[("region", "near", "region")].edge_index = r2r_edge_new
    data[("region", "part_of", "slide")].edge_index = r2s_edge_new
    data[("slide", "contains", "region")].edge_index = s2r_edge_new

    patch_region_labels_new = np.full((n_patches,), -1, dtype=np.int64)
    patch_region_labels_new[new_p2r_src] = new_p2r_dst
    region_feat_np = region_feat_new.detach().cpu().numpy()
    region_centroid_np = region_centroid_new.detach().cpu().numpy()
    region_semantic_ids = (
        region_annotation_new.detach().cpu().numpy().astype(np.int64)
        if region_annotation_new is not None
        else np.arange(n_regions_new, dtype=np.int64)
    )
    patch_feat_np = patch_feat.detach().cpu().numpy()

    slide_store = data["slide"] if "slide" in data.node_types else None
    slide_centroid_np = (
        slide_store.centroid.detach().cpu().numpy()
        if slide_store is not None and torch.is_tensor(getattr(slide_store, "centroid", None))
        else np.zeros((1, region_centroid_np.shape[1]), dtype=np.float64)
    )
    slide_feat_np = (
        slide_store.feat.detach().cpu().numpy()
        if slide_store is not None and torch.is_tensor(getattr(slide_store, "feat", None))
        else None
    )

    data[("patch", "in", "region")].edge_feat = torch.tensor(
        build_edge_features(
            src_xy=patch_pos_np, dst_xy=region_centroid_np,
            src_idx=new_p2r_src, dst_idx=new_p2r_dst,
            src_semantic_ids=patch_region_labels_new, dst_semantic_ids=np.arange(n_regions_new, dtype=np.int64),
            src_feat=patch_feat_np, dst_feat=region_feat_np,
        ),
        dtype=torch.float32,
    )
    data[("region", "has", "patch")].edge_feat = torch.tensor(
        build_edge_features(
            src_xy=region_centroid_np, dst_xy=patch_pos_np,
            src_idx=new_p2r_dst, dst_idx=new_p2r_src,
            src_semantic_ids=np.arange(n_regions_new, dtype=np.int64), dst_semantic_ids=patch_region_labels_new,
            src_feat=region_feat_np, dst_feat=patch_feat_np,
        ),
        dtype=torch.float32,
    )

    r2r_src_np = r2r_edge_new[0].numpy()
    r2r_dst_np = r2r_edge_new[1].numpy()
    r2r_dist = np.linalg.norm(region_centroid_np[r2r_src_np] - region_centroid_np[r2r_dst_np], axis=1) if r2r_src_np.size else np.zeros((0,), dtype=np.float64)
    data[("region", "near", "region")].edge_attr = torch.tensor(r2r_dist[:, None], dtype=torch.float32)
    data[("region", "near", "region")].edge_feat = torch.tensor(
        build_edge_features(
            src_xy=region_centroid_np, dst_xy=region_centroid_np,
            src_idx=r2r_src_np, dst_idx=r2r_dst_np, dist=r2r_dist,
            src_semantic_ids=region_semantic_ids, dst_semantic_ids=region_semantic_ids,
            src_feat=region_feat_np, dst_feat=region_feat_np,
        ),
        dtype=torch.float32,
    )

    data[("region", "part_of", "slide")].edge_feat = torch.tensor(
        build_edge_features(
            src_xy=region_centroid_np, dst_xy=slide_centroid_np,
            src_idx=r2s_src, dst_idx=r2s_dst,
            src_semantic_ids=np.arange(n_regions_new, dtype=np.int64), dst_semantic_ids=np.zeros((1,), dtype=np.int64),
            src_feat=region_feat_np, dst_feat=slide_feat_np,
        ),
        dtype=torch.float32,
    )
    data[("slide", "contains", "region")].edge_feat = torch.tensor(
        build_edge_features(
            src_xy=slide_centroid_np, dst_xy=region_centroid_np,
            src_idx=r2s_dst, dst_idx=r2s_src,
            src_semantic_ids=np.zeros((1,), dtype=np.int64), dst_semantic_ids=np.arange(n_regions_new, dtype=np.int64),
            src_feat=slide_feat_np, dst_feat=region_feat_np,
        ),
        dtype=torch.float32,
    )


def _extra_patch_region_y_fields(region_store, patch_store) -> List[str]:
    region_keys = set(region_store.keys()) if hasattr(region_store, "keys") else set()
    patch_keys = set(patch_store.keys()) if hasattr(patch_store, "keys") else set()
    fields = []
    for key in region_keys:
        if not key.endswith("_y") or key == "annotation_y":
            continue
        if key in patch_keys:
            fields.append(key[: -len("_y")])
    return fields


def _valid_y_tensor(store, field: str, n_expected: int) -> Optional[torch.Tensor]:
    t = getattr(store, f"{field}_y", None)
    return t if torch.is_tensor(t) and t.numel() == n_expected else None


def _majority_label(patch_annotation_y: torch.Tensor, patch_ids: np.ndarray, unknown_id: Optional[int] = None) -> torch.Tensor:
    labels = patch_annotation_y[patch_ids]
    labels = labels[labels >= 0]
    if unknown_id is not None:
        labels = labels[labels != unknown_id]
    if labels.numel() == 0:
        return torch.tensor(-1 if unknown_id is None else unknown_id, dtype=torch.long)
    values, counts = torch.unique(labels, return_counts=True)
    return values[torch.argmax(counts)]


def _extra_field_unknown_ids(data, fields: List[str]) -> dict:
    meta = getattr(data, "meta", None)
    if not isinstance(meta, dict):
        return {field: None for field in fields}
    result = {}
    for field in fields:
        vocab = meta.get(f"{field}_label_vocab") or []
        result[field] = vocab.index("unknown") if "unknown" in vocab else None
    return result


def _is_ambiguous_group(patch_annotation_y: torch.Tensor, patch_ids: np.ndarray) -> bool:
    labels = patch_annotation_y[patch_ids]
    has_unlabeled = bool((labels < 0).any().item())
    labeled = labels[labels >= 0]
    if labeled.numel() == 0:
        return False
    distinct = torch.unique(labeled)
    return bool(distinct.numel() > 1) or has_unlabeled


def _recursive_bisect(xy: np.ndarray, target: int, _depth: int = 0, _max_depth: int = 24) -> Iterator[np.ndarray]:
    n = xy.shape[0]
    if n <= target or _depth >= _max_depth:
        yield np.arange(n)
        return

    ranges = xy.max(axis=0) - xy.min(axis=0)
    axis = int(np.argmax(ranges))
    median = np.median(xy[:, axis])
    left_mask = xy[:, axis] <= median

    if not left_mask.any() or left_mask.all():
        yield np.arange(n)
        return

    left_idx = np.where(left_mask)[0]
    right_idx = np.where(~left_mask)[0]
    for sub in _recursive_bisect(xy[left_idx], target, _depth + 1, _max_depth):
        yield left_idx[sub]
    for sub in _recursive_bisect(xy[right_idx], target, _depth + 1, _max_depth):
        yield right_idx[sub]
