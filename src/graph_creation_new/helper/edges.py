from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.graph_creation_new.enums import EdgeConstructionMode


def _empty_edges() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.float32),
    )


def knn_edges(xy: np.ndarray, k: int, radius_px: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build k-nearest-neighbor edges between points, optionally capped by a distance radius."""
    n_nodes = int(xy.shape[0])
    if n_nodes <= 1:
        return _empty_edges()

    k_eff = min(max(2, int(k) + 1), n_nodes)
    tree = cKDTree(xy)
    distances, indices = tree.query(xy, k=k_eff)

    src: list[int] = []
    dst: list[int] = []
    dist: list[float] = []

    for node_index in range(n_nodes):
        neighbors = indices[node_index][1:]
        neighbor_distances = distances[node_index][1:]
        for neighbor_index, neighbor_distance in zip(neighbors.tolist(), neighbor_distances.tolist()):
            if radius_px is not None and float(neighbor_distance) > float(radius_px):
                continue
            src.append(node_index)
            dst.append(int(neighbor_index))
            dist.append(float(neighbor_distance))

    return (
        np.asarray(src, dtype=np.int64),
        np.asarray(dst, dtype=np.int64),
        np.asarray(dist, dtype=np.float32),
    )


def distance_edges(xy: np.ndarray, radius_px: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build edges between all point pairs within a fixed distance radius."""
    n_nodes = int(xy.shape[0])
    if n_nodes <= 1:
        return _empty_edges()
    if radius_px is None or float(radius_px) <= 0:
        raise ValueError("distance_edges requires radius_px > 0")

    tree = cKDTree(xy)
    neighborhoods = tree.query_ball_point(xy, r=float(radius_px))

    src: list[int] = []
    dst: list[int] = []
    dist: list[float] = []

    for node_index, neighbors in enumerate(neighborhoods):
        for neighbor_index in neighbors:
            if int(neighbor_index) == node_index:
                continue
            d = float(np.linalg.norm(xy[node_index] - xy[int(neighbor_index)]))
            src.append(node_index)
            dst.append(int(neighbor_index))
            dist.append(d)

    if not src:
        return _empty_edges()

    return (
        np.asarray(src, dtype=np.int64),
        np.asarray(dst, dtype=np.int64),
        np.asarray(dist, dtype=np.float32),
    )


def semantic_equality_edges(
    xy: np.ndarray,
    semantic_ids: np.ndarray,
    k: Optional[int] = None,
    radius_px: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build edges only between nodes that share the same semantic id, within a group."""
    n_nodes = int(xy.shape[0])
    if n_nodes <= 1:
        return _empty_edges()

    ids = np.asarray(semantic_ids, dtype=np.int64).reshape(-1)
    if ids.shape[0] != n_nodes:
        raise ValueError("semantic_ids must have one value per node")

    src: list[int] = []
    dst: list[int] = []
    dist: list[float] = []

    valid_semantic_ids = np.unique(ids[ids >= 0])
    for semantic_id in valid_semantic_ids.tolist():
        group_nodes = np.where(ids == int(semantic_id))[0]
        group_size = int(group_nodes.shape[0])
        if group_size <= 1:
            continue

        group_xy = xy[group_nodes]
        group_tree = cKDTree(group_xy)

        if k is not None:
            k_eff = min(max(2, int(k) + 1), group_size)
            distances, indices = group_tree.query(group_xy, k=k_eff)
            for local_index in range(group_size):
                neighbors = indices[local_index][1:]
                neighbor_distances = distances[local_index][1:]
                for neighbor_local_index, neighbor_distance in zip(neighbors.tolist(), neighbor_distances.tolist()):
                    if radius_px is not None and float(neighbor_distance) > float(radius_px):
                        continue
                    src.append(int(group_nodes[local_index]))
                    dst.append(int(group_nodes[int(neighbor_local_index)]))
                    dist.append(float(neighbor_distance))
            continue

        if radius_px is not None:
            neighborhoods = group_tree.query_ball_point(group_xy, r=float(radius_px))
            for local_index, neighbors in enumerate(neighborhoods):
                for neighbor_local_index in neighbors:
                    if int(neighbor_local_index) == local_index:
                        continue
                    d = float(np.linalg.norm(group_xy[local_index] - group_xy[int(neighbor_local_index)]))
                    src.append(int(group_nodes[local_index]))
                    dst.append(int(group_nodes[int(neighbor_local_index)]))
                    dist.append(d)
            continue

        for left_local in range(group_size):
            for right_local in range(group_size):
                if left_local == right_local:
                    continue
                d = float(np.linalg.norm(group_xy[left_local] - group_xy[right_local]))
                src.append(int(group_nodes[left_local]))
                dst.append(int(group_nodes[right_local]))
                dist.append(d)

    if not src:
        return _empty_edges()

    return (
        np.asarray(src, dtype=np.int64),
        np.asarray(dst, dtype=np.int64),
        np.asarray(dist, dtype=np.float32),
    )


EDGE_FEAT_DIM = 8

# Column layout of build_edge_features' output -- see that function's docstring for the
# per-column meaning. Named here so normalize_edge_feat_for_model below (and anything else that
# needs to address a specific column) doesn't repeat magic indices.
EDGE_FEAT_COL_DIST = 0
EDGE_FEAT_COL_INV_DIST = 1
EDGE_FEAT_COL_DELTA_X = 2
EDGE_FEAT_COL_DELTA_Y = 3
EDGE_FEAT_COL_SAME_SEMANTIC = 4
EDGE_FEAT_COL_SEMANTIC_VALID = 5
EDGE_FEAT_COL_COSINE_SIM = 6
EDGE_FEAT_COL_FEAT_VALID = 7

# Pixel scale used to bring dist/delta_x/delta_y down to roughly the same O(1) magnitude as
# this graph's other edge_feat columns (cosine_sim in [-1,1], the *_valid/same_semantic flags
# in {0,1}) -- see normalize_edge_feat_for_model's docstring for why this exists at all.
_SPATIAL_FEAT_SCALE_PX = 1000.0


def normalize_edge_feat_for_model(edge_attr):
    """Rescale edge_feat's spatial columns down to roughly the same magnitude as its other columns."""
    if edge_attr is None:
        return edge_attr
    scaled = edge_attr.clone()
    scaled[:, EDGE_FEAT_COL_DIST] = scaled[:, EDGE_FEAT_COL_DIST] / _SPATIAL_FEAT_SCALE_PX
    scaled[:, EDGE_FEAT_COL_INV_DIST] = scaled[:, EDGE_FEAT_COL_INV_DIST] * _SPATIAL_FEAT_SCALE_PX
    scaled[:, EDGE_FEAT_COL_DELTA_X] = scaled[:, EDGE_FEAT_COL_DELTA_X] / _SPATIAL_FEAT_SCALE_PX
    scaled[:, EDGE_FEAT_COL_DELTA_Y] = scaled[:, EDGE_FEAT_COL_DELTA_Y] / _SPATIAL_FEAT_SCALE_PX
    return scaled


def _cosine_similarity_columns(
    n_edges: int,
    src_idx_arr: np.ndarray,
    dst_idx_arr: np.ndarray,
    src_feat: Optional[np.ndarray],
    dst_feat: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    if src_feat is None or dst_feat is None:
        zeros = np.zeros((n_edges,), dtype=np.float32)
        return zeros, zeros

    src_feat_arr = np.asarray(src_feat, dtype=np.float32)
    dst_feat_arr = np.asarray(dst_feat, dtype=np.float32)

    cosine_sim = np.empty((n_edges,), dtype=np.float32)
    feat_valid = np.empty((n_edges,), dtype=np.float32)

    target_bytes_per_array = 64 * 1024 * 1024
    dim = max(1, int(src_feat_arr.shape[1]))
    chunk_size = max(1, target_bytes_per_array // (dim * 4))

    for start in range(0, n_edges, chunk_size):
        stop = min(start + chunk_size, n_edges)
        src_vecs = src_feat_arr[src_idx_arr[start:stop]]
        dst_vecs = dst_feat_arr[dst_idx_arr[start:stop]]

        src_norm = np.linalg.norm(src_vecs, axis=1)
        dst_norm = np.linalg.norm(dst_vecs, axis=1)
        denom = np.maximum(src_norm * dst_norm, 1e-8)
        sim = np.einsum("ij,ij->i", src_vecs, dst_vecs) / denom
        valid = (src_norm > 1e-8) & (dst_norm > 1e-8)

        cosine_sim[start:stop] = np.clip(sim, -1.0, 1.0) * valid
        feat_valid[start:stop] = valid.astype(np.float32)

    return cosine_sim, feat_valid


def build_edge_features(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    *,
    dist: Optional[np.ndarray] = None,
    src_semantic_ids: Optional[np.ndarray] = None,
    dst_semantic_ids: Optional[np.ndarray] = None,
    src_feat: Optional[np.ndarray] = None,
    dst_feat: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the per-edge feature matrix (distance, direction, semantic and embedding-similarity columns)."""
    src_idx_arr = np.asarray(src_idx, dtype=np.int64).reshape(-1)
    dst_idx_arr = np.asarray(dst_idx, dtype=np.int64).reshape(-1)
    n_edges = int(src_idx_arr.shape[0])

    if int(dst_idx_arr.shape[0]) != n_edges:
        raise ValueError("src_idx and dst_idx must have same length")
    if n_edges == 0:
        return np.zeros((0, EDGE_FEAT_DIM), dtype=np.float32)

    src_xy_arr = np.asarray(src_xy, dtype=np.float32)
    dst_xy_arr = np.asarray(dst_xy, dtype=np.float32)

    src_points = src_xy_arr[src_idx_arr]
    dst_points = dst_xy_arr[dst_idx_arr]
    delta = dst_points - src_points

    if dist is None:
        dist_arr = np.linalg.norm(delta, axis=1).astype(np.float32)
    else:
        dist_arr = np.asarray(dist, dtype=np.float32).reshape(-1)
        if int(dist_arr.shape[0]) != n_edges:
            raise ValueError("dist must match number of edges")

    inv_dist = 1.0 / np.maximum(dist_arr, 1e-6)

    same_semantic = np.zeros((n_edges,), dtype=np.float32)
    semantic_valid = np.zeros((n_edges,), dtype=np.float32)
    if src_semantic_ids is not None and dst_semantic_ids is not None:
        src_ids = np.asarray(src_semantic_ids, dtype=np.int64).reshape(-1)[src_idx_arr]
        dst_ids = np.asarray(dst_semantic_ids, dtype=np.int64).reshape(-1)[dst_idx_arr]
        valid_mask = (src_ids >= 0) & (dst_ids >= 0)
        semantic_valid = valid_mask.astype(np.float32)
        same_semantic = (src_ids == dst_ids).astype(np.float32) * semantic_valid

    cosine_sim, feat_valid = _cosine_similarity_columns(
        n_edges, src_idx_arr, dst_idx_arr, src_feat, dst_feat
    )

    return np.stack(
        [
            dist_arr,
            inv_dist.astype(np.float32),
            delta[:, 0].astype(np.float32),
            delta[:, 1].astype(np.float32),
            same_semantic,
            semantic_valid,
            cosine_sim,
            feat_valid,
        ],
        axis=1,
    ).astype(np.float32)


def build_edges(
    xy: np.ndarray,
    mode: EdgeConstructionMode,
    *,
    k: Optional[int] = None,
    radius_px: Optional[float] = None,
    semantic_ids: Optional[np.ndarray] = None,
    relation_name: str = "edge_relation",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch to the edge-construction function matching the given mode."""
    if mode == EdgeConstructionMode.KNN:
        if k is None:
            raise ValueError(f"KNN mode for {relation_name} requires k")
        return knn_edges(xy=xy, k=int(k), radius_px=radius_px)

    if mode == EdgeConstructionMode.DISTANCE:
        if radius_px is None or float(radius_px) <= 0:
            raise ValueError(f"Distance mode for {relation_name} requires radius_px > 0")
        return distance_edges(xy=xy, radius_px=float(radius_px))

    if mode == EdgeConstructionMode.SEMANTIC_EQUALITY:
        if semantic_ids is None:
            raise ValueError(f"Semantic equality mode for {relation_name} requires semantic_ids")
        return semantic_equality_edges(
            xy=xy,
            semantic_ids=semantic_ids,
            k=int(k) if k is not None else None,
            radius_px=radius_px,
        )

    if mode == EdgeConstructionMode.TARGETHEAD_INTERSECTION:
        raise NotImplementedError(
            f"Edge mode '{mode.value}' for {relation_name} is not implemented yet"
        )

    raise ValueError(f"Unsupported edge construction mode for {relation_name}: {mode}")
