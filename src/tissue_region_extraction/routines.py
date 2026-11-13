from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class TissueRegionTrainingConstraints:
    """Defines only framework conditions; training is done by existing pipelines."""

    architecture: str = "tissue_region_net"
    model_type: str = "gnn"
    input_type: str = "graph"
    embedding_type: str = "pov-gigapath"
    cached: bool = True
    mil: bool = False
    label_mode: str = "binary"
    num_classes: int = 1

    hidden_dim: int = 320
    num_layers: int = 4
    heads: int = 4
    dropout_rate: float = 0.20
    edge_dropout_p: float = 0.05
    region_aux_weight: float = 0.24
    region_aux_pos_weight_cap: float = 6.0

    epochs: int = 80
    batch_size: int = 1
    val_batch_size: int = 1
    grad_accum_steps: int = 4
    mixed_precision: bool = False
    auto_pos_weight: bool = True
    balanced_sampling: bool = True
    strict_graph_name_match: bool = False


DEFAULT_CONSTRAINTS = TissueRegionTrainingConstraints()


def build_training_overrides(
    constraints: TissueRegionTrainingConstraints = DEFAULT_CONSTRAINTS,
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a config override dictionary for existing train_model pipelines."""

    payload: Dict[str, Any] = {
        "architecture": {
            "name": constraints.architecture,
            "pretrained": False,
            "feature_extraction": False,
        },
        "label_mode": constraints.label_mode,
        "num_classes": int(constraints.num_classes),
        "hidden_dim": int(constraints.hidden_dim),
        "num_layers": int(constraints.num_layers),
        "heads": int(constraints.heads),
        "dropout_rate": float(constraints.dropout_rate),
        "edge_dropout_p": float(constraints.edge_dropout_p),
        "region_aux_weight": float(constraints.region_aux_weight),
        "region_aux_pos_weight_cap": float(constraints.region_aux_pos_weight_cap),
        "epochs": int(constraints.epochs),
        "batch_size": int(constraints.batch_size),
        "val_batch_size": int(constraints.val_batch_size),
        "grad_accum_steps": int(constraints.grad_accum_steps),
        "mixed_precision": bool(constraints.mixed_precision),
        "auto_pos_weight": bool(constraints.auto_pos_weight),
        "balanced_sampling": bool(constraints.balanced_sampling),
        "strict_graph_name_match": bool(constraints.strict_graph_name_match),
    }
    if extra_overrides:
        payload.update(extra_overrides)
    return payload


def build_train_model_args(
    task: str = "lymph",
    dataset: str = "camelyon",
    pipeline: Optional[str] = None,
    split_phase: Optional[str] = None,
    seed: int = 42,
    constraints: TissueRegionTrainingConstraints = DEFAULT_CONSTRAINTS,
) -> List[str]:
    """Return CLI args for src/entrypoint/train_model.py using existing routines."""

    args = [
        "--task",
        str(task),
        "--dataset",
        str(dataset),
        "--model_type",
        str(constraints.model_type),
        "--architecture",
        str(constraints.architecture),
        "--input_type",
        str(constraints.input_type),
        "--embedding_type",
        str(constraints.embedding_type),
        "--seed",
        str(int(seed)),
    ]

    if constraints.cached:
        args.append("--cached")
    if constraints.mil:
        args.append("--mil")
    if split_phase:
        args.extend(["--split-phase", str(split_phase)])
    if pipeline:
        args.extend(["--pipeline", str(pipeline)])

    return args


def _extract_region_probs(model_out: Any) -> torch.Tensor:
    if torch.is_tensor(model_out):
        return torch.sigmoid(model_out.view(-1))
    if isinstance(model_out, dict):
        if "region_probs" in model_out and torch.is_tensor(model_out["region_probs"]):
            return model_out["region_probs"].view(-1)
        if "region_logits" in model_out and torch.is_tensor(model_out["region_logits"]):
            return torch.sigmoid(model_out["region_logits"].view(-1))
    raise ValueError("Model output does not contain region logits/probabilities.")


@torch.no_grad()
def infer_region_probabilities(
    data: Any,
    model: torch.nn.Module,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Run the model on data and return per-region tumor probabilities."""
    infer_device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(infer_device)
    model.eval()

    if hasattr(data, "to"):
        data = data.to(infer_device)

    model_out = model(data)
    probs = _extract_region_probs(model_out).detach().cpu()
    return probs


def _get_region_store(data: Any) -> Any:
    if not hasattr(data, "node_types") or "region" not in data.node_types:
        raise ValueError("Expected HeteroData with 'region' node type.")
    return data["region"]


def _get_region_centroids(region_store: Any, n_regions: int) -> np.ndarray:
    centroid = getattr(region_store, "centroid", None)
    if torch.is_tensor(centroid) and centroid.dim() == 2 and centroid.size(0) == n_regions:
        arr = centroid.detach().cpu().numpy().astype(np.float32)
        return arr[:, :2] if arr.shape[1] >= 2 else np.pad(arr, ((0, 0), (0, max(0, 2 - arr.shape[1]))))[:, :2]

    x = getattr(region_store, "x", None)
    if torch.is_tensor(x) and x.dim() == 2 and x.size(0) == n_regions and x.size(1) >= 2:
        return x[:, :2].detach().cpu().numpy().astype(np.float32)

    return np.zeros((n_regions, 2), dtype=np.float32)


def _get_region_edge_index(data: Any, n_regions: int) -> np.ndarray:
    candidates = [
        ("region", "near", "region"),
        ("region", "spatial", "region"),
    ]

    for edge_type in candidates:
        if hasattr(data, "edge_types") and edge_type in data.edge_types:
            edge_index = data[edge_type].edge_index
            if torch.is_tensor(edge_index) and edge_index.dim() == 2 and edge_index.size(0) == 2:
                arr = edge_index.detach().cpu().numpy().astype(np.int64)
                valid = (
                    (arr[0] >= 0)
                    & (arr[0] < n_regions)
                    & (arr[1] >= 0)
                    & (arr[1] < n_regions)
                )
                return arr[:, valid]

    return np.zeros((2, 0), dtype=np.int64)


def _active_components(active: np.ndarray, edge_index: np.ndarray) -> List[List[int]]:
    n_nodes = int(active.shape[0])
    adjacency: List[List[int]] = [[] for _ in range(n_nodes)]

    for src, dst in edge_index.T:
        s = int(src)
        d = int(dst)
        if not (active[s] and active[d]):
            continue
        adjacency[s].append(d)
        adjacency[d].append(s)

    visited = np.zeros((n_nodes,), dtype=bool)
    components: List[List[int]] = []

    for node_idx in range(n_nodes):
        if not active[node_idx] or visited[node_idx]:
            continue
        stack = [node_idx]
        visited[node_idx] = True
        component: List[int] = []

        while stack:
            cur = stack.pop()
            component.append(cur)
            for nxt in adjacency[cur]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)

        components.append(component)

    return components


def _estimate_patch_size_from_centroids(centroids_xy: np.ndarray) -> float:
    if centroids_xy.shape[0] < 2:
        return 1536.0

    diffs = centroids_xy[:, None, :] - centroids_xy[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))
    np.fill_diagonal(dists, np.inf)
    nn = np.min(dists, axis=1)
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:
        return 1536.0
    return float(np.clip(np.median(nn), 512.0, 4096.0))


def _contour_polygons_from_component_regions(
    comp_xy: np.ndarray,
    patch_size_level: float,
    max_grid_side: int = 1024,
) -> List[List[List[float]]]:
    if comp_xy.size == 0:
        return []

    half = float(patch_size_level) / 2.0
    x_min = float(np.min(comp_xy[:, 0]) - half - patch_size_level)
    y_min = float(np.min(comp_xy[:, 1]) - half - patch_size_level)
    x_max = float(np.max(comp_xy[:, 0]) + half + patch_size_level)
    y_max = float(np.max(comp_xy[:, 1]) + half + patch_size_level)

    span_x = max(1.0, x_max - x_min)
    span_y = max(1.0, y_max - y_min)
    span = max(span_x, span_y)

    scale = max(1.0, float(span) / float(max_grid_side))
    scale = min(scale, max(1.0, float(patch_size_level) / 3.0))

    width = int(np.ceil(span_x / scale)) + 3
    height = int(np.ceil(span_y / scale)) + 3

    if width <= 2 or height <= 2:
        return []

    if max(width, height) > 2048:
        upscale = float(max(width, height)) / 2048.0
        scale *= upscale
        width = int(np.ceil(span_x / scale)) + 3
        height = int(np.ceil(span_y / scale)) + 3

    patch_px = max(2, int(round(float(patch_size_level) / scale)))
    half_px = patch_px // 2

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in range(comp_xy.shape[0]):
        cx = int(round((float(comp_xy[i, 0]) - x_min) / scale))
        cy = int(round((float(comp_xy[i, 1]) - y_min) / scale))

        x0 = max(0, cx - half_px)
        x1 = min(width, cx + half_px + 1)
        y0 = max(0, cy - half_px)
        y1 = min(height, cy + half_px + 1)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1

    if int(mask.sum()) == 0:
        return []

    fig, ax = plt.subplots(figsize=(1, 1))
    try:
        cs = ax.contour(mask.astype(np.float32), levels=[0.5])
        polygons: List[List[List[float]]] = []
        for seg in cs.allsegs[0]:
            verts = np.asarray(seg, dtype=np.float64)
            if verts.shape[0] < 3:
                continue

            xs = x_min + verts[:, 0] * scale
            ys = y_min + verts[:, 1] * scale
            ring = np.stack([xs, ys], axis=1).astype(np.float32)

            if not np.allclose(ring[0], ring[-1]):
                ring = np.vstack([ring, ring[0]])

            polygons.append(ring.tolist())

        def _bbox_area(poly: List[List[float]]) -> float:
            arr = np.asarray(poly, dtype=np.float32)
            if arr.shape[0] < 3:
                return 0.0
            return float((np.max(arr[:, 0]) - np.min(arr[:, 0])) * (np.max(arr[:, 1]) - np.min(arr[:, 1])))

        polygons.sort(key=_bbox_area, reverse=True)
        return polygons
    finally:
        plt.close(fig)


def extract_tissue_region_components(
    data: Any,
    region_probabilities: torch.Tensor,
    threshold: float = 0.5,
    min_component_size: int = 2,
    min_mean_probability: float = 0.55,
) -> List[Dict[str, Any]]:
    """Return connected tumor components on the region graph."""

    region_store = _get_region_store(data)

    n_regions_from_store = int(getattr(region_store, "num_nodes", 0) or 0)
    if n_regions_from_store <= 0:
        centroid = getattr(region_store, "centroid", None)
        if torch.is_tensor(centroid):
            n_regions_from_store = int(centroid.size(0))

    probs = region_probabilities.detach().cpu().view(-1).numpy().astype(np.float32)
    n_regions = int(probs.shape[0])
    if n_regions_from_store > 0 and n_regions_from_store != n_regions:
        raise ValueError(
            f"region_probabilities length ({n_regions}) must match region count ({n_regions_from_store})"
        )

    active = probs >= float(threshold)
    edge_index = _get_region_edge_index(data, n_regions)
    centroids = _get_region_centroids(region_store, n_regions)
    patch_size_level = _estimate_patch_size_from_centroids(centroids)

    components = _active_components(active, edge_index)
    out: List[Dict[str, Any]] = []

    for comp_idx, region_ids in enumerate(components):
        if len(region_ids) < int(min_component_size):
            continue

        region_ids_arr = np.asarray(region_ids, dtype=np.int64)
        comp_probs = probs[region_ids_arr]
        mean_prob = float(np.mean(comp_probs))
        if mean_prob < float(min_mean_probability):
            continue

        comp_xy = centroids[region_ids_arr]
        x_min = float(np.min(comp_xy[:, 0]))
        y_min = float(np.min(comp_xy[:, 1]))
        x_max = float(np.max(comp_xy[:, 0]))
        y_max = float(np.max(comp_xy[:, 1]))
        contour_polygons = _contour_polygons_from_component_regions(
            comp_xy=comp_xy,
            patch_size_level=patch_size_level,
        )

        out.append(
            {
                "component_id": int(comp_idx),
                "region_indices": region_ids_arr.tolist(),
                "num_regions": int(len(region_ids)),
                "mean_probability": mean_prob,
                "max_probability": float(np.max(comp_probs)),
                "bbox_level_coords": [x_min, y_min, x_max, y_max],
                "center_level_coords": [float(np.mean(comp_xy[:, 0])), float(np.mean(comp_xy[:, 1]))],
                "contour_polygons_level_coords": contour_polygons,
            }
        )

    return out


def export_components_as_geojson(
    components: Sequence[Dict[str, Any]],
    output_path: Path | str,
    slide_id: Optional[str] = None,
) -> Path:
    """Write tumor region components to a GeoJSON file and return its path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features = []
    for comp in components:
        contour_polygons = comp.get("contour_polygons_level_coords") or []

        geometry: Dict[str, Any]
        if len(contour_polygons) == 1:
            geometry = {
                "type": "Polygon",
                "coordinates": [contour_polygons[0]],
            }
        elif len(contour_polygons) > 1:
            geometry = {
                "type": "MultiPolygon",
                "coordinates": [[ring] for ring in contour_polygons],
            }
        else:
            x_min, y_min, x_max, y_max = comp["bbox_level_coords"]
            polygon = [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
                [x_min, y_min],
            ]
            geometry = {
                "type": "Polygon",
                "coordinates": [polygon],
            }

        properties = {
            "component_id": comp["component_id"],
            "num_regions": comp["num_regions"],
            "mean_probability": comp["mean_probability"],
            "max_probability": comp["max_probability"],
            "region_indices": comp["region_indices"],
        }
        if slide_id is not None:
            properties["slide_id"] = slide_id

        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": properties,
            }
        )

    payload = {"type": "FeatureCollection", "features": features}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def calibrate_threshold_f1(probabilities: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    """Find the probability threshold that maximizes F1 score, returning (threshold, F1)."""
    probs = probabilities.detach().cpu().view(-1).numpy().astype(np.float32)
    y_true = labels.detach().cpu().view(-1).numpy().astype(np.int64)

    if probs.size == 0 or y_true.size == 0:
        return 0.5, 0.0

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.linspace(0.05, 0.95, 91):
        y_pred = (probs >= float(threshold)).astype(np.int64)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)

    return best_threshold, best_f1
