from __future__ import annotations

from pathlib import Path
import sys
from collections import deque
from typing import List, Optional, Sequence, Tuple

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.graph_creation_new.config import BuildContext
from src.graph_creation_new.helper.annotations import annotation_contains_point, annotation_label
from src.graph_creation_new.helper.slide import resolve_patch_size_px, resolve_stride_px


NEGATIVE_REGION_LABEL = "__negative__"
# Region contains a genuine MIX of annotated (any label) and unlabeled patches, or patches
# carrying two or more DIFFERENT annotation labels together -- i.e. this region's true label is
# actually ambiguous, not just unresolved. Only produced by the geometric region-construction
# mode below (region_construction_mode="geometric") -- the annotation_driven mode never mixes
# labels within one region by construction (it only ever connected-components same-label
# patches together in the first place). Deliberately a DIFFERENT sentinel from
# NEGATIVE_REGION_LABEL/None (both of which flow through to annotation_y=-1 and can still get
# resolved to an explicit 0/1 target later via train.py's slide-aware unlabeled_region_policy --
# appropriate for "genuinely no signal here") -- a MIXED region is not "no signal", it's
# "contradictory signal", and must never be silently resolved to a confident 0 or 1. See
# helper/subgraph.py's create_subgraph (region_label_ambiguous) and
# src/pipelines/train.py's _resolve_node_binary_targets for where this distinction is enforced.
MIXED_REGION_LABEL = "__mixed__"


def _connected_components_on_grid(top_left: np.ndarray, stride: int) -> List[np.ndarray]:
    n = int(top_left.shape[0])
    if n == 0:
        return []

    stride = max(1, int(stride))
    coords = [(int(x), int(y)) for x, y in top_left.tolist()]
    index_of = {coord: idx for idx, coord in enumerate(coords)}
    visited = np.zeros((n,), dtype=bool)

    offsets = [
        (-stride, 0),
        (stride, 0),
        (0, -stride),
        (0, stride),
        (-stride, -stride),
        (-stride, stride),
        (stride, -stride),
        (stride, stride),
    ]

    components: List[np.ndarray] = []
    for start in range(n):
        if visited[start]:
            continue
        queue = deque([start])
        visited[start] = True
        members: List[int] = []
        while queue:
            current = queue.popleft()
            members.append(current)
            cx, cy = coords[current]
            for ox, oy in offsets:
                nxt = index_of.get((cx + ox, cy + oy))
                if nxt is None or visited[nxt]:
                    continue
                visited[nxt] = True
                queue.append(nxt)
        components.append(np.asarray(members, dtype=np.int64))

    return components


def _connected_components_fallback(xy: np.ndarray, stride: int) -> List[np.ndarray]:
    n = int(xy.shape[0])
    if n == 0:
        return []
    radius = max(1.0, float(stride) * 1.5)
    visited = np.zeros((n,), dtype=bool)
    components: List[np.ndarray] = []

    for start in range(n):
        if visited[start]:
            continue
        queue = deque([start])
        visited[start] = True
        members: List[int] = []
        while queue:
            current = queue.popleft()
            members.append(current)
            distances = np.linalg.norm(xy - xy[current], axis=1)
            neighbors = np.where((distances <= radius) & (~visited))[0]
            for nxt in neighbors.tolist():
                visited[nxt] = True
                queue.append(nxt)
        components.append(np.asarray(members, dtype=np.int64))

    return components


def _connected_components(patch_xy: np.ndarray, patch_topleft: Optional[np.ndarray], stride: int) -> List[np.ndarray]:
    if patch_topleft is not None and int(patch_topleft.shape[0]) == int(patch_xy.shape[0]):
        return _connected_components_on_grid(np.asarray(patch_topleft, dtype=np.int32), stride)
    return _connected_components_fallback(np.asarray(patch_xy, dtype=np.float32), stride)


def _patch_annotation_membership(
    context: BuildContext,
    patch_xy: np.ndarray,
) -> Tuple[np.ndarray, List[Optional[str]]]:
    annotations: Sequence[dict] = getattr(context, "annotations", None) or []
    n_patches = int(patch_xy.shape[0])
    annotation_ids = np.full((n_patches,), -1, dtype=np.int64)
    annotation_labels: List[Optional[str]] = [None] * n_patches
    if not annotations:
        return annotation_ids, annotation_labels

    downsample = float(context.slide.level_downsamples[int(context.cfg.patch_level)])
    centers_level0 = patch_xy.astype(np.float32) * downsample
    source = context.creation_cfg.annotation_label_source
    classification_field = context.creation_cfg.annotation_classification_field

    for patch_index, center in enumerate(centers_level0):
        for ann_index, annotation in enumerate(annotations):
            if not annotation_contains_point(annotation, (float(center[0]), float(center[1]))):
                continue
            annotation_ids[patch_index] = int(ann_index)
            annotation_labels[patch_index] = annotation_label(annotation, source=source, classification_field=classification_field)
            break

    return annotation_ids, annotation_labels


def _bisect_to_max_size(xy: np.ndarray, max_size: int, _depth: int = 0, _max_depth: int = 24) -> List[np.ndarray]:
    n = int(xy.shape[0])
    if n <= max_size or _depth >= _max_depth:
        return [np.arange(n)]

    ranges = xy.max(axis=0) - xy.min(axis=0)
    axis = int(np.argmax(ranges))
    median = np.median(xy[:, axis])
    left_mask = xy[:, axis] <= median

    if not left_mask.any() or bool(left_mask.all()):
        return [np.arange(n)]

    left_idx = np.where(left_mask)[0]
    right_idx = np.where(~left_mask)[0]
    groups: List[np.ndarray] = []
    for sub in _bisect_to_max_size(xy[left_idx], max_size, _depth + 1, _max_depth):
        groups.append(left_idx[sub])
    for sub in _bisect_to_max_size(xy[right_idx], max_size, _depth + 1, _max_depth):
        groups.append(right_idx[sub])
    return groups


def _build_regions_geometric(
    context: BuildContext,
    patch_xy: np.ndarray,
    patch_topleft: Optional[np.ndarray],
    stride: int,
    annotation_ids: np.ndarray,
    patch_annotation_labels: List[Optional[str]],
) -> None:
    n_patches = int(patch_xy.shape[0])
    region_labels = np.full((n_patches,), -1, dtype=np.int64)
    region_annotation_labels: List[Optional[str]] = []
    next_region_id = 0

    n_mixed = 0
    n_negative = 0
    n_positive = 0

    max_size = getattr(context.cfg, "geometric_max_patches_per_region", None)
    max_size = int(max_size) if max_size is not None and int(max_size) > 0 else None

    for component in _connected_components(patch_xy, patch_topleft, stride):
        if max_size is not None and component.size > max_size:
            sub_groups = [component[local] for local in _bisect_to_max_size(patch_xy[component], max_size)]
        else:
            sub_groups = [component]

        for sub_component in sub_groups:
            member_ann_labels = [patch_annotation_labels[i] for i in sub_component.tolist()]
            distinct_labels = {label for label in member_ann_labels if label is not None}
            has_unlabeled = any(label is None for label in member_ann_labels)

            if not distinct_labels:
                region_label: Optional[str] = NEGATIVE_REGION_LABEL
                n_negative += 1
            elif len(distinct_labels) == 1 and not has_unlabeled:
                region_label = next(iter(distinct_labels))
                n_positive += 1
            else:
                region_label = MIXED_REGION_LABEL
                n_mixed += 1

            region_labels[sub_component] = next_region_id
            region_annotation_labels.append(region_label)
            next_region_id += 1

    n_regions = int(next_region_id)
    region_xy = np.zeros((n_regions, 2), dtype=np.float32)
    for region_id in range(n_regions):
        members = np.where(region_labels == region_id)[0]
        region_xy[region_id] = patch_xy[members].mean(axis=0).astype(np.float32)

    context.region_labels = region_labels
    context.region_xy = region_xy
    context.region_annotation_labels = region_annotation_labels
    context.region_diagnostics = {
        "total_regions": n_regions,
        "annotated_regions": int(n_positive),
        "negative_regions": int(n_negative),
        "mixed_regions": int(n_mixed),
        "unannotated_regions": 0,
        "total_patches": n_patches,
        "annotated_patches": int(np.sum(annotation_ids >= 0)),
        "unannotated_patches": int(np.sum(annotation_ids < 0)),
        "disconnected_region_count": 0,
        "disconnected_region_ids": [],
        "region_policy": (
            "geometric: pure connected-components regardless of annotations; label = "
            "per-region majority/purity over member patches (negative/single-label/mixed)"
        ),
        "negative_region_label": NEGATIVE_REGION_LABEL,
        "mixed_region_label": MIXED_REGION_LABEL,
    }


def build_regions(context: BuildContext) -> None:
    """Group patches into spatial regions and assign each region an annotation label, storing the result on the context."""
    patch_xy = context.patch_xy
    if patch_xy is None or patch_xy.size == 0:
        raise ValueError("Patches must be created before regions")

    patch_xy = np.asarray(patch_xy, dtype=np.float32)
    patch_topleft = None
    if context.patch_topleft is not None:
        patch_topleft = np.asarray(context.patch_topleft, dtype=np.int32)
        if int(patch_topleft.shape[0]) != int(patch_xy.shape[0]):
            patch_topleft = None

    annotation_ids, patch_annotation_labels = _patch_annotation_membership(context, patch_xy)
    region_labels = np.full((patch_xy.shape[0],), -1, dtype=np.int64)
    region_annotation_labels: List[Optional[str]] = []
    next_region_id = 0

    # NOT context.cfg.stride directly: when target_patch_size_um is set (GSDB scanner1/scanner2
    # physical-scale correction), the actual per-slide patch grid stride in pixels is derived
    # from THIS slide's own mpp via resolve_stride_px (same function patches.py's
    # _iter_candidate_top_lefts already uses to place patches) and can differ substantially from
    # the nominal cfg.stride -- e.g. 256px (scanner1) vs 306px (scanner2) at fixed patch_size=256
    # in context_aware_color_normalized. Using cfg.stride here made _connected_components_on_grid's
    # +-stride neighbor-offset check silently fail to match ANY actual neighboring patch for
    # every scanner2 slide (grid step 306 != checked offset 256), so every patch became its own
    # isolated connected component -- i.e. every scanner2 graph had one region per patch (up to
    # ~17000 "regions" instead of the ~50-150 patch_size=256 slides get on scanner1), a massive
    # structural train/test mismatch found via a domain-classifier-AUC=1.0 diagnostic on
    # 2026-08-11, unrelated to color/embedding shift. patch_topleft (checked against this stride)
    # is in the same cfg.patch_level pixel space resolve_stride_px operates in.
    stride = resolve_stride_px(context, resolve_patch_size_px(context))
    has_annotations = bool(getattr(context, "annotations", None))

    # region_construction_mode="geometric" (default "annotation_driven" = current behavior,
    # byte-for-byte unchanged below): region SHAPE no longer depends on has_annotations at all --
    # see _build_regions_geometric's docstring for the full rationale. Checked before the
    # has_annotations branch below since it replaces BOTH of that branch's arms, not just one.
    if str(getattr(context.cfg, "region_construction_mode", "annotation_driven")).strip().lower() == "geometric":
        _build_regions_geometric(context, patch_xy, patch_topleft, stride, annotation_ids, patch_annotation_labels)
        return

    if not has_annotations:
        slide_label = f"slide:{Path(context.slide_path).stem}"
        for component in _connected_components(patch_xy, patch_topleft, stride):
            region_labels[component] = next_region_id
            region_annotation_labels.append(slide_label)
            next_region_id += 1

        n_regions = int(next_region_id)
        region_xy = np.zeros((n_regions, 2), dtype=np.float32)
        for region_id in range(n_regions):
            members = np.where(region_labels == region_id)[0]
            region_xy[region_id] = patch_xy[members].mean(axis=0).astype(np.float32)

        context.region_labels = region_labels
        context.region_xy = region_xy
        context.region_annotation_labels = region_annotation_labels
        context.region_diagnostics = {
            "total_regions": int(n_regions),
            "annotated_regions": 0,
            "unannotated_regions": 0,
            "slide_label_regions": int(n_regions),
            "negative_regions": 0,
            "total_patches": int(patch_xy.shape[0]),
            "annotated_patches": 0,
            "unannotated_patches": int(patch_xy.shape[0]),
            "disconnected_region_count": 0,
            "disconnected_region_ids": [],
            "region_policy": "no annotations -> connected components with slide label",
            "slide_label": slide_label,
        }
        return

    annotated_members = np.where(annotation_ids >= 0)[0]
    annotated_label_values = sorted(
        {
            patch_annotation_labels[idx]
            for idx in annotated_members.tolist()
            if patch_annotation_labels[idx] is not None
        }
    )

    for ann_label in annotated_label_values:
        label_members = np.asarray(
            [idx for idx in annotated_members.tolist() if patch_annotation_labels[idx] == ann_label],
            dtype=np.int64,
        )
        if label_members.size == 0:
            continue
        label_xy = patch_xy[label_members]
        label_topleft = patch_topleft[label_members] if patch_topleft is not None else None
        for component in _connected_components(label_xy, label_topleft, stride):
            global_members = label_members[component]
            region_labels[global_members] = next_region_id
            region_annotation_labels.append(ann_label)
            next_region_id += 1

    unresolved_annotated_members = np.where((annotation_ids >= 0) & (region_labels < 0))[0]
    unique_unresolved_annotation_ids = [
        int(value) for value in np.unique(annotation_ids[unresolved_annotated_members]).tolist() if int(value) >= 0
    ]
    for ann_id in unique_unresolved_annotation_ids:
        ann_members = np.where((annotation_ids == ann_id) & (region_labels < 0))[0]
        if ann_members.size == 0:
            continue
        ann_xy = patch_xy[ann_members]
        ann_topleft = patch_topleft[ann_members] if patch_topleft is not None else None
        for component in _connected_components(ann_xy, ann_topleft, stride):
            global_members = ann_members[component]
            region_labels[global_members] = next_region_id
            label_text = patch_annotation_labels[int(global_members[0])] if global_members.size > 0 else None
            region_annotation_labels.append(label_text)
            next_region_id += 1

    background_members = np.where(annotation_ids < 0)[0]
    if background_members.size > 0:
        bg_xy = patch_xy[background_members]
        bg_topleft = patch_topleft[background_members] if patch_topleft is not None else None
        bg_components = _connected_components(bg_xy, bg_topleft, stride)
        for bg_component in bg_components:
            component_global = background_members[bg_component]
            region_labels[component_global] = next_region_id
            region_annotation_labels.append(NEGATIVE_REGION_LABEL)
            next_region_id += 1

    if np.any(region_labels < 0):
        raise ValueError("No regions could be assigned to the extracted patches")

    n_regions = int(next_region_id)
    region_xy = np.zeros((n_regions, 2), dtype=np.float32)
    for region_id in range(n_regions):
        members = np.where(region_labels == region_id)[0]
        region_xy[region_id] = patch_xy[members].mean(axis=0).astype(np.float32)

    context.region_labels = region_labels
    context.region_xy = region_xy
    context.region_annotation_labels = region_annotation_labels

    annotated_region_count = int(sum(label not in {None, NEGATIVE_REGION_LABEL} for label in region_annotation_labels))
    negative_region_count = int(sum(label == NEGATIVE_REGION_LABEL for label in region_annotation_labels))
    unannotated_region_count = int(sum(label is None for label in region_annotation_labels))
    context.region_diagnostics = {
        "total_regions": int(n_regions),
        "annotated_regions": annotated_region_count,
        "negative_regions": negative_region_count,
        "unannotated_regions": unannotated_region_count,
        "total_patches": int(patch_xy.shape[0]),
        "annotated_patches": int(np.sum(annotation_ids >= 0)),
        "unannotated_patches": int(np.sum(annotation_ids < 0)),
        "disconnected_region_count": 0,
        "disconnected_region_ids": [],
        "region_policy": "annotations->annotation label, non-annotated areas->connected negative regions",
        "negative_region_label": NEGATIVE_REGION_LABEL,
    }
