import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from dataclasses import asdict
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.graph_creation_new.config import BuildConfig, BuildContext, CreationConfig
from src.graph_creation_new.enums import NodeCreationMode, EdgeConstructionMode, DatasetType
from src.graph_creation_new.helper.annotations import (
    annotation_contains_point,
    annotation_label,
    build_annotation_label_vocab,
    is_positive_label_text,
)
from src.graph_creation_new.helper.edges import EDGE_FEAT_DIM, build_edges, build_edge_features
from src.graph_creation_new.helper.features import compute_node_features
from src.graph_creation_new.helper.cells import create_cell_subgraph_structure
from src.graph_creation_new.helper.regions import build_regions, MIXED_REGION_LABEL
from src.graph_creation_new.helper.labels import resolve_label_fields, apply_label_field_data


def create_subgraph(context: BuildContext) -> None:
    """Build the patch/region/slide heterogeneous graph structure for a slide and store it on the context."""
    mode = context.creation_cfg.node_creation_mode
    if mode == NodeCreationMode.CELL_SEGMENTATION:
        create_cell_subgraph_structure(context)
        return
    if mode != NodeCreationMode.PATCH_ENCODER:
        raise ValueError(f"Unsupported node_creation_mode: {mode}")

    if context.patch_xy is None or context.patch_frac is None or context.patch_topleft is None:
        raise ValueError("Patch extraction failed")
    if context.region_labels is None or context.region_xy is None:
        raise ValueError("Region construction failed")

    patch_xy = context.patch_xy
    patch_frac = context.patch_frac
    patch_topleft = context.patch_topleft
    region_labels = context.region_labels
    region_xy = context.region_xy

    p2p_src, p2p_dst, p2p_dist = build_edges(
        xy=patch_xy,
        mode=context.creation_cfg.patch_patch_edge_mode,
        k=int(context.cfg.k_patch),
        radius_px=(
            context.cfg.patch_radius_px
            if context.creation_cfg.patch_patch_edge_mode == EdgeConstructionMode.KNN
            else context.creation_cfg.patch_patch_distance_radius_px
        ),
        semantic_ids=context.region_labels,
        relation_name="patch_near_patch",
    )
    n_regions = int(region_xy.shape[0])
    r2r_src, r2r_dst, r2r_dist = build_edges(
        xy=region_xy,
        mode=context.creation_cfg.region_region_edge_mode,
        k=min(int(context.cfg.k_region), max(1, n_regions - 1)),
        radius_px=context.creation_cfg.region_region_distance_radius_px,
        relation_name="region_near_region",
    )

    context.p2p_src = p2p_src
    context.p2p_dst = p2p_dst
    context.p2p_dist = p2p_dist
    context.r2r_src = r2r_src
    context.r2r_dst = r2r_dst
    context.r2r_dist = r2r_dist

    node_features = compute_node_features(context)
    n_patches = int(patch_xy.shape[0])

    annotation_vocab: List[str] = []
    patch_annotation_ids = None
    region_annotation_ids = None
    region_label_ambiguous = np.zeros((n_regions,), dtype=bool)
    if context.creation_cfg.use_annotation_labels and context.annotations:
        annotation_vocab = build_annotation_label_vocab(
            context.annotations,
            source=context.creation_cfg.annotation_label_source,
            classification_field=context.creation_cfg.annotation_classification_field,
        )
        label_to_id = {label: index for index, label in enumerate(annotation_vocab)}

        patch_annotation_ids = np.full((n_patches,), -1, dtype=np.int64)
        patch_centers_level0 = patch_xy * float(context.slide.level_downsamples[int(context.cfg.patch_level)])
        for patch_index, center in enumerate(patch_centers_level0):
            matched_label = None
            for annotation in context.annotations:
                if annotation_contains_point(annotation, (float(center[0]), float(center[1]))):
                    matched_label = annotation_label(
                        annotation,
                        source=context.creation_cfg.annotation_label_source,
                        classification_field=context.creation_cfg.annotation_classification_field,
                    )
                    if matched_label is not None:
                        break
            if matched_label is not None and matched_label in label_to_id:
                patch_annotation_ids[patch_index] = int(label_to_id[matched_label])

        labeled_patch_count = int(np.sum(patch_annotation_ids >= 0))
        if labeled_patch_count == 0:
            logging.warning(
                "Annotation labels enabled but no patch received a label for %s",
                context.slide_path,
            )

        region_annotation_ids = np.full((n_regions,), -1, dtype=np.int64)
        # True for regions helper/regions.py's geometric mode flagged MIXED_REGION_LABEL --
        # genuinely ambiguous (both annotated and unlabeled patches together, or two+ different
        # labels together), as opposed to plain -1 (no signal at all, safe to resolve later via
        # train.py's slide-aware unlabeled_region_policy). CHANGED 2026-08-12 (v2): mixed regions
        # DO now flow into the majority-vote fallback below like any other unresolved region --
        # i.e. they get a real annotation_y target, contributing to the loss -- rather than being
        # excluded outright. The flag is instead consumed downstream (src/pipelines/train.py's
        # _resolve_node_binary_targets, src/tissue_region_extraction/model.py's region_aux_loss)
        # to down-weight, not drop, these regions: a region that's mostly annotated-positive with
        # a bit of unlabeled tissue mixed in is real, useful (if less certain) signal, not nothing
        # -- only a truly unannotated region (plain -1, no MIXED flag) stays governed purely by
        # the existing slide-aware/pseudo-label policy. Always empty (all False) for
        # annotation_driven-mode graphs, since that mode never produces this sentinel.
        region_annotation_labels = getattr(context, "region_annotation_labels", None)
        if region_annotation_labels is not None and len(region_annotation_labels) == n_regions:
            for region_id, region_label in enumerate(region_annotation_labels):
                if region_label is None:
                    continue
                if region_label == MIXED_REGION_LABEL:
                    region_label_ambiguous[region_id] = True
                    continue
                mapped = label_to_id.get(region_label)
                if mapped is not None:
                    region_annotation_ids[region_id] = int(mapped)

        unresolved_regions = np.where(region_annotation_ids < 0)[0]
        for region_id in unresolved_regions.tolist():
            members = np.where(region_labels == region_id)[0]
            if members.size == 0:
                continue
            region_patch_labels = patch_annotation_ids[members]
            region_patch_labels = region_patch_labels[region_patch_labels >= 0]
            if region_patch_labels.size == 0:
                continue
            values, counts = np.unique(region_patch_labels, return_counts=True)

            # MIXED_REGION_LABEL regions (region_label_ambiguous[region_id]) resolve toward
            # POSITIVE whenever any positive-labeled patch is present, rather than a plain vote
            # count -- see is_positive_label_text's docstring: this is not "an extra label", it's
            # the SAME positive/negative vocabulary, just resolved so a small confirmed-positive
            # focus inside a larger unlabeled/negative region isn't out-voted into "negative".
            # Confidence (not full 1.0) is handled separately via region.label_ambiguous
            # downstream -- this only decides WHICH of the vocab's existing labels wins.
            if region_id < region_label_ambiguous.shape[0] and region_label_ambiguous[region_id]:
                positive_values = [v for v in values.tolist() if is_positive_label_text(annotation_vocab[v])]
                if positive_values:
                    positive_counts = {v: int(counts[values == v][0]) for v in positive_values}
                    region_annotation_ids[region_id] = int(max(positive_counts, key=positive_counts.get))
                    continue

            max_count = int(np.max(counts))
            winners = values[counts == max_count]
            region_annotation_ids[region_id] = int(np.min(winners))

    # Additional classification fields (e.g. GSDB's tissue_type alongside the primary
    # inflammation_status above) -- fully independent of and does not alter anything above.
    # Region-level values here are a plain patch-majority vote over the regions that already
    # exist (built from the PRIMARY field's annotation boundaries in helper/regions.py) -- not a
    # boundary-exact resolution the way the primary field's region_annotation_labels is, since
    # that would require rebuilding regions around this field's own boundaries instead. See
    # CreationConfig.extra_annotation_classification_fields' docstring.
    extra_label_data: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]] = {}
    extra_fields = list(context.creation_cfg.extra_annotation_classification_fields or [])
    if context.creation_cfg.use_annotation_labels and context.annotations and extra_fields:
        patch_centers_level0 = patch_xy * float(context.slide.level_downsamples[int(context.cfg.patch_level)])
        for field_name in extra_fields:
            field_vocab = build_annotation_label_vocab(
                context.annotations,
                source=context.creation_cfg.annotation_label_source,
                classification_field=field_name,
            )
            field_label_to_id = {label: index for index, label in enumerate(field_vocab)}

            field_patch_ids = np.full((n_patches,), -1, dtype=np.int64)
            for patch_index, center in enumerate(patch_centers_level0):
                matched_label = None
                for annotation in context.annotations:
                    if annotation_contains_point(annotation, (float(center[0]), float(center[1]))):
                        matched_label = annotation_label(
                            annotation,
                            source=context.creation_cfg.annotation_label_source,
                            classification_field=field_name,
                        )
                        if matched_label is not None:
                            break
                if matched_label is not None and matched_label in field_label_to_id:
                    field_patch_ids[patch_index] = int(field_label_to_id[matched_label])

            if int(np.sum(field_patch_ids >= 0)) == 0:
                logging.warning(
                    "extra_annotation_classification_fields includes '%s' but no patch received "
                    "a label for it in %s", field_name, context.slide_path,
                )

            field_region_ids = np.full((n_regions,), -1, dtype=np.int64)
            for region_id in range(n_regions):
                members = np.where(region_labels == region_id)[0]
                if members.size == 0:
                    continue
                region_patch_labels = field_patch_ids[members]
                region_patch_labels = region_patch_labels[region_patch_labels >= 0]
                if region_patch_labels.size == 0:
                    continue
                values, counts = np.unique(region_patch_labels, return_counts=True)
                max_count = int(np.max(counts))
                winners = values[counts == max_count]
                field_region_ids[region_id] = int(np.min(winners))

            extra_label_data[field_name] = (field_patch_ids, field_region_ids, field_vocab)

    # Config-driven, per-node-type named label taxonomy (CreationConfig.label_fields) -- e.g. a
    # normalized "status" view (positive/negative/unknown/mixed) on top of annotation_y, or a
    # slide-level "subtype"/"stage" joined from an external CSV. Purely additive: label_fields
    # defaults to an empty list, in which case resolve_label_fields returns {} immediately and
    # nothing below writes anything new onto data/data.meta. See helper/labels.py.
    label_field_data = resolve_label_fields(
        context,
        n_patches=n_patches,
        n_regions=n_regions,
        patch_annotation_ids=patch_annotation_ids,
        region_annotation_ids=region_annotation_ids,
        annotation_vocab=annotation_vocab,
        region_label_ambiguous=region_label_ambiguous,
        extra_label_data=extra_label_data,
    )

    slide_centroid_xy = (
        float(context.slide.dimensions[0]) / 2.0,
        float(context.slide.dimensions[1]) / 2.0,
    )
    meta_extra: Dict[str, Any] = {
        "wsi_path": context.slide_path,
        "level_count": int(context.slide.level_count),
        "level_dimensions": [tuple(map(int, dimensions)) for dimensions in context.slide.level_dimensions],
        "level_downsamples": [float(downsample) for downsample in context.slide.level_downsamples],
        "thumb_shape": tuple(int(x) for x in context.thumb_rgb.shape),
        "tissue_thumb_fraction": float(np.asarray(context.mask, dtype=np.float32).mean()) if context.mask is not None else 0.0,
    }
    if getattr(context, "region_diagnostics", None) is not None:
        meta_extra["region_diagnostics"] = context.region_diagnostics

    context.graph_data = _assemble_hetero_data(
        patch_xy=patch_xy,
        patch_frac=patch_frac,
        patch_topleft=patch_topleft,
        region_labels=region_labels,
        region_xy=region_xy,
        node_features=node_features,
        patch_annotation_ids=patch_annotation_ids,
        region_annotation_ids=region_annotation_ids,
        region_label_ambiguous=region_label_ambiguous,
        annotation_vocab=annotation_vocab,
        extra_label_data=extra_label_data,
        label_field_data=label_field_data,
        creation_cfg=context.creation_cfg,
        cfg=context.cfg,
        slide_centroid_xy=slide_centroid_xy,
        meta_extra=meta_extra,
        p2p_src=p2p_src,
        p2p_dst=p2p_dst,
        p2p_dist=p2p_dist,
        r2r_src=r2r_src,
        r2r_dst=r2r_dst,
        r2r_dist=r2r_dist,
    )


def _assemble_hetero_data(
    *,
    patch_xy: np.ndarray,
    patch_frac: np.ndarray,
    patch_topleft: np.ndarray,
    region_labels: np.ndarray,
    region_xy: np.ndarray,
    node_features: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    patch_annotation_ids: Optional[np.ndarray],
    region_annotation_ids: Optional[np.ndarray],
    region_label_ambiguous: Optional[np.ndarray] = None,
    annotation_vocab: List[str],
    extra_label_data: Optional[Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]]] = None,
    label_field_data: Optional[Dict[str, Dict[str, Any]]] = None,
    creation_cfg: CreationConfig,
    cfg: BuildConfig,
    slide_centroid_xy: Tuple[float, float],
    meta_extra: Dict[str, Any],
    p2p_src: np.ndarray,
    p2p_dst: np.ndarray,
    p2p_dist: np.ndarray,
    r2r_src: np.ndarray,
    r2r_dst: np.ndarray,
    r2r_dist: np.ndarray,
) -> HeteroData:
    n_patches = int(patch_xy.shape[0])
    n_regions = int(region_xy.shape[0])

    p2r_src = np.arange(n_patches, dtype=np.int64)
    p2r_dst = region_labels.astype(np.int64)
    r2p_src = p2r_dst.copy()
    r2p_dst = p2r_src.copy()
    r2s_src = np.arange(n_regions, dtype=np.int64)
    r2s_dst = np.zeros((n_regions,), dtype=np.int64)
    s2r_src = np.zeros((n_regions,), dtype=np.int64)
    s2r_dst = np.arange(n_regions, dtype=np.int64)

    data = HeteroData()
    data.meta = {}
    data["slide"].num_nodes = 1
    slide_spatial = torch.zeros((1, 1), dtype=torch.float32)
    data["slide"].centroid = torch.tensor([list(slide_centroid_xy)], dtype=torch.float32)

    data["region"].num_nodes = n_regions
    region_spatial = torch.tensor(region_xy, dtype=torch.float32)
    data["region"].centroid = region_spatial.clone()

    data["patch"].num_nodes = n_patches
    patch_spatial = torch.tensor(np.concatenate([patch_xy, patch_frac[:, None]], axis=1), dtype=torch.float32)
    data["patch"].centroid = torch.tensor(patch_xy, dtype=torch.float32)
    data["patch"].topleft = torch.tensor(patch_topleft, dtype=torch.int32)

    patch_feat = region_feat = slide_feat = None
    if node_features is None:
        data["slide"].x = slide_spatial
        data["region"].x = region_spatial
        data["patch"].x = patch_spatial
    else:
        patch_feat, region_feat, slide_feat = node_features
        data["patch"].x = torch.tensor(patch_feat, dtype=torch.float32)
        data["region"].x = torch.tensor(region_feat, dtype=torch.float32)
        data["slide"].x = torch.tensor(slide_feat, dtype=torch.float32)

        data["patch"].feat = data["patch"].x.clone()
        data["region"].feat = data["region"].x.clone()
        data["slide"].feat = data["slide"].x.clone()

        data["patch"].spatial_x = patch_spatial
        data["region"].spatial_x = region_spatial
        data["slide"].spatial_x = slide_spatial

    if patch_annotation_ids is not None:
        data["patch"].annotation_y = torch.tensor(patch_annotation_ids, dtype=torch.long)
        data["region"].annotation_y = torch.tensor(region_annotation_ids, dtype=torch.long)
        data.meta["annotation_label_vocab"] = annotation_vocab
        data.meta["annotation_labels_enabled"] = True
        data.meta["annotation_labeled_patch_count"] = int(np.sum(patch_annotation_ids >= 0))
        data.meta["annotation_label_coverage"] = float(np.mean(patch_annotation_ids >= 0))
    else:
        data.meta["annotation_label_vocab"] = []
        data.meta["annotation_labels_enabled"] = False
        data.meta["annotation_labeled_patch_count"] = 0
        data.meta["annotation_label_coverage"] = 0.0

    if region_label_ambiguous is not None and int(np.asarray(region_label_ambiguous).shape[0]) == n_regions:
        data["region"].label_ambiguous = torch.tensor(region_label_ambiguous, dtype=torch.bool)
    else:
        data["region"].label_ambiguous = torch.zeros((n_regions,), dtype=torch.bool)

    data.meta["extra_label_fields"] = sorted((extra_label_data or {}).keys())
    for field_name, (field_patch_ids, field_region_ids, field_vocab) in (extra_label_data or {}).items():
        y_attr = f"{field_name}_y"
        if field_patch_ids is not None and field_region_ids is not None:
            data["patch"][y_attr] = torch.tensor(field_patch_ids, dtype=torch.long)
            data["region"][y_attr] = torch.tensor(field_region_ids, dtype=torch.long)
            data.meta[f"{field_name}_label_vocab"] = field_vocab
            data.meta[f"{field_name}_labeled_patch_count"] = int(np.sum(field_patch_ids >= 0))
            data.meta[f"{field_name}_label_coverage"] = float(np.mean(field_patch_ids >= 0)) if n_patches > 0 else 0.0
        else:
            data.meta[f"{field_name}_label_vocab"] = []
            data.meta[f"{field_name}_labeled_patch_count"] = 0
            data.meta[f"{field_name}_label_coverage"] = 0.0

    apply_label_field_data(data, label_field_data)

    data["patch", "near", "patch"].edge_index = torch.tensor(np.stack([p2p_src, p2p_dst], axis=0), dtype=torch.long)
    data["patch", "near", "patch"].edge_attr = torch.tensor(p2p_dist[:, None], dtype=torch.float32)
    data["patch", "near", "patch"].edge_feat = torch.tensor(
        build_edge_features(
            src_xy=patch_xy,
            dst_xy=patch_xy,
            src_idx=p2p_src,
            dst_idx=p2p_dst,
            dist=p2p_dist,
            src_semantic_ids=region_labels,
            dst_semantic_ids=region_labels,
            src_feat=patch_feat,
            dst_feat=patch_feat,
        ),
        dtype=torch.float32,
    )

    data["region", "near", "region"].edge_index = torch.tensor(np.stack([r2r_src, r2r_dst], axis=0), dtype=torch.long)
    data["region", "near", "region"].edge_attr = torch.tensor(r2r_dist[:, None], dtype=torch.float32)
    data["region", "near", "region"].edge_feat = torch.tensor(
        build_edge_features(
            src_xy=region_xy,
            dst_xy=region_xy,
            src_idx=r2r_src,
            dst_idx=r2r_dst,
            dist=r2r_dist,
            src_semantic_ids=region_annotation_ids if region_annotation_ids is not None else np.arange(n_regions, dtype=np.int64),
            dst_semantic_ids=region_annotation_ids if region_annotation_ids is not None else np.arange(n_regions, dtype=np.int64),
            src_feat=region_feat,
            dst_feat=region_feat,
        ),
        dtype=torch.float32,
    )

    data["patch", "in", "region"].edge_index = torch.tensor(np.stack([p2r_src, p2r_dst], axis=0), dtype=torch.long)
    data["region", "has", "patch"].edge_index = torch.tensor(np.stack([r2p_src, r2p_dst], axis=0), dtype=torch.long)
    data["patch", "in", "region"].edge_feat = torch.tensor(
        build_edge_features(
            src_xy=patch_xy,
            dst_xy=region_xy,
            src_idx=p2r_src,
            dst_idx=p2r_dst,
            src_semantic_ids=region_labels,
            dst_semantic_ids=np.arange(n_regions, dtype=np.int64),
            src_feat=patch_feat,
            dst_feat=region_feat,
        ),
        dtype=torch.float32,
    )
    data["region", "has", "patch"].edge_feat = torch.tensor(
        build_edge_features(
            src_xy=region_xy,
            dst_xy=patch_xy,
            src_idx=r2p_src,
            dst_idx=r2p_dst,
            src_semantic_ids=np.arange(n_regions, dtype=np.int64),
            dst_semantic_ids=region_labels,
            src_feat=region_feat,
            dst_feat=patch_feat,
        ),
        dtype=torch.float32,
    )

    data["region", "part_of", "slide"].edge_index = torch.tensor(np.stack([r2s_src, r2s_dst], axis=0), dtype=torch.long)
    data["slide", "contains", "region"].edge_index = torch.tensor(np.stack([s2r_src, s2r_dst], axis=0), dtype=torch.long)
    data["region", "part_of", "slide"].edge_feat = torch.tensor(
        build_edge_features(
            src_xy=region_xy,
            dst_xy=data["slide"].centroid.cpu().numpy(),
            src_idx=r2s_src,
            dst_idx=r2s_dst,
            src_semantic_ids=np.arange(n_regions, dtype=np.int64),
            dst_semantic_ids=np.zeros((1,), dtype=np.int64),
            src_feat=region_feat,
            dst_feat=slide_feat,
        ),
        dtype=torch.float32,
    )
    data["slide", "contains", "region"].edge_feat = torch.tensor(
        build_edge_features(
            src_xy=data["slide"].centroid.cpu().numpy(),
            dst_xy=region_xy,
            src_idx=s2r_src,
            dst_idx=s2r_dst,
            src_semantic_ids=np.zeros((1,), dtype=np.int64),
            dst_semantic_ids=np.arange(n_regions, dtype=np.int64),
            src_feat=slide_feat,
            dst_feat=region_feat,
        ),
        dtype=torch.float32,
    )

    data.meta.update(meta_extra)
    data.meta.update({
        "config": asdict(cfg),
        "creation_config": asdict(creation_cfg),
        "pipeline": str(cfg.pipeline),
        "node_creation_mode": "patch_encoder",
        "patch_patch_edge_mode": str(creation_cfg.patch_patch_edge_mode.value),
        "region_region_edge_mode": str(creation_cfg.region_region_edge_mode.value),
        "patch_patch_semantic_source": "region_label" if creation_cfg.patch_patch_edge_mode == EdgeConstructionMode.SEMANTIC_EQUALITY else "none",
        "edge_feat_dim": EDGE_FEAT_DIM,
        "num_patches": n_patches,
        "num_regions": n_regions,
        "has_model_features": bool(node_features is not None),
        "feature_dim": int(data["patch"].x.shape[1]),
        "unannotated_region_label_policy": "annotation_y=-1 (only fallback majority if mixed labels exist in a region)",
    })
    return data


def rebuild_edges_only(
    old_data: HeteroData,
    creation_cfg: CreationConfig,
    cfg: BuildConfig,
) -> HeteroData:
    """Recompute only graph topology and edge features for an already-built slide, reusing its cached node data."""
    if "patch" not in old_data.node_types or "region" not in old_data.node_types:
        raise ValueError("rebuild_edges_only expects a patch+region graph (PATCH_ENCODER mode)")
    if ("patch", "in", "region") not in old_data.edge_types:
        raise ValueError(
            "rebuild_edges_only requires an existing ('patch','in','region') edge_index to "
            "recover region_labels from"
        )

    patch_store = old_data["patch"]
    region_store = old_data["region"]
    slide_store = old_data["slide"]
    old_meta = dict(old_data.meta) if hasattr(old_data, "meta") and old_data.meta else {}

    patch_xy = patch_store.centroid.detach().cpu().numpy().astype(np.float64)
    n_patches = int(patch_xy.shape[0])
    spatial_x = getattr(patch_store, "spatial_x", None)
    if torch.is_tensor(spatial_x) and spatial_x.dim() == 2 and spatial_x.shape[1] >= 3:
        patch_frac = spatial_x[:, 2].detach().cpu().numpy().astype(np.float64)
    else:
        patch_frac = np.zeros((n_patches,), dtype=np.float64)
    patch_topleft = patch_store.topleft.detach().cpu().numpy()

    region_xy = region_store.centroid.detach().cpu().numpy().astype(np.float64)
    n_regions = int(region_xy.shape[0])

    p2r_edge_index = old_data[("patch", "in", "region")].edge_index
    sort_order = torch.argsort(p2r_edge_index[0])
    p2r_sorted_src = p2r_edge_index[0][sort_order]
    region_labels = p2r_edge_index[1][sort_order].detach().cpu().numpy().astype(np.int64)
    if region_labels.shape[0] != n_patches or not torch.equal(
        p2r_sorted_src, torch.arange(n_patches, dtype=p2r_sorted_src.dtype)
    ):
        raise ValueError(
            "('patch','in','region') edge_index does not cover every patch exactly once -- "
            "cannot recover region_labels from this graph"
        )

    node_features: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    if bool(old_meta.get("has_model_features", True)):
        feat_source = patch_store if torch.is_tensor(getattr(patch_store, "feat", None)) else None
        if feat_source is not None:
            node_features = (
                patch_store.feat.detach().cpu().numpy(),
                region_store.feat.detach().cpu().numpy(),
                slide_store.feat.detach().cpu().numpy(),
            )

    patch_annotation_ids = None
    region_annotation_ids = None
    annotation_vocab: List[str] = []
    if torch.is_tensor(getattr(patch_store, "annotation_y", None)):
        patch_annotation_ids = patch_store.annotation_y.detach().cpu().numpy().astype(np.int64)
        region_annotation_ids = region_store.annotation_y.detach().cpu().numpy().astype(np.int64)
        annotation_vocab = list(old_meta.get("annotation_label_vocab", []))

    # Carried over unchanged from the cached graph (this function only reruns edge topology --
    # see this function's own docstring -- it has no annotation/geometry context to recompute
    # region ambiguity from). None (not an all-False array) for graphs built before this field
    # existed, so _assemble_hetero_data's own "unknown -> all-False" fallback applies uniformly.
    region_label_ambiguous = None
    if torch.is_tensor(getattr(region_store, "label_ambiguous", None)):
        region_label_ambiguous = region_store.label_ambiguous.detach().cpu().numpy().astype(bool)

    # Extra classification fields (tissue_type, ...) -- carried over from the cached graph
    # unchanged, since this function has no annotation/geometry context to recompute them from
    # (it only reruns edge topology). See CreationConfig.extra_annotation_classification_fields.
    extra_label_data: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]] = {}
    for field_name in old_meta.get("extra_label_fields", []):
        field_patch_attr = getattr(patch_store, f"{field_name}_y", None)
        field_region_attr = getattr(region_store, f"{field_name}_y", None)
        if torch.is_tensor(field_patch_attr) and torch.is_tensor(field_region_attr):
            extra_label_data[field_name] = (
                field_patch_attr.detach().cpu().numpy().astype(np.int64),
                field_region_attr.detach().cpu().numpy().astype(np.int64),
                list(old_meta.get(f"{field_name}_label_vocab", [])),
            )

    p2p_src, p2p_dst, p2p_dist = build_edges(
        xy=patch_xy,
        mode=creation_cfg.patch_patch_edge_mode,
        k=int(cfg.k_patch),
        radius_px=(
            cfg.patch_radius_px
            if creation_cfg.patch_patch_edge_mode == EdgeConstructionMode.KNN
            else creation_cfg.patch_patch_distance_radius_px
        ),
        semantic_ids=region_labels,
        relation_name="patch_near_patch",
    )
    r2r_src, r2r_dst, r2r_dist = build_edges(
        xy=region_xy,
        mode=creation_cfg.region_region_edge_mode,
        k=min(int(cfg.k_region), max(1, n_regions - 1)),
        radius_px=creation_cfg.region_region_distance_radius_px,
        relation_name="region_near_region",
    )

    slide_centroid_xy = tuple(slide_store.centroid.detach().cpu().numpy().reshape(-1)[:2].tolist())
    meta_extra: Dict[str, Any] = {
        "wsi_path": old_meta.get("wsi_path"),
        "level_count": old_meta.get("level_count"),
        "level_dimensions": old_meta.get("level_dimensions"),
        "level_downsamples": old_meta.get("level_downsamples"),
        "thumb_shape": old_meta.get("thumb_shape"),
        "tissue_thumb_fraction": old_meta.get("tissue_thumb_fraction", 0.0),
        "edges_rebuilt_from_config": old_meta.get("patch_patch_edge_mode"),
    }
    if old_meta.get("region_diagnostics") is not None:
        meta_extra["region_diagnostics"] = old_meta["region_diagnostics"]

    return _assemble_hetero_data(
        patch_xy=patch_xy,
        patch_frac=patch_frac,
        patch_topleft=patch_topleft,
        region_labels=region_labels,
        region_xy=region_xy,
        node_features=node_features,
        patch_annotation_ids=patch_annotation_ids,
        region_annotation_ids=region_annotation_ids,
        region_label_ambiguous=region_label_ambiguous,
        annotation_vocab=annotation_vocab,
        extra_label_data=extra_label_data,
        creation_cfg=creation_cfg,
        cfg=cfg,
        slide_centroid_xy=slide_centroid_xy,
        meta_extra=meta_extra,
        p2p_src=p2p_src,
        p2p_dst=p2p_dst,
        p2p_dist=p2p_dist,
        r2r_src=r2r_src,
        r2r_dst=r2r_dst,
        r2r_dist=r2r_dist,
    )


def rebuild_regions_and_edges_only(
    old_data: HeteroData,
    creation_cfg: CreationConfig,
    cfg: BuildConfig,
    dataset: DatasetType,
) -> HeteroData:
    """Recompute region assignment and all derived edges/features for an already-built slide, reusing its cached patch embeddings."""
    from src.graph_creation_new.graph_creation import DynamicalPipeline, _import_openslide
    from src.graph_creation_new.helper.features import _aggregate_region_features
    from src.paths.path_resolver import PathResolver
    from src.paths.paths import get_project_paths

    if "patch" not in old_data.node_types or "region" not in old_data.node_types:
        raise ValueError("rebuild_regions_and_edges_only expects a patch+region graph (PATCH_ENCODER mode)")

    old_meta = dict(old_data.meta) if hasattr(old_data, "meta") and old_data.meta else {}
    wsi_path = old_meta.get("wsi_path")
    if not wsi_path:
        raise ValueError("Cached graph's meta has no 'wsi_path' -- cannot reopen the slide to rebuild regions")

    # cfg.target_patch_size_um is None whenever the caller only ran _load_config() without also
    # running build_graph.py's _resolve_target_patch_size_um* auto-detect step (which picks a
    # value from the FIRST slide in a whole-corpus build run, not something this per-slide
    # function can redo on its own) -- resolve_patch_size_px/resolve_stride_px then silently fall
    # back to the nominal cfg.patch_size/cfg.stride for every slide, reproducing the exact bug
    # this function exists to fix. The value actually used when old_data was built is already
    # recorded verbatim in its own meta (_assemble_hetero_data always stores asdict(cfg)) --
    # reuse that instead of re-deriving it, so this function is correct regardless of whether the
    # caller resolved auto-detection first.
    if getattr(cfg, "target_patch_size_um", None) is None and bool(getattr(cfg, "auto_detect_target_patch_size_um", False)):
        old_target_um = (old_meta.get("config") or {}).get("target_patch_size_um")
        if old_target_um is not None:
            from dataclasses import replace as _dataclass_replace
            cfg = _dataclass_replace(cfg, target_patch_size_um=float(old_target_um))
        else:
            logging.warning(
                "rebuild_regions_and_edges_only: cfg.target_patch_size_um is unresolved and "
                "old_data's cached meta has none either for %s -- proceeding with nominal "
                "cfg.patch_size/cfg.stride, which reproduces the original bug if this slide's "
                "actual grid stride differs.", wsi_path,
            )

    patch_store = old_data["patch"]
    patch_xy = patch_store.centroid.detach().cpu().numpy().astype(np.float32)
    patch_topleft = patch_store.topleft.detach().cpu().numpy().astype(np.int32)
    spatial_x = getattr(patch_store, "spatial_x", None)
    if torch.is_tensor(spatial_x) and spatial_x.dim() == 2 and spatial_x.shape[1] >= 3:
        patch_frac = spatial_x[:, 2].detach().cpu().numpy().astype(np.float32)
    else:
        patch_frac = np.zeros((patch_xy.shape[0],), dtype=np.float32)

    patch_features: Optional[np.ndarray] = None
    if bool(old_meta.get("has_model_features", True)) and torch.is_tensor(getattr(patch_store, "feat", None)):
        patch_features = patch_store.feat.detach().cpu().numpy()

    openslide = _import_openslide()
    handle = openslide.OpenSlide(str(wsi_path))
    try:
        context = BuildContext(
            path_resolver=PathResolver(get_project_paths()),
            dataset=dataset,
            slide_path=str(wsi_path),
            slide=handle,
            cfg=cfg,
            creation_cfg=creation_cfg,
            patch_xy=patch_xy,
            patch_topleft=patch_topleft,
            patch_frac=patch_frac,
        )

        DynamicalPipeline()._load_annoations(context)  # noqa: SLF001 -- intentional reuse of the real pipeline step
        build_regions(context=context)

        if context.region_labels is None or context.region_xy is None:
            raise ValueError(f"build_regions produced no regions for {wsi_path}")

        if patch_features is not None:
            region_features = _aggregate_region_features(
                patch_features, context.region_labels, n_regions=int(context.region_xy.shape[0])
            )
            slide_features = region_features.mean(axis=0, keepdims=True).astype(np.float32)
            context.node_features_cache = (patch_features, region_features, slide_features)

        # create_subgraph reads context.thumb_rgb.shape for meta only (never its pixel content
        # here) -- a placeholder of the original shape avoids re-generating the real thumbnail
        # for no functional benefit. tissue_thumb_fraction is restored from old_meta below since
        # leaving context.mask unset would otherwise report it as 0.0.
        thumb_shape = old_meta.get("thumb_shape") or (1, 1, 3)
        context.thumb_rgb = np.zeros(tuple(int(d) for d in thumb_shape), dtype=np.uint8)

        create_subgraph(context=context)
        new_data = context.graph_data
        if new_data is None:
            raise ValueError(f"create_subgraph produced no graph_data for {wsi_path}")

        new_data.meta["tissue_thumb_fraction"] = old_meta.get("tissue_thumb_fraction", 0.0)
        new_data.meta["regions_rebuilt_from_num_regions"] = old_meta.get("num_regions")
        return new_data
    finally:
        handle.close()
