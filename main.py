from __future__ import annotations

from src.graph_creation_new.config import BuildConfig, CreationConfig
from src.graph_creation_new.enums import MaskType, SlideType, AnnotationType, NodeCreationMode, DatasetType
from src.graph_creation_new.graph_creation import build_graph_from_slide
from src.graph_creation_new.visualization import VisualizationConfig, visualize_graph_structure
from src.paths.path_resolver import PathResolver
from src.paths.paths import get_project_paths
from pathlib import Path
import logging
import torch

def gsdb(slide_path: Path, node_mode: NodeCreationMode = NodeCreationMode.PATCH_ENCODER):
    out_dir = Path("outputs/vis") / slide_path.stem
    cfg = BuildConfig(pipeline="dynamical")

    graph = build_graph_from_slide(
        slide_path,
        cfg=cfg,
        creation_cfg=CreationConfig(
            slide_type=SlideType.MRXS,
            annotation_type=AnnotationType.GEOJSON,
            use_annotations=True,
            mask_type=MaskType.SLIDE,
            use_precomputed_mask=False,
            node_creation_mode=node_mode,
        ),
    )

    return out_dir, cfg, graph

def camelyon(slide_path: Path, node_mode: NodeCreationMode = NodeCreationMode.PATCH_ENCODER):
    out_dir = Path("outputs/vis") / slide_path.stem
    cfg = BuildConfig(pipeline="dynamical")

    graph = build_graph_from_slide(
        slide_path,
        cfg=cfg,
        creation_cfg=CreationConfig(
            slide_type=SlideType.TIFF,
            annotation_type=AnnotationType.ASAP,
            use_annotations=True,
            mask_type=MaskType.SLIDE,
            use_precomputed_mask=False,
            node_creation_mode=node_mode,
            filter_graph_with_annotations=False,
            use_annotation_labels=True,
            annotation_label_source="auto",
        ),
    )

    return out_dir, cfg, graph

def main():
    slide_path = Path("data/raw/camelyon17/images/patient_000_node_4.tif")
    #node_mode = NodeCreationMode.PATCH_ENCODER
    node_mode = NodeCreationMode.CELL_SEGMENTATION

    # out_dir, cfg, graph = gsdb(slide_path, node_mode=node_mode)
    out_dir, cfg, graph = camelyon(slide_path, node_mode=node_mode)

    viz_summary = visualize_graph_structure(
        data=graph,
        out_dir=out_dir,
        slide_path=slide_path,
        build_cfg=cfg,
        viz_cfg=VisualizationConfig(),
    )

    # Speichere Graph mit PathResolver
    path_resolver = PathResolver(get_project_paths())
    graph_path = path_resolver.graph_path(
        slide_path=slide_path,
        dataset=DatasetType.CAMELYON17.value,
        pipeline_name=cfg.pipeline,
    )
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, graph_path)

    print(f"slide: {slide_path}")
    print(f"out_dir: {out_dir}")
    print(f"node_mode: {node_mode.value}")
    print(f"nodes: {viz_summary['node_counts']}")
    print(f"edges: {viz_summary['edge_counts']}")
    print(f"hierarchical_graph: {out_dir / 'hierarchical_graph.png'}")
    print(f"graph_saved: {graph_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()