"""annotate_tissue_regions.py
  python src/entrypoint/annotate_tissue_regions.py \\
      --slide  data/raw/camelyon/images/patient_073_node_1.tif \\
      --checkpoint ~/models/tissue_region_net_best.pt \\
      --out    results/annotations/patient_073_node_1 \\
      --embedding-model prov_gigapath \\
      --threshold 0.45
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.graph_creation_new.config import BuildConfig, CreationConfig
from src.graph_creation_new.enums import DatasetType
from src.graph_creation_new.graph_creation import build_graph_from_slide
from src.core.meta_processor import MetaData
from src.paths.paths import get_project_paths, add_path_args
from src.tissue_region_extraction.model import TissueRegionGraphModel
from src.tissue_region_extraction.routines import (
    extract_tissue_region_components,
    export_components_as_geojson,
    infer_region_probabilities,
)
from src.utils.model_utils import load_model
from src.utils.logging import setup_logging, quiet_external_loggers


def _select_least_busy_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best_idx, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(i)
        if free_bytes > best_free:
            best_idx, best_free = i, free_bytes
    return torch.device(f"cuda:{best_idx}")



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetische Tumorannotationen aus einer rohen WSI erzeugen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--slide", type=Path, required=True,
                        help="Pfad zur WSI-Slide (.tif / .svs / .ndpi / .mrxs)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Ausgabeverzeichnis für Annotationen und Vorschaubilder")

    parser.add_argument("--meta-file", type=str, default=None,
                        help="Name oder Pfad zur Trainings-Meta-YAML. Checkpoint, Modell-"
                             "hyperparameter und Embedding-Typ werden daraus geladen. "
                             "Explizite CLI-Argumente überschreiben den Meta-Wert.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Trainierter TissueRegionGraphModel-Checkpoint (.pt). "
                             "Wird aus --meta-file gelesen wenn nicht explizit angegeben.")

    parser.add_argument("--embedding-model", type=str, default=None,
                        choices=["prov_gigapath", "uni2"],
                        help="Feature-Extractor für die Graph-Knoten. Standard: aus Meta.")

    parser.add_argument("--pipeline", type=str, default=None,
                        help="Graph-Pipeline (dynamical / simple_pipeline / ...). Standard: aus Meta.")
    parser.add_argument("--patch-size", type=int, default=256,
                        help="Patch-Größe in Pixel auf Patch-Level")
    parser.add_argument("--patch-level", type=int, default=0,
                        help="Pyramid-Level für Patch-Extraktion (0 = volle Auflösung)")
    parser.add_argument("--feature-image-size", type=int, default=None,
                    help="Eingabegröße für das Embedding-Modell (Patch wird intern darauf skaliert). "
                        "Standard: automatisch aus dem Modell (z. B. 224 für prov_gigapath/uni2).")
    parser.add_argument("--graph-config", type=Path, default=None,
                        help="Optionale YAML-Konfiguration für den Graph-Bau "
                             "(überschreibt --pipeline / --patch-size etc.)")

    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Wahrscheinlichkeits-Schwellwert für Tumor-Klassifikation")
    parser.add_argument("--min-component-size", type=int, default=2,
                        help="Minimale Anzahl Region-Knoten pro Tumor-Komponente")
    parser.add_argument("--min-mean-prob", type=float, default=0.50,
                        help="Minimale mittlere Wahrscheinlichkeit pro Komponente")

    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="Versteckte Dimension (Standard: aus Meta, sonst 320)")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Anzahl GATv2-Schichten (Standard: aus Meta, sonst 4)")
    parser.add_argument("--heads", type=int, default=None,
                        help="Anzahl Attention-Heads (Standard: aus Meta, sonst 4)")
    parser.add_argument("--dropout-rate", type=float, default=None,
                        help="Dropout-Rate (Standard: aus Meta, sonst 0.20)")
    parser.add_argument("--edge-dropout-p", type=float, default=None,
                        help="Edge-Dropout-Wahrscheinlichkeit (Standard: aus Meta, sonst 0.05)")
    parser.add_argument("--region-aux-weight", type=float, default=None,
                        help="Region-Auxiliary-Loss-Gewicht (Standard: aus Meta, sonst 0.24)")
    parser.add_argument("--no-strict-load", action="store_true",
                        help="Checkpoint nicht strikt laden (für ältere Checkpoints)")

    parser.add_argument("--region-based", action="store_true",
                        help="Älterer Modus: Annotationen basieren direkt auf den Graph-Regionen "
                             "(Standard: patch-basiert, unabhängig von Regiongrenzen)")
    parser.add_argument("--min-component-patches", type=int, default=3,
                        help="[Patch-Modus] Minimale Anzahl Patches pro Tumor-Komponente")

    parser.add_argument("--dataset", type=str, default="camelyon",
                        choices=["camelyon", "gsdb"],
                        help="Datensatz-Typ (beeinflusst Maskenerzeugung)")
    parser.add_argument("--preview", action="store_true",
                        help="Niedrig aufgelöste Vorschau-Overlay auf dem Slide-Thumbnail speichern")
    parser.add_argument("--device", type=str, default=None,
                        help="Gerät: 'cuda', 'cuda:<n>' oder 'cpu'. Standard: automatisch die "
                             "GPU mit dem meisten freien Speicher (geteilte Maschine -- welche "
                             "GPU frei ist, wechselt).")
    parser.add_argument("--keep-graph", action="store_true",
                        help="Erzeugten Graphen als .pt Datei speichern")

    parser = add_path_args(parser)
    return parser.parse_args()



def _load_meta(meta_file: str, paths: Dict[str, Any]) -> MetaData:
    candidate = Path(meta_file)
    if candidate.is_file():
        return MetaData.from_yaml(candidate)
    candidate = Path(paths["META_DIR"]) / "training_metas" / meta_file
    if candidate.is_file():
        return MetaData.from_yaml(candidate)
    raise FileNotFoundError(f"Meta-Datei nicht gefunden: {meta_file}")


def _apply_meta_to_args(args: argparse.Namespace, meta: MetaData) -> argparse.Namespace:
    md = meta.meta_dict or {}

    if args.checkpoint is None and meta.checkpoint_path:
        args.checkpoint = Path(str(meta.checkpoint_path))
        logging.info("Checkpoint aus Meta: %s", args.checkpoint)

    if args.embedding_model is None:
        emb = md.get("embedding_type") or md.get("embedding_model")
        if emb:
            emb_norm = str(emb).replace("-", "_").lower()
            if emb_norm in ("prov_gigapath", "uni2"):
                args.embedding_model = emb_norm
                logging.info("Embedding-Modell aus Meta: %s", args.embedding_model)

    if args.pipeline is None:
        pipeline = md.get("pipeline") or md.get("pipeline_type")
        if pipeline:
            args.pipeline = str(pipeline)
            logging.info("Pipeline aus Meta: %s", args.pipeline)

    model_cfg = md.get("model_config") or {}
    arch_cfg  = model_cfg.get("architecture") or {}

    _HPARAM: Dict[str, tuple] = {
        "hidden_dim":        ("hidden_dim",        320,  int),
        "num_layers":        ("num_layers",         4,    int),
        "heads":             ("heads",              4,    int),
        "dropout_rate":      ("dropout_rate",       0.20, float),
        "edge_dropout_p":    ("edge_dropout_p",     0.05, float),
        "region_aux_weight": ("region_aux_weight",  0.24, float),
    }
    for arg_key, (meta_key, default, cast) in _HPARAM.items():
        if getattr(args, arg_key, None) is None:
            val = arch_cfg.get(meta_key) or model_cfg.get(meta_key) or md.get(meta_key)
            setattr(args, arg_key, cast(val) if val is not None else default)
            if val is not None:
                logging.info("  %s aus Meta: %s", arg_key, getattr(args, arg_key))

    return args


def _fill_arg_defaults(args: argparse.Namespace) -> argparse.Namespace:
    _DEFAULTS = {
        "embedding_model":  "prov_gigapath",
        "pipeline":         "dynamical",
        "hidden_dim":       320,
        "num_layers":       4,
        "heads":            4,
        "dropout_rate":     0.20,
        "edge_dropout_p":   0.05,
        "region_aux_weight":0.24,
    }
    for key, val in _DEFAULTS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)
    return args


def _resolve_feature_image_size(args: argparse.Namespace) -> int:
    if getattr(args, "feature_image_size", None) is not None:
        return int(args.feature_image_size)

    model_name = str(getattr(args, "embedding_model", "") or "").lower()
    if model_name in {"prov_gigapath", "uni2"}:
        return 224

    return int(getattr(args, "patch_size", 256))



def _map_region_probs_to_patches(graph: Any, region_probs: torch.Tensor) -> torch.Tensor:
    try:
        edge_index = graph[("patch", "in", "region")].edge_index
    except Exception:
        logging.warning("Kein patch->region Edge-Index im Graph – Fallback auf region.centroid-Mapping.")
        return torch.zeros(0)

    if not torch.is_tensor(edge_index) or edge_index.shape[0] != 2:
        return torch.zeros(0)

    patch_idx = edge_index[0].cpu().long()
    region_idx = edge_index[1].cpu().long()
    probs_cpu = region_probs.detach().cpu().float().view(-1)

    n_patches = int(graph["patch"].num_nodes or 0)
    if n_patches == 0:
        n_patches = int(patch_idx.max().item()) + 1

    patch_probs = torch.zeros(n_patches, dtype=torch.float32)
    valid = region_idx < len(probs_cpu)
    patch_probs[patch_idx[valid]] = probs_cpu[region_idx[valid]]
    return patch_probs


def _build_patch_annotations(
    graph: Any,
    patch_probs: torch.Tensor,
    threshold: float,
    min_component_patches: int = 3,
) -> List[Dict[str, Any]]:
    import matplotlib.pyplot as plt
    try:
        from scipy.ndimage import label as scipy_label
        _HAS_SCIPY = True
    except ImportError:
        _HAS_SCIPY = False

    patch_store = graph["patch"]
    meta = getattr(graph, "meta", {}) or {}
    config = meta.get("config", {}) or {}

    patch_level = int(config.get("patch_level", 0))
    level_downsamples = meta.get("level_downsamples", [1.0]) or [1.0]
    downsample = float(level_downsamples[min(patch_level, len(level_downsamples) - 1)])
    patch_size = int(config.get("patch_size", 256))
    patch_size_l0 = float(patch_size) * downsample

    topleft = getattr(patch_store, "topleft", None)
    centroid = getattr(patch_store, "centroid", None)

    if torch.is_tensor(topleft) and topleft.dim() == 2 and topleft.shape[1] >= 2:
        tl_arr = topleft[:, :2].cpu().numpy().astype(np.float64) * downsample
    elif torch.is_tensor(centroid) and centroid.dim() == 2 and centroid.shape[1] >= 2:
        c_arr = centroid[:, :2].cpu().numpy().astype(np.float64) * downsample
        tl_arr = c_arr - patch_size_l0 / 2.0
    else:
        logging.warning("Keine patch.topleft / patch.centroid im Graphen – patch-basierte Annotation nicht möglich.")
        return []

    probs_np = patch_probs.cpu().numpy().astype(np.float32)
    if len(probs_np) != len(tl_arr):
        logging.warning(
            "Längenkonflikt: patch_probs=%d, patch_topleft=%d – patch-Annotation übersprungen.",
            len(probs_np), len(tl_arr),
        )
        return []

    positive = probs_np >= float(threshold)
    if not np.any(positive):
        logging.info("Keine Patches über Schwellwert %.3f – keine Annotationen erzeugt.", threshold)
        return []

    pos_tl = tl_arr[positive]
    pos_probs = probs_np[positive]

    x_min = float(np.min(pos_tl[:, 0]))
    x_max = float(np.max(pos_tl[:, 0])) + patch_size_l0
    y_min = float(np.min(pos_tl[:, 1]))
    y_max = float(np.max(pos_tl[:, 1])) + patch_size_l0

    span_x = max(1.0, x_max - x_min)
    span_y = max(1.0, y_max - y_min)
    max_raster = 2048
    scale = max(1.0, max(span_x, span_y) / float(max_raster))

    width  = int(np.ceil(span_x / scale)) + 3
    height = int(np.ceil(span_y / scale)) + 3
    patch_px = max(2, int(round(patch_size_l0 / scale)))

    logging.info(
        "Patch-Annotation: %d positive Patches | raster=%dx%d | patch_px=%d | scale=%.1f",
        int(np.sum(positive)), width, height, patch_px, scale,
    )

    raster = np.zeros((height, width), dtype=np.int32)
    for i in range(pos_tl.shape[0]):
        cx = int(round((float(pos_tl[i, 0]) - x_min) / scale))
        cy = int(round((float(pos_tl[i, 1]) - y_min) / scale))
        x0 = max(0, cx)
        x1 = min(width,  cx + patch_px)
        y0 = max(0, cy)
        y1 = min(height, cy + patch_px)
        if x1 > x0 and y1 > y0:
            raster[y0:y1, x0:x1] = 1

    if _HAS_SCIPY:
        labeled, num_comps = scipy_label(raster)
    else:
        labeled = raster.astype(np.int32)
        num_comps = 1

    min_area_px = int(min_component_patches) * int(patch_px * patch_px)

    components: List[Dict[str, Any]] = []
    out_id = 0
    for comp_id in range(1, num_comps + 1):
        comp_mask = (labeled == comp_id).astype(np.uint8)
        if int(np.sum(comp_mask)) < max(1, min_area_px // 4):
            continue

        fig, ax = plt.subplots(figsize=(1, 1))
        try:
            cs = ax.contour(comp_mask.astype(np.float32), levels=[0.5])
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
        finally:
            plt.close(fig)

        if not polygons:
            continue

        comp_patch_indices: List[int] = []
        for i in range(pos_tl.shape[0]):
            cx = int(round((float(pos_tl[i, 0]) - x_min) / scale))
            cy = int(round((float(pos_tl[i, 1]) - y_min) / scale))
            if 0 <= cy < height and 0 <= cx < width and labeled[cy, cx] == comp_id:
                comp_patch_indices.append(i)

        if not comp_patch_indices:
            continue

        c_idx = np.asarray(comp_patch_indices, dtype=np.int64)
        comp_tl = pos_tl[c_idx]
        comp_patch_probs = pos_probs[c_idx]
        bbox = [
            float(np.min(comp_tl[:, 0])),
            float(np.min(comp_tl[:, 1])),
            float(np.max(comp_tl[:, 0])) + patch_size_l0,
            float(np.max(comp_tl[:, 1])) + patch_size_l0,
        ]

        components.append({
            "component_id": out_id,
            "num_patches": int(len(comp_patch_indices)),
            "num_regions": 0,
            "mean_probability": float(np.mean(comp_patch_probs)),
            "max_probability":  float(np.max(comp_patch_probs)),
            "bbox_level_coords": bbox,
            "center_level_coords": [
                float(np.mean(comp_tl[:, 0]) + patch_size_l0 / 2.0),
                float(np.mean(comp_tl[:, 1]) + patch_size_l0 / 2.0),
            ],
            "contour_polygons_level_coords": polygons,
        })
        out_id += 1

    logging.info(
        "Patch-Annotation: %d Komponenten nach Größenfilter (min_patches=%d)",
        len(components), min_component_patches,
    )
    return components



def _load_feature_model(model_name: str) -> Any:
    import timm
    if model_name == "prov_gigapath":
        logging.info("Lade Feature-Modell: prov_gigapath")
        base = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath",
            pretrained=True,
            num_classes=0,
        ).eval()
        try:
            from src.entrypoint.build_graph import CudaAmpFeatureModel
            model = CudaAmpFeatureModel(base)
        except Exception:
            model = base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model

    if model_name == "uni2":
        logging.info("Lade Feature-Modell: uni2")
        kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kwargs).eval()
        return model

    raise ValueError(f"Unbekanntes Feature-Modell: {model_name}")


def _build_graph_cfg(args: argparse.Namespace) -> Tuple[BuildConfig, CreationConfig]:
    if args.graph_config is not None:
        logging.info("Lade Graph-Konfiguration aus: %s", args.graph_config)
        raw = _load_yaml_with_extends(args.graph_config)
        build_raw = raw.get("graph_build", raw).get("build", raw.get("graph_build", raw))
        creation_raw = raw.get("graph_build", raw).get("creation", {})
        if not isinstance(build_raw, dict):
            build_raw = {}
        if not isinstance(creation_raw, dict):
            creation_raw = {}
        cfg = BuildConfig(**{k: v for k, v in build_raw.items()
                             if k in BuildConfig.__dataclass_fields__})
        creation_cfg = CreationConfig(**{k: v for k, v in creation_raw.items()
                                         if k in CreationConfig.__dataclass_fields__})
        return cfg, creation_cfg

    cfg = BuildConfig(
        pipeline=str(args.pipeline),
        patch_size=int(args.patch_size),
        patch_level=int(args.patch_level),
        feature_image_size=int(_resolve_feature_image_size(args)),
    )
    creation_cfg = CreationConfig(
        use_annotations=False,
        use_annotation_labels=False,
        filter_graph_with_annotations=False,
    )
    return cfg, creation_cfg


def _load_yaml_with_extends(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    with open(resolved, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    extends = raw.pop("extends", None)
    if extends:
        extends_list = extends if isinstance(extends, list) else [extends]
        merged: Dict[str, Any] = {}
        for item in extends_list:
            parent = Path(str(item))
            if not parent.is_absolute():
                parent = resolved.parent / parent
            parent_cfg = _load_yaml_with_extends(parent)
            merged = _deep_merge(merged, parent_cfg)
        return _deep_merge(merged, raw)
    return raw


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result



def _load_tumor_model(
    checkpoint: Path,
    data: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> TissueRegionGraphModel:
    region_store = data["region"]
    region_x = getattr(region_store, "x", None)
    if not torch.is_tensor(region_x) or region_x.dim() != 2:
        raise ValueError("Keine region.x Merkmale im Graphen gefunden.")
    in_dim = int(region_x.size(1))
    logging.info("Region feature dim (in_dim): %d", in_dim)

    model = TissueRegionGraphModel(
        model_name="tissue_region_net",
        num_classes=1,
        in_dim=in_dim,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        heads=int(args.heads),
        dropout_rate=float(args.dropout_rate),
        edge_dropout_p=float(args.edge_dropout_p),
        region_aux_weight=float(args.region_aux_weight),
    )
    model = load_model(
        checkpoint,
        model,
        device=device,
        architecture="tissue_region_net",
        strict=not bool(args.no_strict_load),
    )
    logging.info("Checkpoint geladen: %s", checkpoint)
    return model



def _export_asap_xml(
    components: List[Dict[str, Any]],
    output_path: Path,
    slide_id: str,
    level_downsamples: Optional[List[float]] = None,
) -> Path:
    root = ET.Element("ASAP_Annotations")
    root.set("Name", slide_id)
    root.set("Description", "Synthetische Tumorannotation (TissueRegionGraphModel)")

    groups_el = ET.SubElement(root, "AnnotationGroups")
    group_el = ET.SubElement(groups_el, "Group")
    group_el.set("Name", "tumor")
    group_el.set("PartOfGroup", "None")
    group_el.set("Color", "#F4FA58")

    annotations_el = ET.SubElement(root, "Annotations")

    for comp in components:
        contours = comp.get("contour_polygons_level_coords") or []
        if not contours:
            x0, y0, x1, y1 = comp["bbox_level_coords"]
            contours = [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]

        for ring_idx, ring in enumerate(contours):
            ann_el = ET.SubElement(annotations_el, "Annotation")
            ann_el.set("Name", f"tumor_comp{comp['component_id']}_ring{ring_idx}")
            ann_el.set("PartOfGroup", "tumor")
            ann_el.set("Type", "Polygon")
            ann_el.set("MeanProbability", f"{comp['mean_probability']:.4f}")

            coords_el = ET.SubElement(ann_el, "Coordinates")
            for coord_idx, (cx, cy) in enumerate(ring):
                coord_el = ET.SubElement(coords_el, "Coordinate")
                coord_el.set("Order", str(coord_idx))
                coord_el.set("X", f"{float(cx):.2f}")
                coord_el.set("Y", f"{float(cy):.2f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path



def _save_preview(
    slide_path: Path,
    components: List[Dict[str, Any]],
    output_path: Path,
    threshold: float,
    thumbnail_max_size: int = 1400,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import openslide
    except Exception as exc:
        logging.warning("Vorschau übersprungen: %s", exc)
        return

    try:
        slide = openslide.OpenSlide(str(slide_path))
        w0, h0 = slide.dimensions
        scale = min(1.0, float(thumbnail_max_size) / float(max(1, max(w0, h0))))
        tw = max(64, int(round(w0 * scale)))
        th = max(64, int(round(h0 * scale)))
        thumb = np.asarray(slide.get_thumbnail((tw, th)).convert("RGB"))
        slide.close()
    except Exception as exc:
        logging.warning("Konnte Thumbnail nicht laden: %s", exc)
        return

    fig_w = max(6.0, float(tw) / 220.0)
    fig_h = max(6.0, float(th) / 220.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)
    ax.imshow(thumb)

    tumor_mask = np.zeros((th, tw), dtype=np.uint8)
    for comp in components:
        contours = comp.get("contour_polygons_level_coords") or []
        for ring in contours:
            ring_arr = np.asarray(ring, dtype=np.float32)
            if ring_arr.shape[0] < 3:
                continue
            px = np.clip(np.round(ring_arr[:, 0] * (float(tw) / float(max(1, w0)))).astype(np.int32), 0, tw - 1)
            py = np.clip(np.round(ring_arr[:, 1] * (float(th) / float(max(1, h0)))).astype(np.int32), 0, th - 1)
            from PIL import Image as PILImage, ImageDraw
            poly_img = PILImage.new("L", (tw, th), 0)
            poly_draw = ImageDraw.Draw(poly_img)
            coords_2d = list(zip(px.tolist(), py.tolist()))
            if len(coords_2d) >= 3:
                poly_draw.polygon(coords_2d, fill=1)
            tumor_mask = np.maximum(tumor_mask, np.asarray(poly_img, dtype=np.uint8))

    if np.any(tumor_mask > 0):
        overlay = np.zeros((th, tw, 4), dtype=np.float32)
        overlay[..., 0] = 1.0
        overlay[..., 1] = 0.08
        overlay[..., 2] = 0.08
        overlay[..., 3] = tumor_mask.astype(np.float32) * 0.38
        ax.imshow(overlay, origin="upper")
        ax.contour(tumor_mask, levels=[0.5], colors="white", linewidths=3.0, alpha=0.95, origin="upper")
        ax.contour(tumor_mask, levels=[0.5], colors="#ff2d2d", linewidths=1.8, alpha=0.98, origin="upper")

    n_comps = len(components)
    ax.set_title(
        f"{slide_path.name} | {n_comps} Tumor-Komponenten | thr={threshold:.2f}",
        fontsize=9,
    )
    ax.axis("off")
    fig.tight_layout(pad=0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, format="jpg",
                pil_kwargs={"quality": 88, "optimize": True})
    plt.close(fig)
    logging.info("Vorschau gespeichert: %s", output_path)



def main() -> None:
    """Run the end-to-end tumor-region annotation pipeline on a single slide."""
    args = _parse_args()

    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir, operation="annotate_tissue_regions")
    quiet_external_loggers()

    slide_path = args.slide.expanduser().resolve()
    if not slide_path.is_file():
        logging.error("Slide-Datei nicht gefunden: %s", slide_path)
        sys.exit(1)

    paths = get_project_paths(base_dir=getattr(args, "base_dir", None))
    if args.meta_file:
        try:
            meta = _load_meta(args.meta_file, paths)
            logging.info("Meta-Datei geladen: %s", args.meta_file)
            args = _apply_meta_to_args(args, meta)
        except FileNotFoundError as exc:
            logging.error("%s", exc)
            sys.exit(1)

    args = _fill_arg_defaults(args)

    if args.checkpoint is None:
        logging.error(
            "Kein Checkpoint angegeben. --checkpoint setzen oder "
            "--meta-file mit gespeichertem checkpoint_path verwenden."
        )
        sys.exit(1)

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        logging.error("Checkpoint nicht gefunden: %s", checkpoint)
        sys.exit(1)

    device = torch.device(args.device) if args.device else _select_least_busy_device()
    logging.info("=" * 60)
    logging.info("Slide:            %s", slide_path)
    logging.info("Checkpoint:       %s", checkpoint)
    logging.info("Embedding-Modell: %s", args.embedding_model)
    logging.info("Pipeline:         %s", args.pipeline)
    logging.info("Threshold:        %.3f", args.threshold)
    logging.info("Modellparameter:  hidden_dim=%d layers=%d heads=%d dropout=%.2f",
                 args.hidden_dim, args.num_layers, args.heads, args.dropout_rate)
    logging.info("Patch-Modus:      %s", "region-basiert" if args.region_based else "patch-basiert")
    logging.info("Gerät:            %s", device)
    logging.info("=" * 60)

    logging.info("Lade Feature-Extractor '%s'...", args.embedding_model)
    feature_model = _load_feature_model(args.embedding_model)
    feature_model = feature_model.to(device)

    logging.info("Baue Graph aus Slide (Pipeline='%s', patch_size=%d)...",
                 args.pipeline, args.patch_size)
    build_cfg, creation_cfg = _build_graph_cfg(args)

    dataset_enum = DatasetType.CAMELYON17 if str(args.dataset).lower() == "camelyon" else DatasetType.GSDB
    try:
        graph = build_graph_from_slide(
            slide=slide_path,
            cfg=build_cfg,
            node_feature_model=feature_model,
            creation_cfg=creation_cfg,
            dataset=dataset_enum,
        )
    except Exception as exc:
        logging.error("Graph-Bau fehlgeschlagen: %s\n%s", exc, traceback.format_exc())
        sys.exit(1)

    del feature_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.keep_graph:
        graph_path = out_dir / f"{slide_path.stem}_graph.pt"
        torch.save(graph, graph_path)
        logging.info("Graph gespeichert: %s", graph_path)

    logging.info(
        "Graph gebaut: %d Patch-Knoten, %d Region-Knoten",
        int(getattr(getattr(graph, "patch", None), "num_nodes", 0) or 0),
        int(getattr(getattr(graph, "region", None), "num_nodes", 0) or 0),
    )

    logging.info("Lade TissueRegionGraphModel...")
    tumor_model = _load_tumor_model(checkpoint, graph, args, device)

    logging.info("Inferenz läuft...")
    region_probs = infer_region_probabilities(
        data=graph,
        model=tumor_model,
        device=str(device),
    )
    logging.info(
        "Region-Wahrscheinlichkeiten: n=%d | min=%.3f | mean=%.3f | max=%.3f",
        int(region_probs.numel()),
        float(region_probs.min()),
        float(region_probs.mean()),
        float(region_probs.max()),
    )

    if getattr(args, "region_based", False):
        logging.info("[Modus: region-basiert] Extrahiere Tumor-Komponenten (thr=%.3f)...", args.threshold)
        components = extract_tissue_region_components(
            data=graph,
            region_probabilities=region_probs,
            threshold=float(args.threshold),
            min_component_size=int(args.min_component_size),
            min_mean_probability=float(args.min_mean_prob),
        )
        annotation_mode = "region_based"
    else:
        logging.info("[Modus: patch-basiert] Mappe Region-Probs auf Patches...")
        patch_probs = _map_region_probs_to_patches(graph, region_probs)
        logging.info(
            "Patch-Wahrscheinlichkeiten: n=%d | min=%.3f | mean=%.3f | max=%.3f",
            int(patch_probs.numel()),
            float(patch_probs.min()) if patch_probs.numel() > 0 else 0.0,
            float(patch_probs.mean()) if patch_probs.numel() > 0 else 0.0,
            float(patch_probs.max()) if patch_probs.numel() > 0 else 0.0,
        )
        if patch_probs.numel() == 0:
            logging.error("Patch-Probs konnten nicht erzeugt werden (kein patch->region Edge-Index). "
                          "Verwende --region-based als Fallback.")
            sys.exit(1)
        logging.info("Erzeuge Patch-Annotationen (thr=%.3f, min_patches=%d)...",
                     args.threshold, args.min_component_patches)
        components = _build_patch_annotations(
            graph=graph,
            patch_probs=patch_probs,
            threshold=float(args.threshold),
            min_component_patches=int(args.min_component_patches),
        )
        annotation_mode = "patch_based"

    logging.info("Gefunden: %d Tumor-Komponenten [%s]", len(components), annotation_mode)
    for comp in components:
        unit = "Patches" if annotation_mode == "patch_based" else "Regionen"
        count = comp.get("num_patches") or comp.get("num_regions", 0)
        logging.info(
            "  Komponente %d: %d %s | mean_prob=%.3f | max_prob=%.3f | bbox=%s",
            comp["component_id"],
            count,
            unit,
            comp["mean_probability"],
            comp["max_probability"],
            [f"{v:.0f}" for v in comp["bbox_level_coords"]],
        )

    slide_id = slide_path.stem

    geojson_path = out_dir / f"{slide_id}_tumor_annotation.geojson"
    export_components_as_geojson(
        components=components,
        output_path=geojson_path,
        slide_id=slide_id,
    )
    logging.info("GeoJSON gespeichert: %s", geojson_path)

    xml_path = out_dir / f"{slide_id}_tumor_annotation.xml"
    level_downsamples = None
    if hasattr(graph, "meta") and isinstance(graph.meta, dict):
        level_downsamples = graph.meta.get("level_downsamples")
    _export_asap_xml(
        components=components,
        output_path=xml_path,
        slide_id=slide_id,
        level_downsamples=level_downsamples,
    )
    logging.info("ASAP-XML gespeichert: %s", xml_path)

    summary_path = out_dir / f"{slide_id}_tumor_annotation_summary.json"
    summary = {
        "slide": str(slide_path),
        "checkpoint": str(checkpoint),
        "embedding_model": str(args.embedding_model),
        "annotation_mode": annotation_mode,
        "threshold": float(args.threshold),
        "num_components": int(len(components)),
        "components": [
            {
                "component_id": comp["component_id"],
                "num_regions": comp["num_regions"],
                "mean_probability": comp["mean_probability"],
                "max_probability": comp["max_probability"],
                "bbox_level_coords": comp["bbox_level_coords"],
            }
            for comp in components
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Zusammenfassung: %s", summary_path)

    if args.preview:
        preview_path = out_dir / f"{slide_id}_tumor_annotation_preview.jpg"
        _save_preview(
            slide_path=slide_path,
            components=components,
            output_path=preview_path,
            threshold=float(args.threshold),
        )

    logging.info("Fertig. Alle Ausgaben in: %s", out_dir)
    logging.info(
        "Ausgabedateien:\n  GeoJSON: %s\n  ASAP-XML: %s\n  Zusammenfassung: %s",
        geojson_path,
        xml_path,
        summary_path,
    )


if __name__ == "__main__":
    main()
