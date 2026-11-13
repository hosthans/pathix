import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import xml.etree.ElementTree as ET
import logging
import cv2
import numpy as np

def load_asap_annotation(path: Optional[Path]) -> List[Dict]:
    """Load annotations from an ASAP XML file."""

    if path is None:
        return []

    if not path.exists():
        logging.info("ASAP annotation file not found: %s", path)
        return []

    tree = ET.parse(path)
    root = tree.getroot()

    annotations = []

    annotations_node = root.find("Annotations")
    if annotations_node is None:
        return annotations

    for anno in annotations_node.findall("Annotation"):
        name = anno.get("Name")
        anno_type = anno.get("Type")
        group = anno.get("PartOfGroup")
        color = anno.get("Color")

        coords_node = anno.find("Coordinates")
        if coords_node is None:
            continue

        coords: List[Tuple[float, float]] = []

        coord_elements = coords_node.findall("Coordinate")
        coord_elements_sorted = sorted(
            coord_elements,
            key=lambda c: int(c.get("Order", 0))
        )

        for c in coord_elements_sorted:
            x = float(c.get("X"))
            y = float(c.get("Y"))
            coords.append((x, y))

        annotations.append({
            "name": name,
            "type": anno_type,
            "group": group,
            "color": color,
            "coordinates": coords
        })

    return annotations

def load_geojson_annotations(path: Optional[Path]) -> List[Dict[str, Any]]:
    """Load annotations from a GeoJSON file."""
    if path is None:
        return []

    if not path.exists():
        logging.info("GeoJSON annotation file not found: %s", path)
        return []

    with open(path, "r") as f:
        data = json.load(f)

    features: List[Dict[str, Any]]
    if isinstance(data, dict):
        features_raw = data.get("features", [])
        features = features_raw if isinstance(features_raw, list) else []
    elif isinstance(data, list):
        features = data
    else:
        return []

    annotations: List[Dict[str, Any]] = []

    for feature in features:
        if feature.get("type") != "Feature":
            continue

        geometry = feature.get("geometry")
        if geometry is None:
            continue

        geom_type = geometry.get("type")
        coords_raw = geometry.get("coordinates")

        if geom_type != "Polygon":
            logging.debug("Skipping non-Polygon geometry in %s: %s", path, geom_type)
            continue

        if not coords_raw:
            continue

        outer_ring = coords_raw[0]

        coords: List[Tuple[float, float]] = [
            (float(x), float(y)) for x, y in outer_ring
        ]

        annotations.append({
            "id": feature.get("id"),
            "type": geom_type,
            "classification": feature.get("properties", {}).get("classification"),
            "properties": feature.get("properties", {}),
            "coordinates": coords
        })

    return annotations


def is_positive_label_text(label_text: str) -> bool:
    """Check whether a label string indicates a positive (tumor/inflamed) finding."""
    text = str(label_text).strip().lower()
    return any(token in text for token in ("tumor", "metast", "macro", "micro", "itc", "positive", "status=inflamed"))


def annotation_label(
    annotation: Dict[str, Any],
    source: str = "auto",
    classification_field: Optional[str] = None,
) -> Optional[str]:
    """Derive a normalized label string for an annotation from the given source."""
    if not annotation:
        return None

    normalized_source = str(source or "auto").strip().lower()

    if normalized_source in {"auto", "classification"}:
        classification = annotation.get("classification")
        if isinstance(classification, dict) and classification:
            if classification_field is not None:
                value = classification.get(classification_field)
                if value is not None:
                    field = str(classification_field).strip().lower()
                    return f"classification:{field}={str(value).strip().lower()}"
            else:
                items = [f"{str(key).strip().lower()}={str(value).strip().lower()}" for key, value in sorted(classification.items())]
                return "classification:" + "|".join(items)

    if normalized_source in {"auto", "group"}:
        group = annotation.get("group")
        if isinstance(group, str) and group.strip():
            return "group:" + group.strip().lower()

    if normalized_source in {"auto", "name"}:
        name = annotation.get("name")
        if isinstance(name, str) and name.strip():
            return "name:" + name.strip().lower()

    if annotation.get("id") is not None:
        return f"id:{annotation['id']}"

    return None


def annotation_contains_point(annotation: Dict[str, Any], point_xy: Tuple[float, float]) -> bool:
    """Check whether a point lies inside an annotation's polygon, caching the polygon array on the annotation."""
    polygon = annotation.get("_cached_polygon")
    if polygon is None:
        coordinates = annotation.get("coordinates") or []
        if len(coordinates) < 3:
            return False
        polygon = np.asarray(coordinates, dtype=np.float32)
        annotation["_cached_polygon"] = polygon
    return bool(cv2.pointPolygonTest(polygon, (float(point_xy[0]), float(point_xy[1])), False) >= 0)


def _match_rate(
    annotations: List[Dict[str, Any]],
    points_level0: np.ndarray,
    swap_xy: bool,
) -> float:
    if points_level0.shape[0] == 0:
        return 0.0
    matched = np.zeros((points_level0.shape[0],), dtype=bool)
    for annotation in annotations:
        coordinates = annotation.get("coordinates") or []
        if len(coordinates) < 3:
            continue
        polygon = np.asarray(coordinates, dtype=np.float32)
        if swap_xy:
            polygon = polygon[:, ::-1]
        remaining = np.where(~matched)[0]
        for idx in remaining:
            x, y = float(points_level0[idx, 0]), float(points_level0[idx, 1])
            if cv2.pointPolygonTest(polygon, (x, y), False) >= 0:
                matched[idx] = True
        if matched.all():
            break
    return float(matched.mean())


def assess_annotation_quality(
    annotations: Optional[List[Dict[str, Any]]],
    patch_xy: np.ndarray,
    downsample: float,
    sample_size: Optional[int] = None,
    match_rate_floor: float = 0.0,
    swap_factor: float = 5.0,
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """Assess whether a slide's loaded annotations plausibly match its patch grid, detecting x/y coordinate swaps."""
    n_patches = int(patch_xy.shape[0])
    ann_list = annotations or []
    if not ann_list:
        return {
            "reason": "no_annotations",
            "n_annotations": 0,
            "n_patches": n_patches,
            "match_rate_normal": 0.0,
            "match_rate_swapped": 0.0,
        }

    points = np.asarray(patch_xy, dtype=np.float32) * float(downsample)
    if sample_size is not None and n_patches > int(sample_size):
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(n_patches, size=int(sample_size), replace=False)
        points = points[idx]

    match_rate_normal = _match_rate(ann_list, points, swap_xy=False)
    match_rate_swapped = _match_rate(ann_list, points, swap_xy=True)

    if match_rate_normal > match_rate_floor:
        reason = "ok"
    elif match_rate_swapped > match_rate_floor and match_rate_swapped >= match_rate_normal * swap_factor:
        reason = "mismatched_xy_swap"
    else:
        reason = "mismatched_unknown"

    return {
        "reason": reason,
        "n_annotations": len(ann_list),
        "n_patches": n_patches,
        "match_rate_normal": match_rate_normal,
        "match_rate_swapped": match_rate_swapped,
    }


def build_annotation_label_vocab(
    annotations: List[Dict[str, Any]],
    source: str = "auto",
    classification_field: Optional[str] = None,
) -> List[str]:
    """Build a sorted list of unique annotation labels for a set of annotations."""
    labels = []
    for annotation in annotations:
        label = annotation_label(annotation, source=source, classification_field=classification_field)
        if label is not None:
            labels.append(label)
    return sorted(set(labels))