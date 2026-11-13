from pathlib import Path
import xml.etree.ElementTree as ET
import json
from collections import Counter

def analyze_asap_directory(root_dir: Path):
    """Scan a directory of ASAP XML annotation files and collect summary statistics."""

    stats = {
        "annotation_count": 0,
        "groups": Counter(),
        "types": Counter(),
        "group_set": set(),
        "files_processed": 0,
    }

    for xml_file in root_dir.rglob("*.xml"):

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except Exception:
            continue

        annotations = root.find(".//Annotations")
        if annotations is None:
            continue

        stats["files_processed"] += 1

        for ann in annotations.findall(".//Annotation"):

            stats["annotation_count"] += 1

            group = ann.attrib.get("PartOfGroup")
            ann_type = ann.attrib.get("Type")

            if group:
                stats["groups"][group] += 1
                stats["group_set"].add(group)

            if ann_type:
                stats["types"][ann_type] += 1

    return stats

def analyze_geojson_directory(root_dir: Path):
    """Scan a directory of GeoJSON annotation files and collect summary statistics."""

    stats = {
        "annotation_count": 0,
        "tissue_types": Counter(),
        "inflammation_status": Counter(),
        "cluster_ids": Counter(),
        "files_processed": 0
    }

    for json_file in root_dir.rglob("*.json"):

        try:
            data = json.loads(json_file.read_text())
        except Exception:
            continue

        if isinstance(data, dict) and "features" in data:
            features = data["features"]
        elif isinstance(data, list):
            features = data
        else:
            continue

        stats["files_processed"] += 1

        for f in features:

            if f.get("type") != "Feature":
                continue

            props = f.get("properties", {})
            cls = props.get("classification", {})

            stats["annotation_count"] += 1

            t = cls.get("tissue_type")
            if t:
                stats["tissue_types"][t] += 1

            i = cls.get("inflammation_status")
            if i:
                stats["inflammation_status"][i] += 1

            cid = props.get("cluster_id")
            if cid is not None:
                stats["cluster_ids"][cid] += 1

    return stats

def build_global_stats(asap_root: Path, geo_root: Path):
    """Build combined ASAP and GeoJSON annotation statistics."""

    return {
        "asap": analyze_asap_directory(asap_root),
        "geojson": analyze_geojson_directory(geo_root),
    }