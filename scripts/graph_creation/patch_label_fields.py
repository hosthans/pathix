"""Adds CreationConfig.label_fields tensors (e.g. "status"/"subtype"/"stage"/"tissue_type") onto
ALREADY-BUILT cached graphs, without re-running the WSI/mask/patch-extraction/GPU-feature-encoding
pipeline. Whether this is possible for a given field depends only on what's already stored on the
cached graph -- and for label_fields sourced from "annotation_primary" (e.g. GSDB/CAMELYON
"status") or "annotation_extra" (e.g. GSDB "tissue_type"), everything needed
(annotation_y/{field}_y, region.label_ambiguous, the label vocabs) is ALREADY on every graph built
with use_annotation_labels=True (both production pipelines already set this). "csv_join" fields
(CAMELYON "subtype"/"stage") only need the slide's own file name, also already stored as
data.meta["wsi_path"]. None of this needs the original WSI file, the feature-encoder model, or
edge reconstruction -- see src/graph_creation_new/helper/labels.py's resolve_label_fields for the
actual resolution logic (this script is a thin driver around the exact same function used at
graph-build time, via apply_label_field_data, so both stay numerically identical).

If a field's node_types/source combination genuinely needs data that was never cached (e.g.
requesting "annotation_extra" for a field that wasn't in that graph's
extra_annotation_classification_fields at build time), that field resolves to "unknown"
everywhere for the affected graphs, same as CreationConfig.label_fields already does at
build time when the underlying signal is missing -- this script does not invent data, only
recombines what's already there.

Usage:
    python scripts/graph_creation/patch_label_fields.py \\
        --input-dir /path/to/graphs/camelyon/pl-dynamical_pov-gigapath \\
        --output-dir /path/to/graphs/camelyon/pl-dynamical_pov-gigapath_labeled \\
        --pipeline-config configs/graph_build/context_aware_pipeline_camelyon_labels.yaml

    # Preview without writing anything:
    python scripts/graph_creation/patch_label_fields.py --input-dir ... --output-dir ... \\
        --pipeline-config ... --dry-run

    # Modify the cached graphs in place instead of writing to a new directory (no --output-dir):
    python scripts/graph_creation/patch_label_fields.py --input-dir ... --pipeline-config ... \\
        --overwrite
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.entrypoint.build_graph import _load_yaml_with_extends  # noqa: E402
from src.graph_creation_new.config import CreationConfig  # noqa: E402
from src.graph_creation_new.helper.labels import apply_label_field_data, resolve_label_fields  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of already-cached .pt graphs (searched recursively)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to write patched copies into, mirroring --input-dir's relative structure. Required unless --overwrite.")
    parser.add_argument("--overwrite", action="store_true", help="Write patched graphs back over the originals in --input-dir instead of --output-dir. Mutually exclusive with --output-dir.")
    parser.add_argument("--pipeline-config", type=Path, required=True, help="Graph-build pipeline YAML to read graph_build.creation.label_fields from (e.g. configs/graph_build/context_aware_pipeline_camelyon_labels.yaml)")
    parser.add_argument("--pattern", type=str, default="**/*.pt", help="Glob pattern (relative to --input-dir) selecting which cached files to patch. Default: **/*.pt")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of files processed (for a quick trial run)")
    parser.add_argument("--dry-run", action="store_true", help="Resolve label_fields and report per-field class counts, but write nothing")
    args = parser.parse_args()

    if bool(args.overwrite) == bool(args.output_dir is not None):
        parser.error("Specify exactly one of --output-dir or --overwrite")

    return args


def _load_label_fields(pipeline_config_path: Path) -> List[Dict[str, Any]]:
    raw = _load_yaml_with_extends(pipeline_config_path)
    creation_section = dict(raw.get("graph_build", {}).get("creation", {}) or {})
    cfg = CreationConfig(**creation_section)  # validates label_fields, same as build_graph.py does
    if not cfg.label_fields:
        raise ValueError(
            f"'{pipeline_config_path}' has no graph_build.creation.label_fields entries -- nothing to patch. "
            "Use one of the *_labels.yaml pipeline configs (or add label_fields to a custom one)."
        )
    return cfg.label_fields


def _as_numpy_or_none(store: Any, attr: str):
    if not hasattr(store, attr):
        return None
    return getattr(store, attr).detach().cpu().numpy()


def _build_extra_label_data(data: Any) -> Dict[str, tuple]:
    """Reconstructs the (patch_ids, region_ids, vocab) tuples resolve_label_fields expects for
    source="annotation_extra", straight from what CreationConfig.extra_annotation_classification_
    fields already wrote onto the cached graph at build time (data["patch"/"region"].{field}_y +
    data.meta[f"{field}_label_vocab"] / data.meta["extra_label_fields"]) -- see
    helper/subgraph.py's create_subgraph for the original write side.
    """
    extra_label_data: Dict[str, tuple] = {}
    for field_name in data.meta.get("extra_label_fields", []) or []:
        patch_ids = _as_numpy_or_none(data["patch"], f"{field_name}_y")
        region_ids = _as_numpy_or_none(data["region"], f"{field_name}_y")
        vocab = list(data.meta.get(f"{field_name}_label_vocab", []))
        extra_label_data[field_name] = (patch_ids, region_ids, vocab)
    return extra_label_data


def _patch_one_graph(path: Path, label_fields: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")

    if not hasattr(data, "node_types") or not {"slide", "region", "patch"} <= set(data.node_types):
        logging.warning("Skipping '%s': not a slide/region/patch HeteroData graph", path)
        return None

    if not hasattr(data, "meta") or data.meta is None:
        data.meta = {}

    slide_path = data.meta.get("wsi_path") or path.stem
    context = SimpleNamespace(slide_path=slide_path, creation_cfg=SimpleNamespace(label_fields=label_fields))

    n_patches = int(getattr(data["patch"], "num_nodes", 0) or 0)
    n_regions = int(getattr(data["region"], "num_nodes", 0) or 0)

    label_field_data = resolve_label_fields(
        context,
        n_patches=n_patches,
        n_regions=n_regions,
        patch_annotation_ids=_as_numpy_or_none(data["patch"], "annotation_y"),
        region_annotation_ids=_as_numpy_or_none(data["region"], "annotation_y"),
        annotation_vocab=list(data.meta.get("annotation_label_vocab", [])),
        region_label_ambiguous=_as_numpy_or_none(data["region"], "label_ambiguous"),
        extra_label_data=_build_extra_label_data(data),
    )

    apply_label_field_data(data, label_field_data)

    summary = {
        name: payload["vocab"][payload["slide"]] if payload.get("slide") is not None else None
        for name, payload in label_field_data.items()
    }
    return {"data": data, "summary": summary}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    label_fields = _load_label_fields(args.pipeline_config)
    logging.info("Patching label_fields: %s", [f["name"] for f in label_fields])

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"--input-dir does not exist or is not a directory: {input_dir}")

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None else None

    paths = sorted(input_dir.glob(args.pattern))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No files matched '{args.pattern}' under {input_dir}")

    slide_counters: Dict[str, Counter] = {f["name"]: Counter() for f in label_fields}
    n_ok, n_skipped = 0, 0

    for path in paths:
        result = _patch_one_graph(path, label_fields)
        if result is None:
            n_skipped += 1
            continue

        for field_name, value in result["summary"].items():
            slide_counters[field_name][value] += 1

        if not args.dry_run:
            if args.overwrite:
                out_path = path
            else:
                out_path = output_dir / path.relative_to(input_dir)
                out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(result["data"], out_path)

        n_ok += 1

    logging.info("%s%d graphs patched, %d skipped", "[DRY RUN] " if args.dry_run else "", n_ok, n_skipped)
    for field_name, counter in slide_counters.items():
        logging.info("  slide-level %s counts: %s", field_name, dict(counter))


if __name__ == "__main__":
    main()
