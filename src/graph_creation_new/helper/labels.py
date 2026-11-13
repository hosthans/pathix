from __future__ import annotations

import csv as csv_module
import functools
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.graph_creation_new.helper.annotations import is_positive_label_text

# Matches CAMELYON17's "patient_<id>_node_<n>" slide-stem convention to recover the
# patient-level key ("patient_<id>") for the join_on="patient_zip" case -- stages.csv's
# per-patient pN-stage rows are keyed "patient_<id>.zip" (see resolve_csv_join). Slides from
# other datasets simply won't match, which resolve_csv_join treats as "no value" (-> unknown),
# not an error.
_PATIENT_PREFIX_RE = re.compile(r"^(patient_\d+)", re.IGNORECASE)


def normalize_categorical(
    raw_ids: np.ndarray,
    raw_vocab: List[str],
    ambiguous_mask: Optional[np.ndarray],
    collapse: Optional[str],
) -> Tuple[np.ndarray, List[str]]:
    """Map raw annotation-vocab ids onto a small, fixed output vocab, applying an ambiguous-mask override to "mixed"."""
    raw_ids = np.asarray(raw_ids, dtype=np.int64)
    n = int(raw_ids.shape[0])

    if collapse == "positive_negative":
        vocab = ["negative", "positive", "unknown", "mixed"]
        negative_id, positive_id, unknown_id, mixed_id = 0, 1, 2, 3
        mapping = np.full((len(raw_vocab),), unknown_id, dtype=np.int64)
        for raw_id, label in enumerate(raw_vocab):
            mapping[raw_id] = positive_id if is_positive_label_text(label) else negative_id
    elif collapse is None:
        vocab = list(raw_vocab) + ["unknown", "mixed"]
        unknown_id = len(raw_vocab)
        mixed_id = len(raw_vocab) + 1
        mapping = np.arange(len(raw_vocab), dtype=np.int64)
    else:
        raise ValueError(f"Unsupported collapse: {collapse!r}")

    new_ids = np.full((n,), unknown_id, dtype=np.int64)
    if mapping.size > 0:
        valid = (raw_ids >= 0) & (raw_ids < mapping.size)
        new_ids[valid] = mapping[raw_ids[valid]]

    if ambiguous_mask is not None:
        ambiguous_mask = np.asarray(ambiguous_mask, dtype=bool)
        if int(ambiguous_mask.shape[0]) == n:
            new_ids[ambiguous_mask] = mixed_id

    return new_ids, vocab


def aggregate_majority_mixed(
    child_ids: np.ndarray,
    unknown_id: int,
    mixed_id: int,
    mixed_threshold: float = 0.9,
) -> int:
    """Aggregate a node's children's label ids into a single parent-level class id by majority vote."""
    child_ids = np.asarray(child_ids, dtype=np.int64)
    known = child_ids[child_ids != unknown_id]
    if known.size == 0:
        return int(unknown_id)

    values, counts = np.unique(known, return_counts=True)
    winner_idx = int(np.argmax(counts))
    winner_share = float(counts[winner_idx]) / float(known.size)
    if winner_share >= float(mixed_threshold):
        return int(values[winner_idx])
    return int(mixed_id)


@functools.lru_cache(maxsize=8)
def _load_csv_lookup(
    csv_path: str, key_column: str, value_column: str
) -> Tuple[Dict[str, str], Tuple[str, ...], Tuple[str, ...]]:
    lookup: Dict[str, str] = {}
    tif_values: set = set()
    zip_values: set = set()
    with open(csv_path, "r", newline="") as f:
        reader = csv_module.DictReader(f)
        fieldnames = reader.fieldnames or []
        if key_column not in fieldnames or value_column not in fieldnames:
            raise ValueError(
                f"CSV '{csv_path}' must have columns '{key_column}' and '{value_column}', found: {fieldnames}"
            )
        for row in reader:
            key = str(row.get(key_column, "")).strip().lower()
            value = str(row.get(value_column, "")).strip()
            if not key or not value:
                continue
            lookup[key] = value
            if key.endswith(".tif"):
                tif_values.add(value)
            elif key.endswith(".zip"):
                zip_values.add(value)
    return lookup, tuple(sorted(tif_values)), tuple(sorted(zip_values))


def resolve_csv_join(
    csv_path: str,
    key_column: str,
    value_column: str,
    join_on: str,
    context: Any,
) -> Tuple[Optional[str], List[str]]:
    """Look up this slide's or its patient's value for a field in an external metadata CSV."""
    lookup, tif_vocab, zip_vocab = _load_csv_lookup(str(csv_path), key_column, value_column)
    stem = Path(context.slide_path).stem

    if join_on == "slide_stem":
        vocab = tif_vocab
        key = f"{stem}.tif".strip().lower()
    elif join_on == "patient_zip":
        vocab = zip_vocab
        match = _PATIENT_PREFIX_RE.match(stem)
        if match is None:
            return None, list(vocab)
        key = f"{match.group(1)}.zip".strip().lower()
    else:
        raise ValueError(f"Unsupported join_on: {join_on!r}")

    return lookup.get(key), list(vocab)


def resolve_label_fields(
    context: Any,
    *,
    n_patches: int,
    n_regions: int,
    patch_annotation_ids: Optional[np.ndarray],
    region_annotation_ids: Optional[np.ndarray],
    annotation_vocab: List[str],
    region_label_ambiguous: Optional[np.ndarray],
    extra_label_data: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]],
) -> Dict[str, Dict[str, Any]]:
    """Resolve each configured label field into a per-patch/region/slide id payload."""
    results: Dict[str, Dict[str, Any]] = {}
    label_fields = list(getattr(context.creation_cfg, "label_fields", None) or [])
    if not label_fields:
        return results

    for entry in label_fields:
        name = entry["name"]
        node_types = set(entry["node_types"])
        source = entry["source"]
        collapse = entry.get("collapse")
        mixed_threshold = float(entry.get("mixed_threshold", 0.9))

        payload: Dict[str, Any] = {"patch": None, "region": None, "slide": None, "vocab": []}

        if source == "csv_join":
            value, vocab = resolve_csv_join(
                csv_path=entry["csv_path"],
                key_column=entry["csv_key_column"],
                value_column=entry["csv_value_column"],
                join_on=entry["join_on"],
                context=context,
            )
            full_vocab = list(vocab) + ["unknown"]
            unknown_id = len(vocab)
            slide_id = full_vocab.index(value) if value is not None and value in vocab else unknown_id
            payload["vocab"] = full_vocab
            if "slide" in node_types:
                payload["slide"] = int(slide_id)
            results[name] = payload
            continue

        if source == "annotation_primary":
            raw_patch_ids, raw_region_ids, raw_vocab = patch_annotation_ids, region_annotation_ids, annotation_vocab
        elif source == "annotation_extra":
            field_data = extra_label_data.get(entry["extra_field"])
            raw_patch_ids, raw_region_ids, raw_vocab = field_data if field_data is not None else (None, None, [])
        else:
            raise ValueError(f"Unsupported label_fields source: {source!r}")

        if raw_patch_ids is None or raw_region_ids is None:
            # No annotation signal available at all for this field (e.g. use_annotation_labels
            # is off, or an annotation_extra field never got populated for this slide) --
            # everything is "unknown", but the vocab stays well-formed and consistent with the
            # signal-present case below.
            vocab = ["negative", "positive", "unknown", "mixed"] if collapse == "positive_negative" else ["unknown", "mixed"]
            unknown_id = vocab.index("unknown")
            new_patch_ids = np.full((n_patches,), unknown_id, dtype=np.int64) if n_patches else np.zeros((0,), dtype=np.int64)
            new_region_ids = np.full((n_regions,), unknown_id, dtype=np.int64) if n_regions else np.zeros((0,), dtype=np.int64)
        else:
            new_patch_ids, vocab = normalize_categorical(raw_patch_ids, raw_vocab, ambiguous_mask=None, collapse=collapse)
            new_region_ids, _ = normalize_categorical(
                raw_region_ids, raw_vocab, ambiguous_mask=region_label_ambiguous, collapse=collapse
            )
            unknown_id = vocab.index("unknown")

        mixed_id = vocab.index("mixed")
        payload["vocab"] = vocab
        if "patch" in node_types:
            payload["patch"] = new_patch_ids
        if "region" in node_types:
            payload["region"] = new_region_ids
        if "slide" in node_types:
            source_ids = new_region_ids if new_region_ids.size > 0 else new_patch_ids
            if source_ids.size == 0:
                payload["slide"] = unknown_id
            else:
                payload["slide"] = aggregate_majority_mixed(
                    source_ids, unknown_id=unknown_id, mixed_id=mixed_id, mixed_threshold=mixed_threshold
                )

        results[name] = payload

    return results


def apply_label_field_data(data: Any, label_field_data: Dict[str, Dict[str, Any]]) -> None:
    """Write resolve_label_fields' output onto an already-assembled HeteroData."""
    data.meta["label_fields"] = sorted((label_field_data or {}).keys())
    for field_name, payload in (label_field_data or {}).items():
        y_attr = f"{field_name}_y"
        if payload.get("patch") is not None:
            data["patch"][y_attr] = torch.tensor(payload["patch"], dtype=torch.long)
        if payload.get("region") is not None:
            data["region"][y_attr] = torch.tensor(payload["region"], dtype=torch.long)
        if payload.get("slide") is not None:
            data["slide"][y_attr] = torch.tensor([payload["slide"]], dtype=torch.long)
        data.meta[f"{field_name}_label_vocab"] = payload.get("vocab", [])
