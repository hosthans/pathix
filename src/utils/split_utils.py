"""Helpers for CAMELYON17 split creation (src/entrypoint/create_camelyon_splits.py): stage → tumor-status
mapping, slide-table loading, annotation-composition filtering, 1:1 tumor:normal balancing and
patient-disjointness verification. GSDB splits: src/entrypoint/create_splits.py.
"""
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd

STAGES_CSV = Path("/data/local/hweber/camelyon17/stages.csv")
PROCESSED_CSV = Path("/data/local/hweber/camelyon17/processed_slides.csv")

NORMAL_LABELS = {"negative", "normal", "benign", "none", "0"}
TUMOR_LABELS = {"tumor", "positive", "itc", "micro", "macro", "1", "2", "3"}

SPLIT_COLUMNS = [
    "slide_name", "slide_file", "patient_id", "stage", "inflammation_status",
    "tumor_status", "center", "annotation_used",
]


def stage_to_tumor_status(value: str) -> str:
    """Map a raw stage value to "normal", "tumor", or "unknown"."""
    v = str(value).strip().lower()
    if v in NORMAL_LABELS:
        return "normal"
    if v in TUMOR_LABELS:
        return "tumor"
    return "unknown"


def load_camelyon_slides(
    stages_csv: Path = STAGES_CSV,
    processed_csv: Path = PROCESSED_CSV,
    center: Optional[int] = None,
) -> pd.DataFrame:
    """Load the CAMELYON slide table (stage + annotation_used), optionally filtered to one center."""
    stages = pd.read_csv(stages_csv)
    processed = pd.read_csv(processed_csv)

    stages = stages[stages["patient"].astype(str).str.lower().str.endswith(".tif")].copy()
    stages["slide_name"] = stages["patient"].map(lambda v: Path(v).stem)
    stages["patient_id"] = stages["slide_name"].str.extract(r"^(patient_\d+)_node_\d+$")
    stages["tumor_status"] = stages["stage"].map(stage_to_tumor_status)
    stages["inflammation_status"] = stages["stage"].astype(str).str.strip()
    stages["center"] = stages["center"].astype(int)

    ann = processed[["slide_id", "annotation_used"]].rename(columns={"slide_id": "slide_name"})
    merged = stages.merge(ann, on="slide_name", how="left")
    merged["annotation_used"] = merged["annotation_used"].fillna("no")

    if center is not None:
        merged = merged[merged["center"] == center]

    out = merged[
        ["slide_name", "patient", "patient_id", "stage", "inflammation_status",
         "tumor_status", "center", "annotation_used"]
    ].rename(columns={"patient": "slide_file"}).reset_index(drop=True)

    if out["tumor_status"].eq("unknown").any():
        bad = out.loc[out["tumor_status"] == "unknown", "slide_name"].tolist()
        raise ValueError(f"Unrecognized stage values for slides: {bad}")

    return out


def apply_composition(df: pd.DataFrame, mode: str, target_fraction: float, seed: int) -> pd.DataFrame:
    """Restrict tumor slides per the chosen annotation-composition mode. Normal slides untouched."""
    tumor = df[df.tumor_status == "tumor"]
    normal = df[df.tumor_status == "normal"]
    ann_tumor = tumor[tumor.annotation_used == "yes"]
    unann_tumor = tumor[tumor.annotation_used == "no"]

    if mode == "natural":
        kept_tumor = tumor
    elif mode == "annotated_only":
        kept_tumor = ann_tumor
    elif mode == "majority":
        ann_slides = len(ann_tumor)
        if target_fraction <= 0:
            cap = len(unann_tumor)
        else:
            cap = int(np.floor(ann_slides * (1 - target_fraction) / target_fraction))
        cap = max(0, min(cap, len(unann_tumor)))
        sampled_unann = unann_tumor.sample(n=cap, random_state=seed) if cap > 0 else unann_tumor.iloc[:0]
        kept_tumor = pd.concat([ann_tumor, sampled_unann])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return pd.concat([kept_tumor, normal], ignore_index=True)


def balance_1to1(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Keep all tumor slides, subsample normal slides down to an exact 1:1 count, then shuffle."""
    tumor = df[df.tumor_status == "tumor"]
    normal = df[df.tumor_status == "normal"]
    if len(normal) < len(tumor):
        raise ValueError(f"Not enough normal slides to balance: need {len(tumor)}, have {len(normal)}")
    sampled_normal = normal.sample(n=len(tumor), random_state=seed)
    balanced = pd.concat([tumor, sampled_normal], ignore_index=True)
    return balanced.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)


def assert_patient_disjoint(named_patient_sets: Dict[str, Set[str]]) -> None:
    """Assert every pair of named patient-id sets is disjoint; raises with the offending overlap."""
    names = list(named_patient_sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = named_patient_sets[left] & named_patient_sets[right]
            assert not overlap, f"{left}/{right} patient overlap: {overlap}"
    sizes = ", ".join(f"{name}={len(patients)}" for name, patients in named_patient_sets.items())
    print(f"\nDisjointness verified: {sizes} patients -- zero overlap between any pair.")
