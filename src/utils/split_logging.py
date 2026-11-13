import json
import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def log_split_info(split_name, split_path, split_df, output_dir, target_col=None):
    """Compute and log a split's composition, saving summary plots and a JSON report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_path_str = str(split_path) if isinstance(split_path, (str, Path)) else None

    info = {
        "split_name": split_name,
        "split_path": split_path_str,
        "num_samples": len(split_df),
    }

    if target_col is None:
        for col in ["label", "target", "tumor_status", "stage", "inflammation_status", "tissue_type"]:
            if col in split_df.columns:
                target_col = col
                break

    info["target_column"] = target_col if target_col and target_col in split_df.columns else None

    if target_col and target_col in split_df.columns:
        counts = split_df[target_col].value_counts().sort_index()
        percentages = (counts / len(split_df) * 100).round(2)

        info["target_distribution"] = {
            cls: {
                "count": int(counts[cls]),
                "percentage": float(percentages[cls])
            }
            for cls in counts.index
        }

        plt.figure(figsize=(6, 4))
        counts.plot(kind="bar")
        plt.title(f"{split_name} - {target_col} distribution")
        plt.ylabel("Count")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        for i, v in enumerate(counts):
            plt.text(i, v, str(v), ha='center', va='bottom')

        plot_path = output_dir / f"{split_name}_{target_col}_distribution.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

    else:
        info["target_distribution"] = "unknown"

    if "patient_id" in split_df.columns:
        num_patients = split_df["patient_id"].nunique()
        samples_per_patient = split_df["patient_id"].value_counts()

        info["patients"] = {
            "num_patients": int(num_patients),
            "avg_samples_per_patient": float(samples_per_patient.mean()),
            "max_samples_per_patient": int(samples_per_patient.max()),
        }

        plt.figure(figsize=(6, 4))
        samples_per_patient.hist(bins=20)
        plt.title(f"{split_name} - Samples per patient")
        plt.xlabel("Samples")
        plt.ylabel("Frequency")

        plot_path = output_dir / f"{split_name}_patient_distribution.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

    if "scanner_id" in split_df.columns:
        scanner_counts = split_df["scanner_id"].value_counts().sort_index()
        info["scanner_distribution"] = scanner_counts.to_dict()

        plt.figure()
        scanner_counts.plot(kind="bar")
        plt.title(f"{split_name} - Scanner distribution")
        plt.ylabel("Count")

        plot_path = output_dir / f"{split_name}_scanner_distribution.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

    if "stain" in split_df.columns:
        stain_counts = split_df["stain"].value_counts()
        info["stain_distribution"] = stain_counts.to_dict()

    out_file = output_dir / f"splitinfo_{split_name}.json"
    with open(out_file, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    logging.info(
        f"[SPLIT-INFO] {split_name} | "
        f"Samples: {info['num_samples']} | "
        f"Target column: {info['target_column']} | "
        f"Target: {info['target_distribution']}"
    )

    if "patients" in info:
        logging.info(
            f"[SPLIT-INFO] Patients: {info['patients']['num_patients']} | "
            f"Avg samples/patient: {info['patients']['avg_samples_per_patient']:.2f}"
        )

    logging.info(f"[SPLIT-INFO] Details + Plots gespeichert in: {output_dir}")