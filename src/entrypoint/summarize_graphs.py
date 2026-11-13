"""
Summarize cached graph dataset: node counts, edge counts, annotation coverage, label distribution.
"""
import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.paths.paths import get_project_paths
from src.utils.logging import setup_logging, quiet_external_loggers
from src.graph_creation_new.visualization import graph_structure_summary


def load_graph_safely(graph_path: Path, device: str = "cpu") -> Optional[Any]:
    """Load a cached graph file safely."""
    try:
        return torch.load(graph_path, map_location=device, weights_only=False)
    except Exception as e:
        logging.warning(f"Could not load graph {graph_path}: {e}")
        return None


def extract_graph_stats(data: Any, slide_name: str) -> Optional[Dict[str, Any]]:
    """Extract statistics from a single graph."""
    if data is None:
        return None

    try:
        summary = graph_structure_summary(data)

        stats = {
            "slide_name": slide_name,
            "node_counts": summary.get("node_counts", {}),
            "edge_counts": summary.get("edge_counts", {}),
            "patch_assignment_coverage": summary.get("patch_assignment_coverage"),
            "cell_assignment_coverage": summary.get("cell_assignment_coverage"),
        }

        if hasattr(data, "y") and torch.is_tensor(data.y):
            label_value = data.y.item() if data.y.numel() == 1 else float(data.y.view(-1)[0].item())
            stats["label"] = int(round(label_value))

        if hasattr(data, "node_types"):
            for node_type in ["region", "patch"]:
                if node_type not in data.node_types:
                    continue

                node_store = data[node_type]
                label_tensor = None
                label_source = None

                for candidate in ["y", "annotation_y"]:
                    value = getattr(node_store, candidate, None)
                    if torch.is_tensor(value):
                        label_tensor = value
                        label_source = candidate
                        break

                if label_tensor is None:
                    continue

                labels = label_tensor.detach().cpu().numpy().reshape(-1)
                valid_labels = labels[~np.isnan(labels)] if labels.dtype.kind in {"f"} else labels
                valid_labels = valid_labels[valid_labels != -1]

                present_key = f"{node_type}_labels_present"
                source_key = f"{node_type}_label_source"
                labeled_nodes_key = f"{node_type}_labeled_nodes"
                distribution_key = f"{node_type}_label_distribution"

                stats[present_key] = bool(valid_labels.size > 0)
                stats[source_key] = label_source
                stats[labeled_nodes_key] = int(valid_labels.size)

                if valid_labels.size > 0:
                    uniq, counts = np.unique(valid_labels, return_counts=True)
                    stats[distribution_key] = {int(k): int(v) for k, v in zip(uniq, counts)}

        return stats
    except Exception as e:
        logging.warning(f"Could not extract stats from {slide_name}: {e}")
        return None


def summarize_graph_directory(
    graph_dir: Path,
    output_file: Optional[Path] = None,
    sample_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Summarize all graphs in a directory."""

    graph_files = sorted(graph_dir.glob("*.pt"))
    if not graph_files:
        logging.warning(f"No .pt graph files found in {graph_dir}")
        return {}

    if sample_size is not None:
        graph_files = graph_files[:sample_size]

    logging.info(f"Analyzing {len(graph_files)} graphs from {graph_dir}")

    all_stats = []
    aggregated = {
        "total_graphs": len(graph_files),
        "successful_loads": 0,
        "failed_loads": 0,
        "node_type_totals": defaultdict(int),
        "node_type_mean": defaultdict(float),
        "node_type_std": defaultdict(float),
        "node_type_min": defaultdict(int),
        "node_type_max": defaultdict(int),
        "edge_type_totals": defaultdict(int),
        "edge_type_mean": defaultdict(float),
        "edge_type_std": defaultdict(float),
        "label_distribution": defaultdict(int),
        "label_mode_detected": None,
        "graphs_with_hierarchical_labels": 0,
    }

    for graph_path in tqdm(graph_files, desc="Analyzing graphs"):
        slide_name = graph_path.stem
        graph_data = load_graph_safely(graph_path)
        if graph_data is None:
            aggregated["failed_loads"] += 1
            continue

        aggregated["successful_loads"] += 1
        stats = extract_graph_stats(graph_data, slide_name)
        if stats is None:
            continue

        all_stats.append(stats)

        for node_type, count in stats.get("node_counts", {}).items():
            aggregated["node_type_totals"][node_type] += count

        for edge_type, count in stats.get("edge_counts", {}).items():
            aggregated["edge_type_totals"][edge_type] += count

        if "label" in stats:
            aggregated["label_distribution"][stats["label"]] += 1

        if stats.get("region_labels_present") or stats.get("patch_labels_present"):
            aggregated["graphs_with_hierarchical_labels"] += 1

    for node_type in aggregated["node_type_totals"].keys():
        counts = [s.get("node_counts", {}).get(node_type, 0) for s in all_stats]
        counts = [c for c in counts if c > 0]
        if counts:
            aggregated["node_type_mean"][node_type] = float(np.mean(counts))
            aggregated["node_type_std"][node_type] = float(np.std(counts))
            aggregated["node_type_min"][node_type] = int(np.min(counts))
            aggregated["node_type_max"][node_type] = int(np.max(counts))

    for edge_type in aggregated["edge_type_totals"].keys():
        counts = [s.get("edge_counts", {}).get(edge_type, 0) for s in all_stats]
        counts = [c for c in counts if c > 0]
        if counts:
            aggregated["edge_type_mean"][edge_type] = float(np.mean(counts))
            aggregated["edge_type_std"][edge_type] = float(np.std(counts))

    if aggregated["label_distribution"]:
        unique_labels = sorted(aggregated["label_distribution"].keys())
        if unique_labels == [0, 1]:
            aggregated["label_mode_detected"] = "binary"
        elif len(unique_labels) > 2:
            aggregated["label_mode_detected"] = "multiclass"
        else:
            aggregated["label_mode_detected"] = "single_class"

    aggregated = {
        k: dict(v) if isinstance(v, defaultdict) else v
        for k, v in aggregated.items()
    }

    _print_summary(aggregated, all_stats)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(
                {
                    "summary": aggregated,
                    "per_graph_stats": all_stats,
                },
                f,
                indent=2,
            )
        logging.info(f"Summary saved to {output_file}")

    return aggregated, all_stats


def _print_summary(aggregated: Dict[str, Any], all_stats: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("GRAPH DATASET SUMMARY")
    print("=" * 80)

    print(f"\nTotal graphs: {aggregated['total_graphs']}")
    print(f"Successfully loaded: {aggregated['successful_loads']}")
    print(f"Failed to load: {aggregated['failed_loads']}")

    print(f"\nLabel mode detected: {aggregated.get('label_mode_detected', 'unknown')}")
    if aggregated.get("label_distribution"):
        print("Label distribution:")
        for label, count in sorted(aggregated["label_distribution"].items()):
            pct = 100.0 * count / aggregated["successful_loads"]
            print(f"  Label {label}: {count} graphs ({pct:.1f}%)")

    print(f"\nGraphs with hierarchical labels: {aggregated.get('graphs_with_hierarchical_labels', 0)}")

    print("\nNode Statistics:")
    for node_type in sorted(aggregated.get("node_type_totals", {}).keys()):
        total = aggregated["node_type_totals"][node_type]
        mean = aggregated["node_type_mean"].get(node_type, 0)
        std = aggregated["node_type_std"].get(node_type, 0)
        min_val = aggregated["node_type_min"].get(node_type, 0)
        max_val = aggregated["node_type_max"].get(node_type, 0)
        print(f"  {node_type}: total={total}, mean={mean:.1f}±{std:.1f}, min={min_val}, max={max_val}")

    print("\nEdge Statistics:")
    for edge_type in sorted(aggregated.get("edge_type_totals", {}).keys()):
        total = aggregated["edge_type_totals"][edge_type]
        mean = aggregated["edge_type_mean"].get(edge_type, 0)
        std = aggregated["edge_type_std"].get(edge_type, 0)
        print(f"  {edge_type}: total={total}, mean={mean:.1f}±{std:.1f}")

    print("\n" + "=" * 80 + "\n")


def flatten_graph_stats_for_csv(all_stats: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten nested per-graph statistics into a CSV-ready DataFrame."""
    rows = []

    for stats in all_stats:
        row = {"slide_name": stats.get("slide_name", "unknown")}

        node_counts = stats.get("node_counts", {})
        for node_type in ["patch", "region", "cell", "slide"]:
            row[f"node_count_{node_type}"] = node_counts.get(node_type, 0)

        edge_counts = stats.get("edge_counts", {})
        for edge_type in [
            "patch->near->patch",
            "region->near->region",
            "cell->near->cell",
            "patch->in->region",
            "cell->in->region",
            "region->in->slide",
        ]:
            clean_edge_type = edge_type.replace("->", "_").replace(">", "")
            row[f"edge_count_{clean_edge_type}"] = edge_counts.get(edge_type, 0)

        row["patch_assignment_coverage"] = stats.get("patch_assignment_coverage")
        row["cell_assignment_coverage"] = stats.get("cell_assignment_coverage")

        row["label"] = stats.get("label")
        row["region_labels_present"] = stats.get("region_labels_present", False)
        row["patch_labels_present"] = stats.get("patch_labels_present", False)
        row["region_label_source"] = stats.get("region_label_source")
        row["patch_label_source"] = stats.get("patch_label_source")
        row["region_labeled_nodes"] = stats.get("region_labeled_nodes", 0)
        row["patch_labeled_nodes"] = stats.get("patch_labeled_nodes", 0)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Summarize cached graph dataset")
    parser.add_argument(
        "--graph_dir",
        type=str,
        help="Path to directory containing cached .pt graphs. If not provided, auto-detects from GRAPH_DIR env.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsdb", "camelyon"],
        default="gsdb",
        help="Dataset name to look for in GRAPH_DIR.",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        help="Pipeline name (e.g., 'dynamical', 'tumor_focus'). If provided, only summarizes graphs from that pipeline.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save JSON summary. If not provided, only prints to console.",
    )
    parser.add_argument(
        "--csv_output",
        type=str,
        help="Path to save per-graph statistics as CSV.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        help="Number of graphs to sample. If not provided, analyzes all.",
    )
    return parser.parse_args()


def main() -> None:
    """Summarize a cached graph directory and save the results as JSON and CSV."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    if args.graph_dir:
        graph_dir = Path(args.graph_dir).expanduser().resolve()
    else:
        paths = get_project_paths()
        graph_dir = paths["GRAPH_DIR"] / args.dataset
        if args.pipeline:
            candidates = list(graph_dir.glob(f"*{args.pipeline}*"))
            if candidates:
                graph_dir = candidates[0]

    if not graph_dir.exists():
        logging.error(f"Graph directory not found: {graph_dir}")
        sys.exit(1)

    # Colocated with the graph directory being analyzed (same pattern as build_graph.py) -- this
    # replaces the early basicConfig call above with a proper file+console setup now that
    # graph_dir is known to exist. Anything logged before this point (just the "not found" error
    # path above, if it fired) already went to console via the basicConfig fallback.
    setup_logging(graph_dir, operation="summarize_graphs", force_timestamp=True)
    quiet_external_loggers()

    default_output_stem = f"{graph_dir.name}_graph_summary"
    output_file = Path(args.output) if args.output else (graph_dir / f"{default_output_stem}.json")
    csv_output_file = Path(args.csv_output) if args.csv_output else (graph_dir / f"{default_output_stem}.csv")

    aggregated, all_stats = summarize_graph_directory(
        graph_dir=graph_dir,
        output_file=output_file,
        sample_size=args.sample_size,
    )

    if all_stats:
        csv_output_file.parent.mkdir(parents=True, exist_ok=True)
        df = flatten_graph_stats_for_csv(all_stats)
        df.to_csv(csv_output_file, index=False)
        logging.info(f"Per-graph statistics saved to {csv_output_file} ({len(df)} graphs)")
        print(f"\nCSV saved: {csv_output_file}")
        print(f"Columns: {', '.join(df.columns)}")
        print(f"Sample rows:\n{df.head()}")


if __name__ == "__main__":
    main()
