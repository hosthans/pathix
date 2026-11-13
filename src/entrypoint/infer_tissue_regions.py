from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

import torch
import yaml

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.tissue_region_extraction.model import TissueRegionGraphModel
from src.tissue_region_extraction.routines import (
    export_components_as_geojson,
    extract_tissue_region_components,
    infer_region_probabilities,
)
from src.utils.model_utils import load_model
from src.utils.logging import tee_stdout_to_file


def _select_least_busy_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best_idx, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(i)
        if free_bytes > best_free:
            best_idx, best_free = i, free_bytes
    return torch.device(f"cuda:{best_idx}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Infer tumor regions from a cached graph file.")
    parser.add_argument("--graph", type=Path, required=True, help="Path to cached graph .pt")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained checkpoint")
    parser.add_argument("--out", type=Path, required=True, help="Output GeoJSON path")
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Optional training config .yaml used to recover tissue_region_net hyperparameters",
    )
    parser.add_argument("--slide-id", type=str, default=None, help="Optional slide id for GeoJSON metadata")
    parser.add_argument("--threshold", type=float, default=None, help="Region probability threshold override")
    parser.add_argument(
        "--validation-thresholds",
        type=Path,
        default=None,
        help="Optional validation_thresholds.json from evaluation to auto-select threshold",
    )
    parser.add_argument(
        "--threshold-source",
        type=str,
        choices=["region", "slide"],
        default="region",
        help="Preferred threshold source when --validation-thresholds is provided",
    )
    parser.add_argument("--min-component-size", type=int, default=2, help="Minimum regions per component")
    parser.add_argument("--min-mean-probability", type=float, default=0.55, help="Minimum mean probability per component")

    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--dropout-rate", type=float, default=None)
    parser.add_argument("--edge-dropout-p", type=float, default=None)
    parser.add_argument("--region-aux-weight", type=float, default=None)
    parser.add_argument("--patch-hidden-dim", type=int, default=None)
    parser.add_argument("--patch-num-layers", type=int, default=None)
    parser.add_argument("--patch-aux-weight", type=float, default=None)
    parser.add_argument("--patch-aux-pos-weight-cap", type=float, default=None)
    parser.add_argument("--patch-edge-dropout-p", type=float, default=None)
    parser.add_argument("--max-patch-nodes-for-full-forward", type=int, default=None)
    parser.add_argument("--max-patch-edges-for-full-forward", type=int, default=None)
    parser.add_argument(
        "--strict-load",
        dest="strict_load",
        action="store_true",
        help="Enable strict checkpoint loading (recommended)",
    )
    parser.add_argument(
        "--no-strict-load",
        dest="strict_load",
        action="store_false",
        help="Disable strict checkpoint loading",
    )
    parser.set_defaults(strict_load=True)
    parser.add_argument(
        "--device", type=str, default=None,
        help="cpu | cuda | cuda:<n> (e.g. cuda:1 to pick a specific GPU). If omitted, "
             "auto-selects whichever visible GPU currently has the most free memory -- "
             "this machine's GPUs are shared, so the least-busy one shifts over time.",
    )
    parser.add_argument("--save-prob-json", type=Path, default=None, help="Optional path to save raw region probabilities")
    return parser.parse_args()


def _load_graph(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _infer_region_in_dim(data) -> int:
    if not hasattr(data, "node_types") or "region" not in data.node_types:
        raise ValueError("Expected HeteroData with node type 'region'.")
    region_store = data["region"]
    region_x = getattr(region_store, "x", None)
    if not torch.is_tensor(region_x) or region_x.dim() != 2:
        raise ValueError("Region features data['region'].x are required.")
    return int(region_x.size(1))


def _resolve_model_hparams(args: argparse.Namespace) -> Dict[str, float | int]:
    hparams: Dict[str, float | int] = {
        "hidden_dim": 320,
        "num_layers": 4,
        "heads": 4,
        "dropout_rate": 0.20,
        "edge_dropout_p": 0.05,
        "region_aux_weight": 0.24,
        "patch_num_layers": 2,
        "patch_aux_weight": 0.30,
        "patch_aux_pos_weight_cap": 6.0,
        "patch_edge_dropout_p": 0.05,
        "max_patch_nodes_for_full_forward": 6000,
        "max_patch_edges_for_full_forward": 150000,
    }

    if args.config_file is not None:
        if not args.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {args.config_file}")

        cfg_text = args.config_file.read_text(encoding="utf-8")
        raw_cfg: Dict[str, Any] = yaml.safe_load(cfg_text) or {}
        arch_cfg = (((raw_cfg.get("lymph") or {}).get("tissue_region_net")) or {})

        for key in (
            "hidden_dim",
            "num_layers",
            "heads",
            "dropout_rate",
            "edge_dropout_p",
            "region_aux_weight",
            "patch_hidden_dim",
            "patch_num_layers",
            "patch_aux_weight",
            "patch_aux_pos_weight_cap",
            "patch_edge_dropout_p",
            "max_patch_nodes_for_full_forward",
            "max_patch_edges_for_full_forward",
        ):
            if key in arch_cfg and arch_cfg[key] is not None:
                hparams[key] = arch_cfg[key]

    overrides = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "heads": args.heads,
        "dropout_rate": args.dropout_rate,
        "edge_dropout_p": args.edge_dropout_p,
        "region_aux_weight": args.region_aux_weight,
        "patch_hidden_dim": args.patch_hidden_dim,
        "patch_num_layers": args.patch_num_layers,
        "patch_aux_weight": args.patch_aux_weight,
        "patch_aux_pos_weight_cap": args.patch_aux_pos_weight_cap,
        "patch_edge_dropout_p": args.patch_edge_dropout_p,
        "max_patch_nodes_for_full_forward": args.max_patch_nodes_for_full_forward,
        "max_patch_edges_for_full_forward": args.max_patch_edges_for_full_forward,
    }
    for key, value in overrides.items():
        if value is not None:
            hparams[key] = value

    return hparams


def _resolve_region_threshold(args: argparse.Namespace) -> float:
    if args.threshold is not None:
        return float(args.threshold)

    if args.validation_thresholds is not None:
        if not args.validation_thresholds.exists():
            raise FileNotFoundError(f"Validation thresholds file not found: {args.validation_thresholds}")
        payload = json.loads(args.validation_thresholds.read_text(encoding="utf-8"))
        preferred_block = payload.get(str(args.threshold_source), {}) if isinstance(payload, dict) else {}
        if isinstance(preferred_block, dict) and preferred_block.get("threshold") is not None:
            return float(preferred_block["threshold"])

        region_block = payload.get("region", {}) if isinstance(payload, dict) else {}
        if isinstance(region_block, dict) and region_block.get("threshold") is not None:
            return float(region_block["threshold"])

        slide_block = payload.get("slide", {}) if isinstance(payload, dict) else {}
        if isinstance(slide_block, dict) and slide_block.get("threshold") is not None:
            return float(slide_block["threshold"])

    return 0.5


def main() -> None:
    """Run tumor-region inference on a cached graph and export detected components as GeoJSON."""
    args = parse_args()
    # This script uses plain print() throughout rather than `logging` -- tee_stdout_to_file
    # redirects stdout/stderr so those prints also land in a log file (colocated with the
    # GeoJSON output, same pattern as build_graph.py colocating with its graph output) instead of
    # only ever reaching the console.
    tee_stdout_to_file(args.out.parent, operation="infer_tissue_regions")

    data = _load_graph(args.graph)
    in_dim = _infer_region_in_dim(data)
    hparams = _resolve_model_hparams(args)
    region_threshold = _resolve_region_threshold(args)

    model = TissueRegionGraphModel(
        model_name="tissue_region_net",
        num_classes=1,
        dropout_rate=float(hparams["dropout_rate"]),
        in_dim=in_dim,
        hidden_dim=int(hparams["hidden_dim"]),
        num_layers=int(hparams["num_layers"]),
        heads=int(hparams["heads"]),
        edge_dropout_p=float(hparams["edge_dropout_p"]),
        region_aux_weight=float(hparams["region_aux_weight"]),
        patch_hidden_dim=int(hparams["patch_hidden_dim"]) if hparams.get("patch_hidden_dim") is not None else None,
        patch_num_layers=int(hparams["patch_num_layers"]),
        patch_aux_weight=float(hparams["patch_aux_weight"]),
        patch_aux_pos_weight_cap=float(hparams["patch_aux_pos_weight_cap"]),
        patch_edge_dropout_p=float(hparams["patch_edge_dropout_p"]),
        max_patch_nodes_for_full_forward=int(hparams["max_patch_nodes_for_full_forward"]),
        max_patch_edges_for_full_forward=int(hparams["max_patch_edges_for_full_forward"]),
    )

    device = torch.device(args.device) if args.device else _select_least_busy_device()
    model = load_model(
        args.checkpoint,
        model,
        device=device,
        architecture="tissue_region_net",
        strict=bool(args.strict_load),
    )

    region_probs = infer_region_probabilities(data=data, model=model, device=str(device))

    components = extract_tissue_region_components(
        data=data,
        region_probabilities=region_probs,
        threshold=float(region_threshold),
        min_component_size=int(args.min_component_size),
        min_mean_probability=float(args.min_mean_probability),
    )

    out_path = export_components_as_geojson(
        components=components,
        output_path=args.out,
        slide_id=args.slide_id or args.graph.stem,
    )

    if args.save_prob_json is not None:
        args.save_prob_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_prob_json.write_text(
            json.dumps(
                {
                    "slide_id": args.slide_id or args.graph.stem,
                    "threshold": float(region_threshold),
                    "region_probabilities": [float(v) for v in region_probs.view(-1).tolist()],
                    "num_components": int(len(components)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"Graph: {args.graph}")
    print(f"Checkpoint: {args.checkpoint}")
    if args.config_file is not None:
        print(f"Config file: {args.config_file}")
    if args.validation_thresholds is not None:
        print(f"Validation thresholds file: {args.validation_thresholds}")
    print(f"Region threshold: {float(region_threshold):.4f} (source={args.threshold_source})")
    print(
        "Model params: "
        f"hidden_dim={int(hparams['hidden_dim'])}, "
        f"num_layers={int(hparams['num_layers'])}, "
        f"heads={int(hparams['heads'])}, "
        f"dropout_rate={float(hparams['dropout_rate']):.4f}, "
        f"edge_dropout_p={float(hparams['edge_dropout_p']):.4f}, "
        f"region_aux_weight={float(hparams['region_aux_weight']):.4f}, "
        f"patch_num_layers={int(hparams['patch_num_layers'])}, "
        f"patch_aux_weight={float(hparams['patch_aux_weight']):.4f}, "
        f"strict_load={bool(args.strict_load)}"
    )
    print(f"Region count: {int(region_probs.numel())}")
    print(f"Detected components: {int(len(components))}")
    print(f"GeoJSON: {out_path}")


if __name__ == "__main__":
    main()
