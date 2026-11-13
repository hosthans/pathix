"""Content-level fingerprint for cached graph .pt files, for comparing graph *content*
between two machines/checkouts instead of just file size/mtime (see inspect_raw_structure.py,
which cannot tell "same size" apart from "same content" -- a real concern once two people are
comparing graphs built independently on different machines, e.g. with different library versions).

For each HeteroData .pt file, hashes the raw bytes of every node-type/edge-type tensor
(features, labels, edge_index, positions, ...) EXCLUDING the machine-specific path strings in
`meta` (graph_config_path, graph_output_dir -- see inspect_raw_structure.py's manifest diff,
those differ per-checkout and are irrelevant to whether the graph content matches). Also records
per-node-type/edge-type counts, since a count mismatch alone is enough to prove non-identical
graphs without needing to look at the hash.

Examples:
  # Build a fingerprint manifest of your own graph cache
  python -u scripts/analysis/graph_content_fingerprint.py \\
      --root data/processed/graphs/gsdb/context_aware_gsdb_inflammation \\
      --manifest fingerprint_hweber_graphs_inflammation.json

  # Diff two previously written fingerprint manifests
  python -u scripts/analysis/graph_content_fingerprint.py \\
      --diff fingerprint_hweber_graphs_inflammation.json fingerprint_maarten_graphs_inflammation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fingerprint / diff the actual content of cached graph .pt files")
    parser.add_argument("--root", type=str, default=None, help="Directory of .pt graph files to fingerprint")
    parser.add_argument("--manifest", type=str, default=None, help="Where to write the fingerprint manifest (JSON)")
    parser.add_argument(
        "--diff",
        type=str,
        nargs=2,
        metavar=("MANIFEST_A", "MANIFEST_B"),
        default=None,
        help="Skip scanning; diff two previously written fingerprint manifests instead.",
    )
    return parser.parse_args()


def _tensor_hash(h: "hashlib._Hash", tensor) -> None:
    import torch
    # .contiguous() so two tensors with identical values but a different (e.g. non-contiguous
    # view) memory layout still hash the same -- comparing values, not storage layout.
    h.update(tensor.detach().cpu().contiguous().numpy().tobytes())


def fingerprint_one(path: Path) -> dict:
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    node_counts: Dict[str, int] = {}
    edge_counts: Dict[str, int] = {}
    h = hashlib.sha256()

    for nt in sorted(obj.node_types):
        store = obj[nt]
        node_counts[nt] = int(store.num_nodes) if store.num_nodes is not None else 0
        for attr in sorted(store.keys()):
            tensor = store[attr]
            if not torch.is_tensor(tensor):
                continue
            h.update(f"node|{nt}|{attr}|{tuple(tensor.shape)}|{tensor.dtype}\n".encode("utf-8"))
            _tensor_hash(h, tensor)

    for et in sorted(obj.edge_types, key=str):
        store = obj[et]
        edge_index = store.get("edge_index", None)
        edge_counts[str(et)] = int(edge_index.shape[1]) if edge_index is not None else 0
        for attr in sorted(store.keys()):
            tensor = store[attr]
            if not torch.is_tensor(tensor):
                continue
            h.update(f"edge|{et}|{attr}|{tuple(tensor.shape)}|{tensor.dtype}\n".encode("utf-8"))
            _tensor_hash(h, tensor)

    return {
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "content_sha256": h.hexdigest(),
    }


def build_manifest(root: Path) -> Dict[str, dict]:
    manifest: Dict[str, dict] = {}
    paths = sorted(root.glob("*.pt"))
    total = len(paths)
    for i, path in enumerate(paths, 1):
        try:
            manifest[path.name] = fingerprint_one(path)
        except Exception as exc:  # noqa: BLE001 - report and continue, one bad file shouldn't abort the scan
            manifest[path.name] = {"error": str(exc)}
        if i % 20 == 0 or i == total:
            print(f"  fingerprinted {i}/{total} graphs...", file=sys.stderr)
    return manifest


def diff_payloads(a: dict, b: dict) -> None:
    files_a: Dict[str, dict] = a["files"]
    files_b: Dict[str, dict] = b["files"]

    keys_a, keys_b = set(files_a), set(files_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    common = sorted(keys_a & keys_b)

    differing: List[Tuple[str, dict, dict]] = []
    for k in common:
        ea, eb = files_a[k], files_b[k]
        if ea.get("content_sha256") != eb.get("content_sha256"):
            differing.append((k, ea, eb))

    print(f"\nManifest A: {a.get('root')} ({len(files_a)} graphs, generated {a.get('generated_at')})")
    print(f"Manifest B: {b.get('root')} ({len(files_b)} graphs, generated {b.get('generated_at')})")

    print(f"\nOnly in A: {len(only_a)}")
    for k in only_a[:50]:
        print(f"  - {k}")
    print(f"\nOnly in B: {len(only_b)}")
    for k in only_b[:50]:
        print(f"  - {k}")

    print(f"\nPresent in both but content differs: {len(differing)}")
    for k, ea, eb in differing[:50]:
        if ea.get("node_counts") != eb.get("node_counts") or ea.get("edge_counts") != eb.get("edge_counts"):
            print(f"  - {k}: STRUCTURAL mismatch -- A nodes={ea.get('node_counts')} edges={ea.get('edge_counts')}"
                  f"  B nodes={eb.get('node_counts')} edges={eb.get('edge_counts')}")
        else:
            print(f"  - {k}: same node/edge counts, but tensor VALUES differ "
                  f"(A sha256={ea.get('content_sha256', '-')[:12]}  B sha256={eb.get('content_sha256', '-')[:12]})")
    if len(differing) > 50:
        print(f"  ... and {len(differing) - 50} more")

    if not only_a and not only_b and not differing:
        print("\nAll common graphs are content-identical (same node/edge counts AND same tensor values).")


def main() -> None:
    args = parse_args()

    if args.diff:
        a = json.loads(Path(args.diff[0]).read_text())
        b = json.loads(Path(args.diff[1]).read_text())
        diff_payloads(a, b)
        return

    if not args.root or not args.manifest:
        print("Provide --root and --manifest to build a fingerprint, or --diff to compare two.", file=sys.stderr)
        sys.exit(1)

    root = Path(args.root)
    if not root.exists():
        print(f"Root does not exist: {root.resolve()}", file=sys.stderr)
        sys.exit(1)

    print(f"Fingerprinting graphs in {root.resolve()} ...", file=sys.stderr)
    manifest = build_manifest(root)
    payload = {
        "root": str(root.resolve()),
        "generated_at": datetime.now().isoformat(),
        "files": manifest,
    }
    out_path = Path(args.manifest)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote content fingerprint for {len(manifest)} graphs to {out_path}")


if __name__ == "__main__":
    main()
