"""Bitwise comparison of two directories of cached graph .pt files (HeteroData), file by file.

Unlike graph_content_fingerprint.py (one hash per graph over node/edge tensors only), this loads
both versions of every graph and walks EVERYTHING the graph holds -- every node store, every edge
store, the global store including `meta` -- and reports exactly where they differ:

  - files present in only one directory
  - missing / extra node types, edge types or attributes
  - dtype / shape mismatches
  - tensor value mismatches: number of differing elements and max |a-b|
  - non-tensor mismatches (meta dicts, lists, strings, numbers)

Tensors are compared bit for bit (floats are reinterpreted as same-width ints, so NaN==NaN with the
same payload and 0.0 != -0.0). Only the run-identifying strings in `meta` are ignored by default
(graph_config_path/_name/_stem, graph_output_dir/_name), since two runs of the same config
necessarily live under different names; pass --ignore-key to change that, --ignore-key ''
to ignore nothing. Ignored keys that differ are still listed as INFO.

Exit code: 0 if all graphs are identical (modulo ignored keys), 1 otherwise.

Examples:
  python -u scripts/analysis/compare_graph_dirs.py \\
      data/processed/graphs/gsdb/context_aware_gsdb_inflammation_noaug \\
      data/processed/graphs/gsdb/context_aware_gsdb_inflammation_noaug_o --workers 4

  # only a few slides, write a JSON report
  python -u scripts/analysis/compare_graph_dirs.py DIR_A DIR_B --pattern '10*_*.pt' --report diff.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

# Graph files pickle references to src.* classes, so the project root must be importable (also in workers).
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

DEFAULT_IGNORE_KEYS = (
    "graph_config_path",
    "graph_output_dir",
    "graph_config_name",
    "graph_config_stem",
    "graph_output_dir_name",
)
_FLOAT_AS_INT = {}  # filled lazily (torch import happens in the workers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bitwise comparison of two directories of cached graph .pt files")
    parser.add_argument("dir_a", type=str)
    parser.add_argument("dir_b", type=str)
    parser.add_argument("--pattern", type=str, default="*.pt", help="Glob for the graph files (default: *.pt)")
    parser.add_argument(
        "--ignore-key",
        type=str,
        action="append",
        default=None,
        help=f"Dict key to ignore anywhere in the graph (repeatable). Default: {', '.join(DEFAULT_IGNORE_KEYS)}. "
        "Pass --ignore-key '' to ignore nothing.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel processes (each holds two graphs in RAM)")
    parser.add_argument("--max-diffs-per-file", type=int, default=20, help="Mismatches printed per file")
    parser.add_argument("--report", type=str, default=None, help="Optional JSON report path")
    return parser.parse_args()


def _bit_view(t):
    """Reinterpret floating tensors as same-width ints so equality is bitwise."""
    import torch

    if not _FLOAT_AS_INT:
        _FLOAT_AS_INT.update({
            torch.float64: torch.int64,
            torch.float32: torch.int32,
            torch.float16: torch.int16,
            torch.bfloat16: torch.int16,
        })
    t = t.detach().cpu().contiguous()
    if t.dtype in _FLOAT_AS_INT:
        return t.view(_FLOAT_AS_INT[t.dtype])
    if t.is_complex():
        return torch.view_as_real(t).contiguous().view(_FLOAT_AS_INT[torch.view_as_real(t).dtype])
    return t


def _compare_tensor(path: str, a, b, diffs: List[dict]) -> None:
    import torch

    if a.dtype != b.dtype:
        diffs.append({"path": path, "kind": "dtype", "a": str(a.dtype), "b": str(b.dtype)})
        return
    if tuple(a.shape) != tuple(b.shape):
        diffs.append({"path": path, "kind": "shape", "a": list(a.shape), "b": list(b.shape)})
        return
    va, vb = _bit_view(a), _bit_view(b)
    if torch.equal(va, vb):
        return
    neq = va != vb
    while neq.dim() > a.dim():  # complex -> extra trailing dim
        neq = neq.any(dim=-1)
    entry = {
        "path": path,
        "kind": "values",
        "dtype": str(a.dtype),
        "shape": list(a.shape),
        "n_diff": int(neq.sum().item()),
        "n_total": int(neq.numel()),
    }
    if a.is_floating_point():
        d = (a.detach().cpu().double() - b.detach().cpu().double()).abs()
        d = d[~torch.isnan(d)]
        entry["max_abs_diff"] = float(d.max().item()) if d.numel() else float("nan")
    diffs.append(entry)


def _compare(path: str, a: Any, b: Any, ignore: frozenset, diffs: List[dict], infos: List[dict]) -> None:
    import numpy as np
    import torch

    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            diffs.append({"path": path, "kind": "type", "a": type(a).__name__, "b": type(b).__name__})
            return
        _compare_tensor(path, a, b, diffs)
        return

    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            diffs.append({"path": path, "kind": "type", "a": type(a).__name__, "b": type(b).__name__})
            return
        if a.dtype != b.dtype or a.shape != b.shape:
            diffs.append({"path": path, "kind": "ndarray_meta", "a": f"{a.dtype}{a.shape}", "b": f"{b.dtype}{b.shape}"})
        elif a.dtype == object:
            for i, (x, y) in enumerate(zip(a.ravel().tolist(), b.ravel().tolist())):
                _compare(f"{path}[{i}]", x, y, ignore, diffs, infos)
        elif np.ascontiguousarray(a).tobytes() != np.ascontiguousarray(b).tobytes():
            diffs.append({"path": path, "kind": "values", "dtype": str(a.dtype), "shape": list(a.shape),
                          "n_diff": int((a != b).sum()), "n_total": int(a.size)})
        return

    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            sub = f"{path}/{k}"
            if str(k) in ignore:
                if k in a and k in b and repr(a[k]) != repr(b[k]):
                    infos.append({"path": sub, "a": repr(a[k])[:200], "b": repr(b[k])[:200]})
                continue
            if k not in a:
                diffs.append({"path": sub, "kind": "only_in_b"})
            elif k not in b:
                diffs.append({"path": sub, "kind": "only_in_a"})
            else:
                _compare(sub, a[k], b[k], ignore, diffs, infos)
        return

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            diffs.append({"path": path, "kind": "sequence", "a": f"{type(a).__name__}[{len(a)}]",
                          "b": f"{type(b).__name__}[{len(b)}]"})
            if len(a) != len(b):
                return
        for i, (x, y) in enumerate(zip(a, b)):
            _compare(f"{path}[{i}]", x, y, ignore, diffs, infos)
        return

    if type(a) is not type(b):
        diffs.append({"path": path, "kind": "type", "a": f"{type(a).__name__}:{repr(a)[:80]}",
                      "b": f"{type(b).__name__}:{repr(b)[:80]}"})
        return
    if isinstance(a, float):
        same = (math.isnan(a) and math.isnan(b)) or (a == b and math.copysign(1, a) == math.copysign(1, b))
    else:
        try:
            same = bool(a == b)
        except Exception:  # noqa: BLE001 - objects without a usable __eq__: fall back to repr
            same = repr(a) == repr(b)
    if not same:
        diffs.append({"path": path, "kind": "value", "a": repr(a)[:200], "b": repr(b)[:200]})


def _as_tree(graph: Any) -> Dict[str, Any]:
    """HeteroData/Data -> nested dict {global attrs, node type -> {attr: ...}, edge type -> {attr: ...}}."""
    if hasattr(graph, "node_types") and hasattr(graph, "edge_types"):
        tree: Dict[str, Any] = {f"global:{k}": v for k, v in graph._global_store.to_dict().items()}
        for nt in graph.node_types:
            tree[f"node:{nt}"] = graph[nt].to_dict()
        for et in graph.edge_types:
            tree[f"edge:{'__'.join(et)}"] = graph[et].to_dict()
        return tree
    if hasattr(graph, "to_dict"):
        return dict(graph.to_dict())
    return {"object": graph}


def compare_file(name: str, dir_a: str, dir_b: str, ignore: frozenset) -> dict:
    import torch

    result = {"file": name, "diffs": [], "infos": [], "error": None}
    try:
        a = torch.load(Path(dir_a) / name, map_location="cpu", weights_only=False)
        b = torch.load(Path(dir_b) / name, map_location="cpu", weights_only=False)
        if type(a) is not type(b):
            result["diffs"].append({"path": "", "kind": "type", "a": type(a).__name__, "b": type(b).__name__})
        else:
            _compare("", _as_tree(a), _as_tree(b), ignore, result["diffs"], result["infos"])
    except Exception as exc:  # noqa: BLE001 - report per file, keep scanning
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _fmt(d: dict) -> str:
    if d["kind"] == "values":
        extra = f" max|a-b|={d['max_abs_diff']:.3e}" if "max_abs_diff" in d else ""
        return f"{d['path']}: VALUES differ in {d['n_diff']}/{d['n_total']} elements ({d['dtype']}{d['shape']}){extra}"
    if d["kind"] in ("only_in_a", "only_in_b"):
        return f"{d['path']}: {d['kind'].upper()}"
    return f"{d['path']}: {d['kind'].upper()} a={d.get('a')} b={d.get('b')}"


def main() -> None:
    args = parse_args()
    dir_a, dir_b = Path(args.dir_a), Path(args.dir_b)
    for d in (dir_a, dir_b):
        if not d.is_dir():
            sys.exit(f"Not a directory: {d}")
    ignore = frozenset(k for k in (args.ignore_key if args.ignore_key is not None else DEFAULT_IGNORE_KEYS) if k)

    names_a = {p.name for p in dir_a.glob(args.pattern)}
    names_b = {p.name for p in dir_b.glob(args.pattern)}
    only_a, only_b = sorted(names_a - names_b), sorted(names_b - names_a)
    common = sorted(names_a & names_b)

    print(f"A: {dir_a.resolve()} ({len(names_a)} files)")
    print(f"B: {dir_b.resolve()} ({len(names_b)} files)")
    print(f"Ignored keys: {sorted(ignore) or 'none'}")
    for label, names in (("Only in A", only_a), ("Only in B", only_b)):
        print(f"{label}: {len(names)}")
        for n in names:
            print(f"  - {n}")
    print(f"Comparing {len(common)} common files with {args.workers} workers ...", flush=True)

    results: List[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(compare_file, n, str(dir_a), str(dir_b), ignore) for n in common]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            if r["error"]:
                print(f"[{i}/{len(common)}] ERROR {r['file']}: {r['error']}", flush=True)
            elif r["diffs"]:
                print(f"[{i}/{len(common)}] DIFF  {r['file']} ({len(r['diffs'])} mismatches)", flush=True)
                for d in r["diffs"][: args.max_diffs_per_file]:
                    print(f"      {_fmt(d)}")
                if len(r["diffs"]) > args.max_diffs_per_file:
                    print(f"      ... {len(r['diffs']) - args.max_diffs_per_file} more")
            elif i % 25 == 0 or i == len(common):
                print(f"[{i}/{len(common)}] ok so far", flush=True)

    results.sort(key=lambda r: r["file"])
    n_err = sum(1 for r in results if r["error"])
    n_diff = sum(1 for r in results if r["diffs"])
    n_same = len(results) - n_err - n_diff
    info_paths = sorted({i["path"] for r in results for i in r["infos"]})

    print("\n===== SUMMARY =====")
    print(f"identical: {n_same}/{len(common)}  differing: {n_diff}  errors: {n_err}  "
          f"only in A: {len(only_a)}  only in B: {len(only_b)}")
    if info_paths:
        print(f"INFO: ignored keys that differ (not counted): {', '.join(info_paths)}")
    all_same = n_diff == 0 and n_err == 0 and not only_a and not only_b
    print("RESULT: ALL GRAPHS BITWISE IDENTICAL" if all_same else "RESULT: GRAPHS DIFFER")

    if args.report:
        Path(args.report).write_text(json.dumps({
            "dir_a": str(dir_a.resolve()), "dir_b": str(dir_b.resolve()), "ignored_keys": sorted(ignore),
            "only_in_a": only_a, "only_in_b": only_b, "files": results,
        }, indent=2, default=str))
        print(f"Report written to {args.report}")
    sys.exit(0 if all_same else 1)


if __name__ == "__main__":
    main()
