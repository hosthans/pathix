"""Read-only directory-structure inspector for spotting raw-data drift between machines.

Use case: two people run the exact same graph-build code on their own `data/raw`
tree and get deterministic-but-different results (e.g. a consistent ~5% AUC gap).
That points at the *input data* differing, not the code. This script produces a
quick per-folder summary (file counts, extensions, sizes, mtimes) so such
differences are visible at a glance, and can also write a manifest that can be
diffed directly against a colleague's manifest.

Examples:
  # Quick overview of data/raw, one level of subfolders deep
  python -u scripts/analysis/inspect_raw_structure.py

  # Go deeper / point at a different root
  python -u scripts/analysis/inspect_raw_structure.py --root data/raw --depth 3

  # Write a manifest (per-file: relpath, size, mtime, optional hash) to compare
  # against a colleague's manifest of the same logical directory
  python -u scripts/analysis/inspect_raw_structure.py --root data/raw --manifest raw_manifest_me.json --hash

  # Diff two previously written manifests (e.g. yours and your colleague's)
  python -u scripts/analysis/inspect_raw_structure.py --diff raw_manifest_me.json raw_manifest_colleague.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect / diff a raw data directory structure")
    parser.add_argument("--root", type=str, default="data/raw", help="Root directory to scan")
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="How many levels below --root to report as separate rows (default: 2)",
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Also compute a content hash per file (slow, but catches silently-changed files "
        "with identical name/size). Only used together with --manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="If set, write a full per-file manifest (JSON) to this path, for diffing against "
        "a colleague's manifest of the same logical folder.",
    )
    parser.add_argument(
        "--diff",
        type=str,
        nargs=2,
        metavar=("MANIFEST_A", "MANIFEST_B"),
        default=None,
        help="Skip scanning; instead diff two previously written manifests and print the result.",
    )
    parser.add_argument(
        "--compare-with",
        type=str,
        default=None,
        metavar="MANIFEST_A",
        help="One-shot workflow for the second machine: build a manifest of --root here "
        "(implies --hash), then immediately diff it against a manifest file produced "
        "on another machine (copy it over first, e.g. via scp). Combine with --manifest "
        "to also save the locally built manifest.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Summary mode: fast, no hashing, meant to be run directly on --root
# --------------------------------------------------------------------------- #

@dataclass
class FolderStats:
    num_files: int = 0
    total_bytes: int = 0
    extensions: Counter = field(default_factory=Counter)
    mtime_min: Optional[float] = None
    mtime_max: Optional[float] = None

    def add(self, path: Path, size: int, mtime: float) -> None:
        self.num_files += 1
        self.total_bytes += size
        self.extensions[path.suffix.lower() or "<no ext>"] += 1
        self.mtime_min = mtime if self.mtime_min is None else min(self.mtime_min, mtime)
        self.mtime_max = mtime if self.mtime_max is None else max(self.mtime_max, mtime)


def human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"


def relative_depth(path: Path, root: Path) -> int:
    return len(path.relative_to(root).parts)


def summarize(root: Path, depth: int) -> Dict[Path, FolderStats]:
    stats: Dict[Path, FolderStats] = defaultdict(FolderStats)
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        parent = path.parent
        d = min(relative_depth(parent, root) + 1, depth) if parent != root else 1
        # bucket this file's parent chain up to the reporting depth
        parts = parent.relative_to(root).parts[: max(depth - 1, 0)]
        bucket = root.joinpath(*parts) if parts else root
        stats[bucket].add(path, st.st_size, st.st_mtime)
    return stats


def print_summary(root: Path, stats: Dict[Path, FolderStats]) -> None:
    print(f"\nRaw structure summary for: {root.resolve()}\n")
    if not stats:
        print("  (no files found)")
        return

    rows = sorted(stats.items(), key=lambda kv: str(kv[0]))
    name_w = max(len(str(p.relative_to(root)) or ".") for p, _ in rows)
    name_w = max(name_w, 6)

    header = f"{'folder':<{name_w}}  {'files':>8}  {'size':>10}  top extensions"
    print(header)
    print("-" * len(header))
    for path, s in rows:
        rel = str(path.relative_to(root)) if path != root else "."
        top_ext = ", ".join(f"{ext}:{cnt}" for ext, cnt in s.extensions.most_common(4))
        print(f"{rel:<{name_w}}  {s.num_files:>8}  {human_bytes(s.total_bytes):>10}  {top_ext}")

    total_files = sum(s.num_files for s in stats.values())
    total_bytes = sum(s.total_bytes for s in stats.values())
    all_ext = Counter()
    mtime_min = min((s.mtime_min for s in stats.values() if s.mtime_min is not None), default=None)
    mtime_max = max((s.mtime_max for s in stats.values() if s.mtime_max is not None), default=None)
    for s in stats.values():
        all_ext.update(s.extensions)

    print("-" * len(header))
    print(f"{'TOTAL':<{name_w}}  {total_files:>8}  {human_bytes(total_bytes):>10}")
    print(f"\nAll extensions: {dict(all_ext.most_common())}")
    if mtime_min is not None and mtime_max is not None:
        print(
            f"mtime range: {datetime.fromtimestamp(mtime_min)}  ->  {datetime.fromtimestamp(mtime_max)}"
        )


# --------------------------------------------------------------------------- #
# Manifest mode: per-file records, meant for diffing two machines
# --------------------------------------------------------------------------- #

def file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(root: Path, with_hash: bool) -> Dict[str, dict]:
    manifest: Dict[str, dict] = {}
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    total = len(paths)
    for i, path in enumerate(paths, 1):
        try:
            st = path.stat()
        except OSError:
            continue
        rel = str(path.relative_to(root))
        entry = {"size": st.st_size, "mtime": st.st_mtime}
        if with_hash:
            entry["sha256"] = file_hash(path)
        manifest[rel] = entry
        if with_hash and (i % 200 == 0 or i == total):
            print(f"  hashed {i}/{total} files...", file=sys.stderr)
    return manifest


def root_hash_of(manifest: Dict[str, dict]) -> Optional[str]:
    """Single combined hash over the whole tree: order-independent per file, but
    sensitive to any file being added/removed/renamed/resized/content-changed.
    Only meaningful if all entries carry a sha256 (i.e. built with --hash)."""
    if not manifest or not all("sha256" in e for e in manifest.values()):
        return None
    h = hashlib.sha256()
    for rel in sorted(manifest):
        e = manifest[rel]
        h.update(f"{rel}:{e['size']}:{e['sha256']}\n".encode("utf-8"))
    return h.hexdigest()


def write_manifest(root: Path, manifest: Dict[str, dict], out_path: Path) -> None:
    rhash = root_hash_of(manifest)
    payload = {
        "root": str(root.resolve()),
        "generated_at": datetime.now().isoformat(),
        "root_hash": rhash,
        "files": manifest,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote manifest with {len(manifest)} files to {out_path}")
    if rhash:
        print(f"ROOT HASH (compare this single value with your colleague): {rhash}")


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
        if ea.get("size") != eb.get("size"):
            differing.append((k, ea, eb))
        elif "sha256" in ea and "sha256" in eb and ea["sha256"] != eb["sha256"]:
            differing.append((k, ea, eb))

    print(f"\nManifest A: {a.get('root')} ({len(files_a)} files, generated {a.get('generated_at')})")
    print(f"  root_hash A: {a.get('root_hash')}")
    print(f"Manifest B: {b.get('root')} ({len(files_b)} files, generated {b.get('generated_at')})")
    print(f"  root_hash B: {b.get('root_hash')}")
    if a.get("root_hash") and a.get("root_hash") == b.get("root_hash"):
        print("\nRoot hashes match -> folders are byte-identical. Nothing else to check.")
    print(f"\nOnly in A: {len(only_a)}")
    for k in only_a[:50]:
        print(f"  - {k}")
    if len(only_a) > 50:
        print(f"  ... and {len(only_a) - 50} more")

    print(f"\nOnly in B: {len(only_b)}")
    for k in only_b[:50]:
        print(f"  - {k}")
    if len(only_b) > 50:
        print(f"  ... and {len(only_b) - 50} more")

    print(f"\nPresent in both but differ (size or hash): {len(differing)}")
    for k, ea, eb in differing[:50]:
        print(f"  - {k}: A(size={ea.get('size')}, sha256={ea.get('sha256', '-')[:10]})"
              f"  B(size={eb.get('size')}, sha256={eb.get('sha256', '-')[:10]})")
    if len(differing) > 50:
        print(f"  ... and {len(differing) - 50} more")

    if not only_a and not only_b and not differing:
        print("\nManifests are identical.")


def diff_manifests(path_a: Path, path_b: Path) -> None:
    a = json.loads(path_a.read_text())
    b = json.loads(path_b.read_text())
    diff_payloads(a, b)


def build_manifest_payload(root: Path, with_hash: bool) -> dict:
    manifest = build_manifest(root, with_hash=with_hash)
    return {
        "root": str(root.resolve()),
        "generated_at": datetime.now().isoformat(),
        "root_hash": root_hash_of(manifest),
        "files": manifest,
    }


def main() -> None:
    args = parse_args()

    if args.diff:
        diff_manifests(Path(args.diff[0]), Path(args.diff[1]))
        return

    root = Path(args.root)
    if not root.exists():
        print(f"Root does not exist: {root.resolve()}", file=sys.stderr)
        sys.exit(1)

    if args.compare_with:
        other_path = Path(args.compare_with)
        if not other_path.exists():
            print(f"--compare-with file does not exist: {other_path.resolve()}", file=sys.stderr)
            sys.exit(1)
        other_payload = json.loads(other_path.read_text())

        print(f"Building manifest for {root.resolve()} (with content hashes)...", file=sys.stderr)
        local_payload = build_manifest_payload(root, with_hash=True)
        if local_payload["root_hash"]:
            print(f"ROOT HASH (this machine): {local_payload['root_hash']}")

        if args.manifest:
            out_path = Path(args.manifest)
            out_path.write_text(json.dumps(local_payload, indent=2, sort_keys=True))
            print(f"Wrote local manifest with {len(local_payload['files'])} files to {out_path}")

        diff_payloads(other_payload, local_payload)
        return

    if args.manifest:
        manifest = build_manifest(root, with_hash=args.hash)
        write_manifest(root, manifest, Path(args.manifest))

    stats = summarize(root, depth=args.depth)
    print_summary(root, stats)


if __name__ == "__main__":
    main()
