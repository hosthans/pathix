#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"

DIRS=(
  # --- raw ---
  "raw/camelyon/images"
  "raw/camelyon/annotations"
  "raw/slides"        # gsdb-Slides liegen hier flach (build_graph.py Default, simple_runtime_gsdb.yaml)
  "raw/annotations"   # gsdb-Annotationen liegen hier flach (simple_runtime_gsdb.yaml, create_splits.py)

  # --- processed ---
  "processed/camelyon/tiles"
  "processed/gsdb/tiles"
  "processed/graphs/camelyon"
  "processed/graphs/gsdb"
  "processed/embeddings/prov-gigapath/inflammation/train"

  # --- splits ---
  "splits/camelyon"
  "splits/gsdb"
)

for rel_dir in "${DIRS[@]}"; do
  target="${DATA_DIR}/${rel_dir}"
  mkdir -p "${target}"
  touch "${target}/.gitkeep"
  echo "angelegt: data/${rel_dir}"
done

echo "Fertig. data/-Grundstruktur unter ${DATA_DIR} steht."
