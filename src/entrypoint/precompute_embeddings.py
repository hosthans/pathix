import argparse
from pathlib import Path
import logging
from datetime import datetime
import sys
import os


def _autoselect_least_busy_gpu(default: str = '0') -> str:
    """Pick the physical GPU index with the most free memory, via `nvidia-smi` -- called
    before any project module (several of which import torch transitively) so CUDA_VISIBLE_
    DEVICES is set before CUDA can be initialized. See train_model.py for the full rationale
    (this machine's GPUs are shared; occupancy shifts, hardcoding one index gambles on it).
    Falls back to `default` on any failure -- never raises.
    """
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10, check=True,
        )
        candidates = []
        for line in result.stdout.strip().splitlines():
            idx_str, free_str = (part.strip() for part in line.split(','))
            candidates.append((int(idx_str), int(free_str)))
        if not candidates:
            return default
        best_idx, best_free = max(candidates, key=lambda pair: pair[1])
        sys.stderr.write(
            f"[GPU auto-select] Picking GPU {best_idx} ({best_free} MiB free) "
            f"out of {candidates} -- set CUDA_VISIBLE_DEVICES explicitly to override.\n"
        )
        return str(best_idx)
    except Exception as exc:
        sys.stderr.write(f"[GPU auto-select] Falling back to GPU {default}: {exc}\n")
        return default


os.environ.setdefault('CUDA_VISIBLE_DEVICES', _autoselect_least_busy_gpu())

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.paths.pathconfig import PathConfigurator
from src.paths.paths import add_path_args, get_project_paths
from src.dataset.embeddings import ProvGigaPathEmbedder
from src.utils.logging import setup_logging, quiet_external_loggers

import torch
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from typing import List

cache_dir = os.path.expanduser("~/models")
os.makedirs(cache_dir, exist_ok=True)

assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"

import timm
import pandas as pd

def parse_csv(csv: Path, embedder: ProvGigaPathEmbedder, embedding_dir: Path):
    """Precompute and save embeddings for all slides listed in a split CSV."""
    df = pd.read_csv(csv)
    embedder.precompute_embeddings(df, embedding_dir)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train histology classifier")
    parser.add_argument(
        "--task",
        required=True,
        choices=['inflammation'],
        help="Classification task"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )


    parser = add_path_args(parser)

    args = parser.parse_args()

    return args


def main():
    """Precompute and save embeddings for every split CSV of the given task and seed."""
    args = parse_args()


    paths = get_project_paths(base_dir=args.base_dir)


    logging_dir = PathConfigurator.get_logging_path(args, paths)
    logging_dir.mkdir(parents=True, exist_ok=True)

    operation_name = f"precompute_embeddings_{args.task}_seed{args.seed}"
    setup_logging(logging_dir, operation=operation_name)
    quiet_external_loggers()

    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", cache_dir=cache_dir, pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    embedder = ProvGigaPathEmbedder(model=model, transforms=transform, paths=paths)

    logging.info(f"Embedding output directory: {paths['EMBEDDING_DIR']}")

    for file in os.listdir(paths["SPLITS_DIR"] / f"seed_{args.seed}"):
        logging.info(f"Inspecting split file: {file}")
        if file.endswith(".csv"):
            csv_path = paths["SPLITS_DIR"] / f"seed_{args.seed}" / file
            logging.info(f"Precomputing embeddings for {csv_path}")
            parse_csv(csv_path, embedder, paths["EMBEDDING_DIR"])


if __name__ == '__main__':
    main()
