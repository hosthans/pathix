import os
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import yaml
import argparse

def get_base_dir() -> Path:
    """Get the base directory from an environment variable or default to the project root."""

    env_base_dir = os.environ.get("GASTRIC_BASE_DIR")
    if env_base_dir:
        base_dir = Path(env_base_dir)
        if base_dir.exists():
            return base_dir
        else:
            logging.warning(f"Environment variable GASTRIC_BASE_DIR points to non-existent path: {base_dir}")
            logging.warning("Falling back to default path")


    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent


    default_dir = project_root
    if default_dir.exists():
        return default_dir
    else:

        default_dir.mkdir(parents=True, exist_ok=True)
        return default_dir

def get_project_paths(base_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Build the dictionary of all project paths relative to a base directory."""
    if base_dir is None:
        base_dir = get_base_dir()

    paths = {
        "BASE_DIR": base_dir,
        "DATA_DIR": base_dir / "data",
        "RAW_DIR": base_dir / "data/raw",
        "PROCESSED_DIR": base_dir / "data/processed",
        "SPLITS_DIR": base_dir / "data/splits",
        "RESULTS_DIR": base_dir / "results",
        "LOGS_DIR": base_dir / "results/logs",
        "FIGURES_DIR": base_dir / "results/figures",
        "TABLES_DIR": base_dir / "results/tables",
        "MODELS_DIR": base_dir / "results/models",
        "METRICS_DIR": base_dir / "results/metrics",
        "CONFIG_DIR": base_dir / "configs",
        "TUNING_DIR": base_dir / "results/tuning",
        "EMBEDDING_DIR": base_dir / "data/processed/embeddings/prov-gigapath/inflammation/train",
        "GRAPH_DIR": base_dir / "data/processed/graphs",
        "META_DIR": base_dir / "results/metas",
    }




    return paths

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a configuration dictionary from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading config from {config_path}: {e}")
        return {}

def add_path_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add path-related command-line arguments to an argument parser."""
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory for all data and outputs"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for outputs"
    )
    return parser


DEFAULT_PATHS = get_project_paths()


def check_path_existence(path: [Path]): # type: ignore
    """
    Check if a given list of paths exist.
    """
    for p in path:
        if not p.exists():
            logging.warning(f"Path does not exist: {p}")
            os.makedirs(p, exist_ok=True)

