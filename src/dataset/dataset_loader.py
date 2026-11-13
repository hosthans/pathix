
from collections.abc import Callable
from pathlib import Path
from torch.utils.data import DataLoader
import sys
from typing import Optional

from pyparsing import Any
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.dataset_utils import mil_collate, mil_collate_no_embeddings
from src.dataset.histology_dataset import EmbeddedImageHistologyDataset, HistologyDataset, MILDataset

def get_loader(mil, embeddings, dataset, batch_size, shuffle=True):
    """Builds a DataLoader for the given dataset with MIL/embedding-appropriate settings."""
    collate_fn = (
        mil_collate
        if embeddings
        else mil_collate_no_embeddings
    )
    if mil:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=False,
            collate_fn=collate_fn,
            persistent_workers=False
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )


def load_dataset(split: str = "train", embeddings: bool = False, mil: bool = False, transform_fn: Optional[Callable[[Any], Any]] = None, paths: dict = None, max_tiles: int = 100, batch_size: int = 32):
    """Constructs and returns a DataLoader for the requested dataset configuration."""
    if embeddings:
        if mil:
            dataset = EmbeddedImageHistologyDataset(split=split, transform=transform_fn, paths=paths)
            return get_loader(True, embeddings, MILDataset(base_dataset=dataset, max_tiles=max_tiles, use_embeddings=True), batch_size=batch_size)
        else:
            return get_loader(False, embeddings, EmbeddedImageHistologyDataset(split=split, transform=transform_fn, paths=paths), batch_size=batch_size)
    if not embeddings:
        if mil:
            dataset = HistologyDataset(split=split, transform=transform_fn, paths=paths)
            return get_loader(True, embeddings, MILDataset(base_dataset=dataset, max_tiles=max_tiles, use_embeddings=False), batch_size=batch_size)
        else:
            return get_loader(False, embeddings, HistologyDataset(split=split, transform=transform_fn, paths=paths), batch_size=batch_size)