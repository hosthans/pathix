
from src.dataset.histology_dataset import (
    HistologyDataset,
    EmbeddedHistologyDataset,
    EmbeddedImageHistologyDataset,
    MILDataset
)

from src.dataset.graph_dataset import WSIGraphDataset
from src.dataset.camelyon_dataset import CamelyonHistologyDataset

__all__ = [
    'HistologyDataset',
    'EmbeddedHistologyDataset',
    'EmbeddedImageHistologyDataset',
    'MILDataset',
    'CamelyonHistologyDataset',
    'WSIGraphDataset',
]
