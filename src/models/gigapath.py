"""
GigaPath Model Architecture

This module implements the GigaPathClassifier, which leverages pre-trained
Prov-GigaPath features for histology image classification.
"""

import torch
import torch.nn as nn
import logging

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.core.registry_utils import register_model

@register_model(('cnn', 'gigapath', 'image'))
class GigaPathClassifier(nn.Module):
    """Classifier using Prov-GigaPath features for histology classification.

    This model uses a pre-trained GigaPath tile encoder to extract features
    from histology images, followed by a trainable classification head.
    The encoder weights are frozen during training to leverage the
    pre-trained representations.

    Attributes:
        model_name (str): Identifier for the model architecture
        tile_encoder (nn.Module): Pre-trained GigaPath feature extractor
        classifier (nn.Sequential): Trainable classification head
    """

    def __init__(self, model_name: str, num_classes: int = 1, dropout_rate: float = 0.2, in_dim: int = 512):
        """Initialize the GigaPath classifier with a frozen encoder and trainable head."""
        super().__init__()
        self.model_name = "gigapath"

        try:
            import timm

            self.tile_encoder = timm.create_model(
                "hf_hub:prov-gigapath/prov-gigapath",
                pretrained=True,
                num_classes=0
            )


            for param in self.tile_encoder.parameters():
                param.requires_grad = False


            self.classifier = nn.Sequential(
                nn.Linear(1536, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
        except ImportError:
            logging.error("timm library not found. Please install with: pip install timm")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        with torch.no_grad():
            features = self.tile_encoder(x)
        return self.classifier(features)

    def train(self, mode: bool = True) -> 'GigaPathClassifier':
        """Set training mode for the classifier only, keeping the encoder in eval mode."""
        super().train(mode)

        self.tile_encoder.eval()
        return self

    def configure_optimizers(self, lr: float, weight_decay: float):
        """Configure the optimizer, loss criterion, and gradient scaler for the classifier."""
        self.optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.scaler = torch.amp.GradScaler()
        return self.optimizer