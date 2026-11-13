"""
Base Histology Classification Models

This module implements the base HistologyClassifier class which supports
multiple model architectures for histology image classification.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Tuple, List, Optional

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import os


from src.core.registry_utils import register_model

@register_model(('cnn', 'resnet18', 'image'))
class HistologyClassifier(nn.Module):
    """Advanced histology image classifier supporting multiple architectures.

    This class provides a unified interface for different model architectures
    (ResNet, DenseNet, ConvNeXt, Swin Transformer) to be used for
    histological image classification tasks.

    Attributes:
        model_name (str): Identifier for the model architecture
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout probability for regularization
        backbone (nn.Module): The underlying model backbone
        device (torch.device): Device where the model is located
    """

    def __init__(self, model_name: str, num_classes: int = 1, dropout_rate: float = 0.2, in_dim: int = 512):
        """Initialize the histology classifier with the specified backbone architecture."""
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate


        if model_name == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(in_features, self.num_classes)
            )

        elif model_name == 'densenet121':
            from torchvision.models import densenet121, DenseNet121_Weights
            self.backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)

            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(in_features, self.num_classes)
            )

        elif model_name == 'densenet169':
            from torchvision.models import densenet169, DenseNet169_Weights
            self.backbone = densenet169(weights=DenseNet169_Weights.DEFAULT)

            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(in_features, self.num_classes)
            )

        elif model_name == 'convnext_large':
            from torchvision.models import convnext_large, ConvNeXt_Large_Weights
            self.backbone = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)

            self.backbone.classifier = nn.Sequential(
                nn.Flatten(1),
                nn.LayerNorm(1536),
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(1536, self.num_classes)
            )

        elif model_name == 'swin_v2_b':
            from torchvision.models import swin_v2_b, Swin_V2_B_Weights
            self.backbone = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)


            in_features = self.backbone.head.in_features



            self.backbone.head = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(in_features, self.num_classes)
            )

        else:
            raise ValueError(f"Unsupported model: {model_name}. Use 'resnet18', 'densenet121', 'densenet169', 'swin_v2_b', or 'convnext_large'. For GigaPath, use GigaPathClassifier.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.backbone(x)

    def configure_optimizers(self, lr: float, weight_decay: float):
        """Configure the optimizer, loss criterion, and gradient scaler for training."""
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.scaler = torch.amp.GradScaler()
        return self.optimizer

    def train(self, mode: bool = True) -> 'HistologyClassifier':
        """Set training mode on the model and its backbone."""
        super().train(mode)
        self.backbone.train(mode)
        return self

    def eval(self) -> 'HistologyClassifier':
        """Set evaluation mode on the model and its backbone."""
        super().eval()
        self.backbone.eval()
        return self

    def to(self, device):
        """Move the model and its backbone to the given device."""
        super().to(device)
        self.device = device
        self.backbone = self.backbone.to(device)
        return self