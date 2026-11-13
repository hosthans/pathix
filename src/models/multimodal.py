from collections import defaultdict
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torchvision.models as models

import torch
import torch.nn as nn
import torchvision.models as models
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from src.core.registry_utils import register_model





class FeatureExtractor(nn.Module):
    """
    Generic feature extractor (frozen backbone) for image-to-features.
    Supports: resnet18, densenet121, densenet169, convnext_large, swin_v2_b, gigapath.
    """
    # Tiles per encoder call for the always-frozen GigaPath encoder (bounds peak memory for large bags).
    GIGAPATH_CHUNK = 64

    def __init__(self, model_name: str, freeze: bool = True):
        """Initialize the frozen (optionally) backbone feature extractor."""
        super().__init__()
        self.model_name = model_name
        # True for encoders that are never trained (GigaPath): forward runs in eval mode under no_grad, so
        # neither `freeze=False` nor a later unfreeze_backbone_epoch can change their weights.
        self.always_frozen = False

        if model_name == "gigapath":
            # Same tile encoder as the patch approach (src/models/gigapath.py, GigaPathClassifier, as in
            # Liebel): Prov-GigaPath via timm, pretrained, no head -> 1536-d tile features; needs HF_TOKEN.
            import timm
            backbone = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True, num_classes=0)
            self.feat_dim = 1536
            self.always_frozen = True

        elif model_name == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()

        elif model_name == "densenet121":
            from torchvision.models import densenet121, DenseNet121_Weights
            backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
            self.feat_dim = backbone.classifier.in_features
            backbone.classifier = nn.Identity()

        elif model_name == "densenet169":
            from torchvision.models import densenet169, DenseNet169_Weights
            backbone = densenet169(weights=DenseNet169_Weights.DEFAULT)
            self.feat_dim = backbone.classifier.in_features
            backbone.classifier = nn.Identity()

        elif model_name == "convnext_large":
            from torchvision.models import convnext_large, ConvNeXt_Large_Weights
            backbone = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
            self.feat_dim = 1536
            backbone.classifier = nn.Identity()

        elif model_name == "swin_v2_b":
            from torchvision.models import swin_v2_b, Swin_V2_B_Weights
            backbone = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
            self.feat_dim = backbone.head.in_features
            backbone.head = nn.Identity()

        else:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                "Use 'resnet18', 'densenet121', 'densenet169', "
                "'convnext_large', 'swin_v2_b', or 'gigapath'."
            )

        self.backbone = backbone

        if freeze or self.always_frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        """Set training mode; an always-frozen encoder stays in eval mode (as GigaPathClassifier)."""
        super().train(mode)
        if self.always_frozen:
            self.backbone.eval()
        return self

    def forward(self, x):
        """Extract features from images."""
        if self.always_frozen:
            self.backbone.eval()
            with torch.no_grad():
                return torch.cat([self.backbone(chunk) for chunk in torch.split(x, self.GIGAPATH_CHUNK)], dim=0)
        return self.backbone(x)

@register_model(('mil', 'any', 'multimodal'))
class AttentionMIL(nn.Module):
    """
    Attention-based MIL model with multimodal (image + embedding) attention.
    Uses FeatureExtractor for image backbone.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 1,
        dropout_rate: float = 0.2,
        in_dim: int = 512,
        instance_dim: int = 256,
        tile_embed_dim: int = 1536,
        freeze_backbone: bool = True,
    ):
        """Initialize the multimodal attention-MIL model."""
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate




        self.feature_extractor = FeatureExtractor(model_name, freeze=freeze_backbone)
        img_feat_dim = self.feature_extractor.feat_dim




        self.instance_proj = nn.Sequential(
            nn.Linear(img_feat_dim, instance_dim),
            nn.ReLU(),
        )




        self.attn = nn.Sequential(
            nn.Linear(instance_dim + tile_embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )




        self.classifier = nn.Sequential(
            nn.Linear(instance_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(256, self.num_classes),
        )

    @staticmethod
    def _segment_softmax(scores, bag_idx, num_bags):
        max_per_bag = torch.full(
            (num_bags,), -1e9, device=scores.device
        ).scatter_reduce(
            0, bag_idx, scores, reduce="amax", include_self=True
        )

        stabilized = scores - max_per_bag[bag_idx]
        exp = torch.exp(stabilized)

        denom = torch.zeros(
            (num_bags,), device=scores.device
        ).scatter_add_(0, bag_idx, exp)

        return exp / (denom[bag_idx] + 1e-12)

    def forward(self, images, bag_idx, tile_embeds):
        """Compute bag-level logits and per-tile attention weights."""
        num_bags = int(bag_idx.max().item()) + 1


        feats = self.feature_extractor(images)
        h = self.instance_proj(feats)


        h_cat = torch.cat([h, tile_embeds], dim=1)
        a = self.attn(h_cat).squeeze(1)
        alpha = self._segment_softmax(a, bag_idx, num_bags)


        weighted = h * alpha.unsqueeze(1)
        bag_repr = torch.zeros(
            (num_bags, h.size(1)), device=h.device
        ).scatter_add_(
            0,
            bag_idx.unsqueeze(1).expand(-1, h.size(1)),
            weighted,
        )

        logits = self.classifier(bag_repr)

        if self.num_classes == 1:
            logits = logits.squeeze(1)

        return logits, alpha


from pathlib import Path
import sys
import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.core.registry_utils import register_model


@register_model(('mil', 'any', 'image'))
class AttentionMIL_NoEmbeddings(nn.Module):
    """
    Attention-based MIL model (image-only).
    Uses FeatureExtractor for image backbone.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 1,
        dropout_rate: float = 0.2,
        in_dim: int = 512,
        instance_dim: int = 256,
        tile_embed_dim: int = 1536,
        freeze_backbone: bool = True,
    ):
        """Initialize the image-only attention-MIL model."""
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes




        self.feature_extractor = FeatureExtractor(model_name, freeze=freeze_backbone)
        img_feat_dim = self.feature_extractor.feat_dim




        self.instance_proj = nn.Sequential(
            nn.Linear(img_feat_dim, instance_dim),
            nn.ReLU(),
        )




        self.attn = nn.Sequential(
            nn.Linear(instance_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )




        self.classifier = nn.Sequential(
            nn.Linear(instance_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _segment_softmax(scores, bag_idx, num_bags):
        max_per_bag = torch.full(
            (num_bags,), -1e9, device=scores.device
        ).scatter_reduce(
            0, bag_idx, scores, reduce="amax", include_self=True
        )

        stabilized = scores - max_per_bag[bag_idx]
        exp = torch.exp(stabilized)

        denom = torch.zeros(
            (num_bags,), device=scores.device
        ).scatter_add_(0, bag_idx, exp)

        return exp / (denom[bag_idx] + 1e-12)

    def forward(self, images, bag_idx):
        """Compute bag-level logits and per-tile attention weights from images only."""
        num_bags = int(bag_idx.max().item()) + 1

        feats = self.feature_extractor(images)
        h = self.instance_proj(feats)

        a = self.attn(h).squeeze(1)
        alpha = self._segment_softmax(a, bag_idx, num_bags)

        weighted = h * alpha.unsqueeze(1)

        bag_repr = torch.zeros(
            (num_bags, h.size(1)), device=h.device
        ).scatter_add_(
            0,
            bag_idx.unsqueeze(1).expand(-1, h.size(1)),
            weighted,
        )

        logits = self.classifier(bag_repr)

        if self.num_classes == 1:
            logits = logits.squeeze(1)

        return logits, alpha

@register_model(('cnn', 'any', 'multimodal'))
class MultiModalClassifier(nn.Module):
    """
    Multimodal classifier combining a frozen image backbone with external embeddings.
    Uses FeatureExtractor for image encoding.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 1,
        dropout_rate: float = 0.2,
        in_dim: int = 512,
        embedding_dim: int = 1536,
        proj_dim: int = 256,
    ):
        """Initialize the multimodal classifier combining image and embedding features."""
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate




        self.feature_extractor = FeatureExtractor(model_name, freeze=True)
        img_feat_dim = self.feature_extractor.feat_dim




        self.image_proj = nn.Linear(img_feat_dim, proj_dim)

        self.embedding_proj = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, proj_dim),
        )




        self.classifier = nn.Sequential(
            nn.Linear(2 * proj_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(256, self.num_classes),
        )

    def forward(
        self,
        images: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute classification logits from fused image and embedding features."""
        img_feat = self.feature_extractor(images)
        img_feat = self.image_proj(img_feat)

        emb_feat = self.embedding_proj(embeddings)

        fused = torch.cat([img_feat, emb_feat], dim=1)
        logits = self.classifier(fused)

        if self.num_classes == 1:
            return logits.squeeze(1)

        return logits

def group_tiles_by_slide(dataset):
    """Group dataset tile indices by their parent slide name."""
    slide_to_indices = defaultdict(list)
    for i in range(len(dataset)):
        _, _, _, meta = dataset[i]
        slide_to_indices[meta["slide_name"]].append(i)
    return list(slide_to_indices.values())