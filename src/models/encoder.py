import torch
import torch.nn as nn
import torchvision.models as models

class TileMultiModalEncoder(nn.Module):
    def __init__(self):
        """Initialize the multimodal tile encoder with a frozen ResNet18 image backbone."""
        super().__init__()


        resnet = models.resnet18(weights="IMAGENET1K_V1")
        resnet.fc = nn.Identity()
        for p in resnet.parameters():
            p.requires_grad = False
        self.image_encoder = resnet


        self.image_proj = nn.Linear(512, 256)


        self.embedding_proj = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, image, embedding):
        """Encode and fuse image and embedding features for a tile."""
        img_feat = self.image_encoder(image)
        img_feat = self.image_proj(img_feat)

        emb_feat = self.embedding_proj(embedding)

        fused = torch.cat([img_feat, emb_feat], dim=1)
        return fused


def aggregate_slide(tile_features):
    """Aggregate tile-level features into a slide-level feature vector."""
    return torch.cat([
        tile_features.mean(dim=0),
        tile_features.std(dim=0)
    ])
