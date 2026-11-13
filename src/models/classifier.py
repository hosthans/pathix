import torch.nn as nn

class SlideClassifier(nn.Module):
    def __init__(self):
        """Initialize the slide-level classifier head."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        """Forward pass through the classifier head."""
        return self.net(x).squeeze(1)
