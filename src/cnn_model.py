import torch
import torch.nn as nn


IMG_SIZE = 96


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class FaceCNN(nn.Module):
    """Small CNN for grayscale face identification (input: 1 x 96 x 96)."""

    def __init__(self, num_classes, embedding_dim=128, dropout=0.4):
        super().__init__()

        self.features = nn.Sequential(
            conv_block(1, 32),     # 96 -> 48
            conv_block(32, 64),    # 48 -> 24
            conv_block(64, 128),   # 24 -> 12
            conv_block(128, 256),  # 12 -> 6
            nn.AdaptiveAvgPool2d(1),
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.embedding(self.features(x)))

    def embed(self, x):
        return self.embedding(self.features(x))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
