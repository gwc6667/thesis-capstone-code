import torch
import torch.nn as nn
import torch.nn.functional as F


class I3DLikeModel(nn.Module):
    """
    Lightweight 3D CNN for synthetic video classification.
    Input: [B, 3, T, H, W]
    """

    def __init__(self, num_classes: int = 4, feature_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = nn.Linear(64, feature_dim)
        self.fc2 = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.fc2(features)
        return logits