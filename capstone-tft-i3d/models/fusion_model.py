import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusionModel(nn.Module):
    """
    Improved fusion model:
    - project academic and emotion features to the same hidden space
    - confidence-aware weighting for emotion modality
    - gated fusion
    - modality dropout for robustness
    """

    def __init__(
        self,
        academic_dim=64,
        emotion_dim=128,
        hidden_dim=128,
        num_classes=1,
        modality_dropout=0.2
    ):
        super().__init__()
        self.academic_proj = nn.Linear(academic_dim, hidden_dim)
        self.emotion_proj = nn.Linear(emotion_dim, hidden_dim)
        self.gate_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.modality_dropout = modality_dropout

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, h_academic, h_emotion):
        h_a = F.relu(self.academic_proj(h_academic))
        h_e = F.relu(self.emotion_proj(h_emotion))

        # simple confidence estimate from emotion magnitude
        emotion_conf = torch.sigmoid(h_e.abs().mean(dim=1, keepdim=True))
        h_e = h_e * emotion_conf

        # modality dropout during training
        if self.training and self.modality_dropout > 0:
            mask = (
                torch.rand(h_e.size(0), 1, device=h_e.device) > self.modality_dropout
            ).float()
            h_e = h_e * mask

        z = torch.cat([h_a, h_e], dim=-1)
        gate = torch.sigmoid(self.gate_fc(z))
        fused = gate * h_a + (1 - gate) * h_e

        out = self.classifier(fused).squeeze(-1)
        return out