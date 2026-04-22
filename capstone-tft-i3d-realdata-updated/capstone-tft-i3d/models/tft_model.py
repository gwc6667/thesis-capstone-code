import torch
import torch.nn as nn


class TemporalFusionModel(nn.Module):
    """
    Simplified Temporal Fusion model:
    - LSTM encoder
    - temporal attention
    - binary prediction head
    """

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.attn_layer = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def extract_features(self, x):
        out, _ = self.lstm(x)                # [B, T, H]
        scores = self.attn_layer(out).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(out * weights.unsqueeze(-1), dim=1)
        context = self.dropout(context)
        return context

    def forward(self, x):
        context = self.extract_features(x)
        logits = self.fc(context).squeeze(-1)
        return logits
