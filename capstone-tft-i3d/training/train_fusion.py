import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from models.fusion_model import GatedFusionModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_dummy_fusion_data(num_samples=300, tft_dim=64, i3d_dim=128):
    """
    Generate more realistic multimodal features:
    - academic features are moderately correlated with labels
    - emotion features are weakly correlated with labels
    """
    y = torch.randint(0, 2, (num_samples,)).float()

    tft_vec = torch.randn(num_samples, tft_dim) * 0.9
    tft_signal = (y.unsqueeze(1) * 2 - 1) * 0.45
    tft_vec[:, :10] += tft_signal

    i3d_vec = torch.randn(num_samples, i3d_dim) * 1.0
    emo_signal = (y.unsqueeze(1) * 2 - 1) * 0.12
    i3d_vec[:, :6] += emo_signal

    return tft_vec, i3d_vec, y


def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0

    with torch.no_grad():
        for tft_b, i3d_b, labels in loader:
            tft_b = tft_b.to(device)
            i3d_b = i3d_b.to(device)
            labels = labels.to(device)

            logits = model(tft_b, i3d_b)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader.dataset), correct / total


def train_fusion():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tft_vec, i3d_vec, y = generate_dummy_fusion_data()

    split = int(0.8 * len(y))
    train_ds = TensorDataset(tft_vec[:split], i3d_vec[:split], y[:split])
    val_ds = TensorDataset(tft_vec[split:], i3d_vec[split:], y[split:])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = GatedFusionModel(
        academic_dim=64,
        emotion_dim=128,
        hidden_dim=128,
        num_classes=1
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    save_path = "models/fusion_checkpoint.pth"

    for epoch in range(1, 9):
        model.train()
        total_loss = 0.0

        for tft_b, i3d_b, labels in train_loader:
            tft_b = tft_b.to(device)
            i3d_b = i3d_b.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(tft_b, i3d_b)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        avg_val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

        print(
            f"Epoch [{epoch}/8] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    print(f"Best Val Acc: {best_acc:.4f}")
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train_fusion()