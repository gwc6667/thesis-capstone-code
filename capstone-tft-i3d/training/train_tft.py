import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from models.tft_model import TemporalFusionModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_dummy_data(num_samples=1000, seq_len=12, input_size=5):
    """
    Generate learnable synthetic academic time-series data.
    The label mainly depends on academic patterns in the last few steps.
    """
    x = torch.randn(num_samples, seq_len, input_size)

    trend = x[:, -3:, 0].mean(dim=1)
    engagement = x[:, :, 1].mean(dim=1)
    consistency = -x[:, :, 2].std(dim=1)
    recent_performance = x[:, -1, 3] + 0.5 * x[:, -2, 3]
    noise = 0.08 * torch.randn(num_samples)

    score = (
        1.8 * trend
        + 1.4 * engagement
        + 1.0 * consistency
        + 1.6 * recent_performance
        + noise
    )
    y = (score > 0).float()
    return x, y


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    return total_loss / len(loader.dataset), correct / total


def train_tft():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    input_size = 5
    hidden_size = 64
    seq_len = 12
    batch_size = 32
    num_epochs = 10
    lr = 1e-3

    x, y = generate_dummy_data(
        num_samples=1000,
        seq_len=seq_len,
        input_size=input_size
    )

    split = int(0.8 * len(x))
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = TemporalFusionModel(
        input_size=input_size,
        hidden_size=hidden_size
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    save_path = "models/tft_checkpoint.pth"

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        avg_val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    print(f"Best Val Acc: {best_acc:.4f}")
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train_tft()