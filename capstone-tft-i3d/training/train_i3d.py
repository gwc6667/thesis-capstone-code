import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from models.i3d_model import I3DLikeModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_dummy_video_data(
    num_samples=200,
    num_frames=8,
    height=32,
    width=32,
    num_classes=4
):
    """
    Generate harder synthetic video data.
    Each class still has a weak pattern, but not too easy.
    """
    x = 0.45 * torch.randn(num_samples, 3, num_frames, height, width)
    y = torch.randint(low=0, high=num_classes, size=(num_samples,))

    for i in range(num_samples):
        label = int(y[i].item())
        if label == 0:
            x[i, :, :, :16, :16] += 0.35
        elif label == 1:
            x[i, :, :, 16:, 16:] += 0.35
        elif label == 2:
            for t in range(num_frames):
                x[i, :, t, :, :] += (t / num_frames) * 0.25
        elif label == 3:
            for t in range(num_frames):
                x[i, :, t, :, :] += ((num_frames - t) / num_frames) * 0.25

    return x, y


def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total = 0, 0
    val_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            val_loss += loss.item() * batch_x.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    return val_loss / len(loader.dataset), correct / total


def train_i3d():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_classes = 4
    num_frames = 8
    height = 32
    width = 32
    batch_size = 8
    num_epochs = 6
    lr = 1e-3

    x, y = generate_dummy_video_data(
        num_samples=200,
        num_frames=num_frames,
        height=height,
        width=width,
        num_classes=num_classes,
    )

    split = int(0.8 * len(x))
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = I3DLikeModel(num_classes=num_classes, feature_dim=128).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    save_path = "models/i3d_checkpoint.pth"

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
    train_i3d()