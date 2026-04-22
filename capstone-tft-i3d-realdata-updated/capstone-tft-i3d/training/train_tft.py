from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from eval.metrics import classification_metrics
from models.tft_model import TemporalFusionModel
from preprocess.preprocess_performance import build_oulad_tensors
from utils import get_device, project_root, save_checkpoint, set_seed, append_result_row, write_json


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_features = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_features.append(model.extract_features(batch_x).cpu())

    metrics = classification_metrics(all_labels, all_preds, average='binary')
    avg_loss = total_loss / len(loader.dataset)
    features = torch.cat(all_features, dim=0) if all_features else torch.empty(0)
    labels = torch.tensor(np.asarray(all_labels), dtype=torch.float32)
    return avg_loss, metrics, features, labels


def train_tft():
    set_seed(42)
    device = get_device()
    print('Using device:', device)

    hidden_size = 64
    batch_size = 64
    num_epochs = 10
    lr = 1e-3

    root = project_root()
    perf_dir = root / 'data' / 'performance'
    x, y, meta = build_oulad_tensors(perf_dir, seq_len=12, cache=True)
    input_size = int(meta['input_size'])

    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=y.numpy(),
    )
    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size)

    model = TemporalFusionModel(input_size=input_size, hidden_size=hidden_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    save_path = root / 'checkpoints' / 'tft_checkpoint.pth'
    feature_path = root / 'results' / 'tft_features.pt'
    metrics_csv = root / 'results' / 'training_metrics.csv'

    best_acc = 0.0
    last_train_loss = last_val_loss = None
    last_metrics = None
    best_features = None
    best_labels = None

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
        avg_val_loss, metrics, val_features, val_labels = evaluate(model, val_loader, criterion, device)
        last_train_loss, last_val_loss, last_metrics = avg_train_loss, avg_val_loss, metrics

        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            best_features = val_features.clone()
            best_labels = val_labels.clone()
            save_checkpoint(model, save_path)

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {metrics['accuracy']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1_score']:.4f}"
        )

    if best_features is not None:
        torch.save({'features': best_features, 'labels': best_labels}, feature_path)

    row = {
        'model': 'TFT',
        'dataset': 'OULAD',
        'epochs': num_epochs,
        'final_train_loss': round(last_train_loss, 4),
        'final_val_loss': round(last_val_loss, 4),
        'final_val_acc': round(last_metrics['accuracy'], 4),
        'final_precision': round(last_metrics['precision'], 4),
        'final_recall': round(last_metrics['recall'], 4),
        'final_f1_score': round(last_metrics['f1_score'], 4),
        'best_val_acc': round(best_acc, 4),
        'device': str(device),
        'num_samples': int(meta['num_samples']),
    }
    append_result_row(metrics_csv, row)
    write_json(root / 'results' / 'tft_summary.json', row)
    print(f'Best Val Acc: {best_acc:.4f}')
    print(f'Model saved to {save_path}')


if __name__ == '__main__':
    train_tft()
