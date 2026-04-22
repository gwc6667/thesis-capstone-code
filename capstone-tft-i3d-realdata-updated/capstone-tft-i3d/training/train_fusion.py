from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from eval.metrics import classification_metrics
from models.fusion_model import GatedFusionModel
from utils import get_device, project_root, save_checkpoint, set_seed, append_result_row, write_json


def build_prototype_fusion_pairs(tft_payload, i3d_payload):
    academic_features = tft_payload['features'].float()
    academic_labels = tft_payload['labels'].float()
    emotion_features = i3d_payload['features'].float()
    emotion_labels = i3d_payload['labels'].long()

    label_map = i3d_payload.get('label_map', {})
    positive_ids = {idx for emotion, idx in label_map.items() if emotion in {'happiness', 'surprise'}}
    if not positive_ids:
        positive_ids = {3, 5}
    emotion_positive = torch.tensor([1.0 if int(label.item()) in positive_ids else 0.0 for label in emotion_labels])
    pos_idx = torch.where(emotion_positive == 1)[0]
    neg_idx = torch.where(emotion_positive == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError('Need both positive-like and negative-like emotion samples for fusion prototype.')

    paired_emotion_features = []
    generator = torch.Generator().manual_seed(42)
    for label in academic_labels:
        same_pool = pos_idx if label.item() >= 0.5 else neg_idx
        alt_pool = neg_idx if label.item() >= 0.5 else pos_idx
        use_same = torch.rand(1, generator=generator).item() < 0.7
        pool = same_pool if use_same else alt_pool
        choice = pool[torch.randint(0, len(pool), (1,), generator=generator)].item()
        paired_emotion_features.append(emotion_features[choice])

    paired_emotion_features = torch.stack(paired_emotion_features, dim=0)
    return academic_features, paired_emotion_features, academic_labels


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for tft_b, i3d_b, labels in loader:
            tft_b = tft_b.to(device)
            i3d_b = i3d_b.to(device)
            labels = labels.to(device)
            logits = model(tft_b, i3d_b)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = classification_metrics(all_labels, all_preds, average='binary')
    return total_loss / len(loader.dataset), metrics


def train_fusion():
    set_seed(42)
    device = get_device()
    print('Using device:', device)

    root = project_root()
    tft_feature_path = root / 'results' / 'tft_features.pt'
    i3d_feature_path = root / 'results' / 'i3d_features.pt'
    if not tft_feature_path.exists() or not i3d_feature_path.exists():
        raise FileNotFoundError('Run train_tft.py and train_i3d.py first to create feature files for fusion.')

    tft_payload = torch.load(tft_feature_path, map_location='cpu')
    i3d_payload = torch.load(i3d_feature_path, map_location='cpu')
    tft_vec, i3d_vec, y = build_prototype_fusion_pairs(tft_payload, i3d_payload)

    split = int(0.8 * len(y))
    train_loader = DataLoader(TensorDataset(tft_vec[:split], i3d_vec[:split], y[:split]), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(tft_vec[split:], i3d_vec[split:], y[split:]), batch_size=32)

    model = GatedFusionModel(academic_dim=tft_vec.shape[1], emotion_dim=i3d_vec.shape[1], hidden_dim=128, num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    save_path = root / 'checkpoints' / 'fusion_checkpoint.pth'
    metrics_csv = root / 'results' / 'training_metrics.csv'
    best_acc = 0.0
    last_train_loss = last_val_loss = None
    last_metrics = None

    num_epochs = 8
    for epoch in range(1, num_epochs + 1):
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
        avg_val_loss, metrics = evaluate(model, val_loader, criterion, device)
        last_train_loss, last_val_loss, last_metrics = avg_train_loss, avg_val_loss, metrics
        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
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

    row = {
        'model': 'Fusion',
        'dataset': 'Prototype(OULAD+eNTERFACE05)',
        'epochs': num_epochs,
        'final_train_loss': round(last_train_loss, 4),
        'final_val_loss': round(last_val_loss, 4),
        'final_val_acc': round(last_metrics['accuracy'], 4),
        'final_precision': round(last_metrics['precision'], 4),
        'final_recall': round(last_metrics['recall'], 4),
        'final_f1_score': round(last_metrics['f1_score'], 4),
        'best_val_acc': round(best_acc, 4),
        'device': str(device),
        'num_samples': int(len(y)),
    }
    append_result_row(metrics_csv, row)
    write_json(root / 'results' / 'fusion_summary.json', row)
    print(f'Best Val Acc: {best_acc:.4f}')
    print(f'Model saved to {save_path}')


if __name__ == '__main__':
    train_fusion()
