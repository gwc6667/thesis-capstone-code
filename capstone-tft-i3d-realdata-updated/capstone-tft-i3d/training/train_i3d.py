from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from eval.metrics import classification_metrics
from models.i3d_model import I3DLikeModel
from preprocess.preprocess_emotions import build_emotion_index
from utils import get_device, project_root, save_checkpoint, set_seed, append_result_row, write_json


class EnterfaceVideoDataset(Dataset):
    def __init__(self, samples, num_frames: int = 8, image_size: int = 64):
        self.samples = samples
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def _read_video(self, video_path: str):
        capture = cv2.VideoCapture(video_path)
        frames = []
        success, frame = capture.read()
        while success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.image_size, self.image_size))
            frames.append(frame)
            success, frame = capture.read()
        capture.release()

        if not frames:
            frames = [np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)]

        if len(frames) >= self.num_frames:
            indices = np.linspace(0, len(frames) - 1, self.num_frames).astype(int)
            frames = [frames[idx] for idx in indices]
        else:
            while len(frames) < self.num_frames:
                frames.append(frames[-1])

        arr = np.stack(frames).astype(np.float32) / 255.0
        arr = np.transpose(arr, (3, 0, 1, 2))  # [C, T, H, W]
        return torch.tensor(arr, dtype=torch.float32)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video = self._read_video(sample['path'])
        label = torch.tensor(sample['label'], dtype=torch.long)
        return video, label


def _subject_sort_key(name: str):
    match = re.search(r'(\d+)', name)
    return int(match.group(1)) if match else 0


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
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_features.append(model.extract_features(batch_x).cpu())

    metrics = classification_metrics(all_labels, all_preds, average='weighted')
    avg_loss = total_loss / len(loader.dataset)
    features = torch.cat(all_features, dim=0) if all_features else torch.empty(0)
    labels = torch.tensor(np.asarray(all_labels), dtype=torch.long)
    return avg_loss, metrics, features, labels


def train_i3d():
    set_seed(42)
    device = get_device()
    print('Using device:', device)

    root = project_root()
    emotion_dir = root / 'data' / 'emotions'
    samples, label_map, summary = build_emotion_index(emotion_dir)
    num_classes = len(label_map)

    subjects = sorted({sample['subject'] for sample in samples}, key=_subject_sort_key)
    train_subjects, val_subjects = train_test_split(subjects, test_size=0.2, random_state=42)
    train_subjects = set(train_subjects)
    val_subjects = set(val_subjects)

    train_samples = [sample for sample in samples if sample['subject'] in train_subjects]
    val_samples = [sample for sample in samples if sample['subject'] in val_subjects]

    train_loader = DataLoader(EnterfaceVideoDataset(train_samples), batch_size=4, shuffle=True)
    val_loader = DataLoader(EnterfaceVideoDataset(val_samples), batch_size=4)

    model = I3DLikeModel(num_classes=num_classes, feature_dim=128).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    save_path = root / 'checkpoints' / 'i3d_checkpoint.pth'
    feature_path = root / 'results' / 'i3d_features.pt'
    metrics_csv = root / 'results' / 'training_metrics.csv'

    best_acc = 0.0
    last_train_loss = last_val_loss = None
    last_metrics = None
    best_features = None
    best_labels = None

    num_epochs = 6
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
        torch.save({'features': best_features, 'labels': best_labels, 'label_map': label_map}, feature_path)

    row = {
        'model': 'I3D',
        'dataset': 'eNTERFACE05',
        'epochs': num_epochs,
        'final_train_loss': round(last_train_loss, 4),
        'final_val_loss': round(last_val_loss, 4),
        'final_val_acc': round(last_metrics['accuracy'], 4),
        'final_precision': round(last_metrics['precision'], 4),
        'final_recall': round(last_metrics['recall'], 4),
        'final_f1_score': round(last_metrics['f1_score'], 4),
        'best_val_acc': round(best_acc, 4),
        'device': str(device),
        'num_samples': int(summary['num_samples']),
        'num_classes': int(num_classes),
    }
    append_result_row(metrics_csv, row)
    write_json(root / 'results' / 'i3d_summary.json', row)
    print(f'Best Val Acc: {best_acc:.4f}')
    print(f'Model saved to {save_path}')


if __name__ == '__main__':
    train_i3d()
