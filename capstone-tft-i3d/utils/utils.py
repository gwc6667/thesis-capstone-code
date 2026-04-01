import os
import random
from typing import Sequence

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def get_device() -> torch.device:
    """Return available torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    """Save model checkpoint."""
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved to: {path}")


def load_checkpoint(model: torch.nn.Module, path: str, map_location=None) -> torch.nn.Module:
    """Load model checkpoint."""
    if map_location is None:
        map_location = get_device()
    model.load_state_dict(torch.load(path, map_location=map_location))
    print(f"Checkpoint loaded from: {path}")
    return model


def accuracy_score(y_true: Sequence, y_pred: Sequence) -> float:
    """Compute classification accuracy."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float((y_true == y_pred).mean())


def precision_score_binary(y_true: Sequence, y_pred: Sequence) -> float:
    """Compute binary precision. Positive class is assumed to be 1."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp))


def recall_score_binary(y_true: Sequence, y_pred: Sequence) -> float:
    """Compute binary recall. Positive class is assumed to be 1."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))


def f1_score_binary(y_true: Sequence, y_pred: Sequence) -> float:
    """Compute binary F1 score."""
    precision = precision_score_binary(y_true, y_pred)
    recall = recall_score_binary(y_true, y_pred)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def mean_absolute_error(y_true: Sequence, y_pred: Sequence) -> float:
    """Compute MAE for regression tasks."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: Sequence, y_pred: Sequence) -> float:
    """Compute RMSE for regression tasks."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def print_classification_metrics(y_true: Sequence, y_pred: Sequence) -> None:
    """Print common classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score_binary(y_true, y_pred)
    rec = recall_score_binary(y_true, y_pred)
    f1 = f1_score_binary(y_true, y_pred)
    print("Classification Metrics")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")


def print_regression_metrics(y_true: Sequence, y_pred: Sequence) -> None:
    """Print common regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    print("Regression Metrics")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
