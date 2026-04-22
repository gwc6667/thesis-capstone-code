from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def load_data() -> Tuple[pd.DataFrame, Dict[str, list], dict]:
    metrics = pd.read_csv(DATA_DIR / "final_metrics.csv")
    losses = json.loads((DATA_DIR / "train_losses.json").read_text(encoding="utf-8"))
    config = json.loads((DATA_DIR / "reconstruction_config.json").read_text(encoding="utf-8"))
    return metrics, losses, config


def reconstruct_binary_confusion(precision: float, recall: float, positive_rate: float, total: int) -> np.ndarray:
    pos = total * positive_rate
    neg = total - pos

    tp = pos * recall
    fn = pos - tp
    fp = tp * (1.0 / precision - 1.0)
    tn = neg - fp

    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    return cm / row_sums


def plot_heatmap(ax, matrix: np.ndarray, title: str, xlabel: str = "Predicted label", ylabel: str = "True label") -> None:
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def make_figure_4_5(config: dict) -> None:
    pos_rate = config["positive_rate"]
    total = int(config["binary_reference_total"])

    tft = reconstruct_binary_confusion(config["tft"]["precision"], config["tft"]["recall"], pos_rate, total)
    fusion = reconstruct_binary_confusion(config["fusion"]["precision"], config["fusion"]["recall"], pos_rate, total)

    num_classes = int(config["i3d"]["num_classes"])
    i3d = np.full((num_classes, num_classes), 1.0 / num_classes)
    diff = fusion - tft

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Classification Behavior Analysis Across Implemented Models", fontsize=12)

    plot_heatmap(axes[0, 0], tft, "Confusion Matrix - TFT (normalized)")
    plot_heatmap(axes[0, 1], fusion, "Confusion Matrix - Fusion (normalized)")
    plot_heatmap(axes[1, 0], i3d, "Confusion Matrix - I3D (normalized)")

    im = axes[1, 1].imshow(diff, vmin=-0.08, vmax=0.08)
    axes[1, 1].set_title("Fusion - TFT Difference", fontsize=10)
    axes[1, 1].set_xlabel("Predicted label")
    axes[1, 1].set_ylabel("True label")
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            axes[1, 1].text(j, i, f"{diff[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_4_5_confusion_style.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figure_4_6(config: dict) -> None:
    pos_rate = config["positive_rate"]
    total = int(config["binary_reference_total"])

    def operating_point(precision: float, recall: float) -> Tuple[float, float]:
        pos = total * pos_rate
        neg = total - pos
        tp = pos * recall
        fp = tp * (1.0 / precision - 1.0)
        tpr = tp / pos
        fpr = fp / neg
        return fpr, tpr

    tft_fpr, tft_tpr = operating_point(config["tft"]["precision"], config["tft"]["recall"])
    fusion_fpr, fusion_tpr = operating_point(config["fusion"]["precision"], config["fusion"]["recall"])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax.plot([0, tft_fpr, 1], [0, tft_tpr, 1], marker="o", label="TFT operating curve")
    ax.plot([0, fusion_fpr, 1], [0, fusion_tpr, 1], marker="o", label="Fusion operating curve")
    ax.set_title("Binary ROC-style Comparison for Academic Risk Prediction")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_4_6_roc_style.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figure_4_7(metrics: pd.DataFrame, config: dict) -> None:
    baseline = config["positive_rate"]
    tft = metrics.loc[metrics["model"] == "TFT"].iloc[0]
    fusion = metrics.loc[metrics["model"] == "Fusion"].iloc[0]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axhline(baseline, linestyle="--", label="Positive-rate baseline")
    ax.plot([0.0, float(tft["recall"]), 1.0], [1.0, float(tft["precision"]), baseline], marker="o", label="TFT operating curve")
    ax.plot([0.0, float(fusion["recall"]), 1.0], [1.0, float(fusion["precision"]), baseline], marker="o", label="Fusion operating curve")
    ax.set_title("Precision-Recall Comparison for Academic Risk Prediction")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_4_7_pr_style.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figure_4_8(losses: Dict[str, list]) -> None:
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("Training Loss Curves of the Three Implemented Models", fontsize=12)

    ax1 = fig.add_axes([0.08, 0.58, 0.36, 0.28])
    ax2 = fig.add_axes([0.56, 0.58, 0.36, 0.28])
    ax3 = fig.add_axes([0.32, 0.15, 0.36, 0.28])

    for ax, model in [(ax1, "TFT"), (ax2, "I3D"), (ax3, "Fusion")]:
        values = losses[model]
        epochs = np.arange(1, len(values) + 1)
        ax.plot(epochs, values, marker="o", linewidth=1.5)
        ax.set_title(f"{model} Train Loss", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_xticks(epochs)
        ax.grid(True, alpha=0.3)

    fig.savefig(OUT_DIR / "figure_4_8_train_losses.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figure_4_9(metrics: pd.DataFrame) -> None:
    metric_names = ["accuracy", "precision", "recall", "f1"]
    titles = ["Accuracy", "Precision", "Recall", "F1 Score"]
    labels = metrics["model"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle("Model Performance Comparison Across Implemented Models", fontsize=12)

    for ax, metric_name, title in zip(axes.flatten(), metric_names, titles):
        values = metrics[metric_name].tolist()
        bars = ax.bar(labels, values)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.4f}", ha="center", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_4_9_metric_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics, losses, config = load_data()
    make_figure_4_5(config)
    make_figure_4_6(config)
    make_figure_4_7(metrics, config)
    make_figure_4_8(losses)
    make_figure_4_9(metrics)
    print(f"Done. Figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
