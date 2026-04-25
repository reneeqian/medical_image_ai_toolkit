"""
Figure generation for model testing run artifacts.

Three plots:
  - training_curve.png   : loss per epoch
  - confusion_matrix.png : pixel-level TP/FP/FN/TN with derived metrics
  - patient_samples.png  : 3×2 grid of GT annotation vs model prediction
"""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CT_WL, CT_WW = 40, 400   # cardiac soft-tissue window (HU)


def plot_training_curve(history: list, path: Path) -> Path:
    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, losses, marker="o", linewidth=2, color="#2196F3", markersize=6)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title("Training Loss Curve", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return Path(path)


def plot_confusion_matrix(metrics: dict, path: Path) -> Path:
    tp = int(metrics.get("tp", 0))
    fp = int(metrics.get("fp", 0))
    fn = int(metrics.get("fn", 0))
    tn = int(metrics.get("tn", 0))
    total = tp + fp + fn + tn or 1

    # rows = actual, cols = predicted
    matrix = np.array([[tp, fn], [fp, tn]], dtype=float)
    cell_labels = [["TP", "FN"], ["FP", "TN"]]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues", vmin=0)

    for i in range(2):
        for j in range(2):
            val   = int(matrix[i, j])
            pct   = 100.0 * val / total
            color = "white" if matrix[i, j] > matrix.max() * 0.55 else "black"
            ax.text(j, i, f"{cell_labels[i][j]}\n{val:,}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=12,
                    fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Positive", "Predicted Negative"], fontsize=11)
    ax.set_yticklabels(["Actual Positive", "Actual Negative"], fontsize=11)
    ax.set_title("Pixel-Level Confusion Matrix", fontsize=14, fontweight="bold", pad=12)

    dice = metrics.get("dice")
    iou  = metrics.get("iou")
    prec = metrics.get("precision")
    rec  = metrics.get("recall")
    if None not in (dice, iou, prec, rec):
        fig.text(0.5, 0.01,
                 f"Dice: {dice:.4f}   IoU: {iou:.4f}   Precision: {prec:.4f}   Recall: {rec:.4f}",
                 ha="center", fontsize=10, style="italic")
        plt.tight_layout(rect=[0, 0.04, 1, 1])
    else:
        plt.tight_layout()

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return Path(path)


def plot_patient_samples(sample_viz_data: list, path: Path) -> Path:
    n = len(sample_viz_data)
    if n == 0:
        return None

    fig, axes = plt.subplots(n, 2, figsize=(12, 5.5 * n), facecolor="black")
    if n == 1:
        axes = [axes]

    vmin = CT_WL - CT_WW / 2
    vmax = CT_WL + CT_WW / 2

    for row, s in enumerate(sample_viz_data):
        hu        = s["hu"]
        gt_mask   = s["gt_mask"]
        pred      = s["pred"]
        pid       = s["patient_id"]
        slice_idx = s["slice_idx"]

        for ax, overlay, cmap, col_title in zip(
            axes[row],
            [gt_mask,   pred],
            ["Oranges", "hot"],
            ["GT Annotation", "Model Prediction"],
            strict=True,
        ):
            ax.imshow(hu, cmap="gray", vmin=vmin, vmax=vmax,
                      interpolation="bilinear")
            masked = np.ma.masked_where(overlay < 0.05, overlay)
            ax.imshow(masked, cmap=cmap, alpha=0.55, vmin=0, vmax=1,
                      interpolation="none")
            ax.set_title(
                f"Patient {pid}  |  Slice {slice_idx}\n{col_title}",
                color="white", fontsize=12, fontweight="bold", pad=6,
            )
            ax.axis("off")

    fig.patch.set_facecolor("black")
    plt.tight_layout(pad=1.2)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    return Path(path)
