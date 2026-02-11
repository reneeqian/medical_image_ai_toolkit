from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_sample_visualizations(
    *,
    image: torch.Tensor,       # (1, H, W)
    label: torch.Tensor | None,
    prediction: torch.Tensor,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    img = image.squeeze().cpu().numpy()

    # --- CT image ---
    plt.imshow(img, cmap="gray")
    plt.title("CT Slice")
    plt.axis("off")
    plt.savefig(out_dir / "sample_ct.png", bbox_inches="tight")
    plt.close()

    # --- Label ---
    if label is not None:
        plt.imshow(label.squeeze().cpu().numpy(), cmap="Reds")
        plt.title("Label")
        plt.axis("off")
        plt.savefig(out_dir / "sample_label.png", bbox_inches="tight")
        plt.close()

    # --- Prediction ---
    plt.imshow(img, cmap="gray")
    plt.text(
        5, 15,
        f"Pred: {prediction.item():.3f}",
        color="yellow",
        fontsize=12,
        bbox=dict(facecolor="black", alpha=0.6),
    )
    plt.axis("off")
    plt.savefig(out_dir / "sample_prediction.png", bbox_inches="tight")
    plt.close()

    # --- Overlay (if label exists) ---
    if label is not None:
        plt.imshow(img, cmap="gray")
        plt.imshow(
            label.squeeze().cpu().numpy(),
            cmap="Reds",
            alpha=0.4,
        )
        plt.title("CT + Label Overlay")
        plt.axis("off")
        plt.savefig(out_dir / "sample_overlay.png", bbox_inches="tight")
        plt.close()
