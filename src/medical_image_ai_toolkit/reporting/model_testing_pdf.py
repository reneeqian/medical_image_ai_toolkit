"""
Generate a multi-page PDF report for a model testing run.

Pages
-----
1. Title / key metrics (dice, iou, precision, recall)
2. Confusion matrix
3. Metrics bar chart
4. Per-patient dice score histogram
5. Patient sample images (embedded from patient_samples.png if it exists)
"""
from __future__ import annotations

import json
import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from medical_image_ai_toolkit.reporting._pdf_helpers import open_pdf, text_page
from medical_image_ai_toolkit.reporting._run_finder import find_run_dir
from medical_image_ai_toolkit.visualizations.model_testing_figures import plot_confusion_matrix


def generate_model_testing_pdf(
    runs_root: Path | str,
    run_id: str | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """
    Generate a PDF report for a model testing run.

    Parameters
    ----------
    runs_root:
        Directory containing timestamped model testing run subdirectories.
    run_id:
        Exact or prefix-matched run directory name.  Defaults to most recent.
    output_path:
        Where to write the PDF.  Defaults to
        ``{run_dir}/model_testing_report.pdf``.

    Returns
    -------
    Path
        Absolute path of the written PDF file.
    """
    run_dir = find_run_dir(runs_root, run_id, requires_file="model_testing_report.json")
    report_json = run_dir / "model_testing_report.json"

    data = json.loads(report_json.read_text())
    metrics = data.get("metrics", {})
    per_patient = data.get("per_patient_results", [])

    out_path = Path(output_path) if output_path else run_dir / "model_testing_report.pdf"

    with open_pdf(out_path) as pdf:
        # --- Page 1: title / key metrics ---
        def _fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else ("—" if v is None else str(v))

        summary_lines = [
            ("Run ID", run_dir.name),
            ("Test patients", metrics.get("num_test_patients", "—")),
            ("Test samples", metrics.get("num_test_samples", "—")),
            ("Mean loss", _fmt(metrics.get("mean_loss"))),
            ("Dice", _fmt(metrics.get("dice"))),
            ("IoU", _fmt(metrics.get("iou"))),
            ("Precision", _fmt(metrics.get("precision"))),
            ("Recall", _fmt(metrics.get("recall"))),
            ("Accuracy", _fmt(metrics.get("accuracy"))),
        ]
        text_page(pdf, summary_lines, f"Model Testing Report\n{run_dir.name}")

        # --- Page 2: confusion matrix ---
        _embed_confusion_matrix(pdf, metrics, run_dir)

        # --- Page 3: metrics bar chart ---
        _plot_metrics_bar(pdf, metrics)

        # --- Page 4: per-patient dice histogram ---
        if per_patient:
            _plot_dice_histogram(pdf, per_patient)

        # --- Page 5: patient samples image (if present) ---
        samples_png = run_dir / "patient_samples.png"
        if samples_png.exists():
            _embed_image(pdf, samples_png, "Sample Predictions")

    return out_path.resolve()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _embed_confusion_matrix(pdf, metrics: dict, run_dir: Path) -> None:
    tmp = run_dir / "_tmp_cm.png"
    try:
        plot_confusion_matrix(metrics, tmp)
        _embed_image(pdf, tmp, "Pixel-Level Confusion Matrix")
    finally:
        if tmp.exists():
            tmp.unlink()


def _plot_metrics_bar(pdf, metrics: dict) -> None:
    keys = ["dice", "iou", "precision", "recall"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    vals = [metrics.get(k) for k in keys]

    if all(v is None for v in vals):
        return

    vals_clean = [v if v is not None else 0.0 for v in vals]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(keys, vals_clean, color=colors, edgecolor="white", width=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Key Segmentation Metrics", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, vals):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_dice_histogram(pdf, per_patient: list[dict]) -> None:
    dice_vals = [p["dice"] for p in per_patient if isinstance(p.get("dice"), float)]
    if not dice_vals:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(dice_vals, bins=min(20, len(dice_vals)), color="#2196F3",
            edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(dice_vals), color="#E91E63", linewidth=2, linestyle="--",
               label=f"Mean = {np.mean(dice_vals):.4f}")
    ax.set_xlabel("Dice Score", fontsize=12)
    ax.set_ylabel("Patients", fontsize=12)
    ax.set_title("Per-Patient Dice Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _embed_image(pdf, image_path: Path, title: str) -> None:
    img = plt.imread(str(image_path))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
