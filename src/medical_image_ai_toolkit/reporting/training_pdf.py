"""
Generate a multi-page PDF report for a training run.

Pages
-----
1. Title / summary (run_id, model, lr, epochs, loss)
2. Training loss curve (+ val_loss overlay when available)
3. Per-epoch metrics (dice, iou, precision, recall) — omitted when absent
4. Partition / config summary
"""
from __future__ import annotations

import json
import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt

from medical_image_ai_toolkit.reporting._pdf_helpers import open_pdf, text_page
from medical_image_ai_toolkit.reporting._run_finder import find_run_dir
from medical_image_ai_toolkit.visualizations.model_testing_figures import plot_training_curve

_METRIC_COLORS = {
    "dice": "#2196F3",
    "iou": "#4CAF50",
    "precision": "#FF9800",
    "recall": "#E91E63",
}


def generate_training_pdf(
    runs_root: Path | str,
    run_id: str | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """
    Generate a PDF report for a training run and return the output path.

    Parameters
    ----------
    runs_root:
        Directory containing timestamped training run subdirectories.
    run_id:
        Exact or prefix-matched run directory name.  Defaults to the most
        recent run when ``None``.
    output_path:
        Where to write the PDF.  Defaults to ``{run_dir}/training_report.pdf``.

    Returns
    -------
    Path
        Absolute path of the written PDF file.

    Raises
    ------
    FileNotFoundError
        When *runs_root* is empty or *run_id* is not found.
    ValueError
        When ``training_report.json`` is missing from the run directory.
    """
    run_dir = find_run_dir(runs_root, run_id, requires_file="training_report.json")
    report_json = run_dir / "training_report.json"

    data = json.loads(report_json.read_text())
    metrics = data.get("metrics", {})
    history = data.get("history", [])
    config = data.get("config", {})
    datasource = data.get("datasource", {})

    out_path = Path(output_path) if output_path else run_dir / "training_report.pdf"

    with open_pdf(out_path) as pdf:
        # --- Page 1: title / summary ---
        model_name = str(config.get("task", "—"))
        lr = config.get("learning_rate", "—")
        epochs_trained = metrics.get("epochs_trained", metrics.get("num_epochs", "—"))
        epochs_max = config.get("epochs", "—")
        final_loss = metrics.get("final_loss", "—")
        early_stop = metrics.get("early_stop_reason") or "—"

        summary_lines = [
            ("Run ID", run_dir.name),
            ("Task / model", model_name),
            ("Learning rate", lr),
            ("Epochs", f"{epochs_trained} / {epochs_max}"),
            ("Final loss", f"{final_loss:.6f}" if isinstance(final_loss, float) else final_loss),
            ("Early stop reason", early_stop),
            ("Device", config.get("device", "—")),
            ("Batch size", config.get("batch_size", "—")),
        ]
        text_page(pdf, summary_lines, f"Training Run Report\n{run_dir.name}")

        # --- Page 2: loss curve ---
        if history:
            _plot_loss_curve(pdf, history)

        # --- Page 3: per-epoch metrics (optional) ---
        metric_keys = [k for k in _METRIC_COLORS if any(k in e for e in history)]
        if metric_keys and history:
            _plot_epoch_metrics(pdf, history, metric_keys)

        # --- Page 4: partition / config summary ---
        train_ids = datasource.get("train_ids", [])
        val_ids = datasource.get("val_ids", [])
        test_ids = datasource.get("test_ids", [])
        partition_lines = [
            ("Train patients", len(train_ids) if isinstance(train_ids, list) else "—"),
            ("Val patients", len(val_ids) if isinstance(val_ids, list) else "—"),
            ("Test patients", len(test_ids) if isinstance(test_ids, list) else "—"),
            ("Optimizer", str(config.get("optimizer", "—"))),
            ("Batch size", config.get("batch_size", "—")),
            ("Train samples", metrics.get("num_train_samples", "—")),
        ]
        text_page(pdf, partition_lines, "Partition & Config Summary")

    return out_path.resolve()


# ---------------------------------------------------------------------------
# Private plot helpers
# ---------------------------------------------------------------------------

def _plot_loss_curve(pdf, history: list[dict]) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["loss"] for h in history]
    val_loss = [h["val_loss"] for h in history if "val_loss" in h]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_loss, marker="o", linewidth=2,
            color="#2196F3", markersize=5, label="Train loss")
    if len(val_loss) == len(epochs):
        ax.plot(epochs, val_loss, marker="s", linewidth=2, linestyle="--",
                color="#FF9800", markersize=5, label="Val loss")
        ax.legend(fontsize=11)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_epoch_metrics(pdf, history: list[dict], metric_keys: list[str]) -> None:
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(9, 5))
    for key in metric_keys:
        vals = [h.get(key) for h in history]
        if any(v is not None for v in vals):
            ax.plot(epochs, vals, marker="o", linewidth=2, markersize=5,
                    label=key, color=_METRIC_COLORS.get(key))
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Epoch Metrics", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
