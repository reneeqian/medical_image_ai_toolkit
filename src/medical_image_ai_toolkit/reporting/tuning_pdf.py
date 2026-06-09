"""
Generate a multi-page PDF report for a hyperparameter tuning run.

Pages
-----
1. Title / summary (num_trials, best params, best loss)
2. Trial comparison horizontal bar chart
3. Per-trial params + losses table
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


def generate_tuning_pdf(
    runs_root: Path | str,
    run_id: str | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """
    Generate a PDF report for a hyperparameter tuning run.

    Parameters
    ----------
    runs_root:
        Directory containing timestamped tuning run subdirectories.
    run_id:
        Exact or prefix-matched run directory name.  Defaults to most recent.
    output_path:
        Where to write the PDF.  Defaults to ``{run_dir}/tuning_report.pdf``.

    Returns
    -------
    Path
        Absolute path of the written PDF file.
    """
    run_dir = find_run_dir(runs_root, run_id, requires_file="tuning_report.json")
    report_json = run_dir / "tuning_report.json"

    data = json.loads(report_json.read_text())
    num_trials = data.get("num_trials", 0)
    best = data.get("best_trial") or {}
    trials = data.get("trial_results", [])

    out_path = Path(output_path) if output_path else run_dir / "tuning_report.pdf"

    with open_pdf(out_path) as pdf:
        # --- Page 1: title / summary ---
        best_params = best.get("params", {})
        best_loss = best.get("best_val_loss") or best.get("final_loss")
        summary_lines = [
            ("Run ID", run_dir.name),
            ("Total trials", num_trials),
            ("Best trial ID", best.get("trial_id", "—")),
            ("Best val loss", f"{best_loss:.6f}" if isinstance(best_loss, float) else "—"),
            ("Best epochs", best.get("epochs_trained", "—")),
        ] + [(f"  {k}", v) for k, v in best_params.items()]
        text_page(pdf, summary_lines, f"Hyperparameter Tuning Report\n{run_dir.name}")

        # --- Page 2: trial comparison bar chart ---
        if trials:
            _plot_trial_comparison(pdf, trials, best.get("trial_id"))

        # --- Page 3: per-trial params table ---
        if trials:
            _plot_trial_table(pdf, trials)

    return out_path.resolve()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _metric_for_trial(t: dict) -> float | None:
    v = t.get("best_val_loss")
    if v is not None:
        return float(v)
    v = t.get("final_loss")
    return float(v) if v is not None else None


def _plot_trial_comparison(pdf, trials: list[dict], best_id) -> None:
    ids = [t["trial_id"] for t in trials]
    losses = [_metric_for_trial(t) or 0.0 for t in trials]

    colors = ["#FFD700" if t["trial_id"] == best_id else "#2196F3" for t in trials]
    y_pos = np.arange(len(ids))

    fig, ax = plt.subplots(figsize=(9, max(4, len(ids) * 0.8 + 1)))
    bars = ax.barh(y_pos, losses, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Trial {i}" for i in ids], fontsize=11)
    ax.set_xlabel("Loss (lower is better)", fontsize=12)
    ax.set_title("Trial Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for bar, loss in zip(bars, losses):
        ax.text(
            bar.get_width() + max(losses) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{loss:.5f}",
            va="center",
            fontsize=10,
        )
    # legend
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color="#FFD700", label="Best trial"),
            Patch(color="#2196F3", label="Other trials"),
        ],
        fontsize=10,
    )
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_trial_table(pdf, trials: list[dict]) -> None:
    lines = []
    for t in trials:
        params_str = "  ".join(f"{k}={v}" for k, v in t.get("params", {}).items())
        loss = _metric_for_trial(t)
        loss_str = f"{loss:.5f}" if loss is not None else "—"
        label = f"Trial {t['trial_id']}"
        value = f"{params_str}  |  loss={loss_str}  |  epochs={t.get('epochs_trained', '—')}"
        lines.append((label, value))
    text_page(pdf, lines, "All Trials — Parameters & Losses")
