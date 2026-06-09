from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def open_pdf(path: Path) -> PdfPages:
    """Open a PdfPages context at *path*, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return PdfPages(str(path))


def text_page(pdf: PdfPages, lines: list[tuple[str, str]], title: str) -> None:
    """
    Append a plain key-value summary page to *pdf*.

    Parameters
    ----------
    pdf:
        An open ``PdfPages`` object.
    lines:
        List of ``(label, value)`` tuples rendered as a two-column table.
    title:
        Page title rendered at the top.
    """
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    y = 0.88
    for label, value in lines:
        ax.text(
            0.05, y, f"{label}:", fontsize=11, fontweight="bold", transform=ax.transAxes, va="top"
        )
        ax.text(0.42, y, str(value), fontsize=11, transform=ax.transAxes, va="top")
        y -= 0.07

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
