from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from regulatory_tools.evidence.evidence_report import EvidenceReport

    from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import (
        MedicalImageDataSource,
    )
    from medical_image_ai_toolkit.training.training_config import TrainingConfig

logger = logging.getLogger(__name__)


class MedicalImageTrainingResults:
    """
    Minimal container for outputs of a training run.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        datasource: MedicalImageDataSource,
        run_dir: Path | str,
    ) -> None:

        self.model = model
        self.config = config
        self.datasource = datasource

        self.run_dir = Path(run_dir)

        self.history: list[dict] = []
        self.metrics: dict[str, Any] = {}
        self.artifacts: dict[str, Path] = {}
        self.evidence_report: EvidenceReport | None = None

        self.training_start_time: datetime | None = None
        self.training_end_time: datetime | None = None

    # --------------------------------------------------
    # Training lifecycle
    # --------------------------------------------------

    def mark_training_complete(self) -> None:
        """Record the training end timestamp."""
        self.training_end_time = datetime.now()

    # --------------------------------------------------
    # Reporting
    # --------------------------------------------------

    def summary(self) -> None:
        """Print a concise terminal summary of training metrics."""

        print("\n==============================")
        print("Training Results Summary")
        print("==============================")

        print(f"Model: {self.model.__class__.__name__}")

        if self.training_end_time is not None and self.training_start_time is not None:

            duration = self.training_end_time - self.training_start_time
            print(f"Training time: {duration}")

        else:

            print("Training still running")

        if self.metrics:

            print("\nMetrics")

            for k, v in self.metrics.items():
                print(f"{k}: {v}")

        print("==============================\n")

    def generate_training_report(self) -> Path:
        """
        Write a JSON training report to ``run_dir`` and return its path.

        The report includes final metrics, the full epoch history, and a
        string representation of the training config.
        """

        report_path = self.run_dir / "training_report.json"

        report = {
            "metrics": self.metrics,
            "datasource": vars(self.datasource),
            "config": vars(self.config),
            "history": self.history
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("Training report written to: %s", report_path)
        return report_path

    # --------------------------------------------------
    # Model export
    # --------------------------------------------------

    def export_model(self, path: str | Path) -> None:
        """
        Save the model state dict to ``path``.

        Logs an error and continues (does not raise) if the save fails,
        so the rest of the pipeline artefacts are still written.
        """

        try:
            torch.save(self.model.state_dict(), path)
            logger.info("Model exported to: %s", path)

        except Exception as e:
            logger.error("Failed to export model to %s: %s", path, e)

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------

    def run_inference(self, data: list[np.ndarray]) -> list[torch.Tensor]:
        """
        Run batch inference on a list of raw numpy arrays.

        Each array is converted to a float tensor with a leading batch
        dimension before being passed to the model.

        Parameters
        ----------
        data : list[np.ndarray]
            Input arrays, one per sample.

        Returns
        -------
        list[torch.Tensor]
            Model outputs, one tensor per input sample.
        """

        self.model.eval()

        with torch.no_grad():

            preds = []

            for sample in data:
                x = torch.tensor(sample).float().unsqueeze(0)
                pred = self.model(x)
                preds.append(pred)

        return preds
