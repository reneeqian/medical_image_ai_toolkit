from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch.nn as nn
from regulatory_tools.evidence.evidence_report import EvidenceReport

if TYPE_CHECKING:
    from medical_image_ai_toolkit.results.medical_image_training_results import (
        MedicalImageTrainingResults,
    )
    from medical_image_ai_toolkit.training.training_config import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    trial_id: int
    params: dict[str, Any]
    final_loss: float
    best_val_loss: float | None
    epochs_trained: int
    early_stop_reason: str | None
    run_dir: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "params": self.params,
            "final_loss": self.final_loss,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": self.epochs_trained,
            "early_stop_reason": self.early_stop_reason,
            "run_dir": str(self.run_dir),
        }


def _trial_result_from_training_results(
    trial_id: int,
    params: dict[str, Any],
    training_results: MedicalImageTrainingResults,
) -> TrialResult:
    val_losses = [
        entry["val_loss"]
        for entry in training_results.history
        if "val_loss" in entry
    ]
    return TrialResult(
        trial_id=trial_id,
        params=params,
        final_loss=training_results.metrics["final_loss"],
        best_val_loss=min(val_losses) if val_losses else None,
        epochs_trained=training_results.metrics["epochs_trained"],
        early_stop_reason=training_results.metrics.get("early_stop_reason"),
        run_dir=training_results.run_dir,
    )


class HyperparameterTuningResults:
    """
    Container for the outcomes of a hyperparameter tuning run.

    Attributes
    ----------
    trial_results : list of TrialResult
        One entry per trial that completed.
    best_trial : TrialResult or None
        The trial with the lowest validation loss (or training loss when
        no validation partition was present).
    best_model : nn.Module or None
        The model instance from the best trial, in its trained state.
    best_config : TrainingConfig or None
        The configuration used in the best trial.
    artifacts : dict
        Paths to files written by this tuning run (e.g. ``tuning_report``).
    output_dir : Path
        Directory where tuning run artifacts are written.
    """

    def __init__(
        self,
        trial_results: list[TrialResult],
        best_trial: TrialResult | None,
        best_model: nn.Module | None,
        best_config: TrainingConfig | None,
        artifacts: dict[str, Path],
        output_dir: Path,
    ) -> None:
        self.trial_results = trial_results
        self.best_trial = best_trial
        self.best_model = best_model
        self.best_config = best_config
        self.artifacts = artifacts
        self.output_dir = Path(output_dir)
        self.evidence_report: EvidenceReport | None = None

    def generate_tuning_report(self) -> Path:
        report = {
            "num_trials": len(self.trial_results),
            "best_trial": self.best_trial.to_dict() if self.best_trial else None,
            "trial_results": [tr.to_dict() for tr in self.trial_results],
        }
        path = self.output_dir / "tuning_report.json"
        try:
            with open(path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info("Tuning report saved to %s", path)
        except OSError:
            logger.error("Failed to write tuning report to %s", path)
        return path

    def summary(self) -> None:
        print("\n==============================")
        print("Hyperparameter Tuning Summary")
        print("==============================")
        print(f"Trials run: {len(self.trial_results)}")
        if not self.trial_results:
            print("No trials completed.")
            return

        header = f"{'ID':>3}  {'params':<40}  {'final_loss':>12}  {'best_val_loss':>14}  {'epochs':>6}"
        print(header)
        print("-" * len(header))
        for tr in self.trial_results:
            params_str = ", ".join(f"{k}={v}" for k, v in tr.params.items())
            val_str = f"{tr.best_val_loss:.6f}" if tr.best_val_loss is not None else "     —"
            marker = " *" if self.best_trial and tr.trial_id == self.best_trial.trial_id else ""
            print(
                f"{tr.trial_id:>3}  {params_str:<40}  {tr.final_loss:>12.6f}"
                f"  {val_str:>14}  {tr.epochs_trained:>6}{marker}"
            )

        if self.best_trial:
            print(f"\nBest trial: {self.best_trial.trial_id}  params={self.best_trial.params}")
            metric = (
                f"best_val_loss={self.best_trial.best_val_loss:.6f}"
                if self.best_trial.best_val_loss is not None
                else f"final_loss={self.best_trial.final_loss:.6f}"
            )
            print(f"  {metric}")
