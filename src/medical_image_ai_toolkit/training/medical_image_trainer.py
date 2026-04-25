from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from medical_image_ai_toolkit.results.medical_image_training_results import (
    MedicalImageTrainingResults,
)
from medical_image_ai_toolkit.validation.segmentation_evaluator import SegmentationEvaluator

if TYPE_CHECKING:
    from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import (
        MedicalImageDataSource,
    )
    from medical_image_ai_toolkit.training.training_config import TrainingConfig


class MedicalImageTrainer:
    """
    Training orchestration class.

    Early stopping
    --------------
    Two independent stopping conditions can be enabled together or separately:

    loss_threshold : float, optional
        Stop as soon as epoch loss drops at or below this value.
        Useful when you know the target loss for a task (e.g. 0.05).
        Controlled by ``early_stop`` flag.

    plateau_patience : int
        Stop when epoch loss has not improved by more than
        ``plateau_min_delta`` for this many consecutive epochs.
        Controlled by ``early_stop`` flag.

    Both conditions are checked at the end of each epoch.  Either one
    triggers an early exit.  Set ``early_stop=False`` to disable both
    and always train for the full ``config.epochs``.
    """

    def __init__(
        self,
        datasource: MedicalImageDataSource,
        model: nn.Module,
        training_config: TrainingConfig,
        output_dir: str | Path | None = None,
        random_seed: int | None = 42,
    ) -> None:
        self.datasource = datasource
        self.model = model
        self.config = training_config

        self.output_dir = Path(output_dir or "artifacts/training_runs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if random_seed is not None:
            self._set_seed(random_seed)

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def train(self) -> MedicalImageTrainingResults:
        if not self.datasource.has_partitions():
            raise RuntimeError("Datasource partitions not created")
        if self.config.task is None:
            raise ValueError("TrainingConfig.task must be set")

        train_ids = self.datasource.get_train_ids()

        device = self.config.device

        self.model.to(device)

        optimizer = self.config.optimizer(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        results = MedicalImageTrainingResults(
            self.model,
            self.config,
            self.datasource,
            run_dir
        )
        results.training_start_time = datetime.now()

        early_stop       = self.config.early_stop
        loss_threshold   = self.config.loss_threshold
        plateau_patience = self.config.plateau_patience
        plateau_min_delta = self.config.plateau_min_delta

        print("Starting training")
        if early_stop:
            print(
                f"Early stopping ON  |  loss_threshold={loss_threshold}"
                f"  |  plateau_patience={plateau_patience}"
                f"  |  plateau_min_delta={plateau_min_delta}"
            )

        val_ids = self.datasource.get_val_ids() if self.datasource.has_partitions() else []
        has_val = len(val_ids) > 0

        epoch_losses = []
        epochs_without_improvement = 0
        stop_reason = None

        for epoch in range(self.config.epochs):

            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            running_loss = 0.0
            n_samples = 0

            task = self.config.task

            self.model.train()

            for patient_id in train_ids:

                patient_sample = self.datasource.get_patient(patient_id)

                for sample in task.generate_training_samples(patient_sample):

                    x = sample["input"].to(device)
                    y = sample["target"]

                    if isinstance(y, torch.Tensor):
                        y = y.to(device)

                    optimizer.zero_grad(set_to_none=True)

                    pred = self.model(x)

                    loss = task.compute_loss(pred, y)

                    if not torch.isfinite(loss):
                        raise RuntimeError("NaN loss detected")

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    optimizer.step()

                    running_loss += loss.item()
                    n_samples += 1

            epoch_loss = running_loss / max(n_samples, 1)
            epoch_losses.append(epoch_loss)

            # --- Validation pass ---
            val_metrics = {}
            if has_val:
                val_evaluator = SegmentationEvaluator()
                val_running_loss = 0.0
                val_n_samples = 0

                self.model.eval()
                with torch.no_grad():
                    for patient_id in val_ids:
                        patient_sample = self.datasource.get_patient(patient_id)
                        for sample in task.generate_training_samples(patient_sample):
                            x = sample["input"].to(device)
                            y = sample["target"]
                            if isinstance(y, torch.Tensor):
                                y = y.to(device)
                            pred = self.model(x)
                            loss = task.compute_loss(pred, y)
                            val_running_loss += loss.item()
                            val_n_samples += 1
                            val_evaluator.update(pred, y)
                self.model.train()

                val_loss = val_running_loss / max(val_n_samples, 1)
                seg = val_evaluator.aggregate()
                val_metrics = {
                    "val_loss": val_loss,
                    "val_dice": seg["dice"],
                    "val_recall": seg["recall"],
                    "val_precision": seg["precision"],
                }

            history_entry = {"epoch": epoch + 1, "loss": epoch_loss, **val_metrics}
            results.history.append(history_entry)

            if has_val:
                vm = val_metrics
                print(
                    f"  train loss={epoch_loss:.6f}"
                    f"  |  val loss={vm['val_loss']:.6f}"
                    f"  dice={vm['val_dice']:.4f}"
                    f"  recall={vm['val_recall']:.4f}"
                    f"  prec={vm['val_precision']:.4f}"
                )
            else:
                print(f"  train loss={epoch_loss:.6f}")

            if early_stop:
                if epoch_loss <= loss_threshold:
                    stop_reason = f"loss_threshold reached ({epoch_loss:.6f} <= {loss_threshold})"
                    break

                if len(epoch_losses) >= 2:
                    improvement = epoch_losses[-2] - epoch_loss
                    if improvement < plateau_min_delta:
                        epochs_without_improvement += 1
                    else:
                        epochs_without_improvement = 0

                if epochs_without_improvement >= plateau_patience:
                    stop_reason = (
                        f"plateau: no improvement > {plateau_min_delta}"
                        f" for {plateau_patience} consecutive epochs"
                    )
                    break

        epochs_trained = len(epoch_losses)
        if stop_reason:
            print(f"\nEarly stop after epoch {epochs_trained}: {stop_reason}")

        results.metrics = {
            "final_loss": epoch_losses[-1],
            "epochs_trained": epochs_trained,
            "num_epochs": self.config.epochs,
            "num_train_samples": len(train_ids),
            "early_stop_reason": stop_reason,
        }

        metrics_path = results.run_dir / "metrics.json"

        with open(metrics_path, "w") as f:
            json.dump(results.metrics, f, indent=2)

        results.artifacts["metrics"] = metrics_path

        results.mark_training_complete()

        return results

    def sanity_check(self) -> None:

        print("\n==============================")
        print("MedicalImageTrainer Sanity Check")
        print("==============================")

        # ------------------------------
        # Datasource
        # ------------------------------

        print("\nDatasource")

        try:
            total = self.datasource.get_num_patients()
        except Exception:
            total = len(self.datasource)

        print(f"Total patients: {total}")

        if self.datasource.has_partitions():

            train_n = len(self.datasource.get_train_ids())
            val_n = len(self.datasource.get_val_ids())
            test_n = len(self.datasource.get_test_ids())

            print(f"Train patients: {train_n}")
            print(f"Val patients:   {val_n}")
            print(f"Test patients:  {test_n}")

        else:

            print("Partitions: NOT CREATED")

        # ------------------------------
        # Training Config
        # ------------------------------

        print("\nTraining Configuration")

        for key, value in vars(self.config).items():
            print(f"{key}: {value}")

        # ------------------------------
        # Model
        # ------------------------------

        print("\nModel")

        print(self.model.__class__.__name__)

        try:

            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

            print(f"Total parameters: {total_params}")
            print(f"Trainable parameters: {trainable_params}")

            device = next(self.model.parameters()).device
            print(f"Model device: {device}")

        except Exception:

            print("Could not inspect model parameters.")

        print("\nSanity check complete.")
        print("==============================\n")

    # ---------------------------------------------------------
    # Internal methods
    # ---------------------------------------------------------

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
