from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from regulatory_tools.evidence.evidence_report import EvidenceReport

from medical_image_ai_toolkit.results.medical_image_training_results import (
    MedicalImageTrainingResults,
)

if TYPE_CHECKING:
    from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import (
        MedicalImageDataSource,
    )
    from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition
    from medical_image_ai_toolkit.training.training_config import TrainingConfig

logger = logging.getLogger(__name__)


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
        resume_from: str | Path | None = None,
    ) -> None:
        self.datasource = datasource
        self.model = model
        self.config = training_config
        self.resume_from = Path(resume_from) if resume_from is not None else None

        self.output_dir = Path(output_dir or "artifacts/training_runs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if random_seed is not None:
            self._set_seed(random_seed)

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def train(self) -> MedicalImageTrainingResults:
        """
        Run the training loop and return a populated results object.

        Iterates over training patients for ``config.epochs`` epochs,
        computing per-sample loss and updating model weights.  A
        validation pass (Dice, recall, precision, val loss) is run after
        each epoch when a val partition exists.  Early stopping is
        applied when ``config.early_stop`` is True.

        Returns
        -------
        MedicalImageTrainingResults
            Contains epoch history, final metrics, and references to
            exported artefact paths.

        Raises
        ------
        RuntimeError
            If partitions have not been created on the datasource, or if
            a NaN loss is detected during training.
        ValueError
            If ``config.task`` is None.
        """
        if not self.datasource.has_partitions():
            raise RuntimeError("Datasource partitions not created")
        if self.config.task is None:
            raise ValueError("TrainingConfig.task must be set")

        evidence = EvidenceReport(subject=f"Training run: {self.model.__class__.__name__}")

        train_ids = self.datasource.get_train_ids()

        device = self.config.device

        self.model.to(device)

        optimizer = self.config.optimizer(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        results = MedicalImageTrainingResults(
            self.model,
            self.config,
            self.datasource,
            run_dir
        )
        results.training_start_time = datetime.now()

        early_stop = self.config.early_stop

        start_epoch, epoch_losses, epochs_without_improvement = self._load_checkpoint(
            optimizer, device, results
        )

        logger.info("Starting training")
        if early_stop:
            logger.info(
                "Early stopping ON  |  loss_threshold=%s  |  plateau_patience=%s"
                "  |  plateau_min_delta=%s",
                self.config.loss_threshold, self.config.plateau_patience,
                self.config.plateau_min_delta,
            )

        val_ids = self.datasource.get_val_ids() if self.datasource.has_partitions() else []
        has_val = len(val_ids) > 0

        evidence.info(
            f"model={self.model.__class__.__name__}  device={device}"
            f"  lr={self.config.learning_rate}  optimizer={self.config.optimizer.__name__}"
            f"  epochs={self.config.epochs}  early_stop={self.config.early_stop}"
            f"  checkpoint_every={self.config.checkpoint_every}  resume_from={self.resume_from}",
            "TRN-001",
        )
        evidence.info(
            f"train_patients={len(train_ids)}  val_patients={len(val_ids)}",
            "TRN-008",
        )

        stop_reason = None
        _epoch_start = time.time()

        task = self.config.task

        for epoch in range(start_epoch, self.config.epochs):

            logger.info("Epoch %d/%d", epoch + 1, self.config.epochs)

            epoch_loss = self._run_train_pass(train_ids, task, optimizer, device)
            epoch_losses.append(epoch_loss)

            # --- Validation pass ---
            val_metrics = self._run_val_pass(val_ids, task, device) if has_val else {}

            history_entry = {"epoch": epoch + 1, "loss": epoch_loss, **val_metrics}
            results.history.append(history_entry)

            elapsed = time.time() - _epoch_start
            _epoch_start = time.time()
            self._log_epoch(epoch, epoch_loss, val_metrics, elapsed, evidence)

            self._maybe_save_checkpoint(epoch, optimizer, epoch_losses, epochs_without_improvement, results.history, run_dir)

            if early_stop:
                stop_reason, epochs_without_improvement = self._check_early_stopping(
                    epoch_losses, epochs_without_improvement
                )
                if stop_reason:
                    break

        return self._finalize_training(results, epoch_losses, train_ids, stop_reason, evidence)

    def sanity_check(self) -> None:
        """
        Print a diagnostic summary of the datasource, config, and model.

        Intended to be called before ``train()`` so the user can verify
        partition sizes, hyperparameters, and model architecture at a glance.
        Output goes to stdout because this is an interactive display method.
        """

        print("\n==============================")
        print("MedicalImageTrainer Sanity Check")
        print("==============================")

        # ------------------------------
        # Datasource
        # ------------------------------

        print("\nDatasource")

        try:
            total = self.datasource.get_num_patients()
        except AttributeError:
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

        except (StopIteration, RuntimeError):
            logger.warning("Could not inspect model parameters.")
            print("Could not inspect model parameters.")

        print("\nSanity check complete.")
        print("==============================\n")

    # ---------------------------------------------------------
    # Internal methods
    # ---------------------------------------------------------

    def _log_epoch(
        self,
        epoch: int,
        epoch_loss: float,
        val_metrics: dict,
        elapsed: float,
        evidence: EvidenceReport,
    ) -> None:
        if val_metrics:
            logger.info("  train loss=%.6f  |  val loss=%.6f", epoch_loss, val_metrics["val_loss"])
            print(
                f"  epoch {epoch + 1}/{self.config.epochs}"
                f"  loss={epoch_loss:.6f}"
                f"  val_loss={val_metrics['val_loss']:.6f}"
                f"  ({elapsed:.1f}s)"
            )
            evidence.info(
                f"epoch={epoch + 1}  loss={epoch_loss:.6f}  val_loss={val_metrics['val_loss']:.6f}",
                "TRN-004",
            )
        else:
            logger.info("  train loss=%.6f", epoch_loss)
            print(
                f"  epoch {epoch + 1}/{self.config.epochs}"
                f"  loss={epoch_loss:.6f}"
                f"  ({elapsed:.1f}s)"
            )
            evidence.info(f"epoch={epoch + 1}  loss={epoch_loss:.6f}", "TRN-004")

    def _finalize_training(
        self,
        results: MedicalImageTrainingResults,
        epoch_losses: list[float],
        train_ids: list[str],
        stop_reason: str | None,
        evidence: EvidenceReport,
    ) -> MedicalImageTrainingResults:
        epochs_trained = len(epoch_losses)
        if stop_reason:
            logger.info("Early stop after epoch %d: %s", epochs_trained, stop_reason)
            print(f"  [early stop] {stop_reason}")

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

        evidence.info(
            f"final_loss={epoch_losses[-1]:.6f}  epochs_trained={epochs_trained}"
            f"  num_train_samples={len(train_ids)}",
            "VER-002",
        )
        if stop_reason:
            evidence.warn(f"Early stop triggered: {stop_reason}", "TRN-001")

        evidence_path = results.run_dir / "training_evidence.json"
        evidence.save(evidence_path)
        results.artifacts["evidence"] = evidence_path
        results.evidence_report = evidence

        results.mark_training_complete()
        return results

    def _load_checkpoint(
        self,
        optimizer: torch.optim.Optimizer,
        device: str,
        results: MedicalImageTrainingResults,
    ) -> tuple[int, list[float], int]:
        if self.resume_from is None:
            return 0, [], 0
        if not self.resume_from.exists():
            logger.warning("resume_from path not found, starting fresh: %s", self.resume_from)
            print(f"  [checkpoint] WARNING: {self.resume_from} not found — starting fresh")
            return 0, [], 0
        print(f"  [checkpoint] loading {self.resume_from}")
        ckpt = torch.load(self.resume_from, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        epoch_losses = ckpt["epoch_losses"]
        epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
        results.history = ckpt.get("history", [])
        print(
            f"  [checkpoint] resumed from epoch {ckpt['epoch'] + 1}"
            f"  → continuing from epoch {start_epoch + 1}/{self.config.epochs}"
        )
        logger.info("Resumed from checkpoint %s (epoch %d)", self.resume_from, ckpt["epoch"])
        return start_epoch, epoch_losses, epochs_without_improvement

    def _run_train_pass(
        self,
        train_ids: list[str],
        task: TrainingTaskDefinition,
        optimizer: torch.optim.Optimizer,
        device: str,
    ) -> float:
        self.model.train()
        running_loss = 0.0
        n_samples = 0
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
        return running_loss / max(n_samples, 1)

    def _run_val_pass(
        self,
        val_ids: list[str],
        task: TrainingTaskDefinition,
        device: str,
    ) -> dict:
        val_evaluator = self.config.val_evaluator
        if val_evaluator is not None:
            val_evaluator.reset()
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
                    if val_evaluator is not None:
                        val_evaluator.update(pred, y)
        self.model.train()

        val_loss = val_running_loss / max(val_n_samples, 1)
        metrics: dict = {"val_loss": val_loss}
        if val_evaluator is not None:
            metrics.update(val_evaluator.aggregate())
        return metrics

    def _maybe_save_checkpoint(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        epoch_losses: list[float],
        epochs_without_improvement: int,
        history: list[dict],
        run_dir: Path,
    ) -> None:
        every = self.config.checkpoint_every
        if every > 0 and (epoch + 1) % every == 0:
            self._save_checkpoint(epoch, optimizer, epoch_losses, epochs_without_improvement, history, run_dir)

    def _save_checkpoint(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        epoch_losses: list[float],
        epochs_without_improvement: int,
        history: list[dict],
        run_dir: Path,
    ) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch_losses": epoch_losses,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
        }
        ckpt_path = run_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
        torch.save(ckpt, ckpt_path)
        torch.save(ckpt, run_dir / "checkpoint_latest.pt")
        print(f"  [checkpoint] saved → {ckpt_path.name}")
        logger.info("Checkpoint saved: %s", ckpt_path)

    def _check_early_stopping(
        self,
        epoch_losses: list[float],
        epochs_without_improvement: int,
    ) -> tuple[str | None, int]:
        """Returns (stop_reason | None, updated epochs_without_improvement)."""
        epoch_loss = epoch_losses[-1]

        if epoch_loss <= self.config.loss_threshold:
            return (
                f"loss_threshold reached ({epoch_loss:.6f} <= {self.config.loss_threshold})",
                epochs_without_improvement,
            )

        if len(epoch_losses) >= 2:
            improvement = epoch_losses[-2] - epoch_loss
            if improvement < self.config.plateau_min_delta:
                epochs_without_improvement += 1
            else:
                epochs_without_improvement = 0

        if epochs_without_improvement >= self.config.plateau_patience:
            return (
                f"plateau: no improvement > {self.config.plateau_min_delta}"
                f" for {self.config.plateau_patience} consecutive epochs",
                epochs_without_improvement,
            )

        return None, epochs_without_improvement

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
