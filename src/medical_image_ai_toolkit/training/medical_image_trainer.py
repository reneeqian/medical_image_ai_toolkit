from __future__ import annotations

import json
import time
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from regulatory_tools.evidence.evidence_report import EvidenceReport


class MedicalImageTrainer:
    """
    Owns the full lifecycle of a single training run:
    - Training / validation loops
    - Progress reporting
    - Metrics collection
    - Artifact persistence

    Designed for FDA-auditable SaMD development.
    """

    def __init__(
        self,
        *,
        output_dir: Path | str,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        run_config: Optional[Dict[str, Any]] = None,
        data_splits: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
        show_training_plot: bool = False,
        save_training_plot: bool = True,
        print_every_n_batches: int = 50,
        slow_batch_threshold_sec: float = 30.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Deferred training dependencies ---
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.run_config = run_config
        self.data_splits = data_splits

        # --- Determinism ---
        if random_seed is not None:
            self._set_determinism(random_seed)

        # --- Runtime behavior ---
        self.show_training_plot = show_training_plot
        self.save_training_plot = save_training_plot
        self.print_every_n_batches = print_every_n_batches
        self.slow_batch_threshold_sec = slow_batch_threshold_sec

        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def train(
        self,
        *,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ) -> None:
        self._initialize_defaults_if_needed()
        self._validate_ready_for_training()

        self._save_static_artifacts()

        for epoch in range(num_epochs):
            epoch_record = self._run_epoch(
                epoch=epoch,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
            )
            self.history.append(epoch_record)

        self._save_dynamic_artifacts()
        self._save_evidence_report()
        self._maybe_plot_training_history()
        
        
    # ------------------------------------------------------------------
    # Helper Function
    # ------------------------------------------------------------------
        
    def _initialize_defaults_if_needed(self) -> None:
        """
        Provide safe, deterministic defaults for test and
        development use when components are not supplied.
        """

        if self.device is None:
            self.device = torch.device("cpu")

        if self.model is None:
            # Minimal deterministic model
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(4, 1),
            ).to(self.device)


        if self.loss_fn is None:
            self.loss_fn = nn.MSELoss()

        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=0.01,
            )

        if self.run_config is None:
            self.run_config = {
                "run_intent": "default_test_run",
                "trainer": "MedicalImageTrainer",
            }

        if self.data_splits is None:
            self.data_splits = {
                "train": "in-memory",
                "val": "in-memory",
            }


    # ------------------------------------------------------------------
    # VALIDATION (FDA-CRITICAL)
    # ------------------------------------------------------------------

    def _validate_ready_for_training(self) -> None:
        missing = [
            name
            for name, value in {
                "model": self.model,
                "optimizer": self.optimizer,
                "loss_fn": self.loss_fn,
                "device": self.device,
                "run_config": self.run_config,
                "data_splits": self.data_splits,
            }.items()
            if value is None
        ]

        if missing:
            raise ValueError(
                "MedicalImageTrainer is not fully configured. "
                f"Missing required fields: {', '.join(missing)}"
            )

    # ------------------------------------------------------------------
    # DETERMINISM
    # ------------------------------------------------------------------

    def _set_determinism(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ------------------------------------------------------------------
    # EPOCH LOGIC
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        *,
        epoch: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ) -> Dict[str, Any]:
        epoch_start = time.time()

        print(f"\n🟢 Epoch {epoch + 1}/{num_epochs} — training")
        train_loss = self._train_one_epoch(train_loader, epoch_start)

        print(f"🔵 Epoch {epoch + 1}/{num_epochs} — validation")
        val_loss = self._validate(val_loader)

        epoch_time = time.time() - epoch_start

        print(
            f"✅ Epoch {epoch + 1}/{num_epochs} complete | "
            f"train={train_loss:.4f} | "
            f"val={val_loss:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": {
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
        }

        torch.save(
            checkpoint,
            self.output_dir / f"checkpoint_epoch_{epoch}.pt",
        )

        evidence = EvidenceReport(subject=f"Epoch {epoch}")
        evidence.info(
            f"Training loss: {train_loss:.4f}",
            requirement_id="TSR-001",
        )
        evidence.info(
            f"Validation loss: {val_loss:.4f}",
            requirement_id="TSR-001",
        )
        evidence.save(self.output_dir / f"evidence_epoch_{epoch}.json")

        return {
            "epoch": epoch,
            "train": {"loss": train_loss},
            "val": {"loss": val_loss},
        }

    def _train_one_epoch(self, loader: DataLoader, epoch_start: float) -> float:
        self.model.train()
        total_loss = 0.0
        last_print = epoch_start

        for batch_idx, batch in enumerate(loader):
            batch_start = time.time()

            try:
                images = batch["image"].to(self.device)
                targets = batch["target"].to(self.device).view(-1, 1)

                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.loss_fn(logits, targets)

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss detected "
                        f"(epoch={len(self.history)}, batch={batch_idx})"
                    )

                loss.backward()
                self.optimizer.step()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️ CUDA OOM at batch {batch_idx} — skipping batch")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

            total_loss += loss.item()

            batch_time = time.time() - batch_start
            if batch_time > self.slow_batch_threshold_sec:
                print(
                    f"⚠️ Slow batch detected: "
                    f"{batch_time:.1f}s "
                    f"(epoch={len(self.history)}, batch={batch_idx})"
                )

            if (
                self.print_every_n_batches > 0
                and batch_idx % self.print_every_n_batches == 0
            ):
                now = time.time()
                print(
                    f"  ⏱ batch {batch_idx:5d}/{len(loader)} | "
                    f"loss={loss.item():.4f} | "
                    f"+{now - last_print:.1f}s"
                )
                last_print = now

        return total_loss / len(loader)

    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                targets = batch["target"].to(self.device).view(-1, 1)
                logits = self.model(images)
                loss = self.loss_fn(logits, targets)

                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite validation loss detected")

                total_loss += loss.item()

        return total_loss / len(loader)

    # ------------------------------------------------------------------
    # ARTIFACTS
    # ------------------------------------------------------------------

    def _save_static_artifacts(self) -> None:
        with open(self.output_dir / "run_config.json", "w") as f:
            json.dump(self.run_config, f, indent=2, sort_keys=True)

        with open(self.output_dir / "data_splits.json", "w") as f:
            json.dump(self.data_splits, f, indent=2, sort_keys=True)


    def _save_dynamic_artifacts(self) -> None:
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(
                self.history,
                f,
                indent=2,
                sort_keys=True,  
            )

        torch.save(self.model.state_dict(), self.output_dir / "model.pt")

    def _save_evidence_report(self) -> None:
        evidence = EvidenceReport(
            subject=self.run_config.get("run_intent", "Training run")
        )

        evidence.info(
            "Trainer implementation: MedicalImageTrainer",
            requirement_id="TSR-001",
        )
        evidence.info(
            "Run executed as non-clinical development activity",
            requirement_id="TSR-003",
        )
        evidence.info(
            "No clinical performance claims implied by this run",
            requirement_id="TSR-003",
        )
        evidence.info(
            f"Epochs completed: {len(self.history)}",
            requirement_id="TSR-001",
        )

        evidence.save(self.output_dir / "evidence_report.json")

    def _maybe_plot_training_history(self) -> None:
        if not (self.show_training_plot or self.save_training_plot):
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️ matplotlib not available — skipping plot")
            return

        epochs = [h["epoch"] + 1 for h in self.history]
        train_losses = [h["train"]["loss"] for h in self.history]
        val_losses = [h["val"]["loss"] for h in self.history]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Train")
        plt.plot(epochs, val_losses, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)

        if self.save_training_plot:
            path = self.output_dir / "training_curve.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"📈 Training curve saved to {path}")

        if self.show_training_plot:
            plt.show()
        else:
            plt.close()
