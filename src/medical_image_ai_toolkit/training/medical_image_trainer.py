from __future__ import annotations

import json
import random
from datetime import datetime
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch

from regulatory_tools.evidence.evidence_report import EvidenceReport
from medical_image_ai_toolkit.results.medical_image_training_results import MedicalImageTrainingResults

class MedicalImageTrainer:
    """
    Bare-bones training orchestration class.
    """

    def __init__(
        self,
        datasource,
        model,
        training_config,
        output_dir=None,
        random_seed=42
    ):
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

    def train(self):
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

        print("Starting training")
        
        self.model.train()

        epoch_losses = []
        for epoch in range(self.config.epochs):

            print(f"\nEpoch {epoch+1}")
            running_loss = 0.0
            n_samples = 0
            
            task = self.config.task

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

            results.history.append({
                "epoch": epoch + 1,
                "loss": epoch_loss
            })
            
            print("epoch complete")

        results.metrics = {
            "final_loss": epoch_losses[-1],
            "num_epochs": self.config.epochs,
            "num_train_samples": len(train_ids)
        }

        metrics_path = results.run_dir / "metrics.json"

        with open(metrics_path, "w") as f:
            json.dump(results.metrics, f, indent=2)

        results.artifacts["metrics"] = metrics_path

        results.mark_training_complete()

        return results
    
    def sanity_check(self):

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
            import torch

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

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
