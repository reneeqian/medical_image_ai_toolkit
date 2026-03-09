from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import hashlib
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from regulatory_tools.evidence.evidence_report import EvidenceReport
from medical_image_ai_toolkit.results.medical_image_training_results import MedicalImageTrainingResults


class MedicalImageTrainer:
    """
    Bare-bones training orchestration class.
    """

    def __init__(self, datasource, model, config):

        self.datasource = datasource
        self.model = model
        self.config = config
    
    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def train(self):
        
        results = MedicalImageTrainingResults(
            self.model,
            self.config,
            self.datasource
        )

        print("Starting training...")

        if not self.datasource.has_partitions():
            raise RuntimeError(
                "Datasource partitions not found. "
                "Call datasource.create_partitions() first."
            )

        train_ids = self.datasource.get_train_ids()
        val_ids = self.datasource.get_val_ids()

        print("Starting training...")
        print(f"Train patients: {len(train_ids)}")
        print(f"Val patients: {len(val_ids)}")

        for epoch in range(self.config.epochs):

            print(f"\nEpoch {epoch+1}/{self.config.epochs}")

            self._train_epoch(train_ids)

            self._validate_epoch(val_ids)

        print("Training complete.")

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

    def _train_epoch(self, train_ids):

        for patient_id in train_ids:

            sample = self.datasource.get_patient(patient_id)

            volume = sample.image_volume

            # placeholder
            # model forward + loss + optimizer step
            pass

    def _validate_epoch(self, val_ids):

        for patient_id in val_ids:

            sample = self.datasource.get_patient(patient_id)

            volume = sample.image_volume

            # placeholder validation
            pass

    