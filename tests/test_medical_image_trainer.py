from pathlib import Path
import sys
import json
import hashlib
import torch
import pytest
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from medical_image_ai_toolkit.training.medical_image_trainer import MedicalImageTrainer
from medical_image_ai_toolkit.evidence.evidence_report import EvidenceReport


# -------------------------------
# Tensor-only test fixtures
# -------------------------------

def _dummy_tensor_batch(batch_size: int = 1):
    return {
        "image": torch.zeros(batch_size, 4),
        "target": torch.zeros(batch_size),
    }


def _nan_tensor_batch(batch_size: int = 1):
    return {
        "image": torch.zeros(batch_size, 4),
        "target": torch.full((batch_size,), float("nan")),
    }


class _TensorDataset(Dataset):
    def __init__(self, num_batches: int = 4, nan: bool = False):
        self.num_batches = num_batches
        self.nan = nan

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if self.nan:
            return _nan_tensor_batch(batch_size=1)
        return _dummy_tensor_batch(batch_size=1)

class SmallSliceCNN(nn.Module):
    """
    Minimal 2D CNN for slice-based binary classification.

    This model is intentionally small, transparent, and explainable.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(16, 1)

    def forward(self, x):
        # x: (B, 1, H, W)
        feats = self.features(x)
        feats = feats.view(x.size(0), -1)
        return self.classifier(feats)


# -------------------------------
# Helper Functions
# -------------------------------

def _hash_metrics_file(metrics_path: Path) -> str:
    payload = json.loads(metrics_path.read_text())
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# -------------------------------
# Test Cases
# -------------------------------
@pytest.mark.requirement("MIT-TR-05")
@pytest.mark.requirement("MIT-TR-08")
def test_MIT_TR_05_training_records_metrics_and_evidence(tmp_path):
    report = EvidenceReport(
        subject="MedicalImageTrainer metrics and evidence generation"
    )

    trainer = MedicalImageTrainer(
        output_dir=tmp_path,
        random_seed=123,
    )

    dataset = _TensorDataset(num_batches=2)

    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1,
    )

    metrics_file = tmp_path / "metrics.json"
    evidence_file = tmp_path / "evidence_report.json"

    if not metrics_file.exists():
        report.error(
            message="Metrics file not written",
            requirement_id="MIT-TR-05",
        )

    if not evidence_file.exists():
        report.error(
            message="Evidence report not written",
            requirement_id="MIT-TR-08",
        )

    report.auto_save("MIT_TR_05_training_records_metrics_and_evidence")
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("MIT-TR-09")
def test_MIT_TR_09_training_detects_nan_loss(tmp_path):
    report = EvidenceReport(
        subject="MedicalImageTrainer numerical instability detection"
    )

    trainer = MedicalImageTrainer(
        output_dir=tmp_path,
        random_seed=123,
    )

    dataset = _TensorDataset(num_batches=2, nan=True)

    train_loader = DataLoader(dataset, batch_size=1)
    val_loader = DataLoader(dataset, batch_size=1)

    try:
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,
        )
        report.error(
            message="Trainer did not fail on NaN loss",
            requirement_id="MIT-TR-09",
        )
    except RuntimeError as e:
        report.info(
            message="Trainer failed as expected on NaN loss",
            requirement_id="MIT-TR-09",
            context=str(e),
        )

    report.auto_save("MIT_TR_09_training_detects_nan_loss")
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("MIT-DR-09")
def test_MIT_DR_09_trainer_rejects_unvalidated_input(tmp_path):
    report = EvidenceReport(
        subject="MedicalImageTrainer validation boundary enforcement"
    )

    trainer = MedicalImageTrainer(output_dir=tmp_path)

    class _InvalidDataset(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return None  # invalid batch

    dataset = _InvalidDataset()

    train_loader = DataLoader(dataset, batch_size=1)
    val_loader = DataLoader(dataset, batch_size=1)

    try:
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,
        )
        report.error(
            message="Trainer accepted invalid input batch",
            requirement_id="MIT-DR-09",
        )
    except Exception as e:
        report.info(
            message="Trainer rejected invalid input batch",
            requirement_id="MIT-DR-09",
            context=str(e),
        )

    report.auto_save("MIT_DR_09_trainer_rejects_unvalidated_input")
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("MIT-TR-01")
def test_MIT_TR_01_training_is_deterministic(tmp_path):
    report = EvidenceReport(
        subject="MedicalImageTrainer deterministic training behavior"
    )

    seed = 1337
    dataset = _TensorDataset(num_batches=4)

    train_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    # --- Run 1 ---
    run1_dir = tmp_path / "run1"
    trainer1 = MedicalImageTrainer(
        output_dir=run1_dir,
        random_seed=seed,
    )

    trainer1.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2,
    )

    # --- Run 2 ---
    run2_dir = tmp_path / "run2"
    trainer2 = MedicalImageTrainer(
        output_dir=run2_dir,
        random_seed=seed,
    )

    trainer2.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2,
    )

    hash1 = _hash_metrics_file(run1_dir / "metrics.json")
    hash2 = _hash_metrics_file(run2_dir / "metrics.json")

    report.info(
        message="Computed metric hashes for deterministic comparison",
        requirement_id="MIT-TR-01",
        context=f"run1={hash1}, run2={hash2}",
    )

    if hash1 != hash2:
        report.error(
            message="Training metrics are not deterministic across runs",
            requirement_id="MIT-TR-01",
            context=f"{hash1} != {hash2}",
        )

    report.auto_save("MIT_TR_01_training_is_deterministic")
    assert not report.has_errors, report.summary()

