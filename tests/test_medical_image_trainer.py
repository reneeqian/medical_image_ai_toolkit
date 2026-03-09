from pathlib import Path
import json
import hashlib
import pytest
from torch.utils.data import DataLoader
from tests.conftest import TensorDataset


from medical_image_ai_toolkit.training.medical_image_trainer import MedicalImageTrainer
from regulatory_tools.evidence.evidence_report import EvidenceReport


# -------------------------------
# Helper
# -------------------------------

def _hash_metrics_file(metrics_path: Path) -> str:
    payload = json.loads(metrics_path.read_text())
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# -------------------------------
# Tests
# -------------------------------

@pytest.mark.requirement("VER-002")
def test_training_records_metrics_and_evidence(
    tmp_path,
    evidence_output_dir,
):
    report = EvidenceReport(
        subject="MedicalImageTrainer metrics and evidence generation"
    )

    trainer = MedicalImageTrainer(
        output_dir=tmp_path,
        random_seed=123,
    )

    dataset = TensorDataset(num_batches=2)

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
            requirement_id="VER-002",
        )

    if not evidence_file.exists():
        report.error(
            message="Evidence report not written",
            requirement_id="VER-003",
        )

    report.auto_save(
        "VER_002_training_records_metrics_and_evidence",
        evidence_output_dir,
    )

    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-003")
def test_training_detects_nan_loss(
    tmp_path,
    evidence_output_dir,
):
    report = EvidenceReport(
        subject="MedicalImageTrainer numerical instability detection"
    )

    trainer = MedicalImageTrainer(
        output_dir=tmp_path,
        random_seed=123,
    )

    dataset = TensorDataset(num_batches=2, nan=True)

    train_loader = DataLoader(dataset, batch_size=1)
    val_loader = DataLoader(dataset, batch_size=1)

    with pytest.raises(RuntimeError):
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,
        )

    report.info(
        message="Trainer failed as expected on NaN loss",
        requirement_id="TRN-003",
    )

    report.auto_save(
        "TRN_003_training_detects_nan_loss",
        evidence_output_dir,
    )

    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-001")
def test_trainer_rejects_unvalidated_input(
    tmp_path,
    evidence_output_dir,
):
    report = EvidenceReport(
        subject="MedicalImageTrainer validation boundary enforcement"
    )

    trainer = MedicalImageTrainer(output_dir=tmp_path)

    from torch.utils.data import Dataset

    class InvalidDataset(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return None

    dataset = InvalidDataset()

    train_loader = DataLoader(dataset, batch_size=1)
    val_loader = DataLoader(dataset, batch_size=1)

    with pytest.raises(Exception):
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,
        )

    report.info(
        message="Trainer rejected invalid input batch",
        requirement_id="DAT-001",
    )

    report.auto_save(
        "DAT_001_trainer_rejects_unvalidated_input",
        evidence_output_dir,
    )

    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-001")
def test_training_is_deterministic(
    tmp_path,
    evidence_output_dir,
):
    report = EvidenceReport(
        subject="MedicalImageTrainer deterministic training behavior"
    )

    seed = 1337
    dataset = TensorDataset(num_batches=4)

    train_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Run 1
    run1_dir = tmp_path / "run1"
    trainer1 = MedicalImageTrainer(output_dir=run1_dir, random_seed=seed)

    trainer1.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2)

    # Run 2
    run2_dir = tmp_path / "run2"
    trainer2 = MedicalImageTrainer(output_dir=run2_dir, random_seed=seed)

    trainer2.train(train_loader=train_loader, val_loader=val_loader, num_epochs=2)

    hash1 = _hash_metrics_file(run1_dir / "metrics.json")
    hash2 = _hash_metrics_file(run2_dir / "metrics.json")

    if hash1 != hash2:
        report.error(
            message="Training metrics are not deterministic across runs",
            requirement_id="VER-001",
        )

    report.auto_save(
        "VER_001_training_is_deterministic",
        evidence_output_dir,
    )

    assert not report.has_errors, report.summary()
