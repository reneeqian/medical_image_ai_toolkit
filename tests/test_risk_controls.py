"""Risk control verification tests for medical_image_ai_toolkit.

RSK-001: NaN loss detection halts training with a defined exception.
RSK-002: Model manifest records required fields for artifact integrity checks.
RSK-003: Train/test partition non-overlap is detectable before evaluation.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from medical_image_ai_toolkit.dataobjects.annotation_bundle import AnnotationBundle
from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import (
    MedicalImageDataSource,
)
from medical_image_ai_toolkit.dataobjects.datasources.deterministic_split import (
    DeterministicHoldoutSplit,
)
from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
from medical_image_ai_toolkit.pipeline.training_pipeline import TrainingPipeline
from medical_image_ai_toolkit.training.medical_image_trainer import MedicalImageTrainer
from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition
from medical_image_ai_toolkit.training.training_config import TrainingConfig
from regulatory_tools.evidence.evidence_report import EvidenceReport


# ── Shared helpers ────────────────────────────────────────────────────────────


class _TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


class _SyntheticIngestor:
    def __init__(self, n_patients: int = 6, inject_nan: bool = False):
        self._ids = [f"P{i}" for i in range(n_patients)]
        self._inject_nan = inject_nan

    def list_patient_ids(self):
        return list(self._ids)

    def load_patient_sample(self, patient_id: str) -> PatientSample:
        volume = np.zeros((4, 8, 8), dtype=np.float32)
        if self._inject_nan and patient_id == "P0":
            volume[:] = np.nan
        return PatientSample(
            patient_id=patient_id,
            image_volume=volume,
            spacing=(1.0, 1.0, 1.0),
            annotations=AnnotationBundle(rois={}),
        )

    def get_sample(self, patient_id: str) -> PatientSample:
        return self.load_patient_sample(patient_id)


class _SimpleTask(TrainingTaskDefinition):
    def generate_training_samples(self, patient_sample):
        vol = patient_sample.image_volume
        for i in range(vol.shape[0]):
            img = torch.tensor(vol[i]).unsqueeze(0).unsqueeze(0)
            yield {"input": img, "target": torch.zeros(1)}

    def compute_loss(self, prediction, target):
        return torch.mean((prediction - target) ** 2)


class _FixedSplit:
    def split(self, ids):
        n = len(ids)
        return ids[: n // 2], ids[n // 2 : n // 2 + 1], ids[n // 2 + 1 :]


# ── RSK-001 ───────────────────────────────────────────────────────────────────


@pytest.mark.requirement("RSK-001")
def test_nan_loss_halts_training_with_runtime_error(tmp_path, evidence_output_dir):
    """RSK-001: training shall detect non-finite loss and raise RuntimeError."""
    report = EvidenceReport(
        subject="RSK-001: NaN loss detection halts training with defined exception"
    )

    ingestor = _SyntheticIngestor(inject_nan=True)
    ds = MedicalImageDataSource(ingestor)
    ds.create_partitions(_FixedSplit())

    trainer = MedicalImageTrainer(
        datasource=ds,
        model=_TinyNet(),
        training_config=TrainingConfig(epochs=1, batch_size=2, task=_SimpleTask()),
        output_dir=tmp_path / "runs",
    )

    raised = False
    try:
        trainer.train()
    except RuntimeError:
        raised = True

    if not raised:
        report.error(
            "Training with NaN input did not raise RuntimeError", "RSK-001"
        )
    else:
        report.info("RuntimeError raised on NaN loss as expected", "RSK-001")

    report.auto_save("rsk001_nan_loss_detection", evidence_output_dir)
    assert not report.has_errors, report.summary()


# ── RSK-002 ───────────────────────────────────────────────────────────────────


@pytest.mark.requirement("RSK-002")
def test_model_manifest_contains_required_fields(tmp_path, evidence_output_dir):
    """RSK-002: training shall produce a manifest with model_class, param_count,
    and training_config so artifact-model mismatches can be detected at load time.
    """
    report = EvidenceReport(
        subject="RSK-002: model manifest records required fields for artifact integrity"
    )

    ingestor = _SyntheticIngestor()
    ds = MedicalImageDataSource(ingestor)

    pipeline_out = TrainingPipeline(
        datasource=ds,
        model=_TinyNet(),
        config=TrainingConfig(epochs=1, batch_size=2, task=_SimpleTask(), split_strategy=_FixedSplit()),
        output_dir=tmp_path / "runs",
    ).run()
    results = pipeline_out["results"]

    manifest_path = results.artifacts.get("manifest")
    if manifest_path is None:
        report.error("No 'manifest' key in training artifacts", "RSK-002")
    else:
        manifest = json.loads(Path(manifest_path).read_text())
        required = {"model_class", "param_count", "training_config"}
        missing = required - manifest.keys()
        if missing:
            report.error(f"Manifest missing required fields: {missing}", "RSK-002")
        else:
            report.info(
                f"Manifest contains all required fields: {required}", "RSK-002"
            )

    report.auto_save("rsk002_model_manifest_integrity", evidence_output_dir)
    assert not report.has_errors, report.summary()


# ── RSK-003 ───────────────────────────────────────────────────────────────────


@pytest.mark.requirement("RSK-003")
def test_train_test_partition_no_overlap(evidence_output_dir):
    """RSK-003: train and test partitions shall have no patient identity overlap."""
    report = EvidenceReport(
        subject="RSK-003: evaluation partition isolation — no train/test overlap"
    )

    ingestor = _SyntheticIngestor(n_patients=10)
    ds = MedicalImageDataSource(ingestor)
    ds.create_partitions(DeterministicHoldoutSplit(val_fraction=0.2, test_fraction=0.2))

    train_ids = set(ds.train_ids)
    test_ids = set(ds.test_ids)
    overlap = train_ids & test_ids

    if overlap:
        report.error(
            f"Train/test overlap detected: {overlap}", "RSK-003"
        )
    else:
        report.info("No train/test overlap detected", "RSK-003")

    report.auto_save("rsk003_partition_isolation", evidence_output_dir)
    assert not report.has_errors, report.summary()
