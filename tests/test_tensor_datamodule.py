import pytest
import torch
from pathlib import Path

from medical_image_ai_toolkit.datamodules.tensor_datamodule import TensorDatamodule
from regulatory_tools.evidence.evidence_report import EvidenceReport

@pytest.mark.requirement("MIT-FR-01")
def test_MIT_FR_01_tensor_datamodule_exposes_dataset_length(evidence_output_dir):
    report = EvidenceReport(
        subject="TensorDatamodule dataset length exposure"
    )

    samples = [{"x": torch.randn(1)}, {"x": torch.randn(1)}]
    ds = TensorDatamodule(samples)

    assert len(ds) == 2

    report.info(
        message="TensorDatamodule length matches number of provided samples",
        requirement_id="MIT-FR-01",
        context="len(dataset)=2",
    )

    report.auto_save("MIT_FR_01_tensor_datamodule_exposes_dataset_length",evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("MIT-TR-01")
def test_MIT_TR_01_tensor_datamodule_deterministic_ordering(evidence_output_dir):
    report = EvidenceReport(
        subject="Deterministic ordering of TensorDatamodule samples"
    )

    samples = [
        {"patient_id": "b", "x": torch.tensor(1)},
        {"patient_id": "a", "x": torch.tensor(2)},
    ]

    ds = TensorDatamodule(samples, deterministic=True)

    assert ds[0]["patient_id"] == "a"

    report.info(
        message="Samples are deterministically ordered by patient_id when enabled",
        requirement_id="MIT-TR-01",
        context="first_patient_id=a",
    )

    report.auto_save("MIT_TR_01_tensor_datamodule_deterministic_ordering",evidence_output_dir)
    assert not report.has_errors, report.summary()

@pytest.mark.requirement("MIT-FR-01")
def test_MIT_FR_01_tensor_datamodule_getitem_returns_sample(evidence_output_dir):
    report = EvidenceReport(
        subject="TensorDatamodule __getitem__ behavior"
    )

    sample = {"x": torch.tensor([1, 2, 3])}
    ds = TensorDatamodule([sample])

    out = ds[0]

    assert torch.equal(out["x"], sample["x"])

    report.info(
        message="TensorDatamodule returns original sample via __getitem__",
        requirement_id="MIT-FR-01",
    )

    report.auto_save("MIT_FR_01_tensor_datamodule_getitem_returns_sample",evidence_output_dir)
    assert not report.has_errors, report.summary()

@pytest.mark.requirement("MIT-NFR-01")
def test_MIT_NFR_01_tensor_datamodule_empty_dataset_rejected(evidence_output_dir):
    report = EvidenceReport(
        subject="TensorDatamodule validation of empty dataset"
    )

    with pytest.raises(ValueError):
        TensorDatamodule([])

    report.info(
        message="TensorDatamodule rejects empty dataset at construction time",
        requirement_id="MIT-NFR-01",
    )

    report.auto_save("MIT_NFR_01_tensor_datamodule_empty_dataset_rejected", evidence_output_dir)
    assert not report.has_errors, report.summary()
