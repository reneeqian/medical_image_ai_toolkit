# Requirements (MIT)
# MIT-FR-01 Datasets shall expose samples via a consistent PatientSample interface.
# MIT-DR-01 PatientSample image_volume shall be a NumPy array.
# MIT-DR-02 PatientSample image_volume shall be three-dimensional (z,y,x).


from pathlib import Path
import torch
import sys
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from regulatory_tools.evidence.evidence_report import EvidenceReport
from medical_image_ai_toolkit.adapters.patient_sample_to_tensor import PatientSampleTensorAdapter
from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample

@pytest.fixture
def sample():
    return PatientSample(
        patient_id="TEST-001",
        image_volume=np.zeros((16, 64, 64), dtype=np.float32),
        spacing=(1.0, 1.0, 1.0),
        annotations=np.zeros((16, 64, 64), dtype=np.int64),
    )

def make_invalid_sample():
    return PatientSample(
        patient_id="INVALID-001",
        image_volume=None,  # invalid
        spacing=(1.0, 1.0, 1.0),
        annotations=None,
    )

@pytest.mark.requirement("MIT-DR-03")
def test_MIT_DR_03_patient_sample_converted_to_3d_tensor(sample):
    report = EvidenceReport(
        subject="PatientSample to tensor dimensionality and interface validation"
    )

    adapter = PatientSampleTensorAdapter(require_annotations=True)
    out = adapter(sample)

    image = out["image"]
    target = out["target"]

    # MIT-DR-02
    assert image.ndim == 4
    assert image.shape[0] == 1

    report.info(
        message="PatientSample image_volume converted to 3D tensor with channel dimension",
        requirement_id="MIT-DR-03",
        context=f"shape={tuple(image.shape)}",
    )

    # MIT-DR-01
    assert image.dtype == torch.float32
    assert torch.all(image >= 0)
    assert torch.all(image <= 1)

    report.info(
        message="image_volume represented as normalized float32 tensor",
        requirement_id="MIT-DR-03",
    )

    # MIT-FR-01
    assert target is not None

    report.info(
        message="Adapter preserves PatientSample interface semantics",
        requirement_id="MIT-FR-03",
    )

    report.auto_save("MIT_DR_03_patient_sample_converted_to_3d_tensor")
    assert not report.has_errors, report.summary()

@pytest.mark.requirement("MIT-DR-01")
def test_MIT_DR_01_invalid_patient_sample_rejected():
    report = EvidenceReport(
        subject="Invalid PatientSample rejection"
    )

    sample = make_invalid_sample()
    adapter = PatientSampleTensorAdapter()

    with pytest.raises(ValueError):
        adapter(sample)

    report.info(
        message="Adapter rejects PatientSample with invalid image_volume",
        requirement_id="MIT-DR-01",
    )

    report.auto_save("MIT_DR_01_invalid_patient_sample_rejected")
    assert not report.has_errors, report.summary()
