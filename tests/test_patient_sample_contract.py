import numpy as np
import pytest

from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
from medical_image_ai_toolkit.dataobjects.patient_sample_contract import enforce_patient_sample_contract
from regulatory_tools.evidence.evidence_report import EvidenceReport

@pytest.mark.requirement("DAT-003")
def test_patient_sample_contract_accepts_valid_annotations(evidence_output_dir):

    report = EvidenceReport(
        subject="DAT-003 Valid VectorROI Boundary Validation"
    )

    volume = np.zeros((16, 64, 64), dtype=np.float32)

    annotations = {
        "vector_rois": {
            5: [
                np.array([[10,10],[20,10],[20,20],[10,20]])
            ]
        }
    }

    sample = PatientSample(
        patient_id="TEST-001",
        image_volume=volume,
        spacing=(1,1,1),
        annotations=annotations
    )

    contract = enforce_patient_sample_contract(sample)

    report.issues.extend(contract.issues)
    report.auto_save("DAT003_valid_annotation", evidence_output_dir)

    assert not report.has_errors
    
@pytest.mark.requirement("DAT-003")
def test_patient_sample_contract_rejects_invalid_slice(evidence_output_dir):

    report = EvidenceReport(
        subject="DAT-003 Slice Boundary Validation"
    )

    volume = np.zeros((16,64,64), dtype=np.float32)

    annotations = {
        "vector_rois": {
            30: [  # invalid slice
                np.array([[10,10],[20,10],[20,20],[10,20]])
            ]
        }
    }

    sample = PatientSample(
        patient_id="TEST-002",
        image_volume=volume,
        spacing=(1,1,1),
        annotations=annotations
    )

    contract = enforce_patient_sample_contract(sample)

    report.issues.extend(contract.issues)
    report.auto_save("DAT003_invalid_slice", evidence_output_dir)

    assert report.has_errors
    
@pytest.mark.requirement("DAT-003")
def test_patient_sample_contract_rejects_invalid_coordinates(evidence_output_dir):

    report = EvidenceReport(
        subject="DAT-003 Coordinate Boundary Validation"
    )

    volume = np.zeros((16,64,64), dtype=np.float32)

    annotations = {
        "vector_rois": {
            5: [
                np.array([[10,10],[500,10],[20,20]])  # invalid x
            ]
        }
    }

    sample = PatientSample(
        patient_id="TEST-003",
        image_volume=volume,
        spacing=(1,1,1),
        annotations=annotations
    )

    contract = enforce_patient_sample_contract(sample)

    report.issues.extend(contract.issues)
    report.auto_save("DAT003_invalid_coordinates", evidence_output_dir)

    assert report.has_errors