import numpy as np
import pytest

from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
from medical_image_ai_toolkit.contracts.patient_sample_contract import enforce_patient_sample_contract
from regulatory_tools.evidence.evidence_report import EvidenceReport

@pytest.mark.requirement("MIT-DR-01")
@pytest.mark.requirement("MIT-DR-02")
@pytest.mark.requirement("MIT-VRF-01")
def test_MIT_DR_09_enforce_patient_sample_contract_boundary(evidence_output_dir):
    report = EvidenceReport(subject="PatientSample Contract (Dummy)")

    sample = PatientSample(
        patient_id="DUMMY-001",
        image_volume=np.zeros((16, 64, 64), dtype=np.float32),
        spacing=(1.0, 1.0, 1.0),
        annotations=None,
    )

    contract_report = enforce_patient_sample_contract(
        sample,
        require_annotations=False,
    )

    report.issues.extend(contract_report.issues)
    report.auto_save("patient_sample_contract_dummy", evidence_output_dir)

    assert not report.has_errors, report.summary()
