import numpy as np
import pytest
from regulatory_tools.evidence.evidence_report import EvidenceReport

from medical_image_ai_toolkit.dataobjects.annotation_bundle import AnnotationBundle
from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample


@pytest.mark.requirement("DAT-005")
def test_patient_sample_basic(evidence_output_dir):

    report = EvidenceReport(subject="Patient sample behavior")

    volume = np.zeros((3, 5, 5))

    annotations = AnnotationBundle(vector_rois={})

    sample = PatientSample(
        patient_id="TEST", image_volume=volume, spacing=(1.0, 1.0, 1.0), annotations=annotations
    )

    if sample.image_volume.shape != (3, 5, 5):
        report.error("Volume shape mismatch", "DAT-005")

    # exercise repr
    text = repr(sample)

    if "PatientSample" not in text:
        report.error("repr output missing class name", "DAT-005")

    if "vector_rois: none" not in text:
        report.error("repr annotation summary incorrect", "DAT-005")

    report.info(
        f"PatientSample shape={sample.image_volume.shape}, repr includes class name and annotation summary",
        "DAT-005",
    )
    report.auto_save("DAT005_patient_sample_basic", evidence_output_dir)

    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-005")
def test_patient_sample_repr_with_annotations(evidence_output_dir):

    report = EvidenceReport(subject="Patient sample repr with annotations")

    volume = np.ones((3, 4, 4))

    # fake ROI object
    class DummyROI:
        contour_px = np.array([[0, 0], [1, 1], [2, 2]])

    annotations = AnnotationBundle(vector_rois={1: [DummyROI(), DummyROI()]})

    sample = PatientSample(
        patient_id="PAT123",
        image_volume=volume,
        spacing=(1.0, 1.0, 1.0),
        annotations=annotations,
        metadata={"scanner": "CT"},
    )

    text = repr(sample)

    if "vector_rois" not in text:
        report.error("repr missing ROI summary", "DAT-005")

    if "metadata keys" not in text:
        report.error("repr missing metadata summary", "DAT-005")

    report.info(
        "PatientSample repr includes ROI summary and metadata keys when annotations and metadata are present",
        "DAT-005",
    )
    report.auto_save("DAT005_patient_sample_repr_annotations", evidence_output_dir)

    assert not report.has_errors, report.summary()
