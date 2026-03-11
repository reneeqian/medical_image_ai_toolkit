import numpy as np
import pytest

from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
from medical_image_ai_toolkit.dataobjects.patient_sample_contract import enforce_patient_sample_contract
from medical_image_ai_toolkit.dataobjects.annotation_bundle import AnnotationBundle, VectorROI

from regulatory_tools.evidence.evidence_report import EvidenceReport


# ============================================================
# Helper utilities for synthetic test data
# ============================================================

def make_volume(z=8, y=32, x=32):
    return np.zeros((z, y, x), dtype=np.float32)


def make_valid_roi(slice_index=3, label="lesion"):
    contour = np.array(
        [
            [10, 10],
            [20, 10],
            [20, 20],
            [10, 20],
        ]
    )

    return VectorROI(
        contour_px=contour,
        slice_index=slice_index,
        label=label,
    )


def make_valid_annotations():
    roi = make_valid_roi(slice_index=3)

    return AnnotationBundle(
        vector_rois={
            3: [roi]
        }
    )

# ============================================================
# DAT-001 — PatientSample structural validation
# ============================================================

@pytest.mark.requirement("DAT-001")
def test_valid_patient_sample_passes_contract(evidence_output_dir):

    report = EvidenceReport(subject="DAT-001 Valid PatientSample Structure")

    sample = PatientSample(
        patient_id="TEST-001",
        image_volume=make_volume(),
        spacing=(1.0, 1.0, 1.0),
        annotations=make_valid_annotations(),
    )

    contract = enforce_patient_sample_contract(sample)

    report.issues.extend(contract.issues)
    report.auto_save("DAT001_valid_patient_sample", evidence_output_dir)

    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-001")
def test_volume_must_be_3d(evidence_output_dir):

    report = EvidenceReport(subject="DAT-001 Volume Dimensionality Validation")

    volume = np.zeros((64, 64))  # invalid 2D

    sample = PatientSample(
        patient_id="TEST-002",
        image_volume=volume,
        spacing=(1.0, 1.0, 1.0),
        annotations=None,
    )

    contract = enforce_patient_sample_contract(sample)

    report.issues.extend(contract.issues)
    report.auto_save("DAT001_volume_not_3d", evidence_output_dir)

    assert report.has_errors


@pytest.mark.requirement("DAT-001")
def test_spacing_must_be_positive(evidence_output_dir):

    report = EvidenceReport(subject="DAT-001 Spacing Validation")

    sample = PatientSample(
        patient_id="TEST-003",
        image_volume=make_volume(),
        spacing=(1.0, -1.0, 1.0),  # invalid
        annotations=None,
    )

    contract = enforce_patient_sample_contract(sample)

    report.issues.extend(contract.issues)
    report.auto_save("DAT001_spacing_invalid", evidence_output_dir)

    assert report.has_errors


@pytest.mark.requirement("DAT-001")
def test_patient_id_required(evidence_output_dir):

    report = EvidenceReport(subject="DAT-001 Patient ID Validation")

    sample = PatientSample(
        patient_id="",
        image_volume=make_volume(),
        spacing=(1.0, 1.0, 1.0),
        annotations=None,
    )

    contract = enforce_patient_sample_contract(sample)

    report.issues.extend(contract.issues)
    report.auto_save("DAT001_patient_id_missing", evidence_output_dir)

    assert report.has_errors


# ============================================================
# DAT-002 — Annotation structure compliance
# ============================================================

@pytest.mark.requirement("DAT-002")
def test_dense_mask_annotations_allowed(evidence_output_dir):

    report = EvidenceReport(subject="DAT-002 Dense Mask Annotation Support")

    mask = np.zeros((8, 32, 32))

    sample = PatientSample(
        patient_id="TEST-004",
        image_volume=make_volume(),
        spacing=(1.0, 1.0, 1.0),
        annotations=mask,
    )

    contract = enforce_patient_sample_contract(sample)

    report.issues.extend(contract.issues)
    report.auto_save("DAT002_dense_mask_annotation", evidence_output_dir)

    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-002")
def test_invalid_annotation_type_rejected(evidence_output_dir):

    report = EvidenceReport(subject="DAT-002 Invalid Annotation Type")

    sample = PatientSample(
        patient_id="TEST-005",
        image_volume=make_volume(),
        spacing=(1.0, 1.0, 1.0),
        annotations="invalid_annotation_object",
    )

    contract = enforce_patient_sample_contract(sample)

    report.issues.extend(contract.issues)
    report.auto_save("DAT002_invalid_annotation_type", evidence_output_dir)

    assert report.has_errors


@pytest.mark.requirement("DAT-002")
def test_missing_annotations_allowed_when_not_required(evidence_output_dir):

    report = EvidenceReport(subject="DAT-002 Missing Annotations Allowed")

    sample = PatientSample(
        patient_id="TEST-006",
        image_volume=make_volume(),
        spacing=(1.0, 1.0, 1.0),
        annotations=None,
    )

    contract = enforce_patient_sample_contract(
        sample,
        require_annotations=False,
    )

    report.issues.extend(contract.issues)
    report.auto_save("DAT002_annotations_optional", evidence_output_dir)

    assert not report.has_errors


# ============================================================
# DAT-003 — Annotation boundary validation
# ============================================================

@pytest.mark.requirement("DAT-003")
def test_roi_slice_out_of_bounds(evidence_output_dir):

    report = EvidenceReport(subject="DAT-003 ROI Slice Boundary Validation")

    roi = make_valid_roi(slice_index=50)

    annotations = AnnotationBundle(
        vector_rois={
            50: [roi]
        }
    )

    sample = PatientSample(
        patient_id="TEST-007",
        image_volume=make_volume(),
        spacing=(1.0, 1.0, 1.0),
        annotations=annotations,
    )

    contract = enforce_patient_sample_contract(sample)

    report.issues.extend(contract.issues)
    report.auto_save("DAT003_slice_out_of_bounds", evidence_output_dir)

    assert report.has_errors


@pytest.mark.requirement("DAT-003")
def test_roi_coordinates_out_of_bounds(evidence_output_dir):

    report = EvidenceReport(subject="DAT-003 ROI Coordinate Boundary Validation")

    contour = np.array(
        [
            [10, 10],
            [200, 10],  # invalid x
            [20, 20],
        ]
    )

    roi = VectorROI(
        contour_px=contour,
        slice_index=2,
        label="lesion",
    )
    annotations = AnnotationBundle(
        vector_rois={
            1: [roi]
        }
    )
    sample = PatientSample(
        patient_id="TEST-008",
        image_volume=make_volume(),
        spacing=(1.0, 1.0, 1.0),
        annotations=annotations,
    )

    contract = enforce_patient_sample_contract(sample)

    report.issues.extend(contract.issues)
    report.auto_save("DAT003_coordinates_out_of_bounds", evidence_output_dir)

    assert report.has_errors


@pytest.mark.requirement("DAT-003")
def test_roi_contour_shape_validation(evidence_output_dir):

    report = EvidenceReport(subject="DAT-003 ROI Contour Shape Validation")

    contour = np.array([1, 2, 3])  # invalid contour

    roi = VectorROI(
        contour_px=contour,
        slice_index=1,
        label="lesion",
    )

    annotations = AnnotationBundle(vector_rois=[roi])

    sample = PatientSample(
        patient_id="TEST-009",
        image_volume=make_volume(),
        spacing=(1.0, 1.0, 1.0),
        annotations=annotations,
    )

    contract = enforce_patient_sample_contract(sample)

    report.issues.extend(contract.issues)
    report.auto_save("DAT003_invalid_contour_shape", evidence_output_dir)

    assert report.has_errors

@pytest.mark.requirement("DAT-002")
def test_annotations_required_missing(evidence_output_dir):

    report = EvidenceReport(subject="DAT-002 Required Annotations Missing")

    sample = PatientSample(
        patient_id="TEST-010",
        image_volume=make_volume(),
        spacing=(1.0,1.0,1.0),
        annotations=None,
    )

    contract = enforce_patient_sample_contract(
        sample,
        require_annotations=True,
    )

    report.issues.extend(contract.issues)
    report.auto_save("DAT002_annotations_required_missing", evidence_output_dir)

    assert report.has_errors

@pytest.mark.requirement("DAT-002")
def test_vector_rois_none(evidence_output_dir):

    report = EvidenceReport(subject="DAT-002 vector_rois None")

    annotations = AnnotationBundle(vector_rois=None)

    sample = PatientSample(
        patient_id="TEST-011",
        image_volume=make_volume(),
        spacing=(1.0,1.0,1.0),
        annotations=annotations,
    )

    contract = enforce_patient_sample_contract(sample)

    report.issues.extend(contract.issues)
    report.auto_save("DAT002_vector_rois_none", evidence_output_dir)

    assert not report.has_errors
    
@pytest.mark.requirement("DAT-002")
def test_vector_rois_not_dict(evidence_output_dir):

    report = EvidenceReport(subject="DAT-002 vector_rois must be dict")

    annotations = AnnotationBundle(vector_rois=["not","a","dict"])

    sample = PatientSample(
        patient_id="TEST-012",
        image_volume=make_volume(),
        spacing=(1.0,1.0,1.0),
        annotations=annotations,
    )

    contract = enforce_patient_sample_contract(sample)

    report.issues.extend(contract.issues)
    report.auto_save("DAT002_vector_rois_not_dict", evidence_output_dir)

    assert report.has_errors

@pytest.mark.requirement("DAT-003")
def test_slice_index_not_int(evidence_output_dir):

    report = EvidenceReport(subject="DAT-003 slice index type")

    roi = make_valid_roi()

    annotations = AnnotationBundle(
        vector_rois={
            "bad_index": [roi]
        }
    )

    sample = PatientSample(
        patient_id="TEST-013",
        image_volume=make_volume(),
        spacing=(1.0,1.0,1.0),
        annotations=annotations,
    )

    contract = enforce_patient_sample_contract(sample)

    report.issues.extend(contract.issues)
    report.auto_save("DAT003_slice_index_not_int", evidence_output_dir)

    assert report.has_errors
    
@pytest.mark.requirement("DAT-003")
def test_rois_not_list(evidence_output_dir):

    report = EvidenceReport(subject="DAT-003 rois must be list")

    roi = make_valid_roi()

    annotations = AnnotationBundle(
        vector_rois={
            2: roi
        }
    )

    sample = PatientSample(
        patient_id="TEST-014",
        image_volume=make_volume(),
        spacing=(1.0,1.0,1.0),
        annotations=annotations,
    )

    contract = enforce_patient_sample_contract(sample)

    report.issues.extend(contract.issues)
    report.auto_save("DAT003_rois_not_list", evidence_output_dir)

    assert report.has_errors

@pytest.mark.requirement("DAT-003")
def test_roi_wrong_type(evidence_output_dir):

    report = EvidenceReport(subject="DAT-003 roi wrong type")

    annotations = AnnotationBundle(
        vector_rois={
            2: ["not_roi"]
        }
    )

    sample = PatientSample(
        patient_id="TEST-015",
        image_volume=make_volume(),
        spacing=(1.0,1.0,1.0),
        annotations=annotations,
    )

    contract = enforce_patient_sample_contract(sample)

    report.issues.extend(contract.issues)
    report.auto_save("DAT003_roi_wrong_type", evidence_output_dir)

    assert report.has_errors

@pytest.mark.requirement("DAT-003")
def test_roi_y_out_of_bounds(evidence_output_dir):

    report = EvidenceReport(subject="DAT-003 y bounds")

    contour = np.array([
        [10,10],
        [20,1000],  # invalid y
        [30,10],
    ])

    roi = VectorROI(
        contour_px=contour,
        slice_index=1,
        label="lesion"
    )

    annotations = AnnotationBundle(
        vector_rois={1:[roi]}
    )

    sample = PatientSample(
        patient_id="TEST-016",
        image_volume=make_volume(),
        spacing=(1.0,1.0,1.0),
        annotations=annotations,
    )

    contract = enforce_patient_sample_contract(sample)

    report.issues.extend(contract.issues)
    report.auto_save("DAT003_y_out_of_bounds", evidence_output_dir)

    assert report.has_errors


