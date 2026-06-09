"""Tests for DatasetValidator (DAT-011)."""

from types import SimpleNamespace

import numpy as np
import pytest

from medical_image_ai_toolkit.dataobjects.annotation_bundle import AnnotationBundle
from medical_image_ai_toolkit.dataobjects.data_validation.dataset_validator import DatasetValidator
from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample


def _make_sample(pid="p1", shape=(5, 32, 32), spacing=(1.0, 1.0, 1.0), annotations=None):
    vol = np.zeros(shape, dtype=np.float32)
    ann = annotations if annotations is not None else AnnotationBundle(vector_rois=None)
    return PatientSample(
        patient_id=pid,
        image_volume=vol,
        spacing=spacing,
        annotations=ann,
    )


class FakeDatasource:
    """Minimal fake datasource — no DICOM needed."""

    def __init__(self, patient_ids, samples_or_exc):
        self.ingestor = SimpleNamespace(report=None)
        self._ids = patient_ids
        self._data = samples_or_exc  # dict[pid -> PatientSample | Exception]

    def get_patient_ids(self):
        return list(self._ids)

    def get_patient(self, pid):
        v = self._data[pid]
        if isinstance(v, Exception):
            raise v
        return v


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


from regulatory_tools.evidence.evidence_report import EvidenceReport


@pytest.mark.requirement("DAT-011")
def test_empty_dataset_produces_no_errors(evidence_output_dir):
    evidence = EvidenceReport(subject="DAT-011: empty dataset produces no errors")
    ds = FakeDatasource([], {})
    report = DatasetValidator(ds).run()
    total_msgs = [i.message for i in report.issues if "Total patients" in i.message]
    if report.has_errors:
        evidence.error("DatasetValidator reported errors for empty dataset", "DAT-011")
    if not total_msgs:
        evidence.error("Expected 'Total patients' INFO message not found", "DAT-011")
    evidence.info(
        "Empty dataset validated without errors and produced 'Total patients' summary", "DAT-011"
    )
    evidence.auto_save("DAT011_empty_dataset_no_errors", evidence_output_dir)
    assert not evidence.has_errors, evidence.summary()
    assert not report.has_errors
    assert total_msgs, "Expected 'Total patients' INFO message"


@pytest.mark.requirement("DAT-011")
def test_single_valid_patient_no_errors(evidence_output_dir):
    evidence = EvidenceReport(subject="DAT-011: single valid patient produces no errors")
    sample = _make_sample("p1")
    ds = FakeDatasource(["p1"], {"p1": sample})
    report = DatasetValidator(ds).run()
    if report.has_errors:
        evidence.error("DatasetValidator reported errors for a valid single patient", "DAT-011")
    evidence.info("Single valid PatientSample validated without errors", "DAT-011")
    evidence.auto_save("DAT011_single_valid_patient_no_errors", evidence_output_dir)
    assert not evidence.has_errors, evidence.summary()
    assert not report.has_errors


@pytest.mark.requirement("DAT-011")
def test_multiple_valid_patients_all_processed(evidence_output_dir):
    evidence = EvidenceReport(subject="DAT-011: multiple valid patients all processed")
    samples = {str(i): _make_sample(str(i)) for i in range(5)}
    ds = FakeDatasource(list(samples.keys()), samples)
    report = DatasetValidator(ds).run()
    if report.has_errors:
        evidence.error("DatasetValidator reported errors for 5 valid patients", "DAT-011")
    evidence.info("All 5 valid patients processed without errors", "DAT-011")
    evidence.auto_save("DAT011_multiple_valid_patients", evidence_output_dir)
    assert not evidence.has_errors, evidence.summary()
    assert not report.has_errors


@pytest.mark.requirement("DAT-011")
def test_one_failing_patient_does_not_abort_others(evidence_output_dir):
    evidence = EvidenceReport(
        subject="DAT-011: one failing patient does not abort validation of others"
    )
    ds = FakeDatasource(
        ["p1", "p2", "p3"],
        {
            "p1": _make_sample("p1"),
            "p2": RuntimeError("corrupt DICOM"),
            "p3": _make_sample("p3"),
        },
    )
    report = DatasetValidator(ds).run()
    error_msgs = [i.message for i in report.issues if i.level == "ERROR"]
    if not report.has_errors:
        evidence.error(
            "Expected DatasetValidator to report errors for failing patient p2", "DAT-011"
        )
    if len(error_msgs) != 1:
        evidence.error(f"Expected exactly 1 error message, got {len(error_msgs)}", "DAT-011")
    evidence.info(
        "Single failing patient produced exactly 1 error; other patients were still processed",
        "DAT-011",
    )
    evidence.auto_save("DAT011_one_failing_patient", evidence_output_dir)
    assert not evidence.has_errors, evidence.summary()
    assert report.has_errors
    assert len(error_msgs) == 1


@pytest.mark.requirement("DAT-011")
def test_all_patients_fail_reports_all_errors(evidence_output_dir):
    evidence = EvidenceReport(subject="DAT-011: all patients failing reports all errors")
    n = 3
    ids = [str(i) for i in range(n)]
    ds = FakeDatasource(ids, {pid: RuntimeError("bad") for pid in ids})
    report = DatasetValidator(ds).run()
    errors = [i for i in report.issues if i.level == "ERROR"]
    if len(errors) != n:
        evidence.error(
            f"Expected {n} errors for {n} failing patients, got {len(errors)}", "DAT-011"
        )
    evidence.info(
        f"All {n} failing patients produced {len(errors)} distinct error entries — no silent failures",
        "DAT-011",
    )
    evidence.auto_save("DAT011_all_patients_fail", evidence_output_dir)
    assert not evidence.has_errors, evidence.summary()
    assert len(errors) == n


@pytest.mark.requirement("DAT-011")
def test_none_volume_triggers_contract_error(evidence_output_dir):
    evidence = EvidenceReport(subject="DAT-011: None image_volume triggers contract error")
    sample = _make_sample("p1")
    sample.image_volume = None  # break the contract
    ds = FakeDatasource(["p1"], {"p1": sample})
    report = DatasetValidator(ds).run()
    if not report.has_errors:
        evidence.error("Expected contract error for None image_volume", "DAT-011")
    evidence.info(
        "None image_volume correctly flagged as contract violation by DatasetValidator", "DAT-011"
    )
    evidence.auto_save("DAT011_none_volume_contract_error", evidence_output_dir)
    assert not evidence.has_errors, evidence.summary()
    assert report.has_errors


@pytest.mark.requirement("DAT-011")
def test_2d_volume_triggers_contract_error(evidence_output_dir):
    evidence = EvidenceReport(subject="DAT-011: 2D volume triggers contract error")
    sample = _make_sample("p1")
    sample.image_volume = np.zeros((32, 32), dtype=np.float32)  # 2D, not 3D
    ds = FakeDatasource(["p1"], {"p1": sample})
    report = DatasetValidator(ds).run()
    if not report.has_errors:
        evidence.error("Expected contract error for 2D image_volume", "DAT-011")
    evidence.info(
        "2D (non-3D) image_volume correctly flagged as contract violation by DatasetValidator",
        "DAT-011",
    )
    evidence.auto_save("DAT011_2d_volume_contract_error", evidence_output_dir)
    assert not evidence.has_errors, evidence.summary()
    assert report.has_errors


@pytest.mark.requirement("DAT-011")
def test_negative_spacing_produces_warning_or_error(evidence_output_dir):
    evidence = EvidenceReport(subject="DAT-011: negative spacing produces WARN or ERROR")
    sample = _make_sample("p1", spacing=(-1.0, 1.0, 1.0))
    ds = FakeDatasource(["p1"], {"p1": sample})
    report = DatasetValidator(ds).run()
    non_info = [i for i in report.issues if i.level in ("ERROR", "WARN")]
    if not non_info:
        evidence.error("Expected WARN or ERROR for negative spacing, got none", "DAT-011")
    evidence.info(
        f"Negative spacing produced {len(non_info)} WARN/ERROR issue(s) as expected", "DAT-011"
    )
    evidence.auto_save("DAT011_negative_spacing_warn_or_error", evidence_output_dir)
    assert not evidence.has_errors, evidence.summary()
    assert non_info, "Expected WARN or ERROR for negative spacing"


@pytest.mark.requirement("DAT-011")
def test_has_errors_true_when_load_fails(evidence_output_dir):
    evidence = EvidenceReport(subject="DAT-011: has_errors=True when patient load fails")
    ds = FakeDatasource(["p1"], {"p1": RuntimeError("fail")})
    report = DatasetValidator(ds).run()
    if report.has_errors is not True:
        evidence.error("has_errors should be True when patient load raises an exception", "DAT-011")
    evidence.info("has_errors=True when patient load raises RuntimeError", "DAT-011")
    evidence.auto_save("DAT011_has_errors_true_on_load_fail", evidence_output_dir)
    assert not evidence.has_errors, evidence.summary()
    assert report.has_errors is True


@pytest.mark.requirement("DAT-011")
def test_has_errors_false_when_all_load_cleanly(evidence_output_dir):
    evidence = EvidenceReport(subject="DAT-011: has_errors=False when all patients load cleanly")
    ds = FakeDatasource(["p1"], {"p1": _make_sample("p1")})
    report = DatasetValidator(ds).run()
    if report.has_errors is not False:
        evidence.error("has_errors should be False when all patients load cleanly", "DAT-011")
    evidence.info("has_errors=False when all patients load without error", "DAT-011")
    evidence.auto_save("DAT011_has_errors_false_on_clean_load", evidence_output_dir)
    assert not evidence.has_errors, evidence.summary()
    assert report.has_errors is False


@pytest.mark.requirement("DAT-011")
def test_ingestor_report_is_set_before_iteration(evidence_output_dir):
    evidence = EvidenceReport(subject="DAT-011: ingestor.report is set before patient iteration")
    captured = []

    class TrackingDatasource(FakeDatasource):
        def get_patient_ids(self):
            captured.append(self.ingestor.report)
            return super().get_patient_ids()

    ds = TrackingDatasource(["p1"], {"p1": _make_sample("p1")})
    DatasetValidator(ds).run()
    if not captured or captured[0] is None:
        evidence.error("ingestor.report not set before iteration", "DAT-011")
    evidence.info(
        "ingestor.report is set to the EvidenceReport instance before get_patient_ids() is called",
        "DAT-011",
    )
    evidence.auto_save("DAT011_ingestor_report_set_before_iteration", evidence_output_dir)
    assert not evidence.has_errors, evidence.summary()
    assert captured and captured[0] is not None, "ingestor.report not set before iteration"


@pytest.mark.requirement("DAT-011")
def test_report_auto_save_produces_valid_json(evidence_output_dir):
    import json

    ds = FakeDatasource(["p1"], {"p1": _make_sample("p1")})
    report = DatasetValidator(ds).run()
    report.info("DatasetValidator EvidenceReport auto_save produces valid JSON artifact", "DAT-011")
    report.auto_save("dat011_dataset_validator", evidence_output_dir)
    saved_files = list(evidence_output_dir.glob("dat011_dataset_validator*.json"))
    assert saved_files, "No evidence file written"
    data = json.loads(saved_files[0].read_text())
    assert "subject" in data
