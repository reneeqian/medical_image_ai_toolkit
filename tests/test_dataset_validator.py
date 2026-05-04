"""Tests for DatasetValidator (DAT-011)."""
from types import SimpleNamespace

import numpy as np
import pytest

from medical_image_ai_toolkit.dataobjects.data_validation.dataset_validator import DatasetValidator
from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
from medical_image_ai_toolkit.dataobjects.annotation_bundle import AnnotationBundle


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


@pytest.mark.requirement("DAT-011")
def test_empty_dataset_produces_no_errors():
    ds = FakeDatasource([], {})
    report = DatasetValidator(ds).run()
    assert not report.has_errors
    total_msgs = [i.message for i in report.issues if "Total patients" in i.message]
    assert total_msgs, "Expected 'Total patients' INFO message"


@pytest.mark.requirement("DAT-011")
def test_single_valid_patient_no_errors():
    sample = _make_sample("p1")
    ds = FakeDatasource(["p1"], {"p1": sample})
    report = DatasetValidator(ds).run()
    assert not report.has_errors


@pytest.mark.requirement("DAT-011")
def test_multiple_valid_patients_all_processed():
    samples = {str(i): _make_sample(str(i)) for i in range(5)}
    ds = FakeDatasource(list(samples.keys()), samples)
    report = DatasetValidator(ds).run()
    assert not report.has_errors


@pytest.mark.requirement("DAT-011")
def test_one_failing_patient_does_not_abort_others():
    good = _make_sample("p2")
    ds = FakeDatasource(
        ["p1", "p2", "p3"],
        {
            "p1": _make_sample("p1"),
            "p2": RuntimeError("corrupt DICOM"),
            "p3": _make_sample("p3"),
        },
    )
    report = DatasetValidator(ds).run()
    assert report.has_errors
    error_msgs = [i.message for i in report.issues if i.level == "ERROR"]
    assert len(error_msgs) == 1


@pytest.mark.requirement("DAT-011")
def test_all_patients_fail_reports_all_errors():
    n = 3
    ids = [str(i) for i in range(n)]
    ds = FakeDatasource(ids, {pid: RuntimeError("bad") for pid in ids})
    report = DatasetValidator(ds).run()
    errors = [i for i in report.issues if i.level == "ERROR"]
    assert len(errors) == n


@pytest.mark.requirement("DAT-011")
def test_none_volume_triggers_contract_error():
    sample = _make_sample("p1")
    sample.image_volume = None  # break the contract
    ds = FakeDatasource(["p1"], {"p1": sample})
    report = DatasetValidator(ds).run()
    assert report.has_errors


@pytest.mark.requirement("DAT-011")
def test_2d_volume_triggers_contract_error():
    sample = _make_sample("p1")
    sample.image_volume = np.zeros((32, 32), dtype=np.float32)  # 2D, not 3D
    ds = FakeDatasource(["p1"], {"p1": sample})
    report = DatasetValidator(ds).run()
    assert report.has_errors


@pytest.mark.requirement("DAT-011")
def test_negative_spacing_produces_warning_or_error():
    sample = _make_sample("p1", spacing=(-1.0, 1.0, 1.0))
    ds = FakeDatasource(["p1"], {"p1": sample})
    report = DatasetValidator(ds).run()
    non_info = [i for i in report.issues if i.level in ("ERROR", "WARN")]
    assert non_info, "Expected WARN or ERROR for negative spacing"


@pytest.mark.requirement("DAT-011")
def test_has_errors_true_when_load_fails():
    ds = FakeDatasource(["p1"], {"p1": RuntimeError("fail")})
    report = DatasetValidator(ds).run()
    assert report.has_errors is True


@pytest.mark.requirement("DAT-011")
def test_has_errors_false_when_all_load_cleanly():
    ds = FakeDatasource(["p1"], {"p1": _make_sample("p1")})
    report = DatasetValidator(ds).run()
    assert report.has_errors is False


@pytest.mark.requirement("DAT-011")
def test_ingestor_report_is_set_before_iteration():
    captured = []

    class TrackingDatasource(FakeDatasource):
        def get_patient_ids(self):
            captured.append(self.ingestor.report)
            return super().get_patient_ids()

    ds = TrackingDatasource(["p1"], {"p1": _make_sample("p1")})
    DatasetValidator(ds).run()
    assert captured and captured[0] is not None, "ingestor.report not set before iteration"


@pytest.mark.requirement("DAT-011")
def test_report_auto_save_produces_valid_json(evidence_output_dir):
    import json

    ds = FakeDatasource(["p1"], {"p1": _make_sample("p1")})
    report = DatasetValidator(ds).run()
    report.auto_save("dat011_dataset_validator", evidence_output_dir)
    saved_files = list(evidence_output_dir.glob("dat011_dataset_validator*.json"))
    assert saved_files, "No evidence file written"
    data = json.loads(saved_files[0].read_text())
    assert "subject" in data
