import numpy as np
import pytest

from regulatory_tools.evidence.evidence_report import EvidenceReport
from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import MedicalImageDataSource


class DummyPatient:
    def __init__(self):
        self.image_volume = np.zeros((4,6,6))
        self.annotations = None


class DummyIngestor:

    def __init__(self):
        self.ids = ["A","B","C"]

    def list_patient_ids(self):
        return self.ids

    def load_patient(self, pid):
        return DummyPatient()


@pytest.mark.requirement("DAT-010")
def test_datasource_edge_cases(tmp_path, evidence_output_dir):

    report = EvidenceReport(
        subject="Datasource edge cases"
    )

    ds = MedicalImageDataSource(tmp_path, DummyIngestor())

    # -------------------------------
    # get_sample fallback branch
    # -------------------------------

    sample = ds.get_sample("A")

    if sample.image_volume.shape != (4,6,6):
        report.error("get_sample fallback failed","DAT-010")

    # -------------------------------
    # partition getters before split
    # -------------------------------

    with pytest.raises(RuntimeError):
        ds.get_train_ids()

    with pytest.raises(RuntimeError):
        ds.get_val_ids()

    with pytest.raises(RuntimeError):
        ds.get_test_ids()

    # -------------------------------
    # summary before partition
    # -------------------------------

    ds.partition_summary()

    # -------------------------------
    # show_slice validation error
    # -------------------------------

    with pytest.raises(ValueError):
        ds.show_slice("A")

    # slice bounds validation
    with pytest.raises(IndexError):
        ds.show_slice("A", slice_index=100)

    report.auto_save(
        "DAT010_datasource_edge_cases",
        evidence_output_dir
    )

    assert not report.has_errors, report.summary()