from pathlib import Path
import numpy as np
import pytest

from regulatory_tools.evidence.evidence_report import EvidenceReport
from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import MedicalImageDataSource


# =========================================================
# Shared Dummy Objects
# =========================================================

class DummyPatient:

    def __init__(self):
        self.image_volume = np.zeros((5, 10, 10))
        self.annotations = None


class DummyIngestor:

    def __init__(self):
        self.ids = [f"P{i}" for i in range(10)]

    def list_patient_ids(self):
        return self.ids

    def load_patient(self, patient_id):
        return DummyPatient()


class DummyPartitionStrategy:

    def split(self, patient_ids):

        n = len(patient_ids)

        train = patient_ids[: int(0.6 * n)]
        val = patient_ids[int(0.6 * n): int(0.8 * n)]
        test = patient_ids[int(0.8 * n):]

        return train, val, test


# =========================================================
# Dataset Partition Generation
# =========================================================

@pytest.mark.requirement("DAT-006")
@pytest.mark.requirement("DAT-007")
@pytest.mark.requirement("DAT-008")
@pytest.mark.requirement("SYS-001")
@pytest.mark.requirement("VER-003")
def test_dataset_partition_generation(tmp_path, evidence_output_dir):

    report = EvidenceReport(
        subject="Dataset partition generation"
    )

    ds = MedicalImageDataSource(
        dataset_root=tmp_path,
        ingestor=DummyIngestor()
    )

    # dataset discovery
    ids = ds.get_patient_ids()

    if len(ids) != 10:
        report.error("Incorrect patient discovery", "DAT-006")

    # partition
    train, val, test = ds.create_partitions(DummyPartitionStrategy())

    if not ds.has_partitions():
        report.error("Partitions were not created", "DAT-007")

    # verify no overlap
    overlap = (
        set(train) & set(val)
        or set(train) & set(test)
        or set(val) & set(test)
    )

    if overlap:
        report.error("Dataset partitions overlap", "VER-003")

    # slice validation
    slice_data = ds.load_slice(train[0], 0)

    if slice_data.shape != (10, 10):
        report.error("Slice loading returned incorrect shape", "DAT-008")

    report.auto_save(
        "DAT006_DAT007_DAT008_dataset_partitioning",
        evidence_output_dir
    )

    assert not report.has_errors, report.summary()


# =========================================================
# Datasource API Behavior
# =========================================================

@pytest.mark.requirement("SYS-001")
@pytest.mark.requirement("DAT-004")
@pytest.mark.requirement("DAT-007")
def test_datasource_behaviors(tmp_path, evidence_output_dir):

    report = EvidenceReport(
        subject="Datasource API behavior"
    )

    ds = MedicalImageDataSource(
        dataset_root=tmp_path,
        ingestor=DummyIngestor()
    )

    # len
    if len(ds) != 10:
        report.error("Incorrect dataset length", "SYS-001")

    # getitem
    patient = ds[0]

    if patient.image_volume.shape != (5, 10, 10):
        report.error("getitem failed", "SYS-001")

    # partition dataset
    ds.create_partitions(DummyPartitionStrategy())

    if not ds.has_partitions():
        report.error("Partition flag incorrect", "DAT-007")

    train = ds.get_train_ids()
    val = ds.get_val_ids()
    test = ds.get_test_ids()

    if len(train) + len(val) + len(test) != 10:
        report.error("Partition sizes incorrect", "DAT-007")

    # slice load
    s = ds.load_slice(train[0], 1)

    if s.shape != (10, 10):
        report.error("Slice extraction failed", "DAT-004")

    # slice bounds
    with pytest.raises(IndexError):
        ds.load_slice(train[0], 100)

    # summary (coverage)
    ds.partition_summary()

    report.auto_save(
        "SYS001_DAT004_DAT007_datasource_behavior",
        evidence_output_dir
    )

    assert not report.has_errors, report.summary()