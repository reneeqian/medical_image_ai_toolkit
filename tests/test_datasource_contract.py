from pathlib import Path
import numpy as np
import pytest

from regulatory_tools.evidence.evidence_report import EvidenceReport
from medical_image_ai_toolkit.dataobjects.annotation_bundle import AnnotationBundle, VectorROI
from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import MedicalImageDataSource


# =========================================================
# Shared Dummy Objects
# =========================================================

class DummyPatient:

    def __init__(self):
        self.image_volume = np.zeros((5, 10, 10))
        self.annotations = None


class AnnotatedDummyPatient(DummyPatient):

    def __init__(self):
        super().__init__()
        contour = np.array(
            [
                [1, 1],
                [4, 1],
                [4, 4],
                [1, 4],
            ],
            dtype=np.float32,
        )
        roi = VectorROI(slice_index=2, contour_px=contour, label="lesion")
        self.annotations = AnnotationBundle(vector_rois={2: [roi]})


class DummyIngestor:

    def __init__(self):
        self.ids = [f"P{i}" for i in range(10)]

    def list_patient_ids(self):
        return self.ids

    def load_patient_sample(self, patient_id):
        return DummyPatient()


class AnnotatedDummyIngestor(DummyIngestor):

    def load_patient_sample(self, patient_id):
        return AnnotatedDummyPatient()


class SampleAwareIngestor(DummyIngestor):

    def get_sample(self, patient_id):
        return ("custom", patient_id)


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

@pytest.mark.requirement("SYS-001")
def test_dataset_partition_generation(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="SYS-001: MedicalImageDataSource integrates ingestor, partitioning, and slice access")

    ds = MedicalImageDataSource(dataset_root=tmp_path, ingestor=DummyIngestor())
    ids = ds.get_patient_ids()
    train, val, test = ds.create_partitions(DummyPartitionStrategy())
    slice_data = ds.load_slice(train[0], 0)

    report.info(
        f"End-to-end: discovered {len(ids)} patients, partitioned into train={len(train)}/val={len(val)}/test={len(test)}, slice shape={slice_data.shape}",
        "SYS-001",
    )
    report.auto_save("SYS001_dataset_partition_generation", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-003")
def test_partition_no_overlap(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="VER-003: train/val/test partitions have no patient overlap")

    ds = MedicalImageDataSource(dataset_root=tmp_path, ingestor=DummyIngestor())
    train, val, test = ds.create_partitions(DummyPartitionStrategy())

    for pair, label in (
        (set(train) & set(val), "train/val"),
        (set(train) & set(test), "train/test"),
        (set(val) & set(test), "val/test"),
    ):
        if pair:
            report.error(f"{label} overlap detected: {pair}", "VER-003")
        else:
            report.info(f"No {label} overlap", "VER-003")

    report.auto_save("VER003_partition_no_overlap", evidence_output_dir)
    assert not report.has_errors, report.summary()


# =========================================================
# Datasource API Behavior
# =========================================================

@pytest.mark.requirement("SYS-001")
def test_datasource_len_and_getitem(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="SYS-001: MedicalImageDataSource supports len() and __getitem__ access")

    ds = MedicalImageDataSource(dataset_root=tmp_path, ingestor=DummyIngestor())
    n = len(ds)
    patient = ds[0]

    if n != 10:
        report.error(f"len(ds)={n}, expected 10", "SYS-001")
    if patient.image_volume.shape != (5, 10, 10):
        report.error(f"ds[0].image_volume.shape={patient.image_volume.shape}, expected (5,10,10)", "SYS-001")

    report.info(f"len(ds)={n}, ds[0].image_volume.shape={patient.image_volume.shape}", "SYS-001")
    report.auto_save("SYS001_datasource_len_and_getitem", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-004")
def test_load_slice_shape_and_bounds(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="DAT-004: load_slice returns correct shape and raises IndexError on out-of-bounds index")

    ds = MedicalImageDataSource(dataset_root=tmp_path, ingestor=DummyIngestor())
    ds.create_partitions(DummyPartitionStrategy())
    train = ds.get_train_ids()

    s = ds.load_slice(train[0], 1)
    if s.shape != (10, 10):
        report.error(f"load_slice shape={s.shape}, expected (10,10)", "DAT-004")

    with pytest.raises(IndexError):
        ds.load_slice(train[0], 100)

    report.info(f"load_slice(train[0], 1).shape={s.shape}; OOB index raised IndexError", "DAT-004")
    report.auto_save("DAT004_load_slice_shape_and_bounds", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-007")
def test_partition_ids_cover_all_patients(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="DAT-007: train+val+test ids together cover all patients exactly once")

    ds = MedicalImageDataSource(dataset_root=tmp_path, ingestor=DummyIngestor())
    ds.create_partitions(DummyPartitionStrategy())
    train = ds.get_train_ids()
    val = ds.get_val_ids()
    test = ds.get_test_ids()
    total = len(train) + len(val) + len(test)

    if total != 10:
        report.error(f"train+val+test total={total}, expected 10", "DAT-007")

    report.info(f"train={len(train)}, val={len(val)}, test={len(test)}, total={total}", "DAT-007")
    report.auto_save("DAT007_partition_ids_cover_all_patients", evidence_output_dir)
    assert not report.has_errors, report.summary()

# =========================================================
# Test get_num_patients and get_patient / get_sample
# =========================================================

@pytest.mark.requirement("DAT-006")
def test_get_num_patients_and_get_patient(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="Datasource get_num_patients / get_patient")

    ds = MedicalImageDataSource(tmp_path, DummyIngestor())

    # get_num_patients
    num = ds.get_num_patients()
    if num != 10:
        report.error("get_num_patients returned incorrect count", "DAT-006")

    # get_patient
    patient = ds.get_patient("P0")
    if patient.image_volume.shape != (5, 10, 10):
        report.error("get_patient returned incorrect object", "DAT-006")

    # get_sample fallback
    sample = ds.get_sample("P1")
    if sample.image_volume.shape != (5, 10, 10):
        report.error("get_sample returned incorrect object", "DAT-006")

    report.info(f"get_num_patients={num}, get_patient shape={patient.image_volume.shape}, get_sample shape={sample.image_volume.shape}", "DAT-006")
    report.auto_save("DAT006_get_patient_get_sample", evidence_output_dir)
    assert not report.has_errors, report.summary()


# =========================================================
# Test has_partitions before and after creation
# =========================================================

@pytest.mark.requirement("DAT-007")
def test_has_partitions_flag(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="Datasource has_partitions flag")

    ds = MedicalImageDataSource(tmp_path, DummyIngestor())

    # before partitioning
    if ds.has_partitions():
        report.error("has_partitions True before create_partitions", "DAT-007")

    # after partitioning
    ds.create_partitions(DummyPartitionStrategy())
    if not ds.has_partitions():
        report.error("has_partitions False after create_partitions", "DAT-007")

    report.info("has_partitions()=False before partitioning; True after create_partitions()", "DAT-007")
    report.auto_save("DAT007_has_partitions_flag", evidence_output_dir)
    assert not report.has_errors, report.summary()


# =========================================================
# Test show_slice (basic coverage)
# =========================================================

@pytest.mark.requirement("DAT-004")
def test_show_slice_basic(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="Datasource show_slice validation")

    ds = MedicalImageDataSource(tmp_path, DummyIngestor())

    # slice_index must be valid
    try:
        ds.show_slice("P0", slice_index=0, block = False)
    except Exception as e:
        report.error(f"show_slice raised unexpected exception: {e}", "DAT-004")

    # slice_index out of bounds
    with pytest.raises(IndexError):
        ds.show_slice("P0", slice_index=100, block = False)

    # require ValueError if neither slice_index nor slice_range nor annotated_only
    with pytest.raises(ValueError):
        ds.show_slice("P0", block = False)

    report.info("show_slice valid index succeeded; out-of-bounds raised IndexError; no-selector raised ValueError", "DAT-004")
    report.auto_save("DAT004_show_slice_basic", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-004")
def test_show_slice_with_annotations_renders(
    tmp_path,
    evidence_output_dir,
    monkeypatch,
):
    report = EvidenceReport(subject="DAT-004: show_slice renders annotated slices without raising")

    monkeypatch.setattr(
        "medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource.plt.show",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource.plt.close",
        lambda *args, **kwargs: None,
    )

    annotated_ds = MedicalImageDataSource(tmp_path, AnnotatedDummyIngestor())
    annotated_ds.show_slice("P0", slice_index=2, cols=1, show_annotations=True, show_bboxes=True, block=True)
    annotated_ds.show_slice("P0", slice_range=(1, 3), cols=1, show_annotations=True, show_bboxes=True, block=False)

    no_annotation_ds = MedicalImageDataSource(tmp_path, DummyIngestor())
    no_annotation_ds.show_slice("P0", annotated_only=True, block=False)

    report.info("show_slice with annotations, bboxes, slice_index, and slice_range all rendered without error", "DAT-004")
    report.auto_save("DAT004_show_slice_with_annotations_renders", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-010")
def test_datasource_get_sample_delegates_to_ingestor(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="DAT-010: datasource.get_sample delegates to ingestor.get_sample when defined")

    sample_ds = MedicalImageDataSource(tmp_path, SampleAwareIngestor())
    sample = sample_ds.get_sample("P2")

    if sample != ("custom", "P2"):
        report.error(f"get_sample returned {sample!r}, expected ('custom', 'P2')", "DAT-010")

    report.info(f"get_sample('P2') returned {sample!r} — delegated to ingestor.get_sample", "DAT-010")
    report.auto_save("DAT010_get_sample_delegates_to_ingestor", evidence_output_dir)
    assert not report.has_errors, report.summary()
    
@pytest.mark.requirement("DAT-008")
def test_partitions_are_deterministic(tmp_path, evidence_output_dir):
    
    report = EvidenceReport(subject="Datasource partition determinism")

    ds1 = MedicalImageDataSource(tmp_path, DummyIngestor())
    ds2 = MedicalImageDataSource(tmp_path, DummyIngestor())

    split = DummyPartitionStrategy()

    t1, v1, te1 = ds1.create_partitions(split)
    t2, v2, te2 = ds2.create_partitions(split)

    assert t1 == t2
    assert v1 == v2
    assert te1 == te2

    report.info("Two MedicalImageDataSource instances with the same split strategy produced identical partitions", "DAT-008")
    report.auto_save("DAT008_partitions_are_deterministic", evidence_output_dir)
    assert not report.has_errors, report.summary()

@pytest.mark.requirement("DAT-007")
def test_access_before_partition_raises(tmp_path, evidence_output_dir):
    
    report = EvidenceReport(subject="Datasource access before partitioning")

    ds = MedicalImageDataSource(tmp_path, DummyIngestor())

    with pytest.raises(RuntimeError):
        ds.get_train_ids()

    report.info("get_train_ids() before create_partitions() raised RuntimeError as expected", "DAT-007")
    report.auto_save("DAT007_access_before_partition_raises", evidence_output_dir)
    assert not report.has_errors, report.summary()
    
@pytest.mark.requirement("DAT-004")
def test_invalid_patient_id_raises(tmp_path, evidence_output_dir):
    
    report = EvidenceReport(subject="Datasource invalid patient ID handling")

    ds = MedicalImageDataSource(tmp_path, DummyIngestor())
    ds.create_partitions(DummyPartitionStrategy())

    with pytest.raises(ValueError):
        ds.get_patient("INVALID_ID")

    report.info("get_patient('INVALID_ID') raised exception — unknown patient IDs are rejected", "DAT-004")
    report.auto_save("DAT004_invalid_patient_id_raises", evidence_output_dir)
    assert not report.has_errors, report.summary()
