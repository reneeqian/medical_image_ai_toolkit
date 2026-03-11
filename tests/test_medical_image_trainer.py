from pathlib import Path
import json
import hashlib
import numpy as np
import torch
import pytest

from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
from medical_image_ai_toolkit.dataobjects.annotation_bundle import AnnotationBundle, VectorROI
from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import MedicalImageDataSource
from medical_image_ai_toolkit.training.medical_image_trainer import MedicalImageTrainer
from medical_image_ai_toolkit.training.training_config import TrainingConfig

from regulatory_tools.evidence.evidence_report import EvidenceReport

# ---------------------------------------------------------
# Synthetic Dataset Ingestor
# ---------------------------------------------------------

class SyntheticIngestor:
    """
    Deterministic synthetic datasource for trainer tests.

    Produces valid PatientSample objects that follow the real
    toolkit data contracts.
    """

    def __init__(self, num_patients=10, nan_patient=False):
        self.num_patients = num_patients
        self.nan_patient = nan_patient
        self.patient_ids = [f"P{i}" for i in range(num_patients)]

    def list_patient_ids(self):
        return self.patient_ids

    def load_patient(self, patient_id):

        # deterministic random seed per patient
        rng = np.random.default_rng(abs(hash(patient_id)) % (2**32))

        # synthetic CT volume
        volume = rng.normal(
            loc=0,
            scale=300,
            size=(8, 64, 64)
        ).astype(np.float32)

        # introduce NaNs if requested
        if self.nan_patient and patient_id == "P0":
            volume[:] = np.nan

        spacing = (1.5, 0.7, 0.7)

        # create simple ROI annotation on slice 4
        contour = np.array([
            [20, 20],
            [40, 20],
            [40, 40],
            [20, 40]
        ], dtype=np.float32)

        roi = VectorROI(
            slice_index=4,
            contour_px=contour,
            label="synthetic_lesion",
        )

        annotations = AnnotationBundle(
            vector_rois={4: [roi]},
            segmentation_masks=None,
            label_map={"synthetic_lesion": 1},
        )

        return PatientSample(
            image_volume=volume,
            spacing=spacing,
            annotations=annotations,
            patient_id=patient_id,
            metadata={"source": "synthetic_test"},
        )

# ---------------------------------------------------------
# Synthetic Task Definition
# ---------------------------------------------------------

class SyntheticClassificationTask:
    """
    Minimal task used only for trainer tests.

    Converts a PatientSample into:
    x = center slice cropped to 32x32
    y = binary label indicating lesion presence
    """

    def prepare_training_sample(self, sample):

        volume = sample.image_volume

        # use middle slice
        z = volume.shape[0] // 2
        img = volume[z]

        # crop to 32x32 for TinyConvNet
        img = img[16:48, 16:48]

        x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # label = lesion present
        label = 1 if sample.annotations.vector_rois else 0
        y = torch.tensor([label], dtype=torch.float32)

        return x, y

    def compute_loss(self, pred, y):

        return torch.nn.functional.mse_loss(pred.squeeze(), y.squeeze())

# ---------------------------------------------------------
# Deterministic Partition Strategy
# ---------------------------------------------------------

class DeterministicSplit:

    def split(self, ids):

        n = len(ids)

        train = ids[: int(n * 0.6)]
        val = ids[int(n * 0.6): int(n * 0.8)]
        test = ids[int(n * 0.8):]

        return train, val, test


# ---------------------------------------------------------
# Minimal Model
# ---------------------------------------------------------

class TinyConvNet(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 32, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _hash_metrics_file(metrics_path: Path):

    payload = json.loads(metrics_path.read_text())
    encoded = json.dumps(payload, sort_keys=True).encode()

    return hashlib.sha256(encoded).hexdigest()


def _create_datasource(tmp_path, nan_patient=False):

    ingestor = SyntheticIngestor(
        num_patients=10,
        nan_patient=nan_patient
    )

    return MedicalImageDataSource(
        dataset_root=tmp_path,
        ingestor=ingestor
    )


# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------

@pytest.mark.requirement("SYS-002")
@pytest.mark.requirement("TRN-004")
@pytest.mark.requirement("SYS-004")
@pytest.mark.requirement("VER-002")
def test_training_pipeline_generates_artifacts(
    tmp_path,
    evidence_output_dir,
):

    report = EvidenceReport(
        subject="Training pipeline artifact generation"
    )

    ds = _create_datasource(tmp_path)

    ds.create_partitions(DeterministicSplit())

    model = TinyConvNet()

    config = TrainingConfig(
        epochs=2,
        batch_size=2,
        learning_rate=1e-3
    )

    task = SyntheticClassificationTask()

    trainer = MedicalImageTrainer(
        ds,
        model,
        task,
        config
    )

    results = trainer.train()

    if results is None:
        report.error(
            "Trainer returned no results object",
            "SYS-004"
        )

    artifact_dir = results.run_dir

    if not Path(artifact_dir).exists():
        report.error(
            "Artifact directory not created",
            "TRN-004"
        )

    metrics_file = Path(artifact_dir) / "metrics.json"

    if not metrics_file.exists():
        report.error(
            "Metrics file missing",
            "VER-002"
        )

    report.auto_save(
        "SYS002_TRN004_training_pipeline_generates_artifacts",
        evidence_output_dir
    )

    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-003")
def test_training_detects_nan_loss(
    tmp_path,
    evidence_output_dir,
):

    report = EvidenceReport(
        subject="NaN loss detection during training"
    )

    ds = _create_datasource(
        tmp_path,
        nan_patient=True
    )

    ds.create_partitions(DeterministicSplit())

    model = TinyConvNet()

    config = TrainingConfig(
        epochs=1,
        batch_size=2
    )

    task = SyntheticClassificationTask()

    trainer = MedicalImageTrainer(
        ds,
        model,
        task,
        config
    )

    with pytest.raises(RuntimeError):

        trainer.train()

    report.info(
        "Trainer correctly detected NaN loss",
        "TRN-003"
    )

    report.auto_save(
        "TRN003_nan_loss_detection",
        evidence_output_dir
    )

    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-001")
@pytest.mark.requirement("TRN-006")
def test_training_is_deterministic(
    tmp_path,
    evidence_output_dir,
):

    report = EvidenceReport(
        subject="Deterministic training verification"
    )

    ds1 = _create_datasource(tmp_path)
    ds1.create_partitions(DeterministicSplit())

    ds2 = _create_datasource(tmp_path)
    ds2.create_partitions(DeterministicSplit())

    config = TrainingConfig(
        epochs=2,
        batch_size=2
    )

    model1 = TinyConvNet()
    model2 = TinyConvNet()
    
    task1 = SyntheticClassificationTask()
    task2 = SyntheticClassificationTask()

    trainer1 = MedicalImageTrainer(ds1, model1, task1, config)
    trainer2 = MedicalImageTrainer(ds2, model2, task2, config)

    results1 = trainer1.train()
    results2 = trainer2.train()

    hash1 = _hash_metrics_file(Path(results1.run_dir) / "metrics.json")
    hash2 = _hash_metrics_file(Path(results2.run_dir) / "metrics.json")

    if hash1 != hash2:

        report.error(
            "Training metrics differ across deterministic runs",
            "VER-001"
        )

    report.auto_save(
        "VER001_training_is_deterministic",
        evidence_output_dir
    )

    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-003")
def test_dataset_partitions_do_not_overlap(
    tmp_path,
    evidence_output_dir,
):

    report = EvidenceReport(
        subject="Dataset partition separation enforcement"
    )

    ds = _create_datasource(tmp_path)

    train_ids, val_ids, test_ids = ds.create_partitions(
        DeterministicSplit()
    )

    if set(train_ids) & set(val_ids):
        report.error("Train/Val overlap", "VER-003")

    if set(train_ids) & set(test_ids):
        report.error("Train/Test overlap", "VER-003")

    if set(val_ids) & set(test_ids):
        report.error("Val/Test overlap", "VER-003")

    report.auto_save(
        "VER003_dataset_partition_separation",
        evidence_output_dir
    )

    assert not report.has_errors, report.summary()
    
@pytest.mark.requirement("SYS-002")
@pytest.mark.requirement("SYS-003")
def test_trainer_sanity_check(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="Trainer sanity check")

    class DummyDatasource:

        def has_partitions(self): return False

        def get_num_patients(self): return 0

    class DummyTask:
        pass

    model = torch.nn.Linear(4,2)

    config = TrainingConfig()

    trainer = MedicalImageTrainer(
        DummyDatasource(),
        model,
        DummyTask(),
        config
    )

    trainer.sanity_check()

    report.auto_save(
        "SYS002_SYS003_trainer_sanity_check",
        evidence_output_dir
    )

    assert not report.has_errors, report.summary()