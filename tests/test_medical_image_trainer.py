from pathlib import Path
import json
import hashlib
import numpy as np
import torch
import pytest

from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
from medical_image_ai_toolkit.dataobjects.annotation_bundle import AnnotationBundle, VectorROI
from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import MedicalImageDataSource
from medical_image_ai_toolkit.pipeline.training_pipeline import TrainingPipeline
from medical_image_ai_toolkit.training.medical_image_trainer import MedicalImageTrainer
from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition
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

    def load_patient_sample(self, patient_id):

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
    def get_sample(self, patient_id):

        sample = self.load_patient_sample(patient_id)

        volume = sample.image_volume

        z = volume.shape[0] // 2

        img = volume[z][16:48,16:48]

        x = img.astype(np.float32)[None, None, :, :]  # N,C,H,W

        label = 1 if sample.annotations.vector_rois else 0

        y = np.array([[label]], dtype=np.float32)

        return x, y

# ---------------------------------------------------------
# Dummy Task Definition
# ---------------------------------------------------------

class DummyTask(TrainingTaskDefinition):

    def generate_training_samples(self, patient_sample):
        for _ in range(2):
            yield {
                "input": torch.zeros((1, 1, 32, 32)),
                "target": torch.ones((1, 1, 32, 32))
            }

    def compute_loss(self, prediction, target):
        return torch.mean((prediction - target) ** 2)

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
        learning_rate=1e-3,
        device="cpu",
        task=DummyTask()
    )

    trainer = MedicalImageTrainer(
        ds,
        model,
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

    class NaNTask(TrainingTaskDefinition):

        def generate_training_samples(self, patient_sample):

            # propagate real data (which may contain NaNs)
            vol = patient_sample.image_volume

            for i in range(vol.shape[0]):

                img = torch.tensor(vol[i]).unsqueeze(0).unsqueeze(0)
                target = torch.zeros_like(img)

                yield {
                    "input": img,
                    "target": target
                }

        def compute_loss(self, prediction, target):
            return torch.mean((prediction - target) ** 2)

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
        batch_size=2,
        task=NaNTask()
    )

    trainer = MedicalImageTrainer(
        ds,
        model,
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
        batch_size=2,
        task=DummyTask(),
    )

    model1 = TinyConvNet()
    model2 = TinyConvNet()

    trainer1 = MedicalImageTrainer(ds1, model1, config)
    trainer2 = MedicalImageTrainer(ds2, model2, config)

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

    model = torch.nn.Linear(4,2)

    config = TrainingConfig()

    trainer = MedicalImageTrainer(
        DummyDatasource(),
        model,
        config
    )

    trainer.sanity_check()

    report.auto_save(
        "SYS002_SYS003_trainer_sanity_check",
        evidence_output_dir
    )

    assert not report.has_errors, report.summary()

@pytest.mark.requirement("SYS-002")
def test_training_without_partitions_fails(tmp_path, evidence_output_dir):
    
    report = EvidenceReport(subject="Training without dataset partitions should fail")

    ds = MedicalImageDataSource(tmp_path, SyntheticIngestor())

    model = TinyConvNet()

    config = TrainingConfig(
        epochs=1,
        task=DummyTask()
    )

    trainer = MedicalImageTrainer(ds, model, config)

    with pytest.raises(RuntimeError):
        trainer.train()
    
    report.auto_save(
        "SYS002_training_without_partitions_fails",
        evidence_output_dir
    )
    assert not report.has_errors, report.summary()

@pytest.mark.requirement("TRN-001")
def test_training_without_task_fails(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="Training without task should fail")

    ds = MedicalImageDataSource(tmp_path, SyntheticIngestor())
    ds.create_partitions(DeterministicSplit())

    model = TinyConvNet()

    config = TrainingConfig(
        epochs=1,
        task=None
    )

    trainer = MedicalImageTrainer(ds, model, config)

    with pytest.raises(ValueError):
        trainer.train()
    
    report.auto_save(
        "TRN001_training_without_task_fails",
        evidence_output_dir
    )
    assert not report.has_errors, report.summary()

@pytest.mark.requirement("TRN-001")
def test_training_respects_device(tmp_path, evidence_output_dir):
    
    report = EvidenceReport(subject="Training respects device configuration")

    ds = MedicalImageDataSource(tmp_path, SyntheticIngestor())
    ds.create_partitions(DeterministicSplit())

    model = TinyConvNet()

    config = TrainingConfig(
        epochs=1,
        device="cpu",
        task=DummyTask()
    )

    trainer = MedicalImageTrainer(ds, model, config)

    trainer.train()

    for p in model.parameters():
        assert p.device.type == "cpu"

    report.auto_save(
        "TRN001_training_respects_device",
        evidence_output_dir
    )
    assert not report.has_errors, report.summary()

@pytest.mark.requirement("SYS-002")
def test_training_with_no_patients(tmp_path, evidence_output_dir):
    
    report = EvidenceReport(subject="Training with no patients should complete without error")

    class EmptyIngestor:
        def list_patient_ids(self): return []
        def load_patient_sample(self, pid): pass

    ds = MedicalImageDataSource(tmp_path, EmptyIngestor())
    ds.create_partitions(DeterministicSplit())

    model = TinyConvNet()

    config = TrainingConfig(
        epochs=1,
        task=DummyTask()
    )

    trainer = MedicalImageTrainer(ds, model, config)

    results = trainer.train()

    assert results.metrics["num_train_samples"] == 0
    
    report.auto_save(
        "SYS002_training_with_no_patients",
        evidence_output_dir
    )
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("SYS-005")
def test_training_pipeline_stops_on_dataset_validation_errors(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="Training pipeline should stop on dataset validation errors")

    class InvalidIngestor:
        def list_patient_ids(self):
            return ["P0"]

        def load_patient_sample(self, patient_id):
            return PatientSample(
                image_volume=np.zeros((8, 16, 16), dtype=np.float32),
                spacing=(1.0, -1.0, 1.0),
                annotations=None,
                patient_id=patient_id,
                metadata={},
            )

    ds = MedicalImageDataSource(tmp_path, InvalidIngestor())

    config = TrainingConfig(
        epochs=1,
        task=DummyTask(),
        split_strategy=DeterministicSplit(),
    )

    pipeline = TrainingPipeline(
        datasource=ds,
        model=TinyConvNet(),
        config=config,
    )

    with pytest.raises(RuntimeError, match="Dataset validation failed"):
        pipeline.run()

    report.auto_save(
        "SYS005_training_pipeline_stops_on_invalid_dataset",
        evidence_output_dir
    )
    assert not report.has_errors, report.summary()
