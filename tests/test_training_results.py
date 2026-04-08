import torch
import pytest
from pathlib import Path
import numpy as np

from regulatory_tools.evidence.evidence_report import EvidenceReport
from medical_image_ai_toolkit.results.medical_image_training_results import MedicalImageTrainingResults
from medical_image_ai_toolkit.training.medical_image_trainer import MedicalImageTrainer
from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import MedicalImageDataSource
from medical_image_ai_toolkit.training.training_config import TrainingConfig
from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition
from medical_image_ai_toolkit.dataobjects.annotation_bundle import AnnotationBundle, VectorROI
from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample


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
# Tests
# ---------------------------------------------------------

@pytest.mark.requirement("MOD-003")
@pytest.mark.requirement("MOD-005")
@pytest.mark.requirement("DOC-004")
@pytest.mark.requirement("VER-002")
def test_training_results_artifact_generation(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="Training results artifact generation")

    model = torch.nn.Linear(4, 2)

    class DummyConfig:
        epochs = 2

    class DummyDatasource:
        pass

    results = MedicalImageTrainingResults(
        model,
        DummyConfig(),
        DummyDatasource(),
        tmp_path
    )

    results.metrics = {"loss": 0.1}
    results.history.append({"epoch": 1, "loss": 0.1})

    report_path = results.generate_training_report()

    if not Path(report_path).exists():
        report.error("Training report not generated", "DOC-004")

    model_path = tmp_path / "model.pt"
    results.export_model(model_path)

    if not model_path.exists():
        report.error("Model export failed", "MOD-005")

    report.auto_save(
        "MOD003_MOD005_training_results_artifacts",
        evidence_output_dir
    )

    assert not report.has_errors, report.summary()
    

@pytest.mark.requirement("MOD-006")
@pytest.mark.requirement("VER-007")
def test_inference_determinism(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="Inference determinism")

    model = torch.nn.Linear(4, 2)

    class DummyConfig:
        pass

    class DummyDatasource:
        pass

    results = MedicalImageTrainingResults(
        model,
        DummyConfig(),
        DummyDatasource(),
        tmp_path
    )

    data = [
        [1,2,3,4],
        [4,3,2,1]
    ]

    out1 = results.run_inference(data)
    out2 = results.run_inference(data)

    if len(out1) != len(out2):
        report.error("Inference output length mismatch", "MOD-006")

    for a,b in zip(out1,out2):
        if not torch.allclose(a,b):
            report.error("Inference not deterministic", "VER-007")

    report.auto_save(
        "MOD006_VER007_inference_determinism",
        evidence_output_dir
    )

    assert not report.has_errors, report.summary()


@pytest.mark.requirement("MOD-006")
def test_results_inference(tmp_path, evidence_output_dir):

    report = EvidenceReport(
        subject="Training results inference"
    )

    model = torch.nn.Linear(4,2)

    class Config: pass
    class DS: pass

    results = MedicalImageTrainingResults(
        model,
        Config(),
        DS(),
        tmp_path
    )

    x = torch.randn(1,4)

    y = results.model(x)

    if y.shape != (1,2):
        report.error("Model inference failed", "MOD-006")

    report.auto_save(
        "MOD006_results_inference",
        evidence_output_dir
    )

    assert not report.has_errors, report.summary()

@pytest.mark.requirement("DOC-004")
def test_summary_training_running(tmp_path, evidence_output_dir):
    
    report = EvidenceReport(subject="Summary during training")

    model = torch.nn.Linear(4,2)

    class Config: pass
    class DS: pass

    results = MedicalImageTrainingResults(
        model,
        Config(),
        DS(),
        tmp_path
    )

    # no training_end_time set
    results.summary()
    
    report.auto_save(
        "DOC004_summary_during_training",
        evidence_output_dir
    )
    assert not report.has_errors, report.summary()

@pytest.mark.requirement("DOC-004")
def test_mark_training_complete_summary(tmp_path, evidence_output_dir):
    
    report = EvidenceReport(subject="Mark training complete summary")

    model = torch.nn.Linear(4,2)

    class Config: pass
    class DS: pass

    results = MedicalImageTrainingResults(
        model,
        Config(),
        DS(),
        tmp_path
    )

    from datetime import datetime
    results.training_start_time = datetime.now()

    results.mark_training_complete()

    results.summary()

    assert hasattr(results, "training_end_time")
    
    report.auto_save(
        "DOC004_mark_training_complete_summary",
        evidence_output_dir
    )
    assert not report.has_errors, report.summary()

@pytest.mark.requirement("MOD-005")
def test_export_model_failure(tmp_path, evidence_output_dir):
    
    report = EvidenceReport(subject="Model export failure handling")

    class BadModel:
        def state_dict(self):
            raise RuntimeError("bad model")

    class Config: pass
    class DS: pass

    results = MedicalImageTrainingResults(
        BadModel(),
        Config(),
        DS(),
        tmp_path
    )

    results.export_model(tmp_path / "model.pt")
    
    report.auto_save(
        "MOD005_model_export_failure",
        evidence_output_dir
    )
    assert not report.has_errors, report.summary()

@pytest.mark.requirement("SYS-004")
def test_results_artifact_registration(tmp_path, evidence_output_dir):
    
    report = EvidenceReport(subject="Results artifact registration")

    ds = MedicalImageDataSource(tmp_path, SyntheticIngestor())
    ds.create_partitions(DeterministicSplit())

    model = TinyConvNet()

    config = TrainingConfig(
        epochs=1,
        task=DummyTask()
    )

    trainer = MedicalImageTrainer(ds, model, config)

    results = trainer.train()

    assert "metrics" in results.artifacts
    
    report.auto_save(
        "SYS004_results_artifact_registration",
        evidence_output_dir
    )
    assert not report.has_errors, report.summary()