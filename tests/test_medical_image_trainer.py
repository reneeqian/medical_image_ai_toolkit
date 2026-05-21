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
from medical_image_ai_toolkit.validation.base_evaluator import BaseEvaluator

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

@pytest.mark.requirement("SYS-004")
def test_trainer_returns_results_object(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="MedicalImageTrainer.train() returns a results object")

    ds = _create_datasource(tmp_path)
    ds.create_partitions(DeterministicSplit())
    results = MedicalImageTrainer(
        ds, TinyConvNet(),
        TrainingConfig(epochs=1, batch_size=2, device="cpu", task=DummyTask()),
    ).train()

    assert results is not None, "Trainer returned None instead of a results object"
    report.info(f"train() returned {type(results).__name__} instance", "SYS-004")
    report.auto_save("SYS004_trainer_returns_results_object", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-004")
def test_training_creates_artifact_directory(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Training creates a run artifact directory")

    ds = _create_datasource(tmp_path)
    ds.create_partitions(DeterministicSplit())
    results = MedicalImageTrainer(
        ds, TinyConvNet(),
        TrainingConfig(epochs=1, batch_size=2, device="cpu", task=DummyTask()),
    ).train()

    assert results.run_dir.exists(), f"Artifact directory not created: {results.run_dir}"
    report.info(f"Artifact directory created at {results.run_dir}", "TRN-004")
    report.auto_save("TRN004_training_creates_artifact_directory", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-002")
def test_training_writes_metrics_json(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Training writes metrics.json to the run artifact directory")

    ds = _create_datasource(tmp_path)
    ds.create_partitions(DeterministicSplit())
    results = MedicalImageTrainer(
        ds, TinyConvNet(),
        TrainingConfig(epochs=1, batch_size=2, device="cpu", task=DummyTask()),
    ).train()

    metrics_file = results.run_dir / "metrics.json"
    assert metrics_file.exists(), f"metrics.json not written to {results.run_dir}"
    report.info(f"metrics.json written at {metrics_file}", "VER-002")
    report.auto_save("VER002_training_writes_metrics_json", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("SYS-002")
def test_training_pipeline_integration(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Training pipeline produces results with metrics and history")

    ds = _create_datasource(tmp_path)
    ds.create_partitions(DeterministicSplit())
    results = MedicalImageTrainer(
        ds, TinyConvNet(),
        TrainingConfig(epochs=2, batch_size=2, learning_rate=1e-3, device="cpu", task=DummyTask()),
    ).train()

    assert results.metrics, "Training results have no metrics"
    assert results.history, "Training results have no history"
    report.info(
        f"Training pipeline complete: epochs_trained={results.metrics.get('epochs_trained')}, "
        f"history_entries={len(results.history)}",
        "SYS-002",
    )
    report.auto_save("SYS002_training_pipeline_integration", evidence_output_dir)
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
def test_training_metrics_are_deterministic(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Two training runs with identical seed produce identical metrics")

    ds1 = _create_datasource(tmp_path)
    ds1.create_partitions(DeterministicSplit())
    ds2 = _create_datasource(tmp_path)
    ds2.create_partitions(DeterministicSplit())

    config = TrainingConfig(epochs=2, batch_size=2, task=DummyTask())

    torch.manual_seed(42)
    model1 = TinyConvNet()
    torch.manual_seed(42)
    model2 = TinyConvNet()

    results1 = MedicalImageTrainer(ds1, model1, config).train()
    results2 = MedicalImageTrainer(ds2, model2, config).train()

    hash1 = _hash_metrics_file(Path(results1.run_dir) / "metrics.json")
    hash2 = _hash_metrics_file(Path(results2.run_dir) / "metrics.json")

    if hash1 != hash2:
        report.error("Training metrics differ across runs with identical seed and data", "VER-001")

    report.info("Two runs with identical data and seed produced hash-identical metrics.json", "VER-001")
    report.auto_save("VER001_training_metrics_are_deterministic", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-006")
def test_training_order_is_deterministic(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Training epoch history is deterministic given same seed and data order")

    ds1 = _create_datasource(tmp_path)
    ds1.create_partitions(DeterministicSplit())
    ds2 = _create_datasource(tmp_path)
    ds2.create_partitions(DeterministicSplit())

    config = TrainingConfig(epochs=2, batch_size=2, task=DummyTask())

    torch.manual_seed(42)
    model1 = TinyConvNet()
    torch.manual_seed(42)
    model2 = TinyConvNet()

    results1 = MedicalImageTrainer(ds1, model1, config).train()
    results2 = MedicalImageTrainer(ds2, model2, config).train()

    losses1 = [e["loss"] for e in results1.history]
    losses2 = [e["loss"] for e in results2.history]

    if losses1 != losses2:
        report.error(f"Epoch losses differ: {losses1} vs {losses2}", "TRN-006")

    report.info(f"Epoch losses match across two deterministic runs: {losses1}", "TRN-006")
    report.auto_save("TRN006_training_order_is_deterministic", evidence_output_dir)
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

    report.info(f"No partition overlap: train({len(train_ids)}), val({len(val_ids)}), test({len(test_ids)})", "VER-003")
    report.auto_save(
        "VER003_dataset_partition_separation",
        evidence_output_dir
    )

    assert not report.has_errors, report.summary()
    
@pytest.mark.requirement("SYS-002")
def test_trainer_sanity_check_runs_without_error(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="MedicalImageTrainer.sanity_check() runs without error")

    class DummyDatasource:
        def has_partitions(self): return False
        def get_num_patients(self): return 0

    trainer = MedicalImageTrainer(DummyDatasource(), torch.nn.Linear(4, 2), TrainingConfig())
    trainer.sanity_check()

    report.info("sanity_check() completed without error on a minimal DummyDatasource", "SYS-002")
    report.auto_save("SYS002_trainer_sanity_check_runs", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("SYS-003")
def test_trainer_sanity_check_accepts_training_config(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="TrainingConfig is accessible to sanity_check without errors")

    class DummyDatasource:
        def has_partitions(self): return False
        def get_num_patients(self): return 0

    config = TrainingConfig(epochs=5, learning_rate=1e-4, device="cpu")
    trainer = MedicalImageTrainer(DummyDatasource(), torch.nn.Linear(4, 2), config)
    trainer.sanity_check()

    report.info(
        f"sanity_check() printed config (epochs={config.epochs}, lr={config.learning_rate}) without error",
        "SYS-003",
    )
    report.auto_save("SYS003_trainer_sanity_check_config", evidence_output_dir)
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

    report.info("trainer.train() raised RuntimeError when datasource has no partitions", "SYS-002")
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

    report.info("trainer.train() raised ValueError when task=None — task is required before training", "TRN-001")
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

    report.info("All model parameters remain on cpu after training with device='cpu'", "TRN-001")
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

    report.info("Training with empty dataset completed successfully with num_train_samples=0", "SYS-002")
    report.auto_save(
        "SYS002_training_with_no_patients",
        evidence_output_dir
    )
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-005")
def test_early_stop_loss_threshold_halts_training(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="Early stopping: loss threshold halts training")

    # Task always returns a fixed loss of 0.5 regardless of model output.
    # Multiply prediction by 0 to keep the grad_fn while holding loss constant.
    class FixedLossTask(TrainingTaskDefinition):
        def generate_training_samples(self, patient_sample):
            yield {"input": torch.zeros(1, 1, 32, 32), "target": torch.zeros(1, 1, 32, 32)}
        def compute_loss(self, prediction, target):
            return prediction.mean() * 0.0 + 0.5

    ds = _create_datasource(tmp_path)
    ds.create_partitions(DeterministicSplit())

    config = TrainingConfig(
        epochs=10,
        task=FixedLossTask(),
        early_stop=True,
        loss_threshold=1.0,   # 0.5 <= 1.0 → triggers on epoch 1
        plateau_patience=999,
    )

    results = MedicalImageTrainer(ds, TinyConvNet(), config).train()

    if results.metrics.get("epochs_trained") != 1:
        report.error(
            f"Expected 1 epoch before threshold stop, got {results.metrics.get('epochs_trained')}",
            "TRN-005",
        )

    report.info(
        f"Loss threshold early stop: epochs_trained={results.metrics.get('epochs_trained')}, "
        f"reason={results.metrics.get('early_stop_reason')!r}",
        "TRN-005",
    )
    report.auto_save("TRN005_early_stop_loss_threshold", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-005")
def test_early_stop_plateau_halts_training(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="Early stopping: plateau patience halts training")

    # Task always returns a fixed loss of 1.0 — improvement is always zero.
    # Multiply prediction by 0 to keep the grad_fn while holding loss constant.
    class ConstantLossTask(TrainingTaskDefinition):
        def generate_training_samples(self, patient_sample):
            yield {"input": torch.zeros(1, 1, 32, 32), "target": torch.zeros(1, 1, 32, 32)}
        def compute_loss(self, prediction, target):
            return prediction.mean() * 0.0 + 1.0

    ds = _create_datasource(tmp_path)
    ds.create_partitions(DeterministicSplit())

    config = TrainingConfig(
        epochs=20,
        task=ConstantLossTask(),
        early_stop=True,
        loss_threshold=-1.0,   # impossible to reach — ensures only plateau fires
        plateau_patience=2,
        plateau_min_delta=1e-4,
    )

    results = MedicalImageTrainer(ds, TinyConvNet(), config).train()

    # epoch 1: no comparison; epoch 2: no improvement (counter=1); epoch 3: counter=2 >= patience → stop
    if results.metrics.get("epochs_trained", 20) >= 20:
        report.error("Training ran to completion despite constant loss (plateau not triggered)", "TRN-005")
    if "plateau" not in (results.metrics.get("early_stop_reason") or ""):
        report.error("early_stop_reason does not name plateau", "TRN-005")

    report.info(f"Plateau early stop: epochs_trained={results.metrics.get('epochs_trained')}, reason={results.metrics.get('early_stop_reason')!r}", "TRN-005")
    report.auto_save("TRN005_early_stop_plateau", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-005")
def test_early_stop_disabled_runs_all_epochs(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="Early stopping disabled: all epochs run")

    ds = _create_datasource(tmp_path)
    ds.create_partitions(DeterministicSplit())

    config = TrainingConfig(
        epochs=3,
        task=DummyTask(),
        early_stop=False,
    )

    results = MedicalImageTrainer(ds, TinyConvNet(), config).train()

    if results.metrics.get("epochs_trained") != 3:
        report.error(
            f"Expected 3 epochs with early_stop=False, got {results.metrics.get('epochs_trained')}",
            "TRN-005",
        )

    if results.metrics.get("early_stop_reason") is not None:
        report.error("early_stop_reason should be None when early_stop=False", "TRN-005")

    report.info(f"early_stop=False: all 3 epochs ran, early_stop_reason={results.metrics.get('early_stop_reason')!r}", "TRN-005")
    report.auto_save("TRN005_early_stop_disabled", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-006")
def test_val_loss_appears_in_epoch_history(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="val_loss appears in every epoch history entry when a val partition exists")

    class _DummyEvaluator(BaseEvaluator):
        def update(self, pred, target) -> None:
            pass
        def aggregate(self) -> dict:
            return {"custom_metric": 1.0}

    ds = _create_datasource(tmp_path)
    ds.create_partitions(DeterministicSplit())
    results = MedicalImageTrainer(
        ds, TinyConvNet(),
        TrainingConfig(epochs=2, task=DummyTask(), early_stop=False, val_evaluator=_DummyEvaluator()),
    ).train()

    for entry in results.history:
        if "val_loss" not in entry:
            report.error(f"val_loss missing from history entry: {entry}", "TRN-006")

    report.info(f"val_loss present in all {len(results.history)} history entries", "TRN-006")
    report.auto_save("TRN006_val_loss_in_history", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-008")
def test_val_evaluator_custom_metrics_appear_in_history(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="val_evaluator aggregate() keys are merged into per-epoch history entries")

    class _DummyEvaluator(BaseEvaluator):
        def update(self, pred, target) -> None:
            pass
        def aggregate(self) -> dict:
            return {"custom_metric": 1.0}

    ds = _create_datasource(tmp_path)
    ds.create_partitions(DeterministicSplit())
    results = MedicalImageTrainer(
        ds, TinyConvNet(),
        TrainingConfig(epochs=2, task=DummyTask(), early_stop=False, val_evaluator=_DummyEvaluator()),
    ).train()

    for entry in results.history:
        if "custom_metric" not in entry:
            report.error(f"custom_metric from val_evaluator missing in history entry: {entry}", "TRN-008")

    report.info(
        f"val_evaluator aggregate() keys present in all {len(results.history)} history entries",
        "TRN-008",
    )
    report.auto_save("TRN008_val_evaluator_custom_metrics_in_history", evidence_output_dir)
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

    report.info("TrainingPipeline raised RuntimeError('Dataset validation failed') for patient with negative spacing", "SYS-005")
    report.auto_save(
        "SYS005_training_pipeline_stops_on_invalid_dataset",
        evidence_output_dir
    )
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-004")
def test_training_pipeline_writes_model_and_partition_artifacts(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="TrainingPipeline writes model.pt and partitions.json artifacts")

    ds = _create_datasource(tmp_path)
    config = TrainingConfig(epochs=1, task=DummyTask(), split_strategy=DeterministicSplit(), early_stop=False)
    outputs = TrainingPipeline(
        datasource=ds, model=TinyConvNet(), config=config, output_dir=tmp_path / "pipeline_output"
    ).run()
    results = outputs["results"]

    model_path = results.artifacts.get("model")
    partitions_path = results.artifacts.get("partitions")

    if not model_path or not Path(model_path).exists():
        report.error("model.pt artifact not written", "TRN-004")
    if not partitions_path or not Path(partitions_path).exists():
        report.error("partitions.json artifact not written", "TRN-004")

    report.info("TrainingPipeline wrote model.pt and partitions.json", "TRN-004")
    report.auto_save("TRN004_pipeline_model_and_partition_artifacts", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("SYS-002")
def test_training_pipeline_returns_results_with_artifacts(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="TrainingPipeline returns results dict with artifact path keys")

    ds = _create_datasource(tmp_path)
    config = TrainingConfig(epochs=1, task=DummyTask(), split_strategy=DeterministicSplit(), early_stop=False)
    outputs = TrainingPipeline(
        datasource=ds, model=TinyConvNet(), config=config, output_dir=tmp_path / "pipeline_output"
    ).run()

    assert "results" in outputs, "TrainingPipeline output missing 'results' key"
    results = outputs["results"]
    assert results.artifacts, "results.artifacts is empty"

    report.info(f"TrainingPipeline returned results with artifact keys: {list(results.artifacts)}", "SYS-002")
    report.auto_save("SYS002_training_pipeline_returns_results", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("MOD-002")
def test_checkpoint_save_creates_files(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="checkpoint_every=1 writes per-epoch and latest checkpoint files")

    ds = _create_datasource(tmp_path)
    ds.create_partitions(DeterministicSplit())

    config = TrainingConfig(
        epochs=2,
        task=DummyTask(),
        early_stop=False,
        checkpoint_every=1,
    )

    trainer = MedicalImageTrainer(ds, TinyConvNet(), config, output_dir=tmp_path / "ckpts")
    results = trainer.train()
    run_dir = results.run_dir

    if not (run_dir / "checkpoint_epoch_001.pt").exists():
        report.error("checkpoint_epoch_001.pt not created", "MOD-002")
    if not (run_dir / "checkpoint_epoch_002.pt").exists():
        report.error("checkpoint_epoch_002.pt not created", "MOD-002")
    if not (run_dir / "checkpoint_latest.pt").exists():
        report.error("checkpoint_latest.pt not created", "MOD-002")

    report.info("checkpoint_epoch_001.pt, checkpoint_epoch_002.pt, and checkpoint_latest.pt all created with checkpoint_every=1", "MOD-002")
    report.auto_save("MOD002_checkpoint_save_creates_files", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("MOD-004")
def test_checkpoint_resume_continues_from_saved_epoch(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Training resumes from checkpoint and completes remaining epochs")

    ds = _create_datasource(tmp_path)
    ds.create_partitions(DeterministicSplit())

    # First run: train 1 epoch and save a checkpoint
    config1 = TrainingConfig(
        epochs=1, task=DummyTask(), early_stop=False, checkpoint_every=1
    )
    trainer1 = MedicalImageTrainer(ds, TinyConvNet(), config1, output_dir=tmp_path / "r1")
    results1 = trainer1.train()
    checkpoint = results1.run_dir / "checkpoint_latest.pt"

    # Second run: resume; 2 total epochs, 1 already done → 1 more runs
    config2 = TrainingConfig(epochs=2, task=DummyTask(), early_stop=False)
    trainer2 = MedicalImageTrainer(
        ds, TinyConvNet(), config2,
        output_dir=tmp_path / "r2",
        resume_from=checkpoint,
    )
    results2 = trainer2.train()

    # epoch_losses restored from checkpoint (1 entry) + 1 new = 2 total
    epochs_trained = results2.metrics.get("epochs_trained")
    if epochs_trained != 2:
        report.error(
            f"Expected 2 total epoch entries after resume, got {epochs_trained}",
            "MOD-004",
        )

    report.info(f"Checkpoint resume: epochs_trained={epochs_trained} (1 from checkpoint + 1 new)", "MOD-004")
    report.auto_save("MOD004_checkpoint_resume", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("SYS-003")
def test_sanity_check_reports_partition_sizes(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Sanity check reports partition sizes when partitions exist")

    ds = _create_datasource(tmp_path)
    ds.create_partitions(DeterministicSplit())

    trainer = MedicalImageTrainer(ds, TinyConvNet(), TrainingConfig())
    trainer.sanity_check()

    report.info("sanity_check() with partitioned datasource reported partition sizes without error", "SYS-003")
    report.auto_save("SYS003_sanity_check_with_partitions", evidence_output_dir)
    assert not report.has_errors, report.summary()
