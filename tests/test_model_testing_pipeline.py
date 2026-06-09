import math
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from regulatory_tools.evidence.evidence_report import EvidenceReport

from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import (
    MedicalImageDataSource,
)
from medical_image_ai_toolkit.pipeline.model_testing_pipeline import ModelTestingPipeline
from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition
from medical_image_ai_toolkit.training.training_config import TrainingConfig
from medical_image_ai_toolkit.validation.base_evaluator import BaseEvaluator
from medical_image_ai_toolkit.validation.segmentation_evaluator import SegmentationEvaluator


@dataclass
class ModelTestingPatient:
    patient_id: str
    sample_masks: list[torch.Tensor]


class ModelTestingIngestor:
    def __init__(self):
        self.patient_ids = ["P0", "P1", "P2", "P3"]

    def list_patient_ids(self):
        return self.patient_ids

    def load_patient_sample(self, patient_id):
        mask_a = torch.tensor(
            [[[[1.0, 0.0], [0.0, 1.0]]]],
            dtype=torch.float32,
        )
        mask_b = torch.tensor(
            [[[[0.0, 1.0], [1.0, 0.0]]]],
            dtype=torch.float32,
        )
        return ModelTestingPatient(patient_id=patient_id, sample_masks=[mask_a, mask_b])


class DeterministicSplit:
    def split(self, ids):
        return ids[:2], ids[2:3], ids[3:]


class IdentityTask(TrainingTaskDefinition):
    def generate_training_samples(self, patient_sample):
        for mask in patient_sample.sample_masks:
            yield {
                "input": mask.clone(),
                "target": mask.clone(),
            }

    def compute_loss(self, prediction, target):
        return torch.mean((prediction - target) ** 2)


class IdentityModel(torch.nn.Module):
    def forward(self, x):
        return x


class CountingEvaluator(BaseEvaluator):
    def __init__(self):
        self.count = 0

    def update(self, pred, target) -> None:
        self.count += 1

    def aggregate(self) -> dict:
        return {"sample_updates": self.count}

    def reset(self) -> None:
        self.count = 0


@pytest.mark.requirement("VER-004")
def test_model_testing_pipeline_patient_and_sample_counts(
    tmp_path,
    evidence_output_dir,
):
    report = EvidenceReport(subject="Model testing pipeline — patient and sample counts")

    datasource = MedicalImageDataSource(tmp_path, ModelTestingIngestor())
    datasource.create_partitions(DeterministicSplit())

    results = ModelTestingPipeline(
        datasource=datasource,
        model=IdentityModel(),
        config=TrainingConfig(device="cpu", task=IdentityTask()),
        output_dir=tmp_path / "model_testing_runs",
    ).run()

    if results.metrics.get("num_test_patients") != 1:
        report.error(
            f"Expected num_test_patients=1, got {results.metrics.get('num_test_patients')}",
            "VER-004",
        )
    if results.metrics.get("num_test_samples") != 2:
        report.error(
            f"Expected num_test_samples=2, got {results.metrics.get('num_test_samples')}",
            "VER-004",
        )

    report.info(
        f"num_test_patients={results.metrics.get('num_test_patients')}, "
        f"num_test_samples={results.metrics.get('num_test_samples')}",
        "VER-004",
    )
    report.auto_save("VER004_model_testing_patient_and_sample_counts", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-005")
def test_model_testing_identity_model_produces_perfect_metrics(
    tmp_path,
    evidence_output_dir,
):
    report = EvidenceReport(
        subject="Model testing pipeline — identity model yields zero loss and perfect segmentation metrics"
    )

    datasource = MedicalImageDataSource(tmp_path, ModelTestingIngestor())
    datasource.create_partitions(DeterministicSplit())

    results = ModelTestingPipeline(
        datasource=datasource,
        model=IdentityModel(),
        config=TrainingConfig(device="cpu", task=IdentityTask()),
        output_dir=tmp_path / "model_testing_runs",
    ).run()

    if results.metrics.get("mean_loss") != 0.0:
        report.error(
            f"Expected mean_loss=0.0 for identity model, got {results.metrics.get('mean_loss')}",
            "VER-005",
        )
    for key in ("dice", "iou", "precision", "recall", "accuracy"):
        if results.metrics.get(key) != 1.0:
            report.error(
                f"Expected {key}=1.0 for identity model, got {results.metrics.get(key)}",
                "VER-005",
            )

    report.info(
        f"Identity model: mean_loss={results.metrics.get('mean_loss')}, "
        f"dice={results.metrics.get('dice')}, iou={results.metrics.get('iou')}",
        "VER-005",
    )
    report.auto_save("VER005_model_testing_identity_model_metrics", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-006")
def test_model_testing_pipeline_generates_report_artifact(
    tmp_path,
    evidence_output_dir,
):
    report = EvidenceReport(
        subject="Model testing pipeline — report artifact and per-patient results"
    )

    datasource = MedicalImageDataSource(tmp_path, ModelTestingIngestor())
    datasource.create_partitions(DeterministicSplit())

    results = ModelTestingPipeline(
        datasource=datasource,
        model=IdentityModel(),
        config=TrainingConfig(device="cpu", task=IdentityTask()),
        output_dir=tmp_path / "model_testing_runs",
    ).run()

    report_path = results.artifacts.get("testing_report")
    if report_path is None or not report_path.exists():
        report.error("Model testing report artifact was not generated", "VER-006")

    if len(results.per_patient_results) != 1:
        report.error(
            f"Expected 1 per-patient result, got {len(results.per_patient_results)}",
            "VER-006",
        )

    report.info(
        f"Testing report artifact at {report_path}; "
        f"per_patient_results has {len(results.per_patient_results)} entries",
        "VER-006",
    )
    report.auto_save("VER006_model_testing_report_artifact", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-004")
def test_model_testing_pipeline_requires_existing_partitions(
    tmp_path,
    evidence_output_dir,
):

    report = EvidenceReport(subject="Model testing pipeline partition requirement")

    datasource = MedicalImageDataSource(tmp_path, ModelTestingIngestor())

    pipeline = ModelTestingPipeline(
        datasource=datasource,
        model=IdentityModel(),
        config=TrainingConfig(device="cpu", task=IdentityTask()),
        output_dir=tmp_path / "model_testing_runs",
    )

    with pytest.raises(RuntimeError):
        pipeline.run()

    report.info(
        "ModelTestingPipeline raised RuntimeError when datasource has no partitions", "VER-004"
    )
    report.auto_save(
        "VER004_model_testing_requires_partitions",
        evidence_output_dir,
    )
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-006")
def test_model_testing_pipeline_requires_task_definition(
    tmp_path,
    evidence_output_dir,
):

    report = EvidenceReport(subject="Model testing pipeline task requirement")

    datasource = MedicalImageDataSource(tmp_path, ModelTestingIngestor())
    datasource.create_partitions(DeterministicSplit())

    pipeline = ModelTestingPipeline(
        datasource=datasource,
        model=IdentityModel(),
        config=TrainingConfig(device="cpu", task=None),
        output_dir=tmp_path / "model_testing_runs",
    )

    with pytest.raises(ValueError):
        pipeline.run()

    report.info(
        "ModelTestingPipeline raised ValueError when task=None — task is required for testing",
        "VER-006",
    )
    report.auto_save(
        "VER006_model_testing_requires_task",
        evidence_output_dir,
    )
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-005")
def test_segmentation_evaluator_records_counts_and_reset(
    evidence_output_dir,
):

    report = EvidenceReport(subject="Segmentation evaluator metrics")

    evaluator = SegmentationEvaluator()
    pred = torch.tensor([[[[1.0, -1.0], [1.0, -1.0]]]])
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

    evaluator.update(pred, target)
    metrics = evaluator.aggregate()

    if metrics != {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "dice": 0.5,
        "iou": 1 / 3,
        "accuracy": 0.5,
    }:
        report.error("Segmentation evaluator returned unexpected metrics", "VER-005")

    evaluator.reset()
    reset_metrics = evaluator.aggregate()

    if reset_metrics["tp"] != 0 or reset_metrics["tn"] != 0:
        report.error("Segmentation evaluator reset did not clear counts", "VER-005")

    for key in ("precision", "recall", "dice", "iou", "accuracy"):
        if not math.isnan(reset_metrics[key]):
            report.error(f"Segmentation evaluator {key} should be NaN after reset", "VER-005")

    report.info(
        "SegmentationEvaluator: update/aggregate produced correct metrics; reset() cleared counts and NaN'd ratios",
        "VER-005",
    )
    report.auto_save(
        "VER005_segmentation_evaluator_metrics",
        evidence_output_dir,
    )
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-006")
def test_model_testing_pipeline_generate_figures_false_skips_figures(
    tmp_path,
    evidence_output_dir,
):

    report = EvidenceReport(
        subject="Model testing pipeline: generate_figures=False skips figure files"
    )

    datasource = MedicalImageDataSource(tmp_path, ModelTestingIngestor())
    datasource.create_partitions(DeterministicSplit())

    pipeline = ModelTestingPipeline(
        datasource=datasource,
        model=IdentityModel(),
        config=TrainingConfig(device="cpu", task=IdentityTask()),
        output_dir=tmp_path / "model_testing_runs",
        generate_figures=False,
    )

    results = pipeline.run()

    figure_keys = {"training_curve", "confusion_matrix", "patient_samples"}
    unexpected = figure_keys & set(results.artifacts.keys())
    if unexpected:
        report.error(
            f"Figure artifacts present despite generate_figures=False: {unexpected}",
            "VER-006",
        )

    report.info("generate_figures=False: no figure artifacts in results.artifacts", "VER-006")
    report.auto_save("VER006_model_testing_generate_figures_false", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-006")
def test_model_testing_pipeline_generate_figures_true_creates_files(
    tmp_path,
    evidence_output_dir,
):

    report = EvidenceReport(
        subject="Model testing pipeline: generate_figures=True creates figure files"
    )

    datasource = MedicalImageDataSource(tmp_path, ModelTestingIngestor())
    datasource.create_partitions(DeterministicSplit())

    training_history = [{"epoch": 1, "loss": 0.8}, {"epoch": 2, "loss": 0.5}]

    pipeline = ModelTestingPipeline(
        datasource=datasource,
        model=IdentityModel(),
        config=TrainingConfig(device="cpu", task=IdentityTask()),
        output_dir=tmp_path / "model_testing_runs",
        generate_figures=True,
        training_history=training_history,
    )

    results = pipeline.run()

    for key in ("training_curve", "confusion_matrix"):
        path = results.artifacts.get(key)
        if path is None or not path.exists():
            report.error(f"Expected figure artifact '{key}' not found", "VER-006")

    report.info(
        "generate_figures=True: training_curve and confusion_matrix artifacts created on disk",
        "VER-006",
    )
    report.auto_save("VER006_model_testing_generate_figures_true", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-006")
def test_model_testing_pipeline_supports_custom_evaluator(
    tmp_path,
    evidence_output_dir,
):

    report = EvidenceReport(subject="Model testing pipeline custom evaluator support")

    datasource = MedicalImageDataSource(tmp_path, ModelTestingIngestor())
    datasource.create_partitions(DeterministicSplit())

    pipeline = ModelTestingPipeline(
        datasource=datasource,
        model=IdentityModel(),
        config=TrainingConfig(device="cpu", task=IdentityTask()),
        evaluator=CountingEvaluator(),
        output_dir=tmp_path / "model_testing_runs",
    )

    results = pipeline.run()

    if results.metrics.get("sample_updates") != 2:
        report.error("Custom evaluator did not aggregate all model testing samples", "VER-006")

    if results.per_patient_results[0].get("sample_updates") != 2:
        report.error("Per-patient evaluator did not aggregate patient-local samples", "VER-006")

    report.info(
        f"Custom CountingEvaluator: global sample_updates={results.metrics.get('sample_updates')}, per-patient sample_updates={results.per_patient_results[0].get('sample_updates')}",
        "VER-006",
    )
    report.auto_save(
        "VER006_model_testing_custom_evaluator",
        evidence_output_dir,
    )
    assert not report.has_errors, report.summary()
