from dataclasses import dataclass
import math

import numpy as np
import pytest
import torch

from regulatory_tools.evidence.evidence_report import EvidenceReport
from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import MedicalImageDataSource
from medical_image_ai_toolkit.pipeline.validation_pipeline import ValidationPipeline
from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition
from medical_image_ai_toolkit.training.training_config import TrainingConfig
from medical_image_ai_toolkit.validation.base_evaluator import BaseEvaluator
from medical_image_ai_toolkit.validation.segmentation_evaluator import SegmentationEvaluator


@dataclass
class ValidationPatient:
    patient_id: str
    sample_masks: list[torch.Tensor]


class ValidationIngestor:

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
        return ValidationPatient(patient_id=patient_id, sample_masks=[mask_a, mask_b])


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


@pytest.mark.requirement("VER-004")
@pytest.mark.requirement("VER-005")
@pytest.mark.requirement("VER-006")
def test_validation_pipeline_generates_metrics_and_report(
    tmp_path,
    evidence_output_dir,
):

    report = EvidenceReport(subject="Validation pipeline report generation")

    datasource = MedicalImageDataSource(tmp_path, ValidationIngestor())
    datasource.create_partitions(DeterministicSplit())

    pipeline = ValidationPipeline(
        datasource=datasource,
        model=IdentityModel(),
        config=TrainingConfig(device="cpu", task=IdentityTask()),
        output_dir=tmp_path / "validation_runs",
    )

    results = pipeline.run()

    if results.metrics.get("num_test_patients") != 1:
        report.error("Validation pipeline reported incorrect patient count", "VER-004")

    if results.metrics.get("num_test_samples") != 2:
        report.error("Validation pipeline reported incorrect sample count", "VER-004")

    if results.metrics.get("mean_loss") != 0.0:
        report.error("Validation pipeline mean loss should be zero for identity model", "VER-005")

    for key in ("dice", "iou", "precision", "recall", "accuracy"):
        if results.metrics.get(key) != 1.0:
            report.error(f"Validation metric {key} was not recorded as expected", "VER-005")

    report_path = results.artifacts.get("validation_report")
    if report_path is None or not report_path.exists():
        report.error("Validation report artifact was not generated", "VER-006")

    if len(results.per_patient_results) != 1:
        report.error("Per-patient validation output missing", "VER-006")

    report.auto_save(
        "VER004_VER005_VER006_validation_pipeline",
        evidence_output_dir,
    )
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-004")
def test_validation_pipeline_requires_existing_partitions(
    tmp_path,
    evidence_output_dir,
):

    report = EvidenceReport(subject="Validation pipeline partition requirement")

    datasource = MedicalImageDataSource(tmp_path, ValidationIngestor())

    pipeline = ValidationPipeline(
        datasource=datasource,
        model=IdentityModel(),
        config=TrainingConfig(device="cpu", task=IdentityTask()),
        output_dir=tmp_path / "validation_runs",
    )

    with pytest.raises(RuntimeError):
        pipeline.run()

    report.auto_save(
        "VER004_validation_requires_partitions",
        evidence_output_dir,
    )
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-006")
def test_validation_pipeline_requires_task_definition(
    tmp_path,
    evidence_output_dir,
):

    report = EvidenceReport(subject="Validation pipeline task requirement")

    datasource = MedicalImageDataSource(tmp_path, ValidationIngestor())
    datasource.create_partitions(DeterministicSplit())

    pipeline = ValidationPipeline(
        datasource=datasource,
        model=IdentityModel(),
        config=TrainingConfig(device="cpu", task=None),
        output_dir=tmp_path / "validation_runs",
    )

    with pytest.raises(ValueError):
        pipeline.run()

    report.auto_save(
        "VER006_validation_requires_task",
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

    report.auto_save(
        "VER005_segmentation_evaluator_metrics",
        evidence_output_dir,
    )
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-006")
def test_validation_pipeline_supports_custom_evaluator(
    tmp_path,
    evidence_output_dir,
):

    report = EvidenceReport(subject="Validation pipeline custom evaluator support")

    datasource = MedicalImageDataSource(tmp_path, ValidationIngestor())
    datasource.create_partitions(DeterministicSplit())

    pipeline = ValidationPipeline(
        datasource=datasource,
        model=IdentityModel(),
        config=TrainingConfig(device="cpu", task=IdentityTask()),
        evaluator=CountingEvaluator(),
        output_dir=tmp_path / "validation_runs",
    )

    results = pipeline.run()

    if results.metrics.get("sample_updates") != 2:
        report.error("Custom evaluator did not aggregate all validation samples", "VER-006")

    if results.per_patient_results[0].get("sample_updates") != 2:
        report.error("Per-patient evaluator did not aggregate patient-local samples", "VER-006")

    report.auto_save(
        "VER006_validation_custom_evaluator",
        evidence_output_dir,
    )
    assert not report.has_errors, report.summary()
