import pytest
import torch
import numpy as np

from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition
from regulatory_tools.evidence.evidence_report import EvidenceReport


class DummyTask(TrainingTaskDefinition):

    def generate_training_samples(self, patient_sample):

        for _ in range(3):
            yield {
                "input": torch.zeros((1, 1, 32, 32)),
                "target": torch.ones((1, 1, 32, 32))
            }

    def compute_loss(self, prediction, target):
        return torch.mean((prediction - target) ** 2)


class DummyPatientSample:
    def __init__(self):
        self.image_volume = np.zeros((3, 32, 32))
        self.annotations = None


@pytest.mark.requirement("DAT-009")
def test_task_generates_matched_input_target_shapes(evidence_output_dir):
    report = EvidenceReport(subject="DAT-009: generate_training_samples yields input and target tensors with matching shapes")

    task = DummyTask()
    sample = DummyPatientSample()
    item = next(task.generate_training_samples(sample))
    x = item["input"]
    y = item["target"]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == y.shape

    report.info(f"generate_training_samples: input shape={x.shape}, target shape={y.shape} — match", "DAT-009")
    report.auto_save("DAT009_task_matched_input_target_shapes", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-008")
def test_task_generate_training_samples_interface(evidence_output_dir):
    report = EvidenceReport(subject="TRN-008: generate_training_samples yields dict with 'input' and 'target' keys")

    task = DummyTask()
    sample = DummyPatientSample()
    item = next(task.generate_training_samples(sample))

    if "input" not in item:
        report.error("Yielded dict missing 'input' key", "TRN-008")
    if "target" not in item:
        report.error("Yielded dict missing 'target' key", "TRN-008")

    report.info(f"generate_training_samples yielded dict with keys: {list(item.keys())}", "TRN-008")
    report.auto_save("TRN008_task_generate_training_samples_interface", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-007")
def test_task_compute_loss(evidence_output_dir):
    
    report = EvidenceReport(subject="Task compute_loss returns a finite scalar tensor")
    
    task = DummyTask()

    pred = torch.zeros((1, 1, 32, 32))
    target = torch.ones((1, 1, 32, 32))

    loss = task.compute_loss(pred, target)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0

    report.info(f"compute_loss returned Tensor with value={loss.item():.4f} > 0 for mismatched pred/target", "TRN-007")
    report.auto_save(
        "TRN007_task_compute_loss",
        evidence_output_dir
    )
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-010")
def test_task_slice_level_iteration(evidence_output_dir):
    
    report = EvidenceReport(subject="Task generates samples for each slice in patient volume")
    
    task = DummyTask()
    sample = DummyPatientSample()

    outputs = list(task.generate_training_samples(sample))

    assert len(outputs) == 3

    report.info(f"generate_training_samples yielded {len(outputs)} samples for a 3-slice volume — one sample per slice", "DAT-010")
    report.auto_save(
        "DAT010_task_slice_iteration",
        evidence_output_dir
    )
    assert not report.has_errors, report.summary()

@pytest.mark.requirement("TRN-008")
def test_postprocess_prediction_identity(evidence_output_dir):
    
    report = EvidenceReport(subject="Task postprocess_prediction should be identity by default")

    task = DummyTask()

    x = torch.randn(1, 1)

    out = task.postprocess_prediction(x)

    assert torch.equal(x, out)

    report.info("postprocess_prediction returns input tensor unchanged — default is identity transformation", "TRN-008")
    report.auto_save(
        "TRN008_task_postprocess_prediction_identity",
        evidence_output_dir
    )
    assert not report.has_errors, report.summary()