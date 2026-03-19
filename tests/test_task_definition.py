import pytest
import torch
import numpy as np

from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition


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
@pytest.mark.requirement("TRN-008")
def test_task_generates_aligned_samples():
    task = DummyTask()
    sample = DummyPatientSample()

    gen = task.generate_training_samples(sample)

    item = next(gen)

    x = item["input"]
    y = item["target"]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == y.shape


@pytest.mark.requirement("TRN-007")
def test_task_compute_loss():
    task = DummyTask()

    pred = torch.zeros((1, 1, 32, 32))
    target = torch.ones((1, 1, 32, 32))

    loss = task.compute_loss(pred, target)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0


@pytest.mark.requirement("DAT-010")
def test_task_slice_level_iteration():
    task = DummyTask()
    sample = DummyPatientSample()

    outputs = list(task.generate_training_samples(sample))

    assert len(outputs) == 3