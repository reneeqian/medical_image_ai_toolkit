import pytest
import torch

from medical_image_ai_toolkit.training.training_config import TrainingConfig
from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition


class DummyTask(TrainingTaskDefinition):

    def generate_training_samples(self, patient_sample):
        yield {
            "input": torch.zeros((1, 1)),
            "target": torch.zeros((1, 1)),
        }

    def compute_loss(self, prediction, target):
        return torch.mean((prediction - target) ** 2)


@pytest.mark.requirement("SYS-003")
@pytest.mark.requirement("TRN-001")
def test_training_config_initialization():
    task = DummyTask()

    config = TrainingConfig(
        epochs=5,
        batch_size=4,
        learning_rate=1e-3,
        device="cpu",
        task=task,
    )

    assert config.epochs == 5
    assert config.batch_size == 4
    assert config.learning_rate == 1e-3
    assert config.device == "cpu"
    assert config.task is task


@pytest.mark.requirement("TRN-007")
def test_training_config_uses_task_loss_interface():
    task = DummyTask()
    config = TrainingConfig(task=task)
    loss = config.task.compute_loss(
        torch.ones((1, 1)),
        torch.zeros((1, 1)),
    )

    assert torch.isfinite(loss)


@pytest.mark.requirement("MOD-001")
def test_training_config_optimizer_class():
    config = TrainingConfig(optimizer=torch.optim.Adam)

    assert config.optimizer is torch.optim.Adam
