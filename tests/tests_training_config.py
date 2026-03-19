import pytest
import torch

from medical_image_ai_toolkit.training.training_config import TrainingConfig


@pytest.mark.requirement("SYS-003")
@pytest.mark.requirement("TRN-001")
def test_training_config_initialization():
    config = TrainingConfig(
        epochs=5,
        batch_size=4,
        learning_rate=1e-3,
        device="cpu"
    )

    assert config.epochs == 5
    assert config.batch_size == 4
    assert config.learning_rate == 1e-3
    assert config.device == "cpu"


@pytest.mark.requirement("TRN-007")
def test_training_config_loss_function():
    config = TrainingConfig(loss_fcn=torch.nn.MSELoss())

    assert callable(config.loss_fcn)


@pytest.mark.requirement("MOD-001")
def test_training_config_optimizer_class():
    config = TrainingConfig(optimizer=torch.optim.Adam)

    assert config.optimizer is torch.optim.Adam