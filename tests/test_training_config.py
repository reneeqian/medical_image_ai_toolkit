import pytest
import torch

from medical_image_ai_toolkit.training.training_config import TrainingConfig
from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition
from regulatory_tools.evidence.evidence_report import EvidenceReport


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


@pytest.mark.requirement("TRN-005")
def test_training_config_early_stop_defaults(evidence_output_dir):
    report = EvidenceReport(subject="TrainingConfig early stopping defaults")

    config = TrainingConfig()

    if config.early_stop is not True:
        report.error("early_stop should default to True", "TRN-005")
    if config.loss_threshold != 0.01:
        report.error(f"loss_threshold default wrong: {config.loss_threshold}", "TRN-005")
    if config.plateau_patience != 5:
        report.error(f"plateau_patience default wrong: {config.plateau_patience}", "TRN-005")
    if config.plateau_min_delta != 1e-4:
        report.error(f"plateau_min_delta default wrong: {config.plateau_min_delta}", "TRN-005")

    report.auto_save("TRN005_early_stop_defaults", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-005")
def test_training_config_early_stop_can_be_disabled(evidence_output_dir):
    report = EvidenceReport(subject="TrainingConfig early_stop=False")

    config = TrainingConfig(early_stop=False)

    if config.early_stop is not False:
        report.error("early_stop should be False when set explicitly", "TRN-005")

    report.auto_save("TRN005_early_stop_disabled_config", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-005")
def test_training_config_early_stop_custom_values(evidence_output_dir):
    report = EvidenceReport(subject="TrainingConfig custom early stop thresholds")

    config = TrainingConfig(
        loss_threshold=0.05,
        plateau_patience=10,
        plateau_min_delta=1e-3,
    )

    if config.loss_threshold != 0.05:
        report.error(f"loss_threshold not stored correctly: {config.loss_threshold}", "TRN-005")
    if config.plateau_patience != 10:
        report.error(f"plateau_patience not stored correctly: {config.plateau_patience}", "TRN-005")
    if config.plateau_min_delta != 1e-3:
        report.error(f"plateau_min_delta not stored correctly: {config.plateau_min_delta}", "TRN-005")

    report.auto_save("TRN005_early_stop_custom_values", evidence_output_dir)
    assert not report.has_errors, report.summary()
