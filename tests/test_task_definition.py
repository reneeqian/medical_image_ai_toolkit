import pytest

from regulatory_tools.evidence.evidence_report import EvidenceReport
from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition


@pytest.mark.requirement("TRN-001")
def test_task_definition_interface(evidence_output_dir):

    report = EvidenceReport(subject="TrainingTaskDefinition interface validation")

    class MyTask(TrainingTaskDefinition):

        def prepare_training_sample(self, sample):
            return sample, sample

        def compute_loss(self, pred, y):
            return pred.sum()

    task = MyTask()

    if not hasattr(task, "prepare_training_sample"):
        report.error("Task missing prepare_training_sample", "TRN-001")

    if not hasattr(task, "compute_loss"):
        report.error("Task missing compute_loss", "TRN-001")

    # NEW: cover default postprocess
    result = task.postprocess_prediction(5)

    if result != 5:
        report.error("postprocess_prediction failed", "TRN-001")

    report.auto_save(
        "TRN001_task_definition_interface",
        evidence_output_dir
    )

    assert not report.has_errors, report.summary()

@pytest.mark.requirement("TRN-001")
def test_task_definition_cannot_instantiate_abstract():

    with pytest.raises(TypeError):
        TrainingTaskDefinition()
    

@pytest.mark.requirement("TRN-001")
def test_postprocess_prediction_default(evidence_output_dir):

    report = EvidenceReport(subject="TrainingTaskDefinition postprocess default")

    class MyTask(TrainingTaskDefinition):

        def prepare_training_sample(self, sample):
            return sample, sample

        def compute_loss(self, pred, y):
            return pred

    task = MyTask()

    pred = 123
    result = task.postprocess_prediction(pred)

    if result != pred:
        report.error("postprocess_prediction altered output", "TRN-001")

    report.auto_save(
        "TRN001_postprocess_prediction_default",
        evidence_output_dir
    )

    assert not report.has_errors

@pytest.mark.requirement("TRN-001")
def test_postprocess_prediction_passthrough():

    class MyTask(TrainingTaskDefinition):

        def prepare_training_sample(self, sample):
            return sample, sample

        def compute_loss(self, pred, y):
            return pred

    task = MyTask()

    pred = "test"

    assert task.postprocess_prediction(pred) == pred