import torch
import pytest
from pathlib import Path

from regulatory_tools.evidence.evidence_report import EvidenceReport
from medical_image_ai_toolkit.results.medical_image_training_results import (
    MedicalImageTrainingResults
)


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
def test_summary_training_running(tmp_path):

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

@pytest.mark.requirement("DOC-004")
def test_mark_training_complete_summary(tmp_path):

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

@pytest.mark.requirement("MOD-005")
def test_export_model_failure(tmp_path):

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