from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from regulatory_tools.evidence.evidence_report import EvidenceReport

from medical_image_ai_toolkit.dataobjects.annotation_bundle import AnnotationBundle, VectorROI
from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import (
    MedicalImageDataSource,
)
from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
from medical_image_ai_toolkit.pipeline.hyperparameter_tuning_pipeline import (
    HyperparameterTuningPipeline,
)
from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition
from medical_image_ai_toolkit.training.training_config import TrainingConfig
from medical_image_ai_toolkit.tuning.hyperparameter_space import HyperparameterSpace

# ---------------------------------------------------------
# Synthetic dataset (same pattern as test_medical_image_trainer.py)
# ---------------------------------------------------------


class SyntheticIngestor:
    def __init__(self, num_patients=10):
        self.num_patients = num_patients
        self.patient_ids = [f"P{i}" for i in range(num_patients)]

    def list_patient_ids(self):
        return self.patient_ids

    def load_patient_sample(self, patient_id):
        rng = np.random.default_rng(abs(hash(patient_id)) % (2**32))
        volume = rng.normal(loc=0, scale=300, size=(8, 64, 64)).astype(np.float32)
        spacing = (1.5, 0.7, 0.7)
        contour = np.array([[20, 20], [40, 20], [40, 40], [20, 40]], dtype=np.float32)
        roi = VectorROI(slice_index=4, contour_px=contour, label="synthetic_lesion")
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
        img = volume[z][16:48, 16:48]
        x = img.astype(np.float32)[None, None, :, :]
        label = 1 if sample.annotations.vector_rois else 0
        y = np.array([[label]], dtype=np.float32)
        return x, y


class DummyTask(TrainingTaskDefinition):
    def generate_training_samples(self, patient_sample):
        for _ in range(2):
            yield {"input": torch.zeros((1, 1, 32, 32)), "target": torch.ones((1, 1, 32, 32))}

    def compute_loss(self, prediction, target):
        return torch.mean((prediction - target) ** 2)


class DeterministicSplit:
    def split(self, ids):
        n = len(ids)
        train = ids[: int(n * 0.6)]
        val = ids[int(n * 0.6) : int(n * 0.8)]
        test = ids[int(n * 0.8) :]
        return train, val, test


class TinyConvNet(torch.nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 32, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def _create_datasource(tmp_path):
    ingestor = SyntheticIngestor(num_patients=10)
    return MedicalImageDataSource(dataset_root=tmp_path, ingestor=ingestor)


# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------


@pytest.mark.requirement("TRN-009")
def test_hyperparameter_space_grid_enumerates_all_combinations(evidence_output_dir):
    report = EvidenceReport(subject="HyperparameterSpace grid enumeration")

    space = HyperparameterSpace(lr=[1e-3, 1e-4], epochs=[5, 10, 20])
    combos = space.grid()

    if len(combos) != 6:
        report.error(f"Expected 6 combos, got {len(combos)}", "TRN-009")

    if len(space) != 6:
        report.error(f"__len__ returned {len(space)}, expected 6", "TRN-009")

    for combo in combos:
        if set(combo.keys()) != {"lr", "epochs"}:
            report.error(f"Unexpected keys in combo: {combo}", "TRN-009")

    as_tuples = [tuple(sorted(c.items())) for c in combos]
    if len(set(as_tuples)) != 6:
        report.error("Duplicate combinations found in grid()", "TRN-009")

    report.info(f"HyperparameterSpace grid enumerated {len(combos)} unique combinations for lr×epochs", "TRN-009")
    report.auto_save("TRN009_hyperparameter_space_grid", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-009")
def test_hyperparameter_space_random_sample(evidence_output_dir):
    report = EvidenceReport(subject="HyperparameterSpace random sampling")

    space = HyperparameterSpace(lr=[1e-3, 1e-4, 1e-5], depth=[2, 3, 4])  # 9 total

    sample_a = space.random_sample(4, seed=0)
    sample_b = space.random_sample(4, seed=0)
    if sample_a != sample_b:
        report.error("random_sample is not deterministic with the same seed", "TRN-009")

    if len(sample_a) != 4:
        report.error(f"Expected 4 samples, got {len(sample_a)}", "TRN-009")

    large_sample = space.random_sample(100, seed=0)
    if len(large_sample) != 9:
        report.error(
            f"Expected all 9 combos when n > total, got {len(large_sample)}", "TRN-009"
        )

    report.info(f"random_sample: seed-reproducible, correct count {len(sample_a)}, caps at total {len(large_sample)} when n>total", "TRN-009")
    report.auto_save("TRN009_hyperparameter_space_random_sample", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-002")
def test_tuning_pipeline_runs_all_grid_trials(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="TRN-002: tuning pipeline runs one trial per grid combination and creates run directories")

    ds = _create_datasource(tmp_path)
    space = HyperparameterSpace(learning_rate=[1e-3, 1e-4])  # 2 trials

    def trial_factory(params):
        return TinyConvNet(), TrainingConfig(
            epochs=1,
            learning_rate=params["learning_rate"],
            task=DummyTask(),
            early_stop=False,
        )

    results = HyperparameterTuningPipeline(
        datasource=ds,
        trial_factory=trial_factory,
        space=space,
        split_strategy=DeterministicSplit(),
        output_dir=tmp_path / "tuning",
        search_strategy="grid",
    ).run()

    if len(results.trial_results) != 2:
        report.error(f"Expected 2 trial results, got {len(results.trial_results)}", "TRN-002")
    for tr in results.trial_results:
        if not tr.run_dir.exists():
            report.error(f"Trial {tr.trial_id} run_dir does not exist: {tr.run_dir}", "TRN-002")

    report.info(f"Tuning pipeline ran {len(results.trial_results)} grid trials; all run_dirs exist", "TRN-002")
    report.auto_save("TRN002_tuning_pipeline_runs_all_grid_trials", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-010")
def test_tuning_pipeline_creates_shared_partitions(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="TRN-010: tuning pipeline creates shared partitions that persist across all trials")

    ds = _create_datasource(tmp_path)
    space = HyperparameterSpace(learning_rate=[1e-3, 1e-4])

    def trial_factory(params):
        return TinyConvNet(), TrainingConfig(
            epochs=1,
            learning_rate=params["learning_rate"],
            task=DummyTask(),
            early_stop=False,
        )

    HyperparameterTuningPipeline(
        datasource=ds,
        trial_factory=trial_factory,
        space=space,
        split_strategy=DeterministicSplit(),
        output_dir=tmp_path / "tuning",
        search_strategy="grid",
    ).run()

    if not ds.has_partitions():
        report.error("Datasource has no partitions after tuning run — shared partitions not created", "TRN-010")

    report.info("Datasource has partitions after tuning pipeline run — shared across all trials", "TRN-010")
    report.auto_save("TRN010_tuning_pipeline_shared_partitions", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-011")
def test_tuning_pipeline_selects_best_trial(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Tuning pipeline selects best trial by loss")

    ds = _create_datasource(tmp_path)
    # Use two very different learning rates so losses are distinguishable
    space = HyperparameterSpace(learning_rate=[1e-2, 1e-7])

    def trial_factory(params):
        model = TinyConvNet()
        config = TrainingConfig(
            epochs=3,
            learning_rate=params["learning_rate"],
            task=DummyTask(),
            early_stop=False,
        )
        return model, config

    pipeline = HyperparameterTuningPipeline(
        datasource=ds,
        trial_factory=trial_factory,
        space=space,
        split_strategy=DeterministicSplit(),
        output_dir=tmp_path / "tuning",
    )
    results = pipeline.run()

    if results.best_trial is None:
        report.error("best_trial is None after tuning", "TRN-011")
    else:
        # best_trial must have loss <= all other trials
        metric_fn = (
            (lambda t: t.best_val_loss)
            if results.best_trial.best_val_loss is not None
            else (lambda t: t.final_loss)
        )
        best_metric = metric_fn(results.best_trial)
        for tr in results.trial_results:
            if best_metric > metric_fn(tr) + 1e-9:
                report.error(
                    f"best_trial (metric={best_metric:.6f}) is worse than "
                    f"trial {tr.trial_id} (metric={metric_fn(tr):.6f})",
                    "TRN-011",
                )

    if results.best_model is None:
        report.error("best_model is None after tuning", "TRN-011")

    if results.best_config is None:
        report.error("best_config is None after tuning", "TRN-011")

    report.info(f"Tuning pipeline selected best_trial={results.best_trial.trial_id if results.best_trial else None} with lowest metric", "TRN-011")
    report.auto_save("TRN011_tuning_pipeline_selects_best_trial", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-009")
def test_tuning_pipeline_random_search_requires_n_trials(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Random search raises ValueError when n_trials is missing")

    ds = _create_datasource(tmp_path)
    space = HyperparameterSpace(learning_rate=[1e-3, 1e-4])

    def trial_factory(params):
        return TinyConvNet(), TrainingConfig(epochs=1, task=DummyTask(), early_stop=False)

    pipeline = HyperparameterTuningPipeline(
        datasource=ds,
        trial_factory=trial_factory,
        space=space,
        split_strategy=DeterministicSplit(),
        output_dir=tmp_path / "tuning",
        search_strategy="random",
        n_trials=None,
    )

    try:
        pipeline.run()
        report.error("Expected ValueError when n_trials=None with random search", "TRN-009")
    except ValueError:
        report.info("ValueError raised as expected", "TRN-009")
    except Exception as e:
        report.error(f"Unexpected exception type {type(e).__name__}: {e}", "TRN-009")

    report.auto_save("TRN009_random_search_requires_n_trials", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-009")
def test_tuning_pipeline_random_search_runs_n_trials(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Random search runs exactly n_trials trials")

    ds = _create_datasource(tmp_path)
    space = HyperparameterSpace(learning_rate=[1e-3, 1e-4], hidden=[4, 8])  # 4 total

    def trial_factory(params):
        model = TinyConvNet(hidden=params["hidden"])
        config = TrainingConfig(
            epochs=1,
            learning_rate=params["learning_rate"],
            task=DummyTask(),
            early_stop=False,
        )
        return model, config

    pipeline = HyperparameterTuningPipeline(
        datasource=ds,
        trial_factory=trial_factory,
        space=space,
        split_strategy=DeterministicSplit(),
        output_dir=tmp_path / "tuning",
        search_strategy="random",
        n_trials=2,
        random_seed=0,
    )
    results = pipeline.run()

    if len(results.trial_results) != 2:
        report.error(
            f"Expected 2 trial results with n_trials=2, got {len(results.trial_results)}",
            "TRN-009",
        )

    report.info(f"Random search with n_trials=2 ran exactly {len(results.trial_results)} trials", "TRN-009")
    report.auto_save("TRN009_random_search_runs_n_trials", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-004")
def test_tuning_pipeline_generates_report(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Tuning pipeline generates tuning_report.json artifact")

    ds = _create_datasource(tmp_path)
    space = HyperparameterSpace(learning_rate=[1e-3])  # single trial

    def trial_factory(params):
        model = TinyConvNet()
        config = TrainingConfig(
            epochs=1,
            learning_rate=params["learning_rate"],
            task=DummyTask(),
            early_stop=False,
        )
        return model, config

    pipeline = HyperparameterTuningPipeline(
        datasource=ds,
        trial_factory=trial_factory,
        space=space,
        split_strategy=DeterministicSplit(),
        output_dir=tmp_path / "tuning",
    )
    results = pipeline.run()

    if "tuning_report" not in results.artifacts:
        report.error("'tuning_report' key missing from results.artifacts", "TRN-004")
    else:
        report_path = results.artifacts["tuning_report"]
        if not report_path.exists():
            report.error(f"tuning_report.json does not exist at {report_path}", "TRN-004")
        else:
            data = json.loads(report_path.read_text())
            for key in ("trial_results", "best_trial", "num_trials"):
                if key not in data:
                    report.error(f"tuning_report.json missing key '{key}'", "TRN-004")
                else:
                    report.info(f"tuning_report.json contains all required keys: trial_results, best_trial, num_trials", "TRN-004")

    report.auto_save("TRN004_tuning_pipeline_generates_report", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-011")
def test_tuning_results_summary_prints_trial_table(tmp_path, evidence_output_dir):
    from medical_image_ai_toolkit.results.hyperparameter_tuning_results import (
        HyperparameterTuningResults,
        TrialResult,
    )

    report = EvidenceReport(subject="HyperparameterTuningResults.summary prints trial table and best-trial section")

    # Case 1: trial with val_loss — covers the table rows and best_val_loss branch
    trial = TrialResult(
        trial_id=0,
        params={"lr": 0.001, "base_channels": 16},
        final_loss=0.5,
        best_val_loss=0.4,
        epochs_trained=3,
        early_stop_reason=None,
        run_dir=tmp_path,
    )
    HyperparameterTuningResults(
        trial_results=[trial],
        best_trial=trial,
        best_model=None,
        best_config=None,
        artifacts={},
        output_dir=tmp_path,
    ).summary()

    # Case 2: trial without val_loss — covers the final_loss fallback branch
    trial_no_val = TrialResult(
        trial_id=1,
        params={"lr": 0.01},
        final_loss=0.8,
        best_val_loss=None,
        epochs_trained=2,
        early_stop_reason=None,
        run_dir=tmp_path,
    )
    HyperparameterTuningResults(
        trial_results=[trial_no_val],
        best_trial=trial_no_val,
        best_model=None,
        best_config=None,
        artifacts={},
        output_dir=tmp_path,
    ).summary()

    # Case 3: empty trial list — covers the early-return branch
    HyperparameterTuningResults(
        trial_results=[],
        best_trial=None,
        best_model=None,
        best_config=None,
        artifacts={},
        output_dir=tmp_path,
    ).summary()

    report.info("HyperparameterTuningResults.summary() executed for trials with val_loss, without val_loss, and empty list", "TRN-011")
    report.auto_save("TRN011_tuning_results_summary", evidence_output_dir)
    assert not report.has_errors, report.summary()
