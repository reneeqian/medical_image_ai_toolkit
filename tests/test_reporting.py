"""
Tests for the medical_image_ai_toolkit.reporting module.

Covers:
  - find_run_dir: most-recent, exact ID, prefix match, empty-root error
  - generate_training_pdf: creates PDF file, handles val_loss in history
  - generate_tuning_pdf: creates PDF file
  - generate_model_testing_pdf: creates PDF file
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from regulatory_tools.evidence.evidence_report import EvidenceReport

from medical_image_ai_toolkit.reporting import (
    find_run_dir,
    generate_model_testing_pdf,
    generate_training_pdf,
    generate_tuning_pdf,
)

# ---------------------------------------------------------------------------
# Fixtures — synthetic run directories with minimal JSON artifacts
# ---------------------------------------------------------------------------


def _make_training_run(root: Path, name: str) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    report = {
        "metrics": {"final_loss": 0.25, "epochs_trained": 3, "num_epochs": 5},
        "history": [
            {"epoch": 1, "loss": 0.50},
            {"epoch": 2, "loss": 0.35},
            {"epoch": 3, "loss": 0.25},
        ],
        "config": {
            "learning_rate": 0.001,
            "epochs": 5,
            "batch_size": 8,
            "device": "cpu",
            "optimizer": "Adam",
            "task": "DummyTask",
        },
        "datasource": {
            "train_ids": ["P0", "P1", "P2"],
            "val_ids": ["P3"],
            "test_ids": ["P4", "P5"],
        },
    }
    (run_dir / "training_report.json").write_text(json.dumps(report))
    return run_dir


def _make_training_run_with_val(root: Path, name: str) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    report = {
        "metrics": {"final_loss": 0.20, "epochs_trained": 2, "num_epochs": 2},
        "history": [
            {"epoch": 1, "loss": 0.40, "val_loss": 0.38, "dice": 0.60, "iou": 0.45},
            {"epoch": 2, "loss": 0.20, "val_loss": 0.22, "dice": 0.75, "iou": 0.60},
        ],
        "config": {
            "learning_rate": 0.001,
            "epochs": 2,
            "batch_size": 4,
            "device": "cpu",
            "optimizer": "Adam",
            "task": "DummyTask",
        },
        "datasource": {"train_ids": ["P0"], "val_ids": ["P1"], "test_ids": ["P2"]},
    }
    (run_dir / "training_report.json").write_text(json.dumps(report))
    return run_dir


def _make_tuning_run(root: Path, name: str) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    report = {
        "num_trials": 2,
        "best_trial": {
            "trial_id": 0,
            "params": {"learning_rate": 0.001, "base_channels": 16},
            "final_loss": 0.18,
            "best_val_loss": 0.20,
            "epochs_trained": 3,
            "early_stop_reason": None,
            "run_dir": str(run_dir / "trial_000"),
        },
        "trial_results": [
            {
                "trial_id": 0,
                "params": {"learning_rate": 0.001, "base_channels": 16},
                "final_loss": 0.18,
                "best_val_loss": 0.20,
                "epochs_trained": 3,
                "early_stop_reason": None,
                "run_dir": str(run_dir / "trial_000"),
            },
            {
                "trial_id": 1,
                "params": {"learning_rate": 0.0001, "base_channels": 32},
                "final_loss": 0.30,
                "best_val_loss": 0.32,
                "epochs_trained": 2,
                "early_stop_reason": None,
                "run_dir": str(run_dir / "trial_001"),
            },
        ],
    }
    (run_dir / "tuning_report.json").write_text(json.dumps(report))
    return run_dir


def _make_model_testing_run(root: Path, name: str) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    report = {
        "metrics": {
            "mean_loss": 0.15,
            "num_test_patients": 3,
            "num_test_samples": 30,
            "tp": 500,
            "fp": 100,
            "fn": 80,
            "tn": 9320,
            "precision": 0.833,
            "recall": 0.862,
            "dice": 0.847,
            "iou": 0.735,
            "accuracy": 0.982,
        },
        "per_patient_results": [
            {
                "patient_id": "P0",
                "loss": 0.12,
                "n_samples": 10,
                "dice": 0.85,
                "iou": 0.74,
                "precision": 0.84,
                "recall": 0.87,
            },
            {
                "patient_id": "P1",
                "loss": 0.17,
                "n_samples": 10,
                "dice": 0.80,
                "iou": 0.67,
                "precision": 0.82,
                "recall": 0.79,
            },
            {
                "patient_id": "P2",
                "loss": 0.16,
                "n_samples": 10,
                "dice": 0.89,
                "iou": 0.80,
                "precision": 0.85,
                "recall": 0.94,
            },
        ],
        "config": {"task": "CoronaryCalciumTask", "device": "cpu"},
    }
    (run_dir / "model_testing_report.json").write_text(json.dumps(report))
    return run_dir


# ---------------------------------------------------------------------------
# find_run_dir tests
# ---------------------------------------------------------------------------


@pytest.mark.requirement("REP-002")
def test_find_run_dir_returns_most_recent(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="find_run_dir returns most recent run")

    runs_root = tmp_path / "training_runs"
    _make_training_run(runs_root, "20260101_100000_000000")
    _make_training_run(runs_root, "20260102_100000_000000")
    newest = _make_training_run(runs_root, "20260103_100000_000000")

    result = find_run_dir(runs_root)
    if result != newest:
        report.error(f"Expected most recent run {newest.name}, got {result.name}", "REP-002")

    report.info(f"find_run_dir() returned most recent run: {result.name}", "REP-002")
    report.auto_save("REP002_find_run_dir_most_recent", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("REP-002")
def test_find_run_dir_returns_by_exact_id(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="find_run_dir returns run by exact ID")

    runs_root = tmp_path / "training_runs"
    _make_training_run(runs_root, "20260101_100000_000000")
    target = _make_training_run(runs_root, "20260102_100000_000000")
    _make_training_run(runs_root, "20260103_100000_000000")

    result = find_run_dir(runs_root, run_id="20260102_100000_000000")
    if result != target:
        report.error(f"Expected {target.name}, got {result.name}", "REP-002")

    report.info(
        f"find_run_dir(run_id='20260102_100000_000000') returned exact match: {result.name}",
        "REP-002",
    )
    report.auto_save("REP002_find_run_dir_exact_id", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("REP-002")
def test_find_run_dir_partial_prefix_match(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="find_run_dir accepts prefix match for run ID")

    runs_root = tmp_path / "training_runs"
    _make_training_run(runs_root, "20260427_090000_000000")
    latest_that_day = _make_training_run(runs_root, "20260427_180000_000000")
    _make_training_run(runs_root, "20260428_100000_000000")

    result = find_run_dir(runs_root, run_id="20260427")
    if result != latest_that_day:
        report.error(
            f"Prefix match returned {result.name}, expected {latest_that_day.name}",
            "REP-002",
        )

    report.info(
        f"find_run_dir(run_id='20260427') prefix match returned latest run on that date: {result.name}",
        "REP-002",
    )
    report.auto_save("REP002_find_run_dir_prefix_match", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("REP-002")
def test_find_run_dir_raises_on_empty(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="find_run_dir raises FileNotFoundError on empty root")

    runs_root = tmp_path / "empty_runs"
    runs_root.mkdir()

    try:
        find_run_dir(runs_root)
        report.error("Expected FileNotFoundError, but no exception was raised", "REP-002")
    except FileNotFoundError:
        report.info("FileNotFoundError raised as expected", "REP-002")
    except Exception as e:
        report.error(f"Unexpected exception {type(e).__name__}: {e}", "REP-002")

    report.auto_save("REP002_find_run_dir_raises_on_empty", evidence_output_dir)
    assert not report.has_errors, report.summary()


# ---------------------------------------------------------------------------
# PDF generator tests
# ---------------------------------------------------------------------------


@pytest.mark.requirement("REP-001")
def test_generate_training_pdf_creates_file(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="generate_training_pdf writes a PDF file")

    runs_root = tmp_path / "training_runs"
    _make_training_run(runs_root, "20260427_100000_000000")

    try:
        pdf_path = generate_training_pdf(runs_root)
        if not pdf_path.exists():
            report.error(f"PDF not found at expected path: {pdf_path}", "REP-001")
        elif pdf_path.stat().st_size == 0:
            report.error("PDF file exists but is empty", "REP-001")
        elif not pdf_path.name.endswith(".pdf"):
            report.error(f"Output file has wrong extension: {pdf_path.name}", "REP-001")
        else:
            report.info(
                f"generate_training_pdf() created {pdf_path.name} ({pdf_path.stat().st_size} bytes)",
                "REP-001",
            )
    except Exception as e:
        report.error(f"generate_training_pdf raised {type(e).__name__}: {e}", "REP-001")

    report.auto_save("REP001_training_pdf_creates_file", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("REP-001")
def test_generate_training_pdf_with_val_loss_in_history(tmp_path, evidence_output_dir):
    report = EvidenceReport(
        subject="generate_training_pdf handles val_loss and metric fields in history"
    )

    runs_root = tmp_path / "training_runs"
    _make_training_run_with_val(runs_root, "20260427_120000_000000")

    try:
        pdf_path = generate_training_pdf(runs_root)
        if not pdf_path.exists():
            report.error("PDF not written", "REP-001")
        elif pdf_path.stat().st_size == 0:
            report.error("PDF is empty", "REP-001")
        else:
            report.info(
                f"generate_training_pdf() handled val_loss + metric fields; created {pdf_path.name}",
                "REP-001",
            )
    except Exception as e:
        report.error(
            f"generate_training_pdf raised {type(e).__name__} with val_loss history: {e}",
            "REP-001",
        )

    report.auto_save("REP001_training_pdf_val_loss", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("REP-001")
def test_generate_tuning_pdf_creates_file(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="generate_tuning_pdf writes a PDF file")

    runs_root = tmp_path / "tuning_runs"
    _make_tuning_run(runs_root, "20260427_100000")

    try:
        pdf_path = generate_tuning_pdf(runs_root)
        if not pdf_path.exists():
            report.error(f"PDF not found at: {pdf_path}", "REP-001")
        elif pdf_path.stat().st_size == 0:
            report.error("PDF file is empty", "REP-001")
        else:
            report.info(
                f"generate_tuning_pdf() created {pdf_path.name} ({pdf_path.stat().st_size} bytes)",
                "REP-001",
            )
    except Exception as e:
        report.error(f"generate_tuning_pdf raised {type(e).__name__}: {e}", "REP-001")

    report.auto_save("REP001_tuning_pdf_creates_file", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("REP-001")
def test_generate_model_testing_pdf_creates_file(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="generate_model_testing_pdf writes a PDF file")

    runs_root = tmp_path / "model_testing_runs"
    _make_model_testing_run(runs_root, "20260427_100000_000000")

    try:
        pdf_path = generate_model_testing_pdf(runs_root)
        if not pdf_path.exists():
            report.error(f"PDF not found at: {pdf_path}", "REP-001")
        elif pdf_path.stat().st_size == 0:
            report.error("PDF file is empty", "REP-001")
        else:
            report.info(
                f"generate_model_testing_pdf() created {pdf_path.name} ({pdf_path.stat().st_size} bytes)",
                "REP-001",
            )
    except Exception as e:
        report.error(f"generate_model_testing_pdf raised {type(e).__name__}: {e}", "REP-001")

    report.auto_save("REP001_model_testing_pdf_creates_file", evidence_output_dir)
    assert not report.has_errors, report.summary()
