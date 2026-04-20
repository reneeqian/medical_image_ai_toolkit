import pytest
from regulatory_tools.evidence.evidence_report import EvidenceReport
from medical_image_ai_toolkit.dataobjects.datasources.deterministic_split import DeterministicHoldoutSplit


def _ids(n):
    return [f"P{i}" for i in range(n)]


@pytest.mark.requirement("VER-003")
def test_split_partitions_are_non_overlapping(evidence_output_dir):
    report = EvidenceReport(subject="DeterministicHoldoutSplit non-overlapping partitions")

    train, val, test = DeterministicHoldoutSplit(seed=0).split(_ids(100))

    if set(train) & set(val):
        report.error("Train/Val overlap detected", "VER-003")
    if set(train) & set(test):
        report.error("Train/Test overlap detected", "VER-003")
    if set(val) & set(test):
        report.error("Val/Test overlap detected", "VER-003")

    report.auto_save("VER003_split_non_overlapping", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-001")
def test_split_is_deterministic(evidence_output_dir):
    report = EvidenceReport(subject="DeterministicHoldoutSplit reproducibility")

    ids = _ids(50)
    run1 = DeterministicHoldoutSplit(seed=42).split(ids)
    run2 = DeterministicHoldoutSplit(seed=42).split(ids)

    if run1 != run2:
        report.error("Split results differ across runs with same seed", "VER-001")

    report.auto_save("VER001_split_deterministic", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-001")
def test_different_seeds_produce_different_splits(evidence_output_dir):
    report = EvidenceReport(subject="DeterministicHoldoutSplit seed sensitivity")

    ids = _ids(50)
    train_a, _, _ = DeterministicHoldoutSplit(seed=1).split(ids)
    train_b, _, _ = DeterministicHoldoutSplit(seed=2).split(ids)

    if train_a == train_b:
        report.error("Different seeds produced identical splits", "VER-001")

    report.auto_save("VER001_split_seed_sensitivity", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-003")
def test_split_covers_all_patients(evidence_output_dir):
    report = EvidenceReport(subject="DeterministicHoldoutSplit full coverage")

    ids = _ids(30)
    train, val, test = DeterministicHoldoutSplit().split(ids)

    all_assigned = set(train) | set(val) | set(test)
    if all_assigned != set(ids):
        report.error("Some patients were not assigned to any partition", "VER-003")

    report.auto_save("VER003_split_full_coverage", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-003")
def test_split_max_caps_are_respected(evidence_output_dir):
    report = EvidenceReport(subject="DeterministicHoldoutSplit max caps")

    train, val, test = DeterministicHoldoutSplit(
        max_train=5, max_val=3, max_test=2
    ).split(_ids(100))

    if len(train) > 5:
        report.error(f"Train partition exceeds max_train: {len(train)}", "VER-003")
    if len(val) > 3:
        report.error(f"Val partition exceeds max_val: {len(val)}", "VER-003")
    if len(test) > 2:
        report.error(f"Test partition exceeds max_test: {len(test)}", "VER-003")

    report.auto_save("VER003_split_max_caps", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-003")
def test_split_on_minimal_patient_count(evidence_output_dir):
    report = EvidenceReport(subject="DeterministicHoldoutSplit with very few patients")

    train, val, test = DeterministicHoldoutSplit().split(_ids(3))

    total = len(train) + len(val) + len(test)
    if total != 3:
        report.error(f"Expected 3 patients assigned, got {total}", "VER-003")

    report.auto_save("VER003_split_minimal_patients", evidence_output_dir)
    assert not report.has_errors, report.summary()
