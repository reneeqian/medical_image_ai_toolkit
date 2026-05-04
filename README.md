# Medical Image AI Toolkit

Reusable building blocks for medical imaging AI: data contracts, training orchestration, model evaluation, and structured result artifacts. Engineering demonstration — not a clinical product.

## Install

```bash
pip install -e .
```

## Getting Started

Implement three extension points to build a project on top of the toolkit:

1. **Task definition** — subclass `TrainingTaskDefinition`, implement `generate_training_samples(patient_sample)` (yields `{"input": tensor, "target": tensor}`) and `compute_loss(pred, target)`
2. **Ingestor** — produces `PatientSample` objects (`image_volume`, `spacing`, `annotations`, `metadata`) from your dataset
3. **Evaluator** *(optional)* — subclass `BaseEvaluator`, implement `update(pred, target)` and `aggregate() -> dict`. Default is `SegmentationEvaluator`

Pass a `TrainingConfig(task=..., split_strategy=...)` to `TrainingPipeline.run()`, then `ModelTestingPipeline.run()` to evaluate the saved model.

## Run Reports

Generate a multi-page PDF report from any existing run directory:

```python
from medical_image_ai_toolkit.reporting import (
    generate_training_pdf,
    generate_tuning_pdf,
    generate_model_testing_pdf,
)

# Most recent run (default)
pdf = generate_training_pdf("artifacts/training_runs")

# Specific run by ID or date prefix
pdf = generate_tuning_pdf("artifacts/tuning_runs", run_id="20260427")

# Custom output path
pdf = generate_model_testing_pdf("artifacts/model_testing_runs",
                                  output_path="/tmp/report.pdf")
```

Each PDF includes a title/summary page, training-curve or metric plots, and a
configuration table. Output defaults to `{run_dir}/training_report.pdf` (or
`tuning_report.pdf` / `model_testing_report.pdf`).

## Tests

```bash
python -m pytest
python runtests.py   # also generates traceability matrix and forge health report
```

---

## Forge Health

<!-- forge-health-start -->
*Last run: 2026-05-04*

**Grade: B** (score: 0.90)

| Collector | Score |
|-----------|-------|
| Test Metrics | 0.91 |
| Complexity | 0.74 |
| Dependency Health | 1.00 |
| Requirements Coverage | 1.00 |
| Static Analysis | 0.71 |
| Type Coverage | 0.96 |
<!-- forge-health-end -->
