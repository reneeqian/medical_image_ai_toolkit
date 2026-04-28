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

## Tests

```bash
python -m pytest
python runtests.py   # also generates traceability matrix and forge health report
```

---

## Forge Health

<!-- forge-health-start -->
*Last run: 2026-04-27*

**Grade: B** (score: 0.87)

| Collector | Score |
|-----------|-------|
| Test Metrics | 0.86 |
| Complexity | 0.78 |
| Dependency Health | 1.00 |
| Requirements Coverage | 0.90 |
| Static Analysis | 0.65 |
| Type Coverage | 0.97 |
<!-- forge-health-end -->
