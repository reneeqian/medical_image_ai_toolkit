# Medical Image AI Toolkit

Reusable building blocks for medical imaging AI projects developed with
determinism, explicit contracts, and lightweight traceability in mind.

This is an engineering demonstration, not a clinical product.

## Mission

This module exists to provide a small, reusable core for:

- patient-level data contracts
- lazy dataset access and partitioning
- task-driven training orchestration
- post-training validation
- structured result artifacts

Project-specific behavior should live outside the toolkit wherever
possible. The toolkit should define stable seams, not project-specific
policy.

## Scope

The toolkit is intentionally lightweight. It is meant to support:

- clear interfaces between data, tasks, training, and validation
- deterministic behavior when configuration and inputs are fixed
- machine-readable requirements and test traceability

It is not meant to be a full framework or to carry project-specific
clinical logic.

## Repository Layout

- `src/medical_image_ai_toolkit/`: reusable contracts, training, validation, and results code
- `docs/requirements.yaml`: stable behavioral requirements for the toolkit
- `tests/`: executable verification of those requirements

## Documentation Approach

The primary documentation for this module is:

- this README for mission and boundaries
- `docs/requirements.yaml` for behavioral expectations
- tests for executable examples of intended behavior

Requirements follow the convention defined in
`regulatory_tools/docs/Requirements_Convention.md`.

---

## Forge Health

<!-- forge-health-start -->
*Last run: 2026-04-25*

**Grade: B** (score: 0.83)

| Collector | Score |
|-----------|-------|
| Test Metrics | 0.86 |
| Complexity | 0.78 |
| Dependency Health | 0.85 |
| Requirements Coverage | 0.90 |
| Static Analysis | 0.64 |
| Type Coverage | 0.90 |
<!-- forge-health-end -->
