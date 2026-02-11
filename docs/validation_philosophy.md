# Validation Philosophy

Validation ensures data correctness *before* training begins.

## Key ideas

- Validation is explicit, not implicit
- Training assumes validated inputs
- Validation produces reports, not crashes

## What validation checks

- Structural correctness (shapes, types)
- Semantic invariants (bounds, consistency)
- Cross-field assumptions (e.g. ROI within volume)

## What validation does NOT do

- Modify data
- Perform learning-time checks
- Handle performance or convergence issues

## Why this matters

- Faster debugging
- Clear failure modes
- Regulatory and audit readiness
