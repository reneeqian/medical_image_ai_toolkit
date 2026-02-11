# Training Pipeline Architecture

This module defines a reusable, modality-agnostic training pipeline for medical imaging AI.

## High-level flow

PatientSample
    → Tensor Adapter
    → Torch Dataset
    → Model
    → Trainer
    → Trained Model

## Key concepts

- **PatientSample**
  A validated, project-defined container for patient-level data (images, annotations, metadata).

- **Tensor Adapter**
  Converts a PatientSample into model-ready tensors with consistent shape, dtype, and semantics.

- **Torch Dataset**
  Wraps one or more PatientSamples to support batching, shuffling, and sampling strategies.

- **Trainer**
  Owns the training loop, loss computation, optimization, and metric reporting.

## Design goals

- Separation of concerns
- Explicit interfaces between stages
- Reusable across datasets, tasks, and modalities
- Validation happens *before* training
