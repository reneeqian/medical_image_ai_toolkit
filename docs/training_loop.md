# Training Loop Semantics

This module provides a generic training loop independent of task or modality.

## Trainer responsibilities

- Iterate over batches
- Forward pass through the model
- Compute loss
- Backpropagation and optimization
- Metric aggregation and reporting

## Batch contract

Each batch is expected to be a dictionary containing:
- `image`: torch.Tensor
- `target` (optional)
- `meta` (optional)

## Design intent

- The trainer does not understand clinical meaning
- Task-specific logic lives in:
  - loss functions
  - metrics
  - model heads

# Training Runs

Each model training run produces a unique set of artifacts stored under:

artifacts/training_runs/<run_id>/

Each run directory contains:
- metadata.json: dataset identity, split strategy, and code version
- config.yaml: training hyperparameters
- splits.json: patient IDs assigned to train/validation
- metrics.json: training and validation metrics
- model.pt: serialized model weights

All training runs are immutable once created.
