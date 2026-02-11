# Tensor Interface Contract

Tensor adapters define the boundary between project-specific data and model-agnostic training.

## Core principles

- Models consume **tensors**, not domain objects
- Tensor structure must be explicit and documented
- Adapters are responsible for normalization and layout

## Expected outputs

Adapters should return a dictionary with agreed-upon keys:

- `image`: torch.Tensor  
  Shape: (C, Z, Y, X) or (C, Y, X)  
  Dtype: float32

- `target` (optional): torch.Tensor  
  Task-specific (e.g. mask, label, regression value)

- `meta` (optional): dict  
  Non-tensor metadata (patient_id, spacing, etc.)

## Conventions

- Channel-first tensors
- Physical spacing is *not* encoded unless explicitly required
- Missing targets are allowed for inference-only pipelines
