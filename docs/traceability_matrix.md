# Requirements Traceability Matrix

| Requirement ID | Description | Linked Tests | Evidence Artifacts | Status |
|----------------|-------------|--------------|--------------------|--------|
| MIT-DR-01 | PatientSample image_volume shall be a NumPy array. |  | MIT_DR_01_invalid_patient_sample_rejected_20260212_145439_011312.json | PASS |
| MIT-DR-02 | PatientSample image_volume shall be three-dimensional (z,y,x). |  |  | UNTESTED |
| MIT-DR-03 | PatientSample image_volume spatial dimensions shall be greater than zero. |  | MIT_DR_03_patient_sample_converted_to_3d_tensor_20260212_145439_010587.json | PASS |
| MIT-DR-04 | PatientSample spacing shall be defined as (z,y,x) with all values greater than zero. |  |  | UNTESTED |
| MIT-DR-05 | PatientSample shall define a non-empty patient_id. |  |  | UNTESTED |
| MIT-DR-06 | Annotations may be optional but shall be valid if present. |  |  | UNTESTED |
| MIT-DR-07 | Annotation slice indices shall be within volume bounds. |  |  | UNTESTED |
| MIT-DR-08 | Vector ROI contours shall be valid (N,2) and within image bounds. |  |  | UNTESTED |
| MIT-DR-09 | All PatientSample invariants shall be enforced at a single validation boundary. |  | MIT_DR_09_trainer_rejects_unvalidated_input_20260212_145438_866967.json | PASS |
| MIT-FR-01 | Datasets shall expose samples via a consistent PatientSample interface. |  | MIT_FR_01_tensor_datamodule_exposes_dataset_length_20260212_145439_013619.json, MIT_FR_01_tensor_datamodule_getitem_returns_sample_20260212_145439_016588.json | PASS |
| MIT-FR-02 | Dataset validation logic shall be isolated from dataset iteration logic. |  |  | UNTESTED |
| MIT-FR-03 | Training process shall support configurable models, optimizers, and loss functions. |  | MIT_DR_03_patient_sample_converted_to_3d_tensor_20260212_145439_010587.json | PASS |
| MIT-FR-04 | Training process shall support configurable hyperparameters. |  |  | UNTESTED |
| MIT-FR-05 | Toolkit shall support structured hyperparameter sweep execution. |  |  | UNTESTED |
| MIT-FR-06 | Toolkit shall support model pruning or compression workflows. |  |  | UNTESTED |
| MIT-FR-07 | Optimization workflows shall persist pre- and post-optimization artifacts. |  |  | UNTESTED |
| MIT-MAR-01 | Training runs shall persist model artifacts traceable to a specific run. |  |  | UNTESTED |
| MIT-MAR-02 | Training runs shall persist model checkpoints per epoch. |  |  | UNTESTED |
| MIT-MAR-03 | Training runs shall persist final trained model state. |  |  | UNTESTED |
| MIT-MAR-04 | All persisted artifacts shall be immutable once written. |  |  | UNTESTED |
| MIT-SYS-01 | Training runs shall record non-clinical evidence of execution. |  |  | UNTESTED |
| MIT-SYS-02 | Toolkit shall clearly distinguish research outputs from clinical performance claims. |  |  | UNTESTED |
| MIT-SYS-03 | All configuration inputs shall be persisted as version-controlled artifacts. |  |  | UNTESTED |
| MIT-SYS-04 | Toolkit shall support generation of traceability matrices linking requirements to tests. |  |  | UNTESTED |
| MIT-SYS-05 | Toolkit shall maintain compatibility with FDA SaMD design control documentation practices. |  |  | UNTESTED |
| MIT-TR-01 | Training runs shall be deterministic given identical seeds and configuration. |  | MIT_TR_01_tensor_datamodule_deterministic_ordering_20260212_145439_014482.json | PASS |
| MIT-TR-02 | Each training run shall generate a unique and immutable run identifier. |  |  | UNTESTED |
| MIT-TR-03 | Each training run shall persist dataset identity, split logic, and configuration. |  |  | UNTESTED |
| MIT-TR-04 | Training and validation datasets shall be disjoint. |  |  | UNTESTED |
| MIT-TR-05 | Training process shall record per-epoch training and validation metrics. |  |  | UNTESTED |
| MIT-TR-06 | Training process shall detect and fail on numerical instability. |  |  | UNTESTED |
| MIT-TR-07 | All training runs shall generate a machine-readable metrics file. |  |  | UNTESTED |
| MIT-TR-08 | All training runs shall generate a machine-readable evidence report. |  |  | UNTESTED |
| MIT-VAL-01 | Toolkit shall support structured evaluation of trained models on held-out datasets. |  |  | UNTESTED |
| MIT-VAL-02 | Evaluation outputs shall persist performance metrics in machine-readable form. |  |  | UNTESTED |
| MIT-VAL-03 | Evaluation shall be reproducible given identical model and dataset inputs. |  |  | UNTESTED |


---
Total Requirements: 36

Tested: 6

Failures: 0
