# medical_image_ai_toolkit – Requirements

## 1. Purpose

`medical_image_ai_toolkit` is a reusable training infrastructure module for medical imaging research.

It provides PyTorch-compatible abstractions for converting validated domain-agnostic data objects into tensors and exposing them through training-ready interfaces.

This module **does not own clinical semantics, data ingestion, or dataset curation**.

---

## 2. System Scope

### In Scope
- Conversion of valid `PatientSample` objects to tensors
- PyTorch `Dataset` / DataModule abstractions
- Training-time validation of tensor shape, dtype, and presence
- Infrastructure to support model training and evaluation

### Out of Scope
- Data ingestion or file system access
- Clinical interpretation or labeling semantics
- Dataset curation or annotation validation
- Any domain-specific assumptions (e.g., CAC, organ type)

---

## 3. Functional Requirements

### FR-1: PatientSample Input Contract
`medical_image_ai_toolkit` shall accept `PatientSample` objects as its primary input data unit.

The module shall not accept raw files, paths, or unstructured dictionaries as training inputs.

---

### FR-2: Tensor Adaptation
`medical_image_ai_toolkit` shall convert `PatientSample` objects into PyTorch tensors using adapter components (e.g., `patient_sample_to_tensor`).

Tensor conversion logic shall be isolated from training logic.

---

### FR-3: Torch Dataset Abstraction
`medical_image_ai_toolkit` shall expose PyTorch-compatible `Dataset` (and/or DataModule) abstractions for use in model training loops.

These abstractions shall operate solely on in-memory `PatientSample` objects.

---

## 4. Data Contract Enforcement Requirements

### DR-1: Input Completeness Enforcement
The system shall enforce that all required fields needed for tensor conversion are present in a `PatientSample`.

Missing or malformed inputs shall raise explicit, actionable errors.

---

### DR-2: Tensor Integrity Enforcement
The system shall enforce tensor properties required for training, including:
- Expected dimensionality
- Data type consistency
- Batch compatibility

Enforcement failures shall be detected prior to model execution.

---

## 5. Non-Functional Requirements

### NFR-1: Domain Agnosticism
`medical_image_ai_toolkit` shall not encode domain-specific or clinical assumptions.

All domain semantics shall be enforced upstream.

---

### NFR-2: Testability Without External Data
Core components shall be testable using synthetic or in-memory `PatientSample` instances.

No test shall require access to external storage or real datasets.

---

## 6. Traceability

Each requirement in this document shall be traceable to:
- One or more implementation artifacts
- One or more enforcement or test artifacts

Traceability shall be maintained independently of any consuming project.
