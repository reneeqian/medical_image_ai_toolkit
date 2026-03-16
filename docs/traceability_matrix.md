<!-- AUTO-GENERATED FILE. DO NOT EDIT MANUALLY. -->

# Requirements Traceability Matrix

## Requirement Coverage

**Coverage:** 7.7% (3 / 39 requirements tested)

## Code Coverage

**Line Coverage:** 87.7%

Detailed uncovered lines saved in `artifacts/coverage/uncovered_lines.txt`

| Requirement ID | Title | Linked Tests | Evidence Artifacts | Status |
|----------------|-------------|--------------|--------------------|--------|
| DAT-001 | Patient Sample Structure Validation | tests/test_patient_sample_contract.py::test_patient_id_required, tests/test_patient_sample_contract.py::test_spacing_must_be_positive, tests/test_patient_sample_contract.py::test_valid_patient_sample_passes_contract, tests/test_patient_sample_contract.py::test_volume_must_be_3d |  | UNTESTED |
| DAT-002 | Annotation Structure Compliance | tests/test_patient_sample_contract.py::test_annotations_required_missing, tests/test_patient_sample_contract.py::test_dense_mask_annotations_allowed, tests/test_patient_sample_contract.py::test_invalid_annotation_type_rejected, tests/test_patient_sample_contract.py::test_missing_annotations_allowed_when_not_required, tests/test_patient_sample_contract.py::test_vector_rois_none, tests/test_patient_sample_contract.py::test_vector_rois_not_dict |  | UNTESTED |
| DAT-003 | Annotation Boundary Validation | tests/test_patient_sample_contract.py::test_roi_contour_shape_validation, tests/test_patient_sample_contract.py::test_roi_coordinates_out_of_bounds, tests/test_patient_sample_contract.py::test_roi_slice_out_of_bounds, tests/test_patient_sample_contract.py::test_roi_wrong_type, tests/test_patient_sample_contract.py::test_roi_y_out_of_bounds, tests/test_patient_sample_contract.py::test_rois_not_list, tests/test_patient_sample_contract.py::test_slice_index_not_int |  | UNTESTED |
| DAT-004 | Patient Sample Boundary Validation | tests/test_datasource_contract.py::test_datasource_behaviors, tests/test_datasource_contract.py::test_show_slice_basic |  | UNTESTED |
| DAT-005 | Patient Sample To Tensor Conversion | tests/test_patient_sample.py::test_patient_sample_basic, tests/test_patient_sample.py::test_patient_sample_repr_with_annotations |  | UNTESTED |
| DAT-006 | Patient Indexing without Loading | tests/test_datasource_contract.py::test_dataset_partition_generation, tests/test_datasource_contract.py::test_get_num_patients_and_get_patient |  | UNTESTED |
| DAT-007 | Dataset Partition Generation | tests/test_datasource_contract.py::test_dataset_partition_generation, tests/test_datasource_contract.py::test_datasource_behaviors, tests/test_datasource_contract.py::test_has_partitions_flag |  | UNTESTED |
| DAT-008 | Deterministic Dataset Partitioning | tests/test_datasource_contract.py::test_dataset_partition_generation |  | UNTESTED |
| DAT-009 | Training Tensor Extraction |  |  | UNTESTED |
| DAT-010 | Slice-Level Training Data Exposure | tests/test_datasource_edge_cases.py::test_datasource_edge_cases |  | UNTESTED |
| DOC-001 | Machine-Readable Requirements Definition | tests/test_project_structure.py::test_project_documentation_structure | project_documentation_structure_20260316_103409_266979.json | PASS |
| DOC-002 | Basic Project Documentation | tests/test_project_structure.py::test_project_documentation_structure | project_documentation_structure_20260316_103409_266979.json | PASS |
| DOC-003 | Training Workflow Documentation |  |  | UNTESTED |
| DOC-004 | Training Report Generation | tests/test_training_results.py::test_mark_training_complete_summary, tests/test_training_results.py::test_summary_training_running, tests/test_training_results.py::test_training_results_artifact_generation |  | UNTESTED |
| MOD-001 | Model Optimization Support |  |  | UNTESTED |
| MOD-002 | Artifact Immutability |  |  | UNTESTED |
| MOD-003 | Model Artifact Persistence | tests/test_training_results.py::test_training_results_artifact_generation |  | UNTESTED |
| MOD-004 | Model Artifact Loading |  |  | UNTESTED |
| MOD-005 | Model Export Support | tests/test_training_results.py::test_export_model_failure, tests/test_training_results.py::test_training_results_artifact_generation |  | UNTESTED |
| MOD-006 | Inference Execution Support | tests/test_training_results.py::test_inference_determinism, tests/test_training_results.py::test_results_inference |  | UNTESTED |
| SYS-001 | Dataset Interface Consistency | tests/test_datasource_contract.py::test_dataset_partition_generation, tests/test_datasource_contract.py::test_datasource_behaviors |  | UNTESTED |
| SYS-002 | Training Pipeline Orchestration | tests/test_medical_image_trainer.py::test_trainer_sanity_check, tests/test_medical_image_trainer.py::test_training_pipeline_generates_artifacts |  | UNTESTED |
| SYS-003 | Pipeline Configuration Interface | tests/test_medical_image_trainer.py::test_trainer_sanity_check |  | UNTESTED |
| SYS-004 | Training Result Aggregation | tests/test_medical_image_trainer.py::test_training_pipeline_generates_artifacts |  | UNTESTED |
| TRN-001 | Configurable Training Execution | tests/test_task_definition.py::test_postprocess_prediction_default, tests/test_task_definition.py::test_postprocess_prediction_passthrough, tests/test_task_definition.py::test_task_definition_cannot_instantiate_abstract, tests/test_task_definition.py::test_task_definition_interface |  | UNTESTED |
| TRN-002 | Hyperparameter Sweep Support |  |  | UNTESTED |
| TRN-003 | Training Shall Detect Nan Loss | tests/test_medical_image_trainer.py::test_training_detects_nan_loss | TRN003_nan_loss_detection_20260316_103409_205924.json | PASS |
| TRN-004 | Training Run Artifact Generation | tests/test_medical_image_trainer.py::test_training_pipeline_generates_artifacts |  | UNTESTED |
| TRN-005 | Training Checkpoint Support |  |  | UNTESTED |
| TRN-006 | Deterministic Training Initialization | tests/test_medical_image_trainer.py::test_training_is_deterministic |  | UNTESTED |
| TRN-007 | Loss Function Configuration |  |  | UNTESTED |
| TRN-008 | Training Input Interface |  |  | UNTESTED |
| VER-001 | Deterministic Training Behavior | tests/test_medical_image_trainer.py::test_training_is_deterministic |  | UNTESTED |
| VER-002 | Training Metrics Recording | tests/test_medical_image_trainer.py::test_training_pipeline_generates_artifacts, tests/test_training_results.py::test_training_results_artifact_generation |  | UNTESTED |
| VER-003 | Dataset Separation Enforcement | tests/test_datasource_contract.py::test_dataset_partition_generation, tests/test_medical_image_trainer.py::test_dataset_partitions_do_not_overlap |  | UNTESTED |
| VER-004 | Model Evaluation Execution |  |  | UNTESTED |
| VER-005 | Evaluation Metric Recording |  |  | UNTESTED |
| VER-006 | Post-Training Validation Support |  |  | UNTESTED |
| VER-007 | Inference Consistency Verification | tests/test_training_results.py::test_inference_determinism |  | UNTESTED |


---
### Untested Requirements

- DAT-001
- DAT-002
- DAT-003
- DAT-004
- DAT-005
- DAT-006
- DAT-007
- DAT-008
- DAT-009
- DAT-010
- DOC-003
- DOC-004
- MOD-001
- MOD-002
- MOD-003
- MOD-004
- MOD-005
- MOD-006
- SYS-001
- SYS-002
- SYS-003
- SYS-004
- TRN-001
- TRN-002
- TRN-004
- TRN-005
- TRN-006
- TRN-007
- TRN-008
- VER-001
- VER-002
- VER-003
- VER-004
- VER-005
- VER-006
- VER-007



---
Total Requirements: 39

Tested: 3

Failures: 0
