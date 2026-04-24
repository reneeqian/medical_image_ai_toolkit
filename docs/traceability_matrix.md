<!-- AUTO-GENERATED FILE. DO NOT EDIT MANUALLY. -->

# Requirements Traceability Matrix

## Requirement Coverage

**Coverage:** 90.0% (36 / 40 requirements tested)

## Code Coverage

**Line Coverage:** 85.7%

Detailed uncovered lines saved in `artifacts/coverage/uncovered_lines.txt`

| Requirement ID | Title | Linked Tests | Evidence Artifacts | Status |
|----------------|-------------|--------------|--------------------|--------|
| DAT-001 | Patient Sample Structure Validation | tests/test_patient_sample_contract.py::test_patient_id_required, tests/test_patient_sample_contract.py::test_spacing_must_be_positive, tests/test_patient_sample_contract.py::test_valid_patient_sample_passes_contract, tests/test_patient_sample_contract.py::test_volume_must_be_3d |  | LINKED |
| DAT-002 | Annotation Structure Compliance | tests/test_patient_sample_contract.py::test_annotations_required_missing, tests/test_patient_sample_contract.py::test_dense_mask_annotations_allowed, tests/test_patient_sample_contract.py::test_invalid_annotation_type_rejected, tests/test_patient_sample_contract.py::test_missing_annotations_allowed_when_not_required, tests/test_patient_sample_contract.py::test_vector_rois_none, tests/test_patient_sample_contract.py::test_vector_rois_not_dict |  | LINKED |
| DAT-003 | Annotation Boundary Validation | tests/test_patient_sample_contract.py::test_roi_contour_shape_validation, tests/test_patient_sample_contract.py::test_roi_coordinates_out_of_bounds, tests/test_patient_sample_contract.py::test_roi_slice_out_of_bounds, tests/test_patient_sample_contract.py::test_roi_wrong_type, tests/test_patient_sample_contract.py::test_roi_y_out_of_bounds, tests/test_patient_sample_contract.py::test_rois_not_list, tests/test_patient_sample_contract.py::test_slice_index_not_int |  | LINKED |
| DAT-004 | Patient Sample Boundary Validation | tests/test_datasource_contract.py::test_datasource_behaviors, tests/test_datasource_contract.py::test_invalid_patient_id_raises, tests/test_datasource_contract.py::test_show_slice_basic, tests/test_datasource_contract.py::test_show_slice_with_annotations_and_custom_get_sample |  | LINKED |
| DAT-005 | Patient Sample To Tensor Conversion | tests/test_patient_sample.py::test_patient_sample_basic, tests/test_patient_sample.py::test_patient_sample_repr_with_annotations |  | LINKED |
| DAT-006 | Patient Indexing without Loading | tests/test_datasource_contract.py::test_dataset_partition_generation, tests/test_datasource_contract.py::test_get_num_patients_and_get_patient |  | LINKED |
| DAT-007 | Dataset Partition Generation | tests/test_datasource_contract.py::test_access_before_partition_raises, tests/test_datasource_contract.py::test_dataset_partition_generation, tests/test_datasource_contract.py::test_datasource_behaviors, tests/test_datasource_contract.py::test_has_partitions_flag |  | LINKED |
| DAT-008 | Deterministic Dataset Partitioning | tests/test_datasource_contract.py::test_dataset_partition_generation, tests/test_datasource_contract.py::test_partitions_are_deterministic |  | LINKED |
| DAT-009 | Training Tensor Extraction | tests/test_task_definition.py::test_task_generates_aligned_samples |  | LINKED |
| DAT-010 | Slice-Level Training Data Exposure | tests/test_datasource_contract.py::test_show_slice_with_annotations_and_custom_get_sample, tests/test_datasource_edge_cases.py::test_datasource_edge_cases, tests/test_task_definition.py::test_task_slice_level_iteration |  | LINKED |
| DOC-001 | Machine-Readable Requirements Definition | tests/test_project_structure.py::test_project_documentation_structure |  | LINKED |
| DOC-002 | Basic Project Documentation | tests/test_project_structure.py::test_project_documentation_structure |  | LINKED |
| DOC-003 | Training Workflow Documentation |  |  | UNTESTED |
| DOC-004 | Training Report Generation | tests/test_training_results.py::test_mark_training_complete_summary, tests/test_training_results.py::test_summary_training_running, tests/test_training_results.py::test_training_results_artifact_generation, tests/test_training_results.py::test_training_run_records_lifecycle_timestamps |  | LINKED |
| MOD-001 | Model Optimization Support | tests/test_training_config.py::test_training_config_optimizer_class |  | LINKED |
| MOD-002 | Artifact Immutability |  |  | UNTESTED |
| MOD-003 | Model Artifact Persistence | tests/test_training_results.py::test_training_results_artifact_generation |  | LINKED |
| MOD-004 | Model Artifact Loading |  |  | UNTESTED |
| MOD-005 | Model Export Support | tests/test_training_results.py::test_export_model_failure, tests/test_training_results.py::test_training_results_artifact_generation |  | LINKED |
| MOD-006 | Inference Execution Support | tests/test_training_results.py::test_inference_determinism, tests/test_training_results.py::test_results_inference |  | LINKED |
| SYS-001 | Dataset Interface Consistency | tests/test_datasource_contract.py::test_dataset_partition_generation, tests/test_datasource_contract.py::test_datasource_behaviors |  | LINKED |
| SYS-002 | Training Pipeline Orchestration | tests/test_medical_image_trainer.py::test_trainer_sanity_check, tests/test_medical_image_trainer.py::test_training_pipeline_generates_artifacts, tests/test_medical_image_trainer.py::test_training_with_no_patients, tests/test_medical_image_trainer.py::test_training_without_partitions_fails |  | LINKED |
| SYS-003 | Pipeline Configuration Interface | tests/test_medical_image_trainer.py::test_trainer_sanity_check, tests/test_training_config.py::test_training_config_initialization |  | LINKED |
| SYS-004 | Training Result Aggregation | tests/test_medical_image_trainer.py::test_training_pipeline_generates_artifacts, tests/test_training_results.py::test_results_artifact_registration |  | LINKED |
| SYS-005 | Explicit Dataset Validation | tests/test_medical_image_trainer.py::test_training_pipeline_stops_on_dataset_validation_errors |  | LINKED |
| TRN-001 | Configurable Training Execution | tests/test_medical_image_trainer.py::test_training_respects_device, tests/test_medical_image_trainer.py::test_training_without_task_fails, tests/test_training_config.py::test_training_config_initialization |  | LINKED |
| TRN-002 | Hyperparameter Sweep Support |  |  | UNTESTED |
| TRN-003 | Training Shall Detect Nan Loss | tests/test_medical_image_trainer.py::test_training_detects_nan_loss |  | LINKED |
| TRN-004 | Training Run Artifact Generation | tests/test_medical_image_trainer.py::test_training_pipeline_generates_artifacts |  | LINKED |
| TRN-005 | Training Checkpoint Support | tests/test_medical_image_trainer.py::test_early_stop_disabled_runs_all_epochs, tests/test_medical_image_trainer.py::test_early_stop_loss_threshold_halts_training, tests/test_medical_image_trainer.py::test_early_stop_plateau_halts_training, tests/test_training_config.py::test_training_config_early_stop_can_be_disabled, tests/test_training_config.py::test_training_config_early_stop_custom_values, tests/test_training_config.py::test_training_config_early_stop_defaults |  | LINKED |
| TRN-006 | Deterministic Training Initialization | tests/test_medical_image_trainer.py::test_training_is_deterministic |  | LINKED |
| TRN-007 | Loss Function Configuration | tests/test_task_definition.py::test_task_compute_loss, tests/test_training_config.py::test_training_config_uses_task_loss_interface |  | LINKED |
| TRN-008 | Training Input Interface | tests/test_task_definition.py::test_postprocess_prediction_identity, tests/test_task_definition.py::test_task_generates_aligned_samples |  | LINKED |
| VER-001 | Deterministic Training Behavior | tests/test_deterministic_split.py::test_different_seeds_produce_different_splits, tests/test_deterministic_split.py::test_split_is_deterministic, tests/test_medical_image_trainer.py::test_training_is_deterministic |  | LINKED |
| VER-002 | Training Metrics Recording | tests/test_medical_image_trainer.py::test_training_pipeline_generates_artifacts, tests/test_training_results.py::test_training_results_artifact_generation |  | LINKED |
| VER-003 | Dataset Separation Enforcement | tests/test_datasource_contract.py::test_dataset_partition_generation, tests/test_deterministic_split.py::test_split_covers_all_patients, tests/test_deterministic_split.py::test_split_max_caps_are_respected, tests/test_deterministic_split.py::test_split_on_minimal_patient_count, tests/test_deterministic_split.py::test_split_partitions_are_non_overlapping, tests/test_medical_image_trainer.py::test_dataset_partitions_do_not_overlap |  | LINKED |
| VER-004 | Model Evaluation Execution | tests/test_model_testing_pipeline.py::test_model_testing_pipeline_generates_metrics_and_report, tests/test_model_testing_pipeline.py::test_model_testing_pipeline_requires_existing_partitions |  | LINKED |
| VER-005 | Evaluation Metric Recording | tests/test_model_testing_pipeline.py::test_model_testing_pipeline_generates_metrics_and_report, tests/test_model_testing_pipeline.py::test_segmentation_evaluator_records_counts_and_reset |  | LINKED |
| VER-006 | Post-Training Validation Support | tests/test_model_testing_pipeline.py::test_model_testing_pipeline_generate_figures_false_skips_figures, tests/test_model_testing_pipeline.py::test_model_testing_pipeline_generate_figures_true_creates_files, tests/test_model_testing_pipeline.py::test_model_testing_pipeline_generates_metrics_and_report, tests/test_model_testing_pipeline.py::test_model_testing_pipeline_requires_task_definition, tests/test_model_testing_pipeline.py::test_model_testing_pipeline_supports_custom_evaluator |  | LINKED |
| VER-007 | Inference Consistency Verification | tests/test_training_results.py::test_inference_determinism |  | LINKED |


---

## Untested Requirements

- DOC-003
- MOD-002
- MOD-004
- TRN-002


---
Total Requirements: 40

Tested: 36

Failures: 0
