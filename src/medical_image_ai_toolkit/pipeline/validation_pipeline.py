from datetime import datetime
from pathlib import Path

import torch

from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import MedicalImageDataSource
from medical_image_ai_toolkit.training.training_config import TrainingConfig
from medical_image_ai_toolkit.validation.base_evaluator import BaseEvaluator
from medical_image_ai_toolkit.validation.segmentation_evaluator import SegmentationEvaluator
from medical_image_ai_toolkit.results.medical_image_validation_results import MedicalImageValidationResults


class ValidationPipeline:
    """
    Runs inference on the held-out test partition of a partitioned
    datasource and captures per-patient and aggregate metrics.

    Evaluator
    ---------
    Metric accumulation is delegated to a ``BaseEvaluator`` subclass so
    that the pipeline is not coupled to any particular task type.  Pass
    a custom evaluator to support object detection, classification, or
    any other model type.  When no evaluator is provided, the pipeline
    defaults to ``SegmentationEvaluator``, which computes Dice, IoU,
    precision, and recall — metrics that are robust to the foreground
    class imbalance common in medical segmentation.

    Partition handling
    ------------------
    The training pipeline writes a ``partitions.json`` file into its
    run directory alongside ``model.pt``.  Pass that file as
    ``partitions_path`` so the validation pipeline restores the exact
    same train / val / test split rather than re-splitting independently.

    If the datasource already has partitions loaded (e.g. running
    training and validation back-to-back in the same process),
    ``partitions_path`` may be omitted and the in-memory split is used.

    Parameters
    ----------
    datasource : MedicalImageDataSource
        A datasource pointing at the same dataset used during training.
    model : nn.Module
        A trained model.  The pipeline sets it to eval mode internally;
        the caller is responsible for loading weights beforehand.
    config : TrainingConfig
        The config object used during training.  ``config.task`` is used
        to generate samples and compute loss.
    evaluator : BaseEvaluator, optional
        Evaluator instance responsible for accumulating per-sample
        statistics and producing aggregate metrics.  Defaults to
        ``SegmentationEvaluator()``.
    partitions_path : str | Path, optional
        Path to the ``partitions.json`` produced by the training
        pipeline.  Required when the datasource does not already have
        partitions loaded.
    output_dir : str | Path, optional
        Root directory for validation artefacts.  Defaults to
        ``artifacts/validation_runs``.
    """

    def __init__(
        self,
        datasource,
        model,
        config,
        evaluator: BaseEvaluator = None,
        partitions_path=None,
        output_dir=None,
    ):
        self.datasource = datasource
        self.model = model
        self.config = config
        self.evaluator = evaluator if evaluator is not None else SegmentationEvaluator()
        self.partitions_path = Path(partitions_path) if partitions_path is not None else None

        self.output_dir = Path(output_dir or "artifacts/validation_runs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def run(self) -> MedicalImageValidationResults:
        """
        Execute the validation pipeline and return a results object.

        Steps
        -----
        1. Restore partitions from file, or confirm they are already set.
        2. Reset the evaluator so the pipeline can be safely re-run.
        3. Set the model to eval mode.
        4. Iterate over test patients; accumulate loss and delegate
           per-sample metric counting to the evaluator.
        5. Aggregate metrics from the evaluator.
        6. Write a validation report JSON to the run directory.
        7. Return a populated MedicalImageValidationResults.
        """

        print("\n=== VALIDATION PIPELINE START ===")

        # 1. Resolve partitions
        if self.partitions_path is not None:
            print(f"\nLoading partitions from: {self.partitions_path}")
            MedicalImageDataSource.load_partitions(self.datasource, self.partitions_path)
        elif not self.datasource.has_partitions():
            raise RuntimeError(
                "Datasource has no partitions and no partitions_path was provided. "
                "Either pass partitions_path pointing at the partitions.json from the "
                "training run, or ensure the datasource already has partitions loaded."
            )
        else:
            print("\nUsing in-memory partitions from datasource.")

        test_ids = self.datasource.get_test_ids()
        print(f"Test partition: {len(test_ids)} patient(s)")

        # 2. Prepare run directory and results container
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        results = MedicalImageValidationResults(
            model=self.model,
            config=self.config,
            datasource=self.datasource,
            run_dir=run_dir,
        )
        results.mark_validation_start()

        # 3. Reset evaluator and enter eval mode
        self.evaluator.reset()

        device = self.config.device
        self.model.to(device)
        self.model.eval()

        task = self.config.task
        if task is None:
            raise ValueError("TrainingConfig.task must be set for validation.")

        per_patient_results = []
        total_loss = 0.0
        total_samples = 0

        print("\nRunning inference on test patients...")

        with torch.no_grad():

            for patient_id in test_ids:

                patient_sample = self.datasource.get_patient(patient_id)

                patient_loss = 0.0
                patient_samples = 0

                # Per-patient evaluator to capture patient-level metrics
                patient_evaluator = self._make_patient_evaluator()
                patient_evaluator.reset()

                for sample in task.generate_training_samples(patient_sample):

                    x = sample["input"].to(device)
                    y = sample["target"]

                    if isinstance(y, torch.Tensor):
                        y = y.to(device)

                    pred = self.model(x)
                    loss = task.compute_loss(pred, y)

                    patient_loss += loss.item()
                    patient_samples += 1

                    # Global evaluator — aggregates across all patients
                    self.evaluator.update(pred, y)

                    # Patient-scoped evaluator — produces per-patient metrics
                    patient_evaluator.update(pred, y)

                per_patient_loss = (
                    patient_loss / patient_samples if patient_samples > 0 else None
                )

                patient_metrics = patient_evaluator.aggregate()

                per_patient_results.append(
                    {
                        "patient_id": patient_id,
                        "loss": per_patient_loss,
                        "n_samples": patient_samples,
                        **patient_metrics,
                    }
                )

                total_loss += patient_loss
                total_samples += patient_samples

                loss_str = f"{per_patient_loss:.6f}" if per_patient_loss is not None else "N/A"
                print(f"  {patient_id}: loss={loss_str}  samples={patient_samples}")

        # 5. Aggregate global metrics from evaluator
        aggregate_metrics = self.evaluator.aggregate()

        mean_loss = total_loss / total_samples if total_samples > 0 else None

        results.per_patient_results = per_patient_results
        results.metrics = {
            "mean_loss": mean_loss,
            "num_test_patients": len(test_ids),
            "num_test_samples": total_samples,
            **aggregate_metrics,
        }

        results.mark_validation_complete()

        # 6. Export artefacts
        report_path = results.generate_validation_report()
        results.artifacts["validation_report"] = report_path

        results.summary()

        print("=== VALIDATION PIPELINE COMPLETE ===\n")

        return results

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    def _make_patient_evaluator(self) -> BaseEvaluator:
        """
        Construct a fresh evaluator of the same type and configuration
        as ``self.evaluator`` for per-patient metric scoping.

        SegmentationEvaluator is handled explicitly.  For other
        evaluator types, a no-arg construction is attempted; authors of
        custom evaluators who need init parameters should override this
        method in a ValidationPipeline subclass.
        """
        if isinstance(self.evaluator, SegmentationEvaluator):
            return SegmentationEvaluator(threshold=self.evaluator.threshold)

        return type(self.evaluator)()