from datetime import datetime
from pathlib import Path

import torch

from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import MedicalImageDataSource
from medical_image_ai_toolkit.training.training_config import TrainingConfig
from medical_image_ai_toolkit.results.medical_image_validation_results import MedicalImageValidationResults


class ValidationPipeline:
    """
    Runs inference on the held-out test partition of a partitioned
    datasource and captures per-patient and aggregate metrics.

    Usage
    -----
    The datasource must already have partitions created (i.e.
    ``datasource.has_partitions()`` must return True) before calling
    ``run()``.  The training pipeline guarantees this when it is
    invoked first, but the validation pipeline can also be driven
    standalone by calling ``datasource.create_partitions(strategy)``
    beforehand.

    Parameters
    ----------
    datasource : MedicalImageDataSource
        A partitioned datasource.  The test split is used for
        validation.
    model : nn.Module
        A trained model.  The pipeline sets it to eval mode
        internally; the caller is responsible for loading weights
        before passing it in.
    config : TrainingConfig
        The same config object used during training.  The pipeline
        reads ``config.task`` to generate samples and compute loss.
    output_dir : str | Path, optional
        Root directory for validation artefacts.  Defaults to
        ``artifacts/validation_runs``.
    """

    def __init__(self, datasource, model, config, output_dir=None):

        self.datasource = datasource
        self.model = model
        self.config = config

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
        1. Guard-check that partitions exist.
        2. Set the model to eval mode.
        3. Iterate over test patients, compute per-patient loss.
        4. Aggregate metrics.
        5. Write a validation report JSON to the run directory.
        6. Return a populated MedicalImageValidationResults.
        """

        print("\n=== VALIDATION PIPELINE START ===")

        # 1. Require partitions
        if not self.datasource.has_partitions():
            raise RuntimeError(
                "Datasource has no partitions. "
                "Call datasource.create_partitions(strategy) before running "
                "the validation pipeline."
            )

        test_ids = self.datasource.get_test_ids()
        print(f"\nTest partition: {len(test_ids)} patient(s)")

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

        # 3. Eval mode — no gradient tracking
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

                for sample in task.generate_training_samples(patient_sample):

                    x = sample["input"].to(device)
                    y = sample["target"]

                    if isinstance(y, torch.Tensor):
                        y = y.to(device)

                    pred = self.model(x)
                    loss = task.compute_loss(pred, y)

                    patient_loss += loss.item()
                    patient_samples += 1

                per_patient_loss = (
                    patient_loss / patient_samples if patient_samples > 0 else None
                )

                per_patient_results.append(
                    {
                        "patient_id": patient_id,
                        "loss": per_patient_loss,
                        "n_samples": patient_samples,
                    }
                )

                total_loss += patient_loss
                total_samples += patient_samples

                loss_str = f"{per_patient_loss:.6f}" if per_patient_loss is not None else "N/A"
                print(f"  {patient_id}: loss={loss_str}  samples={patient_samples}")

        # 4. Aggregate
        mean_loss = total_loss / total_samples if total_samples > 0 else None

        results.per_patient_results = per_patient_results
        results.metrics = {
            "mean_loss": mean_loss,
            "num_test_patients": len(test_ids),
            "num_test_samples": total_samples,
        }

        results.mark_validation_complete()

        # 5. Export artefacts
        report_path = results.generate_validation_report()
        results.artifacts["validation_report"] = report_path

        results.summary()

        print("=== VALIDATION PIPELINE COMPLETE ===\n")

        return results