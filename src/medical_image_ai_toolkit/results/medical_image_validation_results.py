from datetime import datetime
from pathlib import Path
import json
import torch


class MedicalImageValidationResults:
    """
    Container for the outputs of a validation run.

    Mirrors MedicalImageTrainingResults in structure and lifecycle,
    but is scoped to the held-out test partition rather than the
    training loop.

    Attributes
    ----------
    model : nn.Module
        The model that was evaluated.
    config : TrainingConfig
        The training configuration associated with the model.
    datasource : MedicalImageDataSource
        The datasource whose test partition was evaluated.
    run_dir : Path
        Directory where validation artefacts are written.
    metrics : dict
        Aggregated metrics produced by the validation run
        (e.g. mean_loss, num_test_samples).
    per_patient_results : list[dict]
        One entry per patient with keys: patient_id, loss, n_samples.
    artifacts : dict
        Paths of any files written during validation.
    """

    def __init__(self, model, config, datasource, run_dir):

        self.model = model
        self.config = config
        self.datasource = datasource

        self.run_dir = Path(run_dir)

        self.metrics: dict = {}
        self.per_patient_results: list = []
        self.artifacts: dict = {}

        # lifecycle timestamps
        self.validation_start_time: datetime | None = None
        self.validation_end_time: datetime | None = None

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------

    def mark_validation_start(self):
        self.validation_start_time = datetime.now()

    def mark_validation_complete(self):
        self.validation_end_time = datetime.now()

    # --------------------------------------------------
    # Reporting
    # --------------------------------------------------

    def summary(self):

        print("\n==============================")
        print("Validation Results Summary")
        print("==============================")

        print(f"Model: {self.model.__class__.__name__}")

        if self.validation_start_time and self.validation_end_time:
            duration = self.validation_end_time - self.validation_start_time
            print(f"Validation time: {duration}")
        else:
            print("Validation still running")

        if self.metrics:
            print("\nMetrics")
            for k, v in self.metrics.items():
                print(f"  {k}: {v}")

        if self.per_patient_results:
            print(f"\nPer-patient results ({len(self.per_patient_results)} patients)")
            for entry in self.per_patient_results:
                pid = entry.get("patient_id", "?")
                loss = entry.get("loss")
                n = entry.get("n_samples", 0)
                loss_str = f"{loss:.6f}" if loss is not None else "N/A"
                print(f"  {pid}: loss={loss_str}  samples={n}")

        print("==============================\n")

    def generate_validation_report(self) -> Path:
        """
        Writes a JSON report to run_dir and returns its path.

        The report structure mirrors the training report so downstream
        tooling can treat both uniformly.
        """

        report_path = self.run_dir / "validation_report.json"

        report = {
            "metrics": self.metrics,
            "per_patient_results": self.per_patient_results,
            "config": {k: str(v) for k, v in vars(self.config).items()},
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Validation report written to: {report_path}")
        return report_path