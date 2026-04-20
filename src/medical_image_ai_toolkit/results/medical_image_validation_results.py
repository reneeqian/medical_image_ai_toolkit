from datetime import datetime
from pathlib import Path
import json


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
        Aggregated metrics including mean_loss, evaluator-specific
        metrics (e.g. dice, iou, precision, recall), and raw confusion
        matrix counts where applicable.
    per_patient_results : list[dict]
        One entry per patient.  Written to the JSON report; not printed
        to the terminal.
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
        self.sample_viz_data: list = []   # populated by ValidationPipeline for figure generation

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
        """
        Print a concise terminal summary.

        Foreground-focused metrics (Dice, IoU, precision, recall) are
        shown first because they are robust to the severe class imbalance
        typical in medical segmentation — where background pixels vastly
        outnumber labelled pixels and can artificially inflate accuracy.
        Overall accuracy is shown last with an explicit warning.

        Per-patient detail is intentionally omitted here; it is written
        to validation_report.json.
        """

        print("\n==============================")
        print("Validation Results Summary")
        print("==============================")

        print(f"Model: {self.model.__class__.__name__}")

        if self.validation_start_time and self.validation_end_time:
            duration = self.validation_end_time - self.validation_start_time
            print(f"Validation time: {duration}")
        else:
            print("Validation still running")

        # --- Run-level counts ---
        mean_loss  = self.metrics.get("mean_loss")
        n_patients = self.metrics.get("num_test_patients", 0)
        n_samples  = self.metrics.get("num_test_samples", 0)

        print(f"\nPatients evaluated : {n_patients}")
        print(f"Total slices       : {n_samples}")
        if mean_loss is not None:
            print(f"Mean loss          : {mean_loss:.6f}")

        # --- Confusion matrix and derived metrics ---
        tp = self.metrics.get("tp", 0)
        fp = self.metrics.get("fp", 0)
        fn = self.metrics.get("fn", 0)
        tn = self.metrics.get("tn", 0)

        if tp + fp + fn + tn > 0:

            # Read pre-computed metrics from the evaluator where present;
            # fall back to computing inline for backwards compatibility.
            def _m(key, num, denom):
                if key in self.metrics:
                    return self.metrics[key]
                return num / denom if denom > 0 else float("nan")

            dice      = _m("dice",      2 * tp,  2 * tp + fp + fn)
            iou       = _m("iou",       tp,      tp + fp + fn)
            precision = _m("precision", tp,      tp + fp)
            recall    = _m("recall",    tp,      tp + fn)
            accuracy  = _m("accuracy",  tp + tn, tp + fp + fn + tn)

            print("\nPixel-level confusion matrix")
            print("  (rows = actual, cols = predicted)")
            print(f"  {'':15s}  {'Pred +':>12s}  {'Pred -':>12s}")
            print(f"  {'Actual +':15s}  {tp:>12,}  {fn:>12,}")
            print(f"  {'Actual -':15s}  {fp:>12,}  {tn:>12,}")

            print("\nForeground-focused metrics  (robust to class imbalance)")
            print(f"  Dice / F1  : {dice:.4f}")
            print(f"  IoU        : {iou:.4f}")
            print(f"  Precision  : {precision:.4f}")
            print(f"  Recall     : {recall:.4f}")

            print("\nOverall metrics  (unreliable when foreground pixels are sparse)")
            print(f"  Accuracy   : {accuracy:.4f}")

        print("\n(Per-patient detail written to validation_report.json)")
        print("==============================\n")

    def generate_validation_report(self) -> Path:
        """
        Writes a JSON report to run_dir and returns its path.

        Includes full metrics dict and per-patient results.  The report
        structure mirrors the training report so downstream tooling can
        treat both uniformly.
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