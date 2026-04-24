import random as _random
from datetime import datetime
from pathlib import Path

import torch

from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import MedicalImageDataSource
from medical_image_ai_toolkit.training.training_config import TrainingConfig
from medical_image_ai_toolkit.validation.base_evaluator import BaseEvaluator
from medical_image_ai_toolkit.validation.segmentation_evaluator import SegmentationEvaluator
from medical_image_ai_toolkit.results.medical_image_model_testing_results import MedicalImageModelTestingResults
from medical_image_ai_toolkit.visualizations.model_testing_figures import (
    plot_training_curve,
    plot_confusion_matrix,
    plot_patient_samples,
)


class ModelTestingPipeline:
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
    ``partitions_path`` so the model testing pipeline restores the exact
    same train / val / test split rather than re-splitting independently.

    If the datasource already has partitions loaded (e.g. running
    training and model testing back-to-back in the same process),
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
        Root directory for model testing artefacts.  Defaults to
        ``artifacts/model_testing_runs``.
    """

    def __init__(
        self,
        datasource,
        model,
        config,
        evaluator: BaseEvaluator = None,
        partitions_path=None,
        output_dir=None,
        training_history: list = None,
        generate_figures: bool = True,
    ):
        self.datasource = datasource
        self.model = model
        self.config = config
        self.evaluator = evaluator if evaluator is not None else SegmentationEvaluator()
        self.partitions_path = Path(partitions_path) if partitions_path is not None else None

        self.output_dir = Path(output_dir or "artifacts/model_testing_runs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_history = training_history
        self.generate_figures = generate_figures

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def run(self) -> MedicalImageModelTestingResults:
        """
        Execute the model testing pipeline and return a results object.

        Steps
        -----
        1. Restore partitions from file, or confirm they are already set.
        2. Reset the evaluator so the pipeline can be safely re-run.
        3. Set the model to eval mode.
        4. Iterate over test patients; accumulate loss and delegate
           per-sample metric counting to the evaluator.
        5. Aggregate metrics from the evaluator.
        6. Write a model testing report JSON to the run directory.
        7. Return a populated MedicalImageModelTestingResults.
        """

        print("\n=== MODEL TESTING PIPELINE START ===")

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

        results = MedicalImageModelTestingResults(
            model=self.model,
            config=self.config,
            datasource=self.datasource,
            run_dir=run_dir,
        )
        results.mark_testing_start()

        # 3. Reset evaluator and enter eval mode
        self.evaluator.reset()

        device = self.config.device
        self.model.to(device)
        self.model.eval()

        task = self.config.task
        if task is None:
            raise ValueError("TrainingConfig.task must be set for model testing.")

        per_patient_results = []
        total_loss = 0.0
        total_samples = 0

        # Select up to 3 random patients for visualization (seeded for reproducibility)
        n_viz = min(3, len(test_ids))
        viz_ids = set(_random.Random(42).sample(test_ids, n_viz))
        _viz_best = {}   # pid -> (score, slice_idx, hu, gt_mask, pred_prob)

        print("\nRunning inference on test patients...")

        with torch.no_grad():

            for patient_id in test_ids:

                patient_sample = self.datasource.get_patient(patient_id)

                patient_loss = 0.0
                patient_samples = 0

                # Per-patient evaluator to capture patient-level metrics
                patient_evaluator = self._make_patient_evaluator()
                patient_evaluator.reset()

                _slice_idx = 0
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

                    # Capture best slice for visualization (prefer annotated slices)
                    if patient_id in viz_ids:
                        gt_sum = float(y.sum().item())
                        pred_prob = torch.sigmoid(pred).squeeze().cpu().detach().numpy()
                        score = gt_sum if gt_sum > 0 else pred_prob.mean()
                        if patient_id not in _viz_best or score > _viz_best[patient_id][0]:
                            _viz_best[patient_id] = (
                                score,
                                _slice_idx,
                                x.squeeze().cpu().numpy(),
                                y.squeeze().cpu().numpy(),
                                pred_prob,
                            )
                    _slice_idx += 1

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

        results.mark_testing_complete()

        # 6. Populate viz data
        for pid, (_, slice_idx, hu, gt_mask, pred_prob) in _viz_best.items():
            results.sample_viz_data.append({
                "patient_id": pid,
                "slice_idx":  slice_idx,
                "hu":         hu,
                "gt_mask":    gt_mask,
                "pred":       pred_prob,
            })

        # 7. Export artefacts
        report_path = results.generate_testing_report()
        results.artifacts["testing_report"] = report_path

        if self.generate_figures:
            results.artifacts.update(self._generate_figures(results, run_dir))

        results.summary()

        print("=== MODEL TESTING PIPELINE COMPLETE ===\n")

        return results

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    def _generate_figures(self, results, run_dir: Path) -> dict:
        artifacts = {}
        try:
            if self.training_history:
                p = plot_training_curve(self.training_history, run_dir / "training_curve.png")
                artifacts["training_curve"] = p
                print(f"Figure saved: {p}")
        except Exception as e:
            print(f"Warning: could not generate training curve: {e}")

        try:
            p = plot_confusion_matrix(results.metrics, run_dir / "confusion_matrix.png")
            artifacts["confusion_matrix"] = p
            print(f"Figure saved: {p}")
        except Exception as e:
            print(f"Warning: could not generate confusion matrix: {e}")

        try:
            if results.sample_viz_data:
                p = plot_patient_samples(results.sample_viz_data, run_dir / "patient_samples.png")
                artifacts["patient_samples"] = p
                print(f"Figure saved: {p}")
        except Exception as e:
            print(f"Warning: could not generate patient samples figure: {e}")

        return artifacts

    def _make_patient_evaluator(self) -> BaseEvaluator:
        """
        Construct a fresh evaluator of the same type and configuration
        as ``self.evaluator`` for per-patient metric scoping.

        SegmentationEvaluator is handled explicitly.  For other
        evaluator types, a no-arg construction is attempted; authors of
        custom evaluators who need init parameters should override this
        method in a ModelTestingPipeline subclass.
        """
        if isinstance(self.evaluator, SegmentationEvaluator):
            return SegmentationEvaluator(threshold=self.evaluator.threshold)

        return type(self.evaluator)()
