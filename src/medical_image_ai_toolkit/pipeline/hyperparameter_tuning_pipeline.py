from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch.nn as nn
from regulatory_tools.evidence.evidence_report import EvidenceReport

from medical_image_ai_toolkit.dataobjects.data_validation.dataset_validator import (
    DatasetValidator,
    summarize_dataset_validation,
)
from medical_image_ai_toolkit.results.hyperparameter_tuning_results import (
    HyperparameterTuningResults,
    _trial_result_from_training_results,
)
from medical_image_ai_toolkit.training.medical_image_trainer import MedicalImageTrainer
from medical_image_ai_toolkit.tuning.hyperparameter_space import HyperparameterSpace

if TYPE_CHECKING:
    from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import (
        MedicalImageDataSource,
    )
    from medical_image_ai_toolkit.results.hyperparameter_tuning_results import TrialResult
    from medical_image_ai_toolkit.training.training_config import TrainingConfig

logger = logging.getLogger(__name__)


class HyperparameterTuningPipeline:
    """
    Grid or random-search over hyperparameter combinations.

    Stages
    ------
    1. **Dataset validation** — run once; abort on errors.
    2. **Partitioning** — create shared partitions once (skipped if they
       already exist), ensuring all trials operate on identical splits.
    3. **Trial loop** — for each parameter combination, instantiate a model
       and config via *trial_factory*, then run ``MedicalImageTrainer.train()``.
    4. **Best-trial selection** — pick the trial with the lowest validation
       loss (or training loss when no validation partition is present).
    5. **Report** — write ``tuning_report.json`` to the run directory.

    Parameters
    ----------
    datasource :
        Dataset to train on.
    trial_factory :
        Callable ``(params: dict) -> (nn.Module, TrainingConfig)``. Called
        once per trial; receives the current parameter combination and must
        return a fresh model instance and a fully configured ``TrainingConfig``
        (``split_strategy`` on the config is ignored — the pipeline's own
        ``split_strategy`` is used).
    space :
        Search space defining parameter names and candidate value lists.
    split_strategy :
        Strategy used to create the shared data partitions (an object with a
        ``split(patient_ids) -> (train_ids, val_ids, test_ids)`` method).
    output_dir :
        Root directory for tuning run artifacts. Defaults to
        ``artifacts/tuning_runs``.
    search_strategy :
        ``"grid"`` (default) exhausts all combinations; ``"random"`` draws
        *n_trials* combinations without replacement.
    n_trials :
        Number of trials for random search. Required when
        ``search_strategy="random"``.
    random_seed :
        Seed for random sampling reproducibility.
    """

    def __init__(
        self,
        datasource: MedicalImageDataSource,
        trial_factory: Callable[[dict[str, Any]], tuple[nn.Module, TrainingConfig]],
        space: HyperparameterSpace,
        split_strategy: Any,
        output_dir: str | Path | None = None,
        search_strategy: str = "grid",
        n_trials: int | None = None,
        random_seed: int | None = None,
    ) -> None:
        self.datasource = datasource
        self.trial_factory = trial_factory
        self.space = space
        self.split_strategy = split_strategy
        self.output_dir = Path(output_dir or "artifacts/tuning_runs")
        self.search_strategy = search_strategy
        self.n_trials = n_trials
        self.random_seed = random_seed

    def run(self) -> HyperparameterTuningResults:
        """Execute all stages and return results."""

        logger.info("=== HYPERPARAMETER TUNING START ===")
        _run_start = time.time()

        # 1. Validate dataset (once)
        logger.info("Validating dataset...")
        ds_validator = DatasetValidator(self.datasource, req_provider=None)
        ds_validation_report = ds_validator.run()
        ds_validation_report.print_summary()
        summarize_dataset_validation(ds_validation_report)

        if ds_validation_report.has_errors:
            raise RuntimeError(
                "Dataset validation failed; aborting tuning pipeline.\n"
                f"{ds_validation_report.to_string()}"
            )

        tuning_evidence = EvidenceReport(subject="Hyperparameter tuning run")
        tuning_evidence.merge(ds_validation_report, "dataset_validation")

        # 2. Create shared partitions (once)
        if not self.datasource.has_partitions():
            logger.info("Creating partitions...")
            self.datasource.create_partitions(self.split_strategy)

        # 3. Determine trial list
        if self.search_strategy == "grid":
            trial_params_list = self.space.grid()
        elif self.search_strategy == "random":
            if self.n_trials is None:
                raise ValueError("n_trials must be set when search_strategy='random'")
            trial_params_list = self.space.random_sample(self.n_trials, self.random_seed)
        else:
            raise ValueError(f"Unknown search_strategy: {self.search_strategy!r}")

        logger.info(
            "Running %d trial(s) (%s search).",
            len(trial_params_list),
            self.search_strategy,
        )
        tuning_evidence.info(
            f"search_strategy={self.search_strategy}  num_trials={len(trial_params_list)}"
            f"  space={dict(self.space._params)}",
            "TRN-002",
        )

        # Per-tuning-run output directory
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tuning_run_dir = self.output_dir / run_ts
        tuning_run_dir.mkdir(parents=True, exist_ok=True)

        _W = 68
        print("\n" + "=" * _W)
        print(f"  HYPERPARAMETER TUNING  |  {len(trial_params_list)} trial(s)  |  {self.search_strategy} search")
        print("=" * _W)
        for k, v in self.space._params.items():
            print(f"  {k}: {v}")
        print(f"  output: {tuning_run_dir}")
        print("-" * _W)

        # 4. Trial loop
        trial_results: list[TrialResult] = []
        trial_models: list[nn.Module] = []
        trial_configs: list[TrainingConfig] = []

        for trial_id, params in enumerate(trial_params_list):
            logger.info("--- Trial %d/%d  params=%s ---", trial_id, len(trial_params_list) - 1, params)
            trial_dir = tuning_run_dir / f"trial_{trial_id:03d}"
            trial_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[Trial {trial_id + 1}/{len(trial_params_list)}]  {params}")
            _trial_start = time.time()

            model, config = self.trial_factory(params)

            trainer = MedicalImageTrainer(
                self.datasource,
                model,
                config,
                output_dir=trial_dir,
            )
            training_results = trainer.train()

            trial_result = _trial_result_from_training_results(trial_id, params, training_results)
            trial_results.append(trial_result)
            trial_models.append(model)
            trial_configs.append(config)

            _trial_elapsed = time.time() - _trial_start
            val_str = (
                f"{trial_result.best_val_loss:.6f}"
                if trial_result.best_val_loss is not None
                else "—"
            )
            print(
                f"  -> loss={trial_result.final_loss:.6f}"
                f"  val_loss={val_str}"
                f"  epochs={trial_result.epochs_trained}"
                f"  ({_trial_elapsed:.1f}s)"
            )

            # Running leaderboard
            sorted_results = sorted(
                trial_results,
                key=lambda t: (t.best_val_loss if t.best_val_loss is not None else t.final_loss),
            )
            print(f"\n  {'#':<5} {'params':<30} {'loss':<12} {'val_loss':<12} {'epochs'}")
            print(f"  {'-'*5} {'-'*30} {'-'*12} {'-'*12} {'-'*6}")
            for rank, t in enumerate(sorted_results):
                v = f"{t.best_val_loss:.6f}" if t.best_val_loss is not None else "—"
                marker = " *" if t.trial_id == sorted_results[0].trial_id else ""
                print(f"  {rank + 1:<5} {str(t.params):<30} {t.final_loss:<12.6f} {v:<12} {t.epochs_trained}{marker}")

            logger.info(
                "Trial %d complete: final_loss=%.6f  best_val_loss=%s",
                trial_id,
                trial_result.final_loss,
                val_str,
            )
            tuning_evidence.info(
                f"trial={trial_id}  params={params}"
                f"  final_loss={trial_result.final_loss:.6f}  best_val_loss={val_str}"
                f"  epochs_trained={trial_result.epochs_trained}",
                "TRN-009",
            )
            if training_results.evidence_report is not None:
                tuning_evidence.merge(
                    training_results.evidence_report, f"trial={trial_id}"
                )

        # 5. Select best trial
        best_trial, best_model, best_config = _select_best_trial(
            trial_results, trial_models, trial_configs
        )

        if best_trial is not None:
            logger.info(
                "Best trial: %d  params=%s",
                best_trial.trial_id,
                best_trial.params,
            )
            metric_key = "best_val_loss" if best_trial.best_val_loss is not None else "final_loss"
            metric_val = best_trial.best_val_loss if best_trial.best_val_loss is not None else best_trial.final_loss
            tuning_evidence.info(
                f"best_trial={best_trial.trial_id}  params={best_trial.params}"
                f"  {metric_key}={metric_val:.6f}",
                "TRN-011",
            )
            if best_trial.best_val_loss is None:
                tuning_evidence.warn(
                    "No validation partition — best trial selected by training loss;"
                    " results may not generalise",
                    "TRN-011",
                )
        else:
            tuning_evidence.warn("No trials completed — best_trial is None", "TRN-011")

        # 6. Build results and write report
        tuning_results = HyperparameterTuningResults(
            trial_results=trial_results,
            best_trial=best_trial,
            best_model=best_model,
            best_config=best_config,
            artifacts={},
            output_dir=tuning_run_dir,
        )
        report_path = tuning_results.generate_tuning_report()
        tuning_results.artifacts["tuning_report"] = report_path

        evidence_path = tuning_run_dir / "tuning_evidence.json"
        tuning_evidence.save(evidence_path)
        tuning_results.artifacts["evidence"] = evidence_path
        tuning_results.evidence_report = tuning_evidence

        _run_elapsed = time.time() - _run_start
        print("\n" + "=" * _W)
        if best_trial is not None:
            best_metric = best_trial.best_val_loss if best_trial.best_val_loss is not None else best_trial.final_loss
            metric_label = "val_loss" if best_trial.best_val_loss is not None else "loss"
            print(
                f"  TUNING COMPLETE  |  {len(trial_results)} trial(s)  |  {_run_elapsed:.1f}s total"
            )
            print(
                f"  best trial: #{best_trial.trial_id + 1}  {metric_label}={best_metric:.6f}  params={best_trial.params}"
            )
        else:
            print(f"  TUNING COMPLETE  |  0 trials  |  {_run_elapsed:.1f}s total")
        print(f"  report: {report_path}")
        print("=" * _W + "\n")

        logger.info("=== HYPERPARAMETER TUNING COMPLETE ===")
        return tuning_results


def _select_best_trial(
    trial_results: list[TrialResult],
    trial_models: list[nn.Module],
    trial_configs: list[TrainingConfig],
) -> tuple[TrialResult | None, nn.Module | None, TrainingConfig | None]:
    if not trial_results:
        return None, None, None

    trials_with_val = [t for t in trial_results if t.best_val_loss is not None]
    if trials_with_val:
        best = min(trials_with_val, key=lambda t: t.best_val_loss)  # type: ignore[arg-type]
    else:
        best = min(trial_results, key=lambda t: t.final_loss)

    idx = best.trial_id
    return best, trial_models[idx], trial_configs[idx]
