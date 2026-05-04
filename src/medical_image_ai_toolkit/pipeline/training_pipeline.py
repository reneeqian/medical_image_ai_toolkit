from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from regulatory_tools.evidence.evidence_report import EvidenceReport
from medical_image_ai_toolkit.dataobjects.data_validation.dataset_validator import DatasetValidator, summarize_dataset_validation
from medical_image_ai_toolkit.training.medical_image_trainer import MedicalImageTrainer

if TYPE_CHECKING:
    import torch.nn as nn

    from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import MedicalImageDataSource
    from medical_image_ai_toolkit.training.training_config import TrainingConfig

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    End-to-end training pipeline.

    Stages
    ------
    1. **Dataset validation** — every patient sample is checked against the
       ``PatientSample`` contract; errors abort the run.
    2. **Partitioning** — the datasource is split into train / val / test
       partitions using ``config.split_strategy`` (skipped if partitions
       already exist).
    3. **Training** — ``MedicalImageTrainer`` runs the epoch loop with
       optional early stopping and validation.
    4. **Export** — model weights (``model.pt``), partitions
       (``partitions.json``), and a training report are written to the run
       directory.

    Returns a dict with keys ``"dataset_validation"`` (EvidenceReport) and
    ``"results"`` (MedicalImageTrainingResults).
    """

    def __init__(
        self,
        datasource: MedicalImageDataSource,
        model: nn.Module,
        config: TrainingConfig,
        req_provider=None,
        output_dir=None,
        resume_from=None,
    ) -> None:
        self.datasource = datasource
        self.model = model
        self.config = config
        self.req_provider = req_provider
        self.output_dir = output_dir
        self.resume_from = resume_from

    def run(self) -> dict:
        """
        Execute all four pipeline stages and return results.

        Returns
        -------
        dict
            ``"dataset_validation"`` : EvidenceReport
            ``"results"``            : MedicalImageTrainingResults
        """

        _W = 68
        _pipeline_start = time.time()
        model_name = self.model.__class__.__name__

        print("\n" + "=" * _W)
        print(f"  TRAINING PIPELINE  |  {model_name}")
        print("=" * _W)
        print(f"  epochs      : {self.config.epochs}")
        print(f"  lr          : {self.config.learning_rate}")
        print(f"  optimizer   : {getattr(self.config.optimizer, '__name__', self.config.optimizer)}")
        print(f"  device      : {self.config.device}")
        print(f"  early_stop  : {self.config.early_stop}")
        if self.config.early_stop:
            print(f"    loss_threshold  = {self.config.loss_threshold}")
            print(f"    plateau_patience= {self.config.plateau_patience}")
        print(f"  checkpoint_every: {self.config.checkpoint_every or 'disabled'}")
        if self.resume_from is not None:
            print(f"  resume_from : {self.resume_from}")
        print("-" * _W)

        logger.info("=== PIPELINE START ===")

        # 1. Validate dataset
        print(f"\n[1/4] Dataset validation")
        _t = time.time()
        logger.info("Validating dataset...")
        ds_validator = DatasetValidator(self.datasource, req_provider=self.req_provider)
        ds_validation_report = ds_validator.run()
        ds_validation_report.print_summary()
        summarize_dataset_validation(ds_validation_report)
        print(f"      done ({time.time() - _t:.1f}s)")

        if ds_validation_report.has_errors:
            raise RuntimeError(
                "Dataset validation failed; aborting training pipeline.\n"
                f"{ds_validation_report.to_string()}"
            )

        # 2. Partition
        print(f"\n[2/4] Partitioning")
        _t = time.time()
        logger.info("Partitioning dataset...")
        if not self.datasource.has_partitions():
            logger.info("Creating partitions...")
            self.datasource.create_partitions(self.config.split_strategy)
        train_n = len(self.datasource.get_train_ids())
        val_n   = len(self.datasource.get_val_ids())
        test_n  = len(self.datasource.get_test_ids())
        print(f"      train={train_n}  val={val_n}  test={test_n}  ({time.time() - _t:.1f}s)")

        # 3. Train
        print(f"\n[3/4] Training")
        logger.info("Training model...")
        trainer = MedicalImageTrainer(
            self.datasource,
            self.model,
            self.config,
            output_dir=self.output_dir,
            resume_from=self.resume_from,
        )

        trainer.sanity_check()

        results = trainer.train()

        final_loss = results.metrics.get("final_loss", float("nan"))
        epochs_trained = results.metrics.get("epochs_trained", "?")
        stop_reason = results.metrics.get("early_stop_reason")
        print(f"\n      final_loss={final_loss:.6f}  epochs_trained={epochs_trained}")
        if stop_reason:
            print(f"      early stop: {stop_reason}")

        # 4. Export results
        print(f"\n[4/4] Exporting artifacts")
        _t = time.time()
        model_path = results.run_dir / "model.pt"

        results.export_model(model_path)
        results.artifacts["model"] = model_path
        manifest_path = _write_model_manifest(results, model_path)
        results.artifacts["manifest"] = manifest_path
        partitions_path = self.datasource.save_partitions(results.run_dir)
        results.artifacts["partitions"] = partitions_path
        results.generate_training_report()
        print(f"      model      : {model_path}")
        print(f"      partitions : {partitions_path}")
        print(f"      run dir    : {results.run_dir}")
        print(f"      ({time.time() - _t:.1f}s)")

        # 5. Pipeline-level evidence: merge dataset validation + training evidence
        pipeline_evidence = EvidenceReport(
            subject=f"Training pipeline: {model_name}"
        )
        pipeline_evidence.merge(ds_validation_report, "dataset_validation")
        if results.evidence_report is not None:
            pipeline_evidence.merge(results.evidence_report, "training")
        pipeline_evidence.info(
            f"artifacts: model={model_path.name}  partitions={partitions_path.name}"
            f"  report=training_report.json  evidence=training_evidence.json",
            "TRN-004",
        )
        pipeline_evidence_path = results.run_dir / "pipeline_evidence.json"
        pipeline_evidence.save(pipeline_evidence_path)
        results.artifacts["pipeline_evidence"] = pipeline_evidence_path

        _elapsed = time.time() - _pipeline_start
        print("\n" + "=" * _W)
        print(f"  PIPELINE COMPLETE  |  {model_name}  |  {_elapsed:.1f}s total")
        print(f"  final_loss={final_loss:.6f}  epochs={epochs_trained}/{self.config.epochs}")
        print("=" * _W + "\n")

        logger.info("=== PIPELINE COMPLETE ===")

        return {
            "dataset_validation": ds_validation_report,
            "results": results,
            "evidence": pipeline_evidence,
        }


def _write_model_manifest(results, model_path: Path) -> Path:
    manifest = {
        "model_class": type(results.model).__name__,
        "param_count": sum(p.numel() for p in results.model.parameters()),
        "run_id": results.run_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_config": {
            "epochs": results.config.epochs,
            "learning_rate": results.config.learning_rate,
            "optimizer": getattr(results.config.optimizer, "__name__", str(results.config.optimizer)),
            "batch_size": results.config.batch_size,
            "task": type(results.config.task).__name__,
        },
        "metrics": results.metrics,
    }
    manifest_path = model_path.parent / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest_path
