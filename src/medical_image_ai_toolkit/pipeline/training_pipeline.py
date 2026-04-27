from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from medical_image_ai_toolkit.dataobjects.data_validation.dataset_validator import (
    DatasetValidator,
    summarize_dataset_validation,
)
from medical_image_ai_toolkit.training.medical_image_trainer import MedicalImageTrainer

if TYPE_CHECKING:
    import torch.nn as nn

    from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import (
        MedicalImageDataSource,
    )
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
    ) -> None:
        self.datasource = datasource
        self.model = model
        self.config = config
        self.req_provider = req_provider
        self.output_dir = output_dir

    def run(self) -> dict:
        """
        Execute all four pipeline stages and return results.

        Returns
        -------
        dict
            ``"dataset_validation"`` : EvidenceReport
            ``"results"``            : MedicalImageTrainingResults
        """

        logger.info("=== PIPELINE START ===")

        # 1. Validate dataset
        logger.info("Validating dataset...")
        ds_validator = DatasetValidator(self.datasource, req_provider=self.req_provider)
        ds_validation_report = ds_validator.run()
        ds_validation_report.print_summary()
        summarize_dataset_validation(ds_validation_report)

        if ds_validation_report.has_errors:
            raise RuntimeError(
                "Dataset validation failed; aborting training pipeline.\n"
                f"{ds_validation_report.to_string()}"
            )

        # 2. Partition
        logger.info("Partitioning dataset...")
        if not self.datasource.has_partitions():
            logger.info("Creating partitions...")
            self.datasource.create_partitions(self.config.split_strategy)

        # 3. Train
        logger.info("Training model...")
        trainer = MedicalImageTrainer(
            self.datasource,
            self.model,
            self.config,
            output_dir=self.output_dir,
        )

        trainer.sanity_check()

        results = trainer.train()

        # 4. Export results
        model_path = results.run_dir / "model.pt"

        results.export_model(model_path)
        results.artifacts["model"] = model_path
        partitions_path = self.datasource.save_partitions(results.run_dir)
        results.artifacts["partitions"] = partitions_path
        results.generate_training_report()

        logger.info("=== PIPELINE COMPLETE ===")

        return {
            "dataset_validation": ds_validation_report,
            "results": results,
        }
