from pathlib import Path

from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import MedicalImageDataSource
from medical_image_ai_toolkit.dataobjects.data_validation.dataset_validator import DatasetValidator, summarize_dataset_validation
from medical_image_ai_toolkit.training.medical_image_trainer import MedicalImageTrainer
from medical_image_ai_toolkit.training.training_config import TrainingConfig
from medical_image_ai_toolkit.results.medical_image_training_results import MedicalImageTrainingResults
from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition

class TrainingPipeline:

    def __init__(self, datasource, model, config, req_provider=None, output_dir=None):
        self.datasource = datasource
        self.model = model
        self.config = config
        self.req_provider = req_provider
        self.output_dir = output_dir

    def run(self):

        print("\n=== PIPELINE START ===")

        # 1. Validate dataset
        print("\nValidating dataset...")
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
        print("\nPartitioning dataset...")
        if not self.datasource.has_partitions():
            print("Creating partitions...")
            self.datasource.create_partitions(self.config.split_strategy)

        # 3. Train
        print("\nTraining model...")
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
        report_path = results.generate_training_report()

        print("\n=== PIPELINE COMPLETE ===")

        return {
            "dataset_validation": ds_validation_report,
            "results": results,
        }
