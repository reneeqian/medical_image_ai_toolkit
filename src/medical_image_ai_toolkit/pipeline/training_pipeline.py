from pathlib import Path

from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import MedicalImageDataSource
from medical_image_ai_toolkit.dataobjects.data_validation.dataset_validator import DatasetValidator, summarize_invalid_annotation_slices
from medical_image_ai_toolkit.training.medical_image_trainer import MedicalImageTrainer
from medical_image_ai_toolkit.training.training_config import TrainingConfig
from medical_image_ai_toolkit.results.medical_image_training_results import MedicalImageTrainingResults
from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition

class TrainingPipeline:

    def __init__(self, datasource, model, config, req_provider=None):
        self.datasource = datasource
        self.model = model
        self.config = config
        self.req_provider = req_provider

    def run(self):

        print("\n=== PIPELINE START ===")

        # 1. Validate dataset
        print("\nValidating dataset...")
        validator = DatasetValidator(self.datasource, req_provider=self.req_provider)
        validation_report = validator.run()
        validation_report.print_summary()
        summarize_invalid_annotation_slices(validation_report)

        if validation_report.has_errors:
            print("Warning: dataset validation contains errors")

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
            self.config
        )
        
        trainer.sanity_check()

        results = trainer.train()

        print("\n=== PIPELINE COMPLETE ===")

        return {
            "validation": validation_report,
            "results": results,
        }