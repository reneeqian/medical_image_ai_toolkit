from datetime import datetime


class MedicalImageTrainingResults:
    """
    Minimal container for outputs of a training run.
    """

    def __init__(self, model, config, datasource):

        self.model = model
        self.config = config
        self.datasource = datasource

        self.training_start_time = datetime.now()
        self.training_end_time = None

        self.metrics = {}

    # --------------------------------------------------
    # Training lifecycle
    # --------------------------------------------------

    def mark_training_complete(self):

        self.training_end_time = datetime.now()

    # --------------------------------------------------
    # Reporting
    # --------------------------------------------------

    def summary(self):

        print("\n==============================")
        print("Training Results Summary")
        print("==============================")

        print(f"Model: {self.model.__class__.__name__}")

        if self.training_end_time:

            duration = self.training_end_time - self.training_start_time

            print(f"Training time: {duration}")

        else:

            print("Training still running")

        if self.metrics:

            print("\nMetrics")

            for k, v in self.metrics.items():
                print(f"{k}: {v}")

        print("==============================\n")

    # --------------------------------------------------
    # Model export
    # --------------------------------------------------

    def export_model(self, path):

        try:
            import torch

            torch.save(self.model.state_dict(), path)

            print(f"Model exported to: {path}")

        except Exception as e:

            print("Failed to export model")
            print(e)