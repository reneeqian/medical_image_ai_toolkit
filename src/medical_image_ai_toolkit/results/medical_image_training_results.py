from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn

class MedicalImageTrainingResults:
    """
    Minimal container for outputs of a training run.
    """

    def __init__(self, model, config, datasource, run_dir):

        self.model = model
        self.config = config
        self.datasource = datasource

        self.run_dir = Path(run_dir)

        self.history = []
        self.metrics = {}
        self.artifacts = {}
        
        # initialize lifecycle state
        self.training_start_time = None
        self.training_end_time = None
        
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

        if self.training_end_time is not None and self.training_start_time is not None:

            duration = self.training_end_time - self.training_start_time
            print(f"Training time: {duration}")

        else:

            print("Training still running")

        if self.metrics:

            print("\nMetrics")

            for k, v in self.metrics.items():
                print(f"{k}: {v}")

        print("==============================\n")
    
    def generate_training_report(self):

        report_path = self.run_dir / "training_report.json"

        import json

        report = {
            "metrics": self.metrics,
            "config": vars(self.config),
            "history": self.history
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report_path

    # --------------------------------------------------
    # Model export
    # --------------------------------------------------

    def export_model(self, path):

        try:
            torch.save(self.model.state_dict(), path)
            print(f"Model exported to: {path}")

        except Exception as e:
            print("Failed to export model")
            print(e)
            
    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    
    def run_inference(self, data):

        self.model.eval()

        with torch.no_grad():

            preds = []

            for sample in data:
                x = torch.tensor(sample).float().unsqueeze(0)
                pred = self.model(x)
                preds.append(pred)

        return preds