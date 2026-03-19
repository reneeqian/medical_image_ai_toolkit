import torch
from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition

class TrainingConfig:
    """
    Configuration container for training parameters.
    """

    def __init__(
            self,
            epochs: int = 10,
            batch_size: int = 2,
            learning_rate: float = 1e-4,
            device: str = "cpu",
            num_workers: int = 0,
            task: TrainingTaskDefinition = None,
            optimizer: torch.optim.Optimizer = torch.optim.Adam,
        ):

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.num_workers = num_workers
        self.task = task
        self.optimizer = optimizer