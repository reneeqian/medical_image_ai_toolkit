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
            split_strategy = None,
            optimizer: torch.optim.Optimizer = torch.optim.Adam,
            early_stop: bool = True,
            loss_threshold: float = 0.01,
            plateau_patience: int = 5,
            plateau_min_delta: float = 1e-4,
        ):

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.num_workers = num_workers
        self.task = task
        self.split_strategy = split_strategy
        self.optimizer = optimizer
        self.early_stop = early_stop
        self.loss_threshold = loss_threshold
        self.plateau_patience = plateau_patience
        self.plateau_min_delta = plateau_min_delta
