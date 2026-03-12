import torch

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
            loss_fcn: torch.nn.Module = torch.nn.MSELoss(),
            optimizer: torch.optim.Optimizer = torch.optim.Adam,
        ):

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.num_workers = num_workers
        self.loss_fcn = loss_fcn
        self.optimizer = optimizer