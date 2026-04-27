from __future__ import annotations

from typing import Any

import torch

from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition


class TrainingConfig:
    """
    Configuration container for training and model-testing parameters.

    Parameters
    ----------
    epochs : int
        Maximum number of training epochs.  Default ``10``.
    batch_size : int
        Reserved for future DataLoader integration; not currently used by
        the training loop, which iterates samples individually.  Default ``2``.
    learning_rate : float
        Optimizer learning rate.  Default ``1e-4``.
    device : str
        PyTorch device string (``"cpu"`` or ``"cuda"``).  Default ``"cpu"``.
    num_workers : int
        Reserved for future DataLoader integration; not currently used.
        Default ``0``.
    task : TrainingTaskDefinition or None
        **Required** for both training and model testing.  Provides
        ``generate_training_samples`` and ``compute_loss``.
    split_strategy : any or None
        Strategy object with a ``split(patient_ids)`` method that returns
        ``(train_ids, val_ids, test_ids)``.  Required by
        ``TrainingPipeline``; unused by ``ModelTestingPipeline``.
    optimizer : type[torch.optim.Optimizer]
        Optimizer *class* (not instance) to use for training.  Called as
        ``optimizer(model.parameters(), lr=learning_rate)``.
        Default ``torch.optim.Adam``.
    early_stop : bool
        Enable early stopping.  When ``True``, both ``loss_threshold`` and
        ``plateau_patience`` are evaluated.  Default ``True``.
    loss_threshold : float
        Stop training when epoch loss falls at or below this value.
        Default ``0.01``.
    plateau_patience : int
        Stop training after this many epochs without improvement greater
        than ``plateau_min_delta``.  Default ``5``.
    plateau_min_delta : float
        Minimum improvement in epoch loss required to reset the plateau
        counter.  Default ``1e-4``.
    """

    def __init__(
            self,
            epochs: int = 10,
            batch_size: int = 2,
            learning_rate: float = 1e-4,
            device: str = "cpu",
            num_workers: int = 0,
            task: TrainingTaskDefinition | None = None,
            split_strategy: Any | None = None,
            optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
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
