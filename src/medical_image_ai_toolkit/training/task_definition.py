from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample


class TrainingTaskDefinition(ABC):

    @abstractmethod
    def generate_training_samples(
        self, patient_sample: PatientSample
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        """
        Yields training samples for a patient.

        Each sample is a dict:
            {
                "input": Tensor,
                "target": Tensor or dict
            }
        """
        pass

    @abstractmethod
    def compute_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Task-specific loss computation.
        """
        pass

    def postprocess_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        return prediction
