from abc import ABC, abstractmethod
from typing import Dict, Any
import torch


class TaskValidationError(RuntimeError):
    """Raised when a TaskDefinition violates required invariants."""


class TaskDefinition(ABC):
    """
    A TaskDefinition specifies the learning objective applied to
    PatientSample-derived tensors.

    TaskDefinition instances are treated as screenable experiment inputs.
    """

    @abstractmethod
    def name(self) -> str:
        """Human-readable task identifier."""
        ...

    @abstractmethod
    def build_loss(self) -> torch.nn.Module:
        """Returns the loss module used for training."""
        ...

    @abstractmethod
    def compute_metrics(
        self,
        model_outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Computes task-appropriate metrics."""
        ...

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        Returns a JSON-serializable description of the task configuration,
        suitable for artifact capture.
        """
        ...

    def validate(self) -> None:
        """
        Validates internal task consistency.

        This is intentionally lightweight and structural; it does not
        validate clinical correctness or dataset semantics.
        """
        loss = self.build_loss()
        if not isinstance(loss, torch.nn.Module):
            raise TaskValidationError(
                "build_loss() must return a torch.nn.Module"
            )

        if not self.name():
            raise TaskValidationError("TaskDefinition.name() must be non-empty")

        meta = self.metadata()
        if not isinstance(meta, dict):
            raise TaskValidationError("TaskDefinition.metadata() must return a dict")
