from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Dict, List, Set, Sequence
import hashlib
from dataclasses import dataclass


class SplitValidationError(RuntimeError):
    """Raised when a SplitStrategy violates required invariants."""


class SplitStrategy(ABC):
    """
    A SplitStrategy defines a deterministic, patient-level assignment
    of patient_ids to logical dataset splits (e.g. train / val / test).

    SplitStrategy instances are treated as first-class, screenable
    experiment inputs.
    """

    @abstractmethod
    def split(
        self,
        patient_ids: Sequence[str],
    ) -> Dict[str, List[str]]:
        """
        Args:
            patient_ids: iterable of unique patient identifiers

        Returns:
            Mapping from split name to list of patient_ids, e.g.:
            {
                "train": [...],
                "val": [...],
            }
        """
        raise NotImplementedError


    @abstractmethod
    def metadata(self) -> Dict[str, object]:
        """
        Returns a JSON-serializable description of the split strategy
        and its configuration, suitable for artifact capture.
        """
        ...

    def validate(self, patient_ids: Iterable[str]) -> None:
        """
        Validates that the split produced by this strategy satisfies
        required invariants.

        Raises:
            SplitValidationError if invariants are violated.
        """
        patient_ids = list(patient_ids)
        split_map = self.split(patient_ids)

        if not split_map:
            raise SplitValidationError("SplitStrategy returned no splits")

        # Ensure all patient_ids are assigned exactly once
        assigned: Set[str] = set()
        for split_name, ids in split_map.items():
            if not ids:
                raise SplitValidationError(f"Split '{split_name}' is empty")

            overlap = assigned.intersection(ids)
            if overlap:
                raise SplitValidationError(
                    f"Patient leakage detected across splits: {overlap}"
                )

            assigned.update(ids)

        missing = set(patient_ids) - assigned
        if missing:
            raise SplitValidationError(
                f"Some patient_ids were not assigned to any split: {missing}"
            )

        # Determinism check (best-effort)
        repeat = self.split(patient_ids)
        if repeat != split_map:
            raise SplitValidationError(
                "SplitStrategy is not deterministic across repeated calls"
            )

@dataclass(frozen=True)
class DeterministicHoldoutSplitStrategy(SplitStrategy):
    """
    Deterministic patient-level holdout split.

    Patients are assigned to splits based on a stable hash of patient_id.
    This guarantees reproducibility across machines and runs.

    Intended use cases:
    - Lightweight end-to-end validation
    - Baseline training
    - Reproducible experiments
    """

    train_fraction: float = 0.8
    val_fraction: float = 0.2
    hash_seed: str = "medical_image_ai_toolkit_v1"

    def __post_init__(self):
        if not (0.0 < self.train_fraction < 1.0):
            raise ValueError("train_fraction must be in (0, 1)")

        if not (0.0 < self.val_fraction < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")

        if abs(self.train_fraction + self.val_fraction - 1.0) > 1e-6:
            raise ValueError("train_fraction + val_fraction must equal 1.0")

    def split(
        self,
        patient_ids: Sequence[str],
    ) -> Dict[str, List[str]]:
        if not patient_ids:
            raise ValueError("patient_ids must be non-empty")

        assignments = {
            "train": [],
            "val": [],
        }

        for pid in patient_ids:
            bucket = self._hash_to_unit_interval(pid)

            if bucket < self.train_fraction:
                assignments["train"].append(pid)
            else:
                assignments["val"].append(pid)

        if not assignments["train"]:
            raise RuntimeError("Train split is empty")

        if not assignments["val"]:
            raise RuntimeError("Validation split is empty")

        return assignments

    def _hash_to_unit_interval(self, patient_id: str) -> float:
        """
        Hash patient_id deterministically into [0, 1).
        """
        h = hashlib.sha256(
            f"{self.hash_seed}:{patient_id}".encode("utf-8")
        ).hexdigest()

        return int(h, 16) / float(2**256)
    
    def metadata(self) -> Dict[str, object]:
        return {
            "type": "DeterministicHoldoutSplitStrategy",
            "train_fraction": self.train_fraction,
            "val_fraction": self.val_fraction,
            "hash_seed": self.hash_seed,
            "hash_function": "sha256",
        }