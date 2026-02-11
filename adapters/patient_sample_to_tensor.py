from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from src.medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
from src.medical_image_ai_toolkit.contracts.patient_sample_contract import enforce_patient_sample_contract


class PatientSampleTensorAdapter:
    """
    Converts a validated PatientSample into PyTorch tensors.

    This class is intentionally stateless and deterministic.
    """

    def __init__(
        self,
        *,
        normalize: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        require_annotations: bool = False,
        default_target: Optional[float] = None,  # 👈 ADD
    ):
        self.normalize = normalize
        self.dtype = dtype
        self.device = device
        self.require_annotations = require_annotations
        self.default_target = default_target

    def __call__(self, sample: PatientSample) -> Dict[str, object]:
        """
        Convert a PatientSample into model-ready tensors.

        Returns
        -------
        dict with keys:
            image: torch.Tensor (1, Z, Y, X)
            target: torch.Tensor or None
            metadata: dict
        """
        # --- Validate ---
        report = enforce_patient_sample_contract(
            sample,
            require_annotations=self.require_annotations,
        )

        if report.has_errors:
            raise ValueError(
                "PatientSample failed validation:\n"
                + report.to_string()
            )

        # --- Image tensor ---
        image = self._image_to_tensor(sample.image_volume)

        # --- Target tensor (task-dependent; placeholder for now) ---
        target = self._build_target(sample)

        # --- Metadata ---
        metadata = {
            "patient_id": sample.patient_id,
            "spacing": sample.spacing,
            **(sample.metadata or {}),
        }

        return {
            "image": image,
            "target": target,
            "metadata": metadata,
        }

    # -------------------------
    # Internal helpers
    # -------------------------

    def _image_to_tensor(self, volume: np.ndarray) -> torch.Tensor:
        """
        Convert image volume to torch tensor.

        Input:  (Z, Y, X)
        Output: (1, Z, Y, X)
        """
        tensor = torch.from_numpy(volume).to(self.dtype)

        if self.normalize:
            tensor = self._normalize_ct(tensor)

        # Add channel dimension
        tensor = tensor.unsqueeze(0)

        if self.device is not None:
            tensor = tensor.to(self.device)

        return tensor

    def _normalize_ct(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Simple CT normalization.

        This is intentionally conservative and explainable.
        """
        # Typical CT window for cardiac structures
        min_hu = -1000.0
        max_hu = 1000.0

        tensor = torch.clamp(tensor, min_hu, max_hu)
        tensor = (tensor - min_hu) / (max_hu - min_hu)

        return tensor

    def _build_target(
        self, sample: PatientSample
    ) -> Optional[torch.Tensor]:
        """
        Placeholder target builder.

        For now:
        - Dense annotations → positive label
        - Vector annotations → positive label if any ROIs exist
        - No annotations → None or error if required
        """
        ann = sample.annotations

        # --- No annotations ---
        if ann is None:
            if self.default_target is not None:
                return torch.tensor(
                    [self.default_target],
                    dtype=torch.float32,
                )
            return None
        
        # --- Dense mask annotations ---
        if isinstance(ann, np.ndarray):
            # Any non-zero pixel = positive label
            return torch.tensor([ann.any()], dtype=torch.float32)

        # --- Vector annotations ---
        if hasattr(ann, "vector_rois"):
            if not ann.vector_rois:
                return None
            return torch.tensor([True], dtype=torch.float32)

        # --- Unsupported annotation type (should never happen if contract enforced) ---
        raise TypeError(f"Unsupported annotation type: {type(ann)}")
