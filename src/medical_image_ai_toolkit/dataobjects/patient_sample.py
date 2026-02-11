from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np

from medical_image_ai_toolkit.dataobjects.annotation_bundle import AnnotationBundle

@dataclass
class PatientSample:
    """
    Canonical, lossless representation of a single patient study.
    """

    image_volume: np.ndarray              # (Z, Y, X), HU
    spacing: tuple[float, float, float]   # (dz, dy, dx)
    annotations: AnnotationBundle

    patient_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
        
    def __repr__(self) -> str:
        lines = [
            "PatientSample(",
            f"  patient_id: {self.patient_id}",
            f"  image_volume:",
            f"    shape: {self.image_volume.shape}",
            f"    dtype: {self.image_volume.dtype}",
            f"    min/max: {self.image_volume.min():.1f} / {self.image_volume.max():.1f}",
            f"  spacing (z, y, x): {self.spacing}",
            f"  annotations:",
        ]

        if self.annotations.vector_rois:
            slices = sorted(self.annotations.vector_rois.keys())
            lines.append(f"    vector_rois: {len(slices)} slices annotated")
            lines.append(f"    slices: {slices[:5]}{'...' if len(slices) > 5 else ''}")
        else:
            lines.append("    vector_rois: none")

        if self.metadata:
            lines.append(f"  metadata keys: {list(self.metadata.keys())}")
        else:
            lines.append("  metadata: none")

        lines.append(")")

        return "\n".join(lines)
