from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class VectorROI:
    slice_index: int
    contour_px: np.ndarray          # shape (N, 2)
    label: str
    metadata: Optional[dict] = None


VectorROISet = Dict[int, List[VectorROI]]


@dataclass
class AnnotationBundle:
    vector_rois: Optional[VectorROISet]
    segmentation_masks: Optional[list] = None
    label_map: Optional[dict] = None
