# Data Requirements (MIT)
# MIT-DR-01 image_volume shall be a NumPy array and 3D
# MIT-DR-02 image_volume spatial dimensions shall be > 0
# MIT-DR-03 spacing shall be (z,y,x) with all values > 0
# MIT-DR-04 patient_id shall be defined and non-empty
# MIT-DR-05 annotations shall be present if required by workflow
# MIT-DR-06 annotation slice indices shall be valid and within bounds
# MIT-DR-07 vector ROI contours shall be (N,2) and within image bounds
# MIT-DR-08 annotation representation shall be supported (vector or dense)
# MIT-DR-09 all PatientSample contracts shall be enforced at a single boundary

import numpy as np

from medical_image_ai_toolkit.evidence.evidence_report import EvidenceReport
from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
from medical_image_ai_toolkit.dataobjects.annotation_bundle import VectorROI

def enforce_patient_sample_contract(
    sample: PatientSample,
    *,
    require_annotations: bool = False,
    report: EvidenceReport | None = None,
) -> EvidenceReport:
    """
    Check structural and semantic correctness of a PatientSample.
    
        Implements data requirements MIT-DR-01 through MIT-DR-09.
    """
    if report is None:
        report = EvidenceReport(subject=f"PatientSample:{sample.patient_id}")

    _check_volume(sample, report)
    _check_spacing(sample, report)
    _check_patient_id(sample, report)
    _check_annotations(sample, report, require_annotations=require_annotations)
    
    return report

def _check_volume(sample: PatientSample, report: EvidenceReport) -> None:
    vol = sample.image_volume

    if not isinstance(vol, np.ndarray):
        report.error(
            message="image_volume is not a numpy array",
            requirement_id="MIT-DR-01",
        )
        return

    if vol.ndim != 3:
        report.error(
            message="image_volume is not 3D",
            requirement_id="MIT-DR-01",
            context=f"shape={vol.shape}",
        )
    else:
        report.info(
            message="volume shape OK",
            requirement_id="MIT-DR-01",
            context=str(vol.shape),
        )

    if vol.shape[1] <= 0 or vol.shape[2] <= 0:
        report.error(
            message="Invalid spatial dimensions",
            requirement_id="MIT-DR-02",
            context=f"shape={vol.shape}",
        )


def _check_spacing(sample: PatientSample, report: EvidenceReport) -> None:
    spacing = sample.spacing

    if spacing is None:
        report.error(
            message="spacing must not be None",
            requirement_id="MIT-DR-03",)
        return

    if len(spacing) != 3:
        report.error(
            message="spacing must be (z, y, x)",
            requirement_id="MIT-DR-03",
            context=f"got {spacing}",
        )
        return

    if any(s <= 0 for s in spacing):
        report.error(
            message="spacing values must be > 0",
            requirement_id="MIT-DR-03",
            context=str(spacing),
        )
    else:
        report.info(
            message="spacing OK",
            requirement_id="MIT-DR-03",
            context=str(spacing),
        )

def _check_patient_id(sample: PatientSample, report: EvidenceReport) -> None:
    if not sample.patient_id:
        report.error(
            message="patient_id must be set",
            requirement_id="MIT-DR-04",
        )
    else:
        report.info(
            message="patient_id OK",
            requirement_id="MIT-DR-04",
            context=sample.patient_id,
        )
        
def _check_annotations(
    sample: PatientSample,
    report: EvidenceReport,
    *,
    require_annotations: bool,
) -> None:
    ann = sample.annotations
    vol = sample.image_volume

    # --- Image volume sanity (needed for downstream checks) ---
    if vol is None or not hasattr(vol, "shape"):
        report.error(
            message="Image volume missing or invalid",
            requirement_id="MIT-DR-09",
        )
        return

    # --- No annotations at all ---
    if ann is None:
        if require_annotations:
            report.error(
                message="Annotations required but none found",
                requirement_id="MIT-DR-05",
            )
        else:
            report.warn(
                message="No annotations present",
                requirement_id="MIT-DR-05",
            )
        return
    
    # --- allow dense mask annotations ---
    if isinstance(ann, np.ndarray):
        report.info(
            message="Dense annotation mask present (raster annotation)",
            requirement_id="MIT-DR-08",
            context=f"shape={ann.shape}",
        )
        return

    # --- Wrong annotation type (common test failure case) ---
    if not hasattr(ann, "vector_rois"):
        report.error(
            message="Annotations object missing vector_rois attribute",
            requirement_id="MIT-DR-05",
            context=f"type={type(ann)}",
        )
        return

    if ann.vector_rois is None:
        if require_annotations:
            report.error(
                message="Annotations required but vector_rois is None",
                requirement_id="MIT-DR-05",
            )
        else:
            report.warn(
                message="vector_rois is None",
                requirement_id="MIT-DR-05",
            )
        return

    if not isinstance(ann.vector_rois, dict):
        report.error(
            message="vector_rois must be a dict",
            requirement_id="MIT-DR-05",
            context=f"type={type(ann.vector_rois)}",
        )
        return

    report.info(
        message="annotations present",
        requirement_id="MIT-DR-05",
        context=f"slices={sorted(ann.vector_rois.keys())}",
    )

    # --- Validate ROIs per slice ---
    depth = vol.shape[0]

    for slice_idx, rois in ann.vector_rois.items():
        if not isinstance(slice_idx, int):
            report.error(
                message="Slice index must be int",
                requirement_id="MIT-DR-06",
                context=f"type={type(slice_idx)}",
            )
            continue

        if slice_idx < 0 or slice_idx >= depth:
            report.error(
                message="ROI slice out of bounds",
                requirement_id="MIT-DR-06",
                context=f"slice={slice_idx}, depth={depth}",
            )
            continue

        if not isinstance(rois, (list, tuple)):
            report.error(
                message="ROIs for slice must be a list",
                requirement_id="MIT-DR-06",
                context=f"type={type(rois)}",
            )
            continue

        for roi in rois:
            if not isinstance(roi, VectorROI):
                report.error(
                    message="ROI must be VectorROI",
                    requirement_id="MIT-DR-07",
                    context=f"type={type(roi)}",
                )
                continue

            _check_vector_roi(
                roi=roi,
                volume=vol,
                report=report,
                slice_idx=slice_idx,
            )


def _check_vector_roi(
    roi: VectorROI,
    volume: np.ndarray,
    report: EvidenceReport,
    slice_idx: int,
) -> None:
    contour = roi.contour_px

    if contour.ndim != 2 or contour.shape[1] != 2:
        report.error(
            message="ROI contour must be (N, 2)",
            requirement_id="MIT-DR-07",
            context=f"slice={slice_idx}, shape={contour.shape}",
        )
        return

    h, w = volume.shape[1:]

    if (contour[:, 0] < 0).any() or (contour[:, 0] >= w).any():
        report.error(
            message="ROI x-coordinates out of bounds",
            requirement_id="MIT-DR-07",
            context=f"slice={slice_idx}, width={w}",
        )

    if (contour[:, 1] < 0).any() or (contour[:, 1] >= h).any():
        report.error(
            message="ROI y-coordinates out of bounds",
            requirement_id="MIT-DR-07",
            context=f"slice={slice_idx}, height={h}",
        )