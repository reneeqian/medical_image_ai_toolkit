import numpy as np

from regulatory_tools.evidence.evidence_report import EvidenceReport
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
    
        Implements data requirements DAT-001 through DAT-001.
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
            requirement_tag="dataset_validation",
        )
        return

    if vol.ndim != 3:
        report.error(
            message="image_volume is not 3D",
            requirement_tag="dataset_validation",
            context=f"shape={vol.shape}",
        )
        return

    report.info(
        message="volume shape OK",
        requirement_tag="dataset_validation",
        context=str(vol.shape),
    )

    if vol.shape[1] <= 0 or vol.shape[2] <= 0:
        report.error(
            message="Invalid spatial dimensions",
            requirement_tag="dataset_validation",
            context=f"shape={vol.shape}",
        )


def _check_spacing(sample: PatientSample, report: EvidenceReport) -> None:
    spacing = sample.spacing

    if spacing is None:
        report.error(
            message="spacing must not be None",
            requirement_tag="dataset_validation",)
        return

    if len(spacing) != 3:
        report.error(
            message="spacing must be (z, y, x)",
            requirement_tag="dataset_validation",
            context=f"got {spacing}",
        )
        return

    if any(s <= 0 for s in spacing):
        report.error(
            message="spacing values must be > 0",
            requirement_tag="dataset_validation",
            context=str(spacing),
        )
    else:
        report.info(
            message="spacing OK",
            requirement_tag="dataset_validation",
            context=str(spacing),
        )

def _check_patient_id(sample: PatientSample, report: EvidenceReport) -> None:
    if not sample.patient_id:
        report.error(
            message="patient_id must be set",
            requirement_tag="dataset_validation",
        )
    else:
        report.info(
            message="patient_id OK",
            requirement_tag="dataset_validation",
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
            requirement_tag="annotation_validation",
        )
        return

    # --- No annotations at all ---
    if ann is None:
        if require_annotations:
            report.error(
                message="Annotations required but none found",
                requirement_tag="annotation_validation",
            )
        else:
            report.warn(
                message="No annotations present",
                requirement_tag="annotation_validation",
            )
        return
    
    # --- allow dense mask annotations ---
    if isinstance(ann, np.ndarray):
        report.info(
            message="Dense annotation mask present (raster annotation)",
            requirement_tag="annotation_validation",
            context=f"shape={ann.shape}",
        )
        return

    # --- Wrong annotation type (common test failure case) ---
    if not hasattr(ann, "vector_rois"):
        report.error(
            message="Annotations object missing vector_rois attribute",
            requirement_tag="annotation_validation",
            context=f"type={type(ann)}",
        )
        return

    if ann.vector_rois is None:
        if require_annotations:
            report.error(
                message="Annotations required but vector_rois is None",
                requirement_tag="annotation_validation",
            )
        else:
            report.info(
                message="No annotations found (vector_rois is None) — patient may not have annotations but this is not required",
                requirement_tag="annotation_validation",
            )
        return

    if not isinstance(ann.vector_rois, dict):
        report.error(
            message="vector_rois must be a dict",
            requirement_tag="annotation_validation",
            context=f"type={type(ann.vector_rois)}",
        )
        return

    report.info(
        message="annotations present",
        requirement_tag="annotation_validation",
        context=f"slices={sorted(ann.vector_rois.keys())}",
    )

    # --- Validate ROIs per slice ---
    depth = vol.shape[0]

    for slice_idx, rois in ann.vector_rois.items():
        if not isinstance(slice_idx, int):
            report.error(
                message="Slice index must be int",
                requirement_tag="annotation_validation",
                context=f"type={type(slice_idx)}",
            )
            continue

        if slice_idx < 0 or slice_idx >= depth:
            report.error(
                message="ROI slice out of bounds",
                requirement_tag="annotation_validation",
                context=f"slice={slice_idx}, depth={depth}",
            )
            continue

        if not isinstance(rois, (list, tuple)):
            report.error(
                message="ROIs for slice must be a list",
                requirement_tag="annotation_validation",
                context=f"type={type(rois)}",
            )
            continue

        for roi in rois:
            if not isinstance(roi, VectorROI):
                report.error(
                    message="ROI must be VectorROI",
                    requirement_tag="annotation_validation",
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
            requirement_tag="annotation_validation",
            context=f"slice={slice_idx}, shape={contour.shape}",
        )
        return

    h, w = volume.shape[1:]

    if (contour[:, 0] < 0).any() or (contour[:, 0] >= w).any():
        report.error(
            message="ROI x-coordinates out of bounds",
            requirement_tag="annotation_validation",
            context=f"slice={slice_idx}, width={w}",
        )

    if (contour[:, 1] < 0).any() or (contour[:, 1] >= h).any():
        report.error(
            message="ROI y-coordinates out of bounds",
            requirement_tag="annotation_validation",
            context=f"slice={slice_idx}, height={h}",
        )