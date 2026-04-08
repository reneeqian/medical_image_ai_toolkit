from collections import defaultdict
from pathlib import Path

from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import MedicalImageDataSource
from medical_image_ai_toolkit.dataobjects.patient_sample_contract import enforce_patient_sample_contract
from regulatory_tools.evidence.evidence_report import EvidenceReport


class DatasetValidator:

    def __init__(self, datasource: MedicalImageDataSource, req_provider=None):
        self.datasource = datasource
        self.req_provider = req_provider
        
    def run(self):

        report = EvidenceReport(
            subject="Dataset Validation",
            requirement_provider=self.req_provider
        )
        self.datasource.ingestor.report = report

        patient_ids = self.datasource.get_patient_ids()

        report.info(
            message=f"Total patients: {len(patient_ids)}",
            requirement_tag="dataset_validation"
        )

        for pid in patient_ids:

            try:
                sample = self.datasource.get_patient(pid)

                sample_report = enforce_patient_sample_contract(
                    sample,
                    require_annotations=False,
                )

                report.merge(sample_report, prefix=f"patient={pid}")

            except Exception as e:
                report.error(
                    message="Failed to load patient",
                    requirement_tag="patient_load_failure",
                    context=f"patient={pid} | error={str(e)}"
                )

        
        return report

def summarize_invalid_annotation_slices(report):
    """
    Extract and group invalid annotation slice warnings
    into a compact summary.
    """

    grouped = defaultdict(set)

    for issue in report.issues:
        if issue.level != "WARN":
            continue

        if "Invalid slice index in annotation" not in issue.message:
            continue

        if not issue.context:
            continue

        # Parse context: "file=... | slice=..."
        parts = dict(
            item.strip().split("=")
            for item in issue.context.split("|")
            if "=" in item
        )

        file_path = parts.get("file")
        slice_idx = parts.get("slice")

        if not file_path or slice_idx is None:
            continue

        filename = Path(file_path).name
        grouped[filename].add(int(slice_idx))

    # Print clean summary
    print("\n=== Invalid Annotation Slice Summary ===")

    if not grouped:
        print("No invalid annotation slices found.")
        return

    for fname, slices in sorted(grouped.items()):
        slice_list = sorted(slices)

        # truncate long lists
        if len(slice_list) > 10:
            display = slice_list[:10]
            suffix = f"... (+{len(slice_list) - 10} more)"
        else:
            display = slice_list
            suffix = ""

        print(f"{fname} → {display} {suffix}")

    print("=== End Summary ===\n")