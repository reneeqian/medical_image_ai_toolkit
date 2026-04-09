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


def summarize_dataset_validation(report):
    print("\n=== Dataset Validation Report ===\n")

    grouped = defaultdict(list)
    severity_counts = {"ERROR": 0, "WARN": 0}

    # Filter + group
    for issue in report.issues:
        if issue.level not in ("ERROR", "WARN"):
            continue

        tag = issue.requirement_tag or "unclassified"
        grouped[tag].append(issue)

        if issue.level in severity_counts:
            severity_counts[issue.level] += 1

    if not grouped:
        print("No WARN or ERROR issues found.\n")
        return

    # Print grouped issues
    for tag, issues in grouped.items():
        print(f"--- {tag.upper()} ({len(issues)} issues) ---")

        for issue in issues[:10]:  # truncate for readability
            ctx = f" | {issue.context}" if issue.context else ""
            print(f"[{issue.level}] {issue.message}{ctx}")

        if len(issues) > 10:
            print(f"... +{len(issues) - 10} more")

        print()

    # Severity summary (very useful for CI / quick scan)
    print("=== Severity Summary ===")
    print(f"ERROR: {severity_counts['ERROR']}")
    print(f"WARN : {severity_counts['WARN']}")
    print("\n=== End Dataset Validation Report ===\n")
    
    
def summarize_invalid_annotation_slices(report):
    """
    Extract and group invalid annotation slice warnings into a compact summary,
    including which patients failed to load entirely.

    Context format emitted by ingestor:
        "file=... | raw_index=... | computed_index=... | num_slices=..."
    """

    # --- Out-of-bounds annotation slices ---
    grouped = defaultdict(lambda: {"raw_indices": set(), "num_slices": None})

    for issue in report.issues:
        if issue.level != "WARN":
            continue
        if "Invalid slice index in annotation" not in issue.message:
            continue
        if not issue.context:
            continue

        parts = {}
        for item in issue.context.split("|"):
            item = item.strip()
            if "=" in item:
                k, v = item.split("=", 1)
                parts[k.strip()] = v.strip()

        file_path = parts.get("file")
        raw_index = parts.get("raw_index")
        num_slices = parts.get("num_slices")

        if not file_path or raw_index is None:
            continue

        filename = Path(file_path).name
        grouped[filename]["raw_indices"].add(int(raw_index))
        if num_slices is not None:
            grouped[filename]["num_slices"] = int(num_slices)

    # --- Failed patient loads ---
    failed_patients = []
    for issue in report.issues:
        if issue.level != "ERROR":
            continue
        if "Failed to load patient" not in issue.message:
            continue
        if issue.context:
            parts = {}
            for item in issue.context.split("|"):
                item = item.strip()
                if "=" in item:
                    k, v = item.split("=", 1)
                    parts[k.strip()] = v.strip()
            pid = parts.get("patient", "unknown")
            err = parts.get("error", "unknown error")
            failed_patients.append((pid, err))

    # --- Print summaries ---
    print("\n=== Invalid Annotation Slice Summary ===")

    if not grouped:
        print("No out-of-bounds annotation slices found.")
    else:
        print(f"{'File':<20} {'Vol Slices':>10}  Out-of-bounds raw ImageIndex values")
        print("-" * 70)
        for fname, data in sorted(grouped.items()):
            raw_list = sorted(data["raw_indices"])
            ns_str = str(data["num_slices"]) if data["num_slices"] is not None else "?"
            if len(raw_list) > 10:
                display = str(raw_list[:10])[:-1] + f", ... +{len(raw_list)-10} more]"
            else:
                display = str(raw_list)
            print(f"{fname:<20} {ns_str:>10}  {display}")

    print()

    if failed_patients:
        print(f"=== Failed Patient Loads ({len(failed_patients)}) ===")
        for pid, err in failed_patients:
            short_err = (err[:120] + "...") if len(err) > 120 else err
            print(f"  patient={pid}: {short_err}")
    else:
        print("No patient load failures.")

    print("\n=== End Summary ===\n")