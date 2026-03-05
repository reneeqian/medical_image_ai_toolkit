import pytest
import yaml
from pathlib import Path

from regulatory_tools.evidence.evidence_report import EvidenceReport


@pytest.mark.requirement("DOC-001")
@pytest.mark.requirement("DOC-002")
def test_project_documentation_structure(
    request,
    evidence_output_dir,
):
    """
    Verifies:
      - docs/requirements.yaml exists and is structurally valid
      - requirements list is non-empty
      - required requirement categories exist (SYS, DAT, VER, DOC)
      - README.md exists and contains at least one heading
    """

    project_root = Path(__file__).resolve().parents[1]
    requirements_path = project_root / "docs" / "requirements.yaml"
    readme_path = project_root / "README.md"

    report = EvidenceReport(
        subject="Project Documentation → Structural Integrity",
        test_id=request.node.nodeid,
    )

    # ============================================================
    # requirements.yaml Checks
    # ============================================================

    report.info(
        message="Checking for requirements.yaml existence",
        requirement_id="DOC-001",
        context=str(requirements_path),
    )

    if not requirements_path.exists():
        report.error(
            message="requirements.yaml not found in docs directory",
            requirement_id="DOC-001",
        )
    else:
        try:
            with open(requirements_path, "r") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            report.error(
                message=f"requirements.yaml failed to parse: {e}",
                requirement_id="DOC-001",
            )
            data = None

        if data:
            # ---- Metadata project title ----
            project_name = (
                data.get("metadata", {})
                .get("project")
            )

            if not project_name:
                report.error(
                    message="metadata.project field missing or empty",
                    requirement_id="DOC-001",
                )

            # ---- Requirements list existence ----
            requirements = data.get("requirements")

            if not requirements or not isinstance(requirements, list):
                report.error(
                    message="requirements list missing or empty",
                    requirement_id="DOC-001",
                )
            else:
                prefixes_present = set()

                for req in requirements:
                    req_id = req.get("id", "")
                    if "-" in req_id:
                        prefixes_present.add(req_id.split("-")[0])

                required_prefixes = {"SYS", "DAT", "VER", "DOC"}

                missing_prefixes = required_prefixes - prefixes_present

                if missing_prefixes:
                    report.error(
                        message=f"Missing required requirement categories: {sorted(missing_prefixes)}",
                        requirement_id="DOC-001",
                    )

    # ============================================================
    # README.md Checks
    # ============================================================

    report.info(
        message="Checking for README.md existence",
        requirement_id="DOC-002",
        context=str(readme_path),
    )

    if not readme_path.exists():
        report.error(
            message="README.md not found in project root",
            requirement_id="DOC-002",
        )
    else:
        content = readme_path.read_text(encoding="utf-8")

        has_heading = any(
            line.strip().startswith("#")
            for line in content.splitlines()
        )

        if not has_heading:
            report.error(
                message="README.md does not contain any Markdown headings",
                requirement_id="DOC-002",
            )

    # ============================================================
    # Save Evidence
    # ============================================================

    report.auto_save(
        "project_documentation_structure",
        evidence_output_dir,
    )

    assert not report.has_errors, report.summary()