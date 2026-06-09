from pathlib import Path

import pytest
import yaml
from regulatory_tools.evidence.evidence_report import EvidenceReport
from regulatory_tools.quality.soup_checker import check_soup_inventory


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
        requirement_tag="DOC-002",
        context=str(requirements_path),
    )

    if not requirements_path.exists():
        report.error(
            message="requirements.yaml not found in docs directory",
            requirement_tag="DOC-002",
        )
    else:
        try:
            with open(requirements_path, "r") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            report.error(
                message=f"requirements.yaml failed to parse: {e}",
                requirement_tag="DOC-002",
            )
            data = None

        if data:
            # ---- Metadata project title ----
            project_name = data.get("metadata", {}).get("project")

            if not project_name:
                report.error(
                    message="metadata.project field missing or empty",
                    requirement_tag="DOC-002",
                )

            # ---- Requirements list existence ----
            requirements = data.get("requirements")

            if not requirements or not isinstance(requirements, list):
                report.error(
                    message="requirements list missing or empty",
                    requirement_tag="DOC-002",
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
                        requirement_tag="DOC-002",
                    )
                else:
                    report.info(
                        message=f"requirements.yaml valid — {len(requirements)} requirements covering {sorted(required_prefixes)}",
                        requirement_tag="DOC-002",
                    )

    # ============================================================
    # README.md Checks
    # ============================================================

    report.info(
        message="Checking for README.md existence",
        requirement_tag="DOC-001",
        context=str(readme_path),
    )

    if not readme_path.exists():
        report.error(
            message="README.md not found in project root",
            requirement_tag="DOC-001",
        )
    else:
        content = readme_path.read_text(encoding="utf-8")

        has_heading = any(line.strip().startswith("#") for line in content.splitlines())

        if not has_heading:
            report.error(
                message="README.md does not contain any Markdown headings",
                requirement_tag="DOC-001",
            )
        else:
            report.info(
                message="README.md exists and contains at least one Markdown heading",
                requirement_tag="DOC-001",
            )

    # ============================================================
    # Save Evidence
    # ============================================================

    report.auto_save(
        "project_documentation_structure",
        evidence_output_dir,
    )

    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DOC-001")
def test_soup_inventory_complete(
    request,
    evidence_output_dir,
):
    """Verify docs/soup.yaml exists and covers all declared pyproject.toml dependencies."""
    project_root = Path(__file__).resolve().parents[1]

    report = EvidenceReport(
        subject="SOUP Inventory → All Dependencies Declared",
        test_id=request.node.nodeid,
    )

    result = check_soup_inventory(project_root)

    if not result["found"]:
        report.error(
            "docs/soup.yaml not found — all third-party dependencies must be inventoried",
            "DOC-001",
            context=str(project_root / "docs" / "soup.yaml"),
        )
    else:
        report.info(
            f"soup.yaml found: {result['soup_path']}",
            "DOC-001",
        )
        if result["unlisted_deps"]:
            report.error(
                "Dependencies in pyproject.toml are missing from soup.yaml: "
                + str(result["unlisted_deps"]),
                "DOC-001",
            )
        else:
            report.info("All pyproject.toml dependencies are listed in soup.yaml", "DOC-001")

    report.auto_save("soup_inventory_complete", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DOC-003")
def test_readme_describes_training_workflow(
    request,
    evidence_output_dir,
):
    """
    DOC-003: docs shall describe the intended role of the toolkit.
    Verify README.md mentions the training workflow.
    """

    project_root = Path(__file__).resolve().parents[1]
    readme_path = project_root / "README.md"

    report = EvidenceReport(
        subject="README describes training workflow",
        test_id=request.node.nodeid,
    )

    if not readme_path.exists():
        report.error("README.md not found", "DOC-003")
    else:
        content = readme_path.read_text(encoding="utf-8").lower()
        keywords = ("train", "trainingpipeline", "medicalimagetrainer")
        if not any(kw in content for kw in keywords):
            report.error(
                f"README.md does not mention training workflow (checked: {keywords})",
                "DOC-003",
            )
        else:
            report.info(
                f"README.md describes training workflow — found one of {keywords}",
                "DOC-003",
            )

    report.auto_save("DOC003_readme_training_workflow", evidence_output_dir)
    assert not report.has_errors, report.summary()
