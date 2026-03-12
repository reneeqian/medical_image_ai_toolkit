import subprocess
from pathlib import Path
import sys
import os
import ast
import xml.etree.ElementTree as ET


from regulatory_tools.traceability.generator import (
    build_trace_matrix,
    write_markdown,
)
from regulatory_tools.traceability.validate_traceability import (
    validate_traceability,
    find_unmarked_tests,
)


PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DIR = PROJECT_ROOT / "tests"
REQUIREMENTS_YAML = PROJECT_ROOT / "docs" /"requirements.yaml"
EVIDENCE_ROOT = PROJECT_ROOT / "artifacts" / "evidence_runs"
OUTPUT_MATRIX = PROJECT_ROOT / "docs" / "traceability_matrix.md"


def run_pytest():
    print("\n[Runner] Running pytest...\n")

    result = subprocess.run(
        ["pytest", str(TEST_DIR)],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        print("\n[Runner] Pytest failed. Aborting traceability generation.")
        sys.exit(1)
        
def run_pytest_with_coverage():
    print("\n[Runner] Running pytest with coverage...\n")

    coverage_dir = PROJECT_ROOT / "artifacts" / "coverage"
    coverage_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "pytest",
            str(TEST_DIR),
            "--cov=medical_image_ai_toolkit",
            "--cov-report=term",
            f"--cov-report=html:{coverage_dir / 'html'}",
            f"--cov-report=xml:{coverage_dir / 'coverage.xml'}",
            "--cov-fail-under=85",

        ],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        print("\n[Runner] Pytest failed. Aborting traceability generation.")
        sys.exit(1)

    print(f"\n[Coverage] Reports saved to: {coverage_dir}\n")

def collect_requirement_markers(test_root: Path):
    """
    Scan pytest files and collect requirement markers.

    Returns:
        dict[str, list[str]]
        { requirement_id: [test_node_ids...] }
    """

    requirement_map = {}

    for test_file in test_root.rglob("test_*.py"):
        module = ast.parse(test_file.read_text())

        for node in module.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            test_name = node.name

            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    if (
                        isinstance(decorator.func, ast.Attribute)
                        and decorator.func.attr == "requirement"
                    ):

                        for arg in decorator.args:
                            req_id = arg.value
                            node_id = f"{test_file.relative_to(PROJECT_ROOT)}::{test_name}"
                            requirement_map.setdefault(req_id, []).append(node_id)

    return requirement_map

def compute_requirement_coverage(matrix):
    """
    Calculate requirement coverage statistics.

    Returns:
        coverage_percent, tested_count, total_count, untested_requirements
    """

    total = len(matrix)

    tested = [
        r for r in matrix
        if r.get("status") in ("PASS", "LINKED")
    ]

    untested = [
        r["requirement_id"]
        for r in matrix
        if r.get("status") == "UNTESTED"
    ]

    tested_count = len(tested)

    coverage = 0.0
    if total > 0:
        coverage = (tested_count / total) * 100

    return coverage, tested_count, total, untested

def compute_code_coverage():
    """
    Parse coverage.xml to compute overall coverage and
    collect uncovered lines per file.
    """

    coverage_xml = PROJECT_ROOT / "artifacts" / "coverage" / "coverage.xml"

    if not coverage_xml.exists():
        return None, {}

    tree = ET.parse(coverage_xml)
    root = tree.getroot()

    coverage_percent = float(root.attrib.get("line-rate", 0)) * 100

    uncovered = {}

    for cls in root.findall(".//class"):

        filename = cls.attrib["filename"]

        missing = []

        for line in cls.findall(".//line"):

            if line.attrib.get("hits") == "0":
                missing.append(int(line.attrib["number"]))

        if missing:
            uncovered[filename] = sorted(missing)

    return coverage_percent, uncovered

def save_uncovered_lines(uncovered_lines):

    output = PROJECT_ROOT / "artifacts" / "coverage" / "uncovered_lines.txt"

    with output.open("w") as f:

        for file, lines in sorted(uncovered_lines.items()):

            f.write(f"{file}\n")

            for ln in lines:
                f.write(f"  line {ln}\n")

            f.write("\n")

    print(f"[Coverage] Uncovered lines saved to {output}")

def generate_traceability_matrix():
    print("\n[Runner] Generating traceability matrix...\n")

    # Validate requirement coverage
    missing, untracked = validate_traceability(
        requirements_yaml=REQUIREMENTS_YAML,
        test_dir=TEST_DIR,
    )

    unmarked_tests = find_unmarked_tests(TEST_DIR)
    marker_links = collect_requirement_markers(TEST_DIR)

    if missing:
        print("[Traceability] Requirements declared but not tested:")
        for r in sorted(missing):
            print(f"  - {r}")

    if untracked:
        print("[Traceability] Tests reference undeclared requirements:")
        for r in sorted(untracked):
            print(f"  - {r}")
            
    if unmarked_tests:
        print("[Traceability] Tests without requirement markers:")
        for t in unmarked_tests:
            print(f"  - {t}")

    matrix = build_trace_matrix(
        requirements_yaml=REQUIREMENTS_YAML,
        evidence_root=EVIDENCE_ROOT,
    )
    
    # ------------------------------------------------------------------
    # Merge pytest requirement markers into the matrix
    # ------------------------------------------------------------------

    for row in matrix:
        req_id = row["requirement_id"]

        # Convert existing tests string -> set
        existing_tests = set(
            t.strip() for t in row.get("tests", "").split(",") if t.strip()
        )

        marker_tests = set(marker_links.get(req_id, []))
        merged = sorted(existing_tests.union(marker_tests))
        row["tests"] = ", ".join(merged)
    
    # ------------------------------------------------------------------
    # Update status based on linked tests
    # ------------------------------------------------------------------

    for entry in matrix:
        tests = entry.get("tests", "")
        evidence = entry.get("evidence_files", "")

        has_tests = bool(tests.strip())
        has_evidence = bool(evidence.strip())

        if has_evidence:
            entry["status"] = "PASS"
        elif has_tests:
            entry["status"] = "LINKED"
        else:
            entry["status"] = "UNTESTED"
    
    # ------------------------------------------------------------------
    # Requirement Coverage Summary
    # ------------------------------------------------------------------

    coverage, tested_count, total_count, untested_requirements = compute_requirement_coverage(matrix)
    
        # ------------------------------------------------------------------
    # Code Coverage
    # ------------------------------------------------------------------

    code_coverage, uncovered_lines = compute_code_coverage()

    if code_coverage is not None:

        print("\n[Coverage] Code Coverage Summary")
        print("-------------------------------------------")
        print(f"Coverage: {code_coverage:.1f}%")

        if uncovered_lines:
            print(f"[Coverage] {sum(len(v) for v in uncovered_lines.values())} uncovered lines detected")

            save_uncovered_lines(uncovered_lines)

    print("\n[Traceability] Requirement Coverage Summary")
    print("-------------------------------------------")
    print(f"Tested requirements: {tested_count} / {total_count}")
    print(f"Coverage: {coverage:.1f}%")

    if untested_requirements:
        print("\n[Traceability] Untested requirements:")
        for r in sorted(untested_requirements):
            print(f"  - {r}")
    
    OUTPUT_MATRIX.parent.mkdir(exist_ok=True)
    write_markdown(
        matrix,
        OUTPUT_MATRIX,
        req_coverage_summary={
            "coverage": coverage,
            "tested": tested_count,
            "total": total_count,
            "untested": untested_requirements,
        },
        code_coverage_summary={
            "coverage": code_coverage
        }
    )

    abs_path = OUTPUT_MATRIX.resolve()

    print("\n[Traceability] Matrix successfully generated.")
    print(f"[Traceability] Saved to: {abs_path}\n")

    # Attempt to open automatically
    try:
        if sys.platform == "darwin":  # macOS
            subprocess.run(["open", str(abs_path)])
        elif sys.platform == "win32":  # Windows
            os.startfile(abs_path)  # type: ignore
        else:  # Linux
            subprocess.run(["xdg-open", str(abs_path)])
    except Exception as e:
        print(f"[Traceability] Could not auto-open file: {e}")


if __name__ == "__main__":
    run_pytest_with_coverage()
    generate_traceability_matrix()
