import subprocess
from pathlib import Path
import sys
import os
import ast


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
            "--cov=Coronary_prj",
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
        
    OUTPUT_MATRIX.parent.mkdir(exist_ok=True)
    write_markdown(matrix, OUTPUT_MATRIX)

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
