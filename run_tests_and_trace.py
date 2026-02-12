import subprocess
from pathlib import Path
import sys
import os

from regulatory_tools.traceability.generator import (
    build_trace_matrix,
    write_markdown,
)
from regulatory_tools.traceability.validate_traceability import (
    validate_traceability,
)


PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DIR = PROJECT_ROOT / "tests"
REQUIREMENTS_YAML = PROJECT_ROOT / "docs" / "requirements.yaml"
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


def generate_traceability_matrix():
    print("\n[Runner] Generating traceability matrix...\n")

    # Validate requirement coverage
    missing, untracked = validate_traceability(
        requirements_yaml=REQUIREMENTS_YAML,
        test_dir=TEST_DIR,
    )

    if missing:
        print("[Traceability] Requirements declared but not tested:")
        for r in sorted(missing):
            print(f"  - {r}")

    if untracked:
        print("[Traceability] Tests reference undeclared requirements:")
        for r in sorted(untracked):
            print(f"  - {r}")


    matrix = build_trace_matrix(
        requirements_yaml=REQUIREMENTS_YAML,
        evidence_root=EVIDENCE_ROOT,
    )

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
    run_pytest()
    generate_traceability_matrix()
