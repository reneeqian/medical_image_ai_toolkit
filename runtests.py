from pathlib import Path

from regulatory_tools.testing.run_tests_and_trace import run_tests_and_trace

PROJECT_ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    run_tests_and_trace(project_root=PROJECT_ROOT, min_grade="B")
