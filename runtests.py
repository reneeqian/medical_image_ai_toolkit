from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

from regulatory_tools.testing import run_tests_and_trace

if __name__ == "__main__":
    run_tests_and_trace(project_root=PROJECT_ROOT)
