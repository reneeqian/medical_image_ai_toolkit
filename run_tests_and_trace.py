from pathlib import Path
from regulatory_tools.testing import run_tests_and_trace

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_ROOT / "src"

if __name__ == "__main__":
    run_tests_and_trace(project_root=PROJECT_ROOT)