# Single authority for where artifacts go.

from pathlib import Path


class TrainingArtifactManager:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.artifacts_root = project_root / "artifacts" / "training_runs"

    def create_run_dir(self, run_id: str) -> Path:
        run_dir = self.artifacts_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir
