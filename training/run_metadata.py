# Defines what must be recorded for every training run.

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any
import json
from pathlib import Path
import hashlib
import uuid


@dataclass(frozen=True)
class TrainingRunMetadata:
    run_id: str
    project_name: str
    dataset_name: str
    dataset_version: str
    inclusion_criteria: str
    exclusion_criteria: str
    split_strategy: str
    code_commit: str
    created_at: str

    @staticmethod
    def create(
        *,
        project_name: str,
        dataset_name: str,
        dataset_version: str,
        inclusion_criteria: str,
        exclusion_criteria: str,
        split_strategy: str,
        code_commit: str,
    ) -> "TrainingRunMetadata":
        run_id = uuid.uuid4().hex[:12]
        created_at = datetime.utcnow().isoformat() + "Z"

        return TrainingRunMetadata(
            run_id=run_id,
            project_name=project_name,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
            split_strategy=split_strategy,
            code_commit=code_commit,
            created_at=created_at,
        )

    def save(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=False)
        path = run_dir / "metadata.json"
        with path.open("w") as f:
            json.dump(asdict(self), f, indent=2)
