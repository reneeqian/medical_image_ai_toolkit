from pathlib import Path
from typing import Iterable
from medical_image_ai_toolkit.training.run_metadata import TrainingRunMetadata
from medical_image_ai_toolkit.training.artifact_manager import TrainingArtifactManager
from medical_image_ai_toolkit.training.splits import HashPatientIDSplit
from medical_image_ai_toolkit.training.split_writer import save_splits


def prepare_training_run(
    *,
    project_root: Path,
    patient_ids: Iterable[str],
    project_name: str,
    dataset_name: str,
    dataset_version: str,
    code_commit: str,
) -> Path:
    metadata = TrainingRunMetadata.create(
        project_name=project_name,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        inclusion_criteria="All patients with valid coronary CT volumes",
        exclusion_criteria="Invalid volumes or missing spacing",
        split_strategy="hash(patient_id)",
        code_commit=code_commit,
    )

    artifact_mgr = TrainingArtifactManager(project_root)
    run_dir = artifact_mgr.create_run_dir(metadata.run_id)

    metadata.save(run_dir)

    splitter = HashPatientIDSplit(val_fraction=0.2)
    splits = splitter.split(patient_ids)
    save_splits(splits, run_dir)

    return run_dir
