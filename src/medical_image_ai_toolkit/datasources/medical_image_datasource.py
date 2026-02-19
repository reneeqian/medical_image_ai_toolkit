from pathlib import Path
from typing import List, Tuple
import torch
from torch.utils.data import Dataset



class MedicalImageDataSource:
    """
    High-level medical imaging datasource controller.
    """

    def __init__(self, dataset_path: Path, ingestor):
        self.dataset_path = Path(dataset_path)
        self.ingestor = ingestor

        self.patient_ids = self._resolve_patient_ids()

    # ----------------------------------------------------------
    # Dataset Introspection
    # ----------------------------------------------------------

    def _resolve_patient_ids(self) -> List[str]:
        return self.ingestor.list_patients()

    def get_num_patients(self) -> int:
        return len(self.patient_ids)

    # ----------------------------------------------------------
    # Data Access
    # ----------------------------------------------------------

    def get_volume(self, patient_index: int) -> torch.Tensor:
        patient_id = self.patient_ids[patient_index]
        sample = self.ingestor.ingest(patient_id)
        return sample["image"]

    def get_slice(self, patient_index: int, slice_index: int) -> torch.Tensor:
        volume = self.get_volume(patient_index)
        return volume[:, slice_index, :, :]

    # ----------------------------------------------------------
    # Splitting
    # ----------------------------------------------------------

    def get_train_val_test_split(self, split_strategy):
        splits = split_strategy.split(self.patient_ids)

        return (
            _DatasetView(self, splits["train"]),
            _DatasetView(self, splits["val"]),
            _DatasetView(self, splits["test"]),
        )



class _DatasetView(Dataset):
    """
    Lightweight view into a subset of patients.
    """

    def __init__(self, parent_ds, patient_subset):
        self.parent = parent_ds
        self.patient_ids = patient_subset

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        return self.parent.ingestor.ingest(patient_id)
