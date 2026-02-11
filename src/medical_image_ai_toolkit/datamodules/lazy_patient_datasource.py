from __future__ import annotations

from typing import Sequence, Callable, Optional, Dict
from torch.utils.data import Dataset

from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample


class LazyPatientDataSource(Dataset):
    """
    Lazily loads PatientSamples via an ingestor and adapts them to tensors.

    Responsibilities:
    - Hold a resolved universe of patient_ids
    - Lazily load one patient at a time via an ingestor
    - Apply an adapter to produce model-ready tensors

    Non-responsibilities:
    - Inferring dataset structure
    - Parsing filesystem layouts
    - Guessing patient identity
    """

    def __init__(
        self,
        *,
        ingestor,
        adapter,
        patient_ids: Optional[Sequence[str]] = None,
        dataroot: Optional[str] = None,
        patient_id_resolver: Optional[Callable[[str], Sequence[str]]] = None,
    ):
        """
        Args:
            ingestor:
                Object responsible for loading a PatientSample given a patient_id.
                Must implement: load_patient(patient_id) -> PatientSample

            adapter:
                Callable that converts a PatientSample into tensors.

            patient_ids:
                Explicit list of patient identifiers (preferred for full control).

            dataroot:
                Root path of the dataset. Required only if patient_ids is not given.

            patient_id_resolver:
                Callable that returns a list of patient_ids given dataroot.
                Required only if patient_ids is not given.
        """

        # ------------------------------------------------------------
        # Resolve patient_ids (explicit > inferred)
        # ------------------------------------------------------------
        if patient_ids is not None:
            resolved_ids = list(patient_ids)

        else:
            if dataroot is None or patient_id_resolver is None:
                raise ValueError(
                    "LazyPatientDataSource requires either:\n"
                    "  - patient_ids, OR\n"
                    "  - (dataroot + patient_id_resolver)"
                )

            resolved_ids = list(patient_id_resolver(dataroot))

        if not resolved_ids:
            raise ValueError("Resolved patient_ids is empty")

        # ------------------------------------------------------------
        # Store state
        # ------------------------------------------------------------
        self.patient_ids = resolved_ids
        self.ingestor = ingestor
        self.adapter = adapter

    # ------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        patient_id = self.patient_ids[idx]

        # --- Lazy load exactly one patient ---
        sample: PatientSample = self.ingestor.load_patient(patient_id)

        # --- Adapt to model-ready tensors ---
        return self.adapter(sample)

    def get_num_slices(self, patient_idx: int) -> int:
            """
            Return number of slices for a patient WITHOUT loading image tensors.

            This must be cheap and metadata-only.
            """
            patient_id = self.patient_ids[patient_idx]
            return self.ingestor.get_num_slices(patient_id)