from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset


class TensorDatamodule(Dataset):
    """
    Torch Dataset wrapping pre-converted tensor samples.

    Assumes:
    - All validation has already occurred
    - All tensors are correctly shaped and typed
    """

    def __init__(
        self,
        tensor_samples: List[Dict[str, Any]],
        deterministic: bool = True,
    ):
        if deterministic:
            tensor_samples = sorted(
                tensor_samples,
                key=lambda x: str(x.get("patient_id", ""))
            )

        for sample in tensor_samples:
            if not isinstance(sample, dict):
                raise TypeError("Each sample must be a dict")

        if not isinstance(tensor_samples, list):
            raise TypeError("tensor_samples must be a list")

        if len(tensor_samples) == 0:
            raise ValueError("tensor_samples must not be empty")

        self.samples = tensor_samples
        
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
