from datetime import datetime
from pathlib import Path
import pytest
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------
# Project Root
# ---------------------------------------------------------------------
@pytest.fixture(scope="session")
def project_root() -> Path:
    """
    Returns the root of the Coronary_prj project.
    """
    return Path(__file__).resolve().parents[1] 

# ---------------------------------------------------------------------
# Evidence Output Directory
# ---------------------------------------------------------------------
@pytest.fixture(scope="session")
def evidence_output_dir():
    root = Path(__file__).resolve().parents[1] / "artifacts" / "evidence_runs"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

# ---------------------------------------------------------------------
# Tensor Batch Helpers
# ---------------------------------------------------------------------
def _dummy_tensor_batch(batch_size: int = 1):
    return {
        "image": torch.zeros(batch_size, 4),
        "target": torch.zeros(batch_size),
    }


def _nan_tensor_batch(batch_size: int = 1):
    return {
        "image": torch.zeros(batch_size, 4),
        "target": torch.full((batch_size,), float("nan")),
    }


# ---------------------------------------------------------------------
# Tensor Batch Fixtures
# ---------------------------------------------------------------------
class TensorDataset(Dataset):
    def __init__(self, num_batches: int = 4, nan: bool = False):
        self.num_batches = num_batches
        self.nan = nan

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if self.nan:
            return _nan_tensor_batch(batch_size=1)
        return _dummy_tensor_batch(batch_size=1)