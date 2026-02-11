#%%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # adjust if needed
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestors.coca_gated_ingestor import COCAGatedIngestor
from medical_image_ai_toolkit.adapters.patient_sample_to_tensor import PatientSampleTensorAdapter

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "coca"
    / "cocacoronarycalciumandchestcts-2"
    / "Gated_release_final"
)
PATIENT_ID = "0"                               # <-- CHANGE IF NEEDED
MAX_PATIENTS_TO_SHOW = 1

# %%

print(DATASET_ROOT)
ingestor = COCAGatedIngestor(dataset_root=DATASET_ROOT)

print("=== Ingesting single patient ===")
patient_dir = DATASET_ROOT / "patient" / PATIENT_ID
sample = ingestor.ingest_patient(PATIENT_ID)


# %%
adapter = PatientSampleTensorAdapter(
    require_annotations=True
)

out = adapter(sample)

print(out["image"].shape)   # (1, Z, Y, X)
print(out["target"])        # tensor(1)
print(out["metadata"])

# %%
