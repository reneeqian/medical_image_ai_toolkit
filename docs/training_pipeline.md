Data flow diagram

Adapter → Dataset → Trainer

How to add a new task

What “training pipeline” means in this project

Inputs: PatientSample

Outputs: trained model + artifacts

## PatientSample

### image_volume
- Type: np.ndarray
- Shape: (Z, Y, X)
- Units: Hounsfield Units
- Orientation: axial slices, inferior → superior

### spacing
- Tuple[float, float, float]
- Order: (z, y, x)
- Units: mm

### annotations.vector_rois
- Dict[int, List[VectorROI]]
- Slice indices are zero-based


## Model Input Tensor

image:
- Type: torch.Tensor
- Shape: (C, Z, Y, X)
- Range: [0, 1]
- Normalization: clipped HU [-1000, 1000]

target:
- Type: torch.Tensor | None
- Shape: task-dependent


┌────────────────────────────┐
│ Coronary_prj (APPLICATION) │
│ - dataset choice           │
│ - split policy             │
│ - run intent               │
│ - artifact location        │
└────────────▲───────────────┘
             │
┌────────────┴───────────────┐
│ medimg_training (LIBRARY)  │
│ - contracts                │
│ - trainers                 │
│ - adapters                 │
│ - abstract training flow   │
└────────────▲───────────────┘
             │
┌────────────┴───────────────┐
│ Data on disk (lazy)        │
│ - patient_id → load sample │
└────────────────────────────┘
