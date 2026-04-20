from pathlib import Path
from typing import List
import json
import matplotlib.pyplot as plt
import numpy as np
import math

class MedicalImageDataSource:
    """
    Lazy wrapper around dataset ingestors.

    Responsibilities:
    - expose dataset size
    - provide lazy patient loading
    - expose patient IDs for partitioning
    """

    def __init__(self, dataset_root: Path, ingestor):

        self.dataset_root = Path(dataset_root)
        self.ingestor = ingestor

        # discover dataset
        self.patient_ids: List[str] = self.ingestor.list_patient_ids()
        
        # these will be populated after partitioning
        self.train_ids: List[str] | None = None
        self.val_ids: List[str] | None = None
        self.test_ids: List[str] | None = None

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    
    def get_num_patients(self) -> int:
        return len(self.patient_ids)

    def get_patient_ids(self) -> List[str]:
        return list(self.patient_ids)
    
    def get_patient(self, patient_id: str):
        if patient_id not in self.patient_ids:
            raise ValueError(f"Patient ID '{patient_id}' not found in dataset")
        return self.ingestor.load_patient_sample(patient_id)

    def get_sample(self, patient_id: str):
        if patient_id not in self.patient_ids:
            raise ValueError(f"Patient ID '{patient_id}' not found in dataset")
        if hasattr(self.ingestor, "get_sample"):
            return self.ingestor.get_sample(patient_id)

        # fallback behavior: return patient object
        return self.get_patient(patient_id)
    
    def load_slice(self, patient_id: str, slice_index: int):
        sample = self.get_patient(patient_id)

        volume = sample.image_volume

        if slice_index < 0 or slice_index >= volume.shape[0]:
            raise IndexError(
                f"Slice index {slice_index} out of bounds for volume with {volume.shape[0]} slices"
            )

        return volume[slice_index]
        
    def create_partitions(self, strategy):
        train_ids, val_ids, test_ids = strategy.split(self.patient_ids)

        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids

        return train_ids, val_ids, test_ids

    
    def partition_summary(self):
        print("\nDataset Summary")
        print("-----------------")

        total = len(self.patient_ids)
        print(f"Total patients: {total}")

        if self.train_ids is None:
            print("\nPartitions have not been created yet.")
            return

        train_n = len(self.train_ids)
        val_n = len(self.val_ids)
        test_n = len(self.test_ids)

        print("\nPartitions")
        print("-----------------")
        print(f"Train: {train_n} ({train_n/total:.2%})")
        print(f"Val:   {val_n} ({val_n/total:.2%})")
        print(f"Test:  {test_n} ({test_n/total:.2%})")
    
    def get_train_ids(self):

        if self.train_ids is None:
            raise RuntimeError(
                "Partitions have not been created. Call create_partitions() first."
            )

        return self.train_ids

    def get_val_ids(self):

        if self.val_ids is None:
            raise RuntimeError(
                "Partitions have not been created. Call create_partitions() first."
            )

        return self.val_ids

    def get_test_ids(self):

        if self.test_ids is None:
            raise RuntimeError(
                "Partitions have not been created. Call create_partitions() first."
            )

        return self.test_ids
    
    def has_partitions(self):

        return (
            self.train_ids is not None
            and self.val_ids is not None
            and self.test_ids is not None
        )

    def save_partitions(self, run_dir: Path) -> Path:
        """
        Write the current partition assignment to ``run_dir/partitions.json``.

        Called by the training pipeline immediately after training so
        that the validation pipeline can reload the exact same split
        without re-running the split strategy.

        Parameters
        ----------
        run_dir : Path
            Directory to write into.  Must already exist.

        Returns
        -------
        Path
            Absolute path of the written file.

        Raises
        ------
        RuntimeError
            If partitions have not been created yet.
        """
        if not self.has_partitions():
            raise RuntimeError(
                "Partitions have not been created. "
                "Call create_partitions() before save_partitions()."
            )

        path = Path(run_dir) / "partitions.json"

        with open(path, "w") as f:
            json.dump(
                {
                    "train_ids": self.train_ids,
                    "val_ids":   self.val_ids,
                    "test_ids":  self.test_ids,
                },
                f,
                indent=2,
            )

        print(f"Partitions saved to: {path}")
        return path

    @classmethod
    def load_partitions(cls, datasource: "MedicalImageDataSource", path: Path) -> "MedicalImageDataSource":
        """
        Restore partition assignments from a ``partitions.json`` file
        onto an existing datasource instance.

        Every patient ID in the file is validated against the
        datasource's known population before being applied, so a
        dataset mismatch is caught immediately rather than silently
        producing wrong test results.

        Parameters
        ----------
        datasource : MedicalImageDataSource
            An already-constructed datasource whose ingestor points at
            the same dataset that was used during training.
        path : Path
            Path to the ``partitions.json`` written by ``save_partitions()``.

        Returns
        -------
        MedicalImageDataSource
            The same ``datasource`` instance with train/val/test IDs populated.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If any saved patient ID is absent from the current datasource.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Partition file not found: {path}")

        with open(path) as f:
            payload = json.load(f)

        known = set(datasource.patient_ids)
        for split in ("train_ids", "val_ids", "test_ids"):
            for pid in payload[split]:
                if pid not in known:
                    raise ValueError(
                        f"Saved partition references patient '{pid}' which is "
                        f"not present in the current datasource. Ensure the "
                        f"datasource points at the same dataset used during training."
                    )

        datasource.train_ids = payload["train_ids"]
        datasource.val_ids   = payload["val_ids"]
        datasource.test_ids  = payload["test_ids"]

        print(f"Partitions loaded from: {path}")
        print(
            f"  train={len(datasource.train_ids)}  "
            f"val={len(datasource.val_ids)}  "
            f"test={len(datasource.test_ids)}"
        )

        return datasource

    # ---------------------------------------------------------
    # Visualization APIs
    # ---------------------------------------------------------
    
    def show_slice(
        self,
        patient_id: str,
        slice_index: int | None = None,
        slice_range: tuple[int, int] | None = None,
        annotated_only: bool = False,
        show_annotations: bool = False,
        show_bboxes: bool = False,
        cols: int = 5,
        block: bool = True,
    ):
        
        sample = self.get_patient(patient_id)
        volume = sample.image_volume
        num_slices = volume.shape[0]

        vector_rois = None
        if sample.annotations is not None:
            vector_rois = getattr(sample.annotations, "vector_rois", None)


        # Determine slice list
        if annotated_only:

            if not vector_rois:
                print("No annotations found for this patient.")
                return

            slice_indices = sorted(vector_rois.keys())

        elif slice_index is not None:

            slice_indices = [slice_index]

        elif slice_range is not None:

            start, end = slice_range
            slice_indices = list(range(start, end))

        else:

            raise ValueError(
                "Provide slice_index, slice_range, or annotated_only=True"
            )

        # Validate indices
        for idx in slice_indices:
            if idx < 0 or idx >= num_slices:
                raise IndexError(f"Slice {idx} out of bounds")

        n = len(slice_indices)

        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

        if not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])

        axes = axes.flatten()

        for i, slice_idx in enumerate(slice_indices):
            if vector_rois and slice_idx in vector_rois:
                print("Slice", slice_idx, "ROIs:", len(vector_rois[slice_idx]))

            ax = axes[i]

            img = volume[slice_idx]

            ax.imshow(img, cmap="gray")
            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(img.shape[0], 0)
            ax.set_title(f"Slice {slice_idx}")
            ax.axis("off")

            if show_annotations and sample.annotations is not None:

                vector_rois = getattr(sample.annotations, "vector_rois", None)

                if vector_rois and slice_idx in vector_rois:

                    for roi in vector_rois[slice_idx]:

                        pts = roi.contour_px

                        if pts is None or len(pts) < 3:
                            continue

                        # draw contour
                        ax.plot(
                            pts[:, 0],
                            pts[:, 1],
                            linewidth=2,
                            color="red"
                        )

                        # ---- ADD BOUNDING BOX ----
                        if show_bboxes:

                            x_min = np.min(pts[:, 0])
                            x_max = np.max(pts[:, 0])
                            y_min = np.min(pts[:, 1])
                            y_max = np.max(pts[:, 1])

                            width = x_max - x_min
                            height = y_max - y_min

                            rect = plt.Rectangle(
                                (x_min, y_min),
                                width,
                                height,
                                linewidth=2,
                                edgecolor="yellow",
                                facecolor="none"
                            )

                            ax.add_patch(rect)

        # Turn off unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle(f"Patient {patient_id}")
        plt.tight_layout()

        if block:
            plt.show()
        else:
            plt.show(block=False)
            plt.close(fig)
    
    # ---------------------------------------------------------
    # Internal utilities
    # ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int):
        patient_id = self.patient_ids[idx]
        return self.get_patient(patient_id)