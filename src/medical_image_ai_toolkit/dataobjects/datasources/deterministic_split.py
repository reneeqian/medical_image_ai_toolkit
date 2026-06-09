from __future__ import annotations

import random
from collections.abc import Iterable


class DeterministicHoldoutSplit:
    def __init__(
        self,
        train: float = 0.7,
        val: float = 0.15,
        seed: int = 42,
        max_train: int | None = None,
        max_val: int | None = None,
        max_test: int | None = None,
    ) -> None:
        self.train = train
        self.val = val
        self.seed = seed
        self.max_train = max_train
        self.max_val = max_val
        self.max_test = max_test

    def split(self, patient_ids: Iterable[str]) -> tuple[list[str], list[str], list[str]]:
        rng = random.Random(self.seed)
        ids = list(patient_ids)
        rng.shuffle(ids)
        n = len(ids)
        train_end = int(self.train * n)
        val_end = train_end + int(self.val * n)
        train_ids = ids[:train_end]
        val_ids = ids[train_end:val_end]
        test_ids = ids[val_end:]
        if self.max_train is not None:
            train_ids = train_ids[: self.max_train]
        if self.max_val is not None:
            val_ids = val_ids[: self.max_val]
        if self.max_test is not None:
            test_ids = test_ids[: self.max_test]
        return train_ids, val_ids, test_ids
