from __future__ import annotations

import itertools
import math
import random
from typing import Any


class HyperparameterSpace:
    """
    Defines a search space as named lists of candidate values.

    Examples
    --------
    space = HyperparameterSpace(learning_rate=[1e-3, 1e-4], epochs=[5, 10])
    space.grid()          # all 4 combinations
    space.random_sample(2, seed=0)  # 2 random picks
    len(space)            # 4
    """

    def __init__(self, **params: list[Any]) -> None:
        for name, values in params.items():
            if not isinstance(values, list):
                raise TypeError(
                    f"Each parameter must be a list of candidate values; "
                    f"got {type(values).__name__!r} for '{name}'"
                )
        self._params: dict[str, list[Any]] = params

    def grid(self) -> list[dict[str, Any]]:
        """Return every combination of parameter values (cartesian product)."""
        if not self._params:
            return [{}]
        keys = list(self._params.keys())
        return [dict(zip(keys, combo, strict=False)) for combo in itertools.product(*self._params.values())]

    def random_sample(self, n: int, seed: int | None = None) -> list[dict[str, Any]]:
        """Return up to *n* combinations drawn without replacement."""
        all_combos = self.grid()
        rng = random.Random(seed)
        return rng.sample(all_combos, min(n, len(all_combos)))

    def __len__(self) -> int:
        if not self._params:
            return 1
        return math.prod(len(v) for v in self._params.values())
