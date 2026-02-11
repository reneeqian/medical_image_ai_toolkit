from pathlib import Path
from typing import Dict, List
import json


def save_splits(
    splits: Dict[str, List[str]],
    run_dir: Path,
) -> None:
    path = run_dir / "splits.json"
    with path.open("w") as f:
        json.dump(splits, f, indent=2)
