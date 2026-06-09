from __future__ import annotations

from pathlib import Path


def find_run_dir(
    runs_root: Path | str,
    run_id: str | None = None,
    requires_file: str | None = None,
) -> Path:
    """
    Return the run directory matching *run_id*, or the most-recent directory
    when *run_id* is ``None``.

    Run directories are named with timestamp strings
    (``YYYYMMDD_HHMMSS_ffffff`` or similar) so lexicographic sort equals
    chronological order.

    Parameters
    ----------
    runs_root:
        Root directory that contains timestamped run subdirectories.
    run_id:
        Exact directory name, or a prefix (e.g. ``"20260427"`` matches the
        latest run whose name starts with that string).
    requires_file:
        When set, only consider directories that contain a file with this
        name.  Useful for skipping empty or incomplete run directories.

    Raises
    ------
    FileNotFoundError
        When *runs_root* does not exist, is empty, or no directory matches
        the requested *run_id* (and *requires_file* constraint).
    """
    runs_root = Path(runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")

    def _qualifies(p: Path) -> bool:
        return p.is_dir() and (requires_file is None or (p / requires_file).exists())

    if run_id is not None:
        candidate = runs_root / run_id
        if _qualifies(candidate):
            return candidate
        matches = sorted(
            p
            for p in runs_root.iterdir()
            if p.is_dir() and p.name.startswith(run_id) and _qualifies(p)
        )
        if matches:
            return matches[-1]
        raise FileNotFoundError(f"No run matching '{run_id}' in {runs_root}")

    dirs = sorted(p for p in runs_root.iterdir() if _qualifies(p))
    if not dirs:
        raise FileNotFoundError(f"No runs found in {runs_root}")
    return dirs[-1]
