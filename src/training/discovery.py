"""Runtime discovery of available datasets in datasets/processed_datasets/."""

from pathlib import Path

_PROCESSED_DIR = Path("datasets/processed_datasets")
_EXCLUDED_SUBDIRS = {"splits"}
_VALID_NAN_LEVELS = {0, 20, 40, 60, 80}


def discover_datasets(nan_level: int = 0, limit: int | None = None) -> list[str]:
    """Return sorted dataset names from *datasets/processed_datasets/*.

    Each subdirectory (except ``splits/``) is treated as a base dataset name.
    The *nan_level* integer is combined with the base name to form the full
    dataset name (e.g. ``vehicle`` + ``40`` → ``vehicle_40nan``).  The
    resulting CSV path is validated before inclusion.

    Parameters
    ----------
    nan_level:
        Missingness percentage.  Must be one of ``{0, 20, 40, 60, 80}``.
    limit:
        If not *None*, return only the first *limit* datasets (alphabetical).

    Returns
    -------
    list[str]
        Sorted list of full dataset names, e.g.
        ``["biodeg_00nan", "credit-g_00nan", …, "vehicle_00nan"]``.

    Raises
    ------
    FileNotFoundError
        If the processed datasets directory does not exist.
    ValueError
        If *nan_level* is not in the allowed set, or if no datasets are found.
    """
    if nan_level not in _VALID_NAN_LEVELS:
        raise ValueError(
            f"Invalid nan_level {nan_level}. Must be one of {sorted(_VALID_NAN_LEVELS)}."
        )

    if not _PROCESSED_DIR.is_dir():
        raise FileNotFoundError(f"Processed datasets directory not found: {_PROCESSED_DIR}")

    suffix = f"{nan_level:02d}nan"

    base_names: list[str] = sorted(
        entry.name
        for entry in _PROCESSED_DIR.iterdir()
        if entry.is_dir() and entry.name not in _EXCLUDED_SUBDIRS
    )

    dataset_names: list[str] = []
    for base in base_names:
        full_name = f"{base}_{suffix}"
        csv_path = _PROCESSED_DIR / base / f"{full_name}.csv"
        if csv_path.is_file():
            dataset_names.append(full_name)
        else:
            # Warn but don't fail — the dataset simply doesn't have this variant.
            import logging

            logging.getLogger(__name__).warning(
                "Skipping %s: CSV not found at %s", full_name, csv_path
            )

    if not dataset_names:
        raise ValueError(
            f"No datasets found with nan_level={nan_level} in {_PROCESSED_DIR}"
        )

    if limit is not None:
        dataset_names = dataset_names[:limit]

    return dataset_names
