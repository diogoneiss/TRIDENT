"""The files an imputation run leaves behind (ADR 0004, decision 8)."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.training.artifacts import ArtifactWriter


def _scored_cells() -> pd.DataFrame:
    """Two rows' worth of scored cells, one of each kind and each population."""
    return pd.DataFrame(
        [
            {"row": 0, "column": "grade", "kind": "categorical", "population": "masked",
             "actual": "a", "imputed": "a", "confidence": 0.9},
            {"row": 0, "column": "amount", "kind": "numerical", "population": "masked",
             "actual": -1.0, "imputed": -0.5, "confidence": float("nan")},
            {"row": 1, "column": "grade", "kind": "categorical", "population": "induced",
             "actual": "b", "imputed": "a", "confidence": 0.4},
            {"row": 1, "column": "amount", "kind": "numerical", "population": "induced",
             "actual": 1.0, "imputed": 0.25, "confidence": float("nan")},
            {"row": 2, "column": "amount", "kind": "numerical", "population": "masked",
             "actual": 0.0, "imputed": 0.1, "confidence": float("nan")},
        ]
    )


def _scaler() -> StandardScaler:
    """Fitted so that a scaled -1.0 is an amount of 500 and 1.0 is 1500."""
    scaler = StandardScaler()
    scaler.fit(pd.DataFrame({"amount": [500.0, 1500.0]}))
    return scaler


def _writer(tmp_path) -> ArtifactWriter:
    return ArtifactWriter(tmp_path / "results", tmp_path / "metrics", "credit-g_20nan")


def test_the_ledger_holds_every_scored_cell_and_says_which_are_previewed(tmp_path) -> None:
    """The preview is a window onto the ledger, never a different set of cells.

    A reader who wonders why a row looks odd has to be able to find it in the full
    record, so the flag is a filter rather than a hunt.
    """
    writer = _writer(tmp_path)
    cells = _scored_cells()

    preview_path, ledger_path = writer.write_imputation_preview(
        "imputation_fold_1", cells, seed=42, fold=1, sample_rows=1,
        scaler=_scaler(), numerical_columns=["amount"],
    )

    ledger = pd.read_csv(ledger_path)
    assert len(ledger) == len(cells)
    previewed = ledger[ledger["in_preview"]]
    assert 0 < len(previewed) < len(ledger)
    assert previewed["row"].nunique() == 1
    text = preview_path.read_text(encoding="utf-8")
    for row in previewed["row"].unique():
        assert f"row {row}" in text


def test_the_preview_reads_in_the_units_a_person_recognises(tmp_path) -> None:
    """A z-score tells a reader nothing about whether an imputation was sensible.

    The scaler here maps -1.0 to an amount of 500 and 1.0 to 1500, so the preview should
    talk about amounts while the ledger keeps the scaled values scoring used.
    """
    writer = _writer(tmp_path)

    preview_path, ledger_path = writer.write_imputation_preview(
        "imputation_fold_1", _scored_cells(), seed=42, fold=1, sample_rows=10,
        scaler=_scaler(), numerical_columns=["amount"],
    )

    text = preview_path.read_text(encoding="utf-8")
    assert "500" in text and "1500" in text
    ledger = pd.read_csv(ledger_path)
    amounts = ledger[ledger["column"] == "amount"]
    # One table holds both kinds, so a value column is text once written; a reader casts
    # by kind, which is the price of keeping the whole record in a single file.
    assert set(amounts["actual"].astype(float)) == {-1.0, 1.0, 0.0}
    assert set(amounts["actual_original_units"].astype(float)) == {500.0, 1500.0, 1000.0}


def test_the_per_column_table_says_which_column_a_poor_fold_struggled_with(tmp_path) -> None:
    """Pooled numbers rank a fold; this says where it went wrong, without crowding MLflow.

    Long form with a fold column, so one file covers every fold of a run rather than one
    file per fold, and spambase's 57 columns never become tracked metric series.
    """
    writer = _writer(tmp_path)
    per_column = pd.DataFrame(
        [
            {"column": "amount", "metric": "rmse_num_z", "value": 5.0},
            {"column": "grade", "metric": "acc_cat", "value": 0.5},
        ]
    )

    path = writer.write_per_column_imputation(
        {1: {"masked": per_column}, 2: {"masked": per_column}}
    )

    table = pd.read_csv(path)
    assert path.relative_to(writer.results_dir) == Path("metrics/per_column_imputation.csv")
    assert set(table.columns) == {"fold", "population", "column", "metric", "value"}
    assert sorted(table["fold"].unique()) == [1, 2]
    assert len(table) == 4


def test_a_wide_row_is_split_so_the_preview_stays_readable(tmp_path) -> None:
    """A row can have a dozen cells filled in, and one table that wide reads as noise.

    Observed on real credit-g output: eleven columns in a single markdown table wraps
    badly, and a 57-column dataset would be worse. Splitting keeps every cell present.
    """
    writer = _writer(tmp_path)
    wide_row = pd.DataFrame(
        [
            {"row": 0, "column": f"field_{index}", "kind": "categorical",
             "population": "masked", "actual": "a", "imputed": "b", "confidence": 0.5}
            for index in range(14)
        ]
    )

    preview_path, _ = writer.write_imputation_preview(
        "imputation_fold_1", wide_row, seed=42, fold=1, sample_rows=1,
    )

    text = preview_path.read_text(encoding="utf-8")
    for index in range(14):
        assert f"field_{index}" in text
    widest = max(line.count("|") for line in text.splitlines() if line.startswith("| actual"))
    assert widest <= 8  # a leading label plus at most six cells
