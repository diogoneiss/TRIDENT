"""The files an imputation run leaves behind (ADR 0004, decision 8)."""

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.training.artifacts import ArtifactWriter
from src.training.config import complete_configuration
from src.training.types import CLASSIFICATION, IMPUTATION, Hyperparameters


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


def _identity_scaler() -> StandardScaler:
    """Fitted so a scaled value and its original unit are the same number.

    Keeps a rendering test about rendering: no scaling arithmetic stands between the
    value written into the ledger and the text the preview is asserted on.
    """
    scaler = StandardScaler()
    scaler.fit(pd.DataFrame({"amount": [-1.0, 1.0]}))
    return scaler


def _row(preview: str, label: str) -> str:
    return next(line for line in preview.splitlines() if line.startswith(f"| {label} |"))


def test_the_preview_rounds_to_three_decimals_and_prints_a_vanishing_value_as_zero(
    tmp_path,
) -> None:
    """Ticket 0004: `1.144e-09` reads as a real measurement when it means zero.

    `spambase` is 77% exact zeros and float32 leaves each one a hair off zero, so the
    old `.4g` format filled the preview with scientific-notation noise that made the
    artifact hard to trust at a glance. Three decimals is the agreed display precision;
    anything too small to show at it is zero rather than a tiny number.
    """
    writer = _writer(tmp_path)
    cells = pd.DataFrame(
        [
            {"row": 0, "column": "amount", "kind": "numerical", "population": "masked",
             "actual": 1.144e-09, "imputed": 123.4567891, "confidence": float("nan")},
        ]
    )

    preview_path, ledger_path = writer.write_imputation_preview(
        "imputation_fold_1", cells, seed=42, fold=1, sample_rows=10,
        scaler=_identity_scaler(), numerical_columns=["amount"],
    )

    preview = preview_path.read_text(encoding="utf-8")
    # A rounding marker may or may not be attached; this test is about the number.
    assert _row(preview, "actual").replace("*", "") == "| actual | 0.000 |"
    assert "123.457" in _row(preview, "imputed")
    assert "1.144e-09" not in preview

    # The ledger is the full record behind the preview, so it keeps what was scored.
    ledger = pd.read_csv(ledger_path)
    assert float(ledger["actual_original_units"].iloc[0]) == pytest.approx(1.144e-09)


def test_a_number_the_preview_had_to_round_is_marked_and_an_exact_one_is_not(
    tmp_path,
) -> None:
    """Ticket 0004: three decimals is a lossy view, so the preview says where it lost.

    Without a marker a reader cannot tell `0.500` that is exactly a half from `0.500`
    that is really 0.4999994, which is the difference between a number they can quote
    and one they have to look up. Marking is decided per cell by whether the rendered
    text reads back as the value it came from, so it never decorates an exact number.
    """
    writer = _writer(tmp_path)
    cells = pd.DataFrame(
        [
            {"row": 0, "column": "amount", "kind": "numerical", "population": "masked",
             "actual": 0.5, "imputed": 123.4567891, "confidence": float("nan")},
        ]
    )

    preview_path, _ = writer.write_imputation_preview(
        "imputation_fold_1", cells, seed=42, fold=1, sample_rows=10,
        scaler=_identity_scaler(), numerical_columns=["amount"],
    )

    preview = preview_path.read_text(encoding="utf-8")
    assert _row(preview, "actual") == "| actual | 0.500 |"
    assert _row(preview, "imputed") == "| imputed | 123.457* |"
    # A marker nobody can decode is just noise, so the legend travels with it.
    assert "*" in preview and "ledger" in preview.lower()


def test_a_truth_the_run_knows_exactly_is_shown_exactly(tmp_path) -> None:
    """Ticket 0004: the preview must not re-derive a number the run already knows.

    Inverting the scaler from the scaled value amplifies its float32 rounding by the
    column's `scale_`, which on `kc2`'s widest column is a tenth of a unit. The scaled
    value here is a hair off -1.0 and the scaler is 500 wide, so re-deriving it would
    print 500.100 where the dataset plainly holds 500. The decode stage carries the exact
    number across in `actual_original`, and the preview prefers it.
    """
    writer = _writer(tmp_path)
    cells = pd.DataFrame(
        [
            {"row": 0, "column": "amount", "kind": "numerical", "population": "masked",
             "actual": -0.9998, "imputed": -0.5, "confidence": float("nan"),
             "actual_original": 500.0},
        ]
    )

    preview_path, ledger_path = writer.write_imputation_preview(
        "imputation_fold_1", cells, seed=42, fold=1, sample_rows=10,
        scaler=_scaler(), numerical_columns=["amount"],
    )

    preview = preview_path.read_text(encoding="utf-8")
    assert _row(preview, "actual") == "| actual | 500.000 |"
    assert "500.100" not in preview

    ledger = pd.read_csv(ledger_path)
    assert float(ledger["actual_original_units"].iloc[0]) == 500.0
    # The imputed side has no exact counterpart: float32 is the model's real precision.
    assert float(ledger["imputed_original_units"].iloc[0]) == pytest.approx(750.0)


def test_a_runs_hyperparameter_file_names_every_value_an_imputation_run_trained_with(tmp_path) -> None:
    """The objective, the patience and the head mode move what an imputation run computes,
    and since ADR 0013 the objective's default depends on the task, so a results directory
    that left them out could not say which configuration it holds. The file carries the same
    key set a promoted configuration does; classification's is unchanged."""
    values = Hyperparameters(pretraining_objective="embedding_normalized", decode_epochs=450)
    imputation = ArtifactWriter(tmp_path / "i", tmp_path / "m", "credit-g_20nan").write_hyperparameters(
        values, IMPUTATION
    )
    classification = ArtifactWriter(tmp_path / "c", tmp_path / "m", "vehicle_00nan").write_hyperparameters(
        Hyperparameters(), CLASSIFICATION
    )

    written = json.loads(imputation.read_text())
    assert written == complete_configuration(values, "imputation")
    assert written["PRETRAIN_OBJECTIVE"] == "embedding_normalized"
    assert list(json.loads(classification.read_text())) == [
        "DIM", "HIDDEN_DIM", "HEADS", "LAYERS", "DIM_FEED", "DROPOUT", "EPOCHS_PRE", "BATCH",
        "LR_PRE", "WEIGHT_DECAY_PRE", "PROB_MASCARA", "LR_SCHEDULER", "EPOCH_FINE", "LR_FINE",
        "WEIGHT_DECAY_FINE", "LABELS",
    ]
