"""Baseline imputers scored on exactly the cells the decoder is scored on (ADR 0007)."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.training.imputation_baselines import (
    KnnImputer,
    MeanModeImputer,
    fit_baseline_imputers,
    score_baselines,
)
from src.training.imputation_metrics import CATEGORICAL, NUMERICAL, mean_mode_baselines

# Interleaved on purpose: a categorical, a numerical, a categorical, a numerical. The KNN
# matrix orders its blocks numerical-first, so a frame that already came numerical-first
# would hide any confusion between the two orders.
CATEGORICAL_COLUMNS = ["colour", "shape"]
NUMERICAL_COLUMNS = ["size", "weight"]


def _frame(rows: int = 40) -> pd.DataFrame:
    colours = ["red", "blue", "green", "red"]
    shapes = ["round", "square", "round", "square"]
    return pd.DataFrame(
        {
            "colour": [colours[index % 4] for index in range(rows)],
            "size": [float(index % 9) for index in range(rows)],
            "shape": [shapes[index % 4] for index in range(rows)],
            "weight": [float(index % 5) / 2 for index in range(rows)],
        }
    )


def _cells(frame: pd.DataFrame, positions: list[tuple[int, str]]) -> pd.DataFrame:
    """The scored-cell table a decode stage would hand over for these positions.

    The model's own guess is irrelevant to a baseline, so it is the truth itself.
    """
    rows = []
    for row, column in positions:
        kind = CATEGORICAL if column in CATEGORICAL_COLUMNS else NUMERICAL
        actual = str(frame.at[row, column]) if kind == CATEGORICAL else float(frame.at[row, column])
        rows.append(
            {
                "row": row, "column": column, "kind": kind, "population": "masked",
                "actual": actual, "imputed": actual, "confidence": float("nan"),
                "actual_original": actual,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "row", "column", "kind", "population",
            "actual", "imputed", "confidence", "actual_original",
        ],
    )


def _split(frame: pd.DataFrame, train_rows: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame.iloc[:train_rows].reset_index(drop=True)
    test = frame.iloc[train_rows:].reset_index(drop=True)
    return train, test


def test_the_mean_mode_baseline_scores_at_parity() -> None:
    """Filling the mean or the mode is what ``impute_score`` is measured against, so the
    same fill scored through it must land on 1.0: the invariant that says the baseline's
    fill values and the score's own denominator agree cell for cell.

    Exact on a table of one kind, where numerator and denominator are the same arithmetic
    on the same arrays; within an ulp on a mixed table, where the categorical half compares
    ``1 - accuracy`` against a directly counted error rate.
    """
    train, test = _split(_frame())
    naive = mean_mode_baselines(train, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS)
    imputer = MeanModeImputer(NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS)
    imputer.fit(train)

    numerical_only = _cells(test, [(0, "size"), (3, "weight"), (7, "size"), (12, "weight")])
    assert score_baselines([imputer], test, numerical_only, naive)[
        "baseline/mean_mode/impute_score"
    ] == 1.0

    mixed = _cells(test, [(0, "size"), (1, "colour"), (5, "shape"), (9, "weight"), (14, "colour")])
    assert score_baselines([imputer], test, mixed, naive)[
        "baseline/mean_mode/impute_score"
    ] == pytest.approx(1.0, abs=1e-12)


def _dependent_frame(rows: int = 40) -> pd.DataFrame:
    """A table where two columns follow a third: ``size`` decides ``weight`` and ``shape``.

    Neighbours in ``size`` therefore know what a hidden ``weight`` or ``shape`` was, while
    the column mean or mode knows nothing about the row.
    """
    colours = ["red", "blue", "green", "red"]
    size = [float(index % 9) for index in range(rows)]
    return pd.DataFrame(
        {
            "colour": [colours[index % 4] for index in range(rows)],
            "size": size,
            "shape": ["round" if value < 4 else "square" for value in size],
            "weight": [2.0 * value for value in size],
        }
    )


def test_knn_beats_the_mean_where_a_column_follows_the_others() -> None:
    """The point of a second baseline is to be harder to beat than the first. Where a
    row's neighbours know the answer, KNN must score well below parity and the mean/mode
    must sit at it, on the very same cells.

    The frame is interleaved, so a fill returned in the KNN matrix's own order (numerical
    first) would land every guess on the wrong column and fail here.
    """
    train, test = _split(_dependent_frame())
    naive = mean_mode_baselines(train, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS)
    knn = KnnImputer(NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, n_neighbors=5)
    knn.fit(train)
    naive_imputer = MeanModeImputer(NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS)
    naive_imputer.fit(train)
    cells = _cells(test, [(row, column) for row in range(0, 20, 3) for column in ("weight", "shape")])

    metrics = score_baselines([naive_imputer, knn], test, cells, naive)

    assert metrics["baseline/knn5/impute_score"] < 0.5
    assert metrics["baseline/mean_mode/impute_score"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["baseline/knn5/rmse_num_z"] < metrics["baseline/mean_mode/rmse_num_z"]
    assert metrics["baseline/knn5/acc_cat"] > metrics["baseline/mean_mode/acc_cat"]


def _fitted(train: pd.DataFrame) -> list:
    imputers = [
        MeanModeImputer(NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS),
        KnnImputer(NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, n_neighbors=5),
    ]
    for imputer in imputers:
        imputer.fit(train)
    return imputers


def test_cells_are_hidden_by_a_mask_and_the_frame_is_never_rewritten() -> None:
    """An integer-coded category (electricity's ``day``) must come back as ``"3"``, never
    ``"3.0"``: writing NaN into such a column to hide a cell would upcast it to float, and
    every categorical guess would then miss the scored truth by its spelling alone.
    """
    frame = _frame()
    frame["colour"] = [1 + index % 7 for index in range(len(frame))]
    train, test = _split(frame)
    before = test.copy()
    naive = mean_mode_baselines(train, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS)
    cells = _cells(test, [(row, "colour") for row in range(0, 20, 2)] + [(1, "size")])
    assert set(cells.loc[cells["column"] == "colour", "actual"]) <= {str(v) for v in range(1, 8)}

    for imputer in _fitted(train):
        filled = imputer.impute(test, np.zeros(test.shape, dtype=bool) | (test.index.to_numpy()[:, None] % 2 == 0))
        assert set(filled["colour"]) <= {str(value) for value in range(1, 8)}, imputer.name
        metrics = score_baselines([imputer], test, cells, naive)
        assert metrics[f"baseline/{imputer.name}/acc_cat"] > 0.0, imputer.name

    assert test.equals(before)
    assert list(test.dtypes) == list(before.dtypes)


def test_the_scored_cells_table_is_left_untouched() -> None:
    """The same table goes on to the cell ledger and the per-column artifact with the
    model's own guesses in it; a baseline that overwrote ``imputed`` in place would be
    reported there as if it were the model.
    """
    train, test = _split(_frame())
    naive = mean_mode_baselines(train, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS)
    cells = _cells(test, [(0, "size"), (1, "colour"), (5, "shape"), (9, "weight")])
    cells["imputed"] = ["7.5", "purple", "hexagon", "-3.0"]
    before = cells.copy()

    score_baselines(_fitted(train), test, cells, naive)

    assert cells.equals(before)


def test_a_baseline_never_reads_the_truth_of_the_cell_it_fills() -> None:
    """A mask that failed to hide the cell would let KNN find its own value in the row and
    look flawless. Changing the truth at a hidden cell must not change the fill.
    """
    train, test = _split(_frame())
    naive = mean_mode_baselines(train, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS)
    positions = [(4, "weight"), (4, "shape")]
    cells = _cells(test, positions)
    imputers = _fitted(train)
    hidden = np.zeros(test.shape, dtype=bool)
    hidden[4, test.columns.get_loc("weight")] = True
    hidden[4, test.columns.get_loc("shape")] = True

    fills_before = [imputer.impute(test, hidden).loc[4, ["weight", "shape"]].tolist() for imputer in imputers]
    metrics_before = score_baselines(imputers, test, cells, naive)

    perturbed = test.copy()
    perturbed.loc[4, "weight"] = 1e6
    perturbed.loc[4, "shape"] = "hexagon"
    fills_after = [imputer.impute(perturbed, hidden).loc[4, ["weight", "shape"]].tolist() for imputer in imputers]
    perturbed_cells = _cells(perturbed, positions)
    metrics_after = score_baselines(imputers, perturbed, perturbed_cells, naive)

    assert fills_after == fills_before
    # The fills did not move, so only the truth did: every error grew or stayed.
    for imputer in imputers:
        assert metrics_after[f"baseline/{imputer.name}/rmse_num_z"] > metrics_before[f"baseline/{imputer.name}/rmse_num_z"]
        assert metrics_after[f"baseline/{imputer.name}/acc_cat"] <= metrics_before[f"baseline/{imputer.name}/acc_cat"]


def test_a_row_hidden_entirely_falls_back_to_the_training_mean_and_mode() -> None:
    """On an ``_80nan`` variant a fair share of rows have nothing left to measure a
    distance from. A NaN fill there would reach the fold's metrics and abort the whole
    cross-validation at the summariser, so the fallback is pinned: the training fold's
    mean for a number, its mode for a category.
    """
    train, test = _split(_frame())
    naive = mean_mode_baselines(train, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS)
    knn = KnnImputer(NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, n_neighbors=5)
    knn.fit(train)
    hidden = np.zeros(test.shape, dtype=bool)
    hidden[6, :] = True

    filled = knn.impute(test, hidden)

    for column in NUMERICAL_COLUMNS:
        assert filled.loc[6, column] == pytest.approx(naive[column])
    for column in CATEGORICAL_COLUMNS:
        assert filled.loc[6, column] == naive[column]
    cells = _cells(test, [(6, column) for column in test.columns])
    metrics = score_baselines([knn], test, cells, naive)
    assert all(np.isfinite(value) for value in metrics.values())


def test_an_all_missing_training_column_keeps_every_other_block_in_place() -> None:
    """scikit-learn drops a feature it saw no value for unless told otherwise, and every
    block after it would then be read as the one before. The fills of the other columns
    must be what a fit without the empty column gives, and the empty column itself must
    still come back finite.
    """
    train, test = _split(_frame())
    train_with_gap = train.copy()
    train_with_gap["weight"] = np.nan
    hidden = np.zeros(test.shape, dtype=bool)
    hidden[[1, 4, 9], test.columns.get_loc("size")] = True
    hidden[[2, 4, 11], test.columns.get_loc("shape")] = True
    hidden[[3, 4], test.columns.get_loc("weight")] = True

    with_gap = KnnImputer(NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, n_neighbors=5)
    with_gap.fit(train_with_gap)
    without = KnnImputer(["size"], CATEGORICAL_COLUMNS, n_neighbors=5)
    without.fit(train.drop(columns=["weight"]))

    filled = with_gap.impute(test, hidden)
    reference = without.impute(test.drop(columns=["weight"]), np.delete(hidden, test.columns.get_loc("weight"), axis=1))

    assert filled["size"].to_numpy() == pytest.approx(reference["size"].to_numpy())
    assert filled["shape"].tolist() == reference["shape"].tolist()
    assert filled["colour"].tolist() == reference["colour"].tolist()
    assert np.isfinite(filled["weight"].to_numpy(dtype=float)).all()


def test_an_all_missing_training_categorical_column_is_refused_clearly() -> None:
    """A category column with no observed value has an empty vocabulary, a zero-width
    block and nothing to decode; better to say which column than to fail inside argmax."""
    train, _ = _split(_frame())
    train["shape"] = np.nan

    with pytest.raises(ValueError, match="shape"):
        KnnImputer(NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, n_neighbors=5).fit(train)


def test_a_table_of_one_kind_produces_only_that_kinds_keys() -> None:
    """Six datasets are all numerical and ``kr-vs-kp`` is all categorical. The summariser
    requires every fold to carry the same keys, so a baseline must emit exactly the keys
    the model emits for that kind, and never the counts the model already reports.
    """
    frame = _frame()
    train, test = _split(frame)

    numerical_only = [MeanModeImputer(NUMERICAL_COLUMNS, []), KnnImputer(NUMERICAL_COLUMNS, [], n_neighbors=5)]
    for imputer in numerical_only:
        imputer.fit(train[NUMERICAL_COLUMNS])
    metrics = score_baselines(
        numerical_only,
        test[NUMERICAL_COLUMNS],
        _cells(test, [(0, "size"), (2, "weight")]),
        mean_mode_baselines(train, NUMERICAL_COLUMNS, []),
    )
    assert set(metrics) == {
        f"baseline/{name}/{metric}"
        for name in ("mean_mode", "knn5")
        for metric in ("rmse_num_z", "mae_num_z", "impute_score")
    }

    categorical_only = [MeanModeImputer([], CATEGORICAL_COLUMNS), KnnImputer([], CATEGORICAL_COLUMNS, n_neighbors=5)]
    for imputer in categorical_only:
        imputer.fit(train[CATEGORICAL_COLUMNS])
    metrics = score_baselines(
        categorical_only,
        test[CATEGORICAL_COLUMNS],
        _cells(test, [(0, "colour"), (2, "shape")]),
        mean_mode_baselines(train, [], CATEGORICAL_COLUMNS),
    )
    assert set(metrics) == {
        f"baseline/{name}/{metric}"
        for name in ("mean_mode", "knn5")
        for metric in ("acc_cat", "macro_f1_cat", "impute_score")
    }


def test_scoring_baselines_leaves_both_global_random_streams_untouched() -> None:
    """Every seeded classification and imputation result depends on the global draw
    sequence; the regression fixtures pin it. A baseline that consumed from either
    stream would move numbers that have nothing to do with it.
    """
    train, test = _split(_frame())
    naive = mean_mode_baselines(train, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS)
    cells = _cells(test, [(0, "size"), (1, "colour"), (5, "shape"), (9, "weight")])
    imputers = _fitted(train)

    np.random.seed(7)
    torch.manual_seed(7)
    expected = (np.random.rand(3).tolist(), torch.rand(3).tolist())
    np.random.seed(7)
    torch.manual_seed(7)
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    score_baselines(imputers, test, cells, naive)

    numpy_after = np.random.get_state()
    assert numpy_after[1].tolist() == numpy_before[1].tolist() and numpy_after[2] == numpy_before[2]
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert (np.random.rand(3).tolist(), torch.rand(3).tolist()) == expected


def test_a_category_only_the_truth_holds_is_a_miss_not_a_crash() -> None:
    """``kr-vs-kp``'s ``spcop`` is ``f`` everywhere but once, and that one ``t`` can sit in
    the test fold or in the sibling's truth. Neither must abort the fold: a category the
    training fold never showed is simply a wrong answer, for the baselines as for the
    model.
    """
    train, test = _split(_frame())
    test.loc[3, "colour"] = "violet"  # observed in the test fold, absent from training
    naive = mean_mode_baselines(train, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS)
    cells = _cells(test, [(0, "colour"), (3, "size")])
    cells.loc[cells["column"] == "colour", ["actual", "actual_original"]] = "violet"

    metrics = score_baselines(_fitted(train), test, cells, naive)

    for name in ("mean_mode", "knn5"):
        assert metrics[f"baseline/{name}/acc_cat"] == 0.0
        assert np.isfinite(metrics[f"baseline/{name}/impute_score"])


class _Spy:
    name = "spy"

    def __init__(self) -> None:
        self.calls = 0

    def fit(self, frame: pd.DataFrame) -> None:
        pass

    def impute(self, frame: pd.DataFrame, hidden: np.ndarray) -> pd.DataFrame:
        self.calls += 1
        return frame


def test_an_empty_population_scores_nothing_and_runs_no_imputer() -> None:
    """No cell, no metric: the model reports no score either, so an empty key set keeps
    the folds' keys aligned, and the KNN transform (a minute on electricity) is not paid
    for nothing.
    """
    _, test = _split(_frame())
    spy = _Spy()

    assert score_baselines([spy], test, _cells(test, []), {}) == {}
    assert spy.calls == 0


def test_each_knn_baseline_fills_from_its_own_number_of_neighbours() -> None:
    """Two KNN baselines at different neighbourhood sizes must be two different bars; a
    size that never reached scikit-learn would log the same numbers under two names.

    Worked by hand: ``y`` equals ``x`` on training rows ``x = 1..20``, and the hidden
    ``y`` of a test row at ``x = 0`` is filled from its nearest training rows by ``x``
    alone. The five nearest are ``x = 1..5``, whose mean is 3.0; the ten nearest are
    ``x = 1..10``, whose mean is 5.5.
    """
    train = pd.DataFrame({"x": [float(v) for v in range(1, 21)], "y": [float(v) for v in range(1, 21)]})
    test = pd.DataFrame({"x": [0.0], "y": [123.0]})
    hidden = np.array([[False, True]])

    fills = {}
    for k in (5, 10):
        imputer = KnnImputer(["x", "y"], [], n_neighbors=k)
        imputer.fit(train)
        fills[imputer.name] = imputer.impute(test, hidden).loc[0, "y"]

    assert fills == {"knn5": pytest.approx(3.0), "knn10": pytest.approx(5.5)}


def test_the_fitted_baselines_are_mean_mode_and_both_knn_sizes() -> None:
    """What every scored population carries: the naive fill, and KNN at five and at ten
    neighbours, each under its own name."""
    train, _ = _split(_frame())

    imputers = fit_baseline_imputers(train, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS)

    assert [imputer.name for imputer in imputers] == ["mean_mode", "knn5", "knn10"]
