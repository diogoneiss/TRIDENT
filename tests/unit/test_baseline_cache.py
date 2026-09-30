"""Caching the baseline imputers' scores across runs that share a fold (ADR 0007, 2026-09-30).

The baselines depend only on the fold's rows and the cells scored on them, never on the
model, so every arm of a study and every trial of a search computes the same numbers. The
cache has to be invisible: a hit must be exactly what a computation gives, and anything that
changes the computation must miss.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.training import baseline_cache
from src.training.baseline_cache import BaselineScorer
from src.training.imputation_baselines import fit_baseline_imputers, score_baselines
from src.training.imputation_metrics import CATEGORICAL, NUMERICAL, mean_mode_baselines

NUMERICAL_COLUMNS = ["size", "weight"]
CATEGORICAL_COLUMNS = ["colour"]


def _frame(rows: int, offset: int = 0) -> pd.DataFrame:
    index = np.arange(offset, offset + rows)
    return pd.DataFrame(
        {
            "size": (index % 9).astype(float) - 4.0,
            "colour": np.array(["red", "blue", "green"])[index % 3],
            "weight": ((index * 7) % 5).astype(float) / 2.0,
        }
    )


def _cells(test: pd.DataFrame) -> pd.DataFrame:
    """Every third row's size and colour, with the truth the test frame holds."""
    records = []
    for row in range(0, len(test), 3):
        records.append({"row": row, "column": "size", "kind": NUMERICAL, "actual": float(test.at[row, "size"]), "imputed": 0.0})
        records.append({"row": row, "column": "colour", "kind": CATEGORICAL, "actual": str(test.at[row, "colour"]), "imputed": "red"})
    return pd.DataFrame(records)


def _direct(train: pd.DataFrame, test: pd.DataFrame, cells: pd.DataFrame):
    baselines = mean_mode_baselines(train, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS)
    imputers = fit_baseline_imputers(train, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS)
    return score_baselines(imputers, test, cells, baselines)


def _scorer(train: pd.DataFrame, cache_dir: Path | None) -> BaselineScorer:
    return BaselineScorer(train, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, cache_dir)


def _hits(train: pd.DataFrame, cache_dir: Path, test: pd.DataFrame, cells: pd.DataFrame, baselines) -> int:
    """How many of a fresh scorer's lookups the cache answered, after scoring once."""
    scorer = _scorer(train, cache_dir)
    scorer.score(test, cells, baselines)
    return scorer.hits


def test_a_cached_score_is_exactly_the_computed_one(tmp_path) -> None:
    train, test = _frame(60), _frame(30, offset=60)
    cells = _cells(test)
    baselines = mean_mode_baselines(train, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS)
    expected = _direct(train, test, cells)

    first = _scorer(train, tmp_path)
    computed = first.score(test, cells, baselines)
    second = _scorer(train, tmp_path)
    cached = second.score(test, cells, baselines)

    assert (first.hits, second.hits) == (0, 1)
    for scores in (computed, cached):
        assert scores.metrics.keys() == expected.metrics.keys()
        assert all(scores.metrics[key] == expected.metrics[key] for key in expected.metrics)
        pdt.assert_frame_equal(scores.per_column, expected.per_column)


def test_anything_that_changes_the_computation_misses(tmp_path) -> None:
    """A different truth, a different training split, or another machine, whose hgb can
    differ in the last digits (the check cell of 2026-09-29), is a different computation."""
    train, test = _frame(60), _frame(30, offset=60)
    cells = _cells(test)
    baselines = mean_mode_baselines(train, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS)
    _scorer(train, tmp_path).score(test, cells, baselines)

    other_truth = cells.copy()
    other_truth.loc[0, "actual"] = 99.0
    other_train = _frame(61)

    hits = [
        _hits(train, tmp_path, test, other_truth, baselines),
        _hits(other_train, tmp_path, test, cells, baselines),
    ]
    original = baseline_cache.socket.gethostname
    baseline_cache.socket.gethostname = lambda: "another-machine"
    try:
        hits.append(_hits(train, tmp_path, test, cells, baselines))
    finally:
        baseline_cache.socket.gethostname = original

    assert hits == [0, 0, 0]
    assert _hits(train, tmp_path, test, cells, baselines) == 1


def test_the_model_s_own_guesses_are_not_part_of_the_key(tmp_path) -> None:
    """Every arm fills the cells differently; the baselines must still be shared."""
    train, test = _frame(60), _frame(30, offset=60)
    cells = _cells(test)
    baselines = mean_mode_baselines(train, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS)
    _scorer(train, tmp_path).score(test, cells, baselines)

    other_guesses = cells.assign(imputed=cells["imputed"].map(lambda v: 1.0 if isinstance(v, float) else "blue"))

    assert _hits(train, tmp_path, test, other_guesses, baselines) == 1


def test_without_a_cache_directory_nothing_is_written_or_read(tmp_path) -> None:
    train, test = _frame(60), _frame(30, offset=60)
    cells = _cells(test)
    baselines = mean_mode_baselines(train, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS)

    scorer = _scorer(train, None)
    scores = scorer.score(test, cells, baselines)

    assert scorer.hits == 0
    assert list(tmp_path.iterdir()) == []
    assert scores.metrics == _direct(train, test, cells).metrics
