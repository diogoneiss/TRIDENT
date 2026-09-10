"""Imputation error and the fold-ranking score (ADR 0004, decision 6)."""

import numpy as np
import pandas as pd
import pytest

from src.training.imputation_metrics import mean_mode_baselines, score_cells


def _cells(rows: list[tuple[str, str, object, object]]) -> pd.DataFrame:
    """A scored-cell table: which column, which kind, the truth, and the guess."""
    return pd.DataFrame(rows, columns=["column", "kind", "actual", "imputed"])


def test_a_baseline_is_the_training_folds_own_mean_or_mode() -> None:
    """What a naive imputer would fill a cell with, learned from the training fold alone.

    Missing training cells are not values, so they cannot pull the mean or win a vote.
    """
    training_fold = pd.DataFrame(
        {
            "amount": [1.0, 2.0, 6.0, np.nan],
            "grade": ["a", "b", "b", np.nan],
        }
    )

    baselines = mean_mode_baselines(training_fold, ["amount"], ["grade"])

    assert baselines == {"amount": 3.0, "grade": "b"}


def test_error_is_pooled_over_the_scored_cells_of_each_kind() -> None:
    """Numbers are scored by how far off they were, categories by whether they were right."""
    cells = _cells(
        [
            ("amount", "numerical", 1.0, 2.0),  # off by 1
            ("amount", "numerical", 10.0, 3.0),  # off by 7
            ("grade", "categorical", "a", "a"),  # right
            ("grade", "categorical", "b", "a"),  # wrong
            ("grade", "categorical", "b", "b"),  # right
        ]
    )

    scores = score_cells(cells, baselines={"amount": 0.0, "grade": "a"})

    assert scores.metrics["rmse_num_z"] == pytest.approx(5.0)  # sqrt(mean(1, 49))
    assert scores.metrics["mae_num_z"] == pytest.approx(4.0)  # mean(1, 7)
    assert scores.metrics["acc_cat"] == pytest.approx(2 / 3)
    assert scores.metrics["n_num_cells"] == 2
    assert scores.metrics["n_cat_cells"] == 3


def test_the_ranking_score_measures_how_much_better_than_naive_the_model_was() -> None:
    """One number, comparable across datasets: the share of the naive imputer's error left.

    Each kind's error is divided by what filling the mean or the mode would have scored
    on the very same cells, then the two are weighted by how many cells of each kind were
    scored. 1.0 means no better than naive; below 1.0 means better.
    """
    cells = _cells(
        [
            # Model is off by 1 on both; filling 0.0 would be off by 1 and by 7.
            ("amount", "numerical", 1.0, 2.0),
            ("amount", "numerical", 7.0, 6.0),
            # Model gets 3 of 4; filling "a" would get 2 of 4.
            ("grade", "categorical", "a", "a"),
            ("grade", "categorical", "a", "a"),
            ("grade", "categorical", "b", "b"),
            ("grade", "categorical", "b", "a"),
        ]
    )

    scores = score_cells(cells, baselines={"amount": 0.0, "grade": "a"})

    # numerical 1.0/5.0 = 0.2 over 2 cells, categorical 0.25/0.5 = 0.5 over 4 cells
    # (2/6) * 0.2 + (4/6) * 0.5 = 0.4
    assert scores.metrics["impute_score"] == pytest.approx(0.4)


def test_a_column_the_naive_imputer_never_gets_wrong_still_ranks() -> None:
    """A constant column leaves no error to reduce, so matching it is parity, not a win.

    Dividing by the naive error would divide by zero here. Scoring the model as perfect
    would claim it beat a flawless baseline, which is not something anyone can do.
    """
    everyone_agrees = _cells(
        [("grade", "categorical", "a", "a"), ("grade", "categorical", "a", "a")]
    )
    model_gets_one_wrong = _cells(
        [("grade", "categorical", "a", "a"), ("grade", "categorical", "a", "b")]
    )

    matched = score_cells(everyone_agrees, baselines={"grade": "a"})
    fell_short = score_cells(model_gets_one_wrong, baselines={"grade": "a"})

    assert matched.metrics["impute_score"] == pytest.approx(1.0)
    assert fell_short.metrics["impute_score"] == pytest.approx(1.5)


def test_a_table_of_one_kind_of_column_is_scored_by_that_kind_alone() -> None:
    """Six of the nine datasets are all numerical and one is all categorical.

    The absent kind contributes no cells, so it takes no share of the score rather than
    contributing an undefined term.
    """
    only_numbers = _cells(
        [("amount", "numerical", 1.0, 2.0), ("amount", "numerical", 7.0, 6.0)]
    )
    only_categories = _cells(
        [
            ("grade", "categorical", "a", "a"),
            ("grade", "categorical", "a", "a"),
            ("grade", "categorical", "b", "b"),
            ("grade", "categorical", "b", "a"),
        ]
    )
    baselines = {"amount": 0.0, "grade": "a"}

    numbers = score_cells(only_numbers, baselines)
    categories = score_cells(only_categories, baselines)

    assert numbers.metrics["impute_score"] == pytest.approx(0.2)  # 1.0 / 5.0
    assert "acc_cat" not in numbers.metrics
    assert categories.metrics["impute_score"] == pytest.approx(0.5)  # 0.25 / 0.5
    assert "rmse_num_z" not in categories.metrics


def test_a_category_the_model_never_reaches_for_scores_zero_rather_than_nothing() -> None:
    """Macro-F1 per column, averaged. At high missingness a rare class can vanish.

    Accuracy alone would flatter a model that always answers with the common category,
    so macro-F1 is reported beside it. A class the model never predicts has no precision
    to speak of; it scores zero rather than making the whole column undefined.
    """
    cells = _cells(
        [
            # "b" is never reached for: F1 is 0.8 for "a" and 0.0 for "b", so 0.4.
            ("grade", "categorical", "a", "a"),
            ("grade", "categorical", "a", "a"),
            ("grade", "categorical", "b", "a"),
            # every cell right, so 1.0
            ("tier", "categorical", "x", "x"),
            ("tier", "categorical", "y", "y"),
        ]
    )

    scores = score_cells(cells, baselines={"grade": "a", "tier": "x"})

    assert scores.metrics["macro_f1_cat"] == pytest.approx(0.7)  # mean(0.4, 1.0)


def test_the_baseline_comes_from_training_not_from_the_cells_being_scored() -> None:
    """Reading the baseline off the scored cells would let the test set set its own bar.

    Here the scored cells are mostly "b" while the training fold said "a". Scoring
    against the training fold's answer, the naive imputer is wrong twice in three and
    the model wrong once, so the model leaves half the naive error. Had the baseline been
    read off these cells it would have been "b", wrong once in three, and the model would
    have scored exactly parity while doing just as well.
    """
    cells = _cells(
        [
            ("grade", "categorical", "b", "b"),
            ("grade", "categorical", "b", "b"),
            ("grade", "categorical", "a", "b"),
        ]
    )

    scores = score_cells(cells, baselines={"grade": "a"})

    assert scores.metrics["impute_score"] == pytest.approx(0.5)


def test_each_column_reports_its_own_error_beside_the_pooled_one() -> None:
    """Pooled numbers rank folds; per-column numbers say which column is the problem."""
    cells = _cells(
        [
            ("amount", "numerical", 1.0, 2.0),  # off by 1
            ("amount", "numerical", 7.0, 0.0),  # off by 7, so rmse 5.0
            ("fee", "numerical", 0.0, 2.0),  # off by 2
            ("fee", "numerical", 10.0, 8.0),  # off by 2, so rmse 2.0
            ("grade", "categorical", "a", "a"),
            ("grade", "categorical", "b", "a"),  # right once in two
        ]
    )

    scores = score_cells(cells, baselines={"amount": 0.0, "fee": 0.0, "grade": "a"})

    per_column = scores.per_column.set_index(["column", "metric"])["value"]
    assert per_column[("amount", "rmse_num_z")] == pytest.approx(5.0)
    assert per_column[("fee", "rmse_num_z")] == pytest.approx(2.0)
    assert per_column[("grade", "acc_cat")] == pytest.approx(0.5)
