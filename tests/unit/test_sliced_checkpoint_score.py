"""``--decode_metric sw_score`` (imputation-protocol/72): the validation scorer's sliced Wasserstein score."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.training.decoding import _ValidationGapScore, _sliced_rows
from src.training.types import validation_objective_key

N_ROWS = 40


def _scorer() -> tuple[_ValidationGapScore, SimpleNamespace, SimpleNamespace]:
    """A scorer over two numerical columns and one categorical (three categories), built by hand: the first 30 rows
    hide their first number and their category; row 0's true category is one the variant never shows (id -1)."""
    rng = np.random.default_rng(3)
    observed = torch.tensor(rng.normal(size=(2, N_ROWS)), dtype=torch.float32)
    observed[0, :30] = float("nan")  # a placeholder at the gaps
    truth_numbers = torch.tensor(rng.normal(size=30), dtype=torch.float64)
    truth_categories = torch.tensor(rng.integers(0, 3, size=30), dtype=torch.long)
    truth_categories[0] = -1
    rows = torch.arange(30)
    scorer = _ValidationGapScore.__new__(_ValidationGapScore)
    scorer._hidden = SimpleNamespace(num_values=observed, cat_indices=torch.full((1, N_ROWS), 7, dtype=torch.long),
                                     masked_positions=torch.zeros((N_ROWS, 3), dtype=torch.bool))
    scorer._numerical = [(0, rows, truth_numbers)]
    scorer._categorical = [(0, rows, truth_categories)]
    scorer._n_numerical = scorer._n_categorical = 30
    embedder = SimpleNamespace(numerical_columns=["a", "b"], categorical_columns=["c"],
                               label_encoders={"c": SimpleNamespace(classes_=np.array(["x", "y", "z"]))})
    scorer._prepare_distribution(embedder, None, {"a": 0.25, "b": 0.0, "c": "y"}, "sw_score")
    truth = SimpleNamespace(numerical_values=torch.zeros((2, N_ROWS)), categorical_ids=torch.zeros((1, N_ROWS), dtype=torch.long))
    truth.numerical_values[0, :30] = truth_numbers.float()
    truth.categorical_ids[0, :30] = truth_categories
    naive = SimpleNamespace(numerical_values=torch.full((2, N_ROWS), 0.25), categorical_ids=torch.ones((1, N_ROWS), dtype=torch.long))
    return scorer, truth, naive


def test_the_truth_scores_zero_and_the_mean_mode_fill_one() -> None:
    scorer, truth, naive = _scorer()
    # float32 predictions against float64 truths: zero up to the cast
    assert scorer._sliced_score(truth) == pytest.approx(0.0, abs=1e-6)
    assert scorer._sliced_score(naive) == pytest.approx(1.0)


def test_the_score_is_the_mean_gap_between_sorted_projections() -> None:
    scorer, _, naive = _scorer()
    numbers, categories = scorer._sw_numbers.clone(), scorer._sw_categories.clone()
    numbers[0, :30] = 0.25
    categories[0, :30] = 1
    filled = _sliced_rows(numbers, categories, [3], scorer._sw_rows).numpy()
    true = _sliced_rows(scorer._sw_numbers, scorer._sw_categories, [3], scorer._sw_rows).numpy()
    along = scorer._sw_along.numpy()
    by_hand = float(np.mean(np.abs(np.sort(filled @ along, axis=0) - np.sort(true @ along, axis=0))))
    assert by_hand > 0
    assert scorer._sliced_distance(numbers, categories) == pytest.approx(by_hand)
    assert scorer._sw_naive == pytest.approx(by_hand)
    assert naive.categorical_ids[0, 0] == 1


def test_a_row_is_its_numbers_and_one_hot_categories_with_a_slot_for_ids_outside_the_vocabulary() -> None:
    rows = _sliced_rows(torch.tensor([[1.0, 2.0]]), torch.tensor([[2, -1]]), [3], torch.tensor([0, 1]))
    assert rows.shape == (2, 5)
    assert rows[0].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0, 2**-0.5])
    assert rows[1].tolist() == pytest.approx([2.0, 2**-0.5, 0.0, 0.0, 0.0])


def test_a_search_cannot_rank_by_a_score_no_run_logs() -> None:
    with pytest.raises(ValueError, match="ranks checkpoints only"):
        validation_objective_key("induced", "sw_score")
