"""E136's label-scarce fine-tuning (``FINETUNE_LABEL_FRACTION``): a fixed stratified share of the training rows."""

import numpy as np
import pytest

from src.training.finetuning import labelled_rows
from src.training.types import Hyperparameters


def test_fraction_one_keeps_every_row_in_order() -> None:
    labels = np.array([0, 1, 0, 1, 2, 2])
    assert np.array_equal(labelled_rows(labels, 1.0), np.arange(6))


def test_the_draw_is_fixed_by_the_rows_alone() -> None:
    labels = np.random.default_rng(1).choice(3, 300)
    a, b = labelled_rows(labels, 0.1), labelled_rows(labels, 0.1)
    assert np.array_equal(a, b)
    assert len(a) == 30 and np.array_equal(a, np.sort(a))


def test_every_class_survives_a_tenth_of_a_small_table() -> None:
    labels = np.array([0] * 300 + [1] * 60 + [2] * 10)  # kc2-like minority shares
    kept = labelled_rows(labels, 0.1)
    assert set(np.unique(labels[kept])) == {0, 1, 2}
    assert len(kept) == 37


def test_the_flag_defaults_to_everything_and_is_validated() -> None:
    assert Hyperparameters.from_mapping({}).finetune_label_fraction == 1.0
    assert Hyperparameters.from_mapping({"FINETUNE_LABEL_FRACTION": 0.25}).finetune_label_fraction == 0.25
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            Hyperparameters.from_mapping({"FINETUNE_LABEL_FRACTION": bad})


def test_scarce_validation_defaults_off_and_is_validated() -> None:
    assert Hyperparameters.from_mapping({}).finetune_scarce_validation == "off"
    assert Hyperparameters.from_mapping({"FINETUNE_SCARCE_VALIDATION": "on"}).finetune_scarce_validation == "on"
    with pytest.raises(ValueError):
        Hyperparameters.from_mapping({"FINETUNE_SCARCE_VALIDATION": "yes"})
