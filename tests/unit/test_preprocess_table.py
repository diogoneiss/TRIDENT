"""The masking primitive, whose draw sequence is protected behaviour.

`AGENTS.md` forbids moving the seeded random-draw sequence, and `preprocess_table` is
where that sequence is consumed: once for the dynamic mask matrix, then once per row that
the matrix left unmasked. Every run ever published went through it, so a change here moves
published results even when nothing else in the pipeline is touched.
"""

import numpy as np
import pandas as pd

from src.utils import preprocess_table


def _frame() -> pd.DataFrame:
    """Small, but shaped like the cliff: at a low `p_base` most rows need the fallback."""
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        rng.integers(0, 100, size=(24, 5)).astype(float), columns=list("abcde")
    )
    return frame.mask(rng.random((24, 5)) < 0.25)


def _mask_positions(frame: pd.DataFrame) -> list[tuple[int, str]]:
    return [
        (row, column)
        for row in range(len(frame))
        for column in frame.columns
        if frame.iloc[row][column] == "[MASK]"
    ]


# Captured from the implementation as it stood before backlog P1 was addressed, which is
# the baseline every published run used. It is a fixed literal, never recomputed the way
# the code computes it, so an optimisation that moves a single draw fails this test.
_PROTECTED_DRAW = [
    (0, "a"), (1, "d"), (2, "d"), (3, "a"), (4, "c"), (5, "d"), (6, "e"), (7, "a"),
    (8, "b"), (9, "b"), (10, "a"), (11, "d"), (12, "e"), (13, "e"), (14, "b"),
    (14, "d"), (15, "c"), (16, "c"), (17, "a"), (18, "a"), (19, "a"), (20, "b"),
    (21, "a"), (22, "b"), (23, "b"),
]


def test_the_seeded_mask_draw_is_exactly_what_it_has_always_been() -> None:
    """P1's speed-up must not cost a single draw.

    The backlog's suggested fix -- one `argmax` over a random matrix -- would be faster
    still but would redraw the fallback, moving every seeded result and the protected
    `vehicle_00nan` baseline with them. Hoisting the row lookup out of the loop keeps the
    `np.random.choice` calls identical in count, order and argument, and this pins that.
    """
    np.random.seed(99)

    masked = preprocess_table(_frame(), p_base=0.05)

    assert _mask_positions(masked) == _PROTECTED_DRAW


def test_an_already_missing_cell_is_never_chosen_as_the_row_guarantee() -> None:
    """The fallback picks among a row's observed cells, so a gap cannot be masked twice.

    `[NULL]` and `[MASK]` mean different things to the model, and a cell that is already
    missing carries no value to hide, so the guarantee has to skip it. A row that is
    entirely missing simply gets nothing.
    """
    frame = pd.DataFrame(
        {"a": [1.0, np.nan], "b": [2.0, np.nan], "c": [np.nan, np.nan]}
    )

    np.random.seed(5)
    masked = preprocess_table(frame, p_base=0.0)

    assert masked.loc[0, "c"] == "[NULL]"
    assert (masked.loc[1] == "[NULL]").all()
    # p_base is zero, so the only mask that can appear is the per-row guarantee.
    assert len(_mask_positions(masked)) == 1
