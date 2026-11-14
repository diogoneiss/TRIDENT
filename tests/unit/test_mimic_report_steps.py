"""E180's reading of a variant from its three calls (docs/tickets/mimic/17-wide-table-matched-steps.md)."""

import pytest

from scripts.experiments.mimic_report import loss_belongs_to


@pytest.mark.parametrize(
    ("capped_full", "steps_capped", "steps_full", "reading"),
    [
        ("worse", "better", "equivalent", "steps"),
        ("worse", "better", "better", "steps"),
        ("worse", "better", "worse", "both"),
        ("worse", "equivalent", "worse", "rows"),
        ("worse", "worse", "worse", "rows"),
        ("worse", "none", "worse", "unclear"),
        ("worse", "better", "none", "unclear"),
        ("none", "better", "equivalent", "no loss to split"),
    ],
)
def test_the_capped_arms_loss_is_read_from_the_three_calls(capped_full: str, steps_capped: str, steps_full: str, reading: str) -> None:
    assert loss_belongs_to(capped_full, steps_capped, steps_full) == reading
