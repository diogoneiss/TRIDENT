"""The shared report reading (T21): which column is K, what the bar is, and the verdict rule."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "experiments"))

import report_common as common  # noqa: E402

P = "impute/induced"


def _run(checkpoint: str) -> common.Run:
    return common.Run("toy_60nan", 1, "file:///nowhere", checkpoint)


def _frame(**columns: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"fold": [1, 2, 3], **columns})


def test_k_is_the_induced_checkpoint_family_on_a_loss_run_and_the_headline_on_an_induced_one() -> None:
    """E34/E35 kept the loss's epoch, so their headline is H and K sits in ``induced_checkpoint``;
    since ADR 0015 the headline is K. Reading the headline as K on E34 would count H (16 of 21 by
    means instead of 19)."""
    loss_run = _frame(**{f"{P}/impute_score": [0.9] * 3, f"{P}/induced_checkpoint/impute_score": [0.8] * 3})
    induced_run = _frame(**{f"{P}/impute_score": [0.8] * 3, f"{P}/loss_checkpoint/impute_score": [0.9] * 3})

    assert common.model(loss_run, _run("loss"), "K").tolist() == [0.8] * 3
    assert common.model(loss_run, _run("loss"), "H").tolist() == [0.9] * 3
    assert common.model(induced_run, _run("induced"), "K").tolist() == [0.8] * 3
    assert common.model(induced_run, _run("induced"), "H").tolist() == [0.9] * 3
    with pytest.raises(KeyError, match="no M readout"):
        common.model(induced_run, _run("induced"), "M")
    with pytest.raises(KeyError, match="no K readout"):
        common.model(_frame(**{f"{P}/impute_score": [0.9] * 3}), _run("loss"), "K")


def test_on_a_calibrated_run_k_is_the_uncalibrated_family_and_kc_the_headline() -> None:
    """ADR 0027 makes the calibrated output the headline at 40% missing and above. Reading that
    headline as K would credit K with KC's gain; older runs keep KC in their diagnostic."""
    calibrated = _frame(**{
        f"{P}/impute_score": [0.7] * 3, f"{P}/uncalibrated/impute_score": [0.8] * 3,
        f"{P}/loss_checkpoint/impute_score": [0.9] * 3,
    })
    older = _frame(**{f"{P}/impute_score": [0.8] * 3, f"{P}/calibrated/impute_score": [0.7] * 3})

    assert common.model(calibrated, _run("induced"), "K").tolist() == [0.8] * 3
    assert common.model(calibrated, _run("induced"), "KC").tolist() == [0.7] * 3
    assert common.model(calibrated, _run("induced"), "H").tolist() == [0.9] * 3
    assert common.model(older, _run("induced"), "K").tolist() == [0.8] * 3
    assert common.model(older, _run("induced"), "KC").tolist() == [0.7] * 3
    with pytest.raises(KeyError, match="no KC readout"):
        common.model(_frame(**{f"{P}/impute_score": [0.8] * 3}), _run("induced"), "KC")


def test_the_bar_never_takes_an_external_or_withdrawn_baseline_however_low() -> None:
    frame = _frame(**{
        f"{P}/baseline/mean_mode/impute_score": [1.0] * 3,
        f"{P}/baseline/hgb/impute_score": [0.7, 0.8, 0.9],
        f"{P}/baseline/lgbm/impute_score": [0.75] * 3,
        f"{P}/baseline/missforest/impute_score": [0.5] * 3,
        f"{P}/baseline/missforest_lgbm/impute_score": [0.4] * 3,
        f"{P}/baseline/best/impute_score": [0.3] * 3,
        f"{P}/baseline/condexp/impute_score": [0.6] * 3,
    })
    assert common.opponent(frame, "bar")[0] == "lgbm"  # 0.75 against hgb's mean 0.8
    assert common.opponent(frame, "strong")[0] == "missforest"
    assert common.opponent(frame, "best_of_all")[0] == "missforest"

    # E41's diagnostic arm and E42's tuned families sit beside them in merged frames; neither is
    # the untuned bar, however low (the first E41 reading let missforest_seq into it).
    merged = frame.assign(**{
        f"{P}/baseline/missforest_seq/impute_score": [0.2] * 3,
        f"{P}/baseline/hgb_tuned/impute_score": [0.1] * 3,
        f"{P}/baseline/tuned_bar/impute_score": [0.1] * 3,
    })
    assert common.opponent(merged, "bar")[0] == "lgbm"
    assert common.opponent(merged, "best_of_all")[0] == "missforest"
    with pytest.raises(KeyError, match="no class names"):
        common.opponent(merged.assign(**{f"{P}/baseline/mystery/impute_score": [0.9] * 3}), "bar")


def test_the_per_type_bar_is_the_bar_on_a_table_of_one_kind_and_no_weaker_on_a_mixed_one() -> None:
    one_kind = _frame(**{
        f"{P}/n_num_cells": [10.0] * 3,
        f"{P}/baseline/mean_mode/rmse_num_z": [1.0] * 3, f"{P}/baseline/mean_mode/impute_score": [1.0] * 3,
        f"{P}/baseline/hgb/rmse_num_z": [0.6] * 3, f"{P}/baseline/hgb/impute_score": [0.6] * 3,
        f"{P}/baseline/knn5/rmse_num_z": [0.7] * 3, f"{P}/baseline/knn5/impute_score": [0.7] * 3,
    })
    assert common.per_type_bar(one_kind) == pytest.approx(common.opponent(one_kind, "bar")[1])

    # hgb wins the numbers, knn5 the categories; each alone scores 0.75, the per-type bar 0.5.
    mixed = _frame(**{
        f"{P}/n_num_cells": [10.0] * 3, f"{P}/n_cat_cells": [10.0] * 3,
        f"{P}/baseline/mean_mode/rmse_num_z": [1.0] * 3, f"{P}/baseline/mean_mode/acc_cat": [0.5] * 3,
        f"{P}/baseline/mean_mode/impute_score": [1.0] * 3,
        f"{P}/baseline/hgb/rmse_num_z": [0.5] * 3, f"{P}/baseline/hgb/acc_cat": [0.5] * 3,
        f"{P}/baseline/hgb/impute_score": [0.75] * 3,
        f"{P}/baseline/knn5/rmse_num_z": [1.0] * 3, f"{P}/baseline/knn5/acc_cat": [0.75] * 3,
        f"{P}/baseline/knn5/impute_score": [0.75] * 3,
    })
    assert common.per_type_bar(mixed) == pytest.approx([0.5] * 3)


def test_the_verdict_needs_the_interval_and_every_seed_and_equivalence_needs_the_margin() -> None:
    better = [np.array([-0.02, -0.03, -0.025, -0.02, -0.03])] * 3
    assert common.verdict(better).call == "better"
    split = [np.array([-0.02] * 5), np.array([-0.03] * 5), np.array([0.001, 0.002, 0.0, 0.001, 0.001])]
    assert common.verdict(split).call == "none"
    small = [np.array([0.001, -0.001, 0.0005, -0.0005, 0.0])] * 3
    assert common.verdict(small).call == "none"
    assert common.verdict(small, sesoi=0.01).call == "equivalent"
    assert common.tally(["better", "better", "none", "worse"]) == "better 2 / no difference 1 / worse 1 (fold-pair t, seed agreement)"


def test_tables_are_tested_by_their_mean_gap_and_holm_steps_down() -> None:
    tables, below, p = common.across_tables({f"t{i}_20nan": -0.05 - i / 100 for i in range(9)})
    assert (tables, below) == (9, 9)
    assert p == pytest.approx(2 / 2**9)
    assert common.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_the_development_cells_follow_the_decisions_since_e34() -> None:
    assert common.development_source("pendigits_20nan", "101/202/303") == common.Source("architecture-large-2026-10-06", "ALL")
    assert common.development_source("kr-vs-kp_40nan", "101/202/303") == common.Source("retune-2026-10-15", "D")
    assert common.development_source("vehicle_20nan", "101/202/303") == common.Source("selection-2026-10-03", None)
    assert common.development_source("letter_20nan", "42/7/13").labelled
    assert not common.development_source("vehicle_20nan", "42/7/13").labelled
