"""C15's seedable generators: matched counts, per-column streams, the outcome tilt, the noise control, the guard."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "datasets"))

import mnar_family  # noqa: E402


def _complete(rows: int = 2000) -> pd.DataFrame:
    stream = np.random.default_rng(0)
    return pd.DataFrame({"a": stream.normal(size=rows), "b": stream.exponential(size=rows),
                         "class": np.where(stream.random(rows) < 0.4, "yes", "no")})


def test_outcome_hides_the_matched_count_and_tilts_toward_class_one() -> None:
    complete = _complete()
    variant = mnar_family.outcome(complete, 0.6, 3.0, seed=5)
    assert variant["class"].equals(complete["class"])
    y = mnar_family.labels(complete)  # "yes" sorts after "no", so it is class 1
    for column in ("a", "b"):
        gaps = variant[column].isna().to_numpy()
        assert gaps.sum() == round(len(complete) * 0.6)
        assert gaps[y == 1].mean() > gaps[y == 0].mean() + 0.1
        observed = ~gaps
        assert (variant.loc[observed, column] == complete.loc[observed, column]).all()


def test_each_column_has_its_own_stream() -> None:
    complete = _complete()
    both = mnar_family.outcome(complete, 0.6, 3.0, seed=5)
    alone = mnar_family.outcome(complete.drop(columns="b"), 0.6, 3.0, seed=5)
    assert both["a"].isna().equals(alone["a"].isna())
    assert not both["a"].isna().equals(mnar_family.outcome(complete, 0.6, 3.0, seed=6)["a"].isna())


def test_noise_adds_one_column_whose_gaps_follow_the_label_and_keeps_the_rest() -> None:
    mcar = _complete()
    mcar.loc[mcar.index[:300], "a"] = np.nan
    variant = mnar_family.noise(mcar, 0.6, 3.0, seed=5)
    assert list(variant.columns) == ["a", "b", mnar_family.NOISE, "class"]
    assert variant[["a", "b", "class"]].equals(mcar[["a", "b", "class"]])
    gaps = variant[mnar_family.NOISE].isna().to_numpy()
    y = mnar_family.labels(mcar)
    assert gaps.sum() == round(len(mcar) * 0.6) and gaps[y == 1].mean() > gaps[y == 0].mean() + 0.1


def test_a_multiclass_label_is_refused() -> None:
    complete = _complete()
    complete["class"] = np.arange(len(complete)) % 3
    with pytest.raises(ValueError):
        mnar_family.outcome(complete, 0.6, 3.0, seed=5)


def test_write_keeps_an_equal_file_and_refuses_a_different_one(tmp_path: Path) -> None:
    frame = _complete(50)
    path = tmp_path / "v.csv"
    mnar_family.write(frame, path)
    mnar_family.write(frame, path)
    with pytest.raises(SystemExit):
        mnar_family.write(frame.assign(a=0.0), path)


def test_mcar_hides_the_matched_count_without_regard_to_class() -> None:
    complete = _complete(6000)
    variant = mnar_family.mcar(complete, 0.6, seed=5)
    y = mnar_family.labels(complete)
    for column in ("a", "b"):
        gaps = variant[column].isna().to_numpy()
        assert gaps.sum() == round(len(complete) * 0.6) and abs(gaps[y == 1].mean() - gaps[y == 0].mean()) < 0.05
