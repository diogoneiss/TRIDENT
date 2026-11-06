"""E150's reader: the four inputs of a fold, built from the table, the fold's fills and its training rows."""

import numpy as np
import pandas as pd

from scripts.experiments import mimic_mortality


def test_the_four_inputs_fill_from_the_training_rows_or_the_decoder_and_flag_the_gapped_columns() -> None:
    features = pd.DataFrame({
        "full": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "sparse": [10.0, np.nan, 30.0, np.nan, 50.0, 1000.0],
        "kind": ["a", "b", None, "a", "a", "b"],
    })
    fills = pd.DataFrame({
        "row": [1, 3, 2], "column": ["sparse", "sparse", "kind"], "kind": ["numerical", "numerical", "categorical"],
        "imputed": [0.1, 0.2, "b"], "imputed_original_units": [22.0, 44.0, np.nan],
    })
    train = np.array([0, 1, 2, 3, 4])  # row 5, with its outlier, is the test row

    tables = mimic_mortality.inputs(features, fills, train)

    mean, spread = 30.0, pd.Series([10.0, 30.0, 30.0, 30.0, 50.0]).std()  # the training rows, once filled
    assert np.allclose(tables["N"]["sparse"], [(v - mean) / spread for v in (10, 30, 30, 30, 50, 1000)])
    assert tables["N"].loc[2, "kind_a"] == 1.0  # the training rows' mode
    decoded = features["sparse"].fillna(pd.Series({1: 22.0, 3: 44.0}))
    assert np.allclose(tables["T"]["sparse"], (decoded - decoded.iloc[train].mean()) / decoded.iloc[train].std())
    assert tables["T"].loc[2, "kind_b"] == 1.0  # the decoder's category
    assert tables["NI"]["sparse_missing"].tolist() == [0, 1, 0, 1, 0, 0]
    assert tables["TI"]["kind_missing"].tolist() == [0, 0, 1, 0, 0, 0]
    assert "full_missing" not in tables["NI"] and list(tables["N"].columns) == list(tables["T"].columns)


def test_the_classifier_scores_the_test_rows_with_the_three_measures() -> None:
    stream = np.random.default_rng(0)
    table = pd.DataFrame({"x": stream.normal(size=200), "noise": stream.normal(size=200)})
    labels = (table["x"] + 0.3 * stream.normal(size=200) > 0.8).to_numpy().astype(int)

    scores = mimic_mortality.score(table, labels, np.arange(150), np.arange(150, 200))

    assert set(scores) == set(mimic_mortality.METRICS)
    assert scores["auroc"] > 0.9 and scores["macro_f1"] > 0.7
