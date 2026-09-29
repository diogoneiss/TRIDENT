"""The decode stage: training a decoder and scoring what it reconstructs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.embedder import TabularEmbedder
from src.models import TridentPretrainer
from src.training.data import evaluation_mask
from src.training.decoding import train_and_evaluate_decoder
from src.training.imputation_metrics import mean_mode_baselines
from src.training.tracking import BufferedFoldTracker
from src.training.types import FoldSplit, Hyperparameters, PreparedDataset, PretrainingOutcome
from src.transformer import TabularTransformerEncoder
from src.utils import preprocess_table

CATEGORICAL = ["colour", "shape"]
NUMERICAL = ["size", "weight"]


def _complete_frame(rows: int = 36) -> pd.DataFrame:
    """The table before any cell was taken away, as a _00nan sibling is stored."""
    colours = ["red", "blue", "green", "red"]
    shapes = ["round", "square", "round", "square"]
    return pd.DataFrame(
        {
            "colour": [colours[index % 4] for index in range(rows)],
            "shape": [shapes[index % 4] for index in range(rows)],
            "size": [float(index % 9) for index in range(rows)],
            "weight": [float(index % 5) / 2 for index in range(rows)],
            "class": [index % 2 for index in range(rows)],
        }
    )


def _dataset(
    rows: int = 36,
    complete: pd.DataFrame | None = None,
    gaps: dict[str, int] | None = None,
) -> PreparedDataset:
    """That table with cells taken away, then scaled as preparation would leave it.

    ``gaps`` maps a column to the modulus of the rows it loses; by default ``size`` loses
    every fifth row and ``colour`` every seventh.
    """
    frame = (complete if complete is not None else _complete_frame(rows)).copy()
    for column, every in (gaps or {"size": 5, "colour": 7}).items():
        frame.loc[frame.index % every == 0, column] = np.nan
    raw_numerical = frame[NUMERICAL].copy()
    scaler = StandardScaler()
    frame[NUMERICAL] = scaler.fit_transform(frame[NUMERICAL])
    frame.attrs["dataset_name"] = "toy_20nan"
    return PreparedDataset(
        frame=frame,
        raw_numerical=raw_numerical,
        label_column="class",
        categorical_columns=CATEGORICAL,
        numerical_columns=NUMERICAL,
        label_classes=np.array([0, 1]),
        source_path=Path("datasets/processed_datasets/toy/toy_20nan.csv"),
        splits_path=Path("datasets/processed_datasets/splits/toy_split.json"),
        scaler=scaler,
    )


def _fold(rows: int = 36) -> FoldSplit:
    every = np.arange(rows)
    return FoldSplit(
        train_indices=every[: rows // 2],
        validation_indices=every[rows // 2 : rows // 2 + rows // 4],
        test_indices=every[rows // 2 + rows // 4 :],
    )


def _pretrained(dataset: PreparedDataset) -> PretrainingOutcome:
    """The encoder a real run would hand over, without paying for pre-training."""
    torch.manual_seed(0)
    features = dataset.frame.drop(columns=[dataset.label_column])
    embedder = TabularEmbedder(features, CATEGORICAL, NUMERICAL, dimensao=8, hidden_dim=4)
    transformer = TabularTransformerEncoder(
        d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.0
    )
    return PretrainingOutcome(
        model=TridentPretrainer(embedder, transformer), train_losses=[], validation_losses=[]
    )


def _hyperparameters(**overrides) -> Hyperparameters:
    return Hyperparameters.from_mapping(
        {
            "DIM": 8, "HIDDEN_DIM": 4, "HEADS": 2, "LAYERS": 1, "DIM_FEED": 16,
            "DROPOUT": 0.0, "EPOCHS_PRE": 1, "BATCH": 16, "PROB_MASCARA": 0.5,
            "EPOCHS_DECODE": 3, "LR_DECODE": 0.01, "EVAL_MASK_RATE": 0.3,
            **overrides,
        }
    )


def _run(dataset=None, hyperparameters=None, sibling=None, **kwargs):
    dataset = dataset if dataset is not None else _dataset()
    tracker = BufferedFoldTracker()
    outcome = train_and_evaluate_decoder(
        dataset=dataset,
        fold=_fold(len(dataset.frame)),
        pretraining=_pretrained(dataset),
        hyperparameters=hyperparameters or _hyperparameters(),
        device=torch.device("cpu"),
        tracker=tracker,
        seed=42,
        fold_ordinal=1,
        complete_sibling=sibling,
        **kwargs,
    )
    return outcome, tracker


def test_the_decode_stage_reports_a_loss_for_every_epoch_it_trained() -> None:
    """Its loss curves are what a reader plots, so every epoch has to be on them."""
    outcome, tracker = _run(hyperparameters=_hyperparameters(EPOCHS_DECODE=3))

    assert len(outcome.train_losses) == 3
    assert len(outcome.validation_losses) == 3
    logged = {event.key for event in tracker.metric_events if event.step is not None}
    assert {"decode/train_loss", "decode/val_loss", "decode/learning_rate"} <= logged


def test_the_decode_stage_scores_the_validation_split_only_when_asked() -> None:
    """A search ranks trials on the validation split, so the test split never chooses
    hyperparameters. Every other run carries no validation score at all, so the
    cross-validation summary never sees one (ADR 0005, decision 3).

    The validation mask is the one the checkpoint already watches; the score is the
    masked-population ``impute_score`` on it, logged in its own family.
    """
    asked, tracker_asked = _run(score_search_objective=True)
    unasked, tracker_unasked = _run()

    key = "validation/impute/masked/impute_score"
    assert key in asked.result.metrics
    assert key in {event.key for event in tracker_asked.metric_events if event.step is None}
    # A different split, so a different number: the score is not the test one relabelled.
    assert asked.result.metrics[key] != asked.result.metrics["impute/masked/impute_score"]
    assert not any(name.startswith("validation/") for name in unasked.result.metrics)
    assert not any(
        "validation/" in event.key for event in tracker_unasked.metric_events
    )
    assert not any(
        event.key.startswith("test/validation/") for event in tracker_asked.metric_events
    )


def test_the_decode_stage_scores_and_lists_every_cell_it_hid() -> None:
    """Two outputs from one pass: numbers that rank the fold, and the cells behind them.

    The count is checked against the mask drawn independently here, so the table cannot
    quietly agree with metrics computed from the same mistake.
    """
    dataset = _dataset()
    hyperparameters = _hyperparameters()
    outcome, tracker = _run(dataset=dataset, hyperparameters=hyperparameters)

    metrics = outcome.result.metrics
    assert "impute/masked/impute_score" in metrics
    assert "impute/masked/rmse_num_z" in metrics
    assert "impute/masked/acc_cat" in metrics

    features = dataset.frame.drop(columns=[dataset.label_column])
    test_frame = features.iloc[_fold(len(dataset.frame)).test_indices].reset_index(drop=True)
    drawn_here = evaluation_mask(test_frame, hyperparameters.eval_mask_rate, 42, 1)
    hidden_here = int((drawn_here == "[MASK]").to_numpy().sum())

    listed = outcome.scored_cells[outcome.scored_cells["population"] == "masked"]
    assert len(listed) == hidden_here
    assert set(listed.columns) >= {"row", "column", "kind", "population", "actual", "imputed"}


def test_cells_the_dataset_is_missing_are_scored_against_the_complete_table() -> None:
    """The gaps in a variant are the real imputation benchmark: their truth is known.

    They reach the model as [MASK], the token the decoder was trained on, rather than as
    [NULL], which it has never been asked to reconstruct.
    """
    dataset = _dataset()
    with_truth, _ = _run(dataset=dataset, sibling=_complete_frame())
    without_truth, _ = _run(dataset=dataset, sibling=None)

    assert "impute/induced/impute_score" in with_truth.result.metrics
    assert not any(key.startswith("impute/induced/") for key in without_truth.result.metrics)

    features = dataset.frame.drop(columns=[dataset.label_column])
    test_rows = features.iloc[_fold(len(dataset.frame)).test_indices]
    gaps_here = int(test_rows.isna().to_numpy().sum())
    listed = with_truth.scored_cells[with_truth.scored_cells["population"] == "induced"]
    assert len(listed) == gaps_here


def test_the_best_epoch_is_the_one_that_is_kept() -> None:
    """Training runs the whole budget, but the model handed back is the best it ever was.

    This only means anything because the validation mask never changes: every epoch is
    graded on the same hidden cells, so the losses are comparable at all.
    """
    dataset = _dataset()
    hyperparameters = _hyperparameters(EPOCHS_DECODE=6, LR_DECODE=0.2)
    outcome, _ = _run(dataset=dataset, hyperparameters=hyperparameters)

    features = dataset.frame.drop(columns=[dataset.label_column])
    validation = features.iloc[_fold(len(dataset.frame)).validation_indices].reset_index(drop=True)
    embedder = outcome.model.embedder
    device = torch.device("cpu")
    hidden = embedder.encode(
        evaluation_mask(validation, hyperparameters.eval_mask_rate, 42, 1), device
    )
    clean = embedder.encode(preprocess_table(validation.copy(), p_base=0.0, fine_tunning=True), device)
    with torch.no_grad():
        kept_model_scores, _ = outcome.model(hidden, clean)

    assert kept_model_scores.item() == pytest.approx(min(outcome.validation_losses), rel=1e-5)


def test_extra_rates_are_scored_as_diagnostics_beside_the_one_that_ranks() -> None:
    """One model, several difficulties. Only the primary rate ranks the fold.

    Scoring a second rate costs one forward pass; training a second model to get it would
    cost the whole pre-training budget again.
    """
    outcome, _ = _run(
        hyperparameters=_hyperparameters(EVAL_MASK_RATES_EXTRA=[0.1, 0.5])
    )

    metrics = outcome.result.metrics
    assert "impute/masked/impute_score" in metrics
    assert "impute/masked/rate_10/rmse_num_z" in metrics
    assert "impute/masked/rate_50/rmse_num_z" in metrics
    assert metrics["impute/masked/rate_10/n_num_cells"] < metrics["impute/masked/rate_50/n_num_cells"]


def test_the_null_path_diagnostic_asks_the_same_cells_through_the_other_token() -> None:
    """The same gaps, shown as [NULL] instead of [MASK], to test whether the head transfers.

    Off unless asked for, and never allowed near the ranking metric.
    """
    dataset = _dataset()
    asked, _ = _run(dataset=dataset, sibling=_complete_frame(), score_null_path=True)
    unasked, _ = _run(dataset=dataset, sibling=_complete_frame())

    assert "impute/induced/null_token/acc_cat" in asked.result.metrics
    assert not any("null_token" in key for key in unasked.result.metrics)
    through_mask = asked.scored_cells["population"] == "induced"
    through_null = asked.scored_cells["population"] == "induced_null_token"
    assert int(through_null.sum()) == int(through_mask.sum())


def test_the_truth_beside_each_guess_is_the_number_the_dataset_actually_holds() -> None:
    """Ticket 0004: a known truth should not be reported at the model's precision.

    The scored table's `actual` stays in the scaled float32 space the metrics compare in,
    but the original-unit truth beside it is read from the frame before scaling, so it is
    exact. The two populations take it from different places, which is the part that can
    be silently wrong: a masked cell's truth is the variant's own value, while an induced
    cell's is the sibling's, because the variant holds nothing but a gap there.

    Expectations come from the raw frames directly rather than from any inverse scaling,
    so a value that round-tripped through float32 cannot satisfy them.
    """
    dataset = _dataset()
    fold = _fold(len(dataset.frame))
    outcome, _ = _run(dataset=dataset, sibling=_complete_frame())
    scored = outcome.scored_cells

    variant_truth = dataset.raw_numerical.iloc[fold.test_indices].reset_index(drop=True)
    masked = scored[(scored["population"] == "masked") & (scored["kind"] == "numerical")]
    assert len(masked) > 0
    for _, cell in masked.iterrows():
        assert cell["actual_original"] == variant_truth.loc[cell["row"], cell["column"]]

    sibling_truth = _complete_frame().iloc[fold.test_indices].reset_index(drop=True)
    induced = scored[(scored["population"] == "induced") & (scored["kind"] == "numerical")]
    assert len(induced) > 0
    for _, cell in induced.iterrows():
        assert cell["actual_original"] == sibling_truth.loc[cell["row"], cell["column"]]

    # `actual` still holds the scaled value the metrics compare in. Scaling it back only
    # approximates the exact column, which is the float32 gap this ticket is about.
    sizes = masked[masked["column"] == "size"]
    index = list(dataset.numerical_columns).index("size")
    rescaled = (
        sizes["actual"].astype(float) * dataset.scaler.scale_[index]
        + dataset.scaler.mean_[index]
    )
    assert rescaled.to_numpy() == pytest.approx(
        sizes["actual_original"].astype(float).to_numpy(), abs=1e-5
    )


def test_an_induced_cell_whose_category_the_variant_never_shows_is_scored_as_a_miss() -> None:
    """kr-vs-kp's `spcop` is `f` everywhere but once, and the 40nan generator took that
    one `t`, so the variant's vocabulary lacks it while the sibling's truth holds it at an
    induced cell. The head can never produce a category its embedder never saw, so the
    cell counts as a miss instead of crashing the fold (fold 4 of a 5-fold run did).
    """
    dataset = _dataset()
    sibling = _complete_frame()
    # Row 28 is a gap in the variant (index % 7 == 0) and sits in the test fold.
    sibling.loc[28, "colour"] = "violet"

    outcome, _ = _run(dataset=dataset, sibling=sibling)

    cells = outcome.scored_cells
    induced_colour = cells[(cells["population"] == "induced") & (cells["column"] == "colour")]
    unseen = induced_colour[induced_colour["actual"] == "violet"]
    assert len(unseen) == 1
    assert unseen.iloc[0]["imputed"] != "violet"
    assert unseen.iloc[0]["actual_original"] == "violet"
    # Rows 28 and 35 are the variant's colour gaps in the test fold: both are scored.
    assert outcome.result.metrics["impute/induced/n_cat_cells"] == 2


def test_every_scored_population_carries_both_baseline_imputers_beside_the_model() -> None:
    """Beside every score of the model's, what filling the mean or mode and what a KNN
    imputer would have scored on the very same cells (ADR 0007), so a reader can tell a
    low bar from a cleared one.

    The naive RMSE is recomputed here from the fold's own masked cells: a hook placed
    after the induced cells are appended to the table would score the wrong population
    and disagree with it.
    """
    dataset = _dataset()
    hyperparameters = _hyperparameters(EVAL_MASK_RATES_EXTRA=[0.1])
    outcome, tracker = _run(
        dataset=dataset, hyperparameters=hyperparameters, sibling=_complete_frame()
    )
    metrics = outcome.result.metrics

    for population in ("impute/masked", "impute/masked/rate_10", "impute/induced"):
        for name in ("mean_mode", "knn5", "knn10", "hgb"):
            assert f"{population}/baseline/{name}/impute_score" in metrics, (population, name)
            assert f"{population}/baseline/{name}/rmse_num_z" in metrics, (population, name)
            assert f"{population}/baseline/{name}/acc_cat" in metrics, (population, name)
            assert f"{population}/baseline/{name}/n_num_cells" not in metrics

    features = dataset.frame.drop(columns=[dataset.label_column])
    fold = _fold(len(dataset.frame))
    naive = mean_mode_baselines(
        features.iloc[fold.train_indices].reset_index(drop=True), NUMERICAL, CATEGORICAL
    )
    masked = outcome.scored_cells[outcome.scored_cells["population"] == "masked"]
    numbers = masked[masked["kind"] == "numerical"]
    naive_rmse = float(
        np.sqrt(
            np.mean(
                (numbers["column"].map(naive).to_numpy(dtype=float) - numbers["actual"].to_numpy(dtype=float)) ** 2
            )
        )
    )
    assert metrics["impute/masked/baseline/mean_mode/rmse_num_z"] == pytest.approx(naive_rmse)
    assert metrics["impute/masked/baseline/mean_mode/impute_score"] == pytest.approx(1.0, abs=1e-12)
    logged = {event.key for event in tracker.metric_events if event.step is None}
    for population in ("masked", "induced"):
        rows = outcome.baseline_per_column[population]
        for name in ("mean_mode", "knn5", "knn10", "hgb"):
            assert f"baseline/{name}/rmse_num_z" in set(rows["metric"]), (population, name)
    for name in ("knn5", "knn10", "hgb"):
        assert f"test/impute/masked/baseline/{name}/impute_score" in logged
        assert f"test/impute/induced/baseline/{name}/impute_score" in logged


def test_baseline_imputers_stay_out_of_the_diagnostic_and_search_families() -> None:
    """The null-token path scores the same induced cells, so its baselines would be the
    induced ones repeated; the search objective is compared across trials that all share
    the same data, so a baseline there would be a constant. Neither carries one.
    """
    outcome, tracker = _run(
        sibling=_complete_frame(), score_null_path=True, score_search_objective=True
    )

    keys = set(outcome.result.metrics) | {event.key for event in tracker.metric_events}
    assert any("/baseline/" in key for key in keys)
    assert not any("null_token/baseline" in key for key in keys)
    assert not any("validation/" in key and "baseline" in key for key in keys)


def test_an_integer_coded_category_is_scored_against_the_truth_it_spells() -> None:
    """electricity's ``day`` is a category written as a number. The variant has gaps in
    it, so it is read as floats and its categories spell ``"3.0"``; the complete sibling
    has none, is read as integers, and spells the same day ``"3"``. Compared as they are
    stored, every induced cell of such a column looked like a category the variant never
    shows and was scored a miss, for the model and every baseline alike.

    The induced truth must be spelt the way the variant spells it, so each one is a
    category the model could answer with, and a guess of the right day counts as right.
    """
    complete = _complete_frame()
    complete["shape"] = [1 + index % 3 for index in range(len(complete))]
    dataset = _dataset(complete=complete, gaps={"size": 5, "colour": 7, "shape": 4})
    assert dataset.frame["shape"].dtype == np.float64
    assert complete["shape"].dtype == np.int64

    outcome, _ = _run(dataset=dataset, sibling=complete)

    induced = outcome.scored_cells[outcome.scored_cells["population"] == "induced"]
    shapes = induced[induced["column"] == "shape"]
    assert len(shapes) > 0
    assert set(shapes["actual"]) <= {"1.0", "2.0", "3.0"}
    # The mode baseline answers every cell with the training mode; it is right exactly on
    # the cells whose truth is that mode, which this toy guarantees are not none.
    features = dataset.frame.drop(columns=[dataset.label_column])
    train = features.iloc[_fold(len(dataset.frame)).train_indices]
    mode = str(train["shape"].mode().iloc[0])
    assert (shapes["actual"] == mode).any()
    assert outcome.result.metrics["impute/induced/baseline/mean_mode/acc_cat"] > 0.0


def test_the_search_also_scores_the_validation_splits_own_gaps() -> None:
    """A search may rank trials by the induced population, the headline, rather than the
    masked one (ADR 0008). On the validation split those are the validation rows' own
    gaps, scored against the complete table as the test ones are, and scoring them must
    not move the masked validation score an existing study ranks by.

    The toy's validation rows (18 to 26) hold three gaps: ``size`` at 20 and 25, ``colour``
    at 21.
    """
    dataset = _dataset()
    np.random.seed(0)
    with_truth, _ = _run(dataset=dataset, sibling=_complete_frame(), score_search_objective=True)
    np.random.seed(0)
    without_truth, _ = _run(dataset=dataset, sibling=None, score_search_objective=True)

    metrics = with_truth.result.metrics
    assert "validation/impute/induced/impute_score" in metrics
    assert (metrics["validation/impute/induced/n_num_cells"], metrics["validation/impute/induced/n_cat_cells"]) == (2, 1)
    assert not any(key.startswith("validation/impute/induced/") for key in without_truth.result.metrics)
    assert not any(key.startswith("validation/impute/induced/baseline/") for key in metrics)
    assert metrics["validation/impute/masked/impute_score"] == without_truth.result.metrics[
        "validation/impute/masked/impute_score"
    ]
