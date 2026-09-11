"""Optuna's search space and failure handling, per task (ADR 0004, decision 13)."""

from pathlib import Path

import optuna
import pytest

from opt import define_search_space


def _sampled(task: str, trials: int = 40, **space) -> list[dict]:
    """Draw a spread of trials so a constraint is tested against many combinations."""
    drawn: list[dict] = []
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

    def objective(trial):
        drawn.append(define_search_space(trial, task=task, **space))
        return 0.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=trials)
    return drawn


def test_a_search_tunes_the_stage_its_task_actually_runs() -> None:
    """Sampling a fine-tuning rate for a run with no classifier wastes the trial.

    It also records a parameter that explains nothing about the result.
    """
    for_classification = _sampled("classification", trials=1)[0]
    for_imputation = _sampled("imputation", trials=1)[0]

    assert {"EPOCH_FINE", "LR_FINE", "WEIGHT_DECAY_FINE"} <= set(for_classification)
    assert {"EPOCHS_DECODE", "LR_DECODE", "WEIGHT_DECAY_DECODE", "LAMBDA_NUM"} <= set(
        for_imputation
    )
    assert not {"EPOCH_FINE", "LR_FINE", "WEIGHT_DECAY_FINE"} & set(for_imputation)
    assert "DIM" in for_classification and "DIM" in for_imputation


def test_the_reduced_profile_samples_only_what_moves_the_decoder() -> None:
    """A held hyperparameter runs at the task default, so it must not appear at all.

    The reduced profile gives freedom to the decode stage and to the corruption it learns
    from, and to the loss balance only where both column types exist (ADR 0005,
    decision 2).
    """
    mixed = _sampled("imputation", trials=1, profile="reduced", mixed_columns=True)[0]
    single_type = _sampled("imputation", trials=1, profile="reduced", mixed_columns=False)[0]

    assert set(mixed) == {
        "PROB_MASCARA", "LR_DECODE", "WEIGHT_DECAY_DECODE", "DROPOUT", "LAMBDA_NUM"
    }
    assert set(single_type) == {"PROB_MASCARA", "LR_DECODE", "WEIGHT_DECAY_DECODE", "DROPOUT"}


def test_the_decode_learning_rate_can_exceed_its_default_in_both_profiles() -> None:
    """The default is 1e-3 and the old range ended there, so nothing above it was reachable.

    Forty log-uniform draws over a range that reaches 1e-2 land above 1e-3 with certainty
    for any practical purpose; a range capped at the default never does.
    """
    for profile in ("full", "reduced"):
        drawn = _sampled("imputation", trials=40, profile=profile)
        assert max(params["LR_DECODE"] for params in drawn) > 1e-3, profile


def test_lambda_num_is_sampled_only_where_both_column_types_exist() -> None:
    """On a single-type table the loss balance is a pure scale, so sampling it wastes a dimension.

    The full profile keeps every other knob; only the loss balance follows the table.
    """
    mixed = _sampled("imputation", trials=1, profile="full", mixed_columns=True)[0]
    single_type = _sampled("imputation", trials=1, profile="full", mixed_columns=False)[0]

    assert "LAMBDA_NUM" in mixed
    assert "LAMBDA_NUM" not in single_type
    assert set(single_type) == set(mixed) - {"LAMBDA_NUM"}
    assert len(mixed) == 15


def test_the_study_entry_point_accepts_every_training_flag() -> None:
    """opt.py's private parser lacked --task, --lr_scheduler, --disable_mlflow and
    --metrics_dir, so a study launched through it could not be an imputation study at all.

    One parser for both entry points cannot drift (ADR 0005, decision 1).
    """
    from opt import build_parser

    args = build_parser().parse_args(
        [
            "--dataset_name", "credit-g_20nan", "--task", "imputation",
            "--search_space", "reduced", "--lr_scheduler", "cosine", "--disable_mlflow",
            "--metrics_dir", "scratch", "--n_trials", "3", "--retrain_best",
        ]
    )

    assert args.task == "imputation" and args.search_space == "reduced"
    assert args.lr_scheduler == "cosine" and args.disable_mlflow is True
    assert args.metrics_dir == "scratch" and args.n_trials == 3 and args.retrain_best is True


def test_a_search_never_tunes_how_hard_its_own_exam_is() -> None:
    """A trial that could lower the evaluation mask rate would win by hiding less."""
    for_imputation = _sampled("imputation", trials=1)[0]

    assert "EVAL_MASK_RATE" not in for_imputation
    assert "EVAL_MASK_RATES_EXTRA" not in for_imputation


def test_every_sampled_shape_is_one_the_model_can_actually_build() -> None:
    """Attention splits the model width across heads, so the width must divide evenly.

    Invalid pairs used to be sampled, crash, and be scored as merely terrible
    hyperparameters, which taught the search to avoid a head count for the wrong reason.
    """
    for params in _sampled("classification", trials=40):
        assert params["DIM"] % params["HEADS"] == 0, params


def test_a_trial_that_crashes_is_discarded_rather_than_scored(tmp_path, monkeypatch) -> None:
    """Scoring a crash teaches the search to seek crashes when lower is better.

    Returning 0.0 was "the worst possible score" only while maximising macro F1. Under a
    minimised error ratio it is the best achievable value, so a study would have chased
    configurations that fail.
    """
    import mlflow

    from opt import ObjectiveFunctionWrapper

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    objective = ObjectiveFunctionWrapper("credit-g_20nan", seed=42, output_dir=str(tmp_path))
    objective.task = "imputation"
    objective.ranking_metric = "impute/masked/impute_score"
    objective.direction = "minimize"
    objective.disable_mlflow = True
    objective.metrics_dir = "metrics"

    def explode(*args, **kwargs):
        raise RuntimeError("shape mismatch")

    import opt as opt_module

    original = opt_module.train_main
    opt_module.train_main = explode
    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=2)
    finally:
        opt_module.train_main = original

    assert [trial.state for trial in study.trials] == [
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.PRUNED,
    ]
    assert study.best_trials == []


def test_an_imputation_study_never_overwrites_a_datasets_shared_config(tmp_path) -> None:
    """Both tasks read datasets/hiperparams/<base>/<dataset>.json, and it names no task.

    An imputation study writing there would silently retune every later classification run
    on that dataset, and nothing would report it.
    """
    from opt import ObjectiveFunctionWrapper

    objective = ObjectiveFunctionWrapper(
        "credit-g_20nan", seed=42, output_dir=str(tmp_path), optuna_dir=tmp_path / "study"
    )
    objective.task = "imputation"
    objective.best_params = {"DIM": 64}
    objective.best_score = 0.5

    written = objective.save_best_params()

    assert written.is_relative_to(tmp_path / "study")
    assert not (Path("datasets/hiperparams/credit-g") / "credit-g_20nan.json").exists()
