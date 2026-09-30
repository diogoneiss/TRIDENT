from argparse import Namespace
import json
from pathlib import Path

import pytest

from src.training.config import (
    build_training_parser,
    logged_hyperparameters,
    resolve_training_request,
    validate_parsed_args,
)
from src.training.types import DEFAULT_LR_SCHEDULER, Hyperparameters, RuntimeOptions


def test_legacy_namespace_uses_legacy_runtime_defaults() -> None:
    args = Namespace(
        dataset_name="vehicle_00nan",
        label_column="class",
        output_dir="results",
        seed=42,
        plot_losses=False,
        save_model=False,
        cv_folds=None,
    )

    request = resolve_training_request(args)

    assert request.runtime == RuntimeOptions(Path("results"), Path("metrics"), True)


def test_parser_exposes_runtime_options() -> None:
    args = build_training_parser().parse_args(
        ["--dataset_name", "vehicle_00nan", "--metrics_dir", "tmp/metrics", "--disable_mlflow"]
    )

    assert args.metrics_dir == "tmp/metrics"
    assert args.disable_mlflow is True


def test_legacy_namespace_accepts_hyperparameter_override() -> None:
    args = Namespace(dataset_name="vehicle_00nan", hyperparams_override={"DIM": 64})

    request = resolve_training_request(args)

    assert request.hyperparameters.dimension == 64


def test_parser_rejects_unknown_scheduler() -> None:
    with pytest.raises(SystemExit):
        build_training_parser().parse_args(
            ["--dataset_name", "vehicle_00nan", "--lr_scheduler", "linear"]
        )


def test_scheduler_defaults_to_legacy_when_flag_is_absent() -> None:
    args = build_training_parser().parse_args(["--dataset_name", "vehicle_00nan"])

    request = resolve_training_request(args)

    assert args.lr_scheduler is None
    assert request.hyperparameters.lr_scheduler == DEFAULT_LR_SCHEDULER == "cosine_legacy"


def test_scheduler_flag_overrides_hyperparameter_override_mapping() -> None:
    args = Namespace(
        dataset_name="vehicle_00nan",
        hyperparams_override={"DIM": 64, "LR_SCHEDULER": "plateau"},
        lr_scheduler="warmup_cosine",
    )

    request = resolve_training_request(args)

    assert request.hyperparameters.dimension == 64
    assert request.hyperparameters.lr_scheduler == "warmup_cosine"


def test_scheduler_flag_overrides_dataset_json_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "datasets" / "hiperparams" / "vehicle"
    config_dir.mkdir(parents=True)
    (config_dir / "vehicle_00nan.json").write_text(json.dumps({"DIM": 32, "LR_SCHEDULER": "cosine"}))

    from_file = resolve_training_request(Namespace(dataset_name="vehicle_00nan"))
    from_flag = resolve_training_request(
        Namespace(dataset_name="vehicle_00nan", lr_scheduler="constant")
    )

    assert from_file.hyperparameters.dimension == 32
    assert from_file.hyperparameters.lr_scheduler == "cosine"
    assert from_flag.hyperparameters.dimension == 32
    assert from_flag.hyperparameters.lr_scheduler == "constant"


def test_hyperparameters_reject_unknown_scheduler() -> None:
    with pytest.raises(ValueError, match="Unknown learning-rate scheduler"):
        Hyperparameters(lr_scheduler="linear")
    with pytest.raises(ValueError, match="Unknown learning-rate scheduler"):
        Hyperparameters.from_mapping({"LR_SCHEDULER": "linear"})


def test_mirroring_is_on_unless_a_run_asks_for_it_off() -> None:
    """ADR 0006, decision 4: default on by the user's explicit decision; ``--disable_mirror``
    and a tracking-disabled run both turn it off."""
    parser = build_training_parser()

    default = resolve_training_request(parser.parse_args(["--dataset_name", "vehicle_00nan"]))
    disabled = resolve_training_request(
        parser.parse_args(["--dataset_name", "vehicle_00nan", "--disable_mirror"])
    )
    untracked = resolve_training_request(
        parser.parse_args(["--dataset_name", "vehicle_00nan", "--disable_mlflow"])
    )

    assert default.runtime.mirror_runs is True
    assert disabled.runtime.mirror_runs is False
    assert untracked.runtime.mirror_runs is False
    assert resolve_training_request(Namespace(dataset_name="vehicle_00nan")).runtime.mirror_runs is True


def test_legacy_namespace_defaults_to_parent_tracking_without_extra_tags() -> None:
    request = resolve_training_request(Namespace(dataset_name="vehicle_00nan"))

    assert request.runtime.tracking_run_role == "parent"
    assert request.runtime.tracking_tags == {}


def test_legacy_namespace_accepts_programmatic_tracking_role_and_tags() -> None:
    args = Namespace(
        dataset_name="vehicle_00nan",
        mlflow_run_role="optuna_trial",
        mlflow_tags={"optuna_study_run_id": "abc"},
    )

    request = resolve_training_request(args)

    assert request.runtime.tracking_run_role == "optuna_trial"
    assert request.runtime.tracking_tags == {"optuna_study_run_id": "abc"}


def test_runtime_options_reject_unknown_tracking_role() -> None:
    with pytest.raises(ValueError, match="tracking run role"):
        RuntimeOptions(tracking_run_role="child")


def test_a_config_written_before_the_imputation_task_still_loads() -> None:
    """Every stored config predates the decode stage and must keep working untouched.

    These are the exact fifteen keys of datasets/hiperparams/vehicle/vehicle_00nan.json,
    which already lacks LR_SCHEDULER and loads today.
    """
    hyperparameters = Hyperparameters.from_mapping(
        {
            "DIM": 128, "HIDDEN_DIM": 16, "HEADS": 16, "LAYERS": 2, "DIM_FEED": 32,
            "DROPOUT": 0.2, "EPOCHS_PRE": 300, "BATCH": 256, "LR_PRE": 0.00034,
            "WEIGHT_DECAY_PRE": 0.005, "PROB_MASCARA": 0.5, "EPOCH_FINE": 150,
            "LR_FINE": 0.001, "WEIGHT_DECAY_FINE": 0.0019, "LABELS": 4,
        }
    )

    assert hyperparameters.dimension == 128
    assert hyperparameters.decode_epochs == 150
    assert hyperparameters.decode_learning_rate == 0.001
    assert hyperparameters.decode_weight_decay == 0.0019
    assert hyperparameters.lambda_num == 1.0
    assert hyperparameters.eval_mask_rate == 0.2
    assert hyperparameters.eval_mask_rates_extra == ()


def test_a_run_trains_classification_unless_it_is_asked_for_imputation() -> None:
    """Classification is what every run did before, so it is what a run does by default."""
    parser = build_training_parser()

    unasked = resolve_training_request(parser.parse_args(["--dataset_name", "vehicle_00nan"]))
    asked = resolve_training_request(
        parser.parse_args(["--dataset_name", "vehicle_00nan", "--task", "imputation"])
    )

    assert unasked.task == "classification"
    assert asked.task == "imputation"


def test_parser_rejects_unknown_task() -> None:
    with pytest.raises(SystemExit):
        build_training_parser().parse_args(
            ["--dataset_name", "vehicle_00nan", "--task", "regression"]
        )


def test_the_null_path_diagnostic_is_refused_without_the_imputation_task() -> None:
    """There is no decoder under classification, so there would be nothing to score.

    Refused when the arguments are read rather than hours later, mid-run.
    """
    parser = build_training_parser()
    with_classification = parser.parse_args(
        ["--dataset_name", "credit-g_20nan", "--score_null_path"]
    )
    with_imputation = parser.parse_args(
        ["--dataset_name", "credit-g_20nan", "--task", "imputation", "--score_null_path"]
    )

    with pytest.raises(SystemExit, match="score_null_path"):
        validate_parsed_args(with_classification)
    assert resolve_training_request(with_imputation).score_null_path is True


def test_a_reduced_search_space_exists_only_for_imputation_until_someone_defines_it() -> None:
    """The reduced profile holds the shared knobs and samples the decode stage, so for
    classification it would sample a space no one has asked to run (ADR 0005, decision 1).

    Refused at parse time, like --score_null_path, rather than mid-study. Left unspecified,
    the flag stays unspecified so the study can resolve it per task.
    """
    parser = build_training_parser()
    with_classification = parser.parse_args(
        ["--dataset_name", "credit-g_20nan", "--search_space", "reduced"]
    )
    with_imputation = parser.parse_args(
        ["--dataset_name", "credit-g_20nan", "--task", "imputation", "--search_space", "reduced"]
    )
    unspecified = parser.parse_args(["--dataset_name", "credit-g_20nan"])

    with pytest.raises(SystemExit, match="search_space"):
        validate_parsed_args(with_classification)
    assert validate_parsed_args(with_imputation).search_space == "reduced"
    assert unspecified.search_space is None


def test_a_search_can_ask_a_request_for_its_validation_score_and_nothing_else_can() -> None:
    """Programmatic only, like the tracking role: no flag exists, so neither a command line
    nor a stored configuration can turn it on, and a run that does not mention it never
    scores its validation split (ADR 0005, decision 3).
    """
    parser = build_training_parser()
    plain = resolve_training_request(
        parser.parse_args(["--dataset_name", "credit-g_20nan", "--task", "imputation"])
    )
    asked = resolve_training_request(
        Namespace(dataset_name="credit-g_20nan", task="imputation", score_search_objective=True)
    )

    assert plain.score_search_objective is False
    assert asked.score_search_objective is True


def test_a_run_records_the_parameters_it_used_and_no_others() -> None:
    """A recorded parameter that did nothing misleads whoever reads the run later.

    An imputation run has no classifier, so the fine-tuning rates and the label count
    would describe nothing; a classification run has no decoder. Classification's own
    record is unchanged, so old and new runs still line up column for column.
    """
    parser = build_training_parser()
    classification = resolve_training_request(parser.parse_args(["--dataset_name", "credit-g_20nan"]))
    imputation = resolve_training_request(
        parser.parse_args(["--dataset_name", "credit-g_20nan", "--task", "imputation"])
    )

    for_classification = logged_hyperparameters(classification)
    for_imputation = logged_hyperparameters(imputation)

    assert set(for_classification) == {
        "DIM", "HIDDEN_DIM", "HEADS", "LAYERS", "DIM_FEED", "DROPOUT", "EPOCHS_PRE",
        "BATCH", "LR_PRE", "WEIGHT_DECAY_PRE", "PROB_MASCARA", "EPOCH_FINE", "LR_FINE",
        "WEIGHT_DECAY_FINE", "LABELS", "LR_SCHEDULER",
    }
    assert set(for_imputation) == {
        "DIM", "HIDDEN_DIM", "HEADS", "LAYERS", "DIM_FEED", "DROPOUT", "EPOCHS_PRE",
        "BATCH", "LR_PRE", "WEIGHT_DECAY_PRE", "PROB_MASCARA", "LR_SCHEDULER",
        "EPOCHS_DECODE", "LR_DECODE", "WEIGHT_DECAY_DECODE", "LAMBDA_NUM", "EVAL_MASK_RATE",
        "DECODE_PATIENCE",
    }


def test_a_single_cross_validation_fold_is_refused_with_a_way_forward() -> None:
    """Backlog B2: `--cv_folds 1` divided by zero several steps into the run.

    One fold is not cross-validation, and the validation ratio `build_folds` derives
    from it is `0.1 / (1 - 1/1)`. Refusing it when the arguments are read costs the
    user nothing, and the message names the mode they actually wanted: omitting the
    flag entirely runs the predefined split, which is the real single-split path.
    """
    parser = build_training_parser()
    single = parser.parse_args(["--dataset_name", "vehicle_00nan", "--cv_folds", "1"])
    two = parser.parse_args(["--dataset_name", "vehicle_00nan", "--cv_folds", "2"])
    predefined = parser.parse_args(["--dataset_name", "vehicle_00nan"])

    with pytest.raises(SystemExit, match="cv_folds"):
        validate_parsed_args(single)
    # The flag's legal values are untouched, and omitting it stays the split mode.
    assert validate_parsed_args(two).cv_folds == 2
    assert validate_parsed_args(predefined).cv_folds is None


def test_an_imputation_run_prefers_its_own_configuration_and_falls_back_to_the_shared_one(
    tmp_path, monkeypatch
) -> None:
    """The shared file names no task, so a promoted imputation configuration lives beside
    it under a task-keyed name (ADR 0005, decision 5).

    Imputation reads the task-keyed file first, then the shared one, then the defaults, so
    a variant with no promoted configuration behaves as it did before. Classification
    never reads the task-keyed file: that is the promise that promoting an imputation
    result cannot retune a classification run.
    """
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "datasets" / "hiperparams" / "vehicle"
    config_dir.mkdir(parents=True)
    shared = config_dir / "vehicle_00nan.json"
    task_keyed = config_dir / "vehicle_00nan.imputation.json"

    def request(task: str):
        return resolve_training_request(Namespace(dataset_name="vehicle_00nan", task=task))

    shared.write_text(json.dumps({"DIM": 32}))
    task_keyed.write_text(json.dumps({"DIM": 64}))
    assert request("imputation").hyperparameters.dimension == 64
    assert request("classification").hyperparameters.dimension == 32

    task_keyed.unlink()
    assert request("imputation").hyperparameters.dimension == 32
    assert request("classification").hyperparameters.dimension == 32

    shared.unlink()
    assert request("imputation").hyperparameters.dimension == 128
    assert request("classification").hyperparameters.dimension == 128


def test_every_request_says_where_its_configuration_came_from(tmp_path, monkeypatch) -> None:
    """A tuned run and a default run look alike in the store unless the run says which
    file it loaded, so the source travels on the request and is logged as a param.

    The path is relative to the repository and posix-style whatever the platform, so
    the same configuration reads the same in every store.
    """
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "datasets" / "hiperparams" / "vehicle"
    config_dir.mkdir(parents=True)

    assert resolve_training_request(Namespace(dataset_name="vehicle_00nan")).config_source == "defaults"

    (config_dir / "vehicle_00nan.json").write_text(json.dumps({"DIM": 32}))
    (config_dir / "vehicle_00nan.imputation.json").write_text(json.dumps({"DIM": 64}))
    classification = resolve_training_request(Namespace(dataset_name="vehicle_00nan"))
    imputation = resolve_training_request(Namespace(dataset_name="vehicle_00nan", task="imputation"))
    assert classification.config_source == "datasets/hiperparams/vehicle/vehicle_00nan.json"
    assert imputation.config_source == "datasets/hiperparams/vehicle/vehicle_00nan.imputation.json"

    # A programmatic override (an Optuna trial) wins over any file and says so.
    overridden = resolve_training_request(
        Namespace(dataset_name="vehicle_00nan", task="imputation", hyperparams_override={"DIM": 8})
    )
    assert overridden.config_source == "override"
    assert overridden.hyperparameters.dimension == 8


def test_the_search_objective_population_is_a_choice_only_imputation_offers() -> None:
    """A study may rank its trials by the validation split's induced gaps instead of its
    masked cells (ADR 0008). Classification ranks by macro F1 and has neither, so the flag
    is refused there at parse time; left unspecified it stays unspecified, and the study
    resolves it to the masked population every earlier study used.
    """
    parser = build_training_parser()
    with_classification = parser.parse_args(
        ["--dataset_name", "credit-g_20nan", "--search_objective", "induced"]
    )
    with_imputation = parser.parse_args(
        ["--dataset_name", "credit-g_20nan", "--task", "imputation", "--search_objective", "induced"]
    )
    unspecified = parser.parse_args(["--dataset_name", "credit-g_20nan"])

    with pytest.raises(SystemExit, match="search_objective"):
        validate_parsed_args(with_classification)
    assert validate_parsed_args(with_imputation).search_objective == "induced"
    assert unspecified.search_objective is None
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset_name", "credit-g_20nan", "--search_objective", "test"])


def test_pretraining_keeps_its_embedding_target_unless_a_run_asks_for_another() -> None:
    """Every run before ADR 0011 regressed onto the clean embedding, so that stays the default.

    The flag wins over the configuration, as ``--lr_scheduler`` does, so one invocation can
    re-run any stored configuration under another objective.
    """
    parser = build_training_parser()
    imputation = ["--dataset_name", "credit-g_20nan", "--task", "imputation"]
    override = {"PRETRAIN_OBJECTIVE": "embedding_normalized"}

    default = resolve_training_request(parser.parse_args(imputation))
    from_flag = resolve_training_request(
        parser.parse_args([*imputation, "--pretrain_objective", "value"])
    )
    from_mapping = resolve_training_request(
        Namespace(dataset_name="credit-g_20nan", hyperparams_override=override)
    )
    flag_over_mapping = resolve_training_request(
        Namespace(
            dataset_name="credit-g_20nan", hyperparams_override=override, pretrain_objective="value"
        )
    )

    assert default.hyperparameters.pretraining_objective == "embedding"
    assert from_flag.hyperparameters.pretraining_objective == "value"
    assert from_mapping.hyperparameters.pretraining_objective == "embedding_normalized"
    assert flag_over_mapping.hyperparameters.pretraining_objective == "value"


def test_an_unknown_pretraining_objective_is_refused() -> None:
    with pytest.raises(SystemExit):
        build_training_parser().parse_args(
            ["--dataset_name", "vehicle_00nan", "--pretrain_objective", "contrastive"]
        )
    with pytest.raises(ValueError, match="Unknown pre-training objective"):
        Hyperparameters.from_mapping({"PRETRAIN_OBJECTIVE": "contrastive"})


def test_imputation_runs_share_the_baseline_cache_unless_told_not_to() -> None:
    """The cache lives at a fixed path under the checkout, never under a run's own output
    directory, which every study runner sets per cell."""
    parser = build_training_parser()
    imputation = ["--dataset_name", "credit-g_20nan", "--task", "imputation", "--output_dir", "cell/42"]

    shared = resolve_training_request(parser.parse_args(imputation))
    off = resolve_training_request(parser.parse_args([*imputation, "--no_baseline_cache"]))

    assert shared.runtime.baseline_cache_dir == Path("results") / "baseline_cache"
    assert off.runtime.baseline_cache_dir is None


def test_decode_patience_is_off_unless_an_imputation_run_asks_for_it() -> None:
    parser = build_training_parser()
    imputation = ["--dataset_name", "credit-g_20nan", "--task", "imputation"]

    default = resolve_training_request(parser.parse_args(imputation))
    from_flag = resolve_training_request(parser.parse_args([*imputation, "--decode_patience", "50"]))
    flag_over_mapping = resolve_training_request(
        Namespace(dataset_name="credit-g_20nan", hyperparams_override={"DECODE_PATIENCE": 20}, decode_patience=50)
    )

    assert default.hyperparameters.decode_patience == 0
    assert from_flag.hyperparameters.decode_patience == 50
    assert flag_over_mapping.hyperparameters.decode_patience == 50
    with pytest.raises(SystemExit, match="decode_patience"):
        validate_parsed_args(parser.parse_args(["--dataset_name", "vehicle_00nan", "--decode_patience", "5"]))
    with pytest.raises(ValueError, match="patience"):
        Hyperparameters(decode_patience=-1)
