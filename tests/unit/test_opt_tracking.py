"""MLflow structure of an Optuna study: one study run, one light run per trial."""

from argparse import Namespace
import json
from pathlib import Path
from typing import Iterator

import mlflow
from mlflow.tracking import MlflowClient
import pytest

import opt


@pytest.fixture
def mlflow_backend(tmp_path, monkeypatch) -> Iterator[str]:
    previous_tracking_uri = mlflow.get_tracking_uri()
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    # A study reads the table's header to learn its column mix before the first trial
    # (ADR 0005). Training is stubbed, so only the header has to exist: vehicle, all
    # numerical, no categorical declaration.
    table = tmp_path / "datasets" / "processed_datasets" / "vehicle" / "vehicle_00nan.csv"
    table.parent.mkdir(parents=True)
    table.write_text("compactness,circularity,class\n")
    # ``run_hyperparameter_optimization`` rebinds the module-level name when
    # tracking is disabled; make sure that never leaks into other tests.
    monkeypatch.setattr(opt, "mlflow", opt.mlflow)
    try:
        yield tracking_uri
    finally:
        while mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous_tracking_uri)


class _TrainingStub:
    """Stand-in for ``train.main`` that records what each trial was asked to do."""

    def __init__(
        self,
        scores: dict[int, float],
        failing: set[int] = frozenset(),
        validation_scores: dict[int, float] | None = None,
        induced_validation_scores: dict[int, float] | None = None,
    ) -> None:
        self.scores = scores
        self.failing = failing
        # The validation split's induced gaps, scored only where a complete sibling exists.
        self.induced_validation_scores = induced_validation_scores or {}
        # An imputation run scores its validation split too when asked; unless a test
        # says otherwise, that score equals the test one.
        self.validation_scores = validation_scores or {}
        self.calls: list[Namespace] = []
        self.active_run_ids: list[str | None] = []

    def __call__(self, args, return_metrics: bool = False):
        self.calls.append(args)
        active = mlflow.active_run()
        self.active_run_ids.append(active.info.run_id if active is not None else None)
        trial_number = self._trial_number(active)
        if trial_number in self.failing:
            raise RuntimeError("boom")
        score = self.scores.get(trial_number, 0.9)
        # The real runner returns the metrics of whichever task the namespace names, so a
        # namespace that forgot its task gets classification metrics and no imputation key.
        if getattr(args, "task", None) == "imputation":
            metrics = {
                "fold": "single_split",
                "dataset": args.dataset_name,
                "impute/masked/impute_score": score,
                "validation/impute/masked/impute_score": self.validation_scores.get(
                    trial_number, score
                ),
            }
            if trial_number in self.induced_validation_scores:
                metrics["validation/impute/induced/impute_score"] = self.induced_validation_scores[trial_number]
            return metrics
        return {"fold": "single_split", "dataset": args.dataset_name, "f1_macro": score}

    @staticmethod
    def _trial_number(active) -> int | None:
        if active is None or "trial_number" not in active.data.tags:
            return None
        return int(active.data.tags["trial_number"])


def _optuna_args(**overrides) -> Namespace:
    values = dict(
        dataset_name="vehicle_00nan",
        n_trials=3,
        seed=42,
        output_dir="results",
        retrain_best=False,
        promote_best=False,
        lr_scheduler=None,
    )
    values.update(overrides)
    return Namespace(**values)


def _runs_by_name(client: MlflowClient) -> dict[str, mlflow.entities.Run]:
    experiment = client.get_experiment_by_name("TRIDENT/vehicle")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    return {run.info.run_name: run for run in runs}


def test_each_trial_trains_inside_its_own_nested_run(mlflow_backend, monkeypatch) -> None:
    stub = _TrainingStub(scores={0: 0.5, 1: 0.7, 2: 0.6}, failing={1})
    monkeypatch.setattr(opt, "train_main", stub)

    opt.run_hyperparameter_optimization(_optuna_args())

    trial_calls = [args for args in stub.calls if getattr(args, "mlflow_run_role", None)]
    assert len(trial_calls) == 3
    assert all(args.mlflow_run_role == "optuna_trial" for args in trial_calls)
    assert all(args.disable_mlflow is False for args in trial_calls)
    assert all(args.lr_scheduler is None for args in trial_calls)
    assert len(set(stub.active_run_ids)) == 3 and None not in stub.active_run_ids

    client = MlflowClient(tracking_uri=mlflow_backend)
    runs = _runs_by_name(client)
    study = next(run for name, run in runs.items() if name.startswith("optuna_vehicle_00nan_"))
    trials = {name: run for name, run in runs.items() if name.startswith("optuna_trial_")}
    assert set(trials) == {"optuna_trial_0", "optuna_trial_1", "optuna_trial_2"}
    assert study.data.tags["run_role"] == "optuna_study"
    assert study.data.tags["run_type"] == "optuna_study"
    assert study.data.tags["lr_scheduler"] == "cosine_legacy"
    # Unspecified, a classification study samples the full space and says so.
    assert study.data.tags["search_space"] == "full"
    assert study.data.params["search_space"] == "full"
    assert study.data.metrics["optuna/best_objective_value"] == pytest.approx(0.6)
    assert study.data.metrics["optuna/best_trial_number"] == 2.0
    assert study.data.tags["best_trial_number"] == "2"
    for name, trial in trials.items():
        assert trial.data.tags["mlflow.parentRunId"] == study.info.run_id
        assert trial.data.tags["run_role"] == "optuna_trial"
        assert trial.data.tags["run_type"] == "optuna_trial"
        assert trial.data.tags["trial_number"] == name.removeprefix("optuna_trial_")
        assert trial.data.tags["lr_scheduler"] == "cosine_legacy"
        assert "run_role" not in trial.data.params
    assert trials["optuna_trial_0"].data.tags["trial_status"] == "success"
    assert trials["optuna_trial_0"].data.metrics["optuna/objective_value"] == pytest.approx(0.5)
    assert trials["optuna_trial_1"].data.tags["trial_status"] == "failed"
    assert trials["optuna_trial_1"].data.tags["error"] == "boom"
    assert "optuna/objective_value" not in trials["optuna_trial_1"].data.metrics
    assert trials["optuna_trial_2"].data.tags["best_trial"] == "true"
    assert "best_trial" not in trials["optuna_trial_0"].data.tags
    assert "best_trial" not in trials["optuna_trial_1"].data.tags
    # A crashed trial is now pruned rather than scored, so MLflow marks its run failed
    # instead of finishing a run that quietly recorded the best possible value.
    assert trials["optuna_trial_1"].info.status == "FAILED"
    assert all(
        run.info.status == "FINISHED"
        for name, run in runs.items()
        if name != "optuna_trial_1"
    )
    assert mlflow.active_run() is None


def test_retraining_the_best_trial_links_a_normal_parent_run_to_the_study(
    mlflow_backend, monkeypatch
) -> None:
    stub = _TrainingStub(scores={0: 0.8, 1: 0.4})
    monkeypatch.setattr(opt, "train_main", stub)

    opt.run_hyperparameter_optimization(_optuna_args(n_trials=2, retrain_best=True))

    retrain = stub.calls[-1]
    assert getattr(retrain, "mlflow_run_role", None) is None
    assert stub.active_run_ids[-1] is None
    client = MlflowClient(tracking_uri=mlflow_backend)
    study = next(
        run
        for name, run in _runs_by_name(client).items()
        if name.startswith("optuna_vehicle_00nan_")
    )
    assert retrain.mlflow_tags == {"optuna_study_run_id": study.info.run_id}
    assert retrain.hyperparams_override == {
        key.removeprefix("best_"): value
        for key, value in study.data.params.items()
        if key.startswith("best_")
    } or retrain.hyperparams_override


def test_an_imputation_study_trains_the_imputation_task_in_every_trial(
    mlflow_backend, monkeypatch
) -> None:
    """The objective builds each trial's namespace itself, so the task has to be copied on.

    Without it every trial trains a classifier, looks up the imputation ranking metric,
    raises and is pruned: a study that finishes with zero scored trials. The retrain
    namespace is built the same way and must carry the task too.
    """
    stub = _TrainingStub(scores={0: 0.9, 1: 0.4})
    monkeypatch.setattr(opt, "train_main", stub)

    opt.run_hyperparameter_optimization(
        _optuna_args(task="imputation", n_trials=2, retrain_best=True)
    )

    assert [args.task for args in stub.calls] == ["imputation", "imputation", "imputation"]


def test_every_run_a_study_opens_says_which_task_and_that_a_search_made_it(
    mlflow_backend, monkeypatch
) -> None:
    """``task`` and ``is_optuna`` are dense on every other run kind (ADR 0004, decision 7).

    A study parent without them slips through any filter on either tag, and a trial that
    crashes before the runner sets its tags would be a run of no known task.
    """
    stub = _TrainingStub(scores={0: 0.9, 1: 0.4})
    monkeypatch.setattr(opt, "train_main", stub)

    opt.run_hyperparameter_optimization(_optuna_args(task="imputation", n_trials=2))

    runs = _runs_by_name(MlflowClient(tracking_uri=mlflow_backend))
    study = next(run for name, run in runs.items() if name.startswith("optuna_vehicle_00nan_"))
    trials = [run for name, run in runs.items() if name.startswith("optuna_trial_")]
    assert len(trials) == 2
    assert study.data.tags["task"] == "imputation"
    assert study.data.tags["is_optuna"] == "true"
    assert all(trial.data.tags["task"] == "imputation" for trial in trials)
    assert all(trial.data.tags["is_optuna"] == "true" for trial in trials)
    assert study.data.tags["is_mirror"] == "false"
    assert all(trial.data.tags["is_mirror"] == "false" for trial in trials)


def test_a_finished_study_mirrors_itself_and_its_trials(mlflow_backend, monkeypatch) -> None:
    """The study is a root, so it mirrors the whole tree when it closes (ADR 0006,
    decision 4); a trial never mirrors on its own, or its parent would not exist yet."""
    stub = _TrainingStub(scores={0: 0.9, 1: 0.4})
    monkeypatch.setattr(opt, "train_main", stub)

    opt.run_hyperparameter_optimization(_optuna_args(task="imputation", n_trials=2))

    client = MlflowClient(tracking_uri=mlflow_backend)
    sources = _runs_by_name(client)
    mirror_experiment = client.get_experiment_by_name("TRIDENT/mirror/imputation")
    assert mirror_experiment is not None
    mirrors = {run.data.tags["source_run_id"]: run for run in client.search_runs([mirror_experiment.experiment_id])}
    assert set(mirrors) == {run.info.run_id for run in sources.values()}
    study = next(run for name, run in sources.items() if name.startswith("optuna_vehicle_00nan_"))
    for name, source in sources.items():
        if name.startswith("optuna_trial_"):
            assert mirrors[source.info.run_id].data.tags["mlflow.parentRunId"] == mirrors[study.info.run_id].info.run_id
    assert mirrors[study.info.run_id].data.tags["is_mirror"] == "true"


def test_a_study_asked_for_no_mirror_gets_none(mlflow_backend, monkeypatch) -> None:
    stub = _TrainingStub(scores={0: 0.9, 1: 0.4})
    monkeypatch.setattr(opt, "train_main", stub)

    opt.run_hyperparameter_optimization(_optuna_args(task="imputation", n_trials=2, disable_mirror=True))

    client = MlflowClient(tracking_uri=mlflow_backend)
    assert client.get_experiment_by_name("TRIDENT/mirror/imputation") is None
    assert len(_runs_by_name(client)) == 3  # the study and its trials still exist


def test_a_finished_imputation_study_leaves_the_shared_config_alone(
    mlflow_backend, monkeypatch
) -> None:
    """Both tasks read datasets/hiperparams/<base>/<dataset>.json, and it names no task.

    The per-trial write already stays inside the study directory; the end-of-study write
    did not, so a *finished* imputation study still retuned every later classification
    run on that dataset, after every trial had carefully avoided doing so.
    """
    stub = _TrainingStub(scores={0: 0.9, 1: 0.4})
    monkeypatch.setattr(opt, "train_main", stub)

    opt.run_hyperparameter_optimization(_optuna_args(task="imputation", n_trials=2))

    assert not (Path("datasets") / "hiperparams").exists()
    study_dirs = list((Path("results") / "vehicle_00nan").glob("optuna_*"))
    assert len(study_dirs) == 1
    assert (study_dirs[0] / "best_hyperparameters.json").exists()


def test_a_study_with_the_same_seed_samples_the_same_trials(mlflow_backend, monkeypatch) -> None:
    """A study is an experiment: the seed that fixes every trial's training fixes the sampler.

    Separate output directories, because the study storage is named to the second and a
    second study in the same second would resume the first instead of starting over.
    """
    sampled: list[list[dict]] = []
    for output_dir in ("first", "second"):
        stub = _TrainingStub(scores={})
        monkeypatch.setattr(opt, "train_main", stub)
        opt.run_hyperparameter_optimization(
            _optuna_args(n_trials=3, disable_mlflow=True, output_dir=output_dir)
        )
        sampled.append([args.hyperparams_override for args in stub.calls])

    assert sampled[0] == sampled[1]


def test_the_configuration_a_study_hands_on_is_the_one_its_best_trial_trained_with(
    mlflow_backend, monkeypatch
) -> None:
    """The width is sampled as a multiple of the head count, so what Optuna records under
    ``DIM`` is that multiplier, not the width the trial trained with.

    Everything a study hands on (the retrain, the saved file, the ``best_`` params on the
    study run) must carry the trained configuration; a retrain from Optuna's record builds
    a model the winning trial never ran, and a head count that no longer divides the width.
    """
    stub = _TrainingStub(scores={0: 0.8, 1: 0.4})
    monkeypatch.setattr(opt, "train_main", stub)

    opt.run_hyperparameter_optimization(_optuna_args(n_trials=2, retrain_best=True))

    winning = stub.calls[0].hyperparams_override
    retrain = stub.calls[-1]
    assert winning["DIM"] % winning["HEADS"] == 0
    assert retrain.hyperparams_override == winning
    saved = next((Path("results") / "vehicle_00nan").glob("optuna_*/best_hyperparameters.json"))
    assert json.loads(saved.read_text()) == winning
    study = next(
        run
        for name, run in _runs_by_name(MlflowClient(tracking_uri=mlflow_backend)).items()
        if name.startswith("optuna_vehicle_00nan_")
    )
    assert study.data.params["best_DIM"] == str(winning["DIM"])


def test_an_imputation_study_records_and_samples_the_reduced_profile_by_default(
    mlflow_backend, monkeypatch
) -> None:
    """Unspecified, an imputation study resolves to ``reduced``; the choice is a sparse tag and
    param on the study parent and every trial, so studies compare within one profile.

    vehicle has no categorical column, so the loss balance is not among the sampled knobs.
    """
    stub = _TrainingStub(scores={0: 0.9, 1: 0.4})
    monkeypatch.setattr(opt, "train_main", stub)

    opt.run_hyperparameter_optimization(_optuna_args(task="imputation", n_trials=2))

    runs = _runs_by_name(MlflowClient(tracking_uri=mlflow_backend))
    study = next(run for name, run in runs.items() if name.startswith("optuna_vehicle_00nan_"))
    trials = [run for name, run in runs.items() if name.startswith("optuna_trial_")]
    assert study.data.tags["search_space"] == "reduced"
    assert study.data.params["search_space"] == "reduced"
    assert all(trial.data.tags["search_space"] == "reduced" for trial in trials)
    assert all(trial.data.params["search_space"] == "reduced" for trial in trials)
    assert all(
        set(args.hyperparams_override)
        == {"PROB_MASCARA", "LR_DECODE", "WEIGHT_DECAY_DECODE", "DROPOUT"}
        for args in stub.calls
    )


def test_an_imputation_trial_is_ranked_by_its_validation_score_not_its_test_score(
    mlflow_backend, monkeypatch
) -> None:
    """The test split never chooses hyperparameters (ADR 0005, decision 3).

    The trial with the lower validation score wins although its test score is the worse
    of the two; every trial asks the runner for the validation score, and the retrain,
    which is a normal comparable run, does not.
    """
    stub = _TrainingStub(scores={0: 0.9, 1: 0.2}, validation_scores={0: 0.3, 1: 0.6})
    monkeypatch.setattr(opt, "train_main", stub)

    opt.run_hyperparameter_optimization(
        _optuna_args(task="imputation", n_trials=2, retrain_best=True)
    )

    study = next(
        run
        for name, run in _runs_by_name(MlflowClient(tracking_uri=mlflow_backend)).items()
        if name.startswith("optuna_vehicle_00nan_")
    )
    assert study.data.metrics["optuna/best_objective_value"] == pytest.approx(0.3)
    assert study.data.metrics["optuna/best_trial_number"] == 0.0
    assert all(args.score_search_objective is True for args in stub.calls[:2])
    assert getattr(stub.calls[-1], "score_search_objective", False) is False


def test_disabled_tracking_runs_the_study_without_touching_the_store(
    mlflow_backend, monkeypatch
) -> None:
    stub = _TrainingStub(scores={0: 0.5, 1: 0.6})
    monkeypatch.setattr(opt, "train_main", stub)

    opt.run_hyperparameter_optimization(_optuna_args(n_trials=2, disable_mlflow=True))

    assert [args.disable_mlflow for args in stub.calls] == [True, True]
    assert [args.mlflow_run_role for args in stub.calls] == ["optuna_trial", "optuna_trial"]
    client = MlflowClient(tracking_uri=mlflow_backend)
    assert client.get_experiment_by_name("TRIDENT/vehicle") is None


def test_a_classification_study_no_longer_writes_the_shared_file_on_its_own(
    mlflow_backend, monkeypatch
) -> None:
    """Backlog I2: the study used to overwrite datasets/hiperparams/<base>/<dataset>.json
    at every improvement and again at the end, destroying any hand-written file there
    and leaving a mid-search best behind an interrupted study.

    Now the running best and the final best stay inside the study's own directory for
    both tasks, and only ``--promote_best`` publishes a result (ADR 0005, decision 5).
    """
    stub = _TrainingStub(scores={0: 0.5, 1: 0.7})
    monkeypatch.setattr(opt, "train_main", stub)

    opt.run_hyperparameter_optimization(_optuna_args(n_trials=2))

    assert not (Path("datasets") / "hiperparams").exists()
    study_dirs = list((Path("results") / "vehicle_00nan").glob("optuna_*"))
    assert len(study_dirs) == 1
    running_best = json.loads((study_dirs[0] / "vehicle_00nan.json").read_text())
    final_best = json.loads((study_dirs[0] / "best_hyperparameters.json").read_text())
    assert running_best == final_best


def test_promotion_writes_a_complete_imputation_configuration_where_the_task_will_find_it(
    mlflow_backend, monkeypatch
) -> None:
    """``--promote_best`` on an imputation study writes ``<dataset>.imputation.json``, the
    file an imputation run reads first, and leaves the shared file alone.

    The file is complete: every key the task uses, the held values written out at the
    defaults they were held at, and ``LR_SCHEDULER`` set to the schedule the study ran
    under, so the file cannot silently move if a default changes (ADR 0005, decision 5).
    """
    stub = _TrainingStub(scores={0: 0.9, 1: 0.4})
    monkeypatch.setattr(opt, "train_main", stub)

    opt.run_hyperparameter_optimization(
        _optuna_args(task="imputation", n_trials=2, promote_best=True, lr_scheduler="cosine")
    )

    config_dir = Path("datasets") / "hiperparams" / "vehicle"
    assert not (config_dir / "vehicle_00nan.json").exists()
    promoted = json.loads((config_dir / "vehicle_00nan.imputation.json").read_text())
    assert set(promoted) == {
        "DIM", "HIDDEN_DIM", "HEADS", "LAYERS", "DIM_FEED", "DROPOUT", "EPOCHS_PRE",
        "BATCH", "LR_PRE", "WEIGHT_DECAY_PRE", "PROB_MASCARA", "LR_SCHEDULER",
        "EPOCHS_DECODE", "LR_DECODE", "WEIGHT_DECAY_DECODE", "LAMBDA_NUM", "EVAL_MASK_RATE",
        "DECODE_PATIENCE", "DECODER_HEADS", "PRETRAIN_OBJECTIVE",
    }
    assert promoted["DECODE_PATIENCE"] == 0
    assert promoted["DECODER_HEADS"] == "batched"
    # Held at the task's defaults by the reduced profile (vehicle is all numerical, so the
    # loss balance is held too), and written out rather than left to the reader.
    assert promoted["DIM"] == 128
    assert promoted["EPOCHS_PRE"] == 300
    assert promoted["EPOCHS_DECODE"] == 450
    assert promoted["PRETRAIN_OBJECTIVE"] == "embedding_normalized"
    assert promoted["LAMBDA_NUM"] == 1.0
    assert promoted["LR_SCHEDULER"] == "cosine"
    # The sampled values are the winning trial's (0.4 beats 0.9 when minimising).
    client = MlflowClient(tracking_uri=mlflow_backend)
    study = next(
        run
        for name, run in _runs_by_name(client).items()
        if name.startswith("optuna_vehicle_00nan_")
    )
    assert study.data.metrics["optuna/best_trial_number"] == 1.0
    assert promoted["LR_DECODE"] == float(study.data.params["best_LR_DECODE"])
    assert promoted["PROB_MASCARA"] == float(study.data.params["best_PROB_MASCARA"])


def test_an_imputation_study_holds_and_promotes_the_task_defaults_unless_flags_say_otherwise(
    mlflow_backend, monkeypatch
) -> None:
    """ADR 0013: a search holds a knob where a plain imputation run would put it.

    A study without ``--lr_scheduler`` follows imputation's per-epoch cosine and says so on
    every run it opens, and its promoted file writes the held objective, decode length and
    schedule out. The configuration flags reach every trial and the promoted file, so the
    search before ADR 0013 stays one command away.
    """
    stub = _TrainingStub(scores={0: 0.9, 1: 0.4})
    monkeypatch.setattr(opt, "train_main", stub)
    promoted_path = Path("datasets") / "hiperparams" / "vehicle" / "vehicle_00nan.imputation.json"

    opt.run_hyperparameter_optimization(_optuna_args(task="imputation", n_trials=2, promote_best=True))

    promoted = json.loads(promoted_path.read_text())
    assert (promoted["PRETRAIN_OBJECTIVE"], promoted["EPOCHS_DECODE"], promoted["LR_SCHEDULER"]) == (
        "embedding_normalized", 450, "cosine",
    )
    runs = _runs_by_name(MlflowClient(tracking_uri=mlflow_backend))
    assert {run.data.tags["lr_scheduler"] for run in runs.values()} == {"cosine"}

    stub.calls.clear()
    opt.run_hyperparameter_optimization(
        _optuna_args(
            task="imputation", n_trials=2, promote_best=True, lr_scheduler="cosine_legacy",
            pretrain_objective="embedding", decode_epochs=150, decode_patience=20,
        )
    )

    assert len(stub.calls) == 2
    assert all(
        (call.pretrain_objective, call.decode_epochs, call.decode_patience, call.lr_scheduler)
        == ("embedding", 150, 20, "cosine_legacy")
        for call in stub.calls
    )
    promoted = json.loads(promoted_path.read_text())
    assert (promoted["PRETRAIN_OBJECTIVE"], promoted["EPOCHS_DECODE"], promoted["LR_SCHEDULER"]) == (
        "embedding", 150, "cosine_legacy",
    )
    assert promoted["DECODE_PATIENCE"] == 20


def test_promotion_writes_a_complete_classification_configuration_to_the_shared_file(
    mlflow_backend, monkeypatch
) -> None:
    """Classification's promotion target is the shared file every classification run
    read before ADR 0005, and it is complete like imputation's: the sampled mapping
    carries neither ``LR_SCHEDULER`` (a command-line choice) nor ``LABELS`` (never
    sampled), so a file made of the sampled keys alone would leave both to the reader.
    """
    stub = _TrainingStub(scores={0: 0.5, 1: 0.7})
    monkeypatch.setattr(opt, "train_main", stub)

    opt.run_hyperparameter_optimization(_optuna_args(n_trials=2, promote_best=True))

    config_dir = Path("datasets") / "hiperparams" / "vehicle"
    assert not (config_dir / "vehicle_00nan.imputation.json").exists()
    promoted = json.loads((config_dir / "vehicle_00nan.json").read_text())
    assert set(promoted) == {
        "DIM", "HIDDEN_DIM", "HEADS", "LAYERS", "DIM_FEED", "DROPOUT", "EPOCHS_PRE",
        "BATCH", "LR_PRE", "WEIGHT_DECAY_PRE", "PROB_MASCARA", "EPOCH_FINE", "LR_FINE",
        "WEIGHT_DECAY_FINE", "LABELS", "LR_SCHEDULER",
    }
    # No schedule was named, so the file pins the one the study ran under: the legacy
    # default, spelled out rather than left to whatever the default is later.
    assert promoted["LR_SCHEDULER"] == "cosine_legacy"
    client = MlflowClient(tracking_uri=mlflow_backend)
    study = next(
        run
        for name, run in _runs_by_name(client).items()
        if name.startswith("optuna_vehicle_00nan_")
    )
    assert study.data.metrics["optuna/best_trial_number"] == 1.0
    assert promoted["DIM"] == int(study.data.params["best_DIM"])
    assert promoted["LR_FINE"] == float(study.data.params["best_LR_FINE"])


def test_a_finished_study_ranks_the_knobs_it_sampled(mlflow_backend, monkeypatch) -> None:
    """A study says which of the knobs it sampled moved the objective, so the reduction
    can be checked against what the search actually found (ADR 0005, decision 6).

    fANOVA importances land on the study parent as one metric per sampled knob (they sum
    to one) and as an ``importance.json`` artifact. Vehicle is all numerical, so the
    reduced profile samples four knobs and the loss balance is not among them.
    """
    stub = _TrainingStub(scores={0: 0.9, 1: 0.4, 2: 0.7})
    monkeypatch.setattr(opt, "train_main", stub)

    opt.run_hyperparameter_optimization(_optuna_args(task="imputation", n_trials=3))

    client = MlflowClient(tracking_uri=mlflow_backend)
    study = next(
        run
        for name, run in _runs_by_name(client).items()
        if name.startswith("optuna_vehicle_00nan_")
    )
    importances = {
        key.removeprefix("optuna/importance/"): value
        for key, value in study.data.metrics.items()
        if key.startswith("optuna/importance/")
    }
    assert set(importances) == {"PROB_MASCARA", "LR_DECODE", "WEIGHT_DECAY_DECODE", "DROPOUT"}
    assert sum(importances.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(value >= 0.0 for value in importances.values())
    artifacts = {info.path for info in client.list_artifacts(study.info.run_id)}
    assert "importance.json" in artifacts


def test_a_study_too_small_to_rank_still_finishes(mlflow_backend, monkeypatch) -> None:
    """One trial gives fANOVA nothing to rank. The study still records its best and
    promotes if asked; it logs no importance and does not fail after the trial ran.
    """
    stub = _TrainingStub(scores={0: 0.9})
    monkeypatch.setattr(opt, "train_main", stub)

    opt.run_hyperparameter_optimization(_optuna_args(task="imputation", n_trials=1))

    client = MlflowClient(tracking_uri=mlflow_backend)
    study = next(
        run
        for name, run in _runs_by_name(client).items()
        if name.startswith("optuna_vehicle_00nan_")
    )
    assert study.data.metrics["optuna/best_trial_number"] == 0.0
    assert not any(key.startswith("optuna/importance/") for key in study.data.metrics)
    assert {info.path for info in client.list_artifacts(study.info.run_id)} == set()


def _with_gaps_variant(tmp_path: Path) -> None:
    """A variant with gaps beside the fixture's complete table, so an induced study has
    a sibling to score against."""
    table = tmp_path / "datasets" / "processed_datasets" / "vehicle" / "vehicle_20nan.csv"
    table.write_text("compactness,circularity,class\n")


def test_an_induced_study_ranks_trials_by_the_validation_splits_own_gaps(
    mlflow_backend, monkeypatch, tmp_path
) -> None:
    """Asked for the induced objective, the study ranks by the validation split's gaps:
    trial 1 wins on them although trial 0 is better on the masked cells. Both run kinds
    say which objective chose them, as the metric key the study read (ADR 0008)."""
    _with_gaps_variant(tmp_path)
    stub = _TrainingStub(
        scores={0: 0.5, 1: 0.5},
        validation_scores={0: 0.3, 1: 0.6},
        induced_validation_scores={0: 0.9, 1: 0.4},
    )
    monkeypatch.setattr(opt, "train_main", stub)

    opt.run_hyperparameter_optimization(
        _optuna_args(dataset_name="vehicle_20nan", task="imputation", n_trials=2, search_objective="induced")
    )

    runs = _runs_by_name(MlflowClient(tracking_uri=mlflow_backend))
    study = next(run for name, run in runs.items() if name.startswith("optuna_vehicle_20nan_"))
    trials = [run for name, run in runs.items() if name.startswith("optuna_trial_")]
    key = "validation/impute/induced/impute_score"
    assert study.data.metrics["optuna/best_objective_value"] == pytest.approx(0.4)
    assert study.data.metrics["optuna/best_trial_number"] == 1.0
    assert study.data.tags["search_objective"] == key
    assert len(trials) == 2 and all(trial.data.tags["search_objective"] == key for trial in trials)
    assert study.data.params["search_objective"] == key


def test_a_study_ranks_by_the_masked_cells_unless_asked_otherwise(mlflow_backend, monkeypatch) -> None:
    """Every study before ADR 0008 ranked by the masked validation cells; one that does not
    ask keeps doing so, and says so."""
    stub = _TrainingStub(scores={0: 0.5, 1: 0.5}, validation_scores={0: 0.3, 1: 0.6})
    monkeypatch.setattr(opt, "train_main", stub)

    opt.run_hyperparameter_optimization(_optuna_args(task="imputation", n_trials=2))

    study = next(
        run
        for name, run in _runs_by_name(MlflowClient(tracking_uri=mlflow_backend)).items()
        if name.startswith("optuna_vehicle_00nan_")
    )
    assert study.data.metrics["optuna/best_trial_number"] == 0.0
    assert study.data.tags["search_objective"] == "validation/impute/masked/impute_score"


def test_an_induced_study_on_a_complete_variant_is_refused_before_any_trial(
    mlflow_backend, monkeypatch
) -> None:
    """A ``_00nan`` table has no gaps, so no induced population to rank by: every trial
    would fail on a missing key. Refused before the study opens, naming the reason."""
    stub = _TrainingStub(scores={})
    monkeypatch.setattr(opt, "train_main", stub)

    with pytest.raises(ValueError, match="induced"):
        opt.run_hyperparameter_optimization(
            _optuna_args(task="imputation", n_trials=2, search_objective="induced")
        )

    assert stub.calls == []
    client = MlflowClient(tracking_uri=mlflow_backend)
    experiment = client.get_experiment_by_name("TRIDENT/vehicle")
    assert experiment is None or client.search_runs([experiment.experiment_id]) == []
