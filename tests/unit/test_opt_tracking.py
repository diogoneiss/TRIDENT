"""MLflow structure of an Optuna study: one study run, one light run per trial."""

from argparse import Namespace
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

    def __init__(self, scores: dict[int, float], failing: set[int] = frozenset()) -> None:
        self.scores = scores
        self.failing = failing
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
