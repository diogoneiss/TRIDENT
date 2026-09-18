"""Mirroring run trees into the derived per-task experiments (ADR 0006)."""

from pathlib import Path
from typing import Iterator

import mlflow
import pandas as pd
import pytest
from mlflow.entities import Run, ViewType
from mlflow.tracking import MlflowClient

import src.training.tracking as tracking
from src.training.mirroring import mirror_run_tree, sync_store
from src.training.summary import summarize_cross_validation
from src.training.tracking import create_tracker
from src.training.types import FoldResult, PreparedDataset, TrackingArtifactPaths

PARENT_RUN_ID_TAG = "mlflow.parentRunId"


@pytest.fixture()
def store(tmp_path, monkeypatch) -> Iterator[str]:
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.chdir(tmp_path)
    try:
        yield tracking_uri
    finally:
        mlflow.end_run()
        mlflow.set_tracking_uri(previous)


def _dataset(tmp_path: Path) -> PreparedDataset:
    frame = pd.DataFrame({"feature": [0.0, 1.0], "class": [0, 1]})
    frame.attrs["dataset_name"] = "vehicle_00nan"
    source_path = tmp_path / "datasets" / "processed_datasets" / "vehicle" / "vehicle_00nan.csv"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(source_path, index=False)
    splits_path = tmp_path / "vehicle_split.json"
    splits_path.write_text("{}")
    return PreparedDataset(
        frame=frame,
        label_column="class",
        categorical_columns=(),
        numerical_columns=("feature",),
        label_classes=("car", "van"),
        source_path=source_path,
        splits_path=splits_path,
    )


def _artifact_paths(tmp_path: Path) -> TrackingArtifactPaths:
    paths = TrackingArtifactPaths(
        raw_fold_metrics_csv=tmp_path / "results" / "metrics" / "raw_fold_metrics.csv",
        summary_json=tmp_path / "results" / "metrics" / "cv_summary.json",
        manifest_json=tmp_path / "results" / "tracking" / "diagnostic_manifest.json",
        provenance_json=tmp_path / "results" / "data" / "provenance.json",
    )
    for path in (paths.raw_fold_metrics_csv, paths.summary_json, paths.manifest_json, paths.provenance_json):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name)
    return paths


def _record(active_tracker, tmp_path: Path, fold: int, f1_macro: float):
    with active_tracker.fold_run(fold=fold, cv_folds=2, dataset_name="vehicle_00nan") as fold_tracker:
        for step in (1, 2):
            fold_tracker.log_metrics(
                {
                    "pretrain/train_loss": 1.0 + fold / step,
                    "pretrain/val_loss": 2.0 + fold / step,
                    "finetune/train_loss": 3.0 + fold / step,
                    "finetune/val_loss": 4.0 + fold / step,
                },
                step=step,
            )
        fold_tracker.log_metrics({"test/f1_macro": f1_macro, "test/accuracy": 0.5 + f1_macro / 2})
        artifact = tmp_path / f"fold_{fold}.txt"
        artifact.write_text(f"fold {fold}")
        fold_tracker.log_artifact(str(artifact), artifact_path="diagnostics")
    return fold_tracker.to_record(
        FoldResult(fold, "vehicle_00nan", {"accuracy": 0.5 + f1_macro / 2, "f1_macro": f1_macro})
    )


def _source_tree(tmp_path: Path, task: str = "imputation", mirror_runs: bool = False) -> None:
    """One comparison parent with two diagnostic children, logged by the real tracker."""
    tracker = create_tracker(enabled=True, mirror_runs=mirror_runs)
    dataset = _dataset(tmp_path)
    with tracker.parent_run(
        dataset_name="vehicle_00nan",
        seed=42,
        cv_folds=2,
        hyperparameters={"DIM": 128, "LR_DECODE": 0.001},
        lr_scheduler="cosine",
        environment={"device": "cpu", "torch_version": "2.0"},
        task=task,
    ) as active:
        records = [_record(active, tmp_path, 1, 0.4), _record(active, tmp_path, 2, 0.6)]
        active.log_prepared_dataset(dataset)
        active.finalize_cross_validation(
            records, summarize_cross_validation(records), dataset, _artifact_paths(tmp_path)
        )


def _runs_of(client: MlflowClient, experiment_name: str) -> list[Run]:
    experiment = client.get_experiment_by_name(experiment_name)
    assert experiment is not None, experiment_name
    return [
        client.get_run(run.info.run_id)
        for run in client.search_runs([experiment.experiment_id], run_view_type=ViewType.ALL)
    ]


def _root(runs: list[Run]) -> Run:
    return next(run for run in runs if PARENT_RUN_ID_TAG not in run.data.tags)


def _history(client: MlflowClient, run: Run) -> dict[str, list[tuple[float, int, int]]]:
    return {
        key: sorted((m.value, m.step, m.timestamp) for m in client.get_metric_history(run.info.run_id, key))
        for key in run.data.metrics
    }


def test_a_mirrored_tree_matches_its_source_field_by_field(store, tmp_path) -> None:
    """A mirror run that differed from its source would be read as a real number.

    Everything but artifacts is copied, and only ``mlflow.parentRunId`` is rewritten, to
    the mirror of the parent, so the UI nests mirrors as it nests sources.
    """
    _source_tree(tmp_path)
    client = MlflowClient(tracking_uri=store)
    sources = _runs_of(client, "TRIDENT/vehicle")
    source_root = _root(sources)

    mirror_run_tree(client, source_root.info.run_id)

    mirrors = _runs_of(client, "TRIDENT/mirror/imputation")
    assert len(mirrors) == len(sources) == 3
    by_source = {run.data.tags["source_run_id"]: run for run in mirrors}
    for source in sources:
        mirror = by_source[source.info.run_id]
        expected_tags = {
            key: value for key, value in source.data.tags.items() if key != PARENT_RUN_ID_TAG
        }
        expected_tags["is_mirror"] = "true"
        expected_tags["source_run_id"] = source.info.run_id
        if PARENT_RUN_ID_TAG in source.data.tags:
            expected_tags[PARENT_RUN_ID_TAG] = by_source[source.data.tags[PARENT_RUN_ID_TAG]].info.run_id
        assert mirror.data.tags == expected_tags
        assert mirror.data.params == source.data.params
        assert mirror.data.metrics == source.data.metrics
        assert _history(client, mirror) == _history(client, source)
        assert mirror.info.run_name == source.info.run_name
        assert mirror.info.status == source.info.status == "FINISHED"
        assert mirror.info.start_time == source.info.start_time
        assert mirror.info.end_time == source.info.end_time
        assert [
            (d.dataset.name, d.dataset.digest, sorted((t.key, t.value) for t in d.tags))
            for d in mirror.inputs.dataset_inputs
        ] == [
            (d.dataset.name, d.dataset.digest, sorted((t.key, t.value) for t in d.tags))
            for d in source.inputs.dataset_inputs
        ]
        assert client.list_artifacts(mirror.info.run_id) == []
    assert len(source_root.inputs.dataset_inputs) == 1  # the seam saw a real lineage


def test_mirroring_the_same_tree_twice_changes_nothing(store, tmp_path) -> None:
    """A second pass is an update, not a duplicate: the store accepts a copy of every
    metric row without a word, and a sync that ran twice would double every series."""
    _source_tree(tmp_path)
    client = MlflowClient(tracking_uri=store)
    source_root = _root(_runs_of(client, "TRIDENT/vehicle"))

    first = mirror_run_tree(client, source_root.info.run_id)
    after_first = {run.info.run_id: _history(client, run) for run in _runs_of(client, "TRIDENT/mirror/imputation")}
    second = mirror_run_tree(client, source_root.info.run_id)
    after_second = {run.info.run_id: _history(client, run) for run in _runs_of(client, "TRIDENT/mirror/imputation")}

    assert len(first.created) == 3 and first.updated == []
    assert second.created == [] and len(second.updated) == 3
    assert after_second == after_first  # same run ids, same series, same lengths


def test_a_refreshed_mirror_follows_its_source(store, tmp_path) -> None:
    """Tags are backfilled onto sources after the fact (``task``, ``lr_scheduler`` were);
    a mirror that kept the first copy would silently disagree with the family."""
    _source_tree(tmp_path)
    client = MlflowClient(tracking_uri=store)
    source_root = _root(_runs_of(client, "TRIDENT/vehicle"))
    mirror_run_tree(client, source_root.info.run_id)
    mirror_root = _root(_runs_of(client, "TRIDENT/mirror/imputation"))

    client.set_tag(source_root.info.run_id, "lr_scheduler_backfilled", "true")
    client.delete_tag(source_root.info.run_id, "torch_version")
    client.delete_run(mirror_root.info.run_id)  # a mirror deleted by hand comes back
    mirror_run_tree(client, source_root.info.run_id)

    refreshed = client.get_run(mirror_root.info.run_id)
    assert refreshed.info.lifecycle_stage == "active"
    assert refreshed.data.tags["lr_scheduler_backfilled"] == "true"
    assert "torch_version" not in refreshed.data.tags
    assert refreshed.data.tags["source_run_id"] == source_root.info.run_id
    assert refreshed.data.tags["is_mirror"] == "true"


def _bare_run(client: MlflowClient, experiment: str, tags: dict[str, str], status: str = "FINISHED") -> str:
    """A run logged outside the tracker, the way the 2026 legacy runs and a crash look."""
    found = client.get_experiment_by_name(experiment)
    experiment_id = found.experiment_id if found is not None else client.create_experiment(experiment)
    run = client.create_run(experiment_id, tags=tags)
    client.log_metric(run.info.run_id, "test/f1_macro", 0.5)
    if status != "RUNNING":
        client.set_terminated(run.info.run_id, status=status)
    return str(run.info.run_id)


def test_a_store_sync_mirrors_every_root_and_only_the_roots_that_finished(store, tmp_path) -> None:
    """The script's contract: every family is scanned, a ``RUNNING`` source waits, a
    ``FAILED`` one is mirrored with its status, a deleted source loses its mirror, an
    untagged source is stamped ``is_mirror=false``, and a mirror is never a source."""
    _source_tree(tmp_path)
    client = MlflowClient(tracking_uri=store)
    running = _bare_run(client, "TRIDENT/vehicle", {"task": "classification"}, status="RUNNING")
    failed = _bare_run(client, "TRIDENT/vehicle", {"task": "classification"}, status="FAILED")
    doomed = _bare_run(client, "TRIDENT/kc2", {"task": "classification", "is_mirror": "false"})
    mirror_run_tree(client, doomed)
    client.delete_run(doomed)

    report = sync_store(client, apply=True)
    again = sync_store(client, apply=True)

    imputation = _runs_of(client, "TRIDENT/mirror/imputation")
    classification = _runs_of(client, "TRIDENT/mirror/classification")
    assert len(imputation) == 3
    active_classification = [run for run in classification if run.info.lifecycle_stage == "active"]
    assert [run.data.tags["source_run_id"] for run in active_classification] == [failed]
    assert active_classification[0].info.status == "FAILED"
    assert report.skipped == [running]
    assert report.deleted == [doomed]
    assert sorted(report.stamped) == sorted([running, failed])
    assert client.get_run(failed).data.tags["is_mirror"] == "false"
    assert client.get_run(running).data.tags["is_mirror"] == "false"
    assert again.created == [] and again.stamped == [] and again.deleted == []
    assert len(_runs_of(client, "TRIDENT/mirror/imputation")) == 3  # no mirror of a mirror


def test_a_dry_run_reports_without_writing(store, tmp_path) -> None:
    _source_tree(tmp_path)
    client = MlflowClient(tracking_uri=store)
    untagged = _bare_run(client, "TRIDENT/vehicle", {"task": "classification"})

    report = sync_store(client, apply=False)

    assert len(report.created) == 4 and report.stamped == [untagged]
    assert client.get_experiment_by_name("TRIDENT/mirror/imputation") is None
    assert client.get_experiment_by_name("TRIDENT/mirror/classification") is None
    assert "is_mirror" not in client.get_run(untagged).data.tags


def test_a_finished_parent_run_is_mirrored_when_it_closes(store, tmp_path) -> None:
    """The live path: no script needed for a run that just finished (ADR 0006, decision 4)."""
    _source_tree(tmp_path, mirror_runs=True)
    client = MlflowClient(tracking_uri=store)

    mirrors = _runs_of(client, "TRIDENT/mirror/imputation")

    sources = {run.info.run_id for run in _runs_of(client, "TRIDENT/vehicle")}
    assert {run.data.tags["source_run_id"] for run in mirrors} == sources
    assert len(mirrors) == 3


def test_a_run_that_asked_for_no_mirror_gets_none(store, tmp_path) -> None:
    _source_tree(tmp_path, mirror_runs=False)
    client = MlflowClient(tracking_uri=store)

    assert client.get_experiment_by_name("TRIDENT/mirror/imputation") is None


def test_a_run_that_failed_is_left_to_the_script(store, tmp_path) -> None:
    """Only the success path mirrors live; the trainer's failure path is not touched."""
    tracker = create_tracker(enabled=True, mirror_runs=True)
    with pytest.raises(RuntimeError, match="boom"):
        with tracker.parent_run(
            dataset_name="vehicle_00nan", seed=42, cv_folds=None, hyperparameters={}, lr_scheduler="cosine"
        ):
            raise RuntimeError("boom")
    client = MlflowClient(tracking_uri=store)

    assert _root(_runs_of(client, "TRIDENT/vehicle")).info.status == "FAILED"
    assert client.get_experiment_by_name("TRIDENT/mirror/classification") is None


def test_a_mirroring_failure_never_fails_the_training(store, tmp_path, monkeypatch, caplog) -> None:
    """The last step of a six-hour run must not be able to lose it: a warning names the
    source run, which stays FINISHED in its family for the script to repair."""
    def explode(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(tracking, "mirror_run_tree", explode)
    with caplog.at_level("WARNING"):
        _source_tree(tmp_path, mirror_runs=True)
    client = MlflowClient(tracking_uri=store)

    source_root = _root(_runs_of(client, "TRIDENT/vehicle"))
    assert source_root.info.status == "FINISHED"
    assert client.get_experiment_by_name("TRIDENT/mirror/imputation") is None
    assert any(source_root.info.run_id in record.message and "database is locked" in record.message for record in caplog.records)


def test_the_script_defaults_to_a_dry_run_and_honours_the_task_filter(store, tmp_path, capsys) -> None:
    """The CLI plumbing over ``sync_store``: dry run by default, ``--apply`` writes, and
    ``--task`` leaves the other task's roots alone."""
    from scripts.mirror_runs import main

    _source_tree(tmp_path)
    client = MlflowClient(tracking_uri=store)
    _bare_run(client, "TRIDENT/vehicle", {"task": "classification"})

    assert main(["--tracking_uri", store]) == 0
    dry = capsys.readouterr().out
    assert main(["--tracking_uri", store, "--task", "imputation", "--apply"]) == 0
    applied = capsys.readouterr().out

    assert "[DRY RUN] mirror runs: 4 created" in dry
    assert "[APPLIED] mirror runs: 3 created" in applied
    assert len(_runs_of(client, "TRIDENT/mirror/imputation")) == 3
    assert client.get_experiment_by_name("TRIDENT/mirror/classification") is None
