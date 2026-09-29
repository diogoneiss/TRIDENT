"""Stamping the search objective on Optuna runs recorded before ADR 0008."""

from pathlib import Path
import sys

import mlflow
from mlflow.tracking import MlflowClient
import pytest

from src.mlflow_utils import SEARCH_OBJECTIVE_BACKFILLED_TAG, SEARCH_OBJECTIVE_TAG

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
import backfill_search_objective_tag as backfill_script  # noqa: E402

MASKED = "validation/impute/masked/impute_score"


@pytest.fixture
def client(tmp_path, monkeypatch) -> MlflowClient:
    previous_tracking_uri = mlflow.get_tracking_uri()
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    try:
        yield MlflowClient(tracking_uri=tracking_uri)
    finally:
        mlflow.end_run()
        mlflow.set_tracking_uri(previous_tracking_uri)


def _seed_runs(client: MlflowClient) -> dict[str, str]:
    credit = client.create_experiment("TRIDENT/credit-g")
    vehicle = client.create_experiment("TRIDENT/vehicle")

    def run(experiment: str, **tags: str) -> str:
        return client.create_run(experiment, tags={"is_mirror": "false", **tags}).info.run_id

    study = run(credit, run_role="optuna_study", task="imputation", is_optuna="true")
    trial = run(credit, run_role="optuna_trial", task="imputation", is_optuna="true", **{"mlflow.parentRunId": study})
    classification_study = run(vehicle, run_role="optuna_study", task="classification", is_optuna="true")
    induced_study = run(credit, run_role="optuna_study", task="imputation", is_optuna="true", search_objective="validation/impute/induced/impute_score")
    comparison = run(credit, run_role="parent", task="imputation", is_optuna="false")
    for run_id in (study, trial, classification_study, induced_study, comparison):
        client.set_terminated(run_id)
    return {
        "study": study,
        "trial": trial,
        "classification_study": classification_study,
        "induced_study": induced_study,
        "comparison": comparison,
    }


def test_every_optuna_run_without_the_tag_gets_its_tasks_default_and_nothing_else_is_touched(
    client: MlflowClient,
) -> None:
    """Every study before ADR 0008 ranked imputation trials by the masked validation
    cells and classification trials by macro F1; the stamp says so, marked as inferred.
    A run already carrying the tag keeps its own value, and runs no search produced get
    nothing."""
    runs = _seed_runs(client)

    report = backfill_script.backfill(client, apply=True)

    assert report.tagged_count == 3
    for name, value in (("study", MASKED), ("trial", MASKED), ("classification_study", "f1_macro")):
        tags = client.get_run(runs[name]).data.tags
        assert (tags[SEARCH_OBJECTIVE_TAG], tags[SEARCH_OBJECTIVE_BACKFILLED_TAG]) == (value, "true"), name
    induced = client.get_run(runs["induced_study"]).data.tags
    assert induced[SEARCH_OBJECTIVE_TAG] == "validation/impute/induced/impute_score"
    assert SEARCH_OBJECTIVE_BACKFILLED_TAG not in induced
    assert SEARCH_OBJECTIVE_TAG not in client.get_run(runs["comparison"]).data.tags


def test_a_dry_run_writes_nothing_and_a_second_apply_finds_nothing(client: MlflowClient) -> None:
    runs = _seed_runs(client)

    dry = backfill_script.backfill(client, apply=False)
    assert dry.tagged_count == 3
    assert SEARCH_OBJECTIVE_TAG not in client.get_run(runs["study"]).data.tags

    backfill_script.backfill(client, apply=True)
    assert backfill_script.backfill(client, apply=True).tagged_count == 0


def test_the_mirror_of_a_stamped_study_carries_the_stamp(client: MlflowClient) -> None:
    """Mirrors copy their source's tags (ADR 0006), so a stamped tree is re-mirrored; a
    cross-experiment query must not find the source stamped and its mirror blank."""
    runs = _seed_runs(client)

    backfill_script.backfill(client, apply=True)

    mirror_experiment = client.get_experiment_by_name("TRIDENT/mirror/imputation")
    assert mirror_experiment is not None
    mirrors = client.search_runs([mirror_experiment.experiment_id], filter_string=f"tags.source_run_id = '{runs['study']}'")
    assert len(mirrors) == 1
    assert mirrors[0].data.tags[SEARCH_OBJECTIVE_TAG] == MASKED
