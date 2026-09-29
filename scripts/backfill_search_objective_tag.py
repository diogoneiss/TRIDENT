#!/usr/bin/env python3
"""Stamp the ``search_objective`` tag on Optuna runs recorded before ADR 0008.

Every study before ADR 0008 ranked its trials by its task's default objective: the
masked validation cells (``validation/impute/masked/impute_score``) for imputation, macro
F1 for classification. This script stamps that value on every Optuna study and trial
(``is_optuna=true``) that lacks the tag, marks it ``search_objective_backfilled=true`` so
an inferred value can be told from a recorded one, and re-mirrors each stamped tree so
the mirror experiments (ADR 0006) carry the same tag. Runs no search produced, runs that
already carry the tag, and deleted runs are left alone; a second run finds nothing.

Usage (dry run by default, honours ``MLFLOW_TRACKING_URI``):

    uv run --python 3.11 python scripts/backfill_search_objective_tag.py
    uv run --python 3.11 python scripts/backfill_search_objective_tag.py --apply
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlflow.entities import Run, ViewType  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src.mlflow_utils import (  # noqa: E402
    IS_OPTUNA_TAG,
    MIRROR_EXPERIMENT_PREFIX,
    SEARCH_OBJECTIVE_BACKFILLED_TAG,
    SEARCH_OBJECTIVE_TAG,
    TASK_TAG,
    setup_mlflow,
)
from src.training.mirroring import mirror_run_tree  # noqa: E402
from src.training.types import DEFAULT_SEARCH_OBJECTIVE_POPULATION, task_spec, validation_objective_key  # noqa: E402

PARENT_RUN_ID_TAG = "mlflow.parentRunId"


def default_objective(task: str) -> str:
    """What a study of this task ranked by before the objective could be chosen."""
    if task == "imputation":
        return validation_objective_key(DEFAULT_SEARCH_OBJECTIVE_POPULATION)
    return task_spec(task).search_objective


@dataclass
class BackfillReport:
    tagged: dict[str, list[str]] = field(default_factory=dict)
    roots: set[str] = field(default_factory=set)

    @property
    def tagged_count(self) -> int:
        return sum(len(runs) for runs in self.tagged.values())


def _optuna_runs(client: MlflowClient) -> list[tuple[str, Run]]:
    """Active Optuna runs in the experiment families, never in the mirror experiments."""
    found: list[tuple[str, Run]] = []
    for experiment in client.search_experiments():
        if experiment.name.startswith(MIRROR_EXPERIMENT_PREFIX):
            continue
        page_token: str | None = None
        while True:
            page = client.search_runs(
                [experiment.experiment_id],
                filter_string=f"tags.{IS_OPTUNA_TAG} = 'true'",
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=1000,
                page_token=page_token,
            )
            found.extend((experiment.name, run) for run in page)
            page_token = page.token
            if not page_token:
                break
    return found


def backfill(client: MlflowClient, apply: bool) -> BackfillReport:
    report = BackfillReport()
    for experiment_name, run in _optuna_runs(client):
        if SEARCH_OBJECTIVE_TAG in run.data.tags:
            continue
        run_id = run.info.run_id
        report.tagged.setdefault(experiment_name, []).append(run_id)
        report.roots.add(run.data.tags.get(PARENT_RUN_ID_TAG, run_id))
        if apply:
            value = default_objective(run.data.tags.get(TASK_TAG, "classification"))
            client.set_tag(run_id, SEARCH_OBJECTIVE_TAG, value)
            client.set_tag(run_id, SEARCH_OBJECTIVE_BACKFILLED_TAG, "true")
    if apply:
        for root in sorted(report.roots):
            mirror_run_tree(client, root)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Write. Without it, only report what would be written.")
    parser.add_argument("--tracking_uri", default=None, help="Override MLFLOW_TRACKING_URI.")
    args = parser.parse_args(argv)

    tracking_uri = setup_mlflow(args.tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    report = backfill(client, apply=args.apply)

    mode = "APPLIED" if args.apply else "DRY RUN"
    print(f"[{mode}] tracking URI: {tracking_uri}")
    print(f"[{mode}] {SEARCH_OBJECTIVE_TAG} stamped on {report.tagged_count} Optuna run(s) in {len(report.roots)} tree(s)")
    for experiment_name in sorted(report.tagged):
        print(f"  {experiment_name:<24} {len(report.tagged[experiment_name])}")
    if not args.apply and report.tagged_count:
        print("Re-run with --apply to write the tags and re-mirror the trees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
