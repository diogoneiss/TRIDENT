#!/usr/bin/env python3
"""Stamp the ``decode_mask_weighting`` tag on imputation runs recorded before it (E65).

Every imputation run before E65 hid its decode-stage training cells uniformly. This script stamps
``decode_mask_weighting=uniform`` on every imputation run that trained a decode stage of its own (a
CV or single-split parent, or an Optuna trial) and lacks the tag. It marks each one
``decode_mask_weighting_backfilled=true``, so an inferred value can be told from a recorded one, and
tags each stamped run's mirror the same way (ADR 0006). Classification runs, diagnostic children,
study runs, runs that already carry the tag and deleted runs are left alone; a second run finds
nothing.

Usage (dry run by default, honours ``MLFLOW_TRACKING_URI``):

    uv run --python 3.11 python scripts/backfill_decode_mask_weighting_tag.py
    uv run --python 3.11 python scripts/backfill_decode_mask_weighting_tag.py --apply
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
    MIRROR_EXPERIMENT_PREFIX,
    SOURCE_RUN_ID_TAG,
    DECODE_MASK_WEIGHTING_BACKFILLED_TAG,
    DECODE_MASK_WEIGHTING_TAG,
    TASK_TAG,
    setup_mlflow,
)
from src.training.mirroring import mirror_run_tree  # noqa: E402

PARENT_RUN_ID_TAG = "mlflow.parentRunId"
# The run roles whose run trained a decode stage of its own.
PRETRAINING_ROLES = ("parent", "optuna_trial")


@dataclass
class BackfillReport:
    tagged: dict[str, list[str]] = field(default_factory=dict)
    roots: set[str] = field(default_factory=set)
    root_of: dict[str, str] = field(default_factory=dict)

    @property
    def tagged_count(self) -> int:
        return sum(len(runs) for runs in self.tagged.values())


def _training_runs(client: MlflowClient) -> list[tuple[str, Run]]:
    """Active parent and trial runs in the experiment families, never in the mirrors."""
    found: list[tuple[str, Run]] = []
    for experiment in client.search_experiments():
        if experiment.name.startswith(MIRROR_EXPERIMENT_PREFIX):
            continue
        # One search per role: MLflow's filter grammar has no IN for tags.
        for role in PRETRAINING_ROLES:
            page_token: str | None = None
            while True:
                page = client.search_runs(
                    [experiment.experiment_id],
                    filter_string=f"tags.run_role = '{role}'",
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
    for experiment_name, run in _training_runs(client):
        if DECODE_MASK_WEIGHTING_TAG in run.data.tags or run.data.tags.get(TASK_TAG) != "imputation":
            continue
        run_id = run.info.run_id
        report.tagged.setdefault(experiment_name, []).append(run_id)
        root = run.data.tags.get(PARENT_RUN_ID_TAG, run_id)
        report.roots.add(root)
        report.root_of[run_id] = root
        if apply:
            _stamp(client, run_id)
    if apply:
        # A stamped run's mirror gets the same tags directly; only a tree with a stamped run
        # that has no mirror yet is mirrored from scratch. Re-mirroring every tree, as the earlier
        # backfills did, took hours on the server's store for the same end state.
        mirrors = _mirrors_by_source(client)
        unmirrored_roots: set[str] = set()
        for runs in report.tagged.values():
            for run_id in runs:
                if run_id in mirrors:
                    for mirror_id in mirrors[run_id]:
                        _stamp(client, mirror_id)
                else:
                    unmirrored_roots.add(report.root_of[run_id])
        for root in sorted(unmirrored_roots):
            mirror_run_tree(client, root)
    return report


def _stamp(client: MlflowClient, run_id: str) -> None:
    client.set_tag(run_id, DECODE_MASK_WEIGHTING_TAG, "uniform")
    client.set_tag(run_id, DECODE_MASK_WEIGHTING_BACKFILLED_TAG, "true")


def _mirrors_by_source(client: MlflowClient) -> dict[str, list[str]]:
    """Every active mirror run, by the run it mirrors."""
    found: dict[str, list[str]] = {}
    for experiment in client.search_experiments():
        if not experiment.name.startswith(MIRROR_EXPERIMENT_PREFIX):
            continue
        page_token: str | None = None
        while True:
            page = client.search_runs(
                [experiment.experiment_id],
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=1000,
                page_token=page_token,
            )
            for run in page:
                source = run.data.tags.get(SOURCE_RUN_ID_TAG)
                if source:
                    found.setdefault(source, []).append(run.info.run_id)
            page_token = page.token
            if not page_token:
                break
    return found


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
    print(f"[{mode}] {DECODE_MASK_WEIGHTING_TAG} stamped on {report.tagged_count} run(s) in {len(report.roots)} tree(s)")
    for experiment_name in sorted(report.tagged):
        print(f"  {experiment_name:<24} {len(report.tagged[experiment_name])}")
    if not args.apply and report.tagged_count:
        print("Re-run with --apply to write the tag and re-mirror the trees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
