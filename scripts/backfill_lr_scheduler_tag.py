#!/usr/bin/env python3
"""Backfill the ``lr_scheduler`` tag on MLflow runs recorded before ADR 0003.

Every run stored before the tag existed trained with the per-batch cosine
schedule now called ``cosine_legacy`` (the scheduler code never changed between
the first commit and ADR 0003). This script stamps that value on each run that
lacks the tag, and marks those runs with ``lr_scheduler_backfilled=true`` so a
reader can tell a recorded value from an inferred one.

Usage (dry run by default, honours ``MLFLOW_TRACKING_URI``):

    uv run --python 3.10 python scripts/backfill_lr_scheduler_tag.py
    uv run --python 3.10 python scripts/backfill_lr_scheduler_tag.py --apply

Runs that already carry the tag are never touched, so the script is idempotent.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import sys
from pathlib import Path

# Allow ``python scripts/backfill_lr_scheduler_tag.py`` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlflow.entities import Run, ViewType  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src.mlflow_utils import (  # noqa: E402
    LR_SCHEDULER_BACKFILLED_TAG,
    LR_SCHEDULER_TAG,
    setup_mlflow,
)
from src.training.types import DEFAULT_LR_SCHEDULER, LR_SCHEDULER_NAMES  # noqa: E402


@dataclass
class BackfillReport:
    tagged: dict[str, list[str]] = field(default_factory=dict)
    skipped: dict[str, list[str]] = field(default_factory=dict)
    deleted: dict[str, list[str]] = field(default_factory=dict)

    @property
    def tagged_count(self) -> int:
        return sum(len(runs) for runs in self.tagged.values())

    @property
    def skipped_count(self) -> int:
        return sum(len(runs) for runs in self.skipped.values())

    @property
    def deleted_count(self) -> int:
        return sum(len(runs) for runs in self.deleted.values())


def iter_all_runs(client: MlflowClient) -> list[tuple[str, Run]]:
    """Every run in every experiment, including deleted ones, as (experiment name, run)."""
    runs: list[tuple[str, Run]] = []
    for experiment in client.search_experiments(view_type=ViewType.ALL):
        page_token: str | None = None
        while True:
            page = client.search_runs(
                [experiment.experiment_id],
                run_view_type=ViewType.ALL,
                max_results=1000,
                page_token=page_token,
            )
            runs.extend((experiment.name, run) for run in page)
            page_token = page.token
            if not page_token:
                break
    return runs


def backfill(client: MlflowClient, value: str, apply: bool) -> BackfillReport:
    if value not in LR_SCHEDULER_NAMES:
        raise ValueError(f"{value!r} is not one of {', '.join(LR_SCHEDULER_NAMES)}")
    report = BackfillReport()
    for experiment_name, run in iter_all_runs(client):
        run_id = run.info.run_id
        if LR_SCHEDULER_TAG in run.data.tags:
            report.skipped.setdefault(experiment_name, []).append(run_id)
            continue
        if run.info.lifecycle_stage != "active":
            # MLflow refuses tag writes on deleted runs; they are reported, not touched.
            report.deleted.setdefault(experiment_name, []).append(run_id)
            continue
        report.tagged.setdefault(experiment_name, []).append(run_id)
        if apply:
            client.set_tag(run_id, LR_SCHEDULER_TAG, value)
            client.set_tag(run_id, LR_SCHEDULER_BACKFILLED_TAG, "true")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the tags. Without this flag the script only reports what it would do.",
    )
    parser.add_argument(
        "--value",
        default=DEFAULT_LR_SCHEDULER,
        choices=LR_SCHEDULER_NAMES,
        help=f"Schedule name to stamp on untagged runs (default: {DEFAULT_LR_SCHEDULER}).",
    )
    parser.add_argument(
        "--tracking_uri",
        default=None,
        help="Override the tracking URI (default: MLFLOW_TRACKING_URI or sqlite:///mlflow.db).",
    )
    args = parser.parse_args(argv)

    tracking_uri = setup_mlflow(args.tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    report = backfill(client, args.value, apply=args.apply)

    mode = "APPLIED" if args.apply else "DRY RUN"
    print(f"[{mode}] tracking URI: {tracking_uri}")
    print(f"[{mode}] {LR_SCHEDULER_TAG}={args.value} on {report.tagged_count} run(s); "
          f"{report.skipped_count} already tagged; {report.deleted_count} deleted (left untouched)")
    for experiment_name in sorted(set(report.tagged) | set(report.skipped) | set(report.deleted)):
        tagged = len(report.tagged.get(experiment_name, []))
        skipped = len(report.skipped.get(experiment_name, []))
        deleted = len(report.deleted.get(experiment_name, []))
        print(f"  {experiment_name:<24} tagged={tagged:<4} already_tagged={skipped:<4} deleted={deleted}")
    if not args.apply and report.tagged_count:
        print("Re-run with --apply to write the tags.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
