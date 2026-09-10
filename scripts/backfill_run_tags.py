#!/usr/bin/env python3
"""Stamp a tag onto MLflow runs recorded before that tag existed.

A tag that only new runs carry cannot be filtered on: a query excluding one value also
excludes every run that predates the tag, silently. Backfilling closes that gap, and this
script is the general form of the one written for ``lr_scheduler`` under ADR 0003.

Two kinds of value need different treatment. One **inferred** from outside knowledge (a
run trained classification because the imputation task did not exist yet) also gets a
marker tag, so a reader can tell it from a value that was recorded at the time. One the
store itself **proves** (no run carries an Optuna role, so none came from a search) needs
no marker, because anyone can re-derive it.

Usage (a dry run by default, honouring ``MLFLOW_TRACKING_URI``):

    uv run --python 3.10 python scripts/backfill_run_tags.py --tag task \\
        --value classification --marker task_backfilled
    uv run --python 3.10 python scripts/backfill_run_tags.py --tag is_optuna --value false

Add ``--apply`` to write. Runs that already carry the tag are never touched, so the
script is idempotent; deleted runs are reported rather than written, because MLflow
refuses tag writes on them.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from mlflow.entities import Run, ViewType
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mlflow_utils import setup_mlflow  # noqa: E402


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


def backfill(
    client: MlflowClient, tag: str, value: str, marker: str | None, apply: bool
) -> BackfillReport:
    """Stamp ``tag=value`` on every run that lacks it, optionally marking it as inferred."""
    report = BackfillReport()
    for experiment_name, run in iter_all_runs(client):
        run_id = run.info.run_id
        if tag in run.data.tags:
            # A recorded value is the truth; an inferred one never overwrites it.
            report.skipped.setdefault(experiment_name, []).append(run_id)
            continue
        if run.info.lifecycle_stage != "active":
            # MLflow refuses tag writes on deleted runs; they are reported, not touched.
            report.deleted.setdefault(experiment_name, []).append(run_id)
            continue
        report.tagged.setdefault(experiment_name, []).append(run_id)
        if apply:
            client.set_tag(run_id, tag, value)
            if marker is not None:
                client.set_tag(run_id, marker, "true")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tag", required=True, help="Tag name to stamp on untagged runs.")
    parser.add_argument("--value", required=True, help="Value to stamp.")
    parser.add_argument(
        "--marker",
        default=None,
        help=(
            "Companion tag marking the value as inferred rather than recorded. Omit when "
            "the store itself proves the value."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the tags. Without this flag the script only reports what it would do.",
    )
    parser.add_argument(
        "--tracking_uri",
        default=None,
        help="Override the tracking URI (default: MLFLOW_TRACKING_URI or sqlite:///mlflow.db).",
    )
    args = parser.parse_args(argv)

    tracking_uri = setup_mlflow(args.tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    report = backfill(client, args.tag, args.value, args.marker, apply=args.apply)

    mode = "APPLIED" if args.apply else "DRY RUN"
    marked = f" (+{args.marker}=true)" if args.marker else ""
    print(f"[{mode}] tracking URI: {tracking_uri}")
    print(
        f"[{mode}] {args.tag}={args.value}{marked} on {report.tagged_count} run(s); "
        f"{report.skipped_count} already tagged; {report.deleted_count} deleted (left untouched)"
    )
    for experiment_name in sorted(set(report.tagged) | set(report.skipped) | set(report.deleted)):
        tagged = len(report.tagged.get(experiment_name, []))
        skipped = len(report.skipped.get(experiment_name, []))
        deleted = len(report.deleted.get(experiment_name, []))
        print(f"  {experiment_name:<24} tagged={tagged:<4} already_tagged={skipped:<4} deleted={deleted}")
    if not args.apply and report.tagged_count:
        print("\nRe-run with --apply to write these tags.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
