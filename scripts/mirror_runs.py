#!/usr/bin/env python3
"""Mirror every run in every experiment family into ``TRIDENT/mirror/<task>`` (ADR 0006).

The trainer mirrors a run tree live when its root closes; this script covers everything
else: the runs recorded before mirroring existed, a run whose live mirror failed, and any
source that changed after it was mirrored. It is an upsert by replay, so running it twice
is the same as running it once, and it reconciles in both directions: a source deleted
from its family, or one that finished ``FAILED``, loses its mirror (ADR 0006, amended
2026-09-30), and a mirror whose source no longer exists anywhere is deleted.

Every active source that lacks the dense ``is_mirror`` tag is stamped ``false`` on the
way, which is the backfill ADR 0006 calls for. Mirror experiments are never sources.

Usage (a dry run by default, honouring ``MLFLOW_TRACKING_URI``):

    uv run --python 3.11 python scripts/mirror_runs.py
    uv run --python 3.11 python scripts/mirror_runs.py --task imputation
    uv run --python 3.11 python scripts/mirror_runs.py --experiment TRIDENT/credit-g --apply

Add ``--apply`` to write. Restricting the scan with ``--experiment`` skips the orphan
pass, because a mirror from an unscanned family would look orphaned. ``RUNNING`` sources
are skipped and reported; they are mirrored once they terminate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mlflow_utils import setup_mlflow  # noqa: E402
from src.training.mirroring import MirrorReport, sync_store  # noqa: E402
from src.training.types import TASK_NAMES  # noqa: E402


def describe(report: MirrorReport, mode: str, tracking_uri: str) -> str:
    lines = [
        f"[{mode}] tracking URI: {tracking_uri}",
        f"[{mode}] mirror runs: {len(report.created)} created, {len(report.updated)} refreshed, "
        f"{len(report.deleted)} deleted with their source, {len(report.orphans)} orphans deleted",
        f"[{mode}] sources: {len(report.stamped)} stamped is_mirror=false, "
        f"{len(report.skipped)} still RUNNING (skipped), {len(report.failed)} not mirrorable",
    ]
    if report.param_conflicts:
        lines.append(f"[{mode}] param conflicts (mirror kept its value): {len(report.param_conflicts)}")
        lines.extend(f"    {entry}" for entry in report.param_conflicts)
    lines.extend(f"    skipped RUNNING: {run_id}" for run_id in report.skipped)
    lines.extend(f"    not mirrorable: {entry}" for entry in report.failed)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the mirrors. Without this flag the script only reports what it would do.",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=TASK_NAMES,
        default=None,
        help="Mirror only roots of this task (repeatable). Default: every task.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=None,
        help="Scan only this experiment family, e.g. TRIDENT/credit-g (repeatable).",
    )
    parser.add_argument(
        "--tracking_uri",
        default=None,
        help="Override the tracking URI (default: MLFLOW_TRACKING_URI or sqlite:///mlflow.db).",
    )
    args = parser.parse_args(argv)

    tracking_uri = setup_mlflow(args.tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    report = sync_store(
        client,
        apply=args.apply,
        tasks=None if args.task is None else frozenset(args.task),
        experiments=None if args.experiment is None else frozenset(args.experiment),
    )
    print(describe(report, "APPLIED" if args.apply else "DRY RUN", tracking_uri))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
