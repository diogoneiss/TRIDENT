#!/usr/bin/env python3
"""Stamp the ``finetune_label_fraction`` and ``finetune_scarce_validation`` tags on classification runs recorded before them (ADR 0038).

Every classification run before ADR 0038 fine-tuned on every training label and chose its epoch on the
whole validation split. This script stamps ``finetune_label_fraction=1.0`` and
``finetune_scarce_validation=off`` on every classification run that fine-tuned (a CV or single-split
parent, or an Optuna trial) and lacks the tags, marks it ``finetune_label_backfilled=true`` so an inferred
value can be told from a recorded one, and tags each stamped run's mirror the same way (ADR 0006).
Imputation runs, diagnostic children, study runs, runs that already carry the tag and deleted runs are
left alone; a second run finds nothing.

Usage (dry run by default, honours ``MLFLOW_TRACKING_URI``):

    uv run --python 3.11 python scripts/backfill_finetune_label_tags.py
    uv run --python 3.11 python scripts/backfill_finetune_label_tags.py --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlflow.tracking import MlflowClient  # noqa: E402

from scripts.backfill_decode_encoder_tags import PARENT_RUN_ID_TAG, BackfillReport, _mirrors_by_source, _training_runs  # noqa: E402
from src.mlflow_utils import (  # noqa: E402
    FINETUNE_LABEL_BACKFILLED_TAG,
    FINETUNE_LABEL_FRACTION_TAG,
    FINETUNE_SCARCE_VALIDATION_TAG,
    TASK_TAG,
    setup_mlflow,
)
from src.training.mirroring import mirror_run_tree  # noqa: E402


def backfill(client: MlflowClient, apply: bool) -> BackfillReport:
    report = BackfillReport()
    for experiment_name, run in _training_runs(client):
        if FINETUNE_LABEL_FRACTION_TAG in run.data.tags or run.data.tags.get(TASK_TAG) != "classification":
            continue
        run_id = run.info.run_id
        report.tagged.setdefault(experiment_name, []).append(run_id)
        root = run.data.tags.get(PARENT_RUN_ID_TAG, run_id)
        report.roots.add(root)
        report.root_of[run_id] = root
        if apply:
            _stamp(client, run_id)
    if apply:
        # A stamped run's mirror gets the same tags in place; only a tree without a mirror is mirrored.
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
    client.set_tag(run_id, FINETUNE_LABEL_FRACTION_TAG, "1.0")
    client.set_tag(run_id, FINETUNE_SCARCE_VALIDATION_TAG, "off")
    client.set_tag(run_id, FINETUNE_LABEL_BACKFILLED_TAG, "true")


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
    print(f"[{mode}] {FINETUNE_LABEL_FRACTION_TAG} and {FINETUNE_SCARCE_VALIDATION_TAG} stamped on {report.tagged_count} run(s) in {len(report.roots)} tree(s)")
    for experiment_name in sorted(report.tagged):
        print(f"  {experiment_name:<24} {len(report.tagged[experiment_name])}")
    if not args.apply and report.tagged_count:
        print("Re-run with --apply to write the tags.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
