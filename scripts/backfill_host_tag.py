#!/usr/bin/env python3
"""Stamp the ``host`` tag on runs recorded before it, from their ``gpu_name`` (ADR 0034).

Each machine has its own GPU model, so ``gpu_name`` names the machine a run trained on. This script
stamps ``host`` on every active run, source or mirror, whose ``gpu_name`` is in ``HOST_OF_GPU`` and
that lacks the tag, and marks it ``host_backfilled=true``, so an inferred value can be told from a
recorded one. A mirror carries its source's ``gpu_name``, so it is stamped directly. Runs with no
``gpu_name`` (recorded before the environment tags) or an unknown one are left alone; a second run
finds nothing.

Usage (dry run by default, honours ``MLFLOW_TRACKING_URI``):

    uv run --python 3.11 python scripts/backfill_host_tag.py
    uv run --python 3.11 python scripts/backfill_host_tag.py --apply
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlflow.entities import ViewType  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src.mlflow_utils import HOST_BACKFILLED_TAG, HOST_TAG, setup_mlflow  # noqa: E402
from src.training.mirroring import _tag  # noqa: E402

# One GPU model per machine (checked 2026-10-28); the value is what the host tag records there.
HOST_OF_GPU = {
    "NVIDIA GeForce RTX 3050 Laptop GPU": "windows",
    "NVIDIA GeForce RTX 3090 Ti": "gorgona8",
    "NVIDIA GeForce RTX 5090": "medusa",
}


def backfill(client: MlflowClient, apply: bool) -> Counter[str]:
    stamped: Counter[str] = Counter()
    experiments = [e.experiment_id for e in client.search_experiments(view_type=ViewType.ALL)]
    for gpu, host in HOST_OF_GPU.items():
        page_token: str | None = None
        while True:
            page = client.search_runs(experiments, filter_string=f"tags.gpu_name = '{gpu}'",
                                      run_view_type=ViewType.ACTIVE_ONLY, max_results=1000, page_token=page_token)
            for run in page:
                if HOST_TAG in run.data.tags:
                    continue
                stamped[host] += 1
                if apply:
                    client.log_batch(run.info.run_id, tags=[_tag(HOST_TAG, host), _tag(HOST_BACKFILLED_TAG, "true")])
            page_token = page.token
            if not page_token:
                break
    return stamped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Write. Without it, only report what would be written.")
    parser.add_argument("--tracking_uri", default=None, help="Override MLFLOW_TRACKING_URI.")
    args = parser.parse_args(argv)

    tracking_uri = setup_mlflow(args.tracking_uri)
    stamped = backfill(MlflowClient(tracking_uri=tracking_uri), apply=args.apply)
    mode = "APPLIED" if args.apply else "DRY RUN"
    print(f"[{mode}] tracking URI: {tracking_uri}")
    print(f"[{mode}] {HOST_TAG} stamped on {sum(stamped.values())} run(s): " + ", ".join(f"{h} {n}" for h, n in sorted(stamped.items())))
    if not args.apply and stamped:
        print("Re-run with --apply to write the tag.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
