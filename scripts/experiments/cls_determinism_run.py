"""Classification step zero, Stage 0: reproduce an old sweep run's fold macro-F1 at HEAD
(docs/tickets/classification-step-zero/01-determinism.md). Nothing is written to the store.

    uv run --python 3.11 python -B scripts/experiments/cls_determinism_run.py <variant> <out_dir>
    uv run --python 3.11 python -B scripts/experiments/cls_determinism_run.py --compare <variant> <out_dir>

The sweep's run is the latest active, non-mirror classification parent of ``<variant>`` with seed
42, 3 folds and ``cosine_legacy`` (review item 13's selection), started before 2026-09-12 and
carrying no ``experiment`` tag, so a study's own runs never match. Its ``hyperparameters.json`` is
passed as ``hyperparams_override``, with ``lr_scheduler='cosine_legacy'``, 3 folds and seed 42,
MLflow off and a private metrics directory. ``--compare`` prints both runs' test metrics fold by
fold and the largest difference.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

METRICS = ("accuracy", "f1_micro", "f1_macro")
# The sweep's last run started on 2026-09-10; a later run, such as E71's own, is never the sweep's.
SWEEP_BEFORE = 1789171200000  # 2026-09-12 00:00 UTC, in MLflow's epoch milliseconds


def sweep_run(variant: str) -> tuple[str, Path]:
    """The sweep's run of ``variant``: its id and its artifact directory."""
    store = sqlite3.connect("file:mlflow.db?mode=ro", uri=True)
    rows = store.execute(
        """
        select r.run_uuid, r.artifact_uri from runs r
        where r.lifecycle_stage = 'active'
          and exists (select 1 from tags t where t.run_uuid = r.run_uuid and t.key = 'task' and t.value = 'classification')
          and exists (select 1 from tags t where t.run_uuid = r.run_uuid and t.key = 'run_role' and t.value = 'parent')
          and exists (select 1 from tags t where t.run_uuid = r.run_uuid and t.key = 'dataset' and t.value = ?)
          and exists (select 1 from tags t where t.run_uuid = r.run_uuid and t.key = 'lr_scheduler' and t.value = 'cosine_legacy')
          and exists (select 1 from params p where p.run_uuid = r.run_uuid and p.key = 'seed' and p.value = '42')
          and exists (select 1 from params p where p.run_uuid = r.run_uuid and p.key = 'cv_folds' and p.value = '3')
          and not exists (select 1 from tags t where t.run_uuid = r.run_uuid and t.key = 'is_mirror' and t.value = 'true')
          and not exists (select 1 from tags t where t.run_uuid = r.run_uuid and t.key = 'experiment')
          and r.start_time < ?
        order by r.start_time desc limit 1
        """,
        (variant, SWEEP_BEFORE),
    ).fetchall()
    if not rows:
        sys.exit(f"no sweep run of {variant}")
    run_id, uri = rows[0]
    return str(run_id), Path(str(uri).removeprefix("file://"))


def run(variant: str, out: Path) -> None:
    from train import main as train_main

    run_id, artifacts = sweep_run(variant)
    override = json.loads((artifacts / "parameters" / "hyperparameters.json").read_text())
    print("sweep run", run_id, "override", json.dumps(override), flush=True)
    args = Namespace(
        dataset_name=variant,
        label_column="class",
        output_dir=str(out / "results"),
        metrics_dir=str(out / "metrics"),
        disable_mlflow=True,
        plot_losses=False,
        save_model=False,
        seed=42,
        cv_folds=3,
        task="classification",
        lr_scheduler="cosine_legacy",
        hyperparams_override=override,
    )
    train_main(args, return_metrics=True)
    print("RUN_DONE", variant, flush=True)


def compare(variant: str, out: Path) -> None:
    run_id, artifacts = sweep_run(variant)
    theirs = pd.read_csv(artifacts / "metrics" / "raw_fold_metrics.csv").sort_values("fold").reset_index(drop=True)
    found = sorted((out / "results").rglob("raw_fold_metrics.csv"))
    if len(found) != 1:
        sys.exit(f"{len(found)} raw_fold_metrics.csv under {out / 'results'}")
    ours = pd.read_csv(found[0]).sort_values("fold").reset_index(drop=True)
    print(f"## `{variant}` (sweep run {run_id})")
    largest = 0.0
    for metric in METRICS:
        diff = (ours[metric] - theirs[metric]).abs()
        largest = max(largest, float(diff.max()))
        print(f"- {metric}: HEAD {[round(x, 6) for x in ours[metric]]} against the sweep's {[round(x, 6) for x in theirs[metric]]}; "
              f"largest difference {diff.max():.2e}")
    print(f"- largest difference over {', '.join(METRICS)}: {largest:.2e}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--compare"] and len(argv) == 3:
        compare(argv[1], Path(argv[2]))
    elif len(argv) == 2:
        run(argv[0], Path(argv[1]))
    else:
        sys.exit(__doc__)
