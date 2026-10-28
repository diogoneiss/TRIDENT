"""E87: does pre-training help classification with ten folds and five seeds? (classification-step-zero/04, T41).

    uv run --python 3.11 python -B scripts/experiments/cls_tenfold.py train <variant> <seed> <A|B>
    uv run --python 3.11 python -B scripts/experiments/cls_tenfold.py --todo <n>
    uv run --python 3.11 python -B scripts/experiments/cls_tenfold.py --tally

E71's arms A (the old sweep's ``hyperparameters.json`` as is) and B (``EPOCHS_PRE 0``), with
``cosine_legacy``, on ten folds (``--cv_folds 10``) and five seeds, MLflow tag
``experiment=cls-tenfold-2026-10-28``. Both ``--todo`` and ``--tally`` read ``raw_fold_metrics.csv``
on disk, as E71 did, so neither waits on a store copy. Run from the checkout root on medusa1.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments.cls_determinism_run import sweep_run  # noqa: E402
from scripts.experiments.cls_stage2_run import ARM_COST, BASES, LEVELS, READ, ROWS, cell_dir, fold_metrics  # noqa: E402

EXPERIMENT = "cls-tenfold-2026-10-28"
STATE = Path("/scratch/diogoneiss/trident-2026-10-28-cls-tenfold")
VARIANTS = tuple(f"{b}_{lv}nan" for b in BASES for lv in LEVELS)
SEEDS = (42, 7, 13, 101, 202)
ARMS = {"A": {}, "B": {"EPOCHS_PRE": 0}}
FOLDS = 10
QUEUES = 8
SESOI = 0.01


def train(variant: str, seed: int, arm: str) -> None:
    from train import main as train_main

    run_id, artifacts = sweep_run(variant)
    override = json.loads((artifacts / "parameters" / "hyperparameters.json").read_text()) | ARMS[arm]
    print("cell", variant, seed, arm, "sweep run", run_id, "override", json.dumps(override), flush=True)
    args = Namespace(
        dataset_name=variant, label_column="class",
        output_dir=str(cell_dir(STATE, variant, seed, arm)), metrics_dir=str(STATE / "metrics" / f"{variant}_s{seed}_{arm}"),
        disable_mlflow=False, plot_losses=False, save_model=False, seed=seed, cv_folds=FOLDS,
        task="classification", lr_scheduler="cosine_legacy", hyperparams_override=override,
        mlflow_tags={"experiment": EXPERIMENT, "arm": arm},
    )
    train_main(args, return_metrics=True)
    frame = fold_metrics(cell_dir(STATE, variant, seed, arm))
    print("CELL_DONE", variant, seed, arm, "f1_macro", [] if frame is None else [round(float(x), 6) for x in frame["f1_macro"]], flush=True)


def todo(which: int) -> None:
    """Seed by seed in SEEDS order; within a seed, the largest tables first, dealt greedily by cost."""
    for seed in SEEDS:
        work = sorted(((ROWS[v.split("_")[0]] * ARM_COST[a], v, a) for v in VARIANTS for a in ARMS), key=lambda w: (-w[0], w[1], w[2]))
        loads = [0.0] * QUEUES
        for cost, variant, arm in work:
            target = loads.index(min(loads))
            loads[target] += cost
            if target == which - 1 and fold_metrics(cell_dir(STATE, variant, seed, arm)) is None:
                print(variant, seed, arm)


def tally() -> None:
    """Per level, B − A on test macro-F1 over every table's fold pairs, one array per seed; then the call."""
    missing = [f"{v} s{s} {a}" for s in SEEDS for v in VARIANTS for a in ARMS if fold_metrics(cell_dir(STATE, v, s, a)) is None]
    if missing:
        print(f"incomplete: {len(missing)} of {len(SEEDS) * len(VARIANTS) * len(ARMS)} cells missing, e.g. {', '.join(missing[:5])}")
        return
    calls = {}
    for level in LEVELS:
        per_seed, means = [], []
        per_table: dict[str, list[float]] = {b: [] for b in BASES}
        for seed in SEEDS:
            diffs = []
            for base in BASES:
                a = fold_metrics(cell_dir(STATE, f"{base}_{level}nan", seed, "A"))
                b = fold_metrics(cell_dir(STATE, f"{base}_{level}nan", seed, "B"))
                assert a is not None and b is not None and len(a) == len(b) == FOLDS
                d = (b["f1_macro"] - a["f1_macro"]).to_numpy(dtype=float)
                diffs.append(d)
                per_table[base].append(float(d.mean()))
                means.append(float(a["f1_macro"].mean()))
            per_seed.append(np.concatenate(diffs))
        v = common.verdict(per_seed, sesoi=SESOI)
        calls[level] = v.call
        tables = "; ".join(f"{b} {np.mean(m):+.4f}" for b, m in per_table.items())
        print(f"- {level}nan (A's mean {np.mean(means):.4f}): B − A {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] seeds "
              f"{', '.join(f'{s:+.4f}' for s in v.seeds)}, {sum(len(d) for d in per_seed)} fold pairs -> {READ[v.call]}")
        print(f"  - by table (mean over seeds): {tables}")
    helps = [lv for lv, c in calls.items() if c == "better"]
    hurts = [lv for lv, c in calls.items() if c == "worse"]
    if len(helps) >= 3 and not hurts:
        reading = "supported: the pre-training gain replicates"
    elif len(helps) <= 1:
        reading = "refuted: no pre-training gain"
    else:
        reading = "mixed"
    print(f"\nPrimary: A higher at {helps or 'no level'}, B higher at {hurts or 'no level'} of {len(LEVELS)} -> {reading} ({common.RULE}, SESOI {SESOI})")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and len(argv) == 4 and argv[3] in ARMS:
        train(argv[1], int(argv[2]), argv[3])
    elif argv[:1] == ["--todo"] and len(argv) == 2:
        todo(int(argv[1]))
    elif argv in (["--tally"], []):
        tally()
    else:
        sys.exit(__doc__)
