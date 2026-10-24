"""Classification step zero, Stage 2: the no-pre-training arms (docs/tickets/classification-step-zero/02-no-pretraining.md).

    uv run --python 3.11 python -B scripts/experiments/cls_stage2_run.py <variant> <seed> <A|B|C> <state_dir> [--no-mlflow]
    uv run --python 3.11 python -B scripts/experiments/cls_stage2_run.py --todo <n> <state_dir>
    uv run --python 3.11 python -B scripts/experiments/cls_stage2_run.py --tally <state_dir>
    uv run --python 3.11 python -B scripts/experiments/cls_stage2_run.py --gate-run <state_dir>
    uv run --python 3.11 python -B scripts/experiments/cls_stage2_run.py --gate <state_dir>

Every arm takes the old sweep's ``hyperparameters.json`` of the variant as ``hyperparams_override``
(review item 13's selection, ``cls_determinism_run.sweep_run``), with ``cosine_legacy``, 3 folds:

- A: as is (``EPOCHS_PRE 300``, ``EPOCH_FINE 150``), the fork's default classification run;
- B: ``EPOCHS_PRE 0``, no pre-training (the paper's Table 4 arm);
- C: ``EPOCHS_PRE 0`` and ``EPOCH_FINE 450``, the same number of epochs as A (as in E24).

Output goes to ``<state_dir>/results/<variant>_s<seed>_<arm>/`` and a private metrics directory;
MLflow tags ``experiment=cls-stage2-2026-10-24`` and ``arm``. Run from the checkout root.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments.cls_determinism_run import sweep_run  # noqa: E402

EXPERIMENT = "cls-stage2-2026-10-24"
BASES = ("biodeg", "credit-g", "electricity", "kc2", "kr-vs-kp", "letter", "pendigits", "spambase", "vehicle")
LEVELS = ("00", "20", "40", "60", "80")
VARIANTS = tuple(f"{b}_{lv}nan" for b in BASES for lv in LEVELS)
SEEDS = (42, 7, 13)
ARMS = {"A": {}, "B": {"EPOCHS_PRE": 0}, "C": {"EPOCHS_PRE": 0, "EPOCH_FINE": 450}}
QUEUES = 4
# Relative cost per table, from the rows each holds; it only orders and balances the queues.
ROWS = {"electricity": 45312, "letter": 20000, "pendigits": 10992, "spambase": 4601, "kr-vs-kp": 3196,
        "biodeg": 1055, "credit-g": 1000, "vehicle": 846, "kc2": 522}
ARM_COST = {"A": 1.0, "B": 0.33, "C": 1.0}
STAGE0 = Path("/scratch2/diogoneiss/trident-2026-10-23-cls-stage0")
# The verdict rule calls a negative difference "better"; on macro-F1 a negative (arm − A) means A,
# the pre-trained arm, scores higher.
READ = {"better": "A higher (pre-training helps)", "worse": "A lower (pre-training hurts)", "none": "no verdict", "equivalent": "equivalent"}


def cell_dir(state: Path, variant: str, seed: int, arm: str) -> Path:
    return state / "results" / f"{variant}_s{seed}_{arm}"


def fold_metrics(directory: Path) -> pd.DataFrame | None:
    found = sorted(directory.rglob("raw_fold_metrics.csv"))
    if len(found) != 1:
        return None
    return pd.read_csv(found[0]).sort_values("fold").reset_index(drop=True)


def run(variant: str, seed: int, arm: str, state: Path, no_mlflow: bool, out: Path | None = None) -> None:
    from train import main as train_main

    run_id, artifacts = sweep_run(variant)
    override = json.loads((artifacts / "parameters" / "hyperparameters.json").read_text()) | ARMS[arm]
    print("cell", variant, seed, arm, "sweep run", run_id, "override", json.dumps(override), flush=True)
    out = out or cell_dir(state, variant, seed, arm)
    args = Namespace(
        dataset_name=variant,
        label_column="class",
        output_dir=str(out),
        metrics_dir=str(state / "metrics" / f"{variant}_s{seed}_{arm}"),
        disable_mlflow=no_mlflow,
        plot_losses=False,
        save_model=False,
        seed=seed,
        cv_folds=3,
        task="classification",
        lr_scheduler="cosine_legacy",
        hyperparams_override=override,
        mlflow_tags={} if no_mlflow else {"experiment": EXPERIMENT, "arm": arm},
    )
    train_main(args, return_metrics=True)
    frame = fold_metrics(out)
    f1 = [] if frame is None else [round(float(x), 6) for x in frame["f1_macro"]]
    print("CELL_DONE", variant, seed, arm, "f1_macro", f1, flush=True)


def todo(which: int, state: Path) -> None:
    """Queue ``which``'s cells not yet done: seed 42 first for every arm, then 7, then 13; within
    a seed, the largest tables first; cells dealt greedily over the queues by cost."""
    for seed in SEEDS:
        work = sorted(((ROWS[v.split("_")[0]] * ARM_COST[a], v, a) for v in VARIANTS for a in ARMS), key=lambda w: (-w[0], w[1], w[2]))
        loads = [0.0] * QUEUES
        for cost, variant, arm in work:
            target = loads.index(min(loads))
            loads[target] += cost
            if target == which - 1 and fold_metrics(cell_dir(state, variant, seed, arm)) is None:
                print(variant, seed, arm)


def gate(state: Path) -> None:
    """gorgona8's run-to-run determinism: the gate cell against Stage 0's rerun of the same run."""
    ours = fold_metrics(state / "gate")
    theirs = fold_metrics(STAGE0 / "vehicle_40nan" / "results")
    if ours is None or theirs is None:
        sys.exit("gate: a run is missing")
    for metric in ("accuracy", "f1_macro"):
        diff = (ours[metric] - theirs[metric]).abs()
        print(f"gate {metric}: {[round(x, 6) for x in ours[metric]]} against Stage 0's {[round(x, 6) for x in theirs[metric]]}; largest difference {diff.max():.2e}")


def tally(state: Path) -> None:
    """Per level, B − A and C − A on test macro-F1 over the complete seeds' fold pairs."""
    complete = [s for s in SEEDS if all(fold_metrics(cell_dir(state, v, s, a)) is not None for v in VARIANTS for a in ARMS)]
    print(f"Complete seeds: {complete or 'none'} of {list(SEEDS)}\n")
    if not complete:
        return
    for other in ("B", "C"):
        print(f"## {other} − A, test macro-F1")
        for level in LEVELS:
            per_seed, per_table = [], []
            for seed in complete:
                diffs = []
                for base in BASES:
                    name = f"{base}_{level}nan"
                    a = fold_metrics(cell_dir(state, name, seed, "A"))
                    o = fold_metrics(cell_dir(state, name, seed, other))
                    assert a is not None and o is not None
                    d = (o["f1_macro"] - a["f1_macro"]).to_numpy(dtype=float)
                    diffs.append(d)
                    per_table.append((base, seed, float(d.mean())))
                per_seed.append(np.concatenate(diffs))
            v = common.verdict(per_seed)
            means = []
            for seed in complete:
                for base in BASES:
                    a = fold_metrics(cell_dir(state, f"{base}_{level}nan", seed, "A"))
                    assert a is not None
                    means.append(float(a["f1_macro"].mean()))
            mean_a = float(np.mean(means))
            tables = {b: float(np.mean([m for t, _, m in per_table if t == b])) for b in BASES}
            ranked = sorted(tables, key=lambda b: tables[b])
            worst, best = ranked[0], ranked[-1]
            print(f"- {level}nan (A's mean {mean_a:.4f}): {other} − A {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {READ[v.call]}; "
                  f"tables from {worst} {tables[worst]:+.4f} to {best} {tables[best]:+.4f}")
        print()


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), Path(argv[2]))
    elif argv[:1] == ["--tally"] and len(argv) == 2:
        tally(Path(argv[1]))
    elif argv[:1] == ["--gate-run"] and len(argv) == 2:
        run("vehicle_40nan", 42, "A", Path(argv[1]), True, out=Path(argv[1]) / "gate")
    elif argv[:1] == ["--gate"] and len(argv) == 2:
        gate(Path(argv[1]))
    elif len(argv) in (4, 5) and argv[2] in ARMS:
        run(argv[0], int(argv[1]), argv[2], Path(argv[3]), argv[4:] == ["--no-mlflow"])
    else:
        sys.exit(__doc__)
