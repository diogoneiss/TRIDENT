"""E86: do gorgona8 and medusa1 give the same fold scores for the same cell? (cross-machine/01).
E94 (cross-machine/02): the same check with enough pairs to show equivalence; every command takes
``--study e94`` first, which switches the state, the cells and the seeds.

    uv run --python 3.11 python -B scripts/experiments/cross_machine.py train <task> <variant> <seed> [dup]
    uv run --python 3.11 python -B scripts/experiments/cross_machine.py --todo <n>
    uv run --python 3.11 python -B scripts/experiments/cross_machine.py --tally

``train`` runs one cell with MLflow off, into ``STATE/<hostname>/``, so nothing reaches the store
and no run or mirror is duplicated. imputation: today's default, five folds, as E84's controls ran.
classification: E71's arm A (the old sweep's ``hyperparameters.json``, ``cosine_legacy``, three
folds, ``cls_stage2_run.run``). ``dup`` runs the same cell again into its own directory, for the
run-to-run check on that machine. ``--tally`` reads both hosts' directories (medusa1's copied
under ``STATE/medusa``) and prints per-fold differences and the verdict per task.
"""

from __future__ import annotations

import socket
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

# medusa1's /scratch2 belongs to root (ADR 0033).
STATE = Path("/scratch2/diogoneiss" if socket.gethostname() == "gorgona8" else "/scratch/diogoneiss") / "trident-2026-10-28-cross-machine"
HOSTS = ("gorgona8", "medusa")
CELLS: dict[str, tuple[tuple[str, ...], tuple[int, ...]]] = {
    "imputation": (("credit-g_20nan", "kc2_60nan", "vehicle_40nan", "biodeg_40nan"), (101, 202, 303)),
    "classification": (("vehicle_40nan", "kc2_40nan", "credit-g_20nan", "biodeg_60nan"), (42, 7, 13)),
}
DUPLICATES: tuple[tuple[str, str, int], ...] = (("imputation", "credit-g_20nan", 101), ("classification", "vehicle_40nan", 42))
QUEUES = 2
# Primary: the default output's induced score (relative, SESOI 1%) and macro-F1 (absolute, SESOI 0.01).
PRIMARY = {"imputation": ("impute/induced/impute_score", 0.01), "classification": ("f1_macro", 0.01)}
SECONDARY = {"imputation": ("impute/masked/impute_score", "impute/induced/baseline/hgb/impute_score",
                            "impute/induced/baseline/knn5/impute_score"), "classification": ("accuracy",)}


def _use_e94() -> None:
    """E94's cells: seven imputation and five classification variants on small tables, seeds 1 to 10, no duplicates."""
    global STATE, CELLS, DUPLICATES, QUEUES
    STATE, QUEUES = STATE.parent / "trident-2026-10-29-cross-machine-e94", 4
    seeds = tuple(range(1, 11))
    CELLS = {
        "imputation": (("credit-g_20nan", "credit-g_60nan", "kc2_20nan", "kc2_60nan", "vehicle_40nan", "vehicle_80nan", "biodeg_40nan"), seeds),
        "classification": (("vehicle_20nan", "vehicle_40nan", "kc2_40nan", "credit-g_20nan", "biodeg_60nan"), seeds),
    }
    DUPLICATES = ()


def cell_dir(host: str, task: str, variant: str, seed: int, dup: bool = False) -> Path:
    return STATE / host / task / f"{variant}_s{seed}{'_dup' if dup else ''}"


def train(task: str, variant: str, seed: int, dup: bool) -> None:
    out = cell_dir(socket.gethostname(), task, variant, seed, dup)
    if task == "classification":
        from scripts.experiments.cls_stage2_run import run

        run(variant, seed, "A", out, no_mlflow=True, out=out / "results")
        return
    from train import main as train_main

    args = Namespace(dataset_name=variant, label_column="class", output_dir=str(out / "results"),
                     metrics_dir=str(out / "metrics"), disable_mlflow=True, plot_losses=False,
                     save_model=False, seed=seed, cv_folds=5, task="imputation", mlflow_tags={})
    train_main(args, return_metrics=True)
    print("CELL_DONE", task, variant, seed, flush=True)


def folds(host: str, task: str, variant: str, seed: int, dup: bool = False) -> pd.DataFrame | None:
    root = cell_dir(host, task, variant, seed, dup)
    pattern = "raw_fold_metrics.csv" if task == "classification" else f"{variant}_metrics.csv"
    found = sorted(root.rglob(pattern))
    if len(found) != 1:
        return None
    frame = pd.read_csv(found[0])
    return frame[pd.to_numeric(frame["fold"], errors="coerce").notna()].sort_values("fold").reset_index(drop=True)


def todo(which: int) -> None:
    host = socket.gethostname()
    work = [(t, v, s, False) for t, (variants, seeds) in CELLS.items() for s in seeds for v in variants]
    work += [(t, v, s, True) for t, v, s in DUPLICATES]
    for k, (task, variant, seed, dup) in enumerate(work):
        if k % QUEUES == which - 1 and folds(host, task, variant, seed, dup) is None:
            print(task, variant, seed, "dup" if dup else "")


def _diff(task: str, metric: str, a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    x, y = a[metric].to_numpy(dtype=float), b[metric].to_numpy(dtype=float)
    return np.asarray((y - x) / x if task == "imputation" else y - x, dtype=float)


def tally() -> None:
    for task, (variants, seeds) in CELLS.items():
        metric, sesoi = PRIMARY[task]
        print(f"## {task}: medusa − gorgona8 on {metric}" + (" (relative)" if task == "imputation" else ""))
        per_seed: dict[int, list[float]] = {s: [] for s in seeds}
        largest: dict[str, float] = {}
        missing = []
        for variant in variants:
            for seed in seeds:
                a, b = folds("gorgona8", task, variant, seed), folds("medusa", task, variant, seed)
                if a is None or b is None:
                    missing.append(f"{variant} s{seed}")
                    continue
                d = _diff(task, metric, a, b)
                per_seed[seed].extend(d.tolist())
                for extra in (metric, *SECONDARY[task]):
                    if extra in a:
                        largest[extra] = max(largest.get(extra, 0.0), float(np.abs(_diff(task, extra, a, b)).max()))
                print(f"- {variant} s{seed}: " + ", ".join(f"{x:+.4f}" for x in d))
        if missing:
            print(f"incomplete: {', '.join(missing)}\n")
            continue
        v = common.verdict([np.array(per_seed[s]) for s in seeds], sesoi=sesoi)
        print(f"- largest |difference| per metric: " + "; ".join(f"{k} {x:.4f}" for k, x in largest.items()))
        print(f"- pooled {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}], seeds {', '.join(f'{s:+.4f}' for s in v.seeds)}, "
              f"{sum(len(per_seed[s]) for s in seeds)} fold pairs -> {v.call} (SESOI {sesoi}; {common.RULE})\n")
    print("## run to run, same machine")
    for task, variant, seed in DUPLICATES:  # none in E94
        metric = PRIMARY[task][0]
        for host in HOSTS:
            a, b = folds(host, task, variant, seed), folds(host, task, variant, seed, dup=True)
            shown = "missing" if a is None or b is None else f"largest |difference| {np.abs(_diff(task, metric, a, b)).max():.2e}"
            print(f"- {host} {task} {variant} s{seed}: {shown}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:2] == ["--study", "e94"]:
        _use_e94()
        argv = argv[2:]
    if argv[:1] == ["train"] and len(argv) in (4, 5) and argv[1] in CELLS:
        train(argv[1], argv[2], int(argv[3]), argv[4:] == ["dup"])
    elif argv[:1] == ["--todo"] and len(argv) == 2:
        todo(int(argv[1]))
    elif argv in (["--tally"], []):
        tally()
    else:
        sys.exit(__doc__)
