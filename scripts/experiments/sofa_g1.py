"""E152: C15's gate G1 on the SOFA table's natural gaps (docs/tickets/mimic/06-sofa-positive-control.md).

    uv run --python 3.11 python -B scripts/experiments/sofa_g1.py train <arm> <seed> [--epochs <pre> <fine>]
    uv run --python 3.11 python -B scripts/experiments/sofa_g1.py --todo <n> <queues>
    uv run --python 3.11 python -B scripts/experiments/sofa_g1.py --g1

A cell is C15's (``c15_run.train``'s arguments) on ``mimic-sofa-num_natural``, read from the study's own data root,
written under ``STATE``. ``--epochs`` is the smoke's override only, into ``STATE/smoke``. ``--g1`` is E128's reader on
one table, with A1's contrast on ``bilirubin_max`` and the sign below 0.
"""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

STATE = Path("/scratch2/diogoneiss/trident-2026-11-06-sofa-g1")
os.environ["TRIDENT_DATASETS_ROOT"] = str(STATE / "data")  # before any dataset is resolved

from scripts.experiments import c15_report as report  # noqa: E402
from scripts.experiments import c15_run as run  # noqa: E402

VARIANT = "mimic-sofa-num_natural"
COLUMN = "bilirubin_max"
ARMS = ("A1", "A2", "A3")
SEEDS = (42, 7, 13)


def train(arm: str, seed: int, epochs: tuple[int, int] | None) -> None:
    from src.training.config import build_training_parser, logged_hyperparameters, resolve_training_request
    from train import main as train_main

    base = logged_hyperparameters(resolve_training_request(build_training_parser().parse_args(["--dataset_name", VARIANT])))
    override = {**base, "PRETRAIN_OBJECTIVE": "embedding_normalized", "C15_ARM": arm}
    if epochs:
        override.update(EPOCHS_PRE=epochs[0], EPOCH_FINE=epochs[1])
    out = run.cell_dir(arm, VARIANT, seed, STATE / "smoke" if epochs else STATE)
    args = Namespace(dataset_name=VARIANT, label_column="class", output_dir=str(out / "results"),
                     metrics_dir=str(out / "metrics"), disable_mlflow=True, plot_losses=False, save_model=False,
                     seed=seed, cv_folds=5, task="classification", mlflow_tags={}, hyperparams_override=override)
    train_main(args, return_metrics=True)
    print("CELL_DONE", arm, VARIANT, seed, flush=True)


def todo(which: int, queues: int) -> None:
    """Seeds dealt over the queues, every arm of a seed in one queue."""
    for k, seed in enumerate(SEEDS):
        if k % queues == which - 1:
            for arm in ARMS:
                if not run.done(arm, VARIANT, seed, STATE):
                    print(arm, seed)


def g1() -> None:
    print(f"## G1 on {VARIANT} (E152, seeds 42/7/13 x 5 folds): (1) A2 − A3 relative test log loss below 0; (2) A1's "
          f"{COLUMN} contrast below 0; Nadeau–Bengio corrected 95% t intervals over 15 cells; passes if both hold")
    rel: list[np.ndarray] = []
    c: list[float] = []
    seeds: list[str] = []
    f1: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
    for seed in SEEDS:
        if not all(run.done(arm, VARIANT, seed, STATE) for arm in ARMS):
            print(f"- s{seed}: missing\n\nPrimary (G1): incomplete")
            return
        folds = {arm: [torch.load(run.cell_dir(arm, VARIANT, seed, STATE) / "metrics" / f"c15_readout_fold_{k}.pt",
                                         weights_only=False) for k in range(1, 6)] for arm in ARMS}
        scored = {arm: report.scores(f) for arm, f in folds.items()}
        for arm in ARMS:
            f1[arm].append(scored[arm][0])
        r = (scored["A2"][1] - scored["A3"][1]) / scored["A3"][1]
        rel.append(r)
        cs = [report.contrast(VARIANT, k + 1, f, seed, COLUMN) for k, f in enumerate(folds["A1"])]
        c.extend(cs)
        seeds.append(f"s{seed} A2 − A3 {r.mean():+.4f}, contrast {np.mean(cs):+.3f}")
    m, lo, hi = report.interval(np.concatenate(rel))
    _, plo, phi = report.interval(np.concatenate(rel), False)
    cm, clo, chi = report.interval(np.asarray(c))
    _, cplo, cphi = report.interval(np.asarray(c), False)
    ok = hi < 0 and chi < 0
    print(f"- A2 − A3 {m:+.4f} [{lo:+.4f}, {hi:+.4f}] (plain [{plo:+.4f}, {phi:+.4f}])")
    print(f"- A1 contrast on {COLUMN} {cm:+.3f} [{clo:+.3f}, {chi:+.3f}] logit (plain [{cplo:+.3f}, {cphi:+.3f}])")
    print(f"- per seed: {'; '.join(seeds)}")
    print("- macro-F1 (descriptive): " + ", ".join(f"{arm} {np.concatenate(v).mean():.4f}" for arm, v in f1.items()))
    print(f"\nPrimary (G1): {'passes' if ok else 'fails'}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and len(argv) == 3:
        train(argv[1], int(argv[2]), None)
    elif argv[:1] == ["train"] and len(argv) == 6 and argv[3] == "--epochs":
        train(argv[1], int(argv[2]), (int(argv[4]), int(argv[5])))
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    elif argv == ["--g1"]:
        g1()
    else:
        sys.exit(__doc__)
