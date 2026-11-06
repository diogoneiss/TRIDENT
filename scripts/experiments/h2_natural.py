"""E155 (docs/tickets/c15-mnar-prediction/11-h2-external-gate.md): the natural front's cells, and the study's reader.

    uv run --python 3.11 python -B scripts/experiments/h2_natural.py train <arm> <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/h2_natural.py --todo <n> <queues>
    uv run --python 3.11 python -B scripts/experiments/h2_natural.py --read

A natural cell is C15's (``c15_run.train``'s arguments) on a numerical copy of a MIMIC-IV table, read from E152's data
root, written under ``STATE``. ``--read`` reads both fronts once: the two primaries with Holm (97.5% first), the
secondary interaction, the cardio table and the sensitivity lines.
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

STATE = Path("/scratch2/diogoneiss/trident-2026-11-06-h2-natural")
os.environ["TRIDENT_DATASETS_ROOT"] = "/scratch2/diogoneiss/trident-2026-11-06-sofa-g1/data"  # before any dataset is resolved

from scripts.experiments import c15_report as report  # noqa: E402
from scripts.experiments import c15_run as run  # noqa: E402

SOFA, CARDIO = "mimic-sofa-num_natural", "mimic-cardio-num_natural"
ARMS = ("A1", "A2")
SEEDS = run.E155_SEEDS
# E139's pairs: each file of P with its MCAR sibling of the same table.
SIBLING = {v: f"{v.split('_')[0]}_mcar_s4848_60nan" for v in run.P_FRESH}


def train(arm: str, variant: str, seed: int) -> None:
    from src.training.config import build_training_parser, logged_hyperparameters, resolve_training_request
    from train import main as train_main

    base = logged_hyperparameters(resolve_training_request(build_training_parser().parse_args(["--dataset_name", variant])))
    override = {**base, "PRETRAIN_OBJECTIVE": "embedding_normalized", "C15_ARM": arm}
    out = run.cell_dir(arm, variant, seed, STATE)
    args = Namespace(dataset_name=variant, label_column="class", output_dir=str(out / "results"),
                     metrics_dir=str(out / "metrics"), disable_mlflow=True, plot_losses=False, save_model=False,
                     seed=seed, cv_folds=5, task="classification", mlflow_tags={}, hyperparams_override=override)
    train_main(args, return_metrics=True)
    print("CELL_DONE", arm, variant, seed, flush=True)


def todo(which: int, queues: int) -> None:
    """(table, seed) pairs dealt over the queues, both arms of a pair in one queue."""
    for k, (variant, seed) in enumerate((v, s) for v in (SOFA, CARDIO) for s in SEEDS):
        if k % queues == which - 1:
            for arm in ARMS:
                if not run.done(arm, variant, seed, STATE):
                    print(arm, variant, seed)


def natural(variant: str) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] | None:
    """Per seed, A1 − A2 per fold: macro-F1, relative and absolute test log loss."""
    f1d, lld, absolute = [], [], []
    for seed in SEEDS:
        if not all(run.done(arm, variant, seed, STATE) for arm in ARMS):
            return None
        (f1_1, ll_1), (f1_2, ll_2) = (report.scores([torch.load(run.cell_dir(arm, variant, seed, STATE) / "metrics"
                                                                 / f"c15_readout_fold_{k}.pt", weights_only=False)
                                                      for k in range(1, 6)]) for arm in ARMS)
        f1d.append(f1_1 - f1_2)
        lld.append((ll_1 - ll_2) / ll_2)
        absolute.append(ll_1 - ll_2)
    return f1d, lld, absolute


def sofa_call(level: float) -> str:
    got = natural(SOFA)
    assert got is not None
    call, text = report.noninferiority(got[0], got[1], level)
    print(f"- {SOFA} ({level:.1%}): {text}; absolute LL {np.concatenate(got[2]).mean():+.4f} -> {call}")
    return call


def interaction() -> None:
    """E139's form: (A1 − A2) on a file of P minus on its MCAR sibling, macro-F1, pooled over the five pairs per cell."""
    def diff(variant: str) -> np.ndarray:
        folds = [report.scores(r)[0] for arm in ARMS for seed in SEEDS if (r := report.readouts(arm, variant, seed))]
        assert len(folds) == 2 * len(SEEDS), variant
        out: np.ndarray = np.concatenate(folds[: len(SEEDS)]) - np.concatenate(folds[len(SEEDS):])
        return out
    per_pair = {v: diff(v) - diff(s) for v, s in SIBLING.items()}
    pooled = np.mean(list(per_pair.values()), axis=0)
    m, lo, hi = report.interval(pooled, False)
    _, nlo, nhi = report.interval(pooled)
    call = "gains" if lo > 0 else "loses" if hi < 0 else "holds" if lo > -0.01 and hi < 0.01 else "no verdict"
    print("\n## Secondary: does A1 gain ground on A2 where the gap is informative? (A1 − A2) on P's file minus on its "
          "MCAR sibling, macro-F1, plain 95% t over 15 cells (Nadeau–Bengio beside)")
    for v, d in per_pair.items():
        print(f"- {v} − {SIBLING[v]}: {d.mean():+.4f}")
    print(f"- pooled (5 pairs): {m:+.4f} [{lo:+.4f}, {hi:+.4f}] (NB [{nlo:+.4f}, {nhi:+.4f}]) -> {call}")


def read() -> None:
    missing = [(a, v, s) for v in (SOFA, CARDIO) for s in SEEDS for a in ARMS if not run.done(a, v, s, STATE)]
    missing += [(a, v, s) for (v, s), arms in run.STAGE5.items() for a in arms if report.readouts(a, v, s) is None]
    if missing:
        sys.exit(f"E155 incomplete, {len(missing)} cells missing, e.g. {missing[:3]}; read only with all 66 done")
    print("## E155: two primaries, Holm: each read at 97.5% first; a favourable call there lets the other be read at 95%")
    synthetic = report.h2(SEEDS, 0.975, "E155")
    print()
    sofa = sofa_call(0.975)
    if synthetic == "supported" and sofa != "non-inferior":
        sofa = sofa_call(0.95)
    elif sofa == "non-inferior" and synthetic != "supported":
        print()
        synthetic = report.h2(SEEDS, 0.95, "E155, Holm's second step")
    print(f"\nPrimary 1 (H2, synthetic, pooled over P): {synthetic}\nPrimary 2 (H2, {SOFA}): {sofa}")
    interaction()
    print("\n## Beside, decides nothing")
    cardio = natural(CARDIO)
    assert cardio is not None
    call, text = report.noninferiority(cardio[0], cardio[1])
    print(f"- {CARDIO} (negative control, 95%): {text}; absolute LL {np.concatenate(cardio[2]).mean():+.4f} -> {call}")
    sofa_f1 = natural(SOFA)
    assert sofa_f1 is not None
    print(f"- (A1 − A2) SOFA minus cardio, macro-F1 per cell: "
          f"{(np.concatenate(sofa_f1[0]) - np.concatenate(cardio[0])).mean():+.4f} (different tables, descriptive)")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and len(argv) == 4:
        train(argv[1], argv[2], int(argv[3]))
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    elif argv == ["--read"]:
        read()
    else:
        sys.exit(__doc__)
