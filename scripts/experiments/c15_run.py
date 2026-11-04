"""C15 Stage 1 cells (docs/tickets/c15-mnar-prediction/03-stage1.md), MLflow off, five folds a cell.

    uv run --python 3.11 python -B scripts/experiments/c15_run.py train <arm> <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/c15_run.py --todo <n> <queues>
    C15_STAGE=2 ... --todo <n> <queues>    # Stage 2's cells (ticket 04); 3 for confirmation (ticket 05)

A cell is one (arm, variant, seed) at ``--cv_folds 5``; the arm is ``C15_ARM`` (``src/training/c15_inputs.py``)
over the variant's resolved classification configuration with ``PRETRAIN_OBJECTIVE embedding_normalized``
(the user's choice, plan). Each fold writes ``c15_readout_fold_<k>.pt`` (per-row probabilities, the kept
state) beside the fold metrics. ``--todo`` deals this machine's pairs over its queues, every arm of a
(variant, seed) pair in one queue (ADR 0035).
"""

from __future__ import annotations

import socket
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

HOST = "gorgona8" if socket.gethostname() == "gorgona8" else "medusa"
STATE = Path("/scratch2/diogoneiss" if HOST == "gorgona8" else "/scratch/diogoneiss") / "trident-2026-11-04-c15-stage1"
F0 = tuple(f"{b}_{r}nan" for b in ("biodeg", "kc2", "vehicle") for r in (20, 60)) + ("spambase_60nan",)
F1 = tuple(f"{b}_probit_{r}nan" for b in ("biodeg", "kc2", "vehicle") for r in (20, 60)) + ("spambase_probit_60nan",)
F2 = tuple(f"biodeg_probit{s}_60nan" for s in ("0p25", "1p0", "2p0"))
F3 = tuple(f"{b}_outcome{o}_s2060_60nan" for b in ("biodeg", "kc2", "spambase") for o in ("1p5", "3p0"))
F5 = tuple(f"{b}_noise3p0_s2060_60nan" for b in ("biodeg", "kc2", "spambase"))
DEVELOPMENT = F0 + F1 + F2 + F3
SEEDS = (42, 7, 13)
A4 = tuple(f"A4:{k}" for k in range(5))
# (variant, seed) -> the arms its pair runs; every pair on one machine (ADR 0035).
PAIRS: dict[tuple[str, int], tuple[str, ...]] = {
    **{(v, 42): ("A1", "A2", *A4) for v in F5},  # short mode for G1
    **{(v, s): ("A1", "A2") for v in DEVELOPMENT for s in SEEDS},
}
# gorgona8 takes seed 42 of the development set; medusa1 takes seeds 7 and 13 and F5.
MACHINE = {pair: ("gorgona8" if pair[1] == 42 and pair[0] in DEVELOPMENT else "medusa") for pair in PAIRS}

# Stage 2 (ticket 04): A4 on F0, F1 and F3; A1 on F7 (the complete tables) and F6 (E63's MAR files and controls).
F6 = tuple(f"{b}_{m}_60nan" for b in ("biodeg", "kc2", "vehicle") for m in ("mar", "marctl"))
F7 = tuple(f"{b}_00nan" for b in ("biodeg", "kc2", "vehicle", "spambase"))
STAGE2: dict[tuple[str, int], tuple[str, ...]] = {
    **{(v, s): A4 for v in F0 + F1 + F3 for s in SEEDS},
    **{(v, s): ("A1",) for v in F7 + F6 for s in SEEDS},
}
# Stage 3 (ticket 05): fresh draws (data seed 4848) of P and of its MCAR siblings, model seeds never used.
CONFIRMATION_SEEDS = (1515, 2626, 3737)
P_FRESH = ("biodeg_probit0p5_s4848_60nan", "vehicle_probit0p5_s4848_60nan",
           *(f"{b}_outcome3p0_s4848_60nan" for b in ("biodeg", "kc2", "spambase")))
SIBLINGS_FRESH = tuple(f"{b}_mcar_s4848_60nan" for b in ("biodeg", "vehicle", "kc2", "spambase"))
STAGE3: dict[tuple[str, int], tuple[str, ...]] = {(v, s): ("A1", "A2") for v in P_FRESH + SIBLINGS_FRESH for s in CONFIRMATION_SEEDS}
STAGES: dict[str, dict[tuple[str, int], tuple[str, ...]]] = {"1": PAIRS, "2": STAGE2, "3": STAGE3}


def machine(pair: tuple[str, int]) -> str:
    """ADR 0035: every arm of a (variant, seed) pair, across stages, on one machine."""
    if pair in MACHINE:
        return MACHINE[pair]
    if pair[1] == 42:
        return "gorgona8"
    return "medusa"


def cell_dir(arm: str, variant: str, seed: int, root: Path = STATE) -> Path:
    return root / arm.replace(":", "_") / f"{variant}_s{seed}"


def done(arm: str, variant: str, seed: int, root: Path = STATE) -> bool:
    d = cell_dir(arm, variant, seed, root) / "metrics"
    return (d / f"{variant}_metrics.csv").exists() and len(list(d.glob("c15_readout_fold_*.pt"))) == 5


def train(arm: str, variant: str, seed: int) -> None:
    from src.training.config import build_training_parser, logged_hyperparameters, resolve_training_request
    from train import main as train_main

    base = logged_hyperparameters(resolve_training_request(build_training_parser().parse_args(["--dataset_name", variant])))
    override = {**base, "PRETRAIN_OBJECTIVE": "embedding_normalized", "C15_ARM": arm}
    out = cell_dir(arm, variant, seed)
    args = Namespace(dataset_name=variant, label_column="class", output_dir=str(out / "results"),
                     metrics_dir=str(out / "metrics"), disable_mlflow=True, plot_losses=False, save_model=False,
                     seed=seed, cv_folds=5, task="classification", mlflow_tags={}, hyperparams_override=override)
    train_main(args, return_metrics=True)
    print("CELL_DONE", arm, variant, seed, flush=True)


def todo(which: int, queues: int) -> None:
    import os

    pairs = STAGES[os.environ.get("C15_STAGE", "1")]
    mine = [pair for pair in pairs if machine(pair) == HOST]
    # Dearest first: spambase's pairs, then the rest, so the queues end together.
    mine.sort(key=lambda p: (not p[0].startswith("spambase"), len(pairs[p]) == 2, p[1], p[0]))
    for k, (variant, seed) in enumerate(mine):
        if k % queues == which - 1:
            for arm in pairs[(variant, seed)]:
                if not done(arm, variant, seed):
                    print(arm, variant, seed)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and len(argv) == 4:
        train(argv[1], argv[2], int(argv[3]))
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    else:
        sys.exit(__doc__)
