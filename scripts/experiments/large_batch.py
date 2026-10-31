"""E101: pendigits at batch 1024 with the learning rates x4, and a longer decode (imputation-protocol/38, T11).

    uv run --python 3.11 python -B scripts/experiments/large_batch.py train <arm> <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/large_batch.py --todo <n> <queues>
    uv run --python 3.11 python -B scripts/experiments/large_batch.py --tally

``train`` runs one imputation cell, today's default on five folds with MLflow off, into
``STATE/<arm>/<variant>_s<seed>``: arm ``M`` as it is; ``P`` with BATCH 1024 and LR_PRE and LR_DECODE
times 4 (E28's Q4) over the variant's resolved configuration; ``P600`` as P with EPOCHS_DECODE 600.
``--todo`` deals the (variant, seed) pairs of this machine's seeds over the queues, every arm of a
pair in one queue. ``--tally`` reads both machines' directories (medusa1's copied under
``STATE/medusa``) and prints each arm − M per variant and the primary reading (P600 − M).
E102 (imputation-protocol/39): ``--study e102`` first, M and P600 on fresh seeds 7878/8989/9090.
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

STATE = Path("/scratch2/diogoneiss" if socket.gethostname() == "gorgona8" else "/scratch/diogoneiss") / "trident-2026-10-30-large-batch"
VARIANTS = tuple(f"pendigits_{r}nan" for r in (20, 40, 60, 80))
SEEDS = {"gorgona8": (42,), "medusa": (7, 13)}  # ADR 0035: split by seed, both arms of a pair on one machine
ARMS: tuple[str, ...] = ("M", "P", "P600")


def _use_e102() -> None:
    """E102's confirmation: M and P600 only, on fresh seeds, in its own state."""
    global STATE, SEEDS, ARMS
    STATE = STATE.parent / "trident-2026-10-30-large-batch-e102"
    SEEDS = {"gorgona8": (7878,), "medusa": (8989, 9090)}
    ARMS = ("M", "P600")


CHANGES = {"P": {"BATCH": 1024, "LR_SCALE": 4}, "P600": {"BATCH": 1024, "LR_SCALE": 4, "EPOCHS_DECODE": 600}}
PRIMARY, SESOI = "impute/induced/impute_score", 0.01
SECONDARY = ("impute/masked/impute_score",)


def cell_dir(root: Path, arm: str, variant: str, seed: int) -> Path:
    return root / arm / f"{variant}_s{seed}"


def train(arm: str, variant: str, seed: int) -> None:
    from src.training.config import build_training_parser, logged_hyperparameters, resolve_training_request
    from train import main as train_main

    out = cell_dir(STATE, arm, variant, seed)
    override = None
    if arm in CHANGES:
        # The variant's resolved configuration (its file, ADR 0018's encoder on 20nan), then the arm's changes.
        base = logged_hyperparameters(resolve_training_request(
            build_training_parser().parse_args(["--dataset_name", variant, "--task", "imputation"])))
        change = dict(CHANGES[arm])
        scale = float(change.pop("LR_SCALE"))
        override = {**base, **change, "LR_PRE": float(base["LR_PRE"]) * scale, "LR_DECODE": float(base["LR_DECODE"]) * scale}  # type: ignore[arg-type]
    args = Namespace(dataset_name=variant, label_column="class", output_dir=str(out / "results"),
                     metrics_dir=str(out / "metrics"), disable_mlflow=True, plot_losses=False,
                     save_model=False, seed=seed, cv_folds=5, task="imputation", mlflow_tags={},
                     **({"hyperparams_override": override} if override else {}))
    train_main(args, return_metrics=True)
    print("CELL_DONE", arm, variant, seed, flush=True)


def folds(root: Path, arm: str, variant: str, seed: int) -> pd.DataFrame | None:
    found = sorted(cell_dir(root, arm, variant, seed).rglob(f"{variant}_metrics.csv"))
    if len(found) != 1:
        return None
    frame = pd.read_csv(found[0])
    return frame[pd.to_numeric(frame["fold"], errors="coerce").notna()].sort_values("fold").reset_index(drop=True)


def todo(which: int, queues: int) -> None:
    pairs = [(v, s) for s in SEEDS[socket.gethostname()] for v in VARIANTS]
    for k, (variant, seed) in enumerate(pairs):
        if k % queues == which - 1:
            for arm in ARMS:
                if folds(STATE, arm, variant, seed) is None:
                    print(arm, variant, seed)


def tally() -> None:
    roots = {seed: STATE if host == "gorgona8" else STATE / "medusa" for host, seeds in SEEDS.items() for seed in seeds}
    calls: dict[str, str] = {}
    missing = []
    for arm in ARMS[1:]:
        print(f"## {arm} − M on {PRIMARY}, relative to M; 15 fold pairs a variant ({common.RULE}, SESOI {SESOI})")
        for variant in VARIANTS:
            per_seed: list[np.ndarray] = []
            extra: dict[str, list[float]] = {m: [] for m in SECONDARY}
            for seed, root in roots.items():
                m, s = folds(root, "M", variant, seed), folds(root, arm, variant, seed)
                if m is None or s is None:
                    missing.append(f"{variant} s{seed}")
                    break
                per_seed.append(np.asarray((s[PRIMARY] - m[PRIMARY]) / m[PRIMARY], dtype=float))
                for k in SECONDARY:
                    extra[k].append(float(((s[k] - m[k]) / m[k]).mean()))
            else:
                v = common.verdict(per_seed, sesoi=SESOI)
                if arm == "P600":
                    calls[variant] = v.call
                print(f"- {variant}: {v.mean:+.2%} [{v.low:+.2%}, {v.high:+.2%}] seeds "
                      + ", ".join(f"{x:+.2%}" for x in v.seeds) + f" -> {v.call}; masked {np.mean(extra[SECONDARY[0]]):+.2%}")
    if missing:
        print(f"\n- missing: {len(missing)}: {', '.join(missing)}\n\nPrimary: incomplete")
        return
    better = sum(c == "better" for c in calls.values())
    worse = sum(c == "worse" for c in calls.values())
    reading = ("supported" if better >= 2 and worse == 0 else "refuted" if better == 0 else "mixed")
    print(f"\nPrimary (P600 − M): {common.tally(calls.values())} of {len(calls)} -> {reading}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:2] == ["--study", "e102"]:
        _use_e102()
        argv = argv[2:]
    if argv[:1] == ["train"] and len(argv) == 4 and argv[1] in ARMS and argv[2] in VARIANTS:
        train(argv[1], argv[2], int(argv[3]))
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    elif argv == ["--tally"]:
        tally()
    else:
        sys.exit(__doc__)
