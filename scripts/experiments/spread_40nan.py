"""E105: does the re-masking spread still rank errors under MNAR and MAR at 40nan? (imputation-protocol/42, ADR 0030).

    uv run --python 3.11 python -B scripts/experiments/spread_40nan.py train <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/spread_40nan.py --todo <n> <queues>
    uv run --python 3.11 python -B scripts/experiments/spread_40nan.py --tally

E84's study (``spread_mechanisms.py``) at 40nan: the same 12 shifted variants, the same seeds, the same
cell (today's default, ``--score_checkpoints --score_remask``, five folds) and the same measure and
rule. Two differences: MLflow is off, and the MNAR variants' MCAR twins (``<base>_40nan``) are trained
here, as the MAR controls are. Everything is under ``STATE``.
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
from scripts.experiments.spread_mechanisms import POPULATIONS, ROWS, SEEDS, _mark, within_column  # noqa: E402

STATE = Path("/scratch2/diogoneiss" if socket.gethostname() == "gorgona8" else "/scratch/diogoneiss") / "trident-2026-10-31-spread-40nan"
MNAR = {"biodeg_probit0p25_40nan": "biodeg_40nan", "biodeg_probit_40nan": "biodeg_40nan",
        "biodeg_probit1p0_40nan": "biodeg_40nan", "biodeg_probit2p0_40nan": "biodeg_40nan",
        "vehicle_probit_40nan": "vehicle_40nan", "kc2_probit_40nan": "kc2_40nan", "spambase_probit_40nan": "spambase_40nan"}
MAR = {f"{table}_marf_40nan": f"{table}_marfctl_40nan" for table in ("biodeg", "kc2", "vehicle", "spambase", "credit-g")}
SHIFTED = [*MNAR, *MAR]
TRAINED = sorted({*MNAR, *MAR, *MNAR.values(), *MAR.values()})


def _ledger_path(variant: str, seed: int, fold: int) -> Path | None:
    found = sorted((STATE / "results" / f"{variant}_s{seed}" / variant).glob(f"*/imputation/imputation_fold_{fold}_cells.csv"))
    return found[0] if len(found) == 1 else None


def train(variant: str, seed: int) -> None:
    from train import main as train_main

    args = Namespace(
        dataset_name=variant, label_column="class",
        output_dir=str(STATE / "results" / f"{variant}_s{seed}"), metrics_dir=str(STATE / "metrics" / f"{variant}_s{seed}"),
        disable_mlflow=True, plot_losses=False, save_model=False, seed=seed, cv_folds=5, task="imputation",
        score_checkpoints=True, score_remask=True, mlflow_tags={},
    )
    train_main(args, return_metrics=True)
    print("CELL_DONE", variant, seed, flush=True)


def todo(which: int, queues: int) -> None:
    """Largest tables first, each cell to the least-loaded queue, as E84 dealt them."""
    work = sorted(((ROWS[v.split("_")[0]], v, s) for v in TRAINED for s in SEEDS), key=lambda w: (-w[0], w[1], w[2]))
    loads = [0.0] * queues
    for cost, variant, seed in work:
        target = loads.index(min(loads))
        loads[target] += cost
        if target == which - 1 and _ledger_path(variant, seed, 5) is None:
            print(variant, seed)


def _per_seed(variant: str, population: str) -> list[np.ndarray] | None:
    paths = [[_ledger_path(variant, seed, f) for f in range(1, 6)] for seed in SEEDS]
    if any(p is None for row in paths for p in row):
        return None
    return [np.array([within_column(pd.read_csv(p, dtype={"actual": object, "imputed": object}), population) for p in row if p])
            for row in paths]


def tally() -> None:
    marks: dict[str, dict[str, str]] = {readout: {} for readout in POPULATIONS}
    for variant in SHIFTED:
        control = {**MNAR, **MAR}[variant]
        parts = []
        for readout, population in POPULATIONS.items():
            ours = _per_seed(variant, population)
            if ours is None:
                parts.append(f"{readout} incomplete")
                continue
            v = common.verdict(ours)
            marks[readout][variant] = _mark(v)
            line = f"{readout} num_rank {v.mean:+.3f} [{v.low:+.3f}, {v.high:+.3f}] -> {marks[readout][variant]}"
            twin = _per_seed(control, population)
            if twin is not None and np.isfinite(np.concatenate(twin)).all():
                d = common.verdict([a - b for a, b in zip(ours, twin)])
                line += f"; minus {control} {d.mean:+.3f} [{d.low:+.3f}, {d.high:+.3f}]"
            parts.append(line)
        print(f"- `{variant}`: " + "; ".join(parts))
    print()
    for readout, by_variant in marks.items():
        print(f"- {readout}: signal on {list(by_variant.values()).count('signal')} of {len(by_variant)}, "
              f"against on {[v for v, m in by_variant.items() if m == 'against'] or 'none'}")
    if any(len(by_variant) < len(SHIFTED) for by_variant in marks.values()):
        print("\nPrimary: incomplete")
        return
    k_signal = [v for v, m in marks["K"].items() if m == "signal"]
    h_signal = [v for v, m in marks["H"].items() if m == "signal"]
    against = [v for v, m in marks["K"].items() if m == "against"]
    if len(k_signal) >= 10 and len(h_signal) >= 10 and not against:
        reading = "supported"
    elif len(k_signal) <= 7 or against:
        reading = "refuted"
    else:
        reading = "mixed"
    print(f"\nPrimary: signal on {len(k_signal)} (K) and {len(h_signal)} (H) of {len(SHIFTED)}, against under K on {against or 'none'} -> {reading} ({common.RULE})")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and len(argv) == 3 and argv[1] in TRAINED:
        train(argv[1], int(argv[2]))
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    elif argv == ["--tally"]:
        tally()
    else:
        sys.exit(__doc__)
