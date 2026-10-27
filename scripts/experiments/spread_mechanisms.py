"""E84: does the re-masking spread still rank errors under MNAR and MAR? (imputation-protocol/32, ADR 0030).

    uv run --python 3.11 python -B scripts/experiments/spread_mechanisms.py train <variant> <seed> [--no-mlflow]
    uv run --python 3.11 python -B scripts/experiments/spread_mechanisms.py --todo <n>
    uv run --python 3.11 python -B scripts/experiments/spread_mechanisms.py --tally

``train`` runs one cell of today's default with ``--score_checkpoints --score_remask``, five folds, MLflow
tag ``experiment=spread-mechanisms-2026-10-27``, as E76 ran. ``--tally`` reads E76's primary measure,
the within-column ``num_rank``: per fold, the Spearman between a numerical cell's spread and its absolute
error within each column, averaged over the columns. It reads it at K (``induced_remask``) and H
(``induced_remask_loss_checkpoint``). The MNAR pairs' MCAR twins are E79's runs at the same seeds, which
carry the spread at K (E79 had no ``--score_checkpoints``).
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

EXPERIMENT = "spread-mechanisms-2026-10-27"
STATE = Path("/scratch2/diogoneiss/trident-2026-10-27-spread-mechanisms")
E79_STATE = Path("/scratch2/diogoneiss/trident-2026-10-27-occlusion-60")
SEEDS = (101, 202, 303)
MNAR = {"biodeg_probit0p25_60nan": "biodeg_60nan", "biodeg_probit_60nan": "biodeg_60nan",
        "biodeg_probit1p0_60nan": "biodeg_60nan", "biodeg_probit2p0_60nan": "biodeg_60nan",
        "vehicle_probit_60nan": "vehicle_60nan", "kc2_probit_60nan": "kc2_60nan", "spambase_probit_60nan": "spambase_60nan"}
MAR = {f"{table}_marf_60nan": f"{table}_marfctl_60nan" for table in ("biodeg", "kc2", "vehicle", "spambase", "credit-g")}
SHIFTED = [*MNAR, *MAR]
TRAINED = [*MNAR, *MAR, *MAR.values()]
ROWS = {"spambase": 4601, "biodeg": 1055, "credit-g": 1000, "vehicle": 846, "kc2": 522}
QUEUES = 4
POPULATIONS = {"K": "induced_remask", "H": "induced_remask_loss_checkpoint"}


def train(variant: str, seed: int, no_mlflow: bool) -> None:
    from src.training.config import complete_configuration, load_hyperparameters
    from train import main as train_main

    resolved, source = load_hyperparameters(Namespace(dataset_name=variant, task="imputation"))
    print("cell", variant, seed, "source", source, "resolved", json.dumps(complete_configuration(resolved, "imputation")), flush=True)
    args = Namespace(
        dataset_name=variant, label_column="class",
        output_dir=str(STATE / "results" / f"{variant}_s{seed}"), metrics_dir=str(STATE / "metrics" / f"{variant}_s{seed}"),
        disable_mlflow=no_mlflow, plot_losses=False, save_model=False, seed=seed, cv_folds=5, task="imputation",
        score_checkpoints=True, score_remask=True,
        mlflow_tags={} if no_mlflow else {"experiment": EXPERIMENT},
    )
    metrics = train_main(args, return_metrics=True) or {}
    print("CELL_DONE", json.dumps({k: v for k, v in metrics.items() if k.startswith("impute/induced/") and k.endswith("/impute_score") and "/baseline/" not in k}), flush=True)


def todo(which: int) -> None:
    done = common.parents(EXPERIMENT)
    work = sorted(((ROWS[v.split("_")[0]], v, s) for v in TRAINED for s in SEEDS), key=lambda w: (-w[0], w[1], w[2]))
    loads = [0.0] * QUEUES
    for cost, variant, seed in work:
        target = loads.index(min(loads))
        loads[target] += cost
        if target == which - 1 and (variant, seed) not in done:
            print(variant, seed)


def _ledger(variant: str, seed: int, fold: int) -> pd.DataFrame:
    state = E79_STATE if variant in MNAR.values() else STATE
    found = sorted((state / "results" / f"{variant}_s{seed}" / variant).glob(f"*/imputation/imputation_fold_{fold}_cells.csv"))
    if len(found) != 1:
        raise FileNotFoundError(f"{variant} s{seed} fold {fold}: {len(found)} ledgers in {state}")
    return pd.read_csv(found[0], dtype={"actual": object, "imputed": object})


def within_column(frame: pd.DataFrame, population: str) -> float:
    """E76's measure on one fold: Spearman(spread, |error|) within each numerical column, averaged."""
    numbers = frame[(frame["population"] == population) & (frame["kind"] == "numerical")]
    if numbers.empty:
        return float("nan")
    error = (numbers["imputed"].astype(float) - numbers["actual"].astype(float)).abs()
    cells = numbers.assign(error=error, spread=numbers["spread"].astype(float))
    ranks = [g["spread"].corr(g["error"], method="spearman") for _, g in cells.groupby("column") if len(g) > 2]
    finite = [r for r in ranks if np.isfinite(r)]
    return float(np.mean(finite)) if finite else float("nan")


def _per_seed(variant: str, population: str) -> list[np.ndarray] | None:
    try:
        return [np.array([within_column(_ledger(variant, seed, f), population) for f in range(1, 6)]) for seed in SEEDS]
    except FileNotFoundError:
        return None


def _mark(v: common.Verdict) -> str:
    if v.low > 0 and all(s > 0 for s in v.seeds):
        return "signal"
    return "against" if v.high < 0 and all(s < 0 for s in v.seeds) else "no signal"


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
    if argv[:1] == ["train"] and len(argv) in (3, 4):
        train(argv[1], int(argv[2]), argv[3:] == ["--no-mlflow"])
    elif argv[:1] == ["--todo"] and len(argv) == 2:
        todo(int(argv[1]))
    elif argv in (["--tally"], []):
        tally()
    else:
        sys.exit(__doc__)
