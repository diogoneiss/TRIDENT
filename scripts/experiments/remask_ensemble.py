"""E75: a test-time re-masking ensemble on E44's confirmation variants (imputation-protocol/28, T36).

    uv run --python 3.11 python -B scripts/experiments/remask_ensemble.py train <variant> <seed> [--no-mlflow]
    uv run --python 3.11 python -B scripts/experiments/remask_ensemble.py --todo <n>
    uv run --python 3.11 python -B scripts/experiments/remask_ensemble.py --tally
    uv run --python 3.11 python -B scripts/experiments/remask_ensemble.py --within-column   (post hoc)

``train`` runs one cell of today's default imputation configuration, read as a default run of the
checkout reads it, five folds, with ``--score_checkpoints`` (H) and ``--score_remask`` (the passes),
MLflow tag ``experiment=remask-ensemble-2026-10-24``. ``--tally`` applies the rule fixed in the
ticket: accuracy from the fold tables, uncertainty and dispersion from the fold ledgers. Run from the
checkout root.
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

EXPERIMENT = "remask-ensemble-2026-10-24"
STATE = Path("/scratch2/diogoneiss/trident-2026-10-24-remask")
SEEDS = (707, 808, 909)
VARIANTS = [
    *(f"{table}_{level}nan" for table in ("pendigits", "letter", "electricity") for level in (40, 60, 80)),
    *(f"{table}_{level}nan" for table in ("vehicle", "biodeg", "kc2") for level in (40, 80)),
]
SESOI = 0.01
QUEUES = 4
ROWS = {"electricity": 45312, "letter": 20000, "pendigits": 10992, "biodeg": 1055, "vehicle": 846, "kc2": 522}
# (remask, single pass) fold-table columns per readout; K and H are uncalibrated, KC is ADR 0027's.
ACCURACY = {
    "K": ("impute/induced/remask/impute_score", "impute/induced/uncalibrated/impute_score"),
    "H": ("impute/induced/loss_checkpoint_remask/impute_score", "impute/induced/loss_checkpoint/impute_score"),
    "KC": ("impute/induced/remask_calibrated/impute_score", "impute/induced/impute_score"),
}
LEDGER = {"K": "induced_remask", "H": "induced_remask_loss_checkpoint"}
Z90 = 1.645


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
    work = sorted(((ROWS[v.split("_")[0]], v, s) for v in VARIANTS for s in SEEDS), key=lambda w: (-w[0], w[1], w[2]))
    loads = [0.0] * QUEUES
    for cost, variant, seed in work:
        target = loads.index(min(loads))
        loads[target] += cost
        if target == which - 1 and (variant, seed) not in done:
            print(variant, seed)


def _ledger(variant: str, seed: int, fold: int) -> pd.DataFrame:
    found = sorted((STATE / "results" / f"{variant}_s{seed}" / variant).glob(f"*/imputation/imputation_fold_{fold}_cells.csv"))
    if len(found) != 1:
        raise FileNotFoundError(f"{variant} s{seed} fold {fold}: {len(found)} ledgers")
    return pd.read_csv(found[0], dtype={"actual": object, "imputed": object})


def _signals(cells: pd.DataFrame) -> dict[str, float]:
    """One fold's uncertainty measures on one population's cells (NaN where a kind is absent or constant)."""
    out: dict[str, float] = {}
    numbers = cells[cells["kind"] == "numerical"]
    if len(numbers):
        error = (numbers["imputed"].astype(float) - numbers["actual"].astype(float)).abs()
        spread = numbers["spread"].astype(float)
        out["num_rank"] = float(spread.corr(error, method="spearman"))
        out["coverage90"] = float((error <= Z90 * spread).mean())
    labels = cells[cells["kind"] == "categorical"]
    if len(labels):
        miss = (labels["imputed"].astype(str) != labels["actual"].astype(str)).astype(float)
        out["cat_rank"] = float(labels["spread"].astype(float).corr(miss, method="spearman"))
    return out


def _dispersion(cells: pd.DataFrame) -> float:
    """metric_panel.py's dispersion: per numerical column, the fills' SD over the truths' SD, averaged."""
    ratios = []
    for _, group in cells[cells["kind"] == "numerical"].groupby("column"):
        actual, imputed = group["actual"].astype(float), group["imputed"].astype(float)
        if len(actual) > 1 and actual.std(ddof=0) > 0:
            ratios.append(float(imputed.std(ddof=0) / actual.std(ddof=0)))
    return float(np.mean(ratios)) if ratios else float("nan")


def _interval(values: list[float]) -> tuple[float, float, float, int]:
    """Mean and 95% t-interval of the usable folds' values, and how many there were."""
    usable = np.array([v for v in values if np.isfinite(v)])
    if len(usable) < 2:
        return float("nan"), float("nan"), float("nan"), len(usable)
    v = common.verdict([usable])
    return v.mean, v.low, v.high, len(usable)


def tally() -> None:
    runs = common.parents(EXPERIMENT)
    reference = common.parents("confirmation-2026-10-17")
    calls: dict[str, dict[str, str]] = {readout: {} for readout in ACCURACY}
    signals: dict[str, dict[str, list[str]]] = {readout: {"num_rank": [], "cat_rank": []} for readout in LEDGER}
    dispersion_calls: list[str] = []
    reproduced: list[float] = []
    for variant in VARIANTS:
        found = [runs.get((variant, seed)) for seed in SEEDS]
        if any(not f for f in found):
            print(f"- `{variant}`: incomplete ({sum(bool(f) for f in found)} of {len(SEEDS)} seeds)")
            continue
        tables = [common.fold_table(f[0]) for f in found if f]
        parts = []
        for readout, (ours, theirs) in ACCURACY.items():
            single = [np.asarray(t[theirs], dtype=float) for t in tables]
            diffs = [np.asarray(t[ours], dtype=float) - s for t, s in zip(tables, single)]
            v = common.verdict(diffs, sesoi=SESOI * float(np.mean([s.mean() for s in single])))
            calls[readout][variant] = v.call
            parts.append(f"{readout} {np.concatenate(single).mean():.4f}, remask − single {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
        print(f"- `{variant}`: " + "; ".join(parts))
        ledgers = {(seed, fold): _ledger(variant, seed, fold) for seed in SEEDS for fold in range(1, 6)}
        lines = []
        for readout, population in LEDGER.items():
            per_fold = [_signals(frame[frame["population"] == population]) for frame in ledgers.values()]
            shown = []
            for measure in ("num_rank", "cat_rank", "coverage90"):
                mean, low, high, n = _interval([f.get(measure, float("nan")) for f in per_fold])
                if n == 0:
                    continue
                shown.append(f"{measure} {mean:+.3f} [{low:+.3f}, {high:+.3f}] ({n} folds)")
                if measure in signals[readout]:
                    signals[readout][measure].append("signal" if np.isfinite(low) and low > 0 else "no signal")
            lines.append(f"{readout}: " + ", ".join(shown))
        single_cat = [_signals(frame[frame["population"] == "induced_uncalibrated"].assign(spread=lambda c: 1.0 - c["confidence"].astype(float))).get("cat_rank", float("nan")) for frame in ledgers.values()]
        mean, low, high, n = _interval(single_cat)
        if n:
            lines.append(f"single pass, 1 − confidence: cat_rank {mean:+.3f} [{low:+.3f}, {high:+.3f}] ({n} folds)")
        print("  - uncertainty, " + "; ".join(lines))
        diffs = [np.array([_dispersion(ledgers[seed, fold][ledgers[seed, fold]["population"] == "induced_remask"])
                           - _dispersion(ledgers[seed, fold][ledgers[seed, fold]["population"] == "induced_uncalibrated"]) for fold in range(1, 6)])
                 for seed in SEEDS]
        v = common.verdict(diffs)
        call = {"better": "lower", "worse": "higher"}.get(v.call, v.call)
        dispersion_calls.append(call)
        print(f"  - dispersion, remask − K: {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {call}")
        for seed, f in zip(SEEDS, found):
            if f and (variant, seed) in reference:
                old = common.model(common.fold_table(reference[variant, seed][0]), reference[variant, seed][0], "K")
                new = common.model(common.fold_table(f[0]), f[0], "K")
                reproduced.append(float(np.max(np.abs(old - new))))
    print()
    for readout, by_variant in calls.items():
        print(f"- remask − single, {readout}: {common.tally(by_variant.values())}")
    for readout, by_measure in signals.items():
        for measure, marks in by_measure.items():
            if marks:
                print(f"- {measure}, {readout}: signal on {marks.count('signal')} of {len(marks)} variants with the kind")
    print(f"- dispersion, remask − K: lower {dispersion_calls.count('lower')}, higher {dispersion_calls.count('higher')}, of {len(dispersion_calls)}")
    if reproduced:
        print(f"- K against E44's run of the same cell, largest fold difference: {max(reproduced):.2e} over {len(reproduced)} cells")
    if len(calls["K"]) < len(VARIANTS):
        print(f"\nPrimary: incomplete ({len(calls['K'])} of {len(VARIANTS)})")
        return
    better = {r: [v for v, c in calls[r].items() if c == "better"] for r in ("K", "H")}
    worse = {r: [v for v, c in calls[r].items() if c == "worse"] for r in ("K", "H")}
    if all(len(better[r]) >= 8 and len(worse[r]) <= 1 for r in ("K", "H")):
        reading = "supported"
    elif len(better["K"]) <= 3 or len(worse["K"]) >= 5:
        reading = "refuted"
    else:
        reading = "mixed"
    print(f"\nPrimary: remask better than the single pass on {len(better['K'])} (K) and {len(better['H'])} (H) of {len(VARIANTS)}, "
          f"worse on {worse['K'] or 'none'} (K) and {worse['H'] or 'none'} (H) -> {reading} ({common.RULE})")


def within_column() -> None:
    """Post hoc, after E75's Outcome (not pre-registered): ``num_rank`` within each numerical column,
    averaged over the columns with a usable value, per fold; does the spread rank cells, or only
    columns? Also the column-level part: Spearman of each column's mean spread against its RMSE."""
    for variant in VARIANTS:
        parts = []
        for readout, population in LEDGER.items():
            within, between = [], []
            for seed in SEEDS:
                for fold in range(1, 6):
                    frame = _ledger(variant, seed, fold)
                    numbers = frame[(frame["population"] == population) & (frame["kind"] == "numerical")]
                    error = (numbers["imputed"].astype(float) - numbers["actual"].astype(float)).abs()
                    cells = numbers.assign(error=error, spread=numbers["spread"].astype(float))
                    ranks = [g["spread"].corr(g["error"], method="spearman") for _, g in cells.groupby("column") if len(g) > 2]
                    within.append(float(np.nanmean(ranks)) if np.isfinite(ranks).any() else float("nan"))
                    by_column = cells.groupby("column").agg(spread=("spread", "mean"), rmse=("error", lambda e: float(np.sqrt(np.mean(e ** 2)))))
                    between.append(float(by_column["spread"].corr(by_column["rmse"], method="spearman")) if len(by_column) > 2 else float("nan"))
            for name, values in (("within", within), ("columns", between)):
                mean, low, high, n = _interval(values)
                parts.append(f"{readout} {name} {mean:+.3f} [{low:+.3f}, {high:+.3f}] ({n})")
        print(f"- `{variant}`: " + "; ".join(parts))


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and len(argv) in (3, 4):
        train(argv[1], int(argv[2]), argv[3:] == ["--no-mlflow"])
    elif argv[:1] == ["--todo"] and len(argv) == 2:
        todo(int(argv[1]))
    elif argv == ["--within-column"]:
        within_column()
    elif argv in (["--tally"], []):
        tally()
    else:
        sys.exit(__doc__)
