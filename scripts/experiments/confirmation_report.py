"""Pre-registered analysis of the confirmation on the 15 unused variants (imputation-protocol/04, E44).

Read-only. Lower impute_score is better; a negative (model - opponent) difference favours the
model. Run from the checkout root:

    uv run --python 3.11 python scripts/experiments/confirmation_report.py [variant ...]
    uv run --python 3.11 python scripts/experiments/confirmation_report.py --tally
    uv run --python 3.11 python scripts/experiments/confirmation_report.py --todo 1|2|3|4

One run per (variant, seed), tag experiment=confirmation-2026-10-17, seeds 707, 808 and 909, never
used before. Each run carries K (the headline), H and M (``--score_checkpoints``) and KC
(``--score_calibrated``) on one trajectory, and the live baselines (the bar of 5 and missForest).
The offline opponents are read from their passes' per-fold CSVs on the same folds: E41's linear
arms (``LINEAR``) and E42's tuned bar (``TUNED``). Every reading goes through ``report_common``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import report_common as common  # noqa: E402

VARIANTS = [
    *(f"{table}_{level}nan" for table in ("pendigits", "letter", "electricity") for level in (40, 60, 80)),
    *(f"{table}_{level}nan" for table in ("vehicle", "biodeg", "kc2") for level in (40, 80)),
]
EIGHTY = [variant for variant in VARIANTS if variant.endswith("_80nan")]
SEEDS = [707, 808, 909]
EXPERIMENT = "confirmation-2026-10-17"
SESOI = 0.01
# The offline passes on E44's folds: E41's driver and E42's, one CSV per (variant, seed).
LINEAR = Path("/scratch2/diogoneiss/trident-2026-10-17-linear-baselines/csv")
TUNED = Path("/scratch2/diogoneiss/trident-2026-10-17-baseline-budget/csv")
TUNED_KEY = "impute/induced/baseline/tuned_bar/impute_score"
LINEAR_ARMS = ("condexp", "softimpute", "ice")
QUEUES = 4
# Minutes a cell, from the store: E34 on the 20nan rungs (default encoder), E40 on the small tables.
CELL_MINUTES = {
    **{f"electricity_{level}nan": 110.0 for level in (40, 60, 80)},
    **{f"letter_{level}nan": 59.0 for level in (40, 60, 80)},
    **{f"pendigits_{level}nan": 38.0 for level in (40, 60, 80)},
    **{f"vehicle_{level}nan": 4.6 for level in (40, 80)},
    **{f"biodeg_{level}nan": 9.9 for level in (40, 80)},
    **{f"kc2_{level}nan": 3.6 for level in (40, 80)},
}
_RUNS: dict[tuple[str, int], list[common.Run]] | None = None


def runs() -> dict[tuple[str, int], list[common.Run]]:
    global _RUNS
    if _RUNS is None:
        _RUNS = common.parents(EXPERIMENT)
    return _RUNS


def cell(variant: str, seed: int) -> tuple[common.Run, pd.DataFrame] | None:
    found = runs().get((variant, seed), [])
    if len(found) > 1:
        sys.exit(f"{variant} s{seed}: {len(found)} finished runs; the design allows one")
    if not found:
        return None
    frame = common.fold_table(found[0])
    for directory in (LINEAR, TUNED):
        path = directory / f"{variant}_s{seed}.csv"
        if path.exists():
            extra = pd.read_csv(path).sort_values("fold").reset_index(drop=True)
            new = [column for column in extra.columns if column.startswith("impute/") and column not in frame.columns]
            frame = pd.concat([frame, extra[new]], axis=1)
    return found[0], frame


def cells(variant: str) -> list[tuple[common.Run, pd.DataFrame]] | None:
    found = [cell(variant, seed) for seed in SEEDS]
    complete = [item for item in found if item is not None]
    return complete if len(complete) == len(SEEDS) else None


def contrast(variant: str, readout: str, opponent: str) -> common.Verdict | None:
    """``readout`` (K, H, M, KC) minus ``opponent`` (an opponent of ``report_common``, ``tuned``,
    or another readout), over the 15 fold pairs."""
    found = cells(variant)
    if found is None:
        return None
    per_seed, scale = [], []
    for run, frame in found:
        if readout == "KC":
            mine = np.asarray(frame["impute/induced/calibrated/impute_score"], dtype=float)
        else:
            mine = common.model(frame, run, readout)
        if opponent in ("K", "H", "M"):
            theirs = common.model(frame, run, opponent)
        elif opponent == "tuned":
            if TUNED_KEY not in frame.columns:
                return None
            theirs = np.asarray(frame[TUNED_KEY], dtype=float)
        else:
            # The strong tier and the best of all need E41's arms on these folds; without them
            # they would silently read missForest alone.
            if opponent in ("strong", "best_of_all") and not set(LINEAR_ARMS) <= set(common.baselines_in(frame.columns, "induced")):
                return None
            try:
                theirs = common.opponent(frame, opponent)[1]
            except KeyError:
                return None
        per_seed.append(mine - theirs)
        scale.append(float(theirs.mean()))
    return common.verdict(per_seed, sesoi=SESOI * float(np.mean(scale)))


def report(variant: str) -> None:
    print(f"=== {variant}")
    found = cells(variant)
    if found is None:
        print("  incomplete")
        return
    for readout, opponent in (
        ("K", "bar"), ("K", "tuned"), ("K", "missforest"), ("K", "strong"), ("K", "best_of_all"), ("K", "per_type"),
        ("H", "bar"), ("M", "bar"), ("K", "H"), ("K", "M"), ("M", "H"), ("KC", "K"),
    ):
        result = contrast(variant, readout, opponent)
        if result is None:
            print(f"  {readout} - {opponent}: not available yet")
            continue
        seeds = ", ".join(f"{d:+.4f}" for d in result.seeds)
        print(f"  {readout} - {opponent}: {result.mean:+.4f} [{result.low:+.4f}, {result.high:+.4f}] ({seeds}) -> {result.call}")
    frames = pd.concat([frame for _, frame in found])
    print(f"  minutes a cell {float(frames['total_seconds'].sum() / 60 / len(SEEDS)):.1f}")


def tally() -> None:
    incomplete = [variant for variant in VARIANTS if cells(variant) is None]
    if incomplete:
        print(f"reading (imputation-protocol/04): incomplete, missing {incomplete}")
        return
    against_bar = {v: contrast(v, "K", "bar") for v in VARIANTS}
    against_tuned = {v: contrast(v, "K", "tuned") for v in VARIANTS}
    if any(result is None for result in against_tuned.values()):
        print("reading (imputation-protocol/04): waits on E42's tuned bar on E44's folds")
    else:
        wins = [v for v in VARIANTS if (b := against_bar[v]) and (u := against_tuned[v]) and b.call == u.call == "better"]
        losses = [v for v in VARIANTS if (b := against_bar[v]) and (u := against_tuned[v]) and "worse" in (b.call, u.call)]
        if len(wins) >= 8 and len(losses) <= 2:
            reading = "supported"
        elif len(losses) >= 5 or len(wins) <= 4:
            reading = "refuted"
        else:
            reading = "mixed"
        print(f"reading (imputation-protocol/04): {reading}: better than both bars on {len(wins)} of 15, worse than either on {len(losses)} ({common.RULE})")
        print(f"  better: {wins}; worse: {losses}")
    for opponent in ("bar", "tuned", "missforest", "strong", "best_of_all", "per_type"):
        results = {v: contrast(v, "K", opponent) for v in VARIANTS}
        if any(result is None for result in results.values()):
            print(f"  K vs {opponent}: not available yet")
            continue
        print(f"  K vs {opponent}: {common.tally(r.call for r in results.values() if r is not None)}")
        eighty = [r.call for v, r in results.items() if v in EIGHTY and r is not None]
        print(f"    at 80nan: {common.tally(eighty)}")
        gaps = {v: r.mean for v, r in results.items() if r is not None}
        tables, below, p = common.across_tables(gaps)
        print(f"    across tables (secondary, mean gap per table, two-sided Wilcoxon): {below} of {tables} below zero, p = {p:.3f}")
    for readout, opponent in (("K", "H"), ("K", "M"), ("M", "H"), ("H", "bar"), ("M", "bar"), ("KC", "K")):
        pairs = [contrast(v, readout, opponent) for v in VARIANTS]
        print(f"  {readout} - {opponent}: {common.tally(r.call for r in pairs if r is not None)}")
    calibration = {v: contrast(v, "KC", "K") for v in VARIANTS}
    k_better = [v for v, r in calibration.items() if r is not None and r.call == "worse"]
    forty = [v for v, r in calibration.items() if v.endswith("_40nan") and r is not None and r.call == "better"]
    higher = [v for v, r in calibration.items() if not v.endswith("_40nan") and r is not None and r.call == "better"]
    if k_better:
        t20 = f"calibration refuted here (K better on {k_better})"
    elif forty:
        t20 = f"threshold 40 supported (KC better at 40nan on {forty})"
    elif higher:
        t20 = f"threshold 60 suggested (KC better only at 60nan or more: {higher})"
    else:
        t20 = "no calibration verdict"
    print(f"  T20's threshold (secondary): {t20}")


def queue(name: str) -> list[tuple[str, int]]:
    order = sorted(
        ((CELL_MINUTES[v], v, s) for v in VARIANTS for s in SEEDS),
        key=lambda item: (-item[0], VARIANTS.index(item[1]), SEEDS.index(item[2])),
    )
    loads = [0.0] * QUEUES
    assigned: list[list[tuple[str, int]]] = [[] for _ in range(QUEUES)]
    for minutes, variant, seed in order:
        target = loads.index(min(loads))
        loads[target] += minutes
        assigned[target].append((variant, seed))
    return assigned[int(name) - 1]


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--tally"]:
        tally()
    elif argv[:1] == ["--todo"]:
        for variant, seed in queue(argv[1]):
            if (variant, seed) not in runs():
                print(variant, seed)
    else:
        for name in argv or VARIANTS:
            report(name)
