"""Pre-registered analysis of the MNAR smoke family (imputation-protocol/05, E45).

Read-only. Run from the checkout root:

    uv run --python 3.11 python scripts/experiments/mnar_smoke_report.py [--tally | --integrity | --todo 1|2|3|4]

Tag experiment=mnar-smoke-2026-10-17, arms G and Mx, seeds 42, 7 and 13, on biodeg, kc2 and vehicle
at 60nan, each as a value-dependent (``_probit_60nan``) and an MCAR (``_60nan``) variant. The two
variants of a table share their folds (``build_folds`` reads only the labels), so differences
between mechanisms are fold-paired. Every reading goes through ``report_common``; the model's
headline here is H (E43), never K, which reads the true values of MNAR-hidden validation cells.
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

TABLES = ("biodeg", "kc2", "vehicle")
MECHANISMS = {"MNAR": "{}_probit_60nan", "MCAR": "{}_60nan"}
ARMS = ("G", "Mx")
SEEDS = [42, 7, 13]
EXPERIMENT = "mnar-smoke-2026-10-17"
REFERENCE = "selection-replication-2026-10-04"  # E35: the same MCAR cells, decode_checkpoint=loss
QUEUES = 4
CELL_MINUTES = {"biodeg": 12.0, "kc2": 4.5, "vehicle": 6.0}
COLUMN_WISE = "impute/induced/column_wise/impute_score"
_FOUND: dict[str, dict[tuple[str, int], list[common.Run]]] = {}


def _runs(arm: str) -> dict[tuple[str, int], list[common.Run]]:
    if arm not in _FOUND:
        _FOUND[arm] = common.parents(EXPERIMENT, arm)
    return _FOUND[arm]


def cells(variant: str, arm: str) -> list[tuple[common.Run, pd.DataFrame]] | None:
    found = []
    for seed in SEEDS:
        runs = _runs(arm).get((variant, seed), [])
        if len(runs) > 1:
            sys.exit(f"{variant} s{seed} {arm}: {len(runs)} finished runs; the design allows one")
        if not runs:
            return None
        found.append((runs[0], common.fold_table(runs[0])))
    return found


def values(item: tuple[common.Run, pd.DataFrame], what: str) -> np.ndarray:
    run, frame = item
    if what in ("K", "H", "M"):
        return common.model(frame, run, what)
    if what == "column_wise":
        return np.asarray(frame[COLUMN_WISE], dtype=float)
    return common.opponent(frame, what)[1]


def gap(table: str, mechanism: str, arm: str, first: str, second: str) -> list[np.ndarray] | None:
    found = cells(MECHANISMS[mechanism].format(table), arm)
    if found is None:
        return None
    return [values(item, first) - values(item, second) for item in found]


def tally() -> None:
    for table in TABLES:
        print(f"=== {table}")
        for mechanism in MECHANISMS:
            for opponent in ("bar", "missforest", "strong"):
                gaps = gap(table, mechanism, "G", "H", opponent)
                if gaps is None:
                    print(f"  {mechanism}: incomplete")
                    break
                v = common.verdict(gaps)
                print(f"  {mechanism} G/H - {opponent}: {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] -> {v.call}")
        for label, (arm_a, first_a, second_a), (arm_b, first_b, second_b) in (
            ("(H - bar), MNAR minus MCAR", ("G", "H", "bar"), ("G", "H", "bar")),
            ("(K - H), MNAR minus MCAR (the oracle's optimism)", ("G", "K", "H"), ("G", "K", "H")),
        ):
            a, b = gap(table, "MNAR", arm_a, first_a, second_a), gap(table, "MCAR", arm_b, first_b, second_b)
            if a is None or b is None:
                continue
            v = common.verdict([x - y for x, y in zip(a, b)])
            print(f"  {label}: {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] -> {v.call}")
        # t2's positive control: Mx asked column-wise (gaps as [NULL]) against G's H (gaps as [MASK]).
        per = {}
        for mechanism in MECHANISMS:
            untied = cells(MECHANISMS[mechanism].format(table), "Mx")
            tied = cells(MECHANISMS[mechanism].format(table), "G")
            if untied is None or tied is None:
                break
            per[mechanism] = [values(a, "column_wise") - values(b, "H") for a, b in zip(untied, tied)]
            v = common.verdict(per[mechanism])
            print(f"  {mechanism} Mx column-wise - G/H: {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] -> {v.call}")
        if len(per) == 2:
            v = common.verdict([x - y for x, y in zip(per["MNAR"], per["MCAR"])])
            print(f"  t2, (Mx column-wise - G/H), MNAR minus MCAR: {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] -> {v.call}")


def integrity() -> None:
    """G on the MCAR variants must reproduce E35's cells: its K is E35's induced_checkpoint and
    its H is E35's headline, fold by fold."""
    reference = common.parents(REFERENCE)
    for table in TABLES:
        variant = MECHANISMS["MCAR"].format(table)
        for seed in SEEDS:
            ours, theirs = _runs("G").get((variant, seed)), reference.get((variant, seed))
            if not ours or not theirs:
                print(f"  {variant} s{seed}: missing")
                continue
            a, b = common.fold_table(ours[0]), common.fold_table(theirs[0])
            worst = 0.0
            for readout in ("K", "H"):
                for population in ("induced", "masked"):
                    for metric in ("impute_score", "rmse_num_z"):
                        x = common.model(a, ours[0], readout, population, metric)
                        y = common.model(b, theirs[0], readout, population, metric)
                        worst = max(worst, float(np.abs(x - y).max()))
            print(f"  {variant} s{seed}: largest difference from E35 {worst:g}")


def queue(name: str) -> list[tuple[str, int, str]]:
    work = [
        (CELL_MINUTES[table] * (1.2 if arm == "Mx" else 1.0), MECHANISMS[m].format(table), seed, arm)
        for table in TABLES for m in MECHANISMS for seed in SEEDS for arm in ARMS
    ]
    work.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    loads = [0.0] * QUEUES
    assigned: list[list[tuple[str, int, str]]] = [[] for _ in range(QUEUES)]
    for minutes, variant, seed, arm in work:
        target = loads.index(min(loads))
        loads[target] += minutes
        assigned[target].append((variant, seed, arm))
    return assigned[int(name) - 1]


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--integrity"]:
        integrity()
    elif argv[:1] == ["--todo"]:
        for variant, seed, arm in queue(argv[1]):
            if (variant, seed) not in _runs(arm):
                print(variant, seed, arm)
    else:
        tally()
