"""Pre-registered analysis of E51, today's imputation default on fresh seeds at 20nan
(imputation-protocol/08).

Read-only. Lower impute_score is better; a negative (model - opponent) difference favours the
model. Run from the checkout root:

    uv run --python 3.11 python scripts/experiments/default_refresh_report.py [variant ...]
    uv run --python 3.11 python scripts/experiments/default_refresh_report.py --tally
    uv run --python 3.11 python scripts/experiments/default_refresh_report.py --todo 1|2|3|4

One run per (variant, seed), tag experiment=default-refresh-2026-10-20, seeds 1001, 2002 and 3003,
never used before. Each run carries K (the headline at 20nan, where ADR 0027 resolves to no
calibration), H and M (``--score_checkpoints``), KC (``--score_calibrated``) and every baseline
live, the strong tier included (ADR 0021). The earlier calls come from the development seed sets
through ``thesis_table`` (``report_common.development_source`` and E41's per-fold CSVs); the
labelled cells (42/7/13 on pendigits_20nan and letter_20nan, the old encoder) are not compared.
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
from scripts.experiments import thesis_table  # noqa: E402

VARIANTS = [f"{table}_20nan" for table in (
    "electricity", "letter", "pendigits", "spambase", "kr-vs-kp", "biodeg", "credit-g", "vehicle", "kc2")]
SEEDS = [1001, 2002, 3003]
EXPERIMENT = "default-refresh-2026-10-20"
EARLIER = ("101/202/303", "42/7/13")
SESOI = 0.01
QUEUES = 4
# Minutes a cell, from the latest default runs with a warm cache: E35 (selection-replication),
# and E38's ALL arm on the two bigger-encoder variants.
CELL_MINUTES = {
    "electricity_20nan": 101.3, "letter_20nan": 101.9, "pendigits_20nan": 72.9, "spambase_20nan": 35.4,
    "kr-vs-kp_20nan": 18.0, "biodeg_20nan": 8.0, "credit-g_20nan": 6.2, "vehicle_20nan": 4.0, "kc2_20nan": 2.1,
}
Cells = list[tuple[common.Run, pd.DataFrame]]
_FOUND: dict[tuple[str, int], list[common.Run]] | None = None


def _runs() -> dict[tuple[str, int], list[common.Run]]:
    global _FOUND
    if _FOUND is None:
        _FOUND = common.parents(EXPERIMENT)
    return _FOUND


def cells(variant: str) -> Cells | None:
    found = []
    for seed in SEEDS:
        runs = _runs().get((variant, seed), [])
        if len(runs) > 1:
            sys.exit(f"{variant} s{seed}: {len(runs)} finished runs; the design allows one")
        if not runs:
            return None
        found.append((runs[0], common.fold_table(runs[0])))
    return found


def contrast(found: Cells, readout: str, opponent: str) -> common.Verdict | None:
    """``readout`` (K, H, M or KC) minus ``opponent`` (an opponent of ``report_common`` or a
    readout), over the 15 fold pairs."""
    per_seed, scale = [], []
    for run, frame in found:
        mine = (np.asarray(frame["impute/induced/calibrated/impute_score"], dtype=float) if readout == "KC"
                else common.model(frame, run, readout))
        if opponent in ("K", "H", "M"):
            theirs = common.model(frame, run, opponent)
        else:
            if opponent in ("strong", "best_of_all") and "condexp" not in common.baselines_in(frame.columns, "induced"):
                return None
            theirs = common.opponent(frame, opponent)[1]
        per_seed.append(mine - theirs)
        scale.append(float(theirs.mean()))
    return common.verdict(per_seed, sesoi=SESOI * float(np.mean(scale)))


def earlier(variant: str, opponent: str) -> dict[str, str]:
    """The K − ``opponent`` call on each earlier seed set that ran today's default."""
    calls = {}
    for seeds in EARLIER:
        source = common.development_source(variant, seeds)
        if source is None or source.labelled:
            continue
        found = thesis_table._cells(variant, seeds, thesis_table.DEFAULT_LINEAR)
        result = None if found is None else thesis_table._contrast(found, opponent)
        if result is not None:
            calls[seeds] = result[1].call
    return calls


def _line(label: str, v: common.Verdict | None) -> str:
    if v is None:
        return f"{label}: not available"
    seeds = ", ".join(f"{d:+.4f}" for d in v.seeds)
    return f"{label}: {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] ({seeds}) -> {v.call}"


def report(variant: str) -> None:
    print(f"=== {variant}")
    found = cells(variant)
    if found is None:
        print("  incomplete")
        return
    k = float(np.concatenate([common.model(f, r, "K") for r, f in found]).mean())
    print(f"  K {k:.4f}; minutes a cell {float(pd.concat([f for _, f in found])['total_seconds'].sum() / 60 / len(SEEDS)):.1f}")
    for readout, opponent in (
        ("K", "bar"), ("K", "strong"), ("K", "missforest"), ("K", "best_of_all"), ("K", "per_type"),
        ("H", "bar"), ("M", "bar"), ("K", "H"), ("K", "M"), ("KC", "K"),
    ):
        print("  " + _line(f"{readout} - {opponent}", contrast(found, readout, opponent)))
    for opponent in ("bar", "strong"):
        print(f"  earlier K - {opponent}: {earlier(variant, opponent) or 'none'}")


REVERSED = {("better", "worse"), ("worse", "better")}


def tally() -> None:
    fresh = {v: found for v in VARIANTS if (found := cells(v)) is not None}
    missing = [v for v in VARIANTS if v not in fresh]
    print(f"complete: {len(fresh)} of {len(VARIANTS)}; not read (incomplete by the deadline): {missing or 'none'}")
    if not fresh:
        return
    for opponent in ("bar", "strong"):
        calls = {v: r.call for v, found in fresh.items() if (r := contrast(found, "K", opponent)) is not None}
        print(f"\nK - {opponent}, fresh seeds 1001/2002/3003: {common.tally(calls.values())}")
        reversals, drifts = [], []
        for variant, call in calls.items():
            before = earlier(variant, opponent)
            print(f"  {variant}: fresh {call}; earlier {before or 'none'}")
            for seeds, old in before.items():
                if (call, old) in REVERSED:
                    reversals.append(f"{variant} ({seeds}: {old} -> {call})")
                elif call != old:
                    drifts.append(f"{variant} ({seeds}: {old} -> {call})")
        if opponent == "bar":
            reading = "holds" if not reversals else "drifts"
            print(f"Primary: K - bar {reading}: reversals {reversals or 'none'} ({common.RULE})")
        else:
            print(f"Secondary: K - strong: reversals {reversals or 'none'}")
        print(f"  changes short of a reversal: {drifts or 'none'}")
    print()
    for readout, opponent in (("H", "bar"), ("M", "bar"), ("K", "H"), ("K", "M"), ("KC", "K"), ("K", "per_type")):
        pairs = [contrast(found, readout, opponent) for found in fresh.values()]
        print(f"{readout} - {opponent}: {common.tally(r.call for r in pairs if r is not None)}")
    kc = [v for v, found in fresh.items() if (r := contrast(found, "KC", "K")) is not None and r.call == "better"]
    print(f"KC better than K at 20nan (ADR 0027 keeps calibration off below 0.3): {kc or 'none'}")


def queue(name: str) -> list[tuple[str, int, float]]:
    order = sorted(
        ((CELL_MINUTES[v], v, s) for v in VARIANTS for s in SEEDS),
        key=lambda item: (-item[0], VARIANTS.index(item[1]), SEEDS.index(item[2])),
    )
    loads = [0.0] * QUEUES
    assigned: list[list[tuple[str, int, float]]] = [[] for _ in range(QUEUES)]
    for minutes, variant, seed in order:
        target = loads.index(min(loads))
        loads[target] += minutes
        assigned[target].append((variant, seed, minutes))
    return assigned[int(name) - 1]


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--todo"] and len(argv) == 2:
        for variant, seed, minutes in queue(argv[1]):
            if (variant, seed) not in _runs():
                print(variant, seed, f"{minutes:.0f}")
    elif argv == ["--tally"]:
        tally()
    elif argv and not argv[0].startswith("--"):
        for variant in argv:
            report(variant)
    elif not argv:
        for variant in VARIANTS:
            report(variant)
    else:
        sys.exit(__doc__)
