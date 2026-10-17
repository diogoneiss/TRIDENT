"""Pre-registered analysis of the full MNAR family (imputation-protocol/06, E46).

Read-only. Run from the checkout root:

    uv run --python 3.11 python scripts/experiments/mnar_family_report.py \
        [--tally | --rung | --dial | --integrity | --todo 1|2|3|4]

Tag experiment=mnar-family-2026-10-17, arms G and Mx, seeds 42, 7 and 13. The new pairs are
spambase at 60nan and biodeg, kc2 and vehicle at 20nan, each as a value-dependent
(``_probit_``) and an MCAR variant; biodeg's tilt dial adds slopes 0.25, 1.0 and 2.0 at 60nan.
E45's cells (tag mnar-smoke-2026-10-17) supply the 60nan rung, slope 0.5 and the dial's MCAR
control. Every variant of a table shares its folds (``build_folds`` reads only the labels), so
every contrast is fold-paired. The model's headline is H (E43), never K.
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

EXPERIMENT = "mnar-family-2026-10-17"
SMOKE = "mnar-smoke-2026-10-17"  # E45
REFERENCE = "selection-replication-2026-10-04"  # E35: the same MCAR cells, decode_checkpoint=loss
PAIRS = (("spambase", 60), ("biodeg", 20), ("kc2", 20), ("vehicle", 20))
RUNG_TABLES = ("biodeg", "kc2", "vehicle")
DIAL = {0.25: "biodeg_probit0p25_60nan", 1.0: "biodeg_probit1p0_60nan", 2.0: "biodeg_probit2p0_60nan"}
DIAL_E45 = {0.5: "biodeg_probit_60nan"}
DIAL_MCAR = "biodeg_60nan"
ARMS = ("G", "Mx")
SEEDS = [42, 7, 13]
QUEUES = 4
CELL_MINUTES = {"spambase": 33.0, "biodeg": 8.0, "kc2": 2.5, "vehicle": 4.0}
COLUMN_WISE = "impute/induced/column_wise/impute_score"
_FOUND: dict[tuple[str, str], dict[tuple[str, int], list[common.Run]]] = {}

Cells = list[tuple[common.Run, pd.DataFrame]]


def mnar(table: str, rung: int) -> str:
    return f"{table}_probit_{rung}nan"


def mcar(table: str, rung: int) -> str:
    return f"{table}_{rung}nan"


def _runs(experiment: str, arm: str) -> dict[tuple[str, int], list[common.Run]]:
    if (experiment, arm) not in _FOUND:
        _FOUND[experiment, arm] = common.parents(experiment, arm)
    return _FOUND[experiment, arm]


def cells(variant: str, arm: str, experiment: str = EXPERIMENT) -> Cells | None:
    found = []
    for seed in SEEDS:
        runs = _runs(experiment, arm).get((variant, seed), [])
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


def gaps(found: Cells | None, first: str, second: str) -> list[np.ndarray] | None:
    if found is None:
        return None
    return [values(item, first) - values(item, second) for item in found]


def minus(a: list[np.ndarray] | None, b: list[np.ndarray] | None) -> list[np.ndarray] | None:
    if a is None or b is None:
        return None
    return [x - y for x, y in zip(a, b)]


def t2(untied: Cells | None, tied: Cells | None) -> list[np.ndarray] | None:
    if untied is None or tied is None:
        return None
    return [values(a, "column_wise") - values(b, "H") for a, b in zip(untied, tied)]


def line(label: str, per_seed: list[np.ndarray] | None) -> str | None:
    if per_seed is None:
        print(f"  {label}: incomplete")
        return None
    v = common.verdict(per_seed)
    print(f"  {label}: {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] -> {v.call}")
    return v.call


def contrasts(mnar_variant: str, mcar_variant: str, mnar_from: str, mcar_from: str) -> str | None:
    """Readings 1-4 for one MNAR variant against its MCAR control; returns the primary call."""
    g = {"MNAR": cells(mnar_variant, "G", mnar_from), "MCAR": cells(mcar_variant, "G", mcar_from)}
    mx = {"MNAR": cells(mnar_variant, "Mx", mnar_from), "MCAR": cells(mcar_variant, "Mx", mcar_from)}
    primary = None
    for mechanism in ("MNAR", "MCAR"):
        for opponent in ("bar", "missforest", "strong"):
            call = line(f"{mechanism} G/H - {opponent}", gaps(g[mechanism], "H", opponent))
            if mechanism == "MNAR" and opponent == "bar":
                primary = call
    line("(H - bar), MNAR minus MCAR", minus(gaps(g["MNAR"], "H", "bar"), gaps(g["MCAR"], "H", "bar")))
    line("(K - H), MNAR minus MCAR (the oracle's optimism)", minus(gaps(g["MNAR"], "K", "H"), gaps(g["MCAR"], "K", "H")))
    for mechanism in ("MNAR", "MCAR"):
        line(f"{mechanism} Mx column-wise - G/H", t2(mx[mechanism], g[mechanism]))
    line("t2, (Mx column-wise - G/H), MNAR minus MCAR", minus(t2(mx["MNAR"], g["MNAR"]), t2(mx["MCAR"], g["MCAR"])))
    return primary


def tally() -> None:
    """Readings 1-4 on the four new pairs, and the primary rule over them."""
    calls = []
    for table, rung in PAIRS:
        print(f"=== {mnar(table, rung)} against {mcar(table, rung)}")
        calls.append(contrasts(mnar(table, rung), mcar(table, rung), EXPERIMENT, EXPERIMENT))
    better, worse = calls.count("better"), calls.count("worse")
    if None in calls:
        reading = "incomplete"
    elif better >= 3 and worse == 0:
        reading = "supported"
    elif worse >= 2 or better <= 1:
        reading = "refuted"
    else:
        reading = "mixed"
    print()
    print(f"Primary, G's H against the weak bar under MNAR on the 4 new pairs: better {better}, worse {worse}, "
          f"other {len(calls) - better - worse} -> {reading} ({common.RULE})")


def rung() -> None:
    """The mechanism effect at 20nan (E46) beside 60nan (E45), per table, and their difference."""
    for table in RUNG_TABLES:
        print(f"=== {table}")
        effect = {}
        for level, source in ((20, EXPERIMENT), (60, SMOKE)):
            effect[level] = minus(
                gaps(cells(mnar(table, level), "G", source), "H", "bar"),
                gaps(cells(mcar(table, level), "G", source), "H", "bar"),
            )
            line(f"(H - bar), MNAR minus MCAR, at {level}nan", effect[level])
        line("the same, 60nan minus 20nan", minus(effect[60], effect[20]))


def dial() -> None:
    """biodeg at 60nan: each slope against E45's MCAR control, and whether the effect grows."""
    control = cells(DIAL_MCAR, "G", SMOKE)
    control_mx = cells(DIAL_MCAR, "Mx", SMOKE)
    means, calls = {}, {}
    for slope, (variant, source) in sorted(
        {**{s: (v, EXPERIMENT) for s, v in DIAL.items()}, **{s: (v, SMOKE) for s, v in DIAL_E45.items()}}.items()
    ):
        print(f"=== slope {slope:g}: {variant}")
        g, mx = cells(variant, "G", source), cells(variant, "Mx", source)
        line("G/H - bar", gaps(g, "H", "bar"))
        line("G/H - missforest", gaps(g, "H", "missforest"))
        effect = minus(gaps(g, "H", "bar"), gaps(control, "H", "bar"))
        calls[slope] = line("(H - bar), minus MCAR's", effect)
        if effect is not None:
            means[slope] = common.verdict(effect).mean
        line("(K - H), minus MCAR's", minus(gaps(g, "K", "H"), gaps(control, "K", "H")))
        line("t2, (Mx column-wise - G/H), minus MCAR's", minus(t2(mx, g), t2(control_mx, control)))
    if len(means) == 4:
        ordered = [means[s] for s in sorted(means)]
        monotone = all(a < b for a, b in zip(ordered, ordered[1:]))
        steepest = calls[max(calls)]
        dose = "yes" if monotone and steepest == "worse" else "no"
        print()
        print(f"Dose-response (the effect's mean rises with the slope, and at 2.0 it is 'worse'): {dose}"
              f" (means {', '.join(f'{m:+.4f}' for m in ordered)}; at 2.0 {steepest})")


def integrity() -> None:
    """G on the four MCAR controls must reproduce E35's cells: its K is E35's induced_checkpoint
    and its H is E35's headline, fold by fold."""
    reference = common.parents(REFERENCE)
    for table, level in PAIRS:
        variant = mcar(table, level)
        for seed in SEEDS:
            ours, theirs = _runs(EXPERIMENT, "G").get((variant, seed)), reference.get((variant, seed))
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


def design() -> list[tuple[float, str, int, str]]:
    """Every E46 cell with its estimated minutes: Mx costs about 1.8x G under MNAR and 1.15x under MCAR (E45)."""
    work = []
    for table, level in PAIRS:
        for variant, factor in ((mnar(table, level), 1.8), (mcar(table, level), 1.15)):
            for seed in SEEDS:
                for arm in ARMS:
                    work.append((CELL_MINUTES[table] * (factor if arm == "Mx" else 1.0), variant, seed, arm))
    for variant in DIAL.values():
        for seed in SEEDS:
            for arm in ARMS:
                work.append((7.3 * (1.8 if arm == "Mx" else 1.0), variant, seed, arm))
    return work


def queue(name: str) -> list[tuple[str, int, str]]:
    work = sorted(design(), key=lambda item: (-item[0], item[1], item[2], item[3]))
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
    elif argv[:1] == ["--todo"] and len(argv) == 2:
        for variant, seed, arm in queue(argv[1]):
            if (variant, seed) not in _runs(EXPERIMENT, arm):
                print(variant, seed, arm)
    elif argv[:1] == ["--rung"]:
        rung()
    elif argv[:1] == ["--dial"]:
        dial()
    elif argv in ([], ["--tally"]):
        tally()
    else:
        sys.exit(__doc__)
