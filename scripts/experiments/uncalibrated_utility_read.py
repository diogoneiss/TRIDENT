"""E188's reads (docs/tickets/imputation-protocol/77): is the calibration of the default output (ADR 0027) why the
decoder's fill serves a downstream forest worse than the gaps left missing on ``kr-vs-kp`` and ``credit-g``?

    uv run --python 3.11 python -B scripts/experiments/uncalibrated_utility_read.py utility <fresh|old> <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/uncalibrated_utility_read.py --jobs    # the cells above not yet computed
    uv run --python 3.11 python -B scripts/experiments/uncalibrated_utility_read.py --gate
    uv run --python 3.11 python -B scripts/experiments/uncalibrated_utility_read.py --tally <core|rest>   # refuses a partial set

Two fills come from one run: ``K`` is the fold ledger's ``induced`` rows (the default output, calibrated) and ``N``
its ``induced_uncalibrated`` rows (the same model before the calibration), which ``--score_remask`` writes. ``fresh``
is this study's runs (``uncalibrated_utility_run.py``, never imported here: it repoints ``sliced_loss_run``); ``old``
is E76's runs of the core variants on seeds 101/202/303, read beside. The downstream cells and their tally are
``sliced_utility.py``'s functions (E186's) with these fills. Run from the checkout root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import c15_report  # noqa: E402
from scripts.experiments import sliced_select_run as e183  # noqa: E402
from scripts.experiments import sliced_utility as u  # noqa: E402
from src.training.imputation_metrics import CATEGORICAL, NUMERICAL  # noqa: E402

STATE = Path("/scratch/diogoneiss/trident-2026-11-23-uncalibrated-utility")
OLD = Path("/scratch2/diogoneiss/trident-2026-10-25-remask-dev")  # E76: the default run with --score_remask
CORE = ("credit-g_40nan", "credit-g_60nan", "credit-g_80nan", "kr-vs-kp_40nan", "kr-vs-kp_60nan", "kr-vs-kp_80nan")
REST = ("kc2_40nan", "kc2_80nan", "vehicle_40nan", "vehicle_80nan", "biodeg_40nan", "biodeg_80nan",
        "pendigits_40nan", "pendigits_60nan", "pendigits_80nan", "electricity_40nan", "electricity_80nan")
PARTS = {"core": CORE, "rest": REST}
SEEDS: dict[str, tuple[int, ...]] = {"fresh": (2718, 6464, 7272), "old": (101, 202, 303)}
READS = (("fresh", "core"), ("old", "core"), ("fresh", "rest"))
POPULATION = {"K": "induced", "N": "induced_uncalibrated"}
P = "impute/induced"
METRIC = {"K": P, "N": f"{P}/uncalibrated"}


def _run(variant: str, seed: int) -> Path | None:
    if seed in SEEDS["old"]:
        found = sorted((OLD / "results" / f"{variant}_s{seed}" / variant).glob("*/metrics/cv_summary.json"))
        return found[-1].parent.parent if found else None
    return e183._run(STATE, "KR", variant, seed)


def _ledgers(run: Path, fold: int) -> dict[str, pd.DataFrame]:
    """The fold's calibrated (K) and uncalibrated (N) rows, as ``sliced_run._ledger`` returns the first."""
    frame = pd.read_csv(run / "imputation" / f"imputation_fold_{fold}_cells.csv", dtype={"actual": object, "imputed": object})
    out: dict[str, pd.DataFrame] = {}
    for arm, population in POPULATION.items():
        part = frame[frame["population"] == population].reset_index(drop=True)
        assert len(part), f"{run}: fold {fold} has no {population} rows (a run without --score_remask, or not calibrated)"
        numeric = part["kind"] == NUMERICAL
        for name in ("actual", "imputed"):
            part.loc[numeric, name] = part.loc[numeric, name].astype(float)
        out[arm] = part
    return out


def _fill(state: Path, arm: str, variant: str, seed: int, fold: int) -> pd.DataFrame:  # noqa: ARG001
    run = _run(variant, seed)
    assert run is not None, f"no finished run of {variant} seed {seed}"
    return _ledgers(run, fold)[arm]


def _out(group: str, part: str) -> Path:
    return STATE / "utility" / f"{group}-{part}" / "csv"


def use(group: str, part: str) -> None:
    """Point ``sliced_utility`` (E186's cells and tally) at one read of this study."""
    assert (group, part) in READS
    u.STATES, u.VARIANTS, u.SEEDS, u.OUT = {STATE if group == "fresh" else OLD: PARTS[part]}, PARTS[part], SEEDS[group], _out(group, part)
    u.STUDY = f"E188, {part}, " + ("fresh seeds" if group == "fresh" else "not fresh: E76's runs")
    u.ARMS, u.FILLS, u._fill = ("K", "N"), ("missing", "naive", "K", "N", "truth"), _fill
    u.PRIMARY, u.TERM, u.BESIDE = (("N", "K"), ("K", "missing"), ("N", "missing")), (), (("naive", "missing"), ("truth", "missing"))


def _part(variant: str) -> str:
    return next(part for part, variants in PARTS.items() if variant in variants)


def _cells(group: str, part: str) -> list[tuple[str, int]]:
    return [(v, s) for v in PARTS[part] for s in SEEDS[group]]


def jobs() -> None:
    for group, part in READS:
        for variant, seed in _cells(group, part):
            if _run(variant, seed) is not None and not (_out(group, part) / f"{variant}_s{seed}.csv").exists():
                print("utility", group, variant, seed)


def _problems(run: Path, fresh: bool) -> list[str]:
    """What makes a run unfit to be read as K and N: its flags, and ledgers that do not reproduce its own scores."""
    config = json.loads((run / "hyperparameters.json").read_text())
    wanted = {"DECODE_CALIBRATION": "gaps", "DECODE_METRIC": "impute_score", "DECODE_SW_WEIGHT": 0.0}
    default = {} if fresh else {"DECODE_METRIC": "impute_score", "DECODE_SW_WEIGHT": 0.0}  # E76 predates the two flags
    problems = [f"{k} {config.get(k, default.get(k))} not {v}" for k, v in wanted.items() if config.get(k, default.get(k)) != v]
    metrics = pd.read_csv(run / "metrics" / "raw_fold_metrics.csv").sort_values("fold")
    assert metrics["fold"].tolist() == [1, 2, 3, 4, 5]
    for fold in range(1, 6):
        ledgers = _ledgers(run, fold)
        k, n = ledgers["K"], ledgers["N"]
        if not (k[["row", "column", "kind"]].equals(n[["row", "column", "kind"]]) and k["actual"].astype(str).equals(n["actual"].astype(str))):
            problems.append(f"fold {fold}: the two outputs are not on the same cells")
        for arm, cells in ledgers.items():
            cat, num = cells[cells["kind"] == CATEGORICAL], cells[cells["kind"] == NUMERICAL]
            mine: dict[str, float] = {}
            if len(cat):
                mine["acc_cat"] = float((cat["imputed"].astype(str) == cat["actual"].astype(str)).mean())
            if len(num):
                mine["rmse_num_z"] = float(np.sqrt(((num["imputed"].astype(float) - num["actual"].astype(float)) ** 2).mean()))
            for name, value in mine.items():
                logged = float(metrics[f"{METRIC[arm]}/{name}"].to_numpy(dtype=float)[fold - 1])
                if abs(value - logged) > 1e-9:
                    problems.append(f"fold {fold}: {arm}'s rows give {name} {value:.6f}, the run logged {logged:.6f}")
    return problems


def gate(reads: tuple[tuple[str, str], ...] = READS) -> int:
    checked = failed = 0
    for group, part in reads:
        for variant, seed in _cells(group, part):
            run = _run(variant, seed)
            if run is None:
                continue
            checked += 1
            problems = _problems(run, group == "fresh")
            if problems:
                failed += 1
                print(f"FAIL {group} {variant} seed {seed}: " + "; ".join(problems))
    print(f"gate: {checked} finished runs checked, {failed} failed (flags; both outputs on the same cells; each reproduces the run's own acc_cat and rmse_num_z)")
    return failed


def _rule(low: float, high: float, loss: float, rest_high: float) -> str:
    """The ticket's rule. ``low``/``high``: the interval of F1(N) − F1(K); ``loss``: the mean of F1(missing) − F1(K),
    above 0; ``rest_high``: the upper end of F1(N) − F1(missing)."""
    if high < 0:
        return "calibration helps here, not a cause"
    if low > 0:
        if high < loss / 2:
            return "a minor cause (under half the loss)"
        return "a cause, explains the loss" if rest_high >= 0 else "a cause, explains part of the loss"
    return "not shown, ruled out as the main cause" if high < loss / 2 else "inconclusive"


assert (_rule(-0.02, -0.01, 0.03, -0.01), _rule(0.001, 0.01, 0.03, -0.01), _rule(0.01, 0.03, 0.03, 0.001), _rule(0.01, 0.03, 0.03, -0.001),
        _rule(-0.002, 0.004, 0.03, -0.01), _rule(-0.002, 0.02, 0.03, 0.01)) == (
    "calibration helps here, not a cause", "a minor cause (under half the loss)", "a cause, explains the loss",
    "a cause, explains part of the loss", "not shown, ruled out as the main cause", "inconclusive")


def _own_forest(folder: Path) -> pd.DataFrame:
    data = pd.concat([pd.read_csv(f) for f in sorted(folder.glob("*.csv"))], ignore_index=True)
    part = data[(data["protocol"] == "own") & (data["learner"] == "forest")]
    return part.pivot_table(index=u.KEY, columns="fill", values="macro_f1").reset_index()


def cause(forest: pd.DataFrame) -> None:
    """The rule on the core pooled, on each table's three variants, and on each variant."""

    def pooled(frame: pd.DataFrame, first: str, second: str) -> NDArray[np.float64]:
        frame = frame.assign(d=frame[first] - frame[second])
        return np.asarray(frame.pivot_table(index=["seed", "fold"], columns="variant", values="d").mean(axis=1), dtype=float)

    def read(name: str, frame: pd.DataFrame) -> None:
        loss, change, rest = pooled(frame, "missing", "K"), pooled(frame, "N", "K"), pooled(frame, "N", "missing")
        verdicts = []
        for corrected in (False, True):
            size, low_loss, _ = c15_report.interval(loss, corrected)
            mean, low, high = c15_report.interval(change, corrected)
            _, _, rest_high = c15_report.interval(rest, corrected)
            verdicts.append("no read (the loss F1(missing) − F1(K) is not above 0)" if low_loss <= 0 else _rule(low, high, size, rest_high))
            if not corrected:
                line = (f"- {name}: loss {size:+.4f} (lower end {low_loss:+.4f}); F1(N) − F1(K) {mean:+.4f} [{low:+.4f}, {high:+.4f}], "
                        f"as a share of the loss {mean / size:+.2f} [{low / size:+.2f}, {high / size:+.2f}]; F1(N) − F1(missing) upper end {rest_high:+.4f}")
        plain, nb = verdicts
        print(f"{line} -> {plain}" + ("" if nb == plain else f" (fragile; corrected: {nb})"))

    print(f"\n### The rule ({u.STUDY}; forest, own): is calibration why K's fill loses to the gaps left missing?")
    read(f"core, {forest['variant'].nunique()} variants pooled per (seed, fold)", forest)
    for table in ("kr-vs-kp", "credit-g"):
        read(f"{table}, its three variants pooled", forest[forest["variant"].str.startswith(table)])
    for variant in u.VARIANTS:
        read(variant, forest[forest["variant"] == variant])


def mechanism(group: str, part: str) -> None:
    """How much the calibration changes the output: the share of gap cells where K and N differ, per kind."""
    print(f"\n### What the calibration changes ({u.STUDY}): gap cells of the test folds where the two outputs differ")
    for variant in PARTS[part]:
        cat_changed, num_changed, shift, cat_n, num_n = 0, 0, 0.0, 0, 0
        summary: list[pd.DataFrame] = []
        for seed in SEEDS[group]:
            run = _run(variant, seed)
            assert run is not None
            summary.append(pd.read_csv(run / "metrics" / "raw_fold_metrics.csv"))
            for fold in range(1, 6):
                ledgers = _ledgers(run, fold)
                k, n = ledgers["K"], ledgers["N"]
                cat, num = (k["kind"] == CATEGORICAL).to_numpy(), (k["kind"] == NUMERICAL).to_numpy()
                cat_changed += int((k.loc[cat, "imputed"].astype(str).to_numpy() != n.loc[cat, "imputed"].astype(str).to_numpy()).sum())
                moved = np.abs(k.loc[num, "imputed"].to_numpy(dtype=float) - n.loc[num, "imputed"].to_numpy(dtype=float))
                num_changed, shift = num_changed + int((moved > 1e-12).sum()), shift + float(moved.sum())
                cat_n, num_n = cat_n + int(cat.sum()), num_n + int(num.sum())
        logged = pd.concat(summary, ignore_index=True)
        notes = [f"{name.rsplit('/', 1)[1]} {logged[name].mean():.2f}" for name in (f"{P}/calibration/n_thresholded_columns", f"{P}/calibration/mean_alpha") if name in logged]
        notes += [f"{name} K {logged[f'{P}/{name}'].mean():.4f}, N {logged[f'{P}/uncalibrated/{name}'].mean():.4f}"
                  for name in ("acc_cat", "rmse_num_z", "impute_score") if f"{P}/uncalibrated/{name}" in logged]
        print(f"- {variant}: categories changed {cat_changed} of {cat_n}" + (f" ({cat_changed / cat_n:.1%})" if cat_n else "")
              + f"; numbers moved {num_changed} of {num_n}" + (f" ({num_changed / num_n:.1%}, mean shift {shift / num_n:.4f} z)" if num_n else "")
              + "; " + ", ".join(notes))


def tally(part: str) -> int:
    reads = tuple(read for read in READS if read[1] == part)
    cells = [(group, v, s) for group, _ in reads for v, s in _cells(group, part)]
    missing = [cell for cell in cells if _run(cell[1], cell[2]) is None]
    files = [cell for cell in cells if (_out(cell[0], part) / f"{cell[1]}_s{cell[2]}.csv").exists()]
    if missing or len(files) != len(cells):
        print(f"incomplete: {len(missing)} of {len(cells)} runs missing, {len(files)} of {len(cells)} downstream files")
        return 1
    if gate(reads):
        return 1
    for group, _ in reads:
        use(group, part)
        if u.tally():
            return 1
        if part == "core":
            cause(_own_forest(u.OUT))
        mechanism(group, part)
        print()
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["utility"] and len(argv) == 4 and argv[1] in SEEDS:
        use(argv[1], _part(argv[2]))
        u.run(argv[2], int(argv[3]))
    elif argv == ["--jobs"]:
        jobs()
    elif argv == ["--gate"]:
        raise SystemExit(1 if gate() else 0)
    elif argv[:1] == ["--tally"] and argv[1:] in (["core"], ["rest"]):
        raise SystemExit(tally(argv[1]))
    else:
        sys.exit(__doc__)
