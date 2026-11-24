"""E189 (docs/tickets/imputation-protocol/78): does a forest lose to the gaps left missing on a filled table because
it cannot tell a filled cell from an observed one? Each fill is given again with one 0/1 flag column per feature
("this cell was a gap"), and the forest of E186's downstream read is fitted on each table's own rows.

    uv run --python 3.11 python -B scripts/experiments/indicator_utility.py run <variant> <seed>     # one CSV under OUT
    uv run --python 3.11 python -B scripts/experiments/indicator_utility.py check <variant> <seed>   # inherited fills only, against the stored F1
    uv run --python 3.11 python -B scripts/experiments/indicator_utility.py --jobs                   # "variant seed" lines not yet computed
    uv run --python 3.11 python -B scripts/experiments/indicator_utility.py --tally                  # the one read; refuses a partial set

It trains nothing of TRIDENT's and uses no GPU. K's fill is the fold ledgers of runs already trained: E184's and
E185's arm K (seeds 101/202/303, read by E186), E187's arm K (1717/8484/9595) and E188's core runs' calibrated rows
(2718/6464/7272). The tables, folds and learners are ``sliced_utility.py``'s (E186's); a cell's inherited fills must
reproduce the macro-F1 those studies stored, or the tally refuses. Never import ``sliced_fresh_read.py`` or a
``*_run.py`` here: they repoint shared modules on import. Run from the checkout root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import f1_score

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import c15_report  # noqa: E402
from scripts.experiments import sliced_run as e182  # noqa: E402
from scripts.experiments import sliced_select_run as e183  # noqa: E402
from scripts.experiments import sliced_utility as u  # noqa: E402
from scripts.experiments import uncalibrated_utility_read as e188  # noqa: E402  (its use() is never called here)

STATE = Path("/scratch/diogoneiss/trident-2026-11-24-indicator-utility")
OUT = STATE / "csv"
E187 = Path("/scratch/diogoneiss/trident-2026-11-22-sliced-fresh")
E186_STATES = dict(u.STATES)  # E184's and E185's states, as sliced_utility.py holds them at import
SETS: dict[str, tuple[int, ...]] = {"E186": (101, 202, 303), "E187": (1717, 8484, 9595), "E188": e188.SEEDS["fresh"]}
STORED = {"E186": u.OUT, "E187": E187 / "utility" / "csv", "E188": e188.STATE / "utility" / "fresh-core" / "csv"}
CORE = e188.CORE
WIDE = u.VARIANTS  # the 26 MCAR variants, on E187's seeds only
CORE_SEEDS = tuple(seed for seeds in SETS.values() for seed in seeds)
CELLS = [(v, s) for v in WIDE for s in SETS["E187"]] + [(v, s) for v in CORE for s in (*SETS["E186"], *SETS["E188"])]
assert len(CELLS) == len(set(CELLS)) == 114 and set(CORE) <= set(WIDE) and len(CORE_SEEDS) == 9
INHERITED = ("missing", "naive", "K", "truth")
FLAGGED = ("naive+ind", "K+ind", "truth+ind")
FILLS = (*INHERITED, *FLAGGED)
FLAG = "__was_gap"
PAIRS = (("K+ind", "K"), ("K+ind", "missing"), ("K", "missing"), ("naive+ind", "missing"), ("K+ind", "naive+ind"),
         ("truth+ind", "truth"), ("naive+ind", "naive"), ("naive", "missing"), ("truth", "missing"))
CALLS = (("K+ind", "K"), ("K+ind", "missing"), ("K+ind", "naive+ind"))
KEY = ["variant", "seed", "fold", "learner", "fill"]


def _study(seed: int) -> str:
    return next(name for name, seeds in SETS.items() if seed in seeds)


def _fill(state: Path, arm: str, variant: str, seed: int, fold: int) -> pd.DataFrame:  # noqa: ARG001
    """K's ledger of one fold, from the study that trained that seed."""
    assert arm == "K"
    study = _study(seed)
    if study == "E188":
        run = e188._run(variant, seed)
        assert run is not None, f"no finished E188 run of {variant} seed {seed}"
        return e188._ledgers(run, fold)["K"]
    root = E187 if study == "E187" else next(s for s, variants in E186_STATES.items() if variant in variants)
    run = e183._run(root, "K", variant, seed)
    assert run is not None, f"no finished K run of {variant} seed {seed} under {root}"
    return e182._ledger(run, fold)


u.STATES, u.ARMS, u.FILLS, u._fill = {STATE: WIDE}, ("K",), INHERITED, _fill


def _scores(variant: str, seed: int, fills: tuple[str, ...]) -> pd.DataFrame:
    """Macro-F1 on each test fold of each learner fitted on each table's own rows outside the fold."""
    assert (variant, seed) in CELLS, (variant, seed)
    tables, labels, folds, numerical, categorical = u._tables(variant, seed)
    flags = tables["missing"].isna().astype(float).add_suffix(FLAG)
    assert flags.to_numpy().sum() > 0 and not set(flags.columns) & set(tables["missing"].columns)
    for fill in FLAGGED:
        tables[fill] = pd.concat([tables[fill.removesuffix("+ind")], flags], axis=1)
    out: list[dict[str, object]] = []
    everything = np.arange(len(labels))
    for ordinal, fold in enumerate(folds, start=1):
        test = np.asarray(fold.test_indices)
        train = np.setdiff1d(everything, test)
        for name in u.LEARNERS:
            for fill in fills:
                numbers = [*numerical, *flags.columns] if fill in FLAGGED else numerical  # a flag passes through as a number
                model = u._learner(name, seed, categorical, numbers).fit(tables[fill].iloc[train], labels.iloc[train])
                score = f1_score(labels.iloc[test], model.predict(tables[fill].iloc[test]), average="macro")
                out.append({"variant": variant, "seed": seed, "fold": ordinal, "learner": name, "fill": fill, "macro_f1": float(score)})
    return pd.DataFrame(out)


def _stored(variant: str, seed: int) -> pd.DataFrame:
    """What the study that read this cell stored for the inherited fills (``own``)."""
    frame = pd.read_csv(STORED[_study(seed)] / f"{variant}_s{seed}.csv")
    frame = frame[(frame["protocol"] == "own") & frame["fill"].isin(INHERITED)]
    assert len(frame) == 5 * len(u.LEARNERS) * len(INHERITED), (variant, seed, len(frame))
    return frame[[*KEY, "macro_f1"]]


def _gap(mine: pd.DataFrame) -> float:
    """The largest distance between this code's F1 on the inherited fills and the stored one."""
    stored = pd.concat([_stored(v, s) for v, s in mine[["variant", "seed"]].drop_duplicates().itertuples(index=False)])
    both = mine[mine["fill"].isin(INHERITED)].merge(stored, on=KEY, suffixes=("", "_stored"), validate="one_to_one")
    assert len(both) == len(stored)
    return float((both["macro_f1"] - both["macro_f1_stored"]).abs().max())


def run(variant: str, seed: int) -> None:
    path = OUT / f"{variant}_s{seed}.csv"
    if path.exists():
        print("SKIP", path.name)
        return
    scores = _scores(variant, seed, FILLS)
    OUT.mkdir(parents=True, exist_ok=True)
    scores.to_csv(path, index=False)
    print(f"DONE {path.name}", flush=True)


def check(variant: str, seed: int) -> None:
    """The inherited fills only, against the stored F1: no flagged table is fitted."""
    print(f"{variant} seed {seed}: inherited fills differ from {_study(seed)}'s stored F1 by at most {_gap(_scores(variant, seed, INHERITED)):.2e}")


def _rule(loss_low: float, loss: float, gate_low: float, low: float, high: float, rest_high: float) -> str:
    """ticket 78's rule: the loss F1(missing) − F1(K), the gate F1(naive+ind) − F1(missing), the change
    F1(K+ind) − F1(K) and what is left, F1(K+ind) − F1(missing)."""
    if loss_low <= 0:
        return "no read (the loss F1(missing) − F1(K) is not above 0)"
    if gate_low <= -loss / 2:
        return "no read (flags do not stand in for the gaps: mean/mode with flags is not within half the loss of the gaps left missing)"
    if high < 0:
        return "flags make K's fill worse, not the cause"
    if low > 0:
        if high < loss / 2:
            return "a minor part (under half the loss)"
        return "accounts for the loss" if rest_high >= 0 else "accounts for part of the loss"
    return "not shown, ruled out as the main cause" if high < loss / 2 else "inconclusive"


assert [_rule(*a).split(" (")[0] for a in ((0, .02, 0, .01, .02, 0), (.01, .02, -.01, .01, .02, 0), (.01, .02, 0, -.02, -.01, 0),
                                           (.01, .02, 0, .001, .009, 0), (.01, .02, 0, .005, .02, 0), (.01, .02, 0, .005, .02, -.001),
                                           (.01, .02, 0, -.001, .009, 0), (.01, .02, 0, -.001, .011, 0))] == [
    "no read", "no read", "flags make K's fill worse, not the cause", "a minor part", "accounts for the loss",
    "accounts for part of the loss", "not shown, ruled out as the main cause", "inconclusive"]


def _pooled(frame: pd.DataFrame, first: str, second: str) -> NDArray[np.float64]:
    """F1(first) − F1(second), pooled over the frame's variants per (seed, fold) replicate."""
    d = frame.assign(d=frame[first] - frame[second]).pivot_table(index=["seed", "fold"], columns="variant", values="d")
    assert not d.isna().any().any()
    return np.asarray(d.mean(axis=1), dtype=float)


def _levels(frame: pd.DataFrame) -> str:
    return ", ".join(f"{fill} {frame[fill].mean():.4f}" for fill in FILLS)


def _verdict(frame: pd.DataFrame, corrected: bool) -> str:
    loss = c15_report.interval(_pooled(frame, "missing", "K"), corrected)
    gate = c15_report.interval(_pooled(frame, "naive+ind", "missing"), corrected)
    change = c15_report.interval(_pooled(frame, "K+ind", "K"), corrected)
    rest = c15_report.interval(_pooled(frame, "K+ind", "missing"), corrected)
    return _rule(loss[1], loss[0], gate[1], change[1], change[2], rest[2])


def _ruled(name: str, frame: pd.DataFrame) -> str:
    loss, lo, _ = c15_report.interval(_pooled(frame, "missing", "K"), False)
    _, gate, _ = c15_report.interval(_pooled(frame, "naive+ind", "missing"), False)
    m, low, high = c15_report.interval(_pooled(frame, "K+ind", "K"), False)
    _, _, rest = c15_report.interval(_pooled(frame, "K+ind", "missing"), False)
    plain, corrected = _verdict(frame, False), _verdict(frame, True)
    share = f", as a share of the loss {m / loss:+.2f} [{low / loss:+.2f}, {high / loss:+.2f}]" if loss > 0 else ""
    return (f"- {name}: loss {loss:+.4f} (lower end {lo:+.4f}); gate F1(naive+ind) − F1(missing) lower end {gate:+.4f} against {-loss / 2:+.4f}; "
            f"F1(K+ind) − F1(K) {m:+.4f} [{low:+.4f}, {high:+.4f}]{share}; F1(K+ind) − F1(missing) upper end {rest:+.4f} -> {plain}"
            + ("" if plain == corrected else f" (fragile; corrected: {corrected})"))


def _pairs(frame: pd.DataFrame, pairs: tuple[tuple[str, str], ...], bullet: str, seed_means: bool = False) -> None:
    for first, second in pairs:
        d = _pooled(frame, first, second)
        line = u._line(f"{bullet}F1({first}) − F1({second})", d, first, second)
        if seed_means:  # beside: a t interval over the nine seeds' means, which no fold dependence touches
            by_seed = frame.assign(d=frame[first] - frame[second]).groupby("seed")["d"].mean().to_numpy(dtype=float)
            _, lo, hi = c15_report.interval(by_seed, False)
            line += f"; over {len(by_seed)} seed means [{lo:+.4f}, {hi:+.4f}]"
        print(line)


def tally() -> int:
    missing = [f"{v}_s{s}" for v, s in CELLS if not (OUT / f"{v}_s{s}.csv").exists()]
    if missing:
        print(f"incomplete: {len(missing)} of {len(CELLS)} cells are not computed")
        return 1
    data = pd.concat([pd.read_csv(OUT / f"{v}_s{s}.csv") for v, s in CELLS], ignore_index=True)
    assert len(data) == len(CELLS) * 5 * len(u.LEARNERS) * len(FILLS) and not data.duplicated(KEY).any()
    gap = _gap(data)
    print(f"gate: {len(CELLS)} cells, the inherited fills (missing, naive, K, truth; forest and logistic regression) differ from the "
          f"F1 stored by E186, E187 and E188 by at most {gap:.2e}")
    if gap > 1e-12:
        print("refused: this code does not reproduce the stored F1 of the inherited fills")
        return 1

    def wide(learner: str, variants: tuple[str, ...], seeds: tuple[int, ...]) -> pd.DataFrame:
        part = data[(data["learner"] == learner) & data["variant"].isin(variants) & data["seed"].isin(seeds)]
        frame = part.pivot_table(index=["variant", "seed", "fold"], columns="fill", values="macro_f1").reset_index()
        assert len(frame) == len(variants) * len(seeds) * 5
        return frame

    core = wide("forest", CORE, CORE_SEEDS)
    print(f"\n## E189, the core: macro-F1 of a forest fitted on each table's own rows, {len(CORE)} variants (credit-g and kr-vs-kp at 40/60/80nan), "
          f"seeds {'/'.join(map(str, CORE_SEEDS))}; a difference is pooled over the variants per (seed, fold) replicate, "
          f"{5 * len(CORE_SEEDS)} replicates, plain 95% t interval (NB: corrected)")
    _pairs(core, PAIRS, "", seed_means=True)
    print(f"Levels: {_levels(core)}")

    print("\n### The rule (forest, own): does the forest lose on K's fill because it cannot tell a filled cell from an observed one?")
    print(_ruled("core, 6 variants pooled per (seed, fold)", core))
    for table in ("kr-vs-kp", "credit-g"):
        print(_ruled(f"{table}, its three variants pooled", core[core["variant"].str.startswith(table)]))
    for variant in CORE:
        print(_ruled(variant, core[core["variant"] == variant]))
    print("Beside, each seed set alone (15 replicates):")
    for study, seeds in SETS.items():
        print(_ruled(f"{study}'s seeds {'/'.join(map(str, seeds))}", core[core["seed"].isin(seeds)]))

    print("\n### Per table and per variant (forest, own)")
    for name, part in [(t, core[core["variant"].str.startswith(t)]) for t in ("kr-vs-kp", "credit-g")] + [(v, core[core["variant"] == v]) for v in CORE]:
        print(f"- {name}: {_levels(part)}")
        _pairs(part, PAIRS[:6], "  - ")

    print("\n### Beside: logistic regression fitted on each table's own rows (a gap is the mean with an indicator, or its own category), the core")
    logistic = wide("logistic", CORE, CORE_SEEDS)
    _pairs(logistic, PAIRS, "- ")
    print(f"- Levels: {_levels(logistic)}")

    seeds = SETS["E187"]
    forest = wide("forest", WIDE, seeds)
    print(f"\n## E189, the 26 MCAR variants on E187's seeds {'/'.join(map(str, seeds))} (forest, own; {5 * len(seeds)} replicates): "
          "is the fill with flags ever worse than the fill alone, or than the gaps left missing?")
    _pairs(forest, PAIRS, "")
    print(f"Levels: {_levels(forest)}")
    print("\n### Per variant, 15 fold values a variant")
    counts = {pair: {pair[0]: 0, "no difference": 0, pair[1]: 0} for pair in CALLS}
    for variant in WIDE:
        part = forest[forest["variant"] == variant]
        print(f"- {variant}: {_levels(part)}")
        for first, second in CALLS:
            line = u._line(f"  - F1({first}) − F1({second})", (part[first] - part[second]).to_numpy(dtype=float), first, second)
            counts[(first, second)][line.rsplit("-> ", 1)[1]] += 1
            print(line)
    for (first, second), count in counts.items():
        print(f"Variants, F1({first}) − F1({second}): " + ", ".join(f"{name} {n}" for name, n in count.items()))
    print("\n### Beside: logistic regression, the 26 variants")
    _pairs(wide("logistic", WIDE, seeds), PAIRS, "- ")
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] in (["run"], ["check"]) and len(argv) == 3:
        (run if argv[0] == "run" else check)(argv[1], int(argv[2]))
    elif argv == ["--jobs"]:
        order = {table: k for k, table in enumerate(e182.TABLES)}
        for variant, seed in sorted(CELLS, key=lambda c: (order[c[0].rsplit("_", 1)[0]], c)):
            if not (OUT / f"{variant}_s{seed}.csv").exists():
                print(variant, seed)
    elif argv == ["--tally"]:
        raise SystemExit(tally())
    else:
        sys.exit(__doc__)
