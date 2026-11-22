"""E187's reads (docs/tickets/imputation-protocol/76) on the fresh-seed K and S runs trained by ``sliced_fresh_run.py``:
E186's three-way downstream classification, and E183's ``sw_score`` read as the manipulation check.

    uv run --python 3.11 python -B scripts/experiments/sliced_fresh_read.py utility <variant> <seed>   # one CSV under OUT
    uv run --python 3.11 python -B scripts/experiments/sliced_fresh_read.py score <K|S> <variant> <seed>  # E182's scorer on the run's ledgers
    uv run --python 3.11 python -B scripts/experiments/sliced_fresh_read.py --jobs     # the cells above not yet computed, one a line
    uv run --python 3.11 python -B scripts/experiments/sliced_fresh_read.py --tally    # both reads; refuses a partial set

The downstream cells and their tally are ``sliced_utility.py``'s functions (E186's) run with this study's state, seeds
and the two arms; the ledger scores are ``sliced_run.py``'s (E182's ``reference`` group). The queues never import this
file and it never imports ``sliced_fresh_run.py`` (which repoints ``sliced_loss_run``). ``FRESH_SELFCHECK=<dir>`` runs
the same code on E186's seeds and runs (already read), writing the downstream CSVs under ``<dir>``: the primary line
must then reproduce E186's. Run from the checkout root.
"""

from __future__ import annotations

import os
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
from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments import sliced_run as e182  # noqa: E402
from scripts.experiments import sliced_select_run as e183  # noqa: E402
from scripts.experiments import sliced_utility as u  # noqa: E402

E186_OUT = u.OUT  # E186's downstream CSVs, for the pooled line
SELFCHECK = os.environ.get("FRESH_SELFCHECK")
STATE = Path("/scratch/diogoneiss/trident-2026-11-22-sliced-fresh")
VARIANTS = u.VARIANTS
ARMS: tuple[str, ...] = ("K", "S")
if SELFCHECK:
    STATES, SEEDS, OUT, STUDY = dict(u.STATES), u.SEEDS, Path(SELFCHECK) / "csv", "self-check of E187's code on E186's seeds"
else:
    STATES, SEEDS, OUT, STUDY = {STATE: VARIANTS}, (1717, 8484, 9595), STATE / "utility" / "csv", "E187"
assert bool(SELFCHECK) == bool(set(SEEDS) & {101, 202, 303})
u.STATES, u.SEEDS, u.OUT, u.STUDY = STATES, SEEDS, OUT, STUDY
u.ARMS, u.FILLS, u.TERM, u.BESIDE = ARMS, ("missing", "naive", *ARMS, "truth"), (), (("naive", "missing"), ("truth", "missing"))

# E186's per-variant calls on F1(K) − F1(missing), forest, ``own`` (ticket 75's printout), written before this study's first cell
PREDICTED: dict[str, tuple[str, ...]] = {
    "K": ("pendigits_20nan", "pendigits_40nan", "pendigits_60nan", "letter_20nan", "vehicle_20nan", "vehicle_40nan",
          "electricity_20nan", "electricity_40nan", "biodeg_40nan"),
    "missing": ("kr-vs-kp_40nan", "kr-vs-kp_60nan", "kr-vs-kp_80nan", "credit-g_60nan", "credit-g_80nan"),
}
CARRIERS = ("credit-g_60nan", "pendigits_80nan")  # the two variants on which E186 called S over K
assert sum(len(v) for v in PREDICTED.values()) == 14 and set(sum(PREDICTED.values(), ())) | set(CARRIERS) <= set(VARIANTS)


def _state(variant: str) -> Path:
    return next(state for state, variants in STATES.items() if variant in variants)


def _run(arm: str, variant: str, seed: int) -> Path | None:
    return e183._run(_state(variant), arm, variant, seed)


def _scored(arm: str, variant: str, seed: int) -> Path:
    """Where the arm's ledger scores live: E184's layout, so the self-check finds E184's and E185's files."""
    return _state(variant) / "csv" / arm / "csv" / f"{variant}_s{seed}__reference.csv"


def score(arm: str, variant: str, seed: int) -> None:
    assert arm in ARMS and variant in VARIANTS and seed in SEEDS, (arm, variant, seed)
    run = _run(arm, variant, seed)
    assert run is not None, f"no finished {arm} run of {variant} seed {seed}"
    e182.STATE, e182._run_dir = _state(variant) / "csv" / arm, lambda variant, seed: run  # noqa: ARG005
    e182.run(variant, seed, "reference")


def jobs() -> None:
    for variant in VARIANTS:
        for seed in SEEDS:
            ready = all(_run(arm, variant, seed) is not None for arm in ARMS)
            if ready and not (OUT / f"{variant}_s{seed}.csv").exists():
                print("utility", variant, seed)
            for arm in ARMS:
                if _run(arm, variant, seed) is not None and not _scored(arm, variant, seed).exists():
                    print("score", arm, variant, seed)


def _rule(repeated: int, opposite: int) -> str:
    return "replicated" if repeated >= 10 and opposite == 0 else "not replicated" if repeated < 7 or opposite >= 2 else "partly"


assert (_rule(14, 0), _rule(10, 0), _rule(10, 1), _rule(9, 0), _rule(7, 1), _rule(6, 0), _rule(12, 2)) == (
    "replicated", "replicated", "partly", "partly", "partly", "not replicated", "not replicated")


def _own_forest(folder: Path) -> pd.DataFrame:
    data = pd.concat([pd.read_csv(f) for f in sorted(folder.glob("*.csv"))], ignore_index=True)
    part = data[(data["protocol"] == "own") & (data["learner"] == "forest")]
    return part.pivot_table(index=u.KEY, columns="fill", values="macro_f1").reset_index()


def _call(values: NDArray[np.float64], first: str, second: str) -> str:
    _, low, high = c15_report.interval(values, False)
    return first if low > 0 else second if high < 0 else "no difference"


def predictions() -> None:
    forest = _own_forest(OUT)
    print(f"\n### E186's per-variant calls on these seeds ({STUDY}; forest, own, 15 fold values a variant)")
    calls = {v: _call((part["K"] - part["missing"]).to_numpy(dtype=float), "K", "missing")
             for v in VARIANTS for part in [forest[forest["variant"] == v]]}
    repeated = opposite = 0
    for was, variants in PREDICTED.items():
        other = "missing" if was == "K" else "K"
        same = [v for v in variants if calls[v] == was]
        flipped = [v for v in variants if calls[v] == other]
        repeated, opposite = repeated + len(same), opposite + len(flipped)
        print(f"- F1(K) − F1(missing), E186 called {was} on {len(variants)}: {len(same)} repeat, {len(flipped)} take the opposite call {flipped}, "
              f"{len(variants) - len(same) - len(flipped)} no difference {[v for v in variants if calls[v] == 'no difference']}")
    new = [f"{v} ({calls[v]})" for v in VARIANTS if v not in sum(PREDICTED.values(), ()) and calls[v] != "no difference"]
    print(f"By-table split: {repeated} of 14 repeat, {opposite} opposite -> {_rule(repeated, opposite)}; calls on variants E186 left at no difference: {new}")
    for variant in CARRIERS:
        part = forest[forest["variant"] == variant]
        print(u._line(f"- S − K carrier {variant}, F1(S) − F1(K)", (part["S"] - part["K"]).to_numpy(dtype=float), "S", "K"))
    if not SELFCHECK:
        both = pd.concat([_own_forest(E186_OUT), forest], ignore_index=True)
        assert both["seed"].nunique() == 6 and len(both) == 6 * 5 * len(VARIANTS)
        d = both.assign(d=both["S"] - both["K"]).pivot_table(index=["seed", "fold"], columns="variant", values="d").mean(axis=1)
        print(u._line("Not fresh, E186's three seeds and these pooled (30 replicates), F1(S) − F1(K)", np.asarray(d, dtype=float), "S", "K"))


def manipulation() -> None:
    """S − K on the output's scores, per variant by the fold-pair rule with seed agreement, and E183's rule over them."""
    frames = {arm: pd.concat([pd.read_csv(_scored(arm, v, s)) for v in VARIANTS for s in SEEDS], ignore_index=True) for arm in ARMS}
    for arm, frame in frames.items():
        assert float(frame.loc[frame["method"] == "trident", "gate"].max()) < 1e-9, f"{arm}: a ledger does not reproduce its run's impute_score"
    print(f"\n## Manipulation check ({STUDY}): S − K on the output's scores (E182's scorer on the test-fold ledgers), seeds {'/'.join(map(str, SEEDS))}")
    calls: dict[str, dict[str, str]] = {}
    for variant in VARIANTS:
        ours = {arm: frame[frame["variant"] == variant] for arm, frame in frames.items()}

        def folds_of(arm: str, method: str, name: str) -> list[NDArray[np.float64]]:
            rows = ours[arm][ours[arm]["method"] == method].sort_values(["seed", "fold"])
            return [rows.loc[rows["seed"] == s, name].to_numpy(dtype=float) for s in SEEDS]

        def level(arm: str, method: str, name: str) -> float:
            return float(np.concatenate(folds_of(arm, method, name)).mean())

        epochs = {arm: float(np.mean([pd.read_csv(run / "metrics" / "raw_fold_metrics.csv")[e183.EPOCH].mean()
                                      for s in SEEDS if (run := _run(arm, variant, s)) is not None])) for arm in ARMS}
        seen = level("K", "shuffled", "sw_score") >= e182.CONTROL
        call = calls.setdefault(variant, {"control": "seen" if seen else "blind"})
        print(f"- `{variant}` ({call['control']}): mean kept epoch K {epochs['K']:.0f} → S {epochs['S']:.0f}")
        for name in e182.SCORES:
            v = common.verdict([s - k for s, k in zip(folds_of("S", "trident", name), folds_of("K", "trident", name))],
                               sesoi=e182.SESOI * level("K", "trident", name))
            call[name] = v.call
            print(f"  - S − K on {name} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call} (K {level('K', 'trident', name):.4f}, S {level('S', 'trident', name):.4f})")
    counted = [v for v in VARIANTS if calls[v]["control"] == "seen"]
    better = [v for v in counted if calls[v]["sw_score"] == "better"]
    worse = [v for v in counted if calls[v]["sw_score"] == "worse"]
    reading = "supported" if 2 * len(better) >= len(counted) and len(worse) <= 2 else "refuted" if 6 * len(better) <= len(counted) else "mixed"
    print(f"\nE183's rule, S − K on sw_score over the {len(counted)} variants the shuffled control passes: better on {len(better)}, "
          f"worse on {len(worse)} {worse} -> {reading}")
    for name in e182.SCORES:
        print(f"S − K on {name}, all variants: {common.tally(calls[v][name] for v in VARIANTS)}")


def tally() -> int:
    cells = [(v, s) for v in VARIANTS for s in SEEDS]
    missing = [(a, v, s) for v, s in cells for a in ARMS if _run(a, v, s) is None]
    unscored = [(a, v, s) for v, s in cells for a in ARMS if not _scored(a, v, s).exists()]
    files = [f.name for f in OUT.glob("*.csv")]
    if missing or unscored or sorted(files) != sorted(f"{v}_s{s}.csv" for v, s in cells):
        print(f"incomplete: {len(missing)} of {len(cells) * len(ARMS)} runs missing, {len(unscored)} ledger scores missing, "
              f"{len(files)} of {len(cells)} downstream files")
        return 1
    if u.tally():
        return 1
    predictions()
    manipulation()
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["utility"] and len(argv) == 3:
        u.run(argv[1], int(argv[2]))
    elif argv[:1] == ["score"] and len(argv) == 4:
        score(argv[1], argv[2], int(argv[3]))
    elif argv == ["--jobs"]:
        jobs()
    elif argv == ["--tally"]:
        raise SystemExit(tally())
    else:
        sys.exit(__doc__)
