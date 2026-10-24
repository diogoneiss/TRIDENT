"""E69: per-column weights for the K + TabPFN blend, cross-fitted, offline
(imputation-protocol/24, T37). Nothing is written to the store.

    uv run --python 3.11 --extra tabpfn python -B scripts/experiments/blend_weights.py <set> <variant> <seed> <out_dir>
    uv run --python 3.11 python -B scripts/experiments/blend_weights.py --tally <out_dir>
    uv run --python 3.11 python -B scripts/experiments/blend_weights.py --weights <out_dir>

``<set>`` is ``confirmation`` (E60's 15 variants at 707/808/909) or ``e40`` (E66's kr-vs-kp and
credit-g at 40/60/80nan, 404/505/606). Per pair, the fills are E60's: the model's induced fills from
the fold ledger (K, asserted fold by fold) and TabPFN v2 refitted on the same training fold. All
five folds' fills are collected before anything is scored, then for fold k and each column:

- **numerical:** the model's weight w from {0, 0.25, 0.5, 0.75, 1} (blend = w·model + (1−w)·TabPFN)
  that minimises the summed squared error of that column's induced cells on the other four folds;
- **categorical:** the model's or TabPFN's category, whichever is right on more of those cells;
- **ties and empty columns:** 0.5 for a numerical column, the model for a categorical one, which is
  E60's equal blend.

Squared error and correct counts add across columns into ``score_cells``'s pooled ``rmse_num_z`` and
``acc_cat``, so a per-column choice is the grid's best on the other folds. Fold k's truth never
enters its own weights. Per fold, ``impute_score`` is written for the model, TabPFN, the equal blend
and the weighted one; the weights go to ``weights/`` and TabPFN's cell fills to ``cells/``.
Threads are capped at 4.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments import tabpfn_blend as blend  # noqa: E402
from src.training.data import build_folds, prepare_dataset  # noqa: E402
from src.training.imputation_metrics import NUMERICAL, mean_mode_baselines  # noqa: E402
from src.training.types import DatasetSpec  # noqa: E402

GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
EQUAL = 0.5
SESOI = 0.01
# (set, variant) pairs; the e40 set is E61's: the two tables where TabPFN leads from 40nan.
SETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "confirmation": ("confirmation", blend.SETS["confirmation"][3]),
    "e40": ("e40", tuple(f"{t}_{level}nan" for t in ("kr-vs-kp", "credit-g") for level in (40, 60, 80))),
}
# Where E60 or E66's equal blend was not better than TabPFN (E60: none or equivalent; E66: worse).
HARD = ("electricity_40nan", "electricity_80nan", "vehicle_80nan", "biodeg_80nan",
        *(f"{t}_{level}nan" for t in ("kr-vs-kp", "credit-g") for level in (40, 60, 80)))
PRIOR = {"confirmation": Path("/scratch2/diogoneiss/trident-2026-10-21-tabpfn-blend/csv"),
         "e40": Path("/scratch2/diogoneiss/trident-2026-10-23-tabpfn-blend-e40/csv")}


def choose(train: pd.DataFrame) -> dict[str, float]:
    """The model's weight per column, from the other folds' cells (columns model, tabpfn, actual)."""
    chosen: dict[str, float] = {}
    for column, group in train.groupby("column", sort=True):
        if group["kind"].iloc[0] == NUMERICAL:
            m, t, a = (group[c].to_numpy(dtype=float) for c in ("model", "tabpfn", "actual"))
            sse = {w: float(np.sum((w * m + (1 - w) * t - a) ** 2)) for w in GRID}
            best = min(sse.values())
            chosen[str(column)] = EQUAL if sse[EQUAL] == best else min(w for w in GRID if sse[w] == best)
        else:
            right_model = int((group["model"] == group["actual"]).sum())
            right_tabpfn = int((group["tabpfn"] == group["actual"]).sum())
            chosen[str(column)] = 0.0 if right_tabpfn > right_model else 1.0
    return chosen


def apply(cells: pd.DataFrame, weights: dict[str, float], default_numerical: float) -> list[object]:
    out: list[object] = []
    for column, kind, m, t in zip(cells["column"], cells["kind"], cells["model"], cells["tabpfn"]):
        if kind == NUMERICAL:
            w = weights.get(column, default_numerical)
            out.append(w * float(m) + (1 - w) * float(t))
        else:
            out.append(m if weights.get(column, 1.0) == 1.0 else t)
    return out


def run_pair(which: str, variant: str, seed: int, out: Path) -> None:
    source = SETS[which][0]
    started = time.time()
    table = score_pair(collect(blend.SETS[source][0], blend.SETS[source][1], "K", variant, seed), variant, seed, out)
    note = ""
    prior = PRIOR[source] / f"{variant}_s{seed}.csv"
    if prior.exists():
        before = pd.read_csv(prior)
        drift = []
        for ours, theirs in (("tabpfn", "tabpfn"), ("equal", "blend")):
            a = table[table.method == ours].sort_values("fold")["impute_score"].to_numpy(dtype=float)
            b = before[before.method == theirs].sort_values("fold")["impute_score"].to_numpy(dtype=float)
            drift.append(f"{ours} vs the earlier run {np.abs(a - b).max():.1e}")
        note = "; " + ", ".join(drift)
    print(f"DONE {variant} s{seed}: model equals the run's K on 5 folds{note}, {time.time() - started:.0f} s", flush=True)


def collect(state: Path, experiment: str, readout: str, variant: str, seed: int) -> list[tuple[pd.DataFrame, dict[str, float | str]]]:
    """Every fold's induced cells with the model's fill (the ledger's, asserted equal to the run's
    ``readout``) and TabPFN v2's, refitted on the same training fold."""
    from src.training.tabpfn_baseline import fit_tabpfn_imputers

    warnings.filterwarnings("ignore")
    found = common.parents(experiment)[(variant, seed)]
    if len(found) != 1:
        raise RuntimeError(f"{variant} s{seed}: {len(found)} runs")
    run = found[0]
    logged = common.model(common.fold_table(run), run, readout)
    dataset = prepare_dataset(DatasetSpec.from_name(variant, None))
    features = dataset.frame.drop(columns=[dataset.label_column])
    numerical, categorical = list(dataset.numerical_columns), list(dataset.categorical_columns)
    folds: list[tuple[pd.DataFrame, dict[str, float | str]]] = []
    for ordinal, fold in enumerate(build_folds(dataset.frame, dataset.label_column, blend.FOLDS, seed), start=1):
        train = features.iloc[np.asarray(fold.train_indices)].reset_index(drop=True)
        test = features.iloc[np.asarray(fold.test_indices)].reset_index(drop=True)
        naive = dict(mean_mode_baselines(train, numerical, categorical))
        cells = blend._ledger(variant, seed, ordinal, state)
        numeric = (cells["kind"] == NUMERICAL).to_numpy()
        for c in ("actual", "imputed"):
            cells[c] = [float(v) if n else v for v, n in zip(cells[c], numeric)]
        positions = cells["row"].to_numpy(dtype=int)
        columns = test.columns.get_indexer(pd.Index(cells["column"]))
        hidden = np.zeros(test.shape, dtype=bool)
        hidden[positions, columns] = True
        imputer = fit_tabpfn_imputers(train, numerical, categorical)[0]
        filled = imputer.impute(test, hidden).to_numpy(dtype=object)[positions, columns]
        model_score = blend._score(cells, naive)
        assert abs(model_score - float(logged[ordinal - 1])) < 1e-9, f"{variant} s{seed} fold {ordinal}: {model_score} against {logged[ordinal - 1]}"
        frame = cells[["row", "column", "kind", "actual"]].assign(
            model=cells["imputed"], tabpfn=[float(g) if k == NUMERICAL else str(g) for g, k in zip(filled, cells["kind"])], fold=ordinal)
        folds.append((frame, naive))
    return folds


def score_pair(folds: list[tuple[pd.DataFrame, dict[str, float | str]]], variant: str, seed: int, out: Path) -> pd.DataFrame:
    """Cross-fit the weights fold by fold and write the scores, the weights and the cells."""
    pooled = pd.concat([f for f, _ in folds], ignore_index=True)
    rows: list[dict[str, object]] = []
    weights: list[dict[str, object]] = []
    for frame, naive in folds:
        k = int(frame["fold"].iloc[0])
        chosen = choose(pooled[pooled["fold"] != k])
        equal = apply(frame, {}, EQUAL)
        weighted = apply(frame, chosen, EQUAL)
        scored = frame.assign(imputed=frame["model"])
        share = float((frame["kind"] == NUMERICAL).mean())
        fills: tuple[tuple[str, list[object]], ...] = (("model", list(frame["model"])), ("tabpfn", list(frame["tabpfn"])), ("equal", equal), ("weighted", weighted))
        for method, values in fills:
            rows.append({"variant": variant, "seed": seed, "fold": k, "method": method,
                         "impute_score": blend._score(scored.assign(imputed=np.array(values, dtype=object)), naive), "numeric_share": share})
        counts = frame.groupby("column").size()
        kinds = frame.groupby("column")["kind"].first()
        for column in sorted(set(frame["column"])):
            weights.append({"variant": variant, "seed": seed, "fold": k, "column": column, "kind": kinds[column],
                            "model_weight": chosen.get(column, EQUAL if kinds[column] == NUMERICAL else 1.0),
                            "from_other_folds": column in chosen, "cells": int(counts[column])})
    for sub in ("csv", "weights", "cells"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "csv" / f"{variant}_s{seed}.csv", index=False)
    pd.DataFrame(weights).to_csv(out / "weights" / f"{variant}_s{seed}.csv", index=False)
    pooled.to_csv(out / "cells" / f"{variant}_s{seed}.csv.gz", index=False)
    return table


def _values(table: pd.DataFrame, variant: str, seeds: list[int], method: str) -> list[np.ndarray]:
    rows = table[(table.variant == variant) & (table.method == method)]
    return [rows[rows.seed == s].sort_values("fold")["impute_score"].to_numpy(dtype=float) for s in seeds]


def tally(out: Path) -> None:
    table = pd.concat([pd.read_csv(p) for p in sorted((out / "csv").glob("*.csv"))], ignore_index=True)
    readings: dict[tuple[str, str], str] = {}
    counts: dict[str, list[str]] = {}
    both: dict[str, list[str]] = {"weighted": [], "equal": []}
    expected = 0
    for which, (source, variants) in SETS.items():
        seeds = list(blend.SETS[source][2])
        print(f"# {which} (seeds {'/'.join(map(str, seeds))})\n")
        for variant in variants:
            expected += 1
            rows = table[table.variant == variant]
            if sorted(rows["seed"].unique()) != sorted(seeds):
                print(f"- `{variant}`: incomplete")
                continue
            values = {m: _values(table, variant, seeds, m) for m in ("model", "tabpfn", "equal", "weighted")}
            parts = []
            calls = {}
            for ours, other in (("weighted", "equal"), ("weighted", "tabpfn"), ("weighted", "model"), ("equal", "tabpfn"), ("equal", "model")):
                diffs = [a - b for a, b in zip(values[ours], values[other])]
                v = common.verdict(diffs, sesoi=SESOI * float(np.mean([o.mean() for o in values[other]])))
                calls[ours, other] = v.call
                counts.setdefault(f"{ours} − {other}", []).append(v.call)
                parts.append(f"{ours} − {other} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
            readings[variant, "weighted − tabpfn"] = calls["weighted", "tabpfn"]
            for arm in both:
                if calls[arm, "tabpfn"] == "better" and calls[arm, "model"] == "better":
                    both[arm].append(variant)
            flag = " (hard)" if variant in HARD else ""
            print(f"- `{variant}`{flag} (numeric share {rows['numeric_share'].mean():.2f}): " + "; ".join(parts))
        print()
    for key, tallied in counts.items():
        print(f"- {key}: {common.tally(tallied)}")
    for arm, names in both.items():
        print(f"- {arm} beats both the model and TabPFN on {len(names)}: {names or 'none'}")
    if len(readings) < expected:
        print(f"\nPrimary: incomplete ({len(readings)} of {expected})")
        return
    primary = {v: readings[v, "weighted − tabpfn"] for v in HARD}
    better = [v for v, c in primary.items() if c == "better"]
    worse = [v for v, c in primary.items() if c == "worse"]
    reading = "supported" if len(better) >= 4 else "refuted" if len(better) <= 1 else "mixed"
    print(f"\nPrimary, weighted − TabPFN on the {len(HARD)} hard variants: better on {better or 'none'}, worse on {worse or 'none'} "
          f"-> {reading} ({common.RULE})")


def weights_table(out: Path) -> None:
    """Descriptive: the share of TabPFN's weight per variant and kind, and how often a column's
    choice agrees across its 15 folds."""
    table = pd.concat([pd.read_csv(p) for p in sorted((out / "weights").glob("*.csv"))], ignore_index=True)
    for (variant, kind), group in table.groupby(["variant", "kind"], sort=False):
        tabpfn_share = float((1 - group["model_weight"]).mean())
        modal = group.groupby("column")["model_weight"].agg(lambda s: s.value_counts().iloc[0] / len(s))
        spread = ", ".join(f"{w:g}: {n}" for w, n in group["model_weight"].value_counts().sort_index().items())
        print(f"- `{variant}` {kind} ({group['column'].nunique()} columns): TabPFN's mean weight {tabpfn_share:.2f}; "
              f"model weights {{{spread}}}; a column's modal choice holds on {modal.mean():.0%} of its folds")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--tally"] and len(argv) == 2:
        tally(Path(argv[1]))
    elif argv[:1] == ["--weights"] and len(argv) == 2:
        weights_table(Path(argv[1]))
    elif len(argv) == 4 and argv[0] in SETS:
        run_pair(argv[0], argv[1], int(argv[2]), Path(argv[3]))
    else:
        sys.exit(__doc__)
