"""E89: can fewer or better-fed choices close the deployable hybrid's gap to TabPFN? (imputation-protocol/33, T39).

    uv run --python 3.11 --extra tabpfn python -B scripts/experiments/hybrid_selection_rules.py cells <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/hybrid_selection_rules.py --tally

``cells`` repeats E82's ``blend`` on E82's own H runs (no training): TabPFN v2 refit on each training
fold fills the test split's induced cells and, group by group, the hidden observed validation cells.
It writes both cell tables, with H's and TabPFN's fills, under ``STATE/cells/``, so every rule below is
scored offline. ``--tally`` scores four ways of choosing the hybrid's weights, none of which reads the
complete table:

- ``column`` (E82's deployable): E69's grid per column, on fold k's validation cells;
- ``table``: one numerical weight and one categorical pick per table, on fold k's validation cells,
  each numerical column's residuals scaled by its training split's sd (no column outweighs the rest);
- ``shrunk`` (the primary): the per-column numerical weight pulled toward the equal blend,
  ``(n w + N0 0.5) / (n + N0)`` with ``n`` the column's validation cells; a categorical column keeps
  its own pick with at least ``N0`` cells and takes the table's pick below;
- ``pooled``: the per-column rule on the validation cells of all five folds.
"""

from __future__ import annotations

import json
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

from scripts.experiments import blend_weights as weights  # noqa: E402
from scripts.experiments import deployable_hybrid as e82  # noqa: E402
from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments import tabpfn_blend as tb  # noqa: E402

STATE = Path("/scratch2/diogoneiss/trident-2026-10-28-hybrid-selection-rules")
N0 = 50
RULES = ("column", "table", "shrunk", "pooled")
SESOI = 0.01


def cells(variant: str, seed: int) -> None:
    """E82's blend, keeping the cells: ``test_fold_<k>.csv`` and ``selection_fold_<k>.csv``."""
    from src.training.data import build_folds, prepare_dataset
    from src.training.imputation_metrics import NUMERICAL, mean_mode_baselines
    from src.training.tabpfn_baseline import fit_tabpfn_imputers
    from src.training.types import DatasetSpec

    warnings.filterwarnings("ignore")
    started = time.time()
    out = STATE / "cells" / f"{variant}_s{seed}"
    found = common.parents(e82.EXPERIMENT)[(variant, seed)]
    if len(found) != 1:
        raise RuntimeError(f"{variant} s{seed}: {len(found)} runs")
    logged = common.model(common.fold_table(found[0]), found[0], "H")
    dataset = prepare_dataset(DatasetSpec.from_name(variant, None))
    features = dataset.frame.drop(columns=[dataset.label_column])
    numerical, categorical = list(dataset.numerical_columns), list(dataset.categorical_columns)
    out.mkdir(parents=True, exist_ok=True)
    for ordinal, fold in enumerate(build_folds(dataset.frame, dataset.label_column, tb.FOLDS, seed), start=1):
        train = features.iloc[np.asarray(fold.train_indices)].reset_index(drop=True)
        validation = features.iloc[np.asarray(fold.validation_indices)].reset_index(drop=True)
        test = features.iloc[np.asarray(fold.test_indices)].reset_index(drop=True)
        naive = dict(mean_mode_baselines(train, numerical, categorical))
        ledger = e82._typed(tb._ledger(variant, seed, ordinal, e82.STATE), ("actual", "imputed"))
        assert abs(tb._score(ledger, naive) - float(logged[ordinal - 1])) < 1e-9, f"{variant} s{seed} fold {ordinal}: ledger is not the run's H"
        imputer = fit_tabpfn_imputers(train, numerical, categorical)[0]
        positions, columns = ledger["row"].to_numpy(dtype=int), test.columns.get_indexer(pd.Index(ledger["column"]))
        hidden = np.zeros(test.shape, dtype=bool)
        hidden[positions, columns] = True
        filled = imputer.impute(test, hidden).to_numpy(dtype=object)[positions, columns]
        ledger[["row", "column", "kind", "actual"]].assign(
            model=ledger["imputed"], tabpfn=[float(g) if k == NUMERICAL else str(g) for g, k in zip(filled, ledger["kind"])]
        ).to_csv(out / f"test_fold_{ordinal}.csv", index=False)
        selection = e82._typed(pd.read_csv(e82.STATE / "selection" / f"{variant}_s{seed}" / f"selection_fold_{ordinal}.csv",
                                           dtype={"actual": object, "model": object}), ("actual", "model"))
        guesses: list[object] = [None] * len(selection)
        for g in range(e82.GROUPS):
            part = np.flatnonzero(selection["group"].to_numpy() == g)
            r = selection["row"].to_numpy(dtype=int)[part]
            c = validation.columns.get_indexer(pd.Index(selection["column"].iloc[part]))
            hide = np.zeros(validation.shape, dtype=bool)
            hide[r, c] = True
            values = imputer.impute(validation, hide).to_numpy(dtype=object)[r, c]
            for i, guess, kind in zip(part, values, selection["kind"].iloc[part]):
                guesses[i] = float(guess) if kind == NUMERICAL else str(guess)
        selection.assign(tabpfn=pd.Series(guesses, index=selection.index, dtype=object)).to_csv(out / f"selection_fold_{ordinal}.csv", index=False)
        (out / f"naive_fold_{ordinal}.json").write_text(json.dumps(naive))
        (out / f"scale_fold_{ordinal}.json").write_text(json.dumps({c: float(train[c].std()) or 1.0 for c in numerical}))
    print(f"DONE {variant} s{seed}: ledger equals the run's H on 5 folds, {time.time() - started:.0f} s", flush=True)


def _load(variant: str, seed: int, kind: str, fold: int) -> pd.DataFrame:
    path = STATE / "cells" / f"{variant}_s{seed}" / f"{kind}_fold_{fold}.csv"
    return e82._typed(pd.read_csv(path, dtype={"actual": object, "model": object, "tabpfn": object}), ("actual", "model", "tabpfn"))


def table_rule(selection: pd.DataFrame, scales: dict[str, float]) -> tuple[float, float]:
    """One numerical weight (grid, residuals scaled per column) and one categorical pick for the table."""
    from src.training.imputation_metrics import NUMERICAL

    numbers = selection[selection["kind"] == NUMERICAL]
    sse = dict.fromkeys(weights.GRID, 0.0)
    for column, group in numbers.groupby("column"):
        m, t, a = (group[c].to_numpy(dtype=float) for c in ("model", "tabpfn", "actual"))
        scale = scales[str(column)]
        for w in weights.GRID:
            sse[w] += float(np.sum(((w * m + (1 - w) * t - a) / scale) ** 2))
    best = min(sse.values())
    numerical = weights.EQUAL if sse[weights.EQUAL] == best else min(w for w in weights.GRID if sse[w] == best)
    labels = selection[selection["kind"] != NUMERICAL]
    categorical = 0.0 if int((labels["tabpfn"] == labels["actual"]).sum()) > int((labels["model"] == labels["actual"]).sum()) else 1.0
    return numerical, categorical


def rule_weights(rule: str, selection: pd.DataFrame, pooled: pd.DataFrame, scales: dict[str, float]) -> dict[str, float]:
    from src.training.imputation_metrics import NUMERICAL

    if rule == "column":
        return weights.choose(selection)
    if rule == "pooled":
        return weights.choose(pooled)
    numerical, categorical = table_rule(selection, scales)
    kinds = {str(c): str(k) for c, k in selection.groupby("column")["kind"].first().items()}
    if rule == "table":
        return {c: numerical if k == NUMERICAL else categorical for c, k in kinds.items()}
    own = weights.choose(selection)
    counts = {str(c): int(n) for c, n in selection.groupby("column").size().items()}
    return {c: (counts[c] * own[c] + N0 * weights.EQUAL) / (counts[c] + N0) if k == NUMERICAL
            else (own[c] if counts[c] >= N0 else categorical) for c, k in kinds.items()}


def scores(variant: str, seed: int) -> pd.DataFrame:
    """Per fold: H, TabPFN, the equal blend and the four rules, on the test induced cells."""
    folds = range(1, tb.FOLDS + 1)
    tests = {k: _load(variant, seed, "test", k) for k in folds}
    selections = {k: _load(variant, seed, "selection", k) for k in folds}
    pooled = pd.concat(selections.values(), ignore_index=True)
    rows = []
    for k in folds:
        test = tests[k]
        naive: dict[str, float | str] = json.loads((STATE / "cells" / f"{variant}_s{seed}" / f"naive_fold_{k}.json").read_text())
        fills = {"model": list(test["model"]), "tabpfn": list(test["tabpfn"]), "equal": weights.apply(test, {}, weights.EQUAL)}
        scales: dict[str, float] = json.loads((STATE / "cells" / f"{variant}_s{seed}" / f"scale_fold_{k}.json").read_text())
        fills |= {rule: weights.apply(test, rule_weights(rule, selections[k], pooled, scales), weights.EQUAL) for rule in RULES}
        for method, values in fills.items():
            rows.append({"variant": variant, "seed": seed, "fold": k, "method": method,
                         "impute_score": tb._score(test.assign(imputed=np.array(values, dtype=object)), naive)})
    return pd.DataFrame(rows)


def tally() -> None:
    e82_table = pd.concat([pd.read_csv(p) for p in sorted((e82.STATE / "blend" / "csv").glob("*.csv"))], ignore_index=True)
    done = [(v, s) for v in e82.VARIANTS for s in e82.SEEDS if (STATE / "cells" / f"{v}_s{s}" / "selection_fold_5.csv").exists()]
    if len(done) < len(e82.VARIANTS) * len(e82.SEEDS):
        print(f"Primary: incomplete ({len(done)} of {len(e82.VARIANTS) * len(e82.SEEDS)} cells)")
        return
    table = pd.concat([scores(v, s) for v, s in done], ignore_index=True)
    drift = max(float(np.max(np.abs(np.array(weights._values(table, v, list(e82.SEEDS), m)) - np.array(e82._values(e82_table, v, e82m)))))
                for v in e82.VARIANTS for m, e82m in (("tabpfn", "tabpfn"), ("column", "deployable")))
    print(f"- reproduction: largest fold difference from E82's TabPFN and deployable scores {drift:.2e}\n")
    calls: dict[str, dict[str, dict[str, str]]] = {rule: {} for rule in RULES}
    for variant in e82.VARIANTS:
        values = {m: weights._values(table, variant, list(e82.SEEDS), m) for m in ("model", "tabpfn", "equal", *RULES)}
        parts = []
        for rule in RULES:
            calls[rule][variant] = {}
            for other in ("model", "tabpfn", "column"):
                if rule == other:
                    continue
                v = common.verdict([a - b for a, b in zip(values[rule], values[other])], sesoi=SESOI * float(np.mean([o.mean() for o in values[other]])))
                calls[rule][variant][other] = v.call
                if rule == "shrunk" or other == "tabpfn":
                    parts.append(f"{rule} − {other} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
        print(f"- `{variant}` (H {np.concatenate(values['model']).mean():.4f}, TabPFN {np.concatenate(values['tabpfn']).mean():.4f}): " + "; ".join(parts))
    print()
    for rule in RULES:
        over = [v for v, c in calls[rule].items() if c["model"] == "better"]
        under = [v for v, c in calls[rule].items() if c["model"] == "worse"]
        behind = [v for v, c in calls[rule].items() if c["tabpfn"] == "worse"]
        if len(over) >= 8 and not under and len(behind) <= 2:
            reading = "supported"
        elif len(over) <= 5 or len(under) >= 2 or len(behind) >= 5:
            reading = "refuted"
        else:
            reading = "mixed"
        vs_column = "" if rule == "column" else f"; against column: {common.tally(c['column'] for c in calls[rule].values())}"
        label = "Primary" if rule == "shrunk" else "Secondary"
        print(f"{label} `{rule}`: better than H on {len(over)} of {len(e82.VARIANTS)}, worse on {under or 'none'}, "
              f"worse than TabPFN on {len(behind)} ({behind or 'none'}) -> {reading}{vs_column} ({common.RULE})")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["cells"] and len(argv) == 3:
        cells(argv[1], int(argv[2]))
    elif argv in (["--tally"], []):
        tally()
    else:
        sys.exit(__doc__)
