"""E175 (imputation-protocol/68, T56): multiple imputation from H's fills, offline, on E174's cells.

    uv run --python 3.11 python -B scripts/experiments/multiple_imputation.py --check
    uv run --python 3.11 python -B scripts/experiments/multiple_imputation.py cells <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/multiple_imputation.py --tally

It trains nothing of TRIDENT's and writes nothing to the store. ``cells`` rebuilds a (variant, seed)'s five folds and,
per fold, completes the test rows five ways, then reads three families of estimands on them (every numerical column's
mean, every categorical column's share of the complete table's commonest category, the slope of each numerical column
on the next one in file order) against the complete table's value, with Rubin's rules:

- ``CD``: the truth in every gap (the ledger's ``actual``); the control that the intervals themselves cover;
- ``SI``: H's fill, once, the complete-data variance alone;
- ``MI-D``: M draws around H's fill from the fold's hidden observed validation cells (deployable): a numerical cell
  takes H's fill plus a residual of its column, a categorical one a truth drawn among the validation cells where H
  predicted the same class; the pool is resampled with replacement for every draw;
- ``MI-O``: the same from the other four folds' induced cells (an oracle pool: it reads the truth of gaps), the
  numerical residuals per tertile of the row's gap count (ADR 0030 decision 8's bins);
- ``MICE``: ICE's chained equations fitted once on the training split, each fill drawn from its conditional's
  predictive distribution (``BayesianRidge`` mean and sd, the logistic model's probabilities).

``--tally`` prints the pre-registered reading. ``--check`` runs the reader's self-check on synthetic data only.
Run from the checkout root.
"""

from __future__ import annotations

import sys
import time
import warnings
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from sklearn.linear_model import BayesianRidge

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import hybrid_selection_rules as rules  # noqa: E402
from scripts.experiments import report_common as common  # noqa: E402
from src.training import linear_baselines as lb  # noqa: E402

STATE = Path("/scratch/diogoneiss/trident-2026-11-12-multiple-imputation")
SOURCE = Path("/scratch/diogoneiss/trident-2026-11-10-hybrid-thesis-seeds")  # E174's cells: H's fills, both populations
VARIANTS = tuple(f"{table}_{rate}nan" for table in ("biodeg", "kc2", "vehicle", "credit-g", "pendigits") for rate in (20, 40, 80))
SEEDS = (101, 202, 303)
FOLDS = 5
M = 20
LEVEL = 0.95
MIN_FIT = 30
SESOI = 0.01
ARMS = ("CD", "SI", "MI-D", "MI-O", "MICE")
Floats = NDArray[np.float64]


def rubin(q: Floats, u: Floats, complete_df: float) -> tuple[float, float, float, float]:
    """Rubin's rules on M estimates and their variances: the pooled estimate, the total variance, Barnard and
    Rubin's degrees of freedom, and lambda, the share of the total variance that is between imputations."""
    m = len(q)
    pooled, within = float(q.mean()), float(u.mean())
    between = float(q.var(ddof=1)) if m > 1 else 0.0
    total = within + (1 + 1 / m) * between
    if between <= 0 or total <= 0:
        return pooled, total, complete_df, 0.0
    lam = (1 + 1 / m) * between / total
    observed = max((complete_df + 1) / (complete_df + 3) * complete_df * (1 - lam), 1e-9)
    return pooled, total, 1 / (lam**2 / (m - 1) + 1 / observed), lam


def estimands(numbers: Floats, labels: NDArray[np.object_], modes: list[str], fpc: float) -> tuple[Floats, Floats, Floats]:
    """Each estimand's value, its complete-data variance (times ``fpc``) and its degrees of freedom, in the order
    means, shares, slopes. ``numbers`` and ``labels`` are a complete table's numerical and categorical blocks."""
    n = len(numbers) if numbers.shape[1] else len(labels)
    q: list[float] = []
    u: list[float] = []
    df: list[float] = []
    for j in range(numbers.shape[1]):
        q.append(float(numbers[:, j].mean()))
        u.append(float(numbers[:, j].var(ddof=1)) / n * fpc)
        df.append(n - 1)
    for j, mode in enumerate(modes):
        share = float((labels[:, j] == mode).mean())
        q.append(share)
        u.append(share * (1 - share) / (n - 1) * fpc)
        df.append(n - 1)
    for j in range(numbers.shape[1] - 1):
        y, x = numbers[:, j], numbers[:, j + 1]
        centred = x - x.mean()
        sxx = float(centred @ centred)
        slope = float(centred @ (y - y.mean())) / sxx if sxx > 0 else 0.0
        residual = (y - y.mean()) - slope * centred
        q.append(slope)
        u.append(float(residual @ residual) / (n - 2) / sxx * fpc if sxx > 0 else 0.0)
        df.append(n - 2)
    return np.array(q), np.array(u), np.array(df)


def names(numerical: list[str], categorical: list[str]) -> list[tuple[str, str]]:
    """(family, name) of every estimand, in ``estimands``' order."""
    return ([("mean", c) for c in numerical] + [("share", c) for c in categorical]
            + [("slope", f"{a}~{b}") for a, b in zip(numerical, numerical[1:])])


def _edges(pool: pd.DataFrame) -> Floats:
    """E151's bins: the tertiles of the row's gap count over the pool's numerical cells."""
    b = pool.loc[pool["kind"] == "numerical", "b"].to_numpy(dtype=float)
    return np.unique(np.quantile(b, [1 / 3, 2 / 3])) if len(b) else np.empty(0)


def draws(test: pd.DataFrame, pool: pd.DataFrame, rng: np.random.Generator, binned: bool) -> tuple[list[NDArray[np.object_]], float]:
    """M fills of ``test``'s cells around H's (``model``), from ``pool``'s cells (``actual`` and ``model``), and the
    share of cells with no pool, which keep H's fill. ``binned`` splits a numerical column's residuals by the
    tertile of ``b`` where the bin holds at least ``MIN_FIT`` of them."""
    edges = _edges(pool) if binned else np.empty(0)
    groups: list[tuple[NDArray[np.intp], Floats | NDArray[np.object_], bool]] = []
    kept = 0
    for column, cells in test.groupby("column"):
        source = pool[pool["column"] == column]
        at = test.index.get_indexer(cells.index)
        if str(cells["kind"].iloc[0]) == "numerical":
            residual = source["actual"].to_numpy(dtype=float) - source["model"].to_numpy(dtype=float)
            parts: list[tuple[NDArray[np.intp], Floats]] = [(at, residual)]
            if binned and len(edges):
                level, source_level = np.searchsorted(edges, cells["b"].to_numpy(dtype=float)), np.searchsorted(edges, source["b"].to_numpy(dtype=float))
                parts = [(at[level == k], residual[source_level == k] if int((source_level == k).sum()) >= MIN_FIT else residual)
                         for k in np.unique(level)]
            for where, values in parts:
                kept += len(where) if not len(values) else 0
                groups.append((where, values, True))
        else:
            guessed, truth = source["model"].to_numpy(dtype=object), source["actual"].to_numpy(dtype=object)
            predicted = cells["model"].to_numpy(dtype=object)
            for label in np.unique(predicted.astype(str)):
                same, truths = at[predicted == label], truth[guessed == label]
                kept += len(same) if not len(truths) else 0
                groups.append((same, truths, False))
    base = test["model"].to_numpy(dtype=object)
    out: list[NDArray[np.object_]] = []
    for _ in range(M):
        fill = base.copy()
        for cells_at, source_values, numerical in groups:
            if len(source_values):
                drawn = rng.choice(rng.choice(source_values, len(source_values)), len(cells_at))
                fill[cells_at] = base[cells_at].astype(float) + drawn if numerical else drawn
        out.append(fill)
    return out, kept / max(len(test), 1)


class _Stochastic:
    """A column model of ICE's that draws from its predictive distribution while its owner holds a generator."""

    def __init__(self, model: Any, owner: MiceImputer) -> None:
        self._model, self._owner = model, owner

    def predict(self, features: Floats) -> Floats:
        rng = self._owner.rng
        if rng is not None and isinstance(self._model, BayesianRidge):
            mean, sd = self._model.predict(features, return_std=True)  # type: ignore[call-arg]  # the stub omits return_std
            return np.asarray(rng.normal(mean, sd), dtype=float)
        if rng is not None and isinstance(self._model, lb._Labels):
            inner = self._model._model
            p = inner.predict_proba(features)
            pick = np.minimum((p.cumsum(axis=1) < rng.random((len(p), 1))).sum(axis=1), p.shape[1] - 1)
            return np.asarray(inner.classes_[pick], dtype=float)
        return np.asarray(self._model.predict(features), dtype=float)


class MiceImputer(lb.IceImputer):
    """ICE, fitted as ICE; with ``rng`` set, ``impute`` draws every fill from its conditional (one chain a call)."""

    name = "mice"
    rng: np.random.Generator | None = None

    def _fit_one(self, index: int, features: Floats, target: Floats) -> Any:
        return _Stochastic(super()._fit_one(index, features, target), self)


def _blocks(frame: pd.DataFrame, numerical: list[str], categorical: list[str]) -> tuple[Floats, NDArray[np.object_]]:
    return frame[numerical].to_numpy(dtype=float), frame[categorical].to_numpy(dtype=object)


def cells(variant: str, seed: int) -> None:
    from src.embedder import as_category_strings
    from src.training.data import build_folds, prepare_dataset
    from src.training.types import DatasetSpec

    warnings.filterwarnings("ignore")
    started = time.time()
    rules.STATE = SOURCE
    dataset = prepare_dataset(DatasetSpec.from_name(variant, None))
    features = dataset.frame.drop(columns=[dataset.label_column]).reset_index(drop=True)
    numerical, categorical = list(dataset.numerical_columns), list(dataset.categorical_columns)
    folds = list(build_folds(dataset.frame, dataset.label_column, FOLDS, seed))
    ledgers: dict[int, pd.DataFrame] = {}
    truth = features.astype(object)
    for c in categorical:
        truth[c] = as_category_strings(features[c]).astype(object)
    block = truth.to_numpy(dtype=object, copy=True)
    for k, fold in enumerate(folds, start=1):
        ledger = rules._load(variant, seed, "test", k).reset_index(drop=True)
        ledger["b"] = ledger["row"].map(ledger.groupby("row").size()).astype(float)
        ledger["at"] = np.asarray(fold.test_indices)[ledger["row"].to_numpy(dtype=int)]
        ledger["position"] = features.columns.get_indexer(pd.Index(ledger["column"]))
        block[ledger["at"].to_numpy(), ledger["position"].to_numpy()] = ledger["actual"].to_numpy(dtype=object)
        ledgers[k] = ledger
    gaps = features.isna().to_numpy()
    filled = np.zeros_like(gaps)
    for ledger in ledgers.values():
        filled[ledger["at"].to_numpy(), ledger["position"].to_numpy()] = True
    assert (gaps == filled).all(), f"{variant} s{seed}: the five ledgers do not hold exactly the table's gaps"
    complete = pd.DataFrame(block, columns=features.columns)
    for c in categorical:
        seen = set(as_category_strings(features[c]).astype(str)[~features[c].isna().to_numpy()])
        hidden = {str(v) for ledger in ledgers.values() for v in ledger.loc[ledger["column"] == c, "actual"]}
        assert not hidden or seen & hidden, f"{c}: the ledger's categories are not written as the table's"
    modes = [str(complete[c].astype(str).mode().iloc[0]) for c in categorical]
    complete[categorical] = complete[categorical].astype(str)
    target = estimands(*_blocks(complete, numerical, categorical), modes, 1.0)[0]
    labelled = names(numerical, categorical)
    rows: list[dict[str, object]] = []
    kept = {"MI-D": 0.0, "MI-O": 0.0}
    for k, fold in enumerate(folds, start=1):
        test_at = np.asarray(fold.test_indices)
        ledger = ledgers[k]
        local, position = ledger["row"].to_numpy(dtype=int), ledger["position"].to_numpy()
        base = complete.iloc[test_at].to_numpy(dtype=object, copy=True)
        fpc = 1 - len(test_at) / len(complete)

        def read(fills: list[NDArray[np.object_]]) -> tuple[Floats, Floats, Floats, Floats]:
            qs, us = [], []
            for fill in fills:
                table = base.copy()
                table[local, position] = fill
                frame = pd.DataFrame(table, columns=features.columns)
                frame[categorical] = frame[categorical].astype(str)
                q, u, df = estimands(*_blocks(frame, numerical, categorical), modes, fpc)
                qs.append(q)
                us.append(u)
            pooled = [rubin(np.array([q[i] for q in qs]), np.array([u[i] for u in us]), float(df[i])) for i in range(len(df))]
            return (np.array([p[0] for p in pooled]), np.array([p[1] for p in pooled]), np.array([p[2] for p in pooled]),
                    np.array([p[3] for p in pooled]))

        def rng(arm: int) -> np.random.Generator:
            return np.random.default_rng([175, zlib.crc32(variant.encode()), seed, k, arm])

        selection = rules._load(variant, seed, "selection", k)
        for c in numerical:  # the ledgers must be written in the frame's space: hidden observed cells are frame values
            hidden_values = selection.loc[selection["column"] == c, "actual"].to_numpy(dtype=float)
            known = np.sort(features[c].dropna().to_numpy(dtype=float))
            if len(hidden_values) and len(known):
                nearest = np.abs(known[np.clip(np.searchsorted(known, hidden_values), 1, len(known) - 1) - 1] - hidden_values)
                nearest = np.minimum(nearest, np.abs(known[np.clip(np.searchsorted(known, hidden_values), 0, len(known) - 1)] - hidden_values))
                assert float((nearest < 1e-4).mean()) > 0.99, f"{c}: the selection ledger is not in the table's units"
        others = pd.concat([ledgers[j] for j in ledgers if j != k], ignore_index=True)
        deployable, kept_d = draws(ledger, selection, rng(2), binned=False)
        oracle, kept_o = draws(ledger, others, rng(3), binned=True)
        kept["MI-D"] += kept_d / FOLDS
        kept["MI-O"] += kept_o / FOLDS
        mice = MiceImputer(numerical, categorical)
        mice.fit(features.iloc[np.asarray(fold.train_indices)].reset_index(drop=True))
        test = features.iloc[test_at].reset_index(drop=True)
        mice.rng = rng(4)
        chained = [mice.impute(test, np.zeros(test.shape, dtype=bool)).to_numpy(dtype=object)[local, position] for _ in range(M)]
        fills = {"CD": [ledger["actual"].to_numpy(dtype=object)], "SI": [ledger["model"].to_numpy(dtype=object)],
                 "MI-D": deployable, "MI-O": oracle, "MICE": chained}
        for arm in ARMS:
            estimate, total, df, lam = read(fills[arm])
            half = stats.t.ppf(0.5 + LEVEL / 2, df) * np.sqrt(total)
            rows.extend({"variant": variant, "seed": seed, "fold": k, "arm": arm, "family": family, "estimand": name,
                         "truth": target[i], "estimate": estimate[i], "half": half[i], "lam": lam[i],
                         "covered": bool(abs(estimate[i] - target[i]) <= half[i])} for i, (family, name) in enumerate(labelled))
    out = STATE / "csv"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{variant}_s{seed}.csv", index=False)
    print(f"DONE {variant} s{seed}: {len(labelled)} estimands, cells keeping H's fill MI-D {kept['MI-D']:.3f} MI-O {kept['MI-O']:.3f}, "
          f"{time.time() - started:.0f} s", flush=True)


def measures(data: pd.DataFrame) -> pd.DataFrame:
    """Per (variant, seed, fold, arm): coverage, shortfall u, and against CD the relative width and the estimate's
    shift in CD standard errors; lambda's mean."""
    key = ["variant", "seed", "fold", "family", "estimand"]
    control = data[data["arm"] == "CD"].set_index(key)[["estimate", "half"]].rename(columns={"estimate": "cd", "half": "cd_half"})
    joined = data.join(control, on=key)
    wide = joined[joined["cd_half"] > 0]
    out = joined.groupby(["variant", "seed", "fold", "arm"]).agg(coverage=("covered", "mean"), lam=("lam", "mean"))
    out["u"] = (LEVEL - out["coverage"]).clip(lower=0)
    out["width"] = (wide["half"] / wide["cd_half"]).groupby([wide[c] for c in ("variant", "seed", "fold", "arm")]).mean()
    out["shift"] = ((wide["estimate"] - wide["cd"]).abs() / (wide["cd_half"] / 1.96)).groupby([wide[c] for c in ("variant", "seed", "fold", "arm")]).mean()
    return out.reset_index()


def tally() -> None:
    missing = [(v, s) for v in VARIANTS for s in SEEDS if not (STATE / "csv" / f"{v}_s{s}.csv").exists()]
    if missing:
        print(f"incomplete: {len(missing)} of {len(VARIANTS) * len(SEEDS)} cells missing, {missing[:6]}")
        return
    data = pd.concat([pd.read_csv(STATE / "csv" / f"{v}_s{s}.csv") for v in VARIANTS for s in SEEDS], ignore_index=True)
    table = measures(data)
    pairs = (("MI-D", "SI"), ("MI-D", "MICE"), ("MI-O", "MI-D"), ("MICE", "SI"))
    calls: dict[str, dict[str, str]] = {f"{a}-{b}": {} for a, b in pairs}
    proper: dict[str, list[str]] = {arm: [] for arm in ARMS}
    for variant in VARIANTS:
        part = table[table["variant"] == variant]
        level = part.groupby("arm")[["coverage", "u", "width", "shift", "lam"]].mean()
        print(f"- `{variant}`: " + "; ".join(
            f"{arm} coverage {level.loc[arm, 'coverage']:.3f} u {level.loc[arm, 'u']:.3f} width {level.loc[arm, 'width']:.2f} "
            f"shift {level.loc[arm, 'shift']:.2f} lambda {level.loc[arm, 'lam']:.2f}" for arm in ARMS))
        families = data[data["variant"] == variant].groupby(["family", "arm"])["covered"].mean().unstack("arm")
        print("  - by family: " + "; ".join(f"{family} " + " ".join(f"{arm} {families.loc[family, arm]:.3f}" for arm in ARMS)
                                              for family in families.index))
        for arm in level.index[level["coverage"].to_numpy(dtype=float) >= 0.90]:
            proper[str(arm)].append(variant)
        u = {arm: [part[(part["arm"] == arm) & (part["seed"] == s)].sort_values("fold")["u"].to_numpy() for s in SEEDS] for arm in ARMS}
        for first, second in pairs:
            v = common.verdict([a - b for a, b in zip(u[first], u[second])], sesoi=SESOI)
            calls[f"{first}-{second}"][variant] = v.call
            print(f"  - u({first}) − u({second}) {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
    print()
    for arm in ARMS:
        print(f"- mean coverage at least 0.90 under {arm}: {len(proper[arm])} of {len(VARIANTS)}")
    for pair, by_variant in calls.items():
        print(f"- u, {pair}: {common.tally(by_variant.values())}")
    better = [v for v, c in calls["MI-D-SI"].items() if c == "better"]
    worse = [v for v, c in calls["MI-D-SI"].items() if c == "worse"]
    reading = "supported" if len(better) >= 10 and len(worse) <= 1 else "refuted" if len(better) <= 5 or len(worse) >= 4 else "mixed"
    print(f"\nE175 PRIMARY 1, u(MI-D) − u(SI): better on {len(better)} of {len(VARIANTS)}, worse on {worse or 'none'} -> {reading} "
          f"({common.RULE}; SESOI {SESOI})")
    print(f"E175 PRIMARY 2, MI-D proper enough (mean coverage at least 0.90 on at least 10 of {len(VARIANTS)}): "
          f"{len(proper['MI-D'])} -> {'yes' if len(proper['MI-D']) >= 10 else 'no'}")
    behind = [v for v, c in calls["MI-D-MICE"].items() if c == "worse"]
    ahead = [v for v, c in calls["MI-D-MICE"].items() if c == "better"]
    print(f"E175 SECONDARY, u(MI-D) − u(MICE): better on {len(ahead)}, worse on {len(behind)} ({behind or 'none'})")


def check() -> None:
    """Rubin's rules on a hand case; then a synthetic finite population where a proper draw (parameters from their
    posterior, then noise) covers the mean near 0.95 and the conditional-mean fill, read once, covers less."""
    pooled, total, df, lam = rubin(np.array([1.0, 2.0, 3.0]), np.ones(3), 1e9)
    assert abs(pooled - 2) < 1e-12 and abs(total - (1 + 4 / 3)) < 1e-12 and abs(lam - (4 / 3) / (7 / 3)) < 1e-12 and 2 < df < 7, (pooled, total, df, lam)
    q, u, dof = estimands(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.5], [4.0, 8.0]]), np.array([["a"], ["a"], ["b"], ["a"]], dtype=object), ["a"], 1.0)
    assert abs(q[0] - 2.5) < 1e-12 and abs(q[2] - 0.75) < 1e-12 and abs(q[3] - np.polyfit([2.0, 4.0, 6.5, 8.0], [1.0, 2.0, 3.0, 4.0], 1)[0]) < 1e-9, q
    assert list(dof) == [3, 3, 3, 2] and (u > 0).all()
    rng = np.random.default_rng(175)
    size, n = 4000, 400
    x = rng.normal(size=size)
    y = 0.3 * x + rng.normal(size=size)
    covered = {"proper": 0, "single": 0}
    for _ in range(400):
        rows = rng.choice(size, n, replace=False)
        xs, ys = x[rows], y[rows].copy()
        gone = rng.random(n) < 0.6
        design = np.column_stack([np.ones(n), xs])
        beta, *_ = np.linalg.lstsq(design[~gone], ys[~gone], rcond=None)
        sigma2 = float(((ys[~gone] - design[~gone] @ beta) ** 2).sum()) / (int((~gone).sum()) - 2)
        spread = np.linalg.inv(design[~gone].T @ design[~gone])
        fpc = 1 - n / size
        qs, us = [], []
        for _ in range(M):
            s2 = sigma2 * (int((~gone).sum()) - 2) / rng.chisquare(int((~gone).sum()) - 2)
            drawn = rng.multivariate_normal(beta, s2 * spread)
            filled = ys.copy()
            filled[gone] = design[gone] @ drawn + rng.normal(0, np.sqrt(s2), int(gone.sum()))
            qs.append(filled.mean())
            us.append(filled.var(ddof=1) / n * fpc)
        once = ys.copy()
        once[gone] = design[gone] @ beta
        for name, (estimate, total, df, _) in (("proper", rubin(np.array(qs), np.array(us), n - 1)),
                                               ("single", rubin(np.array([once.mean()]), np.array([once.var(ddof=1) / n * fpc]), n - 1))):
            covered[name] += abs(estimate - y.mean()) <= stats.t.ppf(0.975, df) * np.sqrt(total)
    share = {name: count / 400 for name, count in covered.items()}
    assert share["proper"] >= 0.92 and share["single"] <= share["proper"] - 0.10, share
    test = pd.DataFrame({"column": ["x"] * 4 + ["c"] * 2, "kind": ["numerical"] * 4 + ["categorical"] * 2, "model": [0.0, 0.0, 0.0, 0.0, "a", "z"], "b": 1.0})
    pool = pd.DataFrame({"column": ["x", "x", "c", "c"], "kind": ["numerical"] * 2 + ["categorical"] * 2, "actual": [1.0, 3.0, "b", "b"], "model": [0.0, 0.0, "a", "a"], "b": 1.0})
    fills, kept = draws(test, pool, np.random.default_rng(1), binned=True)
    assert len(fills) == M and all(set(f[:4]) <= {1.0, 3.0} and f[4] == "b" and f[5] == "z" for f in fills) and abs(kept - 1 / 6) < 1e-12
    print("check passed:", share)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv == ["--check"]:
        check()
    elif argv[:1] == ["cells"] and len(argv) == 3:
        cells(argv[1], int(argv[2]))
    elif argv == ["--tally"]:
        tally()
    else:
        sys.exit(__doc__)
