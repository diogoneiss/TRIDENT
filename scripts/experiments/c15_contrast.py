"""C15's H2.2 and H2b readers (tickets 04 and 05), inference on the kept models of Stages 1-3. CPU.

    uv run --python 3.11 python -B scripts/experiments/c15_contrast.py --h22 dev|confirmation [jobs]
    uv run --python 3.11 python -B scripts/experiments/c15_contrast.py --h2b [jobs]

**H2.2's direct contrast** (A1) for column j on a fold's test rows whose j is a gap: the logit of P(y = 1) with
every gap ``[NULL]``, minus the logit of the mean P(y = 1) over K = 20 MICE draws of x_ij given the row
(``IterativeImputer(sample_posterior=True)``, seeds 0-19, fitted on the fold's training rows' features), the row's
other gaps left ``[NULL]``; on vehicle, the total variation between the two class distributions. C_j is its mean
over the column's gap cells; U_drive and U_nondrive are mean |C_j| over E122's driving columns and the rest
(an outcome-dependent variant: every masked column). Cached per fold under ``<state>/h22``.

**H2b's grid** on P's development counterparts: each arm's test log loss on the same test rows under D1 (the
variant's own gaps), D4 (the MCAR sibling's gaps on those rows) and, on the outcome variants, D6 (the test rows'
gaps redrawn at each column's D1 count with odds 1/3 on the label, seed 4949). A2 and A4 fill the new gaps with
their fold's imputer, refitted on the training rows as in training. s = (LL_regime − LL_D1) / LL_D1 per fold.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from scripts.experiments import c15_report as rep  # noqa: E402
from scripts.experiments import c15_run as run  # noqa: E402

K = 20
DRIVERS = Path("/scratch2/diogoneiss/trident-2026-11-03-c15-stage0/drivers.json")
OUT = run.STATE / "h22"
# P's development counterparts and their MCAR siblings (ticket 04), and P's fresh draws (ticket 05).
DEV_PAIRS = {"biodeg_probit_60nan": "biodeg_60nan", "vehicle_probit_60nan": "vehicle_60nan",
             **{f"{b}_outcome3p0_s2060_60nan": f"{b}_60nan" for b in ("biodeg", "kc2", "spambase")}}
CONFIRMATION_PAIRS = {"biodeg_probit0p5_s4848_60nan": "biodeg_mcar_s4848_60nan",
                      "vehicle_probit0p5_s4848_60nan": "vehicle_mcar_s4848_60nan",
                      **{f"{b}_outcome3p0_s4848_60nan": f"{b}_mcar_s4848_60nan" for b in ("biodeg", "kc2", "spambase")}}
F6_PAIRS = {f"{b}_mar_60nan": f"{b}_marctl_60nan" for b in ("biodeg", "kc2", "vehicle")}


def _draws(train: np.ndarray, rows: np.ndarray) -> np.ndarray:
    from sklearn.experimental import enable_iterative_imputer  # type: ignore[import-untyped]  # noqa: F401
    from sklearn.impute import IterativeImputer  # type: ignore[attr-defined]

    return np.stack([IterativeImputer(sample_posterior=True, random_state=k).fit(train).transform(rows) for k in range(K)])


def column_contrasts(variant: str, seed: int, fold_number: int) -> dict[str, float]:
    """C_j for every column with test gaps, for one A1 fold."""
    from src.training.data import build_folds

    cache = OUT / f"{variant}_s{seed}_f{fold_number}.json"
    if cache.exists():
        return dict(json.loads(cache.read_text()))
    folds = rep.readouts("A1", variant, seed)
    assert folds is not None, f"A1 {variant} s{seed} missing"
    fold = folds[fold_number - 1]
    model, data = rep.model_for(variant, seed, "A1", fold["state_dict"])
    split = build_folds(data.frame, data.label_column, 5, seed)[fold_number - 1]
    assert np.array_equal(np.asarray(split.test_indices), fold["test_indices"])
    columns = list(data.numerical_columns)
    features = data.frame[columns]
    train = features.iloc[np.asarray(split.train_indices)].to_numpy(dtype=float)
    test = features.iloc[np.asarray(split.test_indices)].reset_index(drop=True)
    gaps = test.isna().to_numpy()
    draws = _draws(train, test.to_numpy(dtype=float))  # (K, rows, columns)
    p_null = rep.probabilities(model, test)
    binary = p_null.shape[1] == 2
    out: dict[str, float] = {}
    for j, column in enumerate(columns):
        rows = np.where(gaps[:, j])[0]
        if len(rows) == 0:
            continue
        mean = np.zeros((len(rows), p_null.shape[1]))
        for k in range(K):
            frame = test.iloc[rows].reset_index(drop=True).copy()
            frame[column] = draws[k, rows, j]
            mean += rep.probabilities(model, frame) / K
        if binary:
            q0, q1 = np.clip(p_null[rows, 1], 1e-7, 1 - 1e-7), np.clip(mean[:, 1], 1e-7, 1 - 1e-7)
            out[column] = float(np.mean(np.log(q0 / (1 - q0)) - np.log(q1 / (1 - q1))))
        else:
            out[column] = float(np.mean(0.5 * np.abs(p_null[rows] - mean).sum(1)))
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def u_values(variant: str, contrasts: dict[str, float]) -> tuple[float, float]:
    table = json.loads(DRIVERS.read_text())[variant.split("_")[0]]
    if "_outcome" in variant:
        drive = list(contrasts)
    else:
        drive = [c for c in contrasts if table.get(c, {}).get("drives")]
    rest = [c for c in contrasts if c not in drive]
    def mean(cs: list[str]) -> float:
        return float(np.mean([abs(contrasts[c]) for c in cs])) if cs else float("nan")
    return mean(drive), mean(rest)


def random_effects(estimates: list[np.ndarray]) -> tuple[float, float, float]:
    """DerSimonian–Laird over variants, each a vector of per-cell differences; t on n − 1 df."""
    means = np.asarray([e.mean() for e in estimates])
    var = np.asarray([e.var(ddof=1) * (1 / len(e) + rep.RATIO) for e in estimates])
    w = 1 / var
    fixed = float((w * means).sum() / w.sum())
    q = float((w * (means - fixed) ** 2).sum())
    tau = max(0.0, (q - (len(means) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    w = 1 / (var + tau)
    mean = float((w * means).sum() / w.sum())
    half = float(stats.t.ppf(0.975, len(means) - 1) * np.sqrt(1 / w.sum()))
    return mean, mean - half, mean + half


def h22(which: str, jobs: int) -> None:
    from joblib import Parallel, delayed  # type: ignore[import-untyped]

    pairs = DEV_PAIRS if which == "dev" else CONFIRMATION_PAIRS
    seeds = run.SEEDS if which == "dev" else run.CONFIRMATION_SEEDS
    variants = sorted(set(pairs) | set(pairs.values()) | (set(F6_PAIRS) | set(F6_PAIRS.values()) if which == "dev" else set()))
    todo = [(v, s, f) for v in variants for s in seeds for f in range(1, 6)]
    Parallel(n_jobs=jobs, verbose=2)(delayed(column_contrasts)(*t) for t in todo)
    print(f"## H2.2 ({which}): U_drive(MNAR) − U_drive(MCAR sibling), per cell, random-effects pooled; "
          "(b) sign agreement on sign-test columns; (c) F6's MAR − control within ±0.1 logit")
    diffs = []
    table_all = json.loads(DRIVERS.read_text())
    agree = trials = 0
    for mnar, mcar in pairs.items():
        d = np.asarray([u_values(mnar, column_contrasts(mnar, s, f))[0] - u_values(mcar, column_contrasts(mcar, s, f))[0]
                        for s in seeds for f in range(1, 6)])
        diffs.append(d)
        m, lo, hi = rep.interval(d)
        nd = np.mean([u_values(mnar, column_contrasts(mnar, s, f))[1] for s in seeds for f in range(1, 6)])
        print(f"- {mnar} vs {mcar}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]; U_nondrive(MNAR) {nd:.3f}")
        if "_probit" in mnar and not mnar.startswith("vehicle"):
            table = table_all[mnar.split("_")[0]]
            for column, row in table.items():
                if row.get("sign_test"):
                    cs = [column_contrasts(mnar, s, f).get(column) for s in seeds for f in range(1, 6)]
                    c = float(np.nanmean([x for x in cs if x is not None]))
                    trials += 1
                    agree += int(np.sign(c) == row["sign"])
    m, lo, hi = random_effects(diffs)
    a = "supported" if lo > 0 else "refuted" if (abs(lo) <= 0.1 and abs(hi) <= 0.1) or hi < 0 else "inconclusive"
    p = float(stats.binomtest(agree, trials, 0.5, alternative="greater").pvalue) if trials else float("nan")
    print(f"\n(a) pooled {m:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {a}")
    print(f"(b) sign agreement {agree} of {trials}, one-sided binomial p = {p:.3f}")
    if which == "dev":
        for mar, ctl in F6_PAIRS.items():
            d = np.asarray([u_values(mar, column_contrasts(mar, s, f))[0] - u_values(ctl, column_contrasts(ctl, s, f))[0]
                            for s in seeds for f in range(1, 6)])
            m, lo, hi = rep.interval(d)
            print(f"(c) {mar} − {ctl}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'within' if abs(m) <= 0.1 else 'outside'} ±0.1")


def _regime_ll(arm: str, variant: str, seed: int, fold_number: int, test: pd.DataFrame, data: Any, split: Any) -> float:
    """An arm's test log loss on ``test`` (scaled features with NaN gaps), refilling as the arm does."""
    from sklearn.metrics import log_loss  # type: ignore[attr-defined]

    from src.training.c15_inputs import GAP_PREFIX
    from src.training.missforest import MissForestImputer

    columns = list(data.numerical_columns)
    train = data.frame[columns].iloc[np.asarray(split.train_indices)]
    y = data.frame[data.label_column].iloc[np.asarray(split.test_indices)].to_numpy()
    arms = [f"A4:{k}" for k in range(5)] if arm == "A4" else [arm]
    probs = []
    for a in arms:
        folds = rep.readouts(a, variant, seed)
        assert folds is not None
        model, _ = rep.model_for(variant, seed, a, folds[fold_number - 1]["state_dict"])
        frame = test.copy()
        if a == "A2":
            forest = MissForestImputer(columns, [])
            forest.fit(train)
            filled = forest.impute(test, np.zeros(test.shape, dtype=bool))
            frame = pd.DataFrame(filled[columns].to_numpy(dtype=float), columns=columns)
            for c in columns:
                frame[GAP_PREFIX + c] = test[c].isna().to_numpy(dtype=float)
        elif a.startswith("A4:"):
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer  # type: ignore[attr-defined]

            imputer = IterativeImputer(sample_posterior=True, random_state=int(a.split(":")[1])).fit(train.to_numpy(dtype=float))
            frame = pd.DataFrame(imputer.transform(test.to_numpy(dtype=float)), columns=columns)
        probs.append(rep.probabilities(model, frame))
    p = np.mean(probs, axis=0)
    return float(log_loss(y, p, labels=np.arange(p.shape[1])))


def h2b_fold(variant: str, seed: int, fold_number: int) -> dict[str, float]:
    from src.training.data import build_folds, prepare_dataset
    from src.training.types import DatasetSpec

    cache = run.STATE / "h2b" / f"{variant}_s{seed}_f{fold_number}.json"
    if cache.exists():
        return dict(json.loads(cache.read_text()))
    data = prepare_dataset(DatasetSpec.from_name(variant, "class"))
    split = build_folds(data.frame, data.label_column, 5, seed)[fold_number - 1]
    columns = list(data.numerical_columns)
    idx = np.asarray(split.test_indices)
    base = variant.split("_")[0]
    complete = pd.read_csv(ROOT / "datasets" / "processed_datasets" / base / f"{base}_00nan.csv")[columns]
    scaled = pd.DataFrame(data.scaler.transform(complete), columns=columns).iloc[idx].reset_index(drop=True)
    own = data.frame[columns].iloc[idx].reset_index(drop=True)
    sibling = pd.read_csv(ROOT / "datasets" / "processed_datasets" / base / f"{base}_60nan.csv")[columns].iloc[idx].reset_index(drop=True)
    regimes = {"D1": own, "D4": scaled.mask(sibling.isna())}
    if "_outcome" in variant:
        y = data.frame[data.label_column].iloc[idx].to_numpy()
        stream = np.random.default_rng(4949)
        redrawn = scaled.copy()
        for c in columns:
            count = int(own[c].isna().sum())
            w = (1 / 3) ** y
            hidden = stream.choice(len(redrawn), size=count, replace=False, p=w / w.sum())
            redrawn.loc[redrawn.index[hidden], c] = np.nan
        regimes["D6"] = redrawn
    out = {f"{arm}_{r}": _regime_ll(arm, variant, seed, fold_number, t, data, split)
           for arm in ("A1", "A2", "A4") for r, t in regimes.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def h2b(jobs: int) -> None:
    from joblib import Parallel, delayed

    todo = [(v, s, f) for v in DEV_PAIRS for s in run.SEEDS for f in range(1, 6)]
    rows = Parallel(n_jobs=jobs, verbose=2)(delayed(h2b_fold)(*t) for t in todo)
    print("## H2b (development): shift cost s = (LL_regime − LL_D1) / LL_D1; s_A1 − s_A4 and s_A2 − s_A4 over D4 "
          "(5 variants) and D6 (3), pooled per cell; supported if both > 0 (Holm), refuted if both upper bounds < +0.02")
    by: dict[str, dict[str, list[float]]] = {"A1": {}, "A2": {}}
    for (v, s, f), r in zip(todo, rows):
        for regime in ("D4", "D6"):
            if f"A1_{regime}" not in r:
                continue
            cost = {a: (r[f"{a}_{regime}"] - r[f"{a}_D1"]) / r[f"{a}_D1"] for a in ("A1", "A2", "A4")}
            for a in ("A1", "A2"):
                by[a].setdefault(f"{s}_{f}", []).append(cost[a] - cost["A4"])
    results, pvalues = {}, {}
    for a in ("A1", "A2"):
        cells = np.asarray([np.mean(v) for v in by[a].values()])
        results[a] = rep.interval(cells)
        se = float(np.sqrt(cells.var(ddof=1) * (1 / len(cells) + rep.RATIO)))
        pvalues[a] = float(2 * stats.t.sf(abs(cells.mean() / se), len(cells) - 1))
        m, lo, hi = results[a]
        print(f"- s_{a} − s_A4: {m:+.4f} [{lo:+.4f}, {hi:+.4f}], two-sided p {pvalues[a]:.4f}")
    ordered = sorted(pvalues.values())
    # Holm over the two contrasts at 0.05: the smaller p below 0.025, the larger below 0.05, both means above 0.
    supported = ordered[0] < 0.025 and ordered[1] < 0.05 and all(r[0] > 0 for r in results.values())
    refuted = all(r[2] < 0.02 for r in results.values())
    print(f"\nPrimary (H2b, development): {'supported' if supported else 'refuted' if refuted else 'inconclusive'}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    jobs = int(argv[-1]) if argv and argv[-1].isdigit() else 16
    if argv[:1] == ["--h22"] and len(argv) >= 2:
        h22(argv[1], jobs)
    elif argv[:1] == ["--h2b"]:
        h2b(jobs)
    else:
        sys.exit(__doc__)
