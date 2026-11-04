"""C15 Stage 0 (docs/tickets/c15-mnar-prediction/01-stage0.md): the driving columns and gate G2. CPU only.

    uv run --python 3.11 python -B scripts/experiments/c15_stage0.py --drivers   # the driving-column table
    uv run --python 3.11 python -B scripts/experiments/c15_stage0.py --g2 [jobs]  # G2's 285 cells, resumable
    uv run --python 3.11 python -B scripts/experiments/c15_stage0.py --tally      # G2's rule and the set P

**Driving columns**, per table on its complete ``_00nan`` file: one HistGradientBoosting classifier at fixed
defaults, 5 folds (StratifiedKFold, seed 42); a column drives if dropping it raises held-out log loss, with
the 95% t interval over the 5 folds above 0. Its sign is its coefficient's in an L2 logistic regression on
the standardised table, and it is a sign-test column if the partial dependence of a model fitted on every
row is monotone (|Spearman| >= 0.8 over a 20-point grid). Binary tables only for the sign and the PD.

**G2**, per candidate variant and cell (seed 42/7/13 x fold 1-5 of the fork's own ``build_folds`` at
``--cv_folds 5``, so the decision mode's folds): missForest (``src/training/missforest.py``) fitted on the
training rows' features, never the label, fills the training and validation rows; the same GBDT is fitted
with and without one 0/1 indicator per column with gaps; the cell's reading is the relative drop in
validation log loss, (LL_without - LL_with) / LL_without. An indicators-only logistic probe is read beside
it, against the training class frequencies. A variant is informative if the mean drop over the 15 cells is
at least 0.02 with its 95% t interval above 0. P is the self-masking (F1) and odds-3 outcome (F3) variants
that pass; every positive control (F5) must pass, or G2 is invalid and the study stops.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import partial_dependence  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss  # type: ignore[attr-defined]
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from src.training.data import build_folds  # noqa: E402
from src.training.missforest import MissForestImputer  # noqa: E402

DATA = ROOT / "datasets" / "processed_datasets"
STATE = Path("/scratch2/diogoneiss/trident-2026-11-03-c15-stage0")
TARGET = "class"
SEEDS = (42, 7, 13)
FOLDS = 5
TABLES = ("biodeg", "kc2", "vehicle", "spambase")
F1 = tuple(f"{b}_probit_{r}nan" for b in ("biodeg", "kc2", "vehicle") for r in (20, 60)) + ("spambase_probit_60nan",)
F2 = tuple(f"biodeg_probit{s}_60nan" for s in ("0p25", "1p0", "2p0"))
F3 = tuple(f"{b}_outcome{o}_s2060_60nan" for b in ("biodeg", "kc2", "spambase") for o in ("1p5", "3p0"))
F5 = tuple(f"{b}_noise3p0_s2060_60nan" for b in ("biodeg", "kc2", "spambase"))
CANDIDATES = F1 + F2 + F3 + F5
THRESHOLD = 0.02


def gbdt() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=0)


def load(variant: str) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(DATA / variant.split("_", 1)[0] / f"{variant}.csv")
    return frame.drop(columns=TARGET), np.asarray(LabelEncoder().fit_transform(frame[TARGET]))


def interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def drivers() -> dict[str, dict[str, dict[str, object]]]:
    out: dict[str, dict[str, dict[str, object]]] = {}
    for table in TABLES:
        x, y = load(f"{table}_00nan")
        x = x.astype(float)  # partial_dependence refuses integer columns (spambase)
        classes = np.unique(y)
        rises: dict[str, list[float]] = {c: [] for c in x.columns}
        for train, test in StratifiedKFold(FOLDS, shuffle=True, random_state=42).split(x, y):
            full = log_loss(y[test], gbdt().fit(x.iloc[train], y[train]).predict_proba(x.iloc[test]), labels=classes)
            for column in x.columns:
                kept = x.drop(columns=column)
                rises[column].append(log_loss(y[test], gbdt().fit(kept.iloc[train], y[train]).predict_proba(kept.iloc[test]),
                                              labels=classes) - full)
        rows: dict[str, dict[str, object]] = {}
        binary = len(classes) == 2
        coef = (LogisticRegression(max_iter=5000).fit(StandardScaler().fit_transform(x), y).coef_[0]  # type: ignore[attr-defined]
                if binary else np.zeros(x.shape[1]))
        model = gbdt().fit(x, y) if binary else None
        for j, column in enumerate(x.columns):
            mean, low, high = interval(np.asarray(rises[column]))
            row: dict[str, object] = {"rise": mean, "low": low, "high": high, "drives": bool(mean > 0 and low > 0)}
            if binary and row["drives"] and model is not None:
                pd_ = partial_dependence(model, x, [j], grid_resolution=20)
                rho = float(stats.spearmanr(pd_["grid_values"][0], pd_["average"][0]).statistic)
                row |= {"sign": int(np.sign(coef[j])), "pd_spearman": rho, "sign_test": bool(abs(rho) >= 0.8)}
            rows[column] = row
        out[table] = rows
        d = [c for c, r in rows.items() if r["drives"]]
        print(f"{table}: {len(d)} of {x.shape[1]} columns drive; sign-test {sum(bool(r.get('sign_test')) for r in rows.values())}")
    (STATE / "drivers.json").write_text(json.dumps(out, indent=1))
    return out


def g2_cell(variant: str, seed: int, fold: int) -> dict[str, float]:
    x, y = load(variant)
    folds = build_folds(pd.DataFrame({TARGET: y}), TARGET, FOLDS, seed)
    split = folds[fold - 1]
    train, valid = np.asarray(split.train_indices), np.asarray(split.validation_indices)
    imputer = MissForestImputer(list(x.columns), [])
    imputer.fit(x.iloc[train])
    none = np.zeros((len(train), x.shape[1]), dtype=bool)
    filled_train = imputer.impute(x.iloc[train], none)
    filled_valid = imputer.impute(x.iloc[valid], np.zeros((len(valid), x.shape[1]), dtype=bool))
    gapped = [c for c in x.columns if x[c].isna().any()]
    ind_train = x.iloc[train][gapped].isna().to_numpy(dtype=float)
    ind_valid = x.iloc[valid][gapped].isna().to_numpy(dtype=float)
    classes = np.unique(y)
    without = log_loss(y[valid], gbdt().fit(filled_train, y[train]).predict_proba(filled_valid), labels=classes)
    with_ = log_loss(y[valid], gbdt().fit(np.hstack([filled_train, ind_train]), y[train])
                     .predict_proba(np.hstack([filled_valid, ind_valid])), labels=classes)
    probe = log_loss(y[valid], LogisticRegression(max_iter=5000).fit(ind_train, y[train]).predict_proba(ind_valid), labels=classes)
    prior = np.bincount(y[train], minlength=len(classes)) / len(train)
    base = log_loss(y[valid], np.tile(prior, (len(valid), 1)), labels=classes)
    return {"without": without, "with": with_, "drop": (without - with_) / without, "probe_drop": (base - probe) / base}


def g2(jobs: int) -> None:
    from joblib import Parallel, delayed  # type: ignore[import-untyped]

    (STATE / "g2").mkdir(parents=True, exist_ok=True)
    todo = [(v, s, f) for v in CANDIDATES for s in SEEDS for f in range(1, FOLDS + 1)
            if not (STATE / "g2" / f"{v}_s{s}_f{f}.json").exists()]
    print(f"{len(todo)} of {len(CANDIDATES) * len(SEEDS) * FOLDS} cells to run", flush=True)

    def one(v: str, s: int, f: int) -> None:
        (STATE / "g2" / f"{v}_s{s}_f{f}.json").write_text(json.dumps(g2_cell(v, s, f)))

    Parallel(n_jobs=jobs, verbose=5)(delayed(one)(*t) for t in todo)


def tally() -> None:
    passed: dict[str, bool] = {}
    missing = 0
    print(f"## G2: relative drop in validation log loss from the indicators, 15 cells (seeds {SEEDS} x {FOLDS} folds); "
          f"informative if the mean is >= {THRESHOLD} with the 95% t interval above 0")
    for v in CANDIDATES:
        cells = [STATE / "g2" / f"{v}_s{s}_f{f}.json" for s in SEEDS for f in range(1, FOLDS + 1)]
        if not all(c.exists() for c in cells):
            missing += sum(not c.exists() for c in cells)
            continue
        read = [json.loads(c.read_text()) for c in cells]
        drop = np.asarray([r["drop"] for r in read])
        mean, low, high = interval(drop)
        seeds = [float(drop[i * FOLDS:(i + 1) * FOLDS].mean()) for i in range(len(SEEDS))]
        passed[v] = mean >= THRESHOLD and low > 0
        probe = float(np.mean([r["probe_drop"] for r in read]))
        print(f"- {v}: {mean:+.4f} [{low:+.4f}, {high:+.4f}] seeds {', '.join(f'{x:+.4f}' for x in seeds)}; "
              f"probe {probe:+.4f} -> {'informative' if passed[v] else 'not informative'}")
    if missing:
        print(f"\n- missing: {missing} cells\n\nPrimary: incomplete")
        return
    controls = [v for v in F5 if not passed[v]]
    if controls:
        print(f"\nPrimary (G2): invalid, positive controls failing: {', '.join(controls)} -> stop")
        return
    p = [v for v in F1 + F3 if passed[v] and ("_outcome" not in v or "outcome3p0" in v)]
    print(f"\nPrimary (G2): valid (F5 {len(F5)} of {len(F5)}); P = {len(p)} variants: {', '.join(p) or 'none'}")
    print(f"- F2 (dial, descriptive): {', '.join(f'{v} {passed[v]}' for v in F2)}")


if __name__ == "__main__":
    STATE.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:2] == ["--drivers"]:
        drivers()
    elif sys.argv[1:2] == ["--g2"]:
        g2(int(sys.argv[2]) if len(sys.argv) > 2 else 24)
    elif sys.argv[1:] == ["--tally"]:
        tally()
    else:
        sys.exit(__doc__)
