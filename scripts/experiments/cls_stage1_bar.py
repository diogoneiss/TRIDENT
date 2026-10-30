"""Classification step zero, Stage 1: the CPU bar on the sweep's folds (docs/plans/2026-10-23-classification-step-zero.md).

    uv run --python 3.11 python -B scripts/experiments/cls_stage1_bar.py [--state <dir>] run [<variant> ...]
    uv run --python 3.11 python -B scripts/experiments/cls_stage1_bar.py [--state <dir>] --tally

``run`` fits the bar on each fold of ``build_folds(frame, 'class', 3, 42)``, the sweep's folds, and
writes one CSV per variant to ``<state>/csv/<variant>_s42.csv`` (one row per fold and arm), skipping
a variant already written. The CSV is written once, after the last fold, through a temporary name,
so a killed run leaves nothing a rerun would skip. Every arm trains on the training split alone,
and nothing is written to the store.

- **Native NaN** (``native/hgb``, ``native/lgbm``): HistGradientBoosting and LightGBM classifiers,
  untuned. Numerical columns as they are, categorical ones as their index in the training split's
  vocabulary, declared categorical; NaN for a gap or a category the training split never shows.
- **Impute, then learn** (``<imputer>[+ind]/<learner>``): three imputers, each fit on the training
  split and applied to the training, validation and test splits, then HGB, LightGBM and
  RandomForest on the filled matrix (categories as vocabulary codes: declared to the boosters,
  ordinal to the forest). ``+ind`` appends one ``isna()`` indicator per column with a gap in the
  training split.
  - ``mode_lgbm``: categorical gaps take the training mode, then the ``lgbm`` baseline imputer
    (``LightGbmImputer``) fills the numerical gaps, the paper's protocol approximately. On a table
    with no numerical column (kr-vs-kp) it is the mode fill alone, the same as ``mean_mode``.
  - ``mean_mode``: the training mean or mode (``MeanModeImputer``).
  - ``missforest``: ``MissForestImputer``, ADR 0016's missForest.

Every learner has ``class_weight='balanced'`` and ``random_state`` ``SEED``. The boosters stop early
on the fold's validation split, at most 100 rounds (both libraries' default): HGB's default
``n_iter_no_change`` 10, and LightGBM's ``stopping_rounds`` set to the same 10 (it has no default
early stopping); the validation loss is weighted by the training split's balanced class weights, as
TRIDENT's validation CE is; the forest has no early stopping. Test is scored with
``build_fold_result``. ``THREADS`` caps every pool at 4 (the imputer modules' hard-coded ``THREADS``
included); a forest predicts on one thread, as ``missforest.py`` explains.

``--tally`` prints, per level, each arm's paired (arm − TRIDENT) test macro-F1 against two TRIDENT
references on the same folds: the old sweep (``cls_determinism_run.sweep_run``), which the plan
names, and E71's arm A at seed 42 (``cls-stage2-2026-10-24``), the sweep's configuration rerun at
HEAD on gorgona8 (Stage 0's Outcome keeps the old sweep a descriptive reference only). One seed,
three folds a variant: the intervals are descriptive, and no call is made. Then each arm's mean
macro-F1 by level, and the indicator contrast (with − without), a sanity control under MCAR, not
evidence of informative missingness.
"""

from __future__ import annotations

import os

THREADS = 4
# Before numpy, scikit-learn and LightGBM load: HistGradientBoosting's OpenMP pool reads it once.
os.environ["OMP_NUM_THREADS"] = str(THREADS)

import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import report_common as common  # noqa: E402

E71 = "cls-stage2-2026-10-24"
STATE = Path("/scratch2/diogoneiss/trident-2026-10-29-cls-stage1")
BASES = ("biodeg", "credit-g", "electricity", "kc2", "kr-vs-kp", "letter", "pendigits", "spambase", "vehicle")
LEVELS = ("00", "20", "40", "60", "80")
VARIANTS = tuple(f"{b}_{lv}nan" for b in BASES for lv in LEVELS)
FOLDS = 3
SEED = 42
STOPPING_ROUNDS = 10
IMPUTERS = ("mode_lgbm", "mean_mode", "missforest")
LEARNERS = ("hgb", "lgbm", "rf")
ARMS = ("native/hgb", "native/lgbm") + tuple(
    f"{imputer}{suffix}/{learner}" for imputer in IMPUTERS for suffix in ("", "+ind") for learner in LEARNERS
)

Matrix = NDArray[np.float64]
Labels = NDArray[np.int64]


def _out(state: Path, variant: str) -> Path:
    return state / "csv" / f"{variant}_s{SEED}.csv"


def _codes(frame: pd.DataFrame, categorical: list[str], vocabulary: dict[str, list[str]]) -> Matrix:
    """Numerical columns as they are, categorical ones as their training-vocabulary index; NaN for a
    gap or an unseen category. Never the strings themselves: ``as_category_strings`` renders a gap
    as the category ``'nan'``, which the vocabulary leaves out."""
    from src.embedder import as_category_strings

    columns = []
    for name in frame.columns:
        if name in categorical:
            index = {label: float(k) for k, label in enumerate(vocabulary[name])}
            columns.append(np.array([index.get(label, np.nan) for label in as_category_strings(frame[name])], dtype=float))
        else:
            columns.append(frame[name].to_numpy(dtype=float))
    return np.column_stack(columns)


def _balanced(y_train: Labels, y: Labels) -> NDArray[np.float64]:
    """Each row of ``y`` weighted as ``class_weight='balanced'`` weighs the training split's classes."""
    classes, counts = np.unique(y_train, return_counts=True)
    weight = dict(zip(classes.tolist(), (len(y_train) / (len(classes) * counts)).tolist()))
    return np.array([weight[label] for label in y.tolist()], dtype=float)


def _learn(learner: str, x: tuple[Matrix, Matrix, Matrix], y: tuple[Labels, Labels], declared: list[int]) -> NDArray[Any]:
    """Fit ``learner`` on the training split, stopping on the validation split; predict the test split."""
    import lightgbm
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

    x_train, x_val, x_test = x
    y_train, y_val = y
    if learner == "hgb":
        mask = np.zeros(x_train.shape[1], dtype=bool)
        mask[declared] = True
        hgb = HistGradientBoostingClassifier(
            categorical_features=mask if mask.any() else None, early_stopping=True,
            n_iter_no_change=STOPPING_ROUNDS, class_weight="balanced", random_state=SEED)
        hgb.fit(x_train, y_train, X_val=x_val, y_val=y_val, sample_weight_val=_balanced(y_train, y_val))
        return hgb.predict(x_test)
    if learner == "lgbm":
        booster = lightgbm.LGBMClassifier(class_weight="balanced", random_state=SEED, n_jobs=THREADS,
                                          deterministic=True, force_row_wise=True, verbose=-1)
        booster.fit(x_train, y_train, categorical_feature=declared or "auto", eval_set=[(x_val, y_val)],
                    eval_sample_weight=[_balanced(y_train, y_val)],
                    callbacks=[lightgbm.early_stopping(STOPPING_ROUNDS, verbose=False)])
        return np.asarray(booster.predict(x_test))
    forest = RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=THREADS)
    forest.fit(x_train, y_train)
    # Several threads sum the trees in whatever order they finish, which can flip a near tie.
    forest.n_jobs = 1
    return forest.predict(x_test)


def _impute(imputer: str, splits: list[pd.DataFrame], hidden: list[NDArray[np.bool_]],
            numerical: list[str], categorical: list[str]) -> tuple[list[pd.DataFrame], int]:
    """The three splits filled by ``imputer``, fit on the training split; and missForest's rounds."""
    from src.training.imputation_baselines import MeanModeImputer
    from src.training.lightgbm_baseline import LightGbmImputer
    from src.training.missforest import MissForestImputer

    mode = MeanModeImputer(numerical, categorical)
    mode.fit(splits[0])
    if imputer == "mean_mode":
        return [mode.impute(f, h) for f, h in zip(splits, hidden)], 0
    if imputer == "missforest":
        forest = MissForestImputer(numerical, categorical)
        forest.fit(splits[0])
        return [forest.impute(f, h) for f, h in zip(splits, hidden)], forest.rounds
    # mode_lgbm: the mode fill on the categorical columns first, so the numerical models read
    # filled categories; then LightGBM on the numerical gaps alone, the mode fill left as it is.
    is_numerical = np.isin(np.asarray(splits[0].columns), numerical)
    moded = []
    for frame, h in zip(splits, hidden):
        filled = mode.impute(frame, h)
        filled[numerical] = frame[numerical]  # the mode imputer fills every gap; keep the numerical ones open
        moded.append(filled)
    if not numerical:
        return moded, 0
    boosted = LightGbmImputer(numerical, categorical)
    boosted.fit(moded[0])
    return [boosted.impute(f, h & is_numerical[None, :]) for f, h in zip(moded, hidden)], 0


def run_variant(variant: str, state: Path) -> None:
    from src.embedder import as_category_strings
    from src.training import lightgbm_baseline, missforest
    from src.training.data import build_folds, prepare_dataset
    from src.training.finetuning import build_fold_result
    from src.training.imputation_baselines import _NOT_A_CLASS
    from src.training.types import DatasetSpec

    out = _out(state, variant)
    if out.exists():
        return
    # Hard-coded at 8 in src/; both modules read them at call time.
    lightgbm_baseline.THREADS = THREADS
    missforest.THREADS = THREADS
    warnings.filterwarnings("ignore")
    started = time.time()
    dataset = prepare_dataset(DatasetSpec.from_name(variant, None))
    label = dataset.label_column
    features = dataset.frame.drop(columns=[label])
    categorical = list(dataset.categorical_columns)
    numerical = list(dataset.numerical_columns)
    declared = [list(features.columns).index(c) for c in categorical]
    labels = dataset.frame[label].to_numpy().astype(np.int64)
    rows: list[dict[str, Any]] = []
    for ordinal, fold in enumerate(build_folds(dataset.frame, label, FOLDS, SEED), start=1):
        parts = [np.asarray(fold.train_indices), np.asarray(fold.validation_indices), np.asarray(fold.test_indices)]
        splits = [features.iloc[p].reset_index(drop=True) for p in parts]
        y_train, y_val, y_test = (labels[p] for p in parts)
        vocabulary = {c: sorted(set(as_category_strings(splits[0][c]).tolist()) - _NOT_A_CLASS) for c in categorical}
        native = [_codes(s, categorical, vocabulary) for s in splits]
        # A gap, or a category the training split never shows: what every imputer fills.
        hidden = [np.isnan(m) for m in native]
        empty = [int(h.all(axis=1).sum()) for h in hidden]
        print(f"{variant} fold {ordinal}: rows {[len(s) for s in splits]}, every feature missing {empty}", flush=True)

        def score(arm: str, predicted: NDArray[Any], seconds: float, impute_seconds: float, rounds: int) -> None:
            metrics = build_fold_result(ordinal, variant, y_test, predicted).metrics
            rows.append({"variant": variant, "seed": SEED, "fold": ordinal, "arm": arm,
                         "f1_macro": float(metrics["f1_macro"]), "f1_micro": float(metrics["f1_micro"]),
                         "accuracy": float(metrics["accuracy"]), "seconds": round(seconds, 3),
                         "impute_seconds": round(impute_seconds, 3), "missforest_rounds": rounds})

        for learner in ("hgb", "lgbm"):
            clock = time.perf_counter()
            predicted = _learn(learner, (native[0], native[1], native[2]), (y_train, y_val), declared)
            score(f"native/{learner}", predicted, time.perf_counter() - clock, 0.0, 0)
        gapped = [j for j in range(len(features.columns)) if hidden[0][:, j].any()]
        for imputer in IMPUTERS:
            clock = time.perf_counter()
            if any(h.any() for h in hidden):
                filled, rounds = _impute(imputer, splits, hidden, numerical, categorical)
            else:
                filled, rounds = splits, 0  # nothing to fill: every imputer is the identity
            impute_seconds = time.perf_counter() - clock
            plain = [_codes(f, categorical, vocabulary) for f in filled]
            assert not any(np.isnan(m).any() for m in plain), f"{imputer} left a gap in {variant} fold {ordinal}"
            for suffix in ("", "+ind"):
                x = plain if not suffix else [np.hstack([m, h[:, gapped].astype(float)]) for m, h in zip(plain, hidden)]
                for learner in LEARNERS:
                    clock = time.perf_counter()
                    predicted = _learn(learner, (x[0], x[1], x[2]), (y_train, y_val), declared)
                    score(f"{imputer}{suffix}/{learner}", predicted, time.perf_counter() - clock, impute_seconds, rounds)
            print(f"  {imputer}: {impute_seconds:.1f} s to impute" + (f", {rounds} rounds" if imputer == "missforest" else ""), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial")
    pd.DataFrame(rows).to_csv(partial, index=False)
    partial.rename(out)
    table = pd.DataFrame(rows).groupby("arm", sort=False)["f1_macro"].mean()
    print(f"DONE {variant}: {time.time() - started:.0f} s; " + "; ".join(f"{a} {v:.3f}" for a, v in table.items()), flush=True)


def _references() -> dict[str, dict[str, NDArray[np.float64]]]:
    """Per reference, each variant's TRIDENT test macro-F1, fold by fold, on the same folds."""
    from scripts.experiments.cls_determinism_run import sweep_run

    sweep: dict[str, NDArray[np.float64]] = {}
    rerun: dict[str, NDArray[np.float64]] = {}
    e71 = common.parents(E71, "A")
    for variant in VARIANTS:
        _, artifacts = sweep_run(variant)
        frame = pd.read_csv(artifacts / "metrics" / "raw_fold_metrics.csv").sort_values("fold")
        sweep[variant] = frame["f1_macro"].to_numpy(dtype=float)
        runs = e71.get((variant, SEED))
        if runs:
            rerun[variant] = common.fold_table(runs[0]).sort_values("fold")["f1_macro"].to_numpy(dtype=float)
    return {"sweep": sweep, "E71 A": rerun}


def _interval(diffs: list[NDArray[np.float64]]) -> str:
    v = common.verdict([np.concatenate(diffs)])
    return f"{v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}]"


def tally(state: Path) -> None:
    found = {v: pd.read_csv(_out(state, v)) for v in VARIANTS if _out(state, v).exists()}
    missing = [v for v in VARIANTS if v not in found]
    if missing:
        print(f"incomplete: {len(missing)} of {len(VARIANTS)} variants missing, e.g. {', '.join(missing[:5])}")
    if not found:
        return
    references = _references()

    def folds(variant: str, arm: str) -> NDArray[np.float64]:
        frame = found[variant]
        return frame[frame.arm == arm].sort_values("fold")["f1_macro"].to_numpy(dtype=float)

    print("Descriptive only: one seed, three folds a variant; intervals are the 95% t over fold pairs, no call.\n")
    for level in LEVELS:
        variants = [f"{b}_{level}nan" for b in BASES if f"{b}_{level}nan" in found]
        if not variants:
            continue
        print(f"## {level}nan ({len(variants)} of {len(BASES)} tables)")
        for name, reference in references.items():
            paired = [v for v in variants if v in reference]
            mean_t = float(np.mean([reference[v].mean() for v in paired])) if paired else float("nan")
            print(f"- TRIDENT ({name}) mean macro-F1 {mean_t:.4f} over {len(paired)} tables")
        print("- arm − TRIDENT, test macro-F1 (sweep | E71 A s42), and the arm's mean macro-F1:")
        for arm in ARMS:
            parts = []
            for name, reference in references.items():
                paired = [v for v in variants if v in reference]
                parts.append(_interval([folds(v, arm) - reference[v] for v in paired]) if paired else "n/a")
            mean_arm = float(np.mean([folds(v, arm).mean() for v in variants]))
            print(f"  - {arm:<22} {parts[0]} | {parts[1]}; mean {mean_arm:.4f}")
        print("- indicator contrast (with − without), the MCAR sanity control:")
        for imputer in IMPUTERS:
            cells = [f"{learner} {_interval([folds(v, f'{imputer}+ind/{learner}') - folds(v, f'{imputer}/{learner}') for v in variants])}"
                     for learner in LEARNERS]
            print(f"  - {imputer}: " + "; ".join(cells))
        print()


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--state"] and len(argv) >= 2:
        STATE = Path(argv[1])
        argv = argv[2:]
    if argv[:1] == ["run"]:
        unknown = [v for v in argv[1:] if v not in VARIANTS]
        if unknown:
            sys.exit(f"unknown variants: {', '.join(unknown)}")
        for name in argv[1:] or list(VARIANTS):
            run_variant(name, STATE)
    elif argv == ["--tally"]:
        tally(STATE)
    else:
        sys.exit(__doc__)
