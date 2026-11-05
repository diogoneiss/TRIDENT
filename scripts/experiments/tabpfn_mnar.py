"""TabPFN v2 under informative missingness (T44, E137): two offline arms beside C15's A1 on E124's folds.

    uv run --python 3.11 --extra tabpfn python -B scripts/experiments/tabpfn_mnar.py run [<variant> <seed> ...]
    uv run --python 3.11 --extra tabpfn python -B scripts/experiments/tabpfn_mnar.py --todo <n> <queues>
    uv run --python 3.11 python -B scripts/experiments/tabpfn_mnar.py --tally

One fit a fold and arm, the fold's training split as context (E83's convention; A1 used validation for selection
only), the test split predicted in chunks:
  native  the matrix as E83 (raw NaNs, a missing categorical as NaN);
  mask    the same matrix with one 0/1 missingness column appended per feature, so the in-context learner can use
          the pattern as a predictor.
Each arm predicts twice from the same fit: ``balance_probabilities`` on (E92's outcome: the fairer strong tier
against a class-weighted TRIDENT) is the primary, plain beside. A pair writes ``<STATE>/<variant>_s<seed>.pt``: per
fold the four probability arrays over the label ids, the test indices (asserted equal to A1's readout) and the
context size. The tally pairs them with A1's readouts through ``c15_report.scores``. Pairs run on the machine that
holds their A1 cell (ADR 0035, ``c15_run.machine``). Nothing is trained and the store is untouched.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.experiments import c15_report as c15  # noqa: E402
from scripts.experiments import c15_run as run  # noqa: E402
from scripts.experiments.tabpfn_classification import CHUNK, _matrix  # noqa: E402

STATE = Path(os.environ.get("TABPFN_MNAR_STATE") or
             (Path("/scratch2/diogoneiss" if run.HOST == "gorgona8" else "/scratch/diogoneiss") / "trident-2026-11-04-tabpfn-mnar"))
VARIANTS = run.DEVELOPMENT  # F0 (MCAR controls), F1 (probit MNAR), F2 (probit scales), F3 (outcome-dependent)
ALL = VARIANTS + run.SIBLINGS_2060  # E139 adds F3's same-seed MCAR siblings (c15_run stage h2 trains A1 on them)
SEEDS = run.SEEDS
ARMS = ("native", "mask")
MARGIN = 0.01


def out_path(variant: str, seed: int) -> Path:
    return STATE / f"{variant}_s{seed}.pt"


def _probabilities(classifier: Any, x: np.ndarray, classes: int) -> np.ndarray:
    """The class probabilities over every label id, zero for a class the context never showed."""
    raw = np.concatenate([classifier.predict_proba(x[i:i + CHUNK]) for i in range(0, len(x), CHUNK)])
    p = np.zeros((len(x), classes))
    p[:, np.asarray(classifier.classes_, dtype=int)] = raw
    return np.asarray(p / p.sum(axis=1, keepdims=True))  # float32 rows sum to 1 ± 1e-7; log_loss wants exactly 1


def run_pair(variant: str, seed: int) -> None:
    import importlib

    from sklearn.model_selection import train_test_split

    from src.embedder import as_category_strings
    from src.training.data import build_folds, prepare_dataset
    from src.training.tabpfn_baseline import CONTEXT_ROWS, N_ESTIMATORS, TABPFN_SEED, _NOT_A_CLASS
    from src.training.types import DatasetSpec

    if out_path(variant, seed).exists():
        return
    if not torch.cuda.is_available():  # a CPU fit differs fold by fold from a GPU one (E86, E92)
        sys.exit(f"no CUDA device: {variant} s{seed} not run")
    a1 = c15.readouts("A1", variant, seed)
    if a1 is None:
        sys.exit(f"A1 readouts of {variant} s{seed} are not on this machine")
    warnings.filterwarnings("ignore")
    started = time.time()
    tabpfn = importlib.import_module("tabpfn")
    version = importlib.import_module("tabpfn.constants").ModelVersion.V2
    dataset = prepare_dataset(DatasetSpec.from_name(variant, "class"))
    label = dataset.label_column
    features = dataset.frame.drop(columns=[label])
    categorical = list(dataset.categorical_columns)
    categorical_indices = [list(features.columns).index(c) for c in categorical] or None
    classes = len(dataset.label_classes)
    folds: list[dict[str, Any]] = []
    for k, fold in enumerate(build_folds(dataset.frame, label, 5, seed)):
        assert np.array_equal(np.asarray(fold.test_indices), a1[k]["test_indices"]), f"fold {k + 1} differs from A1"
        train = features.iloc[np.asarray(fold.train_indices)].reset_index(drop=True)
        test = features.iloc[np.asarray(fold.test_indices)].reset_index(drop=True)
        y_train = dataset.frame[label].to_numpy()[np.asarray(fold.train_indices)]
        vocabulary = {c: sorted(set(as_category_strings(train[c]).tolist()) - _NOT_A_CLASS) for c in categorical}
        x_train, x_test = _matrix(train, categorical, vocabulary), _matrix(test, categorical, vocabulary)
        if len(x_train) > CONTEXT_ROWS:
            keep, _ = train_test_split(np.arange(len(x_train)), train_size=CONTEXT_ROWS, stratify=y_train, random_state=TABPFN_SEED)
            x_train, y_train = x_train[np.sort(keep)], y_train[np.sort(keep)]
        record: dict[str, Any] = {"test_indices": np.asarray(fold.test_indices), "context_rows": len(x_train)}
        for arm in ARMS:
            a_train, a_test = x_train, x_test
            if arm == "mask":
                a_train = np.column_stack([x_train, np.isnan(x_train).astype(float)])
                a_test = np.column_stack([x_test, np.isnan(x_test).astype(float)])
            classifier = tabpfn.TabPFNClassifier.create_default_for_version(
                version, n_estimators=N_ESTIMATORS, random_state=TABPFN_SEED,
                categorical_features_indices=categorical_indices, device="cuda")
            numpy_state = np.random.get_state()
            try:
                with torch.random.fork_rng(devices=[0]):
                    classifier.fit(a_train, y_train)
                    record[f"{arm}_plain"] = _probabilities(classifier, a_test, classes)
                classifier.balance_probabilities = True
                with torch.random.fork_rng(devices=[0]):
                    record[f"{arm}_balanced"] = _probabilities(classifier, a_test, classes)
            finally:
                np.random.set_state(numpy_state)
        folds.append(record)
    STATE.mkdir(parents=True, exist_ok=True)
    torch.save(folds, out_path(variant, seed))
    print(f"DONE {variant} s{seed}: {time.time() - started:.0f} s", flush=True)


def todo(which: int, queues: int) -> None:
    mine = [(v, s) for s in SEEDS for v in ALL if run.machine((v, s)) == run.HOST and not out_path(v, s).exists()]
    for k, (variant, seed) in enumerate(mine):
        if k % queues == which - 1:
            print(variant, seed)


def _diff(arm: str, variant: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Per replicate (seed-major, then fold): macro-F1(arm) − macro-F1(A1) and log loss(A1) − log loss(arm)."""
    f1, ll = [], []
    for seed in SEEDS:
        a1 = c15.readouts("A1", variant, seed)
        if a1 is None or not out_path(variant, seed).exists():
            return None
        folds = torch.load(out_path(variant, seed), weights_only=False)
        fa, la = c15.scores(a1)
        fx, lx = c15.scores(a1, [f[arm] for f in folds])
        f1.append(fx - fa)
        ll.append(la - lx)
    return np.concatenate(f1), np.concatenate(ll)


def _line(d: np.ndarray) -> str:
    m, lo, hi = c15.interval(d, False)
    _, nlo, nhi = c15.interval(d)
    return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}] (NB [{nlo:+.4f}, {nhi:+.4f}])"


def _interaction_verdict(d: np.ndarray) -> str:
    m, lo, hi = c15.interval(d, False)
    if hi < 0:
        return "lead SHRINKS under MNAR"
    if lo > 0:
        return "lead GROWS under MNAR"
    if lo > -MARGIN and hi < MARGIN:
        return "lead holds (inside ±0.01)"
    return "no verdict"


def tally() -> None:
    print("## TabPFN v2 under informative missingness (E137, T44): arm − A1 in test macro-F1, 15 paired replicates a file; "
          "primary the interaction (arm − A1) MNAR − MCAR over the seven probit/MCAR sibling pairs, balanced "
          "probabilities, Holm over the two arms; plain beside")
    diffs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for arm in ARMS:
        for mode in ("balanced", "plain"):
            for v in VARIANTS:
                d = _diff(f"{arm}_{mode}", v)
                if d is None:
                    print(f"- {arm} {mode} {v}: missing\n\nTally: incomplete")
                    return
                diffs[(f"{arm}_{mode}", v)] = d
    for mode in ("balanced", "plain"):
        print(f"\n### {mode}{' (primary)' if mode == 'balanced' else ' (beside)'}")
        inter: dict[str, np.ndarray] = {}
        for arm in ARMS:
            key = f"{arm}_{mode}"
            print(f"\n#### {arm} − A1")
            for group, files in (("MCAR controls (F0)", run.F0), ("probit MNAR (F1)", run.F1), ("probit scales (F2)", run.F2),
                                 ("outcome-dependent (F3)", run.F3)):
                for v in files:
                    f1, ll = diffs[(key, v)]
                    print(f"- {v}: F1 {_line(f1)}; LL(A1) − LL({arm}) {c15.interval(ll, False)[0]:+.4f}")
                pooled = np.mean([diffs[(key, v)][0] for v in files], axis=0)
                print(f"  pooled {group}: {_line(pooled)}")
            inter[arm] = np.mean([diffs[(key, v)][0] - diffs[(key, v.replace('_probit_', '_'))][0] for v in run.F1], axis=0)
        p = {arm: float(stats.ttest_1samp(inter[arm], 0.0).pvalue) for arm in ARMS}
        order = sorted(p, key=lambda k: p[k])
        holm = {a: min(1.0, max(p[g] * (2 - j) for j, g in enumerate(order[:k + 1]))) for k, a in enumerate(order)}
        for arm in ARMS:
            print(f"\nInteraction {arm} ({mode}): (arm − A1) MNAR − MCAR, 7 pairs pooled: {_line(inter[arm])}; "
                  f"p {p[arm]:.4f}, Holm {holm[arm]:.4f} -> {_interaction_verdict(inter[arm])}")
        mnar = run.F1 + run.F2 + run.F3
        gap = np.mean([diffs[(f"mask_{mode}", v)][0] - diffs[(f"native_{mode}", v)][0] for v in mnar], axis=0)
        print(f"Mask − native ({mode}) on the {len(mnar)} MNAR files pooled: {_line(gap)}")


def tally_h2() -> None:
    """E139 (ticket 09): the interaction (arm − A1) F3 − same-seed MCAR sibling, 6 outcome files pooled."""
    print("## H2 under outcome-dependent MNAR (E139): (arm − A1) on each F3 file minus (arm − A1) on its same-seed MCAR "
          "sibling (s2060), paired per (table, seed, fold), pooled over the six F3 files; balanced primary, plain beside; "
          "Holm over the two arms")
    sibling = {v: f"{v.split('_')[0]}_mcar_s2060_60nan" for v in run.F3}
    for mode in ("balanced", "plain"):
        print(f"\n### {mode}{' (primary)' if mode == 'balanced' else ' (beside)'}")
        inter: dict[str, np.ndarray] = {}
        for arm in ARMS:
            key = f"{arm}_{mode}"
            parts = []
            for v in run.F3:
                d, ds = _diff(key, v), _diff(key, sibling[v])
                if d is None or ds is None:
                    print(f"- {arm} {v}: missing\n\nTally: incomplete")
                    return
                parts.append(d[0] - ds[0])
                print(f"- {arm}: {v} − sibling: {_line(d[0] - ds[0])}; arm − A1 on F3 {c15.interval(d[0], False)[0]:+.4f}, "
                      f"on the sibling {c15.interval(ds[0], False)[0]:+.4f}")
            inter[arm] = np.mean(parts, axis=0)
        p = {arm: float(stats.ttest_1samp(inter[arm], 0.0).pvalue) for arm in ARMS}
        order = sorted(p, key=lambda k: p[k])
        holm = {a: min(1.0, max(p[g] * (2 - j) for j, g in enumerate(order[:k + 1]))) for k, a in enumerate(order)}
        for arm in ARMS:
            print(f"Interaction {arm} ({mode}), 6 F3 files pooled: {_line(inter[arm])}; p {p[arm]:.4f}, Holm {holm[arm]:.4f} "
                  f"-> {_interaction_verdict(inter[arm])}")
    print("\n### Descriptive: each model on F3 minus on its sibling (macro-F1 of the model itself, A1 then TabPFN native balanced)")
    for v in run.F3:
        own = []
        for s in SEEDS:
            r, rs = c15.readouts("A1", v, s), c15.readouts("A1", sibling[v], s)
            if r is None or rs is None:
                break
            fa, fs = c15.scores(r)[0], c15.scores(rs)[0]
            ft = c15.scores(r, [f["native_balanced"] for f in torch.load(out_path(v, s), weights_only=False)])[0]
            fts = c15.scores(rs, [f["native_balanced"] for f in torch.load(out_path(sibling[v], s), weights_only=False)])[0]
            own.append((fa - fs, ft - fts))
        if len(own) == len(SEEDS):
            a = np.concatenate([x for x, _ in own]); t = np.concatenate([y for _, y in own])
            print(f"- {v}: A1 {c15.interval(a, False)[0]:+.4f}; TabPFN {c15.interval(t, False)[0]:+.4f}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["run"]:
        for variant, seed in list(zip(argv[1::2], map(int, argv[2::2]))) or [(v, s) for s in SEEDS for v in ALL if run.machine((v, s)) == run.HOST]:
            run_pair(variant, seed)
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    elif argv == ["--tally"]:
        tally()
    elif argv == ["--tally-h2"]:
        tally_h2()
    else:
        sys.exit(__doc__)
