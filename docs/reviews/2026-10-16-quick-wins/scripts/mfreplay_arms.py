"""Probe: replay vs sequential test-time fill for the fork's real missForest (forests).

Throwaway, read-only on the TRIDENT repo. Run from the repo root as cwd with the fork's
interpreter, PYTHONDONTWRITEBYTECODE=1 and -B. Writes only to the scratchpad (JSONL per fold).
Never touches MLflow, never passes a cache dir, never constructs BaselineScorer.

PRE-REGISTERED (fixed in this file before any score of this probe was looked at):

Arms, all from ONE fit per fold of MissForestSeqImputer (a MissForestImputer subclass whose
fit is the stock fit: same rounds, same models, same random_state, same stopping rule; the
subclass only observes it through two hooks, _fit_one and _change):
  missforest      the stock transform, MissForestImputer.impute inherited unchanged: test
                  rows start at the training means/modes and the KEPT round's models are
                  replayed, up to MAX_ROUNDS, under still_converging ("replay").
  missforest_seq  sklearn IterativeImputer.transform semantics: test rows start at the
                  training means/modes, then round 1's models, round 2's, ..., round k's
                  (k = the fit's kept rounds, self.rounds), columns in the fit's order
                  (_order), Gauss-Seidel within a round (each column's prediction is written
                  before the next column is predicted), no stopping rule ("sequential").
  mean_mode       MeanModeImputer, beside them (its impute_score is 1.0 by construction).

How the sequential fill is computed: every round's model is applied to the test matrices
the moment it is fitted (inside _fit_one, after super()), which is the same arithmetic as
recording all rounds' models and applying them afterwards in order, without holding
k rounds of forests in memory (electricity/pendigits on a 16 GB machine). The test
matrices are the two the fold will be scored on (masked: evaluation-mask cells plus the
natural gaps; induced: the natural gaps), plus the TRAINING frame itself as a self-check.
The snapshot after round k is the sequential fill.

Self-checks (every fold):
  S1  the sequential fill of the training frame equals, bitwise, the fit's own kept fills
      (captured from _change during fit). Proves start, column order and within-round
      semantics equal the fit's.
Fold-1 checks (mode "check"):
  S2  a separate stock MissForestImputer fitted on the same training fold gives the same
      rounds, and its impute equals the subclass's replay arm exactly
      (pandas assert_frame_equal, check_exact) on both populations' hidden masks.
  S3  (keep_all, not electricity) every round's fitted models are recorded; the kept round's
      models are the same objects as self._models; a recorded-models sequential impute
      (apply round 1..k models after fit) equals the on-the-fly sequential fill exactly.

Thread note: src.training.missforest.THREADS is patched at runtime (module attribute, file
untouched) to MFREPLAY_THREADS when parallel processes share the 12 cores; the module and
ADR 0017 decision 6 state the trees do not depend on it, and prediction is one thread.

Reading rule: tie band 0.003 of impute_score (as the earlier probe). "Sequential beats the
decoder" only when seq < K - 0.003; "replay handicaps missForest" only when
seq < replay - 0.003 on the variant mean. Seed-42 comparisons use seed-42 K; 3-seed claims
only where 3 seeds ran.
"""

import ctypes
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import src.training.missforest as mfmod  # noqa: E402
from scripts.backfill_baseline_imputers import recompute_folds  # noqa: E402
from src.embedder import as_category_strings  # noqa: E402
from src.training.imputation_baselines import MeanModeImputer, _position, score_baselines  # noqa: E402
from src.training.missforest import MissForestImputer  # noqa: E402

if os.environ.get("MFREPLAY_THREADS"):
    mfmod.THREADS = int(os.environ["MFREPLAY_THREADS"])

TIMES: dict[str, float] = {}
INFO: dict[str, object] = {}


def _peak_rss_mb() -> float | None:
    try:
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        fn = ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
        fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return round(pmc.PeakWorkingSetSize / 2**20, 1)
    except Exception:
        return None


def _missing_of(frame: pd.DataFrame, cells: pd.DataFrame) -> np.ndarray:
    """The effective missing mask score_baselines hands an imputer: hidden | natural."""
    hidden = np.zeros(frame.shape, dtype=bool)
    if not cells.empty:
        rows = cells["row"].to_numpy(dtype=int)
        columns = frame.columns.get_indexer(pd.Index(cells["column"]))
        hidden[rows, columns] = True
    return hidden | frame.isna().to_numpy()


class _Target:
    """A frame to be filled sequentially during the fit."""

    def __init__(self, label, frame, missing):
        self.label, self.frame, self.missing = label, frame, missing
        self.fills = None
        self.gaps = None
        self.snapshots: list[np.ndarray] = []
        self.final: np.ndarray | None = None


class MissForestSeqImputer(MissForestImputer):
    """The stock fit, observed; the stock transform inherited (the 'missforest' arm)."""

    name = "missforest"

    def __init__(self, numerical, categorical, keep_all=False):
        super().__init__(numerical, categorical)
        self.keep_all = keep_all
        self._targets: list[_Target] = []

    def set_targets(self, targets):
        self._targets = list(targets)

    def fit(self, frame):
        self._seq: list[dict] = []
        self._calls = 0
        self._fit_rounds: list[np.ndarray] = []
        self._in_fit = True
        for target in self._targets:
            target.fills, target.gaps, target.snapshots, target.final = None, None, [], None
        try:
            super().fit(frame)
        finally:
            self._in_fit = False
        fitted = self._calls // len(self._order)
        INFO["fitted_rounds"] = fitted
        INFO["kept_rounds"] = self.rounds
        k = self.rounds
        for target in self._targets:
            assert len(target.snapshots) == fitted, (len(target.snapshots), fitted)
            target.final = target.snapshots[k - 1]
            target.snapshots = []  # free memory
        self._kept_fit_fills = self._fit_rounds[k - 1]
        self._fit_rounds = []
        if self.keep_all:
            self._seq = self._seq[:k]
            assert len(self._seq) == k
            same = sum(m is self._seq[-1][i] for i, m in enumerate(self._models))
            INFO["S3_kept_models_identical"] = f"{same}/{len(self._models)}"
            assert same == len(self._models)

    def _fit_one(self, index, features, target):
        model = super()._fit_one(index, features, target)
        if self._calls == 0:
            # _classes, _start and _order are set by now; build the targets' matrices.
            for t in self._targets:
                matrix = self._matrix(t.frame, t.missing)
                t.gaps = np.isnan(matrix)
                t.fills = np.where(t.gaps, self._start, matrix)
        r = self._calls // len(self._order)
        if self.keep_all:
            if r == len(self._seq):
                self._seq.append({})
            self._seq[r][index] = model
        others = self._others(index)
        for t in self._targets:
            rows = t.gaps[:, index]
            if rows.any():
                t.fills[rows, index] = model.predict(t.fills[rows][:, others])
        self._calls += 1
        if self._calls % len(self._order) == 0:
            for t in self._targets:
                t.snapshots.append(t.fills.copy())
        return model

    def _change(self, before, after, gaps):
        if getattr(self, "_in_fit", False):
            self._fit_rounds.append(after.copy())
        return super()._change(before, after, gaps)

    # ---- sequential output -------------------------------------------------------
    def _assemble(self, frame, missing, fills):
        """The stock impute's column assembly, verbatim."""
        columns = {}
        for index, column in enumerate(self._numerical):
            numbers = frame[column].to_numpy(dtype=float).copy()
            at = missing[:, _position(frame, column)]
            numbers[at] = fills[at, index]
            columns[column] = numbers
        for offset, column in enumerate(self._categorical):
            index = len(self._numerical) + offset
            strings = as_category_strings(frame[column]).astype(object)
            at = missing[:, _position(frame, column)]
            codes = fills[at, index].astype(int)
            strings[at] = np.array(self._classes[column], dtype=object)[codes]
            columns[column] = strings
        return pd.DataFrame(columns, index=frame.index).reindex(columns=frame.columns)

    def target_fills(self, frame, missing):
        for t in self._targets:
            if t.frame is frame and np.array_equal(t.missing, missing):
                return t.final
        raise KeyError("no sequential target registered for this frame/mask")

    def impute_seq(self, frame, hidden):
        missing = hidden | frame.isna().to_numpy()
        return self._assemble(frame, missing, self.target_fills(frame, missing))

    def recorded_seq_fills(self, frame, missing):
        """S3: apply the recorded rounds' models after the fit, round 1..k."""
        matrix = self._matrix(frame, missing)
        gaps = np.isnan(matrix)
        fills = np.where(gaps, self._start, matrix)
        for models in self._seq[: self.rounds]:
            for index in self._order:
                if gaps[:, index].any():
                    fills[gaps[:, index], index] = models[index].predict(
                        fills[gaps[:, index]][:, self._others(index)]
                    )
        return fills


class _Timed:
    def __init__(self, inner, name=None, method="impute"):
        self._inner = inner
        self.name = name or inner.name
        self._method = method

    def impute(self, frame, hidden):
        t = time.time()
        out = getattr(self._inner, self._method)(frame, hidden)
        TIMES[self.name + "_impute"] = TIMES.get(self.name + "_impute", 0.0) + time.time() - t
        return out


def make_factory(check_first: bool, keep_all_first: bool):
    counter = {"n": 0}

    def factory(train, num, cat):
        counter["n"] += 1
        check = check_first and counter["n"] == 1
        keep_all = keep_all_first and counter["n"] == 1
        INFO["check"] = check
        INFO["keep_all"] = keep_all
        state = {"fitted": None, "calls": 0}
        mm = MeanModeImputer(num, cat)
        mm.fit(train)

        def score(test, cells, naive):
            state["calls"] += 1
            missing = _missing_of(test, cells)
            if state["fitted"] is None:
                natural = test.isna().to_numpy()
                imp = MissForestSeqImputer(num, cat, keep_all=keep_all)
                imp.set_targets([
                    _Target("train", train, train.isna().to_numpy()),
                    _Target("call1", test, missing),
                    _Target("natural", test, natural),
                ])
                t = time.time()
                imp.fit(train)
                TIMES["fit"] = time.time() - t
                # S1: sequential fill of the training frame == the fit's kept fills, bitwise.
                train_final = imp._targets[0].final
                INFO["S1_train_seq_equals_fit"] = bool(np.array_equal(train_final, imp._kept_fit_fills))
                INFO["S1_max_abs"] = float(np.nanmax(np.abs(train_final - imp._kept_fit_fills))) if train_final.size else 0.0
                state["fitted"] = imp
                if check:
                    t = time.time()
                    stock = MissForestImputer(num, cat)
                    stock.fit(train)
                    TIMES["stock_fit"] = time.time() - t
                    state["stock"] = stock
                    INFO["S2_rounds_equal"] = stock.rounds == imp.rounds
            imp = state["fitted"]
            tag = "masked" if state["calls"] == 1 else "induced"
            if check:
                hidden = missing & ~test.isna().to_numpy()
                a = state["stock"].impute(test, hidden)
                b = imp.impute(test, hidden)
                try:
                    pd.testing.assert_frame_equal(a, b, check_exact=True)
                    INFO[f"S2_{tag}_stock_equals_replay"] = True
                except AssertionError as error:
                    INFO[f"S2_{tag}_stock_equals_replay"] = False
                    INFO[f"S2_{tag}_err"] = str(error)[:300]
                if keep_all:
                    rec = imp.recorded_seq_fills(test, missing)
                    fly = imp.target_fills(test, missing)
                    INFO[f"S3_{tag}_recorded_equals_onthefly"] = bool(np.array_equal(rec, fly))
            arms = [
                _Timed(imp, "missforest", "impute"),
                _Timed(imp, "missforest_seq", "impute_seq"),
                _Timed(mm, "mean_mode", "impute"),
            ]
            INFO[f"rss_peak_mb_{tag}"] = _peak_rss_mb()
            return score_baselines(arms, test, cells, naive).metrics

        return score
    return factory


def main() -> None:
    seed = int(sys.argv[1])
    out_path = sys.argv[2]
    folds_arg = sys.argv[3]  # "1", "2-5" or "all"
    check = sys.argv[4] == "check"
    variants = sys.argv[5:]
    rate = float(os.environ.get("MFREPLAY_RATE", "0.2"))
    for variant in variants:
        start = time.time()
        TIMES.clear()
        INFO.clear()
        if folds_arg == "all":
            done = None
        else:
            wanted = {int(x) for x in folds_arg.split(",")}
            done = {f: {} for f in range(1, 6) if f not in wanted}
        keep_all = check and not variant.startswith("electricity")

        def on_fold(ordinal, metrics, variant=variant):
            row = {"variant": variant, "seed": seed, "fold": ordinal,
                   "threads": mfmod.THREADS, "rate": rate,
                   "metrics": {k: v for k, v in metrics.items() if k.endswith("impute_score")
                               or "/rmse_num_z" in k or "/mae_num_z" in k or "/acc_cat" in k},
                   "times": dict(TIMES), "info": json.loads(json.dumps(INFO, default=str)),
                   "elapsed": round(time.time() - start, 2)}
            with open(out_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            TIMES.clear()
            INFO.clear()

        recompute_folds(variant, seed, 5, rate, [], done=done, on_fold=on_fold,
                        fold_scorer=make_factory(check, keep_all))
        print(variant, seed, folds_arg, "seconds", round(time.time() - start, 1), flush=True)


if __name__ == "__main__":
    main()
