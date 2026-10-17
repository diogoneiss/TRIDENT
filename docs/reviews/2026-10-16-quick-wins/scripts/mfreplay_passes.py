"""Add-on to mfreplay_arms.py: stock test-pass counts per fold, and arm C.

Throwaway, read-only on the TRIDENT repo (run from the repo root, PYTHONDONTWRITEBYTECODE=1, -B).
Writes only the scratchpad JSONL given on the command line. No MLflow, no cache dir, no
BaselineScorer. Reuses mfreplay_arms.MissForestSeqImputer (stock fit, observed) unchanged.

Arms per fold, all from ONE fit:
  missforest           stock MissForestImputer.impute (inherited). Its passes are counted by
                       recording every RoundChange the stock loop computes (the _change hook),
                       then replaying the loop's own rule (still_converging) on that list:
                       computed = passes run, kept = passes whose fills were kept.
  missforest_seq       rounds 1..k models once each (as in mfreplay_arms).
  missforest_seqstock  arm C: start from the sequential fill, then the stock kept-model
                       iteration (verbatim copy of the stock loop, MAX_ROUNDS, still_converging).
  mean_mode            beside them.
Self-check S4 (every fold, both populations): the verbatim loop started from the stock start
(training means/modes) gives exactly the stock impute's frame (assert_frame_equal, exact) and
the same pass counts. Proves arm C differs from the stock arm only by its starting fill.
Also re-emits the replay/seq scores, which must equal the 12:33 run's bitwise (determinism).
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
SP = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("mfreplay_arms", os.path.join(SP, "mfreplay_arms.py"))
arms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arms)

from scripts.backfill_baseline_imputers import recompute_folds  # noqa: E402
from src.training.imputation_baselines import MeanModeImputer, score_baselines  # noqa: E402
from src.training.missforest import MAX_ROUNDS, still_converging  # noqa: E402

INFO = arms.INFO
TIMES = arms.TIMES


def _count(changes):
    kept, previous = 0, None
    for change in changes:
        if previous is not None and not still_converging(previous, change):
            break
        kept, previous = kept + 1, change
    return [len(changes), kept]


class PassCounting(arms.MissForestSeqImputer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._recording = None

    def _change(self, before, after, gaps):
        change = super()._change(before, after, gaps)
        if self._recording is not None and not getattr(self, "_in_fit", False):
            self._recording.append(change)
        return change

    def impute(self, frame, hidden):
        self._recording = []
        try:
            out = super().impute(frame, hidden)
        finally:
            changes, self._recording = self._recording, None
        INFO.setdefault("passes_stock", []).append(_count(changes))
        return out

    def _iterate_kept(self, gaps, fills):
        """The stock impute's loop, verbatim, from a given starting fill."""
        previous = None
        changes = []
        for _ in range(MAX_ROUNDS):
            updated = fills.copy()
            for index in self._order:
                if gaps[:, index].any():
                    updated[gaps[:, index], index] = self._models[index].predict(
                        updated[gaps[:, index]][:, self._others(index)]
                    )
            change = super()._change(fills, updated, gaps)
            changes.append(change)
            if previous is not None and not still_converging(previous, change):
                break
            fills, previous = updated, change
        return fills, _count(changes)

    def impute_verbatim_from_start(self, frame, hidden):
        missing = hidden | frame.isna().to_numpy()
        matrix = self._matrix(frame, missing)
        gaps = np.isnan(matrix)
        fills, passes = self._iterate_kept(gaps, np.where(gaps, self._start, matrix))
        return self._assemble(frame, missing, fills), passes

    def impute_seq_then_stock(self, frame, hidden):
        missing = hidden | frame.isna().to_numpy()
        matrix = self._matrix(frame, missing)
        gaps = np.isnan(matrix)
        start = self.target_fills(frame, missing).copy()
        fills, passes = self._iterate_kept(gaps, start)
        INFO.setdefault("passes_seqstock", []).append(passes)
        return self._assemble(frame, missing, fills)


def make_factory():
    def factory(train, num, cat):
        state = {"fitted": None, "calls": 0}
        mm = MeanModeImputer(num, cat)
        mm.fit(train)

        def score(test, cells, naive):
            state["calls"] += 1
            missing = arms._missing_of(test, cells)
            if state["fitted"] is None:
                imp = PassCounting(num, cat, keep_all=False)
                imp.set_targets([
                    arms._Target("train", train, train.isna().to_numpy()),
                    arms._Target("call1", test, missing),
                    arms._Target("natural", test, test.isna().to_numpy()),
                ])
                t = time.time()
                imp.fit(train)
                TIMES["fit"] = time.time() - t
                INFO["S1_train_seq_equals_fit"] = bool(np.array_equal(imp._targets[0].final, imp._kept_fit_fills))
                state["fitted"] = imp
            imp = state["fitted"]
            tag = "masked" if state["calls"] == 1 else "induced"
            hidden = missing & ~test.isna().to_numpy()
            # S4: verbatim loop from the stock start == stock impute (frame and pass counts).
            stock = imp.impute(test, hidden)
            stock_passes = INFO["passes_stock"].pop()
            verb, verb_passes = imp.impute_verbatim_from_start(test, hidden)
            try:
                pd.testing.assert_frame_equal(stock, verb, check_exact=True)
                INFO[f"S4_{tag}_verbatim_equals_stock"] = verb_passes == stock_passes
            except AssertionError as error:
                INFO[f"S4_{tag}_verbatim_equals_stock"] = False
                INFO[f"S4_{tag}_err"] = str(error)[:300]
            n0 = len(INFO.get("passes_stock", []))
            m0 = len(INFO.get("passes_seqstock", []))
            result = score_baselines([
                arms._Timed(imp, "missforest", "impute"),
                arms._Timed(imp, "missforest_seq", "impute_seq"),
                arms._Timed(imp, "missforest_seqstock", "impute_seq_then_stock"),
                arms._Timed(mm, "mean_mode", "impute"),
            ], test, cells, naive).metrics
            INFO[f"passes_stock_{tag}"] = INFO["passes_stock"][n0:]
            INFO[f"passes_seqstock_{tag}"] = INFO["passes_seqstock"][m0:]
            return result

        return score
    return factory


def main() -> None:
    seed = int(sys.argv[1])
    out_path = sys.argv[2]
    variants = sys.argv[3:]
    rate = float(os.environ.get("MFREPLAY_RATE", "0.2"))
    for variant in variants:
        start = time.time()
        TIMES.clear()
        INFO.clear()

        def on_fold(ordinal, metrics, variant=variant):
            info = {k: v for k, v in INFO.items() if k not in ("passes_stock", "passes_seqstock")}
            row = {"variant": variant, "seed": seed, "fold": ordinal, "threads": arms.mfmod.THREADS, "rate": rate,
                   "metrics": {k: v for k, v in metrics.items() if k.endswith("impute_score")},
                   "times": dict(TIMES), "info": json.loads(json.dumps(info, default=str)),
                   "elapsed": round(time.time() - start, 2)}
            with open(out_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            TIMES.clear()
            INFO.clear()

        recompute_folds(variant, seed, 5, rate, [], on_fold=on_fold, fold_scorer=make_factory())
        print(variant, seed, "seconds", round(time.time() - start, 1), flush=True)


if __name__ == "__main__":
    main()
