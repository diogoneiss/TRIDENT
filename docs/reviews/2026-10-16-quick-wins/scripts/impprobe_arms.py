"""Indicative probe: three candidate imputation baselines on the decoder's exact cells.

Throwaway, read-only on the TRIDENT repo. Run from the repo as cwd with the fork's
interpreter and PYTHONDONTWRITEBYTECODE=1. Writes only to the scratchpad (JSONL per fold).

PRE-REGISTERED RULES (fixed in this file before any score was looked at):

ICE      the earlier probe's MiceImputer, unchanged: MissForestImputer's rounds (mean/mode
         start, columns by increasing missingness, up to 10 rounds, still_converging stop),
         each column's model BayesianRidge (numerical target) or
         LogisticRegression(max_iter=1000) (categorical target); categorical PREDICTORS
         enter as vocabulary codes (MissForest's _matrix). Single imputation, posterior mean.

condexp  numerical columns: x_miss <- mu_m + S_mo S_oo^{-1} (x_obs - mu_o) per test row,
         in the frame's scaled space, conditioning on the row's observed numerical cells
         only. mu = training-fold observed means. S = pairwise-complete covariance around
         mu (sum of co-observed centred products / (n_pair - 1); off-diagonal 0 where
         n_pair < 2; variance 1.0 where a column has < 2 observed values), converted to a
         correlation R, clipped to [-1, 1], shrunk R <- (1 - 0.1) R + 0.1 I, eigenvalues
         floored at 1e-3, rescaled back. No EM. Categorical columns: training mode.

softimp  numerical columns, inductive low-rank: centre by training observed means; rank k
         = smallest k whose top-k eigenvalues of the (unshrunk, eigen-clipped at 0)
         pairwise-complete covariance reach 90% of their sum, capped at max(1, p // 2) and
         at p - 1 (k = 1 when p == 1). Fit: hard-impute (iterative rank-k SVD) on the
         training fold, zeros (= the mean) as the start, up to 100 iterations, stop when
         ||Z_new - Z||^2 / ||Z||^2 < 1e-5. Loadings W = V_k diag(s_k) / sqrt(n_train).
         Test rows: PPCA posterior mean z = (W_o'W_o + s2 I)^{-1} W_o' (x_o - mu_o),
         x_m = mu_m + W_m z, with s2 = mean of the discarded covariance eigenvalues
         ((sum - top-k sum) / (p - k)), floored at 1e-6. Categorical columns: training mode.

Tie band for the report (fixed here too): a difference within 0.003 of impute_score is a
tie; "would set the bar" / "beats the model" only beyond it.

POST-HOC DIAGNOSTIC ARMS (added 11:05 after the kc2_20nan/vehicle_20nan seed-42 check showed
ICE at 8-17x mean/mode and condexp at 1.4-2.1x on kc2; NOT pre-registered, never used for
the bar/model flags): ice_clip = ICE with every numerical model's prediction clipped to its
column's observed training [min, max] inside every fit and fill round (sklearn
IterativeImputer's min_value/max_value semantics); condexp_clip / softimpute_clip = the
pre-registered arm's numerical fills clipped to the same range at the end.

REVISION 11:20 (after a cross-check against sklearn IterativeImputer(BayesianRidge) on fold 1
of biodeg_20nan / kc2_20nan, seed 42: the template's per-round fits are identical to
sklearn's to the last digit, but its transform -- MissForestImputer.impute, which replays the
KEPT round's models on a mean-started test matrix -- scores 2.35 / 5.47 against sklearn's
0.586 / 0.560): ice_seq = the same fit, filled by the per-round model sequence as sklearn's
transform does (impute_seq). Not a tuning; a transform defect for learners that extrapolate.
"ice_replay" is the template's ICE exactly as before. ice_clip (replay + clip) is replaced by
ice_seq_clip (sequence + clip inside every round, sklearn min_value/max_value). The run-1
partial outputs (impprobe_run1_s*.jsonl) used the 11:05 version.
"""

import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from sklearn.linear_model import BayesianRidge, LogisticRegression  # noqa: E402

from scripts.backfill_baseline_imputers import recompute_folds  # noqa: E402
from src.training.imputation_baselines import (  # noqa: E402
    KnnImputer,
    MeanModeImputer,
    _Constant,
    _position,
    score_baselines,
)
from src.training.missforest import MissForestImputer  # noqa: E402
from src.embedder import as_category_strings  # noqa: E402

CONDEXP_SHRINK = 0.1
CONDEXP_EIG_FLOOR = 1e-3
SOFT_ENERGY = 0.9
SOFT_MAX_ITERS = 100
SOFT_TOL = 1e-5
SOFT_S2_FLOOR = 1e-6
TIE_BAND = 0.003

TIMES: dict[str, float] = {}
FOLD_INFO: dict[str, object] = {}


def _tick(name: str, start: float) -> None:
    TIMES[name] = TIMES.get(name, 0.0) + time.time() - start


class MiceImputer(MissForestImputer):
    name = "ice"

    def _fit_one(self, index, features, target):
        if len(target) == 0:
            return _Constant(self._start[index])
        values, counts = np.unique(target, return_counts=True)
        if len(values) == 1:
            return _Constant(float(values[0]))
        if features.shape[1] == 0:
            return _Constant(float(target.mean()) if self._is_numerical(index) else float(values[counts.argmax()]))
        if self._is_numerical(index):
            return BayesianRidge().fit(features, target)
        return LogisticRegression(max_iter=1000).fit(features, target.astype(int))


class _Clipped:
    def __init__(self, model, lo, hi):
        self._model, self._lo, self._hi = model, lo, hi

    def predict(self, features):
        return np.clip(self._model.predict(features), self._lo, self._hi)


class MiceSeqImputer(MiceImputer):
    """The template's ICE fit, unchanged, recording every round's models.

    As an arm (name "ice_replay") it fills with MissForestImputer.impute: the kept round's
    models replayed on the test matrix under still_converging (the template's transform).
    ``impute_seq`` fills as sklearn's IterativeImputer.transform does: start at the training
    means/modes and apply round 1's models, then round 2's, ..., up to the kept round,
    Gauss-Seidel within a round, no stop rule.
    """

    name = "ice_replay"

    def fit(self, frame):
        self._seq = []
        self._calls = 0
        super().fit(frame)

    def _make(self, index, features, target):
        return MiceImputer._fit_one(self, index, features, target)

    def _fit_one(self, index, features, target):
        model = self._make(index, features, target)
        r = self._calls // len(self._order)
        if r == len(self._seq):
            self._seq.append({})
        self._seq[r][index] = model
        self._calls += 1
        return model

    def impute_seq(self, frame, hidden):
        missing = hidden | frame.isna().to_numpy()
        matrix = self._matrix(frame, missing)
        gaps = np.isnan(matrix)
        fills = np.where(gaps, self._start, matrix)
        for models in self._seq[: self.rounds]:
            for index in self._order:
                if gaps[:, index].any():
                    fills[gaps[:, index], index] = models[index].predict(
                        fills[gaps[:, index]][:, self._others(index)]
                    )
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


class _SeqView:
    """The 'ice_seq' arm: the same fitted MiceSeqImputer, filled by its round sequence."""

    name = "ice_seq"

    def __init__(self, inner):
        self._inner = inner

    def fit(self, frame):
        pass  # the inner arm is fit by the factory

    def impute(self, frame, hidden):
        t = time.time()
        out = self._inner.impute_seq(frame, hidden)
        _tick(self.name + "_impute", t)
        return out


class MiceSeqClipImputer(MiceSeqImputer):
    """ice_seq with every numerical prediction clipped to its column's observed training
    range in every fit and fill round (sklearn min_value/max_value)."""

    name = "ice_seq_clip"

    def _make(self, index, features, target):
        model = MiceImputer._fit_one(self, index, features, target)
        if self._is_numerical(index) and len(target):
            return _Clipped(model, float(target.min()), float(target.max()))
        return model

    def impute(self, frame, hidden):
        return self.impute_seq(frame, hidden)


class _ClipEnd:
    """Post-hoc: an arm's numerical fills clipped to the training fold's observed range."""

    def __init__(self, inner, numerical):
        self._inner = inner
        self._numerical = list(numerical)
        self.name = inner.name + "_clip"

    def fit(self, frame):
        # The inner arm is already fit by the factory; only the range is learned here.
        self._lo = frame[self._numerical].min()
        self._hi = frame[self._numerical].max()

    def impute(self, frame, hidden):
        out = self._inner.impute(frame, hidden)
        for column in self._numerical:
            out[column] = np.clip(out[column].to_numpy(dtype=float), self._lo[column], self._hi[column])
        return out


def _pairwise_cov(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Training observed means and the pairwise-complete covariance around them."""
    obs = ~np.isnan(X)
    counts = obs.sum(axis=0)
    mu = np.where(counts > 0, np.nansum(X, axis=0) / np.maximum(counts, 1), 0.0)
    Xc = np.where(obs, X - mu, 0.0)
    O = obs.astype(float)
    n_pair = O.T @ O
    S = (Xc.T @ Xc) / np.maximum(n_pair - 1.0, 1.0)
    S[n_pair < 2] = 0.0
    diag = np.diag(S).copy()
    diag[(counts < 2) | (diag <= 0)] = 1.0
    np.fill_diagonal(S, diag)
    return mu, S


class _NumericFill:
    """Shared shell: categoricals by the training mode (MeanModeImputer), numerics by _fill."""

    name = "?"

    def __init__(self, numerical, categorical):
        self._numerical = list(numerical)
        self._categorical = list(categorical)
        self._mm = MeanModeImputer(numerical, categorical)

    def fit(self, frame: pd.DataFrame) -> None:
        t = time.time()
        self._mm.fit(frame)
        if self._numerical:
            self._fit_numeric(frame[self._numerical].to_numpy(dtype=float))
        _tick(self.name + "_fit", t)

    def impute(self, frame: pd.DataFrame, hidden: np.ndarray) -> pd.DataFrame:
        t = time.time()
        out = self._mm.impute(frame, hidden)
        if self._numerical:
            missing = hidden | frame.isna().to_numpy()
            positions = [_position(frame, c) for c in self._numerical]
            gaps = missing[:, positions]
            X = frame[self._numerical].to_numpy(dtype=float).copy()
            X[gaps] = np.nan
            filled = self._fill_numeric(X, gaps)
            for j, column in enumerate(self._numerical):
                out[column] = filled[:, j]
        out = out.reindex(columns=frame.columns)
        _tick(self.name + "_impute", t)
        return out

    def _fill_numeric(self, X: np.ndarray, gaps: np.ndarray) -> np.ndarray:
        filled = X.copy()
        patterns, inverse = np.unique(gaps, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        for p_index, pattern in enumerate(patterns):
            if not pattern.any():
                continue
            rows = np.nonzero(inverse == p_index)[0]
            m = pattern
            o = ~pattern
            filled[np.ix_(rows, np.nonzero(m)[0])] = self._predict(X[np.ix_(rows, np.nonzero(o)[0])], m, o)
        return filled


class CondExpImputer(_NumericFill):
    name = "condexp"

    def _fit_numeric(self, X: np.ndarray) -> None:
        mu, S = _pairwise_cov(X)
        sd = np.sqrt(np.diag(S))
        R = np.clip(S / np.outer(sd, sd), -1.0, 1.0)
        np.fill_diagonal(R, 1.0)
        R = (1 - CONDEXP_SHRINK) * R + CONDEXP_SHRINK * np.eye(len(R))
        w, Q = np.linalg.eigh(R)
        w = np.maximum(w, CONDEXP_EIG_FLOOR)
        R = (Q * w) @ Q.T
        self._mu = mu
        self._S = R * np.outer(sd, sd)

    def _predict(self, Xo: np.ndarray, m: np.ndarray, o: np.ndarray) -> np.ndarray:
        mu_m, mu_o = self._mu[m], self._mu[o]
        if not o.any():
            return np.tile(mu_m, (len(Xo), 1))
        S_oo = self._S[np.ix_(o, o)]
        S_om = self._S[np.ix_(o, m)]
        B = np.linalg.solve(S_oo, S_om)  # |o| x |m|
        return mu_m + (Xo - mu_o) @ B


class SoftImputeImputer(_NumericFill):
    name = "softimpute"

    def _fit_numeric(self, X: np.ndarray) -> None:
        n, p = X.shape
        mu, S = _pairwise_cov(X)
        eig = np.sort(np.maximum(np.linalg.eigvalsh(S), 0.0))[::-1]
        total = float(eig.sum())
        if p == 1:
            k = 1
        else:
            cum = np.cumsum(eig) / total if total > 0 else np.ones(p)
            k = int(np.searchsorted(cum, SOFT_ENERGY) + 1)
            k = max(1, min(k, max(1, p // 2), p - 1))
        s2 = (total - float(eig[:k].sum())) / (p - k) if p > k else 0.0
        s2 = max(s2, SOFT_S2_FLOOR)
        obs = ~np.isnan(X)
        Xc = np.where(obs, X - mu, 0.0)
        Z = Xc.copy()
        iters = 0
        for iters in range(1, SOFT_MAX_ITERS + 1):
            U, s, Vt = np.linalg.svd(Z, full_matrices=False)
            L = (U[:, :k] * s[:k]) @ Vt[:k]
            Z_new = np.where(obs, Xc, L)
            delta = float(((Z_new - Z) ** 2).sum()) / max(float((Z ** 2).sum()), 1e-12)
            Z = Z_new
            if delta < SOFT_TOL:
                break
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        self._W = Vt[:k].T * (s[:k] / np.sqrt(n))
        self._mu = mu
        self._s2 = s2
        info = FOLD_INFO.setdefault("softimpute", [])
        info.append({"p": p, "k": k, "s2": round(s2, 5), "iters": iters,
                     "energy_at_k": round(float(eig[:k].sum() / total), 4) if total > 0 else None})

    def _predict(self, Xo: np.ndarray, m: np.ndarray, o: np.ndarray) -> np.ndarray:
        mu_m, mu_o = self._mu[m], self._mu[o]
        if not o.any():
            return np.tile(mu_m, (len(Xo), 1))
        W_o, W_m = self._W[o], self._W[m]
        k = self._W.shape[1]
        M = np.linalg.solve(W_o.T @ W_o + self._s2 * np.eye(k), W_o.T)  # k x |o|
        return mu_m + ((Xo - mu_o) @ M.T) @ W_m.T


class _Timed:
    """Times an imputer's fit and impute under its own name."""

    def __init__(self, inner):
        self._inner = inner
        self.name = inner.name

    def fit(self, frame):
        t = time.time()
        self._inner.fit(frame)
        _tick(self.name + "_fit", t)
        if isinstance(self._inner, MissForestImputer):
            FOLD_INFO.setdefault(self.name + "_rounds", []).append(self._inner.rounds)
            FOLD_INFO.setdefault(self.name + "_fitted_rounds", []).append(len(self._inner._seq))

    def impute(self, frame, hidden):
        t = time.time()
        out = self._inner.impute(frame, hidden)
        _tick(self.name + "_impute", t)
        return out


def make_factory(with_knn: bool):
    def factory(train, num, cat):
        condexp, soft = CondExpImputer(num, cat), SoftImputeImputer(num, cat)
        ice = MiceSeqImputer(num, cat)
        arms = [
            _Timed(ice),
            condexp,
            soft,
            _Timed(MeanModeImputer(num, cat)),
            _Timed(MiceSeqClipImputer(num, cat)),
        ]
        # The clipped wrappers reuse the fitted condexp/softimpute (fit once, below).
        post = [_ClipEnd(condexp, num), _ClipEnd(soft, num)]
        if with_knn:
            arms.append(_Timed(KnnImputer(num, cat, n_neighbors=5)))
        for arm in arms:
            arm.fit(train)
        for arm in post:
            arm.fit(train)
        scored = arms[:1] + [_SeqView(ice)] + arms[1:] + post
        return lambda test, cells, naive: score_baselines(scored, test, cells, naive).metrics
    return factory


def main() -> None:
    seed = int(sys.argv[1])
    out_path = sys.argv[2]
    with_knn = sys.argv[3] == "knn"
    variants = sys.argv[4:]
    for variant in variants:
        start = time.time()
        TIMES.clear()
        FOLD_INFO.clear()

        def on_fold(ordinal, metrics, variant=variant):
            row = {"variant": variant, "seed": seed, "fold": ordinal,
                   "metrics": {k: v for k, v in metrics.items() if k.endswith("impute_score")
                               or "/rmse_num_z" in k or "/mae_num_z" in k or "/acc_cat" in k},
                   "times": dict(TIMES), "info": json.loads(json.dumps(FOLD_INFO, default=str)),
                   "elapsed": round(time.time() - start, 2)}
            with open(out_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            TIMES.clear()
            FOLD_INFO.clear()

        recompute_folds(variant, seed, 5, 0.2, [], on_fold=on_fold, fold_scorer=make_factory(with_knn))
        print(variant, seed, "seconds", round(time.time() - start, 1), flush=True)


if __name__ == "__main__":
    main()
