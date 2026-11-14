"""E179 (imputation-protocol/69, T56): multiple imputation with draws joint across a row's gaps.

    uv run --python 3.11 python -B scripts/experiments/joint_draws.py --check
    uv run --python 3.11 python -B scripts/experiments/joint_draws.py --todo <queue> <queues>
    uv run --python 3.11 python -B scripts/experiments/joint_draws.py train H <variant> <seed> [--no-mlflow]
    uv run --python 3.11 python -B scripts/experiments/joint_draws.py cells <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/joint_draws.py --tally

``train`` is E82's H cell (``deployable_hybrid.train``: ``--decode_checkpoint loss --decode_calibration none``, the
validation split's hidden observed cells written to ``selection/``), tag ``experiment=joint-draws-2026-11-13``. After
each fold's decode stage the run's model also fills the test split's gaps in place, random states restored, into
``draws/<variant>_s<seed>/draws_fold_<k>.csv``: ``plain`` (its fill, as the run scores it) and M draws of two kinds.
For a draw, every ``Dropout`` module applies one mask to all rows and all passes (one draw of the network):

- ``s_<m>`` (``MI-S``): one pass; a categorical gap takes a class sampled from the head's softmax, a numerical gap the
  pass's fill plus a residual of its column among the validation cells (the pool resampled with replacement a draw);
- ``j_<m>`` (``MI-J``): the same draws chained. Each row's gaps are filled one at a time in a random order, and a draw
  is written into the row (a class id, or a value with its mask flag cleared) before the next pass.

``cells`` (CPU, no store) completes each fold's test rows per arm and reads E175's estimands (``multiple_imputation``:
means, shares, slopes) two ways. The primary reading takes the fold's own complete-data estimate as the target and
the between-draw variance alone: Q̄ ± t(M−1) sqrt((1 + 1/M) B). The secondary is E175's (the whole table's value, the
total variance). Arms: ``CD`` and ``SI`` (secondary only), ``MI-D`` (E175's per-cell residual draws), ``MI-S``,
``MI-R`` (``MI-D`` with a row's numerical residuals drawn together through a Gaussian copula, its correlation that of
the residuals' normal scores over validation rows), ``MI-J`` and ``MICE``.

``--tally`` prints the pre-registered reading. ``--check`` runs the self-check on synthetic data only.
Run from the checkout root.
"""

from __future__ import annotations

import json
import socket
import sys
import time
import warnings
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scipy import stats
from torch import nn

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import deployable_hybrid as e82  # noqa: E402
from scripts.experiments import hybrid_confirmation as confirmation  # noqa: E402
from scripts.experiments import multiple_imputation as mi  # noqa: E402
from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments import tabpfn_blend as tb  # noqa: E402

EXPERIMENT = "joint-draws-2026-11-13"
STATE = Path("/scratch/diogoneiss/trident-2026-11-13-joint-draws")
HOST = "gorgona8"
VARIANTS = mi.VARIANTS
SEEDS = (4545, 5656, 6767)
FOLDS, M, LEVEL, MIN_FIT, SESOI = mi.FOLDS, mi.M, mi.LEVEL, mi.MIN_FIT, mi.SESOI
MULTIPLE = ("MI-D", "MI-S", "MI-R", "MI-J", "MICE")
ARMS = ("CD", "SI") + MULTIPLE
HELD = ("vehicle", "credit-g", "pendigits")  # the tables where E175's control (CD) covered
DRAWN = ["plain"] + [f"{kind}_{m}" for kind in "sj" for m in range(M)]
Floats = NDArray[np.float64]


class SharedDropout:
    """Until ``remove``, every ``Dropout`` module of a model left in eval mode applies one mask to all rows, the same
    one on every pass until ``renew``: one draw of the network a multiple imputation."""

    def __init__(self, model: nn.Module, generator: torch.Generator) -> None:
        self._generator = generator
        self._masks: dict[nn.Module, torch.Tensor] = {}
        self._handles = [m.register_forward_hook(self._apply) for m in model.modules() if isinstance(m, nn.Dropout) and m.p > 0]
        self.modules = len(self._handles)

    def _apply(self, module: nn.Module, inputs: Any, output: torch.Tensor) -> torch.Tensor:
        mask = self._masks.get(module)
        if mask is None or mask.shape[1:] != output.shape[1:]:
            keep = 1 - float(getattr(module, "p"))
            mask = (torch.rand((1, *output.shape[1:]), generator=self._generator, device=output.device) < keep).to(output.dtype) / keep
            self._masks[module] = mask
        return output * mask

    def renew(self) -> None:
        self._masks.clear()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()


def model_draws(model: Any, test: pd.DataFrame, selection: pd.DataFrame, device: torch.device, stream: list[int]) -> pd.DataFrame:
    """Every gap of ``test`` with the model's fill (``plain``) and its ``s_<m>`` and ``j_<m>`` draws (module docstring).
    ``selection`` is the fold's hidden observed validation cells, the pool of numerical residuals."""
    from src.embedder import EncodedTable, EpochMasker
    from src.training.imputation_metrics import CATEGORICAL, NUMERICAL

    embedder = model.embedder
    categorical, numerical = list(embedder.categorical_columns), list(embedder.numerical_columns)
    order, n_cat = categorical + numerical, len(categorical)
    gap = test[order].isna().to_numpy()
    counts = gap.sum(axis=1)
    base = embedder.gaps_as_mask(EpochMasker(embedder, test, device).apply(np.zeros(test.shape, dtype=bool)))
    rng = np.random.default_rng(stream)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(rng.integers(2**31)))
    residual = selection[selection["kind"] == NUMERICAL]
    pools = [residual.loc[residual["column"] == c, "actual"].to_numpy(dtype=float) - residual.loc[residual["column"] == c, "model"].to_numpy(dtype=float)
             for c in numerical]
    valid = [getattr(model, f"valid_ids_{key}") for key in embedder.categorical_keys]
    every = torch.arange(len(test), device=device)

    def sample(context: torch.Tensor, c: int, rows: torch.Tensor) -> torch.Tensor:
        logits = model.categorical_heads[embedder.categorical_keys[c]](context[rows, c, :])
        return valid[c][torch.multinomial(logits.softmax(dim=-1), 1, generator=generator).squeeze(1)]  # type: ignore[no-any-return]

    def value(context: torch.Tensor, j: int, rows: torch.Tensor, pool: Floats) -> torch.Tensor:
        fill = model.numerical_heads[embedder.numerical_keys[j]](context[rows, n_cat + j, :]).squeeze(-1)
        if not len(pool):
            return fill  # type: ignore[no-any-return]
        return fill + torch.as_tensor(rng.choice(pool, len(rows)), dtype=fill.dtype, device=device)  # type: ignore[no-any-return]

    def resampled() -> list[Floats]:
        return [rng.choice(pool, len(pool)) if len(pool) else pool for pool in pools]

    shape = (M, len(order), len(test))
    drawn = {"s": np.empty(shape), "j": np.empty(shape)}  # ids and values alike, as floats
    with torch.no_grad():
        passed = model.predict(base)
        plain = np.concatenate([passed.categorical_ids.cpu().numpy().astype(float), passed.numerical_values.cpu().numpy().astype(float)])
        dropout = SharedDropout(model, generator)
        try:
            for m in range(M):
                dropout.renew()
                pool_of = resampled()
                context = model._encode(base)
                for c in range(n_cat):
                    drawn["s"][m, c] = sample(context, c, every).cpu().numpy()
                for j in range(len(numerical)):
                    drawn["s"][m, n_cat + j] = value(context, j, every, pool_of[j]).cpu().numpy()
                dropout.renew()
                pool_of = resampled()
                ids, values, flags = base.cat_indices.clone(), base.num_values.clone(), base.mask_flags.clone()
                rank = rng.random(gap.shape)
                rank[~gap] = np.inf
                turn = np.argsort(rank, axis=1)
                for step in range(int(counts.max())):
                    active = np.flatnonzero(counts > step)
                    column = turn[active, step]
                    context = model._encode(EncodedTable(ids, values, flags, base.null_flags, base.masked_positions, base.fill_flags))
                    for c in np.unique(column):
                        rows = torch.as_tensor(active[column == c], device=device)
                        if c < n_cat:
                            ids[int(c), rows] = sample(context, int(c), rows)
                        else:
                            values[int(c) - n_cat, rows] = value(context, int(c) - n_cat, rows, pool_of[int(c) - n_cat])
                            flags[int(c) - n_cat, rows] = False
                assert not bool(flags.any()), "the chain left a numerical gap unfilled"
                drawn["j"][m] = np.concatenate([ids.cpu().numpy().astype(float), values.cpu().numpy().astype(float)])
        finally:
            dropout.remove()
    frames = []
    for c, name in enumerate(order):
        at = np.flatnonzero(gap[:, c])
        if not len(at):
            continue
        columns = {"plain": plain[c, at], **{f"{kind}_{m}": drawn[kind][m, c, at] for kind in "sj" for m in range(M)}}
        if c < n_cat:
            decode = embedder.label_encoders[name].inverse_transform
            columns = {key: np.asarray(decode(ids_.astype(int))).astype(str) for key, ids_ in columns.items()}
            assert not any("[MASK]" in set(v) or "[NULL]" in set(v) for v in columns.values()), f"{name}: a draw is a gap token"
        frames.append(pd.DataFrame({"row": at, "column": name, "kind": CATEGORICAL if c < n_cat else NUMERICAL, **columns}))
    print(f"draws: {int(gap.sum())} gaps, at most {int(counts.max())} a row, {dropout.modules} dropout modules", flush=True)
    return pd.concat(frames, ignore_index=True)


def train(variant: str, seed: int, no_mlflow: bool) -> None:
    import src.training.runner as runner

    setattr(e82, "EXPERIMENT", EXPERIMENT)
    setattr(e82, "STATE", STATE)
    real = getattr(runner, "train_and_evaluate_decoder")
    select = e82._selection_cells
    last: dict[str, Any] = {}

    def stash(**kwargs: Any) -> Any:
        last.update(kwargs)
        return real(**kwargs)

    def with_draws(model: Any, frame: pd.DataFrame, device: torch.device, seed_: int, fold_ordinal: int) -> pd.DataFrame:
        # called by e82's wrapper after the fold's decode stage, model in eval mode, random states saved around it
        selection = select(model, frame, device, seed_, fold_ordinal)
        dataset, fold = last["dataset"], last["fold"]
        test = dataset.frame.drop(columns=[dataset.label_column]).iloc[fold.test_indices].reset_index(drop=True)
        out = STATE / "draws" / f"{variant}_s{seed}"
        out.mkdir(parents=True, exist_ok=True)
        model_draws(model, test, selection, device, [179, zlib.crc32(variant.encode()), seed, fold_ordinal]).to_csv(
            out / f"draws_fold_{fold_ordinal}.csv", index=False)
        return selection

    setattr(runner, "train_and_evaluate_decoder", stash)
    setattr(e82, "_selection_cells", with_draws)
    e82.train(variant, seed, no_mlflow)


def todo(which: int, queues: int) -> None:
    if socket.gethostname() != HOST:
        sys.exit(f"E179 runs on {HOST}")
    loads = [0.0] * queues
    for cost, variant, seed in sorted(((confirmation.MINUTES[v.split("_")[0]], v, s) for v in VARIANTS for s in SEEDS),
                                      key=lambda w: (-w[0], w[1], w[2])):
        target = loads.index(min(loads))
        loads[target] += cost
        if target == which - 1 and not (STATE / "draws" / f"{variant}_s{seed}" / f"draws_fold_{FOLDS}.csv").exists():
            print("H", variant, seed)


def between(q: Floats) -> tuple[Floats, Floats]:
    """Over M completed tables' estimates (M, estimands): the pooled estimate and the half width of its interval for
    the complete-data estimate, the missing-data part of Rubin's variance alone, t(M−1) sqrt((1 + 1/M) B)."""
    m = len(q)
    return q.mean(axis=0), stats.t.ppf(0.5 + LEVEL / 2, m - 1) * np.sqrt((1 + 1 / m) * q.var(axis=0, ddof=1))


def correlation(pool: pd.DataFrame, numerical: list[str]) -> tuple[Floats, int]:
    """The correlation between columns of the numerical residuals' normal scores over validation rows, a cell's
    residual averaged over the passes that hid it, and how many column pairs have one (at least ``MIN_FIT`` rows with
    both columns hidden in some pass; the others take 0). Pairs hidden in one same pass are too few: counted on
    E174's selection files, they reach ``MIN_FIT`` on ``pendigits`` at 20 and 40% only."""
    part = pool[pool["kind"] == "numerical"]
    wide = (part.assign(e=part["actual"].astype(float) - part["model"].astype(float))
            .groupby(["row", "column"])["e"].mean().unstack("column").reindex(columns=numerical))
    found = (wide.rank() / (wide.count() + 1)).apply(stats.norm.ppf).corr(min_periods=MIN_FIT).to_numpy(dtype=float)
    pairs = int(np.isfinite(found[np.triu_indices(len(numerical), 1)]).sum())
    found = np.where(np.isfinite(found), found, 0.0)
    np.fill_diagonal(found, 1.0)
    values, vectors = np.linalg.eigh(found)  # pairwise-complete correlations need not be positive definite
    fixed = (vectors * np.clip(values, 1e-6, None)) @ vectors.T
    scale = np.sqrt(np.diag(fixed))
    return fixed / np.outer(scale, scale), pairs


def joint_residual_draws(test: pd.DataFrame, pool: pd.DataFrame, numerical: list[str],
                         rng: np.random.Generator) -> tuple[list[NDArray[np.object_]], int]:
    """``MI-R``: ``mi.draws``' fills with each row's numerical residuals drawn together. A Gaussian copula with
    ``correlation``'s matrix picks each cell's quantile of its column's resampled pool, so a column's draws keep
    ``MI-D``'s distribution. Also the number of column pairs with a correlation."""
    fills, _ = mi.draws(test, pool, rng, binned=False)
    if not numerical:
        return fills, 0
    matrix, pairs = correlation(pool, numerical)
    root = np.linalg.cholesky(matrix)
    row_of, rows = pd.factorize(test["row"])
    base = test["model"].to_numpy(dtype=object)
    groups = []
    for j, column in enumerate(numerical):
        at = np.flatnonzero((test["column"] == column).to_numpy())
        source = pool[pool["column"] == column]
        residual = source["actual"].to_numpy(dtype=float) - source["model"].to_numpy(dtype=float)
        if len(at) and len(residual):
            groups.append((j, at, residual, base[at].astype(float)))
    for fill in fills:
        u = stats.norm.cdf(rng.standard_normal((len(rows), len(numerical))) @ root.T)
        for j, at, residual, centre in groups:
            resampled = np.sort(rng.choice(residual, len(residual)))
            fill[at] = centre + resampled[np.minimum((u[row_of[at], j] * len(resampled)).astype(int), len(resampled) - 1)]
    return fills, pairs


def _typed(values: pd.Series, numeric: NDArray[np.bool_]) -> NDArray[np.object_]:
    out = values.astype(str).to_numpy(dtype=object)
    out[numeric] = values[numeric].astype(float)
    return out


def cells(variant: str, seed: int) -> None:
    from src.embedder import as_category_strings
    from src.training.data import build_folds, prepare_dataset
    from src.training.imputation_metrics import NUMERICAL
    from src.training.types import DatasetSpec

    warnings.filterwarnings("ignore")
    started = time.time()
    dataset = prepare_dataset(DatasetSpec.from_name(variant, None))
    features = dataset.frame.drop(columns=[dataset.label_column]).reset_index(drop=True)
    numerical, categorical = list(dataset.numerical_columns), list(dataset.categorical_columns)
    folds = list(build_folds(dataset.frame, dataset.label_column, FOLDS, seed))
    # the complete table from the five ledgers' truths, as E175 builds it
    ledgers: dict[int, pd.DataFrame] = {}
    truth = features.astype(object)
    for c in categorical:
        truth[c] = as_category_strings(features[c]).astype(object)
    block = truth.to_numpy(dtype=object, copy=True)
    for k, fold in enumerate(folds, start=1):
        ledger = e82._typed(tb._ledger(variant, seed, k, STATE), ("actual", "imputed")).rename(columns={"imputed": "model"})
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
    table_truth = mi.estimands(*mi._blocks(complete, numerical, categorical), modes, 1.0)[0]
    labelled = mi.names(numerical, categorical)
    rows: list[dict[str, object]] = []
    pairs, kept = 0.0, 0.0
    for k, fold in enumerate(folds, start=1):
        test_at = np.asarray(fold.test_indices)
        ledger = ledgers[k]
        local, position = ledger["row"].to_numpy(dtype=int), ledger["position"].to_numpy()
        base = complete.iloc[test_at].to_numpy(dtype=object, copy=True)
        fpc = 1 - len(test_at) / len(complete)
        numeric = (ledger["kind"] == NUMERICAL).to_numpy()
        table = pd.read_csv(STATE / "draws" / f"{variant}_s{seed}" / f"draws_fold_{k}.csv", dtype={c: object for c in DRAWN})
        joined = ledger[["row", "column"]].merge(table, on=["row", "column"], how="left", validate="one_to_one")
        assert len(table) == len(ledger) and bool(joined["plain"].notna().all()), f"fold {k}: the draws are not the ledger's cells"
        plain, model = _typed(joined["plain"], numeric), ledger["model"].to_numpy(dtype=object)
        assert not numeric.any() or float(np.abs(plain[numeric].astype(float) - model[numeric].astype(float)).max()) < 1e-3, f"fold {k}: plain is not H's fill"
        assert numeric.all() or float((plain[~numeric] == model[~numeric]).mean()) >= 0.999, f"fold {k}: plain is not H's class"

        def read(fills: list[NDArray[np.object_]]) -> tuple[Floats, Floats, Floats, Floats, Floats]:
            qs, us = [], []
            for fill in fills:
                completed = base.copy()
                completed[local, position] = fill
                frame = pd.DataFrame(completed, columns=features.columns)
                frame[categorical] = frame[categorical].astype(str)
                q, u, df = mi.estimands(*mi._blocks(frame, numerical, categorical), modes, fpc)
                qs.append(q)
                us.append(u)
            q_all, u_all = np.array(qs), np.array(us)
            pooled = [mi.rubin(q_all[:, i], u_all[:, i], float(df[i])) for i in range(len(df))]
            total_half = stats.t.ppf(0.5 + LEVEL / 2, np.array([p[2] for p in pooled])) * np.sqrt(np.array([p[1] for p in pooled]))
            between_half = between(q_all)[1] if len(fills) > 1 else np.full(len(df), np.nan)
            return q_all.mean(axis=0), between_half, total_half, np.array([p[3] for p in pooled]), u_all.mean(axis=0)

        def rng(arm: int) -> np.random.Generator:
            return np.random.default_rng([179, zlib.crc32(variant.encode()), seed, k, arm])

        selection = e82._typed(pd.read_csv(STATE / "selection" / f"{variant}_s{seed}" / f"selection_fold_{k}.csv",
                                           dtype={"actual": object, "model": object}), ("actual", "model"))
        deployable, kept_d = mi.draws(ledger, selection, rng(2), binned=False)
        together, found = joint_residual_draws(ledger, selection, numerical, rng(5))
        pairs += found / FOLDS
        kept += kept_d / FOLDS
        mice = mi.MiceImputer(numerical, categorical)
        mice.fit(features.iloc[np.asarray(fold.train_indices)].reset_index(drop=True))
        test = features.iloc[test_at].reset_index(drop=True)
        mice.rng = rng(4)
        chained = [mice.impute(test, np.zeros(test.shape, dtype=bool)).to_numpy(dtype=object)[local, position] for _ in range(M)]
        fills = {"CD": [ledger["actual"].to_numpy(dtype=object)], "SI": [model], "MI-D": deployable,
                 "MI-S": [_typed(joined[f"s_{m}"], numeric) for m in range(M)], "MI-R": together,
                 "MI-J": [_typed(joined[f"j_{m}"], numeric) for m in range(M)], "MICE": chained}
        fold_truth, _, _, _, complete_u = read(fills["CD"])
        for arm in ARMS:
            estimate, between_half, total_half, lam, within = read(fills[arm])
            for i, (family, name) in enumerate(labelled):
                rows.append({
                    "variant": variant, "seed": seed, "fold": k, "arm": arm, "family": family, "estimand": name,
                    "fold_truth": fold_truth[i], "table_truth": table_truth[i], "estimate": estimate[i],
                    "half_between": between_half[i], "half_total": total_half[i], "lam": lam[i],
                    "within_ratio": within[i] / complete_u[i] if complete_u[i] > 0 else np.nan,
                    "covered_between": float(abs(estimate[i] - fold_truth[i]) <= between_half[i] + 1e-12) if arm in MULTIPLE else np.nan,
                    "covered_total": float(abs(estimate[i] - table_truth[i]) <= total_half[i])})
    for sub in ("csv", "info"):
        (STATE / sub).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(STATE / "csv" / f"{variant}_s{seed}.csv", index=False)
    of = len(numerical) * (len(numerical) - 1) // 2
    (STATE / "info" / f"{variant}_s{seed}.json").write_text(json.dumps({"pairs": pairs, "of": of, "kept": kept}))
    print(f"DONE {variant} s{seed}: plain equals H's ledger on {FOLDS} folds, {len(labelled)} estimands, MI-R column pairs with a "
          f"correlation {pairs:.1f} of {of} a fold, cells keeping H's fill under MI-D {kept:.3f}, {time.time() - started:.0f} s", flush=True)


def measures(data: pd.DataFrame, covered: str) -> pd.DataFrame:
    """Per (variant, seed, fold, arm): coverage over the fold's estimands and the shortfall u."""
    out = data.groupby(["variant", "seed", "fold", "arm"])[covered].mean().rename("coverage").reset_index()
    out["u"] = (LEVEL - out["coverage"]).clip(lower=0)
    return out


def reading(calls: dict[str, str]) -> tuple[str, list[str], list[str]]:
    """The pre-registered rule on u(arm) − u(MI-D) by variant: the reading, the 80% variants better, the variants worse."""
    better = [v for v, c in calls.items() if v.endswith("_80nan") and c == "better"]
    worse = [v for v, c in calls.items() if c == "worse"]
    return ("supported" if len(better) >= 4 and not worse else "refuted" if len(better) <= 1 or len(worse) >= 3 else "mixed"), better, worse


def _verdicts(table: pd.DataFrame, variant: str, arms: tuple[str, ...], pairs: tuple[tuple[str, str], ...],
              calls: dict[str, dict[str, str]]) -> None:
    part = table[table["variant"] == variant]
    u = {arm: [part[(part["arm"] == arm) & (part["seed"] == s)].sort_values("fold")["u"].to_numpy() for s in SEEDS] for arm in arms}
    for first, second in pairs:
        v = common.verdict([a - b for a, b in zip(u[first], u[second])], sesoi=SESOI)
        calls[f"{first}-{second}"][variant] = v.call
        print(f"  - u({first}) − u({second}) {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")


def tally() -> None:
    missing = [(v, s) for v in VARIANTS for s in SEEDS if not (STATE / "csv" / f"{v}_s{s}.csv").exists()]
    if missing:
        print(f"incomplete: {len(missing)} of {len(VARIANTS) * len(SEEDS)} cells missing, {missing[:6]}")
        return
    data = pd.concat([pd.read_csv(STATE / "csv" / f"{v}_s{s}.csv") for v in VARIANTS for s in SEEDS], ignore_index=True)
    info = {(v, s): json.loads((STATE / "info" / f"{v}_s{s}.json").read_text()) for v in VARIANTS for s in SEEDS}
    multiple = data[data["arm"].isin(MULTIPLE)]
    table = measures(multiple, "covered_between")
    pairs = (("MI-R", "MI-D"), ("MI-J", "MI-D"), ("MI-S", "MI-D"), ("MI-J", "MI-S"), ("MI-R", "MICE"), ("MI-J", "MICE"))
    calls: dict[str, dict[str, str]] = {f"{a}-{b}": {} for a, b in pairs}
    proper: dict[str, list[str]] = {arm: [] for arm in MULTIPLE}
    print("Primary reading: the fold's complete-data estimate, the between-draw variance alone\n")
    for variant in VARIANTS:
        rows = multiple[multiple["variant"] == variant]
        level = table[table["variant"] == variant].groupby("arm")[["coverage", "u"]].mean()
        ratio = rows.groupby("arm")["within_ratio"].median()
        print(f"- `{variant}`: " + "; ".join(
            f"{arm} coverage {level.loc[arm, 'coverage']:.3f} u {level.loc[arm, 'u']:.3f} within/complete {ratio[arm]:.2f}" for arm in MULTIPLE))
        families = rows.groupby(["family", "arm"])["covered_between"].mean().unstack("arm")
        print("  - by family: " + "; ".join(f"{family} " + " ".join(f"{arm} {families.loc[family, arm]:.3f}" for arm in MULTIPLE)
                                              for family in families.index))
        print(f"  - MI-R: {np.mean([info[variant, s]['pairs'] for s in SEEDS]):.1f} of {info[variant, SEEDS[0]]['of']} column pairs with a "
              f"residual correlation a fold; cells keeping H's fill under MI-D {np.mean([info[variant, s]['kept'] for s in SEEDS]):.3f}")
        for arm in level.index[level["coverage"].to_numpy(dtype=float) >= 0.90]:
            proper[str(arm)].append(variant)
        _verdicts(table, variant, MULTIPLE, pairs, calls)
    print()
    for arm in MULTIPLE:
        print(f"- mean coverage at least 0.90 under {arm}: {len(proper[arm])} of {len(VARIANTS)}")
    for pair, by_variant in calls.items():
        print(f"- u, {pair}: {common.tally(by_variant.values())}")
    print()
    for arm in ("MI-R", "MI-J"):
        call, better, worse = reading(calls[f"{arm}-MI-D"])
        print(f"E179 PRIMARY, u({arm}) − u(MI-D): better on {len(better)} of the 5 variants at 80% ({better or 'none'}), worse on "
              f"{worse or 'none'} of {len(VARIANTS)} -> {call} ({common.RULE}; SESOI {SESOI})")
        print(f"E179 PRIMARY, {arm} proper enough (mean coverage at least 0.90 on at least 12 of {len(VARIANTS)}): "
              f"{len(proper[arm])} -> {'yes' if len(proper[arm]) >= 12 else 'no'}")
    held = tuple(v for v in VARIANTS if v.split("_")[0] in HELD)
    total = measures(data[data["variant"].isin(held)], "covered_total")
    second = (("MI-D", "SI"), ("MI-R", "MI-D"), ("MI-J", "MI-D"))
    secondary: dict[str, dict[str, str]] = {f"{a}-{b}": {} for a, b in second}
    print(f"\nSecondary, E175's reading (the whole table's value, the total variance) on {', '.join(HELD)}\n")
    for variant in held:
        mean = total[total["variant"] == variant].groupby("arm")["coverage"].mean()
        print(f"- `{variant}`: " + "; ".join(f"{arm} {mean[arm]:.3f}" for arm in ARMS))
        _verdicts(total, variant, ARMS, second, secondary)
    print()
    for pair, by_variant in secondary.items():
        print(f"- u, {pair}: {common.tally(by_variant.values())}")


def check() -> None:
    """The between-variance interval on a synthetic sample: proper draws (parameters from their posterior, then
    noise) cover the sample's complete-data mean near 0.95, draws with the parameters held fixed cover less. Then the
    copula draws reproduce a planted residual correlation with ``MI-D``'s values, the shared dropout mask, the rule."""
    rng = np.random.default_rng(179)
    n, covered = 400, {"proper": 0, "fixed": 0}
    for _ in range(400):
        x = rng.normal(size=n)
        y = 0.3 * x + rng.normal(size=n)
        gone = rng.random(n) < 0.6
        seen = int((~gone).sum())
        design = np.column_stack([np.ones(n), x])
        beta, *_ = np.linalg.lstsq(design[~gone], y[~gone], rcond=None)
        sigma2 = float(((y[~gone] - design[~gone] @ beta) ** 2).sum()) / (seen - 2)
        spread = np.linalg.inv(design[~gone].T @ design[~gone])
        q: dict[str, list[float]] = {"proper": [], "fixed": []}
        for _ in range(M):
            s2 = sigma2 * (seen - 2) / rng.chisquare(seen - 2)
            for name, (b, s) in (("proper", (rng.multivariate_normal(beta, s2 * spread), s2)), ("fixed", (beta, sigma2))):
                filled = y.copy()
                filled[gone] = design[gone] @ b + rng.normal(0, np.sqrt(s), int(gone.sum()))
                q[name].append(float(filled.mean()))
        for name, values in q.items():
            estimate, half = between(np.array(values)[:, None])
            covered[name] += bool(abs(estimate[0] - y.mean()) <= half[0])
    share = {name: count / 400 for name, count in covered.items()}
    assert share["proper"] >= 0.92 and share["fixed"] <= share["proper"] - 0.08, share

    size = 600
    noise = rng.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], size)
    pool = pd.DataFrame({"row": np.repeat(np.arange(size), 2), "group": 0, "column": ["a", "b"] * size, "kind": "numerical",
                         "actual": noise.ravel(), "model": 0.0})
    pool = pd.concat([pool, pd.DataFrame({"row": [0, 1], "group": 1, "column": "c", "kind": "categorical", "actual": "x", "model": "x"})], ignore_index=True)
    test = pd.DataFrame({"row": np.repeat(np.arange(300), 2), "column": ["a", "b"] * 300, "kind": "numerical", "model": 0.0})
    matrix, pairs = correlation(pool, ["a", "b"])
    assert pairs == 1 and abs(matrix[0, 1] - 0.7) < 0.08, matrix
    fills, _ = joint_residual_draws(test, pool, ["a", "b"], rng)
    drawn = np.array([f.astype(float) for f in fills])
    together = float(np.mean([np.corrcoef(d[0::2], d[1::2])[0, 1] for d in drawn]))
    assert len(fills) == M and abs(together - 0.7) < 0.08 and set(drawn[:, 0::2].ravel()) <= set(noise[:, 0]), together
    apart, _ = mi.draws(test, pool, rng, binned=False)
    assert abs(float(np.mean([np.corrcoef(f[0::2].astype(float), f[1::2].astype(float))[0, 1] for f in apart]))) < 0.1
    sparse = pool[(pool["column"] != "b") | (pool["row"] < MIN_FIT - 1)]
    assert correlation(sparse, ["a", "b"])[1] == 0 and np.allclose(correlation(sparse, ["a", "b"])[0], np.eye(2))

    layer = nn.Sequential(nn.Linear(8, 64), nn.Dropout(0.5)).eval()
    generator = torch.Generator()
    generator.manual_seed(1)
    inputs = torch.ones(5, 8)
    plain = layer(inputs)
    dropout = SharedDropout(layer, generator)
    first, again = layer(inputs), layer(inputs)
    dropout.renew()
    other = layer(inputs)
    dropout.remove()
    assert dropout.modules == 1 and torch.equal(first, again) and torch.allclose(first[0], first[4]) and not torch.equal(first, other)
    assert bool(((first == 0) | torch.isclose(first, 2 * plain)).all()) and bool((first == 0).any()) and torch.equal(layer(inputs), plain)

    heavy = [v for v in VARIANTS if v.endswith("_80nan")]
    same = {v: "equivalent" for v in VARIANTS}
    assert reading({**same, **{v: "better" for v in heavy[:4]}})[0] == "supported"
    assert reading({**same, **{v: "better" for v in heavy[:4]}, VARIANTS[0]: "worse"})[0] == "mixed"
    assert reading({**same, **{v: "better" for v in heavy[:3]}})[0] == "mixed"
    assert reading({**same, heavy[0]: "better"})[0] == "refuted"
    assert reading({**same, **{v: "better" for v in heavy}, **{v: "worse" for v in VARIANTS[:3]}})[0] == "refuted"
    print("check passed:", share, f"copula correlation {together:.3f}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv == ["--check"]:
        check()
    elif argv[:2] == ["train", "H"] and len(argv) in (4, 5) and argv[4:] in ([], ["--no-mlflow"]):
        train(argv[2], int(argv[3]), argv[4:] == ["--no-mlflow"])
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    elif argv[:1] == ["cells"] and len(argv) == 3:
        cells(argv[1], int(argv[2]))
    elif argv == ["--tally"]:
        tally()
    else:
        sys.exit(__doc__)
