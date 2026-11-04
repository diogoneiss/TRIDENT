"""C15 Stage 1's readers (docs/tickets/c15-mnar-prediction/03-stage1.md): gate G1 and the development tally.

    uv run --python 3.11 python -B scripts/experiments/c15_report.py --g1      # on F5, short mode
    uv run --python 3.11 python -B scripts/experiments/c15_report.py --g1c     # G1 again, A2 − A3, decision mode
    uv run --python 3.11 python -B scripts/experiments/c15_report.py --tally   # Stage 1b, descriptive
    uv run --python 3.11 python -B scripts/experiments/c15_report.py --check   # reloaded models equal their readouts
    uv run --python 3.11 python -B scripts/experiments/c15_report.py --h2      # Stage 3: H2, read once (ticket 05)

Cells come from both machines' state directories (medusa1's copied under ``<state>/medusa``). A fold's
readout holds its test rows' probabilities and its kept state; macro-F1 is recomputed from the
probabilities' argmax, which is the run's own prediction. Intervals are t intervals over the cells with the
Nadeau–Bengio correction, variance x (1/J + n_test/n_train) with n_test/n_train = 0.2/0.7 at five folds; the
plain one is printed beside it.

**G1's contrast:** for each test row whose noise cell (``c15_noise``) is a gap, A1's logit P(y = 1) with the
gap as ``[NULL]`` minus the logit of the mean P(y = 1) over K = 20 MICE draws of that cell given the row (an
``IterativeImputer(sample_posterior=True)`` fitted once, seed 0, on the fold's training rows' features, then K
posterior draws by K transforms), the row's other gaps left ``[NULL]``; averaged over the fold's gap cells.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import f1_score, log_loss  # type: ignore[attr-defined]

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import c15_run as run  # noqa: E402

LOCAL = run.STATE
ROOTS = (LOCAL, LOCAL / "medusa") if socket.gethostname() == "gorgona8" else (LOCAL,)
RATIO = 0.2 / 0.7
F1_MARGIN, LL_MARGIN = -0.01, 0.02
NOISE = "c15_noise"
K = 20


def readouts(arm: str, variant: str, seed: int) -> list[dict[str, Any]] | None:
    for root in ROOTS:
        if run.done(arm, variant, seed, root):
            d = run.cell_dir(arm, variant, seed, root) / "metrics"
            return [torch.load(d / f"c15_readout_fold_{k}.pt", weights_only=False) for k in range(1, 6)]
    return None


def interval(values: np.ndarray, corrected: bool = True) -> tuple[float, float, float]:
    n = len(values)
    factor = (1 / n + RATIO) if corrected else 1 / n
    half = float(stats.t.ppf(0.975, n - 1) * np.sqrt(factor * values.var(ddof=1)))
    mean = float(values.mean())
    return mean, mean - half, mean + half


def scores(folds: list[dict[str, Any]], probabilities: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per fold, (macro-F1, unweighted log loss) of the test rows."""
    f1, ll = [], []
    for k, fold in enumerate(folds):
        p = fold["test_probabilities"] if probabilities is None else probabilities[k]
        y = fold["test_labels"]
        classes = np.arange(p.shape[1])
        f1.append(f1_score(y, p.argmax(1), average="macro"))
        ll.append(log_loss(y, p, labels=classes))
    return np.asarray(f1), np.asarray(ll)


def model_for(variant: str, seed: int, arm: str, state: dict[str, torch.Tensor]) -> tuple[Any, Any]:
    """The kept classifier of a fold, rebuilt from its configuration and state, and the prepared dataset."""
    from src.embedder import TabularEmbedder
    from src.models import TridentModel
    from src.training.c15_inputs import has_indicator
    from src.training.config import build_training_parser, logged_hyperparameters, resolve_training_request
    from src.training.data import prepare_dataset
    from src.training.types import DatasetSpec, Hyperparameters, task_defaults
    from src.transformer import TabularTransformerEncoder

    base = logged_hyperparameters(resolve_training_request(build_training_parser().parse_args(["--dataset_name", variant])))
    h = Hyperparameters.from_mapping({**task_defaults("classification"), **base,
                                      "PRETRAIN_OBJECTIVE": "embedding_normalized", "C15_ARM": arm})
    data = prepare_dataset(DatasetSpec.from_name(variant, "class"))
    features = data.frame.drop(columns=[data.label_column])
    embedder = TabularEmbedder(  # type: ignore[no-untyped-call]
        df=features, categorical_columns=data.categorical_columns, numerical_columns=data.numerical_columns,
                               dimensao=h.dimension, hidden_dim=h.hidden_dimension,
                               batched_lookup=h.embedding_lookup == "batched", indicator_channel=has_indicator(arm))
    transformer = TabularTransformerEncoder(  # type: ignore[no-untyped-call]
        d_model=embedder.dimensao, nhead=h.heads, num_layers=h.layers,
                                            dim_feedforward=h.feedforward_dimension, dropout=h.dropout,
                                            final_norm=h.encoder_final_norm)
    from src.training.c15_inputs import blocked_position

    transformer.blocked_key = blocked_position(arm, list(data.categorical_columns), list(data.numerical_columns))
    model = TridentModel(  # type: ignore[no-untyped-call]
        embedder=embedder, transformer=transformer, num_labels=len(data.label_classes),
                         class_weights=torch.ones(len(data.label_classes)))
    model.load_state_dict(state)
    return model.eval(), data


def probabilities(model: Any, frame: Any) -> np.ndarray:
    from src.utils import preprocess_table

    with torch.no_grad():
        encoded = model.embedder.encode(preprocess_table(frame, p_base=0.0, fine_tunning=True))  # type: ignore[no-untyped-call]
        return np.asarray(torch.softmax(model(encoded), dim=1).numpy())


def check() -> None:
    """Every A1 F5 fold reloads to the probabilities it wrote."""
    for variant in run.F5:
        folds = readouts("A1", variant, 42)
        if folds is None:
            print(f"- {variant}: missing")
            continue
        worst = 0.0
        for fold in folds:
            model, data = model_for(variant, 42, "A1", fold["state_dict"])
            frame = data.frame.drop(columns=[data.label_column]).iloc[fold["test_indices"]].reset_index(drop=True)
            worst = max(worst, float(np.abs(probabilities(model, frame) - fold["test_probabilities"]).max()))
        print(f"- {variant}: largest |reloaded - written| test probability {worst:.2e}")


def contrast(variant: str, fold_number: int, fold: dict[str, Any], seed: int = 42) -> float:
    """G1's direct contrast on the noise column for one A1 fold (module docstring)."""
    from sklearn.experimental import enable_iterative_imputer  # type: ignore[import-untyped]  # noqa: F401
    from sklearn.impute import IterativeImputer  # type: ignore[attr-defined]

    from src.training.data import build_folds

    model, data = model_for(variant, seed, "A1", fold["state_dict"])
    split = build_folds(data.frame, data.label_column, 5, seed)[fold_number - 1]
    torch.set_num_threads(1)
    assert np.array_equal(np.asarray(split.test_indices), fold["test_indices"])
    columns = list(data.numerical_columns)
    features = data.frame[columns]
    train = features.iloc[np.asarray(split.train_indices)].to_numpy(dtype=float)
    test = features.iloc[np.asarray(split.test_indices)].reset_index(drop=True)
    rows = np.where(test[NOISE].isna().to_numpy())[0]
    if len(rows) == 0:
        return float("nan")
    j = columns.index(NOISE)
    p_null = probabilities(model, test.iloc[rows].reset_index(drop=True))[:, 1]
    drawn = np.zeros((K, len(rows)))
    imputer = IterativeImputer(sample_posterior=True, random_state=0).fit(train)
    for k in range(K):
        filled = imputer.transform(test.iloc[rows].to_numpy(dtype=float))  # a fresh posterior draw each call
        frame = test.iloc[rows].reset_index(drop=True).copy()
        frame[NOISE] = filled[:, j]
        drawn[k] = probabilities(model, frame)[:, 1]
    eps = 1e-7
    def logit(p: np.ndarray) -> np.ndarray:
        q = np.clip(p, eps, 1 - eps)
        return np.asarray(np.log(q) - np.log(1 - q))
    return float(np.mean(logit(p_null) - logit(drawn.mean(0))))


def g1() -> None:
    print("## G1 (F5, seed 42, 5 folds): (1) A2 − A4 relative test log loss below 0; (2) A1's noise-column contrast "
          "above 0; Nadeau–Bengio corrected 95% t intervals; passes if both hold on >= 2 of 3")
    passes = 0
    for variant in run.F5:
        a1, a2 = readouts("A1", variant, 42), readouts("A2", variant, 42)
        draws = [readouts(f"A4:{k}", variant, 42) for k in range(5)]
        if a1 is None or a2 is None or any(d is None for d in draws):
            print(f"- {variant}: missing\n\nPrimary (G1): incomplete")
            return
        pooled = [np.mean([d[k]["test_probabilities"] for d in draws if d is not None], axis=0) for k in range(5)]
        _, ll_a2 = scores(a2)
        _, ll_a4 = scores(draws[0], pooled)  # type: ignore[arg-type]
        rel = (ll_a2 - ll_a4) / ll_a4
        m, lo, hi = interval(rel)
        c = np.asarray([contrast(variant, k + 1, f) for k, f in enumerate(a1)])
        cm, clo, chi = interval(c)
        ok = hi < 0 and clo > 0
        passes += ok
        print(f"- {variant}: A2 − A4 {m:+.4f} [{lo:+.4f}, {hi:+.4f}] (plain {interval(rel, False)[1]:+.4f}); "
              f"A1 contrast {cm:+.3f} [{clo:+.3f}, {chi:+.3f}] logit -> {'pass' if ok else 'fail'}")
    print(f"\nPrimary (G1): {passes} of {len(run.F5)} -> {'passes' if passes >= 2 else 'fails'}")


def g1c() -> None:
    """G1 again (ticket 07, E128): A2 against A3 and A1's contrast, each over 15 cells (seeds 42/7/13 x 5 folds)."""
    print("## G1 again (E128, F5, seeds 42/7/13 x 5 folds): (1) A2 − A3 relative test log loss below 0; (2) A1's "
          "noise-column contrast above 0; Nadeau–Bengio corrected 95% t intervals over 15 cells; passes if both "
          "hold on >= 2 of 3")
    passes = 0
    for variant in run.F5:
        rel: list[np.ndarray] = []
        c: list[float] = []
        seeds: list[str] = []
        for seed in run.SEEDS:
            a1, a2, a3 = (readouts(arm, variant, seed) for arm in ("A1", "A2", "A3"))
            if a1 is None or a2 is None or a3 is None:
                print(f"- {variant} s{seed}: missing\n\nPrimary (G1 again): incomplete")
                return
            (_, ll_a2), (_, ll_a3) = scores(a2), scores(a3)
            r = (ll_a2 - ll_a3) / ll_a3
            rel.append(r)
            seeds.append(f"s{seed} {r.mean():+.4f}")
            c.extend(contrast(variant, k + 1, f, seed) for k, f in enumerate(a1))
        m, lo, hi = interval(np.concatenate(rel))
        cm, clo, chi = interval(np.asarray(c))
        _, plo, phi = interval(np.concatenate(rel), False)
        ok = hi < 0 and clo > 0
        passes += ok
        print(f"- {variant}: A2 − A3 {m:+.4f} [{lo:+.4f}, {hi:+.4f}] (plain [{plo:+.4f}, {phi:+.4f}]; "
              f"{', '.join(seeds)}); A1 contrast {cm:+.3f} [{clo:+.3f}, {chi:+.3f}] logit -> {'pass' if ok else 'fail'}")
    print(f"\nPrimary (G1 again): {passes} of {len(run.F5)} -> {'passes' if passes >= 2 else 'fails'}")


def tally() -> None:
    print("## Stage 1b (descriptive): A1 − A2 per variant, 15 cells; macro-F1 difference and relative log loss "
          "(LL_A1 − LL_A2) / LL_A2, Nadeau–Bengio corrected (plain); H2's per-variant call at −0.01 / +2%")
    missing = 0
    for variant in run.DEVELOPMENT:
        f1d: list[np.ndarray] = []
        lld: list[np.ndarray] = []
        for seed in run.SEEDS:
            a1, a2 = readouts("A1", variant, seed), readouts("A2", variant, seed)
            if a1 is None or a2 is None:
                missing += 1
                break
            f1_1, ll_1 = scores(a1)
            f1_2, ll_2 = scores(a2)
            f1d.append(f1_1 - f1_2)
            lld.append((ll_1 - ll_2) / ll_2)
        else:
            f, ll = np.concatenate(f1d), np.concatenate(lld)
            fm, flo, fhi = interval(f)
            lm, llo, lhi = interval(ll)
            agree_f = all(x.mean() > F1_MARGIN for x in f1d)
            agree_l = all(x.mean() < LL_MARGIN for x in lld)
            if flo > F1_MARGIN and lhi < LL_MARGIN and agree_f and agree_l:
                call = "superior" if flo > 0 and lhi < 0 else "non-inferior"
            elif fhi < F1_MARGIN or llo > LL_MARGIN:
                call = "inferior"
            else:
                call = "inconclusive"
            print(f"- {variant}: F1 {fm:+.4f} [{flo:+.4f}, {fhi:+.4f}] (plain [{interval(f, False)[1]:+.4f}, "
                  f"{interval(f, False)[2]:+.4f}]); LL {lm:+.4f} [{llo:+.4f}, {lhi:+.4f}] -> {call}")
    if missing:
        print(f"\n- missing: {missing} (variant, seed) pairs\n\nPrimary: incomplete")


def h2() -> None:
    """Ticket 05's rule: per-variant calls over P's fresh draws, then the pooled D_c, the guard and the call."""
    print("## H2 (E126, confirmation, read once): A1 − A2 on P's fresh draws, 15 cells (seeds 1515/2626/3737 x 5 "
          "folds), Nadeau–Bengio corrected 95% t; margins −0.01 macro-F1, +0.02 relative log loss")
    per_f1: dict[str, np.ndarray] = {}
    per_ll: dict[str, np.ndarray] = {}
    calls: dict[str, str] = {}
    for variant in run.P_FRESH:
        f1d, lld = [], []
        for seed in run.CONFIRMATION_SEEDS:
            a1, a2 = readouts("A1", variant, seed), readouts("A2", variant, seed)
            if a1 is None or a2 is None:
                print(f"- {variant} s{seed}: missing\n\nPrimary (H2): incomplete")
                return
            f1_1, ll_1 = scores(a1)
            f1_2, ll_2 = scores(a2)
            f1d.append(f1_1 - f1_2)
            lld.append((ll_1 - ll_2) / ll_2)
        f, ll = np.concatenate(f1d), np.concatenate(lld)
        per_f1[variant], per_ll[variant] = f, ll
        fm, flo, fhi = interval(f)
        lm, llo, lhi = interval(ll)
        agree = all(x.mean() > F1_MARGIN for x in f1d) and all(x.mean() < LL_MARGIN for x in lld)
        if flo > F1_MARGIN and lhi < LL_MARGIN and agree:
            calls[variant] = "non-inferior"
        elif fhi < F1_MARGIN or llo > LL_MARGIN:
            calls[variant] = "inferior"
        else:
            calls[variant] = "inconclusive"
        print(f"- {variant}: F1 {fm:+.4f} [{flo:+.4f}, {fhi:+.4f}]; LL {lm:+.4f} [{llo:+.4f}, {lhi:+.4f}]; seeds F1 "
              + ", ".join(f"{x.mean():+.4f}" for x in f1d) + "; LL " + ", ".join(f"{x.mean():+.4f}" for x in lld)
              + f" -> {calls[variant]}")
    pooled_f1 = np.mean(list(per_f1.values()), axis=0)
    pooled_ll = np.mean(list(per_ll.values()), axis=0)
    fm, flo, fhi = interval(pooled_f1)
    lm, llo, lhi = interval(pooled_ll)
    inferior = [v for v, c in calls.items() if c == "inferior"]
    if flo > F1_MARGIN and lhi < LL_MARGIN and not inferior:
        verdict = "supported"
        if flo > 0 and lhi < 0:
            verdict += "; H2-sup: superior on both"
    elif fhi < F1_MARGIN or llo > LL_MARGIN:
        verdict = "reversed"
    else:
        verdict = "inconclusive" + (f" (guard: inferior on {', '.join(inferior)})" if inferior else "")
    print(f"\nPooled over P ({len(per_f1)} variants): F1 {fm:+.4f} [{flo:+.4f}, {fhi:+.4f}] (plain "
          f"[{interval(pooled_f1, False)[1]:+.4f}, {interval(pooled_f1, False)[2]:+.4f}]); LL {lm:+.4f} [{llo:+.4f}, {lhi:+.4f}] "
          f"(plain [{interval(pooled_ll, False)[1]:+.4f}, {interval(pooled_ll, False)[2]:+.4f}])")
    print(f"Primary (H2): {verdict}")
    n = len(calls)
    noninf = sum(c == "non-inferior" for c in calls.values())
    count = ("supported" if noninf >= -(-2 * n // 3) and not inferior else
             "reversed" if len(inferior) >= -(-n // 3) else "inconclusive")
    print(f"- sensitivity, count rule: non-inferior on {noninf} of {n}, inferior on {len(inferior)} -> {count}")
    tables: dict[str, list[str]] = {}
    for v in per_f1:
        tables.setdefault(v.split("_")[0], []).append(v)
    print("- sensitivity, per table: " + "; ".join(
        f"{t} F1 {np.mean([per_f1[v].mean() for v in vs]):+.4f}, LL {np.mean([per_ll[v].mean() for v in vs]):+.4f}"
        for t, vs in tables.items()))


if __name__ == "__main__":
    if sys.argv[1:] == ["--g1"]:
        g1()
    elif sys.argv[1:] == ["--g1c"]:
        g1c()
    elif sys.argv[1:] == ["--tally"]:
        tally()
    elif sys.argv[1:] == ["--h2"]:
        h2()
    elif sys.argv[1:] == ["--check"]:
        check()
    else:
        sys.exit(__doc__)
