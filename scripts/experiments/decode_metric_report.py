"""E145's reading: arm B (checkpoint by the distributional score) minus arm A (by ``impute_score``)
(docs/tickets/imputation-protocol/51-decode-metric.md). Read-only.

    uv run --python 3.11 python -B scripts/experiments/decode_metric_report.py [<state_dir>]

Both arms are scored the same way, offline, from their five fold ledgers (K's induced fills): the folds
are rebuilt as a run builds them for the mean/mode fills, then ``distribution_scores`` and ``score_cells``
read each ledger. Differences are B − A per (seed, fold) replicate, lower is better for every score;
pooled over the variants per replicate, plain 95% t interval with Nadeau-Bengio beside. Run from the
checkout root.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import c15_report, decode_metric_run as run  # noqa: E402
from src.training.data import build_folds, prepare_dataset  # noqa: E402
from src.training.distribution_metrics import distribution_scores  # noqa: E402
from src.training.imputation_metrics import NUMERICAL, mean_mode_baselines, score_cells  # noqa: E402
from src.training.types import DatasetSpec  # noqa: E402

MARGIN = 0.01  # the project's smallest effect of interest on impute_score
SCORES = ("dist_score", "impute_score", "wd_num_z", "tv_cat", "dispersion")


def _ledger(state: Path, arm: str, variant: str, seed: int, fold: int) -> pd.DataFrame | None:
    found = sorted((state / "results" / f"{variant}_s{seed}_{arm}" / variant).glob(f"*/imputation/imputation_fold_{fold}_cells.csv"))
    if len(found) != 1:
        return None
    frame = pd.read_csv(found[0], dtype={"actual": object, "imputed": object})
    frame = frame[frame["population"] == "induced"].reset_index(drop=True)
    numeric = frame["kind"] == NUMERICAL
    frame.loc[numeric, "actual"] = frame.loc[numeric, "actual"].astype(float)
    frame.loc[numeric, "imputed"] = frame.loc[numeric, "imputed"].astype(float)
    return frame


def _scores(cells: pd.DataFrame, naive: Mapping[str, float | str]) -> dict[str, float]:
    out = {**distribution_scores(cells, naive), "impute_score": float(score_cells(cells, naive).metrics["impute_score"])}
    spreads = []
    for _, group in cells[cells["kind"] == NUMERICAL].groupby("column"):
        actual, imputed = group["actual"].to_numpy(dtype=float), group["imputed"].to_numpy(dtype=float)
        if len(actual) > 1 and actual.std() > 0:
            spreads.append(float(imputed.std() / actual.std()))
    out["dispersion"] = float(np.mean(spreads)) if spreads else float("nan")  # the panel's: sd of fills over sd of truths
    return out


def _kept_epoch(state: Path, arm: str, variant: str, seed: int) -> float:
    found = sorted((state / "results" / f"{variant}_s{seed}_{arm}" / variant).glob("*/metrics/raw_fold_metrics.csv"))
    return float(pd.read_csv(found[0])["decode/induced_checkpoint_epoch"].mean()) if len(found) == 1 else float("nan")


def differences(state: Path, variant: str) -> dict[str, np.ndarray] | None:
    """Per score, B − A over the variant's 15 replicates (seed-major, then fold); None if a ledger is missing."""
    dataset = prepare_dataset(DatasetSpec.from_name(variant, None))
    features = dataset.frame.drop(columns=[dataset.label_column])
    numerical, categorical = list(dataset.numerical_columns), list(dataset.categorical_columns)
    rows: dict[str, list[float]] = {f"{score}{side}": [] for score in SCORES for side in ("", "_A", "_B")}
    for seed in run.SEEDS:
        for ordinal, fold in enumerate(build_folds(dataset.frame, dataset.label_column, 5, seed), start=1):
            naive = mean_mode_baselines(features.iloc[np.asarray(fold.train_indices)].reset_index(drop=True), numerical, categorical)
            a, b = _ledger(state, "A", variant, seed, ordinal), _ledger(state, "B", variant, seed, ordinal)
            if a is None or b is None:
                print(f"- {variant} s{seed} fold {ordinal}: missing")
                return None
            if not (a[["row", "column"]].equals(b[["row", "column"]]) and a["actual"].astype(str).equals(b["actual"].astype(str))):
                sys.exit(f"{variant} s{seed} fold {ordinal}: the arms' ledgers do not hold the same cells")
            sa, sb = _scores(a, naive), _scores(b, naive)
            for score in SCORES:
                va, vb = sa.get(score, float("nan")), sb.get(score, float("nan"))
                rows[f"{score}_A"].append(va)
                rows[f"{score}_B"].append(vb)
                rows[score].append(vb - va)
    return {name: np.asarray(values) for name, values in rows.items()}


def main(argv: list[str]) -> int:
    state = Path(argv[0]) if argv else run.STATE
    print("## Decode metric (E145): B (checkpoint by dist_score) − A (by impute_score), K's induced fills from the fold ledgers, "
          "seeds 42/7/13 × 5 folds; lower is better; pooled over the 12 variants per replicate")
    per_variant = {}
    for variant in run.VARIANTS:
        d = differences(state, variant)
        if d is None:
            print("\nDecode metric: incomplete")
            return 1
        per_variant[variant] = d
        flags = []
        for score in ("dist_score", "impute_score"):
            m, lo, hi = c15_report.interval(d[score], False)
            flags.append(f"{score} {m:+.4f} [{lo:+.4f}, {hi:+.4f}] {'better' if hi < 0 else 'worse' if lo > 0 else 'no difference'}")
        print(f"- {variant}: {'; '.join(flags)}; A dist {d['dist_score_A'].mean():.4f} impute {d['impute_score_A'].mean():.4f} "
              f"dispersion {np.nanmean(d['dispersion_A']):.3f} -> B dispersion {np.nanmean(d['dispersion_B']):.3f}; "
              f"folds with another checkpoint {int((d['impute_score'] != 0).sum())}/15; mean kept epoch "
              f"A {np.nanmean([_kept_epoch(state, 'A', variant, s) for s in run.SEEDS]):.0f} B {np.nanmean([_kept_epoch(state, 'B', variant, s) for s in run.SEEDS]):.0f}")
    pooled = {score: np.nanmean([per_variant[v][score] for v in run.VARIANTS], axis=0) for score in SCORES}
    for score in SCORES:
        m, lo, hi = c15_report.interval(pooled[score], False)
        _, nlo, nhi = c15_report.interval(pooled[score])
        print(f"Pooled B − A {score}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] (NB [{nlo:+.4f}, {nhi:+.4f}])")
    _, _, dist_hi = c15_report.interval(pooled["dist_score"], False)
    _, cost_lo, cost_hi = c15_report.interval(pooled["impute_score"], False)
    improves, within = dist_hi < 0, cost_hi < MARGIN
    verdict = ("SUPPORTED" if improves and within else "improves the distribution at a cost in point accuracy" if improves
               else "not supported")
    print(f"Primary: dist_score {'improves' if improves else 'does not improve'} (interval below 0: {improves}); "
          f"impute_score cost {'inside' if within else 'not inside'} +{MARGIN} (upper bound {cost_hi:+.4f}, lower {cost_lo:+.4f}) -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
