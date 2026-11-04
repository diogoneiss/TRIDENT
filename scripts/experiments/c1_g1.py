"""C1's gate G1, the manipulation check (ticket docs/tickets/c1-split/01-g1-probe.md), CPU only.

    uv run --python 3.11 python -B scripts/experiments/c1_g1.py

Per primary table, on the decision mode's 15 folds (seeds 42/7/13 x ``build_folds`` at 5), a logistic regression
on the 0/1 gap matrix alone is fitted on the fold's training rows and scored on its validation rows (AUROC,
one-vs-rest macro on vehicle; macro-F1 beside), on the MNAR file and on its MCAR sibling, whose folds are the
same. The gate passes if the pooled MNAR − MCAR AUROC difference (the mean over the four tables per replicate)
has its plain 95% t interval above 0 and no table's mean difference is negative. Test rows are never read.
The result is written to ``<state>/g1.json`` for the primary's reader.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score  # type: ignore[attr-defined]

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import c1_run as run  # noqa: E402
from scripts.experiments.c15_stage0 import TARGET, load  # noqa: E402
from src.training.data import build_folds  # noqa: E402


def probe(variant: str, seed: int) -> list[tuple[float, float]]:
    """(AUROC, macro-F1) of the indicators-only probe on each of the seed's 5 validation folds."""
    x, y = load(variant)
    indicators = x.isna().to_numpy(dtype=float)
    classes = np.unique(y)
    out = []
    for split in build_folds(pd.DataFrame({TARGET: y}), TARGET, 5, seed):
        train, valid = np.asarray(split.train_indices), np.asarray(split.validation_indices)
        model = LogisticRegression(max_iter=5000).fit(indicators[train], y[train])
        p = model.predict_proba(indicators[valid])
        auroc = roc_auc_score(y[valid], p[:, 1]) if len(classes) == 2 else roc_auc_score(y[valid], p, multi_class="ovr", average="macro")
        out.append((float(auroc), float(f1_score(y[valid], classes[p.argmax(1)], average="macro"))))
    return out


def main() -> None:
    print("## G1 (indicators-only probe, validation rows, 15 folds a table): MNAR − MCAR AUROC pooled over the four "
          "tables, plain 95% t interval above 0 and no table negative")
    per_table: dict[str, np.ndarray] = {}
    for table in run.TABLES:
        mnar = np.asarray([v for s in run.SEEDS for v in probe(f"{table}_probit_60nan", s)])
        mcar = np.asarray([v for s in run.SEEDS for v in probe(f"{table}_60nan", s)])
        diff = mnar[:, 0] - mcar[:, 0]
        per_table[table] = diff
        half = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
        print(f"- {table}: AUROC MNAR {mnar[:, 0].mean():.3f}, MCAR {mcar[:, 0].mean():.3f}, difference {diff.mean():+.4f} "
              f"[{diff.mean() - half:+.4f}, {diff.mean() + half:+.4f}]; macro-F1 MNAR {mnar[:, 1].mean():.3f}, MCAR {mcar[:, 1].mean():.3f}")
    pooled = np.mean([per_table[t] for t in run.TABLES], axis=0)
    half = stats.t.ppf(0.975, len(pooled) - 1) * pooled.std(ddof=1) / np.sqrt(len(pooled))
    lo, hi = pooled.mean() - half, pooled.mean() + half
    negative = [t for t in run.TABLES if per_table[t].mean() < 0]
    passed = lo > 0 and not negative
    print(f"\nPooled: {pooled.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]; tables with a negative mean: {negative or 'none'}")
    print(f"Primary (G1): {'passes' if passed else 'fails'}")
    run.STATE.mkdir(parents=True, exist_ok=True)
    (run.STATE / "g1.json").write_text(json.dumps({"passed": passed, "pooled": [pooled.mean(), lo, hi],
                                                  "tables": {t: float(per_table[t].mean()) for t in run.TABLES}}, indent=1))


if __name__ == "__main__":
    main()
