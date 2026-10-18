"""Pre-registered analysis of E42, the baselines with a validation-only tuning budget (imputation-protocol/02).

Read-only. Run from the checkout root:

    uv run --python 3.11 python scripts/experiments/baseline_budget_report.py [--tuned DIR]

Per seed set of the ticket's cell map (``report_common.development_source``) and variant, on the
induced ``impute_score`` over 15 fold pairs with the verdict rule and the 1% SESOI (ADR 0020):

1. the tuned bar (the tuned weak tier's best, ``tuned_bar``) against the untuned bar of 5: what the
   budget buys;
2. K against the tuned bar, beside K against the untuned bar; a variant whose verdict changes
   between the two bars is named.

Secondary: the masked-selection sensitivity (each family's setting chosen on the fixed validation
mask, ``<family>_tuned_masked``, and their best on the run's CV mean), the masked population, and
how often each family kept its current setting. The ticket's cells are seed sets 101/202/303 and
404/505/606; the driver also ran 42/7/13, reported after them and never pooled with them. Labelled
variants on 42/7/13 are marked and left out of the counts, as in E41's report.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments.thesis_table import DEVELOPMENT, SESOI, _cells  # noqa: E402
from src.training.tuned_baselines import grid  # noqa: E402

DEFAULT_TUNED = Path("/scratch2/diogoneiss/trident-2026-10-17-baseline-budget")
ORDER = ("101/202/303", "404/505/606", "42/7/13")
PRE_REGISTERED = ("101/202/303", "404/505/606")
FAMILIES = ("knn", "hgb", "lgbm")
Cells = list[tuple[common.Run, pd.DataFrame]]


def _masked_selection_bar(frame: pd.DataFrame, population: str) -> np.ndarray:
    """The best of mean/mode and the masked-selected tuned families, by the run's CV mean."""
    names = ["mean_mode", *(f"{family}_tuned_masked" for family in FAMILIES)]
    keys = [f"impute/{population}/baseline/{name}/impute_score" for name in names]
    best = min(keys, key=lambda key: float(frame[key].mean()))
    return np.asarray(frame[best], dtype=float)


def _values(run: common.Run, frame: pd.DataFrame, name: str, population: str) -> np.ndarray:
    if name == "K":
        return common.model(frame, run, "K", population)
    if name == "tuned":
        return np.asarray(frame[f"impute/{population}/baseline/tuned_bar/impute_score"], dtype=float)
    if name == "tuned_masked":
        return _masked_selection_bar(frame, population)
    return common.opponent(frame, name, population)[1]


def contrast(cells: Cells, first: str, second: str, population: str = "induced") -> common.Verdict:
    per_seed, scale = [], []
    for run, frame in cells:
        a, b = _values(run, frame, first, population), _values(run, frame, second, population)
        per_seed.append(a - b)
        scale.append(float(b.mean()))
    return common.verdict(per_seed, sesoi=SESOI * float(np.mean(scale)))


def _defaults_kept(choices: Path, variant: str, seeds: list[int]) -> dict[str, float]:
    """Per family, the share of folds whose chosen setting is the family's current one, unshrunk."""
    kept: dict[str, list[bool]] = {family: [] for family in FAMILIES}
    for seed in seeds:
        path = choices / f"{variant}_s{seed}.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        for family in FAMILIES:
            first = grid(family)[0].label
            kept[family].extend(str(label) in (first, f"{first}_a0.0") for label in table[f"{family}_tuned"])
    return {family: float(np.mean(flags)) for family, flags in kept.items() if flags}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned", type=Path, default=DEFAULT_TUNED)
    options = parser.parse_args()
    csv, choices = options.tuned / "csv", options.tuned / "choices"
    bar_names: Counter[str] = Counter()
    for seeds in ORDER:
        label = "pre-registered" if seeds in PRE_REGISTERED else "run beside them, not pooled"
        print(f"## Seeds {seeds} ({label})")
        counts: dict[str, list[str]] = {}
        flips: list[str] = []
        for variant in DEVELOPMENT:
            cells = _cells(variant, seeds, csv)
            if cells is None or "impute/induced/baseline/tuned_bar/impute_score" not in cells[0][1].columns:
                continue
            source = common.development_source(variant, seeds)
            labelled = source is not None and source.labelled
            readings = {
                "tuned bar vs bar": contrast(cells, "tuned", "bar"),
                "K vs bar": contrast(cells, "K", "bar"),
                "K vs tuned bar": contrast(cells, "K", "tuned"),
                "K vs masked-selection tuned bar": contrast(cells, "K", "tuned_masked"),
                "masked: tuned bar vs bar": contrast(cells, "tuned", "bar", "masked"),
            }
            relative = 100 * readings["tuned bar vs bar"].mean / float(
                np.mean([common.opponent(frame, "bar")[1].mean() for _, frame in cells])
            )
            kept = _defaults_kept(choices, variant, [run.seed for run, _ in cells])
            for run, _ in cells:
                path = choices / f"{variant}_s{run.seed}.csv"
                if path.exists():
                    bar_names.update(str(name) for name in pd.read_csv(path)["tuned_bar/induced"])
            parts = [f"{name} {v.mean:+.4f} {v.call}" for name, v in readings.items()]
            print(f"- `{variant}`{' †' if labelled else ''}: " + "; ".join(parts)
                  + f"; budget {relative:+.1f}% of the bar; current setting kept "
                  + ", ".join(f"{family} {share:.0%}" for family, share in kept.items()))
            if labelled:
                continue
            for name, v in readings.items():
                counts.setdefault(name, []).append(v.call)
            if readings["K vs bar"].call != readings["K vs tuned bar"].call:
                flips.append(f"{variant} ({readings['K vs bar'].call} → {readings['K vs tuned bar'].call})")
        for name, calls in counts.items():
            print(f"  - {name}: {common.tally(calls)}")
        print(f"  - K's verdict changes between the bars: {flips or 'none'}")
        print()
    total = sum(bar_names.values())
    print("The tuned bar's family, over every fold read: " + ", ".join(
        f"{name} {count / total:.0%}" for name, count in bar_names.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
