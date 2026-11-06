"""T50's two scored studies, read from their state directories (docs/tickets/mimic/03 and 04).

    QUEUE_STATE=<state directory> TRIDENT_DATASETS_ROOT=<root> \\
        uv run --python 3.11 python -B scripts/experiments/mimic_report.py <e148|e149>

- ``e148``: per natural table, the masked population's ``impute_score`` minus the bar's best baseline on the same
  cells, and the scores of the sparse columns.
- ``e149``: per donor table, the same contrast on ``tmatch``'s induced cells, and ``tmatch − trand`` and
  ``trand − mcar`` for the decoder's score and for its gap to the bar.

Each contrast is paired per (seed, fold), 15 replicates, with a plain 95% t interval. A primary is better (or worse)
when its two-sided p, adjusted by Holm over the two tables, is under 0.05 and its mean below (or above) 0. Refuses to
read a study with a cell missing. Run from the checkout root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import report_common  # noqa: E402

Floats = NDArray[np.float64]
SPARSE = ("lactate", "troponin_t", "abp_systolic", "abp_diastolic", "abp_mean", "bilirubin_max", "pf_ratio", "uo_24hr")


def line(name: str, d: Floats) -> tuple[str, float]:
    low, high = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    return f"{name} (n={len(d)}): {d.mean():+.4f} [{low:+.4f}, {high:+.4f}]", float(stats.ttest_1samp(d, 0.0).pvalue)


def verdicts(primaries: dict[str, Floats]) -> None:
    """The primaries, Holm over them: better or worse when the adjusted two-sided p is under 0.05, by the sign."""
    lines = {name: line(name, d) for name, d in primaries.items()}
    adjusted = report_common.holm({name: p for name, (_, p) in lines.items()})
    for name, d in primaries.items():
        call = "no different" if adjusted[name] >= 0.05 else "better" if d.mean() < 0 else "worse"
        print(f"PRIMARY {lines[name][0]}, Holm p {adjusted[name]:.4f} -> {call}")


def main(study: str) -> None:
    from scripts.experiments import mimic_run as run  # needs QUEUE_STATE and TRIDENT_DATASETS_ROOT

    if study == "e149":
        from scripts.experiments import mimic_transplant_run  # noqa: F401  (sets the variants and the arm)
    population, readout = ("masked", "M") if study == "e148" else ("induced", "K")
    arm = next(iter(run.ARMS))
    gaps: dict[str, Floats] = {}
    model: dict[str, Floats] = {}
    for variant in run.VARIANTS:
        own, bar, names = [], [], set()
        for seed in run.SEEDS:
            runs = run.finished(arm, variant, seed)
            if len(runs) != 1:
                sys.exit(f"{variant} seed {seed}: {len(runs)} finished runs; the tickets read only complete studies")
            frame = pd.read_csv(runs[0] / "metrics" / "raw_fold_metrics.csv").sort_values("fold").reset_index(drop=True)
            assert len(frame) == 5
            source = report_common.Run(variant, seed, "", "masked" if study == "e148" else "induced")
            own.append(report_common.model(frame, source, readout, population))
            name, values = report_common.opponent(frame, "bar", population)
            bar.append(values)
            names.add(name)
        model[variant], gaps[variant] = np.concatenate(own), np.concatenate(own) - np.concatenate(bar)
        print(f"{variant}: decoder {model[variant].mean():.4f}, bar {np.concatenate(bar).mean():.4f} "
              f"(best baseline by run: {sorted(names)})")

    if study == "e148":
        verdicts({variant: gaps[variant] for variant in run.VARIANTS})
        for variant in run.VARIANTS:
            columns = pd.concat(
                pd.read_csv(run.finished(arm, variant, seed)[0] / "metrics" / "per_column_imputation.csv")
                for seed in run.SEEDS
            )
            sparse = columns[(columns["population"] == population) & columns["column"].isin(SPARSE)]
            print(f"\n{variant}, sparse columns on the {population} cells (mean over 15 folds):")
            print(sparse.pivot_table(index="column", columns="metric", values="value").round(4).to_string())
    else:
        verdicts({variant: gaps[variant] for variant in run.VARIANTS if "_tmatch_" in variant})
        for table in ("mimic-cardio-donor", "mimic-sofa-donor"):
            for a, b in (("tmatch", "trand"), ("trand", "mcar")):
                for what, scores in (("decoder's score", model), ("gap to the bar", gaps)):
                    difference = scores[f"{table}_{a}_20nan"] - scores[f"{table}_{b}_20nan"]
                    print(line(f"{table} {a} - {b}, {what}", difference)[0])


if __name__ == "__main__":
    if sys.argv[1:] not in (["e148"], ["e149"]):
        sys.exit(__doc__)
    main(sys.argv[1])
