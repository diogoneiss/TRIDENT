"""T50's scored studies, read from their state directories (docs/tickets/mimic/03, 04 and 14).

    QUEUE_STATE=<state directory> TRIDENT_DATASETS_ROOT=<root> \\
        uv run --python 3.11 python -B scripts/experiments/mimic_report.py <e148|e149|e176>

- ``e148``: per natural table, the masked population's ``impute_score`` minus the bar's best baseline on the same
  cells, every scored imputer's mean, the decoder against TabPFN where ``<state>/tabpfn`` holds the offline pass
  (``tabpfn_run.py <state>/tabpfn <variant> <seeds>``), the RMSE of the sparse columns, and the decoder's error by a
  row's count of natural gaps.
- ``e149``: per donor table, the same contrast on ``tmatch``'s induced cells, every scored imputer's mean, the
  decoder against TabPFN on ``tmatch`` where ``<state>/tabpfn`` holds the pass, and ``tmatch − trand`` and
  ``trand − mcar`` for the decoder's score and for its gap to the bar.
- ``e176``: on ``mimic-wide``, the decoder minus TabPFN v2 per variant (``<state>/tabpfn`` is required), by the
  project's rule with an equivalence margin of 1% of TabPFN's mean score, and the decoder minus the bar beside it.
  The primaries are the natural table and ``tmatch``.

Each contrast of ``e148`` and ``e149`` is paired per (seed, fold), 15 replicates, with a plain 95% t interval. A primary is better (or worse)
when its two-sided p, adjusted by Holm over the two tables, is under 0.05 and its mean below (or above) 0. Refuses to
read a study with a cell missing. Run from the checkout root.
"""

from __future__ import annotations

import os
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


def by_row_gaps(variant: str, runs: list[Path]) -> pd.DataFrame:
    """The decoder's error on the masked numerical cells, by how many natural gaps the cell's row has.

    Beside it, the error of filling 0 on the same cells: the cells are in training-set z units, so that is the
    training mean, and the ratio says how much of a group's error is the group's own spread.
    """
    table = pd.read_csv(Path(os.environ["TRIDENT_DATASETS_ROOT"]) / variant.rsplit("_", 1)[0] / f"{variant}.csv")
    row_gaps = table.drop(columns=["class"]).isna().sum(axis=1).to_numpy()
    parts = []
    for directory in runs:
        for fold in range(1, 6):
            cells = pd.read_csv(directory / "imputation" / f"imputation_fold_{fold}_cells.csv")
            split = pd.read_csv(directory / "imputation" / f"imputation_fold_{fold}_partition.csv")
            test = split.loc[split["split"] == "test", "row"].to_numpy()
            cells = cells[(cells["population"] == "masked") & (cells["kind"] == "numerical")]
            # The ledger numbers a row by its position among the fold's test rows.
            gaps = np.minimum(row_gaps[test[cells["row"].to_numpy()]], 4)
            actual, imputed = cells["actual"].to_numpy(float), cells["imputed"].to_numpy(float)
            parts.append(pd.DataFrame({"gaps": gaps, "decoder": (actual - imputed) ** 2, "mean": actual**2}))
    groups = pd.concat(parts).groupby("gaps")
    out: pd.DataFrame = groups[["decoder", "mean"]].mean().pow(0.5)
    out = out.rename(columns={"decoder": "rmse_z", "mean": "mean_fill_rmse_z"})
    out.insert(0, "cells", groups.size())
    out["ratio"] = out["rmse_z"] / out["mean_fill_rmse_z"]
    return out.rename(index={4: "4+"})


def wide() -> None:
    from scripts.experiments import mimic_run as run
    from scripts.experiments import mimic_wide_run

    calls = {}
    for arm, variant in mimic_wide_run.CELLS:
        population, readout = ("masked", "M") if arm == "M" else ("induced", "K")
        own, bar, theirs, seconds = [], [], [], []
        for seed in run.SEEDS:
            runs = run.finished(arm, variant, seed)
            if len(runs) != 1:
                sys.exit(f"{variant} seed {seed}: {len(runs)} finished runs; the ticket reads only a complete study")
            frame = pd.read_csv(runs[0] / "metrics" / "raw_fold_metrics.csv").sort_values("fold").reset_index(drop=True)
            other = pd.read_csv(run.STATE / "tabpfn" / f"{variant}_s{seed}.csv").sort_values("fold")
            assert len(frame) == 5 and len(other) == 5
            own.append(report_common.model(frame, report_common.Run(variant, seed, "", population), readout, population))
            bar.append(report_common.opponent(frame, "bar", population)[1])
            theirs.append(other[f"impute/{population}/baseline/tabpfn/impute_score"].to_numpy(float))
            seconds.append(other["baseline/tabpfn/fit_seconds"].to_numpy(float))
        tabpfn = float(np.concatenate(theirs).mean())
        print(f"{variant} ({population} cells): decoder {np.concatenate(own).mean():.4f}, TabPFN {tabpfn:.4f} "
              f"(fit {np.concatenate(seconds).mean():.0f} s a fold), bar {np.concatenate(bar).mean():.4f}")
        for name, others in (("TabPFN", theirs), ("bar", bar)):
            v = report_common.verdict([a - b for a, b in zip(own, others)], 0.01 * tabpfn)
            print(f"  decoder - {name}: {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}], by seed "
                  + ", ".join(f"{s:+.4f}" for s in v.seeds) + f" -> {v.call}")
            if name == "TabPFN":
                calls[variant] = v.call
    primaries = [calls[variant] for _, variant in mimic_wide_run.CELLS[:2]]
    print(f"PRIMARIES (natural, tmatch) against TabPFN: {primaries} -> "
          + ("refuted" if "worse" in primaries else "supported" if "better" in primaries else "no verdict")
          + f" ({report_common.RULE}, margin 1% of TabPFN's mean)")


def main(study: str) -> None:
    from scripts.experiments import mimic_run as run  # needs QUEUE_STATE and TRIDENT_DATASETS_ROOT

    if study == "e149":
        from scripts.experiments import mimic_transplant_run  # noqa: F401  (sets the variants and the arm)
    population, readout = ("masked", "M") if study == "e148" else ("induced", "K")
    arm = next(iter(run.ARMS))
    gaps: dict[str, Floats] = {}
    model: dict[str, Floats] = {}
    folds: dict[str, pd.DataFrame] = {}
    for variant in run.VARIANTS:
        own, bar, names, frames = [], [], set(), []
        for seed in run.SEEDS:
            runs = run.finished(arm, variant, seed)
            if len(runs) != 1:
                sys.exit(f"{variant} seed {seed}: {len(runs)} finished runs; the tickets read only complete studies")
            frame = pd.read_csv(runs[0] / "metrics" / "raw_fold_metrics.csv").sort_values("fold").reset_index(drop=True)
            assert len(frame) == 5
            frames.append(frame)
            source = report_common.Run(variant, seed, "", "masked" if study == "e148" else "induced")
            own.append(report_common.model(frame, source, readout, population))
            name, values = report_common.opponent(frame, "bar", population)
            bar.append(values)
            names.add(name)
        model[variant], gaps[variant] = np.concatenate(own), np.concatenate(own) - np.concatenate(bar)
        print(f"{variant}: decoder {model[variant].mean():.4f}, bar {np.concatenate(bar).mean():.4f} "
              f"(best baseline by run: {sorted(names)})")
        folds[variant] = pd.concat(frames)
        # every imputer the runs scored, the strong tier's too, as levels
        every = folds[variant].filter(regex=rf"^impute/{population}/(baseline/[^/]+/)?impute_score$").mean()
        every.index = pd.Index([name.split("/")[-2].replace(population, "decoder") for name in every.index])
        levels = ", ".join(f"{name} {value:.4f}" for name, value in every.sort_values().items())
        print(f"  mean impute_score over 15 folds: {levels}")

    primaries = [variant for variant in run.VARIANTS if study == "e148" or "_tmatch_" in variant]
    verdicts({variant: gaps[variant] for variant in primaries})
    passes = {variant: [run.STATE / "tabpfn" / f"{variant}_s{seed}.csv" for seed in run.SEEDS] for variant in primaries}
    # the offline pass (tabpfn_run.py) on the same cells, descriptive; read only once every file of it exists
    if all(path.exists() for paths in passes.values() for path in paths):
        for variant in primaries:
            theirs = pd.concat(pd.read_csv(path).sort_values("fold") for path in passes[variant])
            score = theirs[f"impute/{population}/baseline/tabpfn/impute_score"].to_numpy(float)
            print(f"TabPFN {score.mean():.4f}; " + line(f"{variant}, decoder - TabPFN", model[variant] - score)[0])
            print("  decoder against TabPFN: " + ", ".join(
                f"{metric} {folds[variant][f'impute/{population}/{metric}'].mean():.4f} against "
                f"{theirs[f'impute/{population}/baseline/tabpfn/{metric}'].mean():.4f}"
                for metric in ("rmse_num_z", "mae_num_z", "acc_cat")
                if f"impute/{population}/{metric}" in folds[variant]  # the cardio donors' gaps hold no category
            ))
    if study == "e148":
        for variant in run.VARIANTS:
            columns = pd.concat(
                pd.read_csv(run.finished(arm, variant, seed)[0] / "metrics" / "per_column_imputation.csv")
                for seed in run.SEEDS
            )
            sparse = columns[(columns["population"] == population) & columns["column"].isin(SPARSE)
                             & columns["metric"].str.endswith("rmse_num_z")]
            sparse = sparse.assign(imputer=sparse["metric"].str.split("/").str[-2].fillna("decoder"))
            print(f"\n{variant}, sparse columns on the {population} cells, RMSE in z units (mean over 15 folds):")
            print(sparse.pivot_table(index="imputer", columns="column", values="value").round(4).to_string())
            print(f"\n{variant}, the decoder's numerical cells by the row's count of natural gaps (15 folds pooled):")
            finished = [run.finished(arm, variant, seed)[0] for seed in run.SEEDS]
            print(by_row_gaps(variant, finished).round(4).to_string())
    else:
        for table in ("mimic-cardio-donor", "mimic-sofa-donor"):
            for a, b in (("tmatch", "trand"), ("trand", "mcar")):
                for what, scores in (("decoder's score", model), ("gap to the bar", gaps)):
                    difference = scores[f"{table}_{a}_20nan"] - scores[f"{table}_{b}_20nan"]
                    print(line(f"{table} {a} - {b}, {what}", difference)[0])


if __name__ == "__main__":
    if sys.argv[1:] == ["e176"]:
        wide()
    elif sys.argv[1:] in (["e148"], ["e149"]):
        main(sys.argv[1])
    else:
        sys.exit(__doc__)
