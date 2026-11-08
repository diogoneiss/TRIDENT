"""Pre-registered analysis of E161: ReMasker (the paper's rule) on E148's MIMIC-IV cells (mimic/09). Read-only.

    QUEUE_STATE=/scratch2/diogoneiss/trident-2026-11-06-mimic-e148 TRIDENT_DATASETS_ROOT=/scratch/diogoneiss/trident-data \\
        uv run --python 3.11 python -B scripts/experiments/remasker_mimic_report.py <remasker csv dir>

The masked population of E148's six runs (hidden observed cells, natural gaps left as inputs). ReMasker sees the
numerical columns only and fills the categorical cells with the training mode, so the primary is on the numerical
cells: per table, the decoder's ``rmse_num_z`` minus ReMasker's, paired per (seed, fold), 15 replicates, plain 95% t
interval, Holm over the two tables (E148's rule). Beside it: ``mae_num_z``; ReMasker against the bar's best baseline
and TabPFN on ``rmse_num_z``; ``impute_score`` (ReMasker's carries the mode-filled categorical cells).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import mimic_report as mr  # noqa: E402
from scripts.experiments import mimic_run as run  # noqa: E402  (needs QUEUE_STATE and TRIDENT_DATASETS_ROOT)
from scripts.experiments import report_common as common  # noqa: E402

P = "impute/masked"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.exit(__doc__)
    csv = Path(argv[0])
    arm = next(iter(run.ARMS))
    primaries: dict[str, np.ndarray] = {}
    for variant in run.VARIANTS:
        frames, theirs, pfn = [], [], []
        for seed in run.SEEDS:
            runs = run.finished(arm, variant, seed)
            path = csv / f"{variant}_s{seed}.csv"
            if len(runs) != 1 or not path.exists():
                sys.exit(f"{variant} seed {seed}: incomplete; the ticket reads only a complete study")
            frames.append(pd.read_csv(runs[0] / "metrics" / "raw_fold_metrics.csv").sort_values("fold").reset_index(drop=True))
            mine = pd.read_csv(path).sort_values("fold").reset_index(drop=True)
            assert (mine[f"{P}/baseline/remasker/remask_paper"] == 1).all(), f"{path.name} is not the paper's rule"
            theirs.append(mine)
            pfn.append(pd.read_csv(run.STATE / "tabpfn" / f"{variant}_s{seed}.csv").sort_values("fold").reset_index(drop=True))
        own, rem, tab = pd.concat(frames), pd.concat(theirs), pd.concat(pfn)
        bar = np.concatenate([common.opponent(f, "bar", "masked")[1] for f in frames])

        def col(frame: pd.DataFrame, key: str) -> np.ndarray:
            return frame[key].to_numpy(float)

        print(f"## {variant}")
        for metric in ("rmse_num_z", "mae_num_z", "impute_score"):
            print(f"- {metric}: decoder {col(own, f'{P}/{metric}').mean():.4f}, ReMasker "
                  f"{col(rem, f'{P}/baseline/remasker/{metric}').mean():.4f}, TabPFN {col(tab, f'{P}/baseline/tabpfn/{metric}').mean():.4f}")
        primaries[variant] = col(own, f"{P}/rmse_num_z") - col(rem, f"{P}/baseline/remasker/rmse_num_z")
        print("  " + mr.line("decoder − ReMasker, mae_num_z", col(own, f"{P}/mae_num_z") - col(rem, f"{P}/baseline/remasker/mae_num_z"))[0])
        print("  " + mr.line("decoder − ReMasker, impute_score (ReMasker's categorical cells mode-filled)",
                             col(own, f"{P}/impute_score") - col(rem, f"{P}/baseline/remasker/impute_score"))[0])
        # per fold, the lowest rmse_num_z among the bar's and the strong tier's baselines the run scored
        best = np.concatenate([
            f[[f"{P}/baseline/{n}/rmse_num_z" for n in common.baselines_in(f.columns, "masked")
               if n in common.WEAK or n in common.STRONG]].min(axis=1).to_numpy(float)
            for f in frames])
        print("  " + mr.line("ReMasker − best fixed baseline per fold, rmse_num_z", col(rem, f"{P}/baseline/remasker/rmse_num_z") - best)[0])
        print("  " + mr.line("ReMasker − TabPFN, rmse_num_z",
                             col(rem, f"{P}/baseline/remasker/rmse_num_z") - col(tab, f"{P}/baseline/tabpfn/rmse_num_z"))[0])
        print(f"  ReMasker fit {col(rem, f'{P}/baseline/remasker/fit_seconds').mean():.0f} s a fold; bar by impute_score {bar.mean():.4f}")
    print()
    mr.verdicts({f"{v}, decoder − ReMasker, rmse_num_z": d for v, d in primaries.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
