"""E190's one read (docs/tickets/imputation-protocol/79-wide-table-standing.md), from the study's state directory;
with ``WIDE_TABLE=nomao``, E191's (docs/tickets/imputation-protocol/80-wide-table-nomao.md), which adds the decoder's
``mae_num_z`` against missForest's by the same rule and the categorical columns' means.

    QUEUE_STATE=<state directory> uv run --python 3.11 python -B scripts/experiments/wide_table_report.py
    QUEUE_STATE=<any> uv run --python 3.11 python -B scripts/experiments/wide_table_report.py --check <variant> <run directory>

Per variant: the default output's induced ``impute_score`` (``KC``) minus the bar's best baseline's and minus the
logged strong tier's best, 15 fold pairs, by ``report_common.verdict`` with a margin of 1% of the opponent's mean;
every scored imputer's mean ``impute_score``, ``rmse_num_z`` and ``mae_num_z``; the mean fill's ``rmse_num_z`` per
fold; the cells' wall time from ``progress.log``. The study-level call counts "better" against the bar over the four
variants. Refuses to read with a cell missing. ``--check`` prints one run directory's variant block (a code check on
a cell outside the study). Run from the checkout root.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import report_common  # noqa: E402
from scripts.experiments import wide_table_run as run  # noqa: E402

POPULATION = "induced"


def fold_table(directory: Path) -> pd.DataFrame:
    frame = pd.read_csv(directory / "metrics" / "raw_fold_metrics.csv").sort_values("fold").reset_index(drop=True)
    assert len(frame) == 5, directory
    return frame


def block(variant: str, frames: list[pd.DataFrame]) -> str:
    """Print a variant's reading from its seeds' fold tables; return its call against the bar."""
    source = report_common.Run(variant, 0, "", POPULATION)
    # the run's headline: calibrated from 40nan up (ADR 0027), the uncalibrated output at 20nan
    own = [report_common.model(frame, source, "KC" if f"impute/{POPULATION}/uncalibrated/impute_score" in frame else "K", POPULATION)
           for frame in frames]
    assert all((o == frame[f"impute/{POPULATION}/impute_score"].to_numpy()).all() for o, frame in zip(own, frames))
    print(f"{variant}: decoder {np.concatenate(own).mean():.4f} over {sum(len(o) for o in own)} folds")
    calls = {}
    for which in ("bar", "strong_live"):
        theirs = [report_common.opponent(frame, which, POPULATION) for frame in frames]
        mean = float(np.concatenate([values for _, values in theirs]).mean())
        v = report_common.verdict([a - b for a, (_, b) in zip(own, theirs)], 0.01 * mean)
        calls[which] = v.call
        print(f"  decoder - {which} ({mean:.4f}, best by run: {sorted({name for name, _ in theirs})}): {v.mean:+.4f} "
              f"[{v.low:+.4f}, {v.high:+.4f}], by seed " + ", ".join(f"{s:+.4f}" for s in v.seeds) + f" -> {v.call}")
    pooled = pd.concat(frames)
    if run.TABLE == "nomao":  # E191's secondary: E190's descriptive inversion against missForest, by the same rule
        mine = [report_common.model(frame, source, "KC" if f"impute/{POPULATION}/uncalibrated/mae_num_z" in frame else "K",
                                    POPULATION, "mae_num_z") for frame in frames]
        forest = [frame[f"impute/{POPULATION}/baseline/missforest/mae_num_z"].to_numpy(dtype=float) for frame in frames]
        v = report_common.verdict([a - b for a, b in zip(mine, forest)], 0.01 * float(np.concatenate(forest).mean()))
        print(f"  mae_num_z, decoder - missforest: {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}], by seed "
              + ", ".join(f"{s:+.4f}" for s in v.seeds) + f" -> {v.call}")
    for metric in ("impute_score", "rmse_num_z", "mae_num_z", "acc_cat", "macro_f1_cat"):
        every = pooled.filter(regex=rf"^impute/{POPULATION}/(baseline/[^/]+/)?{metric}$").mean()
        if every.empty:  # a table with no categorical column
            continue
        every.index = pd.Index([name.split("/")[-2].replace(POPULATION, "decoder") for name in every.index])
        print(f"  mean {metric}: " + ", ".join(f"{name} {value:.4f}" for name, value in every.sort_values().items()))
    naive = pooled[f"impute/{POPULATION}/baseline/mean_mode/rmse_num_z"]
    print(f"  mean fill's rmse_num_z per fold: {naive.min():.4f} to {naive.max():.4f}")
    return calls["bar"]


def wall_times() -> None:
    stamps: dict[str, dict[str, datetime]] = {}
    for line in (run.STATE / "progress.log").read_text().splitlines():
        if m := re.match(r"(\S+ \S+) \S+ queue \d+: cell (start|done) (\S+)", line):
            stamps.setdefault(m.group(3), {})[m.group(2)] = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    hours = {cell: (s["done"] - s["start"]).total_seconds() / 3600 for cell, s in stamps.items() if len(s) == 2}
    print("wall time per cell, hours (from progress.log): " + ", ".join(f"{c} {h:.2f}" for c, h in hours.items()))


def main() -> None:
    frames: dict[str, list[pd.DataFrame]] = {}
    for variant in run.VARIANTS:
        for seed in run.SEEDS:
            runs = run.finished(variant, seed)
            if len(runs) != 1:
                sys.exit(f"{variant} seed {seed}: {len(runs)} finished runs; the ticket reads only a complete study")
            frames.setdefault(variant, []).append(fold_table(runs[0]))
    calls = [block(variant, frames[variant]) for variant in run.VARIANTS]
    better = calls.count("better")
    study = "supported" if better == len(calls) else "refuted" if better == 0 else "mixed"
    print(f"AGAINST THE BAR over {len(calls)} variants: {report_common.tally(calls)} -> {study}")
    wall_times()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--check":
        block(sys.argv[2], [fold_table(Path(sys.argv[3]))])
    elif len(sys.argv) == 1:
        main()
    else:
        sys.exit(__doc__)
