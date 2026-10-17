"""Pre-registered analysis of E41, the linear baselines on the decoder's cells (imputation-protocol/01).

Read-only. Run from the checkout root:

    uv run --python 3.11 python scripts/experiments/linear_baselines_report.py [--linear DIR]

Per seed set of the ticket's cell map (``report_common.development_source``) and variant, on the
induced ``impute_score`` over 15 fold pairs with the verdict rule:

1. each arm (condexp, softimpute, ice) against the bar of 5 (the weak tier);
2. each arm against the model's K;
3. K against the strong tier's best (missForest and the three arms).

Also the masked population, SoftImpute's ranks, ICE's rounds and ``missforest_seq`` against
missForest on the small tables. The reading is descriptive (the ticket); every count names its test.
Labelled variants on 42/7/13 are reported apart and never pooled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments.thesis_table import DEFAULT_LINEAR, DEVELOPMENT, SEED_SETS, SESOI, _cells  # noqa: E402

ARMS = ("condexp", "softimpute", "ice")


def _per_seed(cells: list[tuple[common.Run, pd.DataFrame]], first: str, second: str, population: str) -> common.Verdict:
    per_seed, scale = [], []
    for run, frame in cells:
        def values(name: str) -> np.ndarray:
            if name == "K":
                return common.model(frame, run, "K", population)
            return common.opponent(frame, name, population)[1]
        a, b = values(first), values(second)
        per_seed.append(a - b)
        scale.append(float(b.mean()))
    return common.verdict(per_seed, sesoi=SESOI * float(np.mean(scale)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear", type=Path, default=DEFAULT_LINEAR)
    options = parser.parse_args()
    for seeds in SEED_SETS:
        print(f"## Seeds {seeds}")
        counts: dict[str, list[str]] = {}
        flips: list[str] = []
        for variant in DEVELOPMENT:
            cells = _cells(variant, seeds, options.linear)
            if cells is None or "condexp" not in common.baselines_in(cells[0][1].columns, "induced"):
                continue
            source = common.development_source(variant, seeds)
            labelled = source is not None and source.labelled
            parts = []
            for arm in ARMS:
                against_bar = _per_seed(cells, arm, "bar", "induced")
                against_k = _per_seed(cells, arm, "K", "induced")
                parts.append(f"{arm}: bar {against_bar.mean:+.4f} {against_bar.call}, K {against_k.mean:+.4f} {against_k.call}")
                if not labelled:
                    counts.setdefault(f"{arm} vs bar", []).append(against_bar.call)
                    counts.setdefault(f"{arm} vs K", []).append(against_k.call)
                    if against_k.call == "better":
                        flips.append(f"{variant} ({arm})")
            strong = _per_seed(cells, "K", "strong", "induced")
            if not labelled:
                counts.setdefault("K vs strong", []).append(strong.call)
            name = common.opponent(cells[0][1], "strong")[0]
            ranks = sorted({int(r) for _, f in cells if "impute/induced/baseline/softimpute/rank" in f.columns for r in f["impute/induced/baseline/softimpute/rank"]})
            print(f"- `{variant}`{' †' if labelled else ''}: " + "; ".join(parts)
                  + f"; K vs strong ({name}) {strong.mean:+.4f} [{strong.low:+.4f}, {strong.high:+.4f}] {strong.call}; SoftImpute ranks {ranks}")
        for label, calls in counts.items():
            print(f"  - {label}: {common.tally(calls)}")
        print(f"  - arms better than K: {flips or 'none'}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
