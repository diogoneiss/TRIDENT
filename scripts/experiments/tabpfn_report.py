"""Pre-registered analysis of E55, TabPFN v2 as an external foundation-model baseline (imputation-protocol/10).

Read-only. Run from the checkout root:

    uv run --python 3.11 python scripts/experiments/tabpfn_report.py [--tabpfn DIR]

TabPFN's per-fold CSVs (``tabpfn_run.py``) are merged with each run's fold table, beside E41's
linear arms and E42's tuned families, on:

- the development set at seeds 101/202/303 (``report_common.development_source``, the cell map of
  the thesis table);
- E44's confirmation set, 15 variants at seeds 707/808/909.

Per variant, on the induced ``impute_score`` over 15 fold pairs with the verdict rule and the 1%
SESOI (ADR 0020):

1. K − TabPFN (primary);
2. TabPFN − the strong tier's live best (missForest and E41's arms, ``strong_live``), TabPFN − the
   bar, TabPFN − the tuned bar;
3. K − the best of all, TabPFN included (MaCoDE too where its CSVs exist);
4. K − the strong tier before and after ADR 0028 (``strong_live`` and ``strong``, TabPFN in it).

Also the fit time per fold.
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

from scripts.experiments import confirmation_report as confirmation  # noqa: E402
from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments.thesis_table import DEFAULT_LINEAR, DEFAULT_TUNED, DEVELOPMENT, SESOI, TUNED_BAR, _cells  # noqa: E402

DEFAULT_TABPFN = Path("/scratch2/diogoneiss/trident-2026-10-21-tabpfn/csv")
MACODE_CSV = Path("/scratch2/diogoneiss/trident-2026-10-19-macode/csv")
TABPFN = "impute/induced/baseline/tabpfn/impute_score"
Cells = list[tuple[common.Run, pd.DataFrame]]


def _with_tabpfn(cells: Cells | None, directory: Path) -> Cells | None:
    if cells is None:
        return None
    out = []
    for run, frame in cells:
        path = directory / f"{run.variant}_s{run.seed}.csv"
        if not path.exists():
            return None
        macode = MACODE_CSV / f"{run.variant}_s{run.seed}.csv"
        if macode.exists():
            more = pd.read_csv(macode).sort_values("fold").reset_index(drop=True)
            frame = pd.concat([frame.reset_index(drop=True), more[[c for c in more.columns if c.startswith("impute/") and c not in frame.columns]]], axis=1)
        extra = pd.read_csv(path).sort_values("fold").reset_index(drop=True)
        new = [c for c in extra.columns if c.startswith(("impute/", "baseline/")) and c not in frame.columns]
        out.append((run, pd.concat([frame.reset_index(drop=True), extra[new]], axis=1)))
    return out


def _values(run: common.Run, frame: pd.DataFrame, name: str) -> np.ndarray:
    if name == "K":
        return common.model(frame, run, "K")
    if name == "tabpfn":
        return np.asarray(frame[TABPFN], dtype=float)
    if name == "tuned":
        return np.asarray(frame[TUNED_BAR], dtype=float)
    return common.opponent(frame, name)[1]


def contrast(cells: Cells, first: str, second: str) -> common.Verdict:
    per_seed, scale = [], []
    for run, frame in cells:
        a, b = _values(run, frame, first), _values(run, frame, second)
        per_seed.append(a - b)
        scale.append(float(b.mean()))
    return common.verdict(per_seed, sesoi=SESOI * float(np.mean(scale)))


def report(label: str, found: list[tuple[str, Cells | None]]) -> None:
    print(f"## {label}")
    counts: dict[str, list[str]] = {}
    for variant, cells in found:
        if cells is None:
            print(f"- `{variant}`: incomplete")
            continue
        readings = {"K − tabpfn": contrast(cells, "K", "tabpfn"),
                    "tabpfn − strong": contrast(cells, "tabpfn", "strong_live"),
                    "tabpfn − bar": contrast(cells, "tabpfn", "bar")}
        if all(TUNED_BAR in frame.columns for _, frame in cells):
            readings["tabpfn − tuned"] = contrast(cells, "tabpfn", "tuned")
        readings["K − best_of_all"] = contrast(cells, "K", "best_of_all")
        # ADR 0028: K against the strong tier with TabPFN in it, beside the live four.
        readings["K − strong_live"] = contrast(cells, "K", "strong_live")
        readings["K − strong (ADR 0028)"] = contrast(cells, "K", "strong")
        best = sorted({common.opponent(frame, "best_of_all")[0] for _, frame in cells})
        seconds = float(np.mean([frame["impute/induced/baseline/tabpfn/fit_seconds"].mean() for _, frame in cells]))
        print(f"- `{variant}`: " + "; ".join(f"{k} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}" for k, v in readings.items())
              + f"; best of all {'/'.join(best)}; fit {seconds:.0f} s a fold")
        for k, v in readings.items():
            counts.setdefault(k, []).append(v.call)
    for k, calls in counts.items():
        print(f"  - {k}: {common.tally(calls)}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabpfn", type=Path, default=DEFAULT_TABPFN)
    options = parser.parse_args()
    development = [(v, _with_tabpfn(_cells(v, "101/202/303", DEFAULT_LINEAR, DEFAULT_TUNED), options.tabpfn)) for v in DEVELOPMENT]
    report("Development set, seeds 101/202/303 (labelled variants: none on this set)", development)
    confirmation_cells = [(v, _with_tabpfn(confirmation.cells(v), options.tabpfn)) for v in confirmation.VARIANTS]
    report("Confirmation set (E44), seeds 707/808/909", confirmation_cells)
    calls = [contrast(c, "tabpfn", "strong_live").call for _, c in confirmation_cells if c is not None]
    if len(calls) == len(confirmation.VARIANTS):
        better = calls.count("better")
        print(f"Decision input: TabPFN better than the strong tier's best on {better} of {len(calls)} confirmation variants "
              f"({'at least 5: propose it for the strong tier' if better >= 5 else 'fewer than 5: an external baseline only'}).")
    else:
        print(f"Decision input: incomplete, {len(calls)} of {len(confirmation.VARIANTS)} confirmation variants have every seed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
