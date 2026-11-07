"""Pre-registered analysis of E157, ReMasker as an external deep baseline (imputation-protocol/55).

Read-only. Run from the checkout root:

    uv run --python 3.11 python scripts/experiments/remasker_report.py [--remasker DIR]

ReMasker's per-fold CSVs (``remasker_run.py``) are merged with each run's fold table, beside E41's
linear arms and E42's tuned families, on:

E44's confirmation set at seeds 707/808/909, its 9 variants in E157's scope (``SCOPE``). No development set.

Per variant, on the induced ``impute_score`` over 15 fold pairs with the verdict rule and the 1%
SESOI (ADR 0020):

1. K − ReMasker (primary);
2. ReMasker − the strong tier's best (missForest and E41's arms), ReMasker − the bar, ReMasker − the
   tuned bar;
3. K − the best of all, ReMasker included.

Also the fit time per fold.

E158 (imputation-protocol/56) reads its own CSVs, the paper's re-mask count, with ``--remasker DIR --against E157``:
``--against DIR`` merges E157's CSVs as ``remasker_reference`` and adds paper − reference per variant, descriptive.
Every CSV read must carry the rule its directory claims (``remask_paper``), or the report stops.
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
from scripts.experiments.thesis_table import SESOI, TUNED_BAR  # noqa: E402

DEFAULT_REMASKER = Path("/scratch2/diogoneiss/trident-2026-11-06-remasker/csv")
REMASKER = "impute/induced/baseline/remasker/impute_score"
Cells = list[tuple[common.Run, pd.DataFrame]]


def _with_remasker(cells: Cells | None, directory: Path, paper: bool, name: str = "remasker") -> Cells | None:
    if cells is None:
        return None
    out = []
    for run, frame in cells:
        path = directory / f"{run.variant}_s{run.seed}.csv"
        if not path.exists():
            return None
        extra = pd.read_csv(path).sort_values("fold").reset_index(drop=True)
        rule = extra["baseline/remasker/remask_paper"] if "baseline/remasker/remask_paper" in extra else pd.Series([0.0])
        if not (rule == float(paper)).all():
            raise SystemExit(f"{path} does not carry the {'paper' if paper else 'reference'} re-mask rule")
        extra = extra.rename(columns=lambda c: c.replace("/baseline/remasker/", f"/baseline/{name}/").replace("baseline/remasker/", f"baseline/{name}/"))
        new = [c for c in extra.columns if c.startswith(("impute/", "baseline/")) and c not in frame.columns]
        out.append((run, pd.concat([frame.reset_index(drop=True), extra[new]], axis=1)))
    return out


def _values(run: common.Run, frame: pd.DataFrame, name: str) -> np.ndarray:
    if name == "K":
        return common.model(frame, run, "K")
    if name == "remasker":
        return np.asarray(frame[REMASKER], dtype=float)
    if name == "remasker_reference":
        return np.asarray(frame[REMASKER.replace("/remasker/", "/remasker_reference/")], dtype=float)
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
        readings = {"K − remasker": contrast(cells, "K", "remasker"),
                    "remasker − strong": contrast(cells, "remasker", "strong"),
                    "remasker − bar": contrast(cells, "remasker", "bar")}
        if all(TUNED_BAR in frame.columns for _, frame in cells):
            readings["remasker − tuned"] = contrast(cells, "remasker", "tuned")
        readings["K − best_of_all"] = contrast(cells, "K", "best_of_all")
        best = sorted({common.opponent(frame, "best_of_all")[0] for _, frame in cells})
        seconds = float(np.mean([frame["impute/induced/baseline/remasker/fit_seconds"].mean() for _, frame in cells]))
        print(f"- `{variant}`: " + "; ".join(f"{k} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}" for k, v in readings.items())
              + f"; best of all {'/'.join(best)}; fit {seconds:.0f} s a fold")
        for k, v in readings.items():
            counts.setdefault(k, []).append(v.call)
    for k, calls in counts.items():
        print(f"  - {k}: {common.tally(calls)}")
    print()


# E157's scope (imputation-protocol/55): the confirmation set's numerical variants that fit a night at the paper's
# settings; letter (about 25 process-hours) and electricity (a categorical column) are out.
SCOPE = [v for v in confirmation.VARIANTS if v.split("_")[0] in ("pendigits", "vehicle", "biodeg", "kc2")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remasker", type=Path, default=DEFAULT_REMASKER)
    parser.add_argument("--paper", action="store_true", help="the CSVs carry the paper's re-mask rule (E158)")
    parser.add_argument("--against", type=Path, help="E157's CSVs, merged as remasker_reference")
    options = parser.parse_args()
    found = [(v, _with_remasker(confirmation.cells(v), options.remasker, options.paper)) for v in SCOPE]
    study = "E158, the paper's re-mask rule" if options.paper else "E157"
    report(f"Confirmation set (E44), seeds 707/808/909, {study}, 9 variants", found)
    if options.against is not None:
        # Descriptive, after the read and apart from it, so the best of all above never holds E157's arm.
        print("## ReMasker, the paper's rule − the reference code's (E158 − E157), descriptive")
        side = []
        for v, cells in found:
            both = _with_remasker(cells, options.against, False, "remasker_reference")
            if both is None:
                print(f"- `{v}`: incomplete")
                continue
            d = contrast(both, "remasker", "remasker_reference")
            side.append(d.call)
            print(f"- `{v}`: {d.mean:+.4f} [{d.low:+.4f}, {d.high:+.4f}] {d.call}")
        print(f"  - {common.tally(side)}")
        print()
    calls = [contrast(c, "remasker", "strong").call for _, c in found if c is not None]
    if len(calls) == len(SCOPE):
        better = calls.count("better")
        print(f"Decision input: ReMasker better than the strong tier's best on {better} of {len(calls)} variants "
              f"({'at least 3, the third E48 asked for: propose it for the strong tier' if better >= 3 else 'fewer than 3: an external deep baseline only'}).")
    else:
        print(f"incomplete: {len(calls)} of {len(SCOPE)} variants have all three seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
