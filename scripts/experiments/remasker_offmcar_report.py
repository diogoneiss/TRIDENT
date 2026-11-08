"""Pre-registered analysis of E160: ReMasker (the paper's rule) off MCAR, on the MAR and MNAR families'
numerical tables (imputation-protocol/58). Read-only.

    uv run --python 3.11 python -B scripts/experiments/remasker_offmcar_report.py

Cells: E64's MAR family (tag mar-family-2026-10-22, arm G, seeds 404/505/606, TabPFN merged) and E45/E46's MNAR
family as E58 read it (arm G, seeds 42/7/13, TabPFN merged); biodeg, kc2 and vehicle at 20 and 60nan, each mechanism
beside its control (MAR: ``marfctl``; MNAR: the MCAR variant). ReMasker's CSVs (``remasker_run.py``, paper rule) are
merged into each frame. H is the headline those studies read; K beside. On the induced ``impute_score`` over 15 fold
pairs, with the verdict rule:

- **per variant,** with the 1% SESOI: H − ReMasker, K − ReMasker, ReMasker − strong, ReMasker − bar;
- **the mechanism effect, fold-paired:** (H − ReMasker), mechanism minus control, per table-rate;
- **primary,** over the 12 table-rates (6 MAR, 6 MNAR): ReMasker gains ground on TRIDENT off MCAR (``supported``) if
  the effect is worse for H on 4 or more and better on at most 1; ``refuted`` if worse on at most 1; ``mixed``
  otherwise (E64's rule).
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

from scripts.experiments import mar_family_report as mar  # noqa: E402
from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments import tabpfn_seed_sets_report as e58  # noqa: E402
from scripts.experiments.remasker_report import _with_remasker  # noqa: E402

OUT = Path("/scratch2/diogoneiss/trident-2026-11-07-remasker-offmcar")
TABLES = ("biodeg", "kc2", "vehicle")
LEVELS = (20, 60)
SESOI = 0.01
REMASKER = "impute/induced/baseline/remasker/impute_score"
Cells = list[tuple[common.Run, pd.DataFrame]]


def mar_pairs() -> list[tuple[str, str]]:
    return [(mar.name(t, f"{lv}nan", "mar"), mar.name(t, f"{lv}nan", "control")) for t in TABLES for lv in LEVELS]


def mnar_pairs() -> list[tuple[str, str]]:
    return [(f"{t}_probit_{lv}nan", f"{t}_{lv}nan") for t in TABLES for lv in LEVELS]


def _study(variant: str) -> str:
    """E45 ran the 60nan rung (MNAR and MCAR), E46 the 20nan one."""
    return e58.SMOKE if variant.endswith("60nan") else e58.FAMILY


def mar_cells(variant: str) -> Cells | None:
    return _with_remasker(mar.cells(common.parents(mar.EXPERIMENT, mar.ARM), variant), OUT / "mar" / "csv", True)


def mnar_cells(variant: str) -> Cells | None:
    return _with_remasker(e58.cells(variant, [42, 7, 13], _study(variant), "G"), OUT / "mnar" / "csv", True)


def values(run: common.Run, frame: pd.DataFrame, name: str) -> np.ndarray:
    if name in ("H", "K"):
        return common.model(frame, run, name)
    if name == "remasker":
        return np.asarray(frame[REMASKER], dtype=float)
    return common.opponent(frame, name)[1]


def gaps(found: Cells, first: str, second: str) -> tuple[list[np.ndarray], float]:
    per_seed = [values(r, f, first) - values(r, f, second) for r, f in found]
    return per_seed, float(np.mean([values(r, f, second).mean() for r, f in found]))


def main() -> int:
    families = (("MAR (E64's cells, 404/505/606)", mar_pairs(), mar_cells),
                ("MNAR (E45/E46's cells, 42/7/13)", mnar_pairs(), mnar_cells))
    primary: list[str] = []
    for label, pairs, load in families:
        print(f"## {label}")
        counts: dict[str, list[str]] = {}
        for mechanism, control in pairs:
            a, b = load(mechanism), load(control)
            if a is None or b is None:
                print(f"- `{mechanism}` against `{control}`: incomplete")
                continue
            for tag, found in (("mechanism", a), ("control", b)):
                parts = []
                for first, second in (("H", "remasker"), ("K", "remasker"), ("remasker", "strong"), ("remasker", "bar")):
                    per_seed, scale = gaps(found, first, second)
                    v = common.verdict(per_seed, sesoi=SESOI * scale)
                    parts.append(f"{first} − {second} {v.mean:+.4f} {v.call}")
                    counts.setdefault(f"{tag}: {first} − {second}", []).append(v.call)
                print(f"- `{mechanism if tag == 'mechanism' else control}`: " + "; ".join(parts))
            x, _ = gaps(a, "H", "remasker")
            y, _ = gaps(b, "H", "remasker")
            v = common.verdict([p - q for p, q in zip(x, y)])
            primary.append(v.call)
            counts.setdefault("(H − remasker), mechanism minus control", []).append(v.call)
            print(f"  - (H − remasker), mechanism minus control: {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
        for k, calls in counts.items():
            print(f"  - {k}: {common.tally(calls)}")
        print()
    if len(primary) == 12:
        worse, better = primary.count("worse"), primary.count("better")
        call = "supported" if worse >= 4 and better <= 1 else "refuted" if worse <= 1 else "mixed"
        print(f"Primary, (H − ReMasker) mechanism minus control over 12 table-rates: worse {worse}, better {better} -> {call}")
    else:
        print(f"incomplete: {len(primary)} of 12 table-rates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
