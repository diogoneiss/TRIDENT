"""Pre-registered reading of E58: the strong tier with TabPFN on the seed sets E55 did not cover
(imputation-protocol/13). Read-only.

    uv run --python 3.11 python scripts/experiments/tabpfn_seed_sets_report.py

Every frame is a run's fold table, with E41's linear arms merged where their offline CSV exists
and TabPFN's offline CSV merged (ADR 0028). On the induced ``impute_score`` over 15 fold pairs,
with the verdict rule and the 1% SESOI:

- **The thesis table's other seed sets:** 42/7/13 (E35; the five labelled variants left out of the
  counts) and 404/505/606 (E40). K − strong_live, K − strong (ADR 0028), K − TabPFN.
- **E51's fresh seeds** at 20nan (1001/2002/3003): the same three.
- **The MNAR family** (E45, E46; G's H, the readout those studies read): per MNAR variant and its
  MCAR control, H − strong_live, H − strong, H − TabPFN, and (H − TabPFN) MNAR minus MCAR.
  - The MNAR variants have no E41 CSVs, so their ``strong_live`` is missForest alone, and the
    printout names it.
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

from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments.thesis_table import DEFAULT_LINEAR, DEFAULT_TABPFN, DEVELOPMENT, E40_VARIANTS  # noqa: E402

SESOI = 0.01
Cells = list[tuple[common.Run, pd.DataFrame]]
E51 = [f"{t}_20nan" for t in ("electricity", "letter", "pendigits", "spambase", "kr-vs-kp", "biodeg", "credit-g", "vehicle", "kc2")]
SMOKE, FAMILY = "mnar-smoke-2026-10-17", "mnar-family-2026-10-17"
# (MNAR variant, its study, MCAR control, the control's study); the dial's control is E45's, as
# ``mnar_family_report.dial`` reads it.
MNAR_PAIRS = [
    *((f"{t}_probit_60nan", SMOKE, f"{t}_60nan", SMOKE) for t in ("biodeg", "kc2", "vehicle")),
    ("spambase_probit_60nan", FAMILY, "spambase_60nan", FAMILY),
    *((f"{t}_probit_20nan", FAMILY, f"{t}_20nan", FAMILY) for t in ("biodeg", "kc2", "vehicle")),
    *((f"biodeg_probit{slope}_60nan", FAMILY, "biodeg_60nan", SMOKE) for slope in ("0p25", "1p0", "2p0")),
]
_FOUND: dict[tuple[str, str | None], dict[tuple[str, int], list[common.Run]]] = {}


def _merge(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        return frame
    extra = pd.read_csv(path).sort_values("fold").reset_index(drop=True)
    new = [c for c in extra.columns if c.startswith("impute/") and c not in frame.columns]
    return pd.concat([frame.reset_index(drop=True), extra[new]], axis=1)


def cells(variant: str, seeds: list[int], experiment: str, arm: str | None = None) -> Cells | None:
    if (experiment, arm) not in _FOUND:
        _FOUND[experiment, arm] = common.parents(experiment, arm)
    found = []
    for seed in seeds:
        runs = _FOUND[experiment, arm].get((variant, seed), [])
        if len(runs) != 1:
            return None
        frame = common.fold_table(runs[0])
        frame = _merge(frame, DEFAULT_LINEAR / f"{variant}_s{seed}.csv")
        frame = _merge(frame, DEFAULT_TABPFN / f"{variant}_s{seed}.csv")
        if f"impute/induced/baseline/tabpfn/impute_score" not in frame.columns:
            return None
        found.append((runs[0], frame))
    return found


def per_seed(found: Cells, readout: str, opponent: str) -> tuple[list[np.ndarray], float, str]:
    diffs, scale, names = [], [], set()
    for run, frame in found:
        name, theirs = common.opponent(frame, opponent)
        diffs.append(common.model(frame, run, readout) - theirs)
        scale.append(float(theirs.mean()))
        names.add(name)
    return diffs, float(np.mean(scale)), "/".join(sorted(names))


def read(found: Cells, readout: str, opponent: str) -> tuple[common.Verdict, str]:
    diffs, scale, names = per_seed(found, readout, opponent)
    return common.verdict(diffs, sesoi=SESOI * scale), names


def seed_set(label: str, variants: list[str], seeds: list[int], experiment: str, skip: tuple[str, ...] = ()) -> None:
    print(f"## {label}")
    counts: dict[str, list[str]] = {}
    for variant in variants:
        found = cells(variant, seeds, experiment)
        if found is None:
            print(f"- `{variant}`: incomplete")
            continue
        readings = {o: read(found, "K", o) for o in ("strong_live", "strong", "tabpfn")}
        flag = " (labelled: not counted)" if variant in skip else ""
        print(f"- `{variant}`{flag}: " + "; ".join(f"K − {o} ({n}) {v.mean:+.4f} {v.call}" for o, (v, n) in readings.items()))
        if not flag:
            for o, (v, _) in readings.items():
                counts.setdefault(o, []).append(v.call)
    for o, calls in counts.items():
        print(f"  - K − {o}: {common.tally(calls)}")
    print()


def mnar() -> None:
    print("## The MNAR family (G's H; E45 and E46)")
    counts: dict[str, list[str]] = {}
    for mnar_variant, mnar_from, mcar_variant, mcar_from in MNAR_PAIRS:
        a, b = cells(mnar_variant, [42, 7, 13], mnar_from, "G"), cells(mcar_variant, [42, 7, 13], mcar_from, "G")
        if a is None or b is None:
            print(f"- `{mnar_variant}` against `{mcar_variant}`: incomplete")
            continue
        parts = []
        for side, found in (("MNAR", a), ("MCAR", b)):
            for o in ("strong_live", "strong", "tabpfn"):
                v, n = read(found, "H", o)
                parts.append(f"{side} H − {o} ({n}) {v.mean:+.4f} {v.call}")
                counts.setdefault(f"{side} H − {o}", []).append(v.call)
        shift = [x - y for x, y in zip(per_seed(a, "H", "tabpfn")[0], per_seed(b, "H", "tabpfn")[0])]
        v = common.verdict(shift)
        counts.setdefault("(H − tabpfn), MNAR minus MCAR", []).append(v.call)
        print(f"- `{mnar_variant}` against `{mcar_variant}`: " + "; ".join(parts) + f"; (H − tabpfn) MNAR minus MCAR {v.mean:+.4f} {v.call}")
    for k, calls in counts.items():
        print(f"  - {k}: {common.tally(calls)}")


def main() -> int:
    labelled = (*common.FORTY_WITH_REMOVED_FILES, *common.BIG_ENCODER)
    seed_set("Development set, seeds 42/7/13 (E35)", DEVELOPMENT, [42, 7, 13], "selection-replication-2026-10-04", labelled)
    seed_set("Development set at 40nan and above, seeds 404/505/606 (E40)", sorted(E40_VARIANTS), [404, 505, 606], "calibration-2026-10-16")
    seed_set("Fresh seeds at 20nan, 1001/2002/3003 (E51)", E51, [1001, 2002, 3003], "default-refresh-2026-10-20")
    mnar()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
