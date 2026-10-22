"""Pre-registered analysis of E63: a MAR smoke against matched MCAR controls (imputation-protocol/18,
T30). Read-only.

    uv run --python 3.11 python scripts/experiments/mar_smoke_report.py [--tally | --todo 1|2|3|4]

biodeg, kc2, vehicle and kr-vs-kp at 60nan, each as a MAR variant and its matched MCAR control
(``datasets/generate_mar.py``: the same fully observed drivers and per-column counts), today's
default at seeds 42/7/13 (tag experiment=mar-smoke-2026-10-22, arm G), TabPFN's offline CSVs
merged (ADR 0028). On the induced ``impute_score`` over 15 fold pairs, with the verdict rule:

- per mechanism: H − bar, H − strong, H − TabPFN, K − bar, K − strong and KC − strong, with the
  1% SESOI;
- the mechanism effect, fold-paired (a seed builds the same folds on both variants): (H − bar)
  MAR minus control, (H − strong) MAR minus control, and the oracle's optimism, (K − H) MAR minus
  control;
- **primary:** on the four tables, ``holds`` if neither (H − bar) nor (H − strong) MAR minus
  control is worse on any; ``loses ground`` if worse on two or more table-contrasts; ``mixed``
  otherwise.

Under MAR the mean/mode denominator comes from observed values the mechanism biases, so scores are
compared across mechanisms only as differences of within-variant gaps.
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

EXPERIMENT = "mar-smoke-2026-10-22"
ARM = "G"
BASES = ("biodeg", "kc2", "vehicle", "kr-vs-kp")
LEVEL = "60nan"
SEEDS = [42, 7, 13]
SESOI = 0.01
QUEUES = 4
STATE = Path("/scratch2/diogoneiss/trident-2026-10-22-mar-smoke")
TABPFN_DIR = STATE / "tabpfn"
CELL_MINUTES = {"kr-vs-kp": 25.0, "biodeg": 6.0, "kc2": 5.0, "vehicle": 5.0}
PER_MECHANISM = (("H", "bar"), ("H", "strong"), ("H", "tabpfn"), ("K", "bar"), ("K", "strong"), ("KC", "strong"))
Cells = list[tuple[common.Run, pd.DataFrame]]


def variant(base: str, mechanism: str) -> str:
    return f"{base}_{mechanism}_{LEVEL}"


def variants() -> list[str]:
    return [variant(b, m) for b in BASES for m in ("mar", "marctl")]


def _merge(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    extra = pd.read_csv(path).sort_values("fold").reset_index(drop=True)
    new = [c for c in extra.columns if c.startswith("impute/") and c not in frame.columns]
    return pd.concat([frame.reset_index(drop=True), extra[new]], axis=1)


def cells(found: dict[tuple[str, int], list[common.Run]], name: str) -> Cells | None:
    out = []
    for seed in SEEDS:
        runs = found.get((name, seed), [])
        path = TABPFN_DIR / f"{name}_s{seed}.csv"
        if len(runs) != 1 or not path.exists():
            return None
        out.append((runs[0], _merge(common.fold_table(runs[0]), path)))
    return out


def readout(run: common.Run, frame: pd.DataFrame, name: str) -> np.ndarray:
    if name == "KC":
        key = "impute/induced/calibrated/impute_score"
        if key in frame.columns:
            return np.asarray(frame[key], dtype=float)
        return np.asarray(frame["impute/induced/impute_score"], dtype=float)
    return common.model(frame, run, name)


def gaps(found: Cells, name: str, which: str) -> tuple[list[np.ndarray], float, str]:
    per_seed, scale, names = [], [], set()
    for run, frame in found:
        if which in ("bar", "strong", "tabpfn"):
            opponent, theirs = common.opponent(frame, which)
        else:
            opponent, theirs = which, readout(run, frame, which)
        per_seed.append(readout(run, frame, name) - theirs)
        scale.append(float(theirs.mean()))
        names.add(opponent)
    return per_seed, float(np.mean(scale)), "/".join(sorted(names))


def tally() -> None:
    found = common.parents(EXPERIMENT, ARM)
    effects: dict[str, list[str]] = {}
    complete = 0
    for base in BASES:
        mar, ctl = cells(found, variant(base, "mar")), cells(found, variant(base, "marctl"))
        if mar is None or ctl is None:
            print(f"## `{base}`: incomplete\n")
            continue
        complete += 1
        print(f"## `{base}` at {LEVEL}")
        for label, found_m in (("MAR", mar), ("control", ctl)):
            parts = []
            for name, which in PER_MECHANISM:
                per_seed, scale, opponent = gaps(found_m, name, which)
                v = common.verdict(per_seed, sesoi=SESOI * scale)
                parts.append(f"{name} − {which} ({opponent}) {v.mean:+.4f} {v.call}")
            print(f"- {label}: " + "; ".join(parts))
        for name, which in (("H", "bar"), ("H", "strong"), ("K", "H")):
            a, _, _ = gaps(mar, name, which)
            b, _, _ = gaps(ctl, name, which)
            v = common.verdict([x - y for x, y in zip(a, b)])
            key = f"({name} − {which}) MAR minus control"
            effects.setdefault(key, []).append(v.call)
            print(f"  - {key}: {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
            if (name, which) != ("K", "H") and v.call == "worse":
                effects.setdefault("primary worse", []).append(f"{base} {key}")
        print()
    for key, calls in effects.items():
        if key != "primary worse":
            print(f"- {key}: {common.tally(calls)}")
    if complete < len(BASES):
        print(f"\nPrimary: incomplete ({complete} of {len(BASES)})")
        return
    worse = effects.get("primary worse", [])
    reading = "holds" if not worse else "loses ground" if len(worse) >= 2 else "mixed"
    print(f"\nPrimary: worse MAR-minus-control contrasts: {worse or 'none'} -> {reading} ({common.RULE})")


def queue(name: str) -> list[tuple[str, int]]:
    work = sorted(((CELL_MINUTES[v.split("_")[0]], v, s) for v in variants() for s in SEEDS), key=lambda i: (-i[0], i[1], i[2]))
    loads = [0.0] * QUEUES
    assigned: list[list[tuple[str, int]]] = [[] for _ in range(QUEUES)]
    for minutes, name_, seed in work:
        target = loads.index(min(loads))
        loads[target] += minutes
        assigned[target].append((name_, seed))
    return assigned[int(name) - 1]


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--todo"] and len(argv) == 2:
        done = common.parents(EXPERIMENT, ARM)
        for name_, seed in queue(argv[1]):
            if (name_, seed) not in done:
                print(name_, seed)
    elif argv in ([], ["--tally"]):
        tally()
    else:
        sys.exit(__doc__)
