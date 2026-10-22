"""Pre-registered analysis of E64: the MAR family against matched MCAR controls
(imputation-protocol/19, T30). Read-only.

    uv run --python 3.11 python scripts/experiments/mar_family_report.py [--tally | --todo 1|2|3|4]

biodeg, kc2, vehicle, kr-vs-kp, spambase and credit-g at 20 and 60nan, each as a MAR variant (slope
1.5) and its matched MCAR control (``datasets/generate_mar_family.py``), plus biodeg_60nan at
slopes 0.75 and 3 against the same control. Today's default runs at seeds 404/505/606 (tag
experiment=mar-family-2026-10-22, arm G), with TabPFN's offline CSVs merged (ADR 0028). On the
induced ``impute_score`` over 15 fold pairs, with the verdict rule:

- **per mechanism,** with the 1% SESOI: H − bar, H − strong, H − TabPFN, K − bar, K − strong and
  KC − strong;
- **the mechanism effect, fold-paired** (a seed builds the same folds on both variants):
  (H − bar), (H − strong) and (K − H), MAR minus control;
- **primary,** over the 12 table-rates at slope 1.5, (H − strong) MAR minus control:
  - ``supported`` (MAR costs the model ground against the strong tier) if worse on 4 or more and
    better on at most 1;
  - ``refuted`` if worse on at most 1;
  - ``mixed`` otherwise;
- **the dial:** (H − strong) MAR minus control on biodeg_60nan at slopes 0.75, 1.5 and 3,
  descriptive.

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

EXPERIMENT = "mar-family-2026-10-22"
ARM = "G"
BASES = ("biodeg", "kc2", "vehicle", "kr-vs-kp", "spambase", "credit-g")
LEVELS = ("20nan", "60nan")
DIAL = ("biodeg", "60nan", ("0p75", "", "3"))
SEEDS = [404, 505, 606]
SESOI = 0.01
QUEUES = 4
STATE = Path("/scratch2/diogoneiss/trident-2026-10-22-mar-family")
TABPFN_DIR = STATE / "tabpfn"
CELL_MINUTES = {"spambase": 30.0, "kr-vs-kp": 25.0, "credit-g": 6.0, "biodeg": 6.0, "kc2": 5.0, "vehicle": 5.0}
PER_MECHANISM = (("H", "bar"), ("H", "strong"), ("H", "tabpfn"), ("K", "bar"), ("K", "strong"), ("KC", "strong"))
EFFECTS = (("H", "bar"), ("H", "strong"), ("K", "H"))
Cells = list[tuple[common.Run, pd.DataFrame]]


def name(base: str, level: str, mechanism: str, slope: str = "") -> str:
    return f"{base}_{'marfctl' if mechanism == 'control' else 'marf' + slope}_{level}"


def variants() -> list[str]:
    out = [name(b, lv, m) for b in BASES for lv in LEVELS for m in ("mar", "control")]
    base, level, slopes = DIAL
    return out + [name(base, level, "mar", s) for s in slopes if s]


def _merge(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    extra = pd.read_csv(path).sort_values("fold").reset_index(drop=True)
    new = [c for c in extra.columns if c.startswith("impute/") and c not in frame.columns]
    return pd.concat([frame.reset_index(drop=True), extra[new]], axis=1)


def cells(found: dict[tuple[str, int], list[common.Run]], variant: str) -> Cells | None:
    out = []
    for seed in SEEDS:
        runs = found.get((variant, seed), [])
        path = TABPFN_DIR / f"{variant}_s{seed}.csv"
        if len(runs) != 1 or not path.exists():
            return None
        out.append((runs[0], _merge(common.fold_table(runs[0]), path)))
    return out


def readout(run: common.Run, frame: pd.DataFrame, which: str) -> np.ndarray:
    return common.model(frame, run, which)


def gaps(found: Cells, mine: str, which: str) -> tuple[list[np.ndarray], float, str]:
    per_seed, scale, names = [], [], set()
    for run, frame in found:
        if which in ("bar", "strong", "tabpfn"):
            opponent, theirs = common.opponent(frame, which)
        else:
            opponent, theirs = which, readout(run, frame, which)
        per_seed.append(readout(run, frame, mine) - theirs)
        scale.append(float(theirs.mean()))
        names.add(opponent)
    return per_seed, float(np.mean(scale)), "/".join(sorted(names))


def effect(mar: Cells, ctl: Cells, mine: str, which: str) -> common.Verdict:
    a, _, _ = gaps(mar, mine, which)
    b, _, _ = gaps(ctl, mine, which)
    return common.verdict([x - y for x, y in zip(a, b)])


def tally() -> None:
    found = common.parents(EXPERIMENT, ARM)
    calls: dict[str, list[str]] = {}
    primary: dict[str, str] = {}
    for base in BASES:
        for level in LEVELS:
            label = f"{base}_{level}"
            mar, ctl = cells(found, name(base, level, "mar")), cells(found, name(base, level, "control"))
            if mar is None or ctl is None:
                print(f"## `{label}`: incomplete\n")
                continue
            print(f"## `{label}`")
            for tag, found_m in (("MAR", mar), ("control", ctl)):
                parts = []
                for mine, which in PER_MECHANISM:
                    per_seed, scale, opponent = gaps(found_m, mine, which)
                    v = common.verdict(per_seed, sesoi=SESOI * scale)
                    parts.append(f"{mine} − {which} ({opponent}) {v.mean:+.4f} {v.call}")
                print(f"- {tag}: " + "; ".join(parts))
            for mine, which in EFFECTS:
                v = effect(mar, ctl, mine, which)
                key = f"({mine} − {which}) MAR minus control"
                calls.setdefault(key, []).append(v.call)
                if (mine, which) == ("H", "strong"):
                    primary[label] = v.call
                print(f"  - {key}: {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
            print()
    base, level, slopes = DIAL
    ctl = cells(found, name(base, level, "control"))
    print(f"## The dial: `{base}_{level}`, (H − strong) MAR minus control by slope")
    for slope in slopes:
        mar = cells(found, name(base, level, "mar", slope))
        if mar is None or ctl is None:
            print(f"- slope {slope or '1p5'}: incomplete")
            continue
        v = effect(mar, ctl, "H", "strong")
        print(f"- slope {slope or '1p5'}: {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
    print()
    for key, found_calls in calls.items():
        print(f"- {key}: {common.tally(found_calls)}")
    expected = len(BASES) * len(LEVELS)
    if len(primary) < expected:
        print(f"\nPrimary: incomplete ({len(primary)} of {expected})")
        return
    worse = [k for k, c in primary.items() if c == "worse"]
    better = [k for k, c in primary.items() if c == "better"]
    reading = "supported" if len(worse) >= 4 and len(better) <= 1 else "refuted" if len(worse) <= 1 else "mixed"
    print(f"\nPrimary, (H − strong) MAR minus control: worse on {worse or 'none'}, better on {better or 'none'} "
          f"-> {reading} ({common.RULE})")


def queue(which: str) -> list[tuple[str, int]]:
    work = sorted(((CELL_MINUTES[v.split("_")[0]], v, s) for v in variants() for s in SEEDS), key=lambda i: (-i[0], i[1], i[2]))
    loads = [0.0] * QUEUES
    assigned: list[list[tuple[str, int]]] = [[] for _ in range(QUEUES)]
    for minutes, variant, seed in work:
        target = loads.index(min(loads))
        loads[target] += minutes
        assigned[target].append((variant, seed))
    return assigned[int(which) - 1]


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--todo"] and len(argv) == 2:
        done = common.parents(EXPERIMENT, ARM)
        for variant, seed in queue(argv[1]):
            if (variant, seed) not in done:
                print(variant, seed)
    elif argv in ([], ["--tally"]):
        tally()
    else:
        sys.exit(__doc__)
