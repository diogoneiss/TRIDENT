"""Pre-registered analysis of E65: the propensity-mask arm (P) against E64's default (G) on the MAR
family (imputation-protocol/20, T30). Read-only.

    uv run --python 3.11 python scripts/experiments/mar_propensity_report.py [--tally | --todo 1|2]

P (tag experiment=mar-propensity-2026-10-22, arm P) is today's default with
``DECODE_MASK_WEIGHTING propensity``. G is E64's run of the same variant and seed, on the same
folds (``mar_family_report``), with E64's TabPFN CSVs merged into both. On the induced
``impute_score`` over 15 fold pairs, with the verdict rule:

- **per table-rate and mechanism, with the 1% SESOI of G's value:** P − G on H, K and KC;
- **the interaction, fold-paired:** (P − G) on H, MAR minus control;
- **each arm's** (H − strong), MAR minus control, P's beside G's;
- **primary,** over the 12 table-rates at slope 1.5:
  - ``supported`` if P − G on H under MAR is better on 4 or more and worse on at most 1, and
    under the control worse on at most 1;
  - ``refuted`` if under MAR it is better on at most 1;
  - ``mixed`` otherwise.
- **the dial** (biodeg_60nan at slopes 0.75, 1.5 and 3), P − G on H under MAR, descriptive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import mar_family_report as family  # noqa: E402
from scripts.experiments import report_common as common  # noqa: E402

EXPERIMENT = "mar-propensity-2026-10-22"
ARM = "P"
QUEUES = 2


def arm_gap(p: family.Cells, g: family.Cells, readout: str) -> tuple[list[np.ndarray], float]:
    per_seed, scale = [], []
    for (rp, fp), (rg, fg) in zip(p, g):
        assert rp.seed == rg.seed
        theirs = family.readout(rg, fg, readout)
        per_seed.append(family.readout(rp, fp, readout) - theirs)
        scale.append(float(theirs.mean()))
    return per_seed, float(np.mean(scale))


def tally() -> None:
    g_runs, p_runs = common.parents(family.EXPERIMENT, family.ARM), common.parents(EXPERIMENT, ARM)
    under_mar: dict[str, str] = {}
    under_control: dict[str, str] = {}
    for base in family.BASES:
        for level in family.LEVELS:
            label = f"{base}_{level}"
            found = {(arm, m): family.cells(runs, family.name(base, level, m))
                     for arm, runs in (("P", p_runs), ("G", g_runs)) for m in ("mar", "control")}
            if any(v is None for v in found.values()):
                print(f"## `{label}`: incomplete\n")
                continue
            print(f"## `{label}`")
            per_mechanism: dict[str, list[np.ndarray]] = {}
            for mechanism in ("mar", "control"):
                p, g = found["P", mechanism], found["G", mechanism]
                assert p is not None and g is not None
                parts = []
                for readout in ("H", "K", "KC"):
                    per_seed, scale = arm_gap(p, g, readout)
                    v = common.verdict(per_seed, sesoi=family.SESOI * scale)
                    parts.append(f"P − G, {readout} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
                    if readout == "H":
                        per_mechanism[mechanism] = per_seed
                        (under_mar if mechanism == "mar" else under_control)[label] = v.call
                print(f"- {'MAR' if mechanism == 'mar' else 'control'}: " + "; ".join(parts))
            v = common.verdict([a - b for a, b in zip(per_mechanism["mar"], per_mechanism["control"])])
            print(f"  - (P − G on H), MAR minus control: {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
            for arm in ("G", "P"):
                mar, ctl = found[arm, "mar"], found[arm, "control"]
                assert mar is not None and ctl is not None
                e = family.effect(mar, ctl, "H", "strong")
                print(f"  - {arm}'s (H − strong), MAR minus control: {e.mean:+.4f} {e.call}")
            print()
    base, level, slopes = family.DIAL
    print(f"## The dial: `{base}_{level}`, P − G on H under MAR by slope")
    for slope in slopes:
        p = family.cells(p_runs, family.name(base, level, "mar", slope))
        g = family.cells(g_runs, family.name(base, level, "mar", slope))
        if p is None or g is None:
            print(f"- slope {slope or '1p5'}: incomplete")
            continue
        per_seed, scale = arm_gap(p, g, "H")
        v = common.verdict(per_seed, sesoi=family.SESOI * scale)
        print(f"- slope {slope or '1p5'}: {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
    print()
    print(f"- P − G on H under MAR: {common.tally(under_mar.values())}")
    print(f"- P − G on H under the control: {common.tally(under_control.values())}")
    expected = len(family.BASES) * len(family.LEVELS)
    if len(under_mar) < expected:
        print(f"\nPrimary: incomplete ({len(under_mar)} of {expected})")
        return
    better = [k for k, c in under_mar.items() if c == "better"]
    worse = [k for k, c in under_mar.items() if c == "worse"]
    harm = [k for k, c in under_control.items() if c == "worse"]
    reading = ("supported" if len(better) >= 4 and len(worse) <= 1 and len(harm) <= 1
               else "refuted" if len(better) <= 1 else "mixed")
    print(f"\nPrimary, P − G on H: under MAR better on {better or 'none'}, worse on {worse or 'none'}; "
          f"under the control worse on {harm or 'none'} -> {reading} ({common.RULE})")


def queue(which: str) -> list[tuple[str, int]]:
    work = sorted(((family.CELL_MINUTES[v.split("_")[0]], v, s) for v in family.variants() for s in family.SEEDS),
                  key=lambda i: (-i[0], i[1], i[2]))
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
