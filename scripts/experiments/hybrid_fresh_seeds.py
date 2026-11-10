"""E167 and E168: E164's missForest-side hybrid, E165's side pick and E168's cross-fit pick on fresh seeds 42/7/13, on the
12 variants E164 and E165 named (imputation-protocol/62 and /63). E169 (imputation-protocol/64) is E168's cross-fit
pick on the other 26 of E164's 38 variants, same seeds; E172 (imputation-protocol/66) the cross-fit pick on E93's 12
off-MCAR variants, seeds 101/202/303, wholly on gorgona8. Put ``--study e169`` or ``--study e172`` before any command
(default ``e167``).

    uv run --python 3.11 python -B scripts/experiments/hybrid_fresh_seeds.py train H <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/hybrid_fresh_seeds.py --todo <queue> <queues>
    uv run --python 3.11 --extra tabpfn python -B scripts/experiments/hybrid_fresh_seeds.py cells <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/hybrid_fresh_seeds.py --tally

E163's shape (mimic_hybrid.py) on OpenML variants: ``train`` is ``deployable_hybrid.train`` (H with the hidden observed
validation cells), ``cells`` is ``hybrid_selection_rules.cells`` (TabPFN refit, H's ledger asserted equal to the run's)
followed by ``missforest_hybrid.add_missforest`` (missForest refit, asserted equal to the run's logged score). Everything
on medusa1 (ADR 0035). ``--todo`` deals the cells for ``c15_queue.sh``, longest first. ``--tally`` prints every level
and pair, then E167's and E168's pre-registered readings. Run from the checkout root.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import deployable_hybrid as e82  # noqa: E402
from scripts.experiments import hybrid_confirmation as confirmation  # noqa: E402
from scripts.experiments import hybrid_selection_rules as rules  # noqa: E402
from scripts.experiments import missforest_hybrid as mh  # noqa: E402
from scripts.experiments import report_common as common  # noqa: E402

EXPERIMENT = "hybrid-fresh-seeds-2026-11-09"
STATE = Path("/scratch/diogoneiss/trident-2026-11-09-hybrid-fresh-seeds")
SEEDS = (42, 7, 13)
# E164's 7 variants where H trails missForest, E165's 3 losses (against H or `table_tabpfn`) and its 3 other wins over
# `table_tabpfn` (the three probit wins are TRAILS too), named from E164/E165's printouts before any cell.
TRAILS = ("biodeg_80nan", "vehicle_80nan", "credit-g_40nan", "biodeg_probit0p25_60nan", "biodeg_probit_60nan",
          "vehicle_probit_60nan", "electricity_20nan")
LOSSES = ("spambase_probit_60nan", "kc2_marf_60nan", "kc2_probit_60nan")
WINS = ("letter_20nan", "spambase_20nan")
VARIANTS: tuple[str, ...] = TRAILS + LOSSES + WINS
# E169 (imputation-protocol/64): the other 26 of E164's 38 variants, same seeds, same cells
REST = ("electricity_40nan", "electricity_80nan", "kr-vs-kp_40nan", "kr-vs-kp_60nan", "kr-vs-kp_80nan", "credit-g_60nan",
        "credit-g_80nan", "biodeg_40nan", "kc2_40nan", "kc2_80nan", "pendigits_40nan", "pendigits_60nan", "pendigits_80nan",
        "vehicle_40nan", "biodeg_probit1p0_60nan", "biodeg_probit2p0_60nan", "biodeg_marf_60nan", "vehicle_marf_60nan",
        "spambase_marf_60nan", "credit-g_marf_60nan", "pendigits_20nan", "kr-vs-kp_20nan", "biodeg_20nan", "credit-g_20nan",
        "vehicle_20nan", "kc2_20nan")
# E170 (imputation-protocol/65): E93's off-MCAR variants among REST (2 self-masking MNAR, 4 MAR), named before E169's read
OFF_MCAR = ("biodeg_probit1p0_60nan", "biodeg_probit2p0_60nan", "biodeg_marf_60nan", "vehicle_marf_60nan",
            "spambase_marf_60nan", "credit-g_marf_60nan")
# E172 (imputation-protocol/66): E93's 12 off-MCAR variants (7 self-masking MNAR, 5 MAR), wholly on gorgona8
E93 = ("biodeg_probit0p25_60nan", "biodeg_probit_60nan", "biodeg_probit1p0_60nan", "biodeg_probit2p0_60nan", "vehicle_probit_60nan",
       "kc2_probit_60nan", "spambase_probit_60nan", "biodeg_marf_60nan", "kc2_marf_60nan", "vehicle_marf_60nan", "spambase_marf_60nan",
       "credit-g_marf_60nan")
# study: (experiment, state, variants, seeds, host); medusa1's hostname is "medusa"
STUDIES: dict[str, tuple[str, Path, tuple[str, ...], tuple[int, int, int], str]] = {
    "e167": (EXPERIMENT, STATE, VARIANTS, SEEDS, "medusa"),
    "e169": ("hybrid-crossfit-generality-2026-11-09", Path("/scratch/diogoneiss/trident-2026-11-09-hybrid-crossfit-generality"), REST, SEEDS, "medusa"),
    "e172": ("hybrid-crossfit-offmcar-2026-11-09", Path("/scratch/diogoneiss/trident-2026-11-09-hybrid-crossfit-offmcar"), E93, (101, 202, 303), "gorgona8")}
STUDY = "e167"

# letter_20nan trains under ADR 0036's P600 since 2026-10-31: 118 minutes on its latest run, not E82's 61
MINUTES = {**confirmation.MINUTES, "letter": 118}


def use(study: str) -> None:
    global STUDY, EXPERIMENT, STATE, VARIANTS, SEEDS
    STUDY = study
    EXPERIMENT, STATE, VARIANTS, SEEDS, _ = STUDIES[study]
    e82.EXPERIMENT, e82.STATE, e82.SEEDS, e82.VARIANTS = EXPERIMENT, STATE, SEEDS, VARIANTS
    rules.STATE = STATE


use("e167")


def todo(which: int, queues: int) -> None:
    if socket.gethostname() != STUDIES[STUDY][4]:
        sys.exit(f"{STUDY} runs on {STUDIES[STUDY][4]}")
    loads = [0.0] * queues
    for cost, variant, seed in sorted(((MINUTES[v.split("_")[0]], v, s) for v in VARIANTS for s in SEEDS),
                                      key=lambda w: (-w[0], w[1], w[2])):
        target = loads.index(min(loads))
        loads[target] += cost
        if target == which - 1 and not (STATE / "selection" / f"{variant}_s{seed}" / "selection_fold_5.csv").exists():
            print("H", variant, seed)


def cells(variant: str, seed: int) -> None:
    rules.cells(variant, seed)
    mh.add_missforest(EXPERIMENT, STATE / "cells", STATE / "cells-mf", variant, seed)


def tally() -> None:
    missing = [(v, s) for v in VARIANTS for s in SEEDS if not (STATE / "cells-mf" / f"{v}_s{s}" / "selection_fold_5.csv").exists()]
    if missing:
        print(f"incomplete: {missing}")
        return
    pairs = mh.PAIRS + (("pick", "H"), ("pick", "table_tabpfn"), ("pick", "table_missforest"),
                        ("crossfit", "H"), ("crossfit", "table_tabpfn"), ("crossfit", "pick"))
    calls: dict[str, dict[str, str]] = {}
    for variant in VARIANTS:
        per = [mh.both_sides(STATE / "cells-mf" / f"{variant}_s{s}") for s in SEEDS]
        values = {m: [np.asarray(p[m]) for p in per] for m in per[0]}
        print(f"- `{variant}`{' (trails)' if variant in TRAILS else ''}: "
              + ", ".join(f"{m} {np.concatenate(v).mean():.4f}" for m, v in values.items()))
        for first, second in pairs:
            scale = float(np.mean([v.mean() for v in values[second]]))
            v = common.verdict([a - b for a, b in zip(values[first], values[second])], sesoi=mh.SESOI * scale)
            print(f"  - {first} − {second} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
            calls.setdefault(variant, {})[f"{first}-{second}"] = v.call

    def count(key: str, call: str, scope: tuple[str, ...] = VARIANTS) -> list[str]:
        return [v for v in scope if calls[v][key] == call]

    n = len(VARIANTS)
    if STUDY == "e172":
        under = count("crossfit-H", "worse")
        reading = "supported" if not under else "refuted" if len(under) >= 2 else "mixed"
        print(f"\nE172 PRIMARY `crossfit` on E93's {n} off-MCAR variants: worse than H on {under or 'none'} -> {reading}")
        print(f"E172 SECONDARY `crossfit` better than H on {count('crossfit-H', 'better') or 'none'}; `table_tabpfn` worse than H on "
              f"{count('table_tabpfn-H', 'worse') or 'none'}; `crossfit` worse than `table_tabpfn` on {count('crossfit-table_tabpfn', 'worse') or 'none'}, "
              f"better on {count('crossfit-table_tabpfn', 'better') or 'none'}")
        print(f"({common.RULE}; SESOI {mh.SESOI:.0%} of the opponent's mean)")
        return
    if STUDY == "e169":
        under, behind = count("crossfit-H", "worse"), count("crossfit-table_tabpfn", "worse")
        reading = "supported" if not under and len(behind) <= 1 else "refuted" if len(under) >= 2 or len(behind) >= 3 else "mixed"
        print(f"\nE169 PRIMARY `crossfit` on {n}: worse than H on {under or 'none'}, worse than `table_tabpfn` on {behind or 'none'} -> {reading}")
        print(f"E169 SECONDARY `crossfit` better than `table_tabpfn` on {count('crossfit-table_tabpfn', 'better') or 'none'}; better than H "
              f"on {len(count('crossfit-H', 'better'))} of {n}; `crossfit` − `pick` better on {count('crossfit-pick', 'better') or 'none'}, "
              f"worse on {count('crossfit-pick', 'worse') or 'none'}")
        print(f"E169 SECONDARY `table_missforest`: worse than H on {count('table_missforest-H', 'worse') or 'none'}, worse than missForest "
              f"on {count('table_missforest-missforest', 'worse') or 'none'}")
        under = count("crossfit-H", "worse", OFF_MCAR)
        reading = "supported" if not under else "refuted" if len(under) >= 2 else "mixed"
        print(f"E170 PRIMARY `crossfit` on E93's {len(OFF_MCAR)} off-MCAR variants here: worse than H on {under or 'none'} -> {reading}")
        print(f"E170 SECONDARY on them: `crossfit` better than H on {count('crossfit-H', 'better', OFF_MCAR) or 'none'}; `table_tabpfn` "
              f"worse than H on {count('table_tabpfn-H', 'worse', OFF_MCAR) or 'none'}; `crossfit` worse than `table_tabpfn` on "
              f"{count('crossfit-table_tabpfn', 'worse', OFF_MCAR) or 'none'}")
        print(f"({common.RULE}; SESOI {mh.SESOI:.0%} of the opponent's mean)")
        return
    under, behind = count("table_missforest-H", "worse"), count("table_missforest-missforest", "worse")
    reading = "supported" if not under and len(behind) <= 1 else "refuted" if len(under) >= 2 or len(behind) >= 3 else "mixed"
    print(f"\nE167 PRIMARY `table_missforest` on {n}: worse than H on {under or 'none'}, worse than missForest on {behind or 'none'} -> {reading}")
    print(f"E167 SECONDARY better than H on {len(count('table_missforest-H', 'better'))} of {n}; on the {len(TRAILS)} trails: "
          f"{count('table_missforest-H', 'better', TRAILS) or 'none'}")
    for rule, label in (("pick", "E167 PICK"), ("crossfit", "E168 PRIMARY")):
        under, over, behind = count(f"{rule}-H", "worse"), count(f"{rule}-table_tabpfn", "better"), count(f"{rule}-table_tabpfn", "worse")
        reading = ("supported" if not under and len(behind) <= 1 and len(over) >= 2
                   else "refuted" if len(behind) >= 3 or not over else "mixed")
        print(f"{label} `{rule}`: worse than H on {under or 'none'}; against `table_tabpfn` better on {over or 'none'}, "
              f"worse on {behind or 'none'} -> {reading}")
    print(f"E168 SECONDARY `crossfit` − `pick`: better on {count('crossfit-pick', 'better') or 'none'}, worse on {count('crossfit-pick', 'worse') or 'none'}")
    print(f"({common.RULE}; SESOI {mh.SESOI:.0%} of the opponent's mean)")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--study"] and len(argv) >= 2 and argv[1] in STUDIES:
        use(argv[1])
        argv = argv[2:]
    if argv[:1] == ["train"] and len(argv) == 4 and argv[1] == "H":
        e82.train(argv[2], int(argv[3]), False)
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    elif argv[:1] == ["cells"] and len(argv) == 3:
        cells(argv[1], int(argv[2]))
    elif argv == ["--tally"]:
        tally()
    else:
        sys.exit(__doc__)
