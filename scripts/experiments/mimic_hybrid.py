"""E163: T39's deployable hybrid on E149's MIMIC-IV transplant variants, TabPFN or missForest as the other side (mimic/10).
E166 (mimic/11) is the same study on seeds 4545/5656/6767, and E171 (mimic/12) the cross-fit pick on seeds 101/202/303,
wholly on gorgona8. E173 (mimic/13) is decision 6's hybrid on E148's two tables with natural gaps, scored on the
masked population (hidden observed cells; they have no induced ones), on two seed sets, ``e173`` (101/202/303) and
``e173b`` (4545/5656/6767), ``e173`` on gorgona8 and ``e173b`` on medusa1 (split by seed set, ADR 0035). E178
(mimic/16) is the hybrid on E176's ``mimic-wide`` variants, the natural table on its masked population and the donor
table's ``tmatch`` and ``mcar`` on their induced cells, seeds 42/7/13, on medusa1; its primaries are the natural table
and ``tmatch``. Put
``--study <id>`` before any command (default ``e163``).

    uv run --python 3.11 python -B scripts/experiments/mimic_hybrid.py [--study e166] train H <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/mimic_hybrid.py [--study e166] --todo <queue> <queues>
    uv run --python 3.11 --extra tabpfn python -B scripts/experiments/mimic_hybrid.py [--study e166] cells <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/mimic_hybrid.py [--study e166] --tally

``train`` is ``deployable_hybrid.train`` (H, with the hidden observed validation cells read in place) under this
study's tag and state. ``cells`` is ``hybrid_selection_rules.cells`` (TabPFN refit, H's ledger asserted equal to the
run's H) followed by ``missforest_hybrid.add_missforest`` (missForest refit, asserted equal to the run's logged
missForest), so one cell table holds H's, TabPFN's and missForest's fills. Every step runs on medusa1 (ADR 0035: the
whole study on one machine; gorgona8's GPU is the other project's). ``--todo`` deals the cells for ``c15_queue.sh``
with ``QUEUE_RUNNER`` and ``QUEUE_STATE``. ``--tally`` prints every method's level and the fold-pair verdicts, then the
primary reading on the two ``tmatch`` variants. Run from the checkout root with ``TRIDENT_DATASETS_ROOT`` set.
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
from scripts.experiments import hybrid_selection_rules as rules  # noqa: E402
from scripts.experiments import missforest_hybrid as mh  # noqa: E402
from scripts.experiments import report_common as common  # noqa: E402

EXPERIMENT = "mimic-hybrid-2026-11-08"
STATE = Path("/scratch/diogoneiss/trident-2026-11-08-mimic-hybrid")
SEEDS = (42, 7, 13)
VARIANTS = tuple(f"mimic-{t}-donor_{v}_20nan" for t in ("cardio", "sofa") for v in ("tmatch", "trand", "mcar"))
PRIMARY = tuple(v for v in VARIANTS if "_tmatch_" in v)
TRANSPLANT = VARIANTS
NATURAL = ("mimic-cardio_natural", "mimic-sofa_natural")  # E173: E148's tables, both primary
WIDE = ("mimic-wide_natural", "mimic-wide-donor_tmatch_20nan", "mimic-wide-donor_mcar_20nan")  # E178: E176's variants
STUDIES = {"e163": (EXPERIMENT, STATE, SEEDS),
           "e166": ("mimic-hybrid-seeds-2026-11-09", Path("/scratch/diogoneiss/trident-2026-11-09-mimic-hybrid-seeds"), (4545, 5656, 6767)),
           "e171": ("mimic-hybrid-crossfit-2026-11-09", Path("/scratch/diogoneiss/trident-2026-11-09-mimic-crossfit"), (101, 202, 303)),
           "e173": ("mimic-hybrid-natural-2026-11-10", Path("/scratch/diogoneiss/trident-2026-11-10-mimic-natural"), (101, 202, 303)),
           "e173b": ("mimic-hybrid-natural-b-2026-11-10", Path("/scratch/diogoneiss/trident-2026-11-10-mimic-natural-b"), (4545, 5656, 6767)),
           "e178": ("mimic-wide-hybrid-2026-11-13", Path("/scratch/diogoneiss/trident-2026-11-13-mimic-wide-hybrid"), (42, 7, 13))}
HOSTS = {"e163": "medusa", "e166": "medusa", "e171": "gorgona8", "e173": "gorgona8", "e173b": "medusa", "e178": "medusa"}  # the machine each study runs on whole (ADR 0035)
STUDY = "e163"


def use(study: str) -> None:
    global STUDY, EXPERIMENT, STATE, SEEDS, VARIANTS, PRIMARY
    STUDY = study
    EXPERIMENT, STATE, SEEDS = STUDIES[study]
    VARIANTS = NATURAL if study.startswith("e173") else WIDE if study == "e178" else TRANSPLANT
    PRIMARY = tuple(v for v in VARIANTS if "_tmatch_" in v or v.endswith("_natural"))
    e82.EXPERIMENT, e82.STATE, e82.SEEDS, e82.VARIANTS = EXPERIMENT, STATE, SEEDS, VARIANTS
    rules.STATE = STATE


use("e163")


def todo(which: int, queues: int) -> None:
    if socket.gethostname() != HOSTS[STUDY]:  # medusa1's hostname is "medusa"
        sys.exit(f"{STUDY} runs on {HOSTS[STUDY]}")
    for k, (variant, seed) in enumerate((v, s) for s in SEEDS for v in VARIANTS):
        if k % queues == which - 1 and not (STATE / "selection" / f"{variant}_s{seed}" / "selection_fold_5.csv").exists():
            print("H", variant, seed)


def cells(variant: str, seed: int) -> None:
    # a natural table has no induced cells: it is scored on its masked population (hidden observed cells)
    population = "masked" if variant.endswith("_natural") else "induced"
    rules.cells(variant, seed, population)
    mh.add_missforest(EXPERIMENT, STATE / "cells", STATE / "cells-mf", variant, seed, population=population)


def tally() -> None:
    scope = [(v, s) for v in VARIANTS for s in SEEDS]
    missing = [p for p in scope if not (STATE / "cells-mf" / f"{p[0]}_s{p[1]}" / "selection_fold_5.csv").exists()]
    if missing:
        print(f"incomplete: {missing}")
        return
    pairs = mh.PAIRS + (("pick", "H"), ("pick", "table_tabpfn"), ("pick", "table_missforest"), ("crossfit", "H"), ("crossfit", "table_tabpfn"))
    calls: dict[str, dict[str, str]] = {}
    for variant in VARIANTS:
        per = [mh.both_sides(STATE / "cells-mf" / f"{variant}_s{s}") for s in SEEDS]
        values = {m: [np.asarray(p[m]) for p in per] for m in per[0]}
        print(f"- `{variant}`{' (primary)' if variant in PRIMARY else ''}: "
              + ", ".join(f"{m} {np.concatenate(v).mean():.4f}" for m, v in values.items()))
        for first, second in pairs:
            scale = float(np.mean([v.mean() for v in values[second]]))
            v = common.verdict([a - b for a, b in zip(values[first], values[second])], sesoi=mh.SESOI * scale)
            print(f"  - {first} − {second} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
            calls.setdefault(variant, {})[f"{first}-{second}"] = v.call
        print(f"  - the pick took missForest on {int(np.concatenate(values['picked_missforest']).sum())} of 15 folds, the cross-fit pick on "
              f"{int(np.concatenate(values['crossfit_missforest']).sum())}")
    over = [v for v in PRIMARY if calls[v]["table_tabpfn-H"] == "better"]
    under = [v for v in PRIMARY if calls[v]["table_tabpfn-H"] == "worse"]
    behind = [v for v in PRIMARY if calls[v]["table_tabpfn-tabpfn"] == "worse"]
    reading = ("refuted" if not over or under else "supported" if not behind else "mixed")
    label = {"e171": "E163 rule (descriptive in E171)", "e173": "E173 PRIMARY, seed set 101/202/303:",
             "e173b": "E173 PRIMARY, seed set 4545/5656/6767:", "e178": "E178 PRIMARY"}.get(STUDY, "PRIMARY")
    print(f"\n{label} `table` (TabPFN side) on {', '.join(PRIMARY)}: better than H on {over or 'none'}, worse than H on "
          f"{under or 'none'}, worse than TabPFN on {behind or 'none'} -> {reading}")
    if STUDY.startswith("e173"):
        print(f"E173 SECONDARY: `table_tabpfn` better than TabPFN on {[v for v in VARIANTS if calls[v]['table_tabpfn-tabpfn'] == 'better'] or 'none'}; "
              f"`table_missforest` worse than H on {[v for v in VARIANTS if calls[v]['table_missforest-H'] == 'worse'] or 'none'}; "
              f"`crossfit` worse than `table_tabpfn` on {[v for v in VARIANTS if calls[v]['crossfit-table_tabpfn'] == 'worse'] or 'none'}")
        print("E173 reads supported if both seed sets read supported, refuted if either reads refuted, mixed otherwise")
    if STUDY == "e171":
        under = [v for v in PRIMARY if calls[v]["crossfit-H"] == "worse"]
        behind = [v for v in PRIMARY if calls[v]["crossfit-table_tabpfn"] == "worse"]
        reading = "refuted" if under or len(behind) == len(PRIMARY) else "supported" if not behind else "mixed"
        print(f"E171 PRIMARY `crossfit` on {', '.join(PRIMARY)}: worse than H on {under or 'none'}, worse than `table_tabpfn` on "
              f"{behind or 'none'} -> {reading}")
        print(f"E171 SECONDARY on all six: `crossfit` worse than H on {[v for v in VARIANTS if calls[v]['crossfit-H'] == 'worse'] or 'none'}, "
              f"worse than `table_tabpfn` on {[v for v in VARIANTS if calls[v]['crossfit-table_tabpfn'] == 'worse'] or 'none'}")
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
