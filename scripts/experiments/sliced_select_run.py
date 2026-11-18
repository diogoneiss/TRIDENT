"""E183: the decode checkpoint chosen by the sliced Wasserstein score of the validation rows, against the default
run's (docs/tickets/imputation-protocol/72-sliced-checkpoint.md).

    uv run --python 3.11 python -B scripts/experiments/sliced_select_run.py train S <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/sliced_select_run.py --todo <queue> <queues>
    uv run --python 3.11 python -B scripts/experiments/sliced_select_run.py --tally

``S`` is the imputation task's default run with ``--decode_metric sw_score``. Its control is E181's ``Kc`` run of the
same variant and seed on the same machine (the default's epoch), and both are scored offline from their fold ledgers
by E182's code, beside E182's refitted baselines. ``--todo`` deals the cells for ``c15_queue.sh`` (``QUEUE_RUNNER``,
``QUEUE_STATE``), shortest first. ``--tally`` scores the ledgers it has not scored yet and refuses a partial read.
Run from the checkout root, on medusa1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import classcond_run as e181  # noqa: E402
from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments import sliced_run as e182  # noqa: E402
from scripts.experiments.hybrid_fresh_seeds import MINUTES  # noqa: E402

EXPERIMENT = "sliced-checkpoint-2026-11-18"
STATE = Path("/scratch/diogoneiss/trident-2026-11-18-sliced-checkpoint")
CONTROL = e181.STATE  # E181's Kc runs
BASELINES = e182.STATE  # E182's files
VARIANTS, SEEDS, SESOI = e182.VARIANTS, e182.SEEDS, e182.SESOI
SCORES = e182.SCORES
EPOCH = "decode/induced_checkpoint_epoch"


def _run(root: Path, arm: str, variant: str, seed: int) -> Path | None:
    found = sorted((root / "results" / f"{variant}_s{seed}_{arm}" / variant).glob("*/metrics/cv_summary.json"))
    return found[-1].parent.parent if found else None


def train(variant: str, seed: int) -> None:
    e181.EXPERIMENT, e181.STATE, e181.ARMS = EXPERIMENT, STATE, {"S": {"decode_metric": "sw_score"}}
    e181.train("S", variant, seed)


def todo(which: int, queues: int) -> None:
    loads = [0.0] * queues
    for cost, variant, seed in sorted((MINUTES[common.table_of(v)], v, s) for v in VARIANTS for s in SEEDS):
        target = loads.index(min(loads))
        loads[target] += cost
        if target == which - 1 and _run(STATE, "S", variant, seed) is None:
            print("S", variant, seed)


def _score(variant: str, seed: int) -> pd.DataFrame:
    """S's fills scored as E182 scored the control's: the ``reference`` group, with S's ledger in TRIDENT's place."""
    path = STATE / "csv" / f"{variant}_s{seed}__reference.csv"
    if not path.exists():
        run = _run(STATE, "S", variant, seed)
        assert run is not None
        e182.STATE, e182._run_dir = STATE, lambda variant, seed: run  # noqa: ARG005
        e182.run(variant, seed, "reference")
    return pd.read_csv(path)


def tally() -> None:
    missing = [(v, s) for v in VARIANTS for s in SEEDS if _run(STATE, "S", v, s) is None]
    wanted = [BASELINES / "csv" / f"{v}_s{s}__{g}.csv" for v in VARIANTS for s in SEEDS for g in e182.GROUPS]
    if missing or not all(p.exists() for p in wanted):
        print(f"incomplete: {len(missing)} of {len(VARIANTS) * len(SEEDS)} S cells missing, e.g. {missing[:5]}; "
              f"{sum(not p.exists() for p in wanted)} of E182's files missing")
        return
    theirs = pd.concat([pd.read_csv(p) for p in wanted], ignore_index=True)
    mine = pd.concat([_score(v, s) for v in VARIANTS for s in SEEDS], ignore_index=True)
    assert float(mine.loc[mine["method"] == "trident", "gate"].max()) < 1e-9, "a ledger does not reproduce its run's impute_score"
    calls: dict[str, dict[str, str]] = {}
    for variant in VARIANTS:
        ours, part = mine[mine["variant"] == variant], theirs[theirs["variant"] == variant]

        def folds_of(frame: pd.DataFrame, method: str, score: str) -> list[NDArray[np.float64]]:
            rows = frame[frame["method"] == method].sort_values(["seed", "fold"])
            return [rows.loc[rows["seed"] == s, score].to_numpy(dtype=float) for s in SEEDS]

        def level(frame: pd.DataFrame, method: str, score: str) -> float:
            return float(np.concatenate(folds_of(frame, method, score)).mean())

        epochs = {arm: float(np.mean([pd.read_csv(run / "metrics" / "raw_fold_metrics.csv")[EPOCH].mean()
                                      for s in SEEDS if (run := _run(root, arm, variant, s)) is not None]))
                  for arm, root in (("S", STATE), ("Kc", CONTROL))}
        seen = level(part, "shuffled", "sw_score") >= e182.CONTROL
        call = calls.setdefault(variant, {"control": "seen" if seen else "blind"})
        print(f"- `{variant}` ({call['control']}): mean kept epoch K {epochs['Kc']:.0f} → S {epochs['S']:.0f}")
        for score in SCORES:
            v = common.verdict([o - k for o, k in zip(folds_of(ours, "trident", score), folds_of(part, "trident", score))],
                               sesoi=SESOI * level(part, "trident", score))
            call[f"K/{score}"] = v.call
            print(f"  - S − K on {score} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call} (K {level(part, 'trident', score):.4f}, S {level(ours, 'trident', score):.4f})")
        failed = [m for m in e182.STRONG if float(part.loc[part["method"] == m, "gate"].max()) > e182.GATE]
        for score in ("sw_score", "impute_score"):
            name = min((m for m in e182.STRONG if m not in failed), key=lambda m: level(part, m, score))
            for arm, frame in (("S", ours), ("K", part)):
                v = common.verdict([o - b for o, b in zip(folds_of(frame, "trident", score), folds_of(part, name, score))],
                                   sesoi=SESOI * level(part, name, score))
                call[f"{arm}/strong/{score}"] = v.call
                print(f"  - {arm} − strong ({name}) on {score} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
    counted = [v for v in VARIANTS if calls[v]["control"] == "seen"]
    better = [v for v in counted if calls[v]["K/sw_score"] == "better"]
    worse = [v for v in counted if calls[v]["K/sw_score"] == "worse"]
    reading = "supported" if 2 * len(better) >= len(counted) and len(worse) <= 2 else "refuted" if 6 * len(better) <= len(counted) else "mixed"
    print(f"\nPrimary, S − K on sw_score over the {len(counted)} variants E182's control passes: better on {len(better)}, worse on {len(worse)} {worse} -> {reading}")
    for score in SCORES:
        print(f"Secondary, S − K on {score}, all variants: {common.tally(calls[v][f'K/{score}'] for v in VARIANTS)}")
    for score in ("sw_score", "impute_score"):
        moved = [v for v in VARIANTS if calls[v][f"S/strong/{score}"] != calls[v][f"K/strong/{score}"]]
        print(f"Secondary, against the best strong baseline on {score}: S {common.tally(calls[v][f'S/strong/{score}'] for v in VARIANTS)}; "
              f"K {common.tally(calls[v][f'K/strong/{score}'] for v in VARIANTS)}; calls that move: {moved}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:2] == ["train", "S"] and len(argv) == 4:
        train(argv[2], int(argv[3]))
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    elif argv == ["--tally"]:
        tally()
    else:
        sys.exit(__doc__)
