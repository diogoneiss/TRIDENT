"""D1 of T28 (E49): the decode curves, pre-trained against not, read from the store (imputation-pretraining/08).

Read-only. Run from the checkout root:

    uv run --python 3.11 python scripts/experiments/decode_curves_report.py

Each parent's ``cv/decode/val_loss/mean`` per decode epoch, from E24, E25 and E26. A pre-trained
arm (A, N, V at 150 decode epochs; M at 450) is paired by variant and seed with the
no-pre-training arm of the same schedule (B at 150, C at 450): its own study's where it has one,
E24's otherwise. Per pair: the relative gap g(e) = (L_ref − L_arm) / L_ref at epochs 0, 10 and 50,
the first epoch at which the reference is at or below the arm, and the gap between the two minima.
The reading per variant and arm is fixed in the ticket.
"""

from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")

STUDIES = {
    "E24": ("ablation", "pretraining-2026-09-29"),
    "E25": ("experiment", "pretrain-objective-2026-09-30"),
    "E26": ("experiment", "pretrain-decode-budget-2026-09-30"),
}
PAIRS = {"A": "B", "N": "B", "V": "B", "M": "C"}
KEY = "cv/decode/val_loss/mean"
THRESHOLD = 0.01

Curve = np.ndarray


def _curves() -> dict[tuple[str, str, str, int], Curve]:
    """(study, arm, variant, seed) -> the decode validation loss per epoch."""
    store = sqlite3.connect(f"file:{(ROOT / 'mlflow.db').resolve()}?mode=ro", uri=True)
    found: dict[tuple[str, str, str, int], Curve] = {}
    for study, (key, value) in STUDIES.items():
        rows = store.execute(
            f"""
            select r.run_uuid, a.value, d.value, s.value from runs r
            join tags x on x.run_uuid = r.run_uuid and x.key = ? and x.value = ?
            join tags a on a.run_uuid = r.run_uuid and a.key = 'arm'
            join tags d on d.run_uuid = r.run_uuid and d.key = 'dataset'
            join tags s on s.run_uuid = r.run_uuid and s.key = 'seed'
            join tags p on p.run_uuid = r.run_uuid and p.key = 'run_role' and p.value = 'parent'
            join tags m on m.run_uuid = r.run_uuid and m.key = 'is_mirror' and m.value = 'false'
            where r.status = 'FINISHED' and r.lifecycle_stage = 'active'""",
            (key, value),
        ).fetchall()
        for run, arm, variant, seed in rows:
            points = store.execute(
                "select step, value from metrics where run_uuid = ? and key = ? order by step, timestamp", (run, KEY)
            ).fetchall()
            by_step: dict[int, float] = {}
            for step, loss in points:
                by_step[int(step)] = float(loss)  # the last value logged at a step
            if by_step:
                found[study, str(arm), str(variant), int(seed)] = np.array([by_step[s] for s in sorted(by_step)])
    return found


def _reference(curves: dict[tuple[str, str, str, int], Curve], study: str, arm: str, variant: str, seed: int) -> tuple[str, Curve] | None:
    wanted = PAIRS[arm]
    for source in (("E25", study) if arm == "M" else (study,)) + ("E24",):
        curve = curves.get((source, wanted, variant, seed))
        if curve is not None:
            return source, curve
    return None


def main() -> int:
    curves = _curves()
    readings: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    print("| variant | arm | seed | reference | g(0) | g(10) | g(50) | catch-up epoch | gap at the minima |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (study, arm, variant, seed), curve in sorted(curves.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][3], kv[0][0])):
        if arm not in PAIRS:
            continue
        if arm == "A" and study == "E24" and (("E25", "A", variant, seed) in curves):
            continue  # E25's A is the same arm on the same seed; read it once
        found = _reference(curves, study, arm, variant, seed)
        if found is None:
            continue
        source, ref = found
        n = min(len(curve), len(ref))
        gap = (ref[:n] - curve[:n]) / ref[:n]
        behind = np.nonzero(ref[:n] <= curve[:n])[0]
        catch = int(behind[0]) if len(behind) else -1
        minima = float((ref.min() - curve.min()) / ref.min())
        row = {"g0": float(gap[0]), "g10": float(gap[min(10, n - 1)]), "g50": float(gap[min(50, n - 1)]), "catch": catch, "minima": minima}
        readings[variant, arm].append(row)
        print(f"| `{variant}` | {arm} ({study}) | {seed} | {PAIRS[arm]} ({source}) | {row['g0']:+.1%} | {row['g10']:+.1%} | "
              f"{row['g50']:+.1%} | {catch if catch >= 0 else 'never'} | {minima:+.1%} |")
    print()
    print("Reading per variant and arm (fixed in the ticket; threshold ±1%):")
    for (variant, arm), rows in sorted(readings.items()):
        if all(r["minima"] >= THRESHOLD for r in rows):
            call = "real transfer"
        elif all(r["g0"] >= THRESHOLD for r in rows):
            call = "washout"
        elif all(r["g0"] < THRESHOLD for r in rows):
            call = "nothing usable"
        else:
            call = "mixed"
        catches = [r["catch"] for r in rows if r["catch"] >= 0]
        print(f"- `{variant}` {arm} vs {PAIRS[arm]} ({len(rows)} seeds): {call}; head start {np.mean([r['g0'] for r in rows]):+.1%} at epoch 0, "
              f"{np.mean([r['g50'] for r in rows]):+.1%} at 50; gap at the minima {np.mean([r['minima'] for r in rows]):+.1%}; "
              f"caught up by epoch {int(np.median(catches)) if catches else 'never'} (median)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
