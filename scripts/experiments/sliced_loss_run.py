"""E184: a sliced Wasserstein term in the decode loss (``--decode_sw_weight``), read on the output's sliced Wasserstein
score (docs/tickets/imputation-protocol/73), on gorgona8 only so each pair stays on one GPU model.

    uv run --python 3.11 python -B scripts/experiments/sliced_loss_run.py train <S|LS|K|L> <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/sliced_loss_run.py --todo <queue> <queues>
    uv run --python 3.11 python -B scripts/experiments/sliced_loss_run.py --gate
    uv run --python 3.11 python -B scripts/experiments/sliced_loss_run.py --pairing
    uv run --python 3.11 python -B scripts/experiments/sliced_loss_run.py --tally [core]

Core arms: ``S`` (``--decode_metric sw_score``, E183's arm) and ``LS`` (S plus the term at ``WEIGHT``); second batch:
``K`` (the default run) and ``L`` (K plus the term). ``WEIGHT`` is fixed by the weight smoke before launch, and
``train``, ``--gate`` and ``--tally`` refuse while it is None. ``--todo`` deals the cells for ``c15_queue.sh`` (``QUEUE_RUNNER``, ``QUEUE_STATE``):
the core first, then the second batch, a pair's two arms side by side, shortest variants first. ``--gate`` checks each
finished run's flags from its run directory and, read-only from the store, that a run with the term logged it.
``--pairing`` checks each pair's scored cells and hgb's score agree (the loss epoch differs by design). ``--tally``
scores the ledgers by E182's code into ``STATE/csv/<arm>/csv/`` and refuses a partial read; ``core`` reads S and LS
only. Run from the checkout root.
"""

from __future__ import annotations

import json
import sqlite3
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
from scripts.experiments import sliced_select_run as e183  # noqa: E402
from scripts.experiments.hybrid_fresh_seeds import MINUTES  # noqa: E402

EXPERIMENT = "sliced-loss-2026-11-19"
STATE = Path("/scratch/diogoneiss/trident-2026-11-19-sliced-loss")
WEIGHT: float | None = 0.8  # set by the weight smoke of 2026-11-19 18:15 GMT-3 (ticket 73), before any cell
ARMS: dict[str, dict[str, object]] = {"S": {"decode_metric": "sw_score"}, "LS": {"decode_metric": "sw_score"}, "K": {}, "L": {}}
PAIRS: tuple[tuple[str, str], ...] = (("S", "LS"), ("K", "L"))  # (control, with the term): the core, then the second batch
TERM = ("LS", "L")  # the arms that add the term, at WEIGHT read when called (never captured at import)
VARIANTS: tuple[str, ...] = (
    "kc2_20nan", "kc2_40nan", "kc2_80nan", "vehicle_20nan", "vehicle_40nan", "vehicle_80nan",
    "biodeg_20nan", "biodeg_40nan", "biodeg_80nan", "credit-g_20nan", "credit-g_40nan", "credit-g_60nan",
    "credit-g_80nan", "kr-vs-kp_20nan", "kr-vs-kp_40nan", "kr-vs-kp_60nan", "kr-vs-kp_80nan", "spambase_20nan",
)
assert set(VARIANTS) <= set(e181.VARIANTS)
SEEDS: tuple[int, ...] = (101, 202, 303)
SESOI, SCORES, EPOCH = e182.SESOI, e182.SCORES, e183.EPOCH
BASELINES = e182.STATE  # E182's refitted baselines
SAME = ("impute/induced/n_num_cells", "impute/induced/n_cat_cells", "impute/induced/baseline/hgb/impute_score")
STORE = ROOT / "mlflow.db"


def _weight() -> float:
    if WEIGHT is None:
        sys.exit("WEIGHT is not set: the weight smoke fixes it in this file's source before any cell runs")
    return float(WEIGHT)


def flags(arm: str) -> dict[str, object]:
    """The arm's training flags; a term arm carries ``decode_sw_weight`` at the current WEIGHT."""
    return {**ARMS[arm], **({"decode_sw_weight": _weight()} if arm in TERM else {})}


def _run(arm: str, variant: str, seed: int) -> Path | None:
    return e183._run(STATE, arm, variant, seed)


def train(arm: str, variant: str, seed: int) -> None:
    _weight()  # every arm waits for the weight, so the controls never run ahead of it
    assert variant in VARIANTS and seed in SEEDS, (variant, seed)
    e181.EXPERIMENT, e181.STATE, e181.ARMS = EXPERIMENT, STATE, {arm: flags(arm)}
    e181.train(arm, variant, seed)


def todo(which: int, queues: int) -> None:
    by_cost = sorted(VARIANTS, key=lambda v: (MINUTES[common.table_of(v)], v))
    cells = [(arm, v, s) for pair in PAIRS for v in by_cost for s in SEEDS for arm in pair]
    for k, (arm, variant, seed) in enumerate(cells):
        if k % queues == which - 1 and _run(arm, variant, seed) is None:
            print(arm, variant, seed)


def _logged_sliced(store: sqlite3.Connection, arm: str, variant: str, seed: int) -> list[float]:
    """The largest ``decode/train_sliced`` of each diagnostic child (the best and worst folds, the only ones that log
    per-epoch decode metrics) of the cell's latest parent run in the store."""
    parent = store.execute(
        "SELECT r.run_uuid FROM runs r"
        " JOIN tags e ON e.run_uuid = r.run_uuid AND e.key = 'experiment' AND e.value = ?"
        " JOIN tags a ON a.run_uuid = r.run_uuid AND a.key = 'arm' AND a.value = ?"
        " JOIN tags d ON d.run_uuid = r.run_uuid AND d.key = 'dataset' AND d.value = ?"
        " JOIN tags s ON s.run_uuid = r.run_uuid AND s.key = 'seed' AND s.value = ?"
        " JOIN tags m ON m.run_uuid = r.run_uuid AND m.key = 'is_mirror' AND m.value = 'false'"
        " JOIN tags p ON p.run_uuid = r.run_uuid AND p.key = 'run_role' AND p.value = 'parent'"
        " WHERE r.lifecycle_stage = 'active' AND r.status = 'FINISHED' ORDER BY r.start_time DESC LIMIT 1",
        (EXPERIMENT, arm, variant, str(seed)),
    ).fetchone()
    if parent is None:
        return []
    rows = store.execute(
        "SELECT t.run_uuid, MAX(m.value) FROM tags t LEFT JOIN metrics m ON m.run_uuid = t.run_uuid AND m.key = 'decode/train_sliced'"
        " WHERE t.key = 'mlflow.parentRunId' AND t.value = ? GROUP BY t.run_uuid",
        (parent[0],),
    ).fetchall()
    return [float("nan") if value is None else float(value) for _, value in rows]


def gate() -> None:
    weight = _weight()
    store = sqlite3.connect(f"file:{STORE}?mode=ro", uri=True)
    checked = failed = 0
    for arm, wanted in ARMS.items():
        for variant in VARIANTS:
            for seed in SEEDS:
                run = _run(arm, variant, seed)
                if run is None:
                    continue
                checked += 1
                config = json.loads((run / "hyperparameters.json").read_text())
                want_weight = weight if arm in TERM else 0.0
                want_metric = str(wanted.get("decode_metric", "impute_score"))
                problems = []
                if config.get("DECODE_SW_WEIGHT") != want_weight:
                    problems.append(f"DECODE_SW_WEIGHT {config.get('DECODE_SW_WEIGHT')} not {want_weight}")
                if config.get("DECODE_METRIC") != want_metric:
                    problems.append(f"DECODE_METRIC {config.get('DECODE_METRIC')} not {want_metric}")
                logged = _logged_sliced(store, arm, variant, seed)
                if not logged:
                    problems.append("no parent run with diagnostic children in the store")
                elif want_weight > 0 and not all(value > 0 for value in logged):
                    problems.append(f"decode/train_sliced not logged above 0 on every child: {logged}")
                elif want_weight == 0 and not all(np.isnan(value) for value in logged):
                    problems.append(f"a control logged decode/train_sliced: {logged}")
                if problems:
                    failed += 1
                    print(f"FAIL {arm} {variant} seed {seed}: " + "; ".join(problems))
    print(f"gate: {checked} finished runs checked, {failed} failed (the store's term check covers each run's two diagnostic folds)")


def pairing() -> None:
    for control, term in PAIRS:
        for variant in VARIANTS:
            for seed in SEEDS:
                runs = [_run(arm, variant, seed) for arm in (control, term)]
                if runs == [None, None]:
                    continue
                if None in runs:
                    print(f"{control}/{term} {variant} seed {seed}: not both arms finished")
                    continue
                a, b = (pd.read_csv(run / "metrics" / "raw_fold_metrics.csv").sort_values("fold") for run in runs if run)
                same = {c: bool((a[c].to_numpy() == b[c].to_numpy()).all()) for c in SAME}
                print(f"{control}/{term} {variant} seed {seed}: "
                      + ", ".join(f"{c.split('/')[-2] if c.endswith('impute_score') else c.split('/')[-1]} {'same' if ok else 'DIFFERS'}"
                                  for c, ok in same.items()))


def _score(arm: str, variant: str, seed: int) -> pd.DataFrame:
    """The arm's fills scored as E182 scored E181's: the ``reference`` group, with the arm's ledger in TRIDENT's place."""
    home = STATE / "csv" / arm
    path = home / "csv" / f"{variant}_s{seed}__reference.csv"
    if not path.exists():
        run = _run(arm, variant, seed)
        assert run is not None
        e182.STATE, e182._run_dir = home, lambda variant, seed: run  # noqa: ARG005
        e182.run(variant, seed, "reference")
    return pd.read_csv(path)


def _rule(better: int, worse: int, counted: int) -> str:
    if not counted:
        return "no read (the control passes on no variant)"
    return "supported" if 2 * better >= counted and worse <= 2 else "refuted" if 6 * better <= counted else "mixed"


def tally(core: bool) -> None:
    _weight()  # an unedited source would read runs it never trained
    pairs = PAIRS[:1] if core else PAIRS
    arms = [arm for pair in pairs for arm in pair]
    missing = [(a, v, s) for a in arms for v in VARIANTS for s in SEEDS if _run(a, v, s) is None]
    groups = [g for g in e182.GROUPS if g != "reference"]
    wanted = [BASELINES / "csv" / f"{v}_s{s}__{g}.csv" for v in VARIANTS for s in SEEDS for g in groups]
    if missing or not all(p.exists() for p in wanted):
        sys.exit(f"incomplete: {len(missing)} of {len(arms) * len(VARIANTS) * len(SEEDS)} cells missing, e.g. {missing[:5]}; "
                 f"{sum(not p.exists() for p in wanted)} of E182's baseline files missing")
    theirs = pd.concat([pd.read_csv(p) for p in wanted], ignore_index=True)
    mine = {arm: pd.concat([_score(arm, v, s) for v in VARIANTS for s in SEEDS], ignore_index=True) for arm in arms}
    for arm, frame in mine.items():
        assert float(frame.loc[frame["method"] == "trident", "gate"].max()) < 1e-9, f"{arm}: a ledger does not reproduce its run's impute_score"
    calls: dict[str, dict[str, str]] = {}
    for variant in VARIANTS:
        ours = {arm: frame[frame["variant"] == variant] for arm, frame in mine.items()}
        part = theirs[theirs["variant"] == variant]

        def folds_of(frame: pd.DataFrame, method: str, score: str) -> list[NDArray[np.float64]]:
            rows = frame[frame["method"] == method].sort_values(["seed", "fold"])
            return [rows.loc[rows["seed"] == s, score].to_numpy(dtype=float) for s in SEEDS]

        def level(frame: pd.DataFrame, method: str, score: str) -> float:
            return float(np.concatenate(folds_of(frame, method, score)).mean())

        epochs = {arm: float(np.mean([pd.read_csv(run / "metrics" / "raw_fold_metrics.csv")[EPOCH].mean()
                                      for s in SEEDS if (run := _run(arm, variant, s)) is not None])) for arm in arms}
        seen = level(ours["S"], "shuffled", "sw_score") >= e182.CONTROL
        call = calls.setdefault(variant, {"control": "seen" if seen else "blind"})
        print(f"- `{variant}` ({call['control']}): mean kept epoch " + ", ".join(f"{a} {e:.0f}" for a, e in epochs.items()))
        for control, term in pairs:
            for score in SCORES:
                v = common.verdict([o - k for o, k in zip(folds_of(ours[term], "trident", score), folds_of(ours[control], "trident", score))],
                                   sesoi=SESOI * level(ours[control], "trident", score))
                call[f"{term}/{score}"] = v.call
                print(f"  - {term} − {control} on {score} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call} "
                      f"({control} {level(ours[control], 'trident', score):.4f}, {term} {level(ours[term], 'trident', score):.4f})")
        failed = [m for m in e182.STRONG if float(part.loc[part["method"] == m, "gate"].max()) > e182.GATE]
        for score in ("sw_score", "impute_score"):
            name = min((m for m in e182.STRONG if m not in failed), key=lambda m: level(part, m, score))
            for arm in arms:
                v = common.verdict([o - b for o, b in zip(folds_of(ours[arm], "trident", score), folds_of(part, name, score))],
                                   sesoi=SESOI * level(part, name, score))
                call[f"{arm}/strong/{score}"] = v.call
                print(f"  - {arm} − strong ({name}) on {score} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
    counted = [v for v in VARIANTS if calls[v]["control"] == "seen"]
    for control, term in pairs:
        better = [v for v in counted if calls[v][f"{term}/sw_score"] == "better"]
        worse = [v for v in counted if calls[v][f"{term}/sw_score"] == "worse"]
        label = "Primary" if term == "LS" else "Second batch"
        print(f"\n{label}, {term} − {control} on sw_score over the {len(counted)} variants the control passes: "
              f"better on {len(better)}, worse on {len(worse)} {worse} -> {_rule(len(better), len(worse), len(counted))}")
        for score in SCORES[1:]:
            print(f"Secondary, {term} − {control} on {score}, all variants: {common.tally(calls[v][f'{term}/{score}'] for v in VARIANTS)}")
    for score in ("sw_score", "impute_score"):
        print(f"Secondary, against the best strong baseline on {score}: "
              + "; ".join(f"{arm} {common.tally(calls[v][f'{arm}/strong/{score}'] for v in VARIANTS)}" for arm in arms))


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and len(argv) == 4 and argv[1] in ARMS:
        train(argv[1], argv[2], int(argv[3]))
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    elif argv == ["--gate"]:
        gate()
    elif argv == ["--pairing"]:
        pairing()
    elif argv[:1] == ["--tally"] and argv[1:] in ([], ["core"]):
        tally(core=argv[1:] == ["core"])
    else:
        sys.exit(__doc__)
