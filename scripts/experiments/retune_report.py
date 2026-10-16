"""Pre-registered analysis of the re-tuning of the promoted imputation files (imputation-retune/01).

Read-only. Lower impute_score is better, so a negative (X - Y) difference favours X. Run from the
checkout root:

    uv run --python 3.11 python scripts/experiments/retune_report.py [variant ...]
    uv run --python 3.11 python scripts/experiments/retune_report.py --tally
    uv run --python 3.11 python scripts/experiments/retune_report.py --studies
    uv run --python 3.11 python scripts/experiments/retune_report.py --jobs

Three arms on seeds 101, 202 and 303, five folds each. T (the variant's study winner) and D (today's
defaults, no promoted file) come from this study (tag experiment=retune-2026-10-15, tag arm). F, the
variant's current promoted file under today's defaults, is read from E34's cells (tag
experiment=selection-2026-10-03) as their K readout (``impute/*/induced_checkpoint/*``), which equals
a default run to the last digit. The studies are the Optuna study runs of the reduced profile ranked
by the induced validation objective, started after the pre-registration. The integrity check holds
each seed's baseline scores, which never depend on the model, equal across the three arms.
``--jobs`` prints the job list the four workers share, in priority order.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
from scipy.stats import t

# Longest study first, so the two spambase studies, which bound the night, start at once.
VARIANTS = ["spambase_40nan", "spambase_20nan", "kr-vs-kp_40nan", "kr-vs-kp_20nan", "credit-g_40nan"]
SEEDS = [101, 202, 303]
ARMS = ("T", "D", "F")
CONTRASTS = (("T", "F"), ("D", "F"), ("T", "D"))
BASELINES = ("mean_mode", "knn5", "knn10", "hgb")
EXPERIMENT = "retune-2026-10-15"
REFERENCE = "selection-2026-10-03"
OBJECTIVE = "validation/impute/induced/impute_score"
# The pre-registration commit was made after this instant; a study run started before it is
# not this study's.
SINCE_MS = int(datetime(2026, 10, 16, 2, 0, tzinfo=timezone.utc).timestamp() * 1000)
SAMPLED = ("PROB_MASCARA", "LR_DECODE", "WEIGHT_DECAY_DECODE", "DROPOUT", "LAMBDA_NUM")
KEYS = {
    "induced": ("impute/induced/impute_score", "impute/induced/induced_checkpoint/impute_score"),
    "masked": ("impute/masked/impute_score", "impute/masked/induced_checkpoint/impute_score"),
}
STORE = sqlite3.connect("file:mlflow.db?mode=ro", uri=True, timeout=30)
SOURCE = """r.status = 'FINISHED' and r.lifecycle_stage = 'active'
            and r.run_uuid not in (select run_uuid from tags where key = 'is_mirror' and value = 'true')"""


def _path(uri: str) -> Path:
    path = unquote(urlparse(uri).path)
    return Path(path[1:] if re.match(r"^/[A-Za-z]:", path) else path)


def _tagged(experiment: str) -> dict[tuple[str, int, str], list[str]]:
    rows = STORE.execute(
        f"""
        select d.value, s.value, coalesce(a.value, 'F'), r.artifact_uri from runs r
        join tags x on x.run_uuid = r.run_uuid and x.key = 'experiment' and x.value = ?
        join tags d on d.run_uuid = r.run_uuid and d.key = 'dataset'
        join tags s on s.run_uuid = r.run_uuid and s.key = 'seed'
        join tags p on p.run_uuid = r.run_uuid and p.key = 'run_role' and p.value = 'parent'
        left join tags a on a.run_uuid = r.run_uuid and a.key = 'arm'
        where {SOURCE}""",
        (experiment,),
    )
    found: dict[tuple[str, int, str], list[str]] = {}
    for variant, seed, arm, uri in rows:
        if variant in VARIANTS and int(seed) in SEEDS:
            found.setdefault((variant, int(seed), arm), []).append(uri)
    return found


# E34's runs carry no arm tag: every one is an F cell for this study.
RUNS = {**_tagged(EXPERIMENT), **_tagged(REFERENCE)}


def folds(variant: str, seed: int, arm: str) -> pd.DataFrame | None:
    uris = RUNS.get((variant, seed, arm), [])
    if len(uris) > 1:
        sys.exit(f"{variant} s{seed} {arm}: {len(uris)} finished runs; the design allows one")
    if not uris:
        return None
    frame = pd.read_csv(_path(uris[0]) / "metrics" / "raw_fold_metrics.csv")
    return frame.sort_values("fold").reset_index(drop=True)


def score(cell: pd.DataFrame, arm: str, population: str = "induced") -> np.ndarray:
    arm_key, reference_key = KEYS[population]
    return cell[reference_key if arm == "F" else arm_key].to_numpy()


def interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    margin = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - margin, mean + margin


def _cells(variant: str, arm: str) -> list[pd.DataFrame] | None:
    cells = [folds(variant, s, arm) for s in SEEDS]
    complete = [c for c in cells if c is not None]
    return complete if len(complete) == len(SEEDS) else None


def verdict(
    variant: str, x: str, y: str, population: str = "induced"
) -> tuple[str, float, float, float, list[float]] | None:
    xs, ys = _cells(variant, x), _cells(variant, y)
    if xs is None or ys is None:
        return None
    per_seed = [score(a, x, population) - score(b, y, population) for a, b in zip(xs, ys)]
    mean, lo, hi = interval(np.concatenate(per_seed))
    signs = [float(d.mean()) for d in per_seed]
    if hi < 0 and all(s < 0 for s in signs):
        call = f"{x} better"
    elif lo > 0 and all(s > 0 for s in signs):
        call = f"{y} better"
    else:
        call = "no detectable difference"
    return call, mean, lo, hi, signs


def recommendation(variant: str) -> str:
    """The pre-stated rule: an arm qualifies when it beats another arm by verdict and loses to none."""
    calls = {(x, y): verdict(variant, x, y) for x, y in CONTRASTS}
    if any(result is None for result in calls.values()):
        return "incomplete"
    beats: set[tuple[str, str]] = set()
    for (x, y), result in calls.items():
        assert result is not None
        if result[0] == f"{x} better":
            beats.add((x, y))
        elif result[0] == f"{y} better":
            beats.add((y, x))

    def qualifies(arm: str) -> bool:
        others = [a for a in ARMS if a != arm]
        return any((arm, o) in beats for o in others) and not any((o, arm) in beats for o in others)

    if qualifies("T"):
        return "promote T (the study's winner)"
    if qualifies("D") and ("D", "F") in beats:
        return "remove the promoted file (D)"
    return "keep F (the current file)"


def _best_baseline(cell: pd.DataFrame) -> np.ndarray:
    name = min(BASELINES, key=lambda b: float(cell[f"impute/induced/baseline/{b}/impute_score"].mean()))
    return cell[f"impute/induced/baseline/{name}/impute_score"].to_numpy()


def _minutes(variant: str, arm: str) -> float | None:
    cells = [c for s in SEEDS if (c := folds(variant, s, arm)) is not None]
    return float(np.mean([float(c["total_seconds"].sum()) / 60 for c in cells])) if cells else None


def report(variant: str) -> None:
    print(f"=== {variant}")
    present = {arm: _cells(variant, arm) for arm in ARMS}
    means = "  ".join(
        f"{arm} {float(np.concatenate([score(c, arm) for c in cells]).mean()):.4f}"
        for arm, cells in present.items()
        if cells is not None
    )
    reference = present["F"]
    base = float(np.concatenate([_best_baseline(c) for c in reference]).mean()) if reference else float("nan")
    print(f"  induced means  {means}  best baseline {base:.4f}")
    for population, role in (("induced", "primary"), ("masked", "secondary")):
        for x, y in CONTRASTS:
            result = verdict(variant, x, y, population)
            if result is None:
                print(f"    {x} - {y}, {population}: incomplete")
                continue
            call, mean, lo, hi, signs = result
            seeds = ", ".join(f"s{s} {d:+.4f}" for s, d in zip(SEEDS, signs))
            print(f"    {x} - {y}, {population} ({role}) over 15 pairs: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]; {seeds} -> {call}")
    print(f"  recommendation (rule fixed in advance): {recommendation(variant)}")
    minutes = ", ".join(f"{arm} {m:.1f}" for arm in ARMS if (m := _minutes(variant, arm)) is not None)
    print(f"  minutes a cell: {minutes}")


def integrity() -> None:
    """Each seed's baseline scores never depend on the model, so they must be equal across arms."""
    checked, differing = 0, []
    for variant in VARIANTS:
        for seed in SEEDS:
            cells = {arm: folds(variant, seed, arm) for arm in ARMS}
            reference = cells["F"]
            for arm in ("T", "D"):
                cell = cells[arm]
                if cell is None or reference is None:
                    continue
                checked += 1
                for name in BASELINES:
                    key = f"impute/induced/baseline/{name}/impute_score"
                    if not np.array_equal(cell[key].to_numpy(), reference[key].to_numpy()):
                        differing.append(f"{variant} s{seed} {arm} {name}")
    print(f"integrity: {checked} pairs against F, baseline scores differing: {differing or 'none'}")


def tally() -> None:
    for variant in VARIANTS:
        calls = []
        for x, y in CONTRASTS:
            result = verdict(variant, x, y)
            calls.append(f"{x}-{y}: {result[0] if result else 'incomplete'}")
        print(f"{variant}: {'; '.join(calls)} => {recommendation(variant)}")
    integrity()


def _promoted(variant: str) -> dict[str, object]:
    """The promoted file as committed, so a later edit of the working tree cannot change the reading."""
    path = f"datasets/hiperparams/{variant.split('_', 1)[0]}/{variant}.imputation.json"
    text = subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, text=True, check=True).stdout
    values = json.loads(text)
    assert isinstance(values, dict)
    return values


def studies() -> None:
    """Each study's winner beside the current file's values, and the knobs' importances."""
    for variant in VARIANTS:
        rows = STORE.execute(
            f"""
            select r.run_uuid, r.start_time, r.end_time from runs r
            join tags d on d.run_uuid = r.run_uuid and d.key = 'dataset' and d.value = ?
            join tags p on p.run_uuid = r.run_uuid and p.key = 'run_role' and p.value = 'optuna_study'
            join tags q on q.run_uuid = r.run_uuid and q.key = 'search_space' and q.value = 'reduced'
            join tags o on o.run_uuid = r.run_uuid and o.key = 'search_objective' and o.value = ?
            where {SOURCE} and r.start_time >= ?""",
            (variant, OBJECTIVE, SINCE_MS),
        ).fetchall()
        if len(rows) != 1:
            print(f"=== {variant}: {len(rows)} finished studies (expected 1)")
            continue
        run, start, end = rows[0]
        metrics = dict(STORE.execute("select key, value from latest_metrics where run_uuid = ?", (run,)).fetchall())
        params = dict(STORE.execute("select key, value from params where run_uuid = ?", (run,)).fetchall())
        file = _promoted(variant)
        print(
            f"=== {variant}: study {run}, {(end - start) / 3.6e6:.2f} h, best trial "
            f"{metrics['optuna/best_trial_number']:.0f}, objective {metrics['optuna/best_objective_value']:.4f}"
        )
        for knob in SAMPLED:
            if f"best_{knob}" in params:
                importance = metrics.get(f"optuna/importance/{knob}")
                shown = f"{importance:.2f}" if importance is not None else "-"
                print(f"    {knob:20s} winner {float(params[f'best_{knob}']):.4g}  file {float(str(file[knob])):.4g}  importance {shown}")


def jobs() -> None:
    for variant in VARIANTS:
        print("study", variant, "-", "-")
    for arm in ("T", "D"):
        for variant in VARIANTS:
            for seed in SEEDS:
                print("cell", variant, seed, arm)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--tally"]:
        tally()
    elif argv[:1] == ["--studies"]:
        studies()
    elif argv[:1] == ["--jobs"]:
        jobs()
    else:
        for name in argv or VARIANTS:
            report(name)
