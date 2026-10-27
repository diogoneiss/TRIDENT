"""E81: does a decoder that can tell a real gap from its own masks read the gap pattern under MNAR?
(imputation-interpretability/04, T38).

    uv run --python 3.11 python -B scripts/experiments/gap_reliance.py train <variant> <seed> <arm> [--no-mlflow]
    uv run --python 3.11 python -B scripts/experiments/gap_reliance.py --todo <n>
    uv run --python 3.11 python -B scripts/experiments/gap_reliance.py --tally

Under today's default (``DECODE_GAP_TOKEN mask``, ADR 0014) a real gap and a hidden observed cell are
the same input, so no counterfactual can ask whether the default reads the gap pattern. Arm ``null``
(``--decode_gap_token null``) shows real gaps as ``[NULL]`` and its own masks as ``[MASK]``, so it can
learn that ``[NULL]`` at d means d was likely large. Arm ``mask`` is today's default, for accuracy.

``train`` runs one cell, five folds, MLflow tags ``experiment=gap-reliance-2026-10-27`` and ``arm``.
On arm null, after each fold's decode stage, the run's model (K) is read in place, random states
restored, as E78 does: for each numerical target j, the test rows are shown with every gap as
``[NULL]`` and j's as ``[MASK]`` (the column-wise view); then one other numerical column d with
|Spearman(j, d)| >= 0.2 on the training split has its gaps switched to ``[MASK]`` too. d's value is
absent either way; only the "truly missing" label changes. ``reliance_fold_<k>.csv`` holds, per
(j, d) over the rows where both are gaps, the mean of sign(rho) * (fill with d as [NULL] - fill with
d as [MASK]) in scaled units: positive where [NULL] at d is read as "d was large".
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import occlusion_map as om  # noqa: E402
from scripts.experiments import report_common as common  # noqa: E402

EXPERIMENT = "gap-reliance-2026-10-27"
STATE = Path("/scratch2/diogoneiss/trident-2026-10-27-gap-reliance")
SEEDS = (101, 202, 303)
DIAL = {0.0: "biodeg_60nan", 0.25: "biodeg_probit0p25_60nan", 0.5: "biodeg_probit_60nan",
        1.0: "biodeg_probit1p0_60nan", 2.0: "biodeg_probit2p0_60nan"}
PAIRS = [(f"{table}_probit_60nan", f"{table}_60nan") for table in ("vehicle", "kc2", "spambase")]
VARIANTS = [*DIAL.values(), *(v for pair in PAIRS for v in pair)]
ARMS = ("null", "mask")
ROWS = {"spambase": 4601, "biodeg": 1055, "vehicle": 846, "kc2": 522}
MIN_RHO = 0.2
SESOI = 0.01


def _fills(model: Any, frame: pd.DataFrame, target: str, device: torch.device) -> np.ndarray:
    embedder = model.embedder
    index = list(embedder.numerical_columns).index(target)
    with torch.no_grad():
        context = model._encode(embedder.encode(frame, device))
        head = model.numerical_heads[embedder.numerical_keys[index]]
        values = head(context[:, len(embedder.categorical_columns) + index, :]).squeeze(-1)
    return np.asarray(values.double().cpu().numpy())


def _read_in_place(out: Path, model: Any, dataset: Any, fold: Any, device: torch.device,
                   fold_ordinal: int, complete_sibling: pd.DataFrame | None) -> None:
    if complete_sibling is None:
        raise RuntimeError("E81 needs the complete sibling")
    states = (np.random.get_state(), torch.get_rng_state(),
              torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    was_training = model.training
    model.eval()
    try:
        features = dataset.frame.drop(columns=[dataset.label_column])
        frame = features.iloc[fold.test_indices].reset_index(drop=True)
        view = om._View(model, dataset, frame, complete_sibling, fold.test_indices, device)
        view.check(view.outputs([]))
        numerical = list(dataset.numerical_columns)
        rho = features.iloc[fold.train_indices][numerical].corr(method="spearman")
        nulls = frame.isna()
        shown = frame.mask(nulls, "[NULL]")
        records = []
        for target in numerical:
            if not bool(nulls[target].any()):
                continue
            asked = shown.copy()
            asked[target] = frame[target].mask(nulls[target], "[MASK]")
            base = _fills(model, asked, target, device)
            for source in numerical:
                r = float(rho.at[target, source])
                if source == target or not np.isfinite(r) or abs(r) < MIN_RHO:
                    continue
                both = (nulls[target] & nulls[source]).to_numpy()
                if both.sum() < om.MIN_CELLS:
                    continue
                swapped = asked.copy()
                swapped[source] = frame[source].mask(nulls[source], "[MASK]")
                shift = np.sign(r) * (base - _fills(model, swapped, target, device))
                records.append({"target": target, "source": source, "rho": r, "n": int(both.sum()),
                                "shift": float(shift[both].mean())})
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records, columns=["target", "source", "rho", "n", "shift"]).to_csv(out / f"reliance_fold_{fold_ordinal}.csv", index=False)
    finally:
        np.random.set_state(states[0])
        torch.set_rng_state(states[1])
        if states[2] is not None:
            torch.cuda.set_rng_state_all(states[2])
        model.train(was_training)


def train(variant: str, seed: int, arm: str, no_mlflow: bool) -> None:
    import src.training.runner as runner
    from src.training.config import complete_configuration, load_hyperparameters
    from train import main as train_main

    if arm not in ARMS:
        sys.exit(f"arm must be one of {ARMS}")
    out = STATE / "reliance" / f"{variant}_s{seed}"
    if arm == "null":
        original = getattr(runner, "train_and_evaluate_decoder")

        def read_after(**kwargs: Any) -> Any:
            outcome = original(**kwargs)
            _read_in_place(out, outcome.model, kwargs["dataset"], kwargs["fold"], kwargs["device"],
                           kwargs["fold_ordinal"], kwargs["complete_sibling"])
            return outcome

        setattr(runner, "train_and_evaluate_decoder", read_after)
    resolved, source = load_hyperparameters(Namespace(dataset_name=variant, task="imputation", decode_gap_token=arm))
    print("cell", variant, seed, arm, "source", source, "resolved", json.dumps(complete_configuration(resolved, "imputation")), flush=True)
    args = Namespace(
        dataset_name=variant, label_column="class",
        output_dir=str(STATE / "results" / f"{variant}_s{seed}_{arm}"), metrics_dir=str(STATE / "metrics" / f"{variant}_s{seed}_{arm}"),
        disable_mlflow=no_mlflow, plot_losses=False, save_model=False, seed=seed, cv_folds=5, task="imputation",
        decode_gap_token=arm,
        mlflow_tags={} if no_mlflow else {"experiment": EXPERIMENT, "arm": arm},
    )
    metrics = train_main(args, return_metrics=True) or {}
    print("CELL_DONE", json.dumps({k: v for k, v in metrics.items() if k.startswith("impute/induced/") and k.endswith("/impute_score") and "/baseline/" not in k}), flush=True)


def todo(which: int) -> None:
    done = {arm: common.parents(EXPERIMENT, arm) for arm in ARMS}
    work = sorted(((ROWS[v.split("_")[0]], v, s, a) for v in VARIANTS for s in SEEDS for a in ARMS), key=lambda w: (-w[0], w[1], w[2], w[3]))
    loads = [0.0] * om.QUEUES
    for cost, variant, seed, arm in work:
        target = loads.index(min(loads))
        loads[target] += cost
        if target == which - 1 and (variant, seed) not in done[arm]:
            print(variant, seed, arm)


def _reliance(variant: str, seed: int) -> list[float]:
    folder = STATE / "reliance" / f"{variant}_s{seed}"
    return [float(pd.read_csv(folder / f"reliance_fold_{f}.csv")["shift"].mean()) for f in range(1, 6)]


def _above(v: common.Verdict) -> bool:
    return v.low > SESOI and all(s > 0 for s in v.seeds)


def tally() -> None:
    runs = {arm: common.parents(EXPERIMENT, arm) for arm in ARMS}
    reference = common.parents("occlusion-map-60nan-2026-10-27")
    reliance: dict[str, common.Verdict] = {}
    reproduced: list[float] = []
    for variant in VARIANTS:
        found = {arm: [runs[arm].get((variant, seed)) for seed in SEEDS] for arm in ARMS}
        if any(not f for arm in ARMS for f in found[arm]):
            print(f"- `{variant}`: incomplete")
            continue
        v = common.verdict([np.array(_reliance(variant, seed)) for seed in SEEDS])
        reliance[variant] = v
        tables = {arm: [common.fold_table(f[0]) for f in found[arm] if f] for arm in ARMS}
        parts = []
        for readout in ("K", "KC"):
            null = [common.model(t, f[0], readout) for t, f in zip(tables["null"], found["null"]) if f]
            mask = [common.model(t, f[0], readout) for t, f in zip(tables["mask"], found["mask"]) if f]
            d = common.verdict([a - b for a, b in zip(null, mask)], sesoi=0.01 * float(np.mean([m.mean() for m in mask])))
            parts.append(f"{readout} null − mask {d.mean:+.4f} [{d.low:+.4f}, {d.high:+.4f}] {d.call}")
        print(f"- `{variant}`: S {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] seeds {', '.join(f'{s:+.4f}' for s in v.seeds)} "
              f"-> {'above the SESOI' if _above(v) else 'not above'}; accuracy (descriptive): " + "; ".join(parts))
        for seed, f in zip(SEEDS, found["mask"]):
            old = reference.get((variant, seed))
            if f and old:
                reproduced.append(float(np.max(np.abs(common.model(common.fold_table(old[0]), old[0], "K") - common.model(common.fold_table(f[0]), f[0], "K")))))
    print()
    if reproduced:
        print(f"- mask arm's K against E79's run of the same cell, largest fold difference: {max(reproduced):.2e} over {len(reproduced)} cells")
    dial = [reliance[DIAL[s]].mean for s in DIAL if DIAL[s] in reliance]
    if len(dial) == len(DIAL):
        print(f"- dial (descriptive): S at slopes 0/0.25/0.5/1/2: {', '.join(f'{x:+.4f}' for x in dial)}; strictly rising: {all(a < b for a, b in zip(dial, dial[1:]))}")
    paired = [(m, c) for m, c in PAIRS if m in reliance and c in reliance]
    if paired:
        print("- pairs (descriptive): " + "; ".join(f"{m} S above: {_above(reliance[m])}, {c} S above: {_above(reliance[c])}" for m, c in paired))
    if DIAL[2.0] not in reliance or DIAL[0.0] not in reliance:
        print("\nPrimary: incomplete")
        return
    steep, flat = reliance[DIAL[2.0]], reliance[DIAL[0.0]]
    if _above(steep) and not flat.low > SESOI:
        reading = "supported"
    elif not steep.low > 0:
        reading = "refuted"
    else:
        reading = "mixed"
    print(f"\nPrimary: S at slope 2.0 {steep.mean:+.4f} [{steep.low:+.4f}, {steep.high:+.4f}], at MCAR {flat.mean:+.4f} [{flat.low:+.4f}, {flat.high:+.4f}] -> {reading} (SESOI {SESOI})")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"]:
        train(argv[1], int(argv[2]), argv[3], "--no-mlflow" in argv)
    elif argv[:1] == ["--todo"]:
        todo(int(argv[1]))
    elif argv[:1] == ["--tally"]:
        tally()
    else:
        sys.exit(__doc__)
