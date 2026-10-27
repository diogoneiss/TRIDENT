"""E78: which columns does the decoder fill each column from? An occlusion map, tested for truth on a
synthetic table and for faithfulness on nine real ones (imputation-interpretability/01, T38).

    uv run --python 3.11 python -B scripts/experiments/occlusion_map.py train <variant> <seed> [--no-mlflow]
    uv run --python 3.11 python -B scripts/experiments/occlusion_map.py --todo <n>
    uv run --python 3.11 python -B scripts/experiments/occlusion_map.py --tally
    uv run --python 3.11 python -B scripts/experiments/occlusion_map.py --study e79 <any of the above>

``--study e79`` runs and reads E79 (imputation-interpretability/02): the same design at 60nan.
``--study e85`` runs and reads E85 (imputation-interpretability/05): the same design at 20nan and 80nan,
with the rule applied to each rate on its own.

``train`` runs one cell of today's default imputation configuration, five folds, with
``--score_remask`` (its spread kept for a later study, not read here), MLflow tag
``experiment=occlusion-map-2026-10-26``. After each fold's decode stage, before the next fold
starts, the run's own model (K) is read in place: the global random streams are saved and put back,
so training is the run a default run of the checkout trains. ``--tally`` applies the rule fixed in
the ticket. Run from the checkout root.

What one fold writes, under ``STATE/attribution/<variant>_s<seed>/``:

- ``occlusion_fold_<k>.csv``: for each split (validation, test), target column j and source column c,
  the mean change in j's fill at its gaps when c's observed cells are hidden as ``[MASK]``, over
  the gap rows of j where c is observed (``n``); a number's absolute change in scaled units, a
  category's total-variation distance between the two probability vectors. On test, ``attention``:
  the mean attention of j's token to c's, over layers, heads and the same cells, at the unhidden pass.
- ``deletion_fold_<k>.csv``: for each target j, its gap cells' mean error on test with nothing more
  hidden, with the top 3 (and top 1) sources by the validation split's map hidden, and with 3 (and 1)
  sources drawn at random hidden, averaged over 20 draws. A number's error is the absolute error in
  scaled units, a category's 1 - p(true category).
- ``correlation_fold_<k>.csv``: the training split's |Spearman| between numerical columns.
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

from scripts.experiments import report_common as common  # noqa: E402

EXPERIMENT = "occlusion-map-2026-10-26"
STATE = Path("/scratch2/diogoneiss/trident-2026-10-26-occlusion")
CONFIRMATION = (707, 808, 909)
DEVELOPMENT = (101, 202, 303)
SEEDS = {
    "synthblock_40nan": CONFIRMATION,
    **{f"{table}_40nan": CONFIRMATION for table in ("pendigits", "letter", "electricity", "vehicle", "biodeg", "kc2")},
    **{f"{table}_40nan": DEVELOPMENT for table in ("kr-vs-kp", "spambase", "credit-g")},
}
REAL = [variant for variant in SEEDS if not variant.startswith("synthblock")]
ROWS = {"electricity": 45312, "letter": 20000, "pendigits": 10992, "spambase": 4601, "kr-vs-kp": 3196,
        "synthblock": 3000, "credit-g": 1000, "biodeg": 1055, "vehicle": 846, "kc2": 522}
QUEUES = 4
TOP = 3
RANDOM_DRAWS = 20
MIN_CELLS = 10
SESOI = 0.01
AUROC_BAR = 0.8
BLOCKS = {"a": ("a1", "a2", "a3", "a4"), "b": ("b1", "b2", "b3", "b4"), "c": ("c1", "c2", "c3", "c4")}


def _use_grid() -> None:
    """E85's cells: the ten tables at 20nan and 80nan, each at the seed set its earlier runs used."""
    global EXPERIMENT, STATE, SEEDS, REAL
    EXPERIMENT, STATE = "occlusion-map-grid-2026-10-27", Path("/scratch2/diogoneiss/trident-2026-10-27-occlusion-grid")
    confirmation = ("pendigits", "letter", "electricity", "vehicle", "biodeg", "kc2")
    SEEDS = {
        "synthblock_20nan": CONFIRMATION, "synthblock_80nan": CONFIRMATION,
        **{f"{table}_20nan": DEVELOPMENT for table in (*confirmation, "kr-vs-kp", "spambase", "credit-g")},
        **{f"{table}_80nan": CONFIRMATION for table in confirmation},
        **{f"{table}_80nan": DEVELOPMENT for table in ("kr-vs-kp", "spambase", "credit-g")},
    }
    REAL = [variant for variant in SEEDS if not variant.startswith("synthblock")]


def _use_sixty() -> None:
    """E79's cells: the ten tables at 60nan, each at the seed set whose earlier runs K is checked against."""
    global EXPERIMENT, STATE, SEEDS, REAL
    EXPERIMENT, STATE = "occlusion-map-60nan-2026-10-27", Path("/scratch2/diogoneiss/trident-2026-10-27-occlusion-60")
    SEEDS = {
        "synthblock_60nan": CONFIRMATION,
        **{f"{table}_60nan": CONFIRMATION for table in ("pendigits", "letter", "electricity")},
        **{f"{table}_60nan": DEVELOPMENT for table in ("vehicle", "biodeg", "kc2", "kr-vs-kp", "spambase", "credit-g")},
    }
    REAL = [variant for variant in SEEDS if not variant.startswith("synthblock")]


# --- reading the model in place -------------------------------------------------------------------

class _View:
    """One split's gap cells and the model's outputs on them, with chosen source columns hidden."""

    def __init__(self, model: Any, dataset: Any, frame: pd.DataFrame, complete_sibling: pd.DataFrame,
                 rows: Any, device: torch.device) -> None:
        # shortcut: reuses decoding's private scorer for the truth alignment (dtypes, scaling, unseen
        # categories); a rename there breaks this script, not a run.
        from src.embedder import EpochMasker
        from src.training.decoding import _score_induced_missing

        self.model, self.frame, self.device = model, frame, device
        embedder = model.embedder
        self.embedder = embedder
        self.nulls = frame.isna().to_numpy()
        self.columns = list(frame.columns)
        self.masker = EpochMasker(embedder, frame, device)
        ledger = _score_induced_missing(model, dataset, frame, complete_sibling, rows, device)
        self.categorical = list(embedder.categorical_columns)
        self.numerical = list(embedder.numerical_columns)
        self.token = {column: 1 + index for index, column in enumerate(self.categorical + self.numerical)}
        self.targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.ledger_fill: dict[str, np.ndarray] = {}
        for column, cells in ledger.groupby("column"):
            name = str(column)
            if name in self.categorical:
                encoder = embedder.label_encoders[name]
                valid = getattr(model, f"valid_ids_{embedder.categorical_keys[self.categorical.index(name)]}").cpu().numpy()
                position = {int(i): p for p, i in enumerate(valid)}
                known = cells["actual"].astype(str).isin(set(encoder.classes_))
                cells = cells[known]
                ids = encoder.transform(cells["actual"].astype(str).to_numpy())
                truth = np.array([position.get(int(i), -1) for i in ids])
                keep = truth >= 0
                cells, truth = cells[keep], truth[keep]
                self.ledger_fill[name] = cells["imputed"].astype(str).to_numpy()
            else:
                truth = cells["actual"].astype(float).to_numpy()
                self.ledger_fill[name] = cells["imputed"].astype(float).to_numpy()
            if len(cells):
                self.targets[name] = (cells["row"].to_numpy(dtype=np.int64), truth)

    def outputs(self, hidden: list[str]) -> dict[str, np.ndarray]:
        """Each target's fill at its gap rows: a number, or a category's probability vector."""
        hide = np.zeros_like(self.nulls)
        for column in hidden:
            index = self.columns.index(column)
            hide[:, index] = ~self.nulls[:, index]
        encoded = self.embedder.gaps_as_mask(self.masker.apply(hide))
        with torch.no_grad():
            return self._fills(self.model._encode(encoded))

    def _fills(self, context: torch.Tensor) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name, (rows, _) in self.targets.items():
            if name in self.categorical:
                index = self.categorical.index(name)
                key = self.embedder.categorical_keys[index]
                logits = self.model.categorical_heads[key](context[:, index, :])
                out[name] = logits.softmax(dim=-1).double().cpu().numpy()[rows]
            else:
                index = self.numerical.index(name)
                key = self.embedder.numerical_keys[index]
                values = self.model.numerical_heads[key](context[:, len(self.categorical) + index, :]).squeeze(-1)
                out[name] = values.double().cpu().numpy()[rows]
        return out

    def check(self, base: dict[str, np.ndarray]) -> None:
        """The unhidden pass is the run's own fill, cell for cell."""
        for name, fill in base.items():
            if name in self.categorical:
                encoder = self.embedder.label_encoders[name]
                key = self.embedder.categorical_keys[self.categorical.index(name)]
                valid = getattr(self.model, f"valid_ids_{key}").cpu().numpy()
                chosen = encoder.inverse_transform(valid[fill.argmax(axis=1)]).astype(str)
                assert (chosen == self.ledger_fill[name]).all(), name
            else:
                assert np.allclose(fill, self.ledger_fill[name], atol=1e-4), name

    def error(self, name: str, fill: np.ndarray) -> np.ndarray:
        truth = self.targets[name][1]
        if name in self.categorical:
            return np.asarray(1.0 - fill[np.arange(len(truth)), truth.astype(np.int64)])
        return np.asarray(np.abs(fill - truth))

    def observed(self, name: str, source: str) -> np.ndarray:
        return np.asarray(~self.nulls[self.targets[name][0], self.columns.index(source)])


def _change(base: np.ndarray, other: np.ndarray) -> np.ndarray:
    if base.ndim == 2:
        return np.asarray(0.5 * np.abs(other - base).sum(axis=1))
    return np.asarray(np.abs(other - base))


def _attention(view: _View) -> np.ndarray:
    """(rows, tokens, tokens) attention at the unhidden pass, averaged over layers and heads."""
    from src.transformer import MultiHeadAttention

    captured: list[torch.Tensor] = []
    hooks = [module.att_dropout.register_forward_hook(lambda _m, inputs, _o: captured.append(inputs[0].detach().mean(dim=1)))
             for module in view.model.modules() if isinstance(module, MultiHeadAttention)]
    try:
        view.outputs([])
    finally:
        for hook in hooks:
            hook.remove()
    return np.asarray(torch.stack(captured).mean(dim=0).double().cpu().numpy())


def _occlusion(view: _View, split: str, attention: np.ndarray | None) -> pd.DataFrame:
    base = view.outputs([])
    view.check(base)
    records = []
    for source in view.columns:
        hidden = view.outputs([source])
        for name, fill in base.items():
            if name == source:
                continue
            seen = view.observed(name, source)
            n = int(seen.sum())
            effect = float(_change(fill, hidden[name])[seen].mean()) if n >= MIN_CELLS else float("nan")
            record = {"split": split, "target": name, "source": source, "effect": effect, "n": n}
            if attention is not None:
                rows = view.targets[name][0][seen]
                record["attention"] = float(attention[rows, view.token[name], view.token[source]].mean()) if n >= MIN_CELLS else float("nan")
            records.append(record)
    return pd.DataFrame(records)


def _deletion(view: _View, ranking: pd.DataFrame, generator: np.random.Generator) -> pd.DataFrame:
    base = view.outputs([])
    records = []
    for name in view.targets:
        ranked = ranking[(ranking["target"] == name) & ranking["effect"].notna()].sort_values("effect", ascending=False)
        if len(ranked) < TOP:
            continue
        others = [c for c in view.columns if c != name]
        record: dict[str, Any] = {"target": name, "kind": "categorical" if name in view.categorical else "numerical",
                                  "n": len(view.targets[name][1]), "e_base": float(view.error(name, base[name]).mean()),
                                  "top": "|".join(ranked["source"].head(TOP))}
        for k in (TOP, 1):
            top = list(ranked["source"].head(k))
            record[f"e_top{k}"] = float(view.error(name, view.outputs(top)[name]).mean())
            drawn = [list(generator.choice(others, size=k, replace=False)) for _ in range(RANDOM_DRAWS)]
            record[f"e_random{k}"] = float(np.mean([view.error(name, view.outputs(d)[name]).mean() for d in drawn]))
        records.append(record)
    return pd.DataFrame(records)


def _read_in_place(out: Path, model: Any, dataset: Any, fold: Any, device: torch.device, seed: int,
                   fold_ordinal: int, complete_sibling: pd.DataFrame | None) -> None:
    if complete_sibling is None:
        raise RuntimeError("E78 needs the complete sibling: the gaps' truths")
    states = (np.random.get_state(), torch.get_rng_state(),
              torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    was_training = model.training
    model.eval()
    try:
        features = dataset.frame.drop(columns=[dataset.label_column])
        views = {split: _View(model, dataset, features.iloc[rows].reset_index(drop=True), complete_sibling, rows, device)
                 for split, rows in (("validation", fold.validation_indices), ("test", fold.test_indices))}
        validation = _occlusion(views["validation"], "validation", None)
        test = _occlusion(views["test"], "test", _attention(views["test"]))
        out.mkdir(parents=True, exist_ok=True)
        pd.concat([validation, test], ignore_index=True).to_csv(out / f"occlusion_fold_{fold_ordinal}.csv", index=False)
        deletion = _deletion(views["test"], validation, np.random.default_rng([seed, fold_ordinal, 78]))
        deletion.to_csv(out / f"deletion_fold_{fold_ordinal}.csv", index=False)
        train = features.iloc[fold.train_indices][list(dataset.numerical_columns)]
        train.corr(method="spearman").abs().to_csv(out / f"correlation_fold_{fold_ordinal}.csv")
    finally:
        np.random.set_state(states[0])
        torch.set_rng_state(states[1])
        if states[2] is not None:
            torch.cuda.set_rng_state_all(states[2])
        model.train(was_training)


def train(variant: str, seed: int, no_mlflow: bool) -> None:
    import src.training.runner as runner
    from src.training.config import complete_configuration, load_hyperparameters
    from train import main as train_main

    out = STATE / "attribution" / f"{variant}_s{seed}"
    original = getattr(runner, "train_and_evaluate_decoder")

    def read_after(**kwargs: Any) -> Any:
        outcome = original(**kwargs)
        _read_in_place(out, outcome.model, kwargs["dataset"], kwargs["fold"], kwargs["device"], kwargs["seed"],
                       kwargs["fold_ordinal"], kwargs["complete_sibling"])
        return outcome

    setattr(runner, "train_and_evaluate_decoder", read_after)
    resolved, source = load_hyperparameters(Namespace(dataset_name=variant, task="imputation"))
    print("cell", variant, seed, "source", source, "resolved", json.dumps(complete_configuration(resolved, "imputation")), flush=True)
    args = Namespace(
        dataset_name=variant, label_column="class",
        output_dir=str(STATE / "results" / f"{variant}_s{seed}"), metrics_dir=str(STATE / "metrics" / f"{variant}_s{seed}"),
        disable_mlflow=no_mlflow, plot_losses=False, save_model=False, seed=seed, cv_folds=5, task="imputation",
        score_remask=True,
        mlflow_tags={} if no_mlflow else {"experiment": EXPERIMENT},
    )
    metrics = train_main(args, return_metrics=True) or {}
    print("CELL_DONE", json.dumps({k: v for k, v in metrics.items() if k.startswith("impute/induced/") and k.endswith("/impute_score") and "/baseline/" not in k}), flush=True)


def todo(which: int) -> None:
    done = common.parents(EXPERIMENT)
    work = sorted(((ROWS[v.split("_")[0]], v, s) for v, seeds in SEEDS.items() for s in seeds), key=lambda w: (-w[0], w[1], w[2]))
    loads = [0.0] * QUEUES
    for cost, variant, seed in work:
        target = loads.index(min(loads))
        loads[target] += cost
        if target == which - 1 and (variant, seed) not in done:
            print(variant, seed)


# --- the reading fixed in the ticket ---------------------------------------------------------------

def _files(variant: str, seed: int, kind: str) -> list[pd.DataFrame]:
    folder = STATE / "attribution" / f"{variant}_s{seed}"
    return [pd.read_csv(folder / f"{kind}_fold_{fold}.csv") for fold in range(1, 6)]


def _faithfulness(deletion: pd.DataFrame, k: int) -> float:
    """One fold's D: the mean over targets of (top-k error - random-k error) / base error."""
    usable = deletion[(deletion["n"] >= MIN_CELLS) & (deletion["e_base"] > 0)]
    return float(((usable[f"e_top{k}"] - usable[f"e_random{k}"]) / usable["e_base"]).mean())


def _auroc(scores: pd.Series, positive: pd.Series) -> float:
    """AUROC of ``scores`` separating ``positive`` sources from the rest (ties count half)."""
    keep = scores.notna()
    s, p = scores[keep].to_numpy(dtype=float), positive[keep].to_numpy(dtype=bool)
    if p.all() or not p.any():
        return float("nan")
    wins = (s[p][:, None] > s[~p][None, :]).mean() + 0.5 * (s[p][:, None] == s[~p][None, :]).mean()
    return float(wins)


def _block_of(column: str) -> str | None:
    return next((block for block, members in BLOCKS.items() if column in members), None)


def _truth(test: pd.DataFrame, score: str) -> float:
    """One fold's C: the mean over the 12 block columns of the AUROC of ``score`` for in-block sources."""
    values = []
    for target in (c for members in BLOCKS.values() for c in members):
        rows = test[test["target"] == target]
        if len(rows):
            values.append(_auroc(rows[score], rows["source"].map(_block_of) == _block_of(target)))
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def _spearman(a: pd.Series, b: pd.Series) -> float:
    keep = a.notna() & b.notna()
    return float(a[keep].corr(b[keep], method="spearman")) if keep.sum() > 2 else float("nan")


def _with_correlation(test: pd.DataFrame, correlation: pd.DataFrame) -> pd.DataFrame:
    pairs: dict[Any, Any] = correlation.set_index(correlation.columns[0]).stack().to_dict()
    return test.assign(correlation=[float(pairs.get((s, t), float("nan"))) for s, t in zip(test["source"], test["target"])])


def _within_target(test: pd.DataFrame, other: str) -> float:
    """Mean over targets of the Spearman between the occlusion effect and ``other`` across sources."""
    values = [_spearman(rows["effect"], rows[other]) for _, rows in test.groupby("target")]
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def _call(v: common.Verdict, bar: float) -> str:
    if v.low > bar and all(s > (0 if bar == SESOI else bar) for s in v.seeds):
        return "yes"
    return "no" if v.high < bar else "inconclusive"


def tally() -> None:
    runs = common.parents(EXPERIMENT)
    faithful: dict[str, str] = {}
    reproduced: list[float] = []
    truth_calls: dict[str, str] = {}
    for variant, seeds in SEEDS.items():
        found = [runs.get((variant, seed)) for seed in seeds]
        if any(not f for f in found):
            print(f"- `{variant}`: incomplete ({sum(bool(f) for f in found)} of {len(seeds)} seeds)")
            continue
        occlusion = {seed: _files(variant, seed, "occlusion") for seed in seeds}
        deletion = {seed: _files(variant, seed, "deletion") for seed in seeds}
        correlation = {seed: _files(variant, seed, "correlation") for seed in seeds}
        tests = {seed: [o[o["split"] == "test"] for o in occlusion[seed]] for seed in seeds}
        lines = []
        for k in (TOP, 1):
            v = common.verdict([np.array([_faithfulness(d, k) for d in deletion[seed]]) for seed in seeds])
            mark = _call(v, SESOI)
            if k == TOP and variant in REAL:
                faithful[variant] = mark
            lines.append(f"D top{k} {v.mean:+.3f} [{v.low:+.3f}, {v.high:+.3f}] seeds {', '.join(f'{s:+.3f}' for s in v.seeds)} -> faithful: {mark}")
        print(f"- `{variant}`: " + "; ".join(lines))
        if variant.startswith("synthblock"):
            v = common.verdict([np.array([_truth(t, "effect") for t in tests[seed]]) for seed in seeds])
            frames = [common.fold_table(f[0]) for f in found if f]
            k_score = float(np.mean([common.model(t, f[0], "K").mean() for t, f in zip(frames, found) if f]))
            naive = float(np.mean([t["impute/induced/baseline/mean_mode/impute_score"].mean() for t in frames]))
            truth_call = "inconclusive (floor: K not below mean_mode)" if k_score >= naive else _call(v, AUROC_BAR)
            truth_calls[variant] = truth_call
            linear = common.verdict([np.array([_truth(_with_correlation(t, c), "correlation") for t, c in zip(tests[seed], correlation[seed])]) for seed in seeds])
            print(f"  - truth (C): AUROC {v.mean:.3f} [{v.low:.3f}, {v.high:.3f}] seeds {', '.join(f'{s:.3f}' for s in v.seeds)}; "
                  f"K {k_score:.4f} against mean_mode {naive:.4f} -> recovers the blocks: {truth_call}")
            print(f"  - descriptive: |Spearman| on the training split, same AUROC (numerical pairs only): {linear.mean:.3f} [{linear.low:.3f}, {linear.high:.3f}]")
        pooled = pd.concat([t for seed in seeds for t in tests[seed]]).groupby(["target", "source"], as_index=False)[["effect", "attention"]].mean()
        by_seed = [pd.concat(tests[seed]).groupby(["target", "source"])["effect"].mean() for seed in seeds]
        stability = np.nanmean([_spearman(by_seed[i], by_seed[j]) for i in range(3) for j in range(i + 1, 3)])
        linear_agreement = np.nanmean([_within_target(_with_correlation(t, c), "correlation") for seed in seeds for t, c in zip(tests[seed], correlation[seed])])
        print(f"  - map: seed stability {stability:+.3f}; attention against occlusion {_within_target(pooled, 'attention'):+.3f}; "
              f"|Spearman| against occlusion {linear_agreement:+.3f} (within target, mean)")
        tops = []
        for target, rows in pooled.groupby("target"):
            best = rows.sort_values("effect", ascending=False).head(3)
            tops.append(f"{target} <- " + ", ".join(f"{s} {e:.3f}" for s, e in zip(best["source"], best["effect"])))
        print("  - top sources: " + "; ".join(tops[:12]) + (" ..." if len(tops) > 12 else ""))
        for seed, f in zip(seeds, found):
            source = common.development_source(variant, "101/202/303") if seeds == DEVELOPMENT else None
            reference = (common.parents(source.experiment, source.arm) if source else common.parents("confirmation-2026-10-17")).get((variant, seed))
            if f and reference:
                old = common.model(common.fold_table(reference[0]), reference[0], "K")
                new = common.model(common.fold_table(f[0]), f[0], "K")
                reproduced.append(float(np.max(np.abs(old - new))))
    print()
    if reproduced:
        print(f"- K against the earlier run of the same cell, largest fold difference: {max(reproduced):.2e} over {len(reproduced)} cells")
    rates = sorted({v.split("_")[-1] for v in REAL})
    for rate in rates:
        synthetic = [v for v in truth_calls if v.endswith(rate)]
        call = truth_calls[synthetic[0]] if synthetic else "incomplete"
        print(f"- truth (C, gate){'' if len(rates) == 1 else f' at {rate}'}: {call}")
    for rate in rates:
        real = [v for v in REAL if v.endswith(rate)]
        marks = {v: faithful[v] for v in real if v in faithful}
        label = "" if len(rates) == 1 else f" at {rate}"
        if len(marks) < len(real):
            print(f"\nPrimary{label}: incomplete ({len(marks)} of {len(real)})")
            continue
        yes = [v for v, c in marks.items() if c == "yes"]
        no = [v for v, c in marks.items() if c == "no"]
        if len(yes) >= 7 and not no:
            reading = "supported"
        elif len(yes) <= 4:
            reading = "refuted"
        else:
            reading = "mixed"
        print(f"\nPrimary (B){label}: faithful on {len(yes)} of {len(real)}, not faithful on {no or 'none'} -> {reading} ({common.RULE}, SESOI {SESOI})")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:2] == ["--study", "e79"]:
        _use_sixty()
        argv = argv[2:]
    elif argv[:2] == ["--study", "e85"]:
        _use_grid()
        argv = argv[2:]
    if argv[:1] == ["train"]:
        train(argv[1], int(argv[2]), "--no-mlflow" in argv)
    elif argv[:1] == ["--todo"]:
        todo(int(argv[1]))
    elif argv[:1] == ["--tally"]:
        tally()
    else:
        sys.exit(__doc__)
