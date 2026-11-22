"""E186 (ticket imputation-protocol/75): a three-way downstream classification on the 26 MCAR variants of E184 and
E185: the table with its gaps left missing, with the gaps filled by the decoder's checkpoint chosen by
``impute_score`` (arm K), and by the checkpoint chosen by ``sw_score`` (arm S).

    uv run --python 3.11 python -B scripts/experiments/sliced_utility.py run <variant> <seed>   # one CSV under OUT
    uv run --python 3.11 python -B scripts/experiments/sliced_utility.py --jobs                 # "variant seed" lines, small tables first
    uv run --python 3.11 python -B scripts/experiments/sliced_utility.py --tally

It trains nothing of TRIDENT's and uses no GPU: the fills are the fold ledgers of E184's and E185's runs (arms K, S
and, beside, the loss-term arms L and LS). Each row's gaps are filled by the fold in which the row is a test row.
Per fold a fixed classifier is fitted on a table's rows outside the test fold and scored on that table's test fold
(``own``, the primary), or fitted on the truth-filled rows and scored on each table's test fold (``complete``).
Learners: a random forest, which takes the gaps as they are, and logistic regression, for which a gap is the mean
with an indicator column (numbers) or a category of its own. Run from the checkout root.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer  # type: ignore[attr-defined]
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder  # type: ignore[attr-defined]

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import c15_report  # noqa: E402
from scripts.experiments import sliced_loss_run as e184  # noqa: E402  (never sliced_loss_large_run: it repoints e184)
from scripts.experiments import sliced_run as e182  # noqa: E402
from scripts.experiments import sliced_select_run as e183  # noqa: E402
from src.training.data import build_folds, prepare_dataset  # noqa: E402
from src.training.imputation_metrics import mean_mode_baselines  # noqa: E402
from src.training.types import DatasetSpec  # noqa: E402

OUT = Path(os.environ.get("SLICED_UTILITY_OUT", "/scratch/diogoneiss/trident-2026-11-22-sliced-utility")) / "csv"
THREADS = int(os.environ.get("UTILITY_THREADS", "4"))
E185 = ("pendigits_20nan", "pendigits_40nan", "pendigits_60nan", "pendigits_80nan",
        "electricity_20nan", "electricity_40nan", "electricity_80nan", "letter_20nan")
STATES: dict[Path, tuple[str, ...]] = {e184.STATE: e184.VARIANTS, Path("/scratch/diogoneiss/trident-2026-11-20-sliced-loss-large"): E185}
VARIANTS = tuple(v for variants in STATES.values() for v in variants)
assert len(VARIANTS) == len(set(VARIANTS)) == 26 and str(e184.STATE).endswith("trident-2026-11-19-sliced-loss")
STUDY = "E186"  # sliced_fresh_read.py (E187) runs these functions with its own study, states, seeds and arms
SEEDS: tuple[int, ...] = (101, 202, 303)
ARMS: tuple[str, ...] = ("K", "S", "L", "LS")
FILLS: tuple[str, ...] = ("missing", "naive", *ARMS, "truth")
LEARNERS = ("forest", "logistic")
PRIMARY = (("S", "K"), ("K", "missing"), ("S", "missing"))  # the first is the primary line, the other two the three-way
TERM: tuple[tuple[str, str], ...] = (("LS", "S"), ("L", "K"))  # the loss-term arms against their controls
BESIDE: tuple[tuple[str, str], ...] = (*TERM, ("naive", "missing"), ("truth", "missing"))
KEY = ["variant", "seed", "fold"]


def _learner(name: str, seed: int, categorical: list[str], numerical: list[str]) -> Any:
    if name == "forest":  # a gap stays NaN: sklearn's trees split on it
        encode = ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=np.nan), categorical),
            ("num", "passthrough", numerical)])
        return make_pipeline(encode, RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=THREADS))
    own_level = make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"), OneHotEncoder(handle_unknown="ignore"))
    encode = ColumnTransformer([("cat", own_level, categorical), ("num", SimpleImputer(strategy="mean", add_indicator=True), numerical)])
    return make_pipeline(encode, LogisticRegression(max_iter=5000))


def _tables(variant: str, seed: int) -> tuple[dict[str, pd.DataFrame], pd.Series, list[Any], list[str], list[str]]:
    """The variant's features under each fill, its labels, its folds and its column kinds."""
    state = next(s for s, variants in STATES.items() if variant in variants)
    dataset = prepare_dataset(DatasetSpec.from_name(variant, None))
    labels = dataset.frame[dataset.label_column].reset_index(drop=True)
    assert labels.notna().all(), "a label is missing"
    features = dataset.frame.drop(columns=[dataset.label_column]).reset_index(drop=True)
    numerical, categorical = list(dataset.numerical_columns), list(dataset.categorical_columns)
    folds = list(build_folds(dataset.frame, dataset.label_column, 5, seed))
    gaps = features.isna().to_numpy()
    blocks = {fill: features.to_numpy(dtype=object, copy=True) for fill in FILLS}
    filled = np.zeros_like(gaps)
    for ordinal, fold in enumerate(folds, start=1):
        test = np.asarray(fold.test_indices)
        naive = mean_mode_baselines(features.iloc[np.asarray(fold.train_indices)].reset_index(drop=True), numerical, categorical)
        ledgers: dict[str, pd.DataFrame] = {}
        for arm in ARMS:
            run = e183._run(state, arm, variant, seed)
            assert run is not None, f"no finished {arm} run of {variant} seed {seed} under {state}"
            ledgers[arm] = e182._ledger(run, ordinal)
        first = ledgers[ARMS[0]]
        for arm in ARMS[1:]:
            assert first[["row", "column"]].equals(ledgers[arm][["row", "column"]]), f"{arm}: other cells, fold {ordinal}"
            assert first["actual"].astype(str).equals(ledgers[arm]["actual"].astype(str)), f"{arm}: other truths, fold {ordinal}"
        rows = test[first["row"].to_numpy(dtype=int)]
        position = features.columns.get_indexer(pd.Index(first["column"]))
        assert gaps[rows, position].all() and not filled[rows, position].any(), f"fold {ordinal}: a ledger cell is not a gap, or twice"
        filled[rows, position] = True
        values = {"truth": first["actual"], "naive": first["column"].map(naive), **{arm: ledgers[arm]["imputed"] for arm in ARMS}}
        for fill, column in values.items():
            blocks[fill][rows, position] = column.to_numpy(dtype=object)
    assert (filled == gaps).all(), "a gap has no ledger cell"
    tables: dict[str, pd.DataFrame] = {}
    for fill, block in blocks.items():
        table = pd.DataFrame(block, columns=features.columns)
        assert fill == "missing" or not table.isna().any().any(), f"{fill}: a gap is left unfilled"
        table[numerical] = table[numerical].astype(float)
        for name in categorical:  # a category as text; a gap (the missing table only) stays NaN
            table[name] = table[name].map(lambda v: v if pd.isna(v) else str(v))
        tables[fill] = table
    assert int(tables["missing"].isna().to_numpy().sum()) == int(gaps.sum()) > 0
    return tables, labels, folds, numerical, categorical


def run(variant: str, seed: int) -> None:
    assert variant in VARIANTS and seed in SEEDS, (variant, seed)
    path = OUT / f"{variant}_s{seed}.csv"
    if path.exists():
        print("SKIP", path.name)
        return
    tables, labels, folds, numerical, categorical = _tables(variant, seed)
    out: list[dict[str, object]] = []
    everything = np.arange(len(labels))
    seen = 0
    for ordinal, fold in enumerate(folds, start=1):
        test = np.asarray(fold.test_indices)
        train = np.setdiff1d(everything, test)
        for name in LEARNERS:
            scored: dict[tuple[str, str], float] = {}
            for fill in FILLS:
                model = _learner(name, seed, categorical, numerical).fit(tables[fill].iloc[train], labels.iloc[train])
                scored[("own", fill)] = f1_score(labels.iloc[test], model.predict(tables[fill].iloc[test]), average="macro")
                if fill == "missing" and name == "forest":
                    seen += int(np.isnan(np.asarray(model[:-1].transform(tables[fill].iloc[train]), dtype=float)).sum())
                if fill == "truth":
                    for other in FILLS:
                        scored[("complete", other)] = f1_score(labels.iloc[test], model.predict(tables[other].iloc[test]), average="macro")
            out.extend({"variant": variant, "seed": seed, "fold": ordinal, "learner": name, "protocol": protocol, "fill": fill,
                        "macro_f1": float(value)} for (protocol, fill), value in scored.items())
    assert seen > 0, "no gap reached the forest in the missing table"
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(path, index=False)
    print(f"DONE {path.name}: {len(labels)} rows, {int(tables['missing'].isna().to_numpy().sum())} gaps, "
          f"{seen} gaps reached the forest's training rows over the folds", flush=True)


def _line(name: str, d: np.ndarray, first: str, second: str) -> str:
    m, lo, hi = c15_report.interval(d, False)
    _, nlo, nhi = c15_report.interval(d)
    return f"{name}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] (NB [{nlo:+.4f}, {nhi:+.4f}]) -> {first if lo > 0 else second if hi < 0 else 'no difference'}"


def _level(variant: str) -> int:
    return int(variant.rsplit("_", 1)[1][:2])


def tally() -> int:
    files = sorted(OUT.glob("*.csv"))
    expected = len(VARIANTS) * len(SEEDS)
    if len(files) != expected:
        print(f"incomplete: {len(files)} of {expected} files")
        return 1
    data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    assert set(data["variant"]) == set(VARIANTS) and len(data) == expected * 5 * len(LEARNERS) * 2 * len(FILLS)
    print(f"## Downstream utility of the sliced selector ({STUDY}): macro-F1 on the test folds of {len(VARIANTS)} MCAR variants, "
          f"seeds {'/'.join(map(str, SEEDS))}; a difference is pooled over the variants per (seed, fold) replicate, "
          f"{5 * len(SEEDS)} replicates, plain 95% t interval (NB: corrected)")

    def wide(protocol: str, learner: str) -> pd.DataFrame:
        part = data[(data["protocol"] == protocol) & (data["learner"] == learner)]
        return part.pivot_table(index=KEY, columns="fill", values="macro_f1").reset_index()

    def pooled(frame: pd.DataFrame, first: str, second: str) -> np.ndarray:
        frame = frame.assign(d=frame[first] - frame[second])
        return np.asarray(frame.pivot_table(index=["seed", "fold"], columns="variant", values="d").mean(axis=1), dtype=float)

    def levels(frame: pd.DataFrame) -> str:
        return ", ".join(f"{fill} {frame[fill].mean():.4f}" for fill in FILLS)

    forest = wide("own", "forest")
    (first, second), *three_way = PRIMARY
    print(_line(f"Primary, forest fitted on each table's own rows, F1({first}) − F1({second})", pooled(forest, first, second), first, second))
    for first, second in three_way:
        print(_line(f"Three-way, same, F1({first}) − F1({second})", pooled(forest, first, second), first, second))
    for first, second in BESIDE:
        print(_line(f"  beside, F1({first}) − F1({second})", pooled(forest, first, second), first, second))
    print(f"Levels (forest, own): {levels(forest)}")

    print("\n### By missingness level (forest, own)")
    for level in sorted({_level(v) for v in VARIANTS}):
        part = forest[forest["variant"].map(_level) == level]
        print(f"- {level}nan ({part['variant'].nunique()} variants): {levels(part)}")
        for first, second in PRIMARY:
            print(_line(f"  - F1({first}) − F1({second})", pooled(part, first, second), first, second))

    for title, protocol, learner in (("logistic regression, own (a gap is the mean with an indicator, or its own category)", "own", "logistic"),
                                     ("forest fitted on the complete rows, scored on each table's test fold", "complete", "forest"),
                                     ("logistic regression fitted on the complete rows", "complete", "logistic")):
        frame = wide(protocol, learner)
        print(f"\n### Beside: {title}")
        for first, second in (*PRIMARY, *TERM):
            print(_line(f"- F1({first}) − F1({second})", pooled(frame, first, second), first, second))
        print(f"- Levels: {levels(frame)}")

    print("\n### Per variant (forest, own), 15 fold values a variant")
    counts = {pair: {pair[0]: 0, "no difference": 0, pair[1]: 0} for pair in PRIMARY}
    for variant in VARIANTS:
        part = forest[forest["variant"] == variant]
        print(f"- {variant}: {levels(part)}")
        for first, second in PRIMARY:
            line = _line(f"  - F1({first}) − F1({second})", (part[first] - part[second]).to_numpy(dtype=float), first, second)
            counts[(first, second)][line.rsplit("-> ", 1)[1]] += 1
            print(line)
    for (first, second), count in counts.items():
        print(f"Variants, F1({first}) − F1({second}): " + ", ".join(f"{name} {n}" for name, n in count.items()))
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["run"] and len(argv) == 3:
        run(argv[1], int(argv[2]))
    elif argv == ["--jobs"]:
        order = {table: k for k, table in enumerate(e182.TABLES)}
        for variant in sorted(VARIANTS, key=lambda v: (order[v.rsplit("_", 1)[0]], v)):
            for seed in SEEDS:
                if not (OUT / f"{variant}_s{seed}.csv").exists():
                    print(variant, seed)
    elif argv == ["--tally"]:
        raise SystemExit(tally())
    else:
        sys.exit(__doc__)
