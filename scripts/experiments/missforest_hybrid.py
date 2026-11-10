"""E162: the deployable hybrid with missForest in TabPFN's place, where TRIDENT trails missForest (T39 rule, ADR 0031).

    uv run --python 3.11 python -B scripts/experiments/missforest_hybrid.py cells <e89|e90> <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/missforest_hybrid.py --tally

``cells`` reads E89's (seeds 1212/2323/3434) or E90's (4545/5656/6767) cell tables, which hold H's fills of the test
split's induced cells and of the hidden observed validation cells. It refits ADR 0016's missForest on each training
fold, fills the same cells (the validation cells group by group, as TabPFN's were), and asserts that missForest's score
on the test cells equals the run's own logged missForest score, fold by fold. CPU only; no training, nothing in the
store. ``--tally`` applies the ``table`` rule (ADR 0031 decision 6's deployable rule) with missForest as the other side
and prints, per study and variant: H, missForest, the hybrid; hybrid − H and hybrid − missForest (1% SESOI); the
``column`` rule beside it.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import blend_weights as weights  # noqa: E402
from scripts.experiments import deployable_hybrid as e82  # noqa: E402
from scripts.experiments import hybrid_selection_rules as rules  # noqa: E402
from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments import tabpfn_blend as tb  # noqa: E402

STATE = Path("/scratch2/diogoneiss/trident-2026-11-08-missforest-hybrid")
VARIANTS = ("vehicle_80nan", "biodeg_80nan")
STUDIES = {
    "e89": ("deployable-hybrid-2026-10-27", Path("/scratch2/diogoneiss/trident-2026-10-28-hybrid-selection-rules"), (1212, 2323, 3434)),
    "e90": ("hybrid-confirmation-2026-10-28", Path("/scratch2/diogoneiss/trident-2026-10-28-hybrid-confirmation"), (4545, 5656, 6767)),
}
SESOI = 0.01


def _typed_load(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    return e82._typed(pd.read_csv(path, dtype={c: object for c in columns}), columns)


def add_missforest(experiment: str, source_root: Path, out_root: Path, variant: str, seed: int, keep_tabpfn: bool = True,
                   population: str = "induced") -> None:
    """missForest's fills beside the cell table's own (E162; E163–E165 keep TabPFN's column too); ``population`` as
    ``hybrid_selection_rules.cells``."""
    from src.training.data import build_folds, prepare_dataset
    from src.training.imputation_metrics import NUMERICAL
    from src.training.missforest import fit_missforest_imputers
    from src.training.types import DatasetSpec

    warnings.filterwarnings("ignore")
    started = time.time()
    source, out = source_root / f"{variant}_s{seed}", out_root / f"{variant}_s{seed}"
    found = common.parents(experiment)[(variant, seed)]
    if len(found) != 1:
        raise RuntimeError(f"{variant} s{seed}: {len(found)} runs")
    logged = np.asarray(common.fold_table(found[0])[f"impute/{population}/baseline/missforest/impute_score"], dtype=float)
    dataset = prepare_dataset(DatasetSpec.from_name(variant, None))
    features = dataset.frame.drop(columns=[dataset.label_column])
    numerical, categorical = list(dataset.numerical_columns), list(dataset.categorical_columns)
    out.mkdir(parents=True, exist_ok=True)
    for k, fold in enumerate(build_folds(dataset.frame, dataset.label_column, tb.FOLDS, seed), start=1):
        train = features.iloc[np.asarray(fold.train_indices)].reset_index(drop=True)
        validation = features.iloc[np.asarray(fold.validation_indices)].reset_index(drop=True)
        test = features.iloc[np.asarray(fold.test_indices)].reset_index(drop=True)
        imputer = fit_missforest_imputers(train, numerical, categorical)[0]
        naive = json.loads((source / f"naive_fold_{k}.json").read_text())

        def fill(frame: pd.DataFrame, table: pd.DataFrame) -> list[object]:
            r, c = table["row"].to_numpy(dtype=int), frame.columns.get_indexer(pd.Index(table["column"]))
            hide = np.zeros(frame.shape, dtype=bool)
            hide[r, c] = True
            values = imputer.impute(frame, hide).to_numpy(dtype=object)[r, c]
            return [float(g) if kind == NUMERICAL else str(g) for g, kind in zip(values, table["kind"])]

        test_cells = _typed_load(source / f"test_fold_{k}.csv", ("actual", "model", "tabpfn"))
        if not keep_tabpfn:
            test_cells = test_cells.drop(columns="tabpfn")
        test_cells["missforest"] = pd.Series(fill(test, test_cells), dtype=object)
        score = tb._score(test_cells.assign(imputed=test_cells["missforest"]), naive)
        assert abs(score - logged[k - 1]) < 1e-9, f"{variant} s{seed} fold {k}: missForest {score} is not the run's {logged[k - 1]}"
        selection = _typed_load(source / f"selection_fold_{k}.csv", ("actual", "model", "tabpfn"))
        if not keep_tabpfn:
            selection = selection.drop(columns="tabpfn")
        guesses: list[object] = [None] * len(selection)
        for g in range(e82.GROUPS):
            part = np.flatnonzero(selection["group"].to_numpy() == g)
            for i, guess in zip(part, fill(validation, selection.iloc[part])):
                guesses[i] = guess
        selection["missforest"] = pd.Series(guesses, dtype=object)
        test_cells.to_csv(out / f"test_fold_{k}.csv", index=False)
        selection.to_csv(out / f"selection_fold_{k}.csv", index=False)
        for name in (f"naive_fold_{k}.json", f"scale_fold_{k}.json"):
            shutil.copy(source / name, out / name)
    print(f"DONE {variant} s{seed}: missForest equals the run's on 5 folds, {time.time() - started:.0f} s", flush=True)


def cells(study: str, variant: str, seed: int) -> None:
    experiment, state, _ = STUDIES[study]
    add_missforest(experiment, state / "cells", STATE / "cells" / study, variant, seed, keep_tabpfn=False)


def _load(study: str, variant: str, seed: int, kind: str, k: int) -> pd.DataFrame:
    # the rules read the other side as "tabpfn"; here it is missForest
    frame = _typed_load(STATE / "cells" / study / f"{variant}_s{seed}" / f"{kind}_fold_{k}.csv", ("actual", "model", "missforest"))
    return frame.rename(columns={"missforest": "tabpfn"})


def scores(study: str, variant: str, seed: int) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {m: [] for m in ("H", "missforest", "table", "column")}
    directory = STATE / "cells" / study / f"{variant}_s{seed}"
    for k in range(1, tb.FOLDS + 1):
        test, selection = _load(study, variant, seed, "test", k), _load(study, variant, seed, "selection", k)
        naive = json.loads((directory / f"naive_fold_{k}.json").read_text())
        scales = json.loads((directory / f"scale_fold_{k}.json").read_text())
        fills = {"H": list(test["model"]), "missforest": list(test["tabpfn"])}
        fills |= {r: weights.apply(test, rules.rule_weights(r, selection, selection, scales), weights.EQUAL) for r in ("table", "column")}
        for method, values in fills.items():
            out[method].append(tb._score(test.assign(imputed=np.array(values, dtype=object)), naive))
    return out


# E163–E165: both sides in one cell table; the pick rule chooses, per fold, the side whose `table` hybrid scores lower
# on fold k's validation cells (the same cells that choose its weights; a tie keeps TabPFN, ADR 0031 decision 6's side).
PAIRS = (("table_missforest", "H"), ("table_missforest", "missforest"), ("table_tabpfn", "H"), ("table_tabpfn", "tabpfn"))


def _side(frame: pd.DataFrame, other: str) -> pd.DataFrame:
    return frame.drop(columns=[c for c in ("tabpfn", "missforest") if c != other]).rename(columns={other: "tabpfn"})


def _crossfit(chosen: pd.DataFrame, scales: dict[str, float], naive: dict[str, float | str]) -> float:
    """E168: the side's `table` hybrid on fold k's validation cells, each group filled by weights chosen on the other four."""
    fills: list[object] = [None] * len(chosen)
    groups = chosen["group"].to_numpy()
    for g in range(e82.GROUPS):
        rest, part = chosen[groups != g], np.flatnonzero(groups == g)
        for i, value in zip(part, weights.apply(chosen.iloc[part], rules.rule_weights("table", rest, rest, scales), weights.EQUAL)):
            fills[i] = value
    return tb._score(chosen.assign(imputed=np.array(fills, dtype=object)), naive)


def both_sides(directory: Path) -> dict[str, list[float]]:
    """Per fold, on the test cells: H, TabPFN, missForest, each side's `table` hybrid, the pick and the cross-fit pick (E168:
    the side whose held-out-group score is lower; the test fills are that side's `table` hybrid, as the pick's)."""
    out: dict[str, list[float]] = {m: [] for m in ("H", "tabpfn", "missforest", "table_tabpfn", "table_missforest", "pick", "picked_missforest",
                                                   "crossfit", "crossfit_missforest")}
    for k in range(1, tb.FOLDS + 1):
        test = _typed_load(directory / f"test_fold_{k}.csv", ("actual", "model", "tabpfn", "missforest"))
        selection = _typed_load(directory / f"selection_fold_{k}.csv", ("actual", "model", "tabpfn", "missforest"))
        naive = json.loads((directory / f"naive_fold_{k}.json").read_text())
        scales = json.loads((directory / f"scale_fold_{k}.json").read_text())
        fills = {"H": list(test["model"]), "tabpfn": list(test["tabpfn"]), "missforest": list(test["missforest"])}
        validation: dict[str, float] = {}
        held_out: dict[str, float] = {}
        for other in ("tabpfn", "missforest"):
            chosen = _side(selection, other)
            w = rules.rule_weights("table", chosen, chosen, scales)
            fills[f"table_{other}"] = weights.apply(_side(test, other), w, weights.EQUAL)
            validation[other] = tb._score(chosen.assign(imputed=np.array(weights.apply(chosen, w, weights.EQUAL), dtype=object)), naive)
            held_out[other] = _crossfit(chosen, scales, naive)
        side = "missforest" if validation["missforest"] < validation["tabpfn"] else "tabpfn"
        crossfit = "missforest" if held_out["missforest"] < held_out["tabpfn"] else "tabpfn"
        fills["pick"], fills["crossfit"] = fills[f"table_{side}"], fills[f"table_{crossfit}"]
        for method, values in fills.items():
            out[method].append(tb._score(test.assign(imputed=np.array(values, dtype=object)), naive))
        out["picked_missforest"].append(float(side == "missforest"))
        out["crossfit_missforest"].append(float(crossfit == "missforest"))
    return out


def tally() -> None:
    for study, (_, _, seeds) in STUDIES.items():
        missing = [(v, s) for v in VARIANTS for s in seeds if not (STATE / "cells" / study / f"{v}_s{s}" / "selection_fold_5.csv").exists()]
        if missing:
            print(f"## {study}: incomplete, {missing}")
            continue
        print(f"## {study} (seeds {'/'.join(map(str, seeds))})")
        for variant in VARIANTS:
            per = [scores(study, variant, s) for s in seeds]
            values = {m: [np.asarray(p[m]) for p in per] for m in per[0]}
            levels = ", ".join(f"{m} {np.concatenate(v).mean():.4f}" for m, v in values.items())
            parts = []
            for first, second in (("table", "H"), ("table", "missforest"), ("H", "missforest"), ("column", "H"), ("column", "missforest")):
                scale = float(np.mean([v.mean() for v in values[second]]))
                v = common.verdict([a - b for a, b in zip(values[first], values[second])], sesoi=SESOI * scale)
                parts.append(f"{first} − {second} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
            print(f"- `{variant}`: {levels}\n  - " + "\n  - ".join(parts))
        print()


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["cells"] and len(argv) == 4:
        cells(argv[1], argv[2], int(argv[3]))
    elif argv == ["--tally"]:
        tally()
    else:
        sys.exit(__doc__)
