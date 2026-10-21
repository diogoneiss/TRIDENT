"""E60: are TRIDENT's and TabPFN's errors complementary? A blend of their fills, offline
(imputation-protocol/15). Nothing is written to the store.

    uv run --python 3.11 --extra tabpfn python -B scripts/experiments/tabpfn_blend.py <variant> <seed> <out_dir>
    uv run --python 3.11 python -B scripts/experiments/tabpfn_blend.py --tally <out_dir>

On E44's confirmation runs, per fold:

- **The model's fills** are its induced fills from the run's fold ledger: the headline K's, since
  E44 ran before ADR 0027.
- **TabPFN's fills** come from TabPFN v2 refitted on the same training fold, as E55 and E56 did,
  and asked for exactly the ledger's cells.
- **The blend** averages the two on every numerical cell and keeps the model's category on a
  categorical one, since the ledger holds no class probabilities.

``impute_score`` is written for the model, TabPFN and the blend, against the fold's mean/mode fill.
The model's must equal the run's K fold by fold, or the cell stops. ``--tally`` applies the rule
fixed in the ticket. Threads are capped at 4.
"""

from __future__ import annotations

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

from scripts.experiments import confirmation_report as confirmation  # noqa: E402
from scripts.experiments import report_common as common  # noqa: E402
from src.training.data import build_folds, prepare_dataset  # noqa: E402
from src.training.imputation_metrics import NUMERICAL, mean_mode_baselines, score_cells  # noqa: E402
from src.training.types import DatasetSpec  # noqa: E402

STATE = Path("/scratch2/diogoneiss/trident-2026-10-17-confirmation")
EXPERIMENT = "confirmation-2026-10-17"
SEEDS = (707, 808, 909)
FOLDS = 5
SESOI = 0.01


def _ledger(variant: str, seed: int, fold: int) -> pd.DataFrame:
    found = sorted((STATE / "results" / f"{variant}_s{seed}" / variant).glob(f"*/imputation/imputation_fold_{fold}_cells.csv"))
    if len(found) != 1:
        raise FileNotFoundError(f"{variant} s{seed} fold {fold}: {len(found)} ledgers")
    frame = pd.read_csv(found[0], dtype={"actual": object, "imputed": object})
    return frame[frame["population"] == "induced"].reset_index(drop=True)


def _score(cells: pd.DataFrame, naive: dict[str, float | str]) -> float:
    return float(score_cells(cells, naive).metrics.get("impute_score", float("nan")))


def blend(variant: str, seed: int, out: Path) -> None:
    from src.training.tabpfn_baseline import fit_tabpfn_imputers

    warnings.filterwarnings("ignore")
    run = common.parents(EXPERIMENT)[(variant, seed)][0]
    logged = common.model(common.fold_table(run), run, "K")
    dataset = prepare_dataset(DatasetSpec.from_name(variant, None))
    features = dataset.frame.drop(columns=[dataset.label_column])
    numerical, categorical = list(dataset.numerical_columns), list(dataset.categorical_columns)
    rows: list[dict[str, object]] = []
    started = time.time()
    for ordinal, fold in enumerate(build_folds(dataset.frame, dataset.label_column, FOLDS, seed), start=1):
        train = features.iloc[np.asarray(fold.train_indices)].reset_index(drop=True)
        test = features.iloc[np.asarray(fold.test_indices)].reset_index(drop=True)
        naive = dict(mean_mode_baselines(train, numerical, categorical))
        cells = _ledger(variant, seed, ordinal)
        numeric = (cells["kind"] == NUMERICAL).to_numpy()
        cells.loc[numeric, "actual"] = cells.loc[numeric, "actual"].astype(float)
        cells.loc[numeric, "imputed"] = cells.loc[numeric, "imputed"].astype(float)
        positions = cells["row"].to_numpy(dtype=int)
        columns = test.columns.get_indexer(pd.Index(cells["column"]))
        hidden = np.zeros(test.shape, dtype=bool)
        hidden[positions, columns] = True
        imputer = fit_tabpfn_imputers(train, numerical, categorical)[0]
        filled = imputer.impute(test, hidden).to_numpy(dtype=object)[positions, columns]
        tabpfn = [float(g) if k == NUMERICAL else str(g) for g, k in zip(filled, cells["kind"])]
        mixed = [(float(m) + float(t)) / 2 if k == NUMERICAL else m for m, t, k in zip(cells["imputed"], tabpfn, cells["kind"])]
        model_score = _score(cells, naive)
        assert abs(model_score - float(logged[ordinal - 1])) < 1e-9, f"{variant} s{seed} fold {ordinal}: {model_score} against {logged[ordinal - 1]}"
        for method, frame in (("model", cells), ("tabpfn", cells.assign(imputed=tabpfn)), ("blend", cells.assign(imputed=mixed))):
            rows.append({"variant": variant, "seed": seed, "fold": ordinal, "method": method, "impute_score": _score(frame, naive),
                         "numeric_share": float(numeric.mean())})
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{variant}_s{seed}.csv", index=False)
    print(f"DONE {variant} s{seed}: model equals the run's K on 5 folds, {time.time() - started:.0f} s", flush=True)


def tally(out: Path) -> None:
    table = pd.concat([pd.read_csv(p) for p in sorted(out.glob("*.csv"))], ignore_index=True)
    both = []
    counts: dict[str, list[str]] = {}
    for variant in confirmation.VARIANTS:
        rows = table[table.variant == variant]
        if rows["seed"].nunique() < len(SEEDS):
            print(f"- `{variant}`: incomplete")
            continue
        values = {m: [rows[(rows.seed == s) & (rows.method == m)].sort_values("fold")["impute_score"].to_numpy(dtype=float) for s in SEEDS]
                  for m in ("model", "tabpfn", "blend")}
        readings = {}
        for other in ("model", "tabpfn"):
            diffs = [b - o for b, o in zip(values["blend"], values[other])]
            readings[other] = common.verdict(diffs, sesoi=SESOI * float(np.mean([o.mean() for o in values[other]])))
            counts.setdefault(f"blend − {other}", []).append(readings[other].call)
        share = float(rows["numeric_share"].mean())
        print(f"- `{variant}` (numeric share {share:.2f}): " + "; ".join(f"blend − {o} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}" for o, v in readings.items()))
        if readings["model"].call == "better" and readings["tabpfn"].call == "better":
            both.append(variant)
    for k, calls in counts.items():
        print(f"  - {k}: {common.tally(calls)}")
    complete = len(counts.get("blend − model", []))
    if complete < len(confirmation.VARIANTS):
        print(f"\nPrimary: incomplete ({complete} of {len(confirmation.VARIANTS)})")
        return
    reading = "complementary" if len(both) >= 5 else "not complementary" if len(both) <= 1 else "mixed"
    print(f"\nPrimary: the blend beats both on {len(both)} of {complete} ({both or 'none'}) -> {reading} ({common.RULE})")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--tally"] and len(argv) == 2:
        tally(Path(argv[1]))
    elif len(argv) == 3:
        blend(argv[0], int(argv[1]), Path(argv[2]))
    else:
        sys.exit(__doc__)
