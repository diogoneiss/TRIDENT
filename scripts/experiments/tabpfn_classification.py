"""E83: TabPFN v2's classifier against TRIDENT's classification on E71's folds (T40).

    uv run --python 3.11 --extra tabpfn python -B scripts/experiments/tabpfn_classification.py run [<variant> <seed> ...]
    uv run --python 3.11 python -B scripts/experiments/tabpfn_classification.py --tally
    uv run --python 3.11 --extra tabpfn python -B scripts/experiments/tabpfn_classification.py --study e92 run|--todo <n>|--tally

``run`` fits TabPFN v2's classifier, untuned, on each training split of E71's folds (3-fold
``build_folds`` at seeds 42 and 7, as ``cls-stage2-2026-10-24`` arm A trained TRIDENT). It predicts
the test split and writes macro-F1, micro-F1 and accuracy per fold, one CSV per (variant, seed),
skipping a pair already written. Settings are E55's (``tabpfn_baseline``): v2's checkpoints,
8 estimators, ``random_state`` 0, the GPU. Gaps are NaN, which TabPFN reads natively. A categorical
feature is its index in the training split's vocabulary, declared categorical. The context is the
training split, capped at ``CONTEXT_ROWS`` (v2's limit) by a stratified fixed-seed subsample. The
test split is predicted in fixed chunks of ``CHUNK`` rows. letter (26 classes) is out: v2 allows
10 classes, and that limit cannot be raised.

``--study e92`` (classification-step-zero/05) runs the same fits on E87's folds (ten folds, five seeds,
on medusa1) and pairs them with E87's arm A on disk. Each fold also predicts with
``balance_probabilities`` on, from the same fit: the balanced arm differs only in that post-processing.
``--todo <n>`` deals the pairs not yet written to one of ``QUEUES`` GPU queues.
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

from scripts.experiments import report_common as common  # noqa: E402

E71 = "cls-stage2-2026-10-24"
STATE = Path("/scratch2/diogoneiss/trident-2026-10-27-tabpfn-classification")
SEEDS: tuple[int, ...] = (42, 7)
TABLES = ("pendigits", "electricity", "vehicle", "kr-vs-kp", "spambase", "credit-g", "biodeg", "kc2")
LEVELS = (0, 20, 40, 60, 80)
VARIANTS = [f"{table}_{level:02d}nan" for table in TABLES for level in LEVELS]
FOLDS = 3
BALANCED = False
QUEUES = 2
CHUNK = 2000
SESOI = 0.01


def _matrix(frame: pd.DataFrame, categorical: list[str], vocabulary: dict[str, list[str]]) -> np.ndarray:
    from src.embedder import as_category_strings

    columns = []
    for name in frame.columns:
        if name in categorical:
            index = {label: float(k) for k, label in enumerate(vocabulary[name])}
            columns.append(np.array([index.get(label, np.nan) for label in as_category_strings(frame[name])]))
        else:
            columns.append(frame[name].to_numpy(dtype=float))
    return np.column_stack(columns)


def run_pair(variant: str, seed: int) -> None:
    import importlib

    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    from src.embedder import as_category_strings
    from src.training.data import build_folds, prepare_dataset
    from src.training.tabpfn_baseline import CONTEXT_ROWS, N_ESTIMATORS, TABPFN_SEED, _NOT_A_CLASS
    from src.training.types import DatasetSpec

    out = STATE / "csv" / f"{variant}_s{seed}.csv"
    if out.exists():
        return
    # A CPU fit differs fold by fold from a GPU one (E86): with the GPU lost, fail rather than mix devices (E92).
    if not torch.cuda.is_available():
        sys.exit(f"no CUDA device: {variant} s{seed} not run")
    warnings.filterwarnings("ignore")
    started = time.time()
    tabpfn = importlib.import_module("tabpfn")
    version = importlib.import_module("tabpfn.constants").ModelVersion.V2
    dataset = prepare_dataset(DatasetSpec.from_name(variant, None))
    label = dataset.label_column
    features = dataset.frame.drop(columns=[label])
    categorical = list(dataset.categorical_columns)
    rows = []
    for ordinal, fold in enumerate(build_folds(dataset.frame, label, FOLDS, seed), start=1):
        train = features.iloc[np.asarray(fold.train_indices)].reset_index(drop=True)
        test = features.iloc[np.asarray(fold.test_indices)].reset_index(drop=True)
        y_train = dataset.frame[label].to_numpy()[np.asarray(fold.train_indices)]
        y_test = dataset.frame[label].to_numpy()[np.asarray(fold.test_indices)]
        vocabulary = {c: sorted(set(as_category_strings(train[c]).tolist()) - _NOT_A_CLASS) for c in categorical}
        x_train, x_test = _matrix(train, categorical, vocabulary), _matrix(test, categorical, vocabulary)
        if len(x_train) > CONTEXT_ROWS:
            keep, _ = train_test_split(np.arange(len(x_train)), train_size=CONTEXT_ROWS, stratify=y_train, random_state=TABPFN_SEED)
            x_train, y_train = x_train[np.sort(keep)], y_train[np.sort(keep)]
        classifier = tabpfn.TabPFNClassifier.create_default_for_version(
            version, n_estimators=N_ESTIMATORS, random_state=TABPFN_SEED,
            categorical_features_indices=[list(features.columns).index(c) for c in categorical] or None,
            device="cuda" if torch.cuda.is_available() else "cpu")
        numpy_state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[0] if torch.cuda.is_available() else []):
                classifier.fit(x_train, y_train)
                predicted = np.concatenate([classifier.predict(x_test[i:i + CHUNK]) for i in range(0, len(x_test), CHUNK)])
            if BALANCED:
                classifier.balance_probabilities = True
                with torch.random.fork_rng(devices=[0] if torch.cuda.is_available() else []):
                    balanced = np.concatenate([classifier.predict(x_test[i:i + CHUNK]) for i in range(0, len(x_test), CHUNK)])
        finally:
            np.random.set_state(numpy_state)
        row = {"variant": variant, "seed": seed, "fold": ordinal, "context_rows": len(x_train), "test_rows": len(x_test),
               "f1_macro": float(f1_score(y_test, predicted, average="macro")),
               "f1_micro": float(f1_score(y_test, predicted, average="micro")),
               "accuracy": float(accuracy_score(y_test, predicted))}
        if BALANCED:
            row |= {"f1_macro_balanced": float(f1_score(y_test, balanced, average="macro")),
                    "accuracy_balanced": float(accuracy_score(y_test, balanced))}
        rows.append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"DONE {variant} s{seed}: {time.time() - started:.0f} s", flush=True)


def tally() -> None:
    table = pd.concat([pd.read_csv(p) for p in sorted((STATE / "csv").glob("*.csv"))], ignore_index=True)
    trident = common.parents(E71, "A")
    per_level: dict[int, dict[int, list[float]]] = {level: {seed: [] for seed in SEEDS} for level in LEVELS}
    lines: dict[int, list[str]] = {level: [] for level in LEVELS}
    missing = []
    for variant in VARIANTS:
        level = int(variant.split("_")[1].removesuffix("nan"))
        for seed in SEEDS:
            ours = table[(table.variant == variant) & (table.seed == seed)].sort_values("fold")
            run = trident.get((variant, seed))
            if len(ours) != FOLDS or not run:
                missing.append(f"{variant} s{seed}")
                continue
            theirs = common.fold_table(run[0]).sort_values("fold")["f1_macro"].to_numpy(dtype=float)
            per_level[level][seed].extend(list(ours["f1_macro"].to_numpy(dtype=float) - theirs))
        pair = table[table.variant == variant]
        if len(pair) and all(trident.get((variant, s)) for s in SEEDS):
            mean_t = float(np.mean([common.fold_table(trident[(variant, s)][0])["f1_macro"].mean() for s in SEEDS]))
            lines[level].append(f"{variant.split('_')[0]} TabPFN {pair['f1_macro'].mean():.3f} TRIDENT {mean_t:.3f}")
    if missing:
        print(f"incomplete: {', '.join(missing)}")
    for level in LEVELS:
        diffs = [np.array(per_level[level][seed]) for seed in SEEDS]
        if any(len(d) == 0 for d in diffs):
            continue
        # The verdict rule reads lower as better; on (TabPFN − TRIDENT) macro-F1, "worse" means TabPFN higher.
        v = common.verdict(diffs, sesoi=SESOI)
        reading = {"worse": "TabPFN ahead", "better": "TRIDENT ahead", "equivalent": "equivalent", "none": "no verdict"}[v.call]
        print(f"- {level}nan: TabPFN − TRIDENT macro-F1 {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] seeds "
              f"{', '.join(f'{s:+.4f}' for s in v.seeds)}, {sum(len(d) for d in diffs)} fold pairs -> {reading}")
        print("  - " + "; ".join(lines[level]))


def todo(which: int) -> None:
    """Seed by seed, the variants in TABLES order (largest first), dealt round-robin over QUEUES."""
    pairs = [(v, s) for s in SEEDS for v in VARIANTS]
    for k, (variant, seed) in enumerate(pairs):
        if k % QUEUES == which - 1 and not (STATE / "csv" / f"{variant}_s{seed}.csv").exists():
            print(variant, seed)


def tally_e92() -> None:
    """Per level, each TabPFN arm − E87 arm A, fold-paired, pooled over tables with one array per seed; then the calls."""
    from scripts.experiments import cls_tenfold
    from scripts.experiments.cls_stage2_run import cell_dir, fold_metrics

    missing = [f"{v} s{s}" for s in SEEDS for v in VARIANTS
               if not (STATE / "csv" / f"{v}_s{s}.csv").exists() or fold_metrics(cell_dir(cls_tenfold.STATE, v, s, "A")) is None]
    if missing:
        print(f"incomplete: {len(missing)} of {len(SEEDS) * len(VARIANTS)} pairs missing, e.g. {', '.join(missing[:5])}")
        return

    def trident(variant: str, seed: int) -> pd.DataFrame:
        frame = fold_metrics(cell_dir(cls_tenfold.STATE, variant, seed, "A"))
        assert frame is not None
        return frame

    table = pd.concat([pd.read_csv(STATE / "csv" / f"{v}_s{s}.csv") for s in SEEDS for v in VARIANTS], ignore_index=True)
    comparisons = {"plain − TRIDENT": ("f1_macro", None), "balanced − TRIDENT": ("f1_macro_balanced", None),
                   "balanced − plain": ("f1_macro_balanced", "f1_macro")}
    ahead = {name: 0 for name in comparisons}
    behind = {name: 0 for name in comparisons}
    for level in LEVELS:
        print(f"- {level}nan:")
        for name, (left, right) in comparisons.items():
            diffs = []
            for seed in SEEDS:
                d: list[float] = []
                for variant in (v for v in VARIANTS if v.endswith(f"_{level:02d}nan")):
                    ours = table[(table.variant == variant) & (table.seed == seed)].sort_values("fold")
                    theirs = ours[right] if right else trident(variant, seed)["f1_macro"]
                    d.extend(list(ours[left].to_numpy(dtype=float) - theirs.to_numpy(dtype=float)))
                diffs.append(np.array(d))
            # The verdict rule reads lower as better; on (left − right) macro-F1, "worse" means the left side higher.
            v = common.verdict(diffs, sesoi=SESOI)
            ahead[name] += v.call == "worse"
            behind[name] += v.call == "better"
            reading = {"worse": "left ahead", "better": "right ahead", "equivalent": "equivalent", "none": "no verdict"}[v.call]
            print(f"  - {name}: {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] seeds "
                  f"{', '.join(f'{s:+.4f}' for s in v.seeds)}, {sum(len(x) for x in diffs)} fold pairs -> {reading}")
        per_table = []
        for base in TABLES:
            variant = f"{base}_{level:02d}nan"
            mean_t = float(np.mean([trident(variant, s)["f1_macro"].mean() for s in SEEDS]))
            rows = table[table.variant == variant]
            per_table.append(f"{base} plain {rows['f1_macro'].mean():.3f} balanced {rows['f1_macro_balanced'].mean():.3f} TRIDENT {mean_t:.3f}")
        print("  - " + "; ".join(per_table))
    for name in ("plain − TRIDENT", "balanced − TRIDENT"):
        call = "supported" if ahead[name] >= 4 and behind[name] == 0 else "refuted" if ahead[name] <= 1 else "mixed"
        print(f"{name}: TabPFN ahead at {ahead[name]} of 5 levels, TRIDENT at {behind[name]} -> {call}")
    print(f"balanced − plain: balanced ahead at {ahead['balanced − plain']} of 5 levels, plain at {behind['balanced − plain']}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:2] == ["--study", "e92"]:
        STATE = Path("/scratch/diogoneiss/trident-2026-10-28-tabpfn-tenfold")
        SEEDS, FOLDS, BALANCED = (42, 7, 13, 101, 202), 10, True
        argv = argv[2:]
        if argv[:1] == ["--todo"]:
            todo(int(argv[1]))
            sys.exit()
        if argv in (["--tally"], []):
            tally_e92()
            sys.exit()
    if argv[:1] == ["run"]:
        pairs = list(zip(argv[1::2], map(int, argv[2::2]))) or [(v, s) for s in SEEDS for v in VARIANTS]
        for variant, seed in pairs:
            run_pair(variant, seed)
    elif argv in (["--tally"], []):
        tally()
    else:
        sys.exit(__doc__)
