"""The two wide tables, added on a reviewer's request for a table over 100 columns and 10,000 rows.

  gas-drift  OpenML 1476  128 numerical features, 6 classes, 13,895 of its 13,910 rows
  nomao      OpenML 1486  89 numerical and 29 categorical features, 2 classes, 34,465 rows

Roles (the user's decision, 2026-11-25): options are developed on gas-drift and confirmed on nomao;
nothing is tuned or chosen on nomao.

gas-drift loses 15 rows (the user's decision, 2026-11-25): every row holding a cell more than 20
standard deviations from its column's mean, both taken over the full table (population SD), applied
once. With them, a fold that held those rows out scaled them to hundreds of training SDs and every
method, the mean fill included, scored 1.0 there. The pass is not repeated: re-standardised, 14 of
the remaining rows exceed 20 again (at most 50). nomao keeps every row.

    uv run --python 3.11 python datasets/build_wide_tables.py raw <dir>
        datasets_raw/<base>.csv from <dir>/<base>.arff, OpenML's files (md5 checked):
        https://openml.org/data/v1/download/1588715/gas-drift.arff
        https://openml.org/data/v1/download/1592278/nomao.arff
    uv run --python 3.11 python datasets/build_wide_tables.py variants [--root R] [base ...]
        <R>/<base>/<base>_{00,20,40,60,80}nan.csv and <R>/splits/<base>_split.json (R defaults to
        processed_datasets), by generate_splits.py's functions and seeds. Its main() is not used:
        it would rewrite the variants of every other table.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from scipy.io import arff

import generate_splits as g

HERE = Path(__file__).resolve().parent
MD5 = {"gas-drift": "e205264c230a3e15c7546c18ddceedaf", "nomao": "8fc1ac73fbe5236892e166f9f24d7221"}
ROWS = {"gas-drift": (13895, 129), "nomao": (34465, 119)}
EXTREME_SD = 20.0
GAS_DRIFT_DROPPED = [4262, 4263, 4264, 4265, 4266, 4267, 4412, 4417, 4423, 4424, 4425, 4432, 6840, 6908, 13471]


def raw(base: str, arff_dir: Path) -> pd.DataFrame:
    path = arff_dir / f"{base}.arff"
    assert hashlib.md5(path.read_bytes()).hexdigest() == MD5[base], f"{path} is not OpenML's file"
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    nominal = [n for n, t in zip(meta.names(), meta.types()) if t == "nominal"]
    for column in nominal:  # the label and nomao's categorical columns arrive as byte strings of integers
        df[column] = df[column].str.decode("ascii").astype(int)
    df = df.rename(columns={"Class": g.TARGET_COLUMN})
    assert not df.isna().any().any()
    categorical = [n for n in nominal if n != "Class"]
    listed = HERE / "categorical_columns" / f"{base}.txt"
    assert categorical == (listed.read_text().strip().split(",") if listed.exists() else []), base
    if base == "gas-drift":
        features = df.drop(columns=[g.TARGET_COLUMN])
        extreme = (((features - features.mean()) / features.std(ddof=0)).abs() > EXTREME_SD).any(axis=1)
        assert df.index[extreme].tolist() == GAS_DRIFT_DROPPED
        df = df[~extreme].reset_index(drop=True)
    assert df.shape == ROWS[base], df.shape
    return df


def variants(base: str, root: Path) -> None:
    df = pd.read_csv(HERE / "datasets_raw" / f"{base}.csv")
    assert df.shape == ROWS[base], df.shape
    (root / base).mkdir(parents=True, exist_ok=True)
    (root / "splits").mkdir(exist_ok=True)
    split = g.stratified_split_indices(df, g.TARGET_COLUMN)
    (root / "splits" / f"{base}_split.json").write_text(json.dumps(split, indent=4))
    df.to_csv(root / base / f"{base}_00nan.csv", index=False)
    for pct in g.NAN_PERCENTAGES:
        holed = g.inject_nans(df, pct, g.TARGET_COLUMN, seed=g.RANDOM_SEED + int(pct * 100))
        assert holed is not None
        holed.to_csv(root / base / f"{base}_{int(pct * 100):02d}nan.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    steps.add_parser("raw").add_argument("arff_dir", type=Path)
    step = steps.add_parser("variants")
    step.add_argument("--root", type=Path, default=HERE / "processed_datasets")
    step.add_argument("bases", nargs="*", default=list(MD5))
    args = parser.parse_args()
    if args.step == "raw":
        for name in MD5:
            table = raw(name, args.arff_dir)
            table.to_csv(HERE / "datasets_raw" / f"{name}.csv", index=False)
            print(name, table.shape, table[g.TARGET_COLUMN].value_counts().sort_index().to_dict())
    else:
        for name in args.bases:
            variants(name, args.root)
            print(name, "variants under", args.root)
