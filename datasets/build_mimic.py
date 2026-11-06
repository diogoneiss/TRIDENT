"""Build the MIMIC-IV tables of T50 (docs/tickets/mimic/02-tables.md).

Two views of the kg-impute cohort (one row per ICU stay, label: death in hospital), with the
gaps as the hospital left them:

  mimic-cardio   the 19 cardiovascular variables over the first 30 h (kg-impute's tensors)
  mimic-sofa     the SOFA components' raw worst values over the first 24 h

and, cut from each, the rows with every measured column present (``<base>-donor``, the complete
tables E149 transplants gap patterns onto). These go under ``$TRIDENT_DATASETS_ROOT``, outside
the checkout. A 1,000-stay sample of ``mimic-cardio`` is written into the checkout as the toy
problem, ``mimic-cardio-toy``.

Every column of a table but ``class`` is a feature, so the identifiers are not in the tables:
each has a row-aligned ``<base>_index.csv`` beside it (``stay_id``, ``subject_id``, ``hadm_id``).

The training environment has no h5py, and this needs nothing else of it, so from the checkout:

    uv run --no-project --with h5py --with pandas python datasets/build_mimic.py

Inputs exist on gorgona8 only (the tensors and the SOFA component files); the tables reach
medusa1 by rsync.
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parent.parent
VASOPRESSORS = ["dobutamine", "dopamine", "epinephrine", "norepinephrine"]
DEMOGRAPHICS = ["age", "gender", "admission_type"]
IDENTIFIERS = ["stay_id", "subject_id", "hadm_id"]
LABEL = "class"
CLIP_PERCENTILES = (0.5, 99.5)
TOY_ROWS = 1000
TOY_SEED = 42


def stays(mimic: Path, stay_ids: np.ndarray) -> pd.DataFrame:
    """Identifiers, demographics and the label of each stay, in ``stay_ids``' order."""
    icu = pd.read_csv(
        mimic / "icu/icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "los"],
        parse_dates=["intime"],
    )
    patients = pd.read_csv(
        mimic / "hosp/patients.csv.gz", usecols=["subject_id", "gender", "anchor_age", "anchor_year"]
    )
    admissions = pd.read_csv(
        mimic / "hosp/admissions.csv.gz",
        usecols=["hadm_id", "admission_type", "deathtime", "hospital_expire_flag"],
        parse_dates=["deathtime"],
    )
    table = (
        pd.DataFrame({"stay_id": stay_ids})
        .merge(icu, on="stay_id", how="left", validate="one_to_one")
        .merge(patients, on="subject_id", how="left", validate="many_to_one")
        .merge(admissions, on="hadm_id", how="left", validate="many_to_one")
    )
    assert table["hospital_expire_flag"].notna().all(), "a stay without an admission"
    table["age"] = table["anchor_age"] + table["intime"].dt.year - table["anchor_year"]
    table[LABEL] = table["hospital_expire_flag"].astype(int)
    table["hours_to_death"] = (table["deathtime"] - table["intime"]).dt.total_seconds() / 3600
    return table


def cardio(tensors: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Each measured variable's mean over its observed hours; each vasopressor's highest rate.

    The tensors hold the chart as typed, entry errors included (a potassium of 999999). Each
    variable's observed values are first clipped to their 0.5th and 99.5th percentiles, the rule
    kg-impute's own normaliser applies, here over the whole cohort.
    """
    with h5py.File(tensors, "r") as store:
        names = [str(name) for name in store.attrs["variables"]]
        values, observed = store["X"][:], store["M"][:].astype(bool)
        stay_ids = store["stay_ids"][:]
    columns = {}
    for index, name in enumerate(names):
        low, high = np.percentile(values[:, :, index][observed[:, :, index]], CLIP_PERCENTILES)
        values[:, :, index] = np.clip(values[:, :, index], low, high)
        if name in VASOPRESSORS:
            # A drug that was never infused is a dose of zero, not a gap.
            columns[name] = np.where(observed[:, :, index], values[:, :, index], 0.0).max(axis=1)
        else:
            hours = observed[:, :, index].sum(axis=1)
            total = np.where(observed[:, :, index], values[:, :, index], 0.0).sum(axis=1)
            columns[name] = np.where(hours > 0, total / np.maximum(hours, 1), np.nan)
    return stay_ids, pd.DataFrame(columns)


def sofa(directory: Path, stay_ids: np.ndarray) -> pd.DataFrame:
    """The worst raw value of each SOFA component in hours 1 to 24, never the scores.

    The scores count a missing component as normal, which is the rule under study.
    """

    def worst(file: str, how: dict[str, str]) -> pd.DataFrame:
        hourly = pd.read_csv(directory / file, usecols=["stay_id", "hr", *how], na_values=["Null"])
        hourly = hourly[hourly["hr"].between(1, 24)].copy()
        for column in how:
            # The same clip as ``cardio``: a mean pressure of -22767 is a sentinel, not a reading.
            # A rate's zeros are hours without the drug, so only its upper end is clipped.
            rate = column.startswith("rate_")
            seen = hourly[column][hourly[column] > 0] if rate else hourly[column].dropna()
            low, high = np.percentile(seen, CLIP_PERCENTILES)
            hourly[column] = hourly[column].clip(None if rate else low, high)
        return hourly.groupby("stay_id").agg(how)

    circulation = worst(
        "sofa_hourly_scores.csv",
        {"meanbp_min": "min", **{f"rate_{drug}": "max" for drug in VASOPRESSORS}},
    )
    respiration = worst(
        "sofa_hourly_scores_respiratory.csv",
        {
            "pao2fio2ratio_vent_min": "min", "pao2fio2ratio_novent_min": "min",
            "sfratio_vent_min": "min", "sfratio_novent_min": "min",
        },
    )
    parts = [
        circulation,
        worst("sofa_hourly_scores_cns.csv", {"gcs_min": "min"}),
        worst("sofa_hourly_scores_coagulation.csv", {"platelet_min": "min"}),
        worst("sofa_hourly_scores_liver.csv", {"bilirubin_max": "max"}),
        # ``uo_24hr`` is a running 24-hour volume: its last value in the window is the day's.
        worst(
            "sofa_hourly_scores_renal.csv",
            {"creatinine_max": "max", "bun_max": "max", "uo_24hr": "last"},
        ),
        respiration,
    ]
    table = pd.concat(parts, axis=1).reindex(stay_ids).reset_index(drop=True)
    drugs = [f"rate_{drug}" for drug in VASOPRESSORS]
    table[drugs] = table[drugs].fillna(0.0)
    on_vent, off_vent = table["pao2fio2ratio_vent_min"], table["pao2fio2ratio_novent_min"]
    table["pf_ratio"] = np.fmin(on_vent, off_vent)
    # Whether the lowest ratio was taken on invasive ventilation; a gap where there is no ratio.
    table["pf_on_vent"] = np.where(on_vent.fillna(np.inf) <= off_vent.fillna(np.inf), "yes", "no")
    table.loc[table["pf_ratio"].isna(), "pf_on_vent"] = np.nan
    table["sf_ratio"] = np.fmin(table["sfratio_vent_min"], table["sfratio_novent_min"])
    return table.drop(
        columns=[
            "pao2fio2ratio_vent_min", "pao2fio2ratio_novent_min",
            "sfratio_vent_min", "sfratio_novent_min",
        ]
    )


def write(root: Path, base: str, variant: str, table: pd.DataFrame, index: pd.DataFrame) -> None:
    """A table and its row-aligned index, in the layout the training loader reads."""
    assert len(table) == len(index) and LABEL in table.columns
    assert not set(IDENTIFIERS) & set(table.columns), "an identifier would be read as a feature"
    directory = root / base
    directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(directory / f"{base}_{variant}.csv", index=False)
    index[IDENTIFIERS].to_csv(directory / f"{base}_index.csv", index=False)
    categorical = [name for name in table.columns if name != LABEL and table[name].dtype == object]
    (REPOSITORY / "datasets/categorical_columns" / f"{base}.txt").write_text(",".join(categorical))


def describe(name: str, table: pd.DataFrame) -> None:
    features = table.drop(columns=[LABEL])
    gaps = features.isna().mean()
    print(f"\n{name}: {len(table)} rows, {features.shape[1]} features, deaths {table[LABEL].mean():.1%}")
    print(f"  rows with every feature: {int(features.notna().all(axis=1).sum())}")
    if gaps.any():
        print(f"  columns with gaps: {int((gaps > 0).sum())}, their mean gap {gaps[gaps > 0].mean():.1%}")
    for column, share in gaps[gaps > 0].sort_values(ascending=False).items():
        measured = table.loc[features[column].notna(), LABEL].mean()
        unmeasured = table.loc[features[column].isna(), LABEL].mean()
        print(f"  {column:<16} gap {share:6.1%}   deaths measured {measured:.1%} / not {unmeasured:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mimic_root", type=Path, default=os.environ.get("MIMIC_ROOT"))
    parser.add_argument("--out", type=Path, default=os.environ.get("TRIDENT_DATASETS_ROOT"))
    parser.add_argument(
        "--tensors", type=Path,
        default=Path("/scratch2/diogoneiss/calculo-sistemas/cardiokg-impute/data/tensors/tensors.h5"),
    )
    parser.add_argument("--sofa_dir", type=Path, default=Path("/scratch/gabriel-alkmim/disf_pred"))
    args = parser.parse_args()
    if args.mimic_root is None or args.out is None:
        parser.error("set MIMIC_ROOT and TRIDENT_DATASETS_ROOT, or pass --mimic_root and --out")

    stay_ids, cardio_features = cardio(args.tensors)
    index = stays(args.mimic_root, stay_ids)
    assert index["stay_id"].is_unique and len(index) == len(stay_ids)
    tables = {
        "mimic-cardio": cardio_features,
        "mimic-sofa": sofa(args.sofa_dir, stay_ids),
    }
    print(f"stays {len(index)}, patients {index['subject_id'].nunique()}")
    print(f"ICU stay under 24 h: {int((index['los'] < 1).sum())}; "
          f"died within 30 h of admission: {int((index['hours_to_death'] <= 30).sum())}")

    for base, features in tables.items():
        table = pd.concat([features, index[DEMOGRAPHICS + [LABEL]]], axis=1)
        assert not table[VASOPRESSORS if base == "mimic-cardio" else DEMOGRAPHICS].isna().any().any()
        write(args.out, base, "natural", table, index)
        describe(f"{base}_natural", table)

        complete = table.drop(columns=[LABEL]).notna().all(axis=1).to_numpy()
        write(
            args.out, f"{base}-donor", "00nan",
            table[complete].reset_index(drop=True), index[complete].reset_index(drop=True),
        )
        describe(f"{base}-donor_00nan", table[complete])

        if base == "mimic-cardio":
            # The toy problem: a seeded sample, stratified by the label, tracked in the checkout.
            generator = np.random.default_rng(TOY_SEED)
            dead = np.flatnonzero(table[LABEL] == 1)
            alive = np.flatnonzero(table[LABEL] == 0)
            n_dead = round(TOY_ROWS * len(dead) / len(table))
            rows = np.sort(np.concatenate([
                generator.choice(dead, n_dead, replace=False),
                generator.choice(alive, TOY_ROWS - n_dead, replace=False),
            ]))
            toy = table.iloc[rows].reset_index(drop=True)
            write(
                REPOSITORY / "datasets/processed_datasets", "mimic-cardio-toy", "natural",
                toy, index.iloc[rows].reset_index(drop=True),
            )
            describe("mimic-cardio-toy_natural", toy)


if __name__ == "__main__":
    main()
