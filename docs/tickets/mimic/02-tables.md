# Two tables from the kg-impute cohort (T50, stage 0; written 2026-11-05)

The tables E150 to E152 run on. Built by one script from `MIMIC_ROOT` and the files named below, written under
`TRIDENT_DATASETS_ROOT` ([01](01-external-datasets-root.md)); the script prints every count this ticket quotes, and
its output replaces the probe figures here before the first cell.

Both tables hold the same 11,082 ICU stays: the suspected-infection cohort of `calculo-sistemas/cardiokg-impute`
(antibiotic prescribed and given, culture ordered, first stay of each patient). They are two views of one
population, not two replications.

**Label, both tables:** death in hospital (`hosp/admissions.hospital_expire_flag`), column `class`. Probe: 16.5%,
1,830 deaths. The kg-impute paper's own downstream label, circulatory failure, is a function of lactate, MAP and
the vasopressors, columns of these tables, so it is not a target here.

## `mimic-cardio`

From `cardiokg-impute/data/tensors/tensors.h5` (19 variables, 30 one-hour bins after ICU admission).

- 15 measured variables, each the mean of its observed bins, a gap when never observed in the 30 h.
- 4 vasopressors (dobutamine, dopamine, epinephrine, norepinephrine): the highest rate, 0 when never infused. A drug
  not given is a dose of zero, not a gap.
- Age, sex (`hosp/patients`) and admission type (`hosp/admissions`); sex and admission type categorical.

Gaps per stay, from the probe of 2026-11-05 (share of stays with no observation of the variable):

| variable | gap |
|---|---|
| troponin T | 65.8% |
| arterial pressure, systolic and diastolic | 60.3% |
| arterial pressure, mean | 57.4% |
| lactate | 28.5% |
| non-invasive pressure (three) | 5.5–5.6% |
| magnesium | 2.3% |
| heart rate, SpO2, potassium, haemoglobin, haematocrit, sodium | ≤ 0.6% |

About a fifth of the 15 measured columns' cells. The gaps are informative: deaths are
19.9% where lactate was measured and 7.9% where it was not, 24.1% against 12.5% for troponin, 19.9% against 14.0%
for the arterial line.

## `mimic-sofa`

From the hourly SOFA component files in `/scratch/gabriel-alkmim/disf_pred/` (gorgona8), hours 1 to 24: the worst
raw value of each component, not the scores. The scores count a missing component as normal, which is the rule under
study.

- Lowest mean pressure, the four vasopressor rates (highest, 0 when none), lowest Glasgow, lowest platelets, highest
  bilirubin, highest creatinine and urea, urine output of the 24 h, lowest PaO2/FiO2 with whether it was taken on
  invasive ventilation (categorical), lowest SpO2/FiO2.
- Age, sex and admission type as above.

Gaps per stay, same probe: PaO2/FiO2 56.5%, bilirubin 47.0%, urine output 28.6%, platelets 4.1%, creatinine and urea
3.3%, Glasgow 0.4%, mean pressure 0.0%.

## Donor tables, for E151

Rows complete on every measured column, cut from each table: `mimic-cardio-donor` (1,292 rows by the probe) and
`mimic-sofa-donor` (2,561 complete on the six components' raw values by the probe; the built table has more columns,
so its count will be lower). The builder's count stands.

## Names and files

`<base>/<base>_natural.csv` for the two natural tables; `<base>-donor/<base>-donor_00nan.csv` for the donors, with
E151's variants beside them. Base names carry a hyphen, never an underscore, because `DatasetSpec` splits on the
first one. Five-fold cross-validation (`--cv_folds 5`), so no split file. `categorical_columns/<base>.txt` in the
repository.

## Limits

- Deaths and discharges inside the observation window are not removed; the builder reports how many.
- One hospital, one cohort definition. The eICU copy on gorgona8 would give a second hospital; not in T50.
