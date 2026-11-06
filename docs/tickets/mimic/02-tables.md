# Two tables from the kg-impute cohort (T50, stage 0; written 2026-11-05)

The tables E150 to E152 run on. Built by `datasets/build_mimic.py` from `MIMIC_ROOT` and the files named below,
written under `TRIDENT_DATASETS_ROOT` ([01](01-external-datasets-root.md)):

    uv run --no-project --with h5py --with pandas python datasets/build_mimic.py

(the training environment has no h5py, and the builder needs nothing else of it). **Built 2026-11-05 on gorgona8**;
every count below is the builder's, kept in `/scratch/diogoneiss/trident-data/build_report.txt`.

Both tables hold the same 11,082 ICU stays: the suspected-infection cohort of `calculo-sistemas/cardiokg-impute`
(antibiotic prescribed and given, culture ordered, first stay of each patient). They are two views of one
population, not two replications.

**Label, both tables:** death in hospital (`hosp/admissions.hospital_expire_flag`), column `class`: 16.5%. The kg-impute paper's own downstream label, circulatory failure, is a function of lactate, MAP and
the vasopressors, columns of these tables, so it is not a target here.

## `mimic-cardio`

From `cardiokg-impute/data/tensors/tensors.h5` (19 variables, 30 one-hour bins after ICU admission).

- 15 measured variables, each the mean of its observed bins, a gap when never observed in the 30 h.
- The tensors hold the chart as typed, entry errors included (a potassium of 999,999, a negative arterial
  pressure). Each variable's observed values are clipped to their 0.5th and 99.5th percentiles before the mean, the
  rule kg-impute's own normaliser applies, here over the whole cohort. The SOFA files' hourly values get the same
  clip (a mean pressure of −22,767 is a sentinel); a drug's rate is clipped at the top only.
- 4 vasopressors (dobutamine, dopamine, epinephrine, norepinephrine): the highest rate, 0 when never infused. A drug
  not given is a dose of zero, not a gap.
- Age, sex (`hosp/patients`) and admission type (`hosp/admissions`); sex and admission type categorical.

Gaps per stay (share of stays with no observation of the variable):

| variable | gap |
|---|---|
| troponin T | 65.8% |
| arterial pressure, systolic and diastolic | 60.3% |
| arterial pressure, mean | 57.4% |
| lactate | 28.5% |
| non-invasive pressure (three) | 5.5–5.6% |
| magnesium | 2.3% |
| heart rate, SpO2, potassium, haemoglobin, haematocrit, sodium | ≤ 0.6% |

22 features; the 14 columns with gaps miss 21.0% of their cells on average. The gaps are informative: deaths are
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

17 features (`meanbp_min`, `rate_<drug>` ×4, `gcs_min`, `platelet_min`, `bilirubin_max`, `creatinine_max`,
`bun_max`, `uo_24hr`, `pf_ratio`, `pf_on_vent`, `sf_ratio`, age, sex, admission type). Gaps per stay: PaO2/FiO2 and
its ventilation flag 56.5%, bilirubin 47.0%, urine output 28.6%, platelets 4.1%, creatinine and urea 3.3%, Glasgow
0.4%, SpO2/FiO2 0.1%, mean pressure 3 stays. Deaths are 22.2% where a PaO2/FiO2 was taken and 12.1% where not, 22.0%
against 10.3% for bilirubin; urine output runs the other way (15.2% against 19.7%), its gap being mostly a stay
too short to have 22 hours of it.

## Donor tables, for E151

Rows with every feature present, cut from each table: `mimic-cardio-donor` (1,292 rows, 33.5% deaths) and
`mimic-sofa-donor` (2,143 rows, 25.4% deaths). Both pass E151's floor of 1,000, and both are sicker than the cohort
(16.5%).

## Names and files

`<base>/<base>_natural.csv` for the two natural tables and the toy; `<base>-donor/<base>-donor_00nan.csv` for the
donors, with E151's variants beside them. Base names carry a hyphen, never an underscore, because `DatasetSpec` splits on the
first one.

**Row index.** Every column of a table but the label is a feature (`declared_column_types`), so the identifiers do
not sit in the table. Each table has a row-aligned index beside it, `<base>/<base>_index.csv`: `stay_id`,
`subject_id` and `hadm_id` of row *i* on line *i*. It is what joins a row back to MIMIC-IV, the cardio view to the
SOFA view, a donor row to its stay and to the stay whose pattern it took (E151), and a ledger's `row` to a patient.

Five-fold cross-validation (`--cv_folds 5`), so no split file. `categorical_columns/<base>.txt` in the
repository.

## Limits

- Deaths and discharges inside the observation window are not removed: 1,715 stays left the ICU within 24 h and
  333 patients died within 30 h of admission.
- The clip limits are percentiles of the whole cohort, test rows included: a cleaning step that never sees the
  label, stated because it is fitted outside the folds.
- One hospital, one cohort definition. The eICU copy on gorgona8 would give a second hospital; not in T50.
