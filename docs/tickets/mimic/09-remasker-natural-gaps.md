# ReMasker on the MIMIC-IV tables with natural gaps (E161, T53; pre-registered 2026-11-08 10:50 GMT-3)

This ticket was written and committed before the study's first scored cell.

- **Why:** E148 ([03](03-masked-cells-natural-gaps.md)) found the decoder better than the bar's best baseline and than
  TabPFN on hidden observed cells of the two MIMIC-IV tables with natural gaps. No deep baseline was beside it. E157–E160
  measured ReMasker (a masked-autoencoder imputer) only on tables with gaps the project cut. This is the only test against
  a deep baseline on gaps a clinician left.
- **The user's go:** "Execute d e c" (2026-11-08), on "D: o ReMasker nas duas tabelas naturais da MIMIC".
- **Disclosure:** E148's decoder, bar and TabPFN cells are on the record. ReMasker has no score on these tables.

## Design

- **Arm:** ReMasker, `remask="paper"`, E158's settings (600 epochs, batch 64), untuned, offline (`remasker_run.py`,
  nothing written to the store).
- **Cells:** E148's: `mimic-cardio_natural` and `mimic-sofa_natural` × seeds 42/7/13, five folds, the masked
  population (observed cells hidden at rate 0.2, natural gaps left as inputs). 6 pairs, 30 fits.
- **Method difference:** ReMasker has no categorical path. It is fitted on the numerical columns only (cardio 20 of 22,
  sofa 14 of 17), and its categorical cells are filled with the training mode (`baseline/remasker/categorical_mode` 1).
  Hence the primary is on the numerical cells.
- **Folds are the runs' own:** `recompute_folds` reproduced E148's mean/mode `rmse_num_z`, `mae_num_z` and
  `acc_cat` on mimic-cardio_natural s42, mimic-sofa_natural s13 and s7 (max |diff| 2e-16).
- **Smokes, off the record (nothing in the study directory):** a 3-epoch timing smoke gave 4.9 s an epoch on the cardio
  table and 1.8 s on the SOFA one, alone on gorgona8's GPU. A 2-epoch smoke wrote all six CSVs to the scratchpad. It
  asserted that ReMasker's `acc_cat` equals mean/mode's on every fold, so no true categorical value leaks. It also ran
  this ticket's reader end to end. Its numbers describe a 2-epoch model and are not read.
- **Machines (ADR 0035, split by variant):** medusa1 runs `mimic-cardio_natural`, three lanes over its two GPUs, beside
  another user's job. gorgona8 runs `mimic-sofa_natural`, three lanes. Outs `/scratch/diogoneiss/trident-2026-11-08-remasker-mimic`
  (medusa1) and `/scratch2/diogoneiss/trident-2026-11-08-remasker-mimic` (gorgona8). medusa1's CSVs are copied over with
  their SHA-256 checked before the read.
- **Cost:** 600 epochs is about 49 min a fold on the cardio table (4 h a seed) and 18 min on the SOFA one (1.5 h a seed),
  alone on a GPU. Three lanes sharing a GPU took about 1.3× the smoke on E159, so the expected end is about 4–6 h after
  launch for the cardio table and 2–3 h for the SOFA one.
- **Read:** once, with all 6 pairs in:
  `QUEUE_STATE=/scratch2/diogoneiss/trident-2026-11-06-mimic-e148 TRIDENT_DATASETS_ROOT=/scratch/diogoneiss/trident-data
  uv run --python 3.11 python -B scripts/experiments/remasker_mimic_report.py /scratch2/diogoneiss/trident-2026-11-08-remasker-mimic/csv`.

## Measures and rule (fixed now)

- **Primary,** per table: the decoder's `rmse_num_z` (E148's arm M) minus ReMasker's, on the same masked numerical
  cells, paired per (seed, fold), 15 replicates, plain 95% t interval, Holm over the two tables. **Better** when the
  adjusted p is under 0.05 and the mean is below 0; **worse** when it is under 0.05 and the mean is above 0; otherwise
  no different (E148's rule).
- **Not `impute_score`:** it mixes in the categorical cells, which ReMasker fills with the mode. It is printed beside
  the primary with that caveat.
- **Descriptive:** decoder − ReMasker on `mae_num_z`. ReMasker − the best fixed baseline per fold on `rmse_num_z`,
  meaning the lowest `rmse_num_z` among the bar's (`WEAK`) and the strong tier's baselines the run scored.
  ReMasker − TabPFN on `rmse_num_z` (E148's offline pass). Fit time.
- **What it leads to:** a sentence for the user beside E148's. No default, bar or tier change. One seed set, two views
  of one cohort, as E148.

## Outcome (2026-11-08 13:46 GMT-3)

All six pairs were in at 13:41 GMT-3: cardio on medusa1 at 12:05 and SOFA on gorgona8 at 13:41. No lane failed. The
SHA-256 of medusa1's three CSVs matched after the copy. Every row carries `remask_paper` 1 and `categorical_mode` 1.
Read once, at 13:43 GMT-3, with the command above. Unedited printout:

```
## mimic-cardio_natural
- rmse_num_z: decoder 0.8116, ReMasker 1.6203, TabPFN 0.8520
- mae_num_z: decoder 0.4513, ReMasker 1.0186, TabPFN 0.4511
- impute_score: decoder 0.8072, ReMasker 1.5317, TabPFN 0.8459
  decoder − ReMasker, mae_num_z (n=15): -0.5673 [-0.6450, -0.4896]
  decoder − ReMasker, impute_score (ReMasker's categorical cells mode-filled) (n=15): -0.7245 [-0.8747, -0.5742]
  ReMasker − best fixed baseline per fold, rmse_num_z (n=15): +0.8021 [+0.6403, +0.9639]
  ReMasker − TabPFN, rmse_num_z (n=15): +0.7683 [+0.6066, +0.9300]
  ReMasker fit 791 s a fold; bar by impute_score 0.8199
## mimic-sofa_natural
- rmse_num_z: decoder 0.8855, ReMasker 1.7605, TabPFN 0.8999
- mae_num_z: decoder 0.4925, ReMasker 1.1353, TabPFN 0.4665
- impute_score: decoder 0.8848, ReMasker 1.6283, TabPFN 0.8956
  decoder − ReMasker, mae_num_z (n=15): -0.6427 [-0.7610, -0.5244]
  decoder − ReMasker, impute_score (ReMasker's categorical cells mode-filled) (n=15): -0.7435 [-0.9466, -0.5404]
  ReMasker − best fixed baseline per fold, rmse_num_z (n=15): +0.8596 [+0.6296, +1.0896]
  ReMasker − TabPFN, rmse_num_z (n=15): +0.8606 [+0.6295, +1.0917]
  ReMasker fit 1944 s a fold; bar by impute_score 0.8984

PRIMARY mimic-cardio_natural, decoder − ReMasker, rmse_num_z (n=15): -0.8088 [-0.9697, -0.6479], Holm p 0.0000 -> better
PRIMARY mimic-sofa_natural, decoder − ReMasker, rmse_num_z (n=15): -0.8750 [-1.1056, -0.6444], Holm p 0.0000 -> better
```

- **Primary, by the rule:** the decoder is better than ReMasker on both tables (−0.809 on cardio, −0.875 on SOFA,
  Holm p < 0.0001).
- **Arm validity in doubt; no sentence drawn.** The pre-registered descriptive line puts ReMasker +0.80 and +0.86
  `rmse_num_z` above the best fixed baseline. Mean fill alone scores 1.01–1.04 on cardio and 1.00–1.02 on SOFA on
  these cells (E148's `raw_fold_metrics.csv`, `impute/masked/baseline/mean_mode/rmse_num_z`, per seed). ReMasker's 1.62
  and 1.76 are worse than the mean. A baseline worse than the mean says nothing about a competent deep baseline, so this
  result is not a claim that TRIDENT beats ReMasker on clinical gaps.
- **What was checked:** all 600 epochs ran (the loop in `src/training/remasker.py` has no early stop). Cardio's
  791 s a fold on medusa1's 5090s against 49 min in gorgona8's smoke is a speed difference, not a short run.
- **Next:** a diagnostic off the record (scaling against MIMIC's tails, per-column error against the mean). Any rerun
  with a changed arm is a new E-id, never a re-read of this one.

## Diagnostic after the read (2026-11-08 20:24 GMT-3)

Run after the Outcome, on the user's go to attach it ("sim", 2026-11-08). It explains the arm and does not test
anything: fold 1, seed 42, one table each, cardio on medusa1 and SOFA on gorgona8, from 13:50 GMT-3, finished 14:21
GMT-3. `scripts/experiments/remasker_mimic_diagnostic.py` refits ReMasker (paper rule) at 600 and at 60 epochs plus mean
fill on the fold's training split. It prints each numerical column's rmse over the training fold's standard deviation,
on the fold's masked test cells and on 20% of the training split's observed cells hidden after the fit. The committed
file is the scratchpad script that ran, with types added for the mypy gate; the logic is unchanged and it was not
rerun. Unedited printouts:

`mimic-sofa_natural`, s42, fold 1:

```
fit 600 epochs: 1710 s
fit 60 epochs: 177 s
## training cells hidden after fit (rmse / training sd)
                          n  rm600   rm60   mean
meanbp_min           1513.0  0.958  0.954  0.984
rate_dobutamine      1522.0  4.054  3.115  0.877
rate_dopamine        1532.0  1.389  1.105  0.884
rate_epinephrine     1532.0  2.579  2.490  1.002
rate_norepinephrine  1580.0  1.147  1.393  0.972
gcs_min              1649.0  2.361  1.563  1.035
platelet_min         1542.0  1.042  1.151  1.009
bilirubin_max         815.0  1.025  0.959  0.978
creatinine_max       1510.0  0.822  0.829  1.015
bun_max              1537.0  0.929  0.889  1.017
uo_24hr              1103.0  1.008  1.045  1.010
pf_ratio              648.0  0.873  0.864  0.979
sf_ratio             1585.0  1.250  1.233  1.000
age                  1540.0  1.695  1.837  0.983
## test masked cells (rmse / training sd)
                         n  rm600   rm60   mean
meanbp_min           424.0  1.030  1.016  1.085
rate_dobutamine      387.0  4.326  3.268  0.919
rate_dopamine        411.0  1.647  1.337  1.139
rate_epinephrine     413.0  3.026  2.894  0.982
rate_norepinephrine  460.0  1.267  1.484  1.081
gcs_min              371.0  2.438  1.635  1.133
platelet_min         359.0  1.087  1.192  1.018
bilirubin_max        227.0  1.147  1.105  1.101
creatinine_max       373.0  0.817  0.749  0.939
bun_max              398.0  0.927  0.889  0.999
uo_24hr              291.0  1.007  1.046  1.003
pf_ratio             187.0  0.898  0.914  1.069
sf_ratio             394.0  1.278  1.262  1.003
age                  426.0  1.730  1.907  0.999
cell-weighted mean: rm600 1.670, rm60 1.528, mean 1.032
DONE
```

`mimic-cardio_natural`, s42, fold 1:

```
fit 600 epochs: 816 s
fit 60 epochs: 74 s
## training cells hidden after fit (rmse / training sd)
                     n  rm600   rm60   mean
troponin_t       539.0  1.545  0.980  1.024
heart_rate      1498.0  0.955  0.946  1.000
abp_systolic     616.0  0.660  0.744  0.990
abp_diastolic    636.0  0.506  0.557  0.972
abp_mean         662.0  0.675  0.700  1.026
nibp_systolic   1424.0  0.549  0.616  0.991
nibp_diastolic  1472.0  0.393  0.409  1.026
nibp_mean       1447.0  0.471  0.411  0.976
spo2            1585.0  1.077  1.023  1.032
potassium       1587.0  1.255  0.995  1.009
magnesium       1515.0  0.945  0.949  0.953
hemoglobin      1542.0  0.551  0.508  1.014
hematocrit      1547.0  0.555  0.492  1.007
sodium          1570.0  0.980  1.002  1.014
lactate         1111.0  1.628  1.027  1.139
dobutamine      1606.0  2.750  1.060  0.967
dopamine        1555.0  1.833  0.966  1.018
epinephrine     1476.0  3.542  1.468  1.132
norepinephrine  1519.0  1.746  1.252  0.998
age             1575.0  0.912  0.918  1.007
## test masked cells (rmse / training sd)
                    n  rm600   rm60   mean
troponin_t      133.0  1.900  1.105  1.146
heart_rate      372.0  0.972  0.922  0.971
abp_systolic    169.0  0.706  0.741  0.960
abp_diastolic   188.0  0.581  0.577  1.050
abp_mean        163.0  0.710  0.720  1.062
nibp_systolic   363.0  0.562  0.616  1.029
nibp_diastolic  367.0  0.399  0.456  1.085
nibp_mean       372.0  0.508  0.449  1.118
spo2            398.0  1.116  1.003  0.998
potassium       359.0  1.300  0.977  0.997
magnesium       401.0  0.999  1.010  1.007
hemoglobin      399.0  0.494  0.464  0.949
hematocrit      384.0  0.524  0.487  0.960
sodium          425.0  0.964  0.979  0.995
lactate         290.0  1.822  1.088  1.142
dobutamine      399.0  3.821  1.907  1.921
dopamine        403.0  2.048  1.255  1.293
epinephrine     399.0  3.877  1.436  0.548
norepinephrine  378.0  1.915  1.470  1.026
age             396.0  0.955  0.937  1.023
cell-weighted mean: rm600 1.353, rm60 0.951, mean 1.064
DONE
```

- **The model works where columns are correlated.** On the arterial and non-invasive pressures, haemoglobin and
  haematocrit, ReMasker scores 0.39–0.71 against mean fill's 0.95–1.12.
- **The error sits in columns whose values pile at an end of the min-max scale.** These are the four vasopressor
  doses, mostly zero (median at 0.0 of the span), `gcs_min` (median at the top), `troponin_t` and `lactate` (medians
  at 0.002 and 0.067). There ReMasker's 600-epoch fit reaches 1.3–4.3, mostly far above mean fill. The scale and the
  bounded output head (`tanh / 2 + 0.5`) are the reference code's.
- **Not over-fitting:** the training cells hidden after the fit score as badly as the test cells. At 60 epochs the
  cardio fold scores 0.951, under the mean's 1.064, but the SOFA fold still scores 1.528 against 1.032.
- **The OpenML tables (checked after the diagnostic, 2026-11-08 20:30 GMT-3).** E157–E160 measured ReMasker as
  published, so their results stand as read. Two of their tables have such columns too: 19 of `biodeg`'s 41 numerical
  columns and 16 of `kc2`'s 21 have the median within 2% of an end of the span. pendigits, letter and vehicle have
  none. In E158, ReMasker's induced `impute_score` (mean fill = 1) was 0.873 and 1.035 on biodeg 40nan and 80nan,
  against 0.43–0.91 on the tables without them. On kc2 it was 0.61 and 0.745, so an edge-piled column alone does
  not make ReMasker fail.

## The thesis sentence (accepted 2026-11-08 20:24 GMT-3)

On the user's decision ("gosto da decisão possivel 2, isso ir pra tese", then "sim" to the wording):

> Nas tabelas da MIMIC-IV, o ReMasker como publicado fica abaixo até do preenchimento pela média; o erro se concentra nas colunas cujos valores se acumulam no limite da escala, como as doses de vasopressor.

No sentence is drawn on TRIDENT's lead over ReMasker here.
