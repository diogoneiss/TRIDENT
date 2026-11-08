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
