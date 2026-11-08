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
