# A MIMIC-IV table past TabPFN's context: does the decoder hold against TabPFN v2 there? (E176, T57; pre-registered 2026-11-12 23:49 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** every MIMIC-IV table so far sits on kg-impute's cohort of 11,082 stays, where TabPFN v2
  (`src/training/tabpfn_baseline.py`, ADR 0028: v2 checkpoints, 8 estimators, column-wise) sees nearly every training
  row: its context is capped at `CONTEXT_ROWS = 10_000` observed rows per column, a fixed-seed subsample above that.
  TRIDENT trains on every row. This study builds a clinical table several times that cap and asks whether the decoder
  is better than TabPFN v2 where TabPFN cannot use the whole table.
- **The user's go:** "gosto muito da ideia do 2, faça-o primeiro" with "o plano do 1 no caso seria desenhar um novo
  coorte auxiliar pra ele" (2026-11-12), and "inclusive a medusa1 esta com gpu livre, creio que voce pode jogar
  algumas filas lá" (2026-11-12, about 23:15 GMT-3). The table's design below was reported to the user at 23:25
  GMT-3 as the plan to follow barring objection; no answer had come when this was committed.
- **Disclosure:**
  - On the 11,082-stay tables the decoder read better than TabPFN on `mimic-cardio_natural` (−0.0387) and
    `mimic-sofa_natural` (−0.0108) in E148, and under transplanted patterns led on cardio and trailed on SOFA (E149).
  - `electricity` (45,312 rows) is in the benchmark with TabPFN under the same cap; its standing was not re-read for
    this ticket.
  - The table was built and its gaps counted before this ticket (below). One timing cell and one timing TabPFN pass
    ran on `mimic-wide-donor_tmatch_20nan` at seed 999, a seed outside the study, with MLflow off; only their clocks
    were read, no score.
  - The claim is about TabPFN v2 as ADR 0028 pins it. The installed package's v2.5 configuration accepts 50,000
    samples, above both training folds here; this study says nothing about it.

## The table (built 2026-11-12 23:30 GMT-3, commit 8566da7)

`datasets/build_mimic.py --wide`, from the raw MIMIC-IV 3.1 chart and lab events on gorgona8
(`/scratch2/pedrobacelar/mimic/physionet.org/files/mimiciv/3.1`), with kg-impute's item map (`variables_phys.csv`).

- **`mimic-wide_natural`:** every patient's first ICU stay, 65,366 rows, deaths 10.8% (`class`). 23 measured
  variables, each the mean of its hourly means over the stay's first 30 h, clipped to its 0.5th and 99.5th
  percentiles as `cardio()` does, plus age, gender and admission type. The 23 are the mapped variables that under 20%
  of first stays lack: heart rate, SpO2, the three non-invasive pressures, sodium, potassium, chloride, bicarbonate,
  anion gap, creatinine, glucose, calcium, magnesium, hemoglobin, hematocrit, platelets, RBC, MCV, MCH, MCHC, INR,
  PTT. 4.2% of their cells are gaps (PTT 16.4%, INR 15.8%, calcium 8.9%, the pressures 6.2%, magnesium 4.6%).
- **`mimic-wide-donor_00nan`:** its 46,501 complete rows. `datasets/make_mimic_variants.py --bases mimic-wide` cut
  `tmatch`, `trand` and `mcar` at 12.6% of feature cells (label `20nan`): 152,057 gaps each, 177 distinct row
  patterns from 16,175 incomplete stays; INR 56.0%, PTT 58.4%, calcium 29.3%, the pressures 21%.
- **Self-check, passed:** on the kg-impute cohort the raw events reproduce `mimic-cardio_natural` on all 15 shared
  columns, gap for gap (100.0%) and value for value (median relative difference 0.0000).
- **The cap binds on every column:** the training fold is 0.7 of the rows, 45,756 on the natural table and 32,550 on
  the donor table; the sparsest column under `tmatch` (PTT) still has about 13,500 observed training rows.
- Identifiers are in the row-aligned `_index.csv` files, as for the other tables; hashes in
  `datasets/external_tables.sha256`, checked on medusa1.
- **Why 23 columns:** adding lactate, pH, pCO2 and tCO2 leaves 22,731 complete rows, a donor training fold of 15,911,
  where the cap barely binds.

## Design

- **Cells:** `mimic-wide_natural` under E148's arm `M` (`--decode_checkpoint masked`; scored on the masked
  population, observed cells hidden from the test rows), `mimic-wide-donor_tmatch_20nan` and
  `mimic-wide-donor_mcar_20nan` under E149's arm `D` (the imputation defaults; readout K on the induced cells).
  Seeds 42, 7, 13; 9 runs, five folds each, runner `scripts/experiments/mimic_wide_run.py`, tag
  `mimic-wide-2026-11-12`, state `/scratch/diogoneiss/trident-2026-11-12-mimic-wide`.
- **TabPFN v2:** `scripts/experiments/tabpfn_run.py <state>/tabpfn <variant> 42 7 13` on the same folds and cells,
  offline, nothing written to the store.
- **Machine (ADR 0035, the split of studies between servers):** every run and the TabPFN pass on medusa1, the leader;
  two queues of TRIDENT, one a GPU, and TabPFN's pass on GPU 1 beside the second.
- **Cost:** the timing cell's first fold on the donor table took 16 minutes on an RTX 5090 (3 of pre-training, 6 of
  decoding, the rest baselines and scoring), so about 80 minutes a donor run and about 2 h a natural one: 14 GPU-hours,
  7 to 8 h of wall clock in two queues. The timing TabPFN pass took 428 s for five folds: about 1.5 h for the nine files,
  beside the second queue. Ending on the morning of 2026-11-13.
- **Not in this study:** the deployable hybrid (ADR 0031 decision 6) on this table; an arm of TRIDENT on a
  10,000-row training subsample, which would separate "more rows" from "the model" but needs a training flag in
  `src/`, not authorised; `trand`; the bigger encoder of ADR 0018.

## Measures and rule (fixed now)

`impute_score` per fold, lower is better. Per variant, the 15 (seed, fold) differences decoder − TabPFN, read by the
project's fold-pair rule (95% t interval, every seed agreeing), equivalence margin 1% of TabPFN's mean score.

- **Primaries:** `mimic-wide_natural` and `mimic-wide-donor_tmatch_20nan`.
- **Supported** if the decoder is worse than TabPFN on neither primary and better on at least one; **refuted** if
  worse on either; **no verdict** otherwise.
- **Secondary:** the same contrast on `mimic-wide-donor_mcar_20nan`; decoder − the bar's best baseline on the three
  variants; TabPFN's fit seconds a fold.
- Three seeds, fixed; no extension.
- **Read** once with the 9 runs and the 9 TabPFN files in, on medusa1:
  `QUEUE_STATE=/scratch/diogoneiss/trident-2026-11-12-mimic-wide TRIDENT_DATASETS_ROOT=/scratch/diogoneiss/trident-data uv run --python 3.11 python -B scripts/experiments/mimic_report.py e176`
- **What it leads to:** a thesis sentence on the decoder against TabPFN v2 on a clinical table past its context, and
  whether the hybrid is worth running there; both the user's call.

## Outcome (2026-11-13 07:23 GMT-3): supported

The 9 runs finished between 01:34 and 07:19 GMT-3 on 2026-11-13, none failed, and the 9 TabPFN files were in by then.
A first read at 07:22 GMT-3 (f8586bd) stopped on a missing column name (TabPFN's fit seconds are keyed under the
population) before printing any score; the key was fixed (5340d08) and the study read once, on medusa1. The printout,
unedited (its first two lines are the commit and the clock):

```
5340d08 
read 2026-11-13 07:22 GMT-3
mimic-wide_natural (masked cells): decoder 0.6652, TabPFN 0.7644 (fit 19 s a fold), bar 0.6887
  decoder - TabPFN: -0.0991 [-0.1001, -0.0982], by seed -0.0994, -0.0981, -0.0999 -> better
  decoder - bar: -0.0234 [-0.0242, -0.0226], by seed -0.0234, -0.0230, -0.0238 -> better
mimic-wide-donor_tmatch_20nan (induced cells): decoder 0.9086, TabPFN 0.9662 (fit 25 s a fold), bar 0.9352
  decoder - TabPFN: -0.0577 [-0.0592, -0.0562], by seed -0.0582, -0.0576, -0.0572 -> better
  decoder - bar: -0.0266 [-0.0281, -0.0252], by seed -0.0272, -0.0254, -0.0273 -> better
mimic-wide-donor_mcar_20nan (induced cells): decoder 0.7171, TabPFN 0.7216 (fit 26 s a fold), bar 0.7303
  decoder - TabPFN: -0.0044 [-0.0056, -0.0032], by seed -0.0046, -0.0046, -0.0041 -> better
  decoder - bar: -0.0132 [-0.0137, -0.0127], by seed -0.0133, -0.0130, -0.0133 -> better
PRIMARIES (natural, tmatch) against TabPFN: ['better', 'better'] -> supported (fold-pair t, seed agreement, margin 1% of TabPFN's mean)
```

- **Primaries:** the decoder is better than TabPFN v2 on both, by −0.0991 on the natural table (13% of TabPFN's
  score) and −0.0577 under the transplanted patterns (6%), every seed agreeing. Supported.
- **Secondary:** under MCAR at the same gap counts the decoder is also called better, by −0.0044, which is under the
  1% margin (0.0072): a difference too small to matter by the ticket's own measure. The decoder is better than the bar
  on all three variants.
- **What this does not show:** that the lead comes from the row cap. TabPFN v2 scores worse than the bar's best
  baseline on the natural table (0.7644 against 0.6887) and on `tmatch` (0.9662 against 0.9352), and is level with the
  decoder under MCAR, where it has the same 10,000-row context. So the lead tracks the real gap patterns, not the
  table's size; the study has no arm that separates the two (the 10,000-row TRIDENT arm was left out). Why TabPFN
  falls below the bar on real patterns was not examined.
- **Scope:** TabPFN v2 as ADR 0028 pins it; one table, seeds 42/7/13.
- **What it leads to:** the thesis sentence and whether to run the hybrid or the 10,000-row arm here are the user's
  call. No ADR changes.

