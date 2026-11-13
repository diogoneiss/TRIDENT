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
