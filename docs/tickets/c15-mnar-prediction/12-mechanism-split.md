# Does A1 close on A2 where the gap depends on the outcome, not on the value? MIMIC-IV complete tables with known mechanisms (E156, T52; pre-registered 2026-11-06)

This ticket was written and committed before the study's first cell and before its first G2 cell.

- **Why:** [E155](11-h2-external-gate.md)'s secondary read *gains* (+0.0155 on the plain interval; the corrected one
  crossed 0) and split by mechanism. On the three outcome-dependent files of P, A1 (`[NULL]`) was at parity with A2
  (missForest + indicator) while trailing on their MCAR siblings; on the two probit (self-masking) files it trailed
  more than under MCAR. That pattern was found in E155's data, so it is confirmed here on tables E155 never saw, and on
  the correction E155's primaries used.
- **The user's go:** "Vamos fazer o 1 e 3 até amanhã a tarde" (2026-11-06), on the proposal "confirm the pattern the
  secondary showed, a new study with other tables and draws", in its semi-synthetic MIMIC-IV form.
- **Why semi-synthetic:** the claim is about the mechanism, so the mechanism must be known, and a table needs its MCAR
  sibling. The benchmark has no binary numerical table left that P did not use (C15's arms take numerical tables
  only, the outcome family binary labels only), so the tables are cut from MIMIC-IV: real clinical columns, real
  mortality, injected gaps.

## Disclosure (what is known before the first cell)

- E155's per-file means: A1 − A2 macro-F1 +0.000, +0.020, −0.000 on the outcome files, −0.056 and −0.098 on the probit
  ones, −0.012 to −0.078 on the MCAR siblings. E155's SOFA and cardio cells used the natural tables, whose columns
  overlap these; their natural gaps are not used here.
- The odds-3 outcome mechanism nearly separates the classes through the gaps: here deaths lose 90% of their cells,
  survivors 54% (the generator's printout below). Under that mechanism A2's indicator and A1's `[NULL]` both see the
  pattern; E155's parity may be a ceiling where both read it fully.
- No cell, G2 cell or score has been computed on these files.

## Data (written 2026-11-06, before this ticket's commit; the scripts are the record)

- **Complete tables** (`datasets/make_mimic_complete.py`, from `/scratch/diogoneiss/trident-data`): of each natural
  table, the numerical columns other than `gender` and `admission_type` with at most 5% natural gaps and a most
  frequent value covering at most half the rows (the four vasopressor rates, 73–99% at one value, and `gcs_min`, 60%,
  fail it: self-masking on a column that is mostly one value is nearly MCAR), and the rows complete on them.
  - `mimic-sofa-semi`: 10,553 rows; `meanbp_min`, `platelet_min`, `creatinine_max`, `bun_max`, `sf_ratio`, `age`;
    16.4% deaths.
  - `mimic-cardio-semi`: 10,795 rows; `heart_rate`, `spo2`, `potassium`, `magnesium`, `hemoglobin`, `hematocrit`,
    `sodium`, `age`; 16.6% deaths.
  - Both are cut from the same 11,082 stays, so the two tables share most rows, `age` and the label: they are two
    feature sets on one cohort, not two cohorts.
- **Mechanisms** (`datasets/mnar_family.py --root`, data seed **5858**, 60% of each column's cells): `outcome --odds 3`,
  `probit --slope 0.5`, `mcar`. Six files under `/scratch2/diogoneiss/trident-2026-11-06-mechanism-split/data`:
  `{mimic-sofa-semi,mimic-cardio-semi}_{outcome3p0,probit0p5,mcar}_s5858_60nan`. The generator's integrity lines:

```text
mimic-sofa-semi_outcome3p0_s5858_60nan: 0.600 of feature cells missing; by class {0: 0.5403015531118921, 1: 0.9041570438799076}; hidden − observed shift +0.012
mimic-cardio-semi_outcome3p0_s5858_60nan: 0.600 of feature cells missing; by class {0: 0.5395423747639676, 1: 0.9037388392857143}; hidden − observed shift +0.021
mimic-sofa-semi_probit0p5_s5858_60nan: 0.600 of feature cells missing; by class {0: 0.6006310697955636, 1: 0.5969014626635873}; hidden − observed shift +0.501
mimic-cardio-semi_probit0p5_s5858_60nan: 0.600 of feature cells missing; by class {0: 0.59914750638676, 1: 0.6042829241071429}; hidden − observed shift +0.551
mimic-sofa-semi_mcar_s5858_60nan: 0.600 of feature cells missing; by class {0: 0.6014246306163322, 1: 0.5928598922247883}; hidden − observed shift +0.002
mimic-cardio-semi_mcar_s5858_60nan: 0.600 of feature cells missing; by class {0: 0.6000083305564812, 1: 0.5999581473214286}; hidden − observed shift -0.005
```

## Design (fixed now)

- **Arms:** C15's A1 and A2, unchanged, E155's configuration (`embedding_normalized`, the table's resolved
  classification defaults, five folds, MLflow off). Runner `scripts/experiments/mechanism_split.py train`.
- **Seeds:** **2525/4747/6969**, in no log entry, ticket or script before this one. A file's folds depend on the label
  and the seed only, so the three files of a table share their folds and the contrasts pair per (seed, fold).
- **Cells:** 6 files × 2 arms × 3 seeds = **36**, all on gorgona8 (ADR 0035, the rule that keeps every arm of a
  (file, seed) pair on one machine), three queues, both arms of a pair in one queue; chain
  `scripts/experiments/run_e156_chain.sh`, run from a copy in the state dir.
- **G2 (E123's rule, out of fold, seeds 42/7/13, CPU):** `mechanism_split.py --g2`, 90 cells: the indicators must cut
  an external model's out-of-fold log loss by at least 2%, the 95% t interval above 0. It says which files carry
  information in their gaps, outside TRIDENT.
- **Cost, estimated before the first cell:** E155's MIMIC cells took 17–22 min on three queues with 25–30 columns;
  these have 6–8, so at most about 4 h for 36 cells.
- **Not here:** fresh P draws (a replication on E155's tables; left out so ReMasker's study, E157, gets the GPU by
  morning); H2's non-inferiority call (printed per file only as A1 − A2 means).

## Rule, fixed now

Each contrast is a macro-F1 difference of differences per (seed, fold), 15 cells, its mean with the
**Nadeau–Bengio corrected 95% t interval** (variance × (1/15 + 0.286), 14 df), the plain interval beside. A call is
**gains** if the interval is above 0, **loses** if below 0, **holds** if inside ±0.01, otherwise **no verdict**. Pooled
means the mean over the tables per cell.

1. **Primary:** (A1 − A2) on `outcome3p0` minus (A1 − A2) on `mcar`, pooled over the tables whose outcome file passes
   G2. **Predicted: gains.** Confirmed if *gains*; refuted if *loses* or *holds*; otherwise not confirmed. If no outcome
   file passes G2, the primary is not read.
2. **Secondary 1:** the same for `probit0p5` minus `mcar`, pooled over both tables. Predicted: not *gains*. If a
   table's probit file fails G2, that table's probit file is read as a non-informative control, said beside the call.
3. **Secondary 2, the split:** (A1 − A2) on `outcome3p0` minus on `probit0p5`, pooled as the primary. Predicted:
   *gains*.

**Printed beside, decide nothing:** each table's own contrasts, per-seed means, A1 − A2 per file in macro-F1 and
absolute log loss, the G2 drops.

**Read once,** with `mechanism_split.py --read`, which refuses until all 36 cells and 90 G2 cells are in.

**What each reading leads to:**
- **Confirmed (primary gains, with secondary 2 gains):** E155's split replicates on clinical tables: A1 trails A2 under
  MCAR and closes the gap where missingness depends on the outcome, not where it depends on the value. Recorded in T52
  for the thesis sentence on H2, the user's call.
- **Primary confirmed, secondary 2 not:** A1 gains where the gap depends on the outcome; whether self-masking differs is
  open.
- **Not confirmed or refuted:** E155's secondary does not replicate; the H2 sentence keeps only E155's inconclusive
  primaries.
- No ADR changes either way.

## Limits, stated now

- The two tables share a cohort, so their pooling is not two independent replications.
- Odds 3 is a strong mechanism; a parity there may be a ceiling (disclosure). One odds level only.
- The vasopressor rates and GCS are out by the mode rule, so the tables are small (6 and 8 columns).
