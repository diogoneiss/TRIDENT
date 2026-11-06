# C15's positive-control gate on the SOFA table's natural gaps (E152; pre-registered 2026-11-06)

This ticket was written and committed before the study's first cell.

- **Why:** the thesis's main question (RQ2, H2: per-column `[NULL]` tokens predict as well as imputation plus an
  indicator when the gaps carry information) has had no live test since C15 stopped at its gate G1 (E124, E128;
  T45). G1 asks whether the pipeline can see informative missingness at all, and on the injected noise controls the
  indicator arm did not beat the plain fill on 2 of 3. [E150](05-downstream-mortality.md) found a natural case: on
  `mimic-sofa_natural`, a logistic regression with one indicator per column beats one without them by +0.0087 macro-F1
  (Holm p 0.0012). This runs G1, as E128 ran it, on that table.
- **The user's go:** "vou na sua sugestão, vá" (2026-11-06, after 12:21 GMT-3), on the proposal to pre-register G1 on
  the SOFA table and run it when E149 ends.
- **Disclosure:** E150's result and [ticket 02](02-tables.md)'s whole-cohort death rates by gap are known: bilirubin
  22.0% where measured, 10.3% where not. No TRIDENT classification cell has run on these tables.

## Design (fixed before the first cell)

- **Table:** `mimic-sofa-num_natural`, `mimic-sofa_natural` with its three categorical columns made numerical, since
  the C15 arms take numerical tables only (`src/training/c15_inputs.py`): `gender` 0/1 (F/M), `pf_on_vent` 0/1 (no/yes) with
  its gaps kept, `admission_type` one-hot over its 9 values. Same 11,082 rows, label and gaps otherwise. Written by
  `datasets/make_mimic_numeric.py` into the study's own data root (`TRIDENT_DATASETS_ROOT` set to it for these cells
  only), outside the shared root and its manifest.
- **Arms (C15's, unchanged):** A1, the shipped `[NULL]`; A2, missForest's fill plus a gap indicator read by each
  column's value MLP; A3, missForest's fill, no indicator. The fills are fitted on each fold's training rows.
- **Configuration:** the table's resolved classification configuration (no promoted file), `PRETRAIN_OBJECTIVE
  embedding_normalized`, the shipped checkpoint rule, `--cv_folds 5`, MLflow off, as E124 and E128.
- **Cells:** A1, A2, A3 × seeds 42/7/13: 9 cells, 45 folds. Runner and reader: `scripts/experiments/sofa_g1.py`.
- **Machine (ADR 0035):** all 9 on gorgona8, two queues, started when E149's cells end (medusa1's GPUs are taken by
  another user's job). Cost not measured; the smoke's epoch times are recorded before the first cell.

## Rule, fixed now (E128's, one table)

**G1 passes if both hold:**
1. **A2 beats A3 on test log loss:** the per-cell relative difference (LL_A2 − LL_A3) / LL_A3 over 15 cells, paired
   by seed and fold, its 95% t interval with the Nadeau–Bengio correction (1/15 + 0.286) below 0;
2. **A1 reads the gap with the cohort's sign:** E124's direct contrast on `bilirubin_max` (logit P(death) with the
   gap as `[NULL]` minus the logit of the mean P(death) over 20 MICE draws of that cell given the row, the row's
   other gaps left `[NULL]`, averaged over the fold's test rows with that gap), over 15 cells, its corrected interval
   below 0. Bilirubin is fixed now as the one column: the largest gap share among the single numerical columns
   whose gap tracks death in ticket 02 (PaO2/FiO2's gap comes with its ventilation flag's, so its contrast would
   leave half the gap visible).

Plain intervals, per-seed means and macro-F1 are printed beside; they decide nothing. **Read only with all 9 cells
done.**

**What each reading leads to:**
- **passes:** H2's own test on this table (A1 against A2, non-inferiority at −0.01 macro-F1 and +2% relative log
  loss, C15's margins) is pre-registered on fresh seeds, for the user's go.
- **fails:** RQ2 stops at the gate on natural gaps too; recorded in T45 and T50.

**Not here:** `mimic-cardio`, where E150's indicators did not help (a negative control, 9 more cells); A4's MICE
draws; any TabPFN arm.

## Launch note (2026-11-06 13:19 GMT-3, before the first cell)

- **Smoke** (3 pre-training and 3 fine-tuning epochs, seed 42, five folds, into `STATE/smoke`, beside E149's two
  queues): A2 453 s, A1 31 s, both to the end with five readouts each; E124's contrast scorer ran on one smoke fold
  of A1 with `bilirubin_max` and returned a finite number, not read. So about 1 s per epoch and fold: about 37
  minutes an A1 cell at 300 + 150 epochs, and about 7 minutes more for A2's and A3's missForest fits.
- **Queues:** three, one seed each, instead of the two written above, so the 9 cells take about 2.3 h rather than 3.5;
  started by `scripts/experiments/run_sofa_g1_chain.sh` (run from a copy in the state dir), which waits until no
  E149 cell is running. Nothing else changes.

## Outcome (2026-11-06 16:34 GMT-3): G1 fails

All 9 cells ended without a failure (the chain started its queues when E149's cells ended; the last cell ended 16:16
GMT-3). The reader's printout, unedited (the loader's label-mapping lines left out):

```
## G1 on mimic-sofa-num_natural (E152, seeds 42/7/13 x 5 folds): (1) A2 − A3 relative test log loss below 0; (2) A1's bilirubin_max contrast below 0; Nadeau–Bengio corrected 95% t intervals over 15 cells; passes if both hold
- A2 − A3 -0.0153 [-0.1066, +0.0760] (plain [-0.0550, +0.0244])
- A1 contrast on bilirubin_max -0.651 [-0.851, -0.452] logit (plain [-0.738, -0.564])
- per seed: s42 A2 − A3 +0.0311, contrast -0.644; s7 A2 − A3 -0.0399, contrast -0.695; s13 A2 − A3 -0.0371, contrast -0.615
- macro-F1 (descriptive): A1 0.6568, A2 0.6684, A3 0.6664

Primary (G1): fails
```

- **By the rule, G1 fails:** condition (2) holds, condition (1) does not. A1 reads the bilirubin gap with the cohort's
  sign on every seed (−0.651 logit, corrected interval [−0.851, −0.452]). A2 does not beat A3 on test log loss: −0.0153,
  corrected interval [−0.1066, +0.0760], the plain one across 0 too, and the seeds disagree (s42 +0.0311, s7 and s13
  about −0.04).
- **The same split as E128:** the `[NULL]` arm sees the gap; missForest's fill plus an indicator is not shown to use
  it better than the fill alone. E150's logistic regression did gain from the indicators on this table, so the gap
  carries information that this pipeline's indicator arm does not turn into a measurable log-loss gain at 15 cells.
- **Macro-F1, descriptive, decides nothing:** A1 0.6568, A2 0.6684, A3 0.6664.
- **Leads to, as written:** RQ2 stops at the gate on natural gaps too; recorded in T45 and T50. H2's A1-against-A2
  test is not pre-registered.
