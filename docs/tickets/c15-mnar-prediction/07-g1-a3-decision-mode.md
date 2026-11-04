# C15's gate G1 again: A2 against A3 in decision mode (E128; pre-registered 2026-11-04)

This ticket was written and committed before the gate's first cell (07:51 GMT-3).

- **Why:** E124's G1 ([03](03-stage1.md)) failed, 1 of 3. Its condition 1 compared A2 (missForest + indicator) with
  A4 (MICE, no indicator), which differ in the imputer and in the number of models (A4's prediction is the mean of five
  separately trained models) as well as in the indicator. The protocol's Open decision 14a had recommended A3
  (missForest, no indicator) for that condition; the plan kept A4. Five cells under the Nadeau–Bengio factor
  (1/5 + 0.286) left biodeg's and kc2's intervals wide.
- **A post hoc diagnostic, read after E124's failure, descriptive (2026-11-04):** on E124's seed-42 cells, averaging
  A4's five models lowers its test log loss by about 10% on biodeg and spambase and 4% on kc2. Against a single A4
  model, A2 − A4 is −0.106 [−0.261, +0.050] on biodeg, +0.100 [−0.287, +0.486] on kc2, −0.399 [−0.438, −0.360] on
  spambase (the gate's own numbers reproduce exactly). On kc2, A2 trails on 3 of 5 folds, fold 4 most (0.618 against
  0.385). This motivated the choice; it decides nothing.
- **A deviation, decided after E124's result was seen:** the user chose it ("vamos fazer 4. Combinação: A2 contra A3 no
  modo de decisão", 2026-11-04) among four options (accept the stop; G1 with A4 in decision mode; A2 against A3 in
  short mode; this). E124's G1 stays recorded as a failure.
- **Binding now:** if this G1 fails, C15 stops for good before Stage 2; no third form of G1 is tried, and E125–E127
  stay withdrawn. If it passes, Stages 2 and 3 are re-registered in new tickets (E125's and E126's designs) and go to
  the user before any of their cells.
- **Code:** `c15_run.py` stage `1c` (`C15_STAGE=1c`), `c15_report.py --g1c`; the contrast takes the fold's seed (it
  was fixed at 42). No change to `src/`. A3 has not run before: its path is A2's without the indicator columns
  (`fold_dataset`, unit-tested in `tests/unit/test_c15_inputs.py`), the encoder A4's (no indicator channel).

## Design (fixed before the first cell)

- **Controls:** F5's three, `{biodeg,kc2,spambase}_noise3p0_s2060_60nan`; data md5 equal on both machines (07:50 GMT-3).
- **Cells:** seeds 42/7/13 × `--cv_folds 5`, E124's configuration (each variant's resolved classification
  configuration, `PRETRAIN_OBJECTIVE embedding_normalized`, the shipped checkpoint rule, MLflow off). New: A3 at seeds
  42/7/13 (9 cells), A1 and A2 at seeds 7/13 (12 cells), 21 in all. E124's seed-42 A1 and A2 cells are reused as they
  are. A4 does not run.
- **Machines (ADR 0035, every arm of a (variant, seed) pair on one machine):** seed 42's pairs stay on medusa1, where
  E124's A1 and A2 ran; gorgona8 takes biodeg's and kc2's seed-13 pairs (6 cells); medusa1 the other 15.

## Rule, fixed now

- **G1 passes if both hold on at least 2 of the 3 controls (per control, as E124):**
  1. A2 beats A3 on test log loss: the per-cell relative difference (LL_A2 − LL_A3) / LL_A3 over 15 cells (A2 and A3
     paired by seed and fold), its 95% t interval with the Nadeau–Bengio correction (1/15 + 0.286) below 0;
  2. A1's direct contrast on the noise column (E124's scorer, unchanged but for the fold's seed) over 15 cells, its
     corrected interval above 0.
- Plain intervals and per-seed means are printed beside; they decide nothing. **Read only with all 21 cells done.**
- **A threat stated before the reading:** missForest's fill of a pure-noise column may sit near the column's mean, so
  A3 could read part of the gap from the filled value; that would shrink A2 − A3 for a reason other than the
  pipeline's blindness. It is not measured, and the reading is not adjusted for it.

## Outcome (2026-11-04 08:26 GMT-3): G1 fails again, 1 of 3; by the binding clause, C15 stops for good

All 21 cells done (07:52–08:15 GMT-3; gorgona8 6, medusa1 15; none failed). The unedited `c15_report.py --g1c` at
e143137, read 08:25 GMT-3 (only `prepare_dataset`'s label-mapping lines dropped):

```
## G1 again (E128, F5, seeds 42/7/13 x 5 folds): (1) A2 − A3 relative test log loss below 0; (2) A1's noise-column contrast above 0; Nadeau–Bengio corrected 95% t intervals over 15 cells; passes if both hold on >= 2 of 3
- biodeg_noise3p0_s2060_60nan: A2 − A3 -0.0453 [-0.1862, +0.0957] (plain [-0.1066, +0.0161]; s42 -0.0716, s7 -0.0069, s13 -0.0573); A1 contrast +1.775 [+1.464, +2.087] logit -> fail
- kc2_noise3p0_s2060_60nan: A2 − A3 -0.0208 [-0.1600, +0.1184] (plain [-0.0813, +0.0398]; s42 +0.0637, s7 -0.0734, s13 -0.0526); A1 contrast +2.889 [+0.578, +5.201] logit -> fail
- spambase_noise3p0_s2060_60nan: A2 − A3 -0.0983 [-0.1527, -0.0440] (plain [-0.1220, -0.0747]; s42 -0.1211, s7 -0.0909, s13 -0.0830); A1 contrast +1.680 [+1.457, +1.904] logit -> pass
Primary (G1 again): 1 of 3 -> fails
```

- **Condition 2 (A1 reads the informative gap):** holds on all three controls over 15 cells (+1.78 [+1.46, +2.09],
  +2.89 [+0.58, +5.20], +1.68 [+1.46, +1.90] logit); kc2's, which included 0 over E124's 5 cells, now excludes it.
- **Condition 1 (A2 beats A3 on log loss):** only on spambase (−0.098 [−0.153, −0.044]). On biodeg (−0.045) and kc2
  (−0.021) the point estimates are below 0 but the corrected intervals span about ±0.14 and include 0, as do the plain
  ones. The seed means agree in sign on biodeg and spambase (3 of 3), on kc2 2 of 3.
- **What follows, as fixed above:** C15 stops for good before Stage 2; no third form of G1 is tried; E125–E127 stay
  withdrawn; T45 closes. H2 is decided neither way.
- **What can be said, descriptively (it decides nothing):** on every control the shipped `[NULL]` arm reads a gap
  that is informative; the indicator's gain over the same missForest fill is measurable only on spambase, the largest
  of the three tables. Over 15 cells the correction's 0.286 term outweighs 1/15, so more seeds of the same five-fold
  design narrow the intervals little. The threat stated above (A3 reading the gap from missForest's fill) was not
  measured.
