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
