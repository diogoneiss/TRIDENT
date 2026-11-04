# C15 Stage 2: MICE without an indicator (A4), the complete and MAR tables, H2-c2, H2.2 and H2b on development (E125; pre-registered 2026-11-04)

This ticket was written and committed before the stage's first cell (00:31 GMT-3). Its cells run only if G1 passes
(E124, [03](03-stage1.md)); if G1 fails, this ticket is withdrawn unrun.

- **Why:** the plan's Stage 2 ([plan](../../plans/2026-11-03-c15-mnar-prediction.md)): the secondaries H2-c2 (A1 beats
  imputation without an indicator), H2.2's development read (the direct contrast is larger on driving columns
  under MNAR than under the matched MCAR sibling, with E63's MAR as the negative control) and H2b (the
  mask-reading arms lose more than A4 when the test pattern stops tracking y).
- **The user's go:** the full core, Stages 0–3 (2026-11-03), and "comece agora" (2026-11-04).
- **Code:** no `src/` change: `C15_ARM A4:<k>` and `A1` exist (E124). Cells: `C15_STAGE=2 c15_run.py`. The H2.2
  and H2b scorers (`c15_report.py`) are committed before any Stage 2 reading.

## Design (fixed before the first cell)

- **A4, five MICE draws (k = 0–4), one full run each, no indicator; A4's prediction the mean of the five test
  probabilities:** on F0 (7), F1 (7) and F3 (6), seeds 42/7/13. 300 cells.
- **A1 on F7, the complete tables** (biodeg, kc2, vehicle, spambase `_00nan`): H2b's α = 0 reference. 12 cells.
- **A1 on F6, E63's MAR files and their matched controls** (biodeg, kc2, vehicle `_mar_60nan`, `_marctl_60nan`):
  H2.2's negative control. 18 cells. kr-vs-kp's pair stays out (A2 needs numerical tables; the plan's F6 is six).
- **Configuration, folds and checkpoint as E124.** Machines: seed 42 on gorgona8, seeds 7 and 13 on medusa1, so every
  (variant, seed) pair shares a machine with its Stage 1 arms (ADR 0035).

## Measures and rules, fixed now (the protocol's, on development; all descriptive except as stated)

- **H2-c2:** per variant of P's development counterparts (biodeg_probit_60nan, vehicle_probit_60nan, and the
  odds-3 outcome variants of biodeg, kc2, spambase), LL_A1 − LL_A4 over 15 cells, Nadeau–Bengio corrected, Holm over
  the five; read as "A1 beats A4" where the corrected interval lies below 0.
- **H2.2 dry run:** the direct contrast C_j (A1: logit P at `[NULL]` minus the logit of the mean over K = 20 MICE
  draws of the cell given the row, other gaps `[NULL]`; binary tables P(y = 1), vehicle total variation),
  averaged over each column's test gap cells, per fold. U_drive and U_nondrive over E122's driving-column table
  (F3: every masked column). (a) U_drive(MNAR) − U_drive(MCAR sibling) per cell, pooled over P's development
  variants by random effects; (b) sign agreement on sign-test columns, one-sided binomial; (c) F6: U(MAR) −
  U(control) within ±0.1 logit. Development is the dry run; H2.2 decides at confirmation (Stage 3).
- **H2b grid (inference on the kept models of F1 ∩ P and F3 ∩ P, A1, A2, A4):** D1 the variant's own test gaps; D4
  the MCAR sibling's masks on the same test rows; D6 (F3) the test masks redrawn at the same counts with the odds
  reversed (1/3), seed 4949. Shift cost s = (LL_regime − LL_D1) / LL_D1. Supported if s_A1 − s_A4 > 0 and
  s_A2 − s_A4 > 0 (Holm); refuted if both upper bounds are below +0.02; inconclusive otherwise. D2, D3, D5 and the
  α = 0 row are reported as diagnostics.
- **Read only with every cell of a block done.**

## Withdrawn unrun (2026-11-04 01:16 GMT-3)

G1 failed (E124, [03](03-stage1.md), 1 of 3), so by the rule this stage does not run. No cell of it ran.
