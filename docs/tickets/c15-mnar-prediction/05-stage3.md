# C15 Stage 3: confirmation, read once: A1 against A2 on fresh draws of P and of its MCAR siblings (E126; pre-registered 2026-11-04)

This ticket was written and committed before the stage's first cell (00:31 GMT-3). Its cells run only if G1 passes
(E124, [03](03-stage1.md)); if G1 fails, this ticket is withdrawn unrun.

- **Why:** H2, C15's primary: `[NULL]` tokens (A1) are non-inferior to missForest + indicator (A2) in the same
  encoder under informative missingness, pooled over P. Confirmation uses fresh mask draws and never-used seeds,
  read once (the protocol; [plan](../../plans/2026-11-03-c15-mnar-prediction.md)).
- **The user's go:** the full core, Stages 0–3 (2026-11-03); margins fixed (−0.01 macro-F1, +2% relative log loss).
- **Code:** no `src/` change. Data: `mnar_family.py`, data seed 4848, written on both machines before this ticket
  (md5 equal, 2026-11-04 00:33 GMT-3). Cells: `C15_STAGE=3 c15_run.py`. The H2 reader is committed before the
  first confirmation cell is read.

## Data and design (fixed before the first cell)

- **P's fresh draws (5):** `biodeg_probit0p5_s4848_60nan`, `vehicle_probit0p5_s4848_60nan` (self-masking, slope 0.5,
  as E45's), `{biodeg,kc2,spambase}_outcome3p0_s4848_60nan` (odds 3).
- **The MCAR siblings' fresh draws (4):** `{biodeg,vehicle,kc2,spambase}_mcar_s4848_60nan`, the same per-column counts.
- **Arms:** A1 and A2 on all 9; model seeds **1515/2626/3737** (in no log entry); five folds; configuration as E124.
  54 cells, all on medusa1.

## Rule, fixed now (the protocol's H2, pooled tables)

- **Per variant of P:** d_c = A1 − A2 on test macro-F1, and (LL_A1 − LL_A2) / LL_A2, over 15 cells (seed, fold);
  Nadeau–Bengio corrected 95% t intervals (variance × (1/15 + 0.286), 14 df), seed agreement. Calls: non-inferior if
  the F1 lower bound > −0.01 and the LL upper bound < +0.02 with the seeds agreeing; inferior if the F1 upper bound
  < −0.01 or the LL lower bound > +0.02; inconclusive otherwise.
- **Pooled (primary):** D_c, the mean over P's 5 variants of d_{v,c}, per cell; the corrected interval over the 15 D_c.
  **Supported** if the pooled F1 lower bound > −0.01 and the pooled LL upper bound < +0.02, and A1 is inferior on no
  variant of P; **reversed** if the pooled F1 upper bound < −0.01 or the pooled LL lower bound > +0.02;
  **inconclusive** otherwise (including a pooled non-inferiority that fails the guard). P has 5 variants from 4
  tables, so the gate's n ≥ 4 from ≥ 3 tables holds. H2-sup (both intervals in A1's favour) only after "supported".
- **H2.2 (secondary), on P against the siblings:** as ticket 04's (a) and (b), Holm; supported if both pass and E125's
  (c) lies within ±0.1 logit.
- **Read once, only with all 54 cells done.** The sensitivity reads of the protocol (count rule, random effects,
  per-table means, fork tallies) are reported, never deciding.
