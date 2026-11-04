# C15 Stage 4 (optional, secondary): A4 on the dial, the sign arm, A3 and the null model A5 (E127; pre-registered 2026-11-04)

This ticket was written and committed before the stage's first cell (01:07 GMT-3). Its cells run only if G1 passes
(E124) and only on GPUs the approved stages leave idle; they decide nothing about H2.

- **The user's go:** "se ocioso rode o estagio 4" (2026-11-04, about 00:50 GMT-3).
- **Code:** `C15_ARM A5:<column>` (the column's token blocked as a key in every layer, `TabularTransformerEncoder.blocked_key`,
  unit-tested: the classifier's output does not move when the blocked column's values do); `mnar_family.py sign`
  (F4; unit-tested). Cells: `C15_STAGE=4 c15_run.py`.
- **Left out tonight:** the route-B Shapley surrogates (a KernelSHAP game per variant; not written) and the protocol's
  short-mode screen: every block runs in the decision mode at once, and every reading is descriptive.

## Data (generated before the first cell)

- **F4, the sign arm:** `mnar_family.py sign --direction plus|minus --seed 2060` on biodeg, kc2 and spambase at 60nan:
  column j's slope is ±0.5 × sign_j (the sign of its L2 logistic coefficient on the standardised complete table),
  so the hidden tail raises (plus) or lowers (minus) P(y = 1). Realised missing rate by class, class 0 / class 1:
  plus 0.59/0.61, 0.60/0.61, 0.59/0.62; minus 0.61/0.58, 0.60/0.58, 0.61/0.58.

## Design (fixed before the first cell)

- **A4 on F2** (biodeg's dial, slopes 0.25, 1.0, 2.0): A4:0–4, seeds 42/7/13. 45 cells. H2-dial's tilt half.
- **A1 and A2 on F4** (6), seeds 42/7/13. 36 cells. H2.2's sign test on a design built for it.
- **A3** (missForest, no indicator) on biodeg, kc2, vehicle, spambase at 60nan, self-masking and MCAR, seeds 42/7/13.
  24 cells. A2 − A3 is what the indicator adds over the same fill.
- **A5** on the four self-masking 60nan variants, each blocking its table's top driving column in E122's table
  (biodeg V36, kc2 total_Opnd, vehicle PR.AXIS_ASPECT_RATIO, spambase word_freq_george), seeds 42/7/13. 12 cells.
- Configuration as E124; seed 42 on gorgona8, seeds 7 and 13 on medusa1 (ADR 0035). 117 cells.

## Readings (descriptive, fixed now)

- **H2-dial:** LL_A4 − LL_A1 along biodeg's dial (MCAR, 0.25, 0.5, 1.0, 2.0), 15 cells a rung, the corrected interval.
- **F4:** A1 − A2 as E124's development tally; A1's direct contrast on F4's sign-test columns, sign agreement with
  sign_j × sign(slope).
- **A3:** A2 − A3 on macro-F1 and relative log loss, per variant, the corrected interval.
- **A5:** A1's contrast on the blocked column, computed on A5's kept models, must be exactly 0 (an integrity check of
  the contrast code; if not, every H2.2 reading is suspect and that is reported first).
