# H2's fresh-draw confirmation: odds 3 at seed 4848 against its MCAR siblings, confirmation seeds (E142; pre-registered 2026-11-05)

This ticket was written and committed before any cell (00:30 GMT-3). **Conditional:** it runs only if E139
([09](09-h2-outcome-siblings.md)) reads *lead shrinks* for at least one TabPFN arm; otherwise it is void and nothing runs.

- **Files (exist on both machines since C15's stage 3):** `{biodeg,kc2,spambase}_outcome3p0_s4848_60nan` and
  `{biodeg,kc2,spambase}_mcar_s4848_60nan`.
- **Cells:** A1 on the six files × seeds 1515/2626/3737 = 18 (`c15_run.py` stage `h2c`, all on medusa1 by
  `machine`); TabPFN native and mask arms offline on the 18 pairs (`tabpfn_mnar.py run` picks them up once the A1
  cell exists); reader `--tally-h2c`.
- **Rule:** E139's, on the three odds-3 files and their siblings, pooled per replicate (15), Holm over the two arms:
  shrinks / grows / holds (±0.01) / no verdict. *Shrinks* again → the H2 reading is confirmed on a fresh draw and fresh
  seeds, to the user; otherwise E139 stands alone as a first reading.

## Outcome (2026-11-05, 01:24 GMT-3; condition met by E139 at 00:34; 18 of 18 A1 cells and 18 TabPFN pairs on medusa1, done by 01:20 GMT-3, 0 failed)

Read with `tabpfn_mnar.py --tally-h2c`. Unedited tally, the interaction (arm − A1) on the odds-3 file minus (arm − A1)
on its same-seed MCAR sibling (s4848), seeds 1515/2626/3737 × 5 folds, test macro-F1:

| | native | mask |
|---|---|---|
| biodeg | −0.038 [−0.061, −0.016] | (per-file lines in the tally) |
| kc2 | −0.051 [−0.081, −0.021] | |
| spambase | −0.029 [−0.035, −0.023] | |
| **pooled, balanced (primary)** | **−0.0394 [−0.0513, −0.0275]**, NB [−0.067, −0.012], Holm < 0.0001 | **−0.0278 [−0.0398, −0.0159]**, NB [−0.055, −0.000], Holm 0.0002 |
| pooled, plain (beside) | −0.0274 [−0.0482, −0.0067], Holm 0.026 | −0.0219 [−0.0422, −0.0016], Holm 0.036 |

- **Verdict, both arms, both probability modes: the lead SHRINKS under outcome-dependent MNAR**, on a fresh draw and
  fresh seeds; the Nadeau–Bengio intervals of the balanced reading also stay below zero.
- **Each model's own gain from the label-bearing gaps (odds 3, F3 − sibling):** A1 +0.274 / +0.267 / +0.150 (biodeg /
  kc2 / spambase), TabPFN +0.236 / +0.215 / +0.121: both read the pattern, A1 reads 0.03–0.05 more of it.
- **Reading, by the rule:** E139's H2 reading is confirmed. On this project's outcome-dependent family TRIDENT's
  per-column `[NULL]` token gains 0.03–0.05 macro-F1 of ground on TabPFN v2 relative to MCAR, and an explicit 0/1 mask
  gives TabPFN only part of it back. Scope of the claim: three binary tables at 60% missingness, odds 1.5 and 3, two
  draws, six seeds in all; the lead does not vanish everywhere (TabPFN still ahead by up to +0.013 on biodeg/spambase
  outcome files), it shrinks. ADR 0032 (TabPFN in the strong tier) is unchanged; the thesis sentence is the user's call.
