# H2 under outcome-dependent MNAR: TabPFN and A1 on F3 against same-seed MCAR siblings (E139; pre-registered 2026-11-04)

This ticket was written and committed before the stage's first cell (23:37 GMT-3).

- **Why:** E137 ([08](08-tabpfn-arm.md)) found TabPFN's lead over A1 robust to value-dependent (probit) MNAR but gone
  on the outcome-dependent family F3 (−0.003 [−0.009, +0.003], A1 ahead on kc2). F3 had no MCAR sibling drawn the same
  way, so that reading was descriptive. The user agreed to the follow-up ("concordo com voce", 2026-11-04): MCAR
  siblings by the same seed, the same folds, A1 trained on the new files only, TabPFN offline.
- **Files, generated now on both machines (bit-identical, `datasets/mnar_family.py mcar --bases biodeg kc2 spambase
  --pct 0.6 --seed 2060`):** `{biodeg,kc2,spambase}_mcar_s2060_60nan`: uniform weights, the per-column streams
  `SeedSequence([2060, j])` of F3's own draw, 60.0% of feature cells missing, label untouched. The design C15's ticket
  05 fixed for its fresh draws (`_mcar_s4848`).
- **Code:** `c15_run.py` stage `h2` (A1 on the three siblings, seeds 42/7/13, 9 cells); `tabpfn_mnar.py` runs the
  siblings' pairs (its `run`/`--todo` now cover them) and reads with `--tally-h2`. No change to `src/`.

## Design (fixed before the first cell)

- **Cells:** A1 (`[NULL]` tokens, E124's configuration) on the three siblings × seeds 42/7/13 × 5 folds; TabPFN v2
  native and mask arms on the same 9 pairs (one fit a fold, balanced and plain predictions, as E137). F3's A1 cells
  and TabPFN pairs are E124's and E137's.
- **Machines (ADR 0035):** `c15_run.machine`: kc2 and spambase at seed 42 on gorgona8, the rest on medusa1; the TabPFN
  pairs follow their A1 cell.

## Rule, fixed now

- **Primary, two tests as one family:** the interaction (arm − A1) on each F3 file minus (arm − A1) on its sibling, in
  test macro-F1, paired per (table, seed, fold), pooled over the six F3 files (both odds, three tables) per replicate,
  plain 95% t interval, Nadeau–Bengio beside; Holm over the two TabPFN arms, balanced probabilities. **Lead shrinks
  under outcome-dependent MNAR** if the interval is below 0, **grows** if above, **holds** if inside ±0.01, otherwise
  **no verdict**. Plain probabilities beside.
- **Descriptive:** each model's own F3 − sibling difference (does A1 gain from gaps that carry the label; does
  TabPFN), per file.
- **What follows:** *shrinks* is the first positive H2 reading of this project (TRIDENT's per-column `[NULL]` gains
  ground where the gap carries the label), to go to the user with the odds-1.5 / 3.0 lines and a fresh-draw
  confirmation proposal (`_s4848` files exist); *holds* or *grows* closes H2 on this family; no ADR here.
- **Threats stated now:** three tables, one draw seed; kc2's intervals are wide; the siblings are new MCAR draws, not
  the `_60nan` files E137 used as F0, so the F0 reading is not repeated here.
- **Read only with all 9 A1 cells and 9 TabPFN pairs done.**

**Amendment (23:45 GMT-3):** medusa1's GPUs were filled by another user's job at 23:42 GMT-3; its stage-h2 queue failed
its first cell (out of memory) and was stopped. gorgona8's two cells and pairs run now; medusa1's seven cells and
their TabPFN pairs are relaunched when its GPUs free up, so the F3 ↔ sibling pairing stays on one machine per seed.

## Outcome (2026-11-05, 00:34 GMT-3; 9 of 9 A1 cells and 9 of 9 TabPFN pairs; gorgona8 done 00:28 GMT-3, medusa1 by 00:33 GMT-3 after its GPUs freed at 00:21)

Read with `tabpfn_mnar.py --tally-h2`. Unedited tally, balanced probabilities (primary), the interaction
(arm − A1) on the outcome file minus (arm − A1) on its same-seed MCAR sibling, test macro-F1:

| file | native | mask |
|---|---|---|
| biodeg odds 1.5 | −0.054 [−0.078, −0.030] | −0.041 [−0.067, −0.015] |
| biodeg odds 3.0 | −0.041 [−0.061, −0.020] | −0.045 [−0.065, −0.025] |
| kc2 odds 1.5 | −0.070 [−0.095, −0.046] | −0.050 [−0.079, −0.022] |
| kc2 odds 3.0 | −0.067 [−0.094, −0.041] | −0.046 [−0.072, −0.021] |
| spambase odds 1.5 | −0.013 [−0.020, −0.006] | −0.012 [−0.019, −0.005] |
| spambase odds 3.0 | −0.026 [−0.031, −0.020] | −0.024 [−0.030, −0.019] |
| **pooled (6 files)** | **−0.0451 [−0.0555, −0.0347]**, NB [−0.069, −0.021], Holm < 0.0001 | **−0.0364 [−0.0472, −0.0256]**, NB [−0.061, −0.012], Holm < 0.0001 |

- **Verdict, both arms: the lead SHRINKS under outcome-dependent MNAR.** Every one of the twelve per-file intervals
  is below zero. On the siblings TabPFN leads A1 by +0.049 / +0.051 / +0.026 (biodeg / kc2 / spambase); on the outcome
  files the lead is −0.005 to +0.013, and A1 is ahead on kc2 by 0.016–0.019.
- **Plain probabilities (beside):** native −0.020 [−0.034, −0.007] shrinks (Holm 0.012); mask −0.008 [−0.020, +0.005]
  no verdict; the difference from the balanced reading is kc2, where plain TabPFN already trails A1 on the sibling.
- **Each model's own gain from outcome-dependent gaps (F3 − sibling, descriptive):** A1 +0.12 / +0.25 (biodeg odds
  1.5 / 3.0), +0.07 / +0.27 (kc2), +0.08 / +0.15 (spambase); TabPFN +0.07 / +0.21, −0.00 / +0.20, +0.07 / +0.12. Both
  read the pattern; A1 reads more of it: the per-column `[NULL]` token carries the label-bearing gap into the
  classifier more fully than TabPFN's mean fill plus indicator, and an explicit 0/1 mask does not close the difference.
- **Reading, by the rule:** the first positive H2 reading of this project, on one draw (s2060) and the development
  seeds. By the ticket, E142 ([10](10-h2-fresh-draws.md)) runs now: odds 3 at seed 4848 against its MCAR siblings, seeds
  1515/2626/3737. ADR 0032 unchanged; whether this becomes a thesis claim waits for E142.
