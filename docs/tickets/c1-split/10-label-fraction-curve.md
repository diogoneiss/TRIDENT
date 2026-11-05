# The label-fraction curve under a scarce validation split: a tenth and a half, fresh seeds (E140; pre-registered 2026-11-05)

This ticket was written and committed before the cut's first cell (00:30 GMT-3).

- **Why:** E136 opened pre-training's channel at a quarter of the labels and not at a tenth (oracle validation); E138
  ([09](09-scarce-confirmation.md)) confirmed the quarter with the validation split scarce too (+0.022 [+0.009, +0.035]).
  Where does the channel open and close? The user asked for the machines to stay busy through the night (2026-11-05,
  00:25 GMT-3), so the curve runs now, deployable form (`FINETUNE_SCARCE_VALIDATION on`), fresh seeds.
- **Code:** none new. Arms **S10V / B10V** (fraction 0.1) and **S50V / B50V** (0.5), cut `10` in `c1_run.py`
  (routing by seed: 1515 on gorgona8, 2626 and 3737 on medusa1); reader `c1_report.py --curve`.

## Design and rule (fixed before the first cell)

- **Cells:** the eight C1 files × seeds 1515/2626/3737 × 4 arms = 96. Labelled rows at a tenth: 73 / 36 / 59 / 322
  (validation 10 / 5 / 8 / 46); at a half: 369 / 182 / 295 / 1610 (validation 52 / 26 / 42 / 230).
- **Gate per fraction, two tests as one family:** S(f)V − B(f)V pooled over the eight files, 15 replicates, plain 95%
  t interval, Holm over the two fractions; **open** if Holm p < 0.05 with a positive mean, else **closed**. E138's
  quarter is printed beside, so the curve reads 0.10 / 0.25 / 0.50 (and 1.0 from E131: +0.004, closed).
- **What follows:** descriptive for the thesis (where pre-training pays on these tables); no ADR. kc2 at a tenth
  (5 validation rows) is a stated threat.
- **Read only with all 96 cells done.**

## Outcome (2026-11-05, 01:25 GMT-3; 96 of 96 cells, gorgona8 32 and medusa1 64, cut 10 done on both by 01:08 GMT-3, 0 failed)

Read with `c1_report.py --curve`. Unedited tally, S(f)V − B(f)V in test macro-F1, seeds 1515/2626/3737 × 5 folds,
pooled over the eight files:

| labels | pooled S − B | NB | p | Holm | channel | files better / worse |
|---|---|---|---|---|---|---|
| 10% (E140) | **+0.0290 [+0.0165, +0.0416]** | [+0.0002, +0.0579] | 0.0002 | 0.0004 | **OPEN** | kc2_60nan (+0.087), vehicle_60nan (+0.044) / none |
| 25% (E138, beside) | +0.0218 [+0.0090, +0.0346] | [−0.0077, +0.0512] | 0.0026 | — | open | spambase ×2, vehicle_60nan / none |
| 50% (E140) | **+0.0157 [+0.0075, +0.0239]** | [−0.0031, +0.0345] | 0.0010 | 0.0010 | **OPEN** | none / none |
| 100% (E131, oracle selector) | +0.004 [−0.008, +0.016] | | | | closed | |

- **The curve is monotone:** the fewer the labels, the more pre-training is worth, from +0.03 at a tenth to +0.004 at
  every label. No file is worse at any fraction. At a tenth the per-file intervals are wide (kc2 with 36 labelled and
  5 validation rows: +0.087 [+0.010, +0.165]), but every seed's pooled mean is positive at both new fractions.
- **A tenth is open here and was closed in E136** (+0.011 [−0.005, +0.026], oracle validation, seeds 42/7/13). Two
  things differ: the seeds, and the selector. With the whole validation split choosing the kept epoch, B's best
  epoch is found as reliably as S's; with 5–46 validation rows, B's noisier selection costs it more than S's, so the
  deployable reading favours pre-training more. That is a hypothesis from two readings, not a tested claim.
- **Reading, by the rule:** both new fractions open the channel. Descriptive for the thesis: on these four tables at
  60% missingness, pre-training on all rows pays whenever labels are scarce, and pays more the scarcer they are.
  No ADR.
