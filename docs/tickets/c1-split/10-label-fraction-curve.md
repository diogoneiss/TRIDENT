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
