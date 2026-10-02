# Does training the decode stage on the scored row shape close the gap? (pre-registered 2026-10-02)

This ticket was written and committed before the first run, in the morning of 2026-10-02
GMT-3 (the commit and the queues' start time are in the Running notes). It is step 2 of task T02
in [TASKS.md](../../TASKS.md), the follow-up the user chose after step 1
([01](01-column-wise-scoring.md), E31).

## Why

E31 showed that the train/score token-shape mismatch costs score where gaps are many. The decode
stage trains on rows whose real gaps are `[NULL]` and only a share of the other cells `[MASK]`;
the induced headline shows every gap as `[MASK]` at once. Asking the same cells in the trained
shape, one gap column at a time, was better on 5 of the 9 variants at 60nan or more and worse on
none, with the effect growing with the missing level (+0.004 at 20nan, −0.022 at 60nan). It also
cost up to 0.013 on six light variants.

Step 2 fixes the mismatch from the training side instead. With `DECODE_GAP_TOKEN mask`
(`--decode_gap_token mask`) the decode stage shows every real gap as `[MASK]`, the token the
scoring presents it as, on its training rows, its validation rows and the masked test population.
Nothing else changes:

- the cells that enter the loss are still the ones the epoch's mask hides, never a gap (a gap has
  no truth);
- the pre-training stage, the targets, the schedule, the epochs and the heads are untouched;
- the headline scoring is untouched: every gap as `[MASK]`, now the shape the stage trained on.

If the mismatch is what costs the score at high missingness, this should recover E31's gains on
the heavy variants without the price column-wise paid on the light ones, where the training rows
already looked like the scored ones.

## Design (fixed before launch)

- **Arm G:** a plain imputation run under today's defaults (ADR 0013, E30's arm M), with
  `DECODE_GAP_TOKEN mask`. Each variant reads its promoted file or the defaults exactly as any
  run does; five folds.
- **Reference M:** E30's arm-M cells of the same variant and seed, read through E31's cells,
  which re-read the same models and hold both the headline score and the column-wise one (E31
  verified their headline equals E30's to the last digit). G and M are different trained models,
  so a pair is two draws, as in E30, not one model read twice, as in E31.
- **Variants:** all 21 of ADR 0007. **Seeds:** 42, 7, 13. **Size:** 63 new cells, tagged
  `experiment=gap-token-2026-10-02`, `arm=G`.
- **Running it:** four concurrent queues (`scripts/experiments/run_gap_token_queue.sh 1..4`),
  one trainer each, CPU threads capped at 4; cells assigned longest first by E30's arm-M cell
  times, about 313 minutes of those per queue. E31's cells ran at 1.11x those times, so about
  5.8 h of wall time is expected. No cell starts after 22:00 GMT-3 on 2026-10-02 (01:00 UTC on
  2026-10-03); a
  cell not started by then is reported missing. Nothing is rerun or extended on its result; a
  crash is fixed, its cell rerun from the start, and the rerun noted here.
- **Checks before launch:** a default run of credit-g_20nan at seed 42 after the change must
  equal E30's arm-M cell on every `impute/*` metric (the flag off changes nothing); a smoke run
  of the arm at seed 1, outside the study, must finish and record `DECODE_GAP_TOKEN mask`. Its
  scores were not read.

## Measures

- **Primary:** per variant, D = G − M on the induced `impute_score`, paired by seed and fold
  (15 pairs), t-interval; a verdict only when the interval excludes zero and all three seeds'
  mean D agree in sign. A negative D favours G.
- **Reading, fixed now, on the 9 variants at 60nan or more** (credit-g, kr-vs-kp and spambase at
  60 and 80; vehicle, biodeg and kc2 at 60):
  1. **Supported:** "G better" on at least 5 of the 9, and "M better" on none of them.
  2. **Refuted:** "M better" on at least 5 of the 9; or no verdict on at least 7 of the 9.
  3. **Mixed:** anything else.
- **The default, fixed now:** G becomes the recommended imputation default (an ADR, like
  ADR 0013) only if the reading is supported **and** no variant among the 21 shows "M better".
  A default that depends on the missing level would fit this data after the fact and is not
  licensed by it. The user takes the decision.
- **Secondary, descriptive:**
  - D by missing level (20, 40, 60, 80nan): the hypothesis predicts it grows more negative as
    gaps increase and is near zero at 20nan;
  - G − (M scored column-wise), from E31's cells: whether aligning the training reaches what
    aligning the scoring gave;
  - the masked population, G − M;
  - how many variants have a mean below the best baseline under G and under M (M: 11 of 21),
    and how many sit at or above 1.0 (M: 4);
  - fold time per cell, and estimated against real wall time.
- **Analysis:** `scripts/experiments/gap_token_report.py` (`--tally` for the counts).

## Known limits, stated in advance

- **Two draws per pair:** the noise is E30's, not E31's, so the intervals will be wider for the
  same effect; a true effect of E31's size (−0.015 to −0.042 on the heavy variants) was
  detectable at E30's noise on effects of that size.
- **Scope:** the pre-training stage still shows gaps as `[NULL]`; the decode stage retrains the
  encoder, so it can adapt, but a mismatch there is not tested here.
- **The `[NULL]` token's own signal** (the paper's claim, for classification) is not what this
  study measures; E01 already found the null path worse for imputation.
- **Multiplicity:** 21 tests with the seed-agreement rule, as in E30 and E31.
