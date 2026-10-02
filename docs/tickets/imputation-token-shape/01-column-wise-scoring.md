# Does asking the gaps one column at a time, the rest as [NULL], score better? (pre-registered 2026-10-01)

This ticket was written and committed before the first run, at about 23:45 GMT-3 on 2026-10-01.
It is step 1 of task T02 in [TASKS.md](../../TASKS.md).

## Why

The decode stage and the headline induced score show the model two different row shapes.

- **What the decode stage trains on:** a row's real gaps are `[NULL]` and never enter the loss.
  Of the row's other cells, a share is shown as `[MASK]`: `PROB_MASCARA` × (1 − the row's share
  of gaps). At the default 0.5 that is about 32% of the cells on a 20nan variant, but about 2%
  on an 80nan one.
- **What the induced score shows:** every gap in the row is `[MASK]` at once. That is about 80%
  of an 80nan row, a shape the decoder never trained on.

That mismatch is the lead hypothesis for why the model loses to simple imputers where gaps are
many: every arm loses to mean/mode on credit-g_80nan, and the candidate default M is still above
the best baseline on 10 of 21 variants (E30).

Scoring the same induced cells in the trained shape tests the hypothesis without changing
training. The flag `--score_column_wise` does that, and the model, the cells and the truths stay
the same:

- it asks each gap column in turn;
- that column's gaps are `[MASK]`;
- every other gap is the `[NULL]` the variant stores.

## Design (fixed before launch)

- **One arm, two scoring paths on the same trained model:**
  - every run is a plain imputation run under today's defaults, ADR 0013 (E30's arm M);
  - each run reads the promoted file or the defaults exactly as any run does;
  - five folds, with `--score_column_wise` on.
- **The two paths:**
  - *mask*: the headline `impute/induced/impute_score`, every gap as `[MASK]`;
  - *column-wise*: `impute/induced/column_wise/impute_score`, the same cells asked one gap column
    at a time.
- **Variants:** all 21 of ADR 0007.
- **Seeds:** 42, 7, 13.
- **Size:** 63 cells, tagged `experiment=column-wise-2026-10-02`, the date (GMT-3) on which the
  cells run.
- **Running it:**
  - four concurrent queues (`scripts/experiments/run_column_wise_queue.sh 1..4`), one trainer
    each, CPU threads capped at 4;
  - cells are assigned longest first by E30's measured arm-M cell times, about 313 minutes of
    those times per queue: about 5.2 h of wall time, ending near 05:15 GMT-3;
  - no cell starts after 14:00 GMT-3 on 2026-10-02, and a cell not started by then is reported
    missing;
  - nothing is rerun or extended on its result; a crash is fixed, its cell is rerun from the
    start, and the rerun is noted here.
- **Integrity check:** the scores of the mask path, and of the masked population, must equal
  E30's arm-M cells to the last digit, because the configuration, code path and seed are the
  same. A smoke run of credit-g_20nan at seed 42 matched on all 55 `impute/*` metrics, and its
  column-wise values were not looked at.

## Measures

- **Primary:** per variant, D = column-wise − mask on the induced `impute_score`.
  - D is paired by seed and fold, 15 pairs, each pair from one trained model.
  - It is read with a t-interval.
  - A variant gets a verdict only when the interval excludes zero and all three seeds' mean D
    agree in sign.
  - A negative D favours column-wise.
- **Secondary, descriptive:**
  - the mean D at each missing level (20, 40, 60, 80nan);
  - how many variants have a mean below the best baseline under each path (for the mask path,
    E30's M had 11 of 21);
  - the integrity check.
- **Reading, fixed now, on the 9 variants at 60nan or more:** credit-g, kr-vs-kp and spambase at
  60 and 80; vehicle, biodeg and kc2 at 60.
  1. **Supported:** "column-wise better" on at least 5 of the 9, and "mask better" on none of
     them.
  2. **Refuted:**
     - "mask better" on at least 5 of the 9; or
     - no verdict on at least 7 of the 9, which would mean no effect where the hypothesis
       predicts the largest one.
  3. **Mixed:** anything else. The report then lists each variant's verdict.

  The hypothesis also predicts that D gets more negative as the missing level rises. That
  prediction is reported, not ruled on.
- **What each reading leads to. The user decides; this is the recommendation the result will
  carry:**
  - **If supported:** two options follow, and they are separate decisions.
    1. Step 2 of T02, training with decode rows shaped like induced scoring, becomes the next
       pre-registered study.
    2. Column-wise scoring is itself a legitimate way to impute, at one pass per gap column. It
       could become the headline procedure, which would need its own ADR.
  - **If refuted:** the token-shape mismatch is not why the model loses. The next lead is T03,
    the architecture defaults.
- **Analysis:** `scripts/experiments/column_wise_report.py`; `--tally` gives the counts and the
  integrity check.

## Known limits, stated in advance

- **What the test covers:** it changes only how the other gaps are shown (`[NULL]` instead of
  `[MASK]`). The visible cells, the trained model and the scored cells are identical.
- **What it cannot say:** whether a model trained on the induced shape would do better. That is
  step 2.
- **Multiplicity:** 21 tests. With the seed-agreement rule, a false verdict is rarer than 1 in
  20 per test.
- **Noise:** each pair comes from one model, so most fold noise cancels. Its intervals will be
  narrower than E30's for the same effect.
