# Does asking the gaps one column at a time, the rest as [NULL], score better? (pre-registered 2026-10-01)

This ticket was written and committed before the first run, at 23:36 GMT-3 on 2026-10-01 (commit `91fabfd`; the queues started 23:36:48).
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

## Outcome (2026-10-02, 05:40 GMT-3)

All 63 cells ran (23:36 GMT-3 on 2026-10-01 to 05:35 GMT-3 on 2026-10-02), none failed, none
missing. The integrity check passed: every cell's mask-path scores, induced and masked, equal E30's
arm-M cells to the last digit, so the two paths below are read off the same models E30 ran.
Numbers from `scripts/experiments/column_wise_report.py` (kept with `--tally` in the state
directory). Lower is better; a negative difference favours column-wise. ★ marks the nine
variants at 60nan or more, on which the reading is fixed.

| variant | mask | column-wise | best baseline | column-wise − mask (seeds 42, 7, 13) | verdict |
|---|---|---|---|---|---|
| `credit-g_20nan` | 0.8906 | 0.8910 | 0.9180 | +0.0004 [−0.0112, +0.0120] (−0.0017, +0.0107, −0.0078) | none |
| `credit-g_40nan` | 0.9793 | 0.9826 | 0.9638 | +0.0033 [−0.0040, +0.0106] (−0.0027, +0.0092, +0.0033) | none |
| `credit-g_60nan` ★ | 1.0119 | 1.0022 | 1.0000 | −0.0097 [−0.0223, +0.0029] (−0.0051, −0.0095, −0.0144) | none |
| `credit-g_80nan` ★ | 1.0314 | 1.0165 | 1.0000 | −0.0148 [−0.0253, −0.0044] (−0.0074, −0.0091, −0.0280) | column-wise better |
| `kr-vs-kp_20nan` | 0.6154 | 0.6276 | 0.6119 | +0.0122 [+0.0032, +0.0213] (+0.0128, +0.0007, +0.0231) | mask better |
| `kr-vs-kp_40nan` | 0.7704 | 0.7792 | 0.7725 | +0.0088 [+0.0013, +0.0164] (+0.0015, +0.0170, +0.0080) | mask better |
| `kr-vs-kp_60nan` ★ | 0.8942 | 0.8779 | 0.9030 | −0.0163 [−0.0264, −0.0063] (−0.0103, −0.0238, −0.0149) | column-wise better |
| `kr-vs-kp_80nan` ★ | 1.0011 | 1.0119 | 1.0000 | +0.0108 [−0.0103, +0.0320] (+0.0177, +0.0186, −0.0039) | none |
| `spambase_20nan` | 0.8681 | 0.8814 | 0.8664 | +0.0133 [+0.0062, +0.0204] (+0.0113, +0.0212, +0.0074) | mask better |
| `spambase_40nan` | 0.9577 | 0.9240 | 0.9371 | −0.0338 [−0.0682, +0.0007] (−0.0284, −0.0573, −0.0156) | none |
| `spambase_60nan` ★ | 0.9750 | 0.9478 | 0.9845 | −0.0272 [−0.0547, +0.0003] (−0.0117, −0.0270, −0.0431) | none |
| `spambase_80nan` ★ | 1.0266 | 0.9948 | 1.0000 | −0.0319 [−0.0629, −0.0008] (−0.0144, −0.0132, −0.0679) | column-wise better |
| `vehicle_20nan` | 0.4385 | 0.4474 | 0.4731 | +0.0089 [+0.0012, +0.0167] (+0.0078, +0.0097, +0.0093) | mask better |
| `vehicle_60nan` ★ | 0.6693 | 0.6688 | 0.7006 | −0.0005 [−0.0094, +0.0083] (−0.0045, +0.0028, +0.0002) | none |
| `biodeg_20nan` | 0.6210 | 0.6315 | 0.6356 | +0.0105 [−0.0067, +0.0277] (+0.0166, +0.0216, −0.0068) | none |
| `biodeg_60nan` ★ | 0.8267 | 0.7890 | 0.8658 | −0.0378 [−0.0538, −0.0217] (−0.0309, −0.0380, −0.0443) | column-wise better |
| `kc2_20nan` | 0.5843 | 0.5778 | 0.5665 | −0.0066 [−0.0160, +0.0029] (−0.0022, −0.0066, −0.0109) | none |
| `kc2_60nan` ★ | 0.6833 | 0.6417 | 0.7129 | −0.0417 [−0.0795, −0.0038] (−0.0404, −0.0317, −0.0529) | column-wise better |
| `pendigits_20nan` | 0.3343 | 0.3354 | 0.3500 | +0.0011 [+0.0006, +0.0017] (+0.0008, +0.0012, +0.0014) | mask better |
| `letter_20nan` | 0.4613 | 0.4626 | 0.4729 | +0.0013 [+0.0010, +0.0016] (+0.0012, +0.0014, +0.0013) | mask better |
| `electricity_20nan` | 0.6247 | 0.6220 | 0.6058 | −0.0027 [−0.0164, +0.0110] (+0.0007, −0.0105, +0.0017) | none |

**Tally.** Over the 21 variants: column-wise better on 5, mask better on 6, no detectable
difference on 10. On the nine ★ variants: column-wise better on 5 (credit-g_80nan,
kr-vs-kp_60nan, spambase_80nan, biodeg_60nan, kc2_60nan), mask better on none, no detectable
difference on 4. Every "mask better" verdict is on a 20nan or 40nan variant.

**Reading, by the rule fixed in advance: supported, at the rule's threshold.** Column-wise is
better on exactly the 5 heavy variants the rule asked for, and mask on none of them. Two of the
five verdicts sit close to zero (spambase_80nan's upper bound −0.0008, kc2_60nan's −0.0038); the
other three (credit-g_80nan, kr-vs-kp_60nan, biodeg_60nan) are clear. The verdict stands; the
weight it carries is that of a result at the threshold.

**Secondary measures.**
- **The effect grows with the missing level, as the hypothesis predicted:** mean difference
  +0.0043 at 20nan (9 variants), −0.0072 at 40nan (3), −0.0222 at 60nan (6), −0.0120 at 80nan
  (3). On the 12 light variants it is +0.0014; on the 9 heavy ones, −0.0188.
- **Means over the 21 variants:** mask 0.7745, column-wise 0.7673, best baseline 0.7781.
- **Against the best baseline:** the mean is below it on 11 variants under mask and 12 under
  column-wise (spambase_40nan and spambase_80nan join; kr-vs-kp_40nan leaves). At or above 1.0,
  the mean/mode fill: four variants under mask, three under column-wise (spambase_80nan drops to
  0.9948).
- **Where mask wins, the margins are small:** +0.001 on pendigits and letter (tight intervals from
  large tables), +0.009 to +0.013 on kr-vs-kp_20nan and _40nan, spambase_20nan and vehicle_20nan.
  Where column-wise wins, they are larger: −0.015 to −0.042.
- **Time.** Planned about 5.2 h of wall time, ending near 05:15 GMT-3; it took 5 h 59 min,
  1.15x the plan (speedup 0.87). The cells ran 1.11x E30's arm-M times, with the extra forward
  passes of the column-wise path inside that.

**What follows, as pre-stated.** The user decides between the two options, which are separate
decisions:
1. **Step 2 of T02** as the next pre-registered study: train with decode rows shaped like induced
   scoring, so that the headline path sees the shape it was trained on. The gradient above
   says the mismatch costs most where gaps are many and nothing where they are few, which is what
   a training-side fix should remove without the small price column-wise pays at 20nan.
2. **Column-wise scoring as the headline procedure** (its own ADR). It gains 0.007 on the mean
   over 21 variants and loses by up to 0.013 on six light variants. A rule that switches by
   missing level would fit this data after the fact and is not licensed by it.
