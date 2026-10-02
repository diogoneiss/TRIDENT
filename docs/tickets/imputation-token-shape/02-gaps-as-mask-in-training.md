# Does training the decode stage on the scored row shape close the gap? (pre-registered 2026-10-02)

This ticket was written and committed before the first run, at 10:43 GMT-3 on 2026-10-02 (commit
`4058764`; the queues started 10:44:03 GMT-3). It is step 2 of task T02
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

## Running notes

- **Checks before launch, done:** the default run of credit-g_20nan at seed 42 equalled E30's
  arm-M cell on all 55 `impute/*` metrics; the smoke run of the arm at seed 1 finished and
  recorded `DECODE_GAP_TOKEN mask` in its `hyperparameters.json`, its params and its tag (a plain
  run in the store, outside the study; scores not read).
- **The tag backfill was stopped mid-mirroring** so the study could start: all 2137 earlier
  imputation runs are stamped `null`, but only 88 of their 616 trees had their mirrors re-done.
  The rest is re-mirrored after the study, with no trainer running.
- **Re-mirroring finished at about 20:20 GMT-3** (a one-off script, run with no trainer on the
  store). Of the 1737 stamped source runs, 1727 have a mirror carrying the tag; the 10 without
  are 9 `FAILED` sources, which ADR 0006 as amended does not mirror, and one Optuna trial under a
  deleted study, whose subtree the mirroring skips by design.

## Outcome (2026-10-02, 16:40 GMT-3)

All 63 cells ran (10:44 to 16:22 GMT-3), none failed, none missing. Their logged parameters
equal E30's arm-M cells' on every key but `DECODE_GAP_TOKEN` (and `PRETRAIN_OBJECTIVE`, a key
ADR 0013 added to the record after E30 ran; the M cells carry it as the `pretrain_objective` tag,
`embedding_normalized`). Numbers from `scripts/experiments/gap_token_report.py` (kept with
`--tally` in the state directory). Lower is better; a negative difference favours G. ★ marks the
nine variants at 60nan or more, on which the reading is fixed. Induced cells, 15 fold pairs per
variant, two trained models per pair.

| variant | G | M | M column-wise | best baseline | G − M (seeds 42, 7, 13) | verdict | G − M column-wise | verdict |
|---|---|---|---|---|---|---|---|---|
| `credit-g_20nan` | 0.8873 | 0.8906 | 0.8910 | 0.9180 | −0.0033 [−0.0174, +0.0108] (−0.0181, +0.0044, +0.0038) | none | −0.0037 [−0.0186, +0.0113] | none |
| `credit-g_40nan` | 0.9632 | 0.9793 | 0.9826 | 0.9638 | −0.0161 [−0.0302, −0.0019] (+0.0055, −0.0391, −0.0147) | none | −0.0194 [−0.0360, −0.0027] | none |
| `credit-g_60nan` ★ | 1.0030 | 1.0119 | 1.0022 | 1.0000 | −0.0090 [−0.0245, +0.0066] (−0.0058, +0.0016, −0.0226) | none | +0.0007 [−0.0094, +0.0109] | none |
| `credit-g_80nan` ★ | 1.0157 | 1.0314 | 1.0165 | 1.0000 | −0.0157 [−0.0299, −0.0015] (−0.0174, +0.0025, −0.0322) | none | −0.0009 [−0.0115, +0.0098] | none |
| `kr-vs-kp_20nan` | 0.5934 | 0.6154 | 0.6276 | 0.6119 | −0.0220 [−0.0360, −0.0080] (−0.0195, −0.0341, −0.0124) | G better | −0.0342 [−0.0439, −0.0246] | G better |
| `kr-vs-kp_40nan` | 0.7608 | 0.7704 | 0.7792 | 0.7725 | −0.0096 [−0.0196, +0.0004] (−0.0020, −0.0150, −0.0118) | none | −0.0184 [−0.0305, −0.0063] | G better |
| `kr-vs-kp_60nan` ★ | 0.8684 | 0.8942 | 0.8779 | 0.9030 | −0.0258 [−0.0348, −0.0167] (−0.0247, −0.0263, −0.0263) | G better | −0.0095 [−0.0180, −0.0009] | G better |
| `kr-vs-kp_80nan` ★ | 1.0100 | 1.0011 | 1.0119 | 1.0000 | +0.0088 [−0.0137, +0.0314] (+0.0164, +0.0265, −0.0164) | none | −0.0020 [−0.0185, +0.0146] | none |
| `spambase_20nan` | 0.8504 | 0.8681 | 0.8814 | 0.8664 | −0.0177 [−0.0281, −0.0072] (−0.0140, −0.0136, −0.0254) | G better | −0.0310 [−0.0409, −0.0210] | G better |
| `spambase_40nan` | 0.9065 | 0.9577 | 0.9240 | 0.9371 | −0.0512 [−0.0825, −0.0198] (−0.0493, −0.0682, −0.0360) | G better | −0.0174 [−0.0231, −0.0117] | G better |
| `spambase_60nan` ★ | 0.9449 | 0.9750 | 0.9478 | 0.9845 | −0.0301 [−0.0590, −0.0012] (−0.0157, −0.0297, −0.0450) | G better | −0.0029 [−0.0052, −0.0006] | G better |
| `spambase_80nan` ★ | 0.9971 | 1.0266 | 0.9948 | 1.0000 | −0.0295 [−0.0623, +0.0033] (−0.0096, −0.0121, −0.0668) | none | +0.0024 [−0.0036, +0.0083] | none |
| `vehicle_20nan` | 0.4218 | 0.4385 | 0.4474 | 0.4731 | −0.0167 [−0.0309, −0.0025] (−0.0249, −0.0136, −0.0116) | G better | −0.0256 [−0.0370, −0.0143] | G better |
| `vehicle_60nan` ★ | 0.6458 | 0.6693 | 0.6688 | 0.7006 | −0.0235 [−0.0434, −0.0036] (−0.0056, −0.0191, −0.0458) | G better | −0.0230 [−0.0417, −0.0042] | G better |
| `biodeg_20nan` | 0.5954 | 0.6210 | 0.6315 | 0.6356 | −0.0256 [−0.0422, −0.0090] (−0.0169, −0.0133, −0.0466) | G better | −0.0361 [−0.0571, −0.0150] | G better |
| `biodeg_60nan` ★ | 0.7684 | 0.8267 | 0.7890 | 0.8658 | −0.0583 [−0.0744, −0.0422] (−0.0448, −0.0626, −0.0675) | G better | −0.0205 [−0.0281, −0.0130] | G better |
| `kc2_20nan` | 0.5775 | 0.5843 | 0.5778 | 0.5665 | −0.0069 [−0.0306, +0.0169] (+0.0078, −0.0202, −0.0081) | none | −0.0003 [−0.0206, +0.0200] | none |
| `kc2_60nan` ★ | 0.6168 | 0.6833 | 0.6417 | 0.7129 | −0.0666 [−0.1271, −0.0061] (−0.0769, −0.0661, −0.0567) | G better | −0.0249 [−0.0598, +0.0100] | none |
| `pendigits_20nan` | 0.3328 | 0.3343 | 0.3354 | 0.3500 | −0.0015 [−0.0034, +0.0004] (+0.0008, −0.0020, −0.0033) | none | −0.0027 [−0.0048, −0.0006] | G better |
| `letter_20nan` | 0.4598 | 0.4613 | 0.4626 | 0.4729 | −0.0014 [−0.0025, −0.0004] (−0.0012, −0.0012, −0.0019) | G better | −0.0027 [−0.0037, −0.0018] | G better |
| `electricity_20nan` | 0.6201 | 0.6247 | 0.6220 | 0.6058 | −0.0046 [−0.0198, +0.0106] (+0.0011, −0.0147, −0.0003) | none | −0.0019 [−0.0118, +0.0080] | none |

**Tally.** G − M: G better on 11 of 21 variants, M better on none, no detectable difference on
10. On the nine ★ variants: G better on 5 (kr-vs-kp_60nan, spambase_60nan, vehicle_60nan,
biodeg_60nan, kc2_60nan), M better on none, no detectable difference on 4. G − (M scored
column-wise): G better on 12, column-wise better on none.

**Reading, by the rule fixed in advance: supported, and G qualifies as the default.** G is
better on 5 of the 9 heavy variants and worse on none, and no variant among the 21 shows "M
better". The recommendation this result carries is therefore **`DECODE_GAP_TOKEN mask` as the
imputation default**, through an ADR as ADR 0013 was. The user takes the decision.

**Secondary measures.**
- **The effect is there at every missing level, not only where gaps are many:** mean G − M
  −0.0111 at 20nan (9 variants), −0.0256 at 40nan (3), −0.0355 at 60nan (6), −0.0121 at 80nan
  (3). Where column-wise scoring cost up to 0.013 on the light variants (E31), training on the
  scored shape gains there too (kr-vs-kp_20nan −0.022, spambase_20nan −0.018, vehicle_20nan
  −0.017, biodeg_20nan −0.026, letter −0.001 with a tight interval).
- **Aligning the training beats aligning the scoring:** G is below M-scored-column-wise on 12
  variants and above it on none; by level, −0.015 (20nan), −0.018 (40), −0.013 (60), −0.000
  (80). At 80nan the two alignments give the same score.
- **Means over the 21 variants:** G 0.7542, M 0.7745, M column-wise 0.7673, best baseline 0.7781.
- **Against the best baseline:** the mean is below it on 16 variants under G, against 11 under M.
  The five that stay above: credit-g_60nan and _80nan and kr-vs-kp_80nan (at or above 1.0, the
  mean/mode fill, under both arms; spambase_80nan drops below it under G), electricity_20nan
  (0.620 against hgb's 0.606) and kc2_20nan (0.578 against knn5's 0.567).
- **Masked population (secondary):** G better on 10 variants, M better on none.
- **Time.** Planned about 5.8 h of wall time; it took 5 h 38 min, 0.97x the plan (speedup 1.03).
  The cells ran at the same speed as E30's arm M (electricity 100.7 against 100.3 min a cell): the
  token shown at a gap costs nothing. The report's "M" times are E31's runs, which carried the
  column-wise passes, so they read higher than G's.
- **Pairing.** The transformation draws nothing, so a G cell and its M cell consume the same
  random stream: the same folds, masks and initialisation. The pairs are tighter than two
  independent draws would be, which the pre-registration did not anticipate.

**Caveats, stated rather than ruled on.**
- Two of the five heavy verdicts have a wide interval that just excludes zero (spambase_60nan
  −0.030 [−0.059, −0.001], kc2_60nan −0.067 [−0.127, −0.006]); the other three are clear
  (kr-vs-kp_60nan, vehicle_60nan, biodeg_60nan). Four more variants lean G's way without a
  verdict: credit-g_40nan and _80nan have intervals that exclude zero but one seed positive each
  (+0.0055 and +0.0025), which the seed-agreement rule refuses; kr-vs-kp_40nan has all three
  seeds negative and an interval reaching +0.0004; spambase_80nan has all three negative and a
  wide interval.
- The pre-training stage still shows gaps as `[NULL]`; whether aligning it too helps is untested.
- The three variants at or above 1.0 under both arms (credit-g_60nan and _80nan, kr-vs-kp_80nan)
  say the token shape was not the only cause of the model's losses at high missingness.
