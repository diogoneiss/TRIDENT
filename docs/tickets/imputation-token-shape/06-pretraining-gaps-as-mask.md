# Does showing the gaps as [MASK] to pre-training help too? (pre-registered 2026-10-05)

This ticket was written and committed before the first run, at 00:16 GMT-3 on 2026-10-05 (commit
`d0744ad`; the queues started 00:16:30 GMT-3, so about 06:05 GMT-3 is the expected end). The user chose this thread on 2026-10-04, after E35, from the
open questions ADR 0014 left.

## Why

ADR 0014 made the decode stage show a row's real gaps as `[MASK]`, the token the induced scoring
shows them as, and E32 and E33 found it better on 11 and 13 of 21 variants and worse on none.
The pre-training stage still shows them as `[NULL]`, so a model trained under today's defaults
meets `[NULL]` at the gaps for 300 epochs and `[MASK]` for the 450 that follow. Aligning
pre-training too is the remaining piece of the same mismatch.

The honest prior is weak. The decode stage retrains the whole encoder for 450 epochs, and E05
found 300 pre-training epochs no better than 2, so whatever pre-training learns about the gap
token may already be overwritten. A null result is plausible and informative.

## Design (fixed before launch)

- **The change.** `PRETRAIN_GAP_TOKEN` (`--pretrain_gap_token`, `null` | `mask`) shows a row's
  real gaps as `[MASK]` in the pre-training stage's training and validation inputs, through the
  same `TabularEmbedder.gaps_as_mask` the decode stage uses. The targets and the loss positions
  (the epoch's mask, never a gap) do not move, and nothing is drawn. Default `null` for both
  tasks: this is a study, not a switch, so classification and every imputation default are
  unchanged. The flag is accepted for classification too, since pre-training is shared, like
  `--pretrain_objective`; the tag `pretrain_gap_token` is on every run of either task and was
  backfilled `null`.
- **Arm PG:** a plain imputation run under today's defaults (ADRs 0013 to 0015) with
  `--pretrain_gap_token mask`, reading the promoted file or the defaults as any run does; five
  folds.
- **Reference R, read rather than rerun:** E35's cells of the same variant and seed, as their K
  readout (`impute/*/induced_checkpoint/*`). With no decode patience the trajectory is the same
  whichever checkpoint is kept, so that readout equals a default run to the last digit: ADR 0015
  verified it at seed 101, and the pre-launch check below at seed 42. PG and R consume the same
  random stream (`gaps_as_mask` draws nothing), so a pair shares its folds, masks and
  initialisation and differs only by the pre-training token.
- **Variants:** all 21 of ADR 0007. **Seeds:** 42, 7, 13. **Size:** 63 new cells, tagged
  `experiment=pretrain-gap-token-2026-10-05`, `arm=PG`.
- **Integrity check:** each pair's baseline scores, which never depend on the model, must be
  equal; a mismatch means the pairs do not line up.
- **Running it:** four concurrent queues (`scripts/experiments/run_pretrain_gap_token_queue.sh
  1..4`), one trainer each, CPU threads capped at 4, cells assigned as in E34 and E35, which took
  5 h 51 min each, so about 5.8 h is expected. No cell starts after 11:00 GMT-3 on 2026-10-05
  (14:00 UTC), the end of the window the user gave for overnight runs; a cell not started by then
  is reported missing. Nothing is rerun or extended on its
  result; a crash is fixed, its cell rerun from the start, and the rerun noted here.
- **Checks before launch:** default runs of credit-g_40nan (promoted) and vehicle_20nan (no
  promoted file) at seed 42, after the change, must equal E35's K readouts on every induced and
  masked metric that family records, with equal baselines and checkpoint epochs; a smoke run of
  PG at seed 1, outside the study, must record `pretrain_gap_token mask` (its scores not read);
  the unit suite (mypy included) and both regression fixtures must pass.

## Measures

- **Primary:** per variant, PG − R on the induced `impute_score`, paired by seed and fold (15
  pairs), t-interval; a verdict only when the interval excludes zero and all three seeds' mean
  difference agree in sign. A negative difference favours PG.
- **Reading, fixed now, on all 21 variants:**
  1. **Any "R better":** keep `null`.
  2. **No "R better" and "PG better" on at least 3 variants** (the bar ADR 0015's checkpoint
     cleared in E34): PG is recommended as the imputation default, by an ADR (0016) and the
     user's decision; it would then be replicated on other seeds, as E33 and E35 did.
  3. **No "R better" and "PG better" on fewer than 3:** it qualifies but gains nothing worth a
     default; keep `null`.
- **Secondary, descriptive:** the difference by missing level; the masked population; the decode
  stage's chosen epoch under PG and R; means over the 21 variants and how many sit below the best
  baseline (R: 19, from E35); fold time and wall time against the plan.
- **Analysis:** `scripts/experiments/pretrain_gap_token_report.py` (`--tally` prints the reading,
  the counts and the integrity check).

## Known limits, stated in advance

- **R is read from another study.** Its equality with a default run is verified, not assumed, but
  it was run a day earlier on the same machine and code path; the pairs' integrity check guards it.
- **Under PG and ADR 0014 together, no stage ever shows `[NULL]`,** so `--score_null_path` is
  fully off-distribution for such a model.
- **Multiplicity:** 21 tests with the seed-agreement rule; pairs share their random stream, so
  the intervals are tighter than for independent draws.

## Running notes

- **Checks before launch, done:** the default runs of credit-g_40nan and vehicle_20nan at seed 42
  equal E35's K readouts on every recorded metric (14 and 10), with equal baselines (20 and 12)
  and equal checkpoint epochs; the smoke run of PG at seed 1 recorded `pretrain_gap_token mask`
  in its `hyperparameters.json`, its params and its tag (scores not read); the unit suite passes
  (305 tests, mypy included; one classification test's pinned tag set gained the new tag) and
  both regression fixtures pass.
- **Tag backfill applied** before launch, 4 min 37 s: 2140 earlier runs of both tasks stamped
  `null`; 2128 have a mirror carrying the tag, the rest being `FAILED` sources or under a deleted
  study, as for the earlier backfills.

## Outcome (2026-10-05, 06:20 GMT-3)

All 63 cells ran (00:16 to 06:08 GMT-3), none failed, none missing. The integrity check passed:
every pair's baseline scores are equal, so each PG cell lines up with its E35 cell. Numbers from
`scripts/experiments/pretrain_gap_token_report.py` (kept with `--tally` in the state directory).
Lower is better; a negative difference favours PG. Induced cells, 15 fold pairs per variant; the
epochs are the decode stage's chosen epochs, fold means.

| variant | PG | R | best baseline | PG − R (seeds 42, 7, 13) | verdict | epoch, R → PG |
|---|---|---|---|---|---|---|
| `credit-g_20nan` | 0.8813 | 0.8818 | 0.9180 | −0.0005 [−0.0098, +0.0088] (+0.0040, −0.0034, −0.0021) | none | 136 → 144 |
| `credit-g_40nan` | 0.9573 | 0.9556 | 0.9638 | +0.0017 [−0.0165, +0.0200] (−0.0258, +0.0345, −0.0035) | none | 63 → 71 |
| `credit-g_60nan` | 0.9914 | 0.9922 | 1.0000 | −0.0008 [−0.0101, +0.0085] (−0.0069, −0.0029, +0.0075) | none | 43 → 43 |
| `credit-g_80nan` | 1.0027 | 1.0079 | 1.0000 | −0.0052 [−0.0120, +0.0016] (−0.0068, −0.0007, −0.0081) | none | 14 → 11 |
| `kr-vs-kp_20nan` | 0.5874 | 0.5863 | 0.6119 | +0.0011 [−0.0098, +0.0121] (−0.0050, +0.0014, +0.0071) | none | 249 → 264 |
| `kr-vs-kp_40nan` | 0.7576 | 0.7582 | 0.7725 | −0.0006 [−0.0086, +0.0075] (−0.0043, +0.0034, −0.0008) | none | 130 → 152 |
| `kr-vs-kp_60nan` | 0.8675 | 0.8649 | 0.9030 | +0.0026 [−0.0046, +0.0098] (−0.0018, −0.0013, +0.0109) | none | 84 → 82 |
| `kr-vs-kp_80nan` | 0.9775 | 0.9759 | 1.0000 | +0.0017 [−0.0039, +0.0072] (+0.0080, +0.0005, −0.0036) | none | 22 → 16 |
| `spambase_20nan` | 0.8451 | 0.8481 | 0.8664 | −0.0029 [−0.0114, +0.0056] (−0.0127, +0.0016, +0.0023) | none | 295 → 330 |
| `spambase_40nan` | 0.9015 | 0.9030 | 0.9371 | −0.0015 [−0.0062, +0.0033] (+0.0023, −0.0056, −0.0010) | none | 260 → 252 |
| `spambase_60nan` | 0.9410 | 0.9422 | 0.9845 | −0.0012 [−0.0031, +0.0006] (−0.0033, +0.0005, −0.0009) | none | 184 → 165 |
| `spambase_80nan` | 0.9869 | 0.9871 | 1.0000 | −0.0002 [−0.0024, +0.0021] (+0.0025, −0.0006, −0.0025) | none | 55 → 51 |
| `vehicle_20nan` | 0.4197 | 0.4178 | 0.4731 | +0.0019 [−0.0042, +0.0079] (+0.0063, +0.0014, −0.0021) | none | 309 → 278 |
| `vehicle_60nan` | 0.6377 | 0.6449 | 0.7006 | −0.0072 [−0.0175, +0.0031] (−0.0136, −0.0125, +0.0045) | none | 225 → 213 |
| `biodeg_20nan` | 0.5833 | 0.5865 | 0.6356 | −0.0032 [−0.0139, +0.0075] (−0.0038, −0.0189, +0.0131) | none | 285 → 261 |
| `biodeg_60nan` | 0.7639 | 0.7605 | 0.8658 | +0.0033 [−0.0042, +0.0108] (+0.0058, −0.0014, +0.0056) | none | 299 → 294 |
| `kc2_20nan` | 0.5341 | 0.5414 | 0.5665 | −0.0073 [−0.0295, +0.0149] (−0.0223, −0.0040, +0.0043) | none | 195 → 206 |
| `kc2_60nan` | 0.5851 | 0.5723 | 0.7129 | +0.0128 [−0.0040, +0.0296] (+0.0101, +0.0304, −0.0021) | none | 227 → 248 |
| `pendigits_20nan` | 0.3342 | 0.3324 | 0.3500 | +0.0018 [−0.0001, +0.0036] (−0.0000, +0.0033, +0.0021) | none | 406 → 396 |
| `letter_20nan` | 0.4603 | 0.4593 | 0.4729 | +0.0010 [−0.0006, +0.0026] (+0.0015, −0.0009, +0.0024) | none | 421 → 408 |
| `electricity_20nan` | 0.6284 | 0.6179 | 0.6058 | +0.0105 [+0.0020, +0.0190] (+0.0130, +0.0102, +0.0083) | R better | 345 → 342 |

**Tally.** PG − R: R better on 1 variant (electricity_20nan, +0.0105 [+0.0020, +0.0190], all three
seeds), PG better on none, no detectable difference on 20.

**Reading, by the rule fixed in advance: keep `null`.** A variant shows "R better", and no variant
shows "PG better". Showing the gaps as `[MASK]` to pre-training adds nothing to ADR 0014's
decode-stage change and costs a little on electricity.

**Secondary measures.**
- **No effect at any missing level:** mean PG − R +0.0003 at 20nan, −0.0001 at 40nan, +0.0016 at
  60nan, −0.0012 at 80nan.
- **Means over the 21 variants:** PG 0.7450, R 0.7446; below the best baseline on 19 variants
  under both.
- **Masked population:** no detectable difference on all 21.
- **The decode stage's chosen epoch moves little** (for example credit-g_80nan 14 → 11,
  kr-vs-kp_40nan 130 → 152), with no pattern in direction.
- **Time.** Planned about 5.8 h of wall time, ending near 06:05 GMT-3; it took 5 h 52 min and ended
  at 06:08, 1.01x the plan. A PG cell costs the same as a default one.

**Reading against the prior.** The ticket expected a weak effect, since the decode stage retrains
the encoder for 450 epochs: what pre-training learns about the gap token does not survive it.
This is consistent with E05 (300 pre-training epochs no better than 2): the decode stage, not
pre-training, decides how the model treats a gap. The flag stays available, default `null`.
