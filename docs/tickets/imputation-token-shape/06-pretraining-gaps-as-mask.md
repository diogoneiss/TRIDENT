# Does showing the gaps as [MASK] to pre-training help too? (pre-registered 2026-10-05)

This ticket was written and committed before the first run (the commit and the queues' start
time are in the Running notes). The user chose this thread on 2026-10-04, after E35, from the
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
