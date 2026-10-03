# Show the real gaps as [MASK] in the imputation decode stage by default

## Status

Accepted (2026-10-03, 00:30 GMT-3). The user decided this after study
[imputation-token-shape/02](../tickets/imputation-token-shape/02-gaps-as-mask-in-training.md)
(E32), which closed task T02. Like ADRs 0006, 0012 and 0013, it is a deliberate exception to the
rule that old behaviour stays the default. It amends the defaults stated in ADRs
[0004](0004-imputation-decoder-task.md) (how a gap is shown) and
[0013](0013-imputation-defaults-from-the-confirmatory-study.md) (imputation's task defaults), for
the imputation task only. A replication on fresh seeds was pre-registered the same night
([imputation-token-shape/03](../tickets/imputation-token-shape/03-replication-on-fresh-seeds.md),
E33).

## Context

ADR 0004 shows a real gap to the model as `[NULL]` and scores it as `[MASK]`, so the decode stage
trains on one row shape and is scored in another. E31 found that asking the gaps in the trained
shape (one column at a time, the rest `[NULL]`) scores better where gaps are many and a little
worse where they are few. E32 aligned the other side: with `DECODE_GAP_TOKEN mask` the decode
stage shows every real gap as `[MASK]` on its training, validation and masked-test inputs, and
nothing else changes.

On the 21 variants, three seeds each, against ADR 0013's defaults:

- better on 11 variants and worse on none; on the 9 variants at 60nan or more, better on 5 and
  worse on none, which met the pre-stated rule and its clause for a new default;
- better at every missing level (mean difference −0.011 at 20nan, −0.026 at 40nan, −0.036 at
  60nan, −0.012 at 80nan), where E31's column-wise scoring cost a little at 20nan;
- mean score over the 21 variants 0.7745 → 0.7542, below the best baseline's 0.7781, and below
  the best baseline on 16 variants instead of 11;
- a cell costs the same.

## Decision

1. **Imputation shows gaps as `[MASK]` in the decode stage.** `task_defaults("imputation")` in
   `src/training/types.py` gains `DECODE_GAP_TOKEN mask`, beneath every source and every flag
   like the rest of ADR 0013's layer. The dataclass default stays `null`, so classification and
   any mapping read without a task (the regression fixture) are unchanged.
2. **`--decode_gap_token null` recovers the old behaviour.** ADR 0013's command for the defaults
   before both decisions becomes
   `--pretrain_objective embedding --decode_epochs 150 --lr_scheduler cosine_legacy --decode_gap_token null`.
3. **The five promoted `*.imputation.json` files name the value.** Each gains
   `DECODE_GAP_TOKEN mask`, which is what E32's arm G ran on those variants; the rest of each file
   is unchanged. A promoted file names every value (ADR 0005, decision 5).
4. **No new tag.** `decode_gap_token` is on every imputation parent and trial since E32, and
   earlier runs were stamped `null` by `scripts/backfill_decode_gap_token_tag.py`.

## Verification

- **Default runs match E32's arm G.** Imputation runs at seed 42 through `main.py` with no
  configuration flag equal E32's G cells to the last digit on every `impute/*` metric: 55 metrics
  on credit-g_40nan (a promoted file) and 35 on vehicle_20nan (the defaults).
- **Regression fixtures:** both pass unmodified. The credit-g imputation test gained one
  assertion: its mapping, read without a task, stays on `null` (with ADR 0013's `embedding` and
  `cosine_legacy`).
- **Check cell:** the handoff kit's `pretrain_ablation_run.py` now pins `DECODE_GAP_TOKEN null`
  beside ADR 0013's objective pin, and its reference holds: induced 0.8863449097353776, masked
  0.9256312571312231.
- **Typing gate:** the unit suite, mypy included, passes (284 tests). It had been failing since
  the commit that added the flag (`4058764`), on the type of a cache in `gaps_as_mask` and in two
  study reports; fixed here with no change to any value.

## Consequences

- **The masked ranking metric is another system's.** `impute/masked/*`, which ranks folds and is
  the default search objective, now sees the test rows' gaps as `[MASK]`. A masked score from
  before and one from after are two systems on the same cells. Filter on `tags.decode_gap_token`.
  The induced headline's definition does not change: it always showed the gaps as `[MASK]`.
- **The token diagnostics describe the old model.** `--score_null_path` and `--score_column_wise`
  show gaps as `[NULL]`, which a model trained under `mask` never sees. They remain defined, but
  they measure what they were built for only with `--decode_gap_token null`.
- **One more key to pin when re-running an old arm.** Besides ADR 0013's list (the promoted files
  as of `f147c39`, the objective, the decode length), an arm that leaves the gap token to the
  default now trains with `mask`; pin `DECODE_GAP_TOKEN null`. The two runners someone would
  rerun are affected: `scripts/experiments/confirmatory_run.py` (every arm) and
  `scripts/experiments/column_wise_run.py` (E31's integrity check depended on it reproducing
  E30's M). The promoted files as of `c7d67ed` are the ones before this decision.
- **Pre-training still shows gaps as `[NULL]`.** Whether aligning that stage helps too is untested.
- **The three variants at the mean/mode fill stay there.** credit-g_60nan and _80nan and
  kr-vs-kp_80nan are at or above 1.0 under both tokens; the shape was not their only problem.
- **Searches hold the new value.** An imputation Optuna study holds `DECODE_GAP_TOKEN` at `mask`
  unless the flag says otherwise, and promotes it.
