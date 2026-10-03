# Choose the imputation decode checkpoint on the validation rows' own gaps by default

## Status

Accepted (2026-10-03, 19:20 GMT-3). The user decided this after study
[imputation-token-shape/04](../tickets/imputation-token-shape/04-checkpoint-and-calibration.md)
(E34), which measured task T07. Like ADRs 0006 and 0012 to 0014, it is a deliberate exception to
the rule that old behaviour stays the default. It amends ADR [0004](0004-imputation-decoder-task.md)
(how the decode checkpoint is chosen) and ADR [0013](0013-imputation-defaults-from-the-confirmatory-study.md)
(imputation's task defaults), for the imputation task only.

## Context

The decode stage kept the epoch of the lowest loss on a fixed validation mask. That mask hides a
share of the validation rows' visible cells, which shrinks as gaps grow: about 4% of an 80nan
row, some 80 cells on credit-g's validation split. The same split holds an order of magnitude
more of its own gaps, the population the headline scores (ADR 0008 already scores them for
searches). E06 found the two criteria disagree, and E32's P arm blew up on three folds of
spambase_60nan at the epoch the loss chose.

E34 kept a second checkpoint on the same trajectory, the epoch whose validation gaps score best,
and scored the test split there too (K), on the 21 variants and seeds 101, 202 and 303:

- K better than the headline on 3 variants (credit-g_80nan −0.020, kr-vs-kp_80nan −0.029,
  spambase_80nan −0.010), worse on none, which met the pre-stated rule;
- mean over the 21 variants 0.7519 → 0.7446, below the best baseline on 17 variants instead of 15;
- at 80nan the gap criterion stops earlier (credit-g epoch 25 → 12, kr-vs-kp 32 → 21, spambase
  92 → 61), where the loss on a few masked cells kept training into overfitting;
- watching it costs one forward pass an epoch, no measurable time.

## Decision

1. **The decode checkpoint is a named choice.** `Hyperparameters.decode_checkpoint`
   (`DECODE_CHECKPOINT`, `--decode_checkpoint`): `loss`, every run before it and the dataclass
   default, or `induced`, the epoch whose validation rows' own gaps score best against the
   complete sibling. The score is ADR 0008's `validation/impute/induced/impute_score`, computed
   each epoch from tensors prepared once per fold (`_ValidationGapScore` in
   `src/training/decoding.py`, which a test holds equal to the pandas path).
2. **Imputation chooses on the gaps.** `task_defaults("imputation")` gains `DECODE_CHECKPOINT
   induced`, beneath every source and flag. Classification has no decode stage.
3. **Without a complete table the run falls back to the loss, and says so.** A complete variant,
   or real gaps with no truth, leaves nothing to score. The runner then sets the criterion to
   `loss` before anything is recorded, so the parameters, the tag and the hyperparameter file
   name the criterion the run used.
4. **Early stopping counts from the kept criterion's best epoch** (`DECODE_PATIENCE`, off by
   default).
5. **The `decode_checkpoint` tag** is on every imputation parent and trial; earlier runs are
   stamped `loss` by `scripts/backfill_decode_checkpoint_tag.py` and marked
   `decode_checkpoint_backfilled=true`. Its mirrors are stamped in place, and a tree is mirrored
   from scratch only when it has no mirror yet. **Applied 2026-10-03, 19:28 to 19:33 GMT-3:** 1990
   runs stamped; 1980 have a mirror carrying the tag, and the 10 without are 9 `FAILED` sources
   (not mirrored, ADR 0006) and one trial under a deleted study.
6. **The five promoted `*.imputation.json` files** name `DECODE_CHECKPOINT induced`, the
   configuration E34's K readout scored on those variants.
7. **`--decode_checkpoint loss` recovers the old behaviour**; the command for every imputation
   default before ADRs 0013 to 0015 is in the README.

## Verification

- **Default runs match E34's K.** Runs at seed 101 through `main.py` with no flag equal E34's K
  readouts (the `induced_checkpoint` family) of the same cells to the last digit, on every induced
  and masked metric that family records: 14 on credit-g_40nan (a promoted file), 10 on
  vehicle_20nan (the defaults); both record `DECODE_CHECKPOINT induced`.
- **Regression fixtures:** both pass unmodified; the credit-g imputation test asserts that its
  mapping, read without a task, stays on `loss`.
- **Check cell:** the handoff kit's `pretrain_ablation_run.py` now pins `DECODE_CHECKPOINT loss`
  and holds its reference (induced 0.8863449097353776, masked 0.9256312571312231).
- **Unit suite:** 299 tests pass, mypy included; the fallback without a complete table, the
  equality of the per-epoch score with the pandas path, and the in-place mirror stamping each
  have a test.

## Consequences

- **The validation gaps are now spent on model selection.** A search ranked by the induced
  validation objective (ADR 0008's `induced` population) would choose trials on the same cells
  that chose each trial's checkpoint, an optimistic score; the default search objective is
  `masked` and is unaffected. The test split is never touched.
- **`--score_induced_checkpoint`** repeats the headline under the new default, so its
  `induced_checkpoint` family is scored only under `loss`; the two epochs
  (`decode/induced_checkpoint_epoch`, `decode/loss_checkpoint_epoch`) are recorded whenever the
  gap score is watched, the run's own criterion included.
- **One more key to pin when re-running an old arm:** an arm that leaves the checkpoint to the
  default now keeps the gap criterion's epoch; pin `DECODE_CHECKPOINT loss` beside ADR 0014's
  list. The promoted files as of `16e3a09` are the ones before this decision.
- **The gain is concentrated at 80nan** and the effect is small elsewhere; E34 found no variant
  where it hurts, on three seeds. It has not been replicated on other seeds.
