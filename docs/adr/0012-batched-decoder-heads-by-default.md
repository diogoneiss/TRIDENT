# Apply the decoder's per-column heads in batched operations by default

## Status

Accepted (2026-10-01), the user's decision after the check of
[imputation-pretraining/06](../tickets/imputation-pretraining/06-batched-decoder-heads.md). A
deliberate exception to the rule that old behaviour stays the default, as ADR 0006's mirroring
was: training numbers move by a training draw, not by a change of model.

## Context

The decoder (`TridentDecoder`) holds one head per column and applied them one column at a time,
a few small GPU kernels each, in a step that is launch-bound. `--decoder_heads batched`
(`c9be71f`) applies the same parameters with the same arithmetic in a few batched operations:
one matmul and a masked gather for the categorical logits, one batched product for the numerical
MLPs. It gives the per-column loss and every gradient to rounding, and falls back to the
per-column path on a batch that leaves a column without hidden cells, so a head never gets a zero
gradient where it used to get none. A decode epoch runs 1.3x to 1.8x faster, a cell 1.4x to 1.6x.

Rounding still sends training down another path, so the two are different draws of the same
model. The check paired arm A with batched heads against the per-column A cells on four variants
and three seeds: no detectable difference on any of them, induced or masked (pendigits leans
+0.0008 on all three seeds, its interval's lower bound at zero, an order of magnitude below its
machine gap).

## Decision

1. **`DECODER_HEADS` defaults to `batched`** (`--decoder_heads`, `Hyperparameters.decoder_heads`).
2. **`per_column` stays available** and reproduces every run recorded before this decision to the
   last digit; pass `--decoder_heads per_column` to re-run or extend an old study.
3. **The `decoder_heads` tag records the mode** on every imputation parent and trial; runs before
   the flag were stamped `per_column` by `scripts/backfill_decoder_heads_tag.py`. Comparisons that
   pair a new run with an old one pair two draws, like a pair across machines.
4. **The check cell's reference moves.** The pre-training ablation's check cell
   (`pretrain_ablation_run.py credit-g_20nan 42 B ... --no-mlflow`) now gives induced
   0.8863449097353776 and masked 0.9256312571312231; with `--decoder_heads per_column` (through a
   configuration override) it still gives 0.8865289951105849 and 0.9198916682895757.

## Consequences

- Classification is untouched: it has no decoder (only the `value` pre-training objective trains
  one).
- Both regression fixtures pass unmodified within their recorded tolerance, so none was
  regenerated.
- Studies that reuse earlier per-column cells as references should pin `per_column` for their new
  arms, or run their references again.
