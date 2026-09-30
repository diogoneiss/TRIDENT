# Let pre-training regress a masked cell onto its value, or onto a normalised embedding

## Status

Accepted (2026-09-30) as a flag. `embedding`, the objective of every run before this
decision, stays the default, so every run launched as before trains as before. Whether
either new objective should become the imputation default is left to the pre-registered
study [imputation-pretraining/02](../tickets/imputation-pretraining/02-pretraining-objectives.md).

## Context

Pre-training (`TridentPretrainer`) regresses the transformer's output at a `[MASK]` cell
onto that cell's detached clean embedding, with an un-normalised MSE and no predictor head.
The imputation critique found that the loss falls mostly because the output norm shrinks
onto the target, not because the encoder learns what the hidden cell holds: the trained
encoder is 4.2x to 7.7x worse than a per-column constant at predicting those embeddings
(F-13-1), and 300 or 2 pre-training epochs make no detectable difference on the induced
cells (F-13-2). Its verifier added that the embedder also shrinks its own discriminative
variance, which normalising the loss would not stop and a value-space anchor would.

The pre-training ablation ([01](../tickets/imputation-pretraining/01-pretraining-ablation.md))
measures what the stage is worth as it is. Its arm C already spends the pre-training epochs
on the decode loss, continuously; it is no substitute for a pre-training stage whose target
is the value, because it shares the decode stage's learning rate, weight decay, schedule
cycle and checkpoint selection.

## Decision

1. **`--pretrain_objective {embedding,value,embedding_normalized}`**, and the same choice
   as `PRETRAIN_OBJECTIVE` in a configuration file or override mapping. The flag wins over
   the configuration, as `--lr_scheduler` does. Default `embedding`. An unknown name is
   refused at parse time and by `Hyperparameters`.
2. **`value`** trains a `TridentDecoder` (fresh decoder heads on the shared embedder and
   transformer) through the pre-training loop: pre-training's own learning rate, weight
   decay, epochs and schedule cycle, masks drawn at `PROB_MASCARA` as in both stages, and
   the decode stage's own value loss (cross-entropy plus
   `LAMBDA_NUM` times MSE, each averaged over its own hidden cells). The heads are
   discarded with the stage; the decode stage builds its own, as it does after `embedding`.
   The stage still hands over only the embedder and the transformer.
3. **`embedding_normalized`** keeps the detached clean embedding as the target, but
   layer-normalises each target cell over its dimensions (no affine parameters) and reads
   the transformer through a two-layer predictor head (`Linear`, `GELU`, `Linear`, width
   `DIM`), which is discarded with the stage. A predictor that says nothing scores 1 at any
   embedding scale, so the loss can no longer fall by rescaling.
4. **Tag `pretrain_objective`** on every parent and Optuna trial run, the resolved value.
   Runs recorded before it are stamped `embedding` by
   `scripts/backfill_pretrain_objective_tag.py`, marked `pretrain_objective_backfilled=true`,
   and re-mirrored. It is a tag, not a logged parameter: classification's parameter record
   stays column-for-column what it was.
5. **Not in any search space.** No Optuna profile samples the objective; a study runs
   under the one its configuration or flag names.

## Consequences

- Classification with the default is unchanged: the default path builds the same
  `TridentPretrainer` in the same order. The two regression fixtures pass unmodified, and
  a GPU check cell reproduced the ablation's server numbers (study 02, § Checked before
  launch).
- `value` costs more per epoch than `embedding` on tables with many columns, since the
  decoder's loss loops over columns with a host synchronisation each.
- The objective is available to classification runs too, untested there.

## Considered options

- **An EMA teacher for the embedding target** (data2vec, BYOL). The third standard
  anti-collapse ingredient beside stop-gradient and a predictor; left for later, since it
  adds a second copy of the encoder and a decay rate to tune.
- **Keep the value heads as the decoder's initialisation.** Would mix two effects, a
  better encoder and pre-trained heads; discarding them keeps the stage seam and the
  question the ablation asked.
- **A combined embedding plus value loss.** One more weight to choose with no evidence
  yet that either term helps alone.
