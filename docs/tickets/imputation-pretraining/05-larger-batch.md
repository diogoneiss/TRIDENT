# Does a larger batch with a scaled learning rate keep M's scores? (pre-registered 2026-09-30)

Written and committed before the first run, at 13:15 GMT-3. Runs on gorgona8 beside study
[04](04-decode-early-stopping.md); no cell starts after 06:30 GMT-3 on 2026-10-01.

## Why

A training step is launch-bound (overnight report § 5): the GPU mostly waits for small kernels.
Timed on an idle GPU, batch 1024 runs a pendigits decode epoch 2.1x faster than batch 256, and
a kr-vs-kp one 1.55x. A batch four times larger makes four times fewer updates per epoch, so
the learning rate has to change with it; the two usual rules are the square-root rule (x2)
and the linear one (x4).

## Question

With M's configuration, does batch 1024 with both learning rates doubled (Q2) or quadrupled
(Q4) score differently on the induced test cells than batch 256 (M)?

## Arms (fixed before launch)

- **Q2** = M with `BATCH` 1024 and `LR_PRE`, `LR_DECODE` x2. **Q4** = the same with x4.
  Weight decay unchanged (AdamW's is decoupled, so its effective strength scales with the
  learning rate). Everything else as M: `embedding_normalized` pre-training 300 epochs, decode
  450, cosine, five folds.
- **M** = reference, study 03's cells. Configuration as study 04 (`credit-g_20nan` pinned from
  `fcaadfb`).
- Variants and seeds as study 04; 24 new cells tagged `experiment=larger-batch-2026-09-30`,
  `arm=Q2|Q4`. Two queues (`run_decode_patience_batch_queue.sh b2` and `b4`), pendigits
  first, then kr-vs-kp, credit-g_80nan, credit-g_20nan.

## Measures

- **Primary:** `impute/induced/impute_score`, Q2 − M and Q4 − M, paired by seed and fold
  (15 pairs), t-interval, verdict only when the interval excludes zero and all three seeds
  agree in sign.
- **Secondary:** Q4 − Q2; masked cells; best baseline; fold-time per cell, the point of the
  study, though descriptive (three trainers share the GPU; compare against M's own cells,
  which ran two to four to a GPU).

## Known limits, stated in advance

Two scaling rules, no tuning around them; one configuration per variant; three seeds. A null
("no detectable difference") would license the larger batch for speed at this configuration
only; a loss would say the learning rate needs more than a rule.
