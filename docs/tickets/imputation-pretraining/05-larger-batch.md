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

## Outcome (2026-09-30, 15:27 GMT-3)

All 24 cells ran (13:16–15:27 GMT-3, three trainers on the GPU), none missing, none rerun.
Numbers from `scripts/experiments/decode_patience_batch_report.py`. Lower is better; a
negative X − M favours X.

**Primary: induced test cells, 15 fold pairs.**

| variant | Q2 − M (batch 1024, LR x2) | verdict | Q4 − M (batch 1024, LR x4) | verdict |
|---|---|---|---|---|
| `credit-g_20nan` | −0.0033 [−0.0157, +0.0091] | no detectable difference | +0.0190 [+0.0016, +0.0364] (+0.0252, +0.0086, +0.0231) | **M better than Q4** |
| `credit-g_80nan` | +0.0146 [−0.0145, +0.0437] | no detectable difference | +0.0095 [−0.0099, +0.0289] | no detectable difference |
| `kr-vs-kp_40nan` | +0.0115 [+0.0017, +0.0212] (+0.0192, +0.0012, +0.0140) | **M better than Q2** | +0.0012 [−0.0098, +0.0122] | no detectable difference |
| `pendigits_20nan` | −0.0035 [−0.0056, −0.0013] (−0.0038, −0.0048, −0.0018) | **Q2 better than M** | −0.0107 [−0.0127, −0.0087] (−0.0115, −0.0111, −0.0096) | **Q4 better than M** |

**Secondary.** Q4 − Q2: Q2 better on credit-g_20nan (+0.0223), Q4 better on pendigits
(−0.0073), no detectable difference on the other two. Masked cells: Q2 and Q4 better than M on
pendigits (−0.0029, −0.0107), no other masked verdict. Against the best baseline: pendigits Q4
0.323 [0.319, 0.328] beats `knn10` 0.350 by the widest margin of the effort; elsewhere as M.

**Cost** (fold-time sum per cell, descriptive; M ran four to a GPU, Q two to three): pendigits
35.4 → 21.9 min, kr-vs-kp 18.5 → 13.5, credit-g_80nan 5.9 → 3.7, credit-g_20nan 4.1 → 3.9.

**Reading.** A larger batch is not a free speed-up: which learning-rate rule keeps M's scores
depends on the table. With five folds the training split holds about 700 rows on credit-g,
2,240 on kr-vs-kp and 7,690 on pendigits, so batch 1024 means one step per epoch on credit-g
against three, three against nine on kr-vs-kp, and eight against 31 on pendigits. Only on
pendigits, which keeps eight steps an epoch, does the larger batch help, and there the linear
rule (x4) helps most (−0.011, the best pendigits score so far). On the small tables the
quadrupled rate hurts credit-g_20nan and the doubled one hurts kr-vs-kp. Limits as stated in
advance: two rules, no tuning around them, three seeds.
