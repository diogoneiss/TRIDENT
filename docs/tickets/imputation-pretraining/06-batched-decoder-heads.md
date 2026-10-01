# Do batched decoder heads change anything but the training draw? (pre-registered 2026-09-30)

Written and committed before the first run, at 23:18 GMT-3 on 2026-09-30. Decides whether
`--decoder_heads batched` (`c9be71f`) can become the default (task T18).

## Why

The batched heads apply the same parameters with the same arithmetic as the per-column heads,
regrouped into a few operations: the loss and every gradient agree to rounding on a mixed table,
and a cell runs 1.4–1.6x faster. Rounding still sends training down another path, as a run on
another GPU does, so a single cell cannot tell "another draw" from "a systematic change". This
check pairs the two on the cells the studies already hold.

## Question

With arm A's configuration, does the batched decoder score differently on the induced test cells
than the per-column one?

## Arms (fixed before launch)

- **H** = arm A of studies 01–02 (the `embedding` objective, the configured 300 pre-training and
  150 decode epochs, cosine, five folds) with `DECODER_HEADS` batched.
- **A** = the per-column reference on gorgona8: study 02's A cells, and for pendigits and
  kr-vs-kp seed 13 the pre-training ablation's server A cells (the ones study 02 reused). The
  code since then is exact on the per-column path (the check cell, kr-vs-kp A s42 and
  yesterday's pendigits s42 B reproduce to the last digit).
- Configuration: what A's cells read; for `credit-g_20nan` the promoted file removed on
  2026-09-30, pinned from `fcaadfb` in `scripts/experiments/batched_heads_run.py`.
- Variants `credit-g_20nan`, `credit-g_80nan`, `kr-vs-kp_40nan`, `pendigits_20nan`; seeds
  **42, 7, 13**; 12 new cells tagged `experiment=batched-heads-2026-10-01`, `arm=H`; one queue
  (`scripts/experiments/run_batched_heads_queue.sh h`), pendigits first.

## Measures

- **Primary:** `impute/induced/impute_score`, H − A paired by seed and fold (15 pairs),
  t-interval; a verdict only when the interval excludes zero and all three seeds agree in sign,
  as in 01–05.
- **Secondary:** masked cells; fold time per cell (descriptive).
- **Reading, fixed now:** "no detectable difference" on all four variants licenses making
  `batched` the default (behind its tag, as the user decides); a verdict either way on any
  variant means the regrouping does more than change the draw, and the default stays.

## Known limits, stated in advance

Four variants, three seeds, one configuration; the check can only show that a difference, if
any, is smaller than this design detects (about 0.01 on these variants).
