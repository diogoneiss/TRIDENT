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

## Outcome (2026-10-01, 00:08 GMT-3)

All 12 H cells ran (23:19–00:08 GMT-3, one trainer), none missing. Numbers from
`scripts/experiments/batched_heads_report.py`; a positive H − A favours the per-column heads.

| variant | H − A, induced (seeds 42, 7, 13) | verdict | H − A, masked | verdict | fold time per cell, A → H |
|---|---|---|---|---|---|
| `credit-g_20nan` | +0.0026 [−0.0062, +0.0114] (+0.0100, −0.0015, −0.0007) | no detectable difference | +0.0006 [−0.0079, +0.0090] | none | 6.9 → 1.0 min |
| `credit-g_80nan` | +0.0001 [−0.0012, +0.0014] (−0.0000, +0.0000, +0.0003) | no detectable difference | +0.0004 [−0.0017, +0.0025] | none | 2.4 → 1.1 min |
| `kr-vs-kp_40nan` | −0.0006 [−0.0101, +0.0089] (+0.0131, −0.0009, −0.0140) | no detectable difference | +0.0100 [−0.0051, +0.0252] | none | 7.9 → 3.0 min |
| `pendigits_20nan` | +0.0008 [−0.0000, +0.0016] (+0.0009, +0.0009, +0.0006) | no detectable difference | +0.0003 [−0.0005, +0.0011] | none | 13.4 → 10.4 min |

**Reading.** By the rule fixed in advance, "no detectable difference" on all four variants, so
the batched heads may become the default. One caveat, stated rather than ruled on: on
pendigits all three seeds lean the same way (+0.0006 to +0.0009) and the interval's lower bound
sits at zero; the size is 0.2% of the score and a tenth of the pendigits machine gap
(per-fold sd 0.005, study 01), so it is noise-sized but not proven to be noise. The fold times
include every speed-up since A's cells ran (the store on the SSD, the exact decoder fixes, the
epoch masks, the baseline cache), not the batched heads alone.
