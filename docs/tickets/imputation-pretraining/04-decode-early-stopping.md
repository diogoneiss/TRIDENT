# Does stopping the decode stage on a validation plateau lose anything? (pre-registered 2026-09-30)

Written and committed before the first run, at 13:15 GMT-3. Runs on gorgona8 beside study
[05](05-larger-batch.md); no cell starts after 06:30 GMT-3 on 2026-10-01.

## Why

The 450-epoch decode stage (arm C of [01](01-pretraining-ablation.md), arm M of
[03](03-normalised-pretraining-with-a-long-decode.md)) helps on `pendigits_20nan` and costs
time everywhere. Reading the logged validation curves (the overnight report, "Measured the
morning after"): the best decode epoch falls at 9–87 of 450 on credit-g and kr-vs-kp but
around 390 on pendigits, and a patience of 50 was the only value tried that would have kept
pendigits' gain, while saving 73–86% of the decode epochs elsewhere. `--decode_patience`
(`521df3d`) makes it possible.

## Question

With M's configuration, does stopping the decode stage after 50 epochs without a validation
improvement (P) score differently on the induced test cells than training all 450 (M)?

## Arms (fixed before launch)

- **P** = M plus `DECODE_PATIENCE` 50: `embedding_normalized` pre-training for the configured
  300 epochs, a decode stage of up to 450, cosine over 450, five folds.
- **M** = reference, study 03's cells. **N** = study 02's cells (the same pre-training, decode 150).
- Configuration: what M's cells read. For `credit-g_20nan` that is the promoted file removed
  on 2026-09-30, pinned from `fcaadfb` in `scripts/experiments/decode_patience_batch_run.py`.
- Variants `credit-g_20nan`, `credit-g_80nan`, `kr-vs-kp_40nan`, `pendigits_20nan`; seeds
  **42, 7, 13**; 12 new cells tagged `experiment=decode-early-stopping-2026-09-30`, `arm=P`.
  One queue (`scripts/experiments/run_decode_patience_batch_queue.sh p`), pendigits first,
  then kr-vs-kp, credit-g_80nan, credit-g_20nan.

## Measures

- **Primary:** `impute/induced/impute_score`, P − M paired by seed and fold (15 pairs),
  t-interval, verdict only when the interval excludes zero and all three seeds agree in sign
  (as in 01–03); and **decode epochs trained** per fold (`decode/epochs_trained`).
- **Secondary:** P − N; masked cells; best baseline; fold-time per cell (descriptive: three
  trainers share the GPU, and the baseline cache answers repeat folds).
- **Built-in check:** a cell in which no fold stops early must equal its M cell to the last
  digit.

## Known limits, stated in advance

- The schedule still spans 450 epochs, so a run that stops early stops with the learning rate
  still high; a verdict is about this implementation, not about early stopping in general.
- Early stopping changes how much of the global random stream a fold consumes, so after a fold
  that stops early, the later folds of P see different training draws from M's (their
  evaluation masks and splits stay the same). The fold pairing is weaker than in 01–03.
- The checkpoint criterion is the λ-weighted validation loss, which critique F-02-1 found only
  loosely tied to `impute_score`.
- One patience value, three seeds, one configuration per variant.

## Operational note, 2026-09-30 14:05 GMT-3: a crash at finalisation, fixed

The first P cell (`pendigits_20nan` s42) trained all five folds and then failed while
summarising them: the cross-validation summary refused folds whose decode stages ended at
different epochs, which is what early stopping does. The queue was stopped (the second cell,
s7, was interrupted at once), the summary fixed in `fa38425` (training untouched), and the queue
relaunched; both cells run again from the start. A crash is not a result, so this reruns no
finished cell.

## Outcome (2026-09-30, 16:00 GMT-3)

All 12 P cells ran (the two crashed ones again from the start after the fix above), none
missing. Numbers from `scripts/experiments/decode_patience_batch_report.py`. Lower is better; a
negative P − M favours P.

| variant | P − M, induced (primary) | verdict | decode epochs trained, median (range) of 450 | fold time per cell, M → P |
|---|---|---|---|---|
| `credit-g_20nan` | +0.0001 [−0.0122, +0.0125] | no detectable difference | 91 (65–171) | 4.1 → 1.5 min |
| `credit-g_80nan` | −0.0031 [−0.0158, +0.0095] | no detectable difference | 61 (51–85) | 5.9 → 1.4 min |
| `kr-vs-kp_40nan` | −0.0018 [−0.0163, +0.0126] | no detectable difference | 107 (79–135) | 18.5 → 4.9 min |
| `pendigits_20nan` | +0.0057 [+0.0006, +0.0108] (+0.0043, +0.0018, +0.0111) | **M better than P** | 388 (242–450) | 35.4 → 30.5 min |

**Secondary.** Masked cells: the same verdicts (pendigits +0.0057, M better; none elsewhere).
P − N: P better on pendigits (−0.0247, induced), no detectable difference elsewhere. Against the
best baseline: pendigits P 0.340 [0.334, 0.346] still beats `knn10` 0.350; elsewhere as M. Fold
time is descriptive (M ran four to a GPU, P one to three).

**The built-in check** (a cell with no early stop in any fold must equal M) had no cell to
apply to: every cell stopped at least one fold early. Its mechanism shows on fold 1, which still
shares M's random stream: its induced score is identical to M's to the last digit in 9 of 12
cells, exactly those where M's best epoch came before P's stop (the other three are pendigits
s42 and s13 and credit-g_20nan s42). Later folds differ, as stated in advance.

**Reading.** A patience of 50 keeps M's scores on the three smaller tables while training 61 to
107 of the 450 decode epochs, which cuts a kr-vs-kp cell from 18.5 to 4.9 minutes. On pendigits,
the one table whose validation loss is still falling late, it stops some folds as early as epoch
242 and costs 0.006 (still better than the 150-epoch N and still below `knn10`). So early
stopping with this patience is a safe speed-up where the long decode buys nothing, and a small
loss where it buys the most. Limits as stated in advance: one patience value, the schedule still
spans 450 epochs, and later folds see other training draws.
