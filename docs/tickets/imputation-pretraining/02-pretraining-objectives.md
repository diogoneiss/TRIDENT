# Does a value target or a normalised embedding target make pre-training useful? (pre-registered 2026-09-30)

Written and committed before the first run. Runs overnight on gorgona8 (RTX 3090 Ti,
store on the SSD, ADR 0010), unattended until 07:00 GMT-3.

## Why

Pre-training regresses the transformer's output at a masked cell onto the cell's detached
clean embedding. The critique found that objective lets the loss fall by shrinking rather
than by predicting (F-13-1), and 300 or 2 pre-training epochs made no detectable
difference on induced cells (F-13-2). The pre-training ablation
([01](01-pretraining-ablation.md)) measures the stage as it is; its arm C, the pre-training
epochs spent on the decode loss, is the cheap stand-in for a value target and so far loses
to the current pre-training on `kr-vs-kp_40nan` (A better than C). [ADR 0011](../../adr/0011-selectable-pretraining-objective.md)
adds two objectives behind `--pretrain_objective`, one per direction the user asked for
on 2026-09-29: a pre-training architecture that cannot be met by rescaling, and a target
aligned with values.

## Question

With everything else held, does pre-training with the value objective (`value`) or with
the normalised embedding objective (`embedding_normalized`) do better on the induced test
cells than the current objective, and than no pre-training?

## Arms (fixed before launch)

Each variant uses the configuration a run of it loads today (the promoted
`*.imputation.json` where one exists, the defaults otherwise), `--lr_scheduler cosine`,
five folds, `EPOCHS_DECODE` as configured (150). Only the pre-training changes.

| arm | `PRETRAIN_OBJECTIVE` | `EPOCHS_PRE` | what it tests |
|---|---|---|---|
| **A** | `embedding` | as configured (300) | the current pipeline (reference) |
| **B** | (none) | 0 | no pre-training (reference) |
| **C** | (none) | 0, with `EPOCHS_DECODE` = configured decode + pre-training (450) | the ablation's arm C, the same epochs all on the decode loss (reference) |
| **V** | `value` | as configured (300) | the stage trained on the hidden values, heads discarded |
| **N** | `embedding_normalized` | as configured (300) | layer-normalised target read through a predictor head |

Arm C was added on 2026-09-30 after the ablation's outcome and before any cell of this study
ran: on `pendigits_20nan` C beat both A and the best baseline, so whether V beats *C* is now the
question that decides between fixing pre-training and dropping it. V and C spend the same 450
epochs on the value loss; V splits them into two stages. How V differs from C: V is a separate stage with pre-training's own
learning rate (`LR_PRE`, 3.4e-4 by default), weight decay and cosine cycle, followed by a
fresh decode stage with new heads and a checkpoint chosen on validation; C is one longer
decode stage. V ≈ C would say that the stage boundary adds nothing; V better than A would
say that the value target is what the current stage lacks.

- Variants, as in the ablation: `credit-g_20nan`, `credit-g_80nan`, `kr-vs-kp_40nan`,
  `pendigits_20nan`. Seeds **42, 7, 13** for every arm and variant; no seed added after the
  fact.
- **Same-machine references.** The ablation's A and B cells ran on the Windows GPU except
  for the pendigits ones and `kr-vs-kp_40nan` s13 A, and a cross-machine pair adds a
  training-draw gap with a per-fold sd of about 0.044 (ablation, amendment of 2026-09-29).
  So every A, B and C cell is taken from the server: the ablation's server cells are reused
  (all nine pendigits A, B and C cells, and `kr-vs-kp_40nan` s13 A), and the other 26 are run
  now under this study's tag. Tags: `experiment=pretrain-objective-2026-09-30`, `arm`.
- **Two concurrent queues**, one trainer each, from the checkout root
  (`scripts/experiments/run_pretrain_objective_queue.sh`), in this order:
  queue 1 = the A references (credit-g ×2 × 3 seeds, kr-vs-kp s42 and s7), then V on every
  variant and seed; queue 2 = N on every variant and seed, then the C references, then the
  B references (credit-g ×2 × 3 seeds and kr-vs-kp × 3 seeds each). Variants in the order
  above, seeds 42, 7, 13. Two trainers can finalise into the store at once: it is in WAL mode
  with MLflow's 20 s busy timeout, and a run writes its metrics in a handful of batches.
- **Cutoff by the clock, not by a result:** no cell starts after 06:30 GMT-3 on
  2026-09-30. A cell not started by then is reported as missing and may run the next night
  under this same design; nothing is rerun or extended on its result.

## Measures

- **Primary:** `impute/induced/impute_score` per fold. For each variant, V − A and N − A,
  paired by seed and fold (15 pairs), with a t-interval over the pairs (approximate: folds of
  one seed share training data) and the sign of each seed's mean difference. "X better than
  A" only where the interval excludes zero in X's favour **and** all three seeds agree in
  sign; "A better than X" in the mirror case; otherwise "no detectable difference". The same
  rule as the ablation.
- **Secondary:** V − C and N − C, then V − B and N − B, the same way; the masked test cells
  the same way; each arm against the run's best baseline. The reading of V − C is fixed now:
  V ≈ C says the stage boundary and pre-training's own rate add nothing, so the next step is
  to drop pre-training for imputation and lengthen the decode stage; V better than C says a
  separate value stage is worth having; C better than V says to train the decoder longer.
- **Cost, descriptive only:** fold-time sum per arm. Two trainers share the GPU, and profiling
  runs may share it too, so cost is not comparable with the ablation's.
- Analysis: `scripts/experiments/pretrain_objective_report.py`.

## Checked before launch

- The unit suite (248 tests, with the typing gate) and both integration fixtures pass on the
  merged code (`71c6a9b`).
- The ablation's check cell (`credit-g_20nan` s42 B, MLflow off) on the merged code on the GPU
  gives induced 0.9164988203285273 and masked 0.9326723085964879, the same as before the
  change, with the store untouched (3791 runs before and after). The default path is unchanged
  on this GPU, so the reused server cells are valid references.
- The decoder change in `71c6a9b` gives all 56 fold metrics identical on the CPU under the old
  and new code (three-fold `credit-g_20nan`, 8 + 8 epochs, seed 7).

## Known limits, stated in advance

Four variants, three seeds, one configuration per variant, `LR_PRE` at its configured value;
every full-profile winner chose a pre-training rate 2 to 34 times below the default, so a null
bounds these objectives at this rate, not the objectives. V and N add a predictor or heads
that A does not have, so a difference is the objective and its head together. `embedding`
has no EMA teacher, the third usual anti-collapse ingredient (ADR 0011, considered options).

## Operational note, 2026-09-30 01:40 GMT-3: CPU thread limit

After the first cell of each queue, three processes (the two trainers and a check cell)
were fitting the baseline imputers at once, each with 32 OpenMP/BLAS threads on the 32 cores,
and a credit-g fold's scoring stalled for about ten minutes. The fills of all four baseline
imputers are identical to the last value with 1, 4 and 32 threads (`credit-g_20nan` and
`pendigits_20nan`, fold 1, seed 42; `baseline_threads.py` in the night's state folder), and
the model trains on the GPU, so no score depends on the thread count. Both queues were
stopped between cells and relaunched with `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and
`OPENBLAS_NUM_THREADS` at 4 (now set by the launcher). Cost stays descriptive only, as stated
above. At the same stop, `6dd07a5` was merged: the decoder indexes each column by its hidden
rows, found once per batch, instead of by a boolean mask that synchronises with the device per
column. It selects the same cells in the same order; the check cell (`credit-g_20nan` s42 B,
GPU) still gives induced 0.9164988203285273 and masked 0.9326723085964879, and a three-fold
CPU run matches the old code on all 56 fold metrics. Cells finished before the stop: A and N
at `credit-g_20nan` seeds 42 and 7.

## Outcome (2026-09-30, 04:28 GMT-3)

All 50 cells of the design ran on gorgona8 (26 new A/B/C references and 24 V/N cells, plus the
ablation's 10 reused server cells), none missing, none rerun. Numbers from
`scripts/experiments/pretrain_objective_report.py`. Lower is better; a negative X − Y favours X.

**Primary: induced test cells, 15 fold pairs** (per-seed means in the report's output).

| variant | V − A | verdict | N − A | verdict |
|---|---|---|---|---|
| `credit-g_20nan` | −0.0002 [−0.0138, +0.0134] | no detectable difference | +0.0060 [−0.0074, +0.0193] | no detectable difference |
| `credit-g_80nan` | +0.0267 [+0.0057, +0.0477] | **A better than V** | −0.0147 [−0.0283, −0.0010] | **N better than A** |
| `kr-vs-kp_40nan` | +0.0233 [+0.0080, +0.0386] | **A better than V** | +0.0001 [−0.0117, +0.0118] | no detectable difference |
| `pendigits_20nan` | −0.0421 [−0.0446, −0.0397] | **V better than A** | −0.0373 [−0.0397, −0.0348] | **N better than A** |

**Secondary, induced.**

| variant | V − C | V − B | N − C | N − B |
|---|---|---|---|---|
| `credit-g_20nan` | −0.0106 [−0.0280, +0.0069], none | −0.0116 [−0.0279, +0.0047], none | −0.0044 [−0.0167, +0.0079], none | −0.0054 [−0.0228, +0.0119], none |
| `credit-g_80nan` | +0.0315 [+0.0127, +0.0502], **C better** | +0.0124 [−0.0146, +0.0394], none | −0.0099 [−0.0244, +0.0046], none | −0.0290 [−0.0503, −0.0076], **N better** |
| `kr-vs-kp_40nan` | −0.0056 [−0.0211, +0.0099], none | +0.0042 [−0.0084, +0.0167], none | −0.0288 [−0.0456, −0.0120], **N better** | −0.0191 [−0.0298, −0.0084], **N better** |
| `pendigits_20nan` | +0.0243 [+0.0205, +0.0281], **C better** | −0.0206 [−0.0236, −0.0175], **V better** | +0.0291 [+0.0259, +0.0324], **C better** | −0.0157 [−0.0183, −0.0132], **N better** |

**Masked cells.** V − A: A better on kr-vs-kp (+0.0319), V better on pendigits (−0.0377),
none on either credit-g. N − A: N better on pendigits (−0.0290), none elsewhere. N − B: N
better on kr-vs-kp (−0.0283) and pendigits (−0.0096). N − C: N better on kr-vs-kp (−0.0375),
C better on pendigits (+0.0346). No masked verdict contradicts an induced one.

**Against the best baseline** (induced): no V or N cell set clears it. `credit-g_20nan`: every
arm overlaps `hgb` 0.918. `credit-g_80nan`: every arm loses to `mean_mode` 1.000 (N comes
closest, 1.026 [1.015, 1.036]). `kr-vs-kp_40nan`: A 0.765 and N 0.765 overlap `hgb` 0.773
[0.761, 0.784]. `pendigits_20nan`: only C (0.335) beats `knn10` 0.350; V 0.360 and N 0.364
lose to it.

**Cost (fold-time sum per cell, descriptive; two to four trainers shared the GPU).** V costs
most where tables are wide or long (pendigits 29.5 min against A's 13.4), since its
pre-training runs the decoder's per-column loss; N costs what A costs.

**Reading.**
- *The value target (V) is not what pre-training lacks.* It beats the current objective only on
  the all-numerical pendigits, loses to it on credit-g_80nan and kr-vs-kp, and never beats C.
  Read with V − C fixed in advance: V ≈ C on credit-g_20nan and kr-vs-kp, C better on
  credit-g_80nan and pendigits, so a separate value stage is not worth having; on pendigits
  the decoder simply wants more epochs.
- *The normalised target (N) is.* It is never worse than the current objective, better on two
  variants, and better than no pre-training on three of four (credit-g_80nan, kr-vs-kp,
  pendigits), which the current objective is on none (study 01). On kr-vs-kp it also beats C.
  (The ablation's "A better than C" there rested on a cross-machine pair; on the server A 0.765
  against C 0.794 points the same way, descriptively.)
- *The decode budget is variant-specific.* C is best on pendigits and worst on kr-vs-kp.
- Limits as stated in advance: four variants, three seeds, `LR_PRE` at its default, N's
  predictor comes with its objective. Follow-up already pre-registered: N with the 450-epoch
  decode stage ([03](03-normalised-pretraining-with-a-long-decode.md)).
