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
| **V** | `value` | as configured (300) | the stage trained on the hidden values, heads discarded |
| **N** | `embedding_normalized` | as configured (300) | layer-normalised target read through a predictor head |

How V differs from the ablation's arm C: V is a separate stage with pre-training's own
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
  So every A and B cell is taken from the server: the ablation's server cells are reused
  (all six pendigits A and B cells, and `kr-vs-kp_40nan` s13 A), and the other 17 are run
  now under this study's tag. Tags: `experiment=pretrain-objective-2026-09-30`, `arm`.
- **Two concurrent queues**, one trainer each, from the checkout root
  (`scripts/experiments/run_pretrain_objective_queue.sh`), in this order:
  queue 1 = the A references (credit-g ×2 × 3 seeds, kr-vs-kp s42 and s7), then V on every
  variant and seed; queue 2 = N on every variant and seed, then the B references (credit-g
  ×2 × 3 seeds, kr-vs-kp × 3 seeds). Variants in the order above, seeds 42, 7, 13.
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
- **Secondary:** V − B and N − B the same way; the masked test cells the same way; each arm
  against the run's best baseline.
- **Cost, descriptive only:** fold-time sum per arm. Two trainers share the GPU, and profiling
  runs may share it too, so cost is not comparable with the ablation's.
- Analysis: `scripts/experiments/pretrain_objective_report.py`.

## Checked before launch

To be filled in before the first cell, from the runs named here:

- The unit suite (with the typing gate) and both integration fixtures pass on the merged code.
- The ablation's check cell (`credit-g_20nan` s42 B, MLflow off) on the merged code gives
  induced 0.9164988203285273 and masked 0.9326723085964879, as before the change, so the
  default path is unchanged on this GPU and the reused server cells are valid references.

## Known limits, stated in advance

Four variants, three seeds, one configuration per variant, `LR_PRE` at its configured value;
every full-profile winner chose a pre-training rate 2 to 34 times below the default, so a null
bounds these objectives at this rate, not the objectives. V and N add a predictor or heads
that A does not have, so a difference is the objective and its head together. `embedding`
has no EMA teacher, the third usual anti-collapse ingredient (ADR 0011, considered options).
