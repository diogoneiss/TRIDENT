# Does normalised-target pre-training still help once the decoder gets 450 epochs? (pre-registered 2026-09-30)

Written and committed before the first run, at 03:00 GMT-3, while study
[02](02-pretraining-objectives.md) was still running. Runs overnight on gorgona8 beside it;
no cell starts after 06:30 GMT-3.

## Why

In the pre-training ablation ([01](01-pretraining-ablation.md)), spending the 300
pre-training epochs on the decode loss instead (arm C, decode 450) beat every other arm on
`pendigits_20nan` and cleared the best baseline. In study 02's cells finished by 03:00, the
normalised embedding objective (N: layer-normalised target through a predictor head, ADR 0011)
was better than the current objective (A) on `pendigits_20nan` and `credit-g_80nan`, and
better than no pre-training on pendigits, but C still beat N on pendigits. N and C each change
one thing. This asks whether they add up: normalised pre-training followed by the long decode
stage.

## Question

With the decode stage at 450 epochs, does 300 epochs of normalised-target pre-training (M) do
better on the induced test cells than none (C)?

## Arms (fixed before launch)

| arm | `PRETRAIN_OBJECTIVE` | `EPOCHS_PRE` | `EPOCHS_DECODE` | source |
|---|---|---|---|---|
| **M** | `embedding_normalized` | as configured (300) | configured + pre-training (450) | run now |
| **C** | (none) | 0 | 450 | reference: study 02's server C cells; pendigits from the ablation's server cells |
| **N** | `embedding_normalized` | 300 | as configured (150) | reference: study 02 |

Each variant uses the configuration a run of it loads today, `--lr_scheduler cosine`, five
folds. Variants `credit-g_20nan`, `credit-g_80nan`, `kr-vs-kp_40nan`, `pendigits_20nan`; seeds
**42, 7, 13**; 12 new cells tagged `experiment=pretrain-decode-budget-2026-09-30`, `arm=M`.
Every reference ran on the same server. Two queues
(`scripts/experiments/run_pretrain_decode_budget_queue.sh a|b`): queue a = pendigits, then
`credit-g_20nan`; queue b = `kr-vs-kp_40nan`, then `credit-g_80nan`; seeds 42, 7, 13. A cell
not started by 06:30 GMT-3 is reported missing; nothing is rerun or extended on its result. If
a C reference from study 02 is missing at analysis time, its comparison is reported incomplete.

## Measures

- **Primary:** `impute/induced/impute_score` per fold, M − C, paired by seed and fold (15 pairs),
  t-interval, verdict only when the interval excludes zero and all three seeds agree in sign
  ("M better than C" / "C better than M" / "no detectable difference"), as in 01 and 02.
- **Secondary:** M − N (what the longer decode adds after normalised pre-training); the masked
  cells; each arm against the run's best baseline. Cost descriptive only (up to four trainers
  share the GPU tonight).
- Analysis: `scripts/experiments/pretrain_decode_budget_report.py`.

## Known limits, stated in advance

Decided after seeing part of study 02, so it is a follow-up, not a replication: its motivation
is post hoc, its design is not. One configuration per variant, `LR_PRE` at its configured value,
three seeds. M spends 750 epochs against C's 450, so "M better than C" confounds the objective
with the extra training; "no detectable difference" or "C better than M" would say the
pre-training stage does not earn its cost even in its better form.
