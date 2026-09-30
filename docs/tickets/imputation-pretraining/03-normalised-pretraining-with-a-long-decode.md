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

## Outcome (2026-09-30, 05:00 GMT-3)

All 12 M cells ran (03:00–05:00 GMT-3, beside study 02 until 04:28), none missing, none rerun;
every C and N reference was present. Numbers from
`scripts/experiments/pretrain_decode_budget_report.py`. Lower is better; a negative M − X
favours M.

| variant | M | C | N | M − C (primary) | verdict | M − N (secondary) | verdict |
|---|---|---|---|---|---|---|---|
| `credit-g_20nan` | 0.908 | 0.914 | 0.910 | −0.0065 [−0.0158, +0.0029] | no detectable difference | −0.0021 [−0.0152, +0.0110] | none |
| `credit-g_80nan` | 1.031 | 1.036 | 1.026 | −0.0042 [−0.0195, +0.0111] | no detectable difference | +0.0057 [−0.0077, +0.0190] | none |
| `kr-vs-kp_40nan` | 0.773 | 0.794 | 0.765 | −0.0205 [−0.0373, −0.0037] (−0.0232, −0.0172, −0.0211) | **M better than C** | +0.0083 [−0.0060, +0.0226] | none |
| `pendigits_20nan` | 0.334 | 0.335 | 0.364 | −0.0013 [−0.0045, +0.0018] | no detectable difference | −0.0305 [−0.0324, −0.0285] (−0.0292, −0.0303, −0.0319) | **M better than N** |

Masked cells: no detectable M − C difference on any variant (kr-vs-kp −0.0288 [−0.0605,
+0.0030], all three seeds negative); M better than N on pendigits (−0.0341). Against the best
baseline: pendigits M 0.334 [0.331, 0.337] beats `knn10` 0.350 [0.346, 0.354], as C does;
kr-vs-kp M 0.773 [0.762, 0.785] overlaps `hgb` 0.773; credit-g_80nan loses to `mean_mode`;
credit-g_20nan overlaps `hgb`.

**Reading.** Normalised pre-training followed by the long decode stage is never worse than
either of its halves: on pendigits it matches C (the long decode is what matters there) and on
kr-vs-kp it beats C (pre-training is what matters there), where N alone and M are level. So
the two partial wins of the night do add up: no comparison run tonight found M worse than
anything, and it is the only configuration that is never detectably beaten by C or N. It was
not compared formally with A, B or V (not pre-registered); descriptively it sits within noise
of the best arm on each variant (kr-vs-kp A and N 0.765 against M 0.773; credit-g_80nan N
1.026 against 1.031). As stated in advance, M trains 750
epochs against C's 450, so "M better than C" on kr-vs-kp confounds the objective with the
extra training; E25's N − C on kr-vs-kp (−0.0288, N at 450 epochs in total) says the budget
alone does not explain it. One configuration per variant, three seeds.
