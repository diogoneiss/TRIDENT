# What does pre-training contribute to imputation? (pre-registered 2026-09-29)

Written and committed before the first run.

## Why

The pre-training stage regresses the transformer's output at a masked cell onto the
detached clean *embedding* of that cell, not its value. The critique found the resulting
encoder 4.2x to 7.7x worse than a per-column constant at predicting those embeddings, and
no detectable difference on induced cells between 300 and 2 pre-training epochs
(F-13-1, F-13-2, `docs/reviews/imputation-critique/13-pretraining-transfer.md`). Every
winner of the full-profile searches chose a pre-training learning rate 2 to 34 times below
the default ([01](../imputation-optuna-full/01-overnight-full-profile-studies.md)). Before
fixing the stage, measure what it is worth.

## Question

With everything else held, does the imputation model do better on the induced test cells
with the default pre-training (300 epochs) than with none, and than with none but its
epochs given to the decode stage instead?

## Arms (fixed before launch)

Each variant uses the configuration a run of it loads today (the promoted
`*.imputation.json` where one exists, the defaults otherwise), `--lr_scheduler cosine`,
five folds; only the two epoch counts change.

| arm | `EPOCHS_PRE` | `EPOCHS_DECODE` | what it tests |
|---|---|---|---|
| **A** | as configured (300) | as configured (150) | the current pipeline |
| **B** | 0 | as configured (150) | no pre-training at all |
| **C** | 0 | configured decode + pre-training (450) | the same epochs, all spent on the value loss |

C is the cheap stand-in for a value-space pre-training objective: its extra epochs train the
encoder on the values themselves, through the decoder heads, with the decode stage's own
mask rate and learning rate.

- Variants, in this order: `credit-g_20nan`, `credit-g_80nan`, `kr-vs-kp_40nan`,
  `pendigits_20nan` (mixed at low and high missingness, all-categorical, all-numerical).
- Seeds: **42, 7, 13**, all three for every arm and variant; no seed added after the fact.
  Arm A at seed 42 reuses the runs of 2026-09-24/25 (same configuration, same code path for
  the model, same folds and masks); every other cell is run now, tagged
  `ablation=pretraining-2026-09-29` and `arm=A|B|C`.
- One pass; every finished cell reported; nothing rerun or extended on its result.
- Checked before launch: arm A at seed 42 on `credit-g_20nan`, rerun through this
  ablation's runner against a scratch store, reproduced the stored run to the last digit
  (induced 0.9127970009304829, masked 0.9263761033615078), so the reuse is exact.

## Measures

- **Primary:** `impute/induced/impute_score` per fold. For each variant, B − A and C − A
  paired by seed and fold (15 pairs), with a t-interval over the 15 pairs (approximate:
  folds of one seed share training data) and the sign of each seed's mean difference.
  "Pre-training helps" (A better than B) only where the interval excludes zero in A's
  favour **and** all three seeds agree in sign; "no detectable contribution" where the
  interval spans zero; "pre-training hurts" in the mirror case. The same rule for C − A.
- **Secondary:** the masked test cells, the same way; each arm against the same run's best
  baseline.
- **Cost:** wall time per arm.

## Known limits, stated in advance

Four variants, three seeds, one configuration per variant. C equalises epochs, not optimizer
steps or learning-rate schedules. A null result bounds what the current pre-training adds
under this configuration; it does not show that no pre-training objective could help.
