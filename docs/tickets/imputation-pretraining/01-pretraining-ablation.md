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

## Amendment 2026-09-30: the remaining cells run on another machine

Written and committed before any of the remaining cells ran, after 23 of the 32 cells had
finished on the Windows RTX 3050. The other ten run on the Linux server gorgona8
(RTX 3090 Ti), same code, same configuration files: the promoted `*.imputation.json` of
`credit-g_20nan` and `kr-vs-kp_40nan` are tracked, and `credit-g_80nan` and
`pendigits_20nan` have none on either machine, so both run the defaults.

**Check, run before this amendment.** `credit-g_20nan` seed 42 arm B, through this
ablation's runner with MLflow off (store untouched: 3730 runs before and after):

| | Windows (stored run) | server | server − Windows per fold |
|---|---|---|---|
| induced | 0.9186252677917454 | 0.9164988203285273 | mean −0.002, sd 0.044 |
| masked | 0.9341120073750269 | 0.9326723085964879 | mean −0.001, sd 0.041 |

The mean_mode, knn5 and knn10 baselines agree to the last digit on every fold, so folds,
evaluation masks and scaling are shared across machines (they are drawn on the CPU:
`KFold(random_state=seed)` and `evaluation_mask`'s own numpy stream). The model does not
reproduce: different GPUs lay out dropout's draws differently, so a server run is another
training draw of the same cell, with no sign of an offset in this one cell. The hgb
baseline differs on one fold of five (CPU, thread-count dependent), so a run's best
baseline is also per-machine. The per-fold spread of the machine gap (0.044) is of the
order of the per-pair spread of B − A on this variant (about 0.028), which is why the
pairs below are kept on one machine wherever it costs one cell.

**Pairing, decided now.**

- `pendigits_20nan`: arm A at seed 42 runs fresh on the server, tagged like every other
  cell, so all 15 of its pairs are same-machine. The reused Windows run
  `impute_pendigits_20nan_20260925_031545` leaves the verdict; its difference to the
  server's A/42 is reported in the Outcome as a second, descriptive measure of the machine
  gap (same configuration, folds and masks, an all-numerical variant). This adds one cell:
  the queue is ten cells, `kr-vs-kp_40nan 13 A` and then pendigits `42 B, 42 C, 42 A,
  7 B, 7 C, 7 A, 13 B, 13 C, 13 A`.
- `kr-vs-kp_40nan` seed 13 is the one declared cross-machine pair: B and C on Windows,
  A on the server. No finished cell is rerun. The verdict is still taken over all 15
  pairs; the 10 same-machine pairs are shown beside it as a sensitivity view only.
- The verdict rule is unchanged.
- Cells run one at a time on the server as on Windows. Wall time is reported per machine
  and never compared across them; on this check the server took 107 s of fold time
  against Windows' 197 s.
