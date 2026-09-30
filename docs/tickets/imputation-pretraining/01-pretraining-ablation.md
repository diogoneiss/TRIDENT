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

## Amendment 2026-09-29 (21:24 GMT-3): the remaining cells run on another machine

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

## Amendment 2026-09-29 (23:08 GMT-3): the last eight cells write to the SSD store

Written and committed before the batch resumed. The batch was paused at 22:13 GMT-3 at the
user's request, in the middle of `pendigits_20nan` seed 42 arm C; that run ended `FAILED`, is
not counted, and the cell runs from the start (no finished cell is rerun). Done on the server
by then: `kr-vs-kp_40nan` s13 A and `pendigits_20nan` s42 B.

Before resuming, the server store moved to the SSD in WAL mode
([ADR 0010](../../adr/0010-server-store-on-ssd-with-hdd-replica.md)) and run metrics are
written in batches ([ticket 0005](../0005-batched-mlflow-metric-writes.md), `0dde5e2`).
Ticket 0005's exactness check found every metric value, step, param, tag and artifact
identical between the old and the new code; only the `time/*` values and the metric
timestamps differ. The scores of the eight remaining cells are therefore comparable with the
25 finished ones.

Wall time is not. On the hard-disk store, s13 A spent about 13 of its 20.1 minutes and s42 B
about 6 of its 14.0 writing to MLflow. The cost measure for every arm is therefore the sum of
the fold `total_seconds` in `raw_fold_metrics.csv`, which is taken before the store write;
the run's wall clock is reported beside it, per machine and per store.

## Outcome (2026-09-30, 01:10 GMT-3)

All 32 cells of the design finished: 23 on the Windows RTX 3050, then `kr-vs-kp_40nan` s13 A
and pendigits' eight on gorgona8 (RTX 3090 Ti), plus pendigits' fresh A/42 there; the last
seven with the SSD store. Every A/42 is
the reused 2026-09-24/25 run except pendigits', which ran fresh on the server. Numbers from
`pretrain_ablation_report.py` (FINISHED, active, non-mirror runs tagged
`ablation=pretraining-2026-09-29`). Lower is better; a positive difference favours A.

**Primary: induced test cells, 15 fold pairs per comparison.**

| variant | A | B | C | B − A (seeds 42, 7, 13) | verdict | C − A (seeds 42, 7, 13) | verdict |
|---|---|---|---|---|---|---|---|
| `credit-g_20nan` | 0.906 | 0.905 | 0.915 | −0.0015 [−0.0173, +0.0142] (+0.0058, −0.0064, −0.0040) | no detectable difference | +0.0092 [−0.0047, +0.0230] (+0.0068, +0.0080, +0.0127) | no detectable difference |
| `credit-g_80nan` | 1.037 | 1.059 | 1.051 | +0.0218 [−0.0065, +0.0502] (+0.0360, +0.0061, +0.0234) | no detectable difference | +0.0141 [−0.0121, +0.0404] (+0.0355, −0.0065, +0.0134) | no detectable difference |
| `kr-vs-kp_40nan` | 0.769 | 0.780 | 0.788 | +0.0105 [−0.0049, +0.0258] (+0.0095, +0.0171, +0.0048) | no detectable difference | +0.0184 [+0.0046, +0.0323] (+0.0066, +0.0292, +0.0195) | **A better than C** |
| `pendigits_20nan` | 0.402 | 0.380 | 0.335 | −0.0216 [−0.0250, −0.0181] (−0.0264, −0.0165, −0.0217) | **pre-training hurts** | −0.0664 [−0.0701, −0.0627] (−0.0641, −0.0664, −0.0687) | **C better than A** |

**Secondary: masked test cells**, the same way: no detectable difference on either credit-g
variant; `kr-vs-kp_40nan` B − A +0.0136 [−0.0059, +0.0331], none, C − A +0.0255
[+0.0052, +0.0458], A better than C; `pendigits_20nan` B − A −0.0194 [−0.0220, −0.0167],
pre-training hurts, C − A −0.0636 [−0.0674, −0.0598], C better than A. The masked cells
agree with the induced ones on every variant.

**Each arm against the run's best baseline** (induced, mean [95% interval] over 15 folds):

| variant | best baseline | A | B | C |
|---|---|---|---|---|
| `credit-g_20nan` | `hgb` 0.918 [0.899, 0.938] | 0.906 [0.884, 0.929], overlaps | 0.905 [0.885, 0.924], overlaps | 0.915 [0.897, 0.934], overlaps |
| `credit-g_80nan` | `mean_mode` 1.000 | 1.037 [1.021, 1.053], loses | 1.059 [1.039, 1.079], loses | 1.051 [1.034, 1.068], loses |
| `kr-vs-kp_40nan` | `hgb` 0.773 [0.761, 0.784] | 0.769 [0.756, 0.783], overlaps | 0.780 [0.768, 0.792], overlaps | 0.788 [0.777, 0.799], overlaps |
| `pendigits_20nan` | `knn10` 0.350 [0.346, 0.354] | 0.402 [0.396, 0.407], loses | 0.380 [0.376, 0.384], loses | **0.335 [0.331, 0.340], beats the bar** |

Masked, pendigits C 0.371 [0.366, 0.376] also beats `knn10` 0.401 [0.396, 0.406]. It is the
first configuration in this effort to beat the best baseline on pendigits' induced cells; the
full-profile winner of 2026-09-28 (0.345 [0.335, 0.355]) overlapped it.

**Sensitivity, as the amendment of 2026-09-29 required.** kr-vs-kp's C − A verdict rests on
the declared cross-machine seed 13 (+0.0195): over its 10 same-machine pairs (seeds 42 and 7,
Windows) C − A is +0.0179 [−0.0034, +0.0393] induced and +0.0213 [−0.0088, +0.0514] masked,
both spanning zero. Every pendigits pair is same-machine.

**Machine gap, descriptive.** Besides the check cell (`credit-g_20nan` s42 B: per-fold sd of the
gap 0.044, mean −0.002), pendigits A/42 ran on both machines: server − Windows +0.0036 induced
(per-fold sd 0.0050) and +0.0046 masked (sd 0.0069). The gap is about nine times smaller on
the all-numerical table than on the mixed one.

**Cost (fold-time sum per cell, minutes; run wall clock differs only on the hard-disk store).**
Windows: credit-g A 4.4–6.0, B 2.7–3.3, C 7.2–8.1; kr-vs-kp A 17.4–20.7, B 10.8–11.5, C
31.8–35.4. Server: kr-vs-kp A 7.3 (20.1 wall on the hard disk); pendigits A 13.3–13.7, B
7.6–7.7 (s42 B 14.0 wall on the hard disk), C 21.1–21.7. C costs about 1.2x to 1.8x A.

**Reading.** No variant shows "pre-training helps" against no pre-training. Its worth depends
on the table: on the all-numerical pendigits, 300 epochs of the embedding objective cost 0.022
against none, and the same epochs spent on the value loss gain 0.066 and clear the best
baseline; on the all-categorical kr-vs-kp the same move loses 0.018, a verdict that rests on
the one cross-machine pair; on the two credit-g variants nothing is detectable. So the
embedding objective is not what makes pre-training worth its cost where it seems to help, and
on pendigits the decode stage is under-trained at 150 epochs. Limits as stated in advance:
four variants, three seeds, one configuration each, C equalises epochs, not steps or
schedules. Next: the value and normalised-embedding objectives
([02](02-pretraining-objectives.md), ADR 0011), run the same night.
