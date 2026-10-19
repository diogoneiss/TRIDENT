# A deployable calibration: fitted on the validation mask, read with H (E47; pre-registered 2026-10-18)

This ticket was written and committed before the first cell.

- **Why now:** ADR 0027 made the calibrated output (KC) the imputation default at 40% missing and
  above. KC inherits K's dependence on the validation gaps' truth: both are fitted on cells a
  deployed model cannot see.
  - E43 named H, the loss's checkpoint, the deployable headline.
  - The deployable analogue of KC is H after a calibration fitted on the fixed validation mask's
    cells. Those cells are observed, so a deployed model can fit it.
  - It is untested (T20's follow-up in `docs/TASKS.md`). The user agreed to test it on 2026-10-18.
- **Code:** a new diagnostic flag, `--score_masked_calibration`, off by default. It logs
  `impute/induced/masked_calibrated/*` and `impute/masked/masked_calibrated/*` and touches nothing
  else; a unit test holds every other key, the model and the losses equal to a run without it.
  No default changes.

## Design (fixed before launch)

### The readouts, all on one trajectory per run

Runner: `scripts/experiments/deployable_calibration_run.py`; tag
`experiment=deployable-calibration-2026-10-18`.

| readout | what it is | where it is logged |
|---|---|---|
| H | the loss's checkpoint, uncalibrated (the headline) | `impute/<pop>/*` |
| HC_m | H after a calibration fitted on the validation mask's cells | `impute/<pop>/masked_calibrated/*` |
| HC_g | H after the calibration fitted on the validation gaps (an oracle) | `impute/induced/calibrated/*` |
| K | the validation gaps' checkpoint | `impute/<pop>/induced_checkpoint/*` |
| KC | K after the gaps' calibration (ADR 0027's output) | `impute/induced/induced_checkpoint_calibrated/*` |

The runs pin `--decode_checkpoint loss` and `--decode_calibration none`; neither changes training.

- **The calibration** is ADR 0027's, unchanged (`src/training/calibration.py`):
  - each numerical column shrunk toward the training mean by a fitted α, with a prior of 30 cells;
  - the categorical confidence thresholds, on columns with at least 30 fit cells.
- **Only the cells it is fitted on differ.** The validation mask hides about 20% of a row's
  observed cells, so it holds far fewer cells than the validation gaps at heavy missingness (about
  4% of an 80nan row). The prior then keeps α nearer 1, and few categorical columns reach 30 cells.
  This is stated in advance as the likely limit.

### Cells

- **Variants:** E40's 12 at 40% missing and above.
  - kr-vs-kp, spambase and credit-g at 40, 60 and 80nan;
  - vehicle, biodeg and kc2 at 60nan.
- **Seeds:** E40's, 404, 505 and 606: 36 cells.
  - **Why reuse them:** K and KC on these cells must equal E40's fold by fold. That is the
    integrity check (`--integrity`).
  - HC_m and H were never read on these seeds; E40 read only KC − K.
- **Running it:** four queues (`scripts/experiments/run_deployable_calibration_queue.sh`), CPU threads
  capped at 4.
  - The queues balance about 145 minutes each.
  - No cell starts after 08:00 GMT-3 on 2026-10-19; unstarted cells are reported missing.

## Measures

All readings are on `impute_score` over 15 fold pairs, with the verdict rule and the 1% SESOI
(ADR 0020), through `scripts/experiments/deployable_calibration_report.py --tally`.

**Primary:** HC_m − H on the induced population, per variant.

**Secondary (descriptive):**

- HC_g − H, and the share of HC_g's gain that HC_m keeps;
- HC_m − H on the masked population, where the fit is in-distribution;
- KC − K, which replicates E40 by construction;
- HC_m − K, the deployable candidate against today's oracle checkpoint;
- H and HC_m against the weak bar and the strong tier;
- the mean α of each fit, and the validation mask's cells per fold.

## Reading, fixed now

**Primary, on E40's six named variants** (kr-vs-kp_80nan, spambase_60nan, spambase_80nan,
credit-g_60nan, credit-g_80nan, vehicle_60nan):

- **supported** if HC_m is better than H on at least 4 of the 6 named and worse on none of the 12;
- **refuted** if HC_m is worse on at least 2 of the 12, or better on at most 1 of the named;
- **mixed** otherwise.

**What each reading leads to (a proposal, never a default change without the user):**

- **supported:** propose a deployable calibration mode (for example `DECODE_CALIBRATION masked`)
  by an ADR amending 0027, for runs whose claims must transfer;
- **refuted or mixed:** ADR 0027 stays as it is. KC is reported as an oracle readout beside H, as K
  already is.

**If the integrity check fails** (a difference above 1e-6 on K or KC), nothing is read until it is
explained.

## Known limits (note added 2026-10-18, about 23:20 GMT-3, before any score was read)

- **These seeds decided E40.** ADR 0020 retires a seed set once it has decided something, and
  404/505/606 decided that KC beats K. They are reused here only so that K and KC can be held equal
  to E40.
  - So E47 is a development reading.
  - A "supported" result would need fresh seeds before any ADR proposes a deployable mode.
- **This supersedes part of D10** of the plan of 2026-10-17 ("no calibration fitted on masked cells
  for now"), by the user's agreement on 2026-10-18.

## Launch

- **Checked before launch, with `--no-mlflow`, passed (2026-10-18, about 23:06 GMT-3):** kc2_60nan at
  seed 404. Its K and KC equal E40's headline and calibrated readout fold by fold (largest
  difference 0). The validation mask holds 60–65 cells a fold there. No score was read.
- **Launched at 23:07 GMT-3** on 7756f01: four queues of 9 cells in
  `/scratch2/diogoneiss/trident-2026-10-18-deployable-calibration/`, progress in its `progress.log`.

## Outcome (2026-10-19, 01:39 GMT-3)

**Run.** All 36 cells ran from 23:07 GMT-3 on 2026-10-18 to 01:37 on 2026-10-19, before the 08:00
cutoff. None failed.

**Integrity holds:** K and KC equal E40's cells fold by fold on all 12 variants (largest difference
0). The check first stopped on kr-vs-kp, which has no numerical metric; it was fixed to skip a
metric a run does not carry, before the tally was read.

```text
kr-vs-kp_40nan: largest difference from E40 0
  kr-vs-kp_60nan: largest difference from E40 0
  kr-vs-kp_80nan: largest difference from E40 0
  spambase_40nan: largest difference from E40 0
  spambase_60nan: largest difference from E40 0
  spambase_80nan: largest difference from E40 0
  credit-g_40nan: largest difference from E40 0
  credit-g_60nan: largest difference from E40 0
  credit-g_80nan: largest difference from E40 0
  vehicle_60nan: largest difference from E40 0
  biodeg_60nan: largest difference from E40 0
  kc2_60nan: largest difference from E40 0
```

**The printout** (`scripts/experiments/deployable_calibration_report.py --tally`):

- Each reading is a mean difference over the 15 fold pairs of `impute_score` (lower is better), its
  95% interval and the verdict with the 1% SESOI.
- "keeps" is HC_m's gain over H as a share of HC_g's.

```text
- `kr-vs-kp_40nan`: HC_m − H -0.0007 [-0.0029, +0.0016] equivalent; HC_g − H -0.0010 [-0.0077, +0.0057] none; masked: HC_m − H +0.0046 [+0.0017, +0.0074] worse; KC − K +0.0001 [-0.0041, +0.0043] equivalent; HC_m − K +0.0056 [+0.0005, +0.0107] worse; H − bar -0.0320 [-0.0413, -0.0227] better; HC_m − bar -0.0326 [-0.0420, -0.0233] better; HC_m − strong -0.0632 [-0.0703, -0.0561] better; HC_m keeps 64% of HC_g's gain; mean alpha ; validation mask cells per fold 874
- `kr-vs-kp_60nan`: HC_m − H +0.0000 [+0.0000, +0.0000] equivalent; HC_g − H -0.0103 [-0.0165, -0.0040] better; masked: HC_m − H +0.0000 [+0.0000, +0.0000] equivalent; KC − K -0.0029 [-0.0076, +0.0018] equivalent; HC_m − K +0.0101 [-0.0017, +0.0218] none; H − bar -0.0297 [-0.0409, -0.0185] better; HC_m − bar -0.0297 [-0.0409, -0.0185] better; HC_m − strong -0.0341 [-0.0487, -0.0195] better; HC_m keeps -0% of HC_g's gain; mean alpha ; validation mask cells per fold 483
- `kr-vs-kp_80nan` (named): HC_m − H +0.0000 [+0.0000, +0.0000] equivalent; HC_g − H -0.0282 [-0.0375, -0.0189] better; masked: HC_m − H +0.0000 [+0.0000, +0.0000] equivalent; KC − K -0.0083 [-0.0120, -0.0046] better; HC_m − K +0.0226 [+0.0106, +0.0346] worse; H − bar +0.0016 [-0.0098, +0.0129] none; HC_m − bar +0.0016 [-0.0098, +0.0129] none; HC_m − strong +0.0016 [-0.0098, +0.0129] none; HC_m keeps -0% of HC_g's gain; mean alpha ; validation mask cells per fold 339
- `spambase_40nan`: HC_m − H -0.0010 [-0.0024, +0.0004] equivalent; HC_g − H -0.0019 [-0.0032, -0.0006] better; masked: HC_m − H -0.0005 [-0.0034, +0.0024] equivalent; KC − K -0.0012 [-0.0031, +0.0008] equivalent; HC_m − K +0.0012 [-0.0039, +0.0062] equivalent; H − bar -0.0297 [-0.0377, -0.0218] better; HC_m − bar -0.0307 [-0.0385, -0.0229] better; HC_m − strong -0.0358 [-0.0423, -0.0292] better; HC_m keeps 52% of HC_g's gain; mean alpha masked_calibrated 0.841, calibrated 0.784; validation mask cells per fold 1909
- `spambase_60nan` (named): HC_m − H -0.0017 [-0.0037, +0.0003] equivalent; HC_g − H -0.0042 [-0.0069, -0.0014] better; masked: HC_m − H -0.0027 [-0.0048, -0.0007] better; KC − K -0.0020 [-0.0033, -0.0006] better; HC_m − K +0.0031 [+0.0017, +0.0045] worse; H − bar -0.0387 [-0.0449, -0.0324] better; HC_m − bar -0.0404 [-0.0471, -0.0336] better; HC_m − strong -0.0131 [-0.0206, -0.0055] better; HC_m keeps 41% of HC_g's gain; mean alpha masked_calibrated 0.871, calibrated 0.713; validation mask cells per fold 924
- `spambase_80nan` (named): HC_m − H -0.0032 [-0.0043, -0.0021] better; HC_g − H -0.0109 [-0.0152, -0.0066] better; masked: HC_m − H -0.0060 [-0.0127, +0.0008] none; KC − K -0.0054 [-0.0073, -0.0034] better; HC_m − K +0.0022 [-0.0006, +0.0050] equivalent; H − bar -0.0083 [-0.0153, -0.0013] better; HC_m − bar -0.0115 [-0.0178, -0.0051] better; HC_m − strong -0.0099 [-0.0159, -0.0040] better; HC_m keeps 29% of HC_g's gain; mean alpha masked_calibrated 0.896, calibrated 0.528; validation mask cells per fold 511
- `credit-g_40nan`: HC_m − H -0.0000 [-0.0004, +0.0004] equivalent; HC_g − H -0.0024 [-0.0086, +0.0037] equivalent; masked: HC_m − H -0.0007 [-0.0018, +0.0004] equivalent; KC − K -0.0016 [-0.0055, +0.0023] equivalent; HC_m − K +0.0204 [+0.0044, +0.0364] worse; H − bar -0.0170 [-0.0348, +0.0009] none; HC_m − bar -0.0170 [-0.0349, +0.0009] none; HC_m − strong +0.0151 [+0.0010, +0.0292] worse; HC_m keeps 1% of HC_g's gain; mean alpha masked_calibrated 0.938, calibrated 0.856; validation mask cells per fold 174
- `credit-g_60nan` (named): HC_m − H -0.0011 [-0.0017, -0.0005] better; HC_g − H -0.0151 [-0.0218, -0.0085] better; masked: HC_m − H -0.0006 [-0.0011, -0.0001] better; KC − K -0.0061 [-0.0099, -0.0023] better; HC_m − K +0.0086 [-0.0050, +0.0222] none; H − bar -0.0004 [-0.0140, +0.0132] none; HC_m − bar -0.0015 [-0.0150, +0.0120] none; HC_m − strong +0.0103 [-0.0032, +0.0238] none; HC_m keeps 7% of HC_g's gain; mean alpha masked_calibrated 0.955, calibrated 0.693; validation mask cells per fold 120
- `credit-g_80nan` (named): HC_m − H -0.0004 [-0.0006, -0.0001] better; HC_g − H -0.0134 [-0.0221, -0.0047] better; masked: HC_m − H -0.0004 [-0.0007, -0.0001] better; KC − K -0.0040 [-0.0055, -0.0024] better; HC_m − K +0.0130 [-0.0019, +0.0279] none; H − bar +0.0246 [+0.0078, +0.0415] worse; HC_m − bar +0.0242 [+0.0075, +0.0410] worse; HC_m − strong +0.0242 [+0.0079, +0.0405] worse; HC_m keeps 3% of HC_g's gain; mean alpha masked_calibrated 0.956, calibrated 0.574; validation mask cells per fold 101
- `vehicle_60nan` (named): HC_m − H -0.0021 [-0.0033, -0.0009] better; HC_g − H -0.0100 [-0.0130, -0.0069] better; masked: HC_m − H -0.0026 [-0.0045, -0.0008] better; KC − K -0.0081 [-0.0110, -0.0052] better; HC_m − K +0.0025 [-0.0061, +0.0110] none; H − bar -0.0589 [-0.0755, -0.0423] better; HC_m − bar -0.0610 [-0.0780, -0.0440] better; HC_m − strong -0.0110 [-0.0311, +0.0090] none; HC_m keeps 21% of HC_g's gain; mean alpha masked_calibrated 0.979, calibrated 0.915; validation mask cells per fold 98
- `biodeg_60nan`: HC_m − H -0.0025 [-0.0032, -0.0018] better; HC_g − H -0.0096 [-0.0150, -0.0041] better; masked: HC_m − H -0.0035 [-0.0066, -0.0004] better; KC − K -0.0047 [-0.0084, -0.0011] better; HC_m − K +0.0119 [-0.0042, +0.0280] none; H − bar -0.0742 [-0.0888, -0.0597] better; HC_m − bar -0.0767 [-0.0909, -0.0626] better; HC_m − strong +0.0236 [+0.0096, +0.0377] worse; HC_m keeps 26% of HC_g's gain; mean alpha masked_calibrated 0.967, calibrated 0.864; validation mask cells per fold 171
- `kc2_60nan`: HC_m − H -0.0004 [-0.0034, +0.0026] equivalent; HC_g − H -0.0027 [-0.0171, +0.0118] none; masked: HC_m − H -0.0026 [-0.0083, +0.0032] none; KC − K +0.0018 [-0.0065, +0.0102] none; HC_m − K +0.0271 [-0.0036, +0.0578] none; H − bar -0.1625 [-0.2515, -0.0736] better; HC_m − bar -0.1629 [-0.2493, -0.0766] better; HC_m − strong -0.0114 [-0.0767, +0.0539] none; HC_m keeps 15% of HC_g's gain; mean alpha masked_calibrated 0.989, calibrated 0.927; validation mask cells per fold 63

Primary, HC_m − H on the induced impute_score: better on 4 of the 6 named; worse on 0 of 12 (none) -> supported (fold-pair t, seed agreement)
All calls: better 5 / no difference 0 / equivalent 7 / worse 0 (fold-pair t, seed agreement)
```

**Primary, by the rule fixed in advance: supported.**

- HC_m is better than H on 4 of the 6 named variants: spambase_80nan, credit-g_60nan, credit-g_80nan
  and vehicle_60nan. It is also better on biodeg_60nan.
- It is worse on none of the 12; the other 7 are equivalent.

**But the gain is small, and smaller than the oracle's:**

- **Its size:** HC_m − H is −0.0004 to −0.0032 where it has a verdict, at most 0.4% of H.
- **The share of HC_g's gain it keeps, on the named variants with a verdict:**
  - spambase_80nan 29%;
  - vehicle_60nan 21%;
  - credit-g_60nan 7%;
  - credit-g_80nan 3%.
- **On kr-vs-kp at 60nan and 80nan, HC_m equals H exactly.** The table is all categorical, and no
  categorical column reaches the 30 fit cells its threshold needs: the validation mask holds about
  480 and 340 cells a fold over 36 columns.
- **Why:** the validation mask holds few cells, from 63 a fold on kc2_60nan to 1,909 on
  spambase_40nan. The 30-cell prior then keeps α near 1: the mean α of the masked fit is 0.84–0.99,
  against 0.53–0.93 for the gaps' fit. This was the limit stated in advance.
- **HC_m stays below K, the oracle,** on kr-vs-kp_40nan, kr-vs-kp_80nan, spambase_60nan and
  credit-g_40nan, and has no verdict or is equivalent elsewhere.
- **On the masked population** (in-distribution for the fit), HC_m is better on 5 and worse on
  kr-vs-kp_40nan (+0.0046).
- **Against the bars,** HC_m's calls equal H's.

**What the rule leads to, with the known limits:**

- The rule's "supported" would lead to proposing a deployable calibration mode by an ADR.
- The known limits come first: these seeds decided E40, so a replication on fresh seeds would come
  before any ADR.
- And the effect is a few tenths of a percent at most: real, but too small to change any verdict
  against the bars.

**Decided by the user (2026-10-19): (a).** Stop here. KC is reported as an oracle readout, with
HC_m's small deployable gain stated beside it. No deployable mode and no replication.

**The options that were offered:**

- **(a)** Stop here and report KC as an oracle readout, with HC_m's small deployable gain stated
  (recommended).
- **(b)** Replicate on fresh seeds (36 cells, about 2.5 h), then an ADR for a deployable mode.
- **(c)** A new design that gives the deployable fit more cells, such as a lower prior or a second,
  larger fit mask, pre-registered as its own study.

