# T28's D2: collapse and linear probes on the pre-training stage, and the encoder's drift (E53; pre-registered 2026-10-20)

This ticket was written and committed before the first study cell.

- **Why:** T28's report ([2026-10-19](../../reports/2026-10-19-t28-why-pretraining-adds-little.md),
  §3 H1, H2, H4 and H5, §5 step 2) asks the question that ADR 0011 and T13 both assume and nobody
  has measured under today's normalised objective (N). Does pre-training collapse onto the
  column-constant part of its target, and does the encoder end up knowing anything about the
  hidden value?
  - It also takes the drift half of D3: how far today's decode stage moves the encoder, from a
    pre-trained start and from a random one.
- **The user's go:** 2026-10-20, in the afternoon, with D3 (E52, [10](10-frozen-encoder.md)).
- **Offline:** `scripts/experiments/pretrain_probe.py` runs `train_pretrainer` itself and only
  watches it. Nothing is written to the store or to `src/`.
  - **How it watches:** a tracker computes the diagnostics after each epoch's own validation
    pass, in eval mode, with every random stream saved and restored.
  - **Checked before launch:** `fidelity` on credit-g_20nan, seed 101, on the CPU. The watched
    stage's 300 validation losses equal a plain call's exactly (largest difference 0), with
    diagnostics at every epoch.
  - **Smoked on credit-g_20nan at seed 7,** not a study cell.
    - The first logistic probe, at C = 1 on one mask, read worse than the mode (1.09). The probes
      were changed before any study cell: four training masks, and C chosen per column by 3-fold
      CV on the training cells.
    - The drift mode ran end to end.

## Design (fixed before the first cell)

**What one cell is:** the first of the five folds of the variant at one seed, under today's
configuration as a default run reads it, plus the setting's change.

- **probe cells** (on the CPU, 24 in all):
  - **N** (none) and **A** (`PRETRAIN_OBJECTIVE embedding`, the shipped objective), on
    credit-g_20nan, kr-vs-kp_40nan and pendigits_20nan;
  - **N_lr4** (`LR_PRE` ÷ 4) and **N_b64** (`BATCH 64`), on credit-g_20nan only (H5);
  - seeds 101, 202 and 303.
  - **Each cell runs the stage twice from the same seed:**
    - with `EPOCHS_PRE 0`: the Xavier start, the same as E52's FR;
    - and in full: diagnostics every 10 epochs and at the last.
- **drift cells** (on the GPU, after E52's queues end, 8 in all):
  - **N** and **R** (the Xavier start), then today's full decode stage, on E52's four variants;
  - seed 101.

**Measured on a fixed mask of the validation split, at its hidden cells,** overall (`all`) and by
kind:

- **E04's diagnostics, in the loss's own space.** Under N, that is the prediction after the
  predictor and the layer-normalised target:
  - NMSE against the column-mean target;
  - the centred cosine and its standard error 1/sqrt(d·n);
  - the within-column share of the target's second moment;
  - the same share for the raw embedder target;
  - the effective rank of each column's centred targets and predictions.
- **Linear probes,** at the start and at the end:
  - **What they read:** the transformer's output at the hidden cells, the input the decode stage
    reads. Each column gets a `BayesianRidge` for numbers or a logistic regression for categories.
  - **Training cells:** four fixed masks of the training split.
  - **Score:** `impute_score`'s shape, the pooled z-space RMSE over the mean fill's and the error
    over the mode's, weighted by cells. 1.0 means no information.
- **Drift:**
  - the linear CKA between the encoder's outputs on the fixed masked validation input, before and
    after the decode stage (after = the checkpoint it keeps);
  - `‖ΔW‖/‖W‖` per module.

## Reading, fixed now

All readings use setting N, kind `all`, and the three seeds.

- **Collapse on a table:** on all three seeds, three things hold at the last epoch:
  - the within-column share is below the start's;
  - |centred cosine| < 2 standard errors;
  - NMSE ≥ 1.
- **N collapsing:** collapse on 2 or more of the 3 tables.
- **Value information on a table:**
  - the final probe score is below the start's on all three seeds;
  - and the mean gain is over 1% of the start's mean score.
- **Value information under N:** present on 2 or more of the 3 tables. Otherwise, "no value
  information under N".
- **H5, on credit-g:** a diagnostic moves if N_lr4 or N_b64 changes the collapse call, or moves
  the final probe score by more than 1% of N's, all three seeds in one direction.
- **Reported, without a rule:**
  - A's readings, against E04;
  - the split by kind (H4);
  - the drift.

**What it leads to:** E52's decision table ([10](10-frozen-encoder.md), "Reading, fixed now"),
which takes D3's reading and this one together. No default changes; each "propose" is a
pre-registered study and an ADR for the user.

## Launch

- **The probe cells started at 19:34 GMT-3 on 2026-10-20,** on 0bf1ebf. They run in six CPU
  lanes at 4 threads each, in `/scratch2/diogoneiss/trident-2026-10-20-collapse-probes/`.
  - The pendigits cells take about 80 minutes each on the CPU (16 s an epoch with the bigger
    encoder).
  - E52's trainers slowed by about 10% under the load (1.2 s against 1.1 s a pre-training epoch).
- **The drift cells** start on the GPU once E52's four queues have ended
  (`drift_after_e52.sh`, two lanes).
- **The reading** is `scripts/experiments/collapse_probes_report.py`, written before any study cell
  finished.

## Outcome (2026-10-20, 22:31 GMT-3)

- **All 24 probe cells** finished on the CPU between 19:34 and 21:35 GMT-3, none failed.
- **All 8 drift cells** finished on the GPU between 22:20 and 22:29.

**The printout** (`scripts/experiments/collapse_probes_report.py`, unedited):

```text
## Setting N
- credit-g_20nan: no collapse; value information absent
  - s101: within share 0.1949 -> 0.1670, raw 0.1637 -> 0.1359; centred cosine -0.0155 (-4.6 SE); NMSE 9.342 -> 1.033; erank target 2.41, prediction 26.14
  - s202: within share 0.2008 -> 0.1712, raw 0.1546 -> 0.1288; centred cosine -0.0179 (-5.2 SE); NMSE 9.531 -> 1.041; erank target 2.41, prediction 24.79
  - s303: within share 0.2182 -> 0.1875, raw 0.1874 -> 0.1544; centred cosine -0.0590 (-16.7 SE); NMSE 8.258 -> 1.028; erank target 2.34, prediction 24.19
  - probe start [0.9502, 1.0428, 1.0333] -> final [0.9662, 1.0297, 0.9991], mean gain +0.0104 (+1.0% of the start); by kind (final: num 0.9324, cat 1.0352; start: num 0.9619, cat 1.0345)
- kr-vs-kp_40nan: no collapse; value information present
  - s101: within share 0.1380 -> 0.1338, raw 0.1389 -> 0.1338; centred cosine +0.0173 (+8.8 SE); NMSE 14.162 -> 0.907; erank target 1.02, prediction 26.46
  - s202: within share 0.1425 -> 0.1361, raw 0.1422 -> 0.1349; centred cosine +0.0842 (+43.8 SE); NMSE 14.116 -> 0.923; erank target 1.03, prediction 27.37
  - s303: within share 0.1532 -> 0.1447, raw 0.1518 -> 0.1434; centred cosine +0.1225 (+65.3 SE); NMSE 12.723 -> 0.920; erank target 1.03, prediction 27.88
  - probe start [0.9511, 0.927, 0.9825] -> final [0.8587, 0.8942, 0.8928], mean gain +0.0716 (+7.5% of the start); by kind (final: num nan, cat 0.8819; start: num nan, cat 0.9535)
- pendigits_20nan: no collapse; value information present
  - s101: within share 0.1055 -> 0.3086, raw 0.1202 -> 0.3504; centred cosine +0.7729 (+656.6 SE); NMSE 13.587 -> 0.289; erank target 2.80, prediction 6.66
  - s202: within share 0.1106 -> 0.3016, raw 0.1357 -> 0.3531; centred cosine +0.7677 (+656.8 SE); NMSE 12.209 -> 0.298; erank target 2.72, prediction 6.76
  - s303: within share 0.1007 -> 0.2963, raw 0.1221 -> 0.3516; centred cosine +0.7682 (+657.6 SE); NMSE 14.386 -> 0.299; erank target 2.65, prediction 6.65
  - probe start [0.7917, 0.7778, 0.7932] -> final [0.493, 0.5122, 0.5122], mean gain +0.2817 (+35.8% of the start); by kind (final: num 0.5058, cat nan; start: num 0.7875, cat nan)

## Setting A
- credit-g_20nan: no collapse; value information absent
  - s101: within share 0.1754 -> 0.0980, raw 0.1754 -> 0.0980; centred cosine +0.0100 (+3.0 SE); NMSE 218.864 -> 4.591; erank target 2.36, prediction 29.46
  - s202: within share 0.1849 -> 0.1058, raw 0.1849 -> 0.1058; centred cosine +0.0036 (+1.0 SE); NMSE 205.232 -> 4.418; erank target 2.38, prediction 27.75
  - s303: within share 0.1849 -> 0.1067, raw 0.1849 -> 0.1067; centred cosine -0.0079 (-2.2 SE); NMSE 212.848 -> 4.671; erank target 2.32, prediction 26.70
  - probe start [0.9677, 1.0222, 1.0033] -> final [0.9733, 1.0071, 1.0034], mean gain +0.0031 (+0.3% of the start); by kind (final: num 0.9400, cat 1.0247; start: num 0.9556, cat 1.0211)
- kr-vs-kp_40nan: no collapse; value information present
  - s101: within share 0.1375 -> 0.0615, raw 0.1375 -> 0.0615; centred cosine +0.0175 (+8.9 SE); NMSE 454.358 -> 2.265; erank target 1.03, prediction 45.34
  - s202: within share 0.1488 -> 0.0707, raw 0.1488 -> 0.0707; centred cosine +0.0256 (+13.4 SE); NMSE 454.366 -> 2.169; erank target 1.03, prediction 46.81
  - s303: within share 0.1408 -> 0.0636, raw 0.1408 -> 0.0636; centred cosine +0.0150 (+8.0 SE); NMSE 446.369 -> 2.175; erank target 1.03, prediction 48.70
  - probe start [0.9212, 0.9496, 0.9526] -> final [0.9049, 0.9118, 0.9377], mean gain +0.0230 (+2.4% of the start); by kind (final: num nan, cat 0.9181; start: num nan, cat 0.9411)
- pendigits_20nan: no collapse; value information present
  - s101: within share 0.1106 -> 0.2176, raw 0.1106 -> 0.2176; centred cosine +0.6250 (+530.9 SE); NMSE 126.149 -> 0.434; erank target 2.44, prediction 7.83
  - s202: within share 0.1438 -> 0.2266, raw 0.1438 -> 0.2266; centred cosine +0.6328 (+541.4 SE); NMSE 96.421 -> 0.411; erank target 2.39, prediction 7.80
  - s303: within share 0.1367 -> 0.2041, raw 0.1367 -> 0.2041; centred cosine +0.6289 (+538.4 SE); NMSE 90.117 -> 0.404; erank target 2.29, prediction 7.61
  - probe start [0.7986, 0.7751, 0.7788] -> final [0.579, 0.5805, 0.5794], mean gain +0.2045 (+26.1% of the start); by kind (final: num 0.5796, cat nan; start: num 0.7842, cat nan)

Primary: N collapsing: no (0 of 3); value information under N: present (2 of 3)
H5 N_lr4: collapse no collapse (N: no collapse); final probe [0.95, 1.0451, 1.017] against N's [0.9662, 1.0297, 0.9991] (mean +0.0057) -> does not move
H5 N_b64: collapse no collapse (N: no collapse); final probe [0.9622, 1.0196, 0.9648] against N's [0.9662, 1.0297, 0.9991] (mean -0.0162) -> moves

## Drift across today's decode stage (seed 101, fold 1)
        variant start  linear_cka  induced_impute_score  rel_change/embedder  rel_change/transformer.pre_norm  rel_change/transformer.layers  rel_change/transformer.final_norm
 credit-g_20nan     N      0.8733                0.8721               0.0896                           0.0135                         0.1577                                NaN
 credit-g_20nan     R      0.7999                0.8939               0.0653                           0.0137                         0.1208                                NaN
 credit-g_80nan     N      0.9953                1.0157               0.0357                           0.0078                         0.0720                                NaN
 credit-g_80nan     R      0.9862                1.0131               0.0221                           0.0087                         0.0585                                NaN
 kr-vs-kp_40nan     N      0.6026                0.7280               0.2844                           0.0575                         0.4237                                NaN
 kr-vs-kp_40nan     R      0.5253                0.7280               0.3155                           0.0398                         0.3597                                NaN
pendigits_20nan     N      0.3993                0.3227               0.1430                           0.1499                         0.6689                             0.1513
pendigits_20nan     R      0.0307                0.3288               0.2363                           0.1330                         0.7519                             0.1073
```

**Reading, by the rule fixed in advance:**

- **N collapsing: no** (0 of 3 tables).
  - **pendigits_20nan is the opposite of collapse.** The within-column share of the target rises
    from about 0.10 to 0.30, the embedder target's from 0.12 to 0.35. NMSE ends at 0.29, far
    better than a per-column constant, and the centred cosine is +0.77.
  - **kr-vs-kp_40nan:** the share barely falls (0.14 to 0.13), NMSE ends at 0.91 (below 1), and the
    cosine is positive (+9 to +65 SE).
  - **credit-g_20nan comes closest.** The share falls 14% (0.20 to 0.17; the raw target's 17%) and
    NMSE ends at 1.03, no better than the column constant. But the centred cosine is negative at
    −4.6 to −16.7 SE, so the rule's |cos| < 2 SE fails. The predictions point away from the
    within-column part, which is not flat collapse.
- **Value information under N: present** (2 of 3).
  - **pendigits:** the probe goes from 0.787 to 0.506 (+35.8%).
  - **kr-vs-kp:** from 0.954 to 0.882 (+7.5%).
  - **credit-g: absent.** +1.0%, with one seed worse, near no information (about 1.0). Its
    numbers carry a little (0.96 to 0.93); its categories do not (1.03 to 1.04).
- **H5:**
  - `LR_PRE` ÷ 4 does not move a call or the probe (+0.006).
  - **`BATCH 64` moves credit-g's final probe:** −0.016, all three seeds lower. That is just over
    the 1% margin, on a table whose probes sit near 1.0.

**Secondary, without a rule:**

- **The shipped objective (A) erodes its own target.** The within-column share falls 40–55% on
  credit-g and kr-vs-kp (0.18 to 0.10, 0.14 to 0.06), as E04 found. The rule still reads no
  collapse, because the cosine is significant on several seeds.
  - A's probes gain less than N's: +2.4% on kr-vs-kp and +26.1% on pendigits, against N's +7.5%
    and +35.8%.
- **Drift across today's decode stage** (seed 101, fold 1):
  - **pendigits:** the stage rewrites most of the transformer. The weights change 67% from N's
    start and 75% from the random start. The outputs keep a linear CKA of 0.40 with N's start and
    0.03 with the random one.
  - **kr-vs-kp:** CKA 0.60 from N and 0.53 from random; the layers change 42% and 36%.
  - **credit-g_20nan:** CKA 0.87 and 0.80.
  - **credit-g_80nan:** CKA 0.99 from both starts. The kept checkpoint barely moves the encoder,
    at mean/mode parity.
  - **On pendigits and kr-vs-kp, both starts end at about the same score:** 0.323 against 0.329,
    and 0.728 against 0.728.
  - **On credit-g_20nan they do not:** 0.872 from N against 0.894 from random. That is one fold;
    E52's C0 − G is +1.7% over 15.

**What it means, with E52** ([10](10-frozen-encoder.md), Outcome):

- Today's normalised pre-training does not collapse. On the two tables where pre-training matters,
  it leaves a real, linearly decodable representation: frozen, it beats a random encoder by 53% and
  16%.
- The decode stage then keeps only part of it: 40% CKA on pendigits, 60% on kr-vs-kp. From
  random, it builds an encoder about as good: C0 − G within 1.7%.
- So H3, washout, is the mechanism, not H1 or H2. The levers against washout did not help (E50),
  because the decode stage's own encoder is as good as the one it overwrites.

**What it leads to,** through E52's decision table: D3 mixed with value information present, and
a diagnostic moving with the batch. The last row applies: **T12 first, in E26's shape, with a
`BATCH 64` arm.** It is a pre-registered study for the user to decide, with two caveats:

- what moved is credit-g's probe only, near no information;
- by this reading, the decode stage would overwrite a better pre-trained encoder too.

Nothing changes a default. The seeds are development seeds.

