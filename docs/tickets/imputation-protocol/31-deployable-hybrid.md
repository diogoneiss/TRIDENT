# Can the hybrid's weights be chosen without the complete table? (E82; pre-registered 2026-10-27)

This ticket was written and committed before the study's first cell.

- **Why:** ADR 0031 (accepted) makes the KC + TabPFN hybrid a *benchmark* output, because its
  weights read the induced truth through the complete table, as KC does.
  - The user wants the method's viability: "numa implementação real experimentaríamos uma
    combinação de pesos para ver".
  - The deployable route is to try the weight grid on cells that were observed and are hidden on
    purpose, whose truth is known without a complete table. The model side is then a fill that
    reads no complete table either: H (ADR 0024).
- **The user's go:** "vamos no 1 agora então" (2026-10-27, before 13:30 GMT-3).
- **Code:** no change to `src/`.
  - `scripts/experiments/deployable_hybrid.py` has `train`, `blend` and `--tally`.
  - `run_deployable_hybrid_queue.sh` runs four training queues, then two GPU blend lanes, chained,
    as E74 ran.
- **Smoke, outside the study:** credit-g_40nan at seed 1, `--no-mlflow`.
  - `train` ran (5 min 28 s) with `DECODE_CHECKPOINT loss` and `DECODE_CALIBRATION none`, and wrote
    five selection files.
  - `blend` ran (94 s) and asserted the ledger equals the run's H on all 5 folds.
  - Only shapes were printed: 25 score rows over the five methods; 49–60 selection cells per column
    and fold. No score was read.

## Design (fixed before the first cell)

- **Cells:** E74's 10 hard variants (`blend_weights.HARD`) at E74's seeds 1212, 2323 and 3434,
  30 cells.
  - These are electricity at 40 and 80nan; vehicle and biodeg at 80nan; kr-vs-kp and credit-g at
    40, 60 and 80nan.
  - The seeds are E74's on purpose. `build_folds` then gives the same test cells, so E74's KC and
    KC hybrid read beside these, fold by fold.
  - No seed set has seen this selection rule, or H as the model side.
- **Training:** today's default with `--decode_checkpoint loss --decode_calibration none`.
  - The run keeps H, raw. Nothing in training reads the complete table: the checkpoint is the
    loss's, and there is no calibration. The table only scores.
  - Five folds; MLflow tag `experiment=deployable-hybrid-2026-10-27`.
- **The selection cells (read in place after each fold, random states restored):**
  - The validation split's *observed* cells are dealt into 5 groups at random (stream
    `[seed, fold, 82]`).
  - Each group in turn is hidden as `[MASK]`, beside the real gaps, and H fills it. So every
    observed validation cell gets an out-of-sample fill, and its truth is the observed value.
  - TabPFN, refitted on the training fold as in E74, fills the same cells group by group.
- **The five fills of each fold's test induced cells (`blend`):**
  - **H:** the deployable model alone;
  - **TabPFN;**
  - **equal:** the 0.5 blend;
  - **deployable:** E69's grid and rule, with weights chosen on that fold's selection cells. No
    complete table is used;
  - **oracle:** the same rule with weights chosen on the other four folds' induced cells, as in
    E69/E74 but with H as the model side. It is the ceiling for this model side.
- **Running:** four GPU training queues, then two GPU blend lanes, chained; state in
  `/scratch2/diogoneiss/trident-2026-10-27-deployable-hybrid/`. About 5–6 h; E74's identical
  training took 4 h 40 min, and its blend 24 min.

## Measures and reading, fixed now (`deployable_hybrid.py --tally`)

On the induced `impute_score` over 15 fold pairs, with the verdict rule and the 1% SESOI:

- **Primary, viability, over the 10 variants:**
  - **supported:** the deployable hybrid is better than H on 8 or more, worse than H on none, and
    worse than TabPFN on at most 2 (E74's own KC hybrid was worse than TabPFN on 2);
  - **refuted:** better than H on 5 or fewer, or worse than H on 2 or more, or worse than TabPFN on
    5 or more;
  - **mixed:** otherwise.
- **Secondary, the price of deployability:**
  - deployable − oracle per variant, as counts;
  - how often the deployable weight equals the oracle's;
  - the number of variants where the deployable hybrid beats both H and TabPFN (E74's KC hybrid
    beat both on 3).
- **Descriptive:**
  - oracle − H;
  - deployable − KC and deployable − E74's KC hybrid, from E74's blend files on the same folds;
  - the selection cells per column.

**What each reading leads to:**
- **Supported:** a weight chosen on observed cells is enough. A deployable hybrid with H is viable,
  and its cost against the benchmark hybrid is the secondary's numbers.
  - Making it an output would be an amendment to ADR 0031, for the user.
- **Mixed or refuted:** ADR 0031 stays a benchmark output. The study names where selecting without
  the complete table fails.

## Limits fixed now

- MCAR only, on the 10 hard variants.
- The model side is H raw. E47's masked calibration is not tried.
- The selection cells come from the validation split only, about 10% of the rows.

## Outcome (2026-10-27, 18:42 GMT-3): refuted

- **Ran:** 30 of 30 training cells (13:30–16:56 GMT-3) and 30 of 30 blend pairs (to 18:37); none
  failed. Every fold's ledger equals the run's H.
- **Primary: refuted.** The rule's third clause decides it: worse than TabPFN on 5 or more.
  - **Against H:** the deployable hybrid is better on 10 of 10 and worse on none.
  - **Against TabPFN:** worse on 5 of 10: vehicle_80nan, biodeg_80nan, kr-vs-kp_60nan,
    kr-vs-kp_80nan and credit-g_60nan. Better on 2 (electricity 40 and 80nan), no difference on 2,
    equivalent on 1.
- **Secondary, the price of deployability:**
  - The deployable hybrid is worse than the oracle (the same rule, other folds' induced weights) on
    6 of 10, and never better.
  - The deployable weight equals the oracle's on only 27–62% of column-folds.
  - It beats both H and TabPFN on 2 (electricity), against E74's 3.
- **Post hoc (not in the rule; `e82_posthoc` below):** with the oracle's weights, the H-side hybrid
  against TabPFN is better on 3 (the numerical 80nan three), no difference on 2, equivalent on 3 and
  worse on 2. That is E74's level with KC.
  - **So the loss is the selection, not H.**
  - It falls where the selection cells are few: vehicle_80nan and biodeg_80nan have a median of
    18–21 a column a fold.
  - It also falls on categorical columns, where a pick between two categories rests on few
    disagreements: kr-vs-kp at 60/80nan and credit-g_60nan.
  - On electricity (900–2,700 cells a column) the deployable hybrid beats both.
- **Descriptive, and useful:** without the complete table, the deployable hybrid is better than
  today's benchmark default KC on 7 of 10, and worse on none.
- **What it leads to, as fixed:**
  - ADR 0031 stays a benchmark output.
  - Choosing per-column weights on the validation split's observed cells is not enough to reach
    TabPFN on the small and the categorical tables.
  - It is enough to beat H on all 10, and KC on 7.
- **Candidates, not run, each its own ticket if the user wants it:**
  - more selection cells: also hide the test split's own observed cells, or pool the folds;
  - fewer parameters to choose: one weight per table, or a shrinkage toward the equal blend where a
    column has few cells.

The reader's printout (`deployable_hybrid.py --tally`), unedited:

```text
- `electricity_40nan` (H 0.7441, TabPFN 0.7435; selection cells a column a fold: median 2719, min 2634; deployable weight = oracle weight on 53%): deployable − model -0.0150 [-0.0295, -0.0004] better; deployable − tabpfn -0.0144 [-0.0191, -0.0097] better; deployable − oracle -0.0055 [-0.0191, +0.0080] none; oracle − model -0.0094 [-0.0176, -0.0013] better; deployable − KC -0.0203 [-0.0417, +0.0011] none; deployable − E74 hybrid -0.0050 [-0.0158, +0.0058] none
- `electricity_80nan` (H 0.8995, TabPFN 0.8904; selection cells a column a fold: median 902, min 844; deployable weight = oracle weight on 49%): deployable − model -0.0100 [-0.0131, -0.0068] better; deployable − tabpfn -0.0009 [-0.0014, -0.0004] better; deployable − oracle +0.0004 [+0.0001, +0.0008] worse; oracle − model -0.0104 [-0.0135, -0.0074] better; deployable − KC -0.0075 [-0.0109, -0.0041] better; deployable − E74 hybrid +0.0013 [+0.0007, +0.0018] worse
- `vehicle_80nan` (H 0.8178, TabPFN 0.7569; selection cells a column a fold: median 18, min 9; deployable weight = oracle weight on 27%): deployable − model -0.0561 [-0.0630, -0.0493] better; deployable − tabpfn +0.0048 [+0.0005, +0.0090] worse; deployable − oracle +0.0080 [+0.0045, +0.0115] worse; oracle − model -0.0641 [-0.0726, -0.0556] better; deployable − KC -0.0257 [-0.0294, -0.0219] better; deployable − E74 hybrid +0.0117 [+0.0084, +0.0150] worse
- `biodeg_80nan` (H 0.9777, TabPFN 0.9195; selection cells a column a fold: median 21, min 8; deployable weight = oracle weight on 28%): deployable − model -0.0528 [-0.0628, -0.0428] better; deployable − tabpfn +0.0054 [+0.0028, +0.0080] worse; deployable − oracle +0.0138 [+0.0072, +0.0204] worse; oracle − model -0.0666 [-0.0740, -0.0592] better; deployable − KC -0.0161 [-0.0210, -0.0111] better; deployable − E74 hybrid +0.0182 [+0.0144, +0.0220] worse
- `kr-vs-kp_40nan` (H 0.7247, TabPFN 0.7139; selection cells a column a fold: median 191, min 168; deployable weight = oracle weight on 52%): deployable − model -0.0090 [-0.0162, -0.0018] better; deployable − tabpfn +0.0017 [-0.0042, +0.0077] none; deployable − oracle -0.0011 [-0.0076, +0.0054] none; oracle − model -0.0079 [-0.0123, -0.0035] better; deployable − KC -0.0087 [-0.0172, -0.0001] better; deployable − E74 hybrid -0.0036 [-0.0102, +0.0031] none
- `kr-vs-kp_60nan` (H 0.8742, TabPFN 0.8261; selection cells a column a fold: median 129, min 104; deployable weight = oracle weight on 55%): deployable − model -0.0328 [-0.0393, -0.0262] better; deployable − tabpfn +0.0153 [+0.0114, +0.0192] worse; deployable − oracle +0.0118 [+0.0072, +0.0164] worse; oracle − model -0.0446 [-0.0520, -0.0371] better; deployable − KC -0.0158 [-0.0225, -0.0090] better; deployable − E74 hybrid +0.0117 [+0.0064, +0.0170] worse
- `kr-vs-kp_80nan` (H 1.0098, TabPFN 0.9430; selection cells a column a fold: median 64, min 43; deployable weight = oracle weight on 62%): deployable − model -0.0448 [-0.0553, -0.0342] better; deployable − tabpfn +0.0221 [+0.0120, +0.0321] worse; deployable − oracle +0.0210 [+0.0118, +0.0302] worse; oracle − model -0.0658 [-0.0791, -0.0525] better; deployable − KC -0.0049 [-0.0175, +0.0077] none; deployable − E74 hybrid +0.0214 [+0.0121, +0.0307] worse
- `credit-g_40nan` (H 0.9274, TabPFN 0.9042; selection cells a column a fold: median 60, min 46; deployable weight = oracle weight on 46%): deployable − model -0.0211 [-0.0263, -0.0159] better; deployable − tabpfn +0.0021 [-0.0064, +0.0106] none; deployable − oracle -0.0034 [-0.0094, +0.0027] none; oracle − model -0.0177 [-0.0257, -0.0098] better; deployable − KC -0.0172 [-0.0244, -0.0101] better; deployable − E74 hybrid +0.0011 [-0.0070, +0.0092] none
- `credit-g_60nan` (H 0.9982, TabPFN 0.9524; selection cells a column a fold: median 40, min 27; deployable weight = oracle weight on 49%): deployable − model -0.0318 [-0.0443, -0.0193] better; deployable − tabpfn +0.0140 [+0.0060, +0.0220] worse; deployable − oracle +0.0112 [+0.0054, +0.0171] worse; oracle − model -0.0430 [-0.0543, -0.0317] better; deployable − KC -0.0150 [-0.0266, -0.0033] better; deployable − E74 hybrid +0.0157 [+0.0064, +0.0249] worse
- `credit-g_80nan` (H 1.0198, TabPFN 1.0001; selection cells a column a fold: median 20, min 10; deployable weight = oracle weight on 40%): deployable − model -0.0193 [-0.0269, -0.0117] better; deployable − tabpfn +0.0004 [-0.0025, +0.0034] equivalent; deployable − oracle -0.0027 [-0.0069, +0.0016] equivalent; oracle − model -0.0166 [-0.0228, -0.0104] better; deployable − KC -0.0049 [-0.0111, +0.0014] none; deployable − E74 hybrid +0.0013 [-0.0039, +0.0066] equivalent

- deployable − model: better 10 / no difference 0 / worse 0 (fold-pair t, seed agreement)
- deployable − tabpfn: better 2 / no difference 2 / equivalent 1 / worse 5 (fold-pair t, seed agreement)
- deployable − oracle: better 0 / no difference 3 / equivalent 1 / worse 6 (fold-pair t, seed agreement)
- oracle − model: better 10 / no difference 0 / worse 0 (fold-pair t, seed agreement)
- deployable − KC: better 7 / no difference 3 / worse 0 (fold-pair t, seed agreement)
- deployable − E74 hybrid: better 0 / no difference 3 / equivalent 1 / worse 6 (fold-pair t, seed agreement)
- the deployable hybrid beats both H and TabPFN on 2: ['electricity_40nan', 'electricity_80nan']

Primary: the deployable hybrid is better than H on 10 of 10, worse on none, and worse than TabPFN on 5 (['vehicle_80nan', 'biodeg_80nan', 'kr-vs-kp_60nan', 'kr-vs-kp_80nan', 'credit-g_60nan']) -> refuted (fold-pair t, seed agreement)
```

Post hoc, the H-side oracle hybrid against TabPFN (`e82_posthoc`, not in the rule), unedited:

```text
electricity_40nan: oracle − tabpfn -0.0089 [-0.0220, +0.0042] none
electricity_80nan: oracle − tabpfn -0.0013 [-0.0018, -0.0008] better
vehicle_80nan: oracle − tabpfn -0.0032 [-0.0049, -0.0016] better
biodeg_80nan: oracle − tabpfn -0.0084 [-0.0139, -0.0029] better
kr-vs-kp_40nan: oracle − tabpfn +0.0028 [-0.0017, +0.0074] none
kr-vs-kp_60nan: oracle − tabpfn +0.0035 [+0.0010, +0.0060] worse
kr-vs-kp_80nan: oracle − tabpfn +0.0011 [-0.0011, +0.0032] equivalent
credit-g_40nan: oracle − tabpfn +0.0054 [+0.0004, +0.0105] worse
credit-g_60nan: oracle − tabpfn +0.0028 [-0.0024, +0.0079] equivalent
credit-g_80nan: oracle − tabpfn +0.0031 [-0.0018, +0.0080] equivalent
better 3 / no difference 2 / equivalent 3 / worse 2 (fold-pair t, seed agreement)
```
