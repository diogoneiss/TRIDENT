# K and S on fresh seeds: does E186's downstream reading repeat? (E187, T64; pre-registered 2026-11-22 17:27 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** E186 (ticket 75) read a downstream classifier on fills whose seeds (101/202/303) had already been used to
  read `sw_score` (E183 to E185). Its primary gain was small and fragile, and its split by table (filling helps a
  forest on the numerical tables, hurts on `kr-vs-kp` and `credit-g`) was found after the read. Ticket 75 states that
  a claim needs fresh seeds. This study trains them.
- **The user's go** (2026-11-22): "sim, repita k e s em semenstes novas", answering my offer after E186. The reply
  names K and S, so the two arms (ticket 75's reading of the user's "H") are settled for this study. The design
  below is mine.
- **Disclosed:**
  - E186's printout was read before this design; every prediction below comes from it.
  - **Seeds 1717, 8484, 9595:** not among the store's `seed` tags and in no script, ticket or log entry (checked
    2026-11-22). A fresh seed changes the folds, the initialisation and the training masks; a variant's gap pattern
    is its file on disk and stays the same.
  - **Three seeds, fixed now, no extension.** The corrected (Nadeau–Bengio) interval's width is dominated by the
    split-ratio term (0.2/0.7 against 1/15), so more seeds would barely narrow it: a real gain of E186's size is
    expected to stay "fragile" under it. Stated before the first cell.
  - No smoke: the trainer is `sliced_loss_run.py`'s, unchanged. The first cells (`kc2`, minutes each) check the
    mechanics; a failing check is reported, not dropped.
  - Nothing in `src/` changes. `sliced_loss_run.py` gains one type annotation (`PAIRS`), no behaviour.
  - **The reads' script is not in this commit.** It will run `sliced_utility.py`'s functions (E186's) and E182's
    scorer with this study's state, seeds and arms, in a file the queues never import, and is committed with a dated
    note here before any of this study's ledgers is scored.
- **Runner (this commit):** [scripts/experiments/sliced_fresh_run.py](../../../scripts/experiments/sliced_fresh_run.py);
  state `/scratch/diogoneiss/trident-2026-11-22-sliced-fresh`, tag `experiment = sliced-fresh-2026-11-22`.

## Design and rule (fixed before the first cell)

- **Scenarios:** the 26 MCAR variants of E184 and E185 × seeds 1717/8484/9595 × five folds, arms **K** (the
  imputation task's default run: checkpoint by the validation gaps' `impute_score`, ADR 0015) and **S** (the same
  run with `--decode_metric sw_score`, E183). 156 runs, all on gorgona8's RTX 3090 Ti (a pair on one machine, ADR
  0035), four queues (`c15_queue.sh`, two threads each), a pair's two arms side by side, shortest variants first.
  `pendigits` and `letter` use their promoted files (ADR 0018 / 0036 / 0037), as in E183 to E185. Both arms carry the
  default output (calibrated from 40nan, ADR 0027).
  - Cost: about 28 h from E184's and E185's clocks (half of 19.6 h and of 37.4 h), so about 22:00 GMT-3 on
    2026-11-23.
- **Checks before the reads:** `--gate` (each run's `DECODE_METRIC`, no loss term) and `--pairing` (a pair's scored
  cells and hgb's score agree).
- **Downstream read, as E186:** per (variant, seed) the tables `missing`, `K`, `S`, beside `naive` (the fold's
  mean/mode fill) and `truth`; each row's gaps filled by the fold in which it is a test row; macro-F1 on the test
  folds; a random forest of 500 trees (primary, takes the gaps as they are) and logistic regression (beside; a gap
  is the mean with an indicator, or a level of its own); protocol `own` (fitted on the table's own rows outside the
  fold; primary) and `complete` (fitted on the truth-filled rows; beside). A pooled difference is over the 26
  variants per (seed, fold) replicate, 15 replicates, plain 95% t interval; the Nadeau–Bengio interval beside, and
  a verdict it does not share is reported as fragile. On the fresh seeds alone:
  - **Primary: F1(S) − F1(K)**, forest, `own`. E186: +0.0035 [+0.0009, +0.0061]. **Replicated** if the interval is
    above 0; **not replicated** if it includes 0; **reversed** if below 0.
  - **Three-way: F1(K) − F1(missing) and F1(S) − F1(missing)**, same rule. E186: +0.0033 and +0.0068.
  - **The by-table split, K − missing per variant** (15 fold values, plain interval), its first test. E186's calls:
    - K better on 9: `pendigits_20nan`, `pendigits_40nan`, `pendigits_60nan`, `letter_20nan`, `vehicle_20nan`,
      `vehicle_40nan`, `electricity_20nan`, `electricity_40nan`, `biodeg_40nan`;
    - missing better on 5: `kr-vs-kp_40nan`, `kr-vs-kp_60nan`, `kr-vs-kp_80nan`, `credit-g_60nan`, `credit-g_80nan`.
    - **Replicated** if at least 10 of these 14 repeat their call and none takes the opposite call; **not
      replicated** if fewer than 7 repeat or two or more take the opposite call; **partly** otherwise.
  - Beside, no rule: whether S − K's two variants (`credit-g_60nan`, `pendigits_80nan`) repeat; S − missing per
    variant (E186: S on 12, missing on 4); `naive` and `truth` against `missing`; the levels; by missingness level;
    logistic regression; `complete` (E186: S − K +0.0085 forest, +0.0060 logistic, both holding corrected).
  - Beside, labelled not fresh: the primary line pooled over the six seeds (E186's three and these), 30 replicates.
- **Manipulation check, secondary (E183 on fresh seeds):** each run's test-fold ledgers scored by E182's code
  (`sw_score`, `dist_score`, `impute_score`; the `reference` group with the run's ledger in TRIDENT's place, as
  E184). S − K per variant by the fold-pair rule with seed agreement, margin 0.01 of K's mean. E183's rule over the
  variants whose shuffled control reaches 0.10: supported if S is better on `sw_score` on half or more and worse on
  at most 2; refuted if better on a sixth or fewer; mixed otherwise. E183: better on 15 of 26, worse on none;
  `impute_score` worse on 6. Without it a downstream null could not be told from a selector that did nothing.
  The strong-tier comparisons are not repeated (no baseline files for these seeds).
- **Read once:** when all 156 runs are in, after the checks; the reads refuse a partial set. No interim read. No
  default and no ADR changes by this study; a replicated gain is input for the user's decision.
- **Not in this study:** the loss term (L, LS); ADR 0024's H; MAR and MNAR; why filling hurts on `kr-vs-kp` and
  `credit-g`; tuned classifiers.

## Amendment (2026-11-22 17:47 GMT-3): the reads' script, before any of this study's ledgers is scored

- **Script:** [scripts/experiments/sliced_fresh_read.py](../../../scripts/experiments/sliced_fresh_read.py), which the
  queues never import: `utility` (E186's downstream cell), `score` (E182's `reference` scorer on a run's ledgers),
  `--jobs`, and `--tally`, which refuses a partial set and prints E186's tally, the count against E186's 14 calls
  with the rule above, the two S − K carriers, the six-seed pooled line and the manipulation check.
- **`sliced_utility.py` (E186's) is parameterised, behaviour unchanged:** the header takes the study, variants and
  seeds from its globals and the loss-term pairs have a name. E186's `--tally` was rerun after the edit and its 160
  lines are identical to ticket 75's printout (`diff`, no output).
- **Self-check on seeds already read (101/202/303), not a read of this study:** with `FRESH_SELFCHECK` the same code
  recomputed the 78 downstream files from E184's and E185's K and S runs and reproduced E186: primary +0.0035
  [+0.0009, +0.0061], K − missing calls K 9 / missing 5, 14 of 14 repeat (trivially), the two carriers S. Its
  manipulation check, on those seeds with E184/E185's own K runs as control: S better on `sw_score` on 14 of 26,
  worse on none (E183, with E181's control runs: 15 and 0); `impute_score` worse on 5.
- **Mechanics so far:** the first 8 runs (`kc2_20nan`, `kc2_40nan` starting) passed `--gate` (8 checked, 0 failed)
  and `--pairing` (cells and hgb the same). No score of this study has been computed or read.

## Outcome (2026-11-23 19:13 GMT-3): the downstream gain of S over K is not replicated; the by-table split is; the selector's `sw_score` gain is

All 156 runs were in at 2026-11-23 18:45 GMT-3, none failed. `--gate`: 156 finished runs checked, 0 failed.
`--pairing`: cells and hgb's score the same on all 78 pairs. The 78 downstream cells and 156 ledger scores were then
computed (no error), and `--tally` was run once, at 2026-11-23 19:12 GMT-3.

- **Primary, not replicated:** F1(S) − F1(K), forest on each table's own rows, **+0.0018 [−0.0013, +0.0049]** (E186:
  +0.0035 [+0.0009, +0.0061]). Per variant S on 2 (`pendigits_80nan` +0.0142, holding corrected; `electricity_80nan`
  +0.0053, fragile), K on 1 (`kr-vs-kp_80nan` −0.0165, fragile), no difference on 23. Of E186's two carriers,
  `pendigits_80nan` repeats and `credit-g_60nan` does not (+0.0169 [−0.0019, +0.0356]).
  - Not fresh, the six seeds pooled: +0.0027 [+0.0007, +0.0046], corrected [−0.0033, +0.0087]: S, fragile.
- **Three-way:** F1(K) − F1(missing) +0.0036 [−0.0005, +0.0077], no difference: **not replicated** (E186: +0.0033, K).
  F1(S) − F1(missing) +0.0054 [+0.0010, +0.0098]: **replicated**, fragile as in E186 (+0.0068).
- **The by-table split, replicated:** 12 of E186's 14 calls on K − missing repeat and none takes the opposite call.
  - K better again on 8 of 9 (`pendigits` 20/40/60nan up to +0.059, `letter_20nan` +0.041, `vehicle` 20/40nan,
    `electricity_20nan`, `biodeg_40nan`); `electricity_40nan` falls to no difference (+0.0017).
  - Missing better again on 4 of 5 (`kr-vs-kp` 40/60/80nan, `credit-g_80nan` −0.054); `credit-g_60nan` falls to no
    difference (−0.0258 [−0.0518, +0.0002]).
  - New K calls on variants E186 left at no difference: `biodeg_20nan`, `credit-g_20nan`. Counts here: K 10,
    missing 4, no difference 12; for S − missing: S 13, missing 5.
- **Beside, by missingness level (no rule; the levels hold different tables):** K − missing is +0.0123 at 20nan
  (holding corrected), +0.0096 at 40nan, −0.0017 at 60nan and −0.0113 at 80nan (missing, fragile). At 80nan K's fill
  is called better on none of the 7 variants, here and in E186. An observation over two seed sets, not a verdict.
- **Levels (forest, own):** missing 0.7142, mean/mode 0.7054, K 0.7178, S 0.7195, complete table 0.8521. The mean/mode
  fill is worse than leaving the gaps (−0.0088, holding corrected). The fills recover 2.6% (K) and 3.9% (S) of the
  0.138 between the table with gaps and the complete one.
- **Logistic regression, own:** S − K +0.0001, no difference (E186: +0.0029, fragile); K and S are 0.038 above its
  missing arm (mean fill with indicators) again.
- **Fitted on the complete rows (beside):** S − K +0.0062 [+0.0037, +0.0088] with the forest and +0.0075 with logistic
  regression, both holding corrected, as in E186 (+0.0085, +0.0060).
- **Manipulation check, supported:** S − K on `sw_score` better on 16 of 26, worse on none (E183: 15 and 0);
  `dist_score` better on 14, worse on none; `impute_score` better on 4, worse on 5 (`credit-g` 40/80nan,
  `kr-vs-kp_80nan`, `pendigits_80nan`, `letter_20nan`). E183's selector result replicates on fresh seeds, so the
  downstream null is not a selector that did nothing: on `credit-g_80nan` `sw_score` falls from 0.944 to 0.650 and
  macro-F1 moves by +0.004.
- **Reading:** choosing the checkpoint by `sw_score` changes the fills' distribution as designed, and that does not
  reach a classifier fitted on the filled table: the fresh-seed difference is not distinguishable from 0, and six
  seeds put it near +0.003. It does help a classifier fitted on complete rows (+0.006 to +0.008 in both seed sets).
  Whether to fill at all for a forest depends on the table, and that split held on fresh seeds.
- No default and no ADR changes. Limits: MCAR only; two untuned learners; why filling hurts on `kr-vs-kp` and
  `credit-g` is still not tested.

Printout (unedited, all 271 lines):

```
## Downstream utility of the sliced selector (E187): macro-F1 on the test folds of 26 MCAR variants, seeds 1717/8484/9595; a difference is pooled over the variants per (seed, fold) replicate, 15 replicates, plain 95% t interval (NB: corrected)
Primary, forest fitted on each table's own rows, F1(S) − F1(K): +0.0018 [-0.0013, +0.0049] (NB [-0.0054, +0.0090]) -> no difference
Three-way, same, F1(K) − F1(missing): +0.0036 [-0.0005, +0.0077] (NB [-0.0057, +0.0129]) -> no difference
Three-way, same, F1(S) − F1(missing): +0.0054 [+0.0010, +0.0098] (NB [-0.0048, +0.0156]) -> S
  beside, F1(naive) − F1(missing): -0.0088 [-0.0113, -0.0062] (NB [-0.0145, -0.0030]) -> missing
  beside, F1(truth) − F1(missing): +0.1379 [+0.1323, +0.1435] (NB [+0.1251, +0.1508]) -> truth
Levels (forest, own): missing 0.7142, naive 0.7054, K 0.7178, S 0.7195, truth 0.8521

### By missingness level (forest, own)
- 20nan (9 variants): missing 0.8093, naive 0.8042, K 0.8215, S 0.8208, truth 0.8644
  - F1(S) − F1(K): -0.0007 [-0.0028, +0.0013] (NB [-0.0054, +0.0039]) -> no difference
  - F1(K) − F1(missing): +0.0123 [+0.0087, +0.0158] (NB [+0.0041, +0.0205]) -> K
  - F1(S) − F1(missing): +0.0115 [+0.0082, +0.0149] (NB [+0.0039, +0.0192]) -> S
- 40nan (7 variants): missing 0.7442, naive 0.7357, K 0.7539, S 0.7541, truth 0.8371
  - F1(S) − F1(K): +0.0002 [-0.0040, +0.0044] (NB [-0.0095, +0.0099]) -> no difference
  - F1(K) − F1(missing): +0.0096 [+0.0032, +0.0161] (NB [-0.0053, +0.0246]) -> K
  - F1(S) − F1(missing): +0.0098 [+0.0028, +0.0168] (NB [-0.0062, +0.0259]) -> S
- 60nan (3 variants): missing 0.6743, naive 0.6485, K 0.6726, S 0.6776, truth 0.8823
  - F1(S) − F1(K): +0.0050 [-0.0019, +0.0118] (NB [-0.0107, +0.0207]) -> no difference
  - F1(K) − F1(missing): -0.0017 [-0.0121, +0.0087] (NB [-0.0257, +0.0223]) -> no difference
  - F1(S) − F1(missing): +0.0033 [-0.0071, +0.0136] (NB [-0.0206, +0.0271]) -> no difference
- 80nan (7 variants): missing 0.5789, naive 0.5724, K 0.5675, S 0.5728, truth 0.8383
  - F1(S) − F1(K): +0.0053 [-0.0035, +0.0141] (NB [-0.0149, +0.0255]) -> no difference
  - F1(K) − F1(missing): -0.0113 [-0.0182, -0.0045] (NB [-0.0270, +0.0043]) -> missing
  - F1(S) − F1(missing): -0.0061 [-0.0131, +0.0010] (NB [-0.0223, +0.0102]) -> no difference

### Beside: logistic regression, own (a gap is the mean with an indicator, or its own category)
- F1(S) − F1(K): +0.0001 [-0.0018, +0.0020] (NB [-0.0044, +0.0045]) -> no difference
- F1(K) − F1(missing): +0.0379 [+0.0348, +0.0410] (NB [+0.0307, +0.0451]) -> K
- F1(S) − F1(missing): +0.0380 [+0.0346, +0.0414] (NB [+0.0302, +0.0457]) -> S
- Levels: missing 0.6721, naive 0.6584, K 0.7099, S 0.7100, truth 0.8221

### Beside: forest fitted on the complete rows, scored on each table's test fold
- F1(S) − F1(K): +0.0062 [+0.0037, +0.0088] (NB [+0.0004, +0.0121]) -> S
- F1(K) − F1(missing): +0.0435 [+0.0397, +0.0473] (NB [+0.0348, +0.0522]) -> K
- F1(S) − F1(missing): +0.0497 [+0.0466, +0.0528] (NB [+0.0426, +0.0568]) -> S
- Levels: missing 0.6788, naive 0.6177, K 0.7223, S 0.7285, truth 0.8521

### Beside: logistic regression fitted on the complete rows
- F1(S) − F1(K): +0.0075 [+0.0052, +0.0098] (NB [+0.0021, +0.0128]) -> S
- F1(K) − F1(missing): +0.1196 [+0.1149, +0.1242] (NB [+0.1089, +0.1303]) -> K
- F1(S) − F1(missing): +0.1271 [+0.1228, +0.1313] (NB [+0.1172, +0.1369]) -> S
- Levels: missing 0.5832, naive 0.5849, K 0.7028, S 0.7103, truth 0.8221

### Per variant (forest, own), 15 fold values a variant
- kc2_20nan: missing 0.7158, naive 0.7083, K 0.6949, S 0.6930, truth 0.7133
  - F1(S) − F1(K): -0.0019 [-0.0137, +0.0099] (NB [-0.0290, +0.0252]) -> no difference
  - F1(K) − F1(missing): -0.0209 [-0.0421, +0.0002] (NB [-0.0696, +0.0277]) -> no difference
  - F1(S) − F1(missing): -0.0228 [-0.0416, -0.0041] (NB [-0.0660, +0.0203]) -> missing
- kc2_40nan: missing 0.7262, naive 0.7267, K 0.7170, S 0.7150, truth 0.7083
  - F1(S) − F1(K): -0.0019 [-0.0226, +0.0187] (NB [-0.0494, +0.0455]) -> no difference
  - F1(K) − F1(missing): -0.0092 [-0.0356, +0.0173] (NB [-0.0700, +0.0516]) -> no difference
  - F1(S) − F1(missing): -0.0111 [-0.0406, +0.0183] (NB [-0.0789, +0.0566]) -> no difference
- kc2_80nan: missing 0.7074, naive 0.7194, K 0.7037, S 0.7067, truth 0.7119
  - F1(S) − F1(K): +0.0030 [-0.0102, +0.0162] (NB [-0.0273, +0.0333]) -> no difference
  - F1(K) − F1(missing): -0.0037 [-0.0311, +0.0237] (NB [-0.0666, +0.0592]) -> no difference
  - F1(S) − F1(missing): -0.0007 [-0.0287, +0.0274] (NB [-0.0652, +0.0638]) -> no difference
- vehicle_20nan: missing 0.7036, naive 0.6886, K 0.7178, S 0.7192, truth 0.7435
  - F1(S) − F1(K): +0.0014 [-0.0078, +0.0107] (NB [-0.0198, +0.0227]) -> no difference
  - F1(K) − F1(missing): +0.0142 [+0.0013, +0.0272] (NB [-0.0155, +0.0440]) -> K
  - F1(S) − F1(missing): +0.0157 [+0.0042, +0.0271] (NB [-0.0106, +0.0419]) -> S
- vehicle_40nan: missing 0.6559, naive 0.6475, K 0.6861, S 0.6822, truth 0.7464
  - F1(S) − F1(K): -0.0038 [-0.0159, +0.0082] (NB [-0.0315, +0.0239]) -> no difference
  - F1(K) − F1(missing): +0.0302 [+0.0105, +0.0499] (NB [-0.0151, +0.0755]) -> K
  - F1(S) − F1(missing): +0.0263 [+0.0098, +0.0428] (NB [-0.0116, +0.0642]) -> S
- vehicle_80nan: missing 0.4391, naive 0.4482, K 0.4427, S 0.4490, truth 0.7467
  - F1(S) − F1(K): +0.0063 [-0.0172, +0.0298] (NB [-0.0476, +0.0603]) -> no difference
  - F1(K) − F1(missing): +0.0036 [-0.0218, +0.0290] (NB [-0.0548, +0.0621]) -> no difference
  - F1(S) − F1(missing): +0.0099 [-0.0189, +0.0388] (NB [-0.0563, +0.0762]) -> no difference
- biodeg_20nan: missing 0.8290, naive 0.8359, K 0.8446, S 0.8431, truth 0.8531
  - F1(S) − F1(K): -0.0015 [-0.0072, +0.0042] (NB [-0.0146, +0.0116]) -> no difference
  - F1(K) − F1(missing): +0.0155 [+0.0082, +0.0228] (NB [-0.0012, +0.0323]) -> K
  - F1(S) − F1(missing): +0.0140 [+0.0083, +0.0198] (NB [+0.0009, +0.0272]) -> S
- biodeg_40nan: missing 0.7986, naive 0.7886, K 0.8206, S 0.8250, truth 0.8523
  - F1(S) − F1(K): +0.0044 [-0.0043, +0.0132] (NB [-0.0157, +0.0246]) -> no difference
  - F1(K) − F1(missing): +0.0220 [+0.0089, +0.0350] (NB [-0.0080, +0.0519]) -> K
  - F1(S) − F1(missing): +0.0264 [+0.0145, +0.0383] (NB [-0.0010, +0.0537]) -> S
- biodeg_80nan: missing 0.6672, naive 0.6709, K 0.6705, S 0.6913, truth 0.8537
  - F1(S) − F1(K): +0.0208 [-0.0215, +0.0632] (NB [-0.0764, +0.1181]) -> no difference
  - F1(K) − F1(missing): +0.0032 [-0.0365, +0.0430] (NB [-0.0882, +0.0947]) -> no difference
  - F1(S) − F1(missing): +0.0241 [+0.0017, +0.0465] (NB [-0.0274, +0.0756]) -> S
- credit-g_20nan: missing 0.5679, naive 0.5796, K 0.6068, S 0.6046, truth 0.6657
  - F1(S) − F1(K): -0.0022 [-0.0182, +0.0138] (NB [-0.0390, +0.0345]) -> no difference
  - F1(K) − F1(missing): +0.0389 [+0.0196, +0.0582] (NB [-0.0054, +0.0832]) -> K
  - F1(S) − F1(missing): +0.0367 [+0.0213, +0.0521] (NB [+0.0013, +0.0721]) -> S
- credit-g_40nan: missing 0.5535, naive 0.5836, K 0.5540, S 0.5621, truth 0.6642
  - F1(S) − F1(K): +0.0081 [-0.0098, +0.0260] (NB [-0.0330, +0.0492]) -> no difference
  - F1(K) − F1(missing): +0.0005 [-0.0208, +0.0218] (NB [-0.0485, +0.0495]) -> no difference
  - F1(S) − F1(missing): +0.0086 [-0.0110, +0.0283] (NB [-0.0366, +0.0538]) -> no difference
- credit-g_60nan: missing 0.5329, naive 0.5230, K 0.5071, S 0.5240, truth 0.6647
  - F1(S) − F1(K): +0.0169 [-0.0019, +0.0356] (NB [-0.0262, +0.0599]) -> no difference
  - F1(K) − F1(missing): -0.0258 [-0.0518, +0.0002] (NB [-0.0856, +0.0340]) -> no difference
  - F1(S) − F1(missing): -0.0089 [-0.0360, +0.0181] (NB [-0.0712, +0.0533]) -> no difference
- credit-g_80nan: missing 0.5297, naive 0.5194, K 0.4754, S 0.4793, truth 0.6669
  - F1(S) − F1(K): +0.0039 [-0.0107, +0.0185] (NB [-0.0296, +0.0374]) -> no difference
  - F1(K) − F1(missing): -0.0543 [-0.0740, -0.0346] (NB [-0.0995, -0.0091]) -> missing
  - F1(S) − F1(missing): -0.0504 [-0.0722, -0.0286] (NB [-0.1005, -0.0003]) -> missing
- kr-vs-kp_20nan: missing 0.8941, naive 0.8747, K 0.8958, S 0.8938, truth 0.9906
  - F1(S) − F1(K): -0.0020 [-0.0074, +0.0034] (NB [-0.0145, +0.0105]) -> no difference
  - F1(K) − F1(missing): +0.0017 [-0.0032, +0.0067] (NB [-0.0097, +0.0132]) -> no difference
  - F1(S) − F1(missing): -0.0003 [-0.0062, +0.0057] (NB [-0.0140, +0.0134]) -> no difference
- kr-vs-kp_40nan: missing 0.8114, naive 0.7729, K 0.7922, S 0.7855, truth 0.9906
  - F1(S) − F1(K): -0.0067 [-0.0157, +0.0023] (NB [-0.0275, +0.0140]) -> no difference
  - F1(K) − F1(missing): -0.0192 [-0.0291, -0.0093] (NB [-0.0419, +0.0036]) -> missing
  - F1(S) − F1(missing): -0.0259 [-0.0324, -0.0194] (NB [-0.0409, -0.0109]) -> missing
- kr-vs-kp_60nan: missing 0.7072, naive 0.6766, K 0.6692, S 0.6666, truth 0.9906
  - F1(S) − F1(K): -0.0026 [-0.0166, +0.0113] (NB [-0.0347, +0.0294]) -> no difference
  - F1(K) − F1(missing): -0.0380 [-0.0520, -0.0240] (NB [-0.0703, -0.0058]) -> missing
  - F1(S) − F1(missing): -0.0406 [-0.0539, -0.0274] (NB [-0.0711, -0.0101]) -> missing
- kr-vs-kp_80nan: missing 0.6109, naive 0.5872, K 0.5878, S 0.5713, truth 0.9906
  - F1(S) − F1(K): -0.0165 [-0.0285, -0.0045] (NB [-0.0441, +0.0110]) -> K
  - F1(K) − F1(missing): -0.0231 [-0.0339, -0.0123] (NB [-0.0479, +0.0017]) -> missing
  - F1(S) − F1(missing): -0.0396 [-0.0527, -0.0264] (NB [-0.0698, -0.0094]) -> missing
- spambase_20nan: missing 0.9240, naive 0.9206, K 0.9210, S 0.9214, truth 0.9498
  - F1(S) − F1(K): +0.0004 [-0.0014, +0.0022] (NB [-0.0037, +0.0045]) -> no difference
  - F1(K) − F1(missing): -0.0030 [-0.0076, +0.0016] (NB [-0.0135, +0.0075]) -> no difference
  - F1(S) − F1(missing): -0.0026 [-0.0067, +0.0015] (NB [-0.0119, +0.0067]) -> no difference
- pendigits_20nan: missing 0.9682, naive 0.9610, K 0.9837, S 0.9838, truth 0.9916
  - F1(S) − F1(K): +0.0000 [-0.0007, +0.0008] (NB [-0.0018, +0.0019]) -> no difference
  - F1(K) − F1(missing): +0.0155 [+0.0142, +0.0167] (NB [+0.0126, +0.0184]) -> K
  - F1(S) − F1(missing): +0.0155 [+0.0145, +0.0166] (NB [+0.0132, +0.0179]) -> S
- pendigits_40nan: missing 0.9103, naive 0.8856, K 0.9519, S 0.9527, truth 0.9917
  - F1(S) − F1(K): +0.0008 [-0.0004, +0.0020] (NB [-0.0020, +0.0036]) -> no difference
  - F1(K) − F1(missing): +0.0416 [+0.0384, +0.0447] (NB [+0.0344, +0.0488]) -> K
  - F1(S) − F1(missing): +0.0424 [+0.0393, +0.0454] (NB [+0.0353, +0.0494]) -> S
- pendigits_60nan: missing 0.7829, naive 0.7458, K 0.8416, S 0.8422, truth 0.9915
  - F1(S) − F1(K): +0.0007 [-0.0023, +0.0036] (NB [-0.0061, +0.0074]) -> no difference
  - F1(K) − F1(missing): +0.0587 [+0.0532, +0.0641] (NB [+0.0462, +0.0712]) -> K
  - F1(S) − F1(missing): +0.0593 [+0.0542, +0.0645] (NB [+0.0474, +0.0712]) -> S
- pendigits_80nan: missing 0.5024, naive 0.4710, K 0.4991, S 0.5132, truth 0.9915
  - F1(S) − F1(K): +0.0142 [+0.0110, +0.0173] (NB [+0.0069, +0.0214]) -> S
  - F1(K) − F1(missing): -0.0034 [-0.0099, +0.0032] (NB [-0.0185, +0.0118]) -> no difference
  - F1(S) − F1(missing): +0.0108 [+0.0056, +0.0160] (NB [-0.0012, +0.0228]) -> S
- electricity_20nan: missing 0.8227, naive 0.8197, K 0.8298, S 0.8290, truth 0.9063
  - F1(S) − F1(K): -0.0008 [-0.0020, +0.0003] (NB [-0.0035, +0.0018]) -> no difference
  - F1(K) − F1(missing): +0.0071 [+0.0054, +0.0088] (NB [+0.0032, +0.0111]) -> K
  - F1(S) − F1(missing): +0.0063 [+0.0050, +0.0077] (NB [+0.0032, +0.0094]) -> S
- electricity_40nan: missing 0.7538, naive 0.7451, K 0.7555, S 0.7559, truth 0.9063
  - F1(S) − F1(K): +0.0005 [-0.0031, +0.0040] (NB [-0.0076, +0.0085]) -> no difference
  - F1(K) − F1(missing): +0.0017 [-0.0011, +0.0045] (NB [-0.0048, +0.0082]) -> no difference
  - F1(S) − F1(missing): +0.0022 [-0.0007, +0.0051] (NB [-0.0045, +0.0089]) -> no difference
- electricity_80nan: missing 0.5953, naive 0.5911, K 0.5935, S 0.5988, truth 0.9064
  - F1(S) − F1(K): +0.0053 [+0.0015, +0.0090] (NB [-0.0033, +0.0138]) -> S
  - F1(K) − F1(missing): -0.0018 [-0.0055, +0.0018] (NB [-0.0102, +0.0066]) -> no difference
  - F1(S) − F1(missing): +0.0034 [+0.0009, +0.0060] (NB [-0.0023, +0.0092]) -> S
- letter_20nan: missing 0.8581, naive 0.8493, K 0.8994, S 0.8994, truth 0.9654
  - F1(S) − F1(K): -0.0000 [-0.0011, +0.0010] (NB [-0.0024, +0.0024]) -> no difference
  - F1(K) − F1(missing): +0.0413 [+0.0387, +0.0439] (NB [+0.0354, +0.0473]) -> K
  - F1(S) − F1(missing): +0.0413 [+0.0389, +0.0437] (NB [+0.0357, +0.0468]) -> S
Variants, F1(S) − F1(K): S 2, no difference 23, K 1
Variants, F1(K) − F1(missing): K 10, no difference 12, missing 4
Variants, F1(S) − F1(missing): S 13, no difference 8, missing 5

### E186's per-variant calls on these seeds (E187; forest, own, 15 fold values a variant)
- F1(K) − F1(missing), E186 called K on 9: 8 repeat, 0 take the opposite call [], 1 no difference ['electricity_40nan']
- F1(K) − F1(missing), E186 called missing on 5: 4 repeat, 0 take the opposite call [], 1 no difference ['credit-g_60nan']
By-table split: 12 of 14 repeat, 0 opposite -> replicated; calls on variants E186 left at no difference: ['biodeg_20nan (K)', 'credit-g_20nan (K)']
- S − K carrier credit-g_60nan, F1(S) − F1(K): +0.0169 [-0.0019, +0.0356] (NB [-0.0262, +0.0599]) -> no difference
- S − K carrier pendigits_80nan, F1(S) − F1(K): +0.0142 [+0.0110, +0.0173] (NB [+0.0069, +0.0214]) -> S
Not fresh, E186's three seeds and these pooled (30 replicates), F1(S) − F1(K): +0.0027 [+0.0007, +0.0046] (NB [-0.0033, +0.0087]) -> S

## Manipulation check (E187): S − K on the output's scores (E182's scorer on the test-fold ledgers), seeds 1717/8484/9595
- `kc2_20nan` (seen): mean kept epoch K 193 → S 238
  - S − K on sw_score -0.0160 [-0.0307, -0.0013] better (K 0.4119, S 0.3959)
  - S − K on dist_score -0.0106 [-0.0193, -0.0020] better (K 0.2619, S 0.2513)
  - S − K on impute_score -0.0008 [-0.0348, +0.0331] none (K 0.5271, S 0.5262)
- `kc2_40nan` (seen): mean kept epoch K 215 → S 272
  - S − K on sw_score -0.0132 [-0.0295, +0.0031] none (K 0.3828, S 0.3696)
  - S − K on dist_score -0.0111 [-0.0260, +0.0037] none (K 0.2509, S 0.2397)
  - S − K on impute_score -0.0156 [-0.0457, +0.0145] none (K 0.5181, S 0.5025)
- `kc2_80nan` (seen): mean kept epoch K 253 → S 293
  - S − K on sw_score -0.0078 [-0.0155, -0.0001] better (K 0.4156, S 0.4078)
  - S − K on dist_score -0.0057 [-0.0133, +0.0019] none (K 0.3116, S 0.3059)
  - S − K on impute_score -0.0019 [-0.0071, +0.0032] none (K 0.6583, S 0.6564)
- `vehicle_20nan` (seen): mean kept epoch K 289 → S 283
  - S − K on sw_score -0.0027 [-0.0074, +0.0021] none (K 0.3633, S 0.3606)
  - S − K on dist_score -0.0029 [-0.0083, +0.0025] none (K 0.2015, S 0.1986)
  - S − K on impute_score -0.0024 [-0.0071, +0.0023] none (K 0.4128, S 0.4104)
- `vehicle_40nan` (seen): mean kept epoch K 282 → S 291
  - S − K on sw_score -0.0022 [-0.0079, +0.0035] none (K 0.3417, S 0.3395)
  - S − K on dist_score -0.0010 [-0.0059, +0.0040] none (K 0.2100, S 0.2091)
  - S − K on impute_score +0.0007 [-0.0054, +0.0069] none (K 0.4794, S 0.4801)
- `vehicle_80nan` (seen): mean kept epoch K 73 → S 243
  - S − K on sw_score -0.0617 [-0.0845, -0.0388] better (K 0.5125, S 0.4509)
  - S − K on dist_score -0.0501 [-0.0670, -0.0332] better (K 0.3991, S 0.3491)
  - S − K on impute_score -0.0042 [-0.0073, -0.0010] better (K 0.7848, S 0.7806)
- `biodeg_20nan` (seen): mean kept epoch K 275 → S 308
  - S − K on sw_score -0.0061 [-0.0107, -0.0015] better (K 0.4894, S 0.4833)
  - S − K on dist_score -0.0071 [-0.0138, -0.0003] better (K 0.2721, S 0.2651)
  - S − K on impute_score -0.0081 [-0.0146, -0.0016] better (K 0.5994, S 0.5913)
- `biodeg_40nan` (seen): mean kept epoch K 260 → S 304
  - S − K on sw_score -0.0082 [-0.0160, -0.0004] better (K 0.4630, S 0.4548)
  - S − K on dist_score -0.0084 [-0.0188, +0.0020] none (K 0.2974, S 0.2890)
  - S − K on impute_score -0.0028 [-0.0088, +0.0033] none (K 0.6659, S 0.6632)
- `biodeg_80nan` (seen): mean kept epoch K 261 → S 270
  - S − K on sw_score -0.0325 [-0.0756, +0.0106] none (K 0.7382, S 0.7057)
  - S − K on dist_score -0.0361 [-0.0947, +0.0225] none (K 0.6065, S 0.5704)
  - S − K on impute_score -0.0055 [-0.0164, +0.0053] none (K 0.9437, S 0.9382)
- `credit-g_20nan` (seen): mean kept epoch K 102 → S 228
  - S − K on sw_score -0.0461 [-0.0706, -0.0216] better (K 0.7165, S 0.6704)
  - S − K on dist_score -0.0722 [-0.1126, -0.0319] better (K 0.5146, S 0.4424)
  - S − K on impute_score +0.0053 [-0.0067, +0.0172] none (K 0.8850, S 0.8903)
- `credit-g_40nan` (seen): mean kept epoch K 88 → S 269
  - S − K on sw_score -0.0461 [-0.0673, -0.0249] better (K 0.6940, S 0.6479)
  - S − K on dist_score -0.0618 [-0.0912, -0.0325] better (K 0.5829, S 0.5211)
  - S − K on impute_score +0.0140 [+0.0057, +0.0222] worse (K 0.9217, S 0.9357)
- `credit-g_60nan` (seen): mean kept epoch K 43 → S 282
  - S − K on sw_score -0.1440 [-0.1953, -0.0928] better (K 0.7917, S 0.6477)
  - S − K on dist_score -0.1890 [-0.2521, -0.1259] better (K 0.7389, S 0.5499)
  - S − K on impute_score +0.0162 [+0.0018, +0.0306] none (K 0.9873, S 1.0035)
- `credit-g_80nan` (seen): mean kept epoch K 9 → S 252
  - S − K on sw_score -0.2945 [-0.3224, -0.2666] better (K 0.9441, S 0.6496)
  - S − K on dist_score -0.3329 [-0.3668, -0.2990] better (K 0.9429, S 0.6099)
  - S − K on impute_score +0.0357 [+0.0269, +0.0445] worse (K 1.0052, S 1.0409)
- `kr-vs-kp_20nan` (seen): mean kept epoch K 224 → S 288
  - S − K on sw_score -0.0300 [-0.0480, -0.0120] better (K 0.3297, S 0.2998)
  - S − K on dist_score -0.0438 [-0.0673, -0.0203] better (K 0.2190, S 0.1752)
  - S − K on impute_score -0.0124 [-0.0231, -0.0017] better (K 0.5998, S 0.5874)
- `kr-vs-kp_40nan` (seen): mean kept epoch K 302 → S 252
  - S − K on sw_score +0.0032 [-0.0216, +0.0279] none (K 0.4226, S 0.4258)
  - S − K on dist_score +0.0049 [-0.0210, +0.0308] none (K 0.3728, S 0.3777)
  - S − K on impute_score +0.0055 [-0.0019, +0.0129] none (K 0.7173, S 0.7228)
- `kr-vs-kp_60nan` (seen): mean kept epoch K 97 → S 310
  - S − K on sw_score -0.0426 [-0.0718, -0.0135] better (K 0.5384, S 0.4958)
  - S − K on dist_score -0.0407 [-0.0671, -0.0142] better (K 0.5283, S 0.4877)
  - S − K on impute_score +0.0052 [-0.0015, +0.0118] none (K 0.8608, S 0.8660)
- `kr-vs-kp_80nan` (seen): mean kept epoch K 21 → S 262
  - S − K on sw_score -0.2424 [-0.2918, -0.1931] better (K 0.8143, S 0.5718)
  - S − K on dist_score -0.2078 [-0.2508, -0.1649] better (K 0.8421, S 0.6343)
  - S − K on impute_score +0.0358 [+0.0284, +0.0433] worse (K 0.9705, S 1.0063)
- `spambase_20nan` (seen): mean kept epoch K 308 → S 307
  - S − K on sw_score -0.0156 [-0.0238, -0.0075] better (K 0.7305, S 0.7148)
  - S − K on dist_score -0.0034 [-0.0067, -0.0001] better (K 0.5008, S 0.4974)
  - S − K on impute_score +0.0004 [-0.0024, +0.0031] equivalent (K 0.8467, S 0.8471)
- `pendigits_20nan` (seen): mean kept epoch K 514 → S 484
  - S − K on sw_score -0.0006 [-0.0014, +0.0002] none (K 0.1281, S 0.1275)
  - S − K on dist_score -0.0004 [-0.0010, +0.0003] none (K 0.0761, S 0.0757)
  - S − K on impute_score +0.0003 [-0.0005, +0.0011] equivalent (K 0.3018, S 0.3021)
- `pendigits_40nan` (seen): mean kept epoch K 514 → S 514
  - S − K on sw_score -0.0009 [-0.0026, +0.0007] none (K 0.1480, S 0.1471)
  - S − K on dist_score -0.0006 [-0.0016, +0.0004] none (K 0.0966, S 0.0960)
  - S − K on impute_score +0.0000 [-0.0009, +0.0009] equivalent (K 0.3817, S 0.3817)
- `pendigits_60nan` (seen): mean kept epoch K 458 → S 514
  - S − K on sw_score -0.0022 [-0.0043, -0.0001] better (K 0.2658, S 0.2636)
  - S − K on dist_score -0.0017 [-0.0032, -0.0001] better (K 0.1818, S 0.1801)
  - S − K on impute_score -0.0028 [-0.0047, -0.0008] better (K 0.5571, S 0.5543)
- `pendigits_80nan` (seen): mean kept epoch K 51 → S 381
  - S − K on sw_score -0.0163 [-0.0211, -0.0115] better (K 0.6166, S 0.6003)
  - S − K on dist_score -0.0144 [-0.0177, -0.0110] better (K 0.4904, S 0.4760)
  - S − K on impute_score +0.0050 [+0.0038, +0.0061] worse (K 0.8277, S 0.8327)
- `electricity_20nan` (seen): mean kept epoch K 365 → S 352
  - S − K on sw_score -0.0121 [-0.0154, -0.0089] better (K 0.2771, S 0.2650)
  - S − K on dist_score -0.0065 [-0.0101, -0.0030] better (K 0.2511, S 0.2446)
  - S − K on impute_score -0.0051 [-0.0213, +0.0112] none (K 0.6201, S 0.6151)
- `electricity_40nan` (seen): mean kept epoch K 335 → S 291
  - S − K on sw_score +0.0010 [-0.0088, +0.0108] none (K 0.3740, S 0.3750)
  - S − K on dist_score +0.0047 [-0.0034, +0.0127] none (K 0.3305, S 0.3351)
  - S − K on impute_score -0.0014 [-0.0217, +0.0189] none (K 0.7559, S 0.7545)
- `electricity_80nan` (seen): mean kept epoch K 141 → S 249
  - S − K on sw_score -0.0015 [-0.0090, +0.0060] none (K 0.6439, S 0.6424)
  - S − K on dist_score -0.0081 [-0.0163, +0.0001] none (K 0.5818, S 0.5737)
  - S − K on impute_score -0.0072 [-0.0230, +0.0085] none (K 0.9050, S 0.8978)
- `letter_20nan` (seen): mean kept epoch K 522 → S 497
  - S − K on sw_score -0.0009 [-0.0022, +0.0004] none (K 0.1394, S 0.1385)
  - S − K on dist_score -0.0002 [-0.0006, +0.0002] equivalent (K 0.1637, S 0.1636)
  - S − K on impute_score +0.0024 [+0.0011, +0.0038] worse (K 0.4210, S 0.4234)

E183's rule, S − K on sw_score over the 26 variants the shuffled control passes: better on 16, worse on 0 [] -> supported
S − K on sw_score, all variants: better 16 / no difference 10 / worse 0 (fold-pair t, seed agreement)
S − K on dist_score, all variants: better 14 / no difference 11 / equivalent 1 / worse 0 (fold-pair t, seed agreement)
S − K on impute_score, all variants: better 4 / no difference 14 / equivalent 3 / worse 5 (fold-pair t, seed agreement)
```
