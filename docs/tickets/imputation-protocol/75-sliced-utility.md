# A three-way downstream classification: gaps left missing, the default fills, the fills chosen by `sw_score` (E186, T63; pre-registered 2026-11-22 14:38 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** E183 to E185 optimised the fills for the sliced Wasserstein score (`sw_score`) and read that score only.
  This study asks whether that optimisation is useful: does a classifier do better on the table filled that way?
- **The user's go** (2026-11-22): "Quero testar o quão 'útil' é a otimização da imputação por sliced w. Para isso rode
  uma tarefa de classificação 3 vias: dado simplesmente faltante, imputado pelo H otimizado por impute score e H
  optimizado por sliced w. [...] Eu quero um modelo de classificação não tão sofisticado". The design below is mine.
- **Disclosed:**
  - **"H" is read as the decoder's output, not ADR 0024's H.** ADR 0024's H is the checkpoint of the lowest
    validation loss, chosen by neither score. The two arms that fit the request are on disk: **K**, the default run
    (checkpoint by the validation gaps' `impute_score`, ADR 0015), and **S**, the same run with
    `--decode_metric sw_score` (E183). Both carry the default output: calibrated from 40% missing (ADR 0027), raw at
    20nan, the same readout in both arms.
  - **Nothing is trained and no GPU is used.** The fills are the fold ledgers of E184's and E185's runs (tickets 73
    and 74): 26 MCAR variants × seeds 101/202/303 × five folds, on gorgona8. Those studies' `sw_score` and
    `impute_score` were read before this design; no classifier was ever run on these fills.
  - E146 (ticket 52, a pilot on `dist_score` against `impute_score`) was read before this design: +0.014 for the
    distributional checkpoint when the classifier is fitted on complete rows, +0.003 (no difference) when it is
    fitted on the filled rows. The second is this study's primary protocol, so a null is the expected shape.
  - Smoke before this commit: `credit-g_40nan` seed 101, a study cell, run once to check the mechanics. It printed
    `1000 rows, 8000 gaps, 32000 gaps reached the forest's training rows`; its CSV was checked for shape (140 rows,
    no empty score) and deleted without reading a score. The tally was exercised on random numbers.
- **Script (this commit):** [scripts/experiments/sliced_utility.py](../../../scripts/experiments/sliced_utility.py);
  CSVs under `/scratch/diogoneiss/trident-2026-11-22-sliced-utility/csv`.

## Design and rule (fixed before the first cell)

- **Cells:** the 18 variants of E184 and the 8 of E185 × seeds 101/202/303 × five folds: 78 (variant, seed) files.
- **Tables,** per (variant, seed), in the ledgers' scaled space, the variant's features with its gaps:
  - `missing`: left as they are;
  - `K`, `S`: filled by that arm's ledger;
  - beside: `L` and `LS` (K and S with E184's loss term at w = 0.8), `naive` (the fold's mean/mode fill), `truth`.
  - Each row's gaps are filled by the fold in which the row is a test row, so no decoder trained on the row it
    fills, and none saw a label. The arms must hold the same cells and truths, every gap must have a ledger cell,
    and a filled table with a gap left is an error.
- **Classifier, fixed and untuned:**
  - **Primary: a random forest** (500 trees, scikit-learn's defaults otherwise; categories as ordinal codes). It
    takes a gap as it is, so `missing` is truly missing.
  - **Beside: logistic regression** (categories one-hot). It cannot take a gap: in `missing` a number's gap is the
    training mean with an indicator column, a category's gap a level of its own. That arm is a naive fill with
    indicators, stated now; the learners are never averaged.
- **Protocol:** macro-F1 on the imputation's own test folds.
  - **Primary, `own`:** the classifier is fitted on a table's rows outside the test fold and scored on that table's
    test fold (impute, then learn; for `missing`, learn with the gaps). The only protocol in which the missing arm
    is the user's.
  - **Beside, `complete`:** fitted on the truth-filled rows, scored on each table's test fold (E146's primary).
- **Readings,** each a difference pooled over the 26 variants per (seed, fold) replicate, 15 replicates, plain 95% t
  interval (E146's): above 0 the first table is the more useful, below 0 the second, otherwise no difference. The
  Nadeau–Bengio interval is printed beside; a verdict it does not share is reported as fragile.
  - **Primary: F1(S) − F1(K)**, forest, `own`.
  - **Three-way: F1(K) − F1(missing) and F1(S) − F1(missing)**, forest, `own`.
  - Beside, no rule: LS − S and L − K; naive and truth against missing; the levels of all seven tables; the three
    readings by missingness level (20/40/60/80nan), per variant (15 fold values, with counts), for logistic
    regression and under `complete`.
- **Read once:** `--tally`, when all 78 files are in; it refuses a partial read. No default changes whatever the
  result.
- **Limits stated now:** MCAR only; one seed set, already used to read `sw_score`; two untuned learners; a forest
  splits on gaps well, so `missing` is a strong arm under MCAR; macro-F1 only. A claim beyond this needs fresh seeds.

## Outcome (2026-11-22 14:54 GMT-3): S by the rule, small and fragile; whether filling helps a forest depends on the table

**Correction (2026-11-22 14:56 GMT-3), same printout, no new read.** The first text of this Outcome (commit b0afa6d) headed
it "imputing helps a forest at 20–40nan and not above" and read the split by missingness level. The printout does not
support that: the levels hold different tables (the 60nan group is three variants, two of them `credit-g` and
`kr-vs-kp`), `pendigits_60nan` gains +0.057 from K's fill, and `kr-vs-kp_40nan` already loses at 40nan. The split is
by table:

- **Filling helps** (K − missing above 0) on `pendigits` 20/40/60nan, `letter_20nan`, `vehicle` 20/40nan,
  `electricity` 20/40nan and `biodeg_40nan`: 9 variants, all numerical tables but `electricity` (7 numerical, 1
  categorical column).
- **Filling hurts** on `kr-vs-kp` 40/60/80nan (36 categorical columns, no numerical) and `credit-g` 60/80nan (13
  categorical, 7 numerical): 5 variants. S is worse than missing on 4 of them (not `credit-g_60nan`).
- **Not explained here.** The two losing tables are the ones with many categorical columns, and the calibrated
  output (ADR 0027) also starts at 40nan, where `kr-vs-kp` turns negative. This study cannot separate the two, and
  neither was tested.
- **Fragile by level.** Under the corrected interval K − missing holds only at 20nan; at 40, 60 and 80nan it
  includes 0. The two bullets below that state the 60–80nan losses and the last bullet's "about 0.01 at 20–40nan and
  nothing above" are to be read with this.
- **S − K "at 60nan and above"** rests on two variants (`credit-g_60nan`, `pendigits_80nan`); the other eight at
  those levels show no difference.

The 78 files were in at 2026-11-22 14:53 GMT-3 (15 minutes on gorgona8's CPUs), none failed; every run passed its
asserts (same cells and truths in the four arms, every gap with one ledger cell, gaps reaching the forest). Then
`--tally`, once. Macro-F1, the forest fitted on each table's own rows unless stated.

- **Primary, F1(S) − F1(K): +0.0035 [+0.0009, +0.0061], S by the rule, fragile.** The corrected interval
  [−0.0024, +0.0095] includes 0. The gain is all at 60–80nan (+0.0125 at 60nan, +0.0068 at 80nan, the second not
  separated from 0); at 20nan and 40nan it is +0.0005 and +0.0004. Per variant S is better on 2 (`credit-g_60nan`
  +0.029, `pendigits_80nan` +0.016), K on none, no difference on 24.
- **Three-way, against the gaps left missing: K +0.0033 [+0.0002, +0.0064], S +0.0068 [+0.0031, +0.0105],** both
  for the fill by the rule and both fragile. The sign turns with missingness:
  - at 20nan and 40nan filling helps (K +0.011 and +0.013, S +0.012 and +0.014);
  - at 60nan and 80nan K's fill is worse than leaving the gaps (−0.009 and −0.012), and S's is not different
    (+0.004 and −0.005).
  - Per variant: K better than missing on 9, worse on 5; S better on 12, worse on 4. Filling helps on `pendigits`,
    `letter_20nan`, `vehicle`, `electricity` 20–40nan and `biodeg_40nan`; it hurts on `kr-vs-kp` 40–80nan and
    `credit-g` 60–80nan (K) or 80nan (S), the two categorical-heavy tables.
- **Levels:** missing 0.714, mean/mode 0.703, K 0.717, S 0.721, the complete table 0.852. The fills recover 2% (K)
  and 5% (S) of the distance from the missing table to the complete one; the mean/mode fill is worse than leaving
  the gaps (−0.011).
- **The loss term adds nothing downstream:** L − K −0.001 and LS − S −0.002, no difference.
- **Beside, logistic regression (`own`):** S − K +0.0029 [+0.0015, +0.0043], fragile; K and S beat its missing arm
  by 0.036 and 0.039, but that arm is a mean fill with indicators, as stated before the run.
- **Beside, `complete`:** S − K +0.0085 [+0.0067, +0.0103] with the forest and +0.0060 with logistic regression,
  both holding under the corrected interval. As in E146, the selectors differ more when the classifier was fitted
  on complete rows than when it learns from the fills.
- **Reading:** choosing the checkpoint by `sw_score` does not hurt downstream use and helps a little at 60nan and
  above; the effect is a few thousandths of macro-F1, against 0.135 left to the complete table. With a forest,
  filling the gaps is worth about 0.01 at 20–40nan and nothing above. No default changes. One seed set, MCAR only.

Printout (unedited, all 160 lines):

```
## Downstream utility of the sliced selector (E186): macro-F1 on the test folds of 26 MCAR variants, seeds 101/202/303; a difference is pooled over the variants per (seed, fold) replicate, 15 replicates, plain 95% t interval (NB: corrected)
Primary, forest fitted on each table's own rows, F1(S) − F1(K): +0.0035 [+0.0009, +0.0061] (NB [-0.0024, +0.0095]) -> S
Three-way, same, F1(K) − F1(missing): +0.0033 [+0.0002, +0.0064] (NB [-0.0038, +0.0104]) -> K
Three-way, same, F1(S) − F1(missing): +0.0068 [+0.0031, +0.0105] (NB [-0.0017, +0.0154]) -> S
  beside, F1(LS) − F1(S): -0.0017 [-0.0040, +0.0006] (NB [-0.0071, +0.0037]) -> no difference
  beside, F1(L) − F1(K): -0.0008 [-0.0031, +0.0016] (NB [-0.0062, +0.0047]) -> no difference
  beside, F1(naive) − F1(missing): -0.0107 [-0.0136, -0.0077] (NB [-0.0175, -0.0039]) -> missing
  beside, F1(truth) − F1(missing): +0.1375 [+0.1327, +0.1422] (NB [+0.1266, +0.1484]) -> truth
Levels (forest, own): missing 0.7140, naive 0.7033, K 0.7173, S 0.7208, L 0.7165, LS 0.7191, truth 0.8515

### By missingness level (forest, own)
- 20nan (9 variants): missing 0.8096, naive 0.8054, K 0.8210, S 0.8215, L 0.8188, LS 0.8177, truth 0.8635
  - F1(S) − F1(K): +0.0005 [-0.0020, +0.0029] (NB [-0.0052, +0.0061]) -> no difference
  - F1(K) − F1(missing): +0.0114 [+0.0075, +0.0153] (NB [+0.0024, +0.0204]) -> K
  - F1(S) − F1(missing): +0.0119 [+0.0081, +0.0156] (NB [+0.0032, +0.0205]) -> S
- 40nan (7 variants): missing 0.7425, naive 0.7328, K 0.7557, S 0.7561, L 0.7537, LS 0.7558, truth 0.8365
  - F1(S) − F1(K): +0.0004 [-0.0039, +0.0047] (NB [-0.0095, +0.0104]) -> no difference
  - F1(K) − F1(missing): +0.0132 [+0.0072, +0.0191] (NB [-0.0006, +0.0269]) -> K
  - F1(S) − F1(missing): +0.0136 [+0.0084, +0.0188] (NB [+0.0016, +0.0255]) -> S
- 60nan (3 variants): missing 0.6758, naive 0.6497, K 0.6672, S 0.6797, L 0.6701, LS 0.6719, truth 0.8833
  - F1(S) − F1(K): +0.0125 [+0.0053, +0.0196] (NB [-0.0041, +0.0290]) -> S
  - F1(K) − F1(missing): -0.0085 [-0.0148, -0.0022] (NB [-0.0230, +0.0060]) -> missing
  - F1(S) − F1(missing): +0.0039 [-0.0024, +0.0103] (NB [-0.0106, +0.0185]) -> no difference
- 80nan (7 variants): missing 0.5788, naive 0.5656, K 0.5669, S 0.5736, L 0.5676, LS 0.5758, truth 0.8372
  - F1(S) − F1(K): +0.0068 [-0.0011, +0.0147] (NB [-0.0114, +0.0250]) -> no difference
  - F1(K) − F1(missing): -0.0120 [-0.0202, -0.0037] (NB [-0.0309, +0.0070]) -> missing
  - F1(S) − F1(missing): -0.0052 [-0.0129, +0.0025] (NB [-0.0229, +0.0125]) -> no difference

### Beside: logistic regression, own (a gap is the mean with an indicator, or its own category)
- F1(S) − F1(K): +0.0029 [+0.0015, +0.0043] (NB [-0.0003, +0.0061]) -> S
- F1(K) − F1(missing): +0.0362 [+0.0342, +0.0382] (NB [+0.0315, +0.0408]) -> K
- F1(S) − F1(missing): +0.0391 [+0.0366, +0.0415] (NB [+0.0334, +0.0447]) -> S
- F1(LS) − F1(S): -0.0028 [-0.0043, -0.0013] (NB [-0.0063, +0.0007]) -> S
- F1(L) − F1(K): -0.0009 [-0.0025, +0.0007] (NB [-0.0046, +0.0028]) -> no difference
- Levels: missing 0.6722, naive 0.6588, K 0.7084, S 0.7113, L 0.7074, LS 0.7085, truth 0.8226

### Beside: forest fitted on the complete rows, scored on each table's test fold
- F1(S) − F1(K): +0.0085 [+0.0067, +0.0103] (NB [+0.0043, +0.0127]) -> S
- F1(K) − F1(missing): +0.0404 [+0.0364, +0.0443] (NB [+0.0313, +0.0495]) -> K
- F1(S) − F1(missing): +0.0489 [+0.0447, +0.0531] (NB [+0.0393, +0.0585]) -> S
- F1(LS) − F1(S): -0.0026 [-0.0047, -0.0005] (NB [-0.0074, +0.0022]) -> S
- F1(L) − F1(K): +0.0001 [-0.0022, +0.0024] (NB [-0.0052, +0.0053]) -> no difference
- Levels: missing 0.6777, naive 0.6183, K 0.7180, S 0.7265, L 0.7181, LS 0.7240, truth 0.8515

### Beside: logistic regression fitted on the complete rows
- F1(S) − F1(K): +0.0060 [+0.0040, +0.0081] (NB [+0.0013, +0.0107]) -> S
- F1(K) − F1(missing): +0.1185 [+0.1140, +0.1231] (NB [+0.1081, +0.1290]) -> K
- F1(S) − F1(missing): +0.1246 [+0.1200, +0.1291] (NB [+0.1141, +0.1350]) -> S
- F1(LS) − F1(S): -0.0031 [-0.0048, -0.0014] (NB [-0.0070, +0.0009]) -> S
- F1(L) − F1(K): -0.0018 [-0.0038, +0.0001] (NB [-0.0063, +0.0027]) -> no difference
- Levels: missing 0.5839, naive 0.5852, K 0.7024, S 0.7085, L 0.7006, LS 0.7054, truth 0.8226

### Per variant (forest, own), 15 fold values a variant
- kc2_20nan: missing 0.7096, naive 0.7079, K 0.7012, S 0.6985, L 0.6968, LS 0.6873, truth 0.7093
  - F1(S) − F1(K): -0.0027 [-0.0150, +0.0096] (NB [-0.0310, +0.0256]) -> no difference
  - F1(K) − F1(missing): -0.0084 [-0.0295, +0.0127] (NB [-0.0570, +0.0402]) -> no difference
  - F1(S) − F1(missing): -0.0111 [-0.0369, +0.0148] (NB [-0.0705, +0.0483]) -> no difference
- kc2_40nan: missing 0.7195, naive 0.7067, K 0.7115, S 0.7106, L 0.7038, LS 0.7155, truth 0.7056
  - F1(S) − F1(K): -0.0008 [-0.0188, +0.0171] (NB [-0.0421, +0.0405]) -> no difference
  - F1(K) − F1(missing): -0.0080 [-0.0341, +0.0181] (NB [-0.0680, +0.0520]) -> no difference
  - F1(S) − F1(missing): -0.0089 [-0.0302, +0.0125] (NB [-0.0580, +0.0403]) -> no difference
- kc2_80nan: missing 0.6988, naive 0.7080, K 0.7061, S 0.6989, L 0.7054, LS 0.7035, truth 0.7128
  - F1(S) − F1(K): -0.0072 [-0.0337, +0.0194] (NB [-0.0682, +0.0539]) -> no difference
  - F1(K) − F1(missing): +0.0073 [-0.0203, +0.0348] (NB [-0.0561, +0.0706]) -> no difference
  - F1(S) − F1(missing): +0.0001 [-0.0342, +0.0344] (NB [-0.0788, +0.0790]) -> no difference
- vehicle_20nan: missing 0.6976, naive 0.6894, K 0.7238, S 0.7233, L 0.7100, LS 0.7132, truth 0.7387
  - F1(S) − F1(K): -0.0005 [-0.0059, +0.0049] (NB [-0.0129, +0.0118]) -> no difference
  - F1(K) − F1(missing): +0.0262 [+0.0125, +0.0398] (NB [-0.0052, +0.0575]) -> K
  - F1(S) − F1(missing): +0.0257 [+0.0134, +0.0379] (NB [-0.0025, +0.0539]) -> S
- vehicle_40nan: missing 0.6530, naive 0.6403, K 0.6821, S 0.6895, L 0.6820, LS 0.6762, truth 0.7404
  - F1(S) − F1(K): +0.0075 [-0.0048, +0.0197] (NB [-0.0207, +0.0357]) -> no difference
  - F1(K) − F1(missing): +0.0290 [+0.0123, +0.0458] (NB [-0.0094, +0.0675]) -> K
  - F1(S) − F1(missing): +0.0365 [+0.0224, +0.0507] (NB [+0.0040, +0.0691]) -> S
- vehicle_80nan: missing 0.4410, naive 0.4416, K 0.4446, S 0.4623, L 0.4296, LS 0.4588, truth 0.7397
  - F1(S) − F1(K): +0.0177 [-0.0081, +0.0435] (NB [-0.0416, +0.0770]) -> no difference
  - F1(K) − F1(missing): +0.0036 [-0.0104, +0.0175] (NB [-0.0286, +0.0357]) -> no difference
  - F1(S) − F1(missing): +0.0213 [+0.0033, +0.0393] (NB [-0.0201, +0.0627]) -> S
- biodeg_20nan: missing 0.8360, naive 0.8334, K 0.8451, S 0.8465, L 0.8447, LS 0.8457, truth 0.8510
  - F1(S) − F1(K): +0.0014 [-0.0054, +0.0082] (NB [-0.0141, +0.0170]) -> no difference
  - F1(K) − F1(missing): +0.0091 [-0.0012, +0.0194] (NB [-0.0146, +0.0328]) -> no difference
  - F1(S) − F1(missing): +0.0105 [-0.0007, +0.0217] (NB [-0.0154, +0.0364]) -> no difference
- biodeg_40nan: missing 0.7976, naive 0.7895, K 0.8245, S 0.8270, L 0.8270, LS 0.8292, truth 0.8508
  - F1(S) − F1(K): +0.0025 [-0.0041, +0.0091] (NB [-0.0126, +0.0176]) -> no difference
  - F1(K) − F1(missing): +0.0269 [+0.0131, +0.0408] (NB [-0.0049, +0.0588]) -> K
  - F1(S) − F1(missing): +0.0294 [+0.0132, +0.0457] (NB [-0.0080, +0.0668]) -> S
- biodeg_80nan: missing 0.6728, naive 0.6519, K 0.6692, S 0.6885, L 0.6857, LS 0.6940, truth 0.8508
  - F1(S) − F1(K): +0.0194 [-0.0037, +0.0424] (NB [-0.0335, +0.0723]) -> no difference
  - F1(K) − F1(missing): -0.0036 [-0.0376, +0.0305] (NB [-0.0819, +0.0747]) -> no difference
  - F1(S) − F1(missing): +0.0158 [-0.0072, +0.0388] (NB [-0.0371, +0.0687]) -> no difference
- credit-g_20nan: missing 0.5784, naive 0.5935, K 0.5949, S 0.6005, L 0.5959, LS 0.6089, truth 0.6677
  - F1(S) − F1(K): +0.0056 [-0.0118, +0.0231] (NB [-0.0344, +0.0457]) -> no difference
  - F1(K) − F1(missing): +0.0165 [-0.0063, +0.0393] (NB [-0.0359, +0.0689]) -> no difference
  - F1(S) − F1(missing): +0.0221 [+0.0032, +0.0410] (NB [-0.0214, +0.0657]) -> S
- credit-g_40nan: missing 0.5495, naive 0.5853, K 0.5648, S 0.5653, L 0.5624, LS 0.5657, truth 0.6699
  - F1(S) − F1(K): +0.0004 [-0.0188, +0.0197] (NB [-0.0438, +0.0446]) -> no difference
  - F1(K) − F1(missing): +0.0153 [-0.0050, +0.0357] (NB [-0.0314, +0.0621]) -> no difference
  - F1(S) − F1(missing): +0.0158 [-0.0085, +0.0400] (NB [-0.0400, +0.0715]) -> no difference
- credit-g_60nan: missing 0.5322, naive 0.5252, K 0.4929, S 0.5218, L 0.5023, LS 0.5059, truth 0.6668
  - F1(S) − F1(K): +0.0289 [+0.0086, +0.0493] (NB [-0.0179, +0.0757]) -> S
  - F1(K) − F1(missing): -0.0393 [-0.0582, -0.0204] (NB [-0.0827, +0.0042]) -> missing
  - F1(S) − F1(missing): -0.0103 [-0.0290, +0.0083] (NB [-0.0533, +0.0326]) -> no difference
- credit-g_80nan: missing 0.5389, naive 0.4980, K 0.4759, S 0.4895, L 0.4728, LS 0.4917, truth 0.6682
  - F1(S) − F1(K): +0.0136 [-0.0105, +0.0378] (NB [-0.0419, +0.0691]) -> no difference
  - F1(K) − F1(missing): -0.0631 [-0.0846, -0.0415] (NB [-0.1126, -0.0135]) -> missing
  - F1(S) − F1(missing): -0.0494 [-0.0726, -0.0263] (NB [-0.1026, +0.0038]) -> missing
- kr-vs-kp_20nan: missing 0.8932, naive 0.8770, K 0.8914, S 0.8918, L 0.8948, LS 0.8919, truth 0.9910
  - F1(S) − F1(K): +0.0004 [-0.0043, +0.0051] (NB [-0.0104, +0.0112]) -> no difference
  - F1(K) − F1(missing): -0.0018 [-0.0075, +0.0039] (NB [-0.0149, +0.0113]) -> no difference
  - F1(S) − F1(missing): -0.0014 [-0.0092, +0.0065] (NB [-0.0194, +0.0167]) -> no difference
- kr-vs-kp_40nan: missing 0.8130, naive 0.7730, K 0.7948, S 0.7908, L 0.7921, LS 0.7972, truth 0.9910
  - F1(S) − F1(K): -0.0040 [-0.0094, +0.0014] (NB [-0.0163, +0.0084]) -> no difference
  - F1(K) − F1(missing): -0.0182 [-0.0282, -0.0082] (NB [-0.0411, +0.0048]) -> missing
  - F1(S) − F1(missing): -0.0221 [-0.0309, -0.0133] (NB [-0.0424, -0.0019]) -> missing
- kr-vs-kp_60nan: missing 0.7114, naive 0.6770, K 0.6684, S 0.6749, L 0.6686, LS 0.6702, truth 0.9910
  - F1(S) − F1(K): +0.0065 [-0.0075, +0.0206] (NB [-0.0258, +0.0389]) -> no difference
  - F1(K) − F1(missing): -0.0430 [-0.0546, -0.0314] (NB [-0.0697, -0.0163]) -> missing
  - F1(S) − F1(missing): -0.0365 [-0.0466, -0.0263] (NB [-0.0598, -0.0132]) -> missing
- kr-vs-kp_80nan: missing 0.6040, naive 0.5887, K 0.5760, S 0.5646, L 0.5885, LS 0.5682, truth 0.9910
  - F1(S) − F1(K): -0.0115 [-0.0250, +0.0021] (NB [-0.0426, +0.0197]) -> no difference
  - F1(K) − F1(missing): -0.0280 [-0.0366, -0.0194] (NB [-0.0477, -0.0083]) -> missing
  - F1(S) − F1(missing): -0.0394 [-0.0494, -0.0295] (NB [-0.0624, -0.0165]) -> missing
- spambase_20nan: missing 0.9225, naive 0.9190, K 0.9216, S 0.9217, L 0.9204, LS 0.9208, truth 0.9506
  - F1(S) − F1(K): +0.0001 [-0.0018, +0.0020] (NB [-0.0043, +0.0044]) -> no difference
  - F1(K) − F1(missing): -0.0009 [-0.0049, +0.0031] (NB [-0.0101, +0.0083]) -> no difference
  - F1(S) − F1(missing): -0.0009 [-0.0042, +0.0025] (NB [-0.0086, +0.0069]) -> no difference
- pendigits_20nan: missing 0.9682, naive 0.9602, K 0.9832, S 0.9832, L 0.9826, LS 0.9823, truth 0.9919
  - F1(S) − F1(K): +0.0000 [-0.0006, +0.0007] (NB [-0.0014, +0.0015]) -> no difference
  - F1(K) − F1(missing): +0.0150 [+0.0131, +0.0169] (NB [+0.0106, +0.0194]) -> K
  - F1(S) − F1(missing): +0.0151 [+0.0130, +0.0171] (NB [+0.0104, +0.0197]) -> S
- pendigits_40nan: missing 0.9103, naive 0.8873, K 0.9534, S 0.9525, L 0.9515, LS 0.9502, truth 0.9920
  - F1(S) − F1(K): -0.0010 [-0.0021, +0.0002] (NB [-0.0036, +0.0017]) -> no difference
  - F1(K) − F1(missing): +0.0431 [+0.0387, +0.0475] (NB [+0.0330, +0.0532]) -> K
  - F1(S) − F1(missing): +0.0421 [+0.0376, +0.0467] (NB [+0.0317, +0.0526]) -> S
- pendigits_60nan: missing 0.7837, naive 0.7468, K 0.8404, S 0.8424, L 0.8396, LS 0.8397, truth 0.9920
  - F1(S) − F1(K): +0.0019 [-0.0002, +0.0041] (NB [-0.0030, +0.0068]) -> no difference
  - F1(K) − F1(missing): +0.0567 [+0.0521, +0.0614] (NB [+0.0460, +0.0674]) -> K
  - F1(S) − F1(missing): +0.0586 [+0.0539, +0.0634] (NB [+0.0477, +0.0695]) -> S
- pendigits_80nan: missing 0.5030, naive 0.4797, K 0.5027, S 0.5182, L 0.4964, LS 0.5165, truth 0.9919
  - F1(S) − F1(K): +0.0155 [+0.0111, +0.0199] (NB [+0.0054, +0.0256]) -> S
  - F1(K) − F1(missing): -0.0003 [-0.0041, +0.0034] (NB [-0.0089, +0.0082]) -> no difference
  - F1(S) − F1(missing): +0.0152 [+0.0110, +0.0193] (NB [+0.0056, +0.0247]) -> S
- electricity_20nan: missing 0.8230, naive 0.8191, K 0.8294, S 0.8295, L 0.8285, LS 0.8288, truth 0.9062
  - F1(S) − F1(K): +0.0002 [-0.0009, +0.0013] (NB [-0.0024, +0.0027]) -> no difference
  - F1(K) − F1(missing): +0.0064 [+0.0044, +0.0084] (NB [+0.0019, +0.0109]) -> K
  - F1(S) − F1(missing): +0.0066 [+0.0041, +0.0090] (NB [+0.0010, +0.0121]) -> S
- electricity_40nan: missing 0.7547, naive 0.7472, K 0.7586, S 0.7570, L 0.7574, LS 0.7568, truth 0.9061
  - F1(S) − F1(K): -0.0016 [-0.0036, +0.0004] (NB [-0.0062, +0.0030]) -> no difference
  - F1(K) − F1(missing): +0.0039 [+0.0014, +0.0065] (NB [-0.0019, +0.0098]) -> K
  - F1(S) − F1(missing): +0.0023 [+0.0002, +0.0044] (NB [-0.0025, +0.0072]) -> S
- electricity_80nan: missing 0.5932, naive 0.5909, K 0.5937, S 0.5935, L 0.5948, LS 0.5977, truth 0.9062
  - F1(S) − F1(K): -0.0002 [-0.0044, +0.0040] (NB [-0.0098, +0.0095]) -> no difference
  - F1(K) − F1(missing): +0.0005 [-0.0041, +0.0050] (NB [-0.0101, +0.0110]) -> no difference
  - F1(S) − F1(missing): +0.0003 [-0.0040, +0.0045] (NB [-0.0095, +0.0101]) -> no difference
- letter_20nan: missing 0.8580, naive 0.8489, K 0.8988, S 0.8984, L 0.8951, LS 0.8807, truth 0.9651
  - F1(S) − F1(K): -0.0004 [-0.0014, +0.0007] (NB [-0.0028, +0.0021]) -> no difference
  - F1(K) − F1(missing): +0.0408 [+0.0382, +0.0434] (NB [+0.0349, +0.0467]) -> K
  - F1(S) − F1(missing): +0.0404 [+0.0379, +0.0430] (NB [+0.0346, +0.0462]) -> S
Variants, F1(S) − F1(K): S 2, no difference 24, K 0
Variants, F1(K) − F1(missing): K 9, no difference 12, missing 5
Variants, F1(S) − F1(missing): S 12, no difference 10, missing 4
```
