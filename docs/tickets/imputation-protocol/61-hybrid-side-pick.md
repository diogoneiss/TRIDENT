# A deployable rule that picks the hybrid's other side, TabPFN or missForest (E165, T54; pre-registered 2026-11-08 21:14 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** E162 left this open: the missForest side was chosen with the complete table, which a deployment does not
  have. This rule picks the side from the same hidden validation cells that choose the hybrid's weights.
- **The user's go:** "faça 1 2 e 3" (2026-11-08), on "2. Regra que escolhe o outro lado do híbrido, TabPFN ou
  missForest, sem a tabela completa".
- **Disclosure:** the same as E164's ([60](60-missforest-hybrid-generality.md)). Nothing of E164 has been read; both
  are pre-registered before either's first cell.

## Design

- **Cells:** E164's 114 pairs; no cell of its own.
- **The rule (`pick`, `missforest_hybrid.both_sides`), per fold:**
  - fit the `table` rule twice on fold k's validation cells, once with TabPFN and once with missForest as the
    other side;
  - score each hybrid on those same validation cells, with the fold's mean/mode denominators;
  - take the side that scores lower, keeping TabPFN, ADR 0031 decision 6's side, on a tie.
- **Stated:** the side is chosen on the cells that choose the weights, one more in-sample choice on top of the
  `table` rule's. A within-group cross-fit is the upgrade if the result reads optimistic. It is not built now.

## Measures and rule (fixed now)

- **Verdict:** as E164.
- **Primary,** `pick` on the 38 variants, against `table_tabpfn`, the accepted rule, and against H:
  - **supported** if it is worse than H on none, worse than `table_tabpfn` on at most 1, and better than
    `table_tabpfn` on at least 2;
  - **refuted** if it is worse than `table_tabpfn` on 3 or more, or better on none;
  - **mixed** otherwise.
- **Secondary:** `pick` − `table_missforest`, and how often it took missForest.
- **Read:** once, after E164's read, with all 114 pairs in, `missforest_generality.py --tally e165` on medusa1.
- **What it leads to:** whether ADR 0031 decision 6 could name a side rule is the user's call.

## Outcome (2026-11-08 22:05 GMT-3)

**Mixed.** Read once at 22:02 GMT-3 on medusa1, after E164's read, on its 114 pairs. Unedited printout:

```
## e90
- `electricity_40nan`: H 0.7544, tabpfn 0.7427, missforest 0.7899, table_tabpfn 0.7404, table_missforest 0.7488, pick 0.7404; pick took missForest on 27% of folds
  - pick − H -0.0140 [-0.0350, +0.0070] none
  - pick − table_tabpfn +0.0000 [-0.0011, +0.0011] equivalent
  - pick − table_missforest -0.0085 [-0.0192, +0.0023] none
- `electricity_80nan`: H 0.8981, tabpfn 0.8907, missforest 1.0152, table_tabpfn 0.8897, table_missforest 0.8985, pick 0.8897; pick took missForest on 0% of folds
  - pick − H -0.0084 [-0.0101, -0.0066] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0088 [-0.0108, -0.0069] better
- `vehicle_80nan` (trails): H 0.8119, tabpfn 0.7536, missforest 0.7683, table_tabpfn 0.7539, table_missforest 0.7476, pick 0.7514; pick took missForest on 40% of folds
  - pick − H -0.0605 [-0.0654, -0.0557] better
  - pick − table_tabpfn -0.0025 [-0.0063, +0.0014] equivalent
  - pick − table_missforest +0.0038 [-0.0025, +0.0100] none
- `biodeg_80nan` (trails): H 0.9855, tabpfn 0.9188, missforest 0.9255, table_tabpfn 0.9162, table_missforest 0.9137, pick 0.9152; pick took missForest on 40% of folds
  - pick − H -0.0703 [-0.0776, -0.0630] better
  - pick − table_tabpfn -0.0010 [-0.0035, +0.0016] equivalent
  - pick − table_missforest +0.0015 [-0.0018, +0.0047] equivalent
- `kr-vs-kp_40nan`: H 0.7256, tabpfn 0.7118, missforest 0.7996, table_tabpfn 0.7184, table_missforest 0.7256, pick 0.7184; pick took missForest on 0% of folds
  - pick − H -0.0072 [-0.0132, -0.0012] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0072 [-0.0132, -0.0012] better
- `kr-vs-kp_60nan`: H 0.8715, tabpfn 0.8301, missforest 0.9381, table_tabpfn 0.8301, table_missforest 0.8715, pick 0.8301; pick took missForest on 0% of folds
  - pick − H -0.0414 [-0.0474, -0.0355] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0414 [-0.0474, -0.0355] better
- `kr-vs-kp_80nan`: H 1.0096, tabpfn 0.9406, missforest 1.0155, table_tabpfn 0.9445, table_missforest 1.0052, pick 0.9445; pick took missForest on 0% of folds
  - pick − H -0.0652 [-0.0809, -0.0494] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0608 [-0.0747, -0.0469] better
- `credit-g_40nan` (trails): H 0.9378, tabpfn 0.8999, missforest 0.9272, table_tabpfn 0.8997, table_missforest 0.9246, pick 0.9020; pick took missForest on 20% of folds
  - pick − H -0.0357 [-0.0488, -0.0226] better
  - pick − table_tabpfn +0.0024 [-0.0019, +0.0066] equivalent
  - pick − table_missforest -0.0226 [-0.0316, -0.0135] better
- `credit-g_60nan`: H 1.0018, tabpfn 0.9510, missforest 1.0142, table_tabpfn 0.9564, table_missforest 0.9945, pick 0.9564; pick took missForest on 0% of folds
  - pick − H -0.0454 [-0.0590, -0.0317] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0381 [-0.0516, -0.0246] better
- `credit-g_80nan`: H 1.0382, tabpfn 1.0028, missforest 1.0922, table_tabpfn 1.0077, table_missforest 1.0326, pick 1.0069; pick took missForest on 13% of folds
  - pick − H -0.0313 [-0.0494, -0.0131] better
  - pick − table_tabpfn -0.0008 [-0.0019, +0.0004] equivalent
  - pick − table_missforest -0.0257 [-0.0409, -0.0104] better
## e91
- `biodeg_40nan`: H 0.6832, tabpfn 0.7149, missforest 0.6941, table_tabpfn 0.6616, table_missforest 0.6594, pick 0.6581; pick took missForest on 80% of folds
  - pick − H -0.0251 [-0.0347, -0.0154] better
  - pick − table_tabpfn -0.0034 [-0.0096, +0.0027] none
  - pick − table_missforest -0.0013 [-0.0064, +0.0038] equivalent
- `kc2_40nan`: H 0.5057, tabpfn 0.5067, missforest 0.5825, table_tabpfn 0.4883, table_missforest 0.5038, pick 0.4958; pick took missForest on 20% of folds
  - pick − H -0.0099 [-0.0482, +0.0284] none
  - pick − table_tabpfn +0.0075 [-0.0037, +0.0187] none
  - pick − table_missforest -0.0080 [-0.0264, +0.0105] none
- `kc2_80nan`: H 0.6802, tabpfn 0.7396, missforest 0.7511, table_tabpfn 0.6811, table_missforest 0.6806, pick 0.6832; pick took missForest on 20% of folds
  - pick − H +0.0030 [-0.0291, +0.0350] none
  - pick − table_tabpfn +0.0021 [-0.0041, +0.0083] none
  - pick − table_missforest +0.0026 [-0.0082, +0.0134] none
- `pendigits_40nan`: H 0.4274, tabpfn 0.4363, missforest 0.4969, table_tabpfn 0.4151, table_missforest 0.4155, pick 0.4155; pick took missForest on 40% of folds
  - pick − H -0.0119 [-0.0129, -0.0110] better
  - pick − table_tabpfn +0.0003 [-0.0003, +0.0009] equivalent
  - pick − table_missforest -0.0000 [-0.0009, +0.0009] equivalent
- `pendigits_60nan`: H 0.5902, tabpfn 0.6038, missforest 0.7023, table_tabpfn 0.5763, table_missforest 0.5791, pick 0.5763; pick took missForest on 0% of folds
  - pick − H -0.0139 [-0.0155, -0.0124] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0028 [-0.0048, -0.0008] better
- `pendigits_80nan`: H 0.8360, tabpfn 0.8144, missforest 0.9420, table_tabpfn 0.8108, table_missforest 0.8237, pick 0.8108; pick took missForest on 0% of folds
  - pick − H -0.0253 [-0.0295, -0.0210] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0129 [-0.0156, -0.0102] better
- `vehicle_40nan`: H 0.4964, tabpfn 0.5008, missforest 0.5229, table_tabpfn 0.4829, table_missforest 0.4900, pick 0.4857; pick took missForest on 60% of folds
  - pick − H -0.0106 [-0.0207, -0.0006] better
  - pick − table_tabpfn +0.0028 [-0.0034, +0.0090] none
  - pick − table_missforest -0.0042 [-0.0090, +0.0005] none
## e93
- `biodeg_probit0p25_60nan` (trails): H 0.9114, tabpfn 0.9150, missforest 0.8842, table_tabpfn 0.9048, table_missforest 0.8844, pick 0.8952; pick took missForest on 47% of folds
  - pick − H -0.0162 [-0.0232, -0.0093] better
  - pick − table_tabpfn -0.0096 [-0.0164, -0.0028] better
  - pick − table_missforest +0.0108 [+0.0047, +0.0169] worse
- `biodeg_probit_60nan` (trails): H 0.9123, tabpfn 0.9246, missforest 0.8954, table_tabpfn 0.9140, table_missforest 0.8943, pick 0.8986; pick took missForest on 87% of folds
  - pick − H -0.0137 [-0.0196, -0.0078] better
  - pick − table_tabpfn -0.0154 [-0.0223, -0.0084] better
  - pick − table_missforest +0.0043 [-0.0034, +0.0120] none
- `biodeg_probit1p0_60nan`: H 0.9306, tabpfn 0.9379, missforest 0.9339, table_tabpfn 0.9300, table_missforest 0.9257, pick 0.9290; pick took missForest on 20% of folds
  - pick − H -0.0016 [-0.0040, +0.0008] equivalent
  - pick − table_tabpfn -0.0010 [-0.0022, +0.0002] equivalent
  - pick − table_missforest +0.0033 [+0.0005, +0.0061] worse
- `biodeg_probit2p0_60nan`: H 0.9321, tabpfn 0.9545, missforest 0.9527, table_tabpfn 0.9380, table_missforest 0.9337, pick 0.9380; pick took missForest on 0% of folds
  - pick − H +0.0059 [-0.0003, +0.0121] none
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest +0.0043 [-0.0005, +0.0091] equivalent
- `vehicle_probit_60nan` (trails): H 0.6308, tabpfn 0.6225, missforest 0.6212, table_tabpfn 0.6139, table_missforest 0.6103, pick 0.6106; pick took missForest on 87% of folds
  - pick − H -0.0202 [-0.0229, -0.0175] better
  - pick − table_tabpfn -0.0033 [-0.0065, -0.0001] better
  - pick − table_missforest +0.0003 [-0.0017, +0.0023] equivalent
- `kc2_probit_60nan`: H 0.7253, tabpfn 0.7757, missforest 0.7538, table_tabpfn 0.7317, table_missforest 0.7316, pick 0.7364; pick took missForest on 33% of folds
  - pick − H +0.0111 [-0.0168, +0.0389] none
  - pick − table_tabpfn +0.0047 [+0.0001, +0.0094] worse
  - pick − table_missforest +0.0047 [+0.0005, +0.0090] worse
- `spambase_probit_60nan`: H 0.9597, tabpfn 0.9955, missforest 0.9609, table_tabpfn 0.9687, table_missforest 0.9531, pick 0.9659; pick took missForest on 27% of folds
  - pick − H +0.0062 [+0.0009, +0.0116] worse
  - pick − table_tabpfn -0.0028 [-0.0058, +0.0003] equivalent
  - pick − table_missforest +0.0129 [+0.0076, +0.0181] worse
- `biodeg_marf_60nan`: H 0.6864, tabpfn 0.7056, missforest 0.7206, table_tabpfn 0.6643, table_missforest 0.6690, pick 0.6643; pick took missForest on 20% of folds
  - pick − H -0.0221 [-0.0327, -0.0115] better
  - pick − table_tabpfn +0.0000 [-0.0020, +0.0020] equivalent
  - pick − table_missforest -0.0047 [-0.0084, -0.0010] better
- `kc2_marf_60nan`: H 0.5987, tabpfn 0.6701, missforest 0.7134, table_tabpfn 0.6078, table_missforest 0.6200, pick 0.6169; pick took missForest on 47% of folds
  - pick − H +0.0182 [+0.0005, +0.0359] worse
  - pick − table_tabpfn +0.0091 [-0.0023, +0.0205] none
  - pick − table_missforest -0.0031 [-0.0074, +0.0013] none
- `vehicle_marf_60nan`: H 0.4510, tabpfn 0.4445, missforest 0.4947, table_tabpfn 0.4356, table_missforest 0.4348, pick 0.4347; pick took missForest on 40% of folds
  - pick − H -0.0163 [-0.0231, -0.0095] better
  - pick − table_tabpfn -0.0009 [-0.0044, +0.0026] none
  - pick − table_missforest -0.0001 [-0.0064, +0.0061] none
- `spambase_marf_60nan`: H 0.9058, tabpfn 0.9354, missforest 0.9671, table_tabpfn 0.8986, table_missforest 0.8998, pick 0.8983; pick took missForest on 13% of folds
  - pick − H -0.0075 [-0.0100, -0.0049] better
  - pick − table_tabpfn -0.0003 [-0.0015, +0.0009] equivalent
  - pick − table_missforest -0.0015 [-0.0028, -0.0001] better
- `credit-g_marf_60nan`: H 0.9353, tabpfn 0.9054, missforest 0.9745, table_tabpfn 0.9238, table_missforest 0.9341, pick 0.9238; pick took missForest on 0% of folds
  - pick − H -0.0116 [-0.0183, -0.0048] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0104 [-0.0173, -0.0034] better
## e95
- `electricity_20nan` (trails): H 0.6233, tabpfn 0.6259, missforest 0.6159, table_tabpfn 0.6132, table_missforest 0.5954, pick 0.5970; pick took missForest on 87% of folds
  - pick − H -0.0263 [-0.0316, -0.0210] better
  - pick − table_tabpfn -0.0162 [-0.0233, -0.0091] better
  - pick − table_missforest +0.0016 [-0.0007, +0.0039] equivalent
- `letter_20nan`: H 0.4345, tabpfn 0.4542, missforest 0.4634, table_tabpfn 0.4189, table_missforest 0.4158, pick 0.4158; pick took missForest on 100% of folds
  - pick − H -0.0188 [-0.0191, -0.0185] better
  - pick − table_tabpfn -0.0032 [-0.0036, -0.0027] better
  - pick − table_missforest +0.0000 [+0.0000, +0.0000] equivalent
- `pendigits_20nan`: H 0.3200, tabpfn 0.3090, missforest 0.3335, table_tabpfn 0.2986, table_missforest 0.3000, pick 0.2992; pick took missForest on 40% of folds
  - pick − H -0.0208 [-0.0216, -0.0201] better
  - pick − table_tabpfn +0.0005 [+0.0001, +0.0010] equivalent
  - pick − table_missforest -0.0008 [-0.0014, -0.0002] better
- `spambase_20nan`: H 0.8479, tabpfn 0.9367, missforest 0.8563, table_tabpfn 0.8476, table_missforest 0.8371, pick 0.8389; pick took missForest on 80% of folds
  - pick − H -0.0090 [-0.0120, -0.0061] better
  - pick − table_tabpfn -0.0087 [-0.0119, -0.0055] better
  - pick − table_missforest +0.0018 [-0.0004, +0.0040] equivalent
- `kr-vs-kp_20nan`: H 0.5899, tabpfn 0.5847, missforest 0.6530, table_tabpfn 0.5882, table_missforest 0.5899, pick 0.5882; pick took missForest on 0% of folds
  - pick − H -0.0017 [-0.0075, +0.0040] none
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0017 [-0.0075, +0.0040] none
- `biodeg_20nan`: H 0.5930, tabpfn 0.6150, missforest 0.6197, table_tabpfn 0.5763, table_missforest 0.5785, pick 0.5788; pick took missForest on 73% of folds
  - pick − H -0.0142 [-0.0206, -0.0078] better
  - pick − table_tabpfn +0.0025 [-0.0044, +0.0094] none
  - pick − table_missforest +0.0003 [-0.0008, +0.0014] equivalent
- `credit-g_20nan`: H 0.8858, tabpfn 0.8727, missforest 0.8919, table_tabpfn 0.8687, table_missforest 0.8795, pick 0.8703; pick took missForest on 13% of folds
  - pick − H -0.0154 [-0.0287, -0.0022] better
  - pick − table_tabpfn +0.0017 [-0.0050, +0.0083] equivalent
  - pick − table_missforest -0.0092 [-0.0222, +0.0039] none
- `vehicle_20nan`: H 0.4197, tabpfn 0.4170, missforest 0.4494, table_tabpfn 0.4053, table_missforest 0.4162, pick 0.4120; pick took missForest on 60% of folds
  - pick − H -0.0077 [-0.0169, +0.0015] none
  - pick − table_tabpfn +0.0067 [-0.0012, +0.0146] none
  - pick − table_missforest -0.0042 [-0.0086, +0.0003] none
- `kc2_20nan`: H 0.5369, tabpfn 0.5029, missforest 0.5743, table_tabpfn 0.4782, table_missforest 0.5178, pick 0.4897; pick took missForest on 20% of folds
  - pick − H -0.0471 [-0.1228, +0.0285] none
  - pick − table_tabpfn +0.0116 [-0.0164, +0.0395] none
  - pick − table_missforest -0.0281 [-0.0647, +0.0086] none

PRIMARY pick on 38 variants: worse than H on ['spambase_probit_60nan', 'kc2_marf_60nan']; against `table_tabpfn` better on ['biodeg_probit0p25_60nan', 'biodeg_probit_60nan', 'vehicle_probit_60nan', 'electricity_20nan', 'letter_20nan', 'spambase_20nan'], worse on ['kc2_probit_60nan'] -> mixed
SECONDARY against `table_missforest`: better on 13, worse on ['biodeg_probit0p25_60nan', 'biodeg_probit1p0_60nan', 'kc2_probit_60nan', 'spambase_probit_60nan']; the pick took missForest on 33% of folds overall
(fold-pair t, seed agreement; SESOI 1% of the opponent's mean)
```

- **Primary, mixed:** the pick is worse than H on two variants, `spambase_probit_60nan` (+0.0062 [+0.0009,
  +0.0116]) and `kc2_marf_60nan` (+0.0182 [+0.0005, +0.0359]). Against `table_tabpfn`, the accepted rule, it is
  better on 6: biodeg_probit0p25 and biodeg_probit at 60nan, vehicle_probit_60nan, electricity_20nan, letter_20nan and
  spambase_20nan. It is worse on 1, `kc2_probit_60nan` (+0.0047 [+0.0001, +0.0094]). "Supported" needed no variant
  worse than H.
- **Secondary:** against `table_missforest`, better on 13 and worse on 4. The pick took missForest on 33% of folds.
- **Reading:** the side choice on the validation cells helps where missForest is the better side, which are the 6 where
  it beats the accepted rule. It costs on two of the small, noisy tables, where the in-sample choice may be what fails.
  The within-group cross-fit stated in the design is the upgrade, not built.
- **What it leads to:** no ADR change. Whether a cross-fit pick is worth a study is the user's call.
