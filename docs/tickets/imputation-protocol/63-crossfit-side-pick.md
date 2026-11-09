# A cross-fit side pick (E168, T54; pre-registered 2026-11-09 00:57 GMT-3)

This ticket was written and committed before the rule was run on any table.

- **Why:** E165's pick ([61](61-hybrid-side-pick.md)) chooses TabPFN or missForest on the same validation cells that
  choose each hybrid's weights, so a side that overfits those cells is favoured. Its two losses to H were on small
  tables (spambase_probit_60nan, kc2_marf_60nan), where that bias is largest.
- **The user's go:** "C: escolha cross-fit" (2026-11-09 00:40 GMT-3).
- **The rule (`missforest_hybrid.both_sides`, `crossfit`):** per fold k and side s, each of the 5 validation groups is
  filled with the `table` weights chosen on the other 4 groups; the side whose filled validation cells score lower
  (one `impute_score` over all groups) is taken, a tie keeping TabPFN. The test fills are that side's `table` hybrid,
  weights chosen on all 5 groups, as in the pick. Only the side choice differs from E165's.

## Readings (fixed now)

- **Development, descriptive:** on E164/E165's 38 variants (seeds 4545/5656/6767, CPU on medusa1, minutes):
  `uv run --python 3.11 python -B scripts/experiments/missforest_generality.py --tally e168`. It prints E165's rule
  applied to `crossfit`, and `crossfit` − `pick`. It is read after this ticket is committed and before E167's
  cells, and changes nothing below.
- **Primary, on E167's 12 fresh variants** ([62](62-hybrid-fresh-seeds.md)), E165's rule on `crossfit`:
  **supported** if worse than H on none, worse than `table_tabpfn` on at most 1 and better on at least 2; **refuted** if
  worse than `table_tabpfn` on 3 or more, or better on none; **mixed** otherwise. Secondary: `crossfit` − `pick`.
  Read in E167's printout.
- Fold-pair verdict, 1% SESOI. **What it leads to:** whether a side-picking rule is worth the user's ADR call.

## Development reading (read once 2026-11-09 01:02 GMT-3, descriptive)

Read on medusa1 at commit e5b82da, before any E167 cell. The printout, unedited:

```
## e90
- `electricity_40nan`: H 0.7544, tabpfn 0.7427, missforest 0.7899, table_tabpfn 0.7404, table_missforest 0.7488, pick 0.7404, crossfit 0.7404; cross-fit pick took missForest on 33% of folds
  - crossfit − H -0.0140 [-0.0350, +0.0070] none
  - crossfit − table_tabpfn +0.0000 [-0.0011, +0.0011] equivalent
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `electricity_80nan`: H 0.8981, tabpfn 0.8907, missforest 1.0152, table_tabpfn 0.8897, table_missforest 0.8985, pick 0.8897, crossfit 0.8897; cross-fit pick took missForest on 0% of folds
  - crossfit − H -0.0084 [-0.0101, -0.0066] better
  - crossfit − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `vehicle_80nan` (trails): H 0.8119, tabpfn 0.7536, missforest 0.7683, table_tabpfn 0.7539, table_missforest 0.7476, pick 0.7514, crossfit 0.7507; cross-fit pick took missForest on 47% of folds
  - crossfit − H -0.0613 [-0.0662, -0.0564] better
  - crossfit − table_tabpfn -0.0032 [-0.0072, +0.0008] equivalent
  - crossfit − pick -0.0007 [-0.0023, +0.0008] equivalent
- `biodeg_80nan` (trails): H 0.9855, tabpfn 0.9188, missforest 0.9255, table_tabpfn 0.9162, table_missforest 0.9137, pick 0.9152, crossfit 0.9145; cross-fit pick took missForest on 40% of folds
  - crossfit − H -0.0710 [-0.0779, -0.0640] better
  - crossfit − table_tabpfn -0.0017 [-0.0042, +0.0009] equivalent
  - crossfit − pick -0.0007 [-0.0018, +0.0004] equivalent
- `kr-vs-kp_40nan`: H 0.7256, tabpfn 0.7118, missforest 0.7996, table_tabpfn 0.7184, table_missforest 0.7256, pick 0.7184, crossfit 0.7206; cross-fit pick took missForest on 20% of folds
  - crossfit − H -0.0050 [-0.0107, +0.0008] none
  - crossfit − table_tabpfn +0.0022 [-0.0011, +0.0055] equivalent
  - crossfit − pick +0.0022 [-0.0011, +0.0055] equivalent
- `kr-vs-kp_60nan`: H 0.8715, tabpfn 0.8301, missforest 0.9381, table_tabpfn 0.8301, table_missforest 0.8715, pick 0.8301, crossfit 0.8301; cross-fit pick took missForest on 0% of folds
  - crossfit − H -0.0414 [-0.0474, -0.0355] better
  - crossfit − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `kr-vs-kp_80nan`: H 1.0096, tabpfn 0.9406, missforest 1.0155, table_tabpfn 0.9445, table_missforest 1.0052, pick 0.9445, crossfit 0.9561; cross-fit pick took missForest on 27% of folds
  - crossfit − H -0.0535 [-0.0751, -0.0319] better
  - crossfit − table_tabpfn +0.0116 [-0.0022, +0.0255] none
  - crossfit − pick +0.0116 [-0.0022, +0.0255] none
- `credit-g_40nan` (trails): H 0.9378, tabpfn 0.8999, missforest 0.9272, table_tabpfn 0.8997, table_missforest 0.9246, pick 0.9020, crossfit 0.9064; cross-fit pick took missForest on 33% of folds
  - crossfit − H -0.0313 [-0.0464, -0.0163] better
  - crossfit − table_tabpfn +0.0067 [+0.0006, +0.0129] none
  - crossfit − pick +0.0044 [-0.0008, +0.0096] none
- `credit-g_60nan`: H 1.0018, tabpfn 0.9510, missforest 1.0142, table_tabpfn 0.9564, table_missforest 0.9945, pick 0.9564, crossfit 0.9680; cross-fit pick took missForest on 33% of folds
  - crossfit − H -0.0338 [-0.0469, -0.0207] better
  - crossfit − table_tabpfn +0.0116 [-0.0036, +0.0267] none
  - crossfit − pick +0.0116 [-0.0036, +0.0267] none
- `credit-g_80nan`: H 1.0382, tabpfn 1.0028, missforest 1.0922, table_tabpfn 1.0077, table_missforest 1.0326, pick 1.0069, crossfit 1.0123; cross-fit pick took missForest on 33% of folds
  - crossfit − H -0.0259 [-0.0449, -0.0069] better
  - crossfit − table_tabpfn +0.0046 [-0.0029, +0.0121] none
  - crossfit − pick +0.0054 [-0.0020, +0.0127] none
## e91
- `biodeg_40nan`: H 0.6832, tabpfn 0.7149, missforest 0.6941, table_tabpfn 0.6616, table_missforest 0.6594, pick 0.6581, crossfit 0.6601; cross-fit pick took missForest on 87% of folds
  - crossfit − H -0.0231 [-0.0333, -0.0128] better
  - crossfit − table_tabpfn -0.0015 [-0.0092, +0.0063] none
  - crossfit − pick +0.0020 [-0.0023, +0.0063] equivalent
- `kc2_40nan`: H 0.5057, tabpfn 0.5067, missforest 0.5825, table_tabpfn 0.4883, table_missforest 0.5038, pick 0.4958, crossfit 0.4940; cross-fit pick took missForest on 33% of folds
  - crossfit − H -0.0117 [-0.0500, +0.0267] none
  - crossfit − table_tabpfn +0.0058 [-0.0064, +0.0179] none
  - crossfit − pick -0.0017 [-0.0055, +0.0020] none
- `kc2_80nan`: H 0.6802, tabpfn 0.7396, missforest 0.7511, table_tabpfn 0.6811, table_missforest 0.6806, pick 0.6832, crossfit 0.6817; cross-fit pick took missForest on 33% of folds
  - crossfit − H +0.0014 [-0.0300, +0.0328] none
  - crossfit − table_tabpfn +0.0005 [-0.0065, +0.0075] none
  - crossfit − pick -0.0015 [-0.0045, +0.0014] equivalent
- `pendigits_40nan`: H 0.4274, tabpfn 0.4363, missforest 0.4969, table_tabpfn 0.4151, table_missforest 0.4155, pick 0.4155, crossfit 0.4155; cross-fit pick took missForest on 40% of folds
  - crossfit − H -0.0119 [-0.0129, -0.0110] better
  - crossfit − table_tabpfn +0.0003 [-0.0003, +0.0009] equivalent
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `pendigits_60nan`: H 0.5902, tabpfn 0.6038, missforest 0.7023, table_tabpfn 0.5763, table_missforest 0.5791, pick 0.5763, crossfit 0.5769; cross-fit pick took missForest on 13% of folds
  - crossfit − H -0.0134 [-0.0151, -0.0116] better
  - crossfit − table_tabpfn +0.0006 [-0.0003, +0.0014] equivalent
  - crossfit − pick +0.0006 [-0.0003, +0.0014] equivalent
- `pendigits_80nan`: H 0.8360, tabpfn 0.8144, missforest 0.9420, table_tabpfn 0.8108, table_missforest 0.8237, pick 0.8108, crossfit 0.8108; cross-fit pick took missForest on 0% of folds
  - crossfit − H -0.0253 [-0.0295, -0.0210] better
  - crossfit − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `vehicle_40nan`: H 0.4964, tabpfn 0.5008, missforest 0.5229, table_tabpfn 0.4829, table_missforest 0.4900, pick 0.4857, crossfit 0.4861; cross-fit pick took missForest on 53% of folds
  - crossfit − H -0.0103 [-0.0202, -0.0003] better
  - crossfit − table_tabpfn +0.0032 [-0.0029, +0.0093] none
  - crossfit − pick +0.0004 [-0.0004, +0.0012] equivalent
## e93
- `biodeg_probit0p25_60nan` (trails): H 0.9114, tabpfn 0.9150, missforest 0.8842, table_tabpfn 0.9048, table_missforest 0.8844, pick 0.8952, crossfit 0.8936; cross-fit pick took missForest on 53% of folds
  - crossfit − H -0.0177 [-0.0256, -0.0099] better
  - crossfit − table_tabpfn -0.0111 [-0.0180, -0.0043] better
  - crossfit − pick -0.0015 [-0.0048, +0.0018] equivalent
- `biodeg_probit_60nan` (trails): H 0.9123, tabpfn 0.9246, missforest 0.8954, table_tabpfn 0.9140, table_missforest 0.8943, pick 0.8986, crossfit 0.8986; cross-fit pick took missForest on 87% of folds
  - crossfit − H -0.0137 [-0.0196, -0.0078] better
  - crossfit − table_tabpfn -0.0154 [-0.0223, -0.0084] better
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `biodeg_probit1p0_60nan`: H 0.9306, tabpfn 0.9379, missforest 0.9339, table_tabpfn 0.9300, table_missforest 0.9257, pick 0.9290, crossfit 0.9275; cross-fit pick took missForest on 47% of folds
  - crossfit − H -0.0030 [-0.0052, -0.0009] better
  - crossfit − table_tabpfn -0.0025 [-0.0048, -0.0002] better
  - crossfit − pick -0.0015 [-0.0037, +0.0007] equivalent
- `biodeg_probit2p0_60nan`: H 0.9321, tabpfn 0.9545, missforest 0.9527, table_tabpfn 0.9380, table_missforest 0.9337, pick 0.9380, crossfit 0.9373; cross-fit pick took missForest on 20% of folds
  - crossfit − H +0.0052 [-0.0009, +0.0114] none
  - crossfit − table_tabpfn -0.0007 [-0.0017, +0.0003] equivalent
  - crossfit − pick -0.0007 [-0.0017, +0.0003] equivalent
- `vehicle_probit_60nan` (trails): H 0.6308, tabpfn 0.6225, missforest 0.6212, table_tabpfn 0.6139, table_missforest 0.6103, pick 0.6106, crossfit 0.6114; cross-fit pick took missForest on 73% of folds
  - crossfit − H -0.0195 [-0.0220, -0.0170] better
  - crossfit − table_tabpfn -0.0026 [-0.0056, +0.0005] equivalent
  - crossfit − pick +0.0007 [-0.0007, +0.0022] equivalent
- `kc2_probit_60nan`: H 0.7253, tabpfn 0.7757, missforest 0.7538, table_tabpfn 0.7317, table_missforest 0.7316, pick 0.7364, crossfit 0.7355; cross-fit pick took missForest on 47% of folds
  - crossfit − H +0.0102 [-0.0194, +0.0397] none
  - crossfit − table_tabpfn +0.0038 [-0.0015, +0.0091] none
  - crossfit − pick -0.0009 [-0.0028, +0.0010] equivalent
- `spambase_probit_60nan`: H 0.9597, tabpfn 0.9955, missforest 0.9609, table_tabpfn 0.9687, table_missforest 0.9531, pick 0.9659, crossfit 0.9659; cross-fit pick took missForest on 27% of folds
  - crossfit − H +0.0062 [+0.0009, +0.0116] worse
  - crossfit − table_tabpfn -0.0028 [-0.0058, +0.0003] equivalent
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `biodeg_marf_60nan`: H 0.6864, tabpfn 0.7056, missforest 0.7206, table_tabpfn 0.6643, table_missforest 0.6690, pick 0.6643, crossfit 0.6670; cross-fit pick took missForest on 47% of folds
  - crossfit − H -0.0194 [-0.0290, -0.0098] better
  - crossfit − table_tabpfn +0.0027 [-0.0012, +0.0066] none
  - crossfit − pick +0.0027 [-0.0007, +0.0061] equivalent
- `kc2_marf_60nan`: H 0.5987, tabpfn 0.6701, missforest 0.7134, table_tabpfn 0.6078, table_missforest 0.6200, pick 0.6169, crossfit 0.6160; cross-fit pick took missForest on 60% of folds
  - crossfit − H +0.0173 [-0.0001, +0.0347] none
  - crossfit − table_tabpfn +0.0082 [-0.0030, +0.0194] none
  - crossfit − pick -0.0009 [-0.0043, +0.0026] equivalent
- `vehicle_marf_60nan`: H 0.4510, tabpfn 0.4445, missforest 0.4947, table_tabpfn 0.4356, table_missforest 0.4348, pick 0.4347, crossfit 0.4348; cross-fit pick took missForest on 33% of folds
  - crossfit − H -0.0162 [-0.0229, -0.0095] better
  - crossfit − table_tabpfn -0.0008 [-0.0043, +0.0027] equivalent
  - crossfit − pick +0.0001 [-0.0001, +0.0002] equivalent
- `spambase_marf_60nan`: H 0.9058, tabpfn 0.9354, missforest 0.9671, table_tabpfn 0.8986, table_missforest 0.8998, pick 0.8983, crossfit 0.8982; cross-fit pick took missForest on 20% of folds
  - crossfit − H -0.0075 [-0.0101, -0.0050] better
  - crossfit − table_tabpfn -0.0004 [-0.0016, +0.0008] equivalent
  - crossfit − pick -0.0001 [-0.0002, +0.0001] equivalent
- `credit-g_marf_60nan`: H 0.9353, tabpfn 0.9054, missforest 0.9745, table_tabpfn 0.9238, table_missforest 0.9341, pick 0.9238, crossfit 0.9294; cross-fit pick took missForest on 47% of folds
  - crossfit − H -0.0060 [-0.0108, -0.0012] better
  - crossfit − table_tabpfn +0.0056 [-0.0010, +0.0122] none
  - crossfit − pick +0.0056 [-0.0010, +0.0122] none
## e95
- `electricity_20nan` (trails): H 0.6233, tabpfn 0.6259, missforest 0.6159, table_tabpfn 0.6132, table_missforest 0.5954, pick 0.5970, crossfit 0.5970; cross-fit pick took missForest on 87% of folds
  - crossfit − H -0.0263 [-0.0316, -0.0210] better
  - crossfit − table_tabpfn -0.0162 [-0.0233, -0.0091] better
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `letter_20nan`: H 0.4345, tabpfn 0.4542, missforest 0.4634, table_tabpfn 0.4189, table_missforest 0.4158, pick 0.4158, crossfit 0.4158; cross-fit pick took missForest on 100% of folds
  - crossfit − H -0.0188 [-0.0191, -0.0185] better
  - crossfit − table_tabpfn -0.0032 [-0.0036, -0.0027] better
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `pendigits_20nan`: H 0.3200, tabpfn 0.3090, missforest 0.3335, table_tabpfn 0.2986, table_missforest 0.3000, pick 0.2992, crossfit 0.2992; cross-fit pick took missForest on 40% of folds
  - crossfit − H -0.0208 [-0.0216, -0.0201] better
  - crossfit − table_tabpfn +0.0005 [+0.0001, +0.0010] equivalent
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `spambase_20nan`: H 0.8479, tabpfn 0.9367, missforest 0.8563, table_tabpfn 0.8476, table_missforest 0.8371, pick 0.8389, crossfit 0.8383; cross-fit pick took missForest on 87% of folds
  - crossfit − H -0.0096 [-0.0123, -0.0070] better
  - crossfit − table_tabpfn -0.0093 [-0.0121, -0.0064] better
  - crossfit − pick -0.0006 [-0.0018, +0.0006] equivalent
- `kr-vs-kp_20nan`: H 0.5899, tabpfn 0.5847, missforest 0.6530, table_tabpfn 0.5882, table_missforest 0.5899, pick 0.5882, crossfit 0.5865; cross-fit pick took missForest on 47% of folds
  - crossfit − H -0.0034 [-0.0076, +0.0009] none
  - crossfit − table_tabpfn -0.0016 [-0.0051, +0.0019] equivalent
  - crossfit − pick -0.0016 [-0.0051, +0.0019] equivalent
- `biodeg_20nan`: H 0.5930, tabpfn 0.6150, missforest 0.6197, table_tabpfn 0.5763, table_missforest 0.5785, pick 0.5788, crossfit 0.5793; cross-fit pick took missForest on 60% of folds
  - crossfit − H -0.0137 [-0.0204, -0.0071] better
  - crossfit − table_tabpfn +0.0030 [-0.0036, +0.0096] none
  - crossfit − pick +0.0005 [-0.0013, +0.0023] equivalent
- `credit-g_20nan`: H 0.8858, tabpfn 0.8727, missforest 0.8919, table_tabpfn 0.8687, table_missforest 0.8795, pick 0.8703, crossfit 0.8705; cross-fit pick took missForest on 20% of folds
  - crossfit − H -0.0153 [-0.0285, -0.0021] better
  - crossfit − table_tabpfn +0.0018 [-0.0049, +0.0085] equivalent
  - crossfit − pick +0.0001 [-0.0002, +0.0004] equivalent
- `vehicle_20nan`: H 0.4197, tabpfn 0.4170, missforest 0.4494, table_tabpfn 0.4053, table_missforest 0.4162, pick 0.4120, crossfit 0.4123; cross-fit pick took missForest on 60% of folds
  - crossfit − H -0.0074 [-0.0168, +0.0019] none
  - crossfit − table_tabpfn +0.0069 [-0.0008, +0.0147] none
  - crossfit − pick +0.0003 [-0.0010, +0.0015] equivalent
- `kc2_20nan`: H 0.5369, tabpfn 0.5029, missforest 0.5743, table_tabpfn 0.4782, table_missforest 0.5178, pick 0.4897, crossfit 0.4897; cross-fit pick took missForest on 20% of folds
  - crossfit − H -0.0471 [-0.1228, +0.0285] none
  - crossfit − table_tabpfn +0.0116 [-0.0164, +0.0395] none
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent

DEVELOPMENT crossfit on 38 variants: worse than H on ['spambase_probit_60nan']; against `table_tabpfn` better on ['biodeg_probit0p25_60nan', 'biodeg_probit_60nan', 'biodeg_probit1p0_60nan', 'electricity_20nan', 'letter_20nan', 'spambase_20nan'], worse on none -> mixed
`crossfit` − `pick`: better on none, worse on none; the cross-fit pick took missForest on 42% of folds overall
(fold-pair t, seed agreement; SESOI 1% of the opponent's mean)
```

- Under E165's rule the cross-fit pick reads mixed on the development tables: worse than H on one (`spambase_probit_60nan`,
  where the pick was worse too), against `table_tabpfn` better on 6 and worse on none. E165's pick lost to H on
  `kc2_marf_60nan` and to `table_tabpfn` on `kc2_probit_60nan`; the cross-fit pick loses on neither.
- `crossfit` − `pick` has no call on any variant: the difference is small, and it takes missForest more often (42% of
  folds against 33%).
- Descriptive only: the primary is E167's fresh-seed reading, which this changes in no way.
