# The missForest-side hybrid on every variant with T39's cell tables (E164, T54; pre-registered 2026-11-08 21:14 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** E162 ([59](59-missforest-hybrid.md)) found ADR 0031 decision 6's `table` rule, with missForest as the
  other side, better than H and than missForest on `vehicle_80nan` and `biodeg_80nan`. Those two were chosen because
  H trails missForest there, so extending ADR 0031 to missForest needs the rule read where nothing was chosen.
- **The user's go:** "faça 1 2 e 3" (2026-11-08), on "3. Estender o E162 a todas as variantes a 80% em que TRIDENT
  perde para o missForest".
- **The literal scope was empty (disclosed):** at 80nan, the runs' own H trails their logged missForest only on
  vehicle and biodeg, which E162 already read. It leads on credit-g, electricity, kr-vs-kp, kc2 and pendigits
  (E90/E91), and on letter and spambase (E44's and E43's runs: H 0.89 against 0.97, and 0.98–0.99 against 1.03).
  This study is the extension an ADR would need instead: every variant with T39's cell tables, none chosen by score.
- **Disclosure:** the runs' H and missForest scores on these variants are on the record. They name the secondary's
  scope, `TRAILS` in the runner: the 7 variants where H trails missForest by more than 1% of its score. These are
  `biodeg_80nan`, `vehicle_80nan`, `credit-g_40nan`, `biodeg_probit0p25_60nan`, `biodeg_probit_60nan`,
  `vehicle_probit_60nan` and `electricity_20nan`. The two 80nan ones are E162's on E90's seeds.

## Design

- **Cells:** E90's 10, E91's 7, E93's 12 and E95's 9 variants (`hybrid_confirmation.STUDIES`), seeds
  4545/5656/6767, five folds: 38 variants, 114 pairs. H's and TabPFN's fills are the studies' own cell tables.
  missForest (ADR 0016) is refit on each training fold. It fills the test induced cells, and the hidden observed
  validation cells group by group, as TabPFN did. Its test score is asserted equal to the run's logged missForest
  score, fold by fold.
- **Machine:** medusa1, CPU only, nothing in the store. gorgona8's CPU is saturated by the user's other project.
  E90/E91/E93's tables were copied from gorgona8, and all 1,740 files' SHA-256 matched. E95's are medusa1's own. The
  assert also checks that medusa1's missForest reproduces gorgona8's.
- **Runner:** `scripts/experiments/missforest_generality.py` (`missforest_hybrid.add_missforest`, then
  `missforest_hybrid.both_sides`). State `/scratch/diogoneiss/trident-2026-11-08-missforest-generality`.
- **Cost:** a `vehicle_80nan` pair took 187 s on gorgona8 (E162's refactor check). electricity's nine pairs,
  45,312 rows, are unmeasured and dominate. The estimate is several hours over 10 lanes.

## Measures and rule (fixed now)

- **Verdict:** `common.verdict` on 15 (seed, fold) differences of `impute_score`, SESOI 1% of the opponent's mean.
- **Primary,** `table_missforest` on the 38 variants:
  - **supported** if it is worse than H on none and worse than missForest on at most 2;
  - **refuted** if it is worse than H on 2 or more, or worse than missForest on 5 or more;
  - **mixed** otherwise.
- **Secondary:** how many variants it is better than H on, overall and on `TRAILS`.
- **Read:** once, with all 114 pairs in, `missforest_generality.py --tally e164` on medusa1.
- **What it leads to:** extending ADR 0031 decision 6 to missForest is the user's call. E165 reads a rule that picks
  the side on the same cells.

## Outcome (2026-11-08 22:05 GMT-3)

**Mixed.** All 114 pairs were in at 21:47 GMT-3 on 10 lanes started at 21:14, with no failure. The missForest
assert held on every pair, all 570 folds, so medusa1's refit reproduces the runs' missForest. Read once at
22:00 GMT-3 on medusa1. Unedited printout:

```
## e90
- `electricity_40nan`: H 0.7544, tabpfn 0.7427, missforest 0.7899, table_tabpfn 0.7404, table_missforest 0.7488, pick 0.7404; pick took missForest on 27% of folds
  - table_missforest − H -0.0055 [-0.0165, +0.0054] none
  - table_missforest − missforest -0.0410 [-0.0657, -0.0163] better
- `electricity_80nan`: H 0.8981, tabpfn 0.8907, missforest 1.0152, table_tabpfn 0.8897, table_missforest 0.8985, pick 0.8897; pick took missForest on 0% of folds
  - table_missforest − H +0.0005 [-0.0001, +0.0010] equivalent
  - table_missforest − missforest -0.1166 [-0.1301, -0.1031] better
- `vehicle_80nan` (trails): H 0.8119, tabpfn 0.7536, missforest 0.7683, table_tabpfn 0.7539, table_missforest 0.7476, pick 0.7514; pick took missForest on 40% of folds
  - table_missforest − H -0.0643 [-0.0720, -0.0566] better
  - table_missforest − missforest -0.0207 [-0.0262, -0.0152] better
- `biodeg_80nan` (trails): H 0.9855, tabpfn 0.9188, missforest 0.9255, table_tabpfn 0.9162, table_missforest 0.9137, pick 0.9152; pick took missForest on 40% of folds
  - table_missforest − H -0.0718 [-0.0779, -0.0656] better
  - table_missforest − missforest -0.0118 [-0.0162, -0.0073] better
- `kr-vs-kp_40nan`: H 0.7256, tabpfn 0.7118, missforest 0.7996, table_tabpfn 0.7184, table_missforest 0.7256, pick 0.7184; pick took missForest on 0% of folds
  - table_missforest − H +0.0000 [+0.0000, +0.0000] equivalent
  - table_missforest − missforest -0.0740 [-0.0844, -0.0637] better
- `kr-vs-kp_60nan`: H 0.8715, tabpfn 0.8301, missforest 0.9381, table_tabpfn 0.8301, table_missforest 0.8715, pick 0.8301; pick took missForest on 0% of folds
  - table_missforest − H +0.0000 [+0.0000, +0.0000] equivalent
  - table_missforest − missforest -0.0666 [-0.0772, -0.0560] better
- `kr-vs-kp_80nan`: H 1.0096, tabpfn 0.9406, missforest 1.0155, table_tabpfn 0.9445, table_missforest 1.0052, pick 0.9445; pick took missForest on 0% of folds
  - table_missforest − H -0.0044 [-0.0108, +0.0020] none
  - table_missforest − missforest -0.0103 [-0.0248, +0.0042] none
- `credit-g_40nan` (trails): H 0.9378, tabpfn 0.8999, missforest 0.9272, table_tabpfn 0.8997, table_missforest 0.9246, pick 0.9020; pick took missForest on 20% of folds
  - table_missforest − H -0.0131 [-0.0229, -0.0034] better
  - table_missforest − missforest -0.0026 [-0.0116, +0.0065] none
- `credit-g_60nan`: H 1.0018, tabpfn 0.9510, missforest 1.0142, table_tabpfn 0.9564, table_missforest 0.9945, pick 0.9564; pick took missForest on 0% of folds
  - table_missforest − H -0.0073 [-0.0128, -0.0017] better
  - table_missforest − missforest -0.0197 [-0.0281, -0.0113] better
- `credit-g_80nan`: H 1.0382, tabpfn 1.0028, missforest 1.0922, table_tabpfn 1.0077, table_missforest 1.0326, pick 1.0069; pick took missForest on 13% of folds
  - table_missforest − H -0.0056 [-0.0105, -0.0006] better
  - table_missforest − missforest -0.0596 [-0.0771, -0.0421] better
## e91
- `biodeg_40nan`: H 0.6832, tabpfn 0.7149, missforest 0.6941, table_tabpfn 0.6616, table_missforest 0.6594, pick 0.6581; pick took missForest on 80% of folds
  - table_missforest − H -0.0238 [-0.0350, -0.0125] better
  - table_missforest − missforest -0.0347 [-0.0479, -0.0215] better
- `kc2_40nan`: H 0.5057, tabpfn 0.5067, missforest 0.5825, table_tabpfn 0.4883, table_missforest 0.5038, pick 0.4958; pick took missForest on 20% of folds
  - table_missforest − H -0.0019 [-0.0354, +0.0315] none
  - table_missforest − missforest -0.0787 [-0.1231, -0.0344] better
- `kc2_80nan`: H 0.6802, tabpfn 0.7396, missforest 0.7511, table_tabpfn 0.6811, table_missforest 0.6806, pick 0.6832; pick took missForest on 20% of folds
  - table_missforest − H +0.0004 [-0.0258, +0.0266] none
  - table_missforest − missforest -0.0705 [-0.0970, -0.0440] better
- `pendigits_40nan`: H 0.4274, tabpfn 0.4363, missforest 0.4969, table_tabpfn 0.4151, table_missforest 0.4155, pick 0.4155; pick took missForest on 40% of folds
  - table_missforest − H -0.0119 [-0.0133, -0.0105] better
  - table_missforest − missforest -0.0814 [-0.0876, -0.0753] better
- `pendigits_60nan`: H 0.5902, tabpfn 0.6038, missforest 0.7023, table_tabpfn 0.5763, table_missforest 0.5791, pick 0.5763; pick took missForest on 0% of folds
  - table_missforest − H -0.0111 [-0.0131, -0.0092] better
  - table_missforest − missforest -0.1232 [-0.1277, -0.1187] better
- `pendigits_80nan`: H 0.8360, tabpfn 0.8144, missforest 0.9420, table_tabpfn 0.8108, table_missforest 0.8237, pick 0.8108; pick took missForest on 0% of folds
  - table_missforest − H -0.0123 [-0.0142, -0.0105] better
  - table_missforest − missforest -0.1183 [-0.1247, -0.1118] better
- `vehicle_40nan`: H 0.4964, tabpfn 0.5008, missforest 0.5229, table_tabpfn 0.4829, table_missforest 0.4900, pick 0.4857; pick took missForest on 60% of folds
  - table_missforest − H -0.0064 [-0.0160, +0.0032] none
  - table_missforest − missforest -0.0329 [-0.0508, -0.0150] better
## e93
- `biodeg_probit0p25_60nan` (trails): H 0.9114, tabpfn 0.9150, missforest 0.8842, table_tabpfn 0.9048, table_missforest 0.8844, pick 0.8952; pick took missForest on 47% of folds
  - table_missforest − H -0.0270 [-0.0335, -0.0205] better
  - table_missforest − missforest +0.0002 [-0.0040, +0.0045] equivalent
- `biodeg_probit_60nan` (trails): H 0.9123, tabpfn 0.9246, missforest 0.8954, table_tabpfn 0.9140, table_missforest 0.8943, pick 0.8986; pick took missForest on 87% of folds
  - table_missforest − H -0.0180 [-0.0226, -0.0133] better
  - table_missforest − missforest -0.0010 [-0.0043, +0.0023] equivalent
- `biodeg_probit1p0_60nan`: H 0.9306, tabpfn 0.9379, missforest 0.9339, table_tabpfn 0.9300, table_missforest 0.9257, pick 0.9290; pick took missForest on 20% of folds
  - table_missforest − H -0.0049 [-0.0077, -0.0021] better
  - table_missforest − missforest -0.0082 [-0.0113, -0.0051] better
- `biodeg_probit2p0_60nan`: H 0.9321, tabpfn 0.9545, missforest 0.9527, table_tabpfn 0.9380, table_missforest 0.9337, pick 0.9380; pick took missForest on 0% of folds
  - table_missforest − H +0.0016 [-0.0014, +0.0046] equivalent
  - table_missforest − missforest -0.0190 [-0.0265, -0.0115] better
- `vehicle_probit_60nan` (trails): H 0.6308, tabpfn 0.6225, missforest 0.6212, table_tabpfn 0.6139, table_missforest 0.6103, pick 0.6106; pick took missForest on 87% of folds
  - table_missforest − H -0.0205 [-0.0237, -0.0174] better
  - table_missforest − missforest -0.0109 [-0.0161, -0.0057] better
- `kc2_probit_60nan`: H 0.7253, tabpfn 0.7757, missforest 0.7538, table_tabpfn 0.7317, table_missforest 0.7316, pick 0.7364; pick took missForest on 33% of folds
  - table_missforest − H +0.0063 [-0.0219, +0.0345] none
  - table_missforest − missforest -0.0222 [-0.0383, -0.0061] better
- `spambase_probit_60nan`: H 0.9597, tabpfn 0.9955, missforest 0.9609, table_tabpfn 0.9687, table_missforest 0.9531, pick 0.9659; pick took missForest on 27% of folds
  - table_missforest − H -0.0066 [-0.0079, -0.0054] better
  - table_missforest − missforest -0.0078 [-0.0110, -0.0047] better
- `biodeg_marf_60nan`: H 0.6864, tabpfn 0.7056, missforest 0.7206, table_tabpfn 0.6643, table_missforest 0.6690, pick 0.6643; pick took missForest on 20% of folds
  - table_missforest − H -0.0174 [-0.0258, -0.0090] better
  - table_missforest − missforest -0.0516 [-0.0694, -0.0338] better
- `kc2_marf_60nan`: H 0.5987, tabpfn 0.6701, missforest 0.7134, table_tabpfn 0.6078, table_missforest 0.6200, pick 0.6169; pick took missForest on 47% of folds
  - table_missforest − H +0.0213 [+0.0017, +0.0409] worse
  - table_missforest − missforest -0.0934 [-0.1456, -0.0412] better
- `vehicle_marf_60nan`: H 0.4510, tabpfn 0.4445, missforest 0.4947, table_tabpfn 0.4356, table_missforest 0.4348, pick 0.4347; pick took missForest on 40% of folds
  - table_missforest − H -0.0162 [-0.0227, -0.0096] better
  - table_missforest − missforest -0.0599 [-0.0828, -0.0369] better
- `spambase_marf_60nan`: H 0.9058, tabpfn 0.9354, missforest 0.9671, table_tabpfn 0.8986, table_missforest 0.8998, pick 0.8983; pick took missForest on 13% of folds
  - table_missforest − H -0.0060 [-0.0084, -0.0036] better
  - table_missforest − missforest -0.0673 [-0.0759, -0.0587] better
- `credit-g_marf_60nan`: H 0.9353, tabpfn 0.9054, missforest 0.9745, table_tabpfn 0.9238, table_missforest 0.9341, pick 0.9238; pick took missForest on 0% of folds
  - table_missforest − H -0.0012 [-0.0021, -0.0003] better
  - table_missforest − missforest -0.0404 [-0.0587, -0.0220] better
## e95
- `electricity_20nan` (trails): H 0.6233, tabpfn 0.6259, missforest 0.6159, table_tabpfn 0.6132, table_missforest 0.5954, pick 0.5970; pick took missForest on 87% of folds
  - table_missforest − H -0.0279 [-0.0327, -0.0230] better
  - table_missforest − missforest -0.0206 [-0.0304, -0.0107] better
- `letter_20nan`: H 0.4345, tabpfn 0.4542, missforest 0.4634, table_tabpfn 0.4189, table_missforest 0.4158, pick 0.4158; pick took missForest on 100% of folds
  - table_missforest − H -0.0188 [-0.0191, -0.0185] better
  - table_missforest − missforest -0.0477 [-0.0510, -0.0443] better
- `pendigits_20nan`: H 0.3200, tabpfn 0.3090, missforest 0.3335, table_tabpfn 0.2986, table_missforest 0.3000, pick 0.2992; pick took missForest on 40% of folds
  - table_missforest − H -0.0200 [-0.0207, -0.0193] better
  - table_missforest − missforest -0.0335 [-0.0389, -0.0280] better
- `spambase_20nan`: H 0.8479, tabpfn 0.9367, missforest 0.8563, table_tabpfn 0.8476, table_missforest 0.8371, pick 0.8389; pick took missForest on 80% of folds
  - table_missforest − H -0.0109 [-0.0129, -0.0089] better
  - table_missforest − missforest -0.0192 [-0.0258, -0.0127] better
- `kr-vs-kp_20nan`: H 0.5899, tabpfn 0.5847, missforest 0.6530, table_tabpfn 0.5882, table_missforest 0.5899, pick 0.5882; pick took missForest on 0% of folds
  - table_missforest − H +0.0000 [+0.0000, +0.0000] equivalent
  - table_missforest − missforest -0.0631 [-0.0796, -0.0465] better
- `biodeg_20nan`: H 0.5930, tabpfn 0.6150, missforest 0.6197, table_tabpfn 0.5763, table_missforest 0.5785, pick 0.5788; pick took missForest on 73% of folds
  - table_missforest − H -0.0145 [-0.0211, -0.0079] better
  - table_missforest − missforest -0.0412 [-0.0595, -0.0229] better
- `credit-g_20nan`: H 0.8858, tabpfn 0.8727, missforest 0.8919, table_tabpfn 0.8687, table_missforest 0.8795, pick 0.8703; pick took missForest on 13% of folds
  - table_missforest − H -0.0063 [-0.0192, +0.0066] none
  - table_missforest − missforest -0.0124 [-0.0252, +0.0004] none
- `vehicle_20nan`: H 0.4197, tabpfn 0.4170, missforest 0.4494, table_tabpfn 0.4053, table_missforest 0.4162, pick 0.4120; pick took missForest on 60% of folds
  - table_missforest − H -0.0035 [-0.0114, +0.0043] none
  - table_missforest − missforest -0.0332 [-0.0558, -0.0106] better
- `kc2_20nan`: H 0.5369, tabpfn 0.5029, missforest 0.5743, table_tabpfn 0.4782, table_missforest 0.5178, pick 0.4897; pick took missForest on 20% of folds
  - table_missforest − H -0.0191 [-0.0779, +0.0398] none
  - table_missforest − missforest -0.0565 [-0.1003, -0.0127] better

PRIMARY `table_missforest` on 38 variants: worse than H on ['kc2_marf_60nan'], worse than missForest on none -> mixed
SECONDARY better than H on 23 of 38; on the 7 where H trails missForest: better than H on ['vehicle_80nan', 'biodeg_80nan', 'credit-g_40nan', 'biodeg_probit0p25_60nan', 'biodeg_probit_60nan', 'vehicle_probit_60nan', 'electricity_20nan']
(fold-pair t, seed agreement; SESOI 1% of the opponent's mean)
```

- **Primary, mixed:** `table_missforest` is worse than H on one variant, `kc2_marf_60nan` (+0.0213 [+0.0017,
  +0.0409]), and worse than missForest on none. "Supported" needed no variant worse than H.
- **Secondary:** better than H on 23 of 38, and on all 7 of `TRAILS` (−0.013 to −0.072). Never worse than
  missForest: better on 33, equivalent on 2, no different on 3.
- **Beside it, descriptive:** on kc2_marf_60nan, H's 0.599 is far ahead of missForest's 0.713, so a weight toward
  missForest on 522 rows costs. On the tables where H leads by a lot, such as kr-vs-kp, the rule keeps H and the
  hybrid equals it.
- **What it leads to:** extending ADR 0031 decision 6 to missForest is the user's call. This read is one
  loss short of supported.
