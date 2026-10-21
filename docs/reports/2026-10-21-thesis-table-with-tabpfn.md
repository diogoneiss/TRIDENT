# Thesis table with TabPFN v2 in the strong tier (built by script, 2026-10-21, 17:18 GMT-3)

This supersedes [2026-10-17-thesis-table-development](2026-10-17-thesis-table-development.md) for
every column that involves the strong tier. ADR [0028](../adr/0028-tabpfn-joins-the-strong-tier-as-an-offline-member.md)
put TabPFN v2 in that tier as an offline member.

**Sources:**

- the development tables: `uv run --python 3.11 python scripts/experiments/thesis_table.py --seeds <set>`;
- the confirmation summary: `scripts/experiments/tabpfn_report.py`.

Everything is read-only on the store. The old report's columns come back with
`thesis_table.py --tabpfn none`.

**What changed since the old report:**

- **TabPFN's per-fold values** come from E55 (101/202/303 and E44's confirmation seeds) and E58
  (42/7/13 and 404/505/606).
- **Where they enter:**
  - the "strong" and "tuned_strong" tiers and the best of all;
  - a new column, "K − tabpfn".
- **Holm** now runs across eight opponents.
- **Unchanged:** the bar, the tuned bar, missForest, the per-type bar and D4.

**Two caveats travel with every claim against the strong tier:**

- **The distribution (E56).** At 80nan, TabPFN's wins are point-accuracy wins with fills more shrunk
  than TRIDENT's. TRIDENT's dispersion is closer to the truth on 14 of 15 confirmation variants.
- **The default output (E57).** From 40nan on, the default output is KC (ADR 0027). Against TabPFN
  on confirmation it reads better 6, no difference 4, equivalent 1, worse 4. All four losses are at
  80nan.

## The statement, in five lines

- **Against the bar of 5 and the tuned bar,** nothing changes. D4 still gives 13 wins of 21 on
  101/202/303.
- **Against the strong tier with TabPFN,** TRIDENT goes from mostly better to mixed:
  - 101/202/303: 5 better, 9 no difference, 7 worse;
  - 42/7/13: 3 / 9 / 4;
  - 404/505/606: 3 / 2 / 7;
  - confirmation: 6 / 4 / 5.
- **TRIDENT leads TabPFN by table and by rate:** spambase at every rate, letter and pendigits at
  20–60nan (pendigits_20nan excepted), biodeg below 80nan, and kc2 at 60–80nan.
- **TabPFN leads at 80nan, and on kr-vs-kp and credit-g from 40nan.**
- **Under MNAR (E58),** TRIDENT's H is never worse than TabPFN (6 better, 4 no difference), and
  TabPFN gains no ground over MCAR.

## Confirmation set (E44, seeds 707/808/909): summary

```text
## Confirmation set (E44), seeds 707/808/909
  - K − tabpfn: better 6 / no difference 4 / worse 5 (fold-pair t, seed agreement)
  - K − best_of_all: better 6 / no difference 3 / worse 6 (fold-pair t, seed agreement)
  - K − strong_live: better 12 / no difference 1 / worse 2 (fold-pair t, seed agreement)
  - K − strong (ADR 0028): better 6 / no difference 4 / worse 5 (fold-pair t, seed agreement)
## E57: confirmation set (E44), seeds 707/808/909, KC and H against TabPFN
  - K − tabpfn: better 6 / no difference 4 / worse 5 (fold-pair t, seed agreement)
  - KC − tabpfn: better 6 / no difference 4 / equivalent 1 / worse 4 (fold-pair t, seed agreement)
  - H − tabpfn: better 5 / no difference 5 / worse 5 (fold-pair t, seed agreement)
  - calls that change from K to KC: ['letter_80nan: K worse -> KC equivalent']
```

## Development set, seeds 101/202/303 (primary)

## Development set, seeds 101/202/303, induced impute_score, K readout

| variant | K | K − bar | K − tuned | K − missforest | K − tabpfn | K − strong | K − tuned_strong | K − best_of_all | K − per_type |
|---|---|---|---|---|---|---|---|---|---|
| `kr-vs-kp_20nan` | 0.5866 | hgb/lgbm: -0.0193 [-0.0316, -0.0070] better | tuned_bar: -0.0131 [-0.0273, +0.0011] none | missforest: -0.0786 [-0.0933, -0.0639] better | tabpfn: +0.0059 [-0.0053, +0.0172] none | tabpfn: +0.0059 [-0.0053, +0.0172] none | tabpfn: +0.0059 [-0.0053, +0.0172] none | tabpfn: +0.0059 [-0.0053, +0.0172] none | per_type: -0.0193 [-0.0316, -0.0070] better |
| `kr-vs-kp_40nan` | 0.7258 | hgb/lgbm: -0.0446 [-0.0523, -0.0368] better | tuned_bar: -0.0225 [-0.0317, -0.0132] better | missforest: -0.0776 [-0.0876, -0.0677] better | tabpfn: +0.0102 [+0.0013, +0.0191] worse | tabpfn: +0.0102 [+0.0013, +0.0191] worse | tabpfn: +0.0102 [+0.0013, +0.0191] worse | tabpfn: +0.0102 [+0.0013, +0.0191] worse | per_type: -0.0446 [-0.0523, -0.0368] better |
| `kr-vs-kp_60nan` | 0.8711 | hgb: -0.0379 [-0.0463, -0.0296] better | tuned_bar: +0.0137 [+0.0057, +0.0217] worse | missforest: -0.0685 [-0.0805, -0.0565] better | tabpfn: +0.0432 [+0.0352, +0.0512] worse | tabpfn: +0.0432 [+0.0352, +0.0512] worse | tabpfn: +0.0432 [+0.0352, +0.0512] worse | tabpfn: +0.0432 [+0.0352, +0.0512] worse | per_type: -0.0379 [-0.0463, -0.0296] better |
| `kr-vs-kp_80nan` | 0.9815 | mean_mode: -0.0185 [-0.0268, -0.0102] better | tuned_bar: +0.0040 [-0.0039, +0.0119] none | missforest: -0.0298 [-0.0400, -0.0195] better | tabpfn: +0.0404 [+0.0330, +0.0479] worse | tabpfn: +0.0404 [+0.0330, +0.0479] worse | tabpfn: +0.0404 [+0.0330, +0.0479] worse | tabpfn: +0.0404 [+0.0330, +0.0479] worse | per_type: -0.0185 [-0.0268, -0.0102] better |
| `spambase_20nan` | 0.8529 | hgb/lgbm: -0.0061 [-0.0184, +0.0063] none | tuned_bar: +0.0009 [-0.0117, +0.0134] none | missforest: -0.0053 [-0.0151, +0.0045] none | tabpfn: -0.0843 [-0.0987, -0.0698] better | missforest: -0.0053 [-0.0151, +0.0045] none | missforest_tuned: -0.0068 [-0.0160, +0.0024] none | hgb/missforest: -0.0047 [-0.0162, +0.0067] none | per_type: -0.0061 [-0.0184, +0.0063] none |
| `spambase_40nan` | 0.9023 | hgb/lgbm: -0.0317 [-0.0411, -0.0223] better | tuned_bar: -0.0128 [-0.0194, -0.0063] better | missforest: -0.0358 [-0.0428, -0.0287] better | tabpfn: -0.0550 [-0.0672, -0.0428] better | missforest: -0.0358 [-0.0428, -0.0287] better | missforest_tuned: -0.0364 [-0.0431, -0.0298] better | hgb/lgbm: -0.0317 [-0.0411, -0.0223] better | per_type: -0.0317 [-0.0411, -0.0223] better |
| `spambase_60nan` | 0.9404 | hgb: -0.0409 [-0.0479, -0.0339] better | tuned_bar: -0.0139 [-0.0197, -0.0082] better | missforest: -0.0399 [-0.0443, -0.0354] better | tabpfn: -0.0337 [-0.0412, -0.0262] better | condexp: -0.0163 [-0.0200, -0.0126] better | condexp: -0.0163 [-0.0200, -0.0126] better | condexp: -0.0163 [-0.0200, -0.0126] better | per_type: -0.0409 [-0.0479, -0.0339] better |
| `spambase_80nan` | 0.9846 | mean_mode: -0.0154 [-0.0199, -0.0108] better | tuned_bar: -0.0060 [-0.0097, -0.0022] better | missforest: -0.0443 [-0.0518, -0.0368] better | tabpfn: -0.0074 [-0.0115, -0.0033] better | tabpfn: -0.0074 [-0.0115, -0.0033] better | missforest_tuned: -0.0033 [-0.0064, -0.0002] better | tabpfn: -0.0074 [-0.0115, -0.0033] better | per_type: -0.0154 [-0.0199, -0.0108] better |
| `credit-g_20nan` | 0.8751 | hgb: -0.0387 [-0.0556, -0.0217] better | tuned_bar: -0.0198 [-0.0354, -0.0042] better | missforest: -0.0156 [-0.0336, +0.0024] none | tabpfn: -0.0011 [-0.0140, +0.0117] none | tabpfn: -0.0011 [-0.0140, +0.0117] none | tabpfn: -0.0011 [-0.0140, +0.0117] none | tabpfn: -0.0011 [-0.0140, +0.0117] none | per_type: -0.0374 [-0.0539, -0.0210] better |
| `credit-g_40nan` | 0.9208 | hgb: -0.0436 [-0.0567, -0.0306] better | tuned_bar: -0.0175 [-0.0304, -0.0045] better | missforest: -0.0105 [-0.0193, -0.0016] better | tabpfn: +0.0169 [+0.0075, +0.0263] worse | tabpfn: +0.0169 [+0.0075, +0.0263] worse | tabpfn: +0.0169 [+0.0075, +0.0263] worse | tabpfn: +0.0169 [+0.0075, +0.0263] worse | per_type: -0.0425 [-0.0547, -0.0303] better |
| `credit-g_60nan` | 0.9932 | mean_mode: -0.0068 [-0.0224, +0.0088] none | tuned_bar: +0.0130 [-0.0005, +0.0264] none | missforest: -0.0167 [-0.0306, -0.0028] better | tabpfn: +0.0422 [+0.0284, +0.0560] worse | tabpfn: +0.0422 [+0.0284, +0.0560] worse | tabpfn: +0.0422 [+0.0284, +0.0560] worse | tabpfn: +0.0422 [+0.0284, +0.0560] worse | per_type: -0.0068 [-0.0224, +0.0088] none |
| `credit-g_80nan` | 1.0105 | mean_mode: +0.0105 [+0.0043, +0.0167] worse | tuned_bar: +0.0105 [+0.0043, +0.0167] worse | missforest: -0.0874 [-0.1006, -0.0742] better | tabpfn: +0.0141 [+0.0065, +0.0217] worse | condexp/tabpfn: +0.0150 [+0.0080, +0.0219] worse | condexp/tabpfn: +0.0150 [+0.0080, +0.0219] worse | condexp/tabpfn: +0.0150 [+0.0080, +0.0219] worse | per_type: +0.0105 [+0.0043, +0.0167] worse |
| `vehicle_20nan` | 0.4113 | hgb/lgbm: -0.0525 [-0.0663, -0.0388] better | tuned_bar: -0.0477 [-0.0571, -0.0384] better | missforest: -0.0348 [-0.0604, -0.0091] better | tabpfn: -0.0077 [-0.0207, +0.0053] none | tabpfn: -0.0077 [-0.0207, +0.0053] none | tabpfn: -0.0077 [-0.0207, +0.0053] none | tabpfn: -0.0077 [-0.0207, +0.0053] none | per_type: -0.0525 [-0.0663, -0.0388] better |
| `vehicle_60nan` | 0.6398 | hgb/lgbm: -0.0632 [-0.0769, -0.0494] better | tuned_bar: -0.0567 [-0.0682, -0.0452] better | missforest: -0.0111 [-0.0247, +0.0025] none | tabpfn: +0.0052 [-0.0049, +0.0152] none | tabpfn: +0.0052 [-0.0049, +0.0152] none | tabpfn: +0.0052 [-0.0049, +0.0152] none | tabpfn: +0.0052 [-0.0049, +0.0152] none | per_type: -0.0632 [-0.0769, -0.0494] better |
| `biodeg_20nan` | 0.5898 | hgb/lgbm: -0.0589 [-0.0808, -0.0369] better | tuned_bar: -0.0523 [-0.0705, -0.0341] better | missforest: -0.0371 [-0.0586, -0.0156] better | tabpfn: -0.0322 [-0.0616, -0.0027] better | missforest/tabpfn: -0.0302 [-0.0609, +0.0005] none | missforest_tuned/tabpfn: -0.0298 [-0.0560, -0.0035] better | missforest/tabpfn: -0.0302 [-0.0609, +0.0005] none | per_type: -0.0589 [-0.0808, -0.0369] better |
| `biodeg_60nan` | 0.7644 | hgb: -0.0978 [-0.1084, -0.0872] better | tuned_bar: -0.0813 [-0.0915, -0.0711] better | missforest: +0.0115 [-0.0024, +0.0255] none | tabpfn: -0.0200 [-0.0322, -0.0077] better | missforest: +0.0115 [-0.0024, +0.0255] none | missforest_tuned: +0.0083 [-0.0062, +0.0228] none | missforest: +0.0115 [-0.0024, +0.0255] none | per_type: -0.0978 [-0.1084, -0.0872] better |
| `kc2_20nan` | 0.5363 | knn5: +0.0148 [-0.0840, +0.1136] none | tuned_bar: -0.0179 [-0.1091, +0.0733] none | missforest: -0.0354 [-0.1458, +0.0749] none | tabpfn: +0.0434 [-0.0570, +0.1437] none | tabpfn: +0.0434 [-0.0570, +0.1437] none | tabpfn: +0.0434 [-0.0570, +0.1437] none | tabpfn: +0.0434 [-0.0570, +0.1437] none | per_type: +0.0148 [-0.0840, +0.1136] none |
| `kc2_60nan` | 0.5529 | knn5: -0.1713 [-0.2299, -0.1127] better | tuned_bar: -0.1922 [-0.2520, -0.1324] better | missforest: -0.1134 [-0.1505, -0.0763] better | tabpfn: -0.0684 [-0.1178, -0.0189] better | softimpute: -0.0378 [-0.0613, -0.0144] better | softimpute: -0.0378 [-0.0613, -0.0144] better | softimpute: -0.0378 [-0.0613, -0.0144] better | per_type: -0.1713 [-0.2299, -0.1127] better |
| `pendigits_20nan` | 0.3203 | knn10: -0.0307 [-0.0333, -0.0281] better | tuned_bar: -0.0236 [-0.0274, -0.0199] better | missforest: -0.0138 [-0.0182, -0.0094] better | tabpfn: +0.0123 [+0.0080, +0.0165] worse | tabpfn: +0.0123 [+0.0080, +0.0165] worse | tabpfn: +0.0123 [+0.0080, +0.0165] worse | tabpfn: +0.0123 [+0.0080, +0.0165] worse | per_type: -0.0307 [-0.0333, -0.0281] better |
| `letter_20nan` | 0.4331 | knn10: -0.0391 [-0.0419, -0.0364] better | tuned_bar: -0.0355 [-0.0376, -0.0335] better | missforest: -0.0298 [-0.0335, -0.0261] better | tabpfn: -0.0209 [-0.0240, -0.0179] better | tabpfn: -0.0209 [-0.0240, -0.0179] better | tabpfn: -0.0209 [-0.0240, -0.0179] better | tabpfn: -0.0209 [-0.0240, -0.0179] better | per_type: -0.0391 [-0.0419, -0.0364] better |
| `electricity_20nan` | 0.6183 | hgb: +0.0161 [+0.0056, +0.0267] worse | tuned_bar: +0.0387 [+0.0288, +0.0487] worse | missforest: -0.0010 [-0.0093, +0.0074] none | tabpfn: -0.0116 [-0.0237, +0.0004] none | missforest: -0.0010 [-0.0093, +0.0074] none | missforest_tuned: +0.0025 [-0.0068, +0.0118] none | hgb: +0.0161 [+0.0056, +0.0267] worse | per_type: +0.0161 [+0.0056, +0.0267] worse |

† ran a configuration that is no longer the default (removed file or old encoder); left out of the counts.

- K vs bar on 21 variants: better 16 / no difference 3 / worse 2 (fold-pair t, seed agreement); tables below zero 8 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0195)
  - 20nan: better 6 / no difference 2 / worse 1 (fold-pair t, seed agreement)
  - 40nan: better 3 / no difference 0 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 5 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - 80nan: better 2 / no difference 0 / worse 1 (fold-pair t, seed agreement)
- K vs tuned on 21 variants: better 13 / no difference 5 / worse 3 (fold-pair t, seed agreement); tables below zero 8 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0273)
  - 20nan: better 5 / no difference 3 / worse 1 (fold-pair t, seed agreement)
  - 40nan: better 3 / no difference 0 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 4 / no difference 1 / worse 1 (fold-pair t, seed agreement)
  - 80nan: better 1 / no difference 1 / worse 1 (fold-pair t, seed agreement)
- K vs missforest on 21 variants: better 15 / no difference 6 / worse 0 (fold-pair t, seed agreement); tables below zero 9 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0039)
  - 20nan: better 5 / no difference 4 / worse 0 (fold-pair t, seed agreement)
  - 40nan: better 3 / no difference 0 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 4 / no difference 2 / worse 0 (fold-pair t, seed agreement)
  - 80nan: better 3 / no difference 0 / worse 0 (fold-pair t, seed agreement)
- K vs tabpfn on 21 variants: better 8 / no difference 6 / worse 7 (fold-pair t, seed agreement); tables below zero 6 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.4961)
  - 20nan: better 3 / no difference 5 / worse 1 (fold-pair t, seed agreement)
  - 40nan: better 1 / no difference 0 / worse 2 (fold-pair t, seed agreement)
  - 60nan: better 3 / no difference 1 / worse 2 (fold-pair t, seed agreement)
  - 80nan: better 1 / no difference 0 / worse 2 (fold-pair t, seed agreement)
- K vs strong on 21 variants: better 5 / no difference 9 / worse 7 (fold-pair t, seed agreement); tables below zero 5 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.9102)
  - 20nan: better 1 / no difference 7 / worse 1 (fold-pair t, seed agreement)
  - 40nan: better 1 / no difference 0 / worse 2 (fold-pair t, seed agreement)
  - 60nan: better 2 / no difference 2 / worse 2 (fold-pair t, seed agreement)
  - 80nan: better 1 / no difference 0 / worse 2 (fold-pair t, seed agreement)
- K vs tuned_strong on 21 variants: better 6 / no difference 8 / worse 7 (fold-pair t, seed agreement); tables below zero 4 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.8203)
  - 20nan: better 2 / no difference 6 / worse 1 (fold-pair t, seed agreement)
  - 40nan: better 1 / no difference 0 / worse 2 (fold-pair t, seed agreement)
  - 60nan: better 2 / no difference 2 / worse 2 (fold-pair t, seed agreement)
  - 80nan: better 1 / no difference 0 / worse 2 (fold-pair t, seed agreement)
- K vs best_of_all on 21 variants: better 5 / no difference 8 / worse 8 (fold-pair t, seed agreement); tables below zero 4 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.5703)
  - 20nan: better 1 / no difference 6 / worse 2 (fold-pair t, seed agreement)
  - 40nan: better 1 / no difference 0 / worse 2 (fold-pair t, seed agreement)
  - 60nan: better 2 / no difference 2 / worse 2 (fold-pair t, seed agreement)
  - 80nan: better 1 / no difference 0 / worse 2 (fold-pair t, seed agreement)
- K vs per_type on 21 variants: better 16 / no difference 3 / worse 2 (fold-pair t, seed agreement); tables below zero 8 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0195)
  - 20nan: better 6 / no difference 2 / worse 1 (fold-pair t, seed agreement)
  - 40nan: better 3 / no difference 0 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 5 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - 80nan: better 2 / no difference 0 / worse 1 (fold-pair t, seed agreement)
- Holm across opponents: missforest 0.0312, bar 0.1367, per_type 0.1367, tuned 0.1367, tabpfn 1.0000, best_of_all 1.0000, tuned_strong 1.0000, strong 1.0000
- D4, a win needs K better than both the bar and the tuned bar: 13 wins of 21; worse than either on 3 (kr-vs-kp_60nan, credit-g_80nan, electricity_20nan)

## Development set, seeds 42/7/13

## Development set, seeds 42/7/13, induced impute_score, K readout

| variant | K | K − bar | K − tuned | K − missforest | K − tabpfn | K − strong | K − tuned_strong | K − best_of_all | K − per_type |
|---|---|---|---|---|---|---|---|---|---|
| `kr-vs-kp_20nan` | 0.5863 | lgbm: -0.0219 [-0.0321, -0.0118] better | tuned_bar: -0.0216 [-0.0332, -0.0101] better | missforest: -0.0715 [-0.0869, -0.0561] better | tabpfn: +0.0033 [-0.0096, +0.0162] none | tabpfn: +0.0033 [-0.0096, +0.0162] none | tabpfn: +0.0033 [-0.0096, +0.0162] none | tabpfn: +0.0033 [-0.0096, +0.0162] none | per_type: -0.0219 [-0.0321, -0.0118] better |
| `kr-vs-kp_40nan` † | 0.7582 | lgbm: -0.0058 [-0.0160, +0.0043] none | tuned_bar: +0.0127 [+0.0020, +0.0235] worse | missforest: -0.0425 [-0.0524, -0.0325] better | tabpfn: +0.0464 [+0.0390, +0.0537] worse | tabpfn: +0.0464 [+0.0390, +0.0537] worse | tabpfn: +0.0464 [+0.0390, +0.0537] worse | tabpfn: +0.0464 [+0.0390, +0.0537] worse | per_type: -0.0058 [-0.0160, +0.0043] none |
| `kr-vs-kp_60nan` | 0.8649 | hgb/lgbm: -0.0377 [-0.0456, -0.0299] better | tuned_bar: +0.0094 [-0.0007, +0.0195] none | missforest: -0.0705 [-0.0789, -0.0622] better | tabpfn: +0.0380 [+0.0310, +0.0450] worse | tabpfn: +0.0380 [+0.0310, +0.0450] worse | tabpfn: +0.0380 [+0.0310, +0.0450] worse | tabpfn: +0.0380 [+0.0310, +0.0450] worse | per_type: -0.0377 [-0.0456, -0.0299] better |
| `kr-vs-kp_80nan` | 0.9759 | mean_mode: -0.0241 [-0.0299, -0.0184] better | tuned_bar: -0.0051 [-0.0111, +0.0009] none | missforest: -0.0452 [-0.0523, -0.0382] better | tabpfn: +0.0340 [+0.0265, +0.0416] worse | tabpfn: +0.0340 [+0.0265, +0.0416] worse | tabpfn: +0.0340 [+0.0265, +0.0416] worse | tabpfn: +0.0340 [+0.0265, +0.0416] worse | per_type: -0.0241 [-0.0299, -0.0184] better |
| `spambase_20nan` | 0.8481 | lgbm: -0.0167 [-0.0302, -0.0032] better | tuned_bar: -0.0070 [-0.0215, +0.0075] none | missforest: -0.0114 [-0.0230, +0.0003] none | tabpfn: -0.0898 [-0.1069, -0.0726] better | missforest: -0.0114 [-0.0230, +0.0003] none | missforest_tuned: -0.0128 [-0.0249, -0.0007] better | missforest: -0.0114 [-0.0230, +0.0003] none | per_type: -0.0167 [-0.0302, -0.0032] better |
| `spambase_40nan` † | 0.9030 | hgb/lgbm: -0.0336 [-0.0406, -0.0267] better | tuned_bar: -0.0128 [-0.0183, -0.0072] better | missforest: -0.0386 [-0.0461, -0.0312] better | tabpfn: -0.0551 [-0.0644, -0.0458] better | missforest: -0.0386 [-0.0461, -0.0312] better | missforest_tuned: -0.0368 [-0.0466, -0.0270] better | hgb/lgbm: -0.0336 [-0.0406, -0.0267] better | per_type: -0.0336 [-0.0406, -0.0267] better |
| `spambase_60nan` | 0.9422 | hgb/lgbm: -0.0422 [-0.0473, -0.0370] better | tuned_bar: -0.0129 [-0.0176, -0.0082] better | missforest: -0.0384 [-0.0452, -0.0315] better | tabpfn: -0.0323 [-0.0384, -0.0262] better | condexp: -0.0175 [-0.0277, -0.0073] better | condexp/missforest_tuned: -0.0165 [-0.0222, -0.0107] better | condexp: -0.0175 [-0.0277, -0.0073] better | per_type: -0.0422 [-0.0473, -0.0370] better |
| `spambase_80nan` | 0.9871 | mean_mode: -0.0129 [-0.0172, -0.0086] better | tuned_bar: -0.0036 [-0.0075, +0.0004] equivalent | missforest: -0.0461 [-0.0542, -0.0380] better | tabpfn: -0.0049 [-0.0087, -0.0010] better | tabpfn: -0.0049 [-0.0087, -0.0010] better | missforest_tuned: -0.0012 [-0.0036, +0.0012] equivalent | tabpfn: -0.0049 [-0.0087, -0.0010] better | per_type: -0.0129 [-0.0172, -0.0086] better |
| `credit-g_20nan` | 0.8818 | hgb/lgbm: -0.0326 [-0.0462, -0.0191] better | tuned_bar: -0.0094 [-0.0223, +0.0034] none | missforest: -0.0028 [-0.0105, +0.0048] none | tabpfn: +0.0101 [-0.0019, +0.0221] none | tabpfn: +0.0101 [-0.0019, +0.0221] none | tabpfn: +0.0101 [-0.0019, +0.0221] none | tabpfn: +0.0101 [-0.0019, +0.0221] none | per_type: -0.0309 [-0.0461, -0.0158] better |
| `credit-g_40nan` † | 0.9556 | hgb/lgbm: -0.0079 [-0.0276, +0.0117] none | tuned_bar: +0.0145 [-0.0028, +0.0318] none | missforest: +0.0250 [+0.0082, +0.0419] worse | tabpfn: +0.0550 [+0.0418, +0.0681] worse | tabpfn: +0.0550 [+0.0418, +0.0681] worse | tabpfn: +0.0550 [+0.0418, +0.0681] worse | tabpfn: +0.0550 [+0.0418, +0.0681] worse | per_type: -0.0064 [-0.0254, +0.0126] none |
| `credit-g_60nan` | 0.9922 | mean_mode: -0.0078 [-0.0162, +0.0006] none | tuned_bar: +0.0003 [-0.0092, +0.0097] equivalent | missforest: -0.0287 [-0.0395, -0.0178] better | tabpfn: +0.0387 [+0.0300, +0.0475] worse | tabpfn: +0.0387 [+0.0300, +0.0475] worse | tabpfn: +0.0387 [+0.0300, +0.0475] worse | tabpfn: +0.0387 [+0.0300, +0.0475] worse | per_type: -0.0078 [-0.0162, +0.0006] none |
| `credit-g_80nan` | 1.0079 | mean_mode: +0.0079 [+0.0021, +0.0137] worse | tuned_bar: +0.0079 [+0.0021, +0.0137] worse | missforest: -0.0849 [-0.0938, -0.0759] better | tabpfn: +0.0072 [-0.0013, +0.0158] none | condexp/tabpfn: +0.0092 [+0.0019, +0.0164] worse | condexp/tabpfn: +0.0092 [+0.0019, +0.0164] worse | condexp/mean_mode/tabpfn: +0.0092 [+0.0023, +0.0160] worse | per_type: +0.0079 [+0.0021, +0.0137] worse |
| `vehicle_20nan` | 0.4178 | hgb/lgbm: -0.0534 [-0.0706, -0.0363] better | tuned_bar: -0.0493 [-0.0652, -0.0334] better | missforest: -0.0338 [-0.0585, -0.0090] better | tabpfn: -0.0022 [-0.0143, +0.0100] none | tabpfn: -0.0022 [-0.0143, +0.0100] none | tabpfn: -0.0022 [-0.0143, +0.0100] none | tabpfn: -0.0022 [-0.0143, +0.0100] none | per_type: -0.0534 [-0.0706, -0.0363] better |
| `vehicle_60nan` | 0.6449 | hgb/lgbm: -0.0552 [-0.0664, -0.0439] better | tuned_bar: -0.0519 [-0.0643, -0.0396] better | missforest: -0.0018 [-0.0188, +0.0153] none | tabpfn: +0.0098 [-0.0086, +0.0282] none | tabpfn: +0.0098 [-0.0086, +0.0282] none | tabpfn: +0.0098 [-0.0086, +0.0282] none | tabpfn: +0.0098 [-0.0086, +0.0282] none | per_type: -0.0552 [-0.0664, -0.0439] better |
| `biodeg_20nan` | 0.5865 | lgbm: -0.0467 [-0.0679, -0.0256] better | tuned_bar: -0.0381 [-0.0598, -0.0163] better | missforest: -0.0318 [-0.0525, -0.0111] better | tabpfn: -0.0299 [-0.0533, -0.0065] better | missforest/tabpfn: -0.0275 [-0.0497, -0.0053] better | missforest_tuned/tabpfn: -0.0258 [-0.0435, -0.0081] better | missforest/tabpfn: -0.0275 [-0.0497, -0.0053] better | per_type: -0.0467 [-0.0679, -0.0256] better |
| `biodeg_60nan` | 0.7605 | hgb/lgbm: -0.1052 [-0.1181, -0.0923] better | tuned_bar: -0.0872 [-0.1001, -0.0744] better | missforest: -0.0019 [-0.0201, +0.0162] none | tabpfn: -0.0268 [-0.0410, -0.0127] better | missforest: -0.0019 [-0.0201, +0.0162] none | missforest_tuned: -0.0002 [-0.0204, +0.0201] none | missforest: -0.0019 [-0.0201, +0.0162] none | per_type: -0.1052 [-0.1181, -0.0923] better |
| `kc2_20nan` | 0.5414 | knn5: -0.0251 [-0.1011, +0.0508] none | tuned_bar: -0.0251 [-0.1011, +0.0508] none | missforest: -0.0640 [-0.1552, +0.0272] none | tabpfn: +0.0169 [-0.0712, +0.1051] none | tabpfn: +0.0169 [-0.0712, +0.1051] none | tabpfn: +0.0169 [-0.0712, +0.1051] none | knn5/tabpfn: +0.0200 [-0.0620, +0.1020] none | per_type: -0.0251 [-0.1011, +0.0508] none |
| `kc2_60nan` | 0.5723 | knn10/knn5: -0.1406 [-0.2010, -0.0803] better | tuned_bar: -0.1477 [-0.1992, -0.0962] better | missforest: -0.1026 [-0.1380, -0.0672] better | tabpfn: -0.0431 [-0.1023, +0.0161] none | softimpute/tabpfn: -0.0384 [-0.0850, +0.0081] none | softimpute/tabpfn: -0.0384 [-0.0850, +0.0081] none | softimpute/tabpfn: -0.0384 [-0.0850, +0.0081] none | per_type: -0.1406 [-0.2010, -0.0803] better |
| `pendigits_20nan` † | 0.3324 | knn10: -0.0176 [-0.0205, -0.0146] better | tuned_bar: -0.0100 [-0.0128, -0.0073] better | missforest: -0.0020 [-0.0078, +0.0037] none | tabpfn: +0.0245 [+0.0206, +0.0284] worse | tabpfn: +0.0245 [+0.0206, +0.0284] worse | tabpfn: +0.0245 [+0.0206, +0.0284] worse | tabpfn: +0.0245 [+0.0206, +0.0284] worse | per_type: -0.0176 [-0.0205, -0.0146] better |
| `letter_20nan` † | 0.4593 | knn10: -0.0136 [-0.0162, -0.0110] better | tuned_bar: -0.0094 [-0.0117, -0.0072] better | missforest: -0.0029 [-0.0059, +0.0001] none | tabpfn: +0.0053 [+0.0028, +0.0079] worse | tabpfn: +0.0053 [+0.0028, +0.0079] worse | tabpfn: +0.0053 [+0.0028, +0.0079] worse | tabpfn: +0.0053 [+0.0028, +0.0079] worse | per_type: -0.0136 [-0.0162, -0.0110] better |
| `electricity_20nan` | 0.6179 | hgb/lgbm: +0.0132 [-0.0073, +0.0337] none | tuned_bar: +0.0441 [+0.0267, +0.0616] worse | missforest: -0.0015 [-0.0184, +0.0153] none | tabpfn: -0.0083 [-0.0289, +0.0123] none | missforest/tabpfn: -0.0015 [-0.0181, +0.0151] none | missforest_tuned: -0.0040 [-0.0204, +0.0123] none | hgb/lgbm: +0.0132 [-0.0073, +0.0337] none | per_type: +0.0135 [-0.0072, +0.0342] none |

† ran a configuration that is no longer the default (removed file or old encoder); left out of the counts.

- K vs bar on 16 variants: better 12 / no difference 3 / worse 1 (fold-pair t, seed agreement); tables below zero 6 of 7 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0469)
  - 20nan: better 5 / no difference 2 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 5 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - 80nan: better 2 / no difference 0 / worse 1 (fold-pair t, seed agreement)
- K vs tuned on 16 variants: better 7 / no difference 5 / equivalent 2 / worse 2 (fold-pair t, seed agreement); tables below zero 6 of 7 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.1094)
  - 20nan: better 3 / no difference 3 / worse 1 (fold-pair t, seed agreement)
  - 60nan: better 4 / no difference 1 / equivalent 1 / worse 0 (fold-pair t, seed agreement)
  - 80nan: better 0 / no difference 1 / equivalent 1 / worse 1 (fold-pair t, seed agreement)
- K vs missforest on 16 variants: better 10 / no difference 6 / worse 0 (fold-pair t, seed agreement); tables below zero 7 of 7 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0156)
  - 20nan: better 3 / no difference 4 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 4 / no difference 2 / worse 0 (fold-pair t, seed agreement)
  - 80nan: better 3 / no difference 0 / worse 0 (fold-pair t, seed agreement)
- K vs tabpfn on 16 variants: better 5 / no difference 8 / worse 3 (fold-pair t, seed agreement); tables below zero 4 of 7 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.5781)
  - 20nan: better 2 / no difference 5 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 2 / no difference 2 / worse 2 (fold-pair t, seed agreement)
  - 80nan: better 1 / no difference 1 / worse 1 (fold-pair t, seed agreement)
- K vs strong on 16 variants: better 3 / no difference 9 / worse 4 (fold-pair t, seed agreement); tables below zero 4 of 7 (two-sided Wilcoxon on per-table mean relative gaps, p = 1.0000)
  - 20nan: better 1 / no difference 6 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 1 / no difference 3 / worse 2 (fold-pair t, seed agreement)
  - 80nan: better 1 / no difference 0 / worse 2 (fold-pair t, seed agreement)
- K vs tuned_strong on 16 variants: better 3 / no difference 8 / equivalent 1 / worse 4 (fold-pair t, seed agreement); tables below zero 4 of 7 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.9375)
  - 20nan: better 2 / no difference 5 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 1 / no difference 3 / worse 2 (fold-pair t, seed agreement)
  - 80nan: better 0 / no difference 0 / equivalent 1 / worse 2 (fold-pair t, seed agreement)
- K vs best_of_all on 16 variants: better 3 / no difference 9 / worse 4 (fold-pair t, seed agreement); tables below zero 3 of 7 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.6875)
  - 20nan: better 1 / no difference 6 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 1 / no difference 3 / worse 2 (fold-pair t, seed agreement)
  - 80nan: better 1 / no difference 0 / worse 2 (fold-pair t, seed agreement)
- K vs per_type on 16 variants: better 12 / no difference 3 / worse 1 (fold-pair t, seed agreement); tables below zero 6 of 7 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0469)
  - 20nan: better 5 / no difference 2 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 5 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - 80nan: better 2 / no difference 0 / worse 1 (fold-pair t, seed agreement)
- Holm across opponents: missforest 0.1250, bar 0.3281, per_type 0.3281, tuned 0.5469, tabpfn 1.0000, best_of_all 1.0000, tuned_strong 1.0000, strong 1.0000
- D4, a win needs K better than both the bar and the tuned bar: 7 wins of 16; worse than either on 2 (credit-g_80nan, electricity_20nan)

## Development set at 40nan and above, seeds 404/505/606

## Development set, seeds 404/505/606, induced impute_score, K readout

| variant | K | K − bar | K − tuned | K − missforest | K − tabpfn | K − strong | K − tuned_strong | K − best_of_all | K − per_type |
|---|---|---|---|---|---|---|---|---|---|
| `kr-vs-kp_40nan` | 0.7238 | hgb/lgbm: -0.0383 [-0.0461, -0.0304] better | tuned_bar: -0.0224 [-0.0318, -0.0130] better | missforest: -0.0736 [-0.0848, -0.0624] better | tabpfn: +0.0110 [+0.0054, +0.0167] worse | tabpfn: +0.0110 [+0.0054, +0.0167] worse | tabpfn: +0.0110 [+0.0054, +0.0167] worse | tabpfn: +0.0110 [+0.0054, +0.0167] worse | per_type: -0.0383 [-0.0461, -0.0304] better |
| `kr-vs-kp_60nan` | 0.8673 | hgb/lgbm: -0.0398 [-0.0478, -0.0318] better | tuned_bar: +0.0078 [+0.0007, +0.0150] worse | missforest: -0.0721 [-0.0810, -0.0631] better | tabpfn: +0.0435 [+0.0368, +0.0503] worse | tabpfn: +0.0435 [+0.0368, +0.0503] worse | tabpfn: +0.0435 [+0.0368, +0.0503] worse | tabpfn: +0.0435 [+0.0368, +0.0503] worse | per_type: -0.0398 [-0.0478, -0.0318] better |
| `kr-vs-kp_80nan` | 0.9790 | mean_mode: -0.0210 [-0.0305, -0.0116] better | tuned_bar: -0.0049 [-0.0164, +0.0067] none | missforest: -0.0385 [-0.0503, -0.0267] better | tabpfn: +0.0364 [+0.0271, +0.0457] worse | tabpfn: +0.0364 [+0.0271, +0.0457] worse | tabpfn: +0.0364 [+0.0271, +0.0457] worse | tabpfn: +0.0364 [+0.0271, +0.0457] worse | per_type: -0.0210 [-0.0305, -0.0116] better |
| `spambase_40nan` | 0.9038 | lgbm: -0.0319 [-0.0403, -0.0235] better | tuned_bar: -0.0131 [-0.0198, -0.0064] better | missforest: -0.0369 [-0.0452, -0.0286] better | tabpfn: -0.0539 [-0.0634, -0.0444] better | missforest: -0.0369 [-0.0452, -0.0286] better | missforest_tuned: -0.0367 [-0.0442, -0.0292] better | lgbm: -0.0319 [-0.0403, -0.0235] better | per_type: -0.0319 [-0.0403, -0.0235] better |
| `spambase_60nan` | 0.9391 | hgb: -0.0435 [-0.0501, -0.0370] better | tuned_bar: -0.0143 [-0.0189, -0.0097] better | missforest: -0.0364 [-0.0430, -0.0299] better | tabpfn: -0.0346 [-0.0432, -0.0261] better | condexp: -0.0162 [-0.0230, -0.0094] better | condexp: -0.0162 [-0.0230, -0.0094] better | condexp: -0.0162 [-0.0230, -0.0094] better | per_type: -0.0435 [-0.0501, -0.0370] better |
| `spambase_80nan` | 0.9863 | mean_mode: -0.0137 [-0.0195, -0.0078] better | tuned_bar: -0.0041 [-0.0093, +0.0012] equivalent | missforest: -0.0441 [-0.0529, -0.0354] better | tabpfn: -0.0055 [-0.0103, -0.0006] better | tabpfn: -0.0055 [-0.0103, -0.0006] better | missforest_tuned: -0.0020 [-0.0056, +0.0015] equivalent | tabpfn: -0.0055 [-0.0103, -0.0006] better | per_type: -0.0137 [-0.0195, -0.0078] better |
| `credit-g_40nan` | 0.9239 | hgb: -0.0374 [-0.0491, -0.0257] better | tuned_bar: -0.0192 [-0.0354, -0.0029] better | missforest: -0.0053 [-0.0137, +0.0031] none | tabpfn: +0.0220 [+0.0138, +0.0301] worse | tabpfn: +0.0220 [+0.0138, +0.0301] worse | tabpfn: +0.0220 [+0.0138, +0.0301] worse | tabpfn: +0.0220 [+0.0138, +0.0301] worse | per_type: -0.0374 [-0.0491, -0.0257] better |
| `credit-g_60nan` | 0.9899 | mean_mode: -0.0101 [-0.0206, +0.0003] none | tuned_bar: +0.0041 [-0.0105, +0.0188] none | missforest: -0.0218 [-0.0348, -0.0087] better | tabpfn: +0.0392 [+0.0280, +0.0503] worse | tabpfn: +0.0392 [+0.0280, +0.0503] worse | tabpfn: +0.0392 [+0.0280, +0.0503] worse | tabpfn: +0.0392 [+0.0280, +0.0503] worse | per_type: -0.0101 [-0.0206, +0.0003] none |
| `credit-g_80nan` | 1.0112 | mean_mode: +0.0112 [+0.0066, +0.0158] worse | tuned_bar: +0.0112 [+0.0066, +0.0158] worse | missforest: -0.0724 [-0.0795, -0.0654] better | tabpfn: +0.0139 [+0.0083, +0.0194] worse | tabpfn: +0.0139 [+0.0083, +0.0194] worse | tabpfn: +0.0139 [+0.0083, +0.0194] worse | tabpfn: +0.0139 [+0.0083, +0.0194] worse | per_type: +0.0112 [+0.0066, +0.0158] worse |
| `vehicle_60nan` | 0.6378 | hgb/lgbm: -0.0634 [-0.0830, -0.0439] better | tuned_bar: -0.0610 [-0.0812, -0.0408] better | missforest: -0.0135 [-0.0354, +0.0084] none | tabpfn: +0.0018 [-0.0220, +0.0256] none | tabpfn: +0.0018 [-0.0220, +0.0256] none | tabpfn: +0.0018 [-0.0220, +0.0256] none | tabpfn: +0.0018 [-0.0220, +0.0256] none | per_type: -0.0634 [-0.0830, -0.0439] better |
| `biodeg_60nan` | 0.7738 | hgb/lgbm: -0.0887 [-0.1022, -0.0751] better | tuned_bar: -0.0746 [-0.0871, -0.0620] better | missforest: +0.0117 [+0.0009, +0.0225] worse | tabpfn: -0.0135 [-0.0255, -0.0016] better | missforest: +0.0117 [+0.0009, +0.0225] worse | missforest_tuned: +0.0100 [-0.0003, +0.0203] none | missforest: +0.0117 [+0.0009, +0.0225] worse | per_type: -0.0887 [-0.1022, -0.0751] better |
| `kc2_60nan` | 0.5502 | knn5: -0.1900 [-0.2535, -0.1265] better | tuned_bar: -0.1958 [-0.2626, -0.1290] better | missforest: -0.1202 [-0.1685, -0.0718] better | tabpfn: -0.0739 [-0.1333, -0.0145] better | softimpute: -0.0385 [-0.0807, +0.0037] none | softimpute: -0.0385 [-0.0807, +0.0037] none | softimpute: -0.0385 [-0.0807, +0.0037] none | per_type: -0.1900 [-0.2535, -0.1265] better |

† ran a configuration that is no longer the default (removed file or old encoder); left out of the counts.

- K vs bar on 12 variants: better 10 / no difference 1 / worse 1 (fold-pair t, seed agreement); tables below zero 6 of 6 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0312)
  - 40nan: better 3 / no difference 0 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 5 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - 80nan: better 2 / no difference 0 / worse 1 (fold-pair t, seed agreement)
- K vs tuned on 12 variants: better 7 / no difference 2 / equivalent 1 / worse 2 (fold-pair t, seed agreement); tables below zero 6 of 6 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0312)
  - 40nan: better 3 / no difference 0 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 4 / no difference 1 / worse 1 (fold-pair t, seed agreement)
  - 80nan: better 0 / no difference 1 / equivalent 1 / worse 1 (fold-pair t, seed agreement)
- K vs missforest on 12 variants: better 9 / no difference 2 / worse 1 (fold-pair t, seed agreement); tables below zero 5 of 6 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0625)
  - 40nan: better 2 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 4 / no difference 1 / worse 1 (fold-pair t, seed agreement)
  - 80nan: better 3 / no difference 0 / worse 0 (fold-pair t, seed agreement)
- K vs tabpfn on 12 variants: better 5 / no difference 1 / worse 6 (fold-pair t, seed agreement); tables below zero 3 of 6 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.8438)
  - 40nan: better 1 / no difference 0 / worse 2 (fold-pair t, seed agreement)
  - 60nan: better 3 / no difference 1 / worse 2 (fold-pair t, seed agreement)
  - 80nan: better 1 / no difference 0 / worse 2 (fold-pair t, seed agreement)
- K vs strong on 12 variants: better 3 / no difference 2 / worse 7 (fold-pair t, seed agreement); tables below zero 2 of 6 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.8438)
  - 40nan: better 1 / no difference 0 / worse 2 (fold-pair t, seed agreement)
  - 60nan: better 1 / no difference 2 / worse 3 (fold-pair t, seed agreement)
  - 80nan: better 1 / no difference 0 / worse 2 (fold-pair t, seed agreement)
- K vs tuned_strong on 12 variants: better 2 / no difference 3 / equivalent 1 / worse 6 (fold-pair t, seed agreement); tables below zero 2 of 6 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.8438)
  - 40nan: better 1 / no difference 0 / worse 2 (fold-pair t, seed agreement)
  - 60nan: better 1 / no difference 3 / worse 2 (fold-pair t, seed agreement)
  - 80nan: better 0 / no difference 0 / equivalent 1 / worse 2 (fold-pair t, seed agreement)
- K vs best_of_all on 12 variants: better 3 / no difference 2 / worse 7 (fold-pair t, seed agreement); tables below zero 2 of 6 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.8438)
  - 40nan: better 1 / no difference 0 / worse 2 (fold-pair t, seed agreement)
  - 60nan: better 1 / no difference 2 / worse 3 (fold-pair t, seed agreement)
  - 80nan: better 1 / no difference 0 / worse 2 (fold-pair t, seed agreement)
- K vs per_type on 12 variants: better 10 / no difference 1 / worse 1 (fold-pair t, seed agreement); tables below zero 6 of 6 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0312)
  - 40nan: better 3 / no difference 0 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 5 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - 80nan: better 2 / no difference 0 / worse 1 (fold-pair t, seed agreement)
- Holm across opponents: bar 0.2500, tuned 0.2500, per_type 0.2500, missforest 0.3125, tabpfn 1.0000, strong 1.0000, tuned_strong 1.0000, best_of_all 1.0000
- D4, a win needs K better than both the bar and the tuned bar: 7 wins of 12; worse than either on 2 (kr-vs-kp_60nan, credit-g_80nan)
