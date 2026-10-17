# Thesis table on the development set (built by script, 2026-10-17)

Generated at 2026-10-17 01:29 GMT-3 by `uv run --python 3.11 python scripts/experiments/thesis_table.py --seeds <set>` (T21; plan 2026-10-17 §2a), read-only on the store. The baselines a fold CSV lacked come from E41's offline pass, merged only where they reproduce the run's own baselines to 1e-9 (all did). These are **development** numbers: every default was chosen on these 21 variants. The opponent "bar" is the weak tier (mean_mode, knn5, knn10, hgb, lgbm); "strong" is missForest plus E41's arms; "per_type" is item 7b. Each cell: opponent chosen, K − opponent [95% interval], verdict.

## Development set, seeds 101/202/303, induced impute_score, K readout

| variant | K | K − bar | K − missforest | K − strong | K − best_of_all | K − per_type |
|---|---|---|---|---|---|---|
| `kr-vs-kp_20nan` | 0.5866 | hgb/lgbm: -0.0193 [-0.0316, -0.0070] better | missforest: -0.0786 [-0.0933, -0.0639] better | ice/missforest: -0.0749 [-0.0887, -0.0612] better | hgb/lgbm: -0.0193 [-0.0316, -0.0070] better | per_type: -0.0193 [-0.0316, -0.0070] better |
| `kr-vs-kp_40nan` | 0.7258 | hgb/lgbm: -0.0446 [-0.0523, -0.0368] better | missforest: -0.0776 [-0.0876, -0.0677] better | ice: -0.0648 [-0.0726, -0.0570] better | hgb/lgbm: -0.0446 [-0.0523, -0.0368] better | per_type: -0.0446 [-0.0523, -0.0368] better |
| `kr-vs-kp_60nan` | 0.8711 | hgb: -0.0379 [-0.0463, -0.0296] better | missforest: -0.0685 [-0.0805, -0.0565] better | ice: -0.0353 [-0.0419, -0.0286] better | hgb/ice: -0.0344 [-0.0419, -0.0268] better | per_type: -0.0379 [-0.0463, -0.0296] better |
| `kr-vs-kp_80nan` | 0.9815 | mean_mode: -0.0185 [-0.0268, -0.0102] better | missforest: -0.0298 [-0.0400, -0.0195] better | condexp: -0.0185 [-0.0268, -0.0102] better | condexp: -0.0185 [-0.0268, -0.0102] better | per_type: -0.0185 [-0.0268, -0.0102] better |
| `spambase_20nan` | 0.8529 | hgb/lgbm: -0.0061 [-0.0184, +0.0063] none | missforest: -0.0053 [-0.0151, +0.0045] none | missforest: -0.0053 [-0.0151, +0.0045] none | hgb/missforest: -0.0047 [-0.0162, +0.0067] none | per_type: -0.0061 [-0.0184, +0.0063] none |
| `spambase_40nan` | 0.9023 | hgb/lgbm: -0.0317 [-0.0411, -0.0223] better | missforest: -0.0358 [-0.0428, -0.0287] better | missforest: -0.0358 [-0.0428, -0.0287] better | hgb/lgbm: -0.0317 [-0.0411, -0.0223] better | per_type: -0.0317 [-0.0411, -0.0223] better |
| `spambase_60nan` | 0.9404 | hgb: -0.0409 [-0.0479, -0.0339] better | missforest: -0.0399 [-0.0443, -0.0354] better | condexp: -0.0163 [-0.0200, -0.0126] better | condexp: -0.0163 [-0.0200, -0.0126] better | per_type: -0.0409 [-0.0479, -0.0339] better |
| `spambase_80nan` | 0.9846 | mean_mode: -0.0154 [-0.0199, -0.0108] better | missforest: -0.0443 [-0.0518, -0.0368] better | condexp: -0.0144 [-0.0196, -0.0093] better | condexp/mean_mode: -0.0138 [-0.0183, -0.0092] better | per_type: -0.0154 [-0.0199, -0.0108] better |
| `credit-g_20nan` | 0.8751 | hgb: -0.0387 [-0.0556, -0.0217] better | missforest: -0.0156 [-0.0336, +0.0024] none | missforest: -0.0156 [-0.0336, +0.0024] none | missforest: -0.0156 [-0.0336, +0.0024] none | per_type: -0.0374 [-0.0539, -0.0210] better |
| `credit-g_40nan` | 0.9208 | hgb: -0.0436 [-0.0567, -0.0306] better | missforest: -0.0105 [-0.0193, -0.0016] better | missforest: -0.0105 [-0.0193, -0.0016] better | missforest: -0.0105 [-0.0193, -0.0016] better | per_type: -0.0425 [-0.0547, -0.0303] better |
| `credit-g_60nan` | 0.9932 | mean_mode: -0.0068 [-0.0224, +0.0088] none | missforest: -0.0167 [-0.0306, -0.0028] better | condexp: +0.0049 [-0.0098, +0.0197] none | condexp: +0.0049 [-0.0098, +0.0197] none | per_type: -0.0068 [-0.0224, +0.0088] none |
| `credit-g_80nan` | 1.0105 | mean_mode: +0.0105 [+0.0043, +0.0167] worse | missforest: -0.0874 [-0.1006, -0.0742] better | condexp: +0.0124 [+0.0062, +0.0186] worse | condexp: +0.0124 [+0.0062, +0.0186] worse | per_type: +0.0105 [+0.0043, +0.0167] worse |
| `vehicle_20nan` | 0.4113 | hgb/lgbm: -0.0525 [-0.0663, -0.0388] better | missforest: -0.0348 [-0.0604, -0.0091] better | missforest: -0.0348 [-0.0604, -0.0091] better | missforest: -0.0348 [-0.0604, -0.0091] better | per_type: -0.0525 [-0.0663, -0.0388] better |
| `vehicle_60nan` | 0.6398 | hgb/lgbm: -0.0632 [-0.0769, -0.0494] better | missforest: -0.0111 [-0.0247, +0.0025] none | missforest: -0.0111 [-0.0247, +0.0025] none | missforest: -0.0111 [-0.0247, +0.0025] none | per_type: -0.0632 [-0.0769, -0.0494] better |
| `biodeg_20nan` | 0.5898 | hgb/lgbm: -0.0589 [-0.0808, -0.0369] better | missforest: -0.0371 [-0.0586, -0.0156] better | missforest: -0.0371 [-0.0586, -0.0156] better | missforest: -0.0371 [-0.0586, -0.0156] better | per_type: -0.0589 [-0.0808, -0.0369] better |
| `biodeg_60nan` | 0.7644 | hgb: -0.0978 [-0.1084, -0.0872] better | missforest: +0.0115 [-0.0024, +0.0255] none | missforest: +0.0115 [-0.0024, +0.0255] none | missforest: +0.0115 [-0.0024, +0.0255] none | per_type: -0.0978 [-0.1084, -0.0872] better |
| `kc2_20nan` | 0.5363 | knn5: +0.0148 [-0.0840, +0.1136] none | missforest: -0.0354 [-0.1458, +0.0749] none | ice/softimpute: -0.0005 [-0.1130, +0.1121] none | knn5: +0.0148 [-0.0840, +0.1136] none | per_type: +0.0148 [-0.0840, +0.1136] none |
| `kc2_60nan` | 0.5529 | knn5: -0.1713 [-0.2299, -0.1127] better | missforest: -0.1134 [-0.1505, -0.0763] better | softimpute: -0.0378 [-0.0613, -0.0144] better | softimpute: -0.0378 [-0.0613, -0.0144] better | per_type: -0.1713 [-0.2299, -0.1127] better |
| `pendigits_20nan` | 0.3203 | knn10: -0.0307 [-0.0333, -0.0281] better | missforest: -0.0138 [-0.0182, -0.0094] better | missforest: -0.0138 [-0.0182, -0.0094] better | missforest: -0.0138 [-0.0182, -0.0094] better | per_type: -0.0307 [-0.0333, -0.0281] better |
| `letter_20nan` | 0.4331 | knn10: -0.0391 [-0.0419, -0.0364] better | missforest: -0.0298 [-0.0335, -0.0261] better | missforest: -0.0298 [-0.0335, -0.0261] better | missforest: -0.0298 [-0.0335, -0.0261] better | per_type: -0.0391 [-0.0419, -0.0364] better |
| `electricity_20nan` | 0.6183 | hgb: +0.0161 [+0.0056, +0.0267] worse | missforest: -0.0010 [-0.0093, +0.0074] none | missforest: -0.0010 [-0.0093, +0.0074] none | hgb: +0.0161 [+0.0056, +0.0267] worse | per_type: +0.0161 [+0.0056, +0.0267] worse |

† ran a configuration that is no longer the default (removed file or old encoder); left out of the counts.

- K vs bar on 21 variants: better 16 / no difference 3 / worse 2 (fold-pair t, seed agreement); tables below zero 8 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0195)
  - 20nan: better 6 / no difference 2 / worse 1 (fold-pair t, seed agreement)
  - 40nan: better 3 / no difference 0 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 5 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - 80nan: better 2 / no difference 0 / worse 1 (fold-pair t, seed agreement)
- K vs missforest on 21 variants: better 15 / no difference 6 / worse 0 (fold-pair t, seed agreement); tables below zero 9 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0039)
  - 20nan: better 5 / no difference 4 / worse 0 (fold-pair t, seed agreement)
  - 40nan: better 3 / no difference 0 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 4 / no difference 2 / worse 0 (fold-pair t, seed agreement)
  - 80nan: better 3 / no difference 0 / worse 0 (fold-pair t, seed agreement)
- K vs strong on 21 variants: better 13 / no difference 7 / worse 1 (fold-pair t, seed agreement); tables below zero 9 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0039)
  - 20nan: better 5 / no difference 4 / worse 0 (fold-pair t, seed agreement)
  - 40nan: better 3 / no difference 0 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 3 / no difference 3 / worse 0 (fold-pair t, seed agreement)
  - 80nan: better 2 / no difference 0 / worse 1 (fold-pair t, seed agreement)
- K vs best_of_all on 21 variants: better 13 / no difference 6 / worse 2 (fold-pair t, seed agreement); tables below zero 8 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0391)
  - 20nan: better 5 / no difference 3 / worse 1 (fold-pair t, seed agreement)
  - 40nan: better 3 / no difference 0 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 3 / no difference 3 / worse 0 (fold-pair t, seed agreement)
  - 80nan: better 2 / no difference 0 / worse 1 (fold-pair t, seed agreement)
- K vs per_type on 21 variants: better 16 / no difference 3 / worse 2 (fold-pair t, seed agreement); tables below zero 8 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0195)
  - 20nan: better 6 / no difference 2 / worse 1 (fold-pair t, seed agreement)
  - 40nan: better 3 / no difference 0 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 5 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - 80nan: better 2 / no difference 0 / worse 1 (fold-pair t, seed agreement)
- Holm across opponents: missforest 0.0195, strong 0.0195, bar 0.0586, per_type 0.0586, best_of_all 0.0586

## Development set, seeds 42/7/13, induced impute_score, K readout

| variant | K | K − bar | K − missforest | K − strong | K − best_of_all | K − per_type |
|---|---|---|---|---|---|---|
| `kr-vs-kp_20nan` | 0.5863 | lgbm: -0.0219 [-0.0321, -0.0118] better | missforest: -0.0715 [-0.0869, -0.0561] better | ice/missforest: -0.0691 [-0.0803, -0.0580] better | lgbm: -0.0219 [-0.0321, -0.0118] better | per_type: -0.0219 [-0.0321, -0.0118] better |
| `kr-vs-kp_40nan` † | 0.7582 | lgbm: -0.0058 [-0.0160, +0.0043] none | missforest: -0.0425 [-0.0524, -0.0325] better | ice/missforest: -0.0307 [-0.0434, -0.0179] better | lgbm: -0.0058 [-0.0160, +0.0043] none | per_type: -0.0058 [-0.0160, +0.0043] none |
| `kr-vs-kp_60nan` | 0.8649 | hgb/lgbm: -0.0377 [-0.0456, -0.0299] better | missforest: -0.0705 [-0.0789, -0.0622] better | ice: -0.0449 [-0.0526, -0.0371] better | hgb/lgbm: -0.0377 [-0.0456, -0.0299] better | per_type: -0.0377 [-0.0456, -0.0299] better |
| `kr-vs-kp_80nan` | 0.9759 | mean_mode: -0.0241 [-0.0299, -0.0184] better | missforest: -0.0452 [-0.0523, -0.0382] better | condexp: -0.0241 [-0.0299, -0.0184] better | condexp: -0.0241 [-0.0299, -0.0184] better | per_type: -0.0241 [-0.0299, -0.0184] better |
| `spambase_20nan` | 0.8481 | lgbm: -0.0167 [-0.0302, -0.0032] better | missforest: -0.0114 [-0.0230, +0.0003] none | missforest: -0.0114 [-0.0230, +0.0003] none | missforest: -0.0114 [-0.0230, +0.0003] none | per_type: -0.0167 [-0.0302, -0.0032] better |
| `spambase_40nan` † | 0.9030 | hgb/lgbm: -0.0336 [-0.0406, -0.0267] better | missforest: -0.0386 [-0.0461, -0.0312] better | missforest: -0.0386 [-0.0461, -0.0312] better | hgb/lgbm: -0.0336 [-0.0406, -0.0267] better | per_type: -0.0336 [-0.0406, -0.0267] better |
| `spambase_60nan` | 0.9422 | hgb/lgbm: -0.0422 [-0.0473, -0.0370] better | missforest: -0.0384 [-0.0452, -0.0315] better | condexp: -0.0175 [-0.0277, -0.0073] better | condexp: -0.0175 [-0.0277, -0.0073] better | per_type: -0.0422 [-0.0473, -0.0370] better |
| `spambase_80nan` | 0.9871 | mean_mode: -0.0129 [-0.0172, -0.0086] better | missforest: -0.0461 [-0.0542, -0.0380] better | condexp: -0.0129 [-0.0165, -0.0092] better | condexp/mean_mode: -0.0125 [-0.0161, -0.0090] better | per_type: -0.0129 [-0.0172, -0.0086] better |
| `credit-g_20nan` | 0.8818 | hgb/lgbm: -0.0326 [-0.0462, -0.0191] better | missforest: -0.0028 [-0.0105, +0.0048] none | missforest: -0.0028 [-0.0105, +0.0048] none | missforest: -0.0028 [-0.0105, +0.0048] none | per_type: -0.0309 [-0.0461, -0.0158] better |
| `credit-g_40nan` † | 0.9556 | hgb/lgbm: -0.0079 [-0.0276, +0.0117] none | missforest: +0.0250 [+0.0082, +0.0419] worse | missforest: +0.0250 [+0.0082, +0.0419] worse | missforest: +0.0250 [+0.0082, +0.0419] worse | per_type: -0.0064 [-0.0254, +0.0126] none |
| `credit-g_60nan` | 0.9922 | mean_mode: -0.0078 [-0.0162, +0.0006] none | missforest: -0.0287 [-0.0395, -0.0178] better | condexp: +0.0042 [-0.0036, +0.0119] none | condexp: +0.0042 [-0.0036, +0.0119] none | per_type: -0.0078 [-0.0162, +0.0006] none |
| `credit-g_80nan` | 1.0079 | mean_mode: +0.0079 [+0.0021, +0.0137] worse | missforest: -0.0849 [-0.0938, -0.0759] better | condexp: +0.0079 [+0.0015, +0.0144] worse | condexp/mean_mode: +0.0081 [+0.0021, +0.0141] worse | per_type: +0.0079 [+0.0021, +0.0137] worse |
| `vehicle_20nan` | 0.4178 | hgb/lgbm: -0.0534 [-0.0706, -0.0363] better | missforest: -0.0338 [-0.0585, -0.0090] better | condexp/missforest: -0.0304 [-0.0485, -0.0123] better | condexp/missforest: -0.0304 [-0.0485, -0.0123] better | per_type: -0.0534 [-0.0706, -0.0363] better |
| `vehicle_60nan` | 0.6449 | hgb/lgbm: -0.0552 [-0.0664, -0.0439] better | missforest: -0.0018 [-0.0188, +0.0153] none | missforest: -0.0018 [-0.0188, +0.0153] none | missforest: -0.0018 [-0.0188, +0.0153] none | per_type: -0.0552 [-0.0664, -0.0439] better |
| `biodeg_20nan` | 0.5865 | lgbm: -0.0467 [-0.0679, -0.0256] better | missforest: -0.0318 [-0.0525, -0.0111] better | missforest: -0.0318 [-0.0525, -0.0111] better | missforest: -0.0318 [-0.0525, -0.0111] better | per_type: -0.0467 [-0.0679, -0.0256] better |
| `biodeg_60nan` | 0.7605 | hgb/lgbm: -0.1052 [-0.1181, -0.0923] better | missforest: -0.0019 [-0.0201, +0.0162] none | missforest: -0.0019 [-0.0201, +0.0162] none | missforest: -0.0019 [-0.0201, +0.0162] none | per_type: -0.1052 [-0.1181, -0.0923] better |
| `kc2_20nan` | 0.5414 | knn5: -0.0251 [-0.1011, +0.0508] none | missforest: -0.0640 [-0.1552, +0.0272] none | softimpute: -0.0357 [-0.0868, +0.0154] none | knn5/softimpute: -0.0217 [-0.0917, +0.0483] none | per_type: -0.0251 [-0.1011, +0.0508] none |
| `kc2_60nan` | 0.5723 | knn10/knn5: -0.1406 [-0.2010, -0.0803] better | missforest: -0.1026 [-0.1380, -0.0672] better | ice/softimpute: -0.0816 [-0.1030, -0.0601] better | ice/softimpute: -0.0816 [-0.1030, -0.0601] better | per_type: -0.1406 [-0.2010, -0.0803] better |
| `pendigits_20nan` † | 0.3324 | knn10: -0.0176 [-0.0205, -0.0146] better | missforest: -0.0020 [-0.0078, +0.0037] none | missforest: -0.0020 [-0.0078, +0.0037] none | missforest: -0.0020 [-0.0078, +0.0037] none | per_type: -0.0176 [-0.0205, -0.0146] better |
| `letter_20nan` † | 0.4593 | knn10: -0.0136 [-0.0162, -0.0110] better | missforest: -0.0029 [-0.0059, +0.0001] none | missforest: -0.0029 [-0.0059, +0.0001] none | missforest: -0.0029 [-0.0059, +0.0001] none | per_type: -0.0136 [-0.0162, -0.0110] better |
| `electricity_20nan` | 0.6179 | hgb/lgbm: +0.0132 [-0.0073, +0.0337] none | missforest: -0.0015 [-0.0184, +0.0153] none | missforest: -0.0015 [-0.0184, +0.0153] none | hgb/lgbm: +0.0132 [-0.0073, +0.0337] none | per_type: +0.0135 [-0.0072, +0.0342] none |

† ran a configuration that is no longer the default (removed file or old encoder); left out of the counts.

- K vs bar on 16 variants: better 12 / no difference 3 / worse 1 (fold-pair t, seed agreement); tables below zero 6 of 7 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0469)
  - 20nan: better 5 / no difference 2 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 5 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - 80nan: better 2 / no difference 0 / worse 1 (fold-pair t, seed agreement)
- K vs missforest on 16 variants: better 10 / no difference 6 / worse 0 (fold-pair t, seed agreement); tables below zero 7 of 7 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0156)
  - 20nan: better 3 / no difference 4 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 4 / no difference 2 / worse 0 (fold-pair t, seed agreement)
  - 80nan: better 3 / no difference 0 / worse 0 (fold-pair t, seed agreement)
- K vs strong on 16 variants: better 8 / no difference 7 / worse 1 (fold-pair t, seed agreement); tables below zero 6 of 7 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0469)
  - 20nan: better 3 / no difference 4 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 3 / no difference 3 / worse 0 (fold-pair t, seed agreement)
  - 80nan: better 2 / no difference 0 / worse 1 (fold-pair t, seed agreement)
- K vs best_of_all on 16 variants: better 8 / no difference 7 / worse 1 (fold-pair t, seed agreement); tables below zero 5 of 7 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.1094)
  - 20nan: better 3 / no difference 4 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 3 / no difference 3 / worse 0 (fold-pair t, seed agreement)
  - 80nan: better 2 / no difference 0 / worse 1 (fold-pair t, seed agreement)
- K vs per_type on 16 variants: better 12 / no difference 3 / worse 1 (fold-pair t, seed agreement); tables below zero 6 of 7 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0469)
  - 20nan: better 5 / no difference 2 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 5 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - 80nan: better 2 / no difference 0 / worse 1 (fold-pair t, seed agreement)
- Holm across opponents: missforest 0.0781, bar 0.1875, strong 0.1875, per_type 0.1875, best_of_all 0.1875

## Development set, seeds 404/505/606, induced impute_score, K readout

| variant | K | K − bar | K − missforest | K − strong | K − best_of_all | K − per_type |
|---|---|---|---|---|---|---|
| `kr-vs-kp_40nan` | 0.7238 | hgb/lgbm: -0.0383 [-0.0461, -0.0304] better | missforest: -0.0736 [-0.0848, -0.0624] better | ice/missforest: -0.0688 [-0.0761, -0.0615] better | hgb/lgbm: -0.0383 [-0.0461, -0.0304] better | per_type: -0.0383 [-0.0461, -0.0304] better |
| `kr-vs-kp_60nan` | 0.8673 | hgb/lgbm: -0.0398 [-0.0478, -0.0318] better | missforest: -0.0721 [-0.0810, -0.0631] better | ice: -0.0442 [-0.0549, -0.0334] better | ice/lgbm: -0.0387 [-0.0472, -0.0302] better | per_type: -0.0398 [-0.0478, -0.0318] better |
| `kr-vs-kp_80nan` | 0.9790 | mean_mode: -0.0210 [-0.0305, -0.0116] better | missforest: -0.0385 [-0.0503, -0.0267] better | condexp: -0.0210 [-0.0305, -0.0116] better | condexp: -0.0210 [-0.0305, -0.0116] better | per_type: -0.0210 [-0.0305, -0.0116] better |
| `spambase_40nan` | 0.9038 | lgbm: -0.0319 [-0.0403, -0.0235] better | missforest: -0.0369 [-0.0452, -0.0286] better | missforest: -0.0369 [-0.0452, -0.0286] better | lgbm: -0.0319 [-0.0403, -0.0235] better | per_type: -0.0319 [-0.0403, -0.0235] better |
| `spambase_60nan` | 0.9391 | hgb: -0.0435 [-0.0501, -0.0370] better | missforest: -0.0364 [-0.0430, -0.0299] better | condexp: -0.0162 [-0.0230, -0.0094] better | condexp: -0.0162 [-0.0230, -0.0094] better | per_type: -0.0435 [-0.0501, -0.0370] better |
| `spambase_80nan` | 0.9863 | mean_mode: -0.0137 [-0.0195, -0.0078] better | missforest: -0.0441 [-0.0529, -0.0354] better | condexp: -0.0121 [-0.0168, -0.0075] better | condexp: -0.0121 [-0.0168, -0.0075] better | per_type: -0.0137 [-0.0195, -0.0078] better |
| `credit-g_40nan` | 0.9239 | hgb: -0.0374 [-0.0491, -0.0257] better | missforest: -0.0053 [-0.0137, +0.0031] none | missforest: -0.0053 [-0.0137, +0.0031] none | missforest: -0.0053 [-0.0137, +0.0031] none | per_type: -0.0374 [-0.0491, -0.0257] better |
| `credit-g_60nan` | 0.9899 | mean_mode: -0.0101 [-0.0206, +0.0003] none | missforest: -0.0218 [-0.0348, -0.0087] better | condexp: +0.0017 [-0.0083, +0.0116] none | condexp: +0.0017 [-0.0083, +0.0116] none | per_type: -0.0101 [-0.0206, +0.0003] none |
| `credit-g_80nan` | 1.0112 | mean_mode: +0.0112 [+0.0066, +0.0158] worse | missforest: -0.0724 [-0.0795, -0.0654] better | condexp: +0.0111 [+0.0067, +0.0156] worse | condexp/mean_mode: +0.0112 [+0.0066, +0.0159] worse | per_type: +0.0112 [+0.0066, +0.0158] worse |
| `vehicle_60nan` | 0.6378 | hgb/lgbm: -0.0634 [-0.0830, -0.0439] better | missforest: -0.0135 [-0.0354, +0.0084] none | missforest: -0.0135 [-0.0354, +0.0084] none | missforest: -0.0135 [-0.0354, +0.0084] none | per_type: -0.0634 [-0.0830, -0.0439] better |
| `biodeg_60nan` | 0.7738 | hgb/lgbm: -0.0887 [-0.1022, -0.0751] better | missforest: +0.0117 [+0.0009, +0.0225] worse | missforest: +0.0117 [+0.0009, +0.0225] worse | missforest: +0.0117 [+0.0009, +0.0225] worse | per_type: -0.0887 [-0.1022, -0.0751] better |
| `kc2_60nan` | 0.5502 | knn5: -0.1900 [-0.2535, -0.1265] better | missforest: -0.1202 [-0.1685, -0.0718] better | softimpute: -0.0385 [-0.0807, +0.0037] none | softimpute: -0.0385 [-0.0807, +0.0037] none | per_type: -0.1900 [-0.2535, -0.1265] better |

† ran a configuration that is no longer the default (removed file or old encoder); left out of the counts.

- K vs bar on 12 variants: better 10 / no difference 1 / worse 1 (fold-pair t, seed agreement); tables below zero 6 of 6 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0312)
  - 40nan: better 3 / no difference 0 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 5 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - 80nan: better 2 / no difference 0 / worse 1 (fold-pair t, seed agreement)
- K vs missforest on 12 variants: better 9 / no difference 2 / worse 1 (fold-pair t, seed agreement); tables below zero 5 of 6 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0625)
  - 40nan: better 2 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 4 / no difference 1 / worse 1 (fold-pair t, seed agreement)
  - 80nan: better 3 / no difference 0 / worse 0 (fold-pair t, seed agreement)
- K vs strong on 12 variants: better 6 / no difference 4 / worse 2 (fold-pair t, seed agreement); tables below zero 4 of 6 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.1562)
  - 40nan: better 2 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 2 / no difference 3 / worse 1 (fold-pair t, seed agreement)
  - 80nan: better 2 / no difference 0 / worse 1 (fold-pair t, seed agreement)
- K vs best_of_all on 12 variants: better 6 / no difference 4 / worse 2 (fold-pair t, seed agreement); tables below zero 4 of 6 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.1562)
  - 40nan: better 2 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 2 / no difference 3 / worse 1 (fold-pair t, seed agreement)
  - 80nan: better 2 / no difference 0 / worse 1 (fold-pair t, seed agreement)
- K vs per_type on 12 variants: better 10 / no difference 1 / worse 1 (fold-pair t, seed agreement); tables below zero 6 of 6 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0312)
  - 40nan: better 3 / no difference 0 / worse 0 (fold-pair t, seed agreement)
  - 60nan: better 5 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - 80nan: better 2 / no difference 0 / worse 1 (fold-pair t, seed agreement)
- Holm across opponents: bar 0.1562, per_type 0.1562, missforest 0.1875, strong 0.3125, best_of_all 0.3125

