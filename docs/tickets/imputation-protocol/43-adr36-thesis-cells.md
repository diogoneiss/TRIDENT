# The thesis tables' cells that ADR 0036 changed, rerun on today's default (2026-11-01 16:44 GMT-3)

Not a hypothesis test: a rerun so the thesis tables report today's default. Written before the
first cell.

- **Why:** ADR 0036 (accepted 2026-10-31) changed the default of pendigits and letter at 20 and
  40nan (BATCH 1024, LR_PRE and LR_DECODE x4, EPOCHS_DECODE 600). The tables still read those
  variants from runs on the old configuration.
- **The user's go:** "ok, va e faça o item 2 então (isso da adr 36)" (2026-11-01, after 16:20 GMT-3).
- **Which cells:** the development tables (`thesis_table.py`) read pendigits_20nan and letter_20nan
  at 101/202/303 and 42/7/13 (404/505/606 does not include them); the confirmation summary
  (`tabpfn_report.py`, `confirmation_report.py`) reads pendigits_40nan and letter_40nan at
  707/808/909. 18 cells.
- **Each cell:** today's default through the promoted files, five folds, `--score_checkpoints
  --score_calibrated` (every readout the tables use), MLflow tag
  `experiment=adr36-thesis-2026-11-01` (`scripts/experiments/adr36_thesis_runs.py`,
  `run_adr36_thesis_queue.sh`; 6 cells on gorgona8, 12 on medusa1).
- **Then:** `report_common.development_source` and `confirmation_report` read these runs for
  those variants, the tables are rebuilt, and the before/after counts are written here. The offline
  baselines (E41's, E42's, E55's TabPFN) depend only on the folds and are reused.

## Amendment (2026-11-01, 19:09 GMT-3): every cell on gorgona8

- **What happened:** the 18 runs finished (16:44–19:05, none failed, all with ADR 0036's BATCH 1024,
  LR_PRE 0.00136, LR_DECODE 0.004, EPOCHS_DECODE 600). Rebuilding the tables stopped at their check
  that a run's logged baselines equal E41's offline recomputation: the 6 gorgona8 runs match to 0,
  the 12 medusa1 runs differ on knn5 and knn10 only, by 4e-5 to 2.3e-3. A CPU difference in kNN,
  not a fold one; but knn10 is pendigits' bar, and the check is exact by design.
- **The fix:** all 18 cells rerun on gorgona8 under `experiment=adr36-thesis-g8-2026-11-01`, where E41's
  baselines were computed; the tables read that tag. The first tag's runs stay in the store,
  unread. No table number was read from the first runs.

## Second amendment (2026-11-01, 23:53 GMT-3): pendigits_60nan joins

- **Why:** the user added pendigits_60nan to ADR 0036 on E109 ("sim, incluir"). It is in the
  confirmation set, so its cells at 707/808/909 rerun too, on gorgona8 under the same tag, one after
  another in a fourth process beside the three queues. 21 cells in all.

## Outcome

**Written 2026-11-02 02:49 GMT-3.** All 21 cells ran on gorgona8 (19:10–02:46 GMT-3, none failed), with ADR 0036's
configuration; their in-run baselines match E41's offline ones exactly, and the tables rebuilt.
**Only the five ADR 0036 variants moved, all in TRIDENT's favour:**

| variant (seed set) | K before → after | K − TabPFN before → after |
|---|---|---|
| pendigits_20nan (101/202/303) | 0.3203 → 0.3004 | +0.012 worse → **−0.008 better** |
| letter_20nan (101/202/303) | 0.4331 → 0.4206 | −0.021 → −0.033 better |
| pendigits_20nan (42/7/13) | 0.3324 † → 0.3011 | +0.025 worse → **−0.007 better** |
| letter_20nan (42/7/13) | 0.4593 † → 0.4218 | +0.005 worse → **−0.032 better** |
| pendigits_40nan (707/808/909) | | −0.011 → −0.028 better |
| pendigits_60nan (707/808/909) | | −0.018 → −0.024 better |
| letter_40nan (707/808/909) | | −0.034 → −0.043 better |

- **Headline counts that change:**
  - 101/202/303, K vs TabPFN: better 8 / no difference 6 / worse 7 → **9 / 6 / 6**; vs the strong tier
    5 / 9 / 7 → 6 / 9 / 6; vs the best of all 5 / 8 / 8 → 6 / 8 / 7.
  - 42/7/13: the two 20nan variants are no longer labelled (†, old configuration), so the set grows
    from 16 to 18 variants: K vs bar 12 / 3 / 1 → 14 / 3 / 1; vs TabPFN 5 / 8 / 3 → 7 / 8 / 3; D4
    wins 7 of 16 → 9 of 18.
  - Confirmation (E44): the three variants were already better than TabPFN; the counts do not move.
- **Reader changes (no number of another variant moved):** \`report_common.development_source\` and
  \`confirmation_report.cell\` read \`adr36-thesis-g8-2026-11-01\` for the five variants;
  \`tabpfn_report\` reads KC through \`readout_key\` on runs ADR 0027 calibrated (it looked for the
  older \`calibrated\` key and reported "no KC" on them).
- **Still to do:** the consolidated report and the thesis-table report quote the old numbers.

### Before → after, unedited (\`diff\` of the scripts' outputs, 2026-11-02 02:49 GMT-3)

\`thesis_table.py --seeds 101/202/303\`:

```
23,24c23,24
< | `pendigits_20nan` | 0.3203 | knn10: -0.0307 [-0.0333, -0.0281] better | tuned_bar: -0.0236 [-0.0274, -0.0199] better | missforest: -0.0138 [-0.0182, -0.0094] better | tabpfn: +0.0123 [+0.0080, +0.0165] worse | tabpfn: +0.0123 [+0.0080, +0.0165] worse | tabpfn: +0.0123 [+0.0080, +0.0165] worse | tabpfn: +0.0123 [+0.0080, +0.0165] worse | per_type: -0.0307 [-0.0333, -0.0281] better |
< | `letter_20nan` | 0.4331 | knn10: -0.0391 [-0.0419, -0.0364] better | tuned_bar: -0.0355 [-0.0376, -0.0335] better | missforest: -0.0298 [-0.0335, -0.0261] better | tabpfn: -0.0209 [-0.0240, -0.0179] better | tabpfn: -0.0209 [-0.0240, -0.0179] better | tabpfn: -0.0209 [-0.0240, -0.0179] better | tabpfn: -0.0209 [-0.0240, -0.0179] better | per_type: -0.0391 [-0.0419, -0.0364] better |
---
> | `pendigits_20nan` | 0.3004 | knn10: -0.0506 [-0.0541, -0.0472] better | tuned_bar: -0.0436 [-0.0482, -0.0390] better | missforest: -0.0337 [-0.0402, -0.0272] better | tabpfn: -0.0077 [-0.0130, -0.0023] better | tabpfn: -0.0077 [-0.0130, -0.0023] better | tabpfn: -0.0077 [-0.0130, -0.0023] better | tabpfn: -0.0077 [-0.0130, -0.0023] better | per_type: -0.0506 [-0.0541, -0.0472] better |
> | `letter_20nan` | 0.4206 | knn10: -0.0516 [-0.0548, -0.0484] better | tuned_bar: -0.0480 [-0.0504, -0.0456] better | missforest: -0.0423 [-0.0457, -0.0389] better | tabpfn: -0.0334 [-0.0365, -0.0303] better | tabpfn: -0.0334 [-0.0365, -0.0303] better | tabpfn: -0.0334 [-0.0365, -0.0303] better | tabpfn: -0.0334 [-0.0365, -0.0303] better | per_type: -0.0516 [-0.0548, -0.0484] better |
44,45c44,45
< - K vs tabpfn on 21 variants: better 8 / no difference 6 / worse 7 (fold-pair t, seed agreement); tables below zero 6 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.4961)
<   - 20nan: better 3 / no difference 5 / worse 1 (fold-pair t, seed agreement)
---
> - K vs tabpfn on 21 variants: better 9 / no difference 6 / worse 6 (fold-pair t, seed agreement); tables below zero 7 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.1641)
>   - 20nan: better 4 / no difference 5 / worse 0 (fold-pair t, seed agreement)
49,50c49,50
< - K vs strong on 21 variants: better 5 / no difference 9 / worse 7 (fold-pair t, seed agreement); tables below zero 5 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.9102)
<   - 20nan: better 1 / no difference 7 / worse 1 (fold-pair t, seed agreement)
---
> - K vs strong on 21 variants: better 6 / no difference 9 / worse 6 (fold-pair t, seed agreement); tables below zero 6 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.5703)
>   - 20nan: better 2 / no difference 7 / worse 0 (fold-pair t, seed agreement)
54,55c54,55
< - K vs tuned_strong on 21 variants: better 6 / no difference 8 / worse 7 (fold-pair t, seed agreement); tables below zero 4 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.8203)
<   - 20nan: better 2 / no difference 6 / worse 1 (fold-pair t, seed agreement)
---
> - K vs tuned_strong on 21 variants: better 7 / no difference 8 / worse 6 (fold-pair t, seed agreement); tables below zero 5 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.6523)
>   - 20nan: better 3 / no difference 6 / worse 0 (fold-pair t, seed agreement)
59,60c59,60
< - K vs best_of_all on 21 variants: better 5 / no difference 8 / worse 8 (fold-pair t, seed agreement); tables below zero 4 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.5703)
<   - 20nan: better 1 / no difference 6 / worse 2 (fold-pair t, seed agreement)
---
> - K vs best_of_all on 21 variants: better 6 / no difference 8 / worse 7 (fold-pair t, seed agreement); tables below zero 5 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 1.0000)
>   - 20nan: better 2 / no difference 6 / worse 1 (fold-pair t, seed agreement)
69c69
< - Holm across opponents: missforest 0.0312, bar 0.1367, per_type 0.1367, tuned 0.1367, tabpfn 1.0000, best_of_all 1.0000, tuned_strong 1.0000, strong 1.0000
---
> - Holm across opponents: missforest 0.0312, bar 0.1367, per_type 0.1367, tuned 0.1367, tabpfn 0.6562, strong 1.0000, tuned_strong 1.0000, best_of_all 1.0000
```

\`thesis_table.py --seeds 42/7/13\`:

```
23,24c23,24
< | `pendigits_20nan` † | 0.3324 | knn10: -0.0176 [-0.0205, -0.0146] better | tuned_bar: -0.0100 [-0.0128, -0.0073] better | missforest: -0.0020 [-0.0078, +0.0037] none | tabpfn: +0.0245 [+0.0206, +0.0284] worse | tabpfn: +0.0245 [+0.0206, +0.0284] worse | tabpfn: +0.0245 [+0.0206, +0.0284] worse | tabpfn: +0.0245 [+0.0206, +0.0284] worse | per_type: -0.0176 [-0.0205, -0.0146] better |
< | `letter_20nan` † | 0.4593 | knn10: -0.0136 [-0.0162, -0.0110] better | tuned_bar: -0.0094 [-0.0117, -0.0072] better | missforest: -0.0029 [-0.0059, +0.0001] none | tabpfn: +0.0053 [+0.0028, +0.0079] worse | tabpfn: +0.0053 [+0.0028, +0.0079] worse | tabpfn: +0.0053 [+0.0028, +0.0079] worse | tabpfn: +0.0053 [+0.0028, +0.0079] worse | per_type: -0.0136 [-0.0162, -0.0110] better |
---
> | `pendigits_20nan` | 0.3011 | knn10: -0.0488 [-0.0525, -0.0452] better | tuned_bar: -0.0413 [-0.0447, -0.0379] better | missforest: -0.0333 [-0.0392, -0.0274] better | tabpfn: -0.0068 [-0.0112, -0.0024] better | tabpfn: -0.0068 [-0.0112, -0.0024] better | tabpfn: -0.0068 [-0.0112, -0.0024] better | tabpfn: -0.0068 [-0.0112, -0.0024] better | per_type: -0.0488 [-0.0525, -0.0452] better |
> | `letter_20nan` | 0.4218 | knn10: -0.0511 [-0.0545, -0.0477] better | tuned_bar: -0.0469 [-0.0500, -0.0438] better | missforest: -0.0404 [-0.0436, -0.0372] better | tabpfn: -0.0321 [-0.0359, -0.0284] better | tabpfn: -0.0321 [-0.0359, -0.0284] better | tabpfn: -0.0321 [-0.0359, -0.0284] better | tabpfn: -0.0321 [-0.0359, -0.0284] better | per_type: -0.0511 [-0.0545, -0.0477] better |
29,30c29,30
< - K vs bar on 16 variants: better 12 / no difference 3 / worse 1 (fold-pair t, seed agreement); tables below zero 6 of 7 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0469)
<   - 20nan: better 5 / no difference 2 / worse 0 (fold-pair t, seed agreement)
---
> - K vs bar on 18 variants: better 14 / no difference 3 / worse 1 (fold-pair t, seed agreement); tables below zero 8 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0117)
>   - 20nan: better 7 / no difference 2 / worse 0 (fold-pair t, seed agreement)
33,34c33,34
< - K vs tuned on 16 variants: better 7 / no difference 5 / equivalent 2 / worse 2 (fold-pair t, seed agreement); tables below zero 6 of 7 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.1094)
<   - 20nan: better 3 / no difference 3 / worse 1 (fold-pair t, seed agreement)
---
> - K vs tuned on 18 variants: better 9 / no difference 5 / equivalent 2 / worse 2 (fold-pair t, seed agreement); tables below zero 8 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0273)
>   - 20nan: better 5 / no difference 3 / worse 1 (fold-pair t, seed agreement)
37,38c37,38
< - K vs missforest on 16 variants: better 10 / no difference 6 / worse 0 (fold-pair t, seed agreement); tables below zero 7 of 7 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0156)
<   - 20nan: better 3 / no difference 4 / worse 0 (fold-pair t, seed agreement)
---
> - K vs missforest on 18 variants: better 12 / no difference 6 / worse 0 (fold-pair t, seed agreement); tables below zero 9 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0039)
>   - 20nan: better 5 / no difference 4 / worse 0 (fold-pair t, seed agreement)
41,42c41,42
< - K vs tabpfn on 16 variants: better 5 / no difference 8 / worse 3 (fold-pair t, seed agreement); tables below zero 4 of 7 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.5781)
<   - 20nan: better 2 / no difference 5 / worse 0 (fold-pair t, seed agreement)
---
> - K vs tabpfn on 18 variants: better 7 / no difference 8 / worse 3 (fold-pair t, seed agreement); tables below zero 6 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.2031)
>   - 20nan: better 4 / no difference 5 / worse 0 (fold-pair t, seed agreement)
45,46c45,46
< - K vs strong on 16 variants: better 3 / no difference 9 / worse 4 (fold-pair t, seed agreement); tables below zero 4 of 7 (two-sided Wilcoxon on per-table mean relative gaps, p = 1.0000)
<   - 20nan: better 1 / no difference 6 / worse 0 (fold-pair t, seed agreement)
---
> - K vs strong on 18 variants: better 5 / no difference 9 / worse 4 (fold-pair t, seed agreement); tables below zero 6 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.4258)
>   - 20nan: better 3 / no difference 6 / worse 0 (fold-pair t, seed agreement)
49,50c49,50
< - K vs tuned_strong on 16 variants: better 3 / no difference 8 / equivalent 1 / worse 4 (fold-pair t, seed agreement); tables below zero 4 of 7 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.9375)
<   - 20nan: better 2 / no difference 5 / worse 0 (fold-pair t, seed agreement)
---
> - K vs tuned_strong on 18 variants: better 5 / no difference 8 / equivalent 1 / worse 4 (fold-pair t, seed agreement); tables below zero 6 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.3594)
>   - 20nan: better 4 / no difference 5 / worse 0 (fold-pair t, seed agreement)
53,54c53,54
< - K vs best_of_all on 16 variants: better 3 / no difference 9 / worse 4 (fold-pair t, seed agreement); tables below zero 3 of 7 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.6875)
<   - 20nan: better 1 / no difference 6 / worse 0 (fold-pair t, seed agreement)
---
> - K vs best_of_all on 18 variants: better 5 / no difference 9 / worse 4 (fold-pair t, seed agreement); tables below zero 5 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.6523)
>   - 20nan: better 3 / no difference 6 / worse 0 (fold-pair t, seed agreement)
57,58c57,58
< - K vs per_type on 16 variants: better 12 / no difference 3 / worse 1 (fold-pair t, seed agreement); tables below zero 6 of 7 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0469)
<   - 20nan: better 5 / no difference 2 / worse 0 (fold-pair t, seed agreement)
---
> - K vs per_type on 18 variants: better 14 / no difference 3 / worse 1 (fold-pair t, seed agreement); tables below zero 8 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0117)
>   - 20nan: better 7 / no difference 2 / worse 0 (fold-pair t, seed agreement)
61,62c61,62
< - Holm across opponents: missforest 0.1250, bar 0.3281, per_type 0.3281, tuned 0.5469, tabpfn 1.0000, best_of_all 1.0000, tuned_strong 1.0000, strong 1.0000
< - D4, a win needs K better than both the bar and the tuned bar: 7 wins of 16; worse than either on 2 (credit-g_80nan, electricity_20nan)
---
> - Holm across opponents: missforest 0.0312, bar 0.0820, per_type 0.0820, tuned 0.1367, tabpfn 0.8125, tuned_strong 1.0000, strong 1.0000, best_of_all 1.0000
> - D4, a win needs K better than both the bar and the tuned bar: 9 wins of 18; worse than either on 2 (credit-g_80nan, electricity_20nan)
```

\`tabpfn_report.py\`:

```
20,21c20,21
< - `pendigits_20nan`: K − tabpfn +0.0123 [+0.0080, +0.0165] worse; tabpfn − strong -0.0260 [-0.0303, -0.0218] better; tabpfn − bar -0.0430 [-0.0481, -0.0378] better; tabpfn − tuned -0.0359 [-0.0390, -0.0329] better; K − best_of_all +0.0123 [+0.0080, +0.0165] worse; K − strong_live -0.0138 [-0.0182, -0.0094] better; K − strong (ADR 0028) +0.0123 [+0.0080, +0.0165] worse; best of all tabpfn; fit 13 s a fold
< - `letter_20nan`: K − tabpfn -0.0209 [-0.0240, -0.0179] better; tabpfn − strong -0.0089 [-0.0129, -0.0049] better; tabpfn − bar -0.0182 [-0.0214, -0.0150] better; tabpfn − tuned -0.0146 [-0.0163, -0.0128] better; K − best_of_all -0.0041 [-0.0073, -0.0009] better; K − strong_live -0.0298 [-0.0335, -0.0261] better; K − strong (ADR 0028) -0.0209 [-0.0240, -0.0179] better; best of all macode; fit 24 s a fold
---
> - `pendigits_20nan`: K − tabpfn -0.0077 [-0.0130, -0.0023] better; tabpfn − strong -0.0260 [-0.0303, -0.0218] better; tabpfn − bar -0.0430 [-0.0481, -0.0378] better; tabpfn − tuned -0.0359 [-0.0390, -0.0329] better; K − best_of_all -0.0077 [-0.0130, -0.0023] better; K − strong_live -0.0337 [-0.0402, -0.0272] better; K − strong (ADR 0028) -0.0077 [-0.0130, -0.0023] better; best of all tabpfn; fit 13 s a fold
> - `letter_20nan`: K − tabpfn -0.0334 [-0.0365, -0.0303] better; tabpfn − strong -0.0089 [-0.0129, -0.0049] better; tabpfn − bar -0.0182 [-0.0214, -0.0150] better; tabpfn − tuned -0.0146 [-0.0163, -0.0128] better; K − best_of_all -0.0166 [-0.0198, -0.0133] better; K − strong_live -0.0423 [-0.0457, -0.0389] better; K − strong (ADR 0028) -0.0334 [-0.0365, -0.0303] better; best of all macode; fit 24 s a fold
23c23
<   - K − tabpfn: better 8 / no difference 6 / worse 7 (fold-pair t, seed agreement)
---
>   - K − tabpfn: better 9 / no difference 6 / worse 6 (fold-pair t, seed agreement)
27c27
<   - K − best_of_all: better 5 / no difference 8 / worse 8 (fold-pair t, seed agreement)
---
>   - K − best_of_all: better 6 / no difference 8 / worse 7 (fold-pair t, seed agreement)
29c29
<   - K − strong (ADR 0028): better 5 / no difference 9 / worse 7 (fold-pair t, seed agreement)
---
>   - K − strong (ADR 0028): better 6 / no difference 9 / worse 6 (fold-pair t, seed agreement)
32,33c32,33
< - `pendigits_40nan`: K − tabpfn -0.0107 [-0.0137, -0.0078] better; tabpfn − strong -0.0599 [-0.0636, -0.0561] better; tabpfn − bar -0.0546 [-0.0563, -0.0530] better; tabpfn − tuned -0.0421 [-0.0434, -0.0407] better; K − best_of_all -0.0107 [-0.0137, -0.0078] better; K − strong_live -0.0706 [-0.0756, -0.0656] better; K − strong (ADR 0028) -0.0107 [-0.0137, -0.0078] better; best of all tabpfn; fit 9 s a fold
< - `pendigits_60nan`: K − tabpfn -0.0180 [-0.0202, -0.0158] better; tabpfn − strong -0.1035 [-0.1085, -0.0985] better; tabpfn − bar -0.0388 [-0.0403, -0.0373] better; tabpfn − tuned -0.0357 [-0.0371, -0.0343] better; K − best_of_all -0.0180 [-0.0202, -0.0158] better; K − strong_live -0.1215 [-0.1267, -0.1162] better; K − strong (ADR 0028) -0.0180 [-0.0202, -0.0158] better; best of all tabpfn; fit 5 s a fold
---
> - `pendigits_40nan`: K − tabpfn -0.0279 [-0.0311, -0.0248] better; tabpfn − strong -0.0599 [-0.0636, -0.0561] better; tabpfn − bar -0.0546 [-0.0563, -0.0530] better; tabpfn − tuned -0.0421 [-0.0434, -0.0407] better; K − best_of_all -0.0279 [-0.0311, -0.0248] better; K − strong_live -0.0878 [-0.0927, -0.0829] better; K − strong (ADR 0028) -0.0279 [-0.0311, -0.0248] better; best of all tabpfn; fit 9 s a fold
> - `pendigits_60nan`: K − tabpfn -0.0244 [-0.0265, -0.0223] better; tabpfn − strong -0.1035 [-0.1085, -0.0985] better; tabpfn − bar -0.0388 [-0.0403, -0.0373] better; tabpfn − tuned -0.0357 [-0.0371, -0.0343] better; K − best_of_all -0.0244 [-0.0265, -0.0223] better; K − strong_live -0.1279 [-0.1333, -0.1224] better; K − strong (ADR 0028) -0.0244 [-0.0265, -0.0223] better; best of all tabpfn; fit 5 s a fold
35c35
< - `letter_40nan`: K − tabpfn -0.0344 [-0.0369, -0.0319] better; tabpfn − strong -0.0210 [-0.0237, -0.0182] better; tabpfn − bar -0.0211 [-0.0227, -0.0196] better; tabpfn − tuned -0.0070 [-0.0089, -0.0050] better; K − best_of_all -0.0246 [-0.0264, -0.0227] better; K − strong_live -0.0554 [-0.0589, -0.0519] better; K − strong (ADR 0028) -0.0344 [-0.0369, -0.0319] better; best of all macode; fit 17 s a fold
---
> - `letter_40nan`: K − tabpfn -0.0431 [-0.0453, -0.0409] better; tabpfn − strong -0.0210 [-0.0237, -0.0182] better; tabpfn − bar -0.0211 [-0.0227, -0.0196] better; tabpfn − tuned -0.0070 [-0.0089, -0.0050] better; K − best_of_all -0.0332 [-0.0352, -0.0313] better; K − strong_live -0.0641 [-0.0671, -0.0610] better; K − strong (ADR 0028) -0.0431 [-0.0453, -0.0409] better; best of all macode; fit 17 s a fold
58,59c58,59
< - `pendigits_40nan`: K − tabpfn -0.0107 [-0.0137, -0.0078] better; KC − tabpfn -0.0119 [-0.0148, -0.0091] better; H − tabpfn -0.0096 [-0.0129, -0.0064] better
< - `pendigits_60nan`: K − tabpfn -0.0180 [-0.0202, -0.0158] better; KC − tabpfn -0.0216 [-0.0238, -0.0193] better; H − tabpfn -0.0143 [-0.0176, -0.0109] better
---
> - `pendigits_40nan`: K − tabpfn -0.0279 [-0.0311, -0.0248] better; KC − tabpfn -0.0294 [-0.0326, -0.0262] better; H − tabpfn -0.0258 [-0.0290, -0.0226] better
> - `pendigits_60nan`: K − tabpfn -0.0244 [-0.0265, -0.0223] better; KC − tabpfn -0.0286 [-0.0306, -0.0266] better; H − tabpfn -0.0194 [-0.0231, -0.0157] better
61c61
< - `letter_40nan`: K − tabpfn -0.0344 [-0.0369, -0.0319] better; KC − tabpfn -0.0358 [-0.0382, -0.0334] better; H − tabpfn -0.0334 [-0.0357, -0.0311] better
---
> - `letter_40nan`: K − tabpfn -0.0431 [-0.0453, -0.0409] better; KC − tabpfn -0.0449 [-0.0470, -0.0428] better; H − tabpfn -0.0415 [-0.0443, -0.0387] better
```
