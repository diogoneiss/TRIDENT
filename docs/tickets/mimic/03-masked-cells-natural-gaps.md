# Observed cells hidden on tables with natural gaps (E148; pre-registered 2026-11-05 as E150)

Written and committed before either table exists; blocked on [01](01-external-datasets-root.md) and
[02](02-tables.md). The first of three ways to study tables whose gaps have no ground truth (the user's agreement,
2026-11-05); the others are [04](04-pattern-transplant.md) and [05](05-downstream-mortality.md).

- **Why:** every table in the project so far is complete, with gaps cut by a mechanism the project wrote. MIMIC-IV's
  gaps are a clinician's decision not to measure, and no complete sibling exists. The pipeline already has a
  population that needs none: observed cells hidden for scoring (`impute/masked/*`), with the row's natural gaps
  left as `[NULL]` inputs. The masks are drawn on observed cells only (`draw_dynamic_mask`).
- **Hypothesis:** on a clinical table with natural gaps, the decoder's fills of hidden observed cells are better
  than the best baseline of the bar's.

## Design and rule (fixed before the first cell)

- **Tables:** `mimic-cardio_natural` and `mimic-sofa_natural`, 11,082 rows each.
- **Arm:** the imputation defaults with `--decode_checkpoint masked`. The default, `induced` (ADR 0015), needs a
  complete sibling and falls back to the epoch of the lowest loss without one (`decoding.py`, `kept_criterion`);
  `masked` keeps the epoch whose validation mask scores best, the same selection on the population this study
  scores. The baselines of the bar are scored on the same cells by the run itself. The cells also carry
  `--export_gap_fills` (ticket 01): the fills of the natural gaps, which [05](05-downstream-mortality.md) reads.
  It changes no score.
- **Cells:** 2 tables × seeds 42/7/13, five folds each: 6 runs.
- **Primary:** per table, the masked population's `impute_score` minus the best baseline's on the same cells, paired
  per (seed, fold), 15 replicates, plain 95% t interval; Holm over the two tables. **Better** with the upper bound
  below 0, **worse** with the lower bound above 0, otherwise no different.
- **Beside it, descriptive:** the per-column scores of the sparse columns (lactate, troponin and the arterial line;
  bilirubin, PaO2/FiO2 and urine output); the score by a row's count of natural gaps; TabPFN (ADR 0028's tier) if
  the strong-tier script scores this population.
- **What this cannot say:** the hidden cells are a uniform draw among values that were measured. An unmeasured
  lactate is not a measured one (deaths 7.9% against 19.9%), so nothing here says how well the true gaps are filled.
- **Cost:** estimated from the store on 2026-11-06, before the first cell. A default imputation run of the same
  configuration took about 40 minutes on gorgona8 on `pendigits` (10,992 rows, 16 columns; means of 38 to 43
  minutes by level, 75 runs) and about 60 on `letter` (20,000 rows). These cells share the GPU with another
  project's trainers (see Run), so one to two hours a cell was expected. Measured on the first two cells
  (2026-11-06): 33 minutes on the SOFA table and 36 on the cardio one, two cells at a time beside four of the
  other project's trainers.
- **What follows:** no default change and no ADR either way. A first reading on clinical data; a claim needs fresh
  seeds.
- **Read only with all 6 runs done.**

## Run

- **Started 2026-11-06 07:20 GMT-3**, on the user's instruction ("Comece a rodar em paralelo com o kg impute"): the
  cells share gorgona8's GPU with kg-impute's training queue instead of waiting for a free one, as ticket 01's
  item (c) had asked. Nothing in the design changes; only how long a cell takes.
- Runner `scripts/experiments/mimic_run.py` through `c15_queue.sh`, two queues (one table each); state directory
  `/scratch2/diogoneiss/trident-2026-11-06-mimic-e148`. Tracked in MLflow on the leader (medusa1): tag
  `experiment = mimic-natural-gaps-2026-11-06`, arm `M`.
- Read by `scripts/experiments/mimic_report.py e148`, from the state directory's fold tables.

## Outcome (2026-11-06 10:27 GMT-3)

**Supported on both tables.** All 6 runs done, none failed, each cell on its first attempt (the last ended
2026-11-06 10:14 GMT-3); six parent runs FINISHED on the leader. The report, as `mimic_report.py e148` prints it:

```
mimic-cardio_natural: decoder 0.8072, bar 0.8199 (best baseline by run: ['hgb'])
  mean impute_score over 15 folds: decoder 0.8072, ice 0.8191, hgb 0.8199, lgbm 0.8245, missforest 0.8303, condexp 0.8347, knn10 0.8845, softimpute 0.8967, knn5 0.9183, mean_mode 1.0000
mimic-sofa_natural: decoder 0.8848, bar 0.8984 (best baseline by run: ['hgb'])
  mean impute_score over 15 folds: decoder 0.8848, hgb 0.8984, lgbm 0.9083, ice 0.9246, condexp 0.9348, missforest 0.9391, softimpute 0.9706, knn10 0.9802, mean_mode 1.0000, knn5 1.0250
PRIMARY mimic-cardio_natural (n=15): -0.0127 [-0.0177, -0.0076], Holm p 0.0002 -> better
PRIMARY mimic-sofa_natural (n=15): -0.0137 [-0.0202, -0.0072], Holm p 0.0005 -> better
TabPFN 0.8459; mimic-cardio_natural, decoder - TabPFN (n=15): -0.0387 [-0.0443, -0.0331]
  decoder against TabPFN: rmse_num_z 0.8116 against 0.8520, mae_num_z 0.4513 against 0.4511, acc_cat 0.5824 against 0.5685
TabPFN 0.8956; mimic-sofa_natural, decoder - TabPFN (n=15): -0.0108 [-0.0165, -0.0052]
  decoder against TabPFN: rmse_num_z 0.8855 against 0.8999, mae_num_z 0.4925 against 0.4665, acc_cat 0.6099 against 0.6117

mimic-cardio_natural, sparse columns on the masked cells, RMSE in z units (mean over 15 folds):
column      abp_diastolic  abp_mean  abp_systolic  lactate  troponin_t
imputer                                                               
condexp            0.5786    0.5903        0.7130   0.8879      1.0375
decoder            0.5087    0.5964        0.6252   0.8350      1.0162
hgb                0.5279    0.6073        0.6498   0.8752      1.0654
ice                0.5127    0.5723        0.6187   0.8740      1.0067
knn10              0.6642    0.7658        0.7526   0.9076      1.0579
knn5               0.6797    0.8140        0.7765   0.9311      1.0966
lgbm               0.5327    0.6183        0.6518   0.8627      1.0636
mean_mode          1.0096    1.0033        0.9652   1.0298      1.0439
missforest         0.5390    0.6073        0.6539   0.8557      1.0298
softimpute         0.6833    0.7057        0.7508   0.9284      1.3004

mimic-cardio_natural, the decoder's numerical cells by the row's count of natural gaps (15 folds pooled):
      cells  rmse_z  mean_fill_rmse_z   ratio
gaps                                         
0     15680  1.1462            1.3533  0.8469
1     20814  0.7596            0.9987  0.7606
2      6822  0.7436            0.9590  0.7754
3     16148  0.8462            1.0178  0.8314
4+    40946  0.6712            0.8874  0.7563

mimic-sofa_natural, sparse columns on the masked cells, RMSE in z units (mean over 15 folds):
column      bilirubin_max  pf_ratio  uo_24hr
imputer                                     
condexp            0.9599    0.9487   0.9344
decoder            0.9280    0.8197   0.9085
hgb                0.9599    0.7974   0.9284
ice                0.9827    0.9436   0.9388
knn10              0.9862    0.9200   0.9598
knn5               1.0322    0.9519   0.9987
lgbm               0.9599    0.7970   0.9262
mean_mode          0.9950    0.9907   1.0068
missforest         0.9509    0.8223   0.9435
softimpute         1.1465    1.1440   1.0405

mimic-sofa_natural, the decoder's numerical cells by the row's count of natural gaps (15 folds pooled):
      cells  rmse_z  mean_fill_rmse_z   ratio
gaps                                         
0     18244  1.0504            1.1840  0.8872
1     16938  0.9148            1.0423  0.8776
2     17074  0.7800            0.8884  0.8779
3     16951  0.7873            0.9069  0.8682
4+     7179  0.8350            0.9582  0.8714
```

- **Primary.** The decoder's `impute_score` on the hidden observed cells is below the bar's best baseline (`hgb` in
  every run) on both tables: −0.0127 [−0.0177, −0.0076] on the cardio table and −0.0137 [−0.0202, −0.0072] on the
  SOFA one. The rule as applied: both intervals end below 0, and both Holm-adjusted p are under 0.05 (0.0002 and
  0.0005). The difference is about 1.5% of the bar's score on each table.
- **The strong tier, descriptive** (ADR 0021's four, scored by the runs on the same cells; not part of the rule, no
  test): the decoder has the lowest mean `impute_score` of every imputer on both tables. ICE comes within 0.012 on
  the cardio table (0.8191 against 0.8072) and has the lower RMSE on three of its five sparse columns (mean and
  systolic arterial pressure, troponin).
- **Sparse columns, descriptive.** Against the bar's baselines the decoder has the lowest RMSE on seven of the eight
  (not on PaO2/FiO2, where `lgbm` and `hgb` lead, 0.797 against 0.820). Troponin is close to unpredictable for every
  imputer: the best RMSE is 1.007 z (ICE) and the mean fill's is 1.044.
- **By a row's count of natural gaps, descriptive.** The form was chosen after the primary was read: numerical cells
  only, rows grouped at 0 to 3 and 4 or more gaps, the decoder's RMSE beside the training-mean fill's on the same
  cells (the ledger holds the decoder's fills alone). No trend with the count: the ratio stays between 0.76 and 0.85
  on the cardio table and between 0.87 and 0.89 on the SOFA one. Rows with every value have the largest error in z
  units on both tables, and the largest mean-fill error too. Confounded by composition: a row with more gaps has
  fewer and different columns left to hide.
- **TabPFN, descriptive** (added 2026-11-06 11:43 GMT-3; the block above was printed again with it). The offline
  pass (`tabpfn_run.py`, ADR 0028's settings) scores this population, so it ran on both tables, 2026-11-06 10:17 to
  11:41 GMT-3, 30 folds in `<state>/tabpfn/`. Its hidden cells are the runs': the rebuilt folds hold the ledgers'
  cell counts in the ten folds checked. The decoder's `impute_score` is below TabPFN's on both tables: −0.0387
  [−0.0443, −0.0331] on the cardio table (TabPFN 0.8459, above `hgb` too) and −0.0108 [−0.0165, −0.0052] on the SOFA
  one (0.8956); no test in the rule. The lead is in squared error, not in absolute error: TabPFN's RMSE is higher
  on both (0.852 against 0.812, 0.900 against 0.886), while its MAE is level on the cardio table (0.451 both) and
  lower on the SOFA one (0.467 against 0.493). On the project's benchmark tables TabPFN leads the decoder on the
  induced cells (ADR 0028); this is the masked population of one cohort, and no claim against TabPFN follows.
- **What this cannot say** stands as written above: the hidden cells are a draw among measured values, so this is
  not a reading of how well the true gaps are filled. [05](05-downstream-mortality.md) is the only reading of what
  is done with the true gaps.
- **Cost, measured:** 33 and 36 minutes for the first two cells, 60 to 72 for the other four, which ran beside ten
  of kg-impute's trainers; 2 h 53 min of wall time for the study.
- No default change and no ADR, as fixed above. One seed set on two views of one cohort.
