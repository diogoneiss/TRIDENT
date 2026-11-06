# Which score orders imputers as a classifier does, across TRIDENT, the strong baselines and three mechanisms? (E147, a pilot; pre-registered 2026-11-05 as E149)

**Kept on the record, and renumbered (2026-11-06, 06:47 GMT-3).** The user's decision ("quero 146 e 149 mantidos (talvez reindexar o
149 pra ele ser 147)"): this pilot joins `feat/imputation-task` as **E147**, ticket 53. It was pre-registered, run
and read as E149, ticket 55, on the local branch named below (commits 8f009d4 at 22:19 and fbca27d at 22:40 GMT-3 on
2026-11-05, 0a1747a at 06:28 GMT-3 on 2026-11-06), deleted after this merge. E148 and E149 went to T50's MIMIC
experiments the same day, so "E149" below is always this pilot. What
follows is that branch's text; only the title, the links and one cross-reference changed. The unedited tallies keep
"E149" in their heading, and the old names (`55-imputer-utility.py`, `chain_e149.log`) are the ones a reader finds in
the state directories.

- **Scripts**, now in `scripts/experiments/`: [imputer_utility.py](../../../scripts/experiments/imputer_utility.py) (the fits and the tally),
  [imputer_utility_cells.py](../../../scripts/experiments/imputer_utility_cells.py) (the MAR and MNAR cells),
  [imputer_utility_levels.py](../../../scripts/experiments/imputer_utility_levels.py) (the post hoc reading) and the two chains,
  [run_imputer_utility_chain.sh](../../../scripts/experiments/run_imputer_utility_chain.sh) (superseded before any cell) and
  [run_imputer_utility_chain_gorgona8.sh](../../../scripts/experiments/run_imputer_utility_chain_gorgona8.sh) (the one that ran). Typed for
  the gate with their behaviour unchanged: the tally reproduces the one below but for its heading, the post hoc
  reading reproduces exactly, and reruns of kc2_probit_60nan seed 42 for the `trident` and `adr0007` groups are
  byte-identical to the study's files.
- **Data outside the repository:** the 396 per-fold CSVs in
  `/scratch2/diogoneiss/trident-2026-11-05-downstream-utility/csv_imputers` on gorgona8 (medusa1 holds its 180 MCAR
  files under `/scratch/diogoneiss`); the 60 MAR and MNAR cells' ledgers in
  `/scratch2/diogoneiss/trident-2026-11-05-mechanisms` on gorgona8.
- It stays a pilot: a claim needs a pre-registered rerun on fresh seeds.

This ticket was written and committed before the study's first fit and first cell (22:19 GMT-3), on the local branch
`pilot/downstream-utility`, **not pushed**, like E146.

- **Off the shared record still.** The user asked to expand E146 ("quero expandir ele! fazer mais experimentos com o
  dist score, por exemplo comparando trident contra baselines fortes e diferentes tipos de missingness. prepare um
  experimento disso para a madrugada"). That is read as interest, not yet as the verdict to keep the pilots, so the
  branch stays local: no store write, nothing on `feat/imputation-task` or GitHub. It merges in one command.
- **Why:** E146 ([52](52-downstream-utility.md)) compared two TRIDENT checkpoints and got a sign: `dist_score` favoured
  the more useful fills on 5 of 5 verdicts. With a dozen imputers a fold the same question becomes a rank
  correlation, and it can be asked under MAR and MNAR too.
- **Code**, beside this ticket: [55-imputer-utility.py](../../../scripts/experiments/imputer_utility.py) (the fits and the tally),
  [55-mechanism-cells.py](../../../scripts/experiments/imputer_utility_cells.py) (a wrapper around E145's runner for the MAR and MNAR cells; the
  pushed code is unchanged), [55-night-chain.sh](../../../scripts/experiments/run_imputer_utility_chain.sh). Exercised on gorgona8's E145 smoke pair
  (kc2_60nan, seed 42, not a study cell) for the `trident`, `adr0007`, `lgbm` and `tabpfn` groups: every gap filled,
  and TRIDENT's two scores equal E145's reader fold by fold. Numbers seen there before this was written:
  `impute_score` 0.61 (TRIDENT), 0.63 (TabPFN), 0.74 (knn), 0.84 (hgb), 0.85 (lgbm); `dist_score` 0.25, 0.24, 0.36 to
  0.39; one seed of the noisiest table.

## Design and rule (fixed before the first fit)

- **Variants, three mechanisms:** MCAR, E145's 12 (kr-vs-kp, spambase, credit-g at 40/60/80nan; vehicle, biodeg, kc2
  at 60nan), whose cells exist; MAR, `marf_60nan` on the six tables; MNAR, `probit_60nan` on the four numerical
  tables (spambase, vehicle, biodeg, kc2). Seeds 42/7/13, five folds.
- **New TRIDENT cells (the GPU tier):** the MAR and MNAR variants under E145's two arms (A, the checkpoint by
  `impute_score`; B, by `dist_score`; `--decode_calibration none`, MLflow off), 60 cells, every pair's two arms in
  one queue. On medusa1 if its GPUs, held by another user's job tonight, free up before 01:00 GMT-3; otherwise on
  gorgona8 from 01:00, the whole tier there (the user's "use a medusa o máximo possível", and the E144 precedent).
- **Imputers a fold:** TRIDENT A and B (from the ledgers); the mean/mode fill; knn5, knn10, hgb; lgbm; missForest;
  condexp, SoftImpute, ICE and the sequential missForest; TabPFN v2. Each baseline is refitted on the fold's training
  rows and asked for exactly the ledger's cells, in the scaled space, as `metric_panel.py` does. The CPU ones run on
  medusa1; TabPFN on gorgona8's GPU, a deterministic refit on the same folds and cells, so its machine is not E86's
  matter (that is about one model's trajectory on two machines). TabPFN enters a mechanism's reading only if it
  finished on every fold of it; its lane starts no job after 07:30 GMT-3.
- **Per (fold, imputer):** `impute_score`, `dist_score` with its two parts, the dispersion, and the macro-F1 of
  logistic regression and a small MLP (E146's two learners, their mean), fitted on the truth-filled rows outside the
  test fold and scored on the test fold as the imputer filled it (`complete`); fitted on the imputer's own filled
  rows beside (`own`).
- **Primary, one reading a mechanism, Holm over the mechanisms read complete:** within each fold, the Spearman
  correlation across the imputers between each score (sign turned) and the macro-F1; the difference
  rho(`dist_score`) − rho(`impute_score`), pooled over the mechanism's variants per (seed, fold) replicate, 15
  replicates, plain 95% t interval. Above 0: `dist_score` orders the imputers more like the classifier does; below
  0: `impute_score` does; otherwise no difference. MCAR is read when the CPU tier is complete, whatever happens to
  the GPU tier. MCAR at 60nan alone is printed beside, the level MAR and MNAR are read at.
- **Secondary:** the F1 lost by taking each score's best imputer, and how often that imputer is the classifier's
  best; TRIDENT A and B against the best strong baseline of each variant (missForest, condexp, SoftImpute, ICE,
  TabPFN where in) on `dist_score`, with `impute_score` and F1 beside; B against A a mechanism (E145's and E146's
  readings off MCAR); the levels of every imputer, `own` included.
- **Threats stated now:**
  - `dist_score` is marginal and rewards dispersion: arm B, or any fill with spread, can beat the conditional-mean
    imputers (missForest, condexp, TabPFN) on it while losing cell by cell. It is read beside `impute_score` and the
    F1, never alone.
  - F1 differences between imputers within a fold are small; hence the pooling, and the possibility of a null.
  - The fills are transductive on the features; no imputer saw a label. K uncalibrated. One seed set. MAR and MNAR at
    one level, 60nan; the MCAR group mixes 40, 60 and 80nan.
  - Under MNAR, filling a gap erases the pattern TRIDENT's own classifier reads through `[NULL]`; this study does
    not measure that.
  - The best strong baseline of a variant is chosen on the same folds it is then compared on, which favours it.
- **What follows:** the user reads it in the morning and decides whether the pilots are kept.

## Amendment (2026-11-05, 22:40 GMT-3)

On the user's decision ("rode tudo na gorgona mesmo", 22:38 GMT-3) the GPU tier does not wait for medusa1: the first
chain, still waiting for medusa1's GPUs, was stopped before any cell, and the 60 MAR and MNAR cells started on
gorgona8 at 22:39 GMT-3 on four queues (not the three written above), every pair whole on that machine. Their CPU
imputers, the downstream fits and TabPFN run on gorgona8 too
([55-night-chain-gorgona8.sh](../../../scripts/experiments/run_imputer_utility_chain_gorgona8.sh)). The CPU imputers of the 12 MCAR variants had already
finished on medusa1 (180 of 180 files, 0 failed, 22:21 to about 22:35 GMT-3) and are kept; baselines are
deterministic refits, so their machine is not a design matter. Variants, imputers, readings and rules are unchanged.

Two shell defects in the first chain were corrected before any cell ran, and the copy beside this ticket is the
corrected one: a stdin redirect that left `xargs` without its job list, and a remote process check that matched its
own shell.

## Outcome (2026-11-06, 06:27 GMT-3; 60 of 60 cells and 396 of 396 files, none failed)

Complete on every declared cell. The 60 MAR and MNAR cells ran on gorgona8 from 22:39 to 04:54 GMT-3; their CPU
imputers, downstream fits and TabPFN from 04:54 to 06:06 GMT-3; the MCAR TabPFN lane ended at 02:08 GMT-3, before the
07:30 limit, so TabPFN is in every mechanism's reading. Read with `55-imputer-utility.py tally`.

**A deviation, mine, found at the reading:** the design lists "the sequential missForest" among the imputers, but
`fit_linear_imputers` returns condexp, SoftImpute and ICE only, and no group fitted `missforest_seq`. Twelve imputers
a fold were read (TRIDENT A and B, the mean/mode fill, knn5, knn10, hgb, lgbm, missForest, condexp, SoftImpute, ICE,
TabPFN), not thirteen. Nothing was refitted after the numbers were seen.

**Verdict by the rule: mixed; the hypothesis is not supported as a general statement.**

- MCAR (12 variants at 40, 60 and 80nan): rho(`dist_score`) − rho(`impute_score`) +0.1543 [+0.0983, +0.2103], Holm
  0.0001: `dist_score` orders the imputers more like the classifier.
- MAR (6 variants at 60nan): −0.1195 [−0.1830, −0.0559], Holm 0.0012: `impute_score` does. The corrected interval
  printed beside covers 0.
- MNAR (4 variants at 60nan): −0.2206 [−0.2538, −0.1873], Holm 0.0000: `impute_score` does.
- Beside, as declared, MCAR at 60nan alone, the level MAR and MNAR are read at: +0.0475 [−0.0208, +0.1157], no
  difference.

**How to read it.** The MCAR verdict is carried by the missingness level, not by the mechanism: the like-for-like
comparison at 60nan shows no difference under MCAR and `impute_score` ahead under MAR and MNAR. The post hoc
breakdown below (descriptive, not part of the rule) puts numbers on it: inside MCAR the difference is −0.110 at
40nan, +0.048 at 60nan and +0.632 at 80nan, where `impute_score`'s own correlation with the F1 turns negative
(−0.18; −0.65 on credit-g_80nan), the regime in which E145 saw `impute_score` keep near-naive fills. `dist_score`'s
correlation is the steadier one (+0.45 to +0.53 in every group); `impute_score`'s is the higher one wherever the
level is moderate (+0.57 under MAR, +0.76 under MNAR, +0.60 at MCAR 40nan). The spread between tables is wider than
either pooled mean: from −0.88 (spambase_probit_60nan) to +1.33 (credit-g_80nan).

**Secondary readings.**

- *Taking each score's best imputer:* `impute_score` loses less F1 in every mechanism, MCAR included (mean F1 lost
  0.030 against 0.042 under MCAR; 0.016 against 0.025 under MAR; 0.014 against 0.032 under MNAR). Rank correlation and
  the top pick are different questions: under MCAR `dist_score` picks TRIDENT B on 91 of 180 folds and knn5 on 37
  (knn5 is worse than the mean/mode fill on `impute_score`, the dispersion reward stated as a threat), while the
  classifier's best is TabPFN on 55.
- *B against A (E145's and E146's readings off MCAR):* on `dist_score` B is better in all three mechanisms (+0.188,
  +0.053, +0.021); on F1 B is better under MCAR (+0.0141 [+0.0107, +0.0175]) and MNAR (+0.0050 [+0.0015, +0.0086], the
  corrected interval covering 0), and no different under MAR (+0.0020 [−0.0017, +0.0058]).
- *TRIDENT against the best strong baseline of each variant* (chosen on the folds it is compared on, which favours
  it): on `dist_score` A trails in all three mechanisms (−0.133, −0.072, −0.033) and B leads under MCAR only (+0.054;
  6 variants better, 2 no difference, 4 worse), trailing under MAR (−0.018) and MNAR (−0.012). On F1 A trails in all
  three (−0.015, −0.007, −0.011; better on no variant) and B ties under MCAR (−0.001; 2/7/3) and trails under MAR
  (−0.005) and MNAR (−0.006).
- *Levels:* the mean F1 of the fills is highest for TRIDENT B under MCAR (0.696, TabPFN 0.689, A 0.682), for TabPFN
  and B under MAR (0.743 and 0.743, A 0.741) and for missForest under MNAR (0.715, B 0.712, A 0.707, TabPFN 0.702);
  the truth gives 0.80 to 0.82 and the mean/mode fill 0.57 to 0.64.

**What it does not show.** One seed set; MAR and MNAR at one level, so the level effect seen inside MCAR is untested
there; K uncalibrated; two small learners, fitted on complete rows; the fills are transductive on the features. E146's
sign (two TRIDENT checkpoints, MCAR) stands, and it does not extend to a rule about imputers in general. Nothing here
argues for replacing `impute_score` as the reported score (ADR 0025 keeps it the only primary one); the 80nan
failure of its ordering is the one finding in `dist_score`'s favour, and it is post hoc.

Unedited tally:

```text
## Imputers, scores and downstream use (E149, pilot): per fold, every imputer's impute_score, dist_score and the macro-F1 of a classifier fitted on complete rows and scored on the test fold as that imputer filled it (mean of logistic regression and an MLP)

### MCAR: 12 variants, 180 folds, complete; imputers on every fold: condexp, hgb, ice, knn10, knn5, lgbm, missforest, naive, softimpute, tabpfn, trident_A, trident_B
- truth: impute_score 0.0000, dist_score 0.0000, dispersion nan, F1 0.8208, F1 own 0.8208
- trident_B: impute_score 0.8970, dist_score 0.3394, dispersion 0.654, F1 0.6964, F1 own 0.6820
- tabpfn: impute_score 0.8582, dist_score 0.5346, dispersion 0.388, F1 0.6891, F1 own 0.6922
- lgbm: impute_score 0.9380, dist_score 0.4351, dispersion 0.692, F1 0.6861, F1 own 0.6792
- hgb: impute_score 0.9356, dist_score 0.4341, dispersion 0.699, F1 0.6857, F1 own 0.6791
- trident_A: impute_score 0.8573, dist_score 0.5268, dispersion 0.507, F1 0.6823, F1 own 0.6792
- missforest: impute_score 0.9034, dist_score 0.4796, dispersion 0.676, F1 0.6817, F1 own 0.6755
- ice: impute_score 0.9399, dist_score 0.4661, dispersion 0.678, F1 0.6666, F1 own 0.6665
- condexp: impute_score 0.9389, dist_score 0.7411, dispersion 0.546, F1 0.6620, F1 own 0.6759
- knn10: impute_score 0.9575, dist_score 0.5693, dispersion 0.436, F1 0.6504, F1 own 0.6647
- softimpute: impute_score 1.0155, dist_score 0.7425, dispersion 0.831, F1 0.6495, F1 own 0.6656
- knn5: impute_score 1.0066, dist_score 0.4465, dispersion 0.566, F1 0.6466, F1 own 0.6513
- naive: impute_score 1.0000, dist_score 1.0000, dispersion 0.000, F1 0.6166, F1 own 0.6661
Mean Spearman across the 12 imputers of a fold, score against F1: impute_score +0.334, dist_score +0.489
Primary, rho(dist_score) − rho(impute_score): +0.1543 [+0.0983, +0.2103] (NB [+0.0255, +0.2831]) -> dist_score orders them more like the classifier
F1 lost by taking each score's best imputer, impute_score's loss − dist_score's: -0.0122 [-0.0171, -0.0072] (NB [-0.0236, -0.0007]) -> impute_score picks better
Mean F1 lost: by impute_score 0.0297, by dist_score 0.0419; folds where the score's best imputer is the classifier's best: impute_score 24/180, dist_score 28/180
TRIDENT B against A on dist_score (+ favours B): +0.1875 [+0.1808, +0.1941] (NB [+0.1723, +0.2026]) -> B
TRIDENT B against A on impute_score (+ favours B): -0.0397 [-0.0421, -0.0374] (NB [-0.0452, -0.0343]) -> A
TRIDENT B against A on f1 (+ favours B): +0.0141 [+0.0107, +0.0175] (NB [+0.0063, +0.0219]) -> B
Strong baselines in: missforest, condexp, softimpute, ice, tabpfn
trident_A against the best strong baseline of each variant on dist_score (+ favours TRIDENT): -0.1334 [-0.1408, -0.1260] (NB [-0.1504, -0.1164]) -> the baseline; variants better/no difference/worse: 1/2/9
trident_A against the best strong baseline of each variant on impute_score (+ favours TRIDENT): -0.0039 [-0.0105, +0.0027] (NB [-0.0190, +0.0113]) -> no difference; variants better/no difference/worse: 3/4/5
trident_A against the best strong baseline of each variant on f1 (+ favours TRIDENT): -0.0149 [-0.0192, -0.0107] (NB [-0.0246, -0.0052]) -> the baseline; variants better/no difference/worse: 0/7/5
trident_B against the best strong baseline of each variant on dist_score (+ favours TRIDENT): +0.0540 [+0.0498, +0.0583] (NB [+0.0442, +0.0638]) -> TRIDENT; variants better/no difference/worse: 6/2/4
trident_B against the best strong baseline of each variant on impute_score (+ favours TRIDENT): -0.0436 [-0.0500, -0.0372] (NB [-0.0584, -0.0289]) -> the baseline; variants better/no difference/worse: 2/3/7
trident_B against the best strong baseline of each variant on f1 (+ favours TRIDENT): -0.0008 [-0.0043, +0.0027] (NB [-0.0089, +0.0072]) -> no difference; variants better/no difference/worse: 2/7/3

### MCAR at 60nan: 6 variants, 90 folds, complete; imputers on every fold: condexp, hgb, ice, knn10, knn5, lgbm, missforest, naive, softimpute, tabpfn, trident_A, trident_B
- truth: impute_score 0.0000, dist_score 0.0000, dispersion nan, F1 0.8011, F1 own 0.8011
- trident_B: impute_score 0.8122, dist_score 0.3150, dispersion 0.718, F1 0.6985, F1 own 0.6878
- tabpfn: impute_score 0.7988, dist_score 0.4465, dispersion 0.517, F1 0.6907, F1 own 0.6941
- trident_A: impute_score 0.7950, dist_score 0.4127, dispersion 0.654, F1 0.6875, F1 own 0.6793
- missforest: impute_score 0.8368, dist_score 0.4067, dispersion 0.757, F1 0.6856, F1 own 0.6795
- hgb: impute_score 0.8863, dist_score 0.4053, dispersion 0.789, F1 0.6842, F1 own 0.6810
- lgbm: impute_score 0.8931, dist_score 0.4034, dispersion 0.790, F1 0.6815, F1 own 0.6783
- ice: impute_score 0.8766, dist_score 0.4261, dispersion 0.714, F1 0.6674, F1 own 0.6669
- condexp: impute_score 0.8859, dist_score 0.6118, dispersion 0.696, F1 0.6649, F1 own 0.6719
- softimpute: impute_score 0.9377, dist_score 0.5924, dispersion 0.871, F1 0.6570, F1 own 0.6663
- knn10: impute_score 0.9140, dist_score 0.5440, dispersion 0.458, F1 0.6499, F1 own 0.6662
- knn5: impute_score 0.9583, dist_score 0.4425, dispersion 0.574, F1 0.6425, F1 own 0.6575
- naive: impute_score 1.0000, dist_score 1.0000, dispersion 0.000, F1 0.6006, F1 own 0.6498
Mean Spearman across the 12 imputers of a fold, score against F1: impute_score +0.455, dist_score +0.503
Primary, rho(dist_score) − rho(impute_score): +0.0475 [-0.0208, +0.1157] (NB [-0.1095, +0.2044]) -> no difference
F1 lost by taking each score's best imputer, impute_score's loss − dist_score's: -0.0147 [-0.0202, -0.0091] (NB [-0.0273, -0.0020]) -> impute_score picks better
Mean F1 lost: by impute_score 0.0247, by dist_score 0.0393; folds where the score's best imputer is the classifier's best: impute_score 12/90, dist_score 18/90
TRIDENT B against A on dist_score (+ favours B): +0.0977 [+0.0864, +0.1090] (NB [+0.0718, +0.1237]) -> B
TRIDENT B against A on impute_score (+ favours B): -0.0172 [-0.0192, -0.0152] (NB [-0.0218, -0.0125]) -> A
TRIDENT B against A on f1 (+ favours B): +0.0110 [+0.0062, +0.0158] (NB [-0.0001, +0.0220]) -> B
Strong baselines in: missforest, condexp, softimpute, ice, tabpfn
trident_A against the best strong baseline of each variant on dist_score (+ favours TRIDENT): -0.0771 [-0.0896, -0.0645] (NB [-0.1059, -0.0482]) -> the baseline; variants better/no difference/worse: 1/1/4
trident_A against the best strong baseline of each variant on impute_score (+ favours TRIDENT): -0.0028 [-0.0154, +0.0097] (NB [-0.0316, +0.0260]) -> no difference; variants better/no difference/worse: 1/3/2
trident_A against the best strong baseline of each variant on f1 (+ favours TRIDENT): -0.0120 [-0.0183, -0.0056] (NB [-0.0266, +0.0027]) -> the baseline; variants better/no difference/worse: 0/4/2
trident_B against the best strong baseline of each variant on dist_score (+ favours TRIDENT): +0.0207 [+0.0155, +0.0259] (NB [+0.0088, +0.0326]) -> TRIDENT; variants better/no difference/worse: 3/1/2
trident_B against the best strong baseline of each variant on impute_score (+ favours TRIDENT): -0.0200 [-0.0319, -0.0081] (NB [-0.0474, +0.0074]) -> the baseline; variants better/no difference/worse: 1/3/2
trident_B against the best strong baseline of each variant on f1 (+ favours TRIDENT): -0.0010 [-0.0056, +0.0036] (NB [-0.0116, +0.0096]) -> no difference; variants better/no difference/worse: 1/3/2

### MAR: 6 variants, 90 folds, complete; imputers on every fold: condexp, hgb, ice, knn10, knn5, lgbm, missforest, naive, softimpute, tabpfn, trident_A, trident_B
- truth: impute_score 0.0000, dist_score 0.0000, dispersion nan, F1 0.8020, F1 own 0.8020
- tabpfn: impute_score 0.7291, dist_score 0.3664, dispersion 0.598, F1 0.7434, F1 own 0.7538
- trident_B: impute_score 0.7212, dist_score 0.3069, dispersion 0.748, F1 0.7428, F1 own 0.7393
- trident_A: impute_score 0.7154, dist_score 0.3601, dispersion 0.714, F1 0.7408, F1 own 0.7421
- hgb: impute_score 0.7609, dist_score 0.3211, dispersion 0.761, F1 0.7390, F1 own 0.7451
- lgbm: impute_score 0.7645, dist_score 0.3196, dispersion 0.761, F1 0.7384, F1 own 0.7433
- missforest: impute_score 0.7748, dist_score 0.3813, dispersion 0.695, F1 0.7344, F1 own 0.7419
- knn10: impute_score 0.7941, dist_score 0.4209, dispersion 0.556, F1 0.7316, F1 own 0.7403
- knn5: impute_score 0.7980, dist_score 0.3333, dispersion 0.649, F1 0.7294, F1 own 0.7387
- ice: impute_score 0.7998, dist_score 0.3420, dispersion 0.789, F1 0.7221, F1 own 0.7288
- softimpute: impute_score 0.9578, dist_score 0.6420, dispersion 0.931, F1 0.6931, F1 own 0.7221
- condexp: impute_score 0.9524, dist_score 0.6771, dispersion 0.823, F1 0.6852, F1 own 0.7217
- naive: impute_score 1.0000, dist_score 1.0000, dispersion 0.000, F1 0.6406, F1 own 0.7196
Mean Spearman across the 12 imputers of a fold, score against F1: impute_score +0.566, dist_score +0.446
Primary, rho(dist_score) − rho(impute_score): -0.1195 [-0.1830, -0.0559] (NB [-0.2655, +0.0266]) -> impute_score does
F1 lost by taking each score's best imputer, impute_score's loss − dist_score's: -0.0090 [-0.0145, -0.0035] (NB [-0.0216, +0.0036]) -> impute_score picks better
Mean F1 lost: by impute_score 0.0161, by dist_score 0.0250; folds where the score's best imputer is the classifier's best: impute_score 17/90, dist_score 10/90
TRIDENT B against A on dist_score (+ favours B): +0.0533 [+0.0429, +0.0637] (NB [+0.0294, +0.0771]) -> B
TRIDENT B against A on impute_score (+ favours B): -0.0058 [-0.0110, -0.0006] (NB [-0.0177, +0.0061]) -> A
TRIDENT B against A on f1 (+ favours B): +0.0020 [-0.0017, +0.0058] (NB [-0.0065, +0.0106]) -> no difference
Strong baselines in: missforest, condexp, softimpute, ice, tabpfn
trident_A against the best strong baseline of each variant on dist_score (+ favours TRIDENT): -0.0717 [-0.0838, -0.0595] (NB [-0.0996, -0.0437]) -> the baseline; variants better/no difference/worse: 0/0/6
trident_A against the best strong baseline of each variant on impute_score (+ favours TRIDENT): +0.0133 [+0.0008, +0.0257] (NB [-0.0153, +0.0419]) -> TRIDENT; variants better/no difference/worse: 2/2/2
trident_A against the best strong baseline of each variant on f1 (+ favours TRIDENT): -0.0066 [-0.0111, -0.0022] (NB [-0.0169, +0.0036]) -> the baseline; variants better/no difference/worse: 0/3/3
trident_B against the best strong baseline of each variant on dist_score (+ favours TRIDENT): -0.0184 [-0.0249, -0.0119] (NB [-0.0333, -0.0035]) -> the baseline; variants better/no difference/worse: 1/1/4
trident_B against the best strong baseline of each variant on impute_score (+ favours TRIDENT): +0.0075 [-0.0026, +0.0176] (NB [-0.0156, +0.0307]) -> no difference; variants better/no difference/worse: 3/1/2
trident_B against the best strong baseline of each variant on f1 (+ favours TRIDENT): -0.0046 [-0.0080, -0.0012] (NB [-0.0124, +0.0032]) -> the baseline; variants better/no difference/worse: 0/4/2

### MNAR: 4 variants, 60 folds, complete; imputers on every fold: condexp, hgb, ice, knn10, knn5, lgbm, missforest, naive, softimpute, tabpfn, trident_A, trident_B
- truth: impute_score 0.0000, dist_score 0.0000, dispersion nan, F1 0.8043, F1 own 0.8043
- missforest: impute_score 0.7973, dist_score 0.4309, dispersion 0.575, F1 0.7149, F1 own 0.7140
- trident_B: impute_score 0.7935, dist_score 0.4431, dispersion 0.589, F1 0.7124, F1 own 0.7142
- trident_A: impute_score 0.7929, dist_score 0.4638, dispersion 0.580, F1 0.7074, F1 own 0.7152
- tabpfn: impute_score 0.8223, dist_score 0.4785, dispersion 0.455, F1 0.7016, F1 own 0.7225
- lgbm: impute_score 0.8488, dist_score 0.5020, dispersion 0.557, F1 0.6904, F1 own 0.7044
- hgb: impute_score 0.8471, dist_score 0.4988, dispersion 0.552, F1 0.6889, F1 own 0.7050
- ice: impute_score 0.8565, dist_score 0.5538, dispersion 0.517, F1 0.6715, F1 own 0.6898
- condexp: impute_score 0.8587, dist_score 0.5676, dispersion 0.524, F1 0.6574, F1 own 0.6882
- softimpute: impute_score 0.9083, dist_score 0.4760, dispersion 0.772, F1 0.6547, F1 own 0.6791
- knn10: impute_score 0.9012, dist_score 0.6118, dispersion 0.366, F1 0.6198, F1 own 0.6755
- knn5: impute_score 0.9216, dist_score 0.5540, dispersion 0.445, F1 0.6145, F1 own 0.6559
- naive: impute_score 1.0000, dist_score 1.0000, dispersion 0.000, F1 0.5674, F1 own 0.6613
Mean Spearman across the 12 imputers of a fold, score against F1: impute_score +0.755, dist_score +0.534
Primary, rho(dist_score) − rho(impute_score): -0.2206 [-0.2538, -0.1873] (NB [-0.2970, -0.1441]) -> impute_score does
F1 lost by taking each score's best imputer, impute_score's loss − dist_score's: -0.0176 [-0.0271, -0.0081] (NB [-0.0394, +0.0042]) -> impute_score picks better
Mean F1 lost: by impute_score 0.0142, by dist_score 0.0318; folds where the score's best imputer is the classifier's best: impute_score 20/60, dist_score 8/60
TRIDENT B against A on dist_score (+ favours B): +0.0208 [+0.0151, +0.0264] (NB [+0.0078, +0.0338]) -> B
TRIDENT B against A on impute_score (+ favours B): -0.0006 [-0.0039, +0.0027] (NB [-0.0082, +0.0069]) -> no difference
TRIDENT B against A on f1 (+ favours B): +0.0050 [+0.0015, +0.0086] (NB [-0.0031, +0.0132]) -> B
Strong baselines in: missforest, condexp, softimpute, ice, tabpfn
trident_A against the best strong baseline of each variant on dist_score (+ favours TRIDENT): -0.0330 [-0.0412, -0.0247] (NB [-0.0519, -0.0141]) -> the baseline; variants better/no difference/worse: 1/1/2
trident_A against the best strong baseline of each variant on impute_score (+ favours TRIDENT): +0.0043 [-0.0045, +0.0131] (NB [-0.0159, +0.0245]) -> no difference; variants better/no difference/worse: 1/2/1
trident_A against the best strong baseline of each variant on f1 (+ favours TRIDENT): -0.0112 [-0.0147, -0.0076] (NB [-0.0193, -0.0031]) -> the baseline; variants better/no difference/worse: 0/3/1
trident_B against the best strong baseline of each variant on dist_score (+ favours TRIDENT): -0.0122 [-0.0186, -0.0058] (NB [-0.0268, +0.0024]) -> the baseline; variants better/no difference/worse: 2/1/1
trident_B against the best strong baseline of each variant on impute_score (+ favours TRIDENT): +0.0037 [-0.0048, +0.0122] (NB [-0.0158, +0.0232]) -> no difference; variants better/no difference/worse: 1/2/1
trident_B against the best strong baseline of each variant on f1 (+ favours TRIDENT): -0.0061 [-0.0111, -0.0012] (NB [-0.0175, +0.0053]) -> the baseline; variants better/no difference/worse: 0/2/2

### The primary over the mechanisms read complete, Holm over them
- MCAR: rho(dist_score) − rho(impute_score) +0.1543, p 0.0000, Holm 0.0001 -> dist_score
- MAR: rho(dist_score) − rho(impute_score) -0.1195, p 0.0012, Holm 0.0012 -> impute_score
- MNAR: rho(dist_score) − rho(impute_score) -0.2206, p 0.0000, Holm 0.0000 -> impute_score
```

Post hoc, descriptive, not part of the rule ([55-levels-posthoc.py](../../../scripts/experiments/imputer_utility_levels.py)): the primary by level,
the per-variant mean correlations, and the imputer each score picks. Unedited:

```text
MAR 60nan: 6 variants, rho_impute +0.566, rho_dist +0.446, diff -0.1195 [-0.1830, -0.0559]
MCAR 40nan: 3 variants, rho_impute +0.603, rho_dist +0.493, diff -0.1098 [-0.1606, -0.0591]
MCAR 60nan: 6 variants, rho_impute +0.455, rho_dist +0.503, diff +0.0475 [-0.0208, +0.1157]
MCAR 80nan: 3 variants, rho_impute -0.176, rho_dist +0.456, diff +0.6322 [+0.5171, +0.7472]
MNAR 60nan: 4 variants, rho_impute +0.755, rho_dist +0.534, diff -0.2206 [-0.2538, -0.1873]

per variant (MCAR): diff of rho, mean over 15 folds
                            rho_i  rho_d      d
mech variant                                   
MAR  biodeg_marf_60nan      0.671  0.668 -0.003
     credit-g_marf_60nan    0.048 -0.016 -0.064
     kc2_marf_60nan         0.460  0.459 -0.001
     kr-vs-kp_marf_60nan    0.628  0.389 -0.239
     spambase_marf_60nan    0.828  0.467 -0.361
     vehicle_marf_60nan     0.759  0.710 -0.049
MCAR biodeg_60nan           0.853  0.842 -0.011
     credit-g_40nan         0.141  0.523  0.382
     credit-g_60nan        -0.499  0.622  1.121
     credit-g_80nan        -0.654  0.677  1.330
     kc2_60nan              0.362  0.488  0.126
     kr-vs-kp_40nan         0.835  0.692 -0.143
     kr-vs-kp_60nan         0.476  0.407 -0.070
     kr-vs-kp_80nan        -0.328  0.466  0.794
     spambase_40nan         0.832  0.264 -0.568
     spambase_60nan         0.782  0.124 -0.658
     spambase_80nan         0.452  0.225 -0.228
     vehicle_60nan          0.759  0.534 -0.224
MNAR biodeg_probit_60nan    0.757  0.768  0.011
     kc2_probit_60nan       0.694  0.740  0.046
     spambase_probit_60nan  0.704 -0.172 -0.876
     vehicle_probit_60nan   0.864  0.800 -0.063

MAR: picked by impute_score {'tabpfn': 44, 'trident_A': 23, 'trident_B': 20, 'ice': 2, 'hgb': 1}
   picked by dist_score {'tabpfn': 24, 'lgbm': 16, 'knn5': 15, 'trident_B': 14, 'ice': 13, 'hgb': 8}
   classifier's best {'tabpfn': 24, 'lgbm': 12, 'hgb': 12, 'trident_A': 8, 'missforest': 8, 'trident_B': 7, 'knn10': 6, 'knn5': 5, 'ice': 3, 'condexp': 3, 'softimpute': 2}

MCAR: picked by impute_score {'tabpfn': 92, 'trident_A': 48, 'trident_B': 22, 'condexp': 8, 'missforest': 7, 'naive': 3}
   picked by dist_score {'trident_B': 91, 'knn5': 37, 'ice': 16, 'softimpute': 13, 'tabpfn': 9, 'missforest': 8, 'trident_A': 4, 'lgbm': 2}
   classifier's best {'tabpfn': 55, 'trident_B': 32, 'trident_A': 19, 'missforest': 16, 'lgbm': 16, 'hgb': 14, 'knn10': 7, 'ice': 6, 'knn5': 6, 'softimpute': 6, 'condexp': 3}

MNAR: picked by impute_score {'trident_A': 25, 'missforest': 21, 'trident_B': 9, 'tabpfn': 3, 'ice': 1, 'condexp': 1}
   picked by dist_score {'missforest': 22, 'trident_B': 18, 'trident_A': 9, 'knn5': 8, 'tabpfn': 2, 'softimpute': 1}
   classifier's best {'missforest': 18, 'trident_A': 12, 'trident_B': 10, 'tabpfn': 9, 'lgbm': 6, 'hgb': 3, 'condexp': 2}
```
