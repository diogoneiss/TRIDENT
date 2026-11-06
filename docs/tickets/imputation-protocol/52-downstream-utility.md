# Which fills serve a downstream classifier: the checkpoint by `impute_score` or by `dist_score`? (E146, a pilot; pre-registered 2026-11-05)

**Kept on the record (2026-11-06, 06:47 GMT-3).** The user's decision ("quero 146 e 149 mantidos"): this pilot and the imputer
study that followed it ([53](53-imputer-utility.md)) join `feat/imputation-task`. It was pre-registered and read on
the local branch named below (commits 85d9ef0 at 21:10 and 1b8324d at 21:12 GMT-3 on 2026-11-05), deleted after this
merge. What follows is that branch's text; only the link to the script changed. The script is now
[scripts/experiments/downstream_utility.py](../../../scripts/experiments/downstream_utility.py), typed for the gate with its behaviour
unchanged: its tally reproduces the one below but for the word "pilot" in the heading, and a rerun of kc2_60nan seed
42 on gorgona8 equals medusa1's file to 1e-16. The per-fold CSVs are outside the repository, in
`trident-2026-11-05-downstream-utility/csv` under `/scratch/diogoneiss` on medusa1 and `/scratch2/diogoneiss` on
gorgona8. It stays a pilot: a claim needs a pre-registered rerun on fresh seeds.

This ticket was written and committed before the study's first fit (21:10 GMT-3), **on a local branch that is not
pushed** (`pilot/downstream-utility`).

- **Off the shared record, on the user's instruction** (2026-11-05: "Rode os experimentos mas não quero backfill dos
  resultados e findings, quero algo que não deixe rastros se ficar ruim demais"). Nothing of it is on
  `feat/imputation-task` or on GitHub: no store write, no tag, no backfill, no README, ADR, report or TASKS change.
  The CSVs sit in `trident-2026-11-05-downstream-utility` on medusa1. If the user keeps it, this branch is merged and
  pushed; if not, the branch and that directory are deleted. A pilot either way: a claim would need a pre-registered
  rerun on fresh seeds.
- **Why:** the user's question after E145 ([51](51-decode-metric.md)): use the imputations for classification, and see
  which score is the more useful one, `impute_score` or `dist_score`. It is the vault's open todo t4 (selection by
  RMSE/MAE against a downstream metric) and the impute-then-learn arm step zero and T40 left optional.
- **What it reuses:** E145's 72 cells on medusa1, whose five fold ledgers hold K's fills under both selectors on the
  same test cells. No GPU, no TRIDENT training: medusa1's GPUs are held by another user's job again.
- **Code:** [52-downstream-utility.py](../../../scripts/experiments/downstream_utility.py), kept beside this ticket and outside `scripts/`
  until the pilot is kept. Exercised once on gorgona8's E145 smoke pair (kc2_60nan, seed 42, not a study cell):
  every gap filled in all four tables; macro-F1 with logistic regression 0.699 on the truth, 0.700 with A's fills,
  0.704 with B's, 0.568 with the mean/mode fill. Those numbers were seen before this was written.

## Design and rule (fixed before the first fit)

- **Cells:** E145's 12 variants × seeds 42/7/13 × 5 folds, both arms (A: checkpoint by `impute_score`; B: by
  `dist_score`), K's uncalibrated induced fills.
- **Tables:** per (variant, seed), in the ledgers' scaled space, the variant's features with every gap filled four
  ways: the truth (the ledger's `actual`), arm A's fill, arm B's fill, the fold's mean/mode fill. Each row's gaps are
  filled by the fold in which that row is a test row, so no imputer trained on the row it fills, and no imputer ever
  saw a label. A table with a gap left is an error.
- **Classifier:** two fixed, untuned learners on the CPU, logistic regression and a small MLP (64 and 32 units, early
  stopping), categories one-hot, numbers as scaled. Macro-F1 on the imputation's own test folds.
- **Primary protocol, `complete`:** the classifier is fitted on the truth-filled rows outside the test fold and
  scored on the test fold under each fill. Chosen and stated now: a classifier fitted on mean/mode-filled rows
  expects collapsed inputs and would favour A's near-naive fills at 80nan by construction. **Beside, `own`:** fitted
  on a fill's own rows outside the test fold, scored on that fill's test fold (impute, then learn).
- **Primary reading:** F1(B) − F1(A), the mean of the two learners, pooled over the 12 variants per (seed, fold)
  replicate, 15 replicates, plain 95% t interval: above 0, B's fills are the more useful; below 0, A's; otherwise no
  difference. Each learner and the `own` protocol beside.
- **The scores' question:** per variant, the verdict of F1(B) − F1(A) by its fold-pair interval, beside the arm each
  score favoured (the sign of its own B − A), and the count of variants with a verdict where each score was on the
  right side; the replicate-level Spearman correlation of F1(B) − F1(A) with each score's B − A, bootstrapped over
  variants. Stated now: 12 variants cannot separate the two scores statistically. The reading is which one is on the
  right side, and where.
- **Descriptive:** the levels under the four fills and the share of the naive-to-truth gap each arm closes.
- **Limits stated now:** K uncalibrated, not the default output from 30% missing (KC); 12 small and medium tables at
  40nan and above, one seed set; the primary is a test-time-missingness reading, not a deployable pipeline (`own`
  is nearer one); two untuned learners. The literature the vault reviews (Le Morvan and Varoquaux; Shades of Null)
  finds imputation quality weakly coupled to downstream accuracy, so two nulls are the expected shape and would mean
  something.
- **What follows:** the user reads it and decides whether it is kept.

## Outcome (2026-11-05, 21:12 GMT-3; 36 of 36 (variant, seed) files, run on medusa1's CPUs at 21:11 GMT-3)

Read with `52-downstream-utility.py tally`. Unedited tally:

```text
## Downstream utility (E146, pilot): macro-F1 of a fixed classifier on E145's test folds under each fill of their gaps; B − A, 15 replicates a variant, pooled over the 12 variants per replicate; plain 95% t interval
Primary, classifier fitted on the complete rows, mean of the two learners, F1(B) − F1(A): +0.0141 [+0.0107, +0.0175] (NB [+0.0063, +0.0219]) -> B
  logistic: +0.0152 [+0.0114, +0.0189] (NB [+0.0065, +0.0238]) -> B
  mlp: +0.0131 [+0.0081, +0.0180] (NB [+0.0017, +0.0245]) -> B
Beside, classifier fitted on each fill's own table, mean of the two learners, F1(B) − F1(A): +0.0029 [-0.0011, +0.0069] (NB [-0.0063, +0.0121]) -> no difference
Levels (complete, mean of learners): truth 0.8208, A 0.6823, B 0.6964, naive 0.6166; share of the naive-to-truth gap closed: A +0.32, B +0.39
Levels (own): A 0.6792, B 0.6820, naive 0.6661

### Per variant (complete, mean of learners): which score pointed at the more useful fills
- kr-vs-kp_40nan: F1(B) − F1(A) -0.0005 [-0.0052, +0.0043] -> no difference; impute_score said A (+0.0029), dist_score said B (-0.0410); truth 0.975 A 0.821 B 0.821 naive 0.788
- kr-vs-kp_60nan: F1(B) − F1(A) -0.0015 [-0.0087, +0.0057] -> no difference; impute_score said A (+0.0299), dist_score said B (-0.1676); truth 0.975 A 0.701 B 0.700 naive 0.682
- kr-vs-kp_80nan: F1(B) − F1(A) +0.0050 [-0.0106, +0.0206] -> no difference; impute_score said A (+0.1581), dist_score said B (-0.6013); truth 0.975 A 0.570 B 0.575 naive 0.562
- spambase_40nan: F1(B) − F1(A) +0.0000 [-0.0017, +0.0018] -> no difference; impute_score said A (+0.0009), dist_score said B (-0.0258); truth 0.926 A 0.896 B 0.896 naive 0.847
- spambase_60nan: F1(B) − F1(A) +0.0102 [+0.0062, +0.0142] -> B; impute_score said A (+0.0017), dist_score said B (-0.0742); truth 0.929 A 0.850 B 0.861 naive 0.768
- spambase_80nan: F1(B) − F1(A) +0.0280 [+0.0186, +0.0375] -> B; impute_score said A (+0.0200), dist_score said B (-0.2027); truth 0.928 A 0.738 B 0.766 naive 0.608
- credit-g_40nan: F1(B) − F1(A) +0.0137 [-0.0003, +0.0278] -> no difference; impute_score said A (+0.0300), dist_score said B (-0.1525); truth 0.612 A 0.564 B 0.578 naive 0.534
- credit-g_60nan: F1(B) − F1(A) +0.0361 [+0.0095, +0.0627] -> B; impute_score said A (+0.0739), dist_score said B (-0.2892); truth 0.627 A 0.535 B 0.571 naive 0.501
- credit-g_80nan: F1(B) − F1(A) +0.0570 [+0.0364, +0.0777] -> B; impute_score said A (+0.1619), dist_score said B (-0.6398); truth 0.627 A 0.473 B 0.530 naive 0.458
- vehicle_60nan: F1(B) − F1(A) +0.0141 [+0.0041, +0.0241] -> B; impute_score said A (+0.0011), dist_score said B (-0.0227); truth 0.745 A 0.565 B 0.579 naive 0.413
- biodeg_60nan: F1(B) − F1(A) +0.0012 [-0.0029, +0.0053] -> no difference; impute_score said B (-0.0012), dist_score said B (-0.0069); truth 0.843 A 0.789 B 0.790 naive 0.666
- kc2_60nan: F1(B) − F1(A) +0.0058 [-0.0040, +0.0156] -> no difference; impute_score said B (-0.0022), dist_score said B (-0.0257); truth 0.687 A 0.684 B 0.690 naive 0.574
Variants with a verdict where each score was on the right side: {'impute_score': 0, 'dist_score': 5, 'neither': 0}

### Replicate-level Spearman of F1(B) − F1(A) with each score's own B − A, sign turned so that + means agreement
- impute_score: rho -0.233 [-0.509, +0.111] (bootstrap over variants)
- dist_score: rho +0.424 [+0.139, +0.685] (bootstrap over variants)
```

- **Primary, by the rule: B's fills are the more useful.** With the classifier fitted on complete rows, macro-F1 is
  +0.014 higher on the test folds filled by the `dist_score` checkpoint than by the `impute_score` one, for both
  learners. A's fills close 32% of the gap between the mean/mode fill and the truth, B's 39%.
- **Per variant:** B better on 5 (spambase at 60 and 80nan, credit-g at 60 and 80nan, vehicle_60nan), no difference on
  7, A better on none. On the 5 with a verdict `dist_score` favoured the more useful arm every time and
  `impute_score` never; the replicate-level correlation says the same, +0.42 [+0.14, +0.69] against −0.23 [−0.51,
  +0.11]. The largest gains are on the variants where E145 read the largest `impute_score` cost (credit-g_80nan
  +0.057, credit-g_60nan +0.036, spambase_80nan +0.028).
- **Beside, and the limit of it:** when the classifier is fitted on each fill's own table (impute, then learn), the
  difference is +0.003 [−0.001, +0.007], no difference, and the mean/mode fill rises from 0.617 to 0.666: a
  classifier that trains on the fills adapts to them, as the ticket stated before the run. kr-vs-kp shows nothing at
  any level; its fills barely beat the mean/mode fill (0.57 to 0.82 against 0.975 on the truth).
- **Standing:** a pilot on 12 variants and one seed set, K uncalibrated. It is not on the shared record; whether it is
  kept is the user's call, and a claim would need a pre-registered rerun on fresh seeds.

