# C1 cut 1: gate G1, the indicators-only probe (E129; pre-registered 2026-11-04)

This ticket was written and committed before the probe's first fit (12:48 GMT-3).

- **Why:** the protocol's manipulation check ([plan](../../plans/2026-11-04-c1-split.md)): do the gaps of the
  primary MNAR files carry outcome information that their MCAR siblings' gaps do not? If not, a tie between S and T
  is inconclusive, not null (the gate conditions only the Null row; the user's decision 14a, 2026-11-04).
- **Code:** `scripts/experiments/c1_g1.py`, CPU only; `build_folds` at 5 on seeds 42/7/13, the decision mode's folds,
  shared by an MNAR file and its MCAR sibling. No `src/` change.

## Design and rule (fixed before the first fit)

- **Files:** `{biodeg,kc2,vehicle,spambase}_probit_60nan` against `{biodeg,kc2,vehicle,spambase}_60nan`.
- **Per (file, seed, fold):** a logistic regression on the 0/1 gap matrix of every feature column, fitted on the
  training rows, scored on the validation rows: AUROC (one-vs-rest macro on vehicle) and macro-F1. Test rows are
  never read.
- **Reading:** per table, the 15 paired differences MNAR − MCAR in AUROC with a plain 95% t interval; pooled, the
  mean over the four tables per replicate, its plain 95% t interval (14 df).
- **Passes if** the pooled interval lies above 0 and no table's mean difference is negative.
- **What is known before the reading:** E123's G2 (a GBDT indicator gain, another measure) found kc2's and
  spambase's probit_60nan files not informative for prediction, so those two are the likely failures here.

## Outcome (2026-11-04 12:50 GMT-3): G1 passes, 4 of 4 tables above their MCAR siblings

The unedited `c1_g1.py` at 03bdd3f (CPU, gorgona8, about a minute):

```
## G1 (indicators-only probe, validation rows, 15 folds a table): MNAR − MCAR AUROC pooled over the four tables, plain 95% t interval above 0 and no table negative
- biodeg: AUROC MNAR 0.648, MCAR 0.484, difference +0.1637 [+0.1364, +0.1910]; macro-F1 MNAR 0.584, MCAR 0.428
- kc2: AUROC MNAR 0.704, MCAR 0.596, difference +0.1087 [+0.0571, +0.1603]; macro-F1 MNAR 0.599, MCAR 0.458
- vehicle: AUROC MNAR 0.596, MCAR 0.495, difference +0.1019 [+0.0689, +0.1350]; macro-F1 MNAR 0.321, MCAR 0.254
- spambase: AUROC MNAR 0.630, MCAR 0.502, difference +0.1287 [+0.1103, +0.1470]; macro-F1 MNAR 0.570, MCAR 0.425

Pooled: +0.1257 [+0.1071, +0.1444]; tables with a negative mean: none
Primary (G1): passes
```

- Every table's MNAR − MCAR AUROC interval lies above 0 (+0.10 to +0.16); the MCAR probes sit at chance (0.48–0.60),
  the MNAR ones at 0.60–0.70. The primary population has an outcome channel for the gap pattern alone, kc2 and
  spambase included, where E123's GBDT gain (a different measure: indicators added to a missForest fill) had found
  no information for prediction. The gap pattern predicts the class there, but adds little once the filled values
  are in the model, which is the reading the Null row's gate needed either way.
- `g1.json` written for `c1_report.py --primary`.
