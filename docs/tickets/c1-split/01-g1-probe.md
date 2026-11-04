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
