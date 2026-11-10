# Does the cross-fit pick remove the hybrid's weakness off MCAR? (E170, T54; pre-registered 2026-11-09 19:59 GMT-3)

This ticket was written and committed before E169's cells were read, and needs no run of its own.

- **Why:** ADR 0031 decision 6 stays MCAR-only for synthetic missingness because E93 found the accepted TabPFN-side
  hybrid (`table_tabpfn`) mixed off MCAR, worse than H on `spambase_probit_60nan` (+0.0090); E167 repeated that loss
  on fresh seeds (+0.0082). E168's cross-fit pick, which takes missForest's side when its held-out validation score is
  lower, read equivalent to H there (+0.0018). If it is never worse than H off MCAR, it is the form that could move
  decision 6's scope; another TabPFN-side run could not.
- **The user's go:** "Leitura sem GPU" (2026-11-09, just after 19:51 GMT-3), in place of the GPU replication first
  proposed, which E93 and E167 had already answered.
- **Disclosure:** E93's 12 off-MCAR variants split 6 in E167 (already read) and 6 in E169 (unread). Only E169's 6 are the
  test. On them, E168's development printout (seeds 4545/5656/6767) has `crossfit` better than H on 5 and no call on
  `biodeg_probit2p0_60nan` (+0.0052).

## Design

- **Variants (6), `hybrid_fresh_seeds.OFF_MCAR`:** `biodeg_probit1p0_60nan`, `biodeg_probit2p0_60nan` (self-masking
  MNAR); `biodeg_marf_60nan`, `vehicle_marf_60nan`, `spambase_marf_60nan`, `credit-g_marf_60nan` (MAR).
- **Cells:** E169's ([64](64-crossfit-generality.md)), seeds 42/7/13, all on medusa1. Nothing new runs.

## Measures and rule (fixed now)

Fold-pair verdict on 15 (seed, fold) differences, 1% SESOI of the opponent's mean.
- **Primary, `crossfit` against H on the 6:** **supported** if worse on none; **refuted** if worse on 2 or more;
  **mixed** if worse on 1.
- **Secondary:** `crossfit` better than H on which; `table_tabpfn` worse than H on which (does the accepted form's
  weakness show here too); `crossfit` worse than `table_tabpfn` on which. Descriptive: E167's other 6 off-MCAR
  variants, from its printout.
- **Read** once, in E169's printout: `uv run --python 3.11 python -B scripts/experiments/hybrid_fresh_seeds.py --study e169 --tally`
  on medusa1 (the E170 lines).
- **What it leads to:** with E169, whether an ADR 0031 amendment for the cross-fit pick can drop the MCAR-only scope for
  synthetic MAR and MNAR. The user's call.

## Amendment

- **Amendment (2026-11-09 23:20 GMT-3):** E169's offline pass moved to gorgona8 after medusa1's GPU 0 failed (see
  [64](64-crossfit-generality.md)), so this read runs there too, same command, with E169's. Nothing else changes.

## Outcome (read once 2026-11-10 12:41 GMT-3: mixed; written 2026-11-10 12:44 GMT-3)

Read on gorgona8 at commit c11458c, after all 78 pairs' cells were in (gorgona8's offline pass of the amendment above;
no FAILED line; the H-ledger and missForest asserts held on every pair).
The printout is in E169's Outcome ([64](64-crossfit-generality.md)); its E170 lines:

```
E170 PRIMARY `crossfit` on E93's 6 off-MCAR variants here: worse than H on ['biodeg_probit2p0_60nan'] -> mixed
E170 SECONDARY on them: `crossfit` better than H on ['biodeg_marf_60nan', 'vehicle_marf_60nan', 'spambase_marf_60nan', 'credit-g_marf_60nan']; `table_tabpfn` worse than H on ['biodeg_probit2p0_60nan']; `crossfit` worse than `table_tabpfn` on ['credit-g_marf_60nan']
```

- **Primary, mixed.** The cross-fit pick is worse than H on 1 of 6, `biodeg_probit2p0_60nan` (+0.0095
  [+0.0051, +0.0139]). `table_tabpfn` is worse than H there too (+0.0106); E172 found the same on seeds 101/202/303
  (+0.0066, both sides losing).
- **Secondary:** `crossfit` better than H on the four marf variants; `table_tabpfn` worse than H on
  `biodeg_probit2p0_60nan` only; `crossfit` worse than `table_tabpfn` on `credit-g_marf_60nan` (+0.0115).
  Descriptive, E167's other 6: `crossfit` worse than H on none there.
- **What it leads to:** the cross-fit pick does not clear the off-MCAR weakness: under the steepest self-masking every
  hybrid loses to H on two seed sets. ADR 0031's MCAR-only scope for synthetic MAR and MNAR stands unless the user
  decides otherwise.
