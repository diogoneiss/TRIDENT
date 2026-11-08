# ReMasker off MCAR: the MAR and MNAR families (E160, T53; pre-registered 2026-11-07 21:46 GMT-3)

This ticket was written and committed before the study's first scored cell.

- **Why:** TRIDENT's standing against the strong tier off MCAR (E58 under MNAR, E64 under MAR) has no deep baseline
  beside it, and ReMasker's paper reports robustness across missingness mechanisms. E158 measured ReMasker under MCAR
  only.
- **The user's go:** "De acordo, va" (2026-11-07), on "B: ReMasker fora do MCAR, famílias MAR e MNAR no biodeg, kc2,
  vehicle", launched beside E159.
- **Disclosure:** the TRIDENT, strong-tier and TabPFN cells of both families are on the record (E45, E46, E58, E64).
  ReMasker has no score on any of these variants. Before launch, `recompute_folds` reproduced the runs' mean/mode
  scores exactly (max |diff| 0.0) on five of the variants (biodeg_marf_60nan s404, kc2_marfctl_20nan s505,
  vehicle_probit_60nan s42, biodeg_probit_20nan s7, kc2_60nan s13), so the rebuilt folds are the runs' own.

## Design

- **Arm:** ReMasker, `remask="paper"`, E158's settings, untuned.
- **MAR:** E64's cells (tag `mar-family-2026-10-22`, arm G, seeds 404/505/606, TabPFN merged):
  `{biodeg,kc2,vehicle}_{marf,marfctl}_{20,60}nan`, 12 variants, 36 pairs.
- **MNAR:** E45/E46's cells as E58 read them (arm G, seeds 42/7/13, TabPFN merged):
  `{biodeg,kc2,vehicle}_probit_{20,60}nan` and their MCAR controls `{biodeg,kc2,vehicle}_{20,60}nan`, 12 variants,
  36 pairs.
- 72 pairs, all on gorgona8 (beside E159's letter_40nan lanes), four lanes in two launches; outs
  `/scratch2/diogoneiss/trident-2026-11-07-remasker-offmcar/{mar,mnar}`. Expected end about 03:00–05:00 GMT-3 on
  2026-11-08.
- **Read:** once, with all 72 pairs in: `scripts/experiments/remasker_offmcar_report.py`. A failed pair is rerun before
  the read.

## Measures and rule (fixed now; H is the headline those studies read, K beside)

- **Per variant,** 1% SESOI: H − ReMasker, K − ReMasker, ReMasker − strong, ReMasker − bar.
- **Mechanism effect,** fold-paired (a seed builds the same folds on a variant and its control): (H − ReMasker),
  mechanism minus control, per table-rate.
- **Primary,** over the 12 table-rates: ReMasker gains ground on TRIDENT off MCAR, **supported** if the effect is
  worse for H on 4 or more and better on at most 1; **refuted** if worse on at most 1; **mixed** otherwise (E64's
  rule).

**What it leads to:** supported → the thesis's off-MCAR claims carry a stated limit against a deep baseline;
refuted → TRIDENT's lead over ReMasker does not shrink off MCAR, a sentence for the user. No default, bar or tier
change from this study.

## Limits, stated now

- Small numerical tables only (biodeg, kc2, vehicle); the large tables' off-MCAR cells (E119) are left out for cost.
- The two families use different seed sets and different study generations (arm G of their time), so they are read
  side by side, never pooled across families except in the primary's count.
