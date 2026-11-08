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

## Outcome (2026-11-08 02:38 GMT-3)

All 72 pairs finished without a failure (MAR 36, MNAR 36; last at 02:07 GMT-3 on 2026-11-08). Read once.

- **Primary: refuted.** (H − ReMasker), mechanism minus control, is worse for H on 1 of 12 table-rates
  (biodeg probit 60nan, +0.059) and better on 5; ReMasker does not gain ground on TRIDENT off MCAR.
- **MAR:** TRIDENT's lead grows under the mechanism on 4 of 6 table-rates (biodeg 20/60, vehicle 20/60; −0.07 to
  −0.22), none worse. H and K better than ReMasker on 6/6 MAR variants; ReMasker worse than the strong tier on 5/6
  (one no difference).
- **MNAR (probit):** the effect is no difference on 4, better on 1 (vehicle 60nan), worse on 1 (biodeg 60nan). H better
  than ReMasker on 5/6 probit variants (kc2_probit_60nan no difference); ReMasker worse than the strong tier on 5/6.
- **What it leads to (fixed in the design):** a sentence for the user; no default, bar or tier change. Draft, theirs to
  confirm: "Fora do MCAR, a vantagem do TRIDENT sobre o ReMasker não encolhe: sob MAR ela cresce em 4 das 6
  combinações de tabela e nível, e sob MNAR fica estável".
- Stated limits as pre-registered: small tables only; the families are read side by side.

### Printout (unedited)

```
## MAR (E64's cells, 404/505/606)
- `biodeg_marf_20nan`: H − remasker -0.4641 better; K − remasker -0.4548 better; remasker − strong +0.4420 worse; remasker − bar +0.4516 worse
- `biodeg_marfctl_20nan`: H − remasker -0.3640 better; K − remasker -0.3634 better; remasker − strong +0.3430 worse; remasker − bar +0.3423 worse
  - (H − remasker), mechanism minus control: -0.1002 [-0.1602, -0.0401] better
- `biodeg_marf_60nan`: H − remasker -0.4075 better; K − remasker -0.4236 better; remasker − strong +0.4033 worse; remasker − bar +0.3977 worse
- `biodeg_marfctl_60nan`: H − remasker -0.1914 better; K − remasker -0.1995 better; remasker − strong +0.1582 worse; remasker − bar +0.1419 worse
  - (H − remasker), mechanism minus control: -0.2160 [-0.2724, -0.1597] better
- `kc2_marf_20nan`: H − remasker -0.1713 better; K − remasker -0.2305 better; remasker − strong +0.2602 worse; remasker − bar +0.0592 none
- `kc2_marfctl_20nan`: H − remasker -0.1198 none; K − remasker -0.1319 better; remasker − strong +0.2414 worse; remasker − bar +0.1317 worse
  - (H − remasker), mechanism minus control: -0.0515 [-0.1515, +0.0485] none
- `kc2_marf_60nan`: H − remasker -0.0662 better; K − remasker -0.0875 better; remasker − strong +0.0162 none; remasker − bar -0.0226 none
- `kc2_marfctl_60nan`: H − remasker -0.1192 better; K − remasker -0.1399 better; remasker − strong +0.0770 worse; remasker − bar -0.0157 none
  - (H − remasker), mechanism minus control: +0.0530 [-0.0143, +0.1203] none
- `vehicle_marf_20nan`: H − remasker -0.2266 better; K − remasker -0.2279 better; remasker − strong +0.2345 worse; remasker − bar +0.1829 worse
- `vehicle_marfctl_20nan`: H − remasker -0.1597 better; K − remasker -0.1647 better; remasker − strong +0.1702 worse; remasker − bar +0.1080 worse
  - (H − remasker), mechanism minus control: -0.0669 [-0.1039, -0.0298] better
- `vehicle_marf_60nan`: H − remasker -0.2720 better; K − remasker -0.2775 better; remasker − strong +0.2929 worse; remasker − bar +0.2426 worse
- `vehicle_marfctl_60nan`: H − remasker -0.0736 better; K − remasker -0.0775 better; remasker − strong +0.0971 worse; remasker − bar +0.0183 none
  - (H − remasker), mechanism minus control: -0.1984 [-0.2828, -0.1141] better
  - mechanism: H − remasker: better 6 / no difference 0 / worse 0 (fold-pair t, seed agreement)
  - mechanism: K − remasker: better 6 / no difference 0 / worse 0 (fold-pair t, seed agreement)
  - mechanism: remasker − strong: better 0 / no difference 1 / worse 5 (fold-pair t, seed agreement)
  - mechanism: remasker − bar: better 0 / no difference 2 / worse 4 (fold-pair t, seed agreement)
  - control: H − remasker: better 5 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - control: K − remasker: better 6 / no difference 0 / worse 0 (fold-pair t, seed agreement)
  - control: remasker − strong: better 0 / no difference 0 / worse 6 (fold-pair t, seed agreement)
  - control: remasker − bar: better 0 / no difference 2 / worse 4 (fold-pair t, seed agreement)
  - (H − remasker), mechanism minus control: better 4 / no difference 2 / worse 0 (fold-pair t, seed agreement)

## MNAR (E45/E46's cells, 42/7/13)
- `biodeg_probit_20nan`: H − remasker -0.2344 better; K − remasker -0.2369 better; remasker − strong +0.2084 worse; remasker − bar +0.1993 worse
- `biodeg_20nan`: H − remasker -0.2657 better; K − remasker -0.2746 better; remasker − strong +0.2471 worse; remasker − bar +0.2278 worse
  - (H − remasker), mechanism minus control: +0.0313 [-0.0058, +0.0684] none
- `biodeg_probit_60nan`: H − remasker -0.0829 better; K − remasker -0.0846 better; remasker − strong +0.1033 worse; remasker − bar +0.0665 worse
- `biodeg_60nan`: H − remasker -0.1423 better; K − remasker -0.1502 better; remasker − strong +0.1482 worse; remasker − bar +0.0449 worse
  - (H − remasker), mechanism minus control: +0.0593 [+0.0449, +0.0738] worse
- `kc2_probit_20nan`: H − remasker -0.1161 better; K − remasker -0.1167 better; remasker − strong +0.1048 worse; remasker − bar +0.0221 none
- `kc2_20nan`: H − remasker -0.1545 better; K − remasker -0.1905 better; remasker − strong +0.2075 worse; remasker − bar +0.1654 worse
  - (H − remasker), mechanism minus control: +0.0384 [-0.0670, +0.1438] none
- `kc2_probit_60nan`: H − remasker -0.0006 none; K − remasker -0.0140 none; remasker − strong -0.0101 none; remasker − bar -0.1153 better
- `kc2_60nan`: H − remasker -0.0560 none; K − remasker -0.1005 better; remasker − strong +0.0621 worse; remasker − bar -0.0401 none
  - (H − remasker), mechanism minus control: +0.0554 [-0.0079, +0.1187] none
- `vehicle_probit_20nan`: H − remasker -0.1584 better; K − remasker -0.1539 better; remasker − strong +0.1329 worse; remasker − bar +0.0819 worse
- `vehicle_20nan`: H − remasker -0.1297 better; K − remasker -0.1337 better; remasker − strong +0.1315 worse; remasker − bar +0.0803 worse
  - (H − remasker), mechanism minus control: -0.0287 [-0.0642, +0.0069] none
- `vehicle_probit_60nan`: H − remasker -0.0704 better; K − remasker -0.0734 better; remasker − strong +0.0755 worse; remasker − bar +0.0302 worse
- `vehicle_60nan`: H − remasker -0.0353 better; K − remasker -0.0362 better; remasker − strong +0.0460 worse; remasker − bar -0.0190 none
  - (H − remasker), mechanism minus control: -0.0352 [-0.0628, -0.0076] better
  - mechanism: H − remasker: better 5 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - mechanism: K − remasker: better 5 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - mechanism: remasker − strong: better 0 / no difference 1 / worse 5 (fold-pair t, seed agreement)
  - mechanism: remasker − bar: better 1 / no difference 1 / worse 4 (fold-pair t, seed agreement)
  - control: H − remasker: better 5 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - control: K − remasker: better 6 / no difference 0 / worse 0 (fold-pair t, seed agreement)
  - control: remasker − strong: better 0 / no difference 0 / worse 6 (fold-pair t, seed agreement)
  - control: remasker − bar: better 0 / no difference 2 / worse 4 (fold-pair t, seed agreement)
  - (H − remasker), mechanism minus control: better 1 / no difference 4 / worse 1 (fold-pair t, seed agreement)

Primary, (H − ReMasker) mechanism minus control over 12 table-rates: worse 1, better 5 -> refuted
```

## Amendment (2026-11-08 10:50 GMT-3)

The disclosure's fold check compared mean/mode's `impute_score`, which is 1 by construction, so it proved nothing.
Redone on `rmse_num_z` and `mae_num_z` for the same five variants, it holds: max |diff| 4e-16 (biodeg_marf_60nan
s404, kc2_marfctl_20nan s505, vehicle_probit_60nan s42, biodeg_probit_20nan s7, kc2_60nan s13). The rebuilt folds are
the runs' own, and the Outcome stands.
