# The strong tier with TabPFN on the seed sets E55 did not cover (E58; pre-registered 2026-10-21)

This ticket was written and committed before the first cell.

- **Why:** ADR [0028](../../adr/0028-tabpfn-joins-the-strong-tier-as-an-offline-member.md) puts
  TabPFN v2 in the strong tier wherever a study has its offline pass. E55 covered the development
  set at 101/202/303 and E44's confirmation set. Every other seed set the thesis reads still meets
  the live four only.
  - **The other seed sets:** 42/7/13 (E35), 404/505/606 (E40), E51's fresh seeds and the MNAR
    family (E45, E46).
  - **The MNAR claim matters most here:** "the lead over the bar survives MNAR" has not met TabPFN.
- **The user's go:** "Faça 1 e 2 em sequência agora, pode usar a gpu durante o dia" (2026-10-21,
  about 09:10 GMT-3).
- **Code:** `run_tabpfn_lanes.py e58` (the pairs below; logs in their own folder, CSVs beside
  E55's) and the reader `scripts/experiments/tabpfn_seed_sets_report.py`. No change to `src/` or
  to TabPFN's settings (E55's).

## Design (fixed before the first cell)

- **Pairs:** 156 (variant, seed) pairs, each fold rebuilt by `recompute_folds` as its run built it.

  | set | variants | seeds | pairs |
  |---|---|---|---|
  | 42/7/13 (E35) | the 21 development variants | 42, 7, 13 | 63 |
  | 404/505/606 (E40) | its 12 variants at 40nan and above | 404, 505, 606 | 36 |
  | E51's fresh seeds | the nine `_20nan` variants | 1001, 2002, 3003 | 27 |
  | MNAR (E45/E46) | the ten MNAR variants | 42, 7, 13 | 30 |

  - The MNAR family's MCAR controls are development variants at 42/7/13, so the first row covers
    them.
- **Running it:** two GPU lanes, in daytime, with the user's go. The MNAR pairs go first.
  - No pair starts after a pair-by-pair deadline at 23:00 GMT-3.
  - A variant without all three seeds is reported as incomplete.

## Reading, fixed now (descriptive; ADR 0028 stands)

`tabpfn_seed_sets_report.py`, on the induced `impute_score` over 15 fold pairs, with the verdict
rule and the 1% SESOI:

- **42/7/13 and 404/505/606:** K − strong_live, K − strong (with TabPFN) and K − TabPFN, tallied.
  42/7/13's five labelled variants are left out of the counts, as in the thesis table.
- **E51's fresh seeds:** the same three.
- **MNAR:**
  - G's H, the readout E45 and E46 read: H − strong_live, H − strong and H − TabPFN, on each MNAR
    variant and its MCAR control;
  - and (H − TabPFN) MNAR minus MCAR: does TabPFN gain or lose ground under MNAR?
  - The MNAR variants have no E41 CSVs, so their `strong_live` is missForest alone. The printout
    names each opponent.
- **What it leads to:** the thesis table's other seed sets and the MNAR statement are read against
  the strong tier of ADR 0028. No default changes.
