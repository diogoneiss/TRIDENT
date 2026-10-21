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

## Launch

- **Started at 10:06 GMT-3 on 2026-10-21** on 1af3cce, in two lanes.
- **Ended at 14:07:** all 156 pairs, none failed, none skipped.

## Outcome (2026-10-21, 14:09 GMT-3)

**A correction to the reader, before the MNAR section was read in full.**

- The first run printed the three slope-dial pairs (biodeg 0.25, 1.0 and 2.0) as incomplete, and
  the other seven pairs' MNAR tallies.
  - The reader looked for the dial's MCAR control, biodeg_60nan, in E46's runs. It lives in
    E45's, as `mnar_family_report.dial` reads it.
- The fix gives each pair its own source for each side. The design is unchanged, and the
  printout below is the corrected one.

**The printout** (`scripts/experiments/tabpfn_seed_sets_report.py`, unedited):

```text
## Development set, seeds 42/7/13 (E35)
- `kr-vs-kp_20nan`: K − strong_live (ice/missforest) -0.0691 better; K − strong (tabpfn) +0.0033 none; K − tabpfn (tabpfn) +0.0033 none
- `kr-vs-kp_40nan` (labelled: not counted): K − strong_live (ice/missforest) -0.0307 better; K − strong (tabpfn) +0.0464 worse; K − tabpfn (tabpfn) +0.0464 worse
- `kr-vs-kp_60nan`: K − strong_live (ice) -0.0449 better; K − strong (tabpfn) +0.0380 worse; K − tabpfn (tabpfn) +0.0380 worse
- `kr-vs-kp_80nan`: K − strong_live (condexp) -0.0241 better; K − strong (tabpfn) +0.0340 worse; K − tabpfn (tabpfn) +0.0340 worse
- `spambase_20nan`: K − strong_live (missforest) -0.0114 none; K − strong (missforest) -0.0114 none; K − tabpfn (tabpfn) -0.0898 better
- `spambase_40nan` (labelled: not counted): K − strong_live (missforest) -0.0386 better; K − strong (missforest) -0.0386 better; K − tabpfn (tabpfn) -0.0551 better
- `spambase_60nan`: K − strong_live (condexp) -0.0175 better; K − strong (condexp) -0.0175 better; K − tabpfn (tabpfn) -0.0323 better
- `spambase_80nan`: K − strong_live (condexp) -0.0129 better; K − strong (tabpfn) -0.0049 better; K − tabpfn (tabpfn) -0.0049 better
- `credit-g_20nan`: K − strong_live (missforest) -0.0028 none; K − strong (tabpfn) +0.0101 none; K − tabpfn (tabpfn) +0.0101 none
- `credit-g_40nan` (labelled: not counted): K − strong_live (missforest) +0.0250 worse; K − strong (tabpfn) +0.0550 worse; K − tabpfn (tabpfn) +0.0550 worse
- `credit-g_60nan`: K − strong_live (condexp) +0.0042 none; K − strong (tabpfn) +0.0387 worse; K − tabpfn (tabpfn) +0.0387 worse
- `credit-g_80nan`: K − strong_live (condexp) +0.0079 worse; K − strong (condexp/tabpfn) +0.0092 worse; K − tabpfn (tabpfn) +0.0072 none
- `vehicle_20nan`: K − strong_live (condexp/missforest) -0.0304 better; K − strong (tabpfn) -0.0022 none; K − tabpfn (tabpfn) -0.0022 none
- `vehicle_60nan`: K − strong_live (missforest) -0.0018 none; K − strong (tabpfn) +0.0098 none; K − tabpfn (tabpfn) +0.0098 none
- `biodeg_20nan`: K − strong_live (missforest) -0.0318 better; K − strong (missforest/tabpfn) -0.0275 better; K − tabpfn (tabpfn) -0.0299 better
- `biodeg_60nan`: K − strong_live (missforest) -0.0019 none; K − strong (missforest) -0.0019 none; K − tabpfn (tabpfn) -0.0268 better
- `kc2_20nan`: K − strong_live (softimpute) -0.0357 none; K − strong (tabpfn) +0.0169 none; K − tabpfn (tabpfn) +0.0169 none
- `kc2_60nan`: K − strong_live (ice/softimpute) -0.0816 better; K − strong (softimpute/tabpfn) -0.0384 none; K − tabpfn (tabpfn) -0.0431 none
- `pendigits_20nan` (labelled: not counted): K − strong_live (missforest) -0.0020 none; K − strong (tabpfn) +0.0245 worse; K − tabpfn (tabpfn) +0.0245 worse
- `letter_20nan` (labelled: not counted): K − strong_live (missforest) -0.0029 none; K − strong (tabpfn) +0.0053 worse; K − tabpfn (tabpfn) +0.0053 worse
- `electricity_20nan`: K − strong_live (missforest) -0.0015 none; K − strong (missforest/tabpfn) -0.0015 none; K − tabpfn (tabpfn) -0.0083 none
  - K − strong_live: better 8 / no difference 7 / worse 1 (fold-pair t, seed agreement)
  - K − strong: better 3 / no difference 9 / worse 4 (fold-pair t, seed agreement)
  - K − tabpfn: better 5 / no difference 8 / worse 3 (fold-pair t, seed agreement)

## Development set at 40nan and above, seeds 404/505/606 (E40)
- `biodeg_60nan`: K − strong_live (missforest) +0.0117 worse; K − strong (missforest) +0.0117 worse; K − tabpfn (tabpfn) -0.0135 better
- `credit-g_40nan`: K − strong_live (missforest) -0.0053 none; K − strong (tabpfn) +0.0220 worse; K − tabpfn (tabpfn) +0.0220 worse
- `credit-g_60nan`: K − strong_live (condexp) +0.0017 none; K − strong (tabpfn) +0.0392 worse; K − tabpfn (tabpfn) +0.0392 worse
- `credit-g_80nan`: K − strong_live (condexp) +0.0111 worse; K − strong (tabpfn) +0.0139 worse; K − tabpfn (tabpfn) +0.0139 worse
- `kc2_60nan`: K − strong_live (softimpute) -0.0385 none; K − strong (softimpute) -0.0385 none; K − tabpfn (tabpfn) -0.0739 better
- `kr-vs-kp_40nan`: K − strong_live (ice/missforest) -0.0688 better; K − strong (tabpfn) +0.0110 worse; K − tabpfn (tabpfn) +0.0110 worse
- `kr-vs-kp_60nan`: K − strong_live (ice) -0.0442 better; K − strong (tabpfn) +0.0435 worse; K − tabpfn (tabpfn) +0.0435 worse
- `kr-vs-kp_80nan`: K − strong_live (condexp) -0.0210 better; K − strong (tabpfn) +0.0364 worse; K − tabpfn (tabpfn) +0.0364 worse
- `spambase_40nan`: K − strong_live (missforest) -0.0369 better; K − strong (missforest) -0.0369 better; K − tabpfn (tabpfn) -0.0539 better
- `spambase_60nan`: K − strong_live (condexp) -0.0162 better; K − strong (condexp) -0.0162 better; K − tabpfn (tabpfn) -0.0346 better
- `spambase_80nan`: K − strong_live (condexp) -0.0121 better; K − strong (tabpfn) -0.0055 better; K − tabpfn (tabpfn) -0.0055 better
- `vehicle_60nan`: K − strong_live (missforest) -0.0135 none; K − strong (tabpfn) +0.0018 none; K − tabpfn (tabpfn) +0.0018 none
  - K − strong_live: better 6 / no difference 4 / worse 2 (fold-pair t, seed agreement)
  - K − strong: better 3 / no difference 2 / worse 7 (fold-pair t, seed agreement)
  - K − tabpfn: better 5 / no difference 1 / worse 6 (fold-pair t, seed agreement)

## Fresh seeds at 20nan, 1001/2002/3003 (E51)
- `electricity_20nan`: K − strong_live (missforest) +0.0003 none; K − strong (missforest) +0.0003 none; K − tabpfn (tabpfn) -0.0120 none
- `letter_20nan`: K − strong_live (missforest) -0.0301 better; K − strong (tabpfn) -0.0213 better; K − tabpfn (tabpfn) -0.0213 better
- `pendigits_20nan`: K − strong_live (missforest) -0.0143 better; K − strong (tabpfn) +0.0118 worse; K − tabpfn (tabpfn) +0.0118 worse
- `spambase_20nan`: K − strong_live (missforest) -0.0082 none; K − strong (missforest) -0.0082 none; K − tabpfn (tabpfn) -0.0863 better
- `kr-vs-kp_20nan`: K − strong_live (ice/missforest) -0.0703 better; K − strong (tabpfn) +0.0052 none; K − tabpfn (tabpfn) +0.0052 none
- `biodeg_20nan`: K − strong_live (missforest) -0.0276 better; K − strong (tabpfn) -0.0177 none; K − tabpfn (tabpfn) -0.0177 none
- `credit-g_20nan`: K − strong_live (missforest) -0.0135 better; K − strong (tabpfn) +0.0027 none; K − tabpfn (tabpfn) +0.0027 none
- `vehicle_20nan`: K − strong_live (missforest) -0.0269 better; K − strong (tabpfn) -0.0011 none; K − tabpfn (tabpfn) -0.0011 none
- `kc2_20nan`: K − strong_live (softimpute) -0.0252 none; K − strong (softimpute/tabpfn) -0.0103 none; K − tabpfn (tabpfn) -0.0137 none
  - K − strong_live: better 6 / no difference 3 / worse 0 (fold-pair t, seed agreement)
  - K − strong: better 1 / no difference 7 / worse 1 (fold-pair t, seed agreement)
  - K − tabpfn: better 2 / no difference 6 / worse 1 (fold-pair t, seed agreement)

## The MNAR family (G's H; E45 and E46)
- `biodeg_probit_60nan` against `biodeg_60nan`: MNAR H − strong_live (missforest) +0.0204 worse; MNAR H − strong (missforest) +0.0204 worse; MNAR H − tabpfn (tabpfn) -0.0116 better; MCAR H − strong_live (missforest) +0.0059 none; MCAR H − strong (missforest) +0.0059 none; MCAR H − tabpfn (tabpfn) -0.0189 better; (H − tabpfn) MNAR minus MCAR +0.0073 none
- `kc2_probit_60nan` against `kc2_60nan`: MNAR H − strong_live (missforest) -0.0107 none; MNAR H − strong (missforest) -0.0107 none; MNAR H − tabpfn (tabpfn) -0.0454 better; MCAR H − strong_live (ice/softimpute) -0.0371 none; MCAR H − strong (softimpute/tabpfn) +0.0060 none; MCAR H − tabpfn (tabpfn) +0.0014 none; (H − tabpfn) MNAR minus MCAR -0.0469 none
- `vehicle_probit_60nan` against `vehicle_60nan`: MNAR H − strong_live (missforest) +0.0040 none; MNAR H − strong (missforest/tabpfn) +0.0050 none; MNAR H − tabpfn (tabpfn) +0.0044 none; MCAR H − strong_live (missforest) -0.0008 none; MCAR H − strong (tabpfn) +0.0108 none; MCAR H − tabpfn (tabpfn) +0.0108 none; (H − tabpfn) MNAR minus MCAR -0.0064 none
- `spambase_probit_60nan` against `spambase_60nan`: MNAR H − strong_live (missforest) -0.0011 equivalent; MNAR H − strong (missforest) -0.0011 equivalent; MNAR H − tabpfn (tabpfn) -0.0348 better; MCAR H − strong_live (condexp) -0.0148 better; MCAR H − strong (condexp) -0.0148 better; MCAR H − tabpfn (tabpfn) -0.0296 better; (H − tabpfn) MNAR minus MCAR -0.0052 none
- `biodeg_probit_20nan` against `biodeg_20nan`: MNAR H − strong_live (missforest) -0.0260 better; MNAR H − strong (missforest) -0.0260 better; MNAR H − tabpfn (tabpfn) -0.0600 better; MCAR H − strong_live (missforest) -0.0229 better; MCAR H − strong (missforest/tabpfn) -0.0186 none; MCAR H − tabpfn (tabpfn) -0.0210 none; (H − tabpfn) MNAR minus MCAR -0.0390 better
- `kc2_probit_20nan` against `kc2_20nan`: MNAR H − strong_live (missforest) -0.0856 better; MNAR H − strong (tabpfn) -0.0113 none; MNAR H − tabpfn (tabpfn) -0.0113 none; MCAR H − strong_live (softimpute) +0.0004 none; MCAR H − strong (tabpfn) +0.0530 none; MCAR H − tabpfn (tabpfn) +0.0530 none; (H − tabpfn) MNAR minus MCAR -0.0643 none
- `vehicle_probit_20nan` against `vehicle_20nan`: MNAR H − strong_live (missforest) -0.0719 better; MNAR H − strong (tabpfn) -0.0255 none; MNAR H − tabpfn (tabpfn) -0.0255 none; MCAR H − strong_live (condexp/missforest) -0.0264 better; MCAR H − strong (tabpfn) +0.0018 none; MCAR H − tabpfn (tabpfn) +0.0018 none; (H − tabpfn) MNAR minus MCAR -0.0273 none
- `biodeg_probit0p25_60nan` against `biodeg_60nan`: MNAR H − strong_live (missforest) +0.0237 worse; MNAR H − strong (missforest) +0.0237 worse; MNAR H − tabpfn (tabpfn) -0.0066 none; MCAR H − strong_live (missforest) +0.0059 none; MCAR H − strong (missforest) +0.0059 none; MCAR H − tabpfn (tabpfn) -0.0189 better; (H − tabpfn) MNAR minus MCAR +0.0123 none
- `biodeg_probit1p0_60nan` against `biodeg_60nan`: MNAR H − strong_live (missforest) -0.0059 better; MNAR H − strong (missforest) -0.0059 better; MNAR H − tabpfn (tabpfn) -0.0100 better; MCAR H − strong_live (missforest) +0.0059 none; MCAR H − strong (missforest) +0.0059 none; MCAR H − tabpfn (tabpfn) -0.0189 better; (H − tabpfn) MNAR minus MCAR +0.0089 none
- `biodeg_probit2p0_60nan` against `biodeg_60nan`: MNAR H − strong_live (missforest) -0.0245 better; MNAR H − strong (missforest) -0.0245 better; MNAR H − tabpfn (tabpfn) -0.0276 better; MCAR H − strong_live (missforest) +0.0059 none; MCAR H − strong (missforest) +0.0059 none; MCAR H − tabpfn (tabpfn) -0.0189 better; (H − tabpfn) MNAR minus MCAR -0.0087 none
  - MNAR H − strong_live: better 5 / no difference 2 / equivalent 1 / worse 2 (fold-pair t, seed agreement)
  - MNAR H − strong: better 3 / no difference 4 / equivalent 1 / worse 2 (fold-pair t, seed agreement)
  - MNAR H − tabpfn: better 6 / no difference 4 / worse 0 (fold-pair t, seed agreement)
  - MCAR H − strong_live: better 3 / no difference 7 / worse 0 (fold-pair t, seed agreement)
  - MCAR H − strong: better 1 / no difference 9 / worse 0 (fold-pair t, seed agreement)
  - MCAR H − tabpfn: better 5 / no difference 5 / worse 0 (fold-pair t, seed agreement)
  - (H − tabpfn), MNAR minus MCAR: better 1 / no difference 9 / worse 0 (fold-pair t, seed agreement)
```

**Reading:**

- **Against the strong tier with TabPFN,** every seed set moves the way E55's did.

  | set | before (live four) | after (ADR 0028) |
  |---|---|---|
  | 42/7/13 | 8 / 7 / 1 | 3 / 9 / 4 |
  | 404/505/606 | 6 / 4 / 2 | 3 / 2 / 7 |
  | E51's fresh 20nan seeds | 6 / 3 / 0 | 1 / 7 / 1 |

  Each cell is K's better / no difference / worse.

- **TabPFN's wins repeat by table across every seed set:**
  - credit-g and kr-vs-kp from 40nan (404/505/606: TabPFN better on all six);
  - pendigits_20nan (worse at 101/202/303, at 1001/2002/3003 and at 42/7/13's labelled cell).
- **K's wins repeat too:**
  - spambase at every rate (TabPFN worse on all of 404/505/606 and at 1001/2002/3003);
  - letter_20nan, on the two seed sets with today's encoder. 42/7/13's labelled cell, the old
    encoder, is worse (+0.0053);
  - kc2_60nan;
  - biodeg_60nan.
- **The MNAR statement holds against TabPFN.**
  - Under MNAR, G's H against TabPFN: better 6, no difference 4, worse 0. Under MCAR: better 5, no
    difference 5, worse 0.
  - **(H − TabPFN) MNAR minus MCAR:** better 1 (biodeg_probit_20nan), no difference 9, worse 0.
    TabPFN gains no ground on the model under value-dependent missingness.
  - These MNAR tables (biodeg, kc2, vehicle, spambase at 20 and 60nan) are where K stood level with
    or ahead of TabPFN under MCAR too. No MNAR pair is at 80nan.

**What it means for the thesis:**

- With TabPFN in the strong tier, K is level with or behind the strong tier on most seed sets'
  counts.
- K's lead is by table: spambase, letter, biodeg, kc2 and electricity's date-free rows. TabPFN
  leads at 80nan and on kr-vs-kp and credit-g.
- Under MNAR, the model keeps its standing against TabPFN.
- No default changes.

