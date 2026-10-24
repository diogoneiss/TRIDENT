# The tuned bar on the MAR folds: D4 under MAR (E68; pre-registered 2026-10-23)

This ticket was written and committed before the first pair.

- **Why:** this is T34, accepted by the user on 2026-10-22.
  - ADR 0022 decision 5: every study that claims against the bar runs E42's driver on its own
    folds, and D4 counts a win only against both bars.
  - E63 and E64 read the untuned bar only. E64 found (H − bar), MAR minus control, worse on biodeg,
    kc2 and vehicle at 60nan, so "the standing against the bar under MAR" has no D4 reading yet.
- **The user's go:** "vamos na parte de imputacao agora entao" (2026-10-23, about 17:50 GMT-3).
- **The code, with nothing in `src/`:**
  - E42's driver `scripts/experiments/baseline_budget_run.py`, unchanged: the weak tier's families
    knn, hgb and lgbm, threads capped at 4;
  - the new reader `scripts/experiments/mar_tuned_bar_report.py`.

## Design (fixed before the first pair)

- **Pairs:**
  - E64's 24 variants (6 tables × 20 and 60nan × MAR and control) at seeds 404/505/606;
  - E63's 8 variants (4 tables at 60nan × MAR and control) at seeds 42/7/13.
  - That makes 96 (variant, seed) pairs. E64's slope dial is left out.
- **What the driver does:**
  - It rebuilds each run's five folds and evaluation mask.
  - It chooses each family's setting on the validation rows' gaps against the complete sibling.
  - It scores the chosen settings on the test split, and writes `tuned_bar` per fold.
  - Nothing goes to the store.
- **Running:**
  - four CPU lanes, beside E67's GPU queues;
  - about 3–5 h (E44's 45 pairs took about 4 h on 4 lanes);
  - state in `/scratch2/diogoneiss/trident-2026-10-23-mar-tuned-bar/`.

## Measures and reading, fixed now (`mar_tuned_bar_report.py --tally`)

On the induced `impute_score` over 15 fold pairs, with the verdict rule:

- **Per mechanism, with the 1% SESOI:** H − bar, H − tuned bar, K − bar and K − tuned bar. D4 on H
  and on K counts a win where both calls are better.
- **The mechanism effect, fold-paired:** (H − tuned bar) and (K − tuned bar), MAR minus control.
- **Primary,** over E64's 12 table-rates, (H − tuned bar) MAR minus control:
  - **supported** (MAR costs the model ground against the tuned bar): worse on 4 or more, and better
    on at most 1;
  - **refuted:** worse on at most 1;
  - **mixed:** otherwise.
  - E63's four are secondary.
- **The denominator, as in E63 and E64:** under MAR the mean/mode denominator comes from observed
  values the mechanism biases. So scores are compared across mechanisms only as differences of
  within-variant gaps.
- **What each reading leads to (no default changes):**
  - **refuted:** the thesis's MAR statement holds against both bars.
  - **supported:** the thesis states that under MAR the tuned bar gains on the model, naming where.
  - **mixed:** reported, and the user chooses.

## Outcome (2026-10-23, 23:54 GMT-3)

**The run:** 96 of 96 pairs on four CPU lanes, 17:55–23:53 GMT-3, with none failed.

**The printout** (`scripts/experiments/mar_tuned_bar_report.py --tally`, unedited):

```text
# E64 (mar-family-2026-10-22, seeds 404/505/606)

## `biodeg_20nan`
- MAR: H − bar -0.0126 none; H − tuned bar -0.0012 none; K − bar -0.0032 none; K − tuned bar +0.0081 none; D4 on H: no win; D4 on K: no win
- control: H − bar -0.0217 none; H − tuned bar -0.0122 none; K − bar -0.0211 none; K − tuned bar -0.0117 none; D4 on H: no win; D4 on K: no win
  - (H − tuned bar) MAR minus control: +0.0110 [-0.0228, +0.0448] none
  - (K − tuned bar) MAR minus control: +0.0198 [-0.0176, +0.0571] none

## `biodeg_60nan`
- MAR: H − bar -0.0097 none; H − tuned bar -0.0108 none; K − bar -0.0258 better; K − tuned bar -0.0269 better; D4 on H: no win; D4 on K: win
- control: H − bar -0.0495 better; H − tuned bar -0.0473 better; K − bar -0.0576 better; K − tuned bar -0.0554 better; D4 on H: win; D4 on K: win
  - (H − tuned bar) MAR minus control: +0.0365 [+0.0137, +0.0593] worse
  - (K − tuned bar) MAR minus control: +0.0285 [+0.0091, +0.0479] worse

## `kc2_20nan`
- MAR: H − bar -0.1121 better; H − tuned bar -0.1118 better; K − bar -0.1713 better; K − tuned bar -0.1710 better; D4 on H: win; D4 on K: win
- control: H − bar +0.0118 none; H − tuned bar +0.0150 none; K − bar -0.0003 none; K − tuned bar +0.0029 none; D4 on H: no win; D4 on K: no win
  - (H − tuned bar) MAR minus control: -0.1268 [-0.2491, -0.0045] better
  - (K − tuned bar) MAR minus control: -0.1739 [-0.2895, -0.0583] better

## `kc2_60nan`
- MAR: H − bar -0.0888 better; H − tuned bar -0.0878 better; K − bar -0.1101 better; K − tuned bar -0.1091 better; D4 on H: win; D4 on K: win
- control: H − bar -0.1348 better; H − tuned bar -0.1310 better; K − bar -0.1556 better; K − tuned bar -0.1517 better; D4 on H: win; D4 on K: win
  - (H − tuned bar) MAR minus control: +0.0431 [+0.0012, +0.0851] worse
  - (K − tuned bar) MAR minus control: +0.0426 [-0.0026, +0.0878] none

## `vehicle_20nan`
- MAR: H − bar -0.0436 better; H − tuned bar -0.0367 better; K − bar -0.0450 better; K − tuned bar -0.0380 better; D4 on H: win; D4 on K: win
- control: H − bar -0.0517 better; H − tuned bar -0.0543 better; K − bar -0.0567 better; K − tuned bar -0.0593 better; D4 on H: win; D4 on K: win
  - (H − tuned bar) MAR minus control: +0.0176 [-0.0181, +0.0533] none
  - (K − tuned bar) MAR minus control: +0.0213 [-0.0179, +0.0605] none

## `vehicle_60nan`
- MAR: H − bar -0.0295 better; H − tuned bar -0.0288 better; K − bar -0.0349 better; K − tuned bar -0.0342 better; D4 on H: win; D4 on K: win
- control: H − bar -0.0553 better; H − tuned bar -0.0547 better; K − bar -0.0591 better; K − tuned bar -0.0585 better; D4 on H: win; D4 on K: win
  - (H − tuned bar) MAR minus control: +0.0259 [+0.0062, +0.0455] worse
  - (K − tuned bar) MAR minus control: +0.0243 [+0.0062, +0.0424] worse

## `kr-vs-kp_20nan`
- MAR: H − bar -0.0191 better; H − tuned bar -0.0200 better; K − bar -0.0180 better; K − tuned bar -0.0189 better; D4 on H: win; D4 on K: win
- control: H − bar -0.0112 none; H − tuned bar -0.0147 none; K − bar +0.0051 none; K − tuned bar +0.0016 none; D4 on H: no win; D4 on K: no win
  - (H − tuned bar) MAR minus control: -0.0053 [-0.0298, +0.0193] none
  - (K − tuned bar) MAR minus control: -0.0205 [-0.0430, +0.0021] none

## `kr-vs-kp_60nan`
- MAR: H − bar -0.0360 better; H − tuned bar +0.0068 none; K − bar -0.0370 better; K − tuned bar +0.0057 none; D4 on H: no win; D4 on K: no win
- control: H − bar -0.0396 better; H − tuned bar -0.0036 none; K − bar -0.0385 better; K − tuned bar -0.0025 none; D4 on H: no win; D4 on K: no win
  - (H − tuned bar) MAR minus control: +0.0103 [-0.0020, +0.0227] none
  - (K − tuned bar) MAR minus control: +0.0082 [-0.0087, +0.0251] none

## `spambase_20nan`
- MAR: H − bar -0.0064 none; H − tuned bar -0.0047 none; K − bar -0.0072 none; K − tuned bar -0.0054 none; D4 on H: no win; D4 on K: no win
- control: H − bar -0.0017 none; H − tuned bar +0.0018 none; K − bar -0.0086 better; K − tuned bar -0.0051 none; D4 on H: no win; D4 on K: no win
  - (H − tuned bar) MAR minus control: -0.0065 [-0.0191, +0.0061] none
  - (K − tuned bar) MAR minus control: -0.0003 [-0.0139, +0.0132] none

## `spambase_60nan`
- MAR: H − bar -0.0211 better; H − tuned bar -0.0058 none; K − bar -0.0239 better; K − tuned bar -0.0086 better; D4 on H: no win; D4 on K: win
- control: H − bar -0.0147 better; H − tuned bar +0.0044 none; K − bar -0.0190 better; K − tuned bar +0.0001 equivalent; D4 on H: no win; D4 on K: no win
  - (H − tuned bar) MAR minus control: -0.0101 [-0.0194, -0.0008] better
  - (K − tuned bar) MAR minus control: -0.0087 [-0.0164, -0.0010] better

## `credit-g_20nan`
- MAR: H − bar -0.0409 better; H − tuned bar -0.0206 better; K − bar -0.0419 better; K − tuned bar -0.0217 better; D4 on H: win; D4 on K: win
- control: H − bar -0.0268 better; H − tuned bar -0.0245 better; K − bar -0.0148 none; K − tuned bar -0.0126 none; D4 on H: win; D4 on K: no win
  - (H − tuned bar) MAR minus control: +0.0039 [-0.0225, +0.0303] none
  - (K − tuned bar) MAR minus control: -0.0091 [-0.0318, +0.0136] none

## `credit-g_60nan`
- MAR: H − bar -0.0190 better; H − tuned bar -0.0009 none; K − bar -0.0295 better; K − tuned bar -0.0113 none; D4 on H: no win; D4 on K: no win
- control: H − bar -0.0132 better; H − tuned bar +0.0084 none; K − bar -0.0195 better; K − tuned bar +0.0020 none; D4 on H: no win; D4 on K: no win
  - (H − tuned bar) MAR minus control: -0.0093 [-0.0243, +0.0058] none
  - (K − tuned bar) MAR minus control: -0.0133 [-0.0265, -0.0001] better

- MAR D4 on H: 6 wins of 12
- MAR D4 on K: 8 wins of 12
- control D4 on H: 5 wins of 12
- control D4 on K: 4 wins of 12
- (H − tuned bar) MAR minus control: better 2 / no difference 7 / worse 3 (fold-pair t, seed agreement)
- (K − tuned bar) MAR minus control: better 3 / no difference 7 / worse 2 (fold-pair t, seed agreement)

# E63 (mar-smoke-2026-10-22, seeds 42/7/13)

## `biodeg_60nan`
- MAR: H − bar -0.0358 better; H − tuned bar -0.0312 better; K − bar -0.0535 better; K − tuned bar -0.0489 better; D4 on H: win; D4 on K: win
- control: H − bar -0.0462 better; H − tuned bar -0.0397 better; K − bar -0.0596 better; K − tuned bar -0.0531 better; D4 on H: win; D4 on K: win
  - (H − tuned bar) MAR minus control: +0.0085 [-0.0185, +0.0354] none
  - (K − tuned bar) MAR minus control: +0.0042 [-0.0162, +0.0245] none

## `kc2_60nan`
- MAR: H − bar -0.1262 better; H − tuned bar -0.1228 better; K − bar -0.1383 better; K − tuned bar -0.1350 better; D4 on H: win; D4 on K: win
- control: H − bar -0.0537 none; H − tuned bar -0.0537 none; K − bar -0.1023 better; K − tuned bar -0.1023 better; D4 on H: no win; D4 on K: win
  - (H − tuned bar) MAR minus control: -0.0691 [-0.1349, -0.0034] better
  - (K − tuned bar) MAR minus control: -0.0326 [-0.0850, +0.0198] none

## `vehicle_60nan`
- MAR: H − bar -0.0408 better; H − tuned bar -0.0366 better; K − bar -0.0509 better; K − tuned bar -0.0467 better; D4 on H: win; D4 on K: win
- control: H − bar -0.0437 better; H − tuned bar -0.0364 better; K − bar -0.0562 better; K − tuned bar -0.0490 better; D4 on H: win; D4 on K: win
  - (H − tuned bar) MAR minus control: -0.0002 [-0.0156, +0.0152] none
  - (K − tuned bar) MAR minus control: +0.0023 [-0.0154, +0.0200] none

## `kr-vs-kp_60nan`
- MAR: H − bar -0.0282 better; H − tuned bar +0.0170 worse; K − bar -0.0361 better; K − tuned bar +0.0091 worse; D4 on H: no win; D4 on K: no win
- control: H − bar -0.0380 better; H − tuned bar +0.0087 worse; K − bar -0.0441 better; K − tuned bar +0.0026 none; D4 on H: no win; D4 on K: no win
  - (H − tuned bar) MAR minus control: +0.0083 [-0.0021, +0.0186] none
  - (K − tuned bar) MAR minus control: +0.0065 [-0.0049, +0.0179] none

- MAR D4 on H: 3 wins of 4
- MAR D4 on K: 3 wins of 4
- control D4 on H: 2 wins of 4
- control D4 on K: 3 wins of 4
- (H − tuned bar) MAR minus control: better 1 / no difference 3 / worse 0 (fold-pair t, seed agreement)
- (K − tuned bar) MAR minus control: better 0 / no difference 4 / worse 0 (fold-pair t, seed agreement)

Primary, E64's (H − tuned bar) MAR minus control: worse on ['biodeg_60nan', 'kc2_60nan', 'vehicle_60nan'], better on ['kc2_20nan', 'spambase_60nan'] -> mixed (fold-pair t, seed agreement)
```

**Verdict: mixed.** (H − tuned bar), MAR minus control, is worse on 3 of 12 (biodeg_60nan,
kc2_60nan, vehicle_60nan) and better on 2 (kc2_20nan, spambase_60nan). Supported needed worse on 4
or more with better on at most 1; refuted needed worse on at most 1. By the ticket, the user
chooses.

**Reading:**

- **The model is never behind the tuned bar on E64's family, under either mechanism.** H − tuned
  bar is better on 6 of 12 and no different on 6 under MAR, and better on 5 and no different on 7
  under the control. On E63's smoke, kr-vs-kp_60nan is worse on H under both mechanisms.
- **The ground MAR costs is at 60nan on the numerical tables, and against both bars alike.** The
  three worse table-rates are the ones where E64 found (H − bar) worse at 60nan. On them H − bar and
  H − tuned bar barely differ (biodeg_60nan −0.0097 and −0.0108 under MAR), so tuning the bar adds
  nothing there: the model's lead shrinks by 0.026–0.043 from the control to MAR, and it stays
  ahead or level.
- **Where tuning the bar matters, it matters under both mechanisms.** On kr-vs-kp_60nan,
  spambase_60nan and credit-g_60nan the model beats the plain bar but not the tuned one, under MAR
  and under the control. That is ADR 0022's D4 finding at high missingness, not a MAR effect.
- **D4 is not lower under MAR:** on H 6 wins of 12 under MAR against 5 under the control; on K 8
  against 4.
- **The two better table-rates** (kc2_20nan, spambase_60nan) come from the control's weaker lead,
  not from MAR helping the model: kc2_20nan's control has no verdict against either bar.
- **For the user (mixed):** a statement the data support is "under MAR the model stays ahead of or
  level with the tuned bar on all 12 table-rates; at 60nan on biodeg, kc2 and vehicle its lead
  over both bars shrinks by 0.026–0.043 from MCAR". No default changes.

## The user's decision (2026-10-24)

- **"1 ok"**: the thesis states E68's reading as the Outcome proposed it: under MAR the model stays
  ahead of or level with the tuned bar on all 12 table-rates of E64's family; at 60nan on biodeg,
  kc2 and vehicle its lead over both bars shrinks by 0.026–0.043 from MCAR. No default changes.

