# What the imputation studies support, as of 2026-10-23 (consolidated)

This report gathers E44–E65 into one statement for the thesis. It was built 2026-10-23,
05:44 GMT-3, and first built 2026-10-22, 15:12 GMT-3, before E64 and E65. It adds no new
analysis.

- **Every number is a pre-registered reader's output.**
  - The summary lines are in the appendix, built by `scripts/experiments/results_digest.py`, which
    runs each reader unchanged.
  - The per-variant lines stay in each ticket's Outcome, linked below.
- **The per-variant thesis table** is
  [2026-10-21-thesis-table-with-tabpfn](2026-10-21-thesis-table-with-tabpfn.md), and stays the
  source for it.
- **The readouts:**
  - **K:** the checkpoint chosen on the validation gaps (ADR 0015). It is an oracle readout, since it
    uses those gaps' truth.
  - **KC:** K calibrated on the same gaps. It is the default output at 30% missing and above
    (ADR 0027).
  - **H:** the loss checkpoint, the deployable readout (ADR 0024).
- **The tiers:**
  - the **bar** is the best of mean/mode, kNN 5 and 10, HGB and LightGBM;
  - the **strong tier** is missForest, conditional expectation, SoftImpute and ICE, with TabPFN v2
    as an offline member (ADR 0028).

## The statement, in eight lines

1. **TRIDENT beats the bar on most variants and seed sets,** and that holds across fresh seeds
   (E51) and fresh MCAR draws (E62).
2. **Against the strong tier with TabPFN, the result is mixed and table-dependent.**
   - K − strong is 5 / 9 / 7 on development (101/202/303) and 6 / 4 / 5 on confirmation.
   - Without TabPFN it would read 13 / 7 / 1 and 12 / 1 / 2.
3. **TRIDENT leads TabPFN by table and rate:** spambase at every rate, letter and pendigits at
   20–60nan (pendigits_20nan excepted), biodeg below 80nan, and kc2 at 60–80nan. This is the
   thesis table report's reading.
   - **TabPFN leads at 80nan, and on kr-vs-kp and credit-g from 40nan.**
   - At 80nan, TabPFN's wins come with more shrunk fills: TRIDENT's dispersion is closer to the
     truth on 14 of 15 confirmation variants (E56).
4. **TRIDENT's and TabPFN's errors are complementary (E60).** An equal-weight average of their
   fills beats both on 10 of 15 confirmation variants.
5. **Where TabPFN wins on the categorical tables, the deficit is uniform (E61).** TRIDENT learns
   every column TabPFN learns, a few points less. No column is stuck at the marginal, so no
   per-column loss weighting has a target.
6. **Capacity does not close the 80nan gap (E59).** The bigger encoder gains about 4% at 40nan,
   less at 60nan and nothing at 80nan. ADR 0018 stays at 20nan, and the rest is T32.
7. **The standing holds across MCAR draws, under MNAR and under MAR.**
   - **Under MNAR (E45, E46, E58),** H is never worse than TabPFN (6 better, 4 no difference), and
     TabPFN gains no ground over MCAR.
   - **Under MAR (E63–E65):** E63's smoke read "loses ground" against TabPFN.
     - The fuller family (E64, six tables at 20 and 60nan, a new draw and fresh seeds) refutes it.
       (H − strong), MAR minus control, is worse only on biodeg_60nan, 1 of 12.
     - The bar gains at 60nan on the three small numerical tables.
     - Decode masks that follow the missingness (E65) do not help, and cost accuracy.
   - **Across MCAR draws (E62),** no call reverses.
8. **Pre-training adds little because the decode stage washes the encoder out (E49–E53).** The
   pre-trained representation is real: frozen, it beats a random encoder on pendigits and
   kr-vs-kp, and it does not collapse. But decoding rebuilds an encoder as good from random.

## 1. Against the baselines

- **The bar:**
  - D4 gives 13 wins of 21 on 101/202/303
    ([thesis table](2026-10-21-thesis-table-with-tabpfn.md)).
  - E51's fresh seeds at 20nan reverse no call
    ([08](../tickets/imputation-protocol/08-default-refresh.md)).
  - E62's draws keep K − bar better on three of four tables. The fourth, credit-g_60nan, drifts
    from no difference on the shipped draw to better on the fresh two.
- **The strong tier, live four only:** K is mostly better (13 / 7 / 1 on development, 12 / 1 / 2
  on confirmation).
- **With TabPFN,** every seed set reads mixed. The appendix gives E55 and E58's tallies.
  - 42/7/13: 3 / 9 / 4;
  - 404/505/606: 3 / 2 / 7;
  - E51's fresh seeds: 1 / 7 / 1.
- **The default output against TabPFN** on confirmation:
  - **KC:** 6 / 4 / 1 equivalent / 4, with all four losses at 80nan (E57,
    [12](../tickets/imputation-protocol/12-default-output-vs-tabpfn.md)).
  - **H:** 5 / 5 / 5.

## 2. Robustness

- **Seeds:**
  - Three development seed sets and E51's fresh seeds read the same way against each tier (E58,
    [13](../tickets/imputation-protocol/13-tabpfn-seed-sets.md)).
  - TabPFN's wins repeat on the same tables, and so do TRIDENT's.
- **The MCAR draw (E62,** [17](../tickets/imputation-protocol/17-mcar-draws.md)**):** this covers
  pendigits_20nan, kr-vs-kp_40nan, credit-g_60nan and biodeg_80nan, each in three draws.
  - No reversal on K − bar or K − strong, and the shipped draw is not systematically favourable.
  - **A caveat for every interval:** the draw moves a contrast about as much as the
    fold-and-seed interval, and on biodeg_80nan about twice as much. Per-variant intervals are
    conditional on the draw.
- **MNAR (E45, E46, E58):**
  - Value-dependent missingness on biodeg, kc2 and vehicle at 20 and 60nan and on spambase at
    60nan, plus a slope dial on biodeg.
  - (H − TabPFN), MNAR minus MCAR: better 1, no difference 9, worse 0.
- **MAR (E63, E64, E65;** [18](../tickets/imputation-protocol/18-mar-smoke.md),
  [19](../tickets/imputation-protocol/19-mar-family.md),
  [20](../tickets/imputation-protocol/20-mar-propensity.md)**):**
  - **The mechanism:** 30% fully observed driver columns decide which cells go in the others.
    Each MAR variant is read against an MCAR control with the same drivers and counts.
  - **The smoke (E63):** biodeg, kc2, vehicle and kr-vs-kp at 60nan, seeds 42/7/13.
    - (H − strong), MAR minus control, was worse on biodeg and kr-vs-kp and better on kc2: "loses
      ground".
  - **The family (E64):** those four plus spambase and credit-g, at 20 and 60nan, on a new draw
    and seeds 404/505/606.
    - (H − strong), MAR minus control, is worse only on biodeg_60nan (+0.0291), no difference on
      11. The verdict is **refuted**.
    - (H − bar) is worse on biodeg, kc2 and vehicle at 60nan.
    - A slope dial on biodeg_60nan is not monotone.
  - **The propensity-mask arm (E65):**
    - It tests E63's hypothesis, that uniform decode masks miss where MAR's gaps sit, with masks
      weighted by a fitted missingness propensity (`--decode_mask_weighting propensity`, off by
      default).
    - It is never better than the default, and worse on 6 of 12 under MAR and 5 of 12 under the
      control. **Refuted.**

## 3. TRIDENT and TabPFN, closer

- **By table and rate:** see statement line 3 and the thesis table's K − tabpfn column.
- **Point accuracy against distribution (E56,**
  [11](../tickets/imputation-protocol/11-tabpfn-panel.md)**):**
  - RMSE agrees with every call.
  - MAE reverses two of TRIDENT's wins (biodeg_40nan, pendigits_40nan).
  - TRIDENT keeps the spread closer to the truth on 14 of 15.
- **Complementarity (E60,** [15](../tickets/imputation-protocol/15-tabpfn-blend.md)**):**
  - The blend beats both on 8 of 9 variants at 40–60nan, and at 80nan only on pendigits and letter.
    Elsewhere at 80nan it lands on TabPFN's level.
  - The zero "worse" calls are arithmetic (Minkowski), so only "beats the better one" counts.
  - One seed set, and K rather than KC.
- **The categorical deficit (E61,** [16](../tickets/imputation-protocol/16-categorical-deficit.md)**):**
  - On kr-vs-kp and credit-g from 40nan, K's per-column gains over the mode correlate with
    TabPFN's at +0.76 to +0.94, a few points lower.
  - It is a uniform shortfall in conditional structure. T10's weighting half is closed.
- **Capacity (E59,** [14](../tickets/imputation-protocol/14-large-encoder.md)**):**
  - The bigger encoder costs about 2× the training time.
  - It does not move pendigits_80nan's gap to TabPFN (+0.0121 → +0.0104, still worse).

## 4. Limits the thesis states

- **electricity:** the boosters lead because they look values up by `date`, a 933-level key.
  Without the row's date, K is never worse than the bar (E54,
  [09](../tickets/imputation-protocol/09-electricity.md);
  [T29 report](2026-10-20-t29-electricity.md)).
  - A model-side date encoding was set aside by the user (2026-10-21).
- **80nan and the categorical tables:** TabPFN leads there, with no lever found:
  - not capacity (E59);
  - not a column weighting (E61);
  - not the washout levers (E50).
- **Oracle readouts:** K and KC choose and calibrate on the validation gaps' truth, which only a
  benchmark with complete tables has. H is the deployable readout and reads a little weaker
  (E57: H − TabPFN 5 / 5 / 5).
- **Pre-training:** washed out by the decode stage (T28,
  [report](2026-10-19-t28-why-pretraining-adds-little.md); E49–E53).
- **One seed set or one draw:** E60 (the blend) and E61 (the per-column read) each rest on one
  seed set. Every per-variant interval is conditional on the MCAR draw (E62).
- **MAR:** the standing against the strong tier holds on six tables (E64), with biodeg_60nan the
  exception on two draws. The bar gains at 60nan on the small numerical tables.

## 5. Open

- **T27's classification step zero:** a native-NaN GBDT bar, impute-plus-indicator arms and a
  no-pre-training arm. It needs code and an ADR first.
- **T32:** the bigger encoder at 40–60nan, about 4% at 2× the training time, kept for later by
  the user.
- **T10's evaluation-mask half;** T08, T09, T11, T14 and T15 in the backlog.
- **A replication of E60** on another seed set, if the hybrid argument enters the discussion.
- **The decode-mask tag backfill** (`scripts/backfill_decode_mask_weighting_tag.py --apply`):
  stamps `uniform` on runs before E65. It waits on the user's go and a store backup.

## Appendix: every reader's summary lines (`results_digest.py`, unedited)

### E55/E57: confirmation set against TabPFN and the strong tier (imputation-protocol/10, 12)

```text
## Development set, seeds 101/202/303 (labelled variants: none on this set)
  - K − tabpfn: better 8 / no difference 6 / worse 7 (fold-pair t, seed agreement)
  - tabpfn − strong: better 11 / no difference 5 / equivalent 1 / worse 4 (fold-pair t, seed agreement)
  - tabpfn − bar: better 16 / no difference 1 / equivalent 1 / worse 3 (fold-pair t, seed agreement)
  - tabpfn − tuned: better 14 / no difference 1 / equivalent 1 / worse 5 (fold-pair t, seed agreement)
  - K − best_of_all: better 5 / no difference 8 / worse 8 (fold-pair t, seed agreement)
  - K − strong_live: better 13 / no difference 7 / worse 1 (fold-pair t, seed agreement)
  - K − strong (ADR 0028): better 5 / no difference 9 / worse 7 (fold-pair t, seed agreement)
## Confirmation set (E44), seeds 707/808/909
  - K − tabpfn: better 6 / no difference 4 / worse 5 (fold-pair t, seed agreement)
  - tabpfn − strong: better 11 / no difference 4 / worse 0 (fold-pair t, seed agreement)
  - tabpfn − bar: better 13 / no difference 1 / worse 1 (fold-pair t, seed agreement)
  - tabpfn − tuned: better 12 / no difference 0 / equivalent 1 / worse 2 (fold-pair t, seed agreement)
  - K − best_of_all: better 6 / no difference 3 / worse 6 (fold-pair t, seed agreement)
  - K − strong_live: better 12 / no difference 1 / worse 2 (fold-pair t, seed agreement)
  - K − strong (ADR 0028): better 6 / no difference 4 / worse 5 (fold-pair t, seed agreement)
## E57: confirmation set (E44), seeds 707/808/909, KC and H against TabPFN
  - K − tabpfn: better 6 / no difference 4 / worse 5 (fold-pair t, seed agreement)
  - KC − tabpfn: better 6 / no difference 4 / equivalent 1 / worse 4 (fold-pair t, seed agreement)
  - H − tabpfn: better 5 / no difference 5 / worse 5 (fold-pair t, seed agreement)
  - calls that change from K to KC: ['letter_80nan: K worse -> KC equivalent']
## E57: development set at 40nan and above, seeds 101/202/303 (descriptive), KC and H against TabPFN
  - K − tabpfn: better 4 / no difference 1 / worse 4 (fold-pair t, seed agreement)
  - KC − tabpfn: better 3 / no difference 2 / worse 4 (fold-pair t, seed agreement)
  - H − tabpfn: better 1 / no difference 2 / equivalent 1 / worse 5 (fold-pair t, seed agreement)
  - calls that change from K to KC: ['biodeg_60nan: K better -> KC none']
```

### E58: the other seed sets and MNAR, with TabPFN (imputation-protocol/13)

```text
## Development set, seeds 42/7/13 (E35)
  - K − strong_live: better 8 / no difference 7 / worse 1 (fold-pair t, seed agreement)
  - K − strong: better 3 / no difference 9 / worse 4 (fold-pair t, seed agreement)
  - K − tabpfn: better 5 / no difference 8 / worse 3 (fold-pair t, seed agreement)
## Development set at 40nan and above, seeds 404/505/606 (E40)
  - K − strong_live: better 6 / no difference 4 / worse 2 (fold-pair t, seed agreement)
  - K − strong: better 3 / no difference 2 / worse 7 (fold-pair t, seed agreement)
  - K − tabpfn: better 5 / no difference 1 / worse 6 (fold-pair t, seed agreement)
## Fresh seeds at 20nan, 1001/2002/3003 (E51)
  - K − strong_live: better 6 / no difference 3 / worse 0 (fold-pair t, seed agreement)
  - K − strong: better 1 / no difference 7 / worse 1 (fold-pair t, seed agreement)
  - K − tabpfn: better 2 / no difference 6 / worse 1 (fold-pair t, seed agreement)
## The MNAR family (G's H; E45 and E46)
  - MNAR H − strong_live: better 5 / no difference 2 / equivalent 1 / worse 2 (fold-pair t, seed agreement)
  - MNAR H − strong: better 3 / no difference 4 / equivalent 1 / worse 2 (fold-pair t, seed agreement)
  - MNAR H − tabpfn: better 6 / no difference 4 / worse 0 (fold-pair t, seed agreement)
  - MCAR H − strong_live: better 3 / no difference 7 / worse 0 (fold-pair t, seed agreement)
  - MCAR H − strong: better 1 / no difference 9 / worse 0 (fold-pair t, seed agreement)
  - MCAR H − tabpfn: better 5 / no difference 5 / worse 0 (fold-pair t, seed agreement)
  - (H − tabpfn), MNAR minus MCAR: better 1 / no difference 9 / worse 0 (fold-pair t, seed agreement)
```

### E51: fresh seeds at 20nan (imputation-protocol/08)

```text
Primary: K - bar holds: reversals none (fold-pair t, seed agreement)
```

### E62: fresh MCAR draws (imputation-protocol/17)

```text
## `pendigits_20nan`
  - K − bar (primary): better / better / better -> same; the draws' means span 0.0048 against a mean interval half-width of 0.0032
  - K − strong (primary): worse / worse / worse -> same; the draws' means span 0.0047 against a mean interval half-width of 0.0035
  - K − tabpfn: worse / worse / worse -> same; the draws' means span 0.0047 against a mean interval half-width of 0.0035
  - KC − strong: none / worse / worse -> drift; the draws' means span 0.0044 against a mean interval half-width of 0.0034
## `kr-vs-kp_40nan`
  - K − bar (primary): better / better / better -> same; the draws' means span 0.0022 against a mean interval half-width of 0.0097
  - K − strong (primary): worse / worse / worse -> same; the draws' means span 0.0030 against a mean interval half-width of 0.0063
  - K − tabpfn: worse / worse / worse -> same; the draws' means span 0.0030 against a mean interval half-width of 0.0063
  - KC − strong: worse / worse / worse -> same; the draws' means span 0.0064 against a mean interval half-width of 0.0058
## `credit-g_60nan`
  - K − bar (primary): none / better / better -> drift; the draws' means span 0.0051 against a mean interval half-width of 0.0090
  - K − strong (primary): worse / worse / worse -> same; the draws' means span 0.0083 against a mean interval half-width of 0.0088
  - K − tabpfn: worse / worse / worse -> same; the draws' means span 0.0083 against a mean interval half-width of 0.0088
  - KC − strong: worse / worse / worse -> same; the draws' means span 0.0046 against a mean interval half-width of 0.0077
## `biodeg_80nan`
  - K − bar (primary): better / better / better -> same; the draws' means span 0.0170 against a mean interval half-width of 0.0090
  - K − strong (primary): worse / worse / worse -> same; the draws' means span 0.0135 against a mean interval half-width of 0.0089
  - K − tabpfn: worse / worse / worse -> same; the draws' means span 0.0135 against a mean interval half-width of 0.0089
  - KC − strong: worse / worse / worse -> same; the draws' means span 0.0147 against a mean interval half-width of 0.0077
Primary: reversals on K − bar and K − strong: none -> holds (fold-pair t, seed agreement)
```

### E63: MAR against matched MCAR controls (smoke) (imputation-protocol/18)

```text
## `biodeg` at 60nan
  - (H − bar) MAR minus control: +0.0104 [-0.0128, +0.0336] none
  - (H − strong) MAR minus control: +0.0410 [+0.0069, +0.0751] worse
  - (K − H) MAR minus control: -0.0043 [-0.0237, +0.0150] none
## `kc2` at 60nan
  - (H − bar) MAR minus control: -0.0725 [-0.1371, -0.0079] better
  - (H − strong) MAR minus control: -0.1124 [-0.1591, -0.0657] better
  - (K − H) MAR minus control: +0.0365 [-0.0030, +0.0760] none
## `vehicle` at 60nan
  - (H − bar) MAR minus control: +0.0029 [-0.0129, +0.0186] none
  - (H − strong) MAR minus control: +0.0057 [-0.0125, +0.0238] none
  - (K − H) MAR minus control: +0.0025 [-0.0117, +0.0167] none
## `kr-vs-kp` at 60nan
  - (H − bar) MAR minus control: +0.0098 [-0.0036, +0.0233] none
  - (H − strong) MAR minus control: +0.0103 [+0.0011, +0.0196] worse
  - (K − H) MAR minus control: -0.0018 [-0.0106, +0.0071] none
- (H − bar) MAR minus control: better 1 / no difference 3 / worse 0 (fold-pair t, seed agreement)
- (H − strong) MAR minus control: better 1 / no difference 1 / worse 2 (fold-pair t, seed agreement)
- (K − H) MAR minus control: better 0 / no difference 4 / worse 0 (fold-pair t, seed agreement)
Primary: worse MAR-minus-control contrasts: ['biodeg (H − strong) MAR minus control', 'kr-vs-kp (H − strong) MAR minus control'] -> loses ground (fold-pair t, seed agreement)
```

### E64: the MAR family (imputation-protocol/19)

```text
## `biodeg_20nan`
  - (H − bar) MAR minus control: +0.0091 [-0.0220, +0.0402] none
  - (H − strong) MAR minus control: -0.0012 [-0.0575, +0.0552] none
  - (K − H) MAR minus control: +0.0088 [-0.0211, +0.0387] none
## `biodeg_60nan`
  - (H − bar) MAR minus control: +0.0398 [+0.0186, +0.0610] worse
  - (H − strong) MAR minus control: +0.0291 [+0.0081, +0.0500] worse
  - (K − H) MAR minus control: -0.0080 [-0.0242, +0.0081] none
## `kc2_20nan`
  - (H − bar) MAR minus control: -0.1239 [-0.2481, +0.0003] none
  - (H − strong) MAR minus control: -0.0327 [-0.1303, +0.0648] none
  - (K − H) MAR minus control: -0.0471 [-0.1223, +0.0281] none
## `kc2_60nan`
  - (H − bar) MAR minus control: +0.0460 [+0.0058, +0.0862] worse
  - (H − strong) MAR minus control: -0.0079 [-0.0732, +0.0573] none
  - (K − H) MAR minus control: -0.0005 [-0.0532, +0.0521] none
## `vehicle_20nan`
  - (H − bar) MAR minus control: +0.0080 [-0.0271, +0.0431] none
  - (H − strong) MAR minus control: -0.0025 [-0.0433, +0.0382] none
  - (K − H) MAR minus control: +0.0037 [-0.0065, +0.0138] none
## `vehicle_60nan`
  - (H − bar) MAR minus control: +0.0258 [+0.0084, +0.0432] worse
  - (H − strong) MAR minus control: -0.0026 [-0.0329, +0.0277] none
  - (K − H) MAR minus control: -0.0016 [-0.0177, +0.0145] none
## `kr-vs-kp_20nan`
  - (H − bar) MAR minus control: -0.0079 [-0.0335, +0.0176] none
  - (H − strong) MAR minus control: -0.0070 [-0.0265, +0.0125] none
  - (K − H) MAR minus control: -0.0152 [-0.0340, +0.0037] none
## `kr-vs-kp_60nan`
  - (H − bar) MAR minus control: +0.0036 [-0.0097, +0.0170] none
  - (H − strong) MAR minus control: +0.0020 [-0.0091, +0.0131] none
  - (K − H) MAR minus control: -0.0021 [-0.0136, +0.0095] none
## `spambase_20nan`
  - (H − bar) MAR minus control: -0.0048 [-0.0185, +0.0089] none
  - (H − strong) MAR minus control: +0.0004 [-0.0099, +0.0107] none
  - (K − H) MAR minus control: +0.0062 [-0.0046, +0.0170] none
## `spambase_60nan`
  - (H − bar) MAR minus control: -0.0063 [-0.0166, +0.0039] none
  - (H − strong) MAR minus control: -0.0018 [-0.0127, +0.0092] none
  - (K − H) MAR minus control: +0.0014 [-0.0069, +0.0098] none
## `credit-g_20nan`
  - (H − bar) MAR minus control: -0.0142 [-0.0364, +0.0081] none
  - (H − strong) MAR minus control: +0.0160 [-0.0005, +0.0325] none
  - (K − H) MAR minus control: -0.0130 [-0.0375, +0.0114] none
## `credit-g_60nan`
  - (H − bar) MAR minus control: -0.0059 [-0.0275, +0.0157] none
  - (H − strong) MAR minus control: +0.0036 [-0.0123, +0.0196] none
  - (K − H) MAR minus control: -0.0041 [-0.0245, +0.0164] none
## The dial: `biodeg_60nan`, (H − strong) MAR minus control by slope
- (H − bar) MAR minus control: better 0 / no difference 9 / worse 3 (fold-pair t, seed agreement)
- (H − strong) MAR minus control: better 0 / no difference 11 / worse 1 (fold-pair t, seed agreement)
- (K − H) MAR minus control: better 0 / no difference 12 / worse 0 (fold-pair t, seed agreement)
Primary, (H − strong) MAR minus control: worse on ['biodeg_60nan'], better on none -> refuted (fold-pair t, seed agreement)
```

### E65: propensity-weighted decode masks (imputation-protocol/20)

```text
## `biodeg_20nan`
  - (P − G on H), MAR minus control: +0.0052 [-0.0340, +0.0444] none
  - G's (H − strong), MAR minus control: -0.0012 none
  - P's (H − strong), MAR minus control: +0.0040 none
## `biodeg_60nan`
  - (P − G on H), MAR minus control: -0.0063 [-0.0338, +0.0212] none
  - G's (H − strong), MAR minus control: +0.0291 worse
  - P's (H − strong), MAR minus control: +0.0228 none
## `kc2_20nan`
  - (P − G on H), MAR minus control: -0.0137 [-0.1001, +0.0726] none
  - G's (H − strong), MAR minus control: -0.0327 none
  - P's (H − strong), MAR minus control: -0.0464 none
## `kc2_60nan`
  - (P − G on H), MAR minus control: -0.0337 [-0.0811, +0.0136] none
  - G's (H − strong), MAR minus control: -0.0079 none
  - P's (H − strong), MAR minus control: -0.0417 none
## `vehicle_20nan`
  - (P − G on H), MAR minus control: -0.0079 [-0.0417, +0.0259] none
  - G's (H − strong), MAR minus control: -0.0025 none
  - P's (H − strong), MAR minus control: -0.0105 none
## `vehicle_60nan`
  - (P − G on H), MAR minus control: -0.0135 [-0.0362, +0.0092] none
  - G's (H − strong), MAR minus control: -0.0026 none
  - P's (H − strong), MAR minus control: -0.0161 none
## `kr-vs-kp_20nan`
  - (P − G on H), MAR minus control: -0.0012 [-0.0166, +0.0143] none
  - G's (H − strong), MAR minus control: -0.0070 none
  - P's (H − strong), MAR minus control: -0.0081 none
## `kr-vs-kp_60nan`
  - (P − G on H), MAR minus control: +0.0028 [-0.0119, +0.0174] none
  - G's (H − strong), MAR minus control: +0.0020 none
  - P's (H − strong), MAR minus control: +0.0047 none
## `spambase_20nan`
  - (P − G on H), MAR minus control: -0.0085 [-0.0232, +0.0063] none
  - G's (H − strong), MAR minus control: +0.0004 none
  - P's (H − strong), MAR minus control: -0.0081 none
## `spambase_60nan`
  - (P − G on H), MAR minus control: +0.0145 [+0.0041, +0.0248] worse
  - G's (H − strong), MAR minus control: -0.0018 none
  - P's (H − strong), MAR minus control: +0.0127 worse
## `credit-g_20nan`
  - (P − G on H), MAR minus control: +0.0023 [-0.0161, +0.0207] none
  - G's (H − strong), MAR minus control: +0.0160 none
  - P's (H − strong), MAR minus control: +0.0183 none
## `credit-g_60nan`
  - (P − G on H), MAR minus control: +0.0034 [-0.0149, +0.0216] none
  - G's (H − strong), MAR minus control: +0.0036 none
  - P's (H − strong), MAR minus control: +0.0070 none
## The dial: `biodeg_60nan`, P − G on H under MAR by slope
- P − G on H under MAR: better 0 / no difference 6 / worse 6 (fold-pair t, seed agreement)
- P − G on H under the control: better 0 / no difference 7 / worse 5 (fold-pair t, seed agreement)
Primary, P − G on H: under MAR better on none, worse on ['vehicle_20nan', 'kr-vs-kp_60nan', 'spambase_20nan', 'spambase_60nan', 'credit-g_20nan', 'credit-g_60nan']; under the control worse on ['kc2_60nan', 'vehicle_60nan', 'spambase_20nan', 'spambase_60nan', 'credit-g_20nan'] -> refuted (fold-pair t, seed agreement)
```

### E60: a TRIDENT-TabPFN blend (imputation-protocol/15)

```text
  - blend − model: better 14 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - blend − tabpfn: better 11 / no difference 1 / equivalent 3 / worse 0 (fold-pair t, seed agreement)
Primary: the blend beats both on 10 of 15 (['pendigits_40nan', 'pendigits_60nan', 'pendigits_80nan', 'letter_40nan', 'letter_60nan', 'letter_80nan', 'electricity_60nan', 'vehicle_40nan', 'biodeg_40nan', 'kc2_40nan']) -> complementary (fold-pair t, seed agreement)
```

### E61: where K loses to TabPFN, per column (imputation-protocol/16)

```text
## `kr-vs-kp_40nan`: K − TabPFN +0.0110 = categorical +0.0110 + numerical +0.0000
## `kr-vs-kp_60nan`: K − TabPFN +0.0434 = categorical +0.0434 + numerical +0.0000
## `kr-vs-kp_80nan`: K − TabPFN +0.0365 = categorical +0.0365 + numerical +0.0000
## `credit-g_40nan`: K − TabPFN +0.0219 = categorical +0.0155 + numerical +0.0065
## `credit-g_60nan`: K − TabPFN +0.0391 = categorical +0.0324 + numerical +0.0067
## `credit-g_80nan`: K − TabPFN +0.0139 = categorical +0.0113 + numerical +0.0025
Primary: stuck columns hold ≥ 50% of the net excess on 0 of 6 counted variants (none) -> no target
```

### E59: the bigger encoder at 40-80nan (imputation-protocol/14)

```text
Primary, ALL − G on K: better on ['pendigits_40nan', 'pendigits_60nan', 'letter_40nan'], worse on ['letter_80nan'] -> mixed (fold-pair t, seed agreement)
```

### E52: a frozen pre-trained encoder against a random one (imputation-pretraining/10)

```text
Primary, FN − FR on K: a usable pre-trained representation (FN better) on ['pendigits_20nan', 'kr-vs-kp_40nan']; FN worse on none; better 2 / no difference 1 / equivalent 1 / worse 0 (fold-pair t, seed agreement)
```

### E53: collapse and probes (imputation-pretraining/11)

```text
## Setting N
## Setting A
Primary: N collapsing: no (0 of 3); value information under N: present (2 of 3)
## Drift across today's decode stage (seed 101, fold 1)
```
