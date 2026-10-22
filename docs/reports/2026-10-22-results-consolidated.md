# What the imputation studies support, as of 2026-10-22 (consolidated)

This report gathers E44–E63 into one statement for the thesis (built 2026-10-22, 15:12 GMT-3).
It adds no new analysis.

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
7. **The standing holds across MCAR draws and under MNAR, and only partly under MAR.**
   - **Under MNAR (E45, E46, E58),** H is never worse than TabPFN (6 better, 4 no difference), and
     TabPFN gains no ground over MCAR.
   - **Under MAR (E63), a smoke:** H keeps its standing against the bar, but TabPFN gains ground
     on two of four tables (biodeg, kr-vs-kp) and loses it on kc2.
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
- **MAR (E63,** [18](../tickets/imputation-protocol/18-mar-smoke.md)**), a smoke:**
  - **The tables:** biodeg, kc2, vehicle and kr-vs-kp at 60nan.
  - **The mechanism:** 30% fully observed driver columns decide which cells go in the others.
    Each MAR variant is read against an MCAR control with the same drivers and counts.
  - **(H − bar), MAR minus control:** never worse (better on kc2).
  - **(H − strong), MAR minus control:** worse on biodeg (+0.0410) and kr-vs-kp (+0.0103), better
    on kc2, no difference on vehicle. The verdict is "loses ground".
  - Unlike MNAR, missingness that depends on observed columns can favour TabPFN.
  - **An untested hypothesis:** the decode masks are uniform over observed cells, while the test
    gaps follow the drivers.

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
- **MAR:** against TabPFN the model can lose ground (E63; four tables, one rate, one strength).

## 5. Open

- **T27's classification step zero:** a native-NaN GBDT bar, impute-plus-indicator arms and a
  no-pre-training arm. It needs code and an ADR first.
- **T32:** the bigger encoder at 40–60nan, about 4% at 2× the training time, kept for later by
  the user.
- **T10's evaluation-mask half;** T08, T09, T11, T14 and T15 in the backlog.
- **A replication of E60** on another seed set, if the hybrid argument enters the discussion.
- **A fuller MAR family** (E63's proposal, for the user): the 20nan rung, a slope dial, spambase and
  credit-g, and an arm whose decode masks follow the drivers.

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

### E63: MAR against matched MCAR controls (imputation-protocol/18)

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
