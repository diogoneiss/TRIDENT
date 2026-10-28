# What the imputation studies support, as of 2026-10-28 (consolidated)

This report gathers E44–E83 into one statement for the thesis. It was rebuilt 2026-10-28, 00:39 GMT-3 to add
E67–E83. It was built 2026-10-23, 07:59 GMT-3, and first built 2026-10-22, 15:12 GMT-3. It adds no
new analysis.

- **Open studies are left out of the statement:**
  - E84 (the spread under MNAR and MAR) is partial and completing.
  - E85 (the occlusion map at 20/80nan) is running.
  - Neither carries a verdict yet.
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

## The statement, in twelve lines

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
4. **TRIDENT's and TabPFN's errors are complementary, on numerical tables where the two are close
   (E60, E66).** An equal-weight average of their numerical fills beats both on 10 of 15
   confirmation variants. On E40's tables and seeds it does so on 3 of 9 (E66, at the rule's floor).
   Where one method dominates, the blend lands near the better one.
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
9. **The decoder gives a per-cell uncertainty rank, opt-in (ADR 0030; E75–E77, E80).**
   - The re-masking spread ranks numerical errors within a column up to 60nan: 14 of 15
     development variants, replicated (E76).
   - It does not make better intervals than a constant width (E77).
   - It runs the wrong way along a row's missingness (E80): it falls as the row has more gaps,
     while the error rises.
10. **The decoder's column dependencies can be read, and they are faithful (E78, E79).**
    - Hiding each source column as `[MASK]` gives a map.
    - Hiding a target's top 3 sources hurts more than hiding 3 random ones on 9 of 9 tables at
      40nan, and on 7 of 9 at 60nan (kc2 and credit-g inconclusive).
    - On a synthetic table with known blocks the map recovers them: AUROC 0.979 and 0.914, against
      0.847 from |Spearman|.
11. **A per-column hedge with TabPFN helps, but most of its gain is TabPFN's own (ADR 0031; E69,
    E74, E82).**
    - The KC + TabPFN hybrid is never worse than KC on the 10 hard variants (E74, mixed).
    - Its deployable form, with weights chosen without the complete table, beats H on 10 of 10
      and KC on 7, and trails TabPFN on 5 (E82, refuted).
    - On those variants the order is TabPFN ≳ deployable hybrid > KC > H.
12. **On classification, untuned TabPFN v2 leads TRIDENT under the fork's defaults (E83; ADR
    0032).**
    - The lead is +0.017 to +0.024 macro-F1 at every level.
    - Pre-training buys no detectable gain there (E71).
    - TabPFN is a separate strong tier; H1c's bar stays the best GBDT.

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
  - **E66** ([21](../tickets/imputation-protocol/21-tabpfn-blend-replication.md)), on E40's
    tables at 404/505/606: the blend beats both on 3 of 9 (spambase_80nan, vehicle_60nan,
    biodeg_60nan).
    - On spambase at 40–60nan, where K leads, it is equivalent to K.
    - On credit-g, where TabPFN's lead is in the categorical cells, it cannot reach TabPFN.
  - K rather than KC, and numerical cells only.
- **The categorical deficit (E61,** [16](../tickets/imputation-protocol/16-categorical-deficit.md)**):**
  - On kr-vs-kp and credit-g from 40nan, K's per-column gains over the mode correlate with
    TabPFN's at +0.76 to +0.94, a few points lower.
  - It is a uniform shortfall in conditional structure. T10's weighting half is closed.
- **Capacity (E59,** [14](../tickets/imputation-protocol/14-large-encoder.md)**):**
  - The bigger encoder costs about 2× the training time.
  - It does not move pendigits_80nan's gap to TabPFN (+0.0121 → +0.0104, still worse).

## 3b. Since 2026-10-23: E67–E83

Each line cites the study's ticket; the readers' summary lines are in the appendix.

- **Mechanisms and imputers:**
  - **`[NULL]` alone in the decode stage** costs more as the MNAR tilt steepens (E67, refuted;
    [imputation-protocol/22](../tickets/imputation-protocol/22-null-gaps.md)).
  - **The tuned bar on the MAR folds:** the model is never behind it on E64's family; worse on 3
    of 12, better on 2 (E68, mixed; [23](../tickets/imputation-protocol/23-mar-tuned-bar.md)).
  - **not-MIWAE,** which models the mechanism, shrinks the MNAR bias against its own twin on 3 of
    10, but its fills fail on biodeg and spambase (E72, mixed; E73;
    [25](../tickets/imputation-protocol/25-not-miwae.md),
    [26](../tickets/imputation-protocol/26-notmiwae-fill.md)).
- **The hybrid with TabPFN:**
  - E69 (cross-fitted per-column weights, mixed; [24](../tickets/imputation-protocol/24-blend-weights.md));
  - E74 (over KC, fresh seeds: better than KC on 7 of 10, mixed;
    [27](../tickets/imputation-protocol/27-hybrid-output.md));
  - ADR 0031: opt-in, offline, a benchmark output;
  - E82 (deployable: refuted against TabPFN; [31](../tickets/imputation-protocol/31-deployable-hybrid.md)).
- **Uncertainty:**
  - E75 (the averaged fill is not an output; the spread ranks errors 12 of 15;
    [28](../tickets/imputation-protocol/28-remask-ensemble.md));
  - E76 (supported, within columns; [29](../tickets/imputation-protocol/29-remask-development.md));
  - E77 (refuted as intervals; [30](../tickets/imputation-protocol/30-remask-conformal.md));
  - ADR 0030, amended with E80's limit.
- **Interpretability (T38):**
  - E78 (supported at 40nan;
    [imputation-interpretability/01](../tickets/imputation-interpretability/01-occlusion-map.md));
  - E79 (supported at the bar at 60nan; [02](../tickets/imputation-interpretability/02-occlusion-map-60nan.md));
  - E80 (mixed; [03](../tickets/imputation-interpretability/03-spread-and-missing-sources.md));
  - E81 (mixed; [04](../tickets/imputation-interpretability/04-gap-reliance.md)).
  - **E81's structural finding:** under the default (`DECODE_GAP_TOKEN mask`, ADR 0014) a real gap
    and a hidden cell are one token, so the default cannot read the gap pattern.
  - A decoder that can read it (`--decode_gap_token null`) does so under moderate self-masking,
    and is less accurate on 11 of 11.
- **Classification (T27, T40):**
  - E70: the old sweep does not reproduce fold by fold;
  - E71: no pre-training gain detected
    ([classification-step-zero/02](../tickets/classification-step-zero/02-no-pretraining.md));
  - E83: TabPFN v2 ahead ([03](../tickets/classification-step-zero/03-tabpfn-classifier.md));
  - ADR 0032: TabPFN is a separate strong tier.
  - Step zero itself waits on ADR 0029's eight decisions.

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
- **One seed set or one draw:** E61 (the per-column read) rests on one seed set. The blend's
  complementarity (E60, E66) holds on two, narrower on the second. Every per-variant interval is conditional on the MCAR draw (E62).
- **MAR:** the standing against the strong tier holds on six tables (E64), with biodeg_60nan the
  exception on two draws. The bar gains at 60nan on the small numerical tables.
- **The spread (ADR 0030)** is a rank within a column, not an interval (E77). It falls along a row's
  missingness (E80). Under MAR and MNAR it is not yet established, since E84 is partial.
- **The occlusion map** is read on K only, under MCAR, at 40 and 60nan. It is not shown on kc2 and
  credit-g at 60nan, and its effects shrink with missingness.
- **The hybrid's deployable form** does not reach TabPFN on the small and categorical tables (E82).
- **Torch:** every run up to 2026-10-27 used torch 2.5.1. From 2026-10-28 runs use 2.11.0 (ADR 0033's
  preparation), which draws other numbers from the same seed. Studies that span the change say so.

## 5. Open

- **E84 (partial) and E85 (running).** Both are read only when complete, and are not in this
  statement.
- **T27's classification step zero:** ADR 0029 and its eight decisions, deferred by the user on
  2026-10-27. T41: the paper's Table 4 with its parameters.
- **The hybrid's selection:** one weight per table, or shrinkage toward the equal blend, which is
  offline on E82's files; a claim needs fresh seeds.
- **T32:** the bigger encoder at 40–60nan, kept for later by the user.
- **The backlog:** T10's evaluation-mask half, T08, T09, T11, T14 and T15.

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
Primary: the blend beats both on 10 of 15 counted (['pendigits_40nan', 'pendigits_60nan', 'pendigits_80nan', 'letter_40nan', 'letter_60nan', 'letter_80nan', 'electricity_60nan', 'vehicle_40nan', 'biodeg_40nan', 'kc2_40nan']) -> complementary (fold-pair t, seed agreement)
```

### E66: the blend on E40's tables and seeds (imputation-protocol/21)

```text
  - blend − model: better 6 / no difference 1 / equivalent 5 / worse 0 (fold-pair t, seed agreement)
  - blend − tabpfn: better 6 / no difference 0 / worse 6 (fold-pair t, seed agreement)
Primary: the blend beats both on 3 of 9 counted (['spambase_80nan', 'vehicle_60nan', 'biodeg_60nan']) -> complementary (fold-pair t, seed agreement)
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

### E67: the gaps as [NULL] in the decode stage, alone (imputation-protocol/22)

```text
## `biodeg_probit0p25_60nan` against `biodeg_60nan`
## `biodeg_probit_60nan` against `biodeg_60nan`
## `biodeg_probit1p0_60nan` against `biodeg_60nan`
## `biodeg_probit2p0_60nan` against `biodeg_60nan`
## `spambase_probit_60nan` against `spambase_60nan`
## `biodeg_probit_20nan` against `biodeg_20nan`
## `kc2_probit_20nan` against `kc2_20nan`
## `vehicle_probit_20nan` against `vehicle_20nan`
## The dial (biodeg at 60nan), (N − G on H), MNAR minus MCAR by slope
- (N − G on H), MNAR minus MCAR: better 1 / no difference 5 / worse 2 (fold-pair t, seed agreement)
- (N − G on K), MNAR minus MCAR: better 1 / no difference 4 / worse 3 (fold-pair t, seed agreement)
Primary, (N − G on H), MNAR minus MCAR: better on ['biodeg_probit0p25_60nan'], worse on ['biodeg_probit1p0_60nan', 'biodeg_probit2p0_60nan'] -> refuted (fold-pair t, seed agreement)
```

### E68: the tuned bar on the MAR folds (imputation-protocol/23)

```text
## `biodeg_20nan`
  - (H − tuned bar) MAR minus control: +0.0110 [-0.0228, +0.0448] none
  - (K − tuned bar) MAR minus control: +0.0198 [-0.0176, +0.0571] none
## `biodeg_60nan`
  - (H − tuned bar) MAR minus control: +0.0365 [+0.0137, +0.0593] worse
  - (K − tuned bar) MAR minus control: +0.0285 [+0.0091, +0.0479] worse
## `kc2_20nan`
  - (H − tuned bar) MAR minus control: -0.1268 [-0.2491, -0.0045] better
  - (K − tuned bar) MAR minus control: -0.1739 [-0.2895, -0.0583] better
## `kc2_60nan`
  - (H − tuned bar) MAR minus control: +0.0431 [+0.0012, +0.0851] worse
  - (K − tuned bar) MAR minus control: +0.0426 [-0.0026, +0.0878] none
## `vehicle_20nan`
  - (H − tuned bar) MAR minus control: +0.0176 [-0.0181, +0.0533] none
  - (K − tuned bar) MAR minus control: +0.0213 [-0.0179, +0.0605] none
## `vehicle_60nan`
  - (H − tuned bar) MAR minus control: +0.0259 [+0.0062, +0.0455] worse
  - (K − tuned bar) MAR minus control: +0.0243 [+0.0062, +0.0424] worse
## `kr-vs-kp_20nan`
  - (H − tuned bar) MAR minus control: -0.0053 [-0.0298, +0.0193] none
  - (K − tuned bar) MAR minus control: -0.0205 [-0.0430, +0.0021] none
## `kr-vs-kp_60nan`
  - (H − tuned bar) MAR minus control: +0.0103 [-0.0020, +0.0227] none
  - (K − tuned bar) MAR minus control: +0.0082 [-0.0087, +0.0251] none
## `spambase_20nan`
  - (H − tuned bar) MAR minus control: -0.0065 [-0.0191, +0.0061] none
  - (K − tuned bar) MAR minus control: -0.0003 [-0.0139, +0.0132] none
## `spambase_60nan`
  - (H − tuned bar) MAR minus control: -0.0101 [-0.0194, -0.0008] better
  - (K − tuned bar) MAR minus control: -0.0087 [-0.0164, -0.0010] better
## `credit-g_20nan`
  - (H − tuned bar) MAR minus control: +0.0039 [-0.0225, +0.0303] none
  - (K − tuned bar) MAR minus control: -0.0091 [-0.0318, +0.0136] none
## `credit-g_60nan`
  - (H − tuned bar) MAR minus control: -0.0093 [-0.0243, +0.0058] none
  - (K − tuned bar) MAR minus control: -0.0133 [-0.0265, -0.0001] better
- (H − tuned bar) MAR minus control: better 2 / no difference 7 / worse 3 (fold-pair t, seed agreement)
- (K − tuned bar) MAR minus control: better 3 / no difference 7 / worse 2 (fold-pair t, seed agreement)
## `biodeg_60nan`
  - (H − tuned bar) MAR minus control: +0.0085 [-0.0185, +0.0354] none
  - (K − tuned bar) MAR minus control: +0.0042 [-0.0162, +0.0245] none
## `kc2_60nan`
  - (H − tuned bar) MAR minus control: -0.0691 [-0.1349, -0.0034] better
  - (K − tuned bar) MAR minus control: -0.0326 [-0.0850, +0.0198] none
## `vehicle_60nan`
  - (H − tuned bar) MAR minus control: -0.0002 [-0.0156, +0.0152] none
  - (K − tuned bar) MAR minus control: +0.0023 [-0.0154, +0.0200] none
## `kr-vs-kp_60nan`
  - (H − tuned bar) MAR minus control: +0.0083 [-0.0021, +0.0186] none
  - (K − tuned bar) MAR minus control: +0.0065 [-0.0049, +0.0179] none
- (H − tuned bar) MAR minus control: better 1 / no difference 3 / worse 0 (fold-pair t, seed agreement)
- (K − tuned bar) MAR minus control: better 0 / no difference 4 / worse 0 (fold-pair t, seed agreement)
Primary, E64's (H − tuned bar) MAR minus control: worse on ['biodeg_60nan', 'kc2_60nan', 'vehicle_60nan'], better on ['kc2_20nan', 'spambase_60nan'] -> mixed (fold-pair t, seed agreement)
```

### E69: per-column weights for the K + TabPFN blend (imputation-protocol/24)

```text
Primary, weighted − TabPFN on the 10 hard variants: better on ['electricity_80nan', 'vehicle_80nan', 'biodeg_80nan'], worse on ['kr-vs-kp_60nan'] -> mixed (fold-pair t, seed agreement)
```

### E71: classification, what pre-training buys (classification-step-zero/02)

```text
## B − A, test macro-F1
## C − A, test macro-F1
```

### E72: not-MIWAE under MNAR (imputation-protocol/25)

```text
## `biodeg_probit_60nan` against `biodeg_60nan`
  - (notmiwae − miwae on abs_bias), MNAR minus MCAR: -0.0014 [-0.1177, +0.1149] none
## `kc2_probit_60nan` against `kc2_60nan`
  - (notmiwae − miwae on abs_bias), MNAR minus MCAR: +0.0038 [-0.0120, +0.0195] none
## `vehicle_probit_60nan` against `vehicle_60nan`
  - (notmiwae − miwae on abs_bias), MNAR minus MCAR: -0.0727 [-0.0837, -0.0618] better
## `spambase_probit_60nan` against `spambase_60nan`
  - (notmiwae − miwae on abs_bias), MNAR minus MCAR: +0.0258 [-0.0519, +0.1035] none
## `biodeg_probit_20nan` against `biodeg_20nan`
  - (notmiwae − miwae on abs_bias), MNAR minus MCAR: +0.0207 [-0.2803, +0.3218] none
## `kc2_probit_20nan` against `kc2_20nan`
  - (notmiwae − miwae on abs_bias), MNAR minus MCAR: +0.0077 [-0.0118, +0.0271] none
## `vehicle_probit_20nan` against `vehicle_20nan`
  - (notmiwae − miwae on abs_bias), MNAR minus MCAR: -0.0201 [-0.0355, -0.0047] better
## `biodeg_probit0p25_60nan` against `biodeg_60nan`
  - (notmiwae − miwae on abs_bias), MNAR minus MCAR: -0.0337 [-0.1005, +0.0332] none
## `biodeg_probit1p0_60nan` against `biodeg_60nan`
  - (notmiwae − miwae on abs_bias), MNAR minus MCAR: -0.0446 [-0.1851, +0.0959] none
## `biodeg_probit2p0_60nan` against `biodeg_60nan`
  - (notmiwae − miwae on abs_bias), MNAR minus MCAR: -0.1714 [-0.2668, -0.0760] better
Primary, (notmiwae − miwae on abs_bias), MNAR minus MCAR: better on ['vehicle_probit_60nan', 'vehicle_probit_20nan', 'biodeg_probit2p0_60nan'], worse on none -> mixed (fold-pair t, seed agreement)
```

### E74: the KC + TabPFN hybrid as an output (imputation-protocol/27)

```text
Primary: the hybrid is better than KC on 7 of 10 and worse than TabPFN on 2 (['kr-vs-kp_40nan', 'kr-vs-kp_60nan']) -> mixed (fold-pair t, seed agreement)
```

### E75: a test-time re-masking ensemble (imputation-protocol/28)

```text
Primary: remask better than the single pass on 4 (K) and 6 (H) of 15, worse on ['pendigits_40nan', 'pendigits_60nan', 'letter_40nan', 'letter_60nan'] (K) and ['pendigits_40nan', 'pendigits_60nan', 'letter_40nan', 'letter_60nan'] (H) -> mixed (fold-pair t, seed agreement)
```

### E76: the spread on the development variants (imputation-protocol/29)

```text
Primary: within-column signal on 14 (K) and 14 (H) of 15 variants up to 60nan, ranking against the errors on none (K) -> supported
```

### E77: intervals from the spread (imputation-protocol/30)

```text
Primary: the spread interval is narrower than the constant one at the same target on 0 of 15 (K), wider on ['pendigits_40nan', 'pendigits_60nan', 'pendigits_80nan', 'letter_40nan', 'letter_60nan', 'letter_80nan', 'electricity_40nan', 'electricity_60nan', 'electricity_80nan', 'vehicle_40nan', 'vehicle_80nan', 'biodeg_40nan', 'biodeg_80nan', 'kc2_40nan', 'kc2_80nan'] -> refuted (fold-pair t, seed agreement)
```

### E78: the occlusion map at 40nan (imputation-interpretability/01)

```text
  - descriptive: |Spearman| on the training split, same AUROC (numerical pairs only): 0.847 [0.831, 0.863]
Primary (B): faithful on 9 of 9, not faithful on none -> supported (fold-pair t, seed agreement, SESOI 0.01)
```

### E79: the occlusion map at 60nan (imputation-interpretability/02)

```text
  - descriptive: |Spearman| on the training split, same AUROC (numerical pairs only): 0.864 [0.849, 0.880]
Primary (B): faithful on 7 of 9, not faithful on none -> supported (fold-pair t, seed agreement, SESOI 0.01)
```

### E80: the spread and missing top sources (imputation-interpretability/03)

```text
Primary: specific on 7 of 16, against on ['kc2_40nan', 'credit-g_40nan', 'electricity_60nan', 'kc2_60nan', 'spambase_60nan', 'credit-g_60nan'] -> mixed (fold-pair t, seed agreement)
```

### E81: gap-pattern reliance under MNAR (imputation-interpretability/04)

```text
Primary: S at slope 2.0 +0.0049 [+0.0015, +0.0083], at MCAR -0.0022 [-0.0031, -0.0012] -> mixed (SESOI 0.01)
```

### E82: the deployable hybrid (imputation-protocol/31)

```text
Primary: the deployable hybrid is better than H on 10 of 10, worse on none, and worse than TabPFN on 5 (['vehicle_80nan', 'biodeg_80nan', 'kr-vs-kp_60nan', 'kr-vs-kp_80nan', 'credit-g_60nan']) -> refuted (fold-pair t, seed agreement)
```

### E83: TabPFN v2's classifier (classification-step-zero/03)

```text
- 0nan: TabPFN − TRIDENT macro-F1 +0.0212 [+0.0136, +0.0288] seeds +0.0178, +0.0246, 48 fold pairs -> TabPFN ahead
- 20nan: TabPFN − TRIDENT macro-F1 +0.0186 [+0.0108, +0.0264] seeds +0.0160, +0.0212, 48 fold pairs -> TabPFN ahead
- 40nan: TabPFN − TRIDENT macro-F1 +0.0170 [+0.0018, +0.0322] seeds +0.0072, +0.0268, 48 fold pairs -> TabPFN ahead
- 60nan: TabPFN − TRIDENT macro-F1 +0.0240 [+0.0055, +0.0426] seeds +0.0268, +0.0213, 48 fold pairs -> TabPFN ahead
- 80nan: TabPFN − TRIDENT macro-F1 +0.0236 [+0.0064, +0.0408] seeds +0.0260, +0.0212, 48 fold pairs -> TabPFN ahead
```
