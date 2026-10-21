# T29's test: is the boosters' lead on electricity carried by `date` as a key? (E54; pre-registered 2026-10-20)

This ticket was written and committed before any subset in it was computed.

- **Why:** T29's report ([2026-10-20](../../reports/2026-10-20-t29-electricity.md)) finds the model
  trailing hgb and lgbm on most columns, most on `day` and `nswdemand`. `date` is a 933-level key:
  48 rows per date, and `day` is constant within a date.
  - **The hypothesis:** the boosters' lead comes from looking values up by `date`, and the model's
    scalar embedding of `date` cannot do that.
  - **The scope:** T29 is "a short report, then at most one pre-registered test"; this is the test.
- **Offline, on the CPU:** `scripts/experiments/electricity_split.py`. Nothing is written to the
  store or to `src/`.
  - **The model's fills** are the headline K's, from each run's fold ledger.
  - **The boosters' fills** come from `hgb` and `lgbm`, refitted on the same training folds and
    asked for exactly the ledger's cells.
  - **Integrity:** on all the cells, the model's and both boosters' `impute_score` must equal the
    run's fold table to 1e-9, or the cell stops.

## Design (fixed before the first subset)

- **Runs:**
  - electricity_20nan from E51 (seeds 1001/2002/3003);
  - electricity_40, 60 and 80nan from E44 (seeds 707/808/909).
  - 12 (variant, seed) cells, 5 folds each.
- **Subsets of the induced cells,** each scored by `impute_score` against the fold's mean/mode
  fill:
  - **Primary:** the cells of the seven columns other than `date`, split by whether the row's
    `date` is observed. A `date` cell's own row never has its date, so those cells are left out
    of both sides.
  - **Secondary:**
    - the same split for the `day` cells and the `nswdemand` cells alone;
    - the regime split: rows 0–17,423, with the three Victoria columns constant, against the rest;
    - all cells.
- **The opponent:** each run's bar is the booster that the full-cell rule picked for it
  (`report_common.opponent`). It is not chosen again on a subset.

## Reading, fixed now

On K − bar over 15 fold pairs, with the verdict rule and the 1% SESOI (`--tally`), at the two rates
where K − bar is worse on all cells, 20nan and 60nan:

- **supported:** on both rates, the date-observed side reads worse and the date-missing side reads
  not worse (no difference, equivalent or better);
- **refuted:** the date-missing side is also worse on both rates;
- **mixed:** otherwise.
- **Also reported:** 40 and 80nan, the per-column splits and the regime split, without a rule.

**What it leads to (no default changes either way):**

- **supported:** the thesis states the limit with its mechanism.
  - Any response on the model's side is a protected training change, needing an ADR and the user's
    decision: `date` as a category, a periodic encoding, a key-aware input.
- **refuted or mixed:** the limit stays stated without a mechanism. T29 closes.

## Launch

- Four CPU lanes at 4 threads in `/scratch2/diogoneiss/trident-2026-10-20-electricity-split/`,
  launched at 23:13 GMT-3 on 2408f9e. The GPU stayed free.

## Outcome (2026-10-20, 23:16 GMT-3)

- **All 12 (variant, seed) cells finished** between 23:13 and 23:15 GMT-3.
- **Integrity passed on all 60 folds.** The model's, hgb's and lgbm's full-cell `impute_score`
  equal the runs' fold tables to 1e-9.

**The printout** (`scripts/experiments/electricity_split.py --tally`, unedited). Each line is a mean
difference over 15 fold pairs of `impute_score` on that subset's cells (lower is better), with the
95% interval and the call.

```text
=== electricity_20nan (bars: hgb, lgbm, hgb)
  K − bar, all (≈14499 cells a fold): +0.0138 [+0.0023, +0.0254] worse
  K − bar, other7_date_observed (≈10170 cells a fold): +0.0252 [+0.0102, +0.0402] worse
  K − bar, other7_date_missing (≈2516 cells a fold): -0.0341 [-0.0455, -0.0227] better
  K − bar, day_date_observed (≈1439 cells a fold): +0.1325 [+0.1245, +0.1406] worse
  K − bar, day_date_missing (≈373 cells a fold): +0.0056 [-0.0092, +0.0204] none
  K − bar, nswdemand_date_observed (≈1460 cells a fold): +0.0881 [+0.0805, +0.0956] worse
  K − bar, nswdemand_date_missing (≈352 cells a fold): +0.0048 [-0.0010, +0.0107] none
  K − bar, regime (≈5575 cells a fold): +0.0568 [+0.0513, +0.0623] worse
  K − bar, rest (≈8923 cells a fold): -0.0011 [-0.0163, +0.0141] none
=== electricity_40nan (bars: hgb, hgb, lgbm)
  K − bar, all (≈29000 cells a fold): +0.0436 [-0.0013, +0.0885] none
  K − bar, other7_date_observed (≈15205 cells a fold): +0.0253 [+0.0075, +0.0430] worse
  K − bar, other7_date_missing (≈10169 cells a fold): +0.0542 [-0.0481, +0.1566] none
  K − bar, day_date_observed (≈2167 cells a fold): +0.0811 [+0.0720, +0.0903] worse
  K − bar, day_date_missing (≈1457 cells a fold): -0.0103 [-0.0169, -0.0037] better
  K − bar, nswdemand_date_observed (≈2165 cells a fold): +0.0729 [+0.0636, +0.0822] worse
  K − bar, nswdemand_date_missing (≈1459 cells a fold): -0.0027 [-0.0058, +0.0005] equivalent
  K − bar, regime (≈11161 cells a fold): +0.0280 [+0.0247, +0.0313] worse
  K − bar, rest (≈17838 cells a fold): +0.0457 [-0.0123, +0.1036] none
=== electricity_60nan (bars: hgb, lgbm, hgb)
  K − bar, all (≈43499 cells a fold): +0.0112 [+0.0059, +0.0164] worse
  K − bar, other7_date_observed (≈15199 cells a fold): +0.0334 [+0.0227, +0.0440] worse
  K − bar, other7_date_missing (≈22862 cells a fold): -0.0016 [-0.0110, +0.0077] none
  K − bar, day_date_observed (≈2166 cells a fold): +0.0526 [+0.0465, +0.0587] worse
  K − bar, day_date_missing (≈3270 cells a fold): -0.0024 [-0.0087, +0.0039] equivalent
  K − bar, nswdemand_date_observed (≈2158 cells a fold): +0.0672 [+0.0596, +0.0749] worse
  K − bar, nswdemand_date_missing (≈3278 cells a fold): -0.0002 [-0.0032, +0.0028] equivalent
  K − bar, regime (≈16749 cells a fold): +0.0139 [+0.0114, +0.0163] worse
  K − bar, rest (≈26749 cells a fold): +0.0098 [+0.0035, +0.0161] worse
=== electricity_80nan (bars: hgb, lgbm, hgb)
  K − bar, all (≈58000 cells a fold): +0.0004 [-0.0093, +0.0100] none
  K − bar, other7_date_observed (≈10144 cells a fold): +0.0546 [+0.0151, +0.0941] worse
  K − bar, other7_date_missing (≈40605 cells a fold): -0.0127 [-0.0173, -0.0080] better
  K − bar, day_date_observed (≈1446 cells a fold): +0.0451 [+0.0396, +0.0505] worse
  K − bar, day_date_missing (≈5803 cells a fold): -0.0097 [-0.0155, -0.0039] better
  K − bar, nswdemand_date_observed (≈1452 cells a fold): +0.0479 [+0.0366, +0.0591] worse
  K − bar, nswdemand_date_missing (≈5798 cells a fold): +0.0010 [-0.0002, +0.0022] equivalent
  K − bar, regime (≈22286 cells a fold): +0.0008 [-0.0010, +0.0026] equivalent
  K − bar, rest (≈35713 cells a fold): -0.0002 [-0.0126, +0.0121] none

Primary (20 and 60nan, the rates where K − bar is worse): date observed ['worse', 'worse'], date missing ['better', 'none'] -> supported (fold-pair t, seed agreement)
```

**Reading, by the rule fixed in advance: supported.** At 20nan and 60nan, the boosters' lead lives
in the cells whose row has its `date`.

| rate | date observed | date missing |
|---|---|---|
| 20nan | +0.0252, worse | −0.0341, better |
| 60nan | +0.0334, worse | −0.0016, no difference |

At 20nan, without the date, the model beats the boosters.

**Secondary, without a rule:**

- **The pattern holds at every rate.** The date-observed side is worse at 40 and 80nan too (+0.025
  and +0.055). The date-missing side is never worse: no difference at 40nan, better at 80nan
  (−0.013).
- **`day` shows the mechanism most plainly.** With the date observed, the boosters lead by
  +0.13, +0.08, +0.05 and +0.05 at 20, 40, 60 and 80nan. Without it, the lead disappears: no
  difference, then better, equivalent and better.
- **`nswdemand` shows it too:** +0.09 to +0.05 with the date, and no difference or equivalent
  without it.
- **The constant regime is not the mechanism.** The regime rows carry the 20nan loss (+0.057
  against no difference on the rest). But at 60nan both halves are worse, and at 80nan both tie.
  The date split separates the two sides at every rate; the regime split does not.

**What it means:** electricity's 45,312 rows are a half-hourly series, flattened. `date` is a
933-level key that the boosters read with splits, looking each hidden value up in the same day's
other training rows. The model reads `date` as one scalar through a 1 → 16 → 128 MLP, which cannot
resolve 933 keys. Where the key is absent, the model is the better imputer.

**What it leads to (fixed in advance):**

- **The thesis states the limit with its mechanism:** on a table where one column is a
  high-cardinality key shared by groups of rows, the boosters look values up by it, and TRIDENT's
  scalar embedding does not.
- **Any response on the model's side is a protected training change,** with an ADR and the user's
  decision. Examples: `date` as a category, a periodic or positional encoding of numbers, a
  key-aware input.
- **T29 closes.** Nothing changes a default.
- **Limits:**
  - one table;
  - the bar's own booster, untuned (E42's tuned bar is not split here);
  - E51's and E44's seeds.

