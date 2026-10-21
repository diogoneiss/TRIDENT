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

- Four CPU lanes at 4 threads in `/scratch2/diogoneiss/trident-2026-10-20-electricity-split/`.
- The GPU stays free.
