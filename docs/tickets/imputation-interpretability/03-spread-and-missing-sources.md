# Does the spread grow when a cell's top sources are missing? (E80; pre-registered 2026-10-27)

This ticket was written and committed before the reading ran.

- **Why:** ADR 0030's spread says how sure a numerical fill is, and E78/E79's map says what it is
  filled from. If the two describe one mechanism, a cell should be less sure when the columns its
  fill relies on are missing in its row. That should hold beyond the row simply missing more.
- **The user's go:** "otimo, quero fazer E e F agora, vá" (2026-10-27, before 08:29 GMT-3). This is
  study E of T38's proposal.
- **Code:** no change to `src/` and no training. `scripts/experiments/occlusion_spread.py` reads E78's
  and E79's fold ledgers and validation maps.
- **What was already seen (disclosure):** both studies' printouts were read: the maps, D, the AUROC
  and the top sources. No spread-related measure was computed on these cells before this ticket.
  `--check` ran first, and printed only the alignment counts.

## Design (fixed before the reading)

- **Data:** E78's and E79's cells.
  - Scope: the 16 real variants with numerical gaps, the 8 at 40nan and the 8 at 60nan. kr-vs-kp
    has no numerical gaps and is out.
  - synthblock at 40 and 60nan is descriptive.
- **The row's gaps:** read from the fold ledger's `induced` rows, which hold every gap of the test
  split; categories the variant never shows are included.
  - `--check` on pendigits fold 1 at 40 and 60nan: no duplicate cells, and every `induced_remask`
    row is inside the gap rows. The ledger's `row` is the test split's position, the same index the
    occlusion views use.
- **The spread:** each numerical gap's `induced_remask` row, at K.
  - H is not read. These runs did not use `--score_checkpoints`, so there is no
    `induced_remask_loss_checkpoint`. A claim from this study says "K" (ADR 0024).

## Measure and rule, fixed now (`occlusion_spread.py`)

- **One fold's value:** for each numerical target j, take its top 3 sources by the fold's validation
  map, as E78's B ranks them.
  - Compute Spearman, over j's test gap cells, between the spread and how many of those 3 are
    missing in the cell's row.
  - Subtract the mean of the same correlation over 20 draws of 3 random other columns, from a private
    stream seeded by `[seed, fold, 80]`.
  - Average over the targets; a correlation is NaN when either side is constant or there are fewer
    than 10 cells.
- **The verdict on each variant (15 folds):**
  - **specific** if the 95% t-interval lies above 0 and every seed's mean is above 0;
  - **against** if it lies below 0 with every seed below 0;
  - no signal otherwise.
- **The rule over the 16:**
  - **supported:** specific on 12 or more, and against on none;
  - **refuted:** specific on 6 or fewer;
  - **mixed:** otherwise.
- **Descriptive:**
  - Spearman of the spread with the top-3 missing count, and with the row's total gaps;
  - Spearman of the absolute error with the top-3 missing count;
  - on synthblock, the same measure with the true block as the top 3.

**What each reading leads to:**
- **Supported:** the thesis can tie the two outputs together. The spread is higher where the map's
  sources are missing, more than the row's missingness explains.
- **Otherwise:** they are reported as separate signals. No default changes either way.

## Outcome (2026-10-27, 08:33 GMT-3): mixed

- **Primary: mixed.**
  - **Specific (7 of 16):** pendigits, letter, vehicle and biodeg at 40nan; pendigits, letter and
    vehicle at 60nan. Differences +0.014 to +0.056.
  - **Against (6):** kc2 and credit-g at both rates, electricity_60nan and spambase_60nan. Down to
    −0.295 on credit-g_60nan.
  - **No signal:** the other 3.
- **Descriptive, and the part that matters:**
  - **Error rises where the map's sources are missing.** The absolute error's Spearman with the
    top-3 missing count is positive on all 16 (+0.016 to +0.186), which agrees with E78/E79's B.
  - **The spread falls where the row is missing more.** Its Spearman with the row's total gaps is
    negative on all 16 real variants (−0.04 to −0.44) and on synthblock.
  - With the top-3 count it is about 0 or negative on 15 of 16 (biodeg_40nan +0.016).
  - So the spread moves *against* the error along this axis. Where the specific mark appears, it
    only means the top 3 lower the spread *less* than random columns do.
- **A reading of why (post hoc, not tested here):** the re-masking passes hide a share of the row's
  *observed* cells.
  - A row with fewer observed cells gives the passes less to vary, and a fill with little
    information sits near the column's mean whichever cells are hidden.
  - So the passes agree most where the model knows least.
- **What it leads to, as fixed:** the two outputs are reported as separate signals; no default
  changes.
  - The descriptive pattern is a limit on ADR 0030 (accepted): within a column the spread ranks
    errors (E75/E76), but along a row's missingness it runs the wrong way.
  - A cell whose informative sources are all missing can get a *low* spread and a large error.
  - Whether to record that limit in ADR 0030 is the user's decision; this study does not edit the
    ADR.

The reader's printout (`occlusion_spread.py`), unedited:

```text
- `synthblock_40nan`: top-3 − random-3 -0.103 [-0.118, -0.087] seeds -0.108, -0.104, -0.096 -> against; descriptive: ρ(spread, top-3 missing) -0.219, ρ(spread, row's gaps) -0.254, ρ(error, top-3 missing) +0.127; with the true block as the top 3: -0.093 (descriptive)
- `pendigits_40nan`: top-3 − random-3 +0.019 [+0.014, +0.024] seeds +0.020, +0.015, +0.022 -> specific; descriptive: ρ(spread, top-3 missing) -0.002, ρ(spread, row's gaps) -0.044, ρ(error, top-3 missing) +0.154
- `letter_40nan`: top-3 − random-3 +0.036 [+0.033, +0.040] seeds +0.035, +0.038, +0.037 -> specific; descriptive: ρ(spread, top-3 missing) -0.034, ρ(spread, row's gaps) -0.150, ρ(error, top-3 missing) +0.102
- `electricity_40nan`: top-3 − random-3 +0.004 [-0.003, +0.010] seeds +0.005, +0.006, -0.001 -> no signal; descriptive: ρ(spread, top-3 missing) -0.156, ρ(spread, row's gaps) -0.233, ρ(error, top-3 missing) +0.139
- `vehicle_40nan`: top-3 − random-3 +0.056 [+0.040, +0.072] seeds +0.053, +0.053, +0.061 -> specific; descriptive: ρ(spread, top-3 missing) -0.003, ρ(spread, row's gaps) -0.138, ρ(error, top-3 missing) +0.157
- `biodeg_40nan`: top-3 − random-3 +0.033 [+0.020, +0.046] seeds +0.019, +0.049, +0.031 -> specific; descriptive: ρ(spread, top-3 missing) +0.016, ρ(spread, row's gaps) -0.072, ρ(error, top-3 missing) +0.095
- `kc2_40nan`: top-3 − random-3 -0.028 [-0.048, -0.008] seeds -0.032, -0.034, -0.018 -> against; descriptive: ρ(spread, top-3 missing) -0.074, ρ(spread, row's gaps) -0.106, ρ(error, top-3 missing) +0.016
- `spambase_40nan`: top-3 − random-3 -0.001 [-0.009, +0.008] seeds +0.006, +0.001, -0.008 -> no signal; descriptive: ρ(spread, top-3 missing) -0.029, ρ(spread, row's gaps) -0.116, ρ(error, top-3 missing) +0.055
- `credit-g_40nan`: top-3 − random-3 -0.159 [-0.189, -0.129] seeds -0.143, -0.148, -0.187 -> against; descriptive: ρ(spread, top-3 missing) -0.281, ρ(spread, row's gaps) -0.298, ρ(error, top-3 missing) +0.071
- `synthblock_60nan`: top-3 − random-3 -0.260 [-0.282, -0.238] seeds -0.291, -0.248, -0.241 -> against; descriptive: ρ(spread, top-3 missing) -0.468, ρ(spread, row's gaps) -0.453, ρ(error, top-3 missing) +0.161; with the true block as the top 3: -0.235 (descriptive)
- `pendigits_60nan`: top-3 − random-3 +0.026 [+0.022, +0.030] seeds +0.026, +0.027, +0.024 -> specific; descriptive: ρ(spread, top-3 missing) -0.037, ρ(spread, row's gaps) -0.131, ρ(error, top-3 missing) +0.186
- `letter_60nan`: top-3 − random-3 +0.014 [+0.009, +0.019] seeds +0.016, +0.013, +0.012 -> specific; descriptive: ρ(spread, top-3 missing) -0.121, ρ(spread, row's gaps) -0.292, ρ(error, top-3 missing) +0.132
- `electricity_60nan`: top-3 − random-3 -0.034 [-0.037, -0.030] seeds -0.034, -0.035, -0.032 -> against; descriptive: ρ(spread, top-3 missing) -0.330, ρ(spread, row's gaps) -0.441, ρ(error, top-3 missing) +0.179
- `vehicle_60nan`: top-3 − random-3 +0.021 [+0.009, +0.034] seeds +0.028, +0.025, +0.011 -> specific; descriptive: ρ(spread, top-3 missing) -0.080, ρ(spread, row's gaps) -0.233, ρ(error, top-3 missing) +0.145
- `biodeg_60nan`: top-3 − random-3 +0.006 [-0.011, +0.022] seeds +0.016, +0.016, -0.015 -> no signal; descriptive: ρ(spread, top-3 missing) -0.039, ρ(spread, row's gaps) -0.168, ρ(error, top-3 missing) +0.105
- `kc2_60nan`: top-3 − random-3 -0.029 [-0.053, -0.004] seeds -0.056, -0.006, -0.024 -> against; descriptive: ρ(spread, top-3 missing) -0.114, ρ(spread, row's gaps) -0.214, ρ(error, top-3 missing) +0.028
- `spambase_60nan`: top-3 − random-3 -0.060 [-0.082, -0.038] seeds -0.045, -0.081, -0.054 -> against; descriptive: ρ(spread, top-3 missing) -0.105, ρ(spread, row's gaps) -0.194, ρ(error, top-3 missing) +0.050
- `credit-g_60nan`: top-3 − random-3 -0.295 [-0.347, -0.243] seeds -0.333, -0.252, -0.298 -> against; descriptive: ρ(spread, top-3 missing) -0.477, ρ(spread, row's gaps) -0.467, ρ(error, top-3 missing) +0.073

Primary: specific on 7 of 16, against on ['kc2_40nan', 'credit-g_40nan', 'electricity_60nan', 'kc2_60nan', 'spambase_60nan', 'credit-g_60nan'] -> mixed (fold-pair t, seed agreement)
```
