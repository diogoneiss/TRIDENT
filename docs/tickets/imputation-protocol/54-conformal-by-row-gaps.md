# Intervals calibrated per bin of a row's gaps (E151, T43; pre-registered 2026-11-06)

This ticket was written and committed before its reader was run on any ledger.

- **Why:** [ADR 0030](../../adr/0030-an-opt-in-per-cell-uncertainty-signal-from-re-masking.md) keeps the
  re-masking spread as a rank, not an interval. [E77](30-remask-conformal.md) found a per-column constant
  width narrower than the spread-scaled one at the same 90% coverage on 15 of 15 variants. E80
  (imputation-interpretability/03) found why one scale fails: the spread *falls* as a row's gaps grow, while
  the error *rises* when a cell's top sources are missing. Conformal prediction with missing values (Zaffran
  et al., ICML 2023) answers this by calibrating per missingness pattern (Mondrian conformal). Two questions:
  1. **Coverage:** does E77's per-column constant under-cover the rows that miss the most, and does
     calibrating per bin of the row's gaps fix it?
  2. **Width:** once the bin is fixed, does the spread adapt a width better than a constant does? E80's
     confound, spread against the row's gaps, is held fixed inside a bin.
- **The user's go:** "vou na sua sugestão, vá" (2026-11-06, after 12:21 GMT-3), on the proposal that put T43
  first among the lines that need no GPU.
- **Offline, no training:** E75's and E76's fold ledgers (and E84's, descriptive), nothing in `src/` or the
  store. Reader: `scripts/experiments/conformal_row_gaps.py`.
- **Disclosure:** E77's and E80's results are known and motivate the design. No per-bin coverage or width has
  been computed on any ledger.

## Design (fixed now)

- **Cells:**
  - **Primary:** E75's 15 confirmation variants (MCAR, seeds 707/808/909, five folds), numerical gaps only,
    the set E77 read.
  - **Development, same rules, reported beside:** E76's 21 variants at 101/202/303 with numerical gaps
    (kr-vs-kp has none, so 17).
  - **Descriptive:** E84's MNAR and MAR variants at 60nan, 101/202/303.
- **A cell's row gaps:** b = the number of gaps in its row in the fold's ledger (`induced` rows, which hold
  every gap of the test split; the cell itself included), as E80 reads them.
- **Bins:** for held-out fold k of a seed, the tertiles of b over the fit cells (the other four folds of the
  same seed, numerical columns pooled) cut three bins: b ≤ e1, e1 < b ≤ e2, b > e2. Coinciding edges merge
  bins. The **top bin** is the highest one present.
- **Four 90% intervals per numerical cell, cross-fitted as E77** (fit on the same seed's other four folds):
  - **C**, per column, constant: fill ± w, w the conformal quantile of |error| over the column's fit cells;
  - **S**, per column, spread-scaled: fill ± q × spread, q the conformal quantile of |error| / spread (E77's);
  - **MC** and **MS**, the same two fitted on the column's fit cells *in the cell's bin*. A column-and-bin
    group with fewer than 30 fit cells falls back to its column's C or S; the share of such cells is printed.
- **Quantile:** the split-conformal level, the ⌈(n + 1) × 0.9⌉-th smallest of the n fit scores (the largest
  if that exceeds n), for all four arms; E77 used the plain 90% quantile, so C and S are recomputed here.
- **Spread floor:** E77's amendment, per column at the smallest positive spread on the fit cells (1 if none),
  in the column's group and in each bin's.
- **Centres:** K primary, as E77: the ledger's `induced` fill (KC on these calibrated runs) with the spread of
  `induced_remask`. H descriptive: `induced_remask_loss_checkpoint`'s fill and spread.

## Measures

- **Q1, coverage:** per fold, an arm's worst-bin under-coverage u = the largest of 0.90 − coverage over the
  fold's bins, floored at 0. The fold difference is u(MC) − u(C); lower is better. SESOI 0.01, one coverage
  point.
- **Q2, width:** per fold, (mean width of MS − mean width of MC) / mean width of MC; lower is better. SESOI
  0.01 of MC's width, as E77.
- **Per variant:** the project's verdict rule (`report_common.verdict`: the 95% t interval over the 15 fold
  differences, every seed agreeing), better / no difference / equivalent / worse.
- **Printed beside, not ruled on:** each arm's overall coverage and its coverage in each bin; u(C) alone (does
  the per-column constant under-cover the top bin at all?); MS against C, the width of a binned spread
  interval against E77's constant; the fallback share; H.

## Reading, fixed now (on the primary 15, under K)

For each question separately:
- **supported:** better on 10 or more of the 15 and worse on at most 1;
- **refuted:** better on 5 or fewer, or worse on 4 or more;
- **mixed:** otherwise.

The development set is read by the same rule over its 17 and reported as agreeing or not; it does not change
the primary reading.

**What each reading leads to:**
- **Q1 supported:** an interval for the fill, calibrated per bin of the row's gaps, is proposed as an
  amendment to ADR 0030, for the user's decision. It is a conformal wrapper around the fill: it uses the
  spread only if Q2 is supported too.
- **Q1 refuted, with u(C) near 0:** a per-column constant already covers every bin; binning is not needed.
- **Q2 supported:** ADR 0030's "a rank, not an interval" is proposed for amendment: the spread, scaled per
  bin, becomes an interval.
- **Q2 otherwise:** ADR 0030 stays a rank; E77's conclusion extends to binned intervals.

**Limit, stated now:** the calibration cells are other test folds' gaps, whose truth a deployed model would
not have (an oracle reading, as E77). A deployable reading needs observed validation cells hidden on purpose,
which these ledgers do not hold; it would be a follow-up with new runs.

## Outcome (2026-11-06, 13:05 GMT-3)

The reader ran once, started 12:51 GMT-3 and ended before 13:05, on all 225 + 255 + 180 ledgers (E75, E76, E84; no ledger missing, no
group fell back to its column: the fallback share is 0.00 on every variant). The self-check's line is the first.

**Q1: supported** (better on 11 of 15 under K, worse on none; H 11 better). **Q2: refuted** (better on 0 of 15, worse on
15; H worse on 13). The development set agrees on Q2 (0 of 17 better, 15 worse) and reads Q1 mixed (7 of 17 better,
worse on credit-g_80nan by +0.007).

- **The per-column constant under-covers the rows that miss the most** on the large tables and vehicle: its coverage in
  the top tertile of the row's gaps is 0.81 to 0.87 there (electricity_60nan 0.812), against 0.92 to 0.94 in the
  bottom tertile; overall it sits at 0.90. Binning brings the top bin to 0.900 to 0.905 on every one of them.
- **Where it does not, C already covers:** biodeg, kc2, spambase and credit-g under MCAR, top-bin coverage 0.88 to 0.91
  under C. That is the development set's mixed count: its 7 better are pendigits, letter, electricity, vehicle ×2,
  biodeg_60nan and spambase_40nan; the rest are equivalent or no difference, with u(C) at 0.006 to 0.037.
- **Under MNAR the gap is wider** (descriptive, E84): better on all 7 self-masking variants, kc2_probit_60nan's top
  bin 0.776 under C against 0.907 binned. Under MAR (5) C already covers (0.90 to 0.92).
- **Width, post hoc** (not pre-registered; a scratch script over the reader's functions, the arms unchanged): binning
  costs nothing on the large tables, (MC − C) / C between −0.022 and +0.001, and 1 to 12% on the small ones (vehicle
  +0.027, biodeg_40nan +0.053, kc2_40nan +0.116).
- **Q2:** inside a bin the spread-scaled interval is still wider than the binned constant on every variant, +0.06
  (kc2_40nan) to +1.57 (electricity_40nan) at 40nan, and by the floor's orders of magnitude at 60 and 80nan, as in E77.
  Holding the row's gaps fixed does not make the spread adapt a width.

**What it leads to, as fixed above:**
- **Q1 supported:** an interval for the fill, calibrated per numerical column and per tertile of the row's gap count, is
  proposed as an amendment to ADR 0030, for the user's decision. It does not use the spread. The amendment is not
  written; the ADR changes only on the user's word.
- **Q2 refuted:** ADR 0030 stays a rank; E77's conclusion extends to binned intervals.
- **Limit, as stated:** an oracle calibration (other test folds' gaps). The deployable form needs observed validation
  cells hidden on purpose, so new runs.

**The printout** (`scripts/experiments/conformal_row_gaps.py`, unedited):

```text
check passed: {'u_C': 0.17, 'top_C': 0.73, 'width_C': 6.872, 'u_S': 0.007, 'top_S': 0.9, 'width_S': 6.001, 'u_MC': 0.008, 'top_MC': 0.899, 'width_MC': 6.783, 'u_MS': 0.008, 'top_MS': 0.899, 'width_MS': 5.998}

## E75, primary (MCAR, confirmation)

- `pendigits_40nan`: K: Q1 u(MC) − u(C) -0.0358 [-0.0405, -0.0310] better, u(C) 0.043, u(MC) 0.007; Q2 (MS − MC) / MC +0.316 [+0.308, +0.324] worse; (MS − C) / C +0.318 worse; coverage C 0.900 (low bin 0.925, top 0.857), S 0.900 (low bin 0.935, top 0.838), MC 0.900 (low bin 0.900, top 0.901), MS 0.900 (low bin 0.900, top 0.900); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0346 [-0.0392, -0.0299] better, u(C) 0.041, u(MC) 0.007; Q2 (MS − MC) / MC +0.233 [+0.223, +0.244] worse; (MS − C) / C +0.234 worse; coverage C 0.900 (low bin 0.923, top 0.859), S 0.900 (low bin 0.935, top 0.838), MC 0.900 (low bin 0.900, top 0.900), MS 0.900 (low bin 0.900, top 0.901); fallback 0.00
- `pendigits_60nan`: K: Q1 u(MC) − u(C) -0.0614 [-0.0657, -0.0571] better, u(C) 0.067, u(MC) 0.005; Q2 (MS − MC) / MC +0.516 [+0.499, +0.533] worse; (MS − C) / C +0.508 worse; coverage C 0.900 (low bin 0.937, top 0.833), S 0.900 (low bin 0.959, top 0.769), MC 0.900 (low bin 0.900, top 0.900), MS 0.900 (low bin 0.900, top 0.900); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0584 [-0.0630, -0.0538] better, u(C) 0.065, u(MC) 0.006; Q2 (MS − MC) / MC +0.472 [+0.448, +0.496] worse; (MS − C) / C +0.463 worse; coverage C 0.900 (low bin 0.935, top 0.835), S 0.900 (low bin 0.959, top 0.768), MC 0.900 (low bin 0.900, top 0.900), MS 0.900 (low bin 0.900, top 0.901); fallback 0.00
- `pendigits_80nan`: K: Q1 u(MC) − u(C) -0.0452 [-0.0479, -0.0426] better, u(C) 0.052, u(MC) 0.006; Q2 (MS − MC) / MC +338.581 [+93.507, +583.656] worse; (MS − C) / C +335.094 worse; coverage C 0.900 (low bin 0.930, top 0.848), S 0.900 (low bin 1.000, top 0.494), MC 0.900 (low bin 0.900, top 0.900), MS 0.900 (low bin 0.899, top 0.900); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0436 [-0.0482, -0.0389] better, u(C) 0.051, u(MC) 0.008; Q2 (MS − MC) / MC +194.875 [+118.245, +271.504] worse; (MS − C) / C +193.397 worse; coverage C 0.900 (low bin 0.930, top 0.849), S 0.900 (low bin 1.000, top 0.494), MC 0.900 (low bin 0.900, top 0.900), MS 0.899 (low bin 0.898, top 0.900); fallback 0.00
- `letter_40nan`: K: Q1 u(MC) − u(C) -0.0403 [-0.0439, -0.0367] better, u(C) 0.045, u(MC) 0.005; Q2 (MS − MC) / MC +0.431 [+0.424, +0.439] worse; (MS − C) / C +0.428 worse; coverage C 0.900 (low bin 0.924, top 0.855), S 0.900 (low bin 0.941, top 0.818), MC 0.900 (low bin 0.900, top 0.900), MS 0.900 (low bin 0.900, top 0.900); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0328 [-0.0364, -0.0292] better, u(C) 0.038, u(MC) 0.005; Q2 (MS − MC) / MC +0.377 [+0.369, +0.385] worse; (MS − C) / C +0.379 worse; coverage C 0.900 (low bin 0.920, top 0.862), S 0.900 (low bin 0.940, top 0.819), MC 0.900 (low bin 0.900, top 0.900), MS 0.900 (low bin 0.900, top 0.901); fallback 0.00
- `letter_60nan`: K: Q1 u(MC) − u(C) -0.0436 [-0.0460, -0.0412] better, u(C) 0.047, u(MC) 0.004; Q2 (MS − MC) / MC +0.745 [+0.722, +0.768] worse; (MS − C) / C +0.741 worse; coverage C 0.900 (low bin 0.929, top 0.853), S 0.900 (low bin 0.967, top 0.753), MC 0.900 (low bin 0.900, top 0.900), MS 0.900 (low bin 0.900, top 0.900); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0397 [-0.0425, -0.0369] better, u(C) 0.044, u(MC) 0.004; Q2 (MS − MC) / MC +0.785 [+0.735, +0.834] worse; (MS − C) / C +0.784 worse; coverage C 0.900 (low bin 0.926, top 0.856), S 0.900 (low bin 0.967, top 0.752), MC 0.900 (low bin 0.900, top 0.900), MS 0.900 (low bin 0.900, top 0.900); fallback 0.00
- `letter_80nan`: K: Q1 u(MC) − u(C) -0.0277 [-0.0301, -0.0252] better, u(C) 0.031, u(MC) 0.003; Q2 (MS − MC) / MC +52.664 [+45.116, +60.213] worse; (MS − C) / C +52.559 worse; coverage C 0.900 (low bin 0.919, top 0.869), S 0.900 (low bin 1.000, top 0.481), MC 0.900 (low bin 0.900, top 0.900), MS 0.899 (low bin 0.899, top 0.899); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0289 [-0.0322, -0.0256] better, u(C) 0.032, u(MC) 0.003; Q2 (MS − MC) / MC +87.027 [+71.099, +102.955] worse; (MS − C) / C +86.827 worse; coverage C 0.900 (low bin 0.919, top 0.868), S 0.900 (low bin 0.999, top 0.482), MC 0.900 (low bin 0.900, top 0.900), MS 0.899 (low bin 0.898, top 0.899); fallback 0.00
- `electricity_40nan`: K: Q1 u(MC) − u(C) -0.0395 [-0.0418, -0.0372] better, u(C) 0.044, u(MC) 0.004; Q2 (MS − MC) / MC +1.565 [+1.514, +1.616] worse; (MS − C) / C +1.558 worse; coverage C 0.900 (low bin 0.928, top 0.856), S 0.899 (low bin 0.959, top 0.797), MC 0.900 (low bin 0.900, top 0.900), MS 0.901 (low bin 0.901, top 0.899); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0365 [-0.0382, -0.0348] better, u(C) 0.040, u(MC) 0.004; Q2 (MS − MC) / MC +1.459 [+1.417, +1.502] worse; (MS − C) / C +1.467 worse; coverage C 0.900 (low bin 0.925, top 0.860), S 0.900 (low bin 0.960, top 0.795), MC 0.900 (low bin 0.900, top 0.900), MS 0.900 (low bin 0.901, top 0.900); fallback 0.00
- `electricity_60nan`: K: Q1 u(MC) − u(C) -0.0839 [-0.0880, -0.0798] better, u(C) 0.088, u(MC) 0.004; Q2 (MS − MC) / MC +250.346 [+206.091, +294.601] worse; (MS − C) / C +244.836 worse; coverage C 0.900 (low bin 0.932, top 0.812), S 0.900 (low bin 0.988, top 0.514), MC 0.900 (low bin 0.900, top 0.900), MS 0.899 (low bin 0.899, top 0.900); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0787 [-0.0826, -0.0749] better, u(C) 0.083, u(MC) 0.004; Q2 (MS − MC) / MC +171.093 [+121.150, +221.036] worse; (MS − C) / C +167.137 worse; coverage C 0.900 (low bin 0.930, top 0.817), S 0.900 (low bin 0.988, top 0.511), MC 0.900 (low bin 0.900, top 0.901), MS 0.900 (low bin 0.900, top 0.900); fallback 0.00
- `electricity_80nan`: K: Q1 u(MC) − u(C) -0.0378 [-0.0440, -0.0315] better, u(C) 0.045, u(MC) 0.008; Q2 (MS − MC) / MC +1920.137 [+1455.943, +2384.331] worse; (MS − C) / C +1880.600 worse; coverage C 0.900 (low bin 0.927, top 0.855), S 0.900 (low bin 0.998, top 0.657), MC 0.900 (low bin 0.900, top 0.899), MS 0.899 (low bin 0.898, top 0.899); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0350 [-0.0394, -0.0306] better, u(C) 0.043, u(MC) 0.008; Q2 (MS − MC) / MC +5708.147 [+3390.114, +8026.180] worse; (MS − C) / C +5614.947 worse; coverage C 0.900 (low bin 0.926, top 0.857), S 0.900 (low bin 0.998, top 0.657), MC 0.899 (low bin 0.899, top 0.899), MS 0.899 (low bin 0.899, top 0.899); fallback 0.00
- `vehicle_40nan`: K: Q1 u(MC) − u(C) -0.0223 [-0.0337, -0.0108] better, u(C) 0.041, u(MC) 0.019; Q2 (MS − MC) / MC +0.406 [+0.350, +0.462] worse; (MS − C) / C +0.444 worse; coverage C 0.900 (low bin 0.927, top 0.861), S 0.901 (low bin 0.939, top 0.832), MC 0.904 (low bin 0.903, top 0.905), MS 0.905 (low bin 0.903, top 0.910); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0219 [-0.0284, -0.0155] better, u(C) 0.041, u(MC) 0.019; Q2 (MS − MC) / MC +0.422 [+0.350, +0.493] worse; (MS − C) / C +0.460 worse; coverage C 0.900 (low bin 0.926, top 0.862), S 0.900 (low bin 0.938, top 0.828), MC 0.905 (low bin 0.904, top 0.908), MS 0.905 (low bin 0.902, top 0.912); fallback 0.00
- `vehicle_80nan`: K: Q1 u(MC) − u(C) -0.0342 [-0.0463, -0.0221] better, u(C) 0.051, u(MC) 0.017; Q2 (MS − MC) / MC +120.975 [+42.442, +199.508] worse; (MS − C) / C +122.124 worse; coverage C 0.901 (low bin 0.927, top 0.850), S 0.901 (low bin 0.999, top 0.685), MC 0.903 (low bin 0.902, top 0.902), MS 0.896 (low bin 0.894, top 0.899); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0248 [-0.0392, -0.0104] better, u(C) 0.042, u(MC) 0.017; Q2 (MS − MC) / MC +58.497 [+26.865, +90.129] worse; (MS − C) / C +59.231 worse; coverage C 0.901 (low bin 0.925, top 0.858), S 0.900 (low bin 0.998, top 0.686), MC 0.902 (low bin 0.901, top 0.903), MS 0.897 (low bin 0.896, top 0.900); fallback 0.00
- `biodeg_40nan`: K: Q1 u(MC) − u(C) -0.0058 [-0.0120, +0.0003] none, u(C) 0.025, u(MC) 0.019; Q2 (MS − MC) / MC +0.186 [+0.114, +0.258] worse; (MS − C) / C +0.250 worse; coverage C 0.895 (low bin 0.905, top 0.878), S 0.895 (low bin 0.915, top 0.866), MC 0.899 (low bin 0.897, top 0.901), MS 0.898 (low bin 0.898, top 0.899); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0051 [-0.0122, +0.0020] none, u(C) 0.019, u(MC) 0.014; Q2 (MS − MC) / MC +0.124 [-0.031, +0.279] none; (MS − C) / C +0.194 worse; coverage C 0.900 (low bin 0.911, top 0.884), S 0.902 (low bin 0.924, top 0.867), MC 0.903 (low bin 0.902, top 0.903), MS 0.904 (low bin 0.903, top 0.905); fallback 0.00
- `biodeg_80nan`: K: Q1 u(MC) − u(C) +0.0009 [-0.0044, +0.0062] equivalent, u(C) 0.014, u(MC) 0.015; Q2 (MS − MC) / MC +2.397 [+0.636, +4.159] worse; (MS − C) / C +2.459 worse; coverage C 0.894 (low bin 0.893, top 0.900), S 0.874 (low bin 0.921, top 0.798), MC 0.896 (low bin 0.896, top 0.894), MS 0.878 (low bin 0.880, top 0.874); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0008 [-0.0034, +0.0017] equivalent, u(C) 0.018, u(MC) 0.017; Q2 (MS − MC) / MC +17.574 [+4.844, +30.303] worse; (MS − C) / C +17.892 worse; coverage C 0.893 (low bin 0.894, top 0.898), S 0.846 (low bin 0.883, top 0.788), MC 0.895 (low bin 0.894, top 0.894), MS 0.851 (low bin 0.855, top 0.846); fallback 0.00
- `kc2_40nan`: K: Q1 u(MC) − u(C) -0.0018 [-0.0106, +0.0071] none, u(C) 0.024, u(MC) 0.022; Q2 (MS − MC) / MC +0.055 [+0.003, +0.108] worse; (MS − C) / C +0.178 worse; coverage C 0.904 (low bin 0.905, top 0.903), S 0.901 (low bin 0.926, top 0.872), MC 0.909 (low bin 0.907, top 0.908), MS 0.908 (low bin 0.905, top 0.909); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0038 [-0.0096, +0.0019] equivalent, u(C) 0.022, u(MC) 0.018; Q2 (MS − MC) / MC +0.011 [-0.065, +0.087] none; (MS − C) / C +0.113 worse; coverage C 0.903 (low bin 0.908, top 0.894), S 0.901 (low bin 0.926, top 0.871), MC 0.908 (low bin 0.907, top 0.909), MS 0.906 (low bin 0.902, top 0.910); fallback 0.00
- `kc2_80nan`: K: Q1 u(MC) − u(C) +0.0028 [-0.0078, +0.0135] none, u(C) 0.030, u(MC) 0.032; Q2 (MS − MC) / MC +15.695 [+11.815, +19.575] worse; (MS − C) / C +16.342 worse; coverage C 0.901 (low bin 0.910, top 0.888), S 0.900 (low bin 0.995, top 0.635), MC 0.902 (low bin 0.900, top 0.904), MS 0.900 (low bin 0.898, top 0.905); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0033 [-0.0119, +0.0053] none, u(C) 0.030, u(MC) 0.027; Q2 (MS − MC) / MC +43.554 [+14.791, +72.316] worse; (MS − C) / C +45.301 worse; coverage C 0.900 (low bin 0.910, top 0.888), S 0.899 (low bin 0.996, top 0.641), MC 0.905 (low bin 0.904, top 0.905), MS 0.900 (low bin 0.899, top 0.903); fallback 0.00

- Q1, K: better 11 / no difference 3 / equivalent 1 / worse 0 (fold-pair t, seed agreement)
- Q2, K: better 0 / no difference 0 / worse 15 (fold-pair t, seed agreement)
- Q1, H: better 11 / no difference 2 / equivalent 2 / worse 0 (fold-pair t, seed agreement)
- Q2, H: better 0 / no difference 2 / worse 13 (fold-pair t, seed agreement)
E75, primary (MCAR, confirmation) Q1 (K): better on 11 of 15, worse on none -> supported (fold-pair t, seed agreement)
E75, primary (MCAR, confirmation) Q2 (K): better on 0 of 15, worse on ['pendigits_40nan', 'pendigits_60nan', 'pendigits_80nan', 'letter_40nan', 'letter_60nan', 'letter_80nan', 'electricity_40nan', 'electricity_60nan', 'electricity_80nan', 'vehicle_40nan', 'vehicle_80nan', 'biodeg_40nan', 'biodeg_80nan', 'kc2_40nan', 'kc2_80nan'] -> refuted (fold-pair t, seed agreement)

## E76, development (MCAR)

- `spambase_20nan`: K: Q1 u(MC) − u(C) +0.0001 [-0.0009, +0.0011] equivalent, u(C) 0.006, u(MC) 0.006; Q2 (MS − MC) / MC +0.602 [+0.432, +0.771] worse; (MS − C) / C +0.647 worse; coverage C 0.899 (low bin 0.899, top 0.897), S 0.896 (low bin 0.904, top 0.884), MC 0.899 (low bin 0.899, top 0.899), MS 0.898 (low bin 0.898, top 0.896); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0008 [-0.0023, +0.0007] equivalent, u(C) 0.006, u(MC) 0.005; Q2 (MS − MC) / MC +11.251 [-1.141, +23.642] none; (MS − C) / C +11.600 none; coverage C 0.898 (low bin 0.898, top 0.896), S 0.894 (low bin 0.903, top 0.883), MC 0.899 (low bin 0.898, top 0.899), MS 0.896 (low bin 0.895, top 0.898); fallback 0.00
- `spambase_40nan`: K: Q1 u(MC) − u(C) -0.0014 [-0.0025, -0.0002] better, u(C) 0.012, u(MC) 0.010; Q2 (MS − MC) / MC +0.800 [+0.422, +1.179] worse; (MS − C) / C +0.827 worse; coverage C 0.894 (low bin 0.896, top 0.894), S 0.888 (low bin 0.903, top 0.870), MC 0.895 (low bin 0.895, top 0.896), MS 0.889 (low bin 0.889, top 0.889); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0003 [-0.0016, +0.0010] equivalent, u(C) 0.008, u(MC) 0.008; Q2 (MS − MC) / MC +1.424 [+1.067, +1.781] worse; (MS − C) / C +1.462 worse; coverage C 0.897 (low bin 0.899, top 0.896), S 0.889 (low bin 0.904, top 0.870), MC 0.897 (low bin 0.896, top 0.898), MS 0.891 (low bin 0.890, top 0.891); fallback 0.00
- `spambase_60nan`: K: Q1 u(MC) − u(C) +0.0002 [-0.0007, +0.0011] equivalent, u(C) 0.013, u(MC) 0.013; Q2 (MS − MC) / MC +2.803 [+1.111, +4.495] worse; (MS − C) / C +2.850 worse; coverage C 0.891 (low bin 0.891, top 0.891), S 0.887 (low bin 0.908, top 0.858), MC 0.892 (low bin 0.891, top 0.893), MS 0.888 (low bin 0.888, top 0.889); fallback 0.00
  H: Q1 u(MC) − u(C) +0.0009 [-0.0004, +0.0022] equivalent, u(C) 0.011, u(MC) 0.012; Q2 (MS − MC) / MC +2.299 [+1.531, +3.067] worse; (MS − C) / C +2.330 worse; coverage C 0.894 (low bin 0.895, top 0.893), S 0.884 (low bin 0.905, top 0.857), MC 0.895 (low bin 0.894, top 0.896), MS 0.886 (low bin 0.885, top 0.886); fallback 0.00
- `spambase_80nan`: K: Q1 u(MC) − u(C) +0.0024 [-0.0004, +0.0051] equivalent, u(C) 0.013, u(MC) 0.015; Q2 (MS − MC) / MC +30.017 [+17.957, +42.077] worse; (MS − C) / C +30.408 worse; coverage C 0.891 (low bin 0.888, top 0.897), S 0.880 (low bin 0.914, top 0.821), MC 0.891 (low bin 0.892, top 0.889), MS 0.879 (low bin 0.878, top 0.880); fallback 0.00
  H: Q1 u(MC) − u(C) +0.0006 [-0.0003, +0.0015] equivalent, u(C) 0.014, u(MC) 0.014; Q2 (MS − MC) / MC +123.068 [+23.398, +222.739] worse; (MS − C) / C +124.226 worse; coverage C 0.890 (low bin 0.888, top 0.893), S 0.872 (low bin 0.903, top 0.819), MC 0.890 (low bin 0.891, top 0.889), MS 0.873 (low bin 0.872, top 0.873); fallback 0.00
- `credit-g_20nan`: K: Q1 u(MC) − u(C) +0.0078 [-0.0045, +0.0202] none, u(C) 0.024, u(MC) 0.032; Q2 (MS − MC) / MC +0.902 [+0.154, +1.650] worse; (MS − C) / C +0.972 worse; coverage C 0.903 (low bin 0.903, top 0.902), S 0.895 (low bin 0.917, top 0.868), MC 0.905 (low bin 0.906, top 0.907), MS 0.899 (low bin 0.902, top 0.900); fallback 0.04
  H: Q1 u(MC) − u(C) -0.0018 [-0.0137, +0.0100] none, u(C) 0.025, u(MC) 0.024; Q2 (MS − MC) / MC +0.210 [+0.123, +0.298] worse; (MS − C) / C +0.251 worse; coverage C 0.902 (low bin 0.904, top 0.891), S 0.898 (low bin 0.919, top 0.861), MC 0.909 (low bin 0.909, top 0.909), MS 0.907 (low bin 0.907, top 0.909); fallback 0.04
- `credit-g_40nan`: K: Q1 u(MC) − u(C) +0.0018 [-0.0020, +0.0056] equivalent, u(C) 0.014, u(MC) 0.016; Q2 (MS − MC) / MC +0.218 [+0.130, +0.306] worse; (MS − C) / C +0.232 worse; coverage C 0.902 (low bin 0.899, top 0.901), S 0.900 (low bin 0.943, top 0.817), MC 0.905 (low bin 0.903, top 0.907), MS 0.902 (low bin 0.899, top 0.908); fallback 0.00
  H: Q1 u(MC) − u(C) +0.0028 [-0.0046, +0.0103] none, u(C) 0.015, u(MC) 0.018; Q2 (MS − MC) / MC +0.447 [+0.216, +0.677] worse; (MS − C) / C +0.470 worse; coverage C 0.902 (low bin 0.900, top 0.906), S 0.894 (low bin 0.930, top 0.831), MC 0.905 (low bin 0.903, top 0.908), MS 0.897 (low bin 0.895, top 0.899); fallback 0.00
- `credit-g_60nan`: K: Q1 u(MC) − u(C) -0.0014 [-0.0058, +0.0030] equivalent, u(C) 0.023, u(MC) 0.021; Q2 (MS − MC) / MC +3.718 [+1.145, +6.291] worse; (MS − C) / C +3.757 worse; coverage C 0.899 (low bin 0.898, top 0.893), S 0.855 (low bin 0.900, top 0.789), MC 0.901 (low bin 0.899, top 0.902), MS 0.861 (low bin 0.859, top 0.867); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0009 [-0.0063, +0.0046] equivalent, u(C) 0.017, u(MC) 0.016; Q2 (MS − MC) / MC +4.888 [+1.956, +7.821] worse; (MS − C) / C +4.958 worse; coverage C 0.901 (low bin 0.903, top 0.892), S 0.851 (low bin 0.880, top 0.800), MC 0.904 (low bin 0.899, top 0.904), MS 0.865 (low bin 0.865, top 0.862); fallback 0.00
- `credit-g_80nan`: K: Q1 u(MC) − u(C) +0.0074 [+0.0002, +0.0145] worse, u(C) 0.013, u(MC) 0.020; Q2 (MS − MC) / MC +34.251 [+20.238, +48.263] worse; (MS − C) / C +34.629 worse; coverage C 0.903 (low bin 0.899, top 0.913), S 0.889 (low bin 0.990, top 0.613), MC 0.902 (low bin 0.902, top 0.897), MS 0.878 (low bin 0.866, top 0.895); fallback 0.00
  H: Q1 u(MC) − u(C) +0.0002 [-0.0032, +0.0036] equivalent, u(C) 0.018, u(MC) 0.018; Q2 (MS − MC) / MC +88.994 [+12.876, +165.111] worse; (MS − C) / C +90.045 worse; coverage C 0.900 (low bin 0.897, top 0.908), S 0.882 (low bin 0.983, top 0.613), MC 0.903 (low bin 0.900, top 0.905), MS 0.867 (low bin 0.856, top 0.893); fallback 0.00
- `vehicle_20nan`: K: Q1 u(MC) − u(C) -0.0143 [-0.0261, -0.0025] better, u(C) 0.035, u(MC) 0.020; Q2 (MS − MC) / MC +0.385 [+0.354, +0.416] worse; (MS − C) / C +0.417 worse; coverage C 0.903 (low bin 0.919, top 0.871), S 0.903 (low bin 0.926, top 0.856), MC 0.906 (low bin 0.909, top 0.905), MS 0.905 (low bin 0.908, top 0.904); fallback 0.18
  H: Q1 u(MC) − u(C) -0.0089 [-0.0160, -0.0018] better, u(C) 0.036, u(MC) 0.028; Q2 (MS − MC) / MC +0.334 [+0.291, +0.378] worse; (MS − C) / C +0.360 worse; coverage C 0.902 (low bin 0.920, top 0.871), S 0.904 (low bin 0.927, top 0.867), MC 0.902 (low bin 0.908, top 0.901), MS 0.903 (low bin 0.906, top 0.907); fallback 0.18
- `vehicle_60nan`: K: Q1 u(MC) − u(C) -0.0304 [-0.0421, -0.0188] better, u(C) 0.044, u(MC) 0.013; Q2 (MS − MC) / MC +0.530 [+0.426, +0.635] worse; (MS − C) / C +0.574 worse; coverage C 0.900 (low bin 0.927, top 0.856), S 0.899 (low bin 0.955, top 0.792), MC 0.902 (low bin 0.902, top 0.903), MS 0.902 (low bin 0.900, top 0.903); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0255 [-0.0358, -0.0153] better, u(C) 0.042, u(MC) 0.017; Q2 (MS − MC) / MC +0.639 [+0.465, +0.813] worse; (MS − C) / C +0.677 worse; coverage C 0.899 (low bin 0.926, top 0.858), S 0.896 (low bin 0.955, top 0.791), MC 0.902 (low bin 0.900, top 0.904), MS 0.901 (low bin 0.900, top 0.903); fallback 0.00
- `biodeg_20nan`: K: Q1 u(MC) − u(C) -0.0047 [-0.0125, +0.0031] none, u(C) 0.024, u(MC) 0.019; Q2 (MS − MC) / MC +0.159 [-0.040, +0.358] none; (MS − C) / C +0.324 worse; coverage C 0.900 (low bin 0.908, top 0.891), S 0.903 (low bin 0.915, top 0.886), MC 0.904 (low bin 0.902, top 0.908), MS 0.908 (low bin 0.907, top 0.910); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0051 [-0.0113, +0.0010] none, u(C) 0.024, u(MC) 0.019; Q2 (MS − MC) / MC -0.038 [-0.097, +0.021] none; (MS − C) / C +0.084 worse; coverage C 0.900 (low bin 0.911, top 0.887), S 0.905 (low bin 0.923, top 0.879), MC 0.907 (low bin 0.906, top 0.910), MS 0.909 (low bin 0.907, top 0.909); fallback 0.00
- `biodeg_60nan`: K: Q1 u(MC) − u(C) -0.0054 [-0.0108, -0.0001] better, u(C) 0.021, u(MC) 0.016; Q2 (MS − MC) / MC +0.422 [+0.254, +0.589] worse; (MS − C) / C +0.466 worse; coverage C 0.896 (low bin 0.907, top 0.884), S 0.895 (low bin 0.925, top 0.852), MC 0.897 (low bin 0.895, top 0.899), MS 0.897 (low bin 0.896, top 0.897); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0058 [-0.0108, -0.0008] better, u(C) 0.020, u(MC) 0.014; Q2 (MS − MC) / MC +0.131 [+0.083, +0.179] worse; (MS − C) / C +0.171 worse; coverage C 0.898 (low bin 0.911, top 0.886), S 0.899 (low bin 0.931, top 0.853), MC 0.900 (low bin 0.899, top 0.900), MS 0.901 (low bin 0.900, top 0.903); fallback 0.00
- `kc2_20nan`: K: Q1 u(MC) − u(C) -0.0010 [-0.0041, +0.0022] equivalent, u(C) 0.037, u(MC) 0.036; Q2 (MS − MC) / MC -0.101 [-0.231, +0.028] none; (MS − C) / C +0.045 none; coverage C 0.897 (low bin 0.881, top 0.900), S 0.904 (low bin 0.917, top 0.874), MC 0.905 (low bin 0.904, top 0.900), MS 0.905 (low bin 0.914, top 0.874); fallback 0.26
  H: Q1 u(MC) − u(C) -0.0030 [-0.0078, +0.0018] equivalent, u(C) 0.039, u(MC) 0.036; Q2 (MS − MC) / MC -0.071 [-0.166, +0.024] none; (MS − C) / C +0.082 none; coverage C 0.901 (low bin 0.883, top 0.898), S 0.903 (low bin 0.917, top 0.868), MC 0.908 (low bin 0.907, top 0.898), MS 0.902 (low bin 0.911, top 0.868); fallback 0.26
- `kc2_60nan`: K: Q1 u(MC) − u(C) -0.0047 [-0.0141, +0.0046] none, u(C) 0.033, u(MC) 0.028; Q2 (MS − MC) / MC +0.284 [+0.156, +0.413] worse; (MS − C) / C +0.373 worse; coverage C 0.900 (low bin 0.881, top 0.896), S 0.899 (low bin 0.937, top 0.825), MC 0.904 (low bin 0.902, top 0.906), MS 0.903 (low bin 0.902, top 0.903); fallback 0.00
  H: Q1 u(MC) − u(C) +0.0004 [-0.0063, +0.0070] equivalent, u(C) 0.040, u(MC) 0.040; Q2 (MS − MC) / MC +0.236 [+0.076, +0.395] worse; (MS − C) / C +0.312 worse; coverage C 0.897 (low bin 0.887, top 0.882), S 0.899 (low bin 0.935, top 0.831), MC 0.901 (low bin 0.902, top 0.899), MS 0.903 (low bin 0.903, top 0.908); fallback 0.00
- `pendigits_20nan`: K: Q1 u(MC) − u(C) -0.0218 [-0.0263, -0.0174] better, u(C) 0.029, u(MC) 0.007; Q2 (MS − MC) / MC +0.431 [+0.415, +0.447] worse; (MS − C) / C +0.440 worse; coverage C 0.900 (low bin 0.914, top 0.871), S 0.900 (low bin 0.912, top 0.876), MC 0.901 (low bin 0.901, top 0.901), MS 0.901 (low bin 0.900, top 0.902); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0217 [-0.0269, -0.0165] better, u(C) 0.029, u(MC) 0.007; Q2 (MS − MC) / MC +0.310 [+0.296, +0.325] worse; (MS − C) / C +0.317 worse; coverage C 0.901 (low bin 0.913, top 0.871), S 0.900 (low bin 0.913, top 0.876), MC 0.900 (low bin 0.900, top 0.902), MS 0.901 (low bin 0.901, top 0.903); fallback 0.00
- `letter_20nan`: K: Q1 u(MC) − u(C) -0.0246 [-0.0292, -0.0199] better, u(C) 0.031, u(MC) 0.006; Q2 (MS − MC) / MC +0.359 [+0.348, +0.369] worse; (MS − C) / C +0.360 worse; coverage C 0.900 (low bin 0.914, top 0.869), S 0.901 (low bin 0.918, top 0.858), MC 0.900 (low bin 0.900, top 0.901), MS 0.901 (low bin 0.901, top 0.901); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0218 [-0.0266, -0.0169] better, u(C) 0.028, u(MC) 0.007; Q2 (MS − MC) / MC +0.283 [+0.273, +0.294] worse; (MS − C) / C +0.286 worse; coverage C 0.900 (low bin 0.913, top 0.872), S 0.900 (low bin 0.918, top 0.858), MC 0.900 (low bin 0.900, top 0.901), MS 0.901 (low bin 0.901, top 0.900); fallback 0.00
- `electricity_20nan`: K: Q1 u(MC) − u(C) -0.0347 [-0.0383, -0.0311] better, u(C) 0.041, u(MC) 0.006; Q2 (MS − MC) / MC +0.683 [+0.660, +0.706] worse; (MS − C) / C +0.675 worse; coverage C 0.900 (low bin 0.916, top 0.859), S 0.902 (low bin 0.936, top 0.804), MC 0.900 (low bin 0.900, top 0.900), MS 0.901 (low bin 0.901, top 0.901); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0220 [-0.0251, -0.0190] better, u(C) 0.030, u(MC) 0.008; Q2 (MS − MC) / MC +0.613 [+0.588, +0.637] worse; (MS − C) / C +0.613 worse; coverage C 0.900 (low bin 0.912, top 0.870), S 0.900 (low bin 0.933, top 0.808), MC 0.900 (low bin 0.900, top 0.900), MS 0.900 (low bin 0.900, top 0.900); fallback 0.00

- Q1, K: better 7 / no difference 3 / equivalent 6 / worse 1 (fold-pair t, seed agreement)
- Q2, K: better 0 / no difference 2 / worse 15 (fold-pair t, seed agreement)
- Q1, H: better 6 / no difference 3 / equivalent 8 / worse 0 (fold-pair t, seed agreement)
- Q2, H: better 0 / no difference 3 / worse 14 (fold-pair t, seed agreement)
E76, development (MCAR) Q1 (K): better on 7 of 17, worse on ['credit-g_80nan'] -> mixed (fold-pair t, seed agreement)
E76, development (MCAR) Q2 (K): better on 0 of 17, worse on ['spambase_20nan', 'spambase_40nan', 'spambase_60nan', 'spambase_80nan', 'credit-g_20nan', 'credit-g_40nan', 'credit-g_60nan', 'credit-g_80nan', 'vehicle_20nan', 'vehicle_60nan', 'biodeg_60nan', 'kc2_60nan', 'pendigits_20nan', 'letter_20nan', 'electricity_20nan'] -> refuted (fold-pair t, seed agreement)

## E84, descriptive (MNAR and MAR, 60nan)

- `biodeg_probit0p25_60nan`: K: Q1 u(MC) − u(C) -0.0061 [-0.0121, -0.0001] better, u(C) 0.022, u(MC) 0.015; Q2 (MS − MC) / MC +0.352 [+0.234, +0.471] worse; (MS − C) / C +0.434 worse; coverage C 0.897 (low bin 0.908, top 0.880), S 0.894 (low bin 0.918, top 0.855), MC 0.898 (low bin 0.897, top 0.901), MS 0.897 (low bin 0.897, top 0.896); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0057 [-0.0111, -0.0004] better, u(C) 0.023, u(MC) 0.017; Q2 (MS − MC) / MC +0.122 [+0.079, +0.164] worse; (MS − C) / C +0.185 worse; coverage C 0.896 (low bin 0.907, top 0.879), S 0.900 (low bin 0.926, top 0.859), MC 0.898 (low bin 0.896, top 0.901), MS 0.902 (low bin 0.901, top 0.900); fallback 0.00
- `biodeg_probit_60nan`: K: Q1 u(MC) − u(C) -0.0186 [-0.0247, -0.0125] better, u(C) 0.032, u(MC) 0.014; Q2 (MS − MC) / MC +0.251 [+0.142, +0.360] worse; (MS − C) / C +0.411 worse; coverage C 0.896 (low bin 0.917, top 0.868), S 0.891 (low bin 0.923, top 0.839), MC 0.899 (low bin 0.896, top 0.898), MS 0.894 (low bin 0.890, top 0.897); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0149 [-0.0226, -0.0071] better, u(C) 0.032, u(MC) 0.017; Q2 (MS − MC) / MC +0.185 [+0.053, +0.316] worse; (MS − C) / C +0.330 worse; coverage C 0.896 (low bin 0.915, top 0.869), S 0.897 (low bin 0.931, top 0.845), MC 0.898 (low bin 0.896, top 0.899), MS 0.901 (low bin 0.901, top 0.897); fallback 0.00
- `biodeg_probit1p0_60nan`: K: Q1 u(MC) − u(C) -0.0111 [-0.0172, -0.0051] better, u(C) 0.027, u(MC) 0.015; Q2 (MS − MC) / MC +0.465 [+0.378, +0.553] worse; (MS − C) / C +0.569 worse; coverage C 0.897 (low bin 0.916, top 0.873), S 0.896 (low bin 0.931, top 0.838), MC 0.899 (low bin 0.898, top 0.900), MS 0.897 (low bin 0.895, top 0.899); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0116 [-0.0181, -0.0050] better, u(C) 0.026, u(MC) 0.015; Q2 (MS − MC) / MC +0.625 [+0.476, +0.774] worse; (MS − C) / C +0.752 worse; coverage C 0.897 (low bin 0.917, top 0.874), S 0.900 (low bin 0.934, top 0.845), MC 0.900 (low bin 0.897, top 0.902), MS 0.904 (low bin 0.905, top 0.903); fallback 0.00
- `biodeg_probit2p0_60nan`: K: Q1 u(MC) − u(C) -0.0078 [-0.0154, -0.0001] better, u(C) 0.025, u(MC) 0.017; Q2 (MS − MC) / MC +1.676 [+1.361, +1.991] worse; (MS − C) / C +1.812 worse; coverage C 0.897 (low bin 0.921, top 0.877), S 0.894 (low bin 0.930, top 0.845), MC 0.898 (low bin 0.897, top 0.898), MS 0.894 (low bin 0.894, top 0.897); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0098 [-0.0163, -0.0033] better, u(C) 0.030, u(MC) 0.021; Q2 (MS − MC) / MC +2.717 [+1.395, +4.040] worse; (MS − C) / C +2.924 worse; coverage C 0.897 (low bin 0.922, top 0.874), S 0.897 (low bin 0.939, top 0.837), MC 0.898 (low bin 0.898, top 0.899), MS 0.899 (low bin 0.902, top 0.897); fallback 0.00
- `vehicle_probit_60nan`: K: Q1 u(MC) − u(C) -0.0335 [-0.0431, -0.0239] better, u(C) 0.053, u(MC) 0.019; Q2 (MS − MC) / MC +0.826 [+0.678, +0.973] worse; (MS − C) / C +0.859 worse; coverage C 0.900 (low bin 0.933, top 0.848), S 0.899 (low bin 0.962, top 0.754), MC 0.902 (low bin 0.899, top 0.904), MS 0.902 (low bin 0.902, top 0.905); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0321 [-0.0405, -0.0236] better, u(C) 0.045, u(MC) 0.013; Q2 (MS − MC) / MC +0.757 [+0.634, +0.881] worse; (MS − C) / C +0.785 worse; coverage C 0.900 (low bin 0.931, top 0.856), S 0.900 (low bin 0.965, top 0.755), MC 0.904 (low bin 0.904, top 0.907), MS 0.902 (low bin 0.902, top 0.908); fallback 0.00
- `kc2_probit_60nan`: K: Q1 u(MC) − u(C) -0.0937 [-0.1246, -0.0628] better, u(C) 0.124, u(MC) 0.030; Q2 (MS − MC) / MC +0.192 [-0.027, +0.412] none; (MS − C) / C +0.515 worse; coverage C 0.902 (low bin 0.968, top 0.776), S 0.900 (low bin 0.951, top 0.825), MC 0.906 (low bin 0.905, top 0.907), MS 0.905 (low bin 0.903, top 0.907); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0840 [-0.1108, -0.0571] better, u(C) 0.123, u(MC) 0.039; Q2 (MS − MC) / MC +0.438 [+0.148, +0.728] worse; (MS − C) / C +0.793 worse; coverage C 0.901 (low bin 0.966, top 0.777), S 0.897 (low bin 0.944, top 0.820), MC 0.902 (low bin 0.899, top 0.902), MS 0.902 (low bin 0.898, top 0.907); fallback 0.00
- `spambase_probit_60nan`: K: Q1 u(MC) − u(C) -0.0044 [-0.0068, -0.0020] better, u(C) 0.018, u(MC) 0.013; Q2 (MS − MC) / MC +2.087 [+1.000, +3.173] worse; (MS − C) / C +2.146 worse; coverage C 0.892 (low bin 0.900, top 0.882), S 0.887 (low bin 0.901, top 0.867), MC 0.892 (low bin 0.890, top 0.893), MS 0.889 (low bin 0.888, top 0.890); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0055 [-0.0082, -0.0027] better, u(C) 0.017, u(MC) 0.012; Q2 (MS − MC) / MC +45.986 [-5.809, +97.780] none; (MS − C) / C +46.734 none; coverage C 0.893 (low bin 0.903, top 0.883), S 0.891 (low bin 0.905, top 0.870), MC 0.894 (low bin 0.892, top 0.895), MS 0.892 (low bin 0.891, top 0.893); fallback 0.00
- `biodeg_marf_60nan`: K: Q1 u(MC) − u(C) -0.0020 [-0.0067, +0.0027] equivalent, u(C) 0.017, u(MC) 0.015; Q2 (MS − MC) / MC +0.151 [+0.072, +0.230] none; (MS − C) / C +0.182 worse; coverage C 0.897 (low bin 0.885, top 0.907), S 0.900 (low bin 0.917, top 0.871), MC 0.900 (low bin 0.899, top 0.900), MS 0.903 (low bin 0.903, top 0.902); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0002 [-0.0076, +0.0073] equivalent, u(C) 0.021, u(MC) 0.021; Q2 (MS − MC) / MC -0.037 [-0.073, -0.001] better; (MS − C) / C -0.011 none; coverage C 0.898 (low bin 0.884, top 0.909), S 0.901 (low bin 0.915, top 0.872), MC 0.899 (low bin 0.898, top 0.900), MS 0.903 (low bin 0.902, top 0.902); fallback 0.00
- `kc2_marf_60nan`: K: Q1 u(MC) − u(C) +0.0139 [-0.0025, +0.0302] none, u(C) 0.038, u(MC) 0.052; Q2 (MS − MC) / MC -0.067 [-0.132, -0.001] better; (MS − C) / C +0.003 none; coverage C 0.897 (low bin 0.874, top 0.919), S 0.896 (low bin 0.915, top 0.861), MC 0.897 (low bin 0.898, top 0.896), MS 0.900 (low bin 0.901, top 0.900); fallback 0.00
  H: Q1 u(MC) − u(C) +0.0054 [-0.0111, +0.0219] none, u(C) 0.033, u(MC) 0.038; Q2 (MS − MC) / MC -0.133 [-0.207, -0.060] better; (MS − C) / C -0.046 none; coverage C 0.896 (low bin 0.876, top 0.919), S 0.896 (low bin 0.912, top 0.861), MC 0.901 (low bin 0.902, top 0.898), MS 0.898 (low bin 0.898, top 0.895); fallback 0.00
- `vehicle_marf_60nan`: K: Q1 u(MC) − u(C) -0.0043 [-0.0117, +0.0030] none, u(C) 0.024, u(MC) 0.020; Q2 (MS − MC) / MC +0.403 [+0.313, +0.492] worse; (MS − C) / C +0.421 worse; coverage C 0.900 (low bin 0.907, top 0.900), S 0.900 (low bin 0.921, top 0.876), MC 0.902 (low bin 0.899, top 0.903), MS 0.903 (low bin 0.902, top 0.901); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0104 [-0.0170, -0.0038] better, u(C) 0.027, u(MC) 0.017; Q2 (MS − MC) / MC +0.430 [+0.352, +0.509] worse; (MS − C) / C +0.450 worse; coverage C 0.901 (low bin 0.909, top 0.898), S 0.899 (low bin 0.920, top 0.873), MC 0.904 (low bin 0.902, top 0.905), MS 0.902 (low bin 0.901, top 0.903); fallback 0.00
- `spambase_marf_60nan`: K: Q1 u(MC) − u(C) -0.0000 [-0.0022, +0.0021] equivalent, u(C) 0.013, u(MC) 0.013; Q2 (MS − MC) / MC +2.173 [+0.924, +3.423] worse; (MS − C) / C +2.221 worse; coverage C 0.894 (low bin 0.888, top 0.898), S 0.890 (low bin 0.914, top 0.857), MC 0.893 (low bin 0.895, top 0.892), MS 0.891 (low bin 0.893, top 0.888); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0007 [-0.0032, +0.0017] equivalent, u(C) 0.009, u(MC) 0.008; Q2 (MS − MC) / MC +0.820 [+0.438, +1.202] worse; (MS − C) / C +0.840 worse; coverage C 0.897 (low bin 0.891, top 0.902), S 0.892 (low bin 0.916, top 0.859), MC 0.898 (low bin 0.899, top 0.897), MS 0.893 (low bin 0.895, top 0.892); fallback 0.00
- `credit-g_marf_60nan`: K: Q1 u(MC) − u(C) -0.0032 [-0.0115, +0.0051] none, u(C) 0.024, u(MC) 0.020; Q2 (MS − MC) / MC +0.342 [+0.066, +0.619] worse; (MS − C) / C +0.344 worse; coverage C 0.901 (low bin 0.906, top 0.910), S 0.893 (low bin 0.916, top 0.868), MC 0.903 (low bin 0.903, top 0.904), MS 0.896 (low bin 0.895, top 0.896); fallback 0.00
  H: Q1 u(MC) − u(C) -0.0150 [-0.0232, -0.0067] better, u(C) 0.034, u(MC) 0.019; Q2 (MS − MC) / MC +1.732 [+0.152, +3.312] worse; (MS − C) / C +1.741 worse; coverage C 0.903 (low bin 0.911, top 0.912), S 0.881 (low bin 0.904, top 0.857), MC 0.905 (low bin 0.905, top 0.906), MS 0.881 (low bin 0.875, top 0.885); fallback 0.00

- Q1, K: better 7 / no difference 3 / equivalent 2 / worse 0 (fold-pair t, seed agreement)
- Q2, K: better 1 / no difference 2 / worse 9 (fold-pair t, seed agreement)
- Q1, H: better 9 / no difference 1 / equivalent 2 / worse 0 (fold-pair t, seed agreement)
- Q2, H: better 2 / no difference 1 / worse 9 (fold-pair t, seed agreement)
```

## ADR 0030 amended (2026-11-06 16:55 GMT-3)

On the user's decision ("faça a emenda da 030"), ADR 0030 gains decision 8: the interval for a numerical fill is MC,
per column and per tertile of the row's gaps, without the spread, stated with its oracle calibration. Decision 3
(the spread is a rank) stands.
