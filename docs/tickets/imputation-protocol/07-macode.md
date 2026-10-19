# MaCoDE as an external deep baseline (E48; pre-registered 2026-10-19)

This ticket was written and committed before the first scored cell of the study.

- **Plan:** [2026-10-17-experiment-improvements](../../plans/2026-10-17-experiment-improvements.md)
  (§2g, item 10; D21).
  - The user chose MaCoDE as the next deep baseline on 2026-10-19 ("vamos agora para t28 e
    macode"). That is the go for vendoring it.
  - TabPFN v2 and its separate venv are not part of this.
- **Why:** every opponent so far is a classical imputer. MaCoDE (An et al., AAAI 2025) is the
  nearest prior art to TRIDENT.
  - It is a masked transformer over discretised rows: one merged mask-or-missing word per column
    (no `[NULL]`/`[MASK]` split) and quantile-bin targets.
  - It handles mixed and categorical tables natively.
  - It has no pre-training stage: the masked model is the imputer.
- **Code:** `src/training/macode.py`, vendored from github.com/an-seunghwan/MaCoDE at commit
  `8867879` (MIT, notice kept), with an inductive wrapper. The changes, each stated in the module:
  - fit on the training fold alone and fill any frame;
  - the bins passed to the data (the released code kept 10);
  - a table of one column type works;
  - the paper's appendix settings: 50 bins, width 128, 4 heads, 2 layers, 500 epochs, batch 1,024,
    AdamW 1e-3 with weight decay 1e-3;
  - its own random streams;
  - the posterior-mean fill for numbers and the argmax for categories.
  - Unit tests: `tests/unit/test_macode.py`. No training path of TRIDENT changes.

## Design (fixed before the first scored cell)

- **Arm:** MaCoDE as above, untuned, as the paper ran it. TRIDENT's configuration was tuned on the
  development set, so this is stated beside every comparison.
- **Cells:**
  - **The development set** at seeds 101/202/303, the thesis table's primary set: 21 variants and
    63 (variant, seed) pairs. K comes from the runs `report_common.development_source` names.
  - **E44's confirmation set** at seeds 707/808/909: 15 variants and 45 pairs. K comes from E44's
    runs.
  - 108 pairs and 540 fits in all, each fold rebuilt by `recompute_folds` exactly as its run
    built it.
- **Driver:** `scripts/experiments/macode_run.py`, on the GPU, off-store. Its CSVs go to
  `/scratch2/diogoneiss/trident-2026-10-19-macode/csv/` (lanes: `scripts/experiments/run_macode_lanes.py`).
- **Pre-launch smoke (2026-10-19), timing only, no score read:**
  - credit-g_60nan at seed 101: 53 s for 5 folds.
  - electricity_20nan at seed 101: 901 s, about 180 s a fold. The largest table sets the pace.
  - The smoke's CSVs were deleted unread.
- **Cost estimate:** by rows and columns, about 8 process-hours for the 108 pairs: about 3.7 h for
  the development set and 4.4 h for the confirmation set.
  - Three processes share the GPU, so the wall time is about 4–5 h.
  - It is off-store and runs in the GPU gap after E47.
  - No cutoff; a relaunch skips finished files.

## Measures

All on the induced `impute_score` over 15 fold pairs, with the verdict rule and the 1% SESOI
(ADR 0020), through `scripts/experiments/macode_report.py`.

1. **Primary:** K − MaCoDE per variant, tallied on each set.
2. MaCoDE − the strong tier's best (missForest and E41's arms), − the bar, and − the tuned bar
   (E42).
3. K − the best of all, MaCoDE included.
4. MaCoDE's fit time per fold.

## Reading, fixed now

**Descriptive on the development set,** where TRIDENT was tuned and MaCoDE was not.

**On the confirmation set:**

- **K − MaCoDE is the thesis's deep-baseline statement,** reported as a tally beside E44's.
- **Whether MaCoDE joins the strong tier:**
  - If it is better than the strong tier's best on at least 5 of the 15 confirmation variants,
    propose (by an ADR amending 0021) that it join the strong tier as an offline member, like the
    tuned bar.
  - Otherwise it is reported as an external deep baseline only.
- **No default changes either way.**

## Launch

To be filled in at launch.
