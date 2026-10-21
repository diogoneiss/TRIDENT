# ADR 0018's bigger encoder on the large tables at 40–80nan (E59; pre-registered 2026-10-21)

This ticket was written and committed before the first cell.

- **Why:** ADR 0018 put E37's ALL arm on pendigits_20nan and letter_20nan. ALL is 4 layers,
  feed-forward 256 and a final LayerNorm.
  - E38 tested it only on the 21 development variants. The large tables appear there at 20nan
    alone (pendigits −3.8%, letter −5.6%; electricity +0.3%, n.s.).
  - It was never tried at 40–80nan, where E55 and E57 find TabPFN ahead of the default output at
    80nan on pendigits and letter.
- **The user's go:** "sim, vamos" (2026-10-21, about 19:20 GMT-3), to the proposal of the bigger
  encoder at 40–80nan tonight.
- **No new code in `src/`.** ALL is a configuration (`LAYERS 4`, `DIM_FEED 256`,
  `ENCODER_FINAL_NORM layer`). Runner: `scripts/experiments/large_encoder_run.py`; tag
  `experiment=large-encoder-2026-10-21`, arm `ALL`.

## Design (fixed before the first cell)

- **Cells:** pendigits and letter at 40, 60 and 80nan, seeds 707/808/909, arm ALL: 18 cells.
- **G** is E44's run of today's default on the same seed and folds. No promoted file applies to
  these variants.
  - E44 ran before ADR 0027, and the K path has not moved since. E51 reproduced E35 and E50
    reproduced E38 and E39, each to a difference of 0.
  - KC is E44's `--score_calibrated` readout against ALL's ADR 0027 headline: the same calibration
    on the validation gaps.
- **electricity is left out.** ALL did not help it at 20nan (E38), and a cell would take about 3 h.
- **The seeds are E44's confirmation seeds,** so this is a development reading.
  - Any adoption needs a fresh-seed confirmation, pre-registered first, before an ADR amending
    0018.
- **Running it:** four GPU queues (`run_large_encoder_queue.sh`), longest first. About 6.6 h,
  beside E60's one lane. No cutoff.

## Measures and reading, fixed now

`scripts/experiments/large_encoder_report.py --tally`, on the induced `impute_score` over 15 fold
pairs, with the verdict rule and the 1% SESOI:

- **Primary:** ALL − G on K.
  - **supported:** better on 4 or more of the 6, worse on none;
  - **refuted:** better on 1 or none, or worse on 2 or more;
  - **mixed:** otherwise.
- **Secondary:**
  - ALL − G on KC (the default output) and on H;
  - G's and ALL's KC − TabPFN (E55's CSVs): does the bigger encoder close the 80nan gap?
- **What each reading leads to (no default changes from this study):**
  - **supported:** a fresh-seed confirmation, then an ADR amending 0018 for the user;
  - **refuted:** ADR 0018 stays at 20nan;
  - **mixed:** reported, and the user chooses.
