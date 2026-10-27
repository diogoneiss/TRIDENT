# Which columns does the decoder fill each column from? (E78; pre-registered 2026-10-26)

This ticket was written and committed before the study's first cell.

- **Why:** the project has no interpretability measure. ADR 0030 gives a per-cell signal of *how
  sure* a fill is; nothing says *from what* it was filled. T38.
- **The user's go:** asked for interpretability work ("gostaria de trabalhar com
  interpretabilidade", 2026-10-26), was offered studies A–F, and accepted A + B + C ("sim,
  vamos!", 2026-10-26, before 22:45 GMT-3).
- **Code:** no change to `src/`.
  - `scripts/experiments/occlusion_map.py` trains each cell as a default run of the checkout and,
    after each fold's decode stage and before the next fold, reads the run's own model (K) in place.
    It wraps `runner.train_and_evaluate_decoder`, so it gets the fold's exact model, split and
    complete table, and nothing has to be rebuilt offline.
  - The global numpy, torch and CUDA random states are saved before the reading and restored after
    it. The model goes back to its training mode.
  - The plan's `--save_model` was dropped: reading in place needs no saved model.
  - The new table is made by `datasets/generate_synthblock.py`, plus
    `datasets/categorical_columns/synthblock.txt`.
- **Checks run before this ticket, on cells outside the study:**
  - **vehicle_20nan at seed 101** (`--no-mlflow`): K reproduced the thesis table's run
    (`selection-2026-10-03`) to 0.0 on all five folds. The unhidden pass equals the fold ledger's
    fill cell for cell on every fold; the script asserts this.
  - **synthblock_40nan at seed 1:** it ran end to end (7 min 21 s). The table loads under the
    defaults, with its three categorical columns.
  - **The reader's functions** were run on both cells' files. They printed only whether each value
    was finite; no value of any measure was seen.

## The method

**Occlusion.** At the test split's real gaps, shown as `[MASK]` as the induced scoring shows them,
hide one source column c: its observed cells become `[MASK]` too.
- Under today's default (`DECODE_GAP_TOKEN mask`) a hidden cell and a real gap are the same input,
  and the decode stage trains on exactly such hidden cells. So the question stays inside the
  training distribution.
- **The effect of c on target j:** the mean change in j's fill. It is taken over j's gap cells in
  the rows where c is observed. Elsewhere, hiding c changes nothing.
  - For a number: the absolute change, in scaled units.
  - For a category: the total-variation distance between the two probability vectors.
  - It is undefined (NaN) on fewer than 10 such cells.
- It is computed at K's weights and on the raw output. ADR 0027's per-column affine calibration
  scales every source's effect on one target alike, so it reorders nothing.

## Design (fixed before the first cell)

- **Cells:** ten variants at 40nan, three seeds each, 30 cells.
  - The synthetic table `synthblock_40nan` at 707, 808 and 909.
  - E44's six confirmation tables at 40nan, at 707, 808 and 909: pendigits, letter, electricity,
    vehicle, biodeg and kc2.
  - The three development tables at 40nan, at 101, 202 and 303: kr-vs-kp, spambase and credit-g.
  - Each cell is today's default configuration, five folds, with `--score_remask`. Its spread stays
    in the ledgers for a later study (E in the proposal) and is not read here.
- **The synthetic table** (`generate_synthblock.py`, 3,000 rows, MCAR at 40% through the shipped
  `inject_nans`):
  - three independent blocks of four columns, each driven by its own latent, plus three pure-noise
    columns;
  - the relations inside a block are nonlinear, and two pairs have Pearson ≈ 0 (a1 with a2 and a4);
  - two block columns are categorical, and so is one noise column.
  - **The truth:** a block column's informative sources are exactly the other three of its block.
- **Running:** four GPU queues (`run_occlusion_queue.sh`), MLflow tag
  `experiment=occlusion-map-2026-10-26`, state in `/scratch2/diogoneiss/trident-2026-10-26-occlusion/`.
  About 3–4 h; the three electricity cells set it, at about 1.5 h each as in E75.

## Measures and reading, fixed now (`occlusion_map.py --tally`)

**Primary, B: is the map faithful to the model on real data?**
- **Ranking:** for each target j, its sources ranked by their effect on the *validation* split's
  gaps.
- **Deletion:** on the *test* split, hide j's top 3 sources in every row and measure the error of
  j's gap cells. Compare with 3 sources drawn at random from j's others, averaged over 20 draws from
  a private stream seeded by `[seed, fold, 78]`.
  - The error is absolute error in scaled units for a number, and 1 − p(true category) for a
    category.
  - A category the variant never shows is left out.
- **One fold's D:** the mean, over targets with at least 10 gap cells, of
  (top-3 error − random-3 error) / unhidden error. A positive D means the map's top sources matter
  more than random ones.
- **The verdict on each variant (15 folds):**
  - **faithful** if the 95% t-interval lies above the 1% SESOI and every seed's mean is above 0;
  - **not faithful** if the interval lies below the SESOI;
  - inconclusive otherwise.
- **The rule over the 9 real variants:**
  - **supported:** faithful on 7 or more, and not faithful on none;
  - **refuted:** faithful on 4 or fewer;
  - **mixed:** otherwise.

**Gate, C: does the map tell the truth where the truth is known?** (synthblock only)
- **One fold's value:** the mean over the 12 block columns of the AUROC with which the test map's
  effects separate the in-block sources (3) from the rest (11).
- **The verdict:**
  - **recovers the blocks** if the 95% t-interval over 15 folds lies above 0.8 and every seed's
    mean is above 0.8;
  - **does not** if the interval lies below 0.8;
  - inconclusive otherwise.
- **The floor:** if K's mean `impute_score` on synthblock is not below `mean_mode`'s, the model
  learned nothing to read, and C is **inconclusive (floor)**, not failed.

**Descriptive, printed by the same reader (no verdicts):**
- **A**, the map itself: each target's top sources, averaged over the 15 test maps; the maps'
  stability across seeds (Spearman between seed-mean maps); and agreement, within each target,
  with the training split's |Spearman| between numerical columns;
- D with the top 1 source;
- D on synthblock;
- C's AUROC computed from |Spearman| instead of the map, on numerical pairs only. It shows what a
  linear reading misses;
- **D (study D):** attention against occlusion. It is the mean attention from j's token to c's,
  averaged over layers and heads at the unhidden pass, set beside the effect within each target.
  It is never a claim;
- K against the earlier run of the same cell: E44's (`confirmation-2026-10-17`) or the thesis
  table's (`development_source`).

**What each reading leads to:**
- **B supported and C recovers the blocks:** the thesis can say that the occlusion map is faithful
  to the decoder and recovers known structure. It reports the map as an output of the method,
  descriptively, with no ADR.
- **B supported, C does not:** the map is faithful to the model, but the model's dependencies are
  not the data's. That is reported as a finding about the decoder, not as an explanation tool.
- **B mixed or refuted:** the map is not used as an explanation. The tables where it fails are
  named.

## Limits fixed now

- **K only.** ADR 0024 reads a claim that must transfer on H as well. H's weights are internal to
  the decode stage and are not reachable without a `src/` change, so H is not read. A claim from
  this study says "K".
- **MCAR at 40nan only.** MAR, MNAR and other rates are untested. Study F in the proposal (reliance
  on the gap pattern under MNAR) is a separate ticket.
- **One column at a time.** Redundant sources share an effect, so a column with a substitute looks
  weak. C's blocks have three informative sources each, so C tests the map under that redundancy.
