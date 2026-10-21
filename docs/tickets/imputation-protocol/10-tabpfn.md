# TabPFN v2 as an external foundation-model baseline (E55; pre-registered 2026-10-20)

This ticket was written and committed before the first scored cell of the study.

- **Plan:** [2026-10-17-experiment-improvements](../../plans/2026-10-17-experiment-improvements.md)
  (§2g, item 11; D21).
- **The user's go** came in two steps:
  - 2026-10-20 afternoon: add TabPFN through uv instead of a separate venv, after T28 and T29;
  - 2026-10-20 about 23:10 GMT-3: "comece a executar experimentos que não precisam da minha
    supervisão direta, como aquilo do tab", with the machine until 08:30.
- **Why:** MaCoDE (E48) is the nearest deep prior art. TabPFN v2 (Hollmann et al., *Nature* 2025) is
  the tabular foundation model a reader will ask about.

## The code (landed with this ticket)

- **The dependency:** an optional extra, `tabpfn==9.2.0` (Apache-2.0), in `pyproject.toml`.
  - The lock gains packages and changes none: torch 2.5.1+cu121, scikit-learn 1.9.0, numpy 2.4.6,
    pandas 2.3.3, scipy 1.17.1 and lightgbm 4.7.0 are as before.
  - Without `--extra tabpfn` nothing imports it, so the default environment and Windows are
    unaffected.
- **`src/training/tabpfn_baseline.py`:** a column-wise, inductive wrapper. Its docstring has every
  choice; the main ones:
  - TabPFN v2's checkpoints, through `create_default_for_version(ModelVersion.V2)`. The package
    defaults to the gated v3.5.
  - One regressor or classifier a column, fitted on the training fold's rows where the column is
    observed, with the other columns as features. Gaps and hidden cells enter as NaN, and
    categories as training-fold indices declared categorical.
  - A context of at most 10,000 rows (v2's limit), subsampled with a fixed seed on letter and
    electricity.
  - 8 estimators (v2's default, named explicitly), `random_state` 0, the GPU.
  - Its own random streams; constant fills for a column with nothing to learn.
  - **Only the asked-for cells are predicted,** and a fold's masked call also predicts the frame's
    own gaps in the same pass, so its induced call reuses them.
    - TabPFN's fills depend slightly on which rows share a `predict`. On electricity_20nan at
      seed 7, a fold's induced `impute_score` moved by at most 1.6e-4 against separate calls,
      and the masked one by 0.
- **Weights:** downloaded once to `/scratch2/diogoneiss/tabpfn-weights` (`TABPFN_MODEL_CACHE_DIR`).
  They are the package's v2 defaults, Prior Labs License 1.1 (Apache 2.0 with an attribution
  requirement), not gated:
  - `tabpfn-v2-regressor.ckpt` (Hugging Face `Prior-Labs/TabPFN-v2-reg`, revision `4972a65`);
  - `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` (`Prior-Labs/TabPFN-v2-clf`, revision
    `f851f2a`).
  - The thesis cites Hollmann et al. 2025 wherever it reports TabPFN.
- **Checks:**
  - unit tests with a fake estimator (`tests/unit/test_tabpfn_baseline.py`, 5 tests);
  - the unit suite (419 passed, the typing gate included);
  - both regression fixtures, unchanged.
  - No training path of TRIDENT changes.
- **Driver and reading:**
  - `scripts/experiments/tabpfn_run.py` (E48's driver, through `recompute_folds`);
  - `tabpfn_report.py` (E48's report, with MaCoDE in the best of all where its CSVs exist);
  - `run_tabpfn_lanes.py`;
  - `report_common.DEEP` gains `tabpfn`: outside the bar and the strong tier, inside the best of
    all.

## Design (fixed before the first scored cell)

- **The arm:** TabPFN v2 as above, untuned. TRIDENT's configuration was tuned on the development
  set, so this is stated beside every comparison.
- **The cells,** as in E48:
  - **E44's confirmation set** at seeds 707/808/909: 15 variants, 45 pairs, which run first in
    every lane;
  - **the development set** at seeds 101/202/303: 21 variants, 63 pairs. K comes from
    `report_common.development_source`.
- **The smoke (2026-10-20, seed 7, not a study seed), timing only:**
  - electricity_20nan: 121 s for five folds with the one-pass fill;
  - before it: spambase_20nan 1,129 s and credit-g_60nan 40 s.
  - The column count drives the cost: spambase's 57 columns are the slowest pairs.
  - The smoke's CSVs were read only for the old-against-new difference above, then deleted.
- **Running it:** two GPU lanes (`run_tabpfn_lanes.py`), off-store, from 01:00 GMT-3 on
  2026-10-21.
  - About 6.4 h estimated in all; the development pairs may not all finish.
  - **The deadline:** a pair starts only if now + 1.3 × its estimate is before 08:00 GMT-3. A
    skipped pair is logged.
  - **Incomplete variants:** a variant without all three seeds is reported as incomplete and not
    read. The rest may run in a later GPU gap; a relaunch skips finished files.

## Measures

All on the induced `impute_score` over 15 fold pairs, with the verdict rule and the 1% SESOI (ADR
0020), through `scripts/experiments/tabpfn_report.py`:

1. **Primary:** K − TabPFN per variant, tallied on each set.
2. TabPFN − the strong tier's best (missForest and E41's arms), − the bar, and − the tuned bar
   (E42).
3. K − the best of all, TabPFN and MaCoDE included.
4. TabPFN's fit time per fold.

## Reading, fixed now (E48's)

**Descriptive on the development set,** where TRIDENT was tuned and TabPFN was not.

**On the confirmation set:**

- **K − TabPFN is the thesis's foundation-model statement,** reported as a tally beside E44's and
  E48's.
- **Whether TabPFN joins the strong tier:**
  - If it is better than the strong tier's best on at least 5 of the 15 confirmation variants,
    propose (by an ADR amending 0021) that it join the strong tier as an offline member.
  - Otherwise it is reported as an external baseline only.
- **No default changes either way.**
