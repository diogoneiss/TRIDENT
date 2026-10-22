# Do decode masks that follow the missingness help under MAR? A propensity-mask arm (E65; pre-registered 2026-10-22)

This ticket was written and committed before the study's first cell. Two looks came first, and
they are disclosed under "Before the ticket".

- **Why:** E63 found the model losing ground against TabPFN under MAR, and E64 asks whether that
  holds.
  - E63 named an untested hypothesis. The decode stage hides training cells uniformly over the
    observed ones, while under MAR the test gaps sit in rows the other columns pick out.
  - This arm makes the training masks follow the fold's own missingness.
- **The user's go:** "vamos fazer o 1 e 2 agora" (2026-10-22, about 18:55 GMT-3). This is item 2;
  E64 is item 1.
- **The code,** behind a flag, with the old behaviour as the default:
  - **`--decode_mask_weighting propensity`** (`DECODE_MASK_WEIGHTING`), on branch
    `feat/decode-mask-propensity` (commit dd33b81, pushed);
  - **`src/training/mask_propensity.py`:**
    - per feature column with at least 20 gaps and 20 observed cells in the training fold, a
      logistic model (C = 0.1) of whether a cell is missing, on the row's other columns;
    - a numerical column enters standardised with a gap at 0, plus its missing indicator; a
      categorical one as indicators of up to 10 levels, plus its missing indicator;
    - a cell's weight is its propensity over the column's mean on observed cells. The decode stage
      hides it with the uniform probability times that weight, clipped at 1.
    - The unmasked-row fallback stays uniform. With unit weights, the draw equals today's draw
      for draw.
  - **The tag** `decode_mask_weighting`. A backfill script stamps `uniform` on earlier runs; its
    `--apply` waits on the user's go and a store backup.
  - **Checks:**
    - the unit suite (427 tests with the typing gate) passes;
    - the new tests pass (weights, draw, wiring, backfill);
    - both integration fixtures pass unchanged: vehicle_00nan and credit-g_20nan, on the CPU.
- **It runs beside E64, from a worktree** (`/scratch2/diogoneiss/TRIDENT-propensity`, the branch
  above).
  - The worktree has its own virtual environment. Its `mlflow.db` and `results/baseline_cache` are
    links to the checkout's store and cache, and `datasets/processed_datasets` is a link to the
    data.
  - So `src/` in the checkout E64 imports is never touched.
  - The branch is merged into `feat/imputation-task` after both studies end.

## Design (fixed before the first cell)

- **Cells:** E64's 26 variants × seeds 404/505/606 make 78 cells of arm P.
  - Each is today's default plus `DECODE_MASK_WEIGHTING propensity`, with `<base>_<pp>nan`'s
    resolved configuration as an override, `--score_checkpoints` and `--score_calibrated`.
  - The tag is experiment=mar-propensity-2026-10-22, arm P.
- **The comparison is E64's G** on the same variant, seed and folds. TabPFN and the strong tier
  come from E64's runs and CSVs.
- **Running:** two queues (`run_mar_propensity_queue.sh`) beside E64's four, with no cutoff. State
  in `/scratch2/diogoneiss/trident-2026-10-22-mar-propensity/`.

## Measures and reading, fixed now (`mar_propensity_report.py --tally`)

On the induced `impute_score` over 15 fold pairs, with the verdict rule:

- **Per table-rate and mechanism, with the 1% SESOI of G's value:** P − G on H, K and KC.
- **The interaction, fold-paired:** (P − G on H), MAR minus control.
- **Each arm's** (H − strong), MAR minus control: P's beside G's (E64's).
- **Primary,** over the 12 table-rates at slope 1.5:
  - **supported:** P − G on H under MAR is better on 4 or more and worse on at most 1, and under
    the control it is worse on at most 1;
  - **refuted:** under MAR it is better on at most 1;
  - **mixed:** otherwise.
- **The dial,** descriptive: P − G on H under MAR on biodeg_60nan at slopes 0.75, 1.5 and 3.
- **What each reading leads to (no default changes; MAR is never known in practice):**
  - **supported:** the hypothesis stands. Before any ADR, a pre-registered check that P is neutral
    on the shipped MCAR variants, and a fresh-seed MAR confirmation.
  - **refuted:** the mask mismatch is not the mechanism. MAR against TabPFN stays a limit, with its
    cause open.
  - **mixed:** reported, and the user chooses.

## Before the ticket (disclosed)

- **The regularisation C = 0.1 was chosen on the weights alone,** before any score.
  - The measure was their spread on one training fold (seed 404, fold 1) of E64's biodeg and
    kr-vs-kp at 60nan, MAR against control.
  - At 0.1 the MAR spread is 0.44 and 0.41, against 0.22 and 0.11 on the controls. At 1.0 it is
    0.50 and 0.47, against 0.27 and 0.12.
- **One short training smoke ran** to show the path works end to end.
  - biodeg_marf_60nan, seed 404, with EPOCHS_PRE 10 and EPOCHS_DECODE 30, so not E65's
    configuration.
  - K: uniform 0.9219, propensity 0.9086. It is one seed at a tenth of the epochs, so not a
    result, but I have seen it.
