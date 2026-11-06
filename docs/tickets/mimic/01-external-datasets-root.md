# Datasets read from outside the checkout (T50, stage 0; written 2026-11-05)

Stage 0 of T50: what E148 to E150 ([03](03-masked-cells-natural-gaps.md), [04](04-pattern-transplant.md),
[05](05-downstream-mortality.md)) need before their first cell. Not an experiment.

- **Why:** the user wants tables derived from MIMIC-IV studied here (2026-11-05), and set the rule: the full tables
  are always read from a path outside the project folder; a small real cohort lives in the repository as a toy
  problem. The repository is private, exists to sync the user's machines and will be deleted later.
- **What reads a dataset today** (hard-coded, relative to the checkout): `src/training/data.py`
  (`PROCESSED_DATASETS`, the variant's CSV and `splits/<base>_split.json`), `src/training/discovery.py`
  (`_PROCESSED_DIR`), `src/training/data.py` (`datasets/categorical_columns/<base>.txt`) and
  `src/training/config.py` (`datasets/hiperparams/<base>/`).

## The change

- `dataset_directory(base)` in `src/training/data.py`: the checkout's `datasets/processed_datasets/<base>`, or
  `$TRIDENT_DATASETS_ROOT/<base>` when the variable is set and the base is there. A base present in both is an
  error, not a silent choice. The variant's CSV, its `_00nan` sibling, the split file and `opt.py`'s header read all
  go through it.
- The variable unset leaves every path as it is today: classification stays bit-identical and both integration
  fixtures are untouched.
- `discover_datasets` (`main.py --all`) still lists the checkout only. With the variable exported from `~/.bashrc`,
  listing the external root would add the donor tables to every `--all --nan_level 0` run. External tables are run
  by name.
- `categorical_columns/<base>.txt` and `hiperparams/<base>/` stay in the repository: they hold column names and
  settings, no patient value.
- Out of scope: the 20 scripts under `scripts/`, `datasets/` and `tests/` that name `processed_datasets` themselves.
  They belong to earlier studies; the MIMIC studies get their own runner.
- `--export_gap_fills` (off by default, imputation only): each fold writes
  `imputation_fold_<k>_gap_fills.csv`, its kept checkpoint's fill for every gap of the table (training, validation
  and test rows; `row`, `split`, `column`, `kind`, `imputed`, `confidence`, `imputed_original_units`), and
  `imputation_fold_<k>_partition.csv`, every row's split. The gaps are shown as `[MASK]`, as the induced scoring
  shows them. Nothing is scored. E150 needs it: the ledger holds scored cells only.

## Where things live

| what | gorgona8 | medusa1 |
|---|---|---|
| raw MIMIC-IV 3.1 (read-only) | `/scratch2/pedrobacelar/mimic/physionet.org/files/mimiciv/3.1` (another user's tree, not writable) | `/scratch/diogoneiss/mimic/physionet.org/...` |
| derived tables, `TRIDENT_DATASETS_ROOT` | `/scratch/diogoneiss/trident-data` | `/scratch/diogoneiss/trident-data` |

- One path for the derived tables on both machines, as ADR 0033's canonical root; exported from `~/.bashrc`. The
  builder takes the raw tree from `MIMIC_ROOT`.
- The tables are built on gorgona8 (the SOFA component files and the kg-impute tensors exist only there) and reach
  medusa1 by `rsync` over ssh, never through git and never through `sync_remote.py`.
- The runner keeps its state directory outside the checkout, as every study's does.
- Rows, values and identifiers may be read, printed and quoted like any other dataset's (the user's decision of
  2026-11-05, in `AGENTS.md`).

## The toy problem

`mimic-cardio-toy`: 1,000 real stays of `mimic-cardio` ([02](02-tables.md)), drawn with a fixed seed and stratified
by the label (16.5% deaths, 122 rows with every feature), with the table's columns and its natural gaps as they are.
The user's decision of 2026-11-05 (a real cohort, not a synthetic one).

It is tracked where the loader reads it, `datasets/processed_datasets/mimic-cardio-toy/mimic-cardio-toy_natural.csv`,
with its row index ([02](02-tables.md)) beside it, so it reaches Windows and medusa1 through git and needs no
generation step. That directory is ignored (`.gitignore`, and `.git/info/exclude`, which a later rule cannot
undo), so the two files were added with `git add -f`; once tracked, the ignore rules no longer apply to them. Not
under `datasets_raw/`: `generate_splits.py` would cut `_NNnan` variants on top of the natural gaps.

## Done when

- [x] A unit test on the resolver (two roots, the ambiguous base, the variable unset) and one on the export (every
  gap once, each with its split): `tests/unit/test_training_data.py`, `tests/unit/test_training_decoding.py`.
- [x] `pytest -m "not integration"` (497 passed, with the mypy gate; the export's test, added during that run,
  passed on its own) and `pytest -m integration` (both fixtures, the variable unset), 2026-11-06 00:16 GMT-3.
- [x] Smoke on `mimic-cardio-toy_natural` (2026-11-05, CPU, two folds, 40 decode epochs, MLflow off, 9 minutes
  beside the night's trainers): (a) a table with natural gaps and no `_00nan` sibling trains and scores the masked
  population with its baseline bar; (b) `--decode_checkpoint masked` keeps the masked criterion's epoch
  (`decode/masked_checkpoint_epoch` is logged); the export holds 2,894 fills for the table's 2,894 gaps, none
  twice, every split agreeing with the partition file. The fills of so short a run are close to the column mean;
  the smoke checks the plumbing, not the model. Not exercised: the MLflow tags of a name without a `_NNnan`
  suffix (`missingness_percent` will read `unknown`); E148's first cell is the first tracked run.
- [x] Smoke on an external table: `mimic-cardio-donor_00nan` read from `$TRIDENT_DATASETS_ROOT` (CPU, two folds,
  40 decode epochs) trains and scores; a complete table exports no fills and its partition.
- [x] (c) minutes per cell on the 11,082-row tables. Amended 2026-11-06 (the user: "Comece a rodar em paralelo com
  o kg impute"): measured on E148's first cells, which share the GPU with kg-impute's queue, not on a free GPU.
  `mimic-sofa_natural` took 33 minutes (07:22 to 07:54 GMT-3) and `mimic-cardio_natural` 36 (07:21 to 07:56), five
  folds each, two cells at a time beside kg-impute's four trainers. They were also the first tracked runs of a
  name without a `_NNnan` suffix: the parents and their mirrors finished on the leader, `missingness_percent`
  reading `unknown`.

## When

Built in the worktree `feat/mimic-stage0` while the detached chain of 2026-11-05 launched cells from the main
checkout; merged after it ends. Rebased onto the pilots' commit and fast-forwarded into `feat/imputation-task` on
2026-11-06 07:03 GMT-3, after the chain's last cell (04:51 GMT-3).
