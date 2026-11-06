# Datasets read from outside the checkout (T50, stage 0; written 2026-11-05)

Stage 0 of T50: what E150 to E152 ([03](03-masked-cells-natural-gaps.md), [04](04-pattern-transplant.md),
[05](05-downstream-mortality.md)) need before their first cell. Not an experiment.

- **Why:** the user wants tables derived from MIMIC-IV studied here (2026-11-05), and set the rule: the full tables
  are always read from a path outside the project folder; a small real cohort lives in the repository as a toy
  problem. The repository is private, exists to sync the user's machines and will be deleted later.
- **What reads a dataset today** (hard-coded, relative to the checkout): `src/training/data.py`
  (`PROCESSED_DATASETS`, the variant's CSV and `splits/<base>_split.json`), `src/training/discovery.py`
  (`_PROCESSED_DIR`), `src/training/data.py` (`datasets/categorical_columns/<base>.txt`) and
  `src/training/config.py` (`datasets/hiperparams/<base>/`).

## The change

- One resolver for a base dataset's directory, over a list of roots: the in-repo `datasets/processed_datasets`
  first, then `$TRIDENT_DATASETS_ROOT` when it is set. `data.py` (variant and splits) and `discovery.py` use it. A
  base name present in both roots is an error, not a silent choice.
- The variable unset leaves every path as it is today: classification stays bit-identical and both integration
  fixtures are untouched.
- `categorical_columns/<base>.txt` and `hiperparams/<base>/` stay in the repository: they hold column names and
  settings, no patient value.
- Out of scope: the 20 scripts under `scripts/`, `datasets/` and `tests/` that name `processed_datasets` themselves.
  They belong to earlier studies; the MIMIC studies get their own runner.
- Export of the model's fills for a table's **natural** gaps (the ledger holds scored cells only; E152 needs the
  fills). Behind a flag that is off by default. Per fold, it writes that fold's decoder's fills for every row's
  natural gaps (training, validation and test) and the fold's row partition.

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
by the label, with the table's columns and its natural gaps as they are. Written by the builder to
`datasets/datasets_raw/mimic-cardio-toy.csv`, tracked like the other raw tables, so it reaches Windows and medusa1
through git; its row index ([02](02-tables.md)), with the identifiers, is tracked beside the column declarations
as `datasets/row_index/mimic-cardio-toy.csv`. The user's decision of 2026-11-05 (a real cohort, not a synthetic one).

## Done when

- A unit test on the resolver: two roots, the ambiguous base, the variable unset.
- `pytest -m "not integration"` (with the mypy gate) and `pytest -m integration` pass.
- A smoke on `mimic-cardio-toy` and one on an external table, both short imputation runs, answer and record here:
  (a) a table with natural gaps and no `_00nan` sibling trains and scores the masked population with its baseline
  bar, and a name without a `_NNnan` suffix passes the tags and the baseline cache; (b) `--decode_checkpoint masked`
  logs `masked` as the criterion kept; (c) minutes per cell on the 11,082-row tables.

## When

Not while the detached chain of 2026-11-05 launches cells from this checkout: after it ends, or in a worktree.
