# 07. Imputation artifacts

Status: done (this commit) on `feat/imputation-task`
Blocked by: 05, 06
Plan task: 7. ADR 0004 decision 8. Wayfinder ticket 07.

## Goal

The three files an imputation run produces, written by `ArtifactWriter` from the decode
outcome's scored-cell table: the row-triptych preview over a fixed-seed sample, the
cell-ledger CSV holding every scored cell, and the per-column table on the parent.

## Seams under test

- `ArtifactWriter.write_imputation_preview(name, scored_cells, seed, fold, n_rows,
  scaler, label_encoders) -> tuple[Path, Path]`: one sample draw feeds both files.
- `ArtifactWriter.write_per_column_imputation(records) -> Path`.

## Acceptance criteria

- [x] The preview contains exactly the sampled rows, in original units (inverse scaler
      and label decoding), with a model-saw line distinguishing `[NULL]->[MASK]` from
      `[MASK]`.
- [x] The CSV contains every scored cell, is a strict superset of the preview, carries
      both scalings and confidence, and its `in_preview` flag is true for exactly the
      previewed cells.
- [x] The per-column table has one row per (fold, column, population, metric).
- [x] File names follow the plot convention: `imputation_preview_fold_N.md`,
      `imputation_cells_fold_N.csv`, no suffix for a single split.

## Constraints

- The prototype on branch `prototype/imputation-preview` is the visual reference. Rewrite
  it properly; do not promote it.
- Rendering stays in the artifact writer; the decode stage returns data only.

## Comments

- 2026-09-10, four red-green slices in `tests/unit/test_imputation_artifacts.py`:

  | Slice | Behaviour pinned |
  |---|---|
  | 1 | The ledger holds every scored cell and says which are previewed |
  | 2 | The preview reads in the units a person recognises |
  | 3 | The per-column table says which column a poor fold struggled with |
  | 4 | A wide row is split so the preview stays readable |

  **Slice 4 came from looking at real output, not from the plan.** A real
  `credit-g_20nan` row had eleven cells filled in, and one markdown table that wide reads
  as noise; spambase's 57 columns would be worse. The preview now breaks a row into
  blocks of six, which no unit test would have suggested.

  Two findings recorded rather than designed around:

  - The ledger holds both kinds of cell in one table, so a value column round-trips
    through CSV as text and a reader casts by kind. That is the price of keeping the whole
    record in a single file, which ticket 07's wayfinder decision chose deliberately.
  - `write_hyperparameters` still writes the fixed classification key set, so an
    imputation run's `hyperparameters.json` would claim `EPOCH_FINE` and `LABELS`. It
    takes `Hyperparameters` and cannot see the task; **passed to ticket 08**, where the
    runner has the request.

  **Real output**, `credit-g_20nan` fold 1 after 60 pre-training and 60 decode epochs,
  `impute_score` 0.966 over a 3351-row ledger and an 80-row per-column table. One sampled
  row shows the model recovering all three of its hidden cells exactly: `employment`
  `1<=X<4`, `foreign_worker` `yes`, and `installment_commitment` 3.021 against a true 3.

  Unit suite 113 green; the vehicle regression passes unedited.
