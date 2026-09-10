# 07. Imputation artifacts

Status: ready-for-agent
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

- [ ] The preview contains exactly the sampled rows, in original units (inverse scaler
      and label decoding), with a model-saw line distinguishing `[NULL]->[MASK]` from
      `[MASK]`.
- [ ] The CSV contains every scored cell, is a strict superset of the preview, carries
      both scalings and confidence, and its `in_preview` flag is true for exactly the
      previewed cells.
- [ ] The per-column table has one row per (fold, column, population, metric).
- [ ] File names follow the plot convention: `imputation_preview_fold_N.md`,
      `imputation_cells_fold_N.csv`, no suffix for a single split.

## Constraints

- The prototype on branch `prototype/imputation-preview` is the visual reference. Rewrite
  it properly; do not promote it.
- Rendering stays in the artifact writer; the decode stage returns data only.

## Comments
