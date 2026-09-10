# 05. Data support for evaluation

Status: done (this commit) on `feat/imputation-task`
Blocked by: none
Plan task: 5. ADR 0004 decisions 5 and 8. Wayfinder tickets 03, 07.

## Goal

Three data-layer capabilities the decode stage and the preview need: the fitted scaler
retained on the prepared dataset, the row-aligned `_00nan` sibling located and verified,
and a fixed evaluation mask drawn per fold without disturbing the global random state.

## Seams under test

- `prepare_dataset(spec)`: `PreparedDataset.scaler` holds the fitted `StandardScaler`
  (`None` when there are no numerical columns); split-and-scale order unchanged.
- `load_complete_sibling(spec) -> pd.DataFrame | None` in `src/training/data.py`: finds
  `<base>_00nan.csv`; asserts shape, columns and every observed cell equal; returns
  `None` when absent or when the variant is itself `_00nan`.
- `evaluation_mask(frame, rate, seed, fold) -> pd.DataFrame`: one `preprocess_table`
  draw from a generator seeded by `(seed, fold)`.

## Acceptance criteria

- [x] Scaler present after `prepare_dataset`; existing data tests untouched and green.
- [x] Sibling found for a `_20nan` fixture frame; `None` for `_00nan`.
- [x] Alignment assertion raises on a deliberately altered observed cell.
- [x] Evaluation mask identical across two calls with the same `(seed, fold)`, different
      across folds.
- [x] Global `np.random` state is identical before and after the evaluation draw.

## Constraints

- `preprocess_table` uses global `np.random`; wrap the evaluation draw in a
  `get_state` / `set_state` guard around a local reseed so it consumes no global draws.
  This is what keeps the classification draw sequence untouched.
- The sibling is read for scoring only; nothing in training may call it.

## Comments

- 2026-09-10, five red-green slices in `tests/unit/test_training_data.py`:

  | Slice | Behaviour pinned |
  |---|---|
  | 1 | A prepared dataset can return its numbers to original units |
  | 2 | A variant with injected gaps can find the table they were cut from |
  | 3 | A sibling that does not line up is refused |
  | 4 | Scoring the same fold twice asks the same question |
  | 5 | Drawing an evaluation mask leaves the training draws alone |

  Slice 3 needed a fabricated pair of tables on disk, so the dataset root became the
  module constant `PROCESSED_DATASETS` that a test can redirect. `prepare_dataset` now
  uses it too, which removes a duplicated literal path.

  Verified on real `credit-g_20nan`: the sibling loads as 1000x21 with no missing cells
  and covers all 4000 gaps; `duration` reads back as 6.00 from -1.223 and `credit_amount`
  as 1169.00 from -0.736, both matching the CSV.

  **Finding, for the ADR and ticket 06: `EVAL_MASK_RATE` is nominal, not realised.**
  `preprocess_table` scales the masking probability down by each row's null density and
  never masks an already-null cell, so asking for 0.2 hides far fewer cells as
  missingness rises, and not even at a steady rate among the cells it could hide:

  | variant | already missing | asked for | hidden, of all cells | hidden, of observed cells |
  |---|---|---|---|---|
  | `credit-g_00nan` | 0.0% | 0.20 | 20.4% | 20.4% |
  | `credit-g_20nan` | 20.0% | 0.20 | 13.7% | 17.1% |
  | `credit-g_40nan` | 40.0% | 0.20 | 8.7% | 14.6% |
  | `credit-g_60nan` | 60.0% | 0.20 | 6.1% | 15.3% |
  | `credit-g_80nan` | 80.0% | 0.20 | 5.0% | 25.1% |

  The 80% row rises again because the at-least-one-cell-per-row rule dominates once few
  cells remain. Nothing here is broken: the mask is fixed per fold and reproducible, which
  is what this ticket promised. But a self-masked score at "rate 0.2" is not comparable
  across the ladder, and it is not the literature's 20% MCAR either. Ticket 06 should log
  the **realised** rate as a metric beside the nominal one so the comparison is honest,
  and the ADR's decision 5 wording should say nominal.
