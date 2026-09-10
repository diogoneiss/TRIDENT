# 05. Data support for evaluation

Status: ready-for-agent
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

- [ ] Scaler present after `prepare_dataset`; existing data tests untouched and green.
- [ ] Sibling found for a `_20nan` fixture frame; `None` for `_00nan`.
- [ ] Alignment assertion raises on a deliberately altered observed cell.
- [ ] Evaluation mask identical across two calls with the same `(seed, fold)`, different
      across folds.
- [ ] Global `np.random` state is identical before and after the evaluation draw.

## Constraints

- `preprocess_table` uses global `np.random`; wrap the evaluation draw in a
  `get_state` / `set_state` guard around a local reseed so it consumes no global draws.
  This is what keeps the classification draw sequence untouched.
- The sibling is read for scoring only; nothing in training may call it.

## Comments
