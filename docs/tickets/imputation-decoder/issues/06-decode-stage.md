# 06. The decode stage

Status: done (this commit) on `feat/imputation-task`
Blocked by: 01, 02, 03, 04, 05
Plan task: 6. ADR 0004 decisions 2, 4, 5, 10, 11. Wayfinder tickets 03, 04, 11, 12.

## Goal

`train_and_evaluate_decoder` in a new `src/training/decoding.py`, structurally mirroring
`finetuning.py`: trains the decoder on re-rolled masks, selects the best validation epoch
on the fixed draw, and scores both cell populations (plus extra rates and the optional
null path) on the test fold, returning a `DecodingOutcome`.

## Seams under test

- `train_and_evaluate_decoder(dataset, fold, pretraining, hyperparameters, device,
  tracker, seed, score_null_path=False) -> DecodingOutcome`.
- `DecodingOutcome(model, result, train_losses, validation_losses, scored_cells)`.
- **Classification guard**: `run_training` with the default task never constructs a
  decoder (monkeypatch `TridentDecoder.__init__` to raise; the run must still succeed).

## Acceptance criteria

- [x] Per-epoch `decode/train_loss`, `decode/val_loss`, `decode/learning_rate`,
      `decode/val_rmse_num_z`, `decode/val_acc_cat` are logged on the tracker.
- [x] `result.metrics` always holds `impute/masked/*`; holds `impute/induced/*` only when a
      sibling is supplied.
- [x] The best-validation-loss checkpoint is restored.
- [x] `scored_cells` has one row per scored cell with row, column, type, population,
      actual, imputed, confidence, error.
- [x] Extra rates add `impute/masked/rate_<pct>/*`; `score_null_path=True` adds
      `impute/induced/null_token/*` and a `null_path_imputed` column.
- [ ] Classification guard -- **deferred to ticket 08**, which adds the runner's task
      dispatch. Until that exists the guard passes vacuously, so it is not a real test yet.
- [x] Vehicle regression passes with no edit.

## Constraints

- Training masks re-roll every epoch at `mask_probability`; validation and test use one
  fixed draw each at `eval_mask_rate` (ticket 05's helper).
- Induced-missing cells are fed as `[MASK]` on the primary path.
- Same `StageScheduler`, same `lr_scheduler`, same AdamW pattern as fine-tuning.
- **Never** `model.apply(initialize_weights)`.

## Comments

- 2026-09-10 (from ticket 05): `EVAL_MASK_RATE` is **nominal**. `preprocess_table` scales
  the probability by each row's null density and never masks a null, so asking for 0.2
  hides 20.4% of `credit-g_00nan` but only 5.0% of `credit-g_80nan`. Log the **realised**
  rate (hidden cells over scored-eligible cells) as a metric beside the nominal one, or a
  self-masked score cannot be compared across the missingness ladder. See ticket 05's
  comments for the full table.

- 2026-09-10, six red-green slices in `tests/unit/test_training_decoding.py`:

  | Slice | Behaviour pinned |
  |---|---|
  | 1 | The decode stage reports a loss for every epoch it trained |
  | 2 | It scores and lists every cell it hid |
  | 3 | Cells the dataset is missing are scored against the complete table |
  | 4 | The best epoch is the one that is kept |
  | 5 | Extra rates are scored as diagnostics beside the one that ranks |
  | 6 | The null-path diagnostic asks the same cells through the other token |

  Slice 2 counts the listed cells against a mask the test draws for itself, so the table
  cannot quietly agree with metrics computed from the same mistake.

  `DecodedCells` gained `categorical_confidence` here, driven by the ledger's needs rather
  than guessed at in ticket 04.

  **First real end-to-end run**, `credit-g_20nan` fold 1, encoder pre-trained for 40
  epochs then decoded for 40 (6.7s + 7.2s on CUDA), scoring 5327 cells:

  | population | cells | `acc_cat` | `rmse_num_z` | `impute_score` |
  |---|---|---|---|---|
  | self-masked (ranks the fold) | 1375 | 0.597 | 0.948 | **0.958** |
  | induced, shown as `[MASK]` | 1976 | 0.600 | 0.950 | **0.965** |
  | induced, shown as `[NULL]` | 1976 | 0.572 | 1.029 | **1.042** |

  Pre-training loss fell 2.48 -> 0.58 and decode loss 2.35 -> 1.87. Both scored
  populations now beat the mean/mode baseline, which ticket 04's un-pretrained decoder did
  not (0.987 there), so the pretrained encoder is doing work.

  **First evidence on ticket 12's pre-registered question.** The `[MASK]` path scores
  0.965 and the `[NULL]` path 1.042, which is worse than naive imputation. That points
  the same way as ticket 03's decision to substitute `[MASK]`. It is one fold on one
  dataset, so it does not yet meet the criterion (a majority of folds on at least two
  datasets, at 20% and 60%); ticket 11 should gather the rest.

  **Ticket 05's finding confirmed in the real pipeline**: a nominal rate of 0.2 realised
  0.171 on this variant, and the stage logs it as `impute/masked/realised_rate`.

  Unit suite 109 green; the vehicle regression passes unedited.
