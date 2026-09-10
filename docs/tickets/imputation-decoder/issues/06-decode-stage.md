# 06. The decode stage

Status: ready-for-agent
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

- [ ] Per-epoch `decode/train_loss`, `decode/val_loss`, `decode/learning_rate`,
      `decode/val_rmse_num_z`, `decode/val_acc_cat` are logged on the tracker.
- [ ] `result.metrics` always holds `impute/masked/*`; holds `impute/induced/*` only when a
      sibling is supplied.
- [ ] The best-validation-loss checkpoint is restored.
- [ ] `scored_cells` has one row per scored cell with row, column, type, population,
      actual, imputed, confidence, error.
- [ ] Extra rates add `impute/masked/rate_<pct>/*`; `score_null_path=True` adds
      `impute/induced/null_token/*` and a `null_path_imputed` column.
- [ ] Classification guard passes.
- [ ] Vehicle regression passes with no edit.

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
