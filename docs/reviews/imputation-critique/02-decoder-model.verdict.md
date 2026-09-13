# 02 - The decoder model and its heads: verdicts

_Adversarial verification of `02-decoder-model.md`. 2026-09-11._

**Method:** read in full `src/models.py`, `src/training/decoding.py`,
`src/training/imputation_metrics.py`, `tests/unit/test_decoder_model.py`,
`tests/integration/test_credit_g_imputation_regression.py` and
`tests/fixtures/credit-g_20nan_imputation_regression.json`; read the relevant spans of
`src/training/types.py` (`TaskSpec.loss_keys`), `src/training/summary.py`
(`_summarize_loss_bands`, `_loss_events_by_key`), `src/training/finetuning.py` L100-185,
`src/training/data.py` (`prepare_dataset`, `evaluation_mask`), `src/utils.py`
(`preprocess_table`), `opt.py` L80-160 and L400-470, and ADR 0004 decision 4. Oriented with
`graphify query "decoder loss checkpoint selection impute_score lambda_num"`. Read-only
throughout; no repository file outside this one was touched, and no `main.py` run was
launched (my one allowance went unused — the experiments below are cheaper and more
targeted).

Three scripts under
`.../scratchpad/verify02/`, all run as
`PYTHONPATH=<repo> uv run --python 3.10 python <script>` from the repo root.

1. `f4_zero_head.py` — an all-missing categorical column `c` beside a normal one and a
   numerical one, through the real `TabularEmbedder` / `TridentDecoder`:

   ```
   WARN: UserWarning Initializing zero-element tensors is a no-op
   vocab c: ['[MASK]', '[NULL]', 'nan']
   head c out_features: 0
   local_of_c: [-1, -1, -1]
   valid_ids_c: []
   predict RAISED: IndexError argmax(): Expected reduction dim 1 to have non-zero size.
   mean_mode_baselines RAISED: IndexError single positional indexer is out-of-bounds
   forward OK loss: 8.873061180114746 metrics: {'cross_entropy': 0.8374570608139038, 'mse': 8.035604476928711}
   ```

2. `f3_constant_col.py` — a constant categorical column beside a two-category one, plus the
   same constant column pushed through `score_cells`:

   ```
   const head out: 1  bin head out: 2
     col const  cells  20  sum_ce   0.0000  mean_ce 0.0000
     col bin    cells  24  sum_ce  23.3552  mean_ce 0.9731
   pooled cat mean 0.5308
   include_const= False  acc_cat=0.5000  impute_score=1.0000  n_cat=24
   include_const= True   acc_cat=0.7273  impute_score=1.0000  n_cat=44
   ```

3. Two one-liners against `torch`:

   ```
   CE(-1) RAISED: IndexError Target -1 is out of bounds.
   CE over 1 class: 0.0
   ```

I did **not** reproduce the critic's 40+80-epoch dual-criterion run on `credit-g_20nan`
(ρ=0.605, epochs 32 vs 50). Every premise that run rests on is verified in source below;
the numbers themselves are cited as the critic's, unreplicated.

## Verdicts

### F-02-1 - The checkpoint is chosen by minimising the same λ-weighted composite the decoder trains on, so `LAMBDA_NUM` also decides which epoch survives, and that criterion is only loosely aligned with the `impute_score` that ranks the fold

- **Verdict:** SOUND
- **Severity after review:** low (critic said medium)
- **Basis:** every premise holds in source.
  - `src/models.py:200-210` returns `loss = categorical_mean + self.lambda_num * numerical_mean`.
  - `src/training/decoding.py:114` `validation_loss, _ = model(hidden_validation, clean_validation)`
    — that exact scalar — and `:125-127` keeps its argmin:
    `if validation_losses[-1] < best_validation_loss: ... best_state = {...}`.
  - `src/training/imputation_metrics.py:44-57` scores the fold on something else:
    `score += (len(numerical) / total) * _ratio(metrics["rmse_num_z"], naive_rmse)` plus the
    categorical counterpart — cell-count weights over baseline-normalised ratios, with no λ
    anywhere.
  - The Optuna objective really is read off the restored state: `load_state_dict(best_state)`
    at `decoding.py:130`, `_score_population(model, hidden_validation, clean_validation, "masked")`
    at `:204-206`, `validation/impute/masked/impute_score` is `IMPUTATION.search_objective`
    (`types.py:37-43`).
  - λ is sampled `0.1..10` log in **both** profiles when the table is mixed (`opt.py:107` reduced,
    `opt.py:143` full), and `reduced` is the imputation default (`opt.py:408-410`:
    `"reduced" if task.name == "imputation" else "full"`). The shipped configs confirm the
    swing the critic cites: `credit-g_20nan` LAMBDA_NUM 2.738, `credit-g_40nan` 0.231;
    `kr-vs-kp` (all-categorical) sits at the untouched default 1.0, as the `mixed_columns`
    gate predicts.

  So the structural claim — a searched hyperparameter silently reparameterises the
  checkpoint-selection rule, and the selection rule is not the ranking rule — is true and
  worth a sentence in the ADR.
- **Correction:** three things the critic got wrong or overstated, which is why severity drops
  to low.
  1. **"which nothing describes" is false.** ADR 0004 decision 4 states, in the same
     paragraph as the loss definition: *"The best validation-loss epoch is restored."* The
     mechanism is documented. What is undocumented is only that this criterion is λ-dependent
     and is a different objective from the ranking metric.
  2. **The 66.4% / 33.6% vs w_cat 0.644 / w_num 0.356 "close to inverted" comparison is
     rhetoric, not evidence.** Those are different objects in different units: the loss shares
     are magnitudes (nats vs λ·z²) of a static decomposition, while the impute_score weights
     are cell-count fractions applied to baseline-normalised *ratios*. An argmin over epochs
     depends on how each term *moves*, not on its static share, and two terms with a 2:1
     magnitude split can have any relative influence on the argmin. Drop that paragraph; the
     finding does not need it and is weakened by it.
  3. **The select-on-val-loss / rank-on-something-else structure is pre-existing, not
     introduced here.** `finetuning.py:174-178` does exactly the same thing: argmin of the
     validation cross-entropy, while `CLASSIFICATION.ranking_metric` is `f1_macro`. Only the λ
     channel is new to the decode stage.

  On severity: in the critic's own numbers the two criteria land 0.005 apart on test (0.9604
  vs 0.9658) while *both* sit ~0.05 from the oracle 0.9088. The criterion question is an order
  of magnitude smaller than epoch-to-epoch test noise on that fold, and the critic measured it
  on one fold of one dataset. Structural, documented-in-part, unquantified harm: low.
- **What the critic missed here:** λ has a *second* undocumented channel. `decoding.py:124`
  `scheduler.after_epoch(validation_losses[-1])` feeds the same composite to `StageScheduler`,
  and under `--lr_scheduler plateau` (`schedulers.py:103-104`,
  `self._plateau_scheduler.step(validation_loss)`) λ therefore also decides *when the learning
  rate halves*. On `cosine`/`cosine_legacy`/`warmup_cosine` that argument is ignored, so this
  is schedule-conditional — but it is a real third place λ enters a run.

### F-02-2 - The per-kind loss terms are built and detached for reporting and every caller discards them, so no run can decompose the decode loss curve

- **Verdict:** SOUND
- **Severity after review:** low (critic said medium)
- **Basis:** the premise is exact.
  - `models.py:205` `metrics["cross_entropy"] = categorical_mean.detach()` and `:209`
    `metrics["mse"] = numerical_mean.detach()`.
  - Both decode call sites discard the dict: `decoding.py:104`
    `loss, _ = model(hidden_train[batch], clean_train[batch])` and `:114`
    `validation_loss, _ = model(hidden_validation, clean_validation)`.
  - `grep -rn "cross_entropy|\"mse\"|mse_embedding" src/ tests/` returns exactly four lines,
    all inside `src/models.py`. **Nothing in the repository ever reads either key.**
  - The only decode series logged per epoch are `decode/train_loss`, `decode/val_loss`,
    `decode/learning_rate` (`decoding.py:116-123`), all from the combined scalar, and
    `TaskSpec.loss_keys` (`types.py:49-56`) bands only the four `*_loss` keys.
- **Correction:** the consequence is overstated in one specific way, and that is what cuts the
  severity. *"A numerical head that diverges while the categorical head is fine is
  indistinguishable from the reverse"* is false **at the selected checkpoint**:
  `score_cells` emits `rmse_num_z`, `mae_num_z`, `acc_cat`, `macro_f1_cat`, `n_num_cells`,
  `n_cat_cells`, all logged as `test/impute/masked/*` (`decoding.py:151, 213`), written per
  column by `runner._per_column_scores`, and three of them are pinned in the integration
  fixture at `abs=0.01`. The per-kind decomposition of *final performance* exists and is
  tracked. What is genuinely missing is only the per-kind decomposition of the **epoch-wise
  loss curve** — a real but much narrower gap than "a decode run is unreadable".
- **What the critic missed here, and it is the best argument for the fix:** the classification
  stage already logs a per-epoch proxy for its ranking metric beside the loss —
  `finetuning.py:163-168` emits `finetune/val_f1_macro` and `finetune/val_f1_micro` every
  epoch. The decode stage logs no analogous per-epoch quantity at all. That asymmetry is a
  repo-internal precedent for the Direction, and it also makes the gap in F-02-1 visible.
  Second: the Direction is safe to implement as written — `_loss_events_by_key`
  (`summary.py:181-200`) skips any event whose key is not in `loss_keys`
  (`if event.key not in events_by_key: continue`), so adding `decode/train_cross_entropy` and
  friends cannot break `_summarize_loss_bands`. Third, trivially: there is a *third* caller of
  `TridentDecoder.forward`, `tests/unit/test_training_decoding.py:211`
  (`kept_model_scores, _ = outcome.model(hidden, clean)`), which drops the dict too; "grep
  finds no other caller" is true of `src/` only.

### F-02-3 - The categorical term pools per cell across columns, so a column that cannot be got wrong contributes zero loss while still inflating the denominator

- **Verdict:** SOUND
- **Severity after review:** low (unchanged)
- **Basis:** the premise is correct, and my independent reconstruction reproduces the critic's
  numbers to four decimals. `models.py:176-179` accumulates
  `cross_entropy(logits, expected, reduction="sum")` and `categorical_cells += int(selected.sum())`
  for **every** column with a hidden cell, then `:203` divides the total by that total count.
  A column with one real category gets `nn.Linear(d, 1)` and `log_softmax` over a size-1
  dimension is identically zero — verified directly: `CE over 1 class: 0.0`. My run:
  `const` 20 cells / `sum_ce 0.0000`, `bin` 24 cells / `mean_ce 0.9731`, pooled categorical
  term `0.5308` — the informative column's gradient is scaled by 24/44.
- **Correction:** none to the premise; the critic's framing understates the point, and I would
  sharpen rather than cut it. The ranking metric's categorical *ratio* is **invariant** to the
  constant column, because `_ratio(1.0 - acc_cat, naive_error)` divides two quantities that
  both dilute by the same total cell count — measured: `impute_score` is `1.0000` with and
  without the 20 constant cells, while `acc_cat` moves 0.5000 → 0.7273. But the kind's
  *weight*, `len(categorical) / total` (`imputation_metrics.py:52-56`), **rises** with those
  cells. So a constant categorical column makes the ranking metric care *more* about
  categorical error while making the loss care *less* about it. The two move in opposite
  directions, which is a stronger statement than "the effective λ is larger than configured".

  Severity stays low for the reason the critic already gives and I did not disturb: no shipped
  variant has a constant or all-missing categorical column, so this is latent.

### F-02-4 - An all-missing categorical column builds `nn.Linear(d, 0)` and crashes `predict` after the whole decode stage has run; `local_of` maps excluded ids to `-1` with no `ignore_index` guard

- **Verdict:** CONFIRMED
- **Severity after review:** low (unchanged)
- **Basis:** reproduced end to end against the real classes (script 1 above). Vocabulary is
  exactly `['[MASK]', '[NULL]', 'nan']`; `valid_ids` is `[]`; the head's `out_features` is `0`;
  `local_of` is `[-1, -1, -1]`; the only signal at construction is
  `UserWarning: Initializing zero-element tensors is a no-op`; and `predict` raises
  `IndexError: argmax(): Expected reduction dim 1 to have non-zero size` at `models.py:223`.
  The shadowing the critic describes is also confirmed in the same run:
  `mean_mode_baselines` raises `IndexError: single positional indexer is out-of-bounds` on
  `mode().iloc[0]` (`imputation_metrics.py:136`), and it runs first (`decoding.py:134`, before
  `_score_population` at `:145`). Also confirmed: `prepare_dataset` (`data.py:45-85`) has no
  all-null or `dropna(axis=1)` guard, so nothing upstream removes such a column — the crash is
  genuinely reachable from `main.py` once the prior finding is fixed.

  The `ignore_index` half is correct too: `models.py:176` passes no `ignore_index`, and a `-1`
  target raises `IndexError: Target -1 is out of bounds.` The invariant that keeps it safe is
  `preprocess_table:62` `dynamic_mask[null_values] = False`, and `evaluation_mask`
  (`data.py:143-148`) goes through the same function. My run is direct evidence the invariant
  holds on the training path: with the all-missing column present, `forward` did **not** crash
  (`forward OK loss: 8.873...`) precisely because that column is never selected.
- **Correction:** none. Severity stays low: it needs a table the repository does not ship, and
  it is shadowed by an earlier crash on the same trigger. The write-up's own framing ("both
  need fixing together or the fix to #3 just moves the crash 11 lines later") is the right one.

### F-02-5 - No test asserts that decode-stage gradients reach the embedder and transformer, so the ADR's "the encoder is fine-tuned, not frozen" is unguarded

- **Verdict:** SOUND
- **Severity after review:** low (unchanged)
- **Basis:** the premise holds. ADR 0004 decision 4 line 71: *"The encoder is fine-tuned, not
  frozen."* `decoding.py:59-63` delivers it through `optim.AdamW(model.parameters(), ...)`,
  relying entirely on `TridentDecoder` holding `self.embedder` and `self.transformer` as
  submodules (`models.py:113-114`). `test_attaching_the_decoder_leaves_the_pretrained_encoder_untouched`
  (`test_decoder_model.py:143-166`) ends at `TridentDecoder(embedder, transformer)` and then
  asserts `torch.equal(current, original)` — it never calls `backward()`. Reading both test
  files: no `.grad` assertion, no `requires_grad` assertion, no `backward()` anywhere; the only
  gradient-adjacent constructs are `torch.no_grad()` blocks used to zero heads in setup
  (`test_decoder_model.py:96, 112, 134`).
- **Correction:** one overstatement, which does not change the verdict but should be fixed in
  the text. *"would pass every test in the repository"* is not true as written. It is true of
  every **unit** test, which is what the default `pytest -m "not integration"` runs. The one
  guard that could catch a head-only optimizer is
  `tests/integration/test_credit_g_imputation_regression.py`, which pins
  `impute/masked/{impute_score,rmse_num_z,acc_cat}` and the induced counterparts at
  `abs=0.01`. Whether freezing the encoder moves those past 0.01 in the fixture's tiny
  2-pre-train / 2-decode-epoch, DIM-16, LR_DECODE 1e-3 configuration is **unmeasured** — I
  deliberately did not spend a training run on it, and the critic should not claim it either
  way. The honest statement is: the invariant is unguarded in the default suite, and the
  opt-in regression fixture is a coincidental and unverified backstop. The proposed one-line
  gradient test is the right fix and is cheap.

## What this critique missed

Five things, ordered by how much they would change the write-up.

1. **λ's third channel: the learning-rate schedule.** `decoding.py:124`
   `scheduler.after_epoch(validation_losses[-1])` hands the λ-weighted composite to
   `StageScheduler`, and under `--lr_scheduler plateau` that value is what
   `ReduceLROnPlateau.step()` consumes (`schedulers.py:100-104`). So under that schedule λ
   decides the gradient mix, the kept epoch, *and* when the LR halves. F-02-1 claims two
   channels; there are three.

2. **The ADR does document the checkpoint rule.** Decision 4 says "The best validation-loss
   epoch is restored." F-02-1's "which nothing describes" should become "which the ADR
   describes without noting that it is λ-dependent and is not the ranking metric" — and the
   same structure already exists in classification (`finetuning.py:174-178`), so the λ
   dependence is the only novel part.

3. **Classification already logs a per-epoch ranking proxy; decode does not.**
   `finetune/val_f1_macro` and `finetune/val_f1_micro` are logged every epoch
   (`finetuning.py:163-168`). The decode stage logs no per-epoch impute proxy at all. This is
   the strongest single argument for both F-02-1's Direction (score the ranking metric each
   epoch) and F-02-2's (log the two loss terms), and the critique does not mention it.

4. **The constant-column effect is an inversion, not a dilution** (see F-02-3): the ranking
   metric's categorical ratio is invariant to such a column while its cell-count weight rises,
   so the loss and the metric move in opposite directions. Measured, not argued.

5. **`_loss_events_by_key` already tolerates extra keys** (`summary.py:186-187`,
   `if event.key not in events_by_key: continue`), so F-02-2's Direction has no summariser
   blast radius. The critique presents `TaskSpec.loss_keys` as a constraint without checking
   that the consumer ignores unknown keys.

One area the critique's own "Open questions" flags but nobody has measured, which I agree is
the highest-value next experiment and is **not** a finding: `EPOCHS_DECODE` is 150 in all four
shipped imputation configs, and the reduced Optuna profile does not sample it, so it is held
at the `Hyperparameters.decode_epochs = 150` default (`types.py:112`) in every reduced trial
(the full profile samples 20-60, `opt.py:134`). A checkpoint chosen out of 150 epochs on a
criterion that F-02-1 shows is only ρ≈0.6-aligned with the ranking metric has a lot of room. The longer the budget, the more room the
selection rule has to matter. That is the setting in which F-02-1 would stop being structural
and start being measurable.
