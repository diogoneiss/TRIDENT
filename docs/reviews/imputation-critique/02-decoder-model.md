# 02 - The decoder model and its heads

_Critique of the imputation work on branch feat/imputation-task. 2026-09-11._

**Scope:** `src/models.py` L79-240 (`DecodedCells`, `TridentDecoder`) read in full, plus L21-76
(`TridentPretrainer`) and L243-313 (`TridentModel`) for comparison;
`tests/unit/test_decoder_model.py` L1-166 in full. Read as supporting context, not critiqued:
`src/embedder.py` (`EncodedTable`, `TabularEmbedder.encode/forward`), `src/utils.py`
(`preprocess_table`, `split_numeric_and_special`), `src/training/pretraining.py`,
`src/training/finetuning.py` L60-150, `src/training/decoding.py`,
`src/training/imputation_metrics.py`, `src/training/data.py` L45-150,
`src/training/types.py` L34-70 (`TaskSpec.loss_keys`), `src/training/summary.py` L155-215,
`opt.py` L80-150, `docs/adr/0004-imputation-decoder-task.md`.

**Method:** `graphify query` to orient, then the source. Everything quantitative below comes
from scripts run against the real code and the real `credit-g_20nan` data under
`uv run --python 3.10`, in the scratchpad
(`.../scratchpad/decoder-critic/{compose,drift,criterion,masksrc}.py`); no repository file was
modified and no `main.py` run was launched. I did **not** run pytest (instructed not to), did
not exercise a GPU path, and could not establish the trial count behind the two shipped
`*.imputation.json` configs, so the λ figures below are cited descriptively only.

The decoder is faithful to ADR 0004 decision 3 and 4 on every point I could check
mechanically: per-column heads, the three non-category tokens excluded from the output space
by lookup rather than assumption, no Xavier re-init, encoder fine-tuned rather than frozen,
and only hidden cells in the loss. The findings below are about what the loss *means* once it
leaves `forward`, not about whether it computes what the ADR asked for.

## Findings

### F-02-1 - Checkpoint selection runs on a λ-weighted composite, so `LAMBDA_NUM` decides which epoch is kept as well as how the gradient is mixed

- **Kind:** methodology
- **Severity:** medium
- **Where:** `src/models.py:200-210`, consumed at `src/training/decoding.py:114` and `:125-127`
- **Evidence:** `forward` returns `CE_mean + lambda_num * MSE_mean`. `train_and_evaluate_decoder`
  uses exactly that scalar as the validation loss (`validation_loss, _ = model(...)`, L114) and
  keeps the epoch that minimises it (L125-127). The fold is then ranked on something else:
  `impute_score` (`imputation_metrics.py:44-57`) weights the two kinds by their *cell-count
  fractions* and normalises each by the mean/mode baseline.

  Measured on `credit-g_20nan`, fold 0 of 2, with the repo's own tuned config
  (`datasets/hiperparams/credit-g/credit-g_20nan.imputation.json`: DIM 128, HIDDEN_DIM 16,
  HEADS 16, LAYERS 2, PROB_MASCARA 0.4, LR_DECODE 2.54e-3, LAMBDA_NUM 2.738), 40 pre-training
  epochs then 80 decode epochs, scoring both criteria every epoch:

  ```
  epoch chosen by VALIDATION LOSS   : 32 -> val_loss 2.9926 val_impute 0.9168 TEST impute_score 0.9604
  epoch chosen by VALIDATION IMPUTE : 50 -> val_loss 3.1596 val_impute 0.8603 TEST impute_score 0.9658
  best achievable TEST impute_score : 0.9088 at epoch 38
  rank correlation loss vs val_impute: 0.605
  ```

  The two criteria disagree by 18 of 80 epochs and rank the epochs at ρ≈0.61. Note that neither
  criterion reached the oracle 0.9088 and the alternative criterion came out marginally *worse*
  on test (0.9658 vs 0.9604) — on this one fold the disagreement costs nothing measurable. The
  finding is structural, not a demonstrated loss.

  A separate run on the same fold and config (25 pre-training epochs, 60 decode epochs, loss-selected
  epoch 25) decomposes the selected checkpoint's validation loss and compares it to the ranking
  weights:

  ```
  AT SELECTED CHECKPOINT: CE=1.0072 (33.6% of val loss)  lam*MSE=1.9877 (66.4%)
  RANKING WEIGHTS:        w_cat=0.644  w_num=0.356       (impute_score=0.9370)
  ```

  i.e. selection weights numerical:categorical roughly 2:1 while the ranking metric weights them
  roughly 1:2 — the weights are close to inverted. λ is what sets that split, and λ is a searched
  hyperparameter (`opt.py:107`, `opt.py:143`, range 0.1-10 log). The two shipped tuned configs for
  the *same* dataset sit at LAMBDA_NUM 2.738 (`_20nan`) and 0.231 (`_40nan`), a 12x swing; I could
  not recover the trial counts behind them, so I cite the swing only as the range the search
  actually explores, not as evidence of instability.
- **Consequence:** λ enters the reported number twice — once through the gradient, which is what
  ADR 0004 decision 4 describes, and once through which epoch survives, which nothing describes.
  For an ordinary `--task imputation` run this makes the checkpoint arbitrary with respect to the
  metric that is reported; on the one fold I measured the arbitrariness happened to be harmless
  (0.9604 vs 0.9658, both far from the 0.9088 oracle), so the cost is unquantified rather than
  demonstrated. For Optuna under ADR 0005 it is sharper: the trial objective is
  `validation/impute/masked/impute_score` computed on the restored best state
  (`decoding.py:204-210`, after `load_state_dict` at L130), so a trial can win by picking a λ whose
  loss curve happens to bottom out at a lucky epoch rather than by training a better imputer. The
  reduced profile is the default for imputation, and λ is one of only five knobs it samples, so
  this is the common path, not a corner.
- **Direction:** the λ-free option is to select on the quantity the fold is ranked on — the branch
  already computes it (`_score_population(model, hidden_validation, clean_validation, "masked")`
  then `score_cells(..., baselines)`), but only once, at the end, and only when
  `score_search_objective` is set; hoisting the baselines above the epoch loop would let it run
  each epoch. On my one fold that swap did not help, so it should be measured over several folds
  and seeds before being adopted. The minimum either way is to say in the ADR that the checkpoint
  criterion is a second, λ-dependent objective distinct from the ranking metric, and to log both
  terms so the gap is visible at all (see F-02-2).

### F-02-2 - The per-kind loss terms are computed and detached specifically for reporting, and both callers throw them away

- **Kind:** design
- **Severity:** medium
- **Where:** `src/models.py:201-209`; dropped at `src/training/decoding.py:104` and `:114`
- **Evidence:** `forward` builds `metrics["cross_entropy"]` and `metrics["mse"]` as detached
  tensors — the detach exists so that reading them costs no synchronisation, the same pattern
  `TridentPretrainer` uses for `mse_embedding` (`models.py:76`). Both decode call sites discard the
  dict: `loss, _ = model(hidden_train[batch], clean_train[batch])` (L104) and
  `validation_loss, _ = model(hidden_validation, clean_validation)` (L114). `grep` finds no other
  caller. The only decode-stage series that reach the tracker are `decode/train_loss`,
  `decode/val_loss` and `decode/learning_rate` (L116-123), all built from the combined scalar.
  Nothing downstream can decompose it either: `TaskSpec.loss_keys` (`types.py:49-56`) is fixed at
  `pretrain/train_loss`, `pretrain/val_loss`, `<stage>/train_loss`, `<stage>/val_loss`, and
  `_summarize_loss_bands` (`summary.py:161-178`) bands exactly those four keys.
- **Consequence:** a decode run is unreadable in exactly the dimension F-02-1 makes load-bearing.
  Two runs with the same `decode/val_loss` can be 90% numerical error and 90% categorical error;
  on the fold measured above the split is 66/34 and nothing in MLflow or in any artifact says so. A
  numerical head that diverges while the categorical head is fine is indistinguishable from the
  reverse, and a λ that silently switched the stage into a pure regression is invisible. The
  precedent is real — `TridentPretrainer` drops its metrics too — but there the loss is one MSE
  term, so the dict is genuinely redundant; here it is the only decomposition of a composite.
- **Direction:** log the two terms the decoder already hands back, as `decode/train_cross_entropy`,
  `decode/train_mse` and their validation counterparts, beside the combined curve. No new
  computation; the dict is already built, already detached, and already returned.

### F-02-3 - Per-cell pooling inside the categorical term: a column that cannot be got wrong still takes its share of the denominator

- **Kind:** design
- **Severity:** low
- **Where:** `src/models.py:167-179` and `:202-205`
- **Evidence:** the categorical term is `sum over columns of CE(reduction="sum")` divided by
  `categorical_cells`, the total hidden categorical cells across all columns. A column's share of
  the gradient is therefore (its hidden cells) x (its mean surprise). Constructed case — one
  constant categorical column (`const`, all `"a"`) beside a two-category one (`bin`), 40 rows,
  `p_base=0.5`:

  ```
  B const head out: 1   bin head out: 2
    col const  cells 20  sum_ce 0.0000   mean_ce 0.0000
    col bin    cells 24  sum_ce 23.3552  mean_ce 0.9731
  B pooled cat mean 0.5308     <- the reported/optimised categorical term
  ```

  A head with one output class has `log_softmax == 0` identically, so `const` contributes exactly
  zero loss and zero gradient, yet its 20 cells halve the categorical term against the numerical
  one. On real `credit-g_20nan` the graded version of the same effect is visible — mean CE per
  column at the selected checkpoint ranges from `foreign_worker` 0.0321 (K=2, heavily skewed) to
  `purpose` 1.7229 (K=10), a 54x spread. Most of that spread is genuine difficulty, which is what
  cross-entropy should weight, so it is not itself an artifact; the constant-column case is.
- **Consequence:** the protection ADR 0004 decision 4 claims — "each first averaged over its own
  count of scored cells" so that "a table dominated by one kind cannot drown the other's term" —
  holds across kinds but does not hold across columns within a kind. On a table with several
  constant or near-constant categorical columns the categorical term is divided by a denominator
  it did not earn, and the effective λ is larger than the configured one by that factor. I found no
  shipped dataset where this bites: cardinalities are `credit-g` 2-10, `kr-vs-kp` 2-3,
  `electricity` 7, with no constant and no all-missing column at `_00nan`, `_20nan` or `_80nan`.
  So this is a property to know about before the next dataset arrives, not a defect in the
  published numbers.
- **Direction:** either count only columns with more than one real category in `categorical_cells`,
  or skip constructing a head for such a column at all (it has nothing to learn) and record it as
  trivially imputed. Either way, say in the ADR that the within-kind weighting is per cell.

### F-02-4 - A categorical column with no real categories builds a zero-width head and kills `predict` after the whole stage has run

- **Kind:** bug
- **Severity:** low
- **Where:** `src/models.py:120-135` and `:220-227`
- **Evidence:** when every value of a categorical column is missing, its vocabulary is exactly the
  three excluded tokens, so `valid_ids` is empty and the head is `nn.Linear(d, 0)`:

  ```
  A vocab c: ['[MASK]', '[NULL]', 'nan']
  A head c out_features: 0
  A local_of_c: [-1, -1, -1]   valid_ids_c: []
  A predict RAISED: IndexError argmax(): Expected reduction dim 1 to have non-zero size.
  ```

  (`torch` emits one `UserWarning: Initializing zero-element tensors is a no-op` at construction —
  the only signal, and it lands inside the tqdm output.) Note the ordering: this is the *second*
  crash site on that trigger. `mean_mode_baselines` runs first at `decoding.py:134` and already
  raises `IndexError` on `mode().iloc[0]` for an all-missing column — that is prior finding #3,
  which shadows this one. This one only becomes reachable once #3 is fixed.

  Related and unasserted: `local_of` maps the three excluded ids to `-1`, and
  `cross_entropy` at `models.py:176` has no `ignore_index`. The comment at `models.py:130-131`
  states the invariant that keeps this safe — "which can never be a target because masking never
  hides a null" — and it does hold on every path I traced (`preprocess_table:62`
  `dynamic_mask[null_values] = False`; `evaluation_mask` goes through the same function;
  `_select` at `decoding.py:268` overrides `masked_positions` but only ever feeds `predict`). It
  is a load-bearing invariant defended by a comment and nothing else.
- **Consequence:** a user who generates a higher-missingness variant, or brings a table with an
  empty categorical column, loses the fold after pre-training and the entire decode stage have
  already run. With the prior finding #3 in front of it the loss today is bounded; both need
  fixing together or the fix to #3 just moves the crash 11 lines later.
- **Direction:** decide what a column with no real categories means for imputation — most likely
  exclude it from the decoder's head set and from the scored populations, at the same place #3 is
  fixed. Independently, give `cross_entropy` an `ignore_index=-1` so the invariant is enforced by
  the code rather than asserted by a comment.

### F-02-5 - Nothing in the suite asserts that decode gradients reach the embedder and the transformer

- **Kind:** design
- **Severity:** low
- **Where:** `tests/unit/test_decoder_model.py:143-166`; the behaviour under test is
  `src/training/decoding.py:59-63`
- **Evidence:** ADR 0004 decision 4 states "The encoder is fine-tuned, not frozen", and
  `optim.AdamW(model.parameters(), ...)` delivers that, since `TridentDecoder` holds the embedder
  and transformer as submodules. The closest test,
  `test_attaching_the_decoder_leaves_the_pretrained_encoder_untouched`, checks only that
  *constructing* a decoder does not disturb the weights — it never runs a backward pass. `grep` for
  `grad|backward|requires_grad|freeze` across `tests/unit/test_decoder_model.py` and
  `tests/unit/test_training_decoding.py` finds only three `torch.no_grad()` context managers, all
  in test setup. I verified separately that the encoder does move: relative L2 drift of all
  embedder + transformer parameters across a 60-epoch decode stage on `credit-g_20nan` at the
  tuned LR_DECODE is 0.146.
- **Consequence:** the ADR names "freeze the encoder" style ablations as future work, and the
  obvious implementation — narrowing the optimizer to
  `list(decoder.categorical_heads.parameters()) + list(decoder.numerical_heads.parameters())` —
  would pass every test in the repository while silently converting every subsequent imputation
  run to linear-probe numbers. Because the decode stage shares the embedder and transformer
  *objects* with the pre-trainer, the same mistake in reverse (an ablation branch that leaks into
  the default path) is equally invisible.
- **Direction:** one test: run a single decode step and assert a named embedder parameter and a
  named transformer parameter have non-`None`, non-zero `.grad`. That is the ADR sentence, made
  mechanical.

## Checked and cleared

- **Which view supplies the mask.** `forward` reads `hidden.masked_positions`, not
  `targets.masked_positions`. I suspected
  `test_only_hidden_cells_teach_the_decoder` could not tell the two apart, and it can: monkey-
  patching `forward` to read the mask off `targets` gives `with_hidden 0.0` where the shipped code
  gives `11.7413`, so the test's `assert with_hidden_cells.item() > 0.0` fails under that mutant.
  The test is stronger than its two assertions look.
- **Unmasked cells carry no loss.** Only `mask[:, index]`-selected rows enter either term, and a
  frame with nothing hidden short-circuits at `models.py:195-196` to a `requires_grad=True` zero.
  `optimizer.zero_grad()` defaults to `set_to_none=True`, so the subsequent `backward()`/`step()`
  on such a batch is genuinely a no-op rather than a zero-gradient update.
- **The pretrained encoder is fine-tuned, not frozen or re-initialised.** Pre-training's
  `model.apply(initialize_weights)` Xavier sweep is at `pretraining.py:49`, *before* the training
  loop, so it is not a post-hoc erasure; `TridentDecoder.__init__` touches no existing parameter
  (the shipped test proves it, and it passes as written); the decode optimizer takes
  `model.parameters()`, which includes both shared submodules; measured relative L2 drift over a
  60-epoch decode stage is 0.146. This is
  byte-for-byte the same optimizer construction classifier fine-tuning uses
  (`finetuning.py:106-110`, `AdamW(model.parameters(), lr=finetuning_learning_rate, ...)`), so
  "decode-stage training mirrors fine-tuning" is accurate, including the absence of parameter
  groups.
- **Targets never backpropagate.** Unlike `TridentPretrainer`, which must run the embedder on the
  clean view under `no_grad`, the decoder only reads `targets.cat_indices` and `targets.num_values`
  as label tensors. There is no second embedder forward pass and no graph to detach.
- **The excluded-token machinery is right on real vocabularies.** On `credit-g_20nan` every one of
  the 13 categorical columns loses exactly three ids: e.g. `purpose` vocab 13 → head 10,
  `own_telephone` vocab 5 → head 2. `valid_ids` is looked up through `encoder.classes_` rather than
  assumed positional, which matters because `LabelEncoder` sorts, so `"[MASK]"`, `"[NULL]"` and
  `"nan"` land at different indices per column.
- **`LAMBDA_NUM` really is inert on a single-type table.** `opt.py:105-107` and `:141-143` gate the
  suggestion on `mixed_columns`. With one term the loss is `λ * MSE_mean`; a positive scale does not
  move an argmin, and AdamW's adaptive step is scale-invariant with decoupled (therefore
  loss-scale-independent) weight decay, so the trial would be spending a dimension on nothing. The
  gate is correct.
- **Head parameter growth is not a scaling problem.** On `credit-g_20nan` at DIM 128 / HIDDEN_DIM 16:
  categorical heads 6,966 params, numerical heads 14,567, embedder + transformer 181,024. The
  per-column numerical head mirrors the per-column embedder MLP the table already pays for, so the
  decoder adds roughly one embedder's worth of head parameters, not a term that outgrows the model
  as columns are added.

## Open questions

- **Does the numerical head's `HIDDEN_DIM` bottleneck cost anything?** The head is
  `Linear(d, hidden_dim) → ReLU → Linear(hidden_dim, 1)` — at the shipped credit-g config that is a
  128→16→1 compression, while the categorical head is a bare `Linear(128, K)`. `HIDDEN_DIM` was
  tuned for the embedder, where it means "how wide to expand one scalar", and the reduced Optuna
  profile (`opt.py:95-107`) does not sample it, so the decoder inherits it. Settling this needs a
  sweep of the numerical head's width at a fixed encoder — the mirroring in ADR 0004 decision 3 is
  an aesthetic argument, not a measured one.
- **How much does pre-training actually contribute to the decoded numbers?** A one-seed, one-fold
  probe with a deliberately truncated 40-epoch pre-train (configured: 300) scored
  `impute_score` 0.9604 with pre-training and 0.9435 without it — i.e. the no-pre-training arm
  came out slightly ahead. That is far too weak to conclude anything, and a 40-epoch pre-train is
  not the shipped recipe, but a proper ablation (full EPOCHS_PRE, both CV folds, a few seeds) is
  cheap and would tell you whether the decode stage is reading a pretrained representation or
  mostly training one of its own from a warm start.
- **Is LR_DECODE's ceiling safe for the shared encoder?** `LR_DECODE` ranges 1e-4..1e-2 while
  `LR_PRE` tops out at 1e-3, and both shipped configs put the whole encoder through the decode
  optimizer at 7.5x (`_20nan`, 2.54e-3) and 25.6x (`_40nan`, 8.71e-3) their own `LR_PRE` of 3.4e-4.
  Since there are no parameter groups, head LR and encoder LR are the same number by construction.
  I measured drift 0.146 at the 7.5x config but have no baseline to compare it against; the
  discriminating measurement is the same drift figure for classification fine-tuning at `LR_FINE`,
  which would say whether the decode stage is unusually destructive or just normal for this
  codebase.
