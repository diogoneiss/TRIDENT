# 13. Optuna for the imputation task

Type: grilling
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-10
Blocked by: 08

## Question

Ticket 05 fixed the objective and its direction: Optuna should optimise the task's
ranking metric, `impute_score`, with `direction="minimize"` instead of the hardcoded
`f1_macro` / `maximize` in `opt.py`. What remains:

- **Search space.** Do the decode-stage hyperparameters (decode epochs, decode learning
  rate and weight decay, `lambda_num`) join `define_search_space`, and does the
  evaluation mask rate stay fixed so trials remain comparable? Tuning the metric's own
  mask rate would let a trial win by making the problem easier.
- **Shared parameters.** The architecture parameters (`DIM`, `HEADS`, ...) are shared
  with the classification task. One search space with a task switch, or two?
- **Fine-tuning parameters.** A decode run never fine-tunes a classifier, so `LR_FINE`
  and `EPOCH_FINE` are dead parameters in an imputation study; sampling them wastes
  trials and pollutes the recorded parameter set.
- **Known trap** (backlog B3): `HEADS` must divide `DIM`, and invalid combinations are
  currently scored `0.0` by a bare `except`. Under a minimised objective, a crash would
  score `0.0`, which is the *best* possible value rather than the worst. The failure
  path in `ObjectiveFunctionWrapper.__call__` has to change with the direction.

Recommended starting answer: one search space with a task switch, decode parameters
sampled only for the imputation task and fine-tuning parameters only for classification,
the evaluation mask rate held fixed, and the failure path returning a direction-aware
worst value (or raising `TrialPruned`).

## Answer

Resolved 2026-09-10 in one grilling round; every recommendation was accepted.

**Two hazards found in `opt.py` while resolving, beyond what the ticket listed:**

1. **The failure path inverts under a minimised objective.** `ObjectiveFunctionWrapper.__call__`
   ends `return 0.0  # Return worst possible score on failure` (`opt.py:186`). That comment
   is true while maximising `f1_macro` and **false** while minimising `impute_score`, where
   `0.0` is the *best* achievable value. A crashing trial would be recorded as a perfect
   imputer and the search would then hunt for configurations that crash -- the same shape as
   backlog item B4, which made every trial score `0.0`.
2. **`save_best_params` reaches across tasks.** It writes
   `datasets/hiperparams/<base>/<dataset>.json`, a path keyed by dataset with no notion of
   task, and both tasks read that same file. An imputation study on `credit-g_20nan` would
   therefore overwrite the **classification** configuration for that dataset, silently
   changing every later classification run. This is backlog item I2 with a cross-task edge
   the backlog does not mention.

**Decisions:**

1. **What an imputation study samples.** The shared architecture and pre-training block
   stays (`DIM`, `HIDDEN_DIM`, `HEADS`, `LAYERS`, `DIM_FEED`, `DROPOUT`, `EPOCHS_PRE`,
   `BATCH`, `LR_PRE`, `WEIGHT_DECAY_PRE`, `PROB_MASCARA`). The three fine-tuning
   parameters, dead without a classifier, are swapped for the decode ones, and the loss
   balance joins as a genuine tuning knob:

   | Classification study | Imputation study |
   |---|---|
   | `EPOCH_FINE` | `EPOCHS_DECODE` |
   | `LR_FINE` | `LR_DECODE` |
   | `WEIGHT_DECAY_FINE` | `WEIGHT_DECAY_DECODE` |
   | -- | `LAMBDA_NUM` |

   **`EVAL_MASK_RATE` and `EVAL_MASK_RATES_EXTRA` are never sampled**, per ticket 11: a
   trial could otherwise win by making its own evaluation easier. `PROB_MASCARA` stays
   sampled, since ticket 04 confirmed it governs *training* corruption only and never
   touches scoring. Its existing range starts at 0.2, which stays clear of the worst of
   backlog P1's low-rate performance cliff.

2. **One `define_search_space` with a task parameter**, not two functions. Eleven of the
   fourteen parameters are shared, so splitting would duplicate the majority to vary the
   minority, and the two copies would drift.

3. **The failure path never returns a score.** It raises `optuna.TrialPruned`, so the trial
   is discarded rather than ranked, which is correct under either direction. The
   `trial_status` and `error` tags stay, so failures remain visible in the store.
   Alongside it, **`HEADS` is constrained to divisors of `DIM`**, removing the invalid
   combinations of backlog B3 that currently waste roughly a fifth of the reachable grid
   and are scored as legitimately terrible hyperparameters.

   Recorded scope note: constraining the space changes which configurations a
   *classification* study samples. That is acceptable here because no Optuna run exists in
   the tracking store at all (verified in ticket 06), so no published result depends on the
   current sampling sequence, and the training fixture that `AGENTS.md` protects is not an
   Optuna artifact.

4. **Best-trial tracking takes its comparison and starting value from the task's
   direction**, the same `(ranking_metric, direction)` pair ticket 09 threads through the
   summariser. Today `if f1_macro > self.best_score` (`opt.py:171`) is a hardcoded
   greater-than; a minimised objective would never improve on its initial best and would
   therefore save nothing.

5. **An imputation study writes only into its own timestamped study directory**, never into
   `datasets/hiperparams/<base>/<dataset>.json`. That removes the cross-task clobber
   entirely without changing classification's current behaviour, which stays backlog item
   I2's business. Promoting a tuned imputation configuration becomes a deliberate copy, and
   the ADR says so.

Kept unchanged: the objective is logged under `optuna/objective_value` whatever the task's
metric, so one key sorts every trial; and per ticket 06 the study and trial runs carry
`task = imputation` alongside `is_optuna = true`.

## Comments

- 2026-09-10 (ticket 06): runs from an imputation study will carry `task = imputation`
  and the new dense `is_optuna = true` tag, so trials are filterable on both axes and
  need not be excluded by hand from comparison tables.
