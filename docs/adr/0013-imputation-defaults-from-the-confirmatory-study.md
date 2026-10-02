# Give the imputation task the configuration the confirmatory study chose

## Status

Accepted (2026-10-02). The user decided this after the confirmatory study
[imputation-pretraining/07](../tickets/imputation-pretraining/07-confirmatory-21-variants.md)
(E30), which settled task T00. It is a deliberate exception to the rule that old behaviour stays
the default, as ADRs 0006 and 0012 were. It amends the defaults stated in ADRs
[0003](0003-selectable-learning-rate-schedule.md), [0004](0004-imputation-decoder-task.md),
[0005](0005-reduced-optuna-search-for-imputation.md) and
[0011](0011-selectable-pretraining-objective.md), for the imputation task only.

## Context

E30 compared three arms on all 21 imputation variants, three seeds each:

- **A**, the old defaults: the `embedding` pre-training objective and a 150-epoch decode stage.
- **M**: the `embedding_normalized` objective and a 450-epoch decode stage.
- **P**: M with a decode patience of 50.

On induced cells, M was better than A on 9 variants and worse on none. P was worse than A on 2.
By the reading fixed before launch, M is the only candidate that qualifies. M's mean score over
the 21 variants was 0.7745, against 0.7929 for A, and 0.7781 for the best baseline. A cell of M
costs 1.65x A's time.

Every E30 cell ran under the per-epoch `cosine` schedule. The global default is still
`cosine_legacy`, the per-batch sawtooth of ADR 0003. No imputation run in the store has ever used
it: 406 of 424 finished imputation parents ran `cosine`, and 18 ran `plateau`. Moving only the
objective and the decode length would therefore have made the default an untested combination.

## Decision

1. **Imputation has its own defaults.** `task_defaults("imputation")` in
   `src/training/types.py` returns these values:

   | key | value |
   |---|---|
   | `PRETRAIN_OBJECTIVE` | `embedding_normalized` |
   | `EPOCHS_DECODE` | 450 |
   | `LR_SCHEDULER` | `cosine` |

   - **Where they apply:** `src/training/config.py` layers these values beneath every source a run
     reads: the defaults, a promoted file, or an override mapping such as an Optuna trial or a
     study runner. They also sit beneath the flags. A key that a source or a flag names wins.
   - **Classification:** it has no task defaults, so it keeps the dataclass defaults and stays
     bit-identical.
2. **`--decode_epochs`** (imputation only) overrides `EPOCHS_DECODE`, as `--decode_patience` does.
   - **Recovering the old defaults:** imputation's defaults from before this ADR are one command
     away: `--pretrain_objective embedding --decode_epochs 150 --lr_scheduler cosine_legacy`.
   - **Recovering E30's arm A:** use `--lr_scheduler cosine` in that command instead.
3. **The five promoted `*.imputation.json` files now hold M's values:**
   - `PRETRAIN_OBJECTIVE embedding_normalized`;
   - `EPOCHS_DECODE 450`;
   - `DECODE_PATIENCE 0` and `DECODER_HEADS batched`, written out because a promoted file names
     every value (ADR 0005, decision 5).

   The five files are credit-g_40nan, kr-vs-kp_20nan and _40nan, and spambase_20nan and _40nan.
   - **Why their old 150 could be overwritten:** all five held `EPOCHS_DECODE 150`, which the
     full search profile (20–60) never produces. So 150 was the reduced profile's held default,
     not a tuned value.
   - **What the rewrite equals:** it is what their promotion would have written under the new
     default, and it is exactly the configuration of E30's M cells on those variants.
   - **The old contents:** they stay at commit `f147c39`.
4. **Records:**
   - `PRETRAIN_OBJECTIVE` joins imputation's key set. That key set is used by the logged params,
     by promoted files, and now also by a run's `hyperparameters.json`.
   - `hyperparameters.json` is now written through `complete_configuration`, which also adds
     `DECODE_PATIENCE` and `DECODER_HEADS`. Before this, the file left both out.
   - The `pretrain_objective` and `lr_scheduler` tags are already on every run, and
     `EPOCHS_DECODE` is a param. Every earlier run therefore already records what it trained with,
     and no backfill is needed.
5. **Searches hold a knob where a plain run would put it:**
   - **Reduced profile:** an imputation study now holds `EPOCHS_DECODE` at 450, about 3x the
     decode cost per trial. It also holds the normalised objective and the `cosine` schedule.
   - **Full profile:** it still samples `EPOCHS_DECODE` from 20–60.
   - **Configuration flags reach the trials:** all five flags, `CONFIGURATION_FLAGS` in
     `config.py`, now reach every trial, the retrain and the promoted file. Before this, a study
     forwarded only `--lr_scheduler` and silently dropped `--decode_patience`, `--decoder_heads`
     and `--pretrain_objective`.
   - **Schedule tags and promotion:** when `--lr_scheduler` is absent, a study's tags and its
     promoted file take the task's schedule instead of `cosine_legacy`.

## Verification

- **Default runs match E30:** default imputation runs at seed 42, through `main.py` with no
  configuration flag, matched E30's M cells fold by fold to the last digit, on both scored
  populations. One run was on a promoted variant (credit-g_40nan) and one on a variant without a
  promoted file (vehicle_20nan).
- **Regression fixtures:** both fixtures pass unmodified. The credit-g imputation test now asserts
  that its mapping, read without a task, stays on the configuration it was recorded with
  (`embedding`, `cosine_legacy`).
- **Check cell:** the pre-training ablation's check cell (`pretrain_ablation_run.py
  credit-g_20nan 42 B ... --no-mlflow`, in the handoff kit) names no objective, so by default it
  now runs the normalised one. Even arm B, which pre-trains for no epochs, moved: induced
  0.8976199230868737, masked 0.9028980606175766.
  - **The fix:** the kit now pins `PRETRAIN_OBJECTIVE embedding`.
  - **With the pin, the reference holds:** induced 0.8863449097353776, masked 0.9256312571312231.

## Consequences

- **Cost:** an imputation run costs about 1.65x what it did, summed over the 21 variants
  (electricity, 64 → 100 min a cell). This is the price of the 450-epoch decode stage.
- **Promoted variants:** A's point estimate stays ahead of M, without a verdict, on three of the
  five variants whose promoted files were tuned for A's pipeline (kr-vs-kp_20nan and _40nan,
  spambase_40nan). Re-tuning those files under the new defaults is the natural next step for
  them.
- **Re-running finished studies:** the study runners under `scripts/experiments/` read "the
  configuration a run of it loads today". Re-running a finished study after this decision needs
  the promoted files as of `f147c39` (`git show f147c39:datasets/hiperparams/...`). An old arm
  also needs every value it leaves to the defaults pinned, as the kit was. For example, the A
  arms of `confirmatory_run.py` and `batched_heads_run.py` name the objective but not
  `EPOCHS_DECODE`, so on a variant without a promoted file they would now decode for 450 epochs.
- **P is not adopted:** P (patience 50) remains the cheaper candidate. Tasks T07 (checkpoint by
  validation `impute_score`) and T11 (longer patience) are what could make it qualify.
