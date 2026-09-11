# 01. The held set and the ranges

Type: grilling
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-10 (charting session, rounds 1 and 2)
Blocked by: none

## Question

Ticket 13 of the previous map gave an imputation study fifteen sampled hyperparameters:
the eleven shared with classification (`DIM`, `HIDDEN_DIM`, `HEADS`, `LAYERS`, `DIM_FEED`,
`DROPOUT`, `EPOCHS_PRE`, `BATCH`, `LR_PRE`, `WEIGHT_DECAY_PRE`, `PROB_MASCARA`) plus
`EPOCHS_DECODE`, `LR_DECODE`, `WEIGHT_DECAY_DECODE` and `LAMBDA_NUM`. The user wants
degrees of freedom only where they matter for imputation. Which knobs does the reduced
profile sample, which does it hold, at what values, and over what ranges?

## Answer

Resolved in one grilling round; the recommended table was accepted line by line, with one
range widened in round 2.

**Sampled** (the knobs that govern the decode stage or the corruption it learns from):

| Knob | Range | Why it earns a dimension |
|---|---|---|
| `PROB_MASCARA` | 0.2..0.6, step 0.1 | Corruption rate in *both* stages: the decode stage re-rolls masks at this rate every epoch, so the decoder learns from exactly these cells. The floor 0.2 sits at the evaluation rate. |
| `LR_DECODE` | 1e-4..1e-2, log | The stage that produces the output. **Widened**: the old range 1e-5..1e-3 put the default 1e-3 on its upper bound, so nothing above the default was reachable. |
| `WEIGHT_DECAY_DECODE` | 1e-5..1e-2, log | Same stage. |
| `DROPOUT` | 0.1..0.5, step 0.1 | The one regulariser acting on both stages; a cheap dimension. |
| `LAMBDA_NUM` | 0.1..10, log, **mixed tables only** | Weights the numerical term against the categorical one. Of the chosen datasets only credit-g has both column types (electricity, the other mixed table, is out of scope); on the six numerical-only tables it is a pure loss scale that Adam largely normalises away, and on kr-vs-kp it multiplies nothing. |

**Held** at the task default, which is what `hyperparams_override` yields for any key the
profile omits (`Hyperparameters.from_mapping` replaces the base; no configuration JSON
exists under `datasets/hiperparams/`):

| Knob | Held at | Why it is held |
|---|---|---|
| `DIM`, `HEADS`, `LAYERS`, `DIM_FEED`, `HIDDEN_DIM` | 128, 16, 2, 32, 16 | Architecture, shared with classification; the paper's values. |
| `BATCH` | 256 | Cost in both stages. |
| `EPOCHS_PRE`, `LR_PRE`, `WEIGHT_DECAY_PRE` | 300, 0.00034, 0.005 | Pre-training is shared and protected; tuning it doubles the cost of a stage both tasks use. Pre-training keeps its last epoch, so `EPOCHS_PRE` is a real knob, held by decision. |
| `EPOCHS_DECODE` | 150 | A budget: the decode stage keeps the best-validation-loss checkpoint, so more epochs only cost time. |

Five dimensions on credit-g, four elsewhere. The user was asked whether any held value
should be set deliberately rather than inherited (`EPOCHS_PRE` being the candidate) and
kept the defaults.

**Cost consequence, recorded so the budget in ticket 03 is not a surprise**: the old full
space sampled `EPOCHS_PRE` and `EPOCHS_DECODE` in 20..60, so its trials were five to ten
times cheaper than a default run. Holding both at 300 and 150 means a trial costs exactly
what the promoted configuration will cost to run, which is the consistent choice: the
optimum of a learning rate depends on the budget it is run under.

**Round 2 amendment (Q14)**: the `LR_DECODE` widening and the `LAMBDA_NUM` conditional
are corrections, not reductions, so they apply to the `full` profile as well. The two
profiles therefore differ only in which shared knobs are held; ADR 0005 amends ticket 13's
ranges accordingly. No imputation study exists in the store, so nothing published depends
on the old ranges.

**Mechanism note for the plan**: the `LAMBDA_NUM` conditional needs the objective to know
the dataset's column mix before sampling. Read it from the pipeline's own declaration,
the way `src/training/data.py` does: the categorical list is
`datasets/categorical_columns/<base>.txt` and every other feature column is numerical, so
"mixed" means both lists non-empty. Never infer the mix from pandas dtypes: electricity's
integer-coded `day` is declared categorical, which dtype inference misses (the charting
session made exactly that mistake before verifying).

## Comments
