# 11. Multi-rate evaluation: across runs or within one run?

Type: grilling
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-10
Blocked by: none

## Question

Ticket 03 made the evaluation mask rate a hyperparameter (default 0.2). The literature
reports error-vs-rate curves over {0.1, 0.3, 0.5, 0.7}. Two ways to produce one:

- **Across runs**: one run per rate, set through the hyperparameter and a sweep script
  like `experiment.ps1`; runs compared in MLflow by a param or tag. Simple and
  self-describing, but the rate only affects evaluation, so every extra rate re-trains
  the whole model (hundreds of pretraining epochs) for a few forward passes.
- **Within one run**: score the test fold at several rates in the same run and log
  `.../rate_<r>/...` metrics. One training, cheap evaluation, but the hyperparameter
  becomes a list, metric names multiply, and the fold-ranking metric must pin one rate.

Feeds ticket 08 (scalar vs list-valued hyperparameter) and ticket 05 (metric names).

Recommended starting answer: within one run, with 0.2 as the primary rate used for
ranking and every other rate logged as a diagnostic only.

## Answer

Resolved 2026-09-10 in one grilling round; every recommendation was accepted.

**The ticket's premise needed correcting first.** The evaluation mask rate is *not*
evaluation-only: ticket 03 put the validation fold on the same fixed draw and ticket 04
restores the best validation epoch, so changing the rate changes which checkpoint is kept
and therefore changes the model. A naive sweep would compare different models at
different difficulties rather than one model across difficulties.

**A second asymmetry the ticket did not name:** the induced-missing population's rate is
fixed by the dataset variant, so the `_20nan` through `_80nan` ladder *already is* the
across-runs sweep. Only self-masked cells can vary within a run.

**Decisions:**

1. **Multi-rate scoring happens within one run, for self-masked cells only.** Once
   validation is pinned, each extra rate costs one forward pass over a test fold that is
   already scored in a single pass, against re-running hundreds of pre-training epochs per
   rate. The induced-missing side needs no decision.
2. **One designated primary rate drives both validation checkpoint selection and fold
   ranking.** Extra rates are scored on the resulting checkpoint as test-only diagnostics.
   This keeps one model per run, keeps `impute_score` unambiguous, and makes the extra
   rates genuinely free rather than quietly model-changing.
3. **`EVAL_MASK_RATE` stays a scalar** -- the primary, default 0.2, exactly as ticket 08
   decided -- **plus a new optional list key `EVAL_MASK_RATES_EXTRA`, default empty.**
   Folding both into one list would make the primary mean "element zero", which is
   implicit, reorderable by accident, and would change the type of an already-decided key.
4. **Extra rates are named by integer percent**, matching how the dataset variants already
   encode missingness, giving `impute/masked/rate_10/rmse_num_z` and friends. **The primary
   rate's metrics stay unprefixed**, so every name ticket 05 decided is unchanged and the
   ranking metric key does not move.
5. **No sweep by default.** An ordinary run scores the primary rate only and produces
   exactly ticket 05's metric set; sweeping is opt-in through the config. Recorded cost:
   `preprocess_table`'s at-least-one-mask-per-row fallback is a Python loop that is slow at
   low rates (backlog P1 measures ~1.4 s on 45k rows at 0.05), so a sweep including 10%
   pays roughly a second per fold there -- once per fold, not per epoch, because the
   evaluation draw is fixed.

Consequences:

- Ticket 08's open note is resolved: `EVAL_MASK_RATE` does **not** become list-valued.
- Ticket 13 must hold both the primary rate and the extra list fixed across Optuna trials,
  or a trial could win by making its own evaluation easier.

## Comments
