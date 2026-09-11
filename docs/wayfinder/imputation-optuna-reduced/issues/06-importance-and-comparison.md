# 06. Checking the reduction and proving the tuning helped

Type: grilling
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-10 (charting session, round 2)
Blocked by: none

## Question

The held set in ticket 01 is prior-driven. Is there a check that the reduction did not
hold a knob that matters? And is a protocol that shows whether tuning helped part of the
destination, or a later effort?

## Answer

**Checking the reduction**: log `optuna.importance.get_param_importances(study)` (the
default fANOVA evaluator) on the study parent, as a JSON artifact and as
`optuna/importance/<knob>` metrics, computed at the end over completed trials. It ranks
only the sampled knobs, which is accepted: the six studies' own importances are the
evidence worth having. A full-profile pilot on `credit-g_20nan` to rank the held knobs was
**rejected for now**: fifteen dimensions at 50 trials rank noisily and would settle
nothing. It sits in the fog and returns if the reduced importances look flat.

**Proving the tuning helped**: yes, part of the destination, as the ADR's comparison
protocol and the plan's last task. Without it the promoted configurations are six JSON
files with no claim attached.

- For each of the six (dataset, variant) pairs: one 5-fold cross-validation run with the
  promoted configuration against one with the defaults, both `--task imputation
  --cv_folds 5 --lr_scheduler cosine --seed 42`, so the only difference is the
  configuration. The two imputation runs already in the store used `plateau` and 2 folds
  and do not serve; fresh default runs are part of the protocol.
- Headline: `cv/test/impute/induced/impute_score/mean` with its `ci95_lower` /
  `ci95_upper`, the metric ADR 0004 names for `_20nan`..`_80nan` variants; companion
  `cv/test/impute/masked/impute_score/mean`, comparable across the ladder. Lower is better;
  1.0 is baseline parity.
- Filter: `run_role = parent`, `task = imputation`, `is_optuna = false`,
  `lr_scheduler = cosine`, `dataset_variant` per pair. The tuned and default parents are
  told apart by their logged hyperparameters and the loaded-configuration param from
  ticket 04; six pairs do not need a provenance tag.
- The protocol is the glossary's *paired single-seed CV protocol*: fold-level comparisons
  are paired, conclusions do not quantify seed variability. Multi-seed robustness is out
  of scope.
- Cost: twelve 5-fold runs. Scaled from ticket 07's single-split measurements (a 5-fold
  fold trains on about 72% of the rows against the single split's 80%), credit-g is about
  5 min a run, kr-vs-kp about 22, spambase about 37, roughly 2 hours in total.

## Comments
