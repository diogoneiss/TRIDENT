# 05. The search-space profile flag and its record

Type: grilling
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-10 (charting session, rounds 2 and 3)
Blocked by: none

## Question

How is the reduced space expressed: by replacing the imputation branch of
`define_search_space` outright (no imputation study exists in the store to keep
comparable), or behind a profile flag that keeps the full space reachable? If a flag,
what does it mean for classification, and how is the choice recorded on the runs?

## Answer

**Round 2 (user's decision, against the recommendation)**: a profile flag,
`--search_space {full,reduced}`. The charting session recommended replacing the branch
because the flag-and-tag preference exists to keep old runs comparable and there are
none; the user preferred keeping the full space reachable, extending the flag-and-tag
pattern to search spaces regardless. Recorded as a standing preference in the map's notes.

**Round 3 consequences, both recommendations accepted**:

- **Default per task**: the flag defaults to `reduced` for imputation and `full` for
  classification. `--search_space reduced --task classification` is a **parse-time
  rejection** until a separate effort defines that profile; defining it now by analogy
  (hold the same set, sample `LR_FINE` and `WEIGHT_DECAY_FINE`) would sample a space no
  one has asked to run. Out of scope on the map.
- **Recorded as a sparse `search_space` tag and param** on the study parent and every
  trial, absent on non-Optuna runs. Sparse is safe because the dense `is_optuna` tag
  already gates any filter on it; a dense value that is always `none` outside searches
  would carry no information `is_optuna` does not. **No backfill**: the store holds no
  Optuna run.
- The two profiles differ only in which shared knobs are held (ticket 01, Q14): the
  `LR_DECODE` widening and the `LAMBDA_NUM` conditional apply to both.

**Glossary**: *search-space profile* and *held hyperparameter* are in `CONTEXT.md`. The
glossary deliberately avoids "frozen" for a held hyperparameter, because ADR 0004 uses
"frozen encoder" for weights that are not fine-tuned, and the two would collide in the
same sentence.

**For the plan**: `define_search_space(trial, task, profile, mixed_columns)`; a
`SEARCH_SPACE_PROFILES` tuple in `src/training/types.py` in the `LR_SCHEDULER_NAMES`
style; the flag in `src/training/config.py` next to `--n_trials`; the tag constant in
`src/mlflow_utils.py`; the study parent's tags built through `_execution_tags` rather than
`build_run_tags`, which is also how ticket 07's missing `task` and `is_optuna` land.

## Comments
