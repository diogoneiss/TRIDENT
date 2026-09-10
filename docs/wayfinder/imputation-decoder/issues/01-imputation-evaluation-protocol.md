# 01. Research: how do imputation papers score mixed-type tabular imputation?

Type: research
Status: resolved
Blocked by: none
Assignee: research subagent (charting session, 2026-09-10); resolved 2026-09-10
Context pointer: branch `research/imputation-eval-protocol`, file `docs/research/imputation-evaluation-protocol.md`

## Question

For mixed-type (categorical + numerical) tabular imputation, what evaluation protocol and
metrics do the primary sources use?

1. Numerical error: RMSE vs MAE; on standardized / min-max values or original units;
   per-column mean vs pooled over all scored cells.
2. Categorical error: accuracy, error rate, macro-F1; per column vs pooled.
3. How (if at all) a single scalar ranks models when both column types exist.
4. Which cells are scored: artificially masked observed cells (MCAR, which rates) vs
   originally-missing cells when a complete table exists; test split only?
5. Train/validation/test handling and model selection for imputers.

Sources: GAIN, MIWAE, HyperImpute, ReMasker, TabCSDI, Jager et al. 2021 benchmark,
scikit-learn IterativeImputer docs.

Feeds ticket 05 (loss and error metrics).

## Answer

Resolved 2026-09-10 by a research subagent (20 fetches, papers and official code).
Full note: branch `research/imputation-eval-protocol`, file
`docs/research/imputation-evaluation-protocol.md`, commit `a243205`. Read it with
`git show research/imputation-eval-protocol:docs/research/imputation-evaluation-protocol.md`.

Gist, by sub-question:

1. **Numerical metric is RMSE everywhere** (MSE in MIWAE); MAE appears in no primary
   source. Always in a scaled space: min-max for GAIN, HyperImpute, ReMasker and TabCSDI
   (train-split min/max), z-scored for MIWAE (inferred from its Table 2, where mean
   imputation scores about 1.0). Pooled over all masked cells in GAIN, HyperImpute,
   ReMasker and MIWAE; per-column RMSE then averaged in TabCSDI.
2. **Categorical metric**: TabCSDI uses a per-column error rate on the argmax of the
   one-hot group; Jager et al. use macro-F1. GAIN rounds low-cardinality columns and
   HyperImpute/ReMasker fold categoricals into the numeric RMSE.
3. **No source defines a single combined scalar** for mixed data; TabCSDI and Jager
   report both families side by side, Jager ranks methods per condition.
4. **Every source scores only artificially masked, originally-observed cells** against a
   complete table; none scores natively-missing cells. MCAR is the universal default
   (HyperImpute, ReMasker, Jager add MAR/MNAR). Rates: 20% (GAIN default), 0.1-0.2
   (TabCSDI), 50% (MIWAE), {0.1, 0.3, 0.5, 0.7} sweeps (HyperImpute, ReMasker),
   {1, 10, 30, 50}% (Jager). TabCSDI draws the mask per column at an exact fraction,
   the same construction as `datasets/generate_splits.py::inject_nans`.
5. **Splits**: transductive (fit on the incomplete table, score its own masked cells) in
   GAIN's code, MIWAE, HyperImpute, ReMasker; inductive with a held-out test split in
   TabCSDI, Jager and scikit-learn. No deep source uses validation early stopping (fixed
   budgets; TabCSDI initialises `best_valid_loss` but never updates it). Model selection
   exists only in HyperImpute (CV on observed values) and Jager (grid search on train).

Recommended TRIDENT default (note section "What this means for TRIDENT"):

- Primary numerical `rmse_num_z`, pooled, in the z-scored space the model already uses;
  companions `mae_num_z` and `nrmse_num` (ratio to mean imputation on the same cells, so
  1.0 means no better than the column mean); per-column RMSE in original units logged for
  interpretation only, never ranked on.
- Primary categorical `acc_cat`, pooled, argmax over the real vocabulary only
  (`[MASK]`/`[NULL]` predictions count as wrong); secondary `macro_f1_cat` = mean of
  per-column macro-F1 with `zero_division=0`.
- Aggregation: pool within a metric family; log per-column values as diagnostics.
- Fold-ranking scalar, flagged as a **project proposal with no literature source**:
  `impute_score = w_num * (rmse_num_z / mean-baseline) + w_cat * (err_cat / mode-baseline)`
  with cell-fraction weights; lower is better, 1.0 = no better than mean/mode, degrades
  to one family's ratio on single-type tables.
- Which cells: for `_XXnan` variants score the cells NaN in the variant but observed in
  `_00nan`, test fold only, no extra masking (this *is* the literature's protocol, since
  those NaNs were injected MCAR per column). For `_00nan` and real data, add a 20% MCAR
  mask on observed test-fold cells; {0.1, 0.3, 0.5, 0.7} for sweeps. Never score cells
  NaN in `_00nan` itself.
- Stay inductive (train fold fits, test fold scores); any validation-driven early stopping
  is a behaviour change to gate behind a flag and tag; report mean and std over folds on
  `run_role = parent` runs; state the whole-frame scaler leak wherever numbers are
  published.

Caveats the note records: Jager et al. never state whether their RMSE is in original
units; MIWAE's z-scoring is inferred, not stated.

Note for ticket 03: the recommendation to score induced-missing cells is a data-side
statement. At the model those cells arrive as `[NULL]` tokens, and the decoder-heads
note (ticket 02) recommends feeding cells to be imputed as `[MASK]` at inference so the
head sees the token it was trained on. Ticket 03 still has to choose between the two
token paths; the two notes converge on substituting `[MASK]`.

Unblocks: ticket 05 (together with ticket 03).

## Comments
