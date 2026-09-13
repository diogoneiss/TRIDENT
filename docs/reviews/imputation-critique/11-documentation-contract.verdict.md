# 11 - The documentation contract: CONTEXT.md, ARCHITECTURE.md and README against the code: verdicts

_Adversarial verification of `11-documentation-contract.md`. 2026-09-11._

**Method:** read every documented sentence the critique cites against the source it names —
`CONTEXT.md:25-80`, `:100-171`, `docs/ARCHITECTURE.md:340-440`, `README.md:160-170`, `:265-340`,
`:375-385`, `docs/adr/0004-imputation-decoder-task.md:55-64`, `:224-228`,
`docs/adr/0005-reduced-optuna-search-for-imputation.md:41`, `docs/BACKLOG.md` H5 — and against
`src/training/decoding.py` (whole file), `src/training/runner.py:20-110`,
`src/training/types.py:1-180`, `src/training/config.py:20-45`, `:95-115`, `:215-235`,
`src/training/schedulers.py`, `src/training/artifacts.py:55-80`, `src/training/finetuning.py:88-130`,
`src/embedder.py:17-27`, `:100-125`, `src/models.py:10-150`, `opt.py:80-160`, `:180-260`, `:395-405`,
`tests/integration/test_credit_g_imputation_regression.py`,
`tests/fixtures/credit-g_20nan_imputation_regression.json`,
`tests/unit/test_imputation_metrics.py:95-127`, `imputation_studies.ps1`, `experiment_imputation.ps1`.
Oriented with `graphify query "where is the categorical vocabulary built and where does the decoder
head width come from"`. **No training run** — nothing I claim needed one, and the one measurement that
could have moved a severity (F-11-6's scoring share) needs an instrumented stage call, which the
harness's single permitted run form does not allow; I say so where it matters.

Four empirical checks, all in the scratchpad, none touching the tree:

1. **Vocabulary census over the real corpus** (real `as_category_strings` + `models.NON_CATEGORY_TOKENS`):

   ```
   kr-vs-kp_00nan: catcols=36 totalNaN=0      excluded-set=[2] example(V,excl)=(4, 2) nunique_hist={2: 35, 3: 1}
   kr-vs-kp_20nan: catcols=36 totalNaN=23004  excluded-set=[3] example(V,excl)=(5, 3) nunique_hist={2: 35, 3: 1}
   credit-g_00nan: catcols=13 totalNaN=0      excluded-set=[2] example(V,excl)=(6, 2)
   credit-g_20nan: catcols=13 totalNaN=2600   excluded-set=[3] example(V,excl)=(7, 3)
   electricity_00nan: catcols=1 totalNaN=0    excluded-set=[2] example(V,excl)=(9, 2)
   electricity_20nan: catcols=1 totalNaN=9062 excluded-set=[3] example(V,excl)=(10, 3)
   ```

2. **Direct head-width reproduction** — built the real `TabularEmbedder` + `TridentDecoder` on four
   variants and compared `len(encoder.classes_)` against `categorical_heads[key].out_features`:

   ```
   kr-vs-kp_00nan: real V - out_features histogram = {2: 36} | documented V-3 widths = {1: 35, 2: 1}
   kr-vs-kp_20nan: real V - out_features histogram = {3: 36} | documented V-3 widths = {2: 35, 3: 1}
   credit-g_00nan: real V - out_features histogram = {2: 13} | documented V-3 widths = {3: 4, 4: 3, 9: 1, 2: 3, 1: 2}
   credit-g_20nan: real V - out_features histogram = {3: 13} | documented V-3 widths = {4: 4, 5: 3, 10: 1, 3: 3, 2: 2}
   ```

3. **Store censuses** (`sqlite3.connect("file:mlflow.db?mode=ro", uri=True)` — no `sqlite3` binary on
   this box). The store now holds **452** runs, not the critique's 450: two more `optuna_trial` runs
   landed since it was written. Every census reproduces with that drift.

   ```
   run_role census: optuna_trial 186, parent 93, worst_fold 78, best_fold 78, optuna_study 5
   runs with no run_role: 12
   tag coverage: is_optuna 452, lr_scheduler 452, task 452, run_role 440, run_type 287, search_space 191
   test/impute/induced/impute_score rows: 189
   runs carrying any test/impute/%: optuna_trial 185, worst_fold 2, best_fold 2
   validation/impute/% metric rows: 1085
   task x lr_scheduler: classification = cosine_legacy 191, plateau 18, cosine 18, warmup_cosine 12,
                        constant 12; imputation = cosine 193, plateau 8  (cosine_legacy: ZERO)
   datasets with any impute_score: credit-g_20nan 43, kr-vs-kp_40nan 40, kr-vs-kp_20nan 40,
                                   credit-g_40nan 40, spambase_20nan 28
   the 12 untagged: 3 x train_vehicle_00nan_* (0 metrics, run_type=train) + 9 x fold_N_of_3 (248 metric
                    rows each, nested), all carrying task_backfilled and lr_scheduler_backfilled = true
   ```

4. **Fixture resolution and `cosine_legacy` over the decode budget** (real `Hyperparameters.from_mapping`,
   real `CosineAnnealingLR`, counting returns to the base rate):

   ```
   fixture resolves lr_scheduler -> cosine_legacy | decode_epochs 2 | batch 64
   fixture decode (2 epochs, 2-fold, BATCH=64):      steps=16    returns_to_base=4  final_lr=1.000e-03
   150-epoch decode, credit-g cv3, BATCH=256:        steps=450   returns_to_base=1  final_lr=0
   150-epoch decode, credit-g cv3, BATCH=64:         steps=1350  returns_to_base=4  final_lr=0
   150-epoch decode, electricity cv3, BATCH=256:     steps=26550 returns_to_base=88 final_lr=0
   ```

## Verdicts

### F-11-1 - CONTEXT's "Search objective" promises the test split "stays untouched"; every trial scores it and logs it

- **Verdict:** SOUND
- **Severity after review:** medium (critic: high)
- **Basis:** both halves of the premise check out. The narrow guarantee holds — `opt.py:266` reads
  `metrics[self.search_objective]` and `types.py:71` makes that
  `validation/impute/masked/impute_score`. The broad one does not, and nothing in
  `train_and_evaluate_decoder` is conditional on the search: the masked test population
  (`decoding.py:145-157`), every `EVAL_MASK_RATES_EXTRA` rate (`:162-172`), the induced population with
  its sibling load and scaler transform (`:174-182`) and, under `--score_null_path`, the null-token
  population (`:183-196`) are all computed, then `decoding.py:213` publishes them:

  ```python
  tracker.log_metrics({f"test/{name}": float(value) for name, value in metrics.items()})
  ```

  `score_search_objective` at `:202-211` only *adds* a `validation/` family; it gates nothing off. The
  store agrees: 189 `test/impute/induced/impute_score` rows, and of the runs carrying any
  `test/impute/%` key, 185 are `optuna_trial`. The two sibling documents on this branch state the true
  thing — `README.md:381` ("the final metrics ... `test/*` and `time/*` for the predefined split that
  trials use") and ADR 0005:41 ("**A trial scores the test half of the predefined split today**, for both
  tasks"). The reasoning follows: "untouched" is a claim about what exists in the store, and 185 trial
  runs falsify it.
- **Correction:** cut from high to medium. The guarantee that protects the *result* — no key the search
  reads is a `test/` key — is genuinely implemented and genuinely documented elsewhere, so no measured
  quantity in this repository is invalidated by the false sentence. The exposure is a reader forming a
  mid-study view from a column the definition says does not exist, plus per-trial compute nobody reads.
  That is a serious documentation defect and a design smell, not a validity threat. Note also that
  consequence (b) — wasted per-trial work, and the headline sitting beside `optuna/objective_value` — is
  a design argument riding on a methodology finding. It is correct, and it is what makes the *fix*
  attractive; it is not what makes the sentence false.

### F-11-2 - ARCHITECTURE asserts every categorical vocabulary holds `"nan"` and the head is `Linear(d, V_col − 3)`; on `_00nan` variants it is `V_col − 2`

- **Verdict:** CONFIRMED
- **Severity after review:** low (critic: medium)
- **Basis:** `docs/ARCHITECTURE.md:355` draws `cat_heads[key]: Linear(d, V_col − 3)` and `:361-365`
  states "Every categorical vocabulary contains `[MASK]`, `[NULL]` and the placeholder a missing cell
  stringifies to (`"nan"` ...) ... so the head has `V_col − 3` outputs" — an unqualified universal.
  `embedder.py:109-115` builds the vocabulary as `np.unique(as_category_strings(df[col]))` concatenated
  with `["[MASK]", "[NULL]"]` only, so `"nan"` enters exactly when the column holds a missing cell.
  I did not stop at the vocabulary count: I built the real `TabularEmbedder` and the real
  `TridentDecoder` on four variants and read `categorical_heads[key].out_features` back. On both
  `_00nan` variants every head is `V − 2` (36/36 on kr-vs-kp, 13/13 on credit-g); on both `_20nan`
  variants every head is `V − 3`. The documented formula on `kr-vs-kp_00nan` gives **one output on
  35 of its 36 columns** (`nunique_hist={2: 35, 3: 1}`) — a constant predictor per column, with
  `local_of` built one short. The shipped code is right: `models.py:118-127` builds `excluded` with
  `if token in NON_CATEGORY_TOKENS` over `encoder.classes_`, a membership test, never a count.
  ADR 0004:57-60 carries the qualifier ARCHITECTURE dropped: "every vocabulary of a column **with
  missing values** (verified: 13 of 13 categorical columns on `credit-g_20nan`)".
- **Correction:** cut from medium to low. No shipped code is wrong and no run is affected; this is a
  false universal in a reference document whose harm requires someone to edit correct code to match it.
  One minor slip in the critique's own table: it reports `credit-g_20nan total NaN=4000`, which is the
  whole-table figure; among the 13 categorical columns it is 2600 (the other 1400 are the numerical NaNs
  ARCHITECTURE counts two paragraphs later). The finding is unaffected.

### F-11-3 - The hyperparameter wiring table gained an `lr_scheduler` row, no decode rows, and three surviving rows are now task-inaccurate

- **Verdict:** SOUND
- **Severity after review:** low (critic: medium)
- **Basis:** the table at `docs/ARCHITECTURE.md:409-433` has no row for `EPOCHS_DECODE`, `LR_DECODE`,
  `WEIGHT_DECAY_DECODE`, `LAMBDA_NUM`, `EVAL_MASK_RATE` or `EVAL_MASK_RATES_EXTRA` — verified by reading
  every row. The three stale rows check out exactly as claimed:
  - `mask_probability | PROB_MASCARA | "p_base passed to preprocess_table during pretraining"` —
    `decoding.py:91-96` calls `preprocess_table(train_frame.copy(),
    p_base=hyperparameters.mask_probability, fine_tunning=False)` inside the decode epoch loop, and
    `opt.py:96-98` makes it the reduced profile's first sampled knob.
  - `hidden_dimension | HIDDEN_DIM | "... inside TabularEmbedder"` — `models.py:141-147` builds every
    decoder numerical head as `Linear(embedder.dimensao, embedder.hidden_dim) → ReLU →
    Linear(embedder.hidden_dim, 1)`.
  - `pretraining_epochs / finetuning_epochs | EPOCHS_PRE / EPOCH_FINE | "the horizon of each stage's
    StageScheduler"` — `grep -rn finetuning_epochs src/` returns only `finetuning.py:115,129` (the
    classifier loop), `types.py:98,151` and the two serialisers; the decode stage's horizon is
    `hyperparameters.decode_epochs` (`decoding.py:64-69`). On an imputation run `EPOCH_FINE` reaches no
    training code.
  - `batch_size | BATCH | "both training loops"` is three loops now, and the decode evaluation path is
    un-batched (`decoding.py:137-196` encodes and scores whole frames).
- **Correction:** cut from medium to low, and correct one consequence. The critique writes that "for half
  the branch's surface it answers nothing"; that is true of the *table*, but README's Configuration
  System documents all six decode keys with defaults and semantics at `README.md:268-273`, and
  README:272 gets the very point the stale row misses — it calls `EVAL_MASK_RATE` "the evaluation
  counterpart of `PROB_MASCARA`, which governs **training** corruption", not pretraining corruption. The
  table's own intro also delegates JSON-key semantics to README ("documented in README's Configuration
  System"). So the operator-facing documentation is correct and the defect is confined to one reference
  table. The three stale rows stand on their own and are worth fixing. The four drifted code anchors are
  style and I do not carry them; the critique already labels them "supporting, not load-bearing".

### F-11-4 - `--lr_scheduler cosine` is prescribed in two documents, defaulted in none, and the imputation fixture is pinned under `cosine_legacy`

- **Verdict:** SOUND
- **Severity after review:** low (critic: medium)
- **Basis:** every mechanical premise reproduces. `config.py:222-229` parses `--lr_scheduler` with
  `default=None`; `config.py:225-228` replaces the field only when the flag is present; so the effective
  default is `Hyperparameters.lr_scheduler = DEFAULT_LR_SCHEDULER = "cosine_legacy"` (`types.py:17`,
  `:107`), with no task-aware branch anywhere. ADR 0004:227-228 and README:306 both prescribe `cosine`;
  README:303-304's own second example omits the flag. `grep -n LR_SCHEDULER
  tests/fixtures/credit-g_20nan_imputation_regression.json` returns nothing, and I confirmed the
  resolution rather than inferring it: `Hyperparameters.from_mapping(fixture["hyperparameters"])` yields
  `lr_scheduler -> cosine_legacy`. The store shows the prescription is universally followed in practice:
  imputation = 193 `cosine` + 8 `plateau`, **zero** `cosine_legacy`, against 191 `cosine_legacy`
  classification runs. The mechanism the documents warn about is real: `schedulers.py:78-81` builds
  `CosineAnnealingLR(T_max=epochs)` stepped once per batch (`decoding.py:107`), period `2 × T_max`
  steps, so the decode stage completes a dataset-dependent number of complete cosine cycles and ends at a
  learning rate determined by where the batch count lands in the cycle. `config.py:229`'s justification
  for the default ("the schedule of every run before ADR 0003") is indeed empty for a stage with no
  pre-ADR-0003 runs.
- **Correction:** cut from medium to low, plus two evidence corrections. (1) The Evidence reads as if
  "the fixture's `BATCH=64` gives 1500 steps and four returns" described the fixture; it does not — the
  fixture also pins `EPOCHS_DECODE: 2`, so its real trajectory is **16 steps and 4 complete cosine cycles
  at `T_max=2`, ending at the base rate**, which I measured. The 150-epoch figures describe a
  hypothetical run, as the critique's own Direction acknowledges in passing. (2) My step counts differ
  from the critique's (1350 vs 1500 for credit-g at BATCH=64; 26,550 vs 16,050 for electricity) because
  neither of us stated the train-fold row count they depend on; the mechanism and the qualitative
  conclusion are identical and I do not call the critique's numbers wrong. On severity: both launchers
  pass the flag (`imputation_studies.ps1:63,84`, `experiment_imputation.ps1:62`), the store shows no run
  has ever inherited the bad default, and a regression fixture's job is determinism, which
  `cosine_legacy` provides as well as any schedule. The live exposure is a reader copying README:303-304,
  plus the genuine oddity that the one reviewed imputation baseline freezes a schedule no experiment
  uses. Both are worth fixing cheaply; neither is a medium.

### F-11-5 - CONTEXT defines `full` as sampling every hyperparameter the task uses; it permanently holds three

- **Verdict:** SOUND
- **Severity after review:** low (critic: medium)
- **Basis:** `CONTEXT.md:155` reads "`full` samples every hyperparameter the task uses". `opt.py:110-150`'s
  `full` branch samples `HEADS`, `HEAD_DIM`→`DIM`, `HIDDEN_DIM`, `LAYERS`, `DIM_FEED`, `DROPOUT`,
  `EPOCHS_PRE`, `BATCH`, `LR_PRE`, `WEIGHT_DECAY_PRE`, `PROB_MASCARA`, then on imputation
  `EPOCHS_DECODE`, `LR_DECODE`, `WEIGHT_DECAY_DECODE`, and `LAMBDA_NUM` only `if mixed_columns`.
  `EVAL_MASK_RATE`, `EVAL_MASK_RATES_EXTRA` and `LR_SCHEDULER` are never sampled by either profile — all
  three are real `Hyperparameters` fields the imputation task uses (`types.py:107`, `:122`, `:124`), and
  `LR_SCHEDULER` is a first-class knob under ADR 0003 with its own JSON key and MLflow tag.
  `opt.py:128-129` states the omission of the evaluation rate and gives the right reason ("a trial free
  to hide fewer cells would win by making its own exam easier"). The contradiction is internal and exact:
  `CONTEXT.md:160-163` defines "Held hyperparameter" as one a profile does not sample, so by CONTEXT's
  own next entry `full` holds three.
- **Correction:** cut from medium to low. This is a definitional imprecision in a glossary whose
  consequences are soft — the strongest one offered ("an operator will look for an `EVAL_MASK_RATE`
  importance that can never exist") requires someone to audit an importance ranking for a knob nobody
  would tune. The fix is a sentence. The finding is worth keeping because the overclaim is repeated in
  two more places, one of them operator-facing (see *What this critique missed*), not because one
  glossary entry is a medium-severity defect.

### F-11-6 - The stage-timing contract: the imputation window wraps the entire scoring pass and moves with flags that change no training

- **Verdict:** SOUND
- **Severity after review:** low
- **Basis:** `runner.py:78` starts the clock, `runner.py:102` reads it, and the whole
  `train_and_evaluate_decoder` call sits between them; the comment at `:103-105` asserts "the timing
  covers training only". It does not. After the training loop ends at `decoding.py:131`, still inside the
  window: `mean_mode_baselines` (`:134-136`), the masked test encode-and-score (`:137-157`), one extra
  full test population per `EVAL_MASK_RATES_EXTRA` entry (`:162-172`), the induced population with a
  sibling slice and a `scaler.transform` (`:174-182`), optionally the null-token population (`:183-196`),
  optionally the validation objective (`:202-211`), and the per-cell ledger built row by row in
  `_score_population` (`:285-330`) with a `raw_truth.at[row, column]` lookup per numerical cell
  (`:333-341`). Two runs of one configuration differing only in `--score_null_path` therefore report
  different `time/finetune_seconds` while performing identical training — which is the falsifying case,
  independent of the unmeasured share. That share stays unmeasured: settling it needs an instrumented
  stage call, which the single permitted training form does not allow.
- **Correction:** the false sentence is **README:379** ("They cover the two training stages only, not
  data preparation, plotting, or artifact writes"), not CONTEXT. `CONTEXT.md:61-62` says the seconds are
  "measured in the runner **around each stage call**" and excludes "data preparation, plotting, and
  artifact **writes**" — all three literally true, since the ledger is built in memory inside the window
  and written outside it. Reading CONTEXT as false requires reading "the seconds one fold spent in the
  decode stage" as excluding the stage's own scoring, which its next clause contradicts. The critique's
  "same-branch inconsistency" sub-argument is also weak: CONTEXT names no metric key, and the decode
  stage's seconds *are* a tracked, comparable metric — under `time/finetune_seconds`, exactly as
  BACKLOG H5 records — so the two documents do not in fact conflict. Finally, the window has always
  included the second stage's test evaluation (`train_and_evaluate_classifier` evaluates inside the same
  clock), so this is a pre-existing imprecision the decode stage enlarges, not one the branch introduced.

### F-11-7 - "On the test fold only" in two documents, while a trial scores a validation population and `--score_null_path` adds a fourth

- **Verdict:** SOUND
- **Severity after review:** low
- **Basis:** `CONTEXT.md:118-119` ("Imputation error compares the decoder's reconstruction against it on
  the test fold only") and `README.md:310` ("Two populations, on the test fold only"). Both are
  contradicted by code on this branch. `decoding.py:202-211` runs `_score_population(model,
  hidden_validation, clean_validation, "masked")` and logs `validation/impute/masked/*` whenever
  `score_search_objective` is set — 1085 such metric rows in the store. `decoding.py:183-196` scores
  `induced_null_token` on the test fold under `--score_null_path`, a flag README documents at :165 and
  omits from "What gets scored". `CONTEXT.md:146-149` then asserts the imputation search is scored on the
  validation split, which `:118-119` says is impossible. Both sentences are new on this branch and cannot
  both be right.
- **Correction:** "there is no sentence anywhere in the doc set that correctly states which populations a
  given invocation scores and on which split" is overstated. `README.md:381` states it for the search
  objective explicitly ("`validation/impute/masked/impute_score`, a validation-split score that is not
  comparable with any `test/` metric"), and `CONTEXT.md:146-149` states it too. What is missing is a
  single complete population × split × condition statement — exactly what the critique's Direction
  proposes, so the Direction is right and the Consequence oversells it.

### F-11-8 - README's "degrades correctly on the six all-numerical datasets and on all-categorical kr-vs-kp"

- **Verdict:** UNSOUND
- **Severity after review:** none
- **Basis:** the premise is true in every part and I verified it. `imputation_metrics.py:44-56` weights
  each kind by its share of scored cells, so a single-kind table reduces to that kind's ratio; the only
  test is `tests/unit/test_imputation_metrics.py:100`
  (`test_a_table_of_one_kind_of_column_is_scored_by_that_kind_alone`, docstring "Six of the nine datasets
  are all numerical and one is all categorical") over a hand-built frame; `datasets/categorical_columns/`
  holds exactly three files (credit-g, electricity, kr-vs-kp) against nine base datasets in
  `datasets/processed_datasets/`; and the store's only `impute_score`-bearing datasets are
  `credit-g_20nan`, `credit-g_40nan`, `kr-vs-kp_20nan`, `kr-vs-kp_40nan` and `spambase_20nan`.
  The reasoning is what fails. README:321 asserts a property of the metric — that it degenerates
  correctly on single-kind tables — and the critique concedes in its own Evidence that the property is
  true ("Structurally the claim is true and I am not disputing it"). A true statement about a metric's
  construction is properly evidenced by a construction proof plus a unit test; requiring a run on each
  named dataset would make every true mathematical sentence in a README a methodological defect. The
  consequence offered — "a reader ... will not re-check before publishing a table that spans the corpus"
  — has no content, because there is nothing for the re-check to find. The residual point (the two
  genuinely mixed tables go unmentioned) is conceded away in the same paragraph: README states the
  differently-weighted property "two lines earlier".
- **Correction:** the critique's own Direction rewrites a true sentence into a differently-worded true
  sentence. That is wording, which this review explicitly does not treat as a finding. Drop it.

### F-11-9 - Twelve runs carry no `run_role`, so the documented filter recipe and CONTEXT's taxonomy reach none of them

- **Verdict:** SOUND
- **Severity after review:** low
- **Basis:** the census reproduces on the live store. 452 runs: `optuna_trial` 186, `parent` 93,
  `worst_fold` 78, `best_fold` 78, `optuna_study` 5, and **12 with no `run_role` tag**. Enumerated, they
  are three top-level `train_vehicle_00nan_*` runs (2026-07-03, 2026-08-03 ×2, `run_type = train`, zero
  metrics) and their nine nested `fold_N_of_3` children (248 metric rows each: `pretrain/*`,
  `finetune/*`, `test/f1_macro`, `test/accuracy`, ...). All twelve carry `task_backfilled = true` and
  `lr_scheduler_backfilled = true`, so they answer `tags.task` and `tags.lr_scheduler` — the two axes
  ADR 0003 and ADR 0004 tell readers to compare on — while matching no `run_role` value. Tag coverage is
  exactly the partial shape claimed: `task` 452/452, `lr_scheduler` 452/452, `is_optuna` 452/452,
  `run_role` 440/452, `run_type` 287/452. `CONTEXT.md:33-36` says a cross-validation execution retains
  "one of the **two** nested MLflow runs ... `best_fold` or `worst_fold`"; these executions retained all
  three folds, a shape the taxonomy has no term for. The documented recipe is live in the repo:
  `imputation_studies.ps1:132` prints `tags.run_role = 'parent', ...` as the comparison instruction.
- **Correction:** the consequence "the `cosine_legacy` baseline population ... is silently three
  executions short" is weaker than stated. Those three top-level runs hold **zero metrics** — a
  comparison table over parents could not have used them even with a `run_role = parent` tag, because
  there is nothing to compare; and the nine children that hold the numbers are fold runs, which the
  recipe excludes by design. What survives is what the title states: the taxonomy covers 440 of 452 runs
  and the remaining twelve are reachable only by name. Low is right.

## What this critique missed

**1. The same definitional block contains a second false claim, and this one has already bitten.**
`CONTEXT.md:160-163` defines a held hyperparameter as one that "**every trial runs at the task default**
rather than at any stored configuration's value, so trials differ only in what was sampled."
`LR_SCHEDULER` is held by both profiles (F-11-5), and it does *not* run at the task default:
`opt.py:225` sets `args.lr_scheduler = self.lr_scheduler` on the trial's args, and `config.py:225-228`
applies that flag *after* `Hyperparameters.from_mapping(override)` has resolved the held knobs. So a held
knob takes a CLI value the definition says it cannot. This is not hypothetical — it is precisely how all
193 imputation runs in the store are tagged `cosine` while the held default is `cosine_legacy`
(`imputation_studies.ps1:63,84` passes `--lr_scheduler`). It sits in the glossary entry immediately after
the one F-11-5 scrutinises, and it ties F-11-4 and F-11-5 together.

**2. The `full`-samples-everything overclaim has a third instance, in the operator-facing document.**
F-11-5 cites `CONTEXT.md:155` and `opt.py:87-88`. `README.md:335` says it too — "`--search_space full`
samples every knob instead" — in the one paragraph an operator actually reads before launching a study.
A fix that touches only CONTEXT and the docstring leaves the sentence most people will read.

**3. README contradicts itself about how many test populations exist, 47 lines apart.**
`README.md:273` introduces `EVAL_MASK_RATES_EXTRA` as "extra nominal rates to score the test fold at",
and `decoding.py:162-172` duly logs a full `impute/masked/rate_NN/*` family per configured rate.
`README.md:310` then opens "What gets scored" with "**Two populations, on the test fold only**". F-11-7
folds the extra rates into its proposed table but never notes that this contradiction is internal to
README — between a key it documents and a section two pages later — which makes the fix a README edit
rather than a cross-document reconciliation.

**4. The timing contract was already imprecise for classification, which changes F-11-6's fix.**
`train_and_evaluate_classifier`'s test evaluation has always sat inside the same clock, so "cover the two
training stages only" was never true for either task. Moving the scoring pass outside the clock — the
critique's first Direction — is therefore a two-task change with a classification footprint (no metric
moves, but `time/finetune_seconds` does), which AGENTS.md's "preserve training behavior" clause makes
worth stating in whatever ticket picks it up.
