# 11 - The documentation contract: CONTEXT.md, ARCHITECTURE.md and README against the code

_Critique of the imputation work on branch feat/imputation-task. 2026-09-11._

**Scope:** `CONTEXT.md` (all 172 lines, the 14 definitions this branch added at :8-14, :22-26, :59-76, :102-171),
`docs/ARCHITECTURE.md` (the new "TridentDecoder — reconstructing cell values" section, the rewritten
TabularEmbedder implementation diagram, the orchestration diagram, and the "Hyperparameter → code wiring
reference" table at :408-433), `README.md` :150-340 and :379-381, `docs/BACKLOG.md` (all 470 lines, new on
this branch), `docs/adr/0004-imputation-decoder-task.md` :27-90, :209-228, `docs/adr/0005-reduced-optuna-search-for-imputation.md`
:100-195. Read against `src/training/decoding.py:32-346`, `src/training/runner.py:22-192`,
`src/training/summary.py:23-46`, `:80-121`, `:221-236`, `src/training/tracking.py:120-340`,
`src/training/types.py:7-130`, `src/training/config.py:21-112`, `:218-232`,
`src/training/imputation_metrics.py:31-137`, `src/models.py:100-152`, `src/embedder.py:100-116`,
`opt.py:81-150`, `:300-348`, `tests/integration/test_credit_g_imputation_regression.py`,
`tests/fixtures/credit-g_20nan_imputation_regression.json`, `tests/unit/test_imputation_metrics.py:100-127`,
`experiment_imputation.ps1`, `imputation_studies.ps1`.

**Method:** one question per documented claim — does the code do this? Oriented with `graphify query`, then
read the source. Ran read-only SQL against `mlflow.db` (`sqlite3` opened `mode=ro`) for five censuses:
`run_role`/`task`/`lr_scheduler`/`search_space`/`is_optuna` tag coverage over all 450 runs, the
`test/impute/*` population by `run_role`, the datasets that ever produced an `impute_score`, the
task × schedule cross-tab, and the metric inventory of the untagged legacy runs. Ran one `uv run --python
3.10 python` snippet over the real corpus using the real `as_category_strings` and `NON_CATEGORY_TOKENS`
to count how many special tokens each categorical vocabulary actually holds (plus a cardinality census of
`kr-vs-kp_00nan`'s 36 categorical columns), one to verify ARCHITECTURE's "1400 NaN numerical targets on
credit-g_20nan", and one that drives the real `CosineAnnealingLR(T_max=150)` over the decode budget's step
count to derive how many cosine cycles `cosine_legacy` completes there. **No training run** — the task
offered one and nothing I claim needed it. **Not measured:** the share of an imputation fold's second-stage wall clock that is scoring rather than
training (F-11-6 says so explicitly and argues structurally instead). Known findings 9
(`time/decode_seconds` dead), `EVAL_MASK_RATES_EXTRA` dropped by promotion, and the `impute/masked/*`
ladder-comparability claims are not re-reported; where a documented sentence sits on top of one of them I
say which part is mine.

## Findings

### F-11-1 - CONTEXT's "Search objective" promises an untouched test split; every trial scores it, logs it, and parks it one column from the objective

- **Kind:** methodology
- **Severity:** high
- **Where:** `CONTEXT.md:144-151`, `src/training/decoding.py:213`
- **Evidence:** the definition reads "the imputation task scores its search on the validation split of the
  trial's predefined split, so the test split never chooses hyperparameters **and stays untouched until the
  chosen configuration is retrained**". The first half is true: `opt.py:266` reads
  `metrics[self.search_objective]` and `types.py:71` makes that `validation/impute/masked/impute_score`. The
  second half is false, and nothing in the code is conditional. `train_and_evaluate_decoder` scores the test
  split unconditionally — masked population (`decoding.py:145-157`), every extra rate (`:162-172`), the
  induced population against the complete sibling (`:174-182`) and, under `--score_null_path`, the null-token
  population (`:183-196`) — then `decoding.py:213` logs the lot:

  ```python
  tracker.log_metrics({f"test/{name}": float(value) for name, value in metrics.items()})
  ```

  `score_search_objective` only *adds* a `validation/` family at `:202-211`; it gates nothing off. The store
  agrees: `select count(*) from metrics where key='test/impute/induced/impute_score'` returns **187 rows**,
  and grouping the `test/impute/%` family by `run_role` gives `optuna_trial: 183`, `best_fold: 2`,
  `worst_fold: 2`. Every finished trial published a test-split reading. The branch's own operator doc says so
  out loud — README:381 lists a trial's contents as "the final metrics (`cv/test/*` and `cv/time/*`, or
  `test/*` and `time/*` for the predefined split that trials use)" — and ADR 0005's Context section says
  "**A trial scores the test half of the predefined split today**, for both tasks". So CONTEXT contradicts
  both the code and the two documents written beside it.
- **Consequence:** the defect is **both** halves. (a) The definition states a protocol guarantee the system
  does not provide. What the code actually guarantees is narrower and worth saying precisely: *no key the
  search reads is a `test/` key*. "Untouched" is a claim about what exists in the store, and 183 trial runs
  falsify it. (b) Because the test report is computed unconditionally, a trial pays for a sibling load, an
  induced-population forward and one forward per configured extra rate that no ranking will ever read, and
  the headline `test/impute/induced/impute_score` ends up sitting next to `optuna/objective_value` in the
  trial table of every study. That is not leakage — nothing feeds back automatically — but it is exactly the
  material a human uses to form a view about a search while it is still running, and this repository already
  contains a worked instance: the 07 reviewer built a Spearman table of objective-vs-test-headline across
  trials from these very rows. A reader who trusts CONTEXT:148 believes that material does not exist.
- **Direction:** decide which sentence is true and make the other one match it. Cheapest honest fix is to
  gate the test-side scoring on the trial path — when `score_search_objective` is set, score and log the
  validation objective and nothing else — which makes CONTEXT:148 true as written and removes the wasted
  per-trial work. If the test readings are wanted for post-hoc study, keep them but rewrite the definition to
  "no test-split metric is read by the search; test metrics are still computed and logged on every trial",
  and say in ADR 0005 that reading them mid-study is a protocol violation rather than an oversight.

### F-11-2 - ARCHITECTURE states a false universal about categorical vocabularies and the wrong head width; the code is right and the ADR is right

- **Kind:** bug
- **Severity:** medium
- **Where:** `docs/ARCHITECTURE.md:355` (the diagram's `cat_heads[key]: Linear(d, V_col − 3)`) and `:361-368`
  (the paragraph beginning "**The output space excludes three vocabulary entries**")
- **Evidence:** the text says "Every categorical vocabulary contains `[MASK]`, `[NULL]` and the placeholder a
  missing cell stringifies to (`"nan"`) … so the head has `V_col − 3` outputs". Vocabularies are built at
  `embedder.py:109-115` as `unique(as_category_strings(col))` plus `[MASK]`, `[NULL]` — `"nan"` only appears
  when the column actually has a missing cell. Counting the real corpus with the real helpers
  (`uv run --python 3.10 python -c ...` over `as_category_strings` and `models.NON_CATEGORY_TOKENS`):

  ```
  kr-vs-kp_00nan:   cat cols=36  total NaN=0      excluded-per-column set={2}  example V=(4, 2)
  credit-g_00nan:   cat cols=13  total NaN=0      excluded-per-column set={2}  example V=(6, 2)
  electricity_00nan:cat cols=1   total NaN=0      excluded-per-column set={2}  example V=(9, 2)
  kr-vs-kp_20nan:   cat cols=36  total NaN=23004  excluded-per-column set={3}  example V=(5, 3)
  credit-g_20nan:   cat cols=13  total NaN=4000   excluded-per-column set={3}  example V=(7, 3)
  ```

  On a complete variant the width is `V_col − 2`, on a gapped one `V_col − 3`. ADR 0004 decision 3 states this
  correctly — "the dead literal `"nan"` entry that `astype(str)` puts into every vocabulary **of a column with
  missing values** (verified: 13 of 13 categorical columns on `credit-g_20nan`)". ARCHITECTURE drops the
  qualifier and turns a variant-specific verification into a universal. `models.py:118-127` does not: it
  builds `excluded` with a membership test over `encoder.classes_`, so the shipped code is correct on both.
- **Consequence:** three of the nine base tables have categorical columns and all three have `_00nan`
  variants, so the doc is wrong on every complete-variant imputation run. The concrete trap is the
  simplification the text invites: `kr-vs-kp_00nan` is the all-categorical table and 35 of its 36 columns are
  binary (`df[cats].nunique().value_counts()` → `{2: 35, 3: 1}`, so `V=4`: two real categories plus the two
  special tokens). Implementing the documented `V_col − 3` there gives a **one-output** head on those 35
  columns — the decoder becomes a constant predictor on each — and `local_of` would be built one short,
  mapping a real category to `-1`. Nothing in the suite would catch it:
  the only classification regression baseline is `vehicle_00nan`, which has no categorical column at all
  (known finding 15), and the imputation fixture is `credit-g_20nan`, a gapped variant where `V_col − 3` is
  the right answer.
- **Direction:** restore the qualifier and state the width the way the code computes it — "`V_col` minus
  however many of `[MASK]`, `[NULL]`, `"nan"` the column's own encoder holds: two on a complete variant,
  three on a gapped one" — and keep the diagram's label consistent with that. Point the paragraph at
  BACKLOG C2, which is where the `"nan"` entry is tracked.

### F-11-3 - The "Hyperparameter → code wiring reference" was updated for the schedule and not for the task, and three of its surviving rows are now wrong for imputation

- **Kind:** design
- **Severity:** medium
- **Where:** `docs/ARCHITECTURE.md:408-433`
- **Evidence:** the branch edited this table (it added the `lr_scheduler` row and rewrote the epochs row) but
  added no row for any of the six keys it introduced: `EPOCHS_DECODE`, `LR_DECODE`, `WEIGHT_DECAY_DECODE`,
  `LAMBDA_NUM`, `EVAL_MASK_RATE`, `EVAL_MASK_RATES_EXTRA`. Three surviving rows are actively wrong once
  `--task imputation` exists:
  - `mask_probability | PROB_MASCARA | "p_base passed to preprocess_table **during pretraining**"` — the
    decode stage re-rolls its training masks at this rate every epoch (`decoding.py:91-96`), and ADR 0005
    makes it the first knob the `reduced` profile samples (`opt.py:96-98`: "the decode stage re-rolls masks
    at this rate every epoch, so the decoder learns from exactly these cells").
  - `hidden_dimension | HIDDEN_DIM | "…inside TabularEmbedder"` — it also sizes every decoder numerical head
    (`models.py:142-144`, `Linear(dimensao, hidden_dim) → ReLU → Linear(hidden_dim, 1)`), which the same
    document's own decoder diagram draws.
  - `pretraining_epochs / finetuning_epochs | EPOCHS_PRE / EPOCH_FINE | "Epoch counts, and the horizon of
    each stage's StageScheduler"` — on an imputation run `finetuning_epochs` lands nowhere at all and
    `decode_epochs` is the horizon (`decoding.py:64-69`).

  `batch_size | BATCH | "Manual batch-slice size in both training loops"` is now three loops, and on the
  decode path it governs training only — every decode evaluation and scoring forward is un-batched
  (`decoding.py:114`, `:145-196`; known finding 2). Supporting, not load-bearing: four of the section's code
  anchors no longer point at what they name after this branch's edits — `types.py:9` is a comment
  (`Hyperparameters` is at `:90`), `models.py:88` is a dataclass field comment (the hardcoded `0.3`
  classifier dropout moved), `finetuning.py:92` is `present_classes` (`number_of_labels` is `:91`), and
  `runner.py:29` is inside `_per_column_scores`.
- **Consequence:** ARCHITECTURE's closing paragraph tells readers this table and README are "the single
  source of truth" for how a hyperparameter reaches code. For half the branch's surface it answers nothing,
  and where it does answer it misleads: an operator reading it would not expect `PROB_MASCARA` — the knob
  ADR 0005's reduced search exists to tune — to change what the decoder is trained on, nor would they know
  that `EPOCH_FINE` in a promoted imputation file is inert. This is the same drift F-10-6 found in
  `CLAUDE.md`: the branch updated the schedule axis of every reference document and skipped the task axis.
- **Direction:** add the six decode rows, qualify the three stale ones by task, and re-anchor the four
  citations. The table is the one place in the repo where "which knob reaches which line" is supposed to be
  answerable in one read.

### F-11-4 - `--lr_scheduler cosine` is prescribed in two documents, enforced in none, and the only pinned decode artifact is frozen under the schedule the documents warn against

- **Kind:** methodology
- **Severity:** medium
- **Where:** `README.md:306`, `docs/adr/0004-imputation-decoder-task.md:227-228`,
  `src/training/config.py:222-229`, `tests/fixtures/credit-g_20nan_imputation_regression.json`
- **Evidence:** both documents give the same instruction — "Because the decode stage has no published runs to
  preserve and the default schedule is the legacy per-batch cosine, launch imputation experiments with
  `--lr_scheduler cosine`" (ADR 0004:227-228), "prefer `--lr_scheduler cosine` over the legacy default"
  (README:306). Nothing enforces it. `--lr_scheduler` parses with `default=None` (`config.py:222-229`) and the
  effective default is `Hyperparameters.lr_scheduler = DEFAULT_LR_SCHEDULER = "cosine_legacy"`
  (`types.py:107`, `types.py:17`); `config.py:32-34` only overrides when the flag is present. There is no
  task-aware default anywhere. Reachability, correcting the brief I was given: the flag *is* passed by both
  launchers — `imputation_studies.ps1:63,84` **and** `experiment_imputation.ps1:62` — so the prescription is
  reachable; what is unreachable is any default that honours it. The gaps are the ones a reader copies:
  README's own second example, two lines below the prescription, is
  `uv run main.py --dataset_name credit-g_20nan --task imputation --cv_folds 3 --score_null_path` with no
  schedule flag (README:303-304), and the imputation regression fixture pins `cosine_legacy` — the test
  builds `Hyperparameters.from_mapping(fixture["hyperparameters"])`
  (`test_credit_g_imputation_regression.py:38`) and the fixture's hyperparameter block carries no
  `LR_SCHEDULER` key at all (`grep -n LR_SCHEDULER tests/fixtures/credit-g_20nan_imputation_regression.json`
  returns nothing), so `from_mapping` resolves it to `cosine_legacy`. The store shows what people actually
  ran: cross-tabbing `task` against `lr_scheduler` over all 450 runs gives imputation = 192 `cosine` +
  8 `plateau` and **zero** `cosine_legacy`, against 191 `cosine_legacy` classification runs. Meanwhile
  `schedulers.py:78-81` builds `cosine_legacy` as `CosineAnnealingLR(T_max=epochs)` stepped once per batch
  (`decoding.py:107`), whose period is `2 × T_max` steps. Simulating the real scheduler over the shipped
  `EPOCHS_DECODE = 150` (`uv run --python 3.10 python -c "…CosineAnnealingLR(T_max=150)…"`, counting returns
  to the base rate): credit-g at the default `BATCH=256` and 3 folds runs 450 steps and returns to base once,
  ending at lr 0; the fixture's `BATCH=64` gives 1500 steps and four returns, ending at the *base* rate;
  electricity at `BATCH=256` gives 16,050 steps and 53 returns. So the decode stage inherits BACKLOG B1's
  mechanism in full — multiple complete cosine cycles, a per-dataset number of them, and a final learning
  rate that depends on where the batch count lands in the cycle — on a stage that has no legacy results to
  protect.
- **Consequence:** AGENTS.md calls `credit-g_20nan` a reviewed regression baseline for the imputation task,
  so the one artifact that freezes decode behaviour freezes it under a schedule no experiment in the store
  has ever used and both documents tell operators to avoid. Any future decode change is defended by a
  two-epoch run of a configuration nobody runs. Separately, the default's documented rationale is
  vacuous for this task: the flag's own help text says "Default: cosine_legacy (the schedule of every run
  before ADR 0003)" (`config.py:229`), and for the decode stage that population is empty — the reason given
  for the default does not apply to the task the default is being inherited by.
- **Direction:** two independent moves, both small. Make the default task-aware (`imputation` resolves to
  `cosine`), which AGENTS.md permits because it cannot move classification — the classification default stays
  `cosine_legacy` bit-for-bit; and regenerate the imputation fixture (which also pins `EPOCHS_DECODE: 2`, so
  it exercises a two-epoch corner of that schedule) under the schedule imputation actually
  runs, recording the change as the intentional behaviour change AGENTS.md requires. At minimum, put the flag
  in README's second example so the copied command matches the sentence under it.

### F-11-5 - CONTEXT defines the `full` search-space profile as sampling everything; it holds the evaluation rate and the schedule, by design

- **Kind:** design
- **Severity:** medium
- **Where:** `CONTEXT.md:153-158`, `opt.py:110-150`
- **Evidence:** "`full` samples every hyperparameter the task uses; `reduced` samples only the ones that
  govern the task's own stage and the corruption it learns from." `define_search_space`'s `full` branch
  (`opt.py:110-150`) samples `HEADS`, `HEAD_DIM`→`DIM`, `HIDDEN_DIM`, `LAYERS`, `DIM_FEED`, `DROPOUT`,
  `EPOCHS_PRE`, `BATCH`, `LR_PRE`, `WEIGHT_DECAY_PRE`, `PROB_MASCARA`, plus `EPOCHS_DECODE`, `LR_DECODE`,
  `WEIGHT_DECAY_DECODE` and — only on a mixed table — `LAMBDA_NUM`. It never samples `EVAL_MASK_RATE`; the
  code says why, and says it well: "the evaluation rate is deliberately absent, since a trial free to hide
  fewer cells would win by making its own exam easier" (`opt.py:128-129`). It never samples `LR_SCHEDULER`
  either, though ADR 0003 makes it a first-class hyperparameter with a JSON key and a tag, nor
  `EVAL_MASK_RATES_EXTRA`. By CONTEXT's own next definition ("Held hyperparameter: a hyperparameter a
  search-space profile does not sample"), `full` therefore has held hyperparameters — which the definition of
  `full` denies. `opt.py:87-88`'s docstring repeats the overclaim; the comment at `:128` is the honest one.
- **Consequence:** the whole point of ADR 0005 is a decision about what to *stop* sampling, so the vocabulary
  for "what a profile samples" has to be exact. As written, the contrast reads "5 knobs versus everything",
  when on a mixed table it is 5 versus 15 (4 versus 14 on a single-kind one) with `EVAL_MASK_RATE` and
  `LR_SCHEDULER` permanently held in both — and those two permanently held ones are the
  interesting ones: the evaluation rate (held for a protocol reason that deserves to be in the definition,
  not only in a code comment) and the schedule (held while F-11-4 shows the default is wrong for this task).
  An operator auditing "did the reduction lose anything?" against `optuna/importance/<knob>` will look for an
  `EVAL_MASK_RATE` importance that can never exist — and F-07-5 already shows `HEAD_DIM` silently missing
  from that ranking.
- **Direction:** define `full` as "every knob the task's search samples" and name the permanently-held set in
  the definition, with the exam-difficulty reason for `EVAL_MASK_RATE` moved up from the code comment.

### F-11-6 - The stage-timing contract is false for the imputation task: the window is training plus the entire scoring pass, and it moves with flags that change no training

- **Kind:** methodology
- **Severity:** low
- **Where:** `CONTEXT.md:59-68`, `README.md:379`, `src/training/runner.py:78-108`
- **Evidence:** CONTEXT defines stage timing as "the wall-clock seconds one fold spent in pre-training or in
  the task's second stage (fine-tuning or the decode stage) … It excludes data preparation, plotting, and
  artifact writes", and README:379 says the keys "cover the two training stages only, not data preparation,
  plotting, or artifact writes". The clock is started at `runner.py:78` and read at `:102`, around the whole
  `train_and_evaluate_decoder` call, and the runner's own comment claims "the timing covers training only"
  (`:103-105`). It does not. Inside that window, after the training loop ends at `decoding.py:131`, the stage
  still: builds the naive baselines (`:134-136`), encodes and scores the masked test population (`:137-157`),
  encodes and scores one more full test population per entry of `EVAL_MASK_RATES_EXTRA` (`:162-172`), loads
  the sibling slice, rescales it and scores the induced population (`:174-182`), optionally scores the
  null-token population (`:183-196`), optionally scores the validation objective (`:202-211`), and
  constructs the per-cell ledger DataFrame row by row inside `_score_population` (`:285-330`) including a
  pandas `.at[]` lookup per numerical cell (`_exact`, `:333-341`). Classification's equivalent window holds
  one test forward and some sklearn.
  I did **not** measure the share. Structurally it is small — three to six full-split forwards against
  `EPOCHS_DECODE × batches_per_epoch` training steps — but it is not zero, it grows with dataset size on
  the un-batched eval path (known finding 2), and crucially it changes when `--score_null_path` or
  `EVAL_MASK_RATES_EXTRA` change, neither of which changes a single training step.
- **Consequence:** README:379 invites exactly the comparison this breaks — "one column sorts every parent by
  training cost". For an imputation parent that column is training plus evaluation plus ledger construction,
  under a key named after fine-tuning, and two imputation runs of the same configuration differing only in
  `--score_null_path` will not agree. This is the documentation half of known finding 9, which I am not
  re-reporting: the code bug is that no decode key exists; my point is that this branch shipped a normative
  definition (CONTEXT:59-68) describing the decode stage's seconds as a tracked, comparable quantity, while
  the same branch's own BACKLOG H5 records that the key does not exist and that the number lands under
  `time/finetune_seconds`. Two documents from one branch, one presuming what the other records as open.
- **Direction:** either move the scoring pass outside the clock (it is a separate concern from the stage's
  cost and would make both documents true), or describe the window truthfully — "the stage call: training
  plus the stage's own scoring pass" — and have CONTEXT:59-68 point at BACKLOG H5 so a reader looking for a
  decode key learns in one hop why there isn't one.

### F-11-7 - "On the test fold only" appears in two documents while a trial scores the validation fold and `--score_null_path` adds a fourth population

- **Kind:** design
- **Severity:** low
- **Where:** `CONTEXT.md:116-123`, `README.md:310`
- **Evidence:** CONTEXT's "Imputation ground truth" says "Imputation error compares the decoder's
  reconstruction against it **on the test fold only**", and README's "What gets scored" opens at :310 with
  "**Two populations, on the test fold only**". Both are contradicted by code this branch added. `decoding.py:202-211`
  scores a third population — self-masked cells on the *validation* frame — and logs it as
  `validation/impute/masked/*` whenever `score_search_objective` is set, which is every Optuna trial
  (1075 `validation/impute/%` metric rows in the store). `decoding.py:183-196` scores a fourth,
  `induced_null_token`, on the test fold under `--score_null_path`, a flag README documents at :165 but omits
  from "What gets scored". CONTEXT:144-151 then asserts the reverse of :118-119 — that the search's score
  lives on the validation split. Both sentences are new on this branch; they cannot both be right.
- **Consequence:** there is no sentence anywhere in the doc set that correctly states which populations a
  given invocation scores and on which split. The reader of an artifact meets the consequences directly: the
  cell ledger of a `--score_null_path` run carries a population the "two populations" section never names
  (and carries induced cells twice — F-06-3), and a trial run carries a metric family whose split the
  ground-truth definition says is impossible.
- **Direction:** replace the prose with a small table — population × split × the condition that produces it
  (`masked`: test, always; `masked` at each extra rate: test, when `EVAL_MASK_RATES_EXTRA` is set;
  `induced`: test, when the sibling exists; `induced_null_token`: test, under `--score_null_path`;
  `validation/masked`: validation, under a search) — and delete "on the test fold only" from the ground-truth
  definition, which is about *whose value is the truth*, not about which split.

### F-11-8 - README certifies the composite's degenerate behaviour "on the six all-numerical datasets and on all-categorical kr-vs-kp"; the evidence is one synthetic unit test and one of those seven datasets has ever run

- **Kind:** methodology
- **Severity:** low
- **Where:** `README.md:321`
- **Evidence:** the sentence — "It degrades correctly on the six all-numerical datasets and on all-categorical
  `kr-vs-kp`" — reads as a corpus-level verification. Structurally the claim is true and I am not disputing
  it: `score_cells` weights each kind by its share of scored cells (`imputation_metrics.py:44-56`), so a
  single-kind table reduces to that kind's ratio and `_ratio(1.0 - acc_cat, ...)` never fires on an
  all-numerical table. The arithmetic also checks out: three of the nine base datasets have a
  `datasets/categorical_columns/*.txt` file (credit-g, electricity, kr-vs-kp), leaving six all-numerical.
  What does not exist is the measurement. The only test of the property is
  `tests/unit/test_imputation_metrics.py:100`, whose docstring is literally "Six of the nine datasets are all
  numerical and one is all categorical" over a hand-built frame. In the store, grouping every run carrying an
  `impute_score` by dataset gives `credit-g_20nan`, `credit-g_40nan`, `kr-vs-kp_20nan`, `kr-vs-kp_40nan`,
  `spambase_20nan` — one of the six all-numerical datasets, none of the other five, no `_00nan` variant, and
  nothing on `electricity`.
- **Consequence:** a reader takes "degrades correctly on the six all-numerical datasets" as "this was run
  there and behaved", and will not re-check before publishing a table that spans the corpus. The sentence
  also quietly vouches for the easy half: the two tables where the composite genuinely mixes two
  incommensurable ratios (credit-g, with 13 categorical and 7 numerical columns, so `w_cat ≈ 0.65`; and
  electricity, with 1 categorical column of 8, so `w_cat ≈ 0.12`) are the ones it does not mention, and the
  same metric name therefore means a differently-weighted quantity on each — which README states as a
  property two lines earlier but does not connect to this claim.
- **Direction:** say what is true and where it comes from: "a single-kind table reduces to that kind's ratio
  by construction (unit-tested); the mixed tables weight the two ratios by scored-cell share, so
  `impute_score` is not the same quantity across datasets". Name datasets only where a run exists.

### F-11-9 - The run taxonomy has no term for 12 runs in the store, and the documented filter recipe reaches none of them

- **Kind:** methodology
- **Severity:** low
- **Where:** `CONTEXT.md:33-41`, `docs/adr/0004-imputation-decoder-task.md:218-224`
- **Evidence:** CONTEXT defines exactly four kinds of run — parent (implied by "Single-split run" and
  "CV summary"), "Diagnostic fold run" ("**one of the two** nested MLflow runs retained from a
  cross-validation execution"), "Optuna trial run", and the study run — and ADR 0004:218-224 plus CLAUDE.md
  tell every reader to enumerate comparable runs with `tags.run_role = 'parent'`. A census of all 450 runs
  gives `parent 93, optuna_trial 184, best_fold 78, worst_fold 78, optuna_study 5` and **12 runs with no
  `run_role` tag at all**: three top-level `train_vehicle_00nan_*` runs from 2026-07-03 and 2026-08-03, each
  holding zero metrics, plus their nine nested `fold_N_of_3` children, each holding 248 metric rows
  (`pretrain/*`, `finetune/*`, `test/f1_macro`, …). The backfill was partial by tag: `task`, `lr_scheduler`
  and `is_optuna` are present on 450/450 runs (and these twelve carry `task_backfilled = true`,
  `lr_scheduler_backfilled = true`), `run_role` on 438/450, `run_type` on only 285/450.
  `scripts/backfill_run_tags.py`'s own docstring states the problem it exists to solve — "A tag that only new
  runs carry cannot be filtered on: a query excluding one value also excludes every run that predates the
  tag, silently" — and `run_role` is the tag it was not run for.
- **Consequence:** three complete 3-fold `vehicle_00nan` cross-validation executions answer
  `tags.task = 'classification'` and `tags.lr_scheduler = 'cosine_legacy'` — the two axes ADR 0003 and
  ADR 0004 tell a reader to compare on — while being unreachable through `tags.run_role` in any value. The
  `cosine_legacy` baseline population that the schedule sweep is judged against is therefore assembled the
  documented way and silently three executions short (compounding known finding 8, which already says the
  sweep is not comparable to its baselines). Their fold children, which hold the actual numbers, match no
  term in CONTEXT: they are nested fold runs that are neither `best_fold` nor `worst_fold`, a shape
  CONTEXT:34 says a CV execution does not produce.
- **Direction:** either run the backfill for `run_role` with a term for the pre-ADR-0002 shape (parent, and
  something like `legacy_fold` for the nine children, so `run_role != 'legacy_fold'` remains a usable
  filter), or state plainly in CONTEXT and ADR 0004 that the recipe covers runs from ADR 0002 onward and that
  12 earlier runs must be queried by name.

## Checked and cleared

- **CONTEXT.md:8-14, "Fold-ranking metric" — the direction really does travel with the metric.**
  `_TASK_SPECS` (`types.py:64-73`) pairs `impute/masked/impute_score` with `"minimize"`, and
  `_diagnostic_roles` (`summary.py:221-235`) flips the sort with `sign = 1.0 if task.direction == "maximize"
  else -1.0` for both roles. Imputation's `best_fold` is the lowest `impute_score`, not the highest. The
  definition's claim that "the selection inverts without the direction" is exactly what the code implements.
- **CONTEXT.md:39-40, `best_and_worst`.** Real, not aspirational: `summary.py:233-234` returns
  `{fold: "best_and_worst"}` when the two roles land on the same fold. (No run in the store has ever taken
  that branch — 78 `best_fold` and 78 `worst_fold`, no `best_and_worst` — but the definition describes code
  that exists.)
- **CONTEXT.md:131-136, "Decode stage … mirrors classifier fine-tuning in schedule, checkpoint selection and
  stage timing, and never re-initialises the pretrained encoder."** Checkpoint selection matches:
  `finetuning.py:118,174-181` tracks the best validation loss and restores it, `decoding.py:86-130` does the
  same on a mask drawn once. Schedule matches: both build a `StageScheduler` over their own epoch count.
  "Never re-initialises" holds literally — there is no `apply(` anywhere in `models.py` or `decoding.py`;
  `TridentDecoder.__init__` (`models.py:106-148`) stores the embedder and transformer by reference and only
  constructs its own heads. (The "stage timing" clause is the one that does not survive — F-11-6.)
- **ARCHITECTURE's "1400 of them on `credit-g_20nan`".** Verified against the CSV: 7 numerical columns,
  1400 NaN among them, 4000 NaN in the table — so the figure and the argument for encoding targets through
  `preprocess_table(..., fine_tunning=True)` are both right.
- **CONTEXT.md:157, "Chosen per study and recorded on the study and its trials."** True in the store:
  `search_space` is present on exactly 189 runs, which are the 5 `optuna_study` and 184 `optuna_trial` runs
  and nothing else. (It only ever holds `reduced`, but that is F-08-2's hardcoded launcher, not this
  definition.)
- **CONTEXT.md:166-171, "a promoted imputation configuration never changes what a classification run reads."**
  True as written: `promote_best_configuration` (`opt.py:328-347`) writes through `hyperparameter_file(dataset,
  task)`, which is the `.imputation.json` name for imputation, and README's resolution order reads the
  task-keyed file only under `--task imputation`. The converse — a promoted *classification* file being read
  by an imputation run — is real and is F-07-4's; the definition does not claim otherwise.
- **BACKLOG.md's header claims.** "Nothing in this file has been fixed" still holds for the two open items
  this branch could plausibly have closed while rewriting 415 lines of `opt.py` and the training package:
  H1's `create_pretrain_datasets` is still at `src/utils.py:135`, and H4's rebinding of the module-level
  `mlflow` name is still at `opt.py:381`. C2's harm argument ("masking never overwrites an already-null
  cell, so those positions can never be selected by the reconstruction loss") is contradicted in letter by
  `decoding.py:259-262`, where `--score_null_path` points the selection tensor straight at the null cells —
  but the claim's substance survives, because that path only scores and its targets come from the complete
  sibling, so no `"nan"` id ever reaches a loss.
- **The backfill itself, for the axes it targeted.** `task` is on 450/450 runs (251 classification, 199
  imputation) and `lr_scheduler` on 450/450, so ADR 0004's "or rely on the backfilled value" is sound for
  those two tags. Only `run_role`/`run_type` were left partial (F-11-9).

## Open questions

- **What share of an imputation fold's `time/finetune_seconds` is scoring rather than training?** I argued it
  structurally and declined to run training for it. Settling it costs one instrumented fold: time
  `train_and_evaluate_decoder` up to `decoding.py:131` and from `:133` to the return, on `credit-g_20nan`
  with and without `--score_null_path` and with two extra rates. If the scoring half is more than a few
  percent, F-11-6 moves from a wording defect to a measurement defect.
- **Was the "untouched test split" wording written before the unconditional test scoring, or in spite of it?**
  ADR 0005's Context section states the true behaviour, so the ADR author knew. If the intent was to gate the
  scoring and the gate was never added, F-11-1's fix is the code; if the intent was always to keep the test
  report for post-hoc analysis, it is the definition. The ticket history under
  `docs/wayfinder/imputation-optuna-reduced/issues/02-search-objective-on-validation.md` would say which.
- **Is anything downstream reading the promoted `EPOCH_FINE` value out of an imputation configuration file?**
  `complete_configuration` omits fine-tuning keys for imputation, but a hand-written or sweep-written shared
  file carries them, and the wiring table (F-11-3) still presents `EPOCH_FINE` as the second stage's epoch
  count. A grep of the shipped `datasets/hiperparams/` tree against each task's consumed key set would show
  whether any file in the corpus is currently misread.
