# 12 - Reproducibility and provenance of an imputation run end to end

_Critique of the imputation work on branch feat/imputation-task. 2026-09-11._

**Scope:** `src/training/artifacts.py:27-207,240-290`, `src/training/environment.py:1-21`,
`src/training/runner.py:35-192`, `src/training/tracking.py:99-241,325-425`,
`src/training/data.py:45-149`, `src/training/config.py:21-137`, `src/utils.py:8-82`,
`src/training/decoding.py:32-146,227-341`, `src/training/pretraining.py:64-121`,
`src/training/finetuning.py:63-90`, `opt.py:81-150,304-346,495-582`,
`docs/adr/0005-reduced-optuna-search-for-imputation.md:140-195`,
`docs/adr/0004-imputation-decoder-task.md:212-232`, `README.md:305-332`,
`tests/integration/test_credit_g_imputation_regression.py`,
`tests/fixtures/credit-g_20nan_imputation_regression.json`,
plus the live store `mlflow.db` (read-only) and the published tree `results/credit-g_20nan/20260910_195312/`.

**Method:** `graphify query` to orient, then the source. Branch-vs-main separation with
`git diff main...HEAD` / `git show main:<path>` so an inherited gap is not reported as
branch-new. Read-only SQL against `mlflow.db` (`file:...?mode=ro`) for the 450 runs, the
6 imputation parents and the 45 dataset-input rows. Four `uv run --python 3.10` experiments:
(1) MLflow's `compute_pandas_digest` against the four `kr-vs-kp` variants;
(2) `_assert_row_aligned` against a re-drawn `credit-g_20nan`;
(3) global-numpy-stream drift across epoch budgets and mask rates, plus a call counter on
the floor loop;
(4) `StandardScaler` fit across the missingness ladder.
I did not run a training run — nothing below needs one — and did not run pytest.
I could not check whether the six decision-6 comparison runs exist, because they do not:
the store holds 6 imputation parents, of which 4 FAILED and 2 predate `config_source`.

## Findings

### F-12-1 - The `_XXnan` missingness draw, which *is* the induced benchmark, is recorded nowhere and no check in the pipeline can detect its substitution

- **Kind:** methodology
- **Severity:** high
- **Where:** `src/training/artifacts.py:177-207`, `src/training/data.py:92-129`, `.gitignore:19`
- **Evidence:**

  `write_tracking_provenance` writes exactly this payload (verified against the real file
  `results/credit-g_20nan/20260910_195312/data/provenance.json`): `source_path`,
  `splits_path`, `dataset_name`, `prepared_schema` (label column, the two column lists,
  `label_classes`), `preparation` = `{"label_encoding": "LabelEncoder",
  "numerical_scaling": "StandardScaler"}`, `split_strategy`, `cv_folds`, `seed`. No hash,
  no size, no mtime, no row count, no generator seed. The payload is unchanged from `main`
  (`git diff main...HEAD -- src/training/artifacts.py` contains no hunk touching it).

  The branch is what made a *second* file load-bearing. `load_complete_sibling`
  (`data.py:92-109`) is branch-new and resolves `<base>_00nan.csv` by naming convention.
  The complete sibling supplies the truth of every `impute/induced/*` number — the
  headline that ADR 0004:219 and ADR 0005:163 both name — and it appears in **no**
  record: not in `provenance.json`, not as an MLflow dataset input, not as a param or
  tag. I confirmed against the store: the successful `credit-g_20nan` parent
  (`1873c2d756ee…`) has exactly one `inputs` row, for `credit-g_20nan`; `credit-g_00nan`
  is absent from that run's inputs.

  The data is outside version control. `git check-ignore -v
  datasets/processed_datasets/credit-g/credit-g_20nan.csv` →
  `.gitignore:19:datasets/processed_datasets/`, and `git ls-files
  datasets/processed_datasets | wc -l` → `0`. So MLflow's auto-recorded
  `mlflow.source.git.commit` (present on all 450 runs) pins the *code* and nothing about
  the data.

  `_assert_row_aligned` (`data.py:112-129`) certifies that two files agree on observed
  cells. It cannot see a re-drawn variant, because a re-draw from the same complete table
  leaves every surviving cell equal to the sibling's:

  ```
  $ uv run --python 3.10 python .../redraw.py
  original variant  nan: 4000
  re-drawn  variant nan: 4007
  cells missing in BOTH: 769 of 4000
  _assert_row_aligned on re-drawn variant: PASSED
  shape equal: True | columns equal: True
  ```

  81% of the induced population changed identity and every check passed. `provenance.json`
  would be byte-identical.

  The one content identity anywhere in the record is MLflow's dataset digest, and on an
  all-categorical table it is blind to missingness. `mlflow.data.digest_utils.compute_pandas_digest`
  (mlflow 3.14.0) keeps only columns where **every** value is `str` plus numeric-dtype
  columns; a categorical column containing NaN is neither, so it is dropped before hashing:

  ```
  kr-vs-kp_20nan shape (3196, 37) nan 23004 digest 0d37f6f0
  kr-vs-kp_40nan shape (3196, 37) nan 46008 digest 0d37f6f0
  kr-vs-kp_80nan shape (3196, 37) nan 92052 digest 0d37f6f0
  credit-g_20nan ... digest 85f29a2f
  credit-g_40nan ... digest 988fde43
  ```

  The live store shows the same collision on the *prepared* frames:
  `kr-vs-kp_20nan`, `_40nan`, `_60nan`, `_80nan` all carry digest `3f2c2e1d`.
  `kr-vs-kp` is one of the six pairs that has a promoted imputation configuration
  (`datasets/hiperparams/kr-vs-kp/kr-vs-kp_20nan.imputation.json`,
  `kr-vs-kp_40nan.imputation.json`), and ADR 0005:177 records that kr-vs-kp already scores
  0.72 masked at the defaults.

  One more thing the record loses: `provenance.json` names the *class* `StandardScaler`,
  never the fitted `mean_`/`scale_`. The scaler is fit on the variant, so the z-space that
  `actual`, `imputed` and `rmse_num_z` live in differs per variant —
  `credit-g` `credit_amount`: `scale_` = 2821.33 (`_00nan`), 2826.48 (`_20nan`),
  2697.03 (`_40nan`) — and nothing in the record lets that space be rebuilt without
  re-reading the exact CSV.

- **Consequence:** Every `impute/induced/*` number — `cv/test/impute/induced/impute_score/mean`,
  the number ADR 0005 decision 6 uses to decide whether the tuning helped — is a score
  against one unrecorded draw of a random generator over an ungoverned file. If
  `datasets/processed_datasets/credit-g/credit-g_20nan.csv` is regenerated on another
  machine, or between the defaults arm and the tuned arm, or between today and the
  re-run in six months, the two numbers are measured on different cell populations and
  nothing — not the alignment check, not the provenance file, not the git commit, not the
  MLflow digest on an all-categorical table — will say so. The published committed fixture
  `credit-g_20nan_imputation_regression.json` has the same exposure: it pins six metrics
  per fold to `abs=0.01` against two files that are in no commit and identified by no hash,
  so a fixture failure cannot be attributed to code or to data.
- **Direction:** Put a content identity of both files in the record the run itself writes.
  A SHA-256 of `source_path` and of the `_00nan` sibling, their byte sizes and row counts,
  in `provenance.json` and as MLflow params, and the same two hashes in the regression
  fixture's `environment` block. Extend `provenance.json` with a `ground_truth` section
  naming the sibling path explicitly rather than leaving it implied by convention. If the
  variant generator has a seed, record it. Separately, `_assert_row_aligned`'s docstring
  claims more than the check delivers ("Regenerating one alone would break that") — it
  detects a regenerated *sibling*, never a regenerated *variant*; a hash comparison is what
  closes that gap.

### F-12-2 - Training corruption is drawn from one global numpy stream whose position at fold *k* is a function of every earlier fold's draws, so ADR 0005 decision 6's "the configuration is the only difference" is false from fold 2 and no single fold is reproducible on its own

- **Kind:** methodology
- **Severity:** high
- **Where:** `src/utils.py:58,70-74`, `src/training/pretraining.py:70-75`, `src/training/decoding.py:91-96`, `src/training/runner.py:37,67`
- **Evidence:**

  `run_training` seeds once, at `runner.py:37`, and then runs every fold inside one process
  (`runner.py:67`). `preprocess_table` draws `np.random.rand(*data.shape)` (`utils.py:58`)
  and then an unbounded number of `np.random.choice` calls in the floor loop
  (`utils.py:70-74`) — one per row the rate happened to leave unmasked. That count is
  data-dependent, not a function of the hyperparameters:

  ```
  np.random.choice calls per epoch, p=0.5: [0, 2, 0, 0, 0]
  np.random.choice calls per epoch, p=0.2: [23, 40, 42, 32, 35]
  ```

  Pre-training consumes two of these per epoch (`pretraining.py:70-75`, train and
  validation frames); the decode stage consumes one (`decoding.py:91-96`). So the stream
  offset entering fold 2 is a function of `EPOCHS_PRE`, `EPOCHS_DECODE`, `PROB_MASCARA`,
  the fold's row count *and the realised draws*. Measured, at seed 42, on the first draws
  fold 2 would take:

  ```
  EPOCHS_PRE=10, PROB_MASCARA=0.5 -> [0.19512634 0.33764692 0.89533745 0.69032892]
  EPOCHS_PRE=10, PROB_MASCARA=0.4 -> [0.15674464 0.08003204 0.71421234 0.45785192]
  EPOCHS_PRE=11, PROB_MASCARA=0.5 -> [0.13349282 0.00554774 0.84324513 0.80343530]
  ```

  Draw the boundary for decision 6 ("one 5-fold cross-validation run with the promoted
  configuration against one with the defaults, both `--task imputation --cv_folds 5
  --lr_scheduler cosine --seed 42`, **so the configuration is the only difference**", ADR
  0005:151-165, called "the glossary's *paired single-seed CV protocol*"):

  *Genuinely common between the two arms.* The fold partition — `build_folds`
  (`runner.py:46`) runs before any training and uses `random_state=seed` throughout, so
  both arms get the identical `train/validation/test_indices` for all five folds. The
  evaluation masks — `evaluation_mask` (`data.py:143-148`) reseeds on
  `(seed * 1_000_003 + fold)` and restores the global state, so the test mask, the
  validation mask and the extra-rate masks are identical per fold in both arms. The
  induced-missing population — deterministic from the CSV. Torch weight init in *all*
  folds: the reduced profile samples only `PROB_MASCARA`, `LR_DECODE`,
  `WEIGHT_DECAY_DECODE`, `DROPOUT`, `LAMBDA_NUM` (`opt.py:81-101`), so architecture,
  `BATCH` and both epoch budgets are held; `torch.randperm` call counts match, `DROPOUT`
  changes the Bernoulli *p* and not the draw count, so the torch generator stays in step.

  *Not common.* The training corruption, from epoch 1 of fold 1 — because
  `PROB_MASCARA` is itself a sampled knob, and the promoted `credit-g_20nan.imputation.json`
  carries `PROB_MASCARA: 0.4` against the code default of 0.5 (`types.py:103`). From
  fold 2 onward the two arms also enter each fold at an *arbitrary, unseeded* stream
  offset determined by the earlier folds' realised draws.

  The known crossings, verified:
  1. **No fold is reproducible alone.** There is no mechanism to re-enter fold 3's
     training-RNG state; only folds 1 and 2 replayed at the identical configuration put it
     there. The evaluation side is per-fold reproducible; the training side is not.
  2. **Imputation and classification diverge from fold 2 at the same seed.** Classification's
     fine-tuning calls `preprocess_table(..., fine_tunning=True)` (`finetuning.py:81-89`),
     which returns before any draw — measured, the MT19937 position is unchanged
     (`624 -> 624`). The decode stage consumes `EPOCHS_DECODE` frames per fold. So the two
     tasks' pre-training sees identical corruption in fold 1 and different corruption in
     every fold after it.
  3. **Changing an epoch budget is not a truncation.** A `EPOCHS_PRE: 200` run is not the
     `EPOCHS_PRE: 300` run stopped early; from fold 2 it is a different experiment. The
     same applies to the `--cv_folds 2` regression fixture against a `--cv_folds 5`
     production run: different splits *and* different per-fold row counts, so the fixture
     exercises none of the stream arithmetic the production shape depends on.

- **Consequence:** For the six decision-6 pairs the *question* per fold is identical
  (same split, same evaluation mask, same induced cells — the 08 verifier's "common random
  numbers" credit is right about the search objective, which is scored on the isolated
  validation mask). What is not identical is the training randomness, and at one seed
  there is no way to separate "the promoted configuration is better" from "the tuned arm
  drew luckier corruption in folds 2-5". The `ci95_lower`/`ci95_upper` that ADR 0005 asks
  the reader to compare is a t-interval over five folds whose only within-arm variation is
  the split plus this uncontrolled offset; it is not an uncertainty over the thing being
  compared. The gap also means an author who wants to re-inspect fold 4's ledger — folds
  other than `best_fold`/`worst_fold` upload nothing to MLflow (`tracking.py:214-234`) —
  must re-run the entire 5-fold job at exactly the same budget, and cannot shorten it.
- **Direction:** Give training corruption the treatment `evaluation_mask` already has: a
  stream derived per `(seed, fold, stage, epoch)` rather than the ambient global one, so a
  fold is a closed experiment and two arms differ only where a knob differs. That is a
  behaviour change and belongs behind a flag plus an MLflow tag under the project's own
  rule. Until it exists, decision 6's comparison needs more than one seed, and the ADR's
  "the configuration is the only difference" should be narrowed to "the configuration and
  the fold-entry RNG offset".

### F-12-3 - `config_source` is a mutable path into an untracked directory; the commit the run records does not contain the file, and re-running that commit silently trains the defaults instead

- **Kind:** design
- **Severity:** medium
- **Where:** `src/training/config.py:38-69`, `opt.py:328-346`
- **Evidence:**

  `promote_best_configuration` writes to `hyperparameter_file(dataset_name, task)` — a
  fixed path per `(dataset, task)` with no version, no study id and no timestamp
  (`opt.py:342-345`). `config_source` is that path string (`config.py:56`).
  `datasets/hiperparams/` is untracked and *not* gitignored: `git ls-files
  datasets/hiperparams` → empty; `git check-ignore` → exit 1; `git status --porcelain` →
  `?? datasets/hiperparams/`. So the entire output of the ADR 0005 search effort — the six
  promoted files that every "tuned arm" run reads — exists only in one working tree.

  The failure is silent, not loud. `_load_base_hyperparameters` (`config.py:49-58`) walks
  its candidate list and, when no candidate exists, returns `Hyperparameters(), "defaults"`
  with no warning. Check out the commit a run recorded, re-run the same command line, and
  the run trains the code defaults while logging `config_source='defaults'` — a *different*
  configuration reported as a legitimate one. The auto-tagged `mlflow.source.git.commit`
  makes this worse rather than better, because it invites exactly that re-run.

  The store already shows the dangling pointer. Three imputation parents record
  `config_source = datasets/hiperparams/electricity/electricity_40nan.json` or
  `…_20nan.json`; `datasets/hiperparams/electricity/` is now empty. (Those three runs
  FAILED, for the known dtype crash — I am not re-reporting that; what I am reporting is
  that the pointer survives the file.) Note also the path they name. Two of those three
  ran at `23c9548`, before `a24dcec` introduced the task-keyed lookup, so for them
  `<dataset>.json` was the only candidate; the third, at `a051258`, is the current
  semantics — an imputation run falling back to the *classification*-keyed file
  (`config.py:49-51`). Under that fallback the recorded `config_source` does not say which
  of the two key sets came from the file and which came from defaults: a classification
  file names no `EPOCHS_DECODE`, `LR_DECODE`, `LAMBDA_NUM` or `EVAL_MASK_RATE`, so an
  imputation run reading one trains its entire decode stage at the code defaults while
  logging a path that suggests a tuned configuration.

  What is *not* lost: `_log_execution_params` logs `complete_configuration` as params, so
  the values themselves are recoverable from any MLflow-tracked run (I read them back:
  the `electricity_40nan` parent carries all 17 keys). The loss is which study produced
  them, and re-runnability from the recorded commit. `--retrain_best` tags the study with
  `optuna_study_run_id` (`opt.py:562`); the decision-6 comparison runs, launched through
  `main.py` against the promoted file, carry no such link, so nothing ties a "tuned arm"
  number back to the study that chose its configuration.

  One sentence on the CWD-relative path, which is pre-existing and which I am dropping
  as a finding: `hyperparameter_file` returns a relative `Path` (`config.py:69`), so a run
  launched from anywhere but the repo root resolves nothing and logs `'defaults'`.
  `promote_best_configuration` writes through the same relative path, so a study launched
  elsewhere promotes into a directory nobody reads.

- **Direction:** Make the promoted file content-addressed in the record, not path-addressed:
  log a hash of the loaded file alongside `config_source`, and have `--promote_best` stamp
  the promoted JSON with the study's MLflow run id and write it under a versioned name with
  the canonical path as a pointer. Commit `datasets/hiperparams/` (it is small and it is a
  research result, not a build output). Make the fallback loud: when `--task imputation`
  resolves to the classification-keyed file, say so on stdout and in `config_source`
  (e.g. `…/electricity_40nan.json#classification-fallback`).

### F-12-4 - The branch-new environment record names torch and CUDA only, omitting exactly the libraries whose behaviour the imputation path depends on

- **Kind:** design
- **Severity:** medium
- **Where:** `src/training/environment.py:7-21`, `tests/fixtures/credit-g_20nan_imputation_regression.json:2-9`
- **Evidence:** `runtime_environment_tags` returns four keys: `device`, `gpu_name`,
  `torch_version`, `cuda_version`. Nothing else. `git show main:src/training/environment.py`
  is empty and `git show main:src/training/runner.py | grep environment` is empty — the
  whole mechanism is branch-new, so which descriptors get recorded is this branch's
  decision, not an inherited one. The committed fixture's `environment` block records
  `python`, `platform`, `torch`, `device`, `seed`, `cv_folds`, `runs` — again no numpy,
  pandas or scikit-learn.

  Those three are load-bearing for this task in a way they are not for classification:
  the categorical vocabulary is a `LabelEncoder` fit and the z-space is a `StandardScaler`
  fit (`data.py:58-71`), the fold partition is `StratifiedKFold`/`StratifiedShuffleSplit`
  (`data.py:187-220`), and the decode stage re-encodes the `_00nan` sibling through both of
  them before it can score a single induced cell (`decoding.py:246-255`). numpy is the
  least of the three risks: the legacy `RandomState` sequence is stable across versions by
  numpy's own compatibility policy, so the corruption stream is safe. scikit-learn carries
  no such policy for either the splitters or `StandardScaler`'s NaN handling, and both sit
  directly under every imputation number.
- **Consequence:** A fixture regression or a changed `impute/induced/*` number cannot be
  attributed. Two machines that both report `torch 2.5.1+cu121, cuda 12.1, device cuda`
  can disagree on the fold partition (a scikit-learn splitter change) or on the z-space
  that every `rmse_num_z` and every ledger `actual` lives in (a `StandardScaler`
  NaN-handling change) — and the record gives the reader no way to tell which, because it
  does not say what those versions were.
- **Direction:** Add `python_version`, `numpy_version`, `pandas_version` and
  `sklearn_version` to `runtime_environment_tags`, and to the fixture's `environment`
  block when it is next regenerated. They cost one tag each and they are the versions a
  reproduction attempt actually has to match.

### F-12-5 - A run's on-disk artifact tree and its MLflow run are joined only by a timestamp drawn twice, and both successful imputation parents in the store are off by one second

- **Kind:** bug
- **Severity:** low
- **Where:** `src/training/artifacts.py:29`, `src/training/tracking.py:120,137`
- **Evidence:** `ArtifactWriter.__init__` takes `datetime.now().strftime("%Y%m%d_%H%M%S")`
  and builds `results/<dataset>/<timestamp>/`; `MlflowTracker.parent_run` takes its *own*
  `datetime.now().strftime(...)` and builds `run_name = f"impute_{dataset}_{timestamp}"`.
  The second call happens after `setup_mlflow()` and `get_or_create_experiment()`
  (`tracking.py:114-115`) — SQLite round-trips against a 140 MB store — which is why the
  results directory carries the *earlier* stamp in both cases below. Nothing else links
  the two: the results directory
  contains no run id, and no param or tag on the run names the results path (grep over
  `tracking.py`/`runner.py` for `results_dir` finds only the `log_artifact` call at
  `runner.py:162`). Both successful imputation parents in the store are skewed:

  | results directory | MLflow run name |
  |---|---|
  | `results/credit-g_20nan/20260910_195312` | `impute_credit-g_20nan_20260910_195313` |
  | `results/spambase_20nan/20260910_200756` | `impute_spambase_20nan_20260910_200757` |

  This is inherited from `main` (`ArtifactWriter.__init__`'s timestamp is unchanged), but
  the imputation task gives it new weight: only the `best_fold`/`worst_fold` children
  replay their artifacts (`tracking.py:214-234`), so in a 5-fold run three folds' cell
  ledgers and previews exist *only* under `results/` — which is gitignored — behind a name
  that does not match the run.
- **Consequence:** Someone reading `cv/test/impute/induced/impute_score/mean` in MLflow
  and wanting fold 3's ledger has to guess the directory by ±1 second, and on a machine
  running several jobs the guess is ambiguous.
- **Direction:** Draw the timestamp once and pass it to both, or — better — log
  `str(artifacts.results_dir)` as a param on the parent run and write the MLflow run id
  into `provenance.json`, making the join explicit in both directions.

## The cell ledger's `row`, and what it costs

`_score_population` takes `torch.nonzero(selected)` over a frame built as
`features.iloc[fold.test_indices].reset_index(drop=True)` (`decoding.py:133`), so the `row`
written at `decoding.py:298` and `:313` is a position in *this fold's test block*, not a
dataset row. `fold.test_indices` is written into neither the ledger, the preview, nor
`provenance.json`, and `FoldSplit` never reaches `ArtifactWriter`.

Within one fold this costs nothing that matters. All three populations — `masked`,
`induced`, `induced_null_token` — are produced from the same `test_frame` with the same
positional basis, so `--score_null_path`'s two views of the same gap pair correctly on
`(row, column)`: `_score_induced_missing` scores exactly `test_frame.isna()` in both
calls (`decoding.py:243,262`), so every `induced` row has its `induced_null_token`
counterpart at the same key. The preview sampler draws over the same `row` space
(`artifacts.py:244`), so the `in_preview` flag is consistent.

Across folds and across runs it costs the joins the documentation invites. `row 42` in
`imputation_fold_2_cells.csv` and `row 42` in `imputation_fold_3_cells.csv` are different
dataset rows, so the five per-fold ledgers of a CV run cannot be concatenated into one
table of the dataset's cells, and a cell cannot be traced back to the source CSV to ask
what it actually was. The comparison README:321 and ADR 0004:222 both invite — the same
`impute_score` read across `_20nan`…`_80nan` — is a metric-level comparison only; the
underlying cells cannot be matched, so "the model got worse on the same hard columns"
cannot be checked against "the model got a different, harder set of cells". `fold.test_indices`
is regenerable in principle (it is deterministic from the frame, label column, `cv_folds`
and seed) but only by re-running `prepare_dataset` and `build_folds` under the same
scikit-learn — which F-12-4 says is unrecorded and F-12-1 says is against an unrecorded
file. Writing `dataset_row = int(fold.test_indices[row])` as one extra ledger column would
close it for a few bytes per row.

## Checked and cleared

- **Cross-validation *does* record dataset lineage.** The premise that the CV branch skips
  provenance is wrong. `write_cv_tracking_artifacts` calls
  `write_tracking_provenance(dataset, seed, cv_folds)` at `artifacts.py:107` as part of
  building `TrackingArtifactPaths`, and `_log_parent_artifacts` uploads it under `data/`
  (`tracking.py:209`). Confirmed on disk:
  `results/credit-g_20nan/20260910_195312/data/provenance.json` exists for a `cv_folds: 2`
  run. `log_prepared_dataset` also runs in CV mode, from `finalize_cross_validation`
  (`tracking.py:170`). A five-fold imputation run records exactly what a single-split run
  records; what it records is the problem (F-12-1), not where.
- **The git commit *is* recorded.** MLflow's own context providers tag every run:
  `mlflow.source.git.commit`, `mlflow.source.git.branch`, `mlflow.source.git.repoURL`,
  `mlflow.source.name`, `mlflow.source.type` are present on all **450** runs in the store.
  The code state of an MLflow-tracked run is identifiable. (It pins the code only —
  `datasets/` and `datasets/hiperparams/` are outside it, which is F-12-1 and F-12-3 —
  and `--disable_mlflow` runs and the `results/` tree carry none of it.)
- **`EVAL_MASK_RATE` is not searched, in either profile.** I expected a search that tunes
  its own exam. `define_search_space`'s reduced profile samples `PROB_MASCARA`,
  `LR_DECODE`, `WEIGHT_DECAY_DECODE`, `DROPOUT` and (on mixed tables) `LAMBDA_NUM`; the
  full profile deliberately omits the evaluation rate with a comment saying why
  (`opt.py:128-130`). So the evaluation mask really is the same question across trials and
  across both arms of decision 6 — which is what makes the `evaluation_mask` isolation at
  `data.py:143-148` worth what it costs.
- **The torch stream stays paired across the two decision-6 arms.** Weight init happens at
  fold start (`pretraining.py:45-49`) and `torch.randperm` once per epoch in each stage;
  the reduced profile holds architecture, `BATCH`, `EPOCHS_PRE` and `EPOCHS_DECODE`, and
  `DROPOUT` changes the Bernoulli probability without changing the number of draws. So
  every fold's initial weights are identical between the two arms. Only the numpy
  corruption stream drifts (F-12-2).
- **The configuration values a run trained with are reconstructable from MLflow.**
  `_log_execution_params` logs `complete_configuration`, so the full task key set lands as
  params. I read all 17 back off the `electricity_40nan` parent. Reconstruction does not
  depend on the promoted file still existing — only the attribution does (F-12-3). The one
  documented hole, `EVAL_MASK_RATES_EXTRA`, is already reported as F-10-5/F-06-8.
- **The store contains 6 imputation parents, not 4.** Four of them (`electricity_40nan` ×2,
  `electricity_20nan` ×2) have `status = FAILED`, consistent with the known dtype crash
  (I read the status, not an error message). Only
  `impute_credit-g_20nan_20260910_195313` and `impute_spambase_20nan_20260910_200757`
  finished; both carry `LR_SCHEDULER: plateau`, `cv_folds: 2`, `seed: 42` and no
  `config_source` param, which matches ADR 0005:160-161's note that the two store runs
  "used `plateau` and 2 folds". None of the twelve decision-6 comparison runs exists yet,
  so no published number currently rests on that protocol.
- **The fixture's own repeatability evidence is sound as far as it goes.**
  `"device": "cuda", "runs": 3, "observed_max_deviation": 0.0` is consistent with what I
  read: with `evaluation_mask` isolated and the whole run in one process from one
  `set_global_seed`, a bit-identical repeat at an identical configuration is exactly what
  the code should produce. My findings are not about a run repeating itself; they are
  about what the record says when it does not.

## Open questions

- **Does the `_XXnan` generator have a recorded seed anywhere?** The generator itself is
  not in this repository (`datasets/processed_datasets/` is gitignored and I found no
  script that writes the variants). If it lives elsewhere and is seeded, F-12-1's fix is
  cheap — record the seed. If the variants were drawn once, unseeded, and only exist as
  files, then a hash is the *only* identity available and F-12-1 is the more urgent for it.
  Settled by finding the generator.
- **Would per-fold seeding of the training corruption change published numbers?** It would,
  for every run with `cv_folds >= 2`, since fold 2 onward would draw differently. Whether
  that is acceptable depends on how many imputation numbers the project intends to keep;
  today the answer looks like "two finished parents, both at `plateau`, neither part of the
  decision-6 protocol", which is the cheapest moment this change will ever have. Settled by
  the author deciding whether those two runs are baselines.
- **How far does the digest blind spot reach?** I verified `kr-vs-kp` (all-categorical) and
  `credit-g` (mixed). The six pairs in decision 6 include tables I did not check. A
  one-line loop over every variant's prepared frame would say which of the six carry a
  digest that distinguishes their missingness level and which do not; I checked the store
  rather than recomputing all 45.
