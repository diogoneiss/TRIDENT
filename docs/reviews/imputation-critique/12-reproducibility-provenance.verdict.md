# 12 - Reproducibility and provenance of an imputation run end to end: verdicts

_Adversarial verification of `12-reproducibility-provenance.md`. 2026-09-11._

**Method:** Read `src/training/artifacts.py`, `src/training/data.py`, `src/training/runner.py`,
`src/training/tracking.py`, `src/training/config.py`, `src/training/environment.py`,
`src/training/pretraining.py`, `src/training/decoding.py`, `src/training/finetuning.py`,
`src/utils.py`, `src/training/types.py`, `opt.py:78-135,300-350,550-575`,
`tests/integration/test_credit_g_imputation_regression.py`,
`tests/fixtures/credit-g_20nan_imputation_regression.json`,
`docs/adr/0005-reduced-optuna-search-for-imputation.md:140-196`, and
`datasets/generate_splits.py`. Read-only SQL against `mlflow.db` (`file:mlflow.db?mode=ro`).
Read-only git. Five `uv run --python 3.10` experiments, listed under the findings they settle.

I did **not** spend the permitted training run: nothing here needed one.
I did not run pytest.

Where my measured numbers differ from the critic's (the `np.random.choice` call counts), it is
because I ran `preprocess_table` over the whole `credit-g_20nan` feature frame rather than a
fold subset. The mechanism reproduced exactly; the specific counts were never expected to match.

The single most consequential thing I found is in the critic's own **Open questions**: the
`_XXnan` generator *is* in this repository, *is* committed, and *is* seeded. That was the
load-bearing assumption behind F-12-1's `high`, and it is false.

## Verdicts

### F-12-1 - The induced-missing benchmark rests on an unrecorded, ungoverned missingness draw that nothing can detect the substitution of

- **Verdict:** UNSOUND
- **Severity after review:** low
- **Basis:**

  Every *code-level* premise checks out. I verified each:

  - `write_tracking_provenance` (`artifacts.py:177-207`) writes `source_path`, `splits_path`,
    `dataset_name`, `prepared_schema`, the two class-name strings, `split_strategy`,
    `cv_folds`, `seed` — no hash, no size, no row count. Confirmed byte for byte against the
    real `results/credit-g_20nan/20260910_195312/data/provenance.json`.
  - The `_00nan` sibling appears in no record. The finished `credit-g_20nan` parent
    (`1873c2d756ee`) has exactly one MLflow `inputs` row:
    `credit-g_20nan 32b277ac datasets\processed_datasets\credit-g\credit-g_20nan.csv`.
  - `git check-ignore -v datasets/processed_datasets/credit-g/credit-g_20nan.csv` ->
    `.gitignore:19:datasets/processed_datasets/`; `git ls-files datasets/processed_datasets` -> 0.
  - `_assert_row_aligned` passes a re-drawn variant. Reproduced the critic's numbers exactly:

    ```
    variant  shape (1000, 21) nan 4000
    re-drawn nan 4007
    overlap of missing cells: 769 of 4000
    _assert_row_aligned on re-drawn variant: PASSED
    shape equal: True | columns equal: True
    ```
  - The MLflow digest collides across missingness levels on an all-categorical table.
    Confirmed both by recomputation (mlflow 3.14.0 `compute_pandas_digest`:
    `kr-vs-kp_20nan/_40nan/_80nan` all `0d37f6f0`, `_00nan` `aac829fb`) **and** in the live
    store on the prepared frames: `kr-vs-kp_20nan`, `_40nan`, `_60nan`, `_80nan` all carry
    `3f2c2e1d`.

  What does **not** follow is the consequence. The critic wrote, in Open questions: *"The
  generator itself is not in this repository ... and I found no script that writes the
  variants."* It is `datasets/generate_splits.py`, it is **tracked**
  (`git ls-files datasets/generate_splits.py` -> hit), and it is seeded:

  ```python
  NAN_PERCENTAGES = [0.20, 0.40, 0.60, 0.80]
  RANDOM_SEED     = 42
  ...
  df_nan = inject_nans(df, pct, TARGET_COLUMN, seed=RANDOM_SEED + int(pct*100))
  ...
  rng = np.random.default_rng(seed)
  to_nan = rng.choice(non_nan_idx, size=n_here, replace=False)
  ```

  Its input, `datasets/datasets_raw/`, is **also tracked** (9 CSVs, not gitignored). So the
  whole chain raw table -> seeded generator -> `_XXnan.csv` sits inside the commit that
  `mlflow.source.git.commit` already records. I regenerated the variants from the committed
  inputs and compared against the files on disk:

  ```
  raw == _00nan frame-equal: True
  regenerated == on-disk _20nan: True
  missingness pattern identical: True | cells differing: 0
  kr-vs-kp_20nan: frame-equal=True pattern-identical=True nan=23004
  kr-vs-kp_40nan: frame-equal=True pattern-identical=True nan=46008
  spambase_20nan: frame-equal=True pattern-identical=True nan=52440
  ```

  Three of three, including the all-categorical table the digest argument rests on.

- **Correction:** The scenario the finding is built on — *"if
  `datasets/processed_datasets/credit-g/credit-g_20nan.csv` is regenerated on another machine,
  or between the defaults arm and the tuned arm, or between today and the re-run in six
  months, the two numbers are measured on different cell populations"* — does not hold.
  Regenerating from the committed raw table with the committed generator reproduces the
  identical missingness pattern, cell for cell. The `.csv` is gitignored as the *build output*
  of a committed, seeded, deterministic script, not as ungoverned data. The fixture claim
  falls with it: a fixture failure **can** be attributed to code or data, by regenerating and
  diffing. The scaler point falls too — `scale_` is derivable from a file that is itself
  derivable.

  Two smaller corrections. (a) For **mixed** tables MLflow's digest does distinguish a redraw
  (original `credit-g_20nan` `85f29a2f` vs my re-drawn `4d732d19`), so "no check in the
  pipeline can detect its substitution" needs the qualifier *on an all-categorical table, and
  even there only as a passive record, never as a guard*. (b) `_assert_row_aligned`'s docstring
  really does claim more than the check delivers; that sub-point stands on its own.

  What genuinely survives, and why it is `low`: the run's own record names neither the
  generator, its seed, nor the `_00nan` sibling it reads, so a reader must know the naming
  convention to reconstruct the benchmark; and the one residual reproduction risk the critic
  never named is that `numpy.random.Generator` carries **no** cross-version stream guarantee
  (unlike the legacy `RandomState` the critic correctly cleared in F-12-4), so
  `default_rng(...).choice` is the one link in this chain that a numpy upgrade could move.
  Recording a SHA-256 of both files is still a cheap, correct improvement — it is just a
  nice-to-have, not a hole under every published number.

### F-12-2 - One global numpy stream means folds >= 2 enter at an offset set by earlier folds, so decision 6's "the configuration is the only difference" is not what it sounds like

- **Verdict:** SOUND
- **Severity after review:** medium
- **Basis:** Every premise verified in source and by measurement.

  `run_training` seeds once (`runner.py:37`, `set_global_seed(request.seed)`) and runs all
  folds in one process; `grep -rn set_global_seed src/ main.py train.py opt.py` shows
  `runner.py:37` is the **only** call site. The only global-numpy consumers in `src/` are
  `utils.py:58` (`np.random.rand(*data.shape)`) and `utils.py:73` (`np.random.choice` in the
  floor loop). Pre-training calls `preprocess_table` twice per epoch
  (`pretraining.py:70-75`), the decode stage once (`decoding.py:91-96`).

  Measured (my own run, full `credit-g_20nan` feature frame):

  ```
  np.random.choice calls per epoch, p=0.5: [0, 0, 3, 0, 1]
  np.random.choice calls per epoch, p=0.2: [63, 77, 63, 82, 67]
  EPOCHS_PRE=10, PROB_MASCARA=0.5 -> [0.06397063 0.66196355 0.24007852 0.65292426]
  EPOCHS_PRE=10, PROB_MASCARA=0.4 -> [0.86410157 0.73181109 0.26695575 0.89779092]
  EPOCHS_PRE=11, PROB_MASCARA=0.5 -> [0.5070653  0.7613947  0.99786015 0.20341169]
  fine_tunning=True MT19937 pos: 624 -> 624 | key identical: True
  evaluation_mask state restored: True
  ```

  The paired components are confirmed as the critic describes. `build_folds` runs before any
  training and uses `random_state=seed` throughout (`data.py:170-220`);
  `evaluation_mask` reseeds on `(seed * 1_000_003 + fold)` and restores the global state
  (`data.py:143-148`) — measured restored; the preview sampler uses an isolated
  `np.random.default_rng` (`artifacts.py:244`).

  I confirmed the arms really are matched on everything but the sampled knobs. The reduced
  profile samples `PROB_MASCARA`, `LR_DECODE`, `WEIGHT_DECAY_DECODE`, `DROPOUT`, `LAMBDA_NUM`
  (`opt.py:94-107`), and the promoted `credit-g_20nan.imputation.json` matches
  `Hyperparameters` defaults on every held knob — `DIM 128`, `HIDDEN_DIM 16`, `HEADS 16`,
  `LAYERS 2`, `DIM_FEED 32`, `EPOCHS_PRE 300`, `BATCH 256`, `LR_PRE 0.00034`,
  `WEIGHT_DECAY_PRE 0.005`, `EPOCHS_DECODE 150`, `EVAL_MASK_RATE 0.2` — differing only on
  `PROB_MASCARA 0.4` (default 0.5, `types.py:103`), `DROPOUT 0.3` (default 0.2), `LR_DECODE`,
  `WEIGHT_DECAY_DECODE`, `LAMBDA_NUM`. So `torch.randperm` counts match and only the numpy
  corruption stream drifts, exactly as claimed.

  ADR 0005:157-158 does say *"both `--task imputation --cv_folds 5 --lr_scheduler cosine
  --seed 42`, so the configuration is the only difference"*, and calls it *the glossary's
  paired single-seed CV protocol*.

- **Correction:** Two wording fixes, and a severity cut from `high`.

  (a) "an *arbitrary, unseeded* stream offset" is wrong. The offset is fully deterministic
  from `(seed, configuration)` and reproduces exactly on a re-run — I verified a fixed
  `(EPOCHS_PRE, PROB_MASCARA)` gives the same fold-2 draws every time. It is **unmatched
  between arms**, not unseeded. Likewise "the configuration is the only difference" is
  literally true as a statement about *inputs*; what is false is the inference it invites,
  that the outcome difference is attributable to the configuration's mechanism rather than to
  an incidental RNG reshuffle the configuration change causes. The ADR sentence needs
  narrowing for that reason, not because a second uncontrolled input exists.

  (b) The finding is sharper than the critic wrote it, and worth restating: fold 1 epoch 1 is
  the **only** genuine common-random-numbers moment. Both arms consume the same
  `np.random.rand(*shape)` matrix at the same stream position and merely threshold it
  differently, so at `p=0.4` the masked set is nested inside the `p=0.5` set. That pairing is
  lost the moment the floor loop runs a different number of `choice` calls — within fold 1,
  not at fold 2.

  Severity `medium`, not `high`, for three reasons. The first-order confound is unavoidable:
  changing `PROB_MASCARA` changes the realised corruption under *any* seeding scheme, so the
  branch-specific harm is only the loss of the shared underlying uniforms past the first
  epoch. No published number rests on it — the store holds 6 imputation parents, 4 FAILED, the
  2 FINISHED ones at `plateau`/`cv_folds=2` with no `config_source`; none of the twelve
  decision-6 runs exists. And the remedy is a documentation narrowing plus a flagged
  behaviour change, which is the cheapest it will ever be. The two collateral observations —
  "changing an epoch budget is not a truncation" and "no fold is reproducible on its own" —
  are both correct and are the parts most likely to bite day to day.

### F-12-3 - `config_source` is a mutable path into an untracked directory, and the fallback is silent

- **Verdict:** SOUND
- **Severity after review:** medium
- **Basis:** `promote_best_configuration` writes to `hyperparameter_file(dataset_name, task)`
  (`opt.py:342-345`) — `datasets/hiperparams/<base>/<dataset>[.task].json`, one fixed path per
  `(dataset, task)`, no version, no study id, no timestamp. `config_source` is that path
  string (`config.py:56`).

  ```
  $ git ls-files datasets/hiperparams | wc -l   -> 0
  $ git check-ignore -v datasets/hiperparams/   -> exit 1
  $ git status --porcelain                      -> ?? datasets/hiperparams/
  ```

  Untracked **and** not gitignored: the six promoted files — the entire output of the ADR 0005
  search — exist in one working tree and are one `git clean -fd` from gone. That is the
  sharpest consequence here and the critic under-sold it.

  `_load_base_hyperparameters` (`config.py:38-58`) returns `Hyperparameters(), "defaults"`
  with no warning when no candidate exists. The dangling pointer is already in the store, as
  claimed: three parents record `config_source = datasets/hiperparams/electricity/
  electricity_{20,40}nan.json` (`10cdd533fee5`, `631c3b64a184`, `7ed05b09f090`) and
  `ls datasets/hiperparams/electricity` is now empty.

  The classification-keyed fallback is real. `candidates = [hyperparameter_file(dataset_name,
  task)]` then, for a non-default task, `hyperparameter_file(dataset_name, DEFAULT_TASK)`
  (`config.py:49-51`). An imputation run with no `.imputation.json` but an existing
  `<dataset>.json` loads the classification file and logs its path, while
  `Hyperparameters.from_mapping` leaves `EPOCHS_DECODE`, `LR_DECODE`, `WEIGHT_DECAY_DECODE`,
  `LAMBDA_NUM` and `EVAL_MASK_RATE` at dataclass defaults — a `config_source` that names a
  tuned file over a decode stage that is entirely untuned, with nothing in the record saying
  which half came from where.

  Recoverability is as the critic says: `_log_execution_params` logs `complete_configuration`
  (`config.py:78-113`), and `--retrain_best` tags `optuna_study_run_id` (`opt.py:562`) while
  a `main.py`-launched comparison run does not.

- **Correction:** "re-running that commit silently trains the defaults instead ... a
  *different* configuration reported as a legitimate one" overstates the silence. The run does
  log `config_source='defaults'`, so a reader comparing params against the original run sees
  the mismatch immediately. What is genuinely silent is the *load*: no warning on stdout, and
  no disclosure when the classification-keyed fallback fires. Severity stays `medium` — the
  untracked-and-unignored promoted files and the demonstrated dangling pointers carry it.

### F-12-4 - The environment record omits the scikit-learn version

- **Verdict:** UNSOUND
- **Severity after review:** low
- **Basis:** The premise is true and I verified it: `runtime_environment_tags`
  (`environment.py:13-21`) returns exactly `device`, `gpu_name`, `torch_version`,
  `cuda_version`; `git show main:src/training/environment.py` -> *"exists on disk, but not in
  'main'"*, so the mechanism is branch-new; the fixture's `environment` block is
  `python/platform/torch/device/seed/cv_folds/runs` with no numpy, pandas or scikit-learn. The
  load-bearing-ness of scikit-learn is also true (`LabelEncoder`/`StandardScaler` at
  `data.py:58-71`, the splitters at `data.py:170-220`, and `_score_induced_missing`
  re-encoding the sibling before scoring).

  The consequence does not follow. `uv.lock` **is tracked**
  (`git ls-files uv.lock pyproject.toml` -> both), and it pins the exact versions:

  ```
  uv.lock:3044  name = "scikit-learn"
  uv.lock:3045  version = "1.7.2"
  uv.lock:1658  { name = "scikit-learn", version = "1.7.2", ... marker = "python_full_version < '3.11'" }
  ```

  AGENTS.md mandates `uv run --python 3.10`, which resolves to that pin. The critic's own
  "Checked and cleared" section establishes that `mlflow.source.git.commit` is present on all
  450 runs — so for every MLflow-tracked run the record *does* say what scikit-learn, numpy
  and pandas versions the run was meant to have, via the commit it already names.

- **Correction:** *"the record gives the reader no way to tell which"* is contradicted by the
  critic's own cleared item. The commit tag plus a tracked lockfile is the dependency
  manifest; `runtime_environment_tags`' docstring scopes the four keys to *"so its timings are
  comparable across machines"*, which is what device/gpu/torch/cuda are for. The residual gap
  is narrower than the finding states and is why this survives at `low`: (1) nothing verifies
  the running interpreter actually matched the lock, so a run outside `uv` records a commit
  that lies about its own environment — a `sklearn_version` tag would catch precisely that, and
  is the one argument for the change that still holds; (2) `--disable_mlflow` runs record no
  commit and no environment at all; (3) the fixture's `environment` block is hand-maintained,
  so it inherits nothing from the lock.

### F-12-5 - The results directory and the MLflow run are joined only by a timestamp drawn twice, and both finished parents are off by a second

- **Verdict:** CONFIRMED
- **Severity after review:** low
- **Basis:** Two independent calls, exactly as described: `ArtifactWriter.__init__` takes
  `datetime.now().strftime("%Y%m%d_%H%M%S")` (`artifacts.py:29`) and builds
  `results/<dataset>/<timestamp>/`; `MlflowTracker.parent_run` takes its own at
  `tracking.py:120`, after `setup_mlflow()` and `get_or_create_experiment(dataset_name)`
  (`tracking.py:114-115`) round-trip the SQLite store, then builds
  `run_name=f"{prefix}_{dataset_name}_{timestamp}"` (`tracking.py:137`).

  Both finished imputation parents in the store are skewed, 2 of 2:

  | results directory | MLflow run name |
  |---|---|
  | `results/credit-g_20nan/20260910_195312` | `impute_credit-g_20nan_20260910_195313` |
  | `results/spambase_20nan/20260910_200756` | `impute_spambase_20nan_20260910_200757` |

  Nothing else joins them. `mlflow.log_artifact` uploads *content*, never the source path;
  `runner.py` references `artifacts.results_dir` only to hand `metrics.csv` to that call
  (`runner.py:161-163`), and no param or tag on the run names the results path, nor does
  `provenance.json` carry a run id. The weight the critic gives it is right: `BufferedFoldTracker`
  buffers every fold but `_log_diagnostic_children` (`tracking.py:212-234`) replays only the
  `best_fold`/`worst_fold` records, so in a 5-fold run three folds' cell ledgers and previews
  exist only under the gitignored `results/` tree, behind a directory name that does not match
  the run.

- **Correction:** None to the facts. `low` is right — the skew is a second, the directory is
  findable by dataset and date, and the metrics themselves are all in MLflow. Calling it a
  `bug` rather than a design gap is generous but defensible: the two names are meant to be the
  same string and demonstrably are not.

## What this critique missed

1. **`provenance.json` records a pointer to a file the run never opens, beside no pointer to
   the one it does.** `prepare_dataset` always sets `splits_path`
   (`data.py:48`), and `write_tracking_provenance` always writes it — but `build_folds` reads
   it only when `cv_folds is None` (`data.py:155-172`). The real
   `results/credit-g_20nan/20260910_195312/data/provenance.json` is a `cv_folds: 2` run and
   still names `datasets\processed_datasets\splits\credit-g_split.json`, which had no bearing
   on a single fold. So the lineage record carries a dead pointer to an unused file while
   omitting the `_00nan` sibling the run actually read — a sharper framing of F-12-1's
   surviving half, and one the critic did not spot.

2. **The environment record and `config_source` reach MLflow only; the on-disk tree gets
   neither.** `runtime_environment_tags` is passed to `tracker.parent_run` and nowhere else
   (`runner.py:62`), and `config_source` likewise goes only to `_log_execution_params`.
   `write_hyperparameters` writes values but not their origin, and `provenance.json` carries
   no device, no library version and no config source. A `--disable_mlflow` run — the form
   AGENTS.md recommends for isolated runs, and the form every integration test uses — leaves a
   `results/` tree with no record of where it ran or where its configuration came from. F-12-4
   argues about *which* descriptors are recorded; the larger gap is *where*.

3. **The critic cleared numpy for the wrong reason, and the right one cuts the other way.**
   F-12-4 credits numpy's `RandomState` compatibility policy for the training corruption
   stream, which is correct (`utils.py:58,73` use the legacy global functions). But
   `datasets/generate_splits.py:30` draws the benchmark's missingness with
   `np.random.default_rng`, and NEP 19 explicitly exempts `Generator` from that policy. The one
   place in this whole chain where a numpy upgrade could silently change a published number is
   the one the critic assumed did not exist.

4. **The promoted configuration files are untracked and *not* ignored.** F-12-3 states both
   facts but frames the loss as attribution. The louder consequence is destruction: `??
   datasets/hiperparams/` in `git status` means any routine `git clean -fd` erases the entire
   output of the ADR 0005 search effort, and the store already demonstrates the aftermath in
   `datasets/hiperparams/electricity/`, which is empty while three runs still point into it.

5. **The `row` section's fix is understated.** The critique correctly identifies that ledger
   `row` is a position in the fold's test block and proposes one extra column. Worth adding:
   `fold.test_indices` is also what would let a reader check the induced population is the
   same across a `_20nan`...`_80nan` comparison, which is the exact comparison README:321 and
   ADR 0004:222 invite — and with F-12-1 cut to low, that is now the *only* thing standing
   between the record and that check, since the cells themselves are fully regenerable.
