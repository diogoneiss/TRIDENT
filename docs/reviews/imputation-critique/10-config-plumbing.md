# 10 - Configuration plumbing and classification bit-identity

_Critique of the imputation work on branch feat/imputation-task. 2026-09-11._

**Scope:** `src/training/config.py` (all 278 lines), `src/training/types.py:1-380`,
`src/training/runner.py` (all 192 lines), `main.py`, `train.py`,
`tests/unit/test_training_config.py` (all 335 lines), plus the plumbing these reach:
`src/training/cli.py`, `src/training/data.py:45-149`, `src/training/artifacts.py:27-90`,
`src/training/summary.py:31-46,242-257`, `src/training/decoding.py:32-224`,
`src/utils.py:8-16`, `opt.py:79-620`, `experiment_imputation.ps1` (working-tree version),
`README.md` §Configuration System / §MLflow, `CLAUDE.md`.

**Method:** `git diff main...HEAD` on every file above, plus `git show main:<file>` for the
pre-branch loader and `opt.py`. Four empirical probes, none of which wrote inside the
repository:
(1) `Hyperparameters.from_mapping({})` and a partial mapping, against the `main` source;
(2) `resolve_training_request` for `--task imputation` against a fabricated shared
classification config in a scratch CWD;
(3) two `ArtifactWriter.write_metrics` calls for the same dataset under different task
schemas, into a scratchpad metrics dir;
(4) `git log -S`/`git show` to date the epoch-default change.
Corroborated against the real working tree (`metrics/`, `results/credit-g_20nan/`).
I did **not** run pytest (instructed) and did **not** run a training run — none of the
findings below needed one; every claim is either static or closed by a probe.
I could not check what hyperparameter JSON files exist on the author's other machines,
which bounds the blast radius of F-10-2 (see Open questions).

## Findings

### F-10-1 - The flat metrics aggregate is not task-keyed, so the two tasks overwrite each other's record for the same dataset variant

- **Kind:** design
- **Severity:** high
- **Where:** `src/training/artifacts.py:86` (written from `src/training/runner.py:47,157`)
- **Evidence:** `ArtifactWriter.write_metrics` writes a second copy of every run's fold
  table to a flat, non-timestamped path built from the dataset name alone:

  ```python
  root_path = self.metrics_dir / f"{self.dataset_name}_metrics.csv"
  frame.to_csv(root_path, index=False)
  ```

  The row schema is `{"fold", "dataset", **result.metrics}` (`artifacts.py:80`) — there
  is no `task` column and no task component in the filename, while `result.metrics` is
  entirely task-dependent (`f1_macro`/`confusion_matrix_*` vs `impute/masked/*`).
  `runner.py:47` constructs the writer identically for both tasks.

  This has already happened in the working tree. Same directory, two schemas:

  ```
  $ head -1 metrics/credit-g_00nan_metrics.csv
  fold,dataset,accuracy,f1_micro,f1_macro,precision_micro,...,confusion_matrix_tp
  $ head -1 metrics/credit-g_20nan_metrics.csv
  fold,dataset,impute/masked/n_num_cells,impute/masked/n_cat_cells,impute/masked/rmse_num_z,...
  $ cut -d, -f1-2 metrics/credit-g_20nan_metrics.csv
  fold,dataset
  single_split,credit-g_20nan
  ```

  and a classification run for that same variant demonstrably existed before the
  imputation work reached it:

  ```
  $ head -1 results/credit-g_20nan/20260909_065141/metrics.csv
  fold,dataset,accuracy,f1_micro,f1_macro,...
  $ head -1 results/kr-vs-kp_20nan/20260909_072520/metrics.csv
  fold,dataset,accuracy,f1_micro,f1_macro,...
  ```

  Probe (3) reproduces the overwrite directly, into the scratchpad:

  ```
  $ uv run --python 3.10 python -c "...ArtifactWriter(root/'results', root/'metrics', 'credit-g_20nan')..."
  FINAL FLAT FILE: fold,dataset,f1_macro
  1,credit-g_20nan,0.5
  ```

  The imputation row written first is gone, with no error and nothing in the file saying
  a different task ever wrote there. (Pre-existing on `main`, an Optuna trial's
  single-split row could already overwrite a CV parent's rows at this path — but every
  writer was then the same task with the same columns, so the file stayed
  apples-to-apples. Two tasks sharing it is new on this branch.)

- **Consequence:** `metrics/` is the only cross-run, cross-dataset table the repository
  keeps outside MLflow, and it is now silently schema-heterogeneous. Anyone who globs
  `metrics/*_metrics.csv` to build a results table — the obvious use of a directory of
  per-dataset CSVs — gets classification numbers for `credit-g_00nan` and imputation
  numbers for `credit-g_20nan`, joined on nothing, with the column difference the only
  hint. The classification record for `credit-g_20nan` at that path is already lost; on
  `kr-vs-kp_20nan` too. `experiment_imputation.ps1` passes no `--metrics_dir`
  (`experiment_imputation.ps1:60-62`), so each of its four schedule arms overwrites the
  previous arm and the dataset's classification record with it.
- **Direction:** put the task in the flat filename (`<dataset>.<task>_metrics.csv`, the
  scheme `hyperparameter_file` already uses for configs), or add a `task` column to the
  frame `write_metrics` builds and make readers filter on it. The timestamped
  `results/<dataset>/<ts>/metrics.csv` copy is already unambiguous and needs nothing.

### F-10-2 - The JSON/override loader's epoch fallbacks moved from 40 to 300/150, changing classification training length for any partial configuration

- **Kind:** bug
- **Severity:** medium
- **Where:** `src/training/types.py:142` and `src/training/types.py:151`
- **Evidence:** the branch diff of `from_mapping`:

  ```diff
  -    pretraining_epochs=int(values.get("EPOCHS_PRE", values.get("pretraining_epochs", 40))),
  +    pretraining_epochs=int(values.get("EPOCHS_PRE", values.get("pretraining_epochs", 300))),
  -    finetuning_epochs=int(values.get("EPOCH_FINE", values.get("finetuning_epochs", 40))),
  +    finetuning_epochs=int(values.get("EPOCH_FINE", values.get("finetuning_epochs", 150))),
  ```

  Probe (1), on the branch:

  ```
  branch from_mapping({}): 300 150
  partial file EPOCHS_PRE/EPOCH_FINE: 300 150     # a 13-key file with the two epoch keys absent
  ```

  On `main` both of those read `40 40`. This is not an unreachable path: `README.md`
  §Configuration System states "Every key is optional and defaulted, so a file written
  before a key existed keeps loading", and the README's own example config omits `LABELS`,
  so partial files are a sanctioned shape. The same fallback governs every programmatic
  `hyperparams_override` mapping — `AGENTS.md` pins `train.main(args, return_metrics=...)`
  as a stable public API, and the repository's own unit test for it passes `{"DIM": 64}`
  (`tests/unit/test_training_config.py:42`).

  The change was deliberate (`fc5666d` "fix: align pre-training and fine-tuning epoch
  defaults", then `c0a909b` "chore: restore 300/150 epoch defaults to match the August
  batch run"), but both commit messages reason exclusively about the *dataclass* path and
  `main.py --all`; neither mentions that the from_mapping fallback was reachable from a
  partial file or override. Nothing outside those two commit messages records it: no ADR,
  no README note, no test. The one test that claims to guard exactly this,
  `test_a_config_written_before_the_imputation_task_still_loads`
  (`tests/unit/test_training_config.py:127-148`), hard-codes `"EPOCHS_PRE": 300` and
  `"EPOCH_FINE": 150` in its fixture mapping — it passes *because* it supplies the two
  values that moved, so it can never catch this.

- **Consequence:** a classification run driven by a partial config file or a partial
  override mapping now pre-trains 300 epochs instead of 40 and fine-tunes 150 instead of
  40 — 7.5× and 3.75× the compute, and a different model, with no flag, no tag, and no
  line in the run record saying the budget changed. The run's MLflow parameter table shows
  `EPOCHS_PRE = 300` (`config.py:91`, resolved values, not file contents), so the run looks
  like a deliberate 300-epoch run, which makes it uncomparable-but-indistinguishable
  against pre-branch runs of the same config file. This is precisely the class of change
  the project's own rule gates behind a flag + MLflow tag + backfill.
- **Direction:** either keep the pre-branch fallbacks and accept the dataclass/from_mapping
  split (documenting why), or keep the alignment and treat it as the training-behaviour
  change it is: note it in an ADR, and add a test that pins
  `Hyperparameters.from_mapping({}) == Hyperparameters()` so the next move is visible.
  A stronger fix removes the duplicated literal defaults from `from_mapping` entirely and
  derives them from the dataclass fields, so the two can never disagree again.

### F-10-3 - The imputation launcher configures the decode stage through classification's keys, edits classification's config file, and tells the operator to compare a metric imputation never emits

- **Kind:** methodology
- **Severity:** medium
- **Where:** `experiment_imputation.ps1:15,24-39,52,99-100` (working-tree version)
- **Evidence:** the file is new on this branch (`git diff --stat main...HEAD` shows 101
  insertions), and the committed version at `HEAD` does **not** pass `--task` at all
  (`git show HEAD:experiment_imputation.ps1 | grep -n task` returns nothing) — so the
  launcher named "imputation" is, as committed, a second classification sweep. The
  uncommitted working-tree edit adds exactly `"--task", "imputation"` to the argument list
  and nothing else; the config keys and the summary query below were not adapted with it.

  The script's header explains it writes a hyperparameter JSON because
  "Epoch counts cannot be set from the CLI". It writes the file at
  `datasets/hiperparams/<base>/<dataset>.json` (`experiment_imputation.ps1:24-28`) — the
  **shared** file, the one `config.py:49-51` gives to classification and only falls back to
  for imputation — and the body it writes is the classification key set:

  ```powershell
  EPOCHS_PRE = $PretrainEpochs; BATCH = 256
  LR_PRE = 0.00034; WEIGHT_DECAY_PRE = 0.005; PROB_MASCARA = 0.5
  EPOCH_FINE = $FinetuneEpochs; LR_FINE = 0.001; WEIGHT_DECAY_FINE = 0.0019
  LABELS = 4
  ```

  It then runs `main.py --task imputation` (`experiment_imputation.ps1:60`). Under
  `--task imputation` the decode stage reads `EPOCHS_DECODE` / `LR_DECODE` /
  `WEIGHT_DECAY_DECODE` (`decoding.py:59-69`, key set in `config.py:97-105`); `EPOCH_FINE`,
  `LR_FINE`, `WEIGHT_DECAY_FINE` and `LABELS` are parsed and then consumed by nothing.
  The `-FinetuneEpochs` parameter (`:15`) is therefore inert, while the setup banner at
  `:52` announces `EPOCH_FINE=$FinetuneEpochs` as though it took effect. Its default of
  150 happens to equal `decode_epochs`'s default (`types.py:112`), so the mistake is
  invisible until someone passes `-FinetuneEpochs 40` and the decode stage still runs 150.

  Separately, the script's closing instruction (`:99-100`) is classification's:

  ```
  Compare in MLflow: tags.run_role = 'parent' and tags.missingness_percent = '0', grouped by tags.lr_scheduler
  Metric: cv/test/f1_macro/mean with cv/test/f1_macro/ci95_lower and ci95_upper
  ```

  No imputation run emits `f1_macro`, and the filter carries no `tags.task`, so it
  selects the classification runs on the same datasets instead.

- **Consequence:** the imputation schedule sweep — the evidence this branch will be judged
  on — holds the decode stage at its defaults while reporting that it set the stage's epoch
  budget, and hands the operator a query that returns the wrong task's runs and a metric
  the sweep did not produce. It also writes an imputation experiment's configuration into
  the file every classification run reads, which is exactly the coupling ADR 0005's
  task-keyed `<dataset>.imputation.json` exists to prevent; the `finally` block
  (`:80-93`) restores it, but only for a process that reaches `finally` (a `kill`, a
  bluescreen or a reboot mid-sweep leaves `EPOCHS_PRE = 200` in classification's file).
- **Direction:** write the task-keyed file (`hyperparameter_file(dataset, "imputation")`)
  with `complete_configuration(..., "imputation")`'s key set, rename the parameter to the
  stage it drives (`EPOCHS_DECODE`), and replace the summary query with the imputation
  ranking metric plus `tags.task = 'imputation'`. Longer term, the header's premise —
  "epoch counts cannot be set from the CLI" — is what forces every launcher to hand-write
  a config file into a shared, read-by-another-task path.

### F-10-4 - `config_source` over-claims: an imputation run reports a file that supplied none of its decode parameters, and silently inherits that file's schedule

- **Kind:** methodology
- **Severity:** medium
- **Where:** `src/training/config.py:49-58` (with `config.py:72-112`, `runner.py:60-65`)
- **Evidence:** the fallback itself is documented (README §Configuration System item 3,
  "read by both tasks") and is not the defect. What is undocumented is what the run then
  records. Probe (2), in a scratch CWD holding only a classification-shaped shared file:

  ```
  config_source: datasets/hiperparams/credit-g/credit-g_20nan.json
  logged params: {"DIM":192, "HIDDEN_DIM":8, "HEADS":8, "LAYERS":4, "DIM_FEED":96,
                  "DROPOUT":0.4, "EPOCHS_PRE":40, "BATCH":128, "LR_PRE":0.0002,
                  "WEIGHT_DECAY_PRE":0.001, "PROB_MASCARA":0.3, "LR_SCHEDULER":"cosine",
                  "EPOCHS_DECODE":150, "LR_DECODE":0.001, "WEIGHT_DECAY_DECODE":0.0019,
                  "LAMBDA_NUM":1.0, "EVAL_MASK_RATE":0.2}
  ```

  All five decode values are defaults; the file contained none of them.
  `complete_configuration` logs *resolved* values under config-file names
  (`config.py:77-112`), and `ArtifactWriter.write_hyperparameters` writes the same resolved
  set to `results/.../hyperparameters.json` (`artifacts.py:35-77`), so neither the run
  record nor the artifact preserves which keys the named file actually carried. ADR 0005
  decision 5 introduced `config_source` so "tuned and default runs can be told apart"; here
  it says "tuned" for a run whose entire decode stage is at defaults.

  The same line also shows the second half: `LR_SCHEDULER: "cosine"` came from a
  classification study's promoted file, with no `--lr_scheduler` on the command line. It is
  tagged correctly (`runner.py:61` passes `request.hyperparameters.lr_scheduler`), so it is
  discoverable — but the README's documented default for the flag is `cosine_legacy`
  (`README.md:172`) and nothing warned that a file for another task chose the schedule.

- **Consequence:** across an `--all` imputation batch or a multi-dataset sweep, variants
  whose base dataset happens to have a classification config file train at that file's
  architecture, pre-training budget, mask probability and schedule, while variants without
  one train at defaults — and every one of them logs a `config_source` that looks equally
  authoritative. Comparing `impute/masked/impute_score` across those datasets is then
  comparing different encoders under different schedules. Concretely: with
  `datasets/hiperparams/credit-g/credit-g_20nan.json` present and no
  `credit-g_20nan.imputation.json`, an imputation run pre-trains 40 epochs under `cosine`;
  the promoted `credit-g_20nan.imputation.json` in the tree today specifies
  `EPOCHS_PRE: 300`. The two are not comparable and nothing in the run record says so.
- **Direction:** either record which task's file was read (a `config_source_task` param, or
  suffix the path with the fallback: `...json (classification fallback)`), or log the raw
  file contents alongside the resolved values so a reader can see which keys were defaulted.
  A blunter option is to refuse the fallback and make imputation read only its own file,
  since a classification-tuned encoder is a strange starting point for a decode study.

### F-10-5 - `EVAL_MASK_RATES_EXTRA` is in no run record and is silently dropped by promotion, contradicting the documented "a promoted file is complete"

- **Kind:** design
- **Severity:** low
- **Where:** `src/training/config.py:97-105`
- **Evidence:** `complete_configuration`'s imputation branch returns exactly
  `EPOCHS_DECODE, LR_DECODE, WEIGHT_DECAY_DECODE, LAMBDA_NUM, EVAL_MASK_RATE` on top of the
  shared keys. `EVAL_MASK_RATES_EXTRA` is a real, loaded, consumed key —
  `types.py:124,171-176` parses it, `decoding.py:162` iterates it to emit
  `impute/masked/rate_<N>/*` metrics, and `README.md:273` documents it under
  "Imputation (`--task imputation` only)" alongside the five keys that are included.
  `complete_configuration` is both what every run logs (`config.py:72-74` →
  `runner.py:60`) and what `--promote_best` writes (`opt.py:328-345`), and README
  §Configuration System asserts of the latter: "A promoted file is complete: every key the
  task uses, held values included." It is not.
- **Consequence:** a user who sets `EVAL_MASK_RATES_EXTRA: [0.1, 0.5]` in
  `<dataset>.imputation.json` and then runs a study with `--promote_best` gets that key
  deleted by the promotion; the next run emits no `rate_10/*` or `rate_50/*` metrics, and
  the flat `metrics/<dataset>_metrics.csv` and `cv/test/*` column sets shrink between two
  runs that both claim `config_source` = the same file path. Independently, a run that
  *does* score extra rates records no parameter saying which rates were asked for — the
  only trace is the metric key itself, so a run whose extra rates produced zero scored
  cells is indistinguishable from one that requested none.
- **Direction:** include the key in the imputation branch of `complete_configuration`
  (serialising the tuple as a list), or state in the docstring and README that diagnostic
  rates are deliberately excluded from both the record and the promoted file, and say why.

### F-10-6 - The agent-facing MLflow comparison rule was updated for the schedule but not for the task

- **Kind:** methodology
- **Severity:** low
- **Where:** `CLAUDE.md` §"MLflow run analysis"
- **Evidence:** the branch diff of `CLAUDE.md` adds only the schedule clause:

  ```diff
  -Filter `tags.run_role = parent` before comparing runs — ... Full tagging/CI-metric layout: ...
  +Filter `tags.run_role = parent` before comparing runs — ... Also compare within one
  +`tags.lr_scheduler` value (`cosine_legacy` is the pre-ADR-0003 schedule; ...). Full ...
  ```

  `README.md:369` — the reference the rule points at — does carry the missing clause:
  "Filter `tags.run_role = parent` and `tags.task` (`classification` or `imputation`)". The
  rule file that is actually loaded into every session does not.
- **Consequence:** the branch put two tasks into one per-dataset MLflow experiment
  (`opt.py:385`, `get_or_create_experiment(args.dataset_name)`), so `run_role = parent`
  alone now returns a mixture. An agent or a person following the rule as written compares
  a classification parent against an imputation parent on the same dataset; since the two
  share no metric key, the likely outcome is silently empty or half-populated comparison
  tables rather than a visible error. `experiment_imputation.ps1:99` makes exactly this
  mistake (F-10-3).
- **Direction:** add `tags.task` to the `CLAUDE.md` filter, matching `README.md:369`.

## Checked and cleared

- **RNG isolation between the two tasks inside one process (the Optuna scenario).**
  `run_training`'s first statement is `set_global_seed(request.seed)` (`runner.py:37`),
  which reseeds `random`, `numpy.random`, `torch` and, when present, all CUDA generators
  (`src/utils.py:8-16`). Every trial calls it, so trial *N+1* starts from an RNG state
  identical to trial *N*'s regardless of which task trial *N* ran. Between the reseed and
  `build_folds` (`runner.py:46`) the only task-conditional work is `load_complete_sibling`
  (`runner.py:43-45`), which is two `pd.read_csv` calls and a numpy comparison
  (`data.py:92-129`) — no draws. Inside the decode stage, the only extra global consumer
  relative to fine-tuning is `evaluation_mask`, which explicitly saves `np.random`'s state,
  seeds a private stream from `(seed, fold)` and restores it in a `finally`
  (`data.py:143-148`); the clean-target encodings go through
  `preprocess_table(..., fine_tunning=True)`, whose `else` branch (`src/utils.py:79-82`)
  draws nothing. I found no module-level cache or memo anywhere under `src/` that either
  task could populate for the other (`grep` for `lru_cache`, module-level `global`, and
  module-level dict literals returns only `_PREVIEW_WIDTH`-style constants, `discovery.py`
  path constants, and `opt.py:379`'s `global mlflow`, which is process-wide by design and
  set once before any trial). Classification results cannot move because an imputation run
  preceded them in the same process.
- **`train.main(args, return_metrics=True)` keeps the flat Optuna shape for both tasks.**
  `run_from_namespace` (`cli.py:9-18`) is unchanged in structure: CV returns
  `{"dataset", **mean_metrics}`, single-split returns `{"fold", "dataset", **fold.metrics}`.
  The CV branch's aggregator is schema-agnostic — `compute_cv_summary`
  (`summary.py:242-257`) takes "every column that is not `fold` or `dataset`", so imputation
  keys aggregate without a list to extend. The imputation `FoldResult` is rebuilt by the
  runner with `request.dataset.dataset_name` (`runner.py:144-148`), so the `dataset` field
  is the same value classification returns and `decoding.py:218`'s `frame.attrs` lookup
  never reaches the caller. Both branches' guards agree between `runner.py:185` and
  `cli.py:15`.
- **Classification cannot read the task-keyed file.** `_load_base_hyperparameters` builds
  `candidates = [hyperparameter_file(dataset_name, task)]` and appends the shared file only
  when `task != DEFAULT_TASK` (`config.py:49-51`); for classification the list has one
  entry, whose suffix is `.json` (`config.py:68`). Pinned by
  `test_an_imputation_run_prefers_its_own_configuration_and_falls_back_to_the_shared_one`
  (`tests/unit/test_training_config.py:276-307`), which asserts all three states including
  the "task-keyed file present, classification unaffected" one. Promoting an imputation
  study cannot retune classification.
- **`--lr_scheduler` precedence is correct and defaults to the legacy schedule.**
  `load_hyperparameters` applies the flag *after* the base load, so it beats both the
  override mapping and the file (`config.py:29-35`); the parser's default is `None`, not a
  name (`config.py:222-231`), so an absent flag leaves whatever the file said and an absent
  file leaves `cosine_legacy` (`types.py:17,107`). Verified against
  `tests/unit/test_training_config.py:56-92`. `opt.py:225` copies the same value onto every
  trial namespace and onto the trial's tag (`opt.py:243`), so trials and their tag agree.
- **`score_search_objective` is genuinely programmatic-only.** No parser argument exists
  (`config.py:186-278`); `resolve_training_request` reads it only via `getattr`
  (`config.py:135`); `opt.py:221` sets it, and only for imputation. So no ordinary run
  carries `validation/*` keys into the CV summariser. (See Open questions for the one
  construction that would.)
- **Base-name derivation is consistent across the three places that compute it.**
  `hyperparameter_file` (`config.py:67`), `DatasetSpec.from_name` (`types.py:190`) and
  `opt.py:417` all split on the first underscore, so the config path, the CSV path
  (`data.py:88-89`) and the study's column-type probe agree for every name in
  `datasets/processed_datasets/` (`credit-g`, `kr-vs-kp` included).
- **The new parse-time guards fire where they should.** `--score_null_path` without
  `--task imputation` and `--search_space reduced` without it both `SystemExit`
  (`config.py:156-181`), and `--cv_folds 1` is refused with a message naming the flag
  rather than `n_splits`; `build_folds` repeats the `cv_folds < 2` guard
  (`data.py:177-181`) for programmatic callers that bypass `validate_parsed_args`, which is
  the right place for it given `train.main` and `opt.py` build namespaces by hand.
- **New `PreparedDataset` fields do not move the pipeline.** `scaler` and `raw_numerical`
  are captured before `fit_transform` overwrites the columns (`data.py:67-71`) and both
  default to `None` (`types.py:243,248`), so positional construction sites and the
  classification path are untouched; the copy is memory, not behaviour.

## Open questions

- **How many partial classification configs exist in the wild?** F-10-2's severity is
  entirely a function of that. `datasets/hiperparams/` is gitignored
  (it is untracked here and holds only two `.imputation.json` pairs, both with
  `EPOCHS_PRE: 300`), and `main`'s `save_best_params` always wrote `EPOCHS_PRE` and
  `EPOCH_FINE`, so machine-written files are safe. What would settle it: listing
  `datasets/hiperparams/**/*.json` on whatever machine ran the August batch and the
  pre-branch Optuna studies, and grepping for files missing either key.
- **`score_search_objective=True` together with `cv_folds`** would put `validation/*` keys
  into the frame `write_metrics` builds and hence into `cv/test/validation/impute/...` —
  a validation number under a test prefix. No entry point reaches it today (`opt.py`'s
  trial namespace never sets `cv_folds`, so trials are always single-split), so it is a
  latent trap rather than a finding. A `__post_init__` check on `TrainingRequest` rejecting
  the combination would close it for good.
- **`ArtifactWriter`'s results directory is second-resolution and not task-keyed**
  (`artifacts.py:29-30`). Probe (3) produced two writers in the same second and they shared
  `results/credit-g_20nan/20260911_094800/`, so the second run's `metrics.csv` replaced the
  first's. Real training takes longer than a second, so this needs a parallel or trivially
  short launch to bite; I could not determine whether anything in the workflow launches two
  runs concurrently. A `results/<dataset>/<task>_<ts>/` layout would remove the question.
- **Whether the `main.py --all --task imputation` default of `--nan_level 0` is intended.**
  Every `_00nan` variant is its own complete sibling, so `load_complete_sibling` returns
  `None` (`data.py:104`) and the batch scores only artificially masked cells — no
  `impute/induced/*` at all. That is a reasonable diagnostic mode, but `run_all` prints the
  same table either way and nothing says the induced benchmark was skipped. Whether that is
  a defect depends on how the `--all` imputation mode is meant to be used, which I could
  not establish from the docs.
