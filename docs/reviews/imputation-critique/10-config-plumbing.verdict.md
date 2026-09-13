# 10 - Configuration plumbing and classification bit-identity: verdicts

_Adversarial verification of `10-config-plumbing.md`. 2026-09-11._

**Method:** `graphify query "how are hyperparameter config files loaded and metrics written
per task"` to orient, then the source itself: `src/training/config.py` (all 279 lines),
`src/training/types.py:85-200`, `src/training/artifacts.py:27-110`,
`src/training/runner.py:40-75`, `src/training/tracking.py:100-175,360-390`,
`src/training/decoding.py:150-175`, `opt.py:79-160,195-250,295-400`,
`experiment_imputation.ps1` (working tree, all 101 lines), `README.md:205-280,369-385`,
`docs/adr/0005-reduced-optuna-search-for-imputation.md`. Branch history via
`git diff main...HEAD`, `git show main:src/training/types.py`, `git show fc5666d --stat`,
`git show -s fc5666d c0a909b`, `git show fc5666d -- tests/unit/test_training_runtime.py`,
`git show main:src/training/config.py`, `git diff -- experiment_imputation.ps1`,
`git check-ignore -v metrics/... results/...`, plus a repo-wide grep for readers of
`_metrics.csv`.

Six probes, all read-only on the repo, writing only into the scratchpad. No pytest, no
training run (none was needed).

Probe P1-P3, `uv run --python 3.10 python .../probe.py`:

```
=== P1: from_mapping fallbacks (branch) ===
from_mapping({}) EPOCHS_PRE/EPOCH_FINE: 300 150
dataclass defaults          : 300 150
from_mapping({}) == Hyperparameters(): True
13-key partial (no epoch keys): 300 150

=== P2: flat metrics overwrite across tasks ===
after imputation write: fold,dataset,impute/masked/impute_score | single_split,credit-g_20nan,0.42
after classification write: fold,dataset,f1_macro | 1,credit-g_20nan,0.5

=== P3: complete_configuration key sets ===
eval_mask_rates_extra parsed: (0.1, 0.5)
imputation keys: [... 'EVAL_MASK_RATE' ...]   EVAL_MASK_RATES_EXTRA present: False
```

Probe P4 (`resolve_training_request` from a scratch CWD holding one fabricated
classification-shaped shared file):

```
['--dataset_name','credit-g_20nan','--task','imputation']
  config_source: datasets/hiperparams/credit-g/credit-g_20nan.json
  logged: {... 'EPOCHS_PRE': 40, ... 'LR_SCHEDULER': 'cosine',
           'EPOCHS_DECODE': 150, 'LR_DECODE': 0.001, 'WEIGHT_DECAY_DECODE': 0.0019,
           'LAMBDA_NUM': 1.0, 'EVAL_MASK_RATE': 0.2}
['--dataset_name','kc2_20nan','--task','imputation']
  config_source: defaults
  logged: {... 'EPOCHS_PRE': 300, ... 'LR_SCHEDULER': 'cosine_legacy', ...}
```

Probe P5 (`define_search_space` under both profiles, resolved through `from_mapping`):

```
reduced -> sampled: DROPOUT, LAMBDA_NUM, LR_DECODE, PROB_MASCARA, WEIGHT_DECAY_DECODE
   resolved EPOCHS_PRE = 300  EPOCHS_DECODE = 150
full    -> sampled: ... EPOCHS_PRE, EPOCHS_DECODE ...
   resolved EPOCHS_PRE = 40   EPOCHS_DECODE = 50
```

Probe P6, read-only sqlite against `mlflow.db` (`mode=ro`), parents grouped by experiment
and `tags.task`:

```
('TRIDENT/credit-g', 'classification', 15)   ('TRIDENT/credit-g', 'imputation', 1)
('TRIDENT/electricity', 'classification', 7) ('TRIDENT/electricity', 'imputation', 4)
('TRIDENT/spambase', 'classification', 7)    ('TRIDENT/spambase', 'imputation', 1)
```

## Verdicts

### F-10-1 - The flat `metrics/<dataset>_metrics.csv` carries no task key, so the two tasks silently overwrite each other's record for one dataset variant

- **Verdict:** SOUND
- **Severity after review:** medium (critic said high)
- **Basis:** Premise confirmed at the source. `artifacts.py:80-88` builds the flat path
  from the dataset name alone and the rows from `{"fold", "dataset", **result.metrics}`,
  with no `task` anywhere:

  ```python
  rows = [{"fold": result.fold, "dataset": result.dataset_name, **result.metrics} for result in fold_results]
  root_path = self.metrics_dir / f"{self.dataset_name}_metrics.csv"
  frame.to_csv(root_path, index=False)
  ```

  `runner.py:47` constructs the writer identically for both tasks, and probe P2 reproduces
  the cross-task clobber end to end. The working-tree state is as described:
  `metrics/credit-g_00nan_metrics.csv` is classification-schema,
  `metrics/credit-g_20nan_metrics.csv` is imputation-schema, both directories are
  `.gitignore`d (`git check-ignore -v` → `.gitignore:18 results/`, `.gitignore:20 metrics/`),
  so the overwritten flat rows are not recoverable from git.
  `experiment_imputation.ps1:60-62` indeed passes no `--metrics_dir`.
- **Correction:** the severity. Two things cut it. First, the flat file is a convenience
  duplicate, not the record of truth: the per-run copy at
  `results/credit-g_20nan/20260909_065141/metrics.csv` still holds the classification folds
  and MLflow holds them too, so nothing was actually lost — the critique says so itself and
  then rates the finding as if something had been. Nor does the repository contain the
  reader whose breakage would justify high: a grep for `_metrics.csv` across `*.py`,
  `*.ipynb`, `*.ps1` and `*.md` returns only `artifacts.py:86` (the writer) and matches on
  the unrelated `raw_fold_metrics.csv`; `main.py` names neither `metrics_dir` nor
  `metrics.csv`, and `scripts/` holds only the two backfill scripts. The hazard is real for
  a human globbing the directory, which is what the critique describes — but it is a
  latent-reader hazard, not a broken pipeline.
  Second, the overwrite that actually happened in this tree is not the cross-task one the
  critique reconstructs. `metrics/credit-g_20nan_metrics.csv` holds a single
  `fold = single_split` row, while `results/credit-g_20nan/20260910_195312/metrics.csv`
  holds a multi-fold **imputation** CV run and `results/credit-g_20nan/optuna_20260911_011020/`
  is a later study. Every Optuna trial namespace gets `args.metrics_dir = "metrics"`
  (`opt.py:208`, `opt.py:399`), so the last trial of that study reduced the file to one row,
  destroying a same-task CV run as well as the classification one. The file has been a
  last-writer-wins scratch pad since before this branch; cross-task heterogeneity is a new
  facet of it, not a new failure mode. Medium, and the fix the critique proposes (task in
  the name or a `task` column) is still the right one — it just should also say the file
  needs a run discriminator at all, not only a task one.

### F-10-2 - `from_mapping`'s epoch fallbacks moved 40/40 → 300/150, changing classification training length for partial configs

- **Verdict:** REFUTED
- **Severity after review:** low (critic said medium)
- **Basis:** to be unambiguous: **the behaviour change is real**. What is refuted is that it
  is a defect rather than a repair. Probe P1 shows `from_mapping({})` and a 13-key partial
  both yield 300/150 on the branch, and `git show main:src/training/types.py:37,47` confirms
  40/40 there. But the defect framing does not survive contact with the sources the critique
  cites:

  1. **It was an inconsistency, not a behaviour.** `main`'s dataclass already read
     `pretraining_epochs: int = 300` / `finetuning_epochs: int = 150`, with
     `# pretraining_epochs: int = 40` commented out above each. So on `main`
     `Hyperparameters()` gave 300/150 while `from_mapping({})` gave 40/40 — the default
     depended on which constructor ran. Probe P1 shows the branch makes
     `from_mapping({}) == Hyperparameters()` **True**. The branch removed a split; it did
     not introduce one. README's own precedence list (`README.md:207`, "**Default values**:
     the `Hyperparameters` defaults when no file exists") documents 300/150 as the default,
     and the example config at `README.md:227,234` names them.
  2. **"no flag, tag, ADR or test recording it" is false on the ADR count.** ADR 0005
     decision 2's held table states it outright:

     ```
     | held | `EPOCHS_PRE`, `LR_PRE`, `WEIGHT_DECAY_PRE` | 300, 0.00034, 0.005 | ... |
     | held | `EPOCHS_DECODE` | 150 | a budget under best-checkpoint selection |
     ```

     and that table is *load-bearing on this branch*: the `reduced` profile — the new
     default for `--task imputation` — returns only five keys and relies on `from_mapping`
     resolving the held ones to the task default. Probe P5 shows a reduced trial resolving
     to `EPOCHS_PRE = 300, EPOCHS_DECODE = 150`. Under `main`'s fallback of 40 the ADR's
     held table would have been wrong and the promoted file (which really does carry
     `"EPOCHS_PRE": 300` — see `datasets/hiperparams/credit-g/credit-g_20nan.imputation.json`)
     would not have matched what the study ran. The critique never opened ADR 0005 on this
     point.
  3. **"both commit messages reason exclusively about the dataclass path" is false.**
     `git show -s fc5666d` reads: *"the default a run got depended on whether it was built
     from a JSON config or from the dataclass"* — the JSON-config path is the subject of
     the sentence that justifies the change. That same commit also moved
     `README.md:227,234` (`git show fc5666d --stat`: `README.md | 4 ++--`).
  4. **Blast radius is empty.** Every `datasets/hiperparams/**/*.json` in the tree carries
     the epoch keys (checked by grep over all four files); `opt.py`'s `full` profile samples
     `EPOCHS_PRE` and `EPOCH_FINE` explicitly (`opt.py:124,146`), so no classification
     Optuna trial reaches the fallback.
  5. **The test hunk in `fc5666d` shows a red test being made green, not a guard being
     bumped.** `git show fc5666d -- tests/unit/test_training_runtime.py` edits the expected
     mapping in `test_runner_uses_fold_buffers_and_finalizes_cross_validation_once`, which
     is an *assertion* on what the runner logged
     (`assert fake_tracker.parent_run_kwargs == {... "hyperparameters": {...}}`) for a
     request with `"config_source": "defaults"` — i.e. no file and no override. On `main`
     that path returns `Hyperparameters()` (`git show main:src/training/config.py:22`,
     `return Hyperparameters()`), which was already 300/150, while the assertion read
     `EPOCHS_PRE: 40, EPOCH_FINE: 40`. The test was failing on `main`, exactly as fc5666d's
     body states. The value the commits edited is the *dataclass* default that never moved
     on that path, so nothing that pinned the old behaviour was relaxed.
  6. The other test critique is also overstated.
     `test_a_config_written_before_the_imputation_task_still_loads`'s own docstring says it
     pins that a pre-decode-stage config still loads; it does not "claim to guard exactly
     this."
- **Correction:** what deserves to survive is one line of the critique's own Direction, at
  low severity: nothing pins `Hyperparameters.from_mapping({}) == Hyperparameters()`, and
  that invariant has already broken once (fc5666d's body records that the mismatch made
  `test_runner_uses_fold_buffers_and_finalizes_cross_validation_once` fail). A single
  assertion, or deriving the fallbacks from the dataclass fields, closes it. Framing it as
  an ungated training-behaviour change under the project's flag+tag+backfill rule is wrong:
  the rule protects the *old default*, and the old default for a run with no config file
  was already 300/150.

### F-10-3 - `experiment_imputation.ps1` drives the decode stage through classification's keys, writes the shared classification file, and closes with an f1_macro query

- **Verdict:** SOUND
- **Severity after review:** medium (unchanged)
- **Basis:** every premise verified line by line in the working-tree file.
  `Get-HyperparamsPath` (`:24-28`) returns `datasets/hiperparams/<base>/<dataset>.json` —
  the shared file, which `config.py:49-51` gives to classification unconditionally and to
  imputation only as a fallback. `New-HyperparamsJson` (`:32-38`) emits
  `EPOCHS_PRE / BATCH / LR_PRE / WEIGHT_DECAY_PRE / PROB_MASCARA / EPOCH_FINE / LR_FINE /
  WEIGHT_DECAY_FINE / LABELS` and no decode key, while `:60` runs
  `main.py --task imputation`. `complete_configuration`'s imputation branch
  (`config.py:97-105`) consumes `EPOCHS_DECODE / LR_DECODE / WEIGHT_DECAY_DECODE /
  LAMBDA_NUM / EVAL_MASK_RATE` — none of which the script writes — so `-FinetuneEpochs`
  (`:15`) is inert while `:52` prints `EPOCH_FINE=$FinetuneEpochs` as applied. `:99-100`:

  ```
  Compare in MLflow: tags.run_role = 'parent' and tags.missingness_percent = '0', grouped by tags.lr_scheduler
  Metric: cv/test/f1_macro/mean with cv/test/f1_macro/ci95_lower and ci95_upper
  ```

  no `tags.task`, and `f1_macro` is not in any imputation run's metric set. The committed
  version really does pass no `--task` (`git show HEAD:experiment_imputation.ps1 | grep task`
  exits 1) and the sole uncommitted edit is the `"--task", "imputation"` insertion
  (`git diff -- experiment_imputation.ps1`, a one-line hunk).
- **Correction:** one nuance in the critique's favour, and one correction to my own first
  reading. In its favour: the sweep is not wholly inert — `EPOCHS_PRE` *is* shared, both
  default datasets are `_00nan` with no `.imputation.json`, so the fallback fires and
  `EPOCHS_PRE = 200` does take effect; only the fine-tune knob is dead. The correction: the
  launcher is **committed on this branch** (101 insertions in `git diff --stat main...HEAD`),
  so anyone checking out `HEAD` gets a script named `experiment_imputation.ps1` that runs a
  classification sweep and writes classification configs; only the one-line
  `"--task", "imputation"` fix is uncommitted, and neither the config key set nor the
  summary query was adapted alongside it. Medium holds because the sweep still produces
  usable pre-training/schedule arms and the damage is a misreported knob plus a wrong-task
  query, not a wrong result — but not because the file is scratch.

### F-10-4 - An imputation run falling back to the shared file logs a `config_source` naming a file that supplied none of its decode parameters, and silently takes its `LR_SCHEDULER`

- **Verdict:** UNSOUND
- **Severity after review:** low (critic said medium)
- **Basis:** the premise reproduces exactly — probe P4 prints
  `config_source: datasets/hiperparams/credit-g/credit-g_20nan.json` alongside five decode
  values that are all defaults and `LR_SCHEDULER: 'cosine'` with no `--lr_scheduler` on the
  command line. `complete_configuration` (`config.py:77-112`) and
  `ArtifactWriter.write_hyperparameters` (`artifacts.py:35-71`) both log resolved values,
  not file contents, exactly as claimed.

  The reasoning is what does not follow. The critique's consequence is that the fallback is
  invisible — "every one of them logs a `config_source` that looks equally authoritative"
  and "nothing in the run record says so". Both are false against the run record the probe
  printed:

  - `config_source` is the **full posix path including the suffix**
    (`config.py:56`, `hyperparameters_path.as_posix()`). For an imputation run, a path
    ending `.json` rather than `.imputation.json` *is* the statement that the task-keyed
    file was absent and the shared one was read. The discriminator the critique asks to be
    added is already in the string it quotes.
  - Every value that differs between a fallback run and a defaults run is a logged
    parameter. Probe P4's two rows differ visibly: `EPOCHS_PRE 40` vs `300`, `DIM 192` vs
    `128`, `LR_SCHEDULER 'cosine'` vs `'cosine_legacy'`. Anyone comparing
    `impute/masked/impute_score` across variants sees the encoder differ in the same
    parameter table they are comparing in.

  What genuinely survives is much narrower: for a *partial* file you cannot tell which keys
  it supplied and which were defaulted. True — and equally true of classification since
  before this branch, so it is not something the imputation work introduced.
- **Correction:** the critique confuses "the record does not label the fallback as a
  fallback" with "the record does not show it". Only the first is true, and it is a
  legibility nit rather than a methodology defect that makes results uncomparable. ADR 0005
  decision 5's purpose — telling tuned runs from default ones — is served: `defaults` vs a
  path, and the path's suffix says which task's file. Low.

### F-10-5 - `EVAL_MASK_RATES_EXTRA` is in no run record and is deleted by `--promote_best`, against README's "a promoted file is complete"

- **Verdict:** SOUND
- **Severity after review:** low (unchanged)
- **Basis:** probe P3 confirms `from_mapping` parses the key
  (`eval_mask_rates_extra: (0.1, 0.5)`) and that `complete_configuration(..., "imputation")`
  drops it. `decoding.py:162` consumes it (`for extra in hyperparameters.eval_mask_rates_extra:`
  → `impute/masked/rate_<N>/*`), `README.md:273` documents it, and
  `promote_best_configuration` (`opt.py:340-345`) writes
  `complete_configuration(resolved, task)` verbatim — so promotion deletes the key from a
  file that held it, against `README.md:213` ("A promoted file is complete: every key the
  task uses, held values included").
- **Correction:** none to the finding; one addition to the fix. The key table is
  **duplicated**: `complete_configuration` (`config.py:87-112`) and
  `ArtifactWriter.write_hyperparameters` (`artifacts.py:41-71`) are two independent copies
  of the same seventeen entries, and `EVAL_MASK_RATES_EXTRA` is missing from both
  (`artifacts.py:57-64`). Fixing only `config.py` would leave
  `results/<ts>/hyperparameters.json` disagreeing with the MLflow params. The critique
  treats the second as a mirror; nothing makes it one.

### F-10-6 - `CLAUDE.md`'s comparison rule gained `tags.lr_scheduler` but not `tags.task`

- **Verdict:** SOUND
- **Severity after review:** low (unchanged)
- **Basis:** `git diff main...HEAD -- CLAUDE.md` is a single-line hunk adding only the
  `tags.lr_scheduler` clause; `README.md:369`, the reference that line points at, reads
  "Filter `tags.run_role = parent` and `tags.task` (`classification` or `imputation`)".
  `tags.task` is a real filterable tag, not only a param — `tracking.py:121-131` passes
  `task=task` into `execution_tags` for the run's tag set (`_log_execution_params` at
  `:382` logs it as a param *as well*), so the fix the critique proposes actually works.
  Probe P6 shows the mixture already exists in the store: `TRIDENT/credit-g` holds 15
  classification parents and 1 imputation parent, `TRIDENT/electricity` 7 and 4.
- **Correction:** one detail is understated rather than wrong. The critique blames
  `opt.py:385`, but every ordinary run keys the experiment the same way
  (`tracking.py:115`, `get_or_create_experiment(dataset_name)`), and the resulting
  experiment name is the **base** dataset (`TRIDENT/credit-g`), so `run_role = parent`
  returns a mixture spanning every missingness variant of a dataset, not just one variant.
  That widens the blast radius; it stays low because the remedy is a one-line docs edit.

## What this critique missed

- **`save_best_params` writes a study's raw partial mapping under the shared
  classification config's exact basename** (`opt.py:304-320`:
  `save_path = save_dir / f"{self.dataset_name}.json"`, for both tasks). The file sitting
  in the tree right now,
  `results/credit-g_20nan/optuna_20260911_011020/credit-g_20nan.json`, contains an
  *imputation* study's five sampled keys:

  ```json
  {"PROB_MASCARA": 0.4, "LR_DECODE": 0.0025386…, "WEIGHT_DECAY_DECODE": 0.0023168…,
   "DROPOUT": 0.30000000000000004, "LAMBDA_NUM": 2.738287…}
  ```

  Its name is byte-identical to the file `config.py:49-51` hands every classification run
  for that variant. Copy it into `datasets/hiperparams/credit-g/` — the obvious manual
  promotion, and the filename invites it — and every later classification run on
  `credit-g_20nan` trains at an imputation study's dropout and mask probability. That is
  precisely the coupling the function's own docstring says it exists to prevent
  ("for imputation it would also have retuned every later classification run, since that
  file names no task"); the docstring solved the directory and left the filename. It also
  answers the critique's own open question — a partial classification-named config in the
  wild exists, and the branch generates one per study.

- **Configuration resolution is CWD-relative and fails silently** (pre-existing, widened by
  the branch). `hyperparameter_file` returns `Path("datasets/hiperparams") / base / name`
  (`config.py:69`) — a relative path, resolved against the process working directory. Probe
  P4 is the proof: run from a scratch directory, the loader read a *fabricated* file placed
  there. A run launched from anywhere but the repo root resolves `config_source = "defaults"`
  and trains at 300/150 instead of the promoted configuration, with no warning. The launcher
  only works because it does `Set-Location $PSScriptRoot` (`experiment_imputation.ps1:22`);
  `results/` and `metrics/` are relative for the same reason. This is **not new** —
  `git show main:src/training/config.py:18` has the same relative literal — so it is out of
  the critique's declared scope of the imputation work. It is worth naming anyway because
  the branch turned one silently-missable file into two (`config.py:49-51` now probes a
  task-keyed candidate first), and because F-10-4's whole argument is about what
  `config_source` means: a missed CWD makes it mean `defaults` when a tuned file exists
  three directories away.

- **The flat-metrics failure the tree actually exhibits is Optuna-driven, not cross-task**
  (see F-10-1's correction). `opt.py:208` and `opt.py:399` point every trial at the real
  `metrics/` directory, so an N-trial study leaves one single-split row where a CV run's
  folds used to be. The critique buries this in a parenthesis as "pre-existing on `main`"
  and then reconstructs a cross-task story around a row that a trial wrote.

- **The `reduced` profile's held epoch budget was worth checking and checks out.** I
  expected to find that "reduced" silently costs 5-10x a `full` trial (probe P5:
  `EPOCHS_PRE 300 / EPOCHS_DECODE 150` held vs `40 / 50` sampled). ADR 0005 anticipates it
  exactly — decision 2's held table names both values with reasoning, and `:50-54` and
  `:124` quantify the per-trial cost. Recording the negative because it is the single place
  on this branch where F-10-2's changed fallback governs a real run, and the critique
  reached the opposite conclusion about it without looking.
