# Consolidated report — the imputation work on `feat/imputation-task`

_Thirteen adversarially-verified critique dimensions, consolidated. 2026-09-11._

Every item below is attributed to the critique file that produced it and the verdict file
that survived or corrected it. Severities are **after** adversarial review, not as claimed.
Findings whose verdict was `severity: none` are in §5, not dropped.

> **Revised 2026-09-15 — see [`ADDENDUM-2026-09-15-optuna-results.md`](ADDENDUM-2026-09-15-optuna-results.md).**
> Twelve Optuna studies and twenty-four five-fold comparison runs now exist, where this
> report had five studies and no comparison. The addendum revises §1's characterisation of
> the search objective, §3.3, and §4's comparison subsection, and adds five findings to the
> ledger that the new runs speak to. **F-07-1's conclusion is upheld and its argument
> replaced**; nothing in this report is withdrawn. The text below is left as written on
> 2026-09-11.
>
> **Revised again 2026-09-18 — addendum §8.** Nine further runs plus a second overnight
> batch: the typing refactor is bit-identical (§8.1); a five-seed noise floor exists
> (§8.2: one run carries ±0.013 to ±0.027); the search mis-ranks its own trials by a median
> 0.036 on the study split, of which about a fifth survives five-fold for the single-split-best
> trial (§8.14: ≈0.016 on `credit-g_20nan`, inside noise on `credit-g_40nan`; that trial is
> not distinguishable from the defaults) — and no validation-side statistic recovers it
> (§8.3, §8.8);
> `full` and `reduced` are disjoint profiles, not nested (§8.4); D-1 reproduces at HEAD to
> the cell (§8.9); the one 95% separation does not replicate (§8.10); and the **committed
> `credit-g_20nan` promoted file scores worse than the defaults** (§8.11: Welch p ≈ 0.01,
> paired p ≈ 0.008, three of three shared seeds); `credit-g_40nan` leans the same way on four
> of four seeds but stays "hold" (p ≈ 0.04, fragile to any one seed, reached past the
> pre-stated stopping seed).

---

## 1. Verdict

**The decode machinery is sound; the experiment that is being run around it is not.** Exactly
one *bug* survives verification at high severity — everything else at high or critical is
methodology, which means the code mostly does what the ADRs say and the problem is what the
numbers are being asked to mean.

The single most consequential finding is that **the Optuna search cannot identify the
configuration it promotes, and the four promoted files in `datasets/hiperparams/` are
therefore not evidence-backed** (`07-optuna-imputation.md` F-07-1, SOUND/critical). The
objective `validation/impute/masked/impute_score` is computed on 283 cells for
`credit-g_20nan` and 179 for `credit-g_40nan`; on the all-categorical tables it collapses to a
lattice of one-cell steps, so both finished `kr-vs-kp` studies produced 20 distinct values
across 40 trials and **both promoted configurations were decided by `study.best_trial`'s
tie-break on trial number**. The verifier then went further than the critique and settled its
own open question from data already in `mlflow.db`: rank-correlating each trial's objective
against `test/impute/induced/impute_score` — the population ADR 0005 decision 6 declares the
headline — gives ρ = **+0.124** on `credit-g_20nan` against a ±0.31 noise floor at n=40 (the
matching low reading on `kr-vs-kp_40nan`, ρ = **+0.071**, is against `test/impute/masked`; its
induced ρ is +0.372, borderline — the full table is in §3.3), and the promoted winners rank
**31/40, 21/40, 33/40, 1/40 and 2/20** in their own studies on that metric (single-seed,
single-split trial runs: the verifier
calls the direction real and the precision loose). So `credit-g` is not the safe case: its
objective resolves, never ties, and still selects near the middle of the pack.

Three further things are load-bearing. (a) `evaluation_mask` reuses the *training* corruption
helper, so the realised exam rate is 0.2085/0.1685/0.1548/0.1572/0.2500 across the `credit-g`
ladder at a configured 0.2 and rises monotonically to 0.52 on `electricity_80nan` — which
falsifies ADR 0004:222's verbatim claim that `cv/test/impute/masked/impute_score/mean` "is
comparable across the ladder", a claim the repo's *own ticket 05* already recorded as false
(`01`, `03`, `04`, `09`). (b) `decoding.py:262` feeds a CSV-ordered mask where an
embedder-ordered one is required, so `--score_null_path` scored 76.4% observed cells on
`credit-g` — and ADR 0004 decision 11 records those numbers as the measured outcome of a
pre-registered decision gate (`04`, `05`, `09`; CONFIRMED/high in all three). (c) Pre-training
ends several times worse than a twenty-vector constant lookup on its own objective while its
logged loss falls ~98% — NMSE 4.21 on the critic's fold, **7.70** on the verifier's independent
reproduction, against loss drops of 98.5% and 97.4% — and *the critic's* paired ablation, which
the verifier did **not** re-run, could not detect any benefit from 298 of the 300 default
epochs (`13`, SOUND/high) — no reported imputation number is invalidated by this, because the
decode stage trains on real cell values, but two ADRs protect the stage as load-bearing and
every reduced trial spends about half its wall clock there.

What is **clean**, and verified: the decode stage does not leak. Train/validation/test indices
are disjoint, the `_00nan` sibling is touched only after `best_state` is restored, the
checkpoint is selected on validation, `impute_score` is exactly scale-invariant to the
transductive `StandardScaler`, the induced gaps are genuinely MCAR at a fixed count per column
with zero NaN in all nine `_00nan` siblings, and no `cv/test/*` metric key is shared between
the two tasks. The variants are also fully reproducible — `datasets/generate_splits.py` is
tracked and seeded and regenerates the shipped CSVs byte-identically (3/3 checked), which
refutes the largest provenance claim in the review.

The honest summary: **sound-but-for-one-high-severity-bug in the code, methodologically
compromised in the tuning and comparison protocol.** Nothing published so far rests on the
broken parts — the store holds two finished imputation parents at `plateau`/`cv_folds 2` and
none of the twelve ADR 0005 decision-6 comparison runs exists yet. That is the good news: this
is the cheapest moment the fixes will ever have.

---

## 2. Confirmed defects

Every finding whose verdict was **CONFIRMED**, ordered by severity after review. Duplicates
found by more than one dimension are merged into one entry citing all of them. D-3 is the one
entry here that carries no CONFIRMED verdict line, because no critique filed it: it was
reproduced by two verifiers in their "What this critique missed" sections, and its severity is
flagged as a synthesis judgement in place.

### D-1 — `_select` hands the decoder a frame-ordered mask where a token-ordered one is required (**high**)

- **Where:** `src/training/decoding.py:262`, against `src/embedder.py:189`
- **Sources:** `04-ground-truth-protocol.md` F-04-1, `05-embedder-tokens.md` F-05-1,
  `09-test-adequacy.md` F-09-1 — CONFIRMED/high by all three verifiers, independently
  reproduced against the real classes.

`TabularEmbedder.encode` fixes `masked_positions` in embedder order at `src/embedder.py:189`,
`columns = list(self.categorical_columns) + list(self.numerical_columns)`, and every consumer
reads it that way (`src/models.py:170,185`; `src/training/decoding.py:288,307`).
`_score_induced_missing` then overwrites that field with `test_frame.isna().to_numpy()`, which
is CSV header order. The orders coincide only when the declared categorical columns are a
CSV-order prefix — true for the six all-numerical tables and for `kr-vs-kp`, **false for
`credit-g` (19/20 positions misaligned) and `electricity` (2/8)**.

Measured three ways. A real `TabularEmbedder` + real `_select` round trip on a mixed frame
gives exactly transposed marks. On the real `credit-g_20nan` feature frame, 4000 cells are
missing, `_select` marks 4000, and the overlap is 944 = **23.6%** — chance overlap at a 20%
rate. Calling `decoding._score_induced_missing` directly on a real `prepare_dataset` +
`load_complete_sibling` + `build_folds` + real `TridentDecoder`: `induced` scored 1976 cells,
1976 of them actually missing; `induced_null_token` scored 1976 cells, **478 (24.2%)** actually
missing.

Nothing raises, because the cell counts match. The one guard,
`tests/unit/test_training_decoding.py:246`, asserts
`int(through_null.sum()) == int(through_mask.sum())` — a count, invariant under exactly this
permutation — and the shared `_complete_frame()` helper is ordered categorical-first, so the
permutation is the identity in the fixture.

Two corrections from the verifiers sharpen it rather than soften it. `TridentDecoder.forward`
computes loss only at masked positions (`src/models.py:161-190`), so the head has never been
optimised on a context vector from an observed position: the `credit-g` null-path numbers are a
trained readout on an out-of-distribution input, **direction indeterminate**, not simply
"flattered". And `_assert_row_aligned` guarantees the sibling equals the variant at every
observed cell, so at a wrongly selected position the "truth" is the value sitting in the
model's own input token.

Severity is held at high despite the flag being off by default and the metric never ranking a
fold: `git log -S"_select" -- src/training/decoding.py` returns only `345574b` (2026-09-10
19:10) and ADR 0004's decision-11 verdict table landed in `74229fd` the same evening (19:46),
so the four `credit-g` rows at `docs/adr/0004-imputation-decoder-task.md:176-179` went through
this code. Decision 11's own criterion requires a win "on at least two datasets"; only the
all-categorical `kr-vs-kp` rows were validly measured, so the recorded conclusion rests on one
dataset, and specifically not on the mixed table where `[MASK]`-vs-`[NULL]` is hardest to
justify.

**Fix direction:** build the selection tensor in embedder order —
`test_frame[list(cat_columns) + list(num_columns)].isna()` — or expose the order on
`EncodedTable` so no caller has to guess it (`_select` is the only place in the decode path
that addresses columns positionally). Change the guard from a count to set equality of
`(row, column)` pairs, re-order one fabricated frame in `test_training_decoding.py` so
categoricals are not first, and re-measure or strike ADR 0004:176-179's `credit-g` rows.
`electricity`'s induced path is unreachable until D-3 is fixed, so the two travel together.

### D-2 — `experiment_imputation.ps1` at HEAD is byte-identical to `experiment.ps1` and runs classification (**medium**)

- **Where:** `experiment_imputation.ps1:60` (committed in `9a38b90`)
- **Sources:** `08-experimentation-design.md` F-08-1 (CONFIRMED/medium),
  `10-config-plumbing.md` F-10-3 (SOUND/medium). Symptoms F-08-6, F-08-7 and F-08-10 below are
  the same unadapted copy; the verifier of 08 says so explicitly and asks that they be counted
  once.

`git show HEAD:experiment_imputation.ps1 | diff` against `experiment.ps1` is empty. The only
thing making it an imputation launcher is the **uncommitted** worktree edit, which inserts
exactly `"--task", "imputation"` at line 60. Header, usage lines, datasets, JSON builder and
the closing `cv/test/f1_macro` recipe are all unadapted.

The digest claim that the committed file *produced* eight classification runs does not hold:
the newest result directory for either default dataset is `20260909_21xx`, two days before the
commit (2026-09-11 00:21:58), `.github/workflows` does not exist, none of the three launchers
is on `main`, and ticket 05's Comments record the file as "work in progress in the working
copy". So it is a real committed defect that silently runs the wrong task, already fixed in the
worktree, never executed.

Even fixed, three things are wrong and are verified separately:

- **F-08-6 / F-10-3 (CONFIRMED/low, SOUND/medium):** `Get-HyperparamsPath` writes
  `datasets/hiperparams/<base>/<dataset>.json` — the **shared** file that
  `src/training/config.py:49-51` gives classification unconditionally and imputation only as a
  fallback. Line 52 prints `EPOCH_FINE=$FinetuneEpochs` as applied while the decode stage reads
  `EPOCHS_DECODE`/`LR_DECODE`/`WEIGHT_DECAY_DECODE`/`LAMBDA_NUM`/`EVAL_MASK_RATE`
  (`src/training/config.py:97-105`), so `-FinetuneEpochs` is inert — masked by its 150 default
  equalling `decode_epochs`' default. On a nan variant with a promoted file, the sweep's JSON
  is shadowed entirely. It is latent today only because the script's defaults are `_00nan`
  variants with no `.imputation.json`, where `config_source` reads `defaults`; `EPOCHS_PRE` is
  shared so *that* knob does take effect.
- **F-08-7 (CONFIRMED/low):** the datasets are `vehicle_00nan` and `credit-g_00nan`;
  `load_complete_sibling` returns `None` for both (verified by calling it), so
  `src/training/decoding.py:174` skips the induced block and the ADR 0005 headline population
  **does not exist**. Lines 99-100 then print a `cv/test/f1_macro` recipe that no imputation
  run emits, with no `tags.task` filter, so it returns the *other* task's runs.
- **F-08-10 (CONFIRMED/low):** the backup-and-write loop is lines 43-53, `try` opens at 57 and
  the restoring `finally` at 80-93, with `$ErrorActionPreference='Stop'` set at 21. A failure on
  the second dataset after the first is written terminates before any `finally` exists. Worse
  than the critique states: `git ls-tree -r` on `main` and `HEAD` under `datasets/hiperparams`
  both return nothing, so every `$backups` entry is `$null` and even the successful restore is
  a delete — a crash leaves a file nothing can distinguish from an intentional configuration,
  which then feeds the fallback above.

**Fix direction:** adapt the file rather than annotate it — task-keyed config path built from
`complete_configuration(..., "imputation")`, `-DecodeEpochs` instead of `-FinetuneEpochs`, nan
variants as defaults, the imputation ranking metric plus `tags.task` in the closing recipe, and
the setup loop inside the `try`.

### D-3 — Two shipped variants crash inside the shipped configuration space before scoring an induced cell (**medium**)

- **Where:** `src/training/decoding.py:264-265` → `src/embedder.py:177`
- **Sources:** `01-decode-stage.verdict.md` "What this critique missed" (the `kr-vs-kp` case,
  reproduced and filed by nobody); `05-embedder-tokens.verdict.md` "missed" item 3 and
  `09-test-adequacy.md` F-09-4 (the `electricity` case, reproduced).
- **Severity note:** no verifier issued a verdict line for this, because no critique filed it.
  The 01 verifier wrote "I would file it low-to-medium rather than high" for the `kr-vs-kp`
  half; the 05 verifier reproduced the `electricity` half and assigned it no severity at all.
  **Medium here is a synthesis judgement**, taken because two independent triggers share one
  mechanism, F-09-4 (SOUND/medium) turns on the `electricity` crash, and F-04-4 (SOUND/low) is
  the same dtype mismatch seen from the guard's side.

Two independent triggers, one mechanism: the embedder's `LabelEncoder`s are fitted on the
**variant** (`src/embedder.py:109-114` via `src/training/pretraining.py:29`) and the `_00nan`
sibling is transformed against them at `src/embedder.py:177`, reached from
`src/training/decoding.py:264`.

- **`electricity_*`:** `day` is `int64` in `_00nan` and `float64` in `_20nan`, so the variant
  vocabulary is `['2.0'..'6.0','[MASK]','[NULL]','nan']` and the sibling stringifies to `'2'`.
  `ValueError: y contains previously unseen labels: '2'` — reproduced by driving the real
  `TabularEmbedder`. `--task imputation` on `electricity_20nan` therefore crashes before
  scoring an induced cell, and three `electricity` imputation parents in `mlflow.db` are
  `FAILED` consistently with this.
- **`kr-vs-kp_40nan`:** a *category erased by the generator*. `spcop='t'` occurs in exactly one
  row of `kr-vs-kp_00nan` (row 2891) and the 40% draw removed it, so the variant vocabulary is
  `['[MASK]','[NULL]','f','nan']` and the sibling's `'t'` raises. Row 2891 falls in the test
  split of fold 4 of 5 at seed 42, so `kr-vs-kp_40nan --cv_folds 5 --seed 42` completes folds
  1-3 and then raises **after** pre-training and decode training have run for that fold.

`_assert_row_aligned` (`src/training/data.py:112-129`) certifies both as aligned: it compares
values only, through an object-dtype `to_numpy()` where `1.0 == 1` is `True`, and checks
neither of the two properties scoring actually depends on — encodability into the variant's
vocabulary, and a non-NaN sibling wherever the variant has a gap
(`04-ground-truth-protocol.md` F-04-4, SOUND/low as a design point).

**Fix direction:** extend the guard to what scoring needs — per-column dtype equality or a
shared canonicalisation, a refusal when the sibling is NaN where the variant is NaN, and a
check that the sibling's categorical values are a subset of the variant's observed categories.
That last one turns both crashes into a message naming the column. Note that fixing them makes
`electricity` the second live dataset for D-1.

### D-4 — Pre-training's re-rolled validation mask drives real `plateau` learning-rate decisions (**medium**)

- **Where:** `src/training/pretraining.py:70-75` and `:120`; `src/training/schedulers.py:90-93,103-104`
- **Source:** `13-pretraining-transfer.md` F-13-5 — CONFIRMED/medium, every store figure
  reproduced against a scratchpad copy of `mlflow.db`.

The decode stage draws its validation mask once per fold and says why in its own module
docstring — "so that the loss moves only when the model does"
(`src/training/decoding.py:3-5`, implemented at `:71-82`). Pre-training re-rolls **both** masks
every epoch and feeds the result to `scheduler.after_epoch(...)`, which under
`--lr_scheduler plateau` is `ReduceLROnPlateau(mode='min', factor=0.5, patience=10).step(...)`.

Four of the six imputation parents in the store ran `plateau`. Fold `e764df47`
(`spambase_20nan`) halved **11 times** at epochs 154, 178, 189, 209, 227, 238, 249, 260, 271,
282, 293 — from epoch 238 onward every 11 epochs exactly, i.e. `patience` + 1, meaning the loop
never saw a new best over the last 62 epochs — finishing at **1.66e-07**, 2048x below the
configured 3.4e-4. The noise floor is measurable on that same fold: across the 29 epochs where
the rate is under 1e-6 and the model is effectively frozen, `pretrain/val_loss` still ranges
0.01265..0.01762, a 9.2% CV and a 1.39x spread, produced entirely by the mask re-roll. The
schedule was chasing a lucky mask draw. Both `credit-g` folds triggered zero halvings, so it is
dataset-dependent.

The verifier adds the fact that settles who owns it: `src/training/schedulers.py` **does not
exist on `main`** (`git show main:src/training/schedulers.py` fails; its adding commit `87d8c82`
is on no remote branch). The noisy signal is inherited, but the consumer that turns it into a
training decision is this branch's own addition — made in the same branch that gave its new
decode stage a fixed validation mask and wrote down the reason.

**Fix direction:** draw pre-training's *validation* mask once per fold exactly as
`src/training/decoding.py:71-82` does; keep the training-mask re-roll, which is the pretext
task. That is a behaviour change to a protected stage and wants the flag-and-tag treatment.
The cheap intermediate is to average the validation loss over several mask draws before handing
it to the scheduler, or to document that `plateau` is unsupported for pre-training.

### D-5 — The flat per-dataset metrics file is task-blind and last-writer-wins (**medium**)

- **Where:** `src/training/artifacts.py:80-88`, called from `src/training/runner.py:47,157`
- **Sources:** `06-artifacts-tracking.md` F-06-1 (CONFIRMED/medium),
  `10-config-plumbing.md` F-10-1 (SOUND/medium), `08-experimentation-design.md` F-08-8
  (SOUND/medium — the Optuna half).

`write_metrics` writes a second copy of every run's fold table to
`metrics_dir / f"{dataset_name}_metrics.csv"` with `frame.to_csv(root_path)` — a plain
overwrite. The name carries the dataset variant and nothing else; the rows are
`{"fold", "dataset", **result.metrics}` with no `task` column. `src/training/config.py:61-69`
keys the *config* file by task and `results_dir` is timestamped, so this is the only
dataset-only key in the codebase whose column set varies with the task.

Five files in `metrics/` are already overwritten — `credit-g_20nan`, `credit-g_40nan`,
`kr-vs-kp_20nan`, `kr-vs-kp_40nan`, `spambase_20nan` — each now holding **one**
`fold=single_split` row carrying a `validation/impute/...` family, i.e. an Optuna trial. The
overwrite that actually happened is Optuna-driven rather than cross-task: `opt.py:206` isolates
trial *results* into `temp_trials` but `opt.py:208` passes `metrics_dir` straight through, and
the launcher never sets it, so each of 40 trials overwrites the file and the last one wins,
destroying a same-task CV run as well as the classification one. In `-Compare` mode the
defaults arm runs second and overwrites the promoted arm's rows. AGENTS.md asks explicitly to
preserve generated `metrics/`.

Two corrections keep this at medium rather than high. The timestamped copies survive —
`results/credit-g_20nan/20260909_065141/metrics.csv` and its four siblings all still hold 3
classification fold rows — and each of the five variants has exactly one
`run_role=parent, task=classification` run in the store carrying `cv/test/f1_macro/mean`
(0.6292, 0.6361, 0.8843, 0.8009, 0.9064, all `fold_count=3`). And grepping every `.py`/`.ps1`/
`.ipynb`/`.md` for `_metrics.csv` finds only the writer: no programmatic consumer reads the root
file back, so the harm is a human opening `metrics/` and getting a different question answered.

**Fix direction:** the filename is the run's identity claim, so it has to carry everything that
changes the columns — task *and* a run discriminator, so a trial cannot land on a comparison
run's file. Give trials their own metrics directory beside `temp_trials` in `opt.py`, and have
`imputation_studies.ps1 -Compare` pass a per-arm `--metrics_dir`.

### D-6 — `--score_null_path` writes every induced cell twice and mislabels it in the preview (**low**)

- **Where:** `src/training/artifacts.py:327` (`_render_preview`) and `:359`, fed by
  `src/training/decoding.py:178,190`
- **Sources:** `06-artifacts-tracking.md` F-06-3 (CONFIRMED/low);
  `05-embedder-tokens.verdict.md` "missed" item 1 for the label.

Both induced populations select the same cell set — `as_mask=True` builds
`test_frame.mask(test_frame.isna(), "[MASK]")` and `src/embedder.py:192` sets
`masked_array[:, index] = (df[col] == "[MASK]")`, while `as_mask=False` sets the same tensor
explicitly at `src/training/decoding.py:262`. Both concatenate into `cells`. Running the real
`ArtifactWriter.write_imputation_preview` reproduces the defect byte for byte: a header reading
"row 0 — 3 cell(s) filled in" for two distinct cells, `duration` printed twice in one block with
identical "model saw" labels and different imputed values, and a ledger of 3 rows over 2
distinct `(row, column)` pairs.

The label is independently wrong: `src/training/artifacts.py:359` renders
`"[NULL]->[MASK]" if population.startswith("induced") else "[MASK]"`, and the null-path
population is named `induced_null_token` — so the one population whose entire purpose is that
the model saw `[NULL]` **and it was not converted** is labelled `[NULL]->[MASK]`. That is
exactly the artifact a human would use to sanity-check D-1, and it would have hidden it.

Blast radius is small and worth stating: no metric is affected (`score_cells` runs at
`src/training/decoding.py:150` *before* the induced concat at `:178`, and
`runner._per_column_scores` groups by `population` first), and the flag has never been used —
every `*_cells.csv` on disk has distinct `(row, column)` equal to its row count.

**Fix direction:** `population == "induced"` for the label; pivot the preview on
`(row, column, population)` with the population named rather than prefix-tested; count distinct
cells in the block header.

### D-7 — Extra evaluation rates collide silently on a rounded percent key (**low**)

- **Where:** `src/training/decoding.py:169-172`
- **Sources:** `01-decode-stage.md` F-01-5 and `06-artifacts-tracking.md` F-06-7 — CONFIRMED/low
  by both.

`prefix = f"impute/masked/rate_{round(extra * 100)}"` feeding a `metrics.update`, which is
last-wins. `0.02`, `0.015`, `0.024` and `0.025` all map to `rate_2`; `0.004` to `rate_0`; `0.2`
and `0.205` to `rate_20`; banker's rounding makes `0.115` and `0.125` collide too. Nothing
validates the tuple at configuration time (`src/training/types.py:171-176` only coerces to
float), and `_validate_final_metrics` cannot catch it because every fold collides identically.

Latent, and I checked rather than assumed: `grep -rn "EVAL_MASK" datasets/hiperparams/` returns
exactly four lines, each setting only `EVAL_MASK_RATE: 0.2`, and those four `.imputation.json`
files are the entire shipped imputation config surface. The low-end collapse and the ~0.87
ceiling that 01 reports in the same finding are D-9's root cause restated in the extra-rate
family and must not be counted twice.

**Fix direction:** key on the exact rate (`rate_0p025`) and refuse a duplicate prefix at the
point it is built.

### D-8 — `EVAL_MASK_RATES_EXTRA` is in no run record and is deleted by `--promote_best` (**low**)

- **Where:** `src/training/config.py:97-105` and `src/training/artifacts.py:57-64`
- **Sources:** `10-config-plumbing.md` F-10-5 (SOUND/low) and `06-artifacts-tracking.md` F-06-8
  (CONFIRMED/low) — the 06 verifier notes this is one defect reported in three places
  (`01-decode-stage.verdict.md:318` is the third) and asks for one entry.

Both key tables list `EPOCHS_DECODE, LR_DECODE, WEIGHT_DECAY_DECODE, LAMBDA_NUM,
EVAL_MASK_RATE` and stop, under docstrings claiming "the values this run used, and only those"
and "a promoted file names every value the task will train with"; `README.md:213` repeats it as
"A promoted file is complete". But `src/training/types.py:171-176` parses the key
(probe: `eval_mask_rates_extra -> (0.1, 0.5)`) and `src/training/decoding.py:162` consumes it,
and `opt.py:344-345` opens the task's config with `'w'` and writes
`complete_configuration(resolved, task)` — so a hand-written extras list is erased by the next
promotion. All four promoted files on disk carry `EVAL_MASK_RATE 0.2` and no extras, exactly as
predicted. `--score_null_path` is likewise a CLI flag and `TrainingRequest` field with no param
and no tag.

**Fix direction:** add the key to **both** tables — fixing only `src/training/config.py` would
leave `results/<ts>/hyperparameters.json` disagreeing with the MLflow params — and log
`score_null_path` as a run tag.

### D-9 — Nearby confirmed low-severity defects

Grouped because each is a one-line fix with a verified reproduction and no current victim.

- **An all-missing categorical column builds `nn.Linear(d, 0)` and crashes `predict` after the
  whole stage has run** (`02-decoder-model.md` F-02-4, CONFIRMED/low). Reproduced end to end:
  vocabulary `['[MASK]','[NULL]','nan']`, `valid_ids []`, `out_features 0`, `local_of
  [-1,-1,-1]`, the only signal a `UserWarning: Initializing zero-element tensors is a no-op`,
  and `IndexError: argmax(): Expected reduction dim 1 to have non-zero size` at
  `src/models.py:223`. `mean_mode_baselines` raises first on `mode().iloc[0]`
  (`src/training/imputation_metrics.py:136`, called at `src/training/decoding.py:134`), which
  shadows it — both need fixing together or the fix moves the crash 11 lines later.
  `prepare_dataset` has no all-null guard, so it is genuinely reachable from `main.py`.
  Independently: `cross_entropy` at `src/models.py:176` has no `ignore_index`, and a `-1` target
  raises; the invariant that prevents it is `src/utils.py:62`
  `dynamic_mask[null_values] = False`, defended by a comment and nothing else.
- **`encode` has a third representation of a missing cell** (`05-embedder-tokens.md` F-05-2,
  CONFIRMED/low). `split_numeric_and_special`'s non-object branch (`src/utils.py:124-126`)
  keeps literal `NaN` with both flags `False`, and `as_category_strings`
  (`src/embedder.py:27`) yields the string `'nan'`, which `TabularEmbedder.__init__` fits as an
  ordinary class. `src/training/pretraining.py:66-67` encodes the raw frame for its clean
  targets: measured on a 50-row `credit-g_20nan` slice, 62 NaN in `num_values`, 496 NaN in the
  embedding output, `{'nan'}` for `checking_status`'s missing cells. Cut from medium to low
  because both escape routes were checked: the NaN cannot reach the loss (`preprocess_table`
  zeroes `dynamic_mask` at nulls, measured overlap of `[MASK]` with NaN positions: 0) and
  cannot spread (the target is `embedder(original)` only, `embedder.forward` is strictly
  per-column, and the whole target path is under `no_grad`). It is a latent trap plus an
  untrained vocabulary entry that `src/models.py:12` has to blacklist — and the fix moves every
  `_XXnan` pre-training target, so it is a protected behaviour change with no numerical payoff
  today.
- **`score_cells` has no baseline-coverage contract** (`09-test-adequacy.md` F-09-9,
  CONFIRMED/low). A baseline mapping missing a scored column gives `impute_score = nan` on the
  numerical side (`column.map(baselines)` → NaN → `naive_rmse`). The critique's categorical
  demo was degenerate and the verifier rebuilt it: with the covered column's baseline wrong,
  the full baseline gives 1.0 and omitting `tier` gives 0.5 — an inflated denominator makes the
  model look twice as good. A missing *key* is unreachable from the pipeline; a present key
  with a **NaN value** is not — an all-missing numerical train column gives
  `float(series.mean()) = nan` with identical propagation, and the NaN then reaches
  `_diagnostic_roles`, whose min over NaN picks an arbitrary best fold.
- **fANOVA silently drops `HEAD_DIM`** (`07-optuna-imputation.md` F-07-5, CONFIRMED/low).
  Reproduced on the installed optuna 4.9.0 with `opt.py:115-116`'s exact construction: ranked
  knobs `['DROPOUT','HEADS']`, intersection space the same, sampled knobs
  `['DROPOUT','HEADS','HEAD_DIM']`. No exception, so `opt.py:360-362`'s
  `except (ValueError, RuntimeError)` never fires and `importance.json` is one key short.
  Cut from medium because the reduced profile never samples `HEAD_DIM` and no full-profile
  study exists; it bites only the deferred pilot ADR 0005 nominates to check the reduction,
  which would then "find" that model width does not matter, by omission.
- **`opt.py` hardcodes `label_column='class'` for trials and the retrain** (`07` F-07-8,
  CONFIRMED/low). `opt.py:422` computes `mixed_columns` — and therefore whether `LAMBDA_NUM`
  gets a dimension — from `getattr(args,'label_column',None) or 'class'`, honouring the flag;
  `opt.py:203` and `opt.py:544` then set the literal on the trial and retrain namespaces. The
  parser's default is `None` (`src/training/config.py:214`), so the literal overrides a
  user-supplied value. Two paths in one file disagree about the same input. Nothing today
  trips it — every processed dataset uses `class`.
- **ARCHITECTURE states a false universal about head width** (`11-documentation-contract.md`
  F-11-2, CONFIRMED/low). `docs/ARCHITECTURE.md:355,361-365` says every categorical vocabulary
  holds `[MASK]`, `[NULL]` and `"nan"` and the head is `Linear(d, V_col − 3)`. Measured by
  reading `categorical_heads[key].out_features` off a real `TridentDecoder`: `V − out_features`
  is **2** on every column of `kr-vs-kp_00nan` (36/36) and `credit-g_00nan` (13/13), 3 on the
  `_20nan` variants — `"nan"` enters only when a cell is missing, and all nine `_00nan` tables
  have zero NaN. Implementing the documented width on `kr-vs-kp_00nan` gives a **one-output
  head on 35 of 36 columns** (`nunique` histogram `{2: 35, 3: 1}`). The shipped code is right
  (`src/models.py:118-127` uses a membership test over `encoder.classes_`) and so is ADR 0004
  decision 3, which keeps the qualifier; only the doc is wrong.
- **A run's artifact tree and its MLflow run are joined only by a timestamp drawn twice**
  (`12-reproducibility-provenance.md` F-12-5, CONFIRMED/low).
  `src/training/artifacts.py:29` takes its own `datetime.now()`;
  `src/training/tracking.py:120` takes a second one after `setup_mlflow()` and
  `get_or_create_experiment()` round-trip a 140 MB SQLite store. Both finished imputation
  parents are skewed by one second (`results/credit-g_20nan/20260910_195312` vs
  `impute_credit-g_20nan_20260910_195313`; same for `spambase_20nan`). Nothing else joins them —
  `mlflow.log_artifact` uploads content, never paths, and `provenance.json` carries no run id.
  It matters more on this branch than before, because `_log_diagnostic_children`
  (`src/training/tracking.py:212-234`) replays only `best_fold`/`worst_fold`, so three folds of
  a 5-fold run have cell ledgers *only* under the gitignored `results/` tree behind a
  mismatched directory name.

---

## 3. Design and methodology critiques judged sound

Grouped by what a fix would change. Findings that came back **UNSOUND with surviving residue**
appear here under their *corrected* claim, with the refuted headline noted in one sentence;
those that came back UNSOUND with nothing left are in §5.

### 3.1 The evaluation mask is the training corruption helper, and one ADR sentence rests on it

**Severity: high.** Sources: `01-decode-stage.md` F-01-1 (SOUND/high),
`04-ground-truth-protocol.md` F-04-2 (SOUND/high), `03-imputation-metrics.md` F-03-1
(SOUND/medium), `09-test-adequacy.md` F-09-2 (SOUND/low). One defect, four measurements.

`evaluation_mask` (`src/training/data.py:137-148`) is a seeded wrapper around
`preprocess_table`, which does two things no evaluation mask should do: it deflates the rate by
the row's null density (`src/utils.py:55`, `p_dynamic = p_base * (1 - prop_nulls)`) and then
**forces at least one mask into every row that drew none** (`src/utils.py:64-74`). The
arithmetic is the whole finding: for null fraction `q` over `C` columns, expected masks per row
are `p_base*(1-q)*C*(1-q)` over a `C*(1-q)` observed-cell denominator, giving `p_base*(1-q)` —
a falling term — plus `1/(C*(1-q))` per floored row, a term that grows without bound. They
cross somewhere in the middle.

Measured through the real path (`prepare_dataset` → `build_folds(5, 42)` → fold 1 test frame →
`evaluation_mask(., 0.2, 42, 1)` → exactly `src/training/decoding.py:154-157`'s definition):

| variant | `_00nan` | `_20nan` | `_40nan` | `_60nan` | `_80nan` |
|---|---|---|---|---|---|
| credit-g realised | 0.2085 | 0.1685 | 0.1548 | 0.1572 | **0.2500** |
| electricity realised (8 features) | 0.2230 | 0.2198 | 0.2437 | 0.3246 | **0.5242** |

Down-then-up on a 20-column table, monotonically up to **2.4x** on an 8-column one. The floor's
share of the scored cells on `credit-g` at nominal 0.2 is 0.2% / 2.1% / 12.7% / 44.3% /
**81.4%**, so at `_80nan` the estimand has changed character: "a near-uniform 20% sample"
becomes "one uniformly chosen observed cell per row" for four rows in five. A real
`kc2_20nan --cv_folds 2` run reported `impute/masked/realised_rate = 0.1729` for a configured
0.2. At `_60nan` and `_80nan` the extra-rate ladder is absorbed too: a 4x change in the nominal
rate moves the realised exam on `credit-g_80nan` by 7% (198 → 212 cells).

The consequence is a document defect, precisely locatable.
`docs/adr/0004-imputation-decoder-task.md:222` says verbatim that "on any variant,
`cv/test/impute/masked/impute_score/mean` is comparable across the ladder", and
`docs/wayfinder/imputation-optuna-reduced/issues/06-importance-and-comparison.md:35` propagates
it. It is not: `impute_score` is a ratio to a *context-free* mean/mode baseline, so hiding a
larger share of a row removes context from the model and none from the baseline — exam
difficulty moves the score directly. The sharpest framing is the one the 03 verifier found and
the critiques missed: **`docs/tickets/imputation-decoder/issues/05-evaluation-data-support.md:65-77`
already carries this table and the sentence "a self-masked score at rate 0.2 is not comparable
across the ladder"**, and `06-decode-stage.md:47-51` repeats it. The prescribed mitigation —
log the realised rate — was implemented. So ADR 0004:222 contradicts its own ticket, and
ADR 0005 inherited the wrong half.

Two things must **not** survive into a fix. The code's explanation at
`src/training/decoding.py:152-153` ("which falls as missingness rises because it never hides an
already-missing cell") is wrong in two independent ways — the denominator at `:154` already
nets out already-missing cells, and the quantity does not monotonically fall — but "the sign is
backwards" overstates it, since the error direction is not even constant across datasets.
`src/training/types.py:118-120` documents the down-weight correctly and then predicts ~5% where
`credit-g_80nan` delivers 25% and `electricity_80nan` 52%; both comments need the same fix. And
the directional claim that "a model equally good at every rung will show a score that improves
from `_20nan` to `_40nan`" is unsupported: at `_40nan` each row also offers 40% fewer observed
cells as context, the two effects push opposite ways, and nobody ran a model. The defensible
statement is that the exam and the yardstick both move, so the reading is uninterpretable.

`README.md:272` is wrong in a related way (`04` F-04-5, SOUND/low): it describes only the
downscaling mechanism, omits the floor, and quotes "about 5%" beside the name of a metric that
reads 0.251 on `credit-g_80nan` — a denominator mismatch between prose and metric, not a false
statement about the helper. Under the paragraph's own implied denominator (share of all cells)
the direction is right.

**Fix direction:** give evaluation its own draw — a flat Bernoulli over observed cells, no
`(1 - prop_nulls)` deflation, no per-row floor (the floor exists so a pre-training batch is
never loss-free; a scoring pass has no such need). `evaluation_mask` already owns a private RNG
stream, so classification stays bit-identical and only the imputation fixture moves. Emit
`realised_rate` per rate segment, including every `rate_N` — today it is emitted only for the
primary rate (`src/training/decoding.py:154-157`, outside the loop at `:162-172`), which is
exactly the population where the nominal rate lies most. If the draw is not changed, strike
ADR 0004:222 and report the induced population for cross-ladder reads instead.

### 3.2 The two scored populations are not a pair, and the induced view is out of distribution

**Severity: medium.** Sources: `01-decode-stage.md` F-01-2 (SOUND/medium),
`05-embedder-tokens.md` F-05-3 (SOUND/medium), `04-ground-truth-protocol.md` F-04-3 (SOUND/low).

ADR 0004:221-222 and ADR 0005:161-162 present `impute/induced/impute_score` as headline and
`impute/masked/impute_score` as companion. Three mechanisms make the gap between them
non-attributable, all verified on shipped data.

**Different rows.** The masked population inherits `p_base*(1-prop_nulls)`, so the gappier a row
is the less of it is scored; `_score_induced_missing` is the exact mirror
(`src/training/decoding.py:243` masks *every* NaN, and `encode` marks every `[MASK]` as scored),
so a row contributes exactly its null count. On `credit-g` fold 1 at rate 0.2,
`corr(nulls_in_row, masks_in_row)` is **−0.359 / −0.406 / −0.336 / −0.345** at
`_20/_40/_60/_80nan`, and the two populations move in opposite directions with missingness:
masked 541→374→249→203 cells against induced 789→1584→2416→3188.

**Different context tokens.** The masked view keeps the row's natural gaps as `[NULL]`; the
induced view converts all of them to `[MASK]`, so the model sees a gappy row with **zero**
`[NULL]` tokens and zero null flags. Census of the full `credit-g_20nan` feature frame:
decode-train view `[MASK]=0.328 [NULL]=0.200` with 1.1% of rows `[NULL]`-free; masked eval view
`[MASK]=0.136 [NULL]=0.200`, same 1.1%; induced scoring view `[MASK]=0.200 [NULL]=0.000` with
**100%** of rows `[NULL]`-free. `[NULL]` is a trained per-column input feature — a distinct
vocabulary id and embedding row for categoricals, a distinct
`special_embeddings[f'{key}_null']` `nn.Parameter` for numericals (`src/embedder.py:147-148`) —
so erasing it shifts the encoder's input distribution, not just which cells are held out. The
effect widens with the ladder, exactly where the imputation question is most interesting.

One sub-claim is corrected: the *cell count* is in distribution. `mask_probability` defaults to
0.5 (`src/training/types.py:103`, used at `src/training/decoding.py:92-94`), so a `_40nan`
training row shows ~6 masked cells against ~8 at induced scoring. The genuinely
out-of-distribution part is the absence of `[NULL]`, not the number hidden.

**Context-free rows.** `datasets/generate_splits.py` injects gaps independently per column, so
at high rates whole rows vanish: `electricity_80nan` has 7593/45312 rows fully missing =
60744/290000 induced cells = **20.9%**; `electricity_60nan` 2.9%; `credit-g_80nan` 1.2%; every
other corpus variant 0.0%. For such a row the input carries only column-identity, mask and
positional embeddings, so the head emits a constant whose optimum is the marginal mean/mode —
which is the `mean_mode_baselines` denominator. The 04 verifier sharpens the direction: an
all-`[MASK]` row is a shape decode training essentially never produces, so those cells are
bounded *below* by ~1.0 in expectation rather than pinned at it. The ratio stays internally
fair; what is at risk is the causal attribution ADR decision 10 invites.

The fixture makes the hazard concrete: a reader comparing `impute/masked` 1.3996 against
`impute/induced` 1.3402 in `tests/fixtures/credit-g_20nan_imputation_regression.json` reads
"the model does better on real gaps", when the two populations differ in which cells they score,
how much context those cells' rows retain, and whether `[NULL]` was present at all. The gap is
**over-determined**, so no effect size can be attributed to any one mechanism without a paired
measurement — which nobody has.

**Fix direction:** the comparability caveat is the finding and belongs in ADR 0004 decision 6
and the README metric section. Beyond that: log the per-row mask-count distribution and the
share of scored cells from context-free rows for both populations; and either train the decoder
on rows shaped like the scoring view, or score the induced population one gap at a time with
the row's other gaps left as `[NULL]` — which is also what a deployed imputer would see.

### 3.3 The search cannot identify a winner, and the winner it publishes is selection-biased

> **Revised 2026-09-15 — [addendum §2](ADDENDUM-2026-09-15-optuna-results.md).** The
> `rho = +0.124` below is the weakest of the twelve studies now available; the Fisher-z
> means are 0.442 (cosine) and 0.524 (plateau), and ten of twelve clear the noise floor.
> The objective is **not** noise — but selection is still near chance (mean winner rank
> 17.3 and 15.2 against 20.5), because the correlation lives entirely in the bad tail:
> restricted to the twenty best-by-objective trials it averages -0.029 and +0.076. The
> proxy is an effective filter and a useless ranker, and promotion uses it as a ranker.
> The selection-bias half of this section is unchanged.

**Severity: critical (F-07-1), high (F-07-2).** Sources: `07-optuna-imputation.md` F-07-1,
F-07-2, F-07-6, F-07-7; `03-imputation-metrics.md` F-03-3; `08-experimentation-design.md`
F-08-3, F-08-5.

**Resolution (F-07-1, SOUND/critical).** Every number reproduces against the user's own study
databases (read-only sqlite) and the real code. Scored validation cells at seed 42, fold 1:
`credit-g_20nan` **283** (187 cat / 96 num), `credit-g_40nan` **179**, `kr-vs-kp_20nan` 1511,
`kr-vs-kp_40nan` 910, `spambase_20nan` 3390. On an all-categorical table `impute_score` reduces
to `wrong_cells / (n * naive_error)` — a lattice. Consecutive distinct objective values differ
by exactly `1/273` on `kr-vs-kp_20nan` and `1/179` on `kr-vs-kp_40nan`; both studies have 40
trials, **20 distinct values**, and `best == 2nd` bit-for-bit. The promoted files hold the
tie-break winners' parameters (trial 17's `LR_DECODE 0.0017418753352176958`; trial 32's
`0.0016986087972445047`). The verifier strengthened it independently: on `kr-vs-kp_40nan` the
tie-break promoted the configuration that is **0.028 worse** on
`test/impute/induced/impute_score` (0.8092 vs 0.7814) — direction, not significance, but it
shows tied configurations are not interchangeable on the metric decision 6 reports.

The verifier also settled the critique's own open question — *does the objective predict the
headline?* — from 180 trial runs already in `mlflow.db`:

| study | ρ(objective, test masked) | ρ(objective, test **induced**) |
|---|---|---|
| credit-g_20nan (n=40) | +0.611 | **+0.124** |
| credit-g_40nan (n=40) | +0.563 | +0.568 |
| kr-vs-kp_20nan (n=40) | +0.418 | +0.340 |
| kr-vs-kp_40nan (n=40) | +0.071 | +0.372 |
| spambase_20nan (n=20) | +0.783 | +0.812 |

At n=40 a Spearman ρ is distinguishable from zero only beyond about ±0.31. So on
`credit-g_20nan` the objective and the population decision 6 headlines are, on this evidence,
unrelated — and the promoted winners rank **31/40, 21/40, 33/40, 1/40, 2/20** on that
population, giving up 52%, 42%, 66% and 0% of their own study's achievable induced range. The
lattice is not the only failure mode: `credit-g_20nan`'s objective is continuous, never ties,
and still selects near the middle.

The damage is not only at promotion. `TPESampler` splits observed trials at a quantile of the
objective; with 20 distinct values over 40 trials that boundary falls inside a block of ties, so
the 30 trials after TPE's random startup were steered by a signal with ~20 levels. A coarse
objective degrades the 2h39m of search that preceded the pick.

**Selection bias (F-07-2 SOUND/high; F-03-3 SOUND/low — same mechanism, two severities).**
`hidden_validation`/`clean_validation` are encoded once per fold
(`src/training/decoding.py:71-82`); each epoch computes `model(hidden_validation,
clean_validation)` and keeps `best_state` at its argmin (`:112-127`); the loop restores that
checkpoint and scores **the same two tensors** at `:204-206`. Not merely the same rows — the
same cells, since the loss is restricted to `hidden.masked_positions`
(`src/models.py:165`). Min-over-150-epochs on a fixed sample, then reporting a correlated
functional on it, is optimistically biased. Across all 180 reduced-profile trials the objective
reads better than the same trial's test masked score on 36/40, 34/40, 25/40, 38/40 and 20/20
trials, mean gaps −0.012 to −0.064 (suggestive only: val and test are different rows).

The two dimensions disagreed on severity and the disagreement is worth stating rather than
silently resolved. `03` cut it to low on two grounds, both correct: the "longer trials take a
deeper minimum" channel does **not** apply to the search that ships, because the reduced
profile does not sample `EPOCHS_DECODE` (`opt.py:94-108`; every trial takes the min over the
same 150 epochs), and the bias never reaches a comparison — `score_search_objective` is set
only at `opt.py:221`, `src/training/decoding.py:202-211` emits the `validation/` family only
under it, so no `cv/test/*` key, no regression fixture and neither leg of the promotion
comparison carries it. `07` kept it at high because `optuna/best_objective_value` is a
*published* number that will read systematically better than the promoted configuration's real
performance, so anyone reading the study parent and the CV comparison side by side sees the CV
number come out worse and reaches for a story about folds. **Settle at medium**: a contaminated
published number and a real search-efficiency cost, against a demonstrably clean comparison.
The `LR_DECODE → noisier curve → deeper minimum` channel remains plausible and unmeasured by
anyone.

**Selection rows reappear as test rows (F-07-6 SOUND/low, F-08-3 SOUND/medium).** The trial
namespace is a bare `class Args: pass` that never sets `cv_folds`, so `build_folds` takes the
predefined branch: 800/100/100, 2556/320/320, 3679/461/461, with val ∩ test = 0. ADR 0005
decision 6 then runs `--cv_folds 5`, which partitions every row into exactly one test fold
(`src/training/data.py:183-191`), so the predefined validation rows make up **8.5-13%** of each
CV test fold (measured `[.085,.090,.130,.110,.085]` on `credit-g_20nan`). One-directional — it
can only flatter the promoted arm. Three dampeners are real and should be stated with it: the
channel is four or five hyperparameters, the CV run re-masks at a different fold ordinal so no
scored *cell* recurs, and the headline population is `induced`, which the search never touched
in any split. The 08 verifier adds the credit the critiques never gave: all 40 trials share one
seed, hence one mask, one split and one initialisation, so the noise is almost perfectly
correlated across trials and the *ranking* is far more reliable than 283 independent cells
suggest — which weakens the winner's-curse half while leaving the leakage half intact.

**Promotion reads nothing held out (F-07-7 — verdict UNSOUND, corrected version medium).** The
headline claim, "nothing *produces* a held-out number for the winner before it is published",
is false: `train_and_evaluate_decoder` always scores the test split, and all 180 trial runs
carry `test/impute/masked/impute_score` and `test/impute/induced/impute_score` on 419-5202
cells. **The corrected finding is sharper than the original**: the held-out number exists on the
winning trial's own run and *nothing reads it*. `--promote_best` writes the file immediately
from `study.best_trial.user_attrs` (`opt.py:525-531`); `--retrain_best` runs afterwards and
gates nothing; `best_metrics.json` (`opt.py:568`) is read by no file in the repository. That is
what produces the 31/40 ranking above.

**Objective population ≠ headline population (F-08-5 — verdict UNSOUND/low).** The premise is
true — `TaskSpec` ranks on `validation/impute/masked/impute_score` while ADR 0005 headlines
`cv/test/impute/induced/impute_score/mean`, with no evidence offered for the agreement — but
the critique's consequence does not hold. Its snapshot (ρ 0.945, pick 3/11, regret 0.0334) was
read from an in-flight study's `temp_trials`; nine further trials of the same study moved the
regret to 0.0009 and ρ to 0.812. The claim that 0.033 "is systematic, not noise" is contradicted
by the study that produced it. What survives is a documentation gap, now superseded by the
07 verifier's full-study table above, which is the measurement this finding wanted.

**Fix direction, cheapest first.** The verifier found a fourth lever the critique missed and it
is free in cells: the **induced** population on the *validation* rows is 419/771/2326/4594/5202
cells — 1.5x to 5x the masked validation population — it is the exact population the headline
uses, and `_score_induced_missing` (`src/training/decoding.py:227-265`) already takes the frame
as a parameter and only hardcodes `fold.test_indices` for the truth rows. Moving it to the
validation split fixes the resolution problem, aligns the objective with the headline, and
costs one forward pass. Beside it: draw a *second* validation mask so the checkpoint watches one
and the objective reads the other; log the count of trials within one objective quantum of the
winner on the study parent and refuse `--promote_best` when that count exceeds one; log the
winner's own held-out `test/impute/induced/impute_score` on the study parent; and record the
8.5-13% overlap in ADR 0005's "How to compare".

### 3.4 Checkpoint selection is a different objective from the one that ranks the fold

**Severity: low.** Sources: `02-decoder-model.md` F-02-1 and `01-decode-stage.md` F-01-4 — one
finding, merged; plus `02` F-02-2 and `06-artifacts-tracking.md` F-06-5.

`TridentDecoder.forward` returns `CE_mean + lambda_num * MSE_mean` (`src/models.py:200-210`);
`src/training/decoding.py:114` uses that exact scalar as the validation loss and `:125-127`
keeps its argmin. The fold is then ranked on `impute_score`, which weights each kind by its
share of scored cells with no λ anywhere
(`src/training/imputation_metrics.py:44-57`). `LAMBDA_NUM` defaults to 1.0 and is sampled
0.1-10 log in both profiles when the table is mixed, so on `credit-g` a trial drawing 0.1
selects the epoch best at categories while one drawing 10.0 selects the epoch best at numbers,
and both are compared on one λ-free ratio. Column mixes confirmed: `electricity.txt` declares
only `day` (1 cat / 7 num), `credit-g.txt` declares 13 (13/7). On the critic's own 80-epoch run
the two criteria disagree by 18 epochs at ρ≈0.61.

Three corrections cut this to low and must travel with it. (a) "Nothing describes this" is
false — ADR 0004 decision 4 says "The best validation-loss epoch is restored"; only the
λ-dependence and the divergence from the ranking metric are undocumented. (b) The
"50% vs 12.5%, a 4x overweight" and "66.4%/33.6% vs w_cat 0.644/w_num 0.356, close to inverted"
quantifications are not shares of anything: a mean cross-entropy in nats and a mean squared
error in z-units added with equal coefficients do not have equal influence, and an argmin
depends on how terms *move*, not on their static magnitudes. The defensible claim is
directional: the criterion is not cell-count weighted and the reported score is. (c) The
structure pre-exists — `src/training/finetuning.py:174-178` argmins validation CE while ranking
on `f1_macro` — so only the λ channel is new. On the one fold measured, the two criteria land
0.005 apart on test while both sit ~0.05 from the oracle.

The verifier found a **third λ channel** the critique missed: `src/training/decoding.py:124`
hands the same composite to `scheduler.after_epoch(...)`, so under `--lr_scheduler plateau`
(`src/training/schedulers.py:103-104`) λ also decides when the learning rate halves.

**Nothing can decompose the curve (F-02-2, SOUND/low).** `src/models.py:205,209` build
detached `cross_entropy` and `mse` entries specifically for reporting;
`src/training/decoding.py:104` and `:114` both discard the dict, and a repo-wide grep returns
exactly four lines, all inside `src/models.py`. Only `decode/train_loss`, `decode/val_loss` and
`decode/learning_rate` reach the tracker. The consequence is narrower than claimed — at the
selected checkpoint `score_cells` emits `rmse_num_z`, `mae_num_z`, `acc_cat`, `macro_f1_cat`
and both cell counts per population, so a diverging numerical head *is* distinguishable — but
the per-epoch decomposition of the loss curve does not exist. Two verifier additions make the
fix obviously right: classification already logs `finetune/val_f1_macro` and
`finetune/val_f1_micro` every epoch (`src/training/finetuning.py:163-168`) while decode logs no
per-epoch proxy at all, and `_loss_events_by_key` (`src/training/summary.py:186-187`) skips
unknown keys, so extra `decode/*` series cannot break `_summarize_loss_bands`.

**No step-less test loss (F-06-5 — verdict UNSOUND, residue low).** The headline — "the
decoder's own objective is never scored on held-out data" — is false three times over: the
composite *is* evaluated on the held-out validation frame every epoch on every fold
(`src/training/decoding.py:51,114`), `cv/decode/val_loss/mean` reaches the parent for **every**
fold (verified in the store at 150 steps with `fold_count=2`), and a trial whose loss never came
down is visible in `validation/impute/masked/{rmse_num_z, acc_cat, ...}`. What survives: the
quantity selected on is not an unbiased held-out estimate, and `best_validation_loss` and the
selected epoch are not step-less scalars, so **which epoch produced this number is unrecoverable
for folds 2..n−1** of a CV run.

**Fix direction:** score the validation split's `impute_score` each epoch and select on it — the
machinery already exists at `src/training/decoding.py:202-211`, gated behind
`score_search_objective` and run once at the end. Log the two loss terms the decoder already
returns, plus step-less `decode/best_val_loss` and `decode/best_epoch`. Do **not** fix it by
making `LAMBDA_NUM` mandatory: that hides the mismatch behind a tuned constant. Earning a
higher severity needs a per-epoch validation `impute_score` trace against the loss trace on
`electricity_20nan`, which nobody has.

### 3.5 What `impute_score` normalises, and what the record keeps of it

**Severity: medium (F-06-2), low (the rest).** Sources: `06-artifacts-tracking.md` F-06-2,
`03-imputation-metrics.md` F-03-2, F-03-4, F-03-6, `02-decoder-model.md` F-02-3.

**The denominator is never persisted (F-06-2, SOUND/medium).** `naive_rmse` and `naive_error`
are locals in `score_cells` (`src/training/imputation_metrics.py:48-56`); grepping `baseline`
over `src/`, `opt.py`, `main.py` and `train.py` returns only call sites, so
`mean_mode_baselines` is never returned, logged, tagged or written. `_error_metrics` builds both
the pooled metrics and every `per_column` row and emits nothing about the baseline
(`03` F-03-6, SOUND/low), while the dataclass docstring at `:22-24` claims the artifact "says
where a poor fold went wrong". With the measured per-column mode errors on `credit-g`
(`foreign_worker` 0.062, `purpose` 0.738), `acc_cat = 0.93` on `foreign_worker` is a **loss** to
that column's baseline accuracy of 0.938 while `acc_cat = 0.45` on `purpose` is a clear win over
its 0.262 — and the artifact shows only 0.93 and 0.45, which order them backwards. On
`kr-vs-kp_20nan`, **14 of 36** columns have a mode baseline above 0.90 accuracy, so sorting the
artifact ascending by `acc_cat` points at the easiest column every time.

Two sub-premises are false and must not be repeated. There is no per-column `_ratio`:
`per_column` is built from `_error_metrics` alone and `_ratio` is called exactly twice, on the
population-pooled numbers, so the zero-baseline branch can fire only if the naive error over the
*entire* scored population is exactly 0. And `spcop` is not constant —
`value_counts(dropna=False)` on `kr-vs-kp_00nan` gives `f 3195, t 1`. Two mitigations the
critique missed: on single-kind tables the denominator is algebraically recoverable from logged
metrics (`naive_error = (1-acc_cat)/impute_score`, which on the logged `kr-vs-kp_20nan` row
gives 0.184402), so the cross-variant blind spot bites only on mixed tables; and
`artifacts.write_imputation_preview` writes the full cell ledger for **every** fold, so
per-column baselines are rebuildable offline from a `results/` tree. Low-not-none for the
per-column half survives because `src/training/tracking.py:214-236` replays a fold's artifacts
only for `best_fold`/`worst_fold`, so the baseline-free `per_column_imputation.csv` on the
parent is the only per-column view in the tracking store for a *middling* fold — which is
exactly the fold the artifact is for.

**Parity is 1.0 and the normalisation is per-fold scalar (F-03-2, F-03-4 — both UNSOUND,
residue low).** The headline of F-03-2 is measurably false: running the real `score_cells` on a
model whose predictions **are** the mean/mode imputer on both halves gives
`impute_score = 1.000000` exactly. The composite is a convex combination of two ratios each
anchored at 1.0, so ADR 0004's gloss is correct as written; the demo's mode-collapsed 0.756 is a
truthful aggregate of a model that really did halve the naive RMSE on half the cells, and it is
ranked strictly worse than a learned model. `acc_cat` is also pinned per population in the
fixture at 0.01, so a change trading macro-F1 for mode-biased accuracy cannot pass silently.
What survives at low is that **`macro_f1_cat` is computed, commented on at
`src/training/imputation_metrics.py:88-90` as the reason accuracy is the wrong quantity, and
used by nothing** — not the ranking, not the objective, not promotion, not the comparison.

F-03-4's premise reproduces in full on real data (naive RMSE per fold
[1.032, 0.918, 1.022, 1.052, 1.080], cv 6.0%; naive cat error [0.389…0.429], cv 5.4%;
per-column mode error spanning 0.062-0.738; per-column naive RMSE `credit_amount` 0.658-1.545
against `residence_since` 1.009-1.083). The load-bearing consequence does not follow: numerator
and denominator are computed on the *same cells*, so the sign depends on their covariance, and
the verifier measured both directions through the real `score_cells` — for `pred = 0.6*actual`
the ratio's fold-to-fold cv is 0.15% against 5.9% raw (a 40x *reduction*), for
`pred = actual + N(0,0.6)` it is 9.1% against 4.3% raw. The effect is indeterminate without
measuring the trained model. **What survives, and should have led:** the CV statistic is a mean
of five ratios with five different denominators, and a Student-t interval assumes neither that
nor a ratio of pooled errors — and the weights themselves are random across folds (`w_num`
measured 0.371 down to 0.325, a 14% swing), so the CV mean is a mean of five
differently-weighted composites. The 03 verifier also names the design question nobody asked:
`ratio_num` is a ratio of RMSEs and `ratio_cat` a ratio of 0/1 error rates, summed with
cell-count weights as though "30% less RMSE" and "30% fewer wrong categories" were the same
amount of better. Nothing in ADR 0004 or the tickets argues that they are.

**Within-kind pooling (F-02-3, SOUND/low).** The categorical term is a sum of per-column
`cross_entropy(reduction="sum")` over one pooled `categorical_cells` denominator
(`src/models.py:176-179,203`). A constant categorical column gets `nn.Linear(d, 1)`, whose
cross-entropy is exactly 0.0 (verified directly), yet its cells still fill the denominator:
measured, `const` 20 cells / `sum_ce 0.0000` beside `bin` 24 cells / `mean_ce 0.9731`, pooled
0.5308 — the informative column's gradient scaled by 24/44. The verifier sharpens it into an
**inversion**: the ranking metric's categorical ratio is *invariant* to the constant column
(measured `impute_score` 1.0000 with and without, while `acc_cat` moves 0.5000 → 0.7273) while
the kind's weight `len(categorical)/total` **rises**. The loss down-weights the categorical kind
exactly where the ranking metric up-weights it. Latent: no shipped variant has a constant or
all-missing categorical column.

**Fix direction:** carry `baselines` out of `score_cells` — it is already in hand at
`src/training/imputation_metrics.py:59` — as `baseline_rmse`/`baseline_error` in
`ImputationScores.metrics` and as a per-column ratio in the long-form table; emit `ratio_num`
and `ratio_cat` beside the composite (two lines, already computed); count only columns with more
than one real category in `categorical_cells`; and state the ratio-estimator caveat wherever the
CI is read.

### 3.6 Pre-training is not measurably contributing, and nothing in the record could say so

**Severity: high (F-13-1, F-13-2), medium (F-13-3, F-13-4).** Source:
`13-pretraining-transfer.md`, all four SOUND.

**The objective is not measuring learning (F-13-1, SOUND/high — cut from critical).** The loss
is `mse_loss` on un-normalised vectors (`src/models.py:72`) with the two
`nn.functional.normalize` calls sitting **commented out** two lines above (`:66-67`) and a
now-dead `self.eps` at `:35`; there is no LayerNorm anywhere in `src/embedder.py`, no predictor
head, no EMA target, no variance term — the only asymmetry is the stop-gradient. The verifier
reproduced the measurement independently, with no monkeypatching, on the real `train_pretrainer`
over `credit-g_20nan` fold 0 at `Hyperparameters()` defaults, 120.6s CPU, on a single fixed
mask: `val_loss` 1.70959 → 0.04488 (**−97.4%**) while `||enc||` falls 15.98 → 3.59 and
`||tgt||` **rises** 2.388 → 2.561; NMSE against a per-column constant lookup 242.7 → **7.70**
(the critic got 4.21 on their fold); centred cosine −0.0060 → +0.0068, 1.9 SE from zero; 89.5%
of the target's energy is the column-constant part, which the transformer is *given* in its own
input. Two sharper numbers from the verifier: the trained encoder (0.04564) is only **19%**
better than predicting the zero vector (second moment 0.05627), and the *raw* cosine rises
0.20 → 0.75 — the encoder learns the free column-constant part and nothing else.

The verifier also found a **second collapse mechanism the critique missed, and it changes the
repair**: the target's within-column variance — the part that distinguishes one value from
another in the same column — **shrank 26%** (0.00807 → 0.00593) while the target's total second
moment grew 16% and the column-constant share of its energy rose 83.3% → 89.5%. Weight
decay cannot explain it (AdamW's decoupled decay over 900 steps is 0.15%), which leaves the
gradient through the input path: the optimisation makes its own target easier. Normalising the
loss does not stop that; only a value-space anchor does.

Cut from critical to high for four reasons, all of which belong in the write-up: no reported
imputation number is invalidated, because the decode stage trains on real cell values;
`git diff main...HEAD` confirms the objective, the commented `normalize` calls and the dead
`eps` arrive unchanged from `main` and the stage is shared with classification, so this is a
project-level problem the branch inherited; 300 epochs at `BATCH 256` on a 500-row fold is only
900 optimizer steps and the critic's own `BATCH=64` rerun moves NMSE to 1.51, so "undertrained
and the metric cannot show it" is established while "the objective cannot work" is not; and it
is one dataset, one fold. The heading's "not learning" is also stronger than the body.

**No detectable benefit (F-13-2, SOUND/high).** The premise that carries the finding needs no
run and holds completely: **all 186 imputation Optuna trials in `mlflow.db` carry
`EPOCHS_PRE=300`** (`Counter({'300': 186})`), as do all 6 imputation parents, so the store
contains zero imputation runs at any other value; and the recorded justification is cost and
sharing, never evidence —
`docs/wayfinder/imputation-optuna-reduced/issues/01-held-set-and-ranges.md:40` ("Pre-training is
shared and protected; tuning it doubles the cost of a stage both tasks use… held by decision"),
echoed at ADR 0005:86, with ADR 0004 decision 4 forbidding the decode stage from disturbing the
encoder. `src/training/decoding.py:54-57` is the whole seam. The critic's own paired ablation
(300 vs 2 epochs at fixed decode budget, two seeds) found arm means of 0.9196 and 0.9203 on
`impute/induced/impute_score` — a 0.0007 difference with the sign flipping between seeds —
against a 0.030 seed spread and 0.042-0.047 fold spread. The verifier did **not** re-run it and
says so: those four numbers are unverified here, and 2 seeds × 2 folds on one dataset cannot
distinguish "no benefit" from "a benefit this design cannot see". The one consistent signal is
that the 300-epoch arm leads on the *masked* population in both seeds (0.0096, 0.0183) and on
`induced` not at all — which is what F-13-1 predicts: familiarity with `[MASK]`, nothing about
values. Severity is high because the fix is cheap **only before** the nineteen GPU-hours of
studies run.

The verifier adds one counterweight the critique owes a sentence to: `EPOCHS_PRE` across 87
**classification** parents is not constant (300 ×56, 200 ×20, 40 ×11), and the one
within-dataset, within-scheduler contrast (`vehicle_00nan`, `cosine_legacy`, 3 folds, seed 42)
is `EPOCHS_PRE=40` → accuracy 0.7210 against `EPOCHS_PRE=300` → 0.7577/0.7530/0.7530. It is
confounded (`EPOCH_FINE` moves with it) and settles nothing, but it points the other way.

**Unfalsifiable from the record (F-13-3, SOUND/medium).** `src/training/pretraining.py:112-119`
logs exactly `pretrain/train_loss`, `pretrain/val_loss`, `pretrain/learning_rate`; grepping
`norm()`/`cosine_similarity`/`normalize` over `src/` returns only the two commented-out lines;
and no unit test ever executes `train_pretrainer` — it is monkeypatched at
`tests/unit/test_training_runtime.py:134,177,449`, the suite's only `TridentPretrainer` is an
untrained container at `tests/unit/test_training_decoding.py:79`, and the imputation fixture
pins `EPOCHS_PRE 2` at `DIM 16`. Cut from high because "unfalsifiable from a run's record" is
too broad — the record carries `impute/induced/impute_score` against a mean/mode baseline; what
is unfalsifiable is specifically "the pre-training stage worked" — and because as a finding it
is the instrumentation half of F-13-1's own Direction. The genuine residue is the coverage gap:
a future change to the shared stage has no feedback beyond a loss whose units move with the
change.

**The reduced profile cannot attribute its win (F-13-4, SOUND/medium).** The reduced branch
samples `PROB_MASCARA`, `LR_DECODE`, `WEIGHT_DECAY_DECODE`, `DROPOUT` and conditionally
`LAMBDA_NUM` (`opt.py:96-112`); `EPOCHS_PRE` 300 and `EPOCHS_DECODE` 150 are held defaults;
`LR_PRE`/`WEIGHT_DECAY_PRE` are held, so **no sampled dimension moves pre-training alone** and
fANOVA structurally cannot rank a held knob. Three corrections: `DROPOUT` is cross-stage too and
more tightly so (the transformer is constructed once in `src/training/pretraining.py:36-42` and
the decode stage reuses that object), so two of five sampled dimensions are cross-stage, not
one; the GPU-hours arithmetic is ~30% overstated — every trial logs `time/pretrain_seconds` and
across 185 trials the median pre-training share of stage time is **0.514**, so "twelve and a
half of the nineteen hours" should be at most ~9.8; and ADR 0005 explicitly names the fANOVA gap
and defers the full-profile pilot, so it is an acknowledged trade-off. The attribution problem
is real at medium: if the best trial lands at `PROB_MASCARA=0.6`, nothing says whether 60%
corruption suited the decoder's masks or pre-training's, and the importance figure sums two
effects that may have opposite signs. The verifier adds that the decode stage cannot set its own
dropout even in the `full` profile, because it reuses the pre-trainer's transformer instance.

**Fix direction:** log `pretrain/nmse_vs_column_mean` and `pretrain/centred_cosine` on the fixed
validation mask each epoch (a few lines on tensors already in hand) plus one unit test asserting
NMSE drops clearly below 1.0 on a synthetic table with a deterministic column relationship —
that is the assertion that would have failed on day one. Split `PROB_MASCARA` into
`PROB_MASCARA_PRE`/`PROB_MASCARA_DECODE`, defaulting both to the current field so every existing
run stays comparable. And run the ablation — same design at `cv_folds=5`, three seeds, three
datasets, roughly one study's cost — **before** the nineteen hours, recording the result in
ADR 0005 next to the sentence that holds `EPOCHS_PRE` at 300.

### 3.7 The comparison protocol cannot support a generalisation claim

**Severity: medium.** Sources: `08-experimentation-design.md` F-08-4,
`12-reproducibility-provenance.md` F-12-2, `08` F-08-9, F-08-2.

**Two unpaired intervals where the protocol declares a paired design (F-08-4, SOUND/medium).**
`_summarize_metric` → `_student_t_interval` over one arm's five folds is the **only** interval
in the repository, and `grep -rn "paired" --include=*.py .` returns nothing outside `.venv` —
while ADR 0005:163 and ticket 06 both call the design "the glossary's paired single-seed CV
protocol: fold-level comparisons are paired". The declared protocol is paired and neither the
code nor `imputation_studies.ps1:133`'s printed recipe produces a paired quantity. That mismatch
is the strongest part of the finding and the critique reaches it only as a remedy. Two further
limits: the five training sets overlap 75% pairwise, so a t-interval over folds systematically
understates how much the number moves under a different partition (there is no unbiased
estimator of k-fold CV variance), and six pairs are read simultaneously with no correction. Two
corrections: the single-seed limit is **declared scope**, not a hidden flaw — ticket 06 says
"conclusions do not quantify seed variability. Multi-seed robustness is out of scope"; and the
`_20nan`/`_40nan` masks are **independent draws, not nested** (2459 of `credit-g`'s 4000
20%-missing cells are observed at 40%), so the family is effectively three-to-six rather than
six — which softens the multiplicity arithmetic and, more usefully, means **a second missingness
draw is already available as a repetition axis**.

**"The configuration is the only difference" is true about inputs and false about the inference
it invites (F-12-2, SOUND/medium).** `set_global_seed` is called exactly once, at
`src/training/runner.py:37`, and all folds run in one process. The only global-numpy consumers
in `src/` are `src/utils.py:58` (`np.random.rand`) and `:73` (`np.random.choice` in the floor
loop), and the floor's call count is **data-dependent**: measured `[0,0,3,0,1]` per epoch at
p=0.5 against `[63,77,63,82,67]` at p=0.2 on the real `credit-g_20nan` frame. Pre-training
consumes two frames per epoch and the decode stage one, so the stream offset entering fold *k*
is a function of every earlier fold's *realised* draws. What **is** genuinely paired between the
two decision-6 arms is substantial and was verified: `build_folds` uses `random_state=seed`
throughout; `evaluation_mask` reseeds on `(seed, fold)` and restores the global state exactly;
the induced population is deterministic from the CSV; and the reduced profile holds every knob
that would change `torch.randperm` counts, with the promoted `credit-g_20nan.imputation.json`
matching the dataclass defaults on all of them (DIM 128, HEADS 16, LAYERS 2, EPOCHS_PRE 300,
BATCH 256, EPOCHS_DECODE 150, EVAL_MASK_RATE 0.2). Only the numpy corruption stream drifts.

Two wording corrections. The offset is **not unseeded**: it is deterministic from
`(seed, configuration)` and reproduces on re-run — it is *unmatched between arms*. And the
finding is sharper than written: fold 1 epoch 1 is the only genuine common-random-numbers moment
(both arms threshold the same `rand` matrix, so at p=0.4 the mask is nested in the p=0.5 mask),
and that pairing is lost the moment the floor loop makes a different number of `choice` calls —
**within fold 1, not at fold 2**. Severity is medium, not high, because the first-order confound
(changing `PROB_MASCARA` changes realised corruption) is unavoidable under any seeding, no
published number rests on it, and the collateral points are what will actually bite: **no fold
is reproducible alone**, and an epoch-budget change is a different experiment rather than a
truncation.

**`-Scheduler X -Compare` answers nothing about X (F-08-9, SOUND/low).** `-Compare` reuses
`-Scheduler` as `--lr_scheduler` at `imputation_studies.ps1:84`, and
`src/training/config.py:30-33` applies `dataclasses.replace(..., lr_scheduler=...)` so the flag
beats the file; all four promoted files read `"LR_SCHEDULER": "cosine"`. So
`-Scheduler plateau -Compare` compares a cosine-tuned configuration under plateau against
defaults under plateau. The exploration actually queued is `-Scheduler plateau -NoPromote`, the
study branch, which never touches promoted files — so this is a plausible next step rather than
something already scheduled.

**The twelve runs cannot test the reduction (F-08-2 — verdict UNSOUND, residue low).**
`imputation_studies.ps1:62` is the literal `"--search_space", "reduced"` with no `-SearchSpace`
parameter, and `full` is a valid imputation profile — that much is true. The reasoning fails:
ADR 0005 decision 6 is titled "Checking the reduction, and proving the tuning helped" and names
**two** mechanisms — fANOVA checks the reduction, the twelve runs prove the tuning helped — and
its Considered options explicitly defers "a full-profile pilot with fANOVA to rank the held
knobs" with a reason. The proposed remedy is also not a matched arm: `full` samples `EPOCHS_PRE`
and `EPOCHS_DECODE` in 20..60 (`opt.py:124,134`) while `reduced` holds them at 300 and 150, so a
same-budget `full` arm would promote a configuration pre-trained a fifth as long and confound
space width with epoch budget. **Residue at low:** fANOVA ranks only sampled knobs, so it can
expose an over-inclusion but never an under-inclusion — which the ADR itself states, and which
F-07-5's `HEAD_DIM` bug narrows further.

**Fix direction:** report the per-fold **paired difference** and its interval instead of two
overlapping single-arm intervals — free, and it uses runs you already have. State the family
size. Use the second missingness draw as the repetition axis. Narrow ADR 0005:157-158 to "the
configuration and the fold-entry RNG offset", and give training corruption the treatment
`evaluation_mask` already has — a stream derived per `(seed, fold, stage, epoch)` — behind a
flag and a tag.

### 3.8 Configuration precedence, promotion and provenance

**Severity: medium (F-07-4, F-12-3), low (the rest).**

**An imputation run with no file of its own reads classification's tuned file (F-07-4,
SOUND/medium).** `src/training/config.py:49-51` builds
`candidates = [task-keyed, classification-keyed]`. Reproduced in an isolated scratch cwd holding
only a classification file: an imputation run took `dim 64, dropout 0.45, prob_mask 0.11,
epochs_pre 7, sched plateau`, with `epochs_decode` falling back to 150 — a configuration nobody
chose. `PROB_MASCARA`, `EPOCHS_PRE` and `LR_SCHEDULER` are all in `complete_configuration`'s
shared block, and `--lr_scheduler` only wins when explicitly passed. Cut from high because the
blast radius today is **zero** (all four files under `datasets/hiperparams` are
`.imputation.json`; no shared `<dataset>.json` exists anywhere), because the committed
launcher's own comment shows the author reasoning about this case, and because the critique's
prescription — "add the missing test" — is wrong: `tests/unit/test_training_config.py:276`
(`test_an_imputation_run_prefers_its_own_configuration_and_falls_back_to_the_shared_one`) pins
the opposite **deliberately**, with a docstring stating the rationale. The right ask is to change
the decision, not to add coverage. Medium rather than low because the inheritance is quietly
large — the full classification profile samples `EPOCHS_PRE` 20..60 against the held 300, a five-
to fifteen-fold cut to pre-training — and it is one `--promote_best` away from being live.

**`config_source` points into an untracked directory (F-12-3, SOUND/medium).**
`promote_best_configuration` writes to `hyperparameter_file(dataset_name, task)` — one fixed
path per `(dataset, task)`, no version, no study id (`opt.py:342-345`) — and `config_source` is
that path string (`src/training/config.py:56`). `git ls-files datasets/hiperparams` → 0;
`git check-ignore` → exit 1; `git status --porcelain` → `?? datasets/hiperparams/`. **Untracked
and not ignored**, so the entire output of the ADR 0005 search effort is one `git clean -fd` from
gone — the verifier calls this the sharpest consequence and notes the critique under-sold it by
framing the loss as attribution. The dangling pointer is already in the store: three parents
(`10cdd533fee5`, `631c3b64a184`, `7ed05b09f090`) record
`config_source = datasets/hiperparams/electricity/electricity_{20,40}nan.json` and that
directory is now empty. `_load_base_hyperparameters` then returns `Hyperparameters(), "defaults"`
with no warning, so checking out the recorded commit and re-running the recorded command trains
the defaults. One correction: `config_source='defaults'` *is* logged, so a reader comparing
params against the original sees the mismatch; what is genuinely silent is the load itself and
the undisclosed classification fallback. The 10 verifier adds the second way that path goes
quiet: `hyperparameter_file` returns a **CWD-relative** `Path("datasets/hiperparams")/...`
(`src/training/config.py:69`), so a run launched from anywhere but the repo root resolves
`config_source = "defaults"` and trains at 300/150 with a promoted file sitting three
directories away — proved by pointing the loader at a fabricated file in a scratch cwd. It is
pre-existing (`git show main:src/training/config.py` carries the same literal) and outside `10`'s
declared scope, but the branch widened it by turning one silently-missable file into two, and
the launchers only work because they `Set-Location $PSScriptRoot`. Values are recoverable — `_log_execution_params` logs
`complete_configuration`, all 17 keys read back off the `electricity_40nan` parent — so what is
lost is attribution and re-runnability.

**`config_source` over-claims for a partial file (F-10-4 — verdict UNSOUND, residue low).** The
premise reproduces (a fallback run logs a `.json` path beside five decode defaults), but both
consequences are falsified by the record the critique quotes. `config_source` is the full posix
path **including the suffix** (`src/training/config.py:56`), so for an imputation run a path
ending `.json` rather than `.imputation.json` *is* the statement that the task-keyed file was
absent — the discriminator being asked for is already in the string. And "nothing in the run
record says so" is false: every value that differs between a fallback run and a defaults run is
a logged parameter (the probe's two rows differ visibly at `EPOCHS_PRE` 40 vs 300, `DIM` 192 vs
128, `LR_SCHEDULER` cosine vs cosine_legacy). What survives is narrow: for a *partial* file you
cannot tell which keys it supplied and which were defaulted — equally true of classification
since before the branch.

**The defaults arm is runnable (F-07-3 — verdict UNSOUND, residue low).** The premise is true —
no CLI flag skips a promoted file, and `hyperparams_override` is programmatic only — but the
conclusion is false, and the reason is a scope omission worth recording: the critique's scope
list names the uncommitted `experiment_imputation.ps1` and the launch **log**, but never
`imputation_studies.ps1`, the committed launcher that produced that log and that ADR 0005
decision 6 refers to. Its `-Compare` mode **is** decision 6's protocol and already does the file
move as code — `$aside = "$promoted.aside"; Move-Item ...` inside a `try/finally` that restores
on Ctrl+C, plus a `[skip]` guard when no promoted file exists. So three of four claims fail.
Residue at low: a file move is weaker than a flag (a hard kill between the two `Move-Item` calls
strands the `.aside`), and any entry point other than that launcher still has no escape.

**Smaller, all SOUND/low.** A study **cannot be resumed** (F-07-9): `storage_name` is built from
a per-invocation timestamp (`opt.py:389,431`) so `load_if_exists=True` at `:437` can never match,
and `study.optimize` is called without `catch=` (`opt.py:478`) so anything raised outside the
objective's own try — including `KeyboardInterrupt` — terminates the study and strands its
completed trials. (One wording slip corrected: `opt.py:302` *raises* `TrialPruned`, it does not
catch.) The study's **running-best file uses classification's promoted filename** (F-07-10):
`opt.py:317` writes `save_dir / f'{self.dataset_name}.json'`, character-for-character what
`hyperparameter_file(dataset, 'classification')` produces, and each of the five study
directories now holds a partial `<dataset>.json` (5 of the task's 17 keys for `credit-g`, 4 for
`kr-vs-kp`) against decision 5's "a promoted file is complete". The file sitting in the tree
right now, `results/credit-g_20nan/optuna_20260911_011020/credit-g_20nan.json`, contains an
*imputation* study's five sampled keys under classification's exact filename — copy it one
directory over, the obvious manual promotion, and every later classification run on that variant
trains at an imputation study's dropout and mask probability. A **study deletes every trial's
cell ledger including the winner's** (`opt.py:575-582` `rmtree`s `temp_trials`), while the one
artifact that survives is the clobbered root `metrics/<dataset>_metrics.csv` row — the retention
is exactly inverted.

**Fix direction:** commit `datasets/hiperparams/` (it is small and it is a research result, not
a build output). Add `--hyperparams {path|defaults}`, which would also let
`imputation_studies.ps1 -Compare` drop its `Move-Item` pair — and whoever touches
`_load_base_hyperparameters` must keep that launcher working, because the comparison it runs is
the effort's deliverable. Make the fallback loud. Stamp the promoted JSON with the study's MLflow
run id. Name the running-best file after what it is. Add `--study_dir`/`--resume`.

### 3.9 Test adequacy

**Severity: medium (F-09-3, F-09-4, F-09-5), low (the rest).** Source: `09-test-adequacy.md`,
plus `02-decoder-model.md` F-02-5.

**Nothing asserts the decoder beats the naive baseline (F-09-3, SOUND/medium).** Verified by
exhaustion: every occurrence of `impute_score`/`acc_cat`/`rmse_num_z` in the decode unit tests
and the integration test is a membership check, one `!=` between two splits, or a
`pytest.approx` pin — not one inequality against a baseline anywhere. Every pinned `impute_score`
in the fixture exceeds 1.0, which by construction means *worse than mean-and-mode*; the columns
are z-scored, so a constant-0 predictor scores `sqrt(mean(z^2)) ≈ 1.0` and the pinned
`rmse_num_z` values 1.0563 and 0.9997 sit 0.06 and **0.0003** from it. The class of bug this
misses — a prediction tensor paired with the wrong truth tensor, a head wired to the wrong
column token, a selection mask off by one — is exactly the class this branch is most exposed to,
and D-1 is a demonstrated live instance. One framing correction: the fixture pins regressions
from the recorded run and never claimed to certify it, so the exposure is to **day-one wiring
errors**, not to defects introduced later (a later pairing break would move the pinned means).
The critic's flakiness warning should be kept verbatim: `preprocess_table` draws from the global
numpy stream, so any such test must seed it.

**The fixture's justification is false, and the excluded dataset is the one that crashes
(F-09-4, SOUND/medium).** `tests/integration/test_credit_g_imputation_regression.py:3` and
AGENTS.md both claim `credit-g_20nan` is "the only variant that exercises both column types and
both scored populations". `datasets/categorical_columns/` holds exactly three files (credit-g
13, electricity `day`, kr-vs-kp 36), and `electricity_20nan` exists with a `_00nan` sibling — so
it qualifies too, and it is the dataset whose induced path crashes (D-3). A second fixture there
would have been red on day one. Six of nine datasets carry no categorical declaration and
`kr-vs-kp` none numerical, so **two thirds of the corpus never reaches the decode stage in any
test** — coverage, not a second crash: both single-kind shapes were driven end to end and
complete.

**The fixture pins six pooled averages and no structural invariant (F-09-5, SOUND/medium).**
`PINNED` is six means, while the same run computes `n_num_cells`, `n_cat_cells`, `mae_num_z`,
`macro_f1_cat` for both populations plus `realised_rate` and discards them. `macro_f1_cat` —
introduced on this branch *because* accuracy flatters a majority predictor — has exactly one
assertion in the suite (`tests/unit/test_imputation_metrics.py:149`) and no end-to-end pin. The
integration test opens neither preview, ledger nor `per_column_imputation.csv`. Cell counts are
integers fixed by the seed, the split and the mask draw: they either match exactly or the
population moved, which is strictly stronger than a mean inside a 0.01 tolerance. One sub-claim
is **premise-false and corrected in the file**: "`cv_folds=None` under `--task imputation` is
untested everywhere" is wrong — `tests/unit/test_training_runtime.py:487` and `:537` run
`run_training(_minimal_request(..., 'imputation'))` with `cv_folds=None` and take the
`src/training/runner.py:176-186` else branch. The decoder is stubbed and nothing
imputation-specific is asserted, so the accurate word is **unasserted**. The row-count and
MCAR-uniformity sub-bullets stand: the un-batched full-split evaluation path is only ever
exercised at 500 test rows.

**No test guards that decode gradients reach the encoder (F-02-5, SOUND/low).** ADR 0004
decision 4 says "The encoder is fine-tuned, not frozen";
`src/training/decoding.py:59-63` delivers it only through `optim.AdamW(model.parameters(), ...)`
relying on `TridentDecoder` holding both as submodules (`src/models.py:113-114`), and
`test_attaching_the_decoder_leaves_the_pretrained_encoder_untouched` ends at construction and
never calls `backward`. Verified universal: grepping `backward|named_parameters|\.grad\b` across
`tests/unit/` matches only `test_decoder_model.py`, which uses `named_parameters` solely in that
construction test. One overstatement corrected: "would pass every test in the repository" is
false — the integration fixture pins `impute_score`, `rmse_num_z` and `acc_cat` at 0.01 and is
the one guard that *could* catch a head-only optimizer, though whether a 2/2-epoch, DIM-16 run
moves those past 0.01 is unmeasured. Honest statement: unguarded in the default
`pytest -m "not integration"` suite, with a coincidental and unverified opt-in backstop.

**A test certifies a path no run takes (F-09-6, SOUND/low).**
`test_imputation_summary_carries_the_decode_stage_timing`
(`tests/unit/test_training_tasks.py:100`) hand-builds a `time/decode_seconds` `LoggedMetric`.
Closed with a literal grep the critic did not run: `decode_seconds` appears **exactly once** in
`src/`, `opt.py`, `main.py` and `train.py` — at `src/training/summary.py:35`, the
`TIMING_METRIC_KEYS` constant itself. `stage_timing_metrics` returns only pretrain/finetune/total
for both tasks, so no run has ever produced the key, and the test's name states something false
about every imputation run. Cut to low because the suite would be green without this test, so
its existence did not *cause* the shipped gap, and because it does exercise a real pass-through
behaviour.

**The search objective key is typed twice and cross-checked nowhere (F-09-7, SOUND/low).**
`opt.py:405` assigns `task.search_objective`, `opt.py:266` does
`metrics[self.search_objective]`, the producer builds the key as an f-string at
`src/training/decoding.py:207-210`, and two tests type the literal. Nothing asserts
`task_spec('imputation').search_objective in outcome.result.metrics`. A rename surfaces at
`opt.py:266` as a `KeyError` *after* a full trial has trained, `opt.py:295-302` converts it to
`TrialPruned`, and `opt.py:502` then raises once every trial is pruned — the user sees "the
search produced no result" with no metric name anywhere. One premise correction: the producer
connection does exist (`test_training_decoding.py:122-144` runs the real
`train_and_evaluate_decoder` and asserts the key is present); what is missing is the
spec-to-producer cross-check, and the edit that slips through is a rename on one side with that
side's test updated in lockstep.

**A `validation/` key reaches the `cv/test/` family (F-09-8, SOUND/low).**
`src/training/decoding.py:219` returns `metrics={**metrics, **validation_metrics}`;
`final_metrics_for_tracking` copies `result.metrics` wholesale; `_validate_final_metrics` accepts
any finite numeric key; `summarize_cross_validation` summarises all of them; and
`src/training/tracking.py:392-394` logs each as `cv/test/{name}`. `opt.py:221` sets
`score_search_objective` for every imputation trial and `opt.py:209` leaves MLflow on, so **every
imputation Optuna trial run carries `cv/test/validation/impute/masked/impute_score`**. Bounded:
trials are `run_role=optuna_trial` and the project compares `run_role=parent`. The verifier
extends it one caller further: `write_metrics` spreads `result.metrics` straight into the row,
so these become **columns of `metrics.csv`** — the deterministic artifact the fixture's whole
design is built around — on every imputation trial.

**A data-dependent key set aborts a CV run after every fold has trained (F-06-6, SOUND/low).**
`score_cells` key sets are genuinely data-dependent (cat-only, num-only and empty all differ),
and feeding two mismatched folds to the real `summarize_cross_validation` raises "All folds must
have the same final metric keys." The raise is at `src/training/runner.py:165`, after
`metrics.csv`, `hyperparameters.json` and `per_column_imputation.csv` are already on disk and
before the CV summary, manifest, provenance and every `cv/*` metric. Classification's key set is
fixed by the sklearn metric functions, so the asymmetry is real. Could not be triggered from
shipped data.

**A tidy-up, not a finding (F-09-10 — verdict UNSOUND/low).** The factual claims hold —
`test_a_numerical_cell_is_imputed_with_one_number_per_column` is strictly subsumed by
`test_columns_never_share_head_parameters`, and the loss-curve length assertion restates the
loop — but the recommendation does not follow: the finding's own consequence line says "none
directly", its causal story points at the wrong file, and the two ranking-tuple tests **should
stay**, because they pin a documented ADR 0004 contract for two lines each and fail on a direct
edit to `_TASK_SPECS`, which the summariser tests cannot distinguish from a summariser bug.

**One thing nobody tested and the verifier found:** a CV run has **two** metric aggregators and
the one Optuna reads is not the one the tests cover. `run_training` returns
`compute_cv_summary(frame)` (`src/training/runner.py:183-190`) over the frame `write_metrics`
built, which averages every non-`fold`/`dataset` column with **no key validation at all**
(`src/training/summary.py:246-249`), while every summariser test exercises
`summarize_cross_validation`'s `_summarize_metric`. Both run, so a NaN fold is still caught — the
objective path is guarded by a different function than the one it reads from, and that pairing
is untested.

**And a second gap the same verifier named:** no test opens an imputation run's `metrics.csv`
at all. The regression fixture reads `result.fold_results[*].metrics` in memory and never
touches the file, so the artifact AGENTS.md treats as the deterministic record is unexercised
for this task — the same file F-09-8's `cv/test/validation/...` keys land in. (The same
verifier's other small note, that
`test_the_decode_stage_scores_the_validation_split_only_when_asked` uses a float `!=` where it
means "a different split", is test wording and is out of this review's scope.)

### 3.10 The documentation contract

**Severity: medium (F-11-1), low (the rest).** Source: `11-documentation-contract.md`, plus
`06` F-06-4 / `10` F-10-6 and `04` F-04-5.

**CONTEXT promises an untouched test split (F-11-1, SOUND/medium).** `CONTEXT.md:148-149` says
the test split "stays untouched until the chosen configuration is retrained". The narrow
guarantee holds — `opt.py:266` reads `validation/impute/masked/impute_score` — but
`train_and_evaluate_decoder` scores the test split with **no condition**: masked
(`src/training/decoding.py:145-157`), each extra rate (`:162-172`), induced with sibling load
and scaler transform (`:174-182`), null-token under the flag (`:183-196`), all logged as `test/*`
at `:213`. `score_search_objective` only *adds* a `validation/` family. The store: 189
`test/impute/induced/impute_score` rows, of which 185 are `optuna_trial`. `README.md:381` and
ADR 0005:41 both state the true behaviour, so CONTEXT contradicts two sibling documents. Cut
from high because the guarantee that protects the result is genuinely implemented, so no
measured quantity is invalidated; the exposure is a reader forming a mid-study view from a column
the definition says does not exist — and this repository already contains a worked instance, the
07 verifier's Spearman table, built from exactly those rows.

The verifier found a **second false claim in the same glossary block, and it has already
bitten**: `CONTEXT.md:160-163` defines a held hyperparameter as one that "every trial runs at
the task default". `LR_SCHEDULER` is held by both profiles and does **not** run at the task
default — `opt.py:225` sets `args.lr_scheduler` and `src/training/config.py:225-228` applies it
*after* `from_mapping` has resolved the held knobs. That is precisely why all 193 imputation runs
in the store are tagged `cosine` while the held default is `cosine_legacy`.

**Smaller, all SOUND/low, each verified line by line.**

- **The wiring table skipped the task axis (F-11-3).** `docs/ARCHITECTURE.md:409-433` gained an
  `lr_scheduler` row and no row for any of `EPOCHS_DECODE`, `LR_DECODE`, `WEIGHT_DECAY_DECODE`,
  `LAMBDA_NUM`, `EVAL_MASK_RATE`, `EVAL_MASK_RATES_EXTRA`; three surviving rows are wrong for
  imputation (`PROB_MASCARA` is not "during pretraining" — `src/training/decoding.py:91-96`
  re-rolls at that rate every decode epoch; `HIDDEN_DIM` also sizes every decoder numerical head
  at `src/models.py:141-147`; `finetuning_epochs` reaches **no** training code on imputation
  while `decode_epochs` is the horizon). Confined to one reference table: `README.md:268-273`
  documents all six decode keys correctly and calls `PROB_MASCARA` the knob that "governs
  training corruption".
- **`--lr_scheduler cosine` is prescribed, enforced nowhere, and the fixture pins the schedule
  the documents warn against (F-11-4).** `src/training/config.py:222-229` parses with
  `default=None` and the effective default stays `cosine_legacy`
  (`src/training/types.py:17,107`), with no task-aware branch. `Hyperparameters.from_mapping`
  on the fixture's block resolves to `cosine_legacy` — while the store holds imputation = 193
  `cosine` + 8 `plateau` and **zero** `cosine_legacy`, against 191 `cosine_legacy`
  classification runs. Live exposure is small (both launchers pass the flag; no run ever
  inherited the bad default) but a reviewed baseline is frozen under a schedule nobody runs, and
  `README.md:303-304`'s own example omits the flag two lines below the sentence prescribing it.
  The critique's "1500 steps and four returns" describes a hypothetical 150-epoch run; the
  fixture pins `EPOCHS_DECODE: 2` and runs 16 steps.
- **`full` does not sample everything (F-11-5).** `opt.py:110-150` never samples
  `EVAL_MASK_RATE`, `EVAL_MASK_RATES_EXTRA` or `LR_SCHEDULER`, and drops `LAMBDA_NUM` on
  single-kind tables, while `CONTEXT.md:155` says `full` "samples every hyperparameter the task
  uses" and `CONTEXT.md:160-163` defines a held hyperparameter as one a profile does not sample —
  an exact internal contradiction. `opt.py:128-129` states the exam-difficulty reason for holding
  the evaluation rate, and states it well; it belongs in the definition. The overclaim is
  repeated at `README.md:335`, the paragraph an operator actually reads before launching a study.
- **Stage timing covers the entire scoring pass (F-11-6).** `src/training/runner.py:78` starts
  the clock and `:102` reads it around the whole `train_and_evaluate_decoder` call, while the
  comment at `:103-105` claims "the timing covers training only". After training ends at
  `src/training/decoding.py:131` the window still holds the baselines, the masked scoring, one
  full population per extra rate, the induced population with a sibling slice and
  `scaler.transform`, the optional null-token population, the optional validation objective, and
  a per-cell ledger built row by row with a `.at[]` lookup per numerical cell. The falsifying
  case needs no measurement: two runs differing only in `--score_null_path` report different
  `time/finetune_seconds` while training identically. Corrections: the false sentence is
  `README.md:379`, not CONTEXT (`CONTEXT.md:61-62` excludes only data prep, plotting and artifact
  writes, all literally true); and `train_and_evaluate_classifier`'s test evaluation has always
  sat inside the same clock, so moving the scoring pass out is a two-task change with a
  classification footprint.
- **"On the test fold only" is contradicted by the branch's own objective (F-11-7).**
  `CONTEXT.md:118-119` and `README.md:310` both say it, while
  `src/training/decoding.py:202-211` scores a masked population on the *validation* frame (1085
  such metric rows in the store) and `:183-196` adds a fourth test-fold population under a flag
  `README.md:165` documents and "What gets scored" omits. `CONTEXT.md:146-149` asserts the
  reverse of `:118-119`; both sentences are new on this branch. `README.md:273` and `:310` also
  contradict each other about how many populations exist, 47 lines apart. Correction: "no
  sentence anywhere states it correctly" is overstated — `README.md:381` does, for the search
  objective. What is missing is one complete population × split × condition statement.
- **The run taxonomy has no term for 12 runs (F-11-9).** Census on the live store (452 runs):
  `optuna_trial` 186, `parent` 93, `worst_fold` 78, `best_fold` 78, `optuna_study` 5, and **12
  with no `run_role`** — three top-level `train_vehicle_00nan_*` runs and their nine nested
  `fold_N_of_3` children holding 248 metric rows each. All twelve carry `task_backfilled` and
  `lr_scheduler_backfilled`, so they answer the two axes ADR 0003 and ADR 0004 name while
  matching no `run_role` value; coverage is `task` 452/452, `lr_scheduler` 452/452,
  `run_role` 440/452, `run_type` 287/452. Correction: the three parents hold **zero metrics**, so
  a parent-filtered comparison could not have used them anyway, and the nine children the recipe
  excludes by design hold the numbers. What survives is taxonomy coverage — and a shape
  (`CONTEXT.md:33-36` says a CV execution retains *one of two* children; these retained all
  three) the vocabulary cannot name.
- **`CLAUDE.md`'s MLflow rule was updated for the schedule and not for the task**
  (`06` F-06-4 and `10` F-10-6, SOUND/low, one defect). `git diff main...HEAD -- CLAUDE.md` is a
  one-line rewrite adding the `lr_scheduler` clause and no `tags.task`, while `README.md:369`
  says "Filter `tags.run_role = parent` and `tags.task`" — two checked-in documents disagreeing,
  with `CLAUDE.md` the one loaded into every session. The mixture exists: `TRIDENT/credit-g`
  holds 15 classification parents and 1 imputation parent, `TRIDENT/electricity` 7 and 4,
  `TRIDENT/spambase` 7 and 1 — and because `get_or_create_experiment` keys on the **base**
  dataset, `run_role=parent` returns a mixture spanning every missingness variant. The 28 metric
  keys both tasks share on `run_role=parent` are exactly the pre-training loss bands, the full
  `cv/time/*` family and `time/training_seconds`; **no `cv/test/*` key is shared**, so the harm
  is confined to timing aggregates and the legitimately-shared pre-training bands. `tags.task` is
  a real filterable tag (`src/training/tracking.py:121-131`), so the fix is one line.

### 3.11 Residual leaks and units

All **low**, all verified, none changing a published number — listed so they are not
rediscovered.

- **Validation and test masks come from one RNG stream** (`01` F-01-3 SOUND/low; `04` F-04-6
  UNSOUND/low — same mechanism). `src/training/data.py:145` seeds on `(seed, fold)` only, and
  `preprocess_table`'s first act is `np.random.rand(*data.shape)` filled row-major, so the
  validation frame's uniforms are the leading block of the test frame's draw. On `credit-g_00nan`
  the coupling is near-total: 187/200 identical masked-column sets, cellwise agreement 0.9892,
  `P(masked in test | masked in validation)` 0.999 against a 0.208 base rate. **But it is an
  `_00nan` property**: the per-row deflation and null-zeroing break it on gapped variants —
  both-masked falls to 66% at `_20nan`, 29% at `_60nan`, 3% at `_80nan`. The rows are different
  rows, so only the column composition is shared; the decoder trains on masks re-rolled every
  epoch across all columns; and only `Nv/Nt` of test rows fall in the overlapping index range.
  The extra rates are also **nested** inside the primary rate rather than independent
  difficulties (535/682 cells at 0.05 are also masked at 0.20), which is undocumented and is the
  more concretely checkable half. One argument fixes both.
- **`rmse_num_z`'s unit is fit on the rows it scores** (`04` F-04-7 and `05` F-05-4, both
  SOUND/low). `scaler.fit_transform` runs at `src/training/data.py:71` inside `prepare_dataset`
  and `build_folds` is called afterwards at `src/training/runner.py:46`, so folds do not exist
  when the scaler is fit; `StandardScaler` is NaN-aware here, so the statistics come from every
  observed row including the test fold, and `src/training/decoding.py:251-255` pushes the
  sibling's truth through that same fitted scaler. `impute_score` is provably immune —
  the train-fold-mean baseline RMSE in scaled space equals the raw one over the global sigma to
  six decimals (1.070101 both ways) — but `rmse_num_z`/`mae_num_z` are pinned in the fixture and
  logged per population, and each variant fits its own scaler, so they are not on a common axis
  across the ladder. Two opposite corrections worth keeping: the cross-ladder unit drift is
  **smaller** than framed (credit-g `duration` sigma 12.0528 / 11.8835 / 12.2502 / 12.1850, a 3%
  spread across a 5x change in sample), and the within-run contamination is **larger** (at
  `--cv_folds 2`, which the ADR's own verdict runs used, the test fold is half the table). The
  scaling order is AGENTS.md-protected, so the honest fix is prose. The same whole-table fit
  applies to the categorical **vocabulary**, which on this branch became load-bearing in a new
  way: `TridentDecoder.__init__` derives each head's output space from it, so the set of answers
  the decoder may give is fixed in part by the test fold's values — the same argument F-05-4
  makes about the numerical unit, unstated.
- **The row-alignment guard certifies what it cannot check** (`04` F-04-4, SOUND/low) — covered
  under D-3, where its consequence lives.
- **The ledger's `row` is a position in the fold's test block**, not a dataset row
  (`12`, the standalone section). `fold.test_indices` reaches neither the ledger, the preview nor
  `provenance.json`. Within a fold this costs nothing — all three populations share one
  positional basis — but the five per-fold ledgers of a CV run cannot be concatenated, and a cell
  cannot be traced to the source CSV. One extra column, `dataset_row = int(fold.test_indices[row])`,
  closes it, and with F-12-1 cut to low it is now the **only** thing standing between the record
  and checking whether a cross-ladder comparison scored the same cells.
- **Cell counts are summarised as measurements**: `n_num_cells` and `n_cat_cells` are in
  `FoldResult.metrics`, so the parent publishes a two-sided Student-t interval on a count of
  cells beside the real ones (`06`, "missed").

---

## 4. What the experiments can and cannot support

Read this before launching anything else.

### `imputation_studies.ps1` — the study branch (five studies run, one in flight, one not started)

**Can support:** "On `<variant>`, at seed 42, on one predefined 80/10/10 split, the
configuration that minimised `validation/impute/masked/impute_score` over 40 trials of the
reduced space is X." Nothing more. The search is internally consistent — all trials share one
seed, hence one mask, one split and one initialisation, which is deliberate variance reduction
and makes the *ranking* more reliable than 283 cells suggests — and `mixed_columns` gating,
`EVAL_MASK_RATE` being unsampled, and the promoted-filename/loader agreement all check out.

**Cannot support:** that X is better than another trial's configuration on the headline. The
objective is decided on 283 cells (`credit-g_20nan`) or 179 (`_40nan`); on `kr-vs-kp` it is a
lattice where best and second are bit-identical and the winner is a **tie-break on trial
number**; and its rank correlation with `test/impute/induced/impute_score` is +0.124 on
`credit-g_20nan` against a ±0.31 noise floor, with the promoted winners ranking 31/40, 21/40,
33/40, 1/40 and 2/20. **The four promoted files currently in `datasets/hiperparams/` should not
be described as "the tuned configuration" in any write-up.**

**Minimal change that fixes it:** move the *induced* scoring to the validation rows and make it
the objective. `_score_induced_missing` (`src/training/decoding.py:227-265`) already takes the
frame as a parameter and only hardcodes `fold.test_indices` for the truth rows; the population
is 1.5x-5x larger, it is the population decision 6 headlines, and it costs one forward pass.
Log the number of trials within one objective quantum of the winner and refuse `--promote_best`
when that exceeds one. Both are small enough to land before the remaining studies.

**Two operational facts to act on today.** `spambase_20nan` was last seen at 20 of 40 trials and
`spambase_40nan` has not started; **a study cannot be resumed** (`opt.py:431` stamps the storage
URI per invocation, so `load_if_exists=True` is inert, and `study.optimize` is called without
`catch=`), so an interruption strands ~5.5 hours of completed trials in a store nothing will
reopen, with ~16 hours of `spambase` still to run. And if either study dies, `-Compare` silently
**skips** that pair (`imputation_studies.ps1:86-90`) and the summary table records
`Status = SKIPPED` with no reason — while `-DryRun` cannot warn you, because the promoted-file
check is guarded by `if (-not $DryRun -and ...)`. Make that `Test-Path` unconditional before
spending the GPU time.

### `imputation_studies.ps1 -Compare` — the twelve comparison runs (none has been run)

> **Revised 2026-09-15 — [addendum §3](ADDENDUM-2026-09-15-optuna-results.md).** The
> parenthetical is superseded: **twenty-four** comparison runs now exist, twelve under
> cosine and twelve under plateau. Across the twelve tuned-versus-default comparisons
> **exactly one separates at 95%** — about what chance produces. Every "cannot support"
> item below still stands, and item 1 is precisely the caveat the new table inherits.

**Can support:** "On these six tables, with this one missingness draw, this one 5-fold partition
at seed 42, under the cosine schedule, the promoted configuration's mean induced `impute_score`
is X and the defaults arm's is Y." A fully conditional descriptive statement. The mechanics are
right: the arms genuinely share the schedule, both arms' fold partitions and evaluation masks are
identical per fold, the induced cells are deterministic from the CSV, `results/` directories do
not collide, and the `-Compare` move-aside/restore is exception-safe in every ordering.

**Cannot support:**

1. **A 95% statement about the difference.** Two overlapping single-arm Student-t intervals are
   not a test of a difference, and they discard the pairing the shared folds give for free —
   while ADR 0005:163 and ticket 06 call the design "paired". *Fix:* report the per-fold paired
   difference and its interval. Free, on runs you already plan to make.
2. **That the advantage survives a different partition or seed.** The five training sets overlap
   75% pairwise, so the fold interval understates partition variance, and one seed is declared
   scope. *Fix:* use the second missingness draw (`_20nan` vs `_40nan` masks are **independent**,
   not nested) as a repetition axis, and say in the write-up that the family is three-to-six
   pairs read simultaneously.
3. **"The configuration is the only difference" (ADR 0005:157-158).** True about inputs, false
   about the inference: the training corruption stream is unmatched between arms from fold 1's
   floor loop onward, because `PROB_MASCARA` is itself sampled and the `np.random.choice` count
   is data-dependent. *Fix:* narrow the ADR sentence now; seed corruption per
   `(seed, fold, stage, epoch)` behind a flag and a tag later.
4. **A clean held-out reading of the promoted arm.** The predefined validation rows that chose
   the configuration are 8.5-13% of every CV test fold. One-directional, small, unremovable
   without a third split. *Fix:* one sentence in "How to compare".
5. **Anything about a schedule other than cosine.** `-Scheduler plateau -Compare` would run a
   cosine-tuned configuration under plateau against defaults under plateau.
6. **Anything, if the metrics files matter to you afterwards.** Neither arm isolates
   `--metrics_dir`, so the defaults arm (second) overwrites the promoted arm's row in
   `metrics/<dataset>_metrics.csv`, on top of what the studies already did. *Fix:* per-arm
   `--metrics_dir metrics/compare/<dataset>/{promoted,defaults}`.

### Cross-ladder reads (`_20nan` → `_80nan`)

**`cv/test/impute/masked/impute_score/mean` is not comparable across the ladder**, despite
ADR 0004:222. The realised exam rate is 0.209/0.169/0.155/0.157/0.250 on `credit-g` and rises to
0.52 on `electricity_80nan`, and at `_80nan` 81% of `credit-g`'s scored cells come from the
one-mask-per-row floor. The `rate_N` segments are mislabeled — `rate_10` realises 0.2548 on
`credit-g_80nan`, 2.5x its own label, and on a narrow 4-feature frame `rate_5` realises 0.3086,
6x — and they carry no realised rate at all. **`cv/test/impute/induced/impute_score/mean` *is* the comparable one** — the
generator's gaps are a fixed cell set, and the naive denominators are stable across the ladder
(numerical 0.998/1.020/0.963/0.952, categorical mode error 0.406/0.416/0.392/0.405). ADR 0004
has this exactly backwards, and ticket 05 already said so. Two caveats on the induced read:
`electricity_80nan` has 20.9% of its induced cells in rows with **no** observed cell at all
(2.9% at `_60nan`, 1.2% on `credit-g_80nan`, 0.0% everywhere else), where the ceiling is
marginal-baseline parity; and `rmse_num_z`/`mae_num_z` are in a per-variant z-unit and are
diagnostic only.

### `experiment_imputation.ps1` — do not run it as it stands

At HEAD it runs classification. With the worktree's two-token edit it runs imputation on
`_00nan` variants, where `load_complete_sibling` returns `None` and **the ADR 0005 headline
population does not exist at all**; it writes classification's key set into the shared config
file (so `-FinetuneEpochs` is inert while the banner says it applied, and on a nan variant a
promoted file shadows the whole thing); and it closes by telling the operator to query
`cv/test/f1_macro/mean`, which no imputation run emits, with no `tags.task` filter. Its setup
loop is outside its `try`, so a mid-loop failure leaves an unrecoverable config file. Fix all
four together (D-2) or use `imputation_studies.ps1` for the schedule question too.

### Two crashes to expect

`--task imputation` on any `electricity_*nan` variant raises
`ValueError: y contains previously unseen labels: '2'` at `src/training/decoding.py:264`.
`kr-vs-kp_40nan --cv_folds 5 --seed 42` completes folds 1-3 and raises the same way in fold 4,
after that fold's pre-training and decode training have run (D-3).

---

## 5. Checked and cleared

**Rule for this section:** REFUTED findings and UNSOUND findings whose severity after review is
`none` appear here in full. UNSOUND findings with surviving residue appear in §3 under their
corrected claim, with the refuted headline noted there.

### Refuted, and severity-none

- **The `from_mapping` epoch fallbacks are a repair, not an ungated behaviour change**
  (`10-config-plumbing.md` F-10-2, **REFUTED**). The behaviour change is real —
  `from_mapping({})` and a 13-key partial both yield 300/150 on the branch against 40/40 on
  `main` — but five things break the defect framing, all verified. It *removed* an
  inconsistency: `main`'s dataclass already read 300/150 and `main`'s loader returns
  `Hyperparameters()` when no file exists, so a no-file run was **already** 300/150 on `main`
  while `from_mapping({})` gave 40/40. "No ADR records it" is false — ADR 0005 decision 2's held
  table names `EPOCHS_PRE=300` and `EPOCHS_DECODE=150` as the task defaults and that table is
  load-bearing, because the default reduced profile returns only 5 keys and relies on
  `from_mapping` resolving held keys to the task default. "Both commit messages reason
  exclusively about the dataclass path" is false — `fc5666d`'s body names the JSON path as the
  subject of its justifying sentence. The test hunk in that commit is a **red test being made
  green**, not a guard relaxed. And the blast radius is empty: all four
  `datasets/hiperparams` files carry the epoch keys and `opt.py`'s full profile samples them
  explicitly. What survives, at low, is one line of the critic's own Direction: nothing pins
  `Hyperparameters.from_mapping({}) == Hyperparameters()`, and that invariant has already broken
  once.
- **`_ratio`'s zero guard does not invert any ranking** (`03-imputation-metrics.md` F-03-5,
  UNSOUND, **severity none**). Demo B reproduces exactly through the real `score_cells` (perfect
  model: baseline wrong on 0/200 → 1.0, on 1/200 → 0.0, on 2/200 → 0.0) and shows something
  else entirely. When the baseline is flawless, "the perfect model" and the mode imputer emit
  identical predictions on every cell — they are the same predictor — so 1.0 is correct for
  both, confirmed by running the mode-copier over the same three baselines and getting
  1.0/1.0/1.0. When the baseline errs, the perfect model scores 0.0 and the mode-copier 1.0,
  correctly ordered. This is the documented intent
  (`src/training/imputation_metrics.py:108-113`) and is pinned red-green by
  `test_a_column_the_naive_imputer_never_gets_wrong_still_ranks`. The discontinuity is real
  (`_ratio(0.1, 0.0) = 1.1` against `_ratio(0.1, 1e-9) = 1.0e8`) but unreachable, and not for
  the stated reason: the categorical denominator is a mean of 0/1 indicators, so it is exactly 0
  or at least `1/n`, never `1e-9` at any population size. All that is left is an imprecise ADR
  sentence, which the brief excludes.
- **README's "degrades correctly on the six all-numerical datasets" is a construction claim, not
  a measurement claim** (`11` F-11-8, UNSOUND, **severity none**). Every premise is true and was
  verified, and the critique concedes in its own Evidence that the property holds
  ("Structurally the claim is true and I am not disputing it"). A true statement about a
  metric's construction is properly evidenced by the construction plus a unit test; requiring a
  run on each named dataset would make every true mathematical sentence in a README a
  methodological defect. The offered consequence has no content, because there is nothing for a
  re-check to find, and the Direction rewrites a true sentence into another true sentence.

### Suspected and disproved — do not spend time here

The first two entries are `12`'s two UNSOUND findings (F-12-1, F-12-4, both with residue kept
at low); each states in bold what survives, and that residue is a real item, not a cleared one.
Everything from the third entry onward is cleared outright.

- **The `_XXnan` variants are fully reproducible.** This is the largest false start in the
  review and it deserves space. `12-reproducibility-provenance.md` F-12-1 argued (at claimed
  high) that the missingness draw is recorded nowhere and its substitution is undetectable, on
  the explicit assumption — stated in its own Open questions — that "the generator itself is not
  in this repository … I found no script that writes the variants." **It is
  `datasets/generate_splits.py`, it is tracked, and it is seeded** (`RANDOM_SEED = 42`;
  `seed = RANDOM_SEED + int(pct*100)`; `rng = np.random.default_rng(seed)`), and its input
  `datasets/datasets_raw/` is also tracked (9 CSVs, not gitignored). The verifier regenerated the
  variants from the committed inputs and diffed against disk: `raw == _00nan` frame-equal,
  `regenerated == on-disk _20nan` frame-equal, missingness pattern identical, **0 cells
  differing**; `kr-vs-kp_20nan`, `kr-vs-kp_40nan` and `spambase_20nan` all frame-equal and
  pattern-identical. So the `.csv` is gitignored as the build output of a committed,
  deterministic script, and the whole chain sits inside the commit
  `mlflow.source.git.commit` already records on all 450 runs. The finding's central scenario —
  regenerate elsewhere and the two arms measure different cells — is false, and the
  fixture-attribution and scaler claims fall with it. Every code-level premise did hold
  (`provenance.json` carries no hash or row count, the `_00nan` sibling is in no record,
  `_assert_row_aligned` passes a re-drawn variant with 769/4000 overlap, and MLflow's digest
  collides across `kr-vs-kp_20/40/60/80nan` because `compute_pandas_digest` drops every column
  holding a NaN — though for *mixed* tables it does distinguish a redraw). What survives at low:
  the run record names neither the generator nor its seed nor the sibling, and
  `np.random.Generator` carries no cross-version stream guarantee under NEP 19 — **the one link
  a numpy upgrade could move, and the one the critique assumed did not exist.** The verifier
  sharpens the omission half into a dead pointer: `prepare_dataset` always sets `splits_path`
  (`src/training/data.py:48`) and `write_tracking_provenance` always writes it, but
  `build_folds` reads it only when `cv_folds is None` (`src/training/data.py:155-172`) — so
  `results/credit-g_20nan/20260910_195312/data/provenance.json`, a `cv_folds: 2` run, names a
  split file that had no bearing on a single fold while omitting the `_00nan` sibling the run
  actually read.
- **The environment record's omissions are covered by the lockfile** (`12` F-12-4, UNSOUND,
  residue low). `runtime_environment_tags` really does return only device, gpu_name,
  torch_version, cuda_version, and scikit-learn really is load-bearing (LabelEncoder,
  StandardScaler, both splitters, the sibling re-encode). But `uv.lock` **is tracked** and pins
  `scikit-learn 1.7.2` for `python_full_version < '3.11'`, AGENTS.md mandates
  `uv run --python 3.10`, and the critique's own cleared item establishes
  `mlflow.source.git.commit` on all 450 runs — so for every tracked run the record already says
  which scikit-learn, numpy and pandas the run was *meant* to have. Residue: nothing verifies
  the running interpreter matched the lock, and `--disable_mlflow` runs record no commit and no
  environment at all.
- **The decode stage does not leak.** Every consumer of `fold.test_indices` inside the stage —
  `src/training/decoding.py:133`, `:141`, `:246` — is first touched after the epoch loop has
  ended and `best_state` has been restored at `:129-130`. `build_folds` partitions with an outer
  `KFold`/`StratifiedKFold` and splits train/validation *inside* the non-test indices, so the
  three index sets are disjoint by construction. `complete_sibling` enters as a parameter and is
  referenced only inside the post-training scoring block; it never touches the embedder, the
  optimizer, the baselines or the validation loss.
- **`impute_score` is exactly invariant to the transductive scaler** — verified numerically in
  four dimensions (§3.11).
- **The induced gaps are genuinely MCAR and their truth is genuine.**
  `datasets/generate_splits.py:43-48` draws `n_inject = round(n_rows * pct)` row indices per
  feature column with `rng.choice(..., replace=False)` — a fixed count per column, independent
  of every cell's value, label column excluded, no value-dependent step anywhere. All nine
  `_00nan` tables have zero NaN, so every induced gap has a real value in the sibling.
- **A scored cell's truth is always a real value; the `-1` sentinel is unreachable.**
  `preprocess_table` zeroes the dynamic mask at null positions (`src/utils.py:62`) so the masked
  population can never select a cell whose clean target is `[NULL]`, and the induced population's
  truth is the complete sibling. Confirmed by a real forward pass completing normally with an
  all-missing column present.
- **The decoder cannot see a hidden cell's value.** Numerical: the embedder computes the MLP
  output for every cell and discards it wherever the mask or null flag is set, and
  `split_numeric_and_special` writes `0.0` into the value slot at every special token.
  Categorical: the hidden cell carries the `[MASK]` vocabulary id.
- **`[MASK]` and `[NULL]` are genuinely two tokens with two trained parameters**, the
  cat-then-num token order survives the tensor-native rewrite everywhere except
  `src/training/decoding.py:262` (D-1), the positional embedding is numerically identical to the
  old construction, the `ModuleDict` key-collision guard fires in both directions and cannot be
  defeated by `_mask`/`_null` suffixes, and `as_category_strings` round-trips **every** dtype
  `read_csv` can produce byte-identically (including `9007199254740993`, `0.1+0.2`, `1e25`,
  bools and strings with leading spaces).
- **No `cv/test/*` metric key is shared between the two tasks.** Census over the real store on
  `run_role=parent`: zero shared `cv/test/*` keys. Stage loss bands are keyed by the task's own
  stage. The overlap is timing plus the genuinely shared pre-training bands.
- **`_diagnostic_roles` inverts correctly for a minimised metric** — `worst` selects the largest
  `impute_score`, `best` the smallest, ties to the lowest fold number. Reversing this would have
  silently swapped the two diagnostic children on every imputation run.
- **The induced cell accounting reconciles exactly against the real data.** From the logged
  `credit-g_20nan` row: masked 98+185 = 283 at `realised_rate 0.17900063251106893` → eligible
  1581.0 exactly; induced 151+268 = 419; 1581+419 = 2000 = 100 test rows × 20 features; and the
  real NaN count of that test split is 419.
- **RNG isolation across tasks in one process is airtight.** `run_training`'s first statement is
  `set_global_seed(request.seed)` (`src/training/runner.py:37`), `evaluation_mask` saves and
  restores the global numpy state in a `finally`, `preprocess_table(..., fine_tunning=True)`
  draws nothing (MT19937 position 624 → 624), and there is no module-level cache either task
  could populate for the other. Classification results cannot move because an imputation run
  preceded them.
- **`train.main(args, return_metrics=True)` keeps its flat Optuna shape for both tasks**, and
  `compute_cv_summary` is schema-agnostic, so imputation keys aggregate without a list to extend.
- **Classification cannot read the task-keyed file**, `--lr_scheduler` precedence is correct and
  defaults to `cosine_legacy`, `score_search_objective` is genuinely programmatic-only, the
  base-name derivation agrees across all three places that compute it, and the new parse-time
  guards fire where they should.
- **Promotion cannot move either regression fixture** — both integration tests construct
  `TrainingRequest` directly from the fixture mapping and never touch `load_hyperparameters`.
- **The pruner is inert** (`trial.should_prune()` is never called) and **no trial has failed** —
  all four finished studies show 40/40 COMPLETE.
- **`mean_mode_baselines` learns from the training fold only**, and its categorical mode is
  stringified the same way the vocabulary is, so an integer-coded categorical column does not
  silently get a baseline that can never be correct.
- **Two populations are never pooled into one number** — `score_cells` is called separately per
  population before the `pd.concat`, and `runner._per_column_scores` groups by population.
- **`_to_original_units` indexes the scaler correctly**, `actual_original` takes its raw value
  from the right frame for each population, and the ledger has enough precision to recompute
  every metric except `impute_score`.
- **An empty scored-cell frame does not crash the preview.**
- **The critique-level "no recovery" claims were checkable and are false**: the five clobbered
  `metrics/*.csv` files have intact timestamped copies and matching classification parents in
  the store (§D-5).

---

## 6. Coverage map

| # | Dimension | Critique file | Findings filed → survived | Not examined |
|---|---|---|---|---|
| 01 | The decode stage | `01-decode-stage.md` + verdict | 5 → 5 (1 high, 1 med, 3 low) | Whether `_XXnan` is MCAR (settled by 04/12); the magnitude of the shared-stream bias on real data; GPU paths. **Cleared:** no test-split or sibling leakage, `best_state` restoration, `realised_rate`'s denominator, scale-invariance, the `-1` sentinel. Its verdict additionally settled the erased-category crash (D-3) and flagged a silent `nan`-loss degradation to "last epoch" (`src/training/decoding.py:87,129-130`) as a note, not a finding. |
| 02 | The decoder model and heads | `02-decoder-model.md` + verdict | 5 → 5 (all low) | The 40+80-epoch dual-criterion run was not replicated (premises verified in source, numbers cited as the critic's); `HIDDEN_DIM`'s bottleneck cost on the numerical head; whether `LR_DECODE`'s ceiling is safe for the shared encoder. **Cleared:** which view supplies the mask (mutation-tested), unmasked cells carry no loss, no Xavier re-init, targets never backpropagate, excluded-token machinery on real vocabularies, `LAMBDA_NUM` inert on single-type tables, head parameter growth. |
| 03 | Imputation metrics | `03-imputation-metrics.md` + verdict | 6 → 5 (1 med, 4 low); 3 UNSOUND, of which 1 at none | No trained model was measured, which is what leaves F-03-4's sign indeterminate. **Cleared:** the induced population *is* ladder-comparable, model and baseline share a space, the weights match ADR 0004 decision 6, `_00nan` lacking induced keys is by design, promotion is a human reading two CIs. Its verdict added the commensurability question (a ratio of RMSEs summed with a ratio of 0/1 rates) and the random fold weights. |
| 04 | Ground truth and protocol | `04-ground-truth-protocol.md` + verdict | 7 → 7 (2 high, 5 low) | No training run; no MLflow store inspection, so ADR decision 11's verdict table was taken at face value as a record of what was run. **Cleared:** MCAR by construction, positional row alignment, population disjointness, scale-invariance, no meaningful vocabulary leak (1 category across all `kr-vs-kp` folds), `evaluation_mask`'s RNG save/restore, `actual_original`'s source frame. |
| 05 | Embedder, `[MASK]`/`[NULL]` | `05-embedder-tokens.md` + verdict | 4 → 4 (1 high, 1 med, 2 low) | The `baddbmm` rewrite (cleared by a prior review, not re-verified); what the 2026-09-10 null-path runs actually logged. **Cleared:** token ordering after the rewrite, positional embedding equivalence, the collision guard, `[MASK]`≠`[NULL]`, every `read_csv` dtype round-trip, `StandardScaler`+NaN, fine-tuning never reaching the NaN encode path. Its verdict added the `induced_null_token` preview mislabel and the categorical-vocabulary whole-table fit. |
| 06 | Artifacts, keys, MLflow identity | `06-artifacts-tracking.md` + verdict | 8 → 8 (2 med, 6 low; 1 UNSOUND with residue) | Whether the `results/` tree matches what MLflow received (no artifact-store walk); decoder numerics. **Cleared:** no shared `cv/test/*` key, stage-keyed loss bands, `_diagnostic_roles` orientation, the manifest and diagnostic children, every population and rate reaching MLflow, the induced cell reconciliation, the preview's `[MASK]` claim for the masked population, `_to_original_units`, ledger precision, empty-frame handling. Its verdict added the trial-ledger `rmtree` and the `cv/test/validation/...` reachability. |
| 07 | The Optuna search | `07-optuna-imputation.md` + verdict | 10 → 10 (1 critical, 1 high, 2 med, 6 low); 2 UNSOUND with residue, one of them sharper than the original | No training run, so no configuration was re-scored at a second mask draw — still the single most informative experiment available. The critique never opened `imputation_studies.ps1`, the committed launcher, which is what made F-07-3 wrong. **Cleared:** promoted filename/loader agreement, promotion cannot move the fixtures, objective and test metric in the same units, inert pruner, no failed trials, disjoint predefined splits, schedule consistency, retrain isolation, `mixed_columns` gating, the held values being dataclass defaults, trial-to-trial differences not being seed noise. |
| 08 | Launcher experimental design | `08-experimentation-design.md` + verdict | 10 → 10 (4 med, 6 low); 2 UNSOUND with residue; 4 of the 10 are one unadapted file | No seed-to-seed spread measured; the masked/induced rank correlation was measured only on one all-numerical in-flight study (superseded by 07's verdict). **Cleared:** untracked `datasets/hiperparams/` is normal in-flight output; `-Compare`'s move-aside restore is exception-safe; both arms share the schedule; today's defaults arm really is defaults; `LAMBDA_NUM` correctly held; run artifacts not clobbered between arms; per-dataset experiments do not pool; no trials failed; `$PSNativeCommandUseErrorActionPreference` is not a live hazard on this host. |
| 09 | Test adequacy | `09-test-adequacy.md` + verdict | 10 → 10 (1 high, 3 med, 6 low); 1 UNSOUND, surviving only as a tidy-up | **Scope was the imputation test files only.** `tests/unit/test_opt_tracking.py` (509 new lines), `test_training_tracking.py` (+320), `test_training_runtime.py` (+200), `test_training_summary.py` (+87), `test_training_config.py` (+301), `test_training_schedulers.py` (132 new), `test_training_environment.py` (31 new) and both `test_backfill_*.py` were **not** reviewed. No pytest was run by anyone. Whether the fixture reproduces on CPU is unknown (it records `device: cuda`). **Cleared:** both single-kind table shapes survive the decode stage end to end; the masked population's `actual` is never fabricated; the masked path's own column indexing is correct; the `actual_original` test is real coverage; `evaluation_mask`'s RNG isolation is properly tested; the imputation metric unit tests are well-chosen. Its verdict found the two-aggregator gap nothing tests. |
| 10 | Config plumbing, classification bit-identity | `10-config-plumbing.md` + verdict | 6 → 5 as defects (2 med, 3 low); 1 REFUTED (residue low), 1 UNSOUND with residue | How many partial classification configs exist on other machines (bounds F-10-2, now refuted anyway). **Cleared:** RNG isolation between tasks in one process, `train.main`'s flat Optuna shape, classification cannot read the task-keyed file, `--lr_scheduler` precedence, `score_search_objective` programmatic-only, base-name derivation, the new parse-time guards, the new `PreparedDataset` fields. Its verdict found the study's classification-named partial config and the CWD-relative silent-fallback path. |
| 11 | Documentation contract | `11-documentation-contract.md` + verdict | 9 → 8 (1 med, 7 low; 1 UNSOUND at none) | The share of an imputation fold's `time/finetune_seconds` that is scoring rather than training (argued structurally, never measured). **Cleared:** the fold-ranking direction travels with the metric, `best_and_worst` is real code, the decode stage really does mirror fine-tuning on checkpoint selection and schedule and never re-initialises the encoder, ARCHITECTURE's "1400 NaN" figure, `search_space` tag coverage, "a promoted imputation configuration never changes what a classification run reads", BACKLOG's header claims, the `task`/`lr_scheduler` backfill. |
| 12 | Reproducibility and provenance | `12-reproducibility-provenance.md` + verdict | 5 → 5 (2 med, 3 low; 2 UNSOUND with residue) | The digest blind spot was checked on `kr-vs-kp` and `credit-g` only, not all 45 prepared frames. **Cleared:** CV *does* record dataset lineage (the prompt's premise was wrong); the git commit *is* recorded on all 450 runs; `EVAL_MASK_RATE` is not searched in either profile; the torch stream stays paired across the two decision-6 arms; the configuration values are reconstructable from MLflow params; the store holds 6 imputation parents (4 FAILED, 2 finished at plateau/2 folds), so **none of the twelve decision-6 runs exists yet**; the fixture's repeatability evidence is sound. |
| 13 | Pre-training transfer | `13-pretraining-transfer.md` + verdict | 5 → 5 (2 high, 3 med — one of the three a CONFIRMED bug) | The paired ablation was **not** re-run by the verifier, so its four numbers are unverified here; both probes are `credit-g_20nan` only; GPU behaviour. Whether a differently designed objective would help was out of scope. **Cleared:** the embedder does not collapse in the BYOL sense (`‖tgt‖` grew, and `pre_norm` explains why); the per-kind loss balance is not skewed by the Xavier sweep; the Xavier coverage gap has no reachable consequence through the decoder; the probe measures the shipped objective (0.4% agreement with the loop's own logged loss); pre-training keeps its last epoch and nothing else selects on the noisy signal. Its verdict found the 26% within-column variance collapse and that `schedulers.py` is branch-new. |

### Source files the branch touched that **no dimension opened**

- **`scripts/backfill_run_tags.py` (149 new lines) and `scripts/backfill_lr_scheduler_tag.py`
  (136 new)** — only the former's docstring was quoted, in `11`. Their correctness was never
  reviewed, which matters because F-11-9 shows `run_role` is the one tag the backfill was *not*
  run for, leaving 12 of 452 runs outside every documented filter. Their tests
  (`tests/unit/test_backfill_*.py`, 180 lines) were likewise not opened.
- **`src/training/schedulers.py` (113 lines, entirely new on this branch)** — read incidentally
  by `13` for D-4 and by `02`'s verifier for λ's third channel, but **no dimension owned the
  ADR 0003 schedule work**. Two findings already touch it from the outside (D-4; λ feeding
  `ReduceLROnPlateau`), which is a weak signal that a dedicated pass would find more.
- **`src/mlflow_utils.py` (19 new lines)** — listed as context by `06`, never critiqued.
- **`src/training/finetuning.py` (+49)** — read only as the classification comparison point by
  `02`, `05`, `09` and `12`. The classification-bit-identity question was answered through
  `src/training/config.py` (dimension 10) and the `vehicle_00nan` fixture, not by auditing this
  diff line by line.
- **`main.py` (+15)** — read by `10` only, as plumbing.
- The **`graphify-out/` regeneration** (44,950 changed lines in `graph.json`) was used for
  orientation by every dimension and audited by none; it is generated output, so this is noted
  rather than filed.

`src/transformer.py` and `datasets/generate_splits.py` were both read (by `05`/`13` and
`04`/`12` respectively) and are **not** touched by the branch.
