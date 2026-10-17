# Plan: improve the imputation experiment protocol and baselines (2026-10-17)

**Status: accepted on 2026-10-16 at about 22:45 GMT-3.** The user agreed with every recommendation
in section 7 ("concordo com suas recomendacoes"):

- D1–D21 as recommended;
- path B for the schedule (D19): E43 first, then the full E44.

The code changes it names (D9's checkpoint flag, D20's `.gitattributes`) are therefore authorized.
The plan turns the 2026-10-16 review into workstreams, experiment drafts, ADR options and a
two-week schedule for gorgona8. Each experiment still gets its own ticket and log entry before its
first cell.

Written on 2026-10-16 at about 22:30 GMT-3 on gorgona8, after `git pull`:

- TRIDENT at `0f9fc6a` on `feat/imputation-task`;
- the wiki at `db9c5e5`.

## 0. What this plan rests on

**Sources read:**

- the wiki page `projects/trident/findings-2026-10-16-baselines-methodology.md`. Its threat numbers
  (1 to 7) are the ones used here;
- `docs/reviews/2026-10-16-quick-wins/`:
  - `README.md`;
  - `digests/review_digest.txt`;
  - `digests/merge_digest.txt`;
  - the template scripts;
- `AGENTS.md`, `CLAUDE.md`, `CONTEXT.md` and `docs/TASKS.md`;
- ADRs 0004, 0007, 0015, 0016, 0017 and 0019;
- the log entries E39 and E40, the E40 ticket and
  [the calibration proposal](../proposals/calibration-as-imputation-output.md) (T20).

**The review's probe numbers are indicative.** They came from scratch runs on the Windows laptop:
not pre-registered and not in the store. This plan cites them as the review's, with the file to
look in, and never as store numbers.

**Facts checked on gorgona8, read-only, on 2026-10-16.** Some of them update the review.

1. **Source unchanged since the review.** `git diff b2b50fd..HEAD -- src scripts tests` shows only:
   - the three JSON files ADR 0019 removed;
   - the E40 scripts.

   Every line number the review cites in `src/` is still valid.
2. **Seeds already used by imputation parents** (source runs only, `is_mirror` excluded):

   | seeds | runs | note |
   |---|---|---|
   | 42, 7, 13 | 287, 224, 219 | |
   | 101, 202, 303 | 94 each | |
   | 404, 505, 606 | 12 each | E40 |
   | 420, 99, 1 | 14, 2, 2 | |

   The review suggested 404/505/606 for the confirmation set. E40 has spent them, so the
   confirmation needs a fresh triple.
3. **The 15 variants never used for a decision.** Fourteen have no imputation run at all:
   - pendigits and letter at 40, 60 and 80nan;
   - electricity at 60 and 80nan;
   - vehicle, biodeg and kc2 at 40 and 80nan.

   electricity_40nan has two `FAILED` runs. Both ran on the laptop's RTX 3050 on 2026-09-11, at
   00:25 and 00:37 GMT-3.

   None of the 15 has a promoted file. The only inheritance question is ADR 0018's encoder on
   pendigits and letter at 40–80nan.
4. **All five fold ledgers are on the server.** The queue state directories
   `/scratch2/diogoneiss/trident-2026-10-*/results/.../imputation/imputation_fold_N_cells.csv` hold
   every fold of every cell:

   | studies | ledgers |
   |---|---|
   | E34, E35, E36, E38 | 315 each |
   | E39 | 150 |
   | E40 | 180 |

   This overturns the review's "not quick" verdict on distributional metrics, which assumed only
   the best and worst folds survived.

   **What the ledgers hold:** the model's headline fills only, for the induced and masked
   populations. They hold no baseline fills and no calibrated fills. The headline is:
   - H on E34 and E35 (`decode_checkpoint=loss`);
   - K on E38, E39 and E40.
5. **Which fold CSVs carry which baselines.**
   - E34, E35 and E36 carry ADR 0007's four.
   - E40, E39 and the E38 cells resumed after ADR 0017 carry all six, lgbm and missForest
     included.
   - So per-fold lgbm and missForest beside K already exist for 12 variants on seeds 404/505/606.
     The review's item 3 gap is now only:
     - the 20nan variants on 101/202/303;
     - all of 42/7/13.
6. **Baseline cache.** `results/baseline_cache` holds 7,218 entries (70 MB). All three families
   (`adr0007`, `lgbm`, `missforest`) put the bytes of `imputation_baselines.py` and
   `imputation_metrics.py` into their key: `_FAMILIES` at `baseline_cache.py:64`, and
   `_environment_digest` at `:77`.
   - Changing either file misses every entry.
   - A new module misses nothing.
7. **Thread cap.** `missforest.py:70` and `lightgbm_baseline.py:27` hard-code `THREADS = 8` and pass
   it as `n_jobs`, which overrides `OMP_NUM_THREADS`. A CPU pass beside the GPU queue must set both
   module constants to 4 in its driver. Fills are thread-invariant (ADR 0017 decision 6), so the
   cache keys do not change.
8. **Line endings.** On gorgona8 all 16 tracked `.sh` files (15 queue scripts and `start_mlflow.sh`)
   are LF in the index and in the working tree,
   and there is no `.gitattributes`. The CRLF trap lives in the Windows working tree and in its
   `code --apply` push (section 6).
9. **The machine:** 32 cores, 62 GB of RAM and one RTX 3090 Ti with 24 GB. The four GPU queues use
   16 threads, which leaves about 16 cores for CPU passes.
10. **Cell times,** five folds, mean of three seeds, from the store:

    | table | default encoder | ALL encoder |
    |---|---|---|
    | electricity_20nan | 101 min (E34) | 180 min (E38) |
    | letter_20nan | 59 min | 111 min |
    | pendigits_20nan | 38 min | 73 min |

    The light tables take 3 to 10 minutes: vehicle 4.6, biodeg 9.9, kc2 3.6 (E40 at 60nan).
11. **The review's ORDER section numbers are stale.** It calls the pre-registration of items 2–5
    "E40" and the MNAR smoke "E41". E40 is the calibration study, so the next free number is
    **E41**, and this plan renumbers from there.
12. **Item 7b does not appear in the wiki or in the digests.** Section 2d defines it from the
    review's own evidence; decision D1 asks whether that is the intended meaning.

## 1. Where E40 changes or overlaps the review

The review was written before E40 and assumed seeds 404/505/606 were unused. E40 left the defaults
unchanged (T20 is parked), so the frozen post-E40 configuration is:

- ADRs 0013–0015;
- ADR 0018 on pendigits_20nan and letter_20nan only;
- ADR 0019.

| review item | what E40 changes |
|---|---|
| 1 (thesis table) | **E40 is a third seed set** on 12 variants (all ≥40nan), run on exactly today's default, with all six baselines per fold. It enters the table. Its own Outcome table read a bar of four (`calibration_report.py` names no lgbm), so its "best baseline" column and its "below the best baseline: 11 of 12" need the shared helper. No decision rested on them. |
| 2–3 (linear arms) | E40's folds host the linear arms in the same pass. |
| 5 (deployable checkpoint) | **KC uses the validation gaps' truth a second time:** K picks the epoch on them and the calibration is fitted on them. Adopting T20 would stack a second oracle step on ADR 0015's. Any deployable arm has to say what happens to the calibration (D10). T20 should wait for E43. |
| 6 (confirmation) | **404/505/606 are spent**, so the confirmation needs a fresh triple (D11). **All 15 confirmation variants are at ≥40nan**, so a `--score_calibrated` readout in every confirmation cell tests T20's rule out of sample at no cost. It also tests its open threshold question, 40 vs 60. |
| 7 (metrics) | **KC shrinks numbers toward the mean** (mean α 0.60 at 80nan) and answers low-confidence categories with the mode. That lowers RMSE but should lower dispersion and balanced accuracy. Adopting T20 should need the per-kind panel to agree, not `impute_score` alone. `impute/induced/calibrated/mean_alpha` is already a signed-shrinkage readout to show beside the signed-bias term. |
| 8 (MNAR) | Fitting the calibration on MNAR-shifted truth is an oracle, like K. MNAR arms read neither as a headline. |
| 9–12 | No overlap. |

## 2. Workstreams

Status in [TASKS.md](../TASKS.md): T21 to T27.

**"Protected"** means it changes training behaviour or the split order, which AGENTS.md reserves.
Such a change needs a flag, a tag, a backfill, an ADR and separate authorization.

**Code authorization.** Every workstream with code needs the user's authorization, protected or
not (the standing rule that code changes are authorized separately).

### 2a · Reporting: one shared helper and the thesis table (T21)

**Goal.** One module that every report reads through. It closes threat 5 and the three coexisting
bar definitions behind finding 1. It also advances T16 (one study report).

`scripts/experiments/report_common.py`, under strict mypy:

- **Selects** parents by tags: `experiment`, `arm`, `dataset`, `seed`, `run_role=parent`, plus
  `is_mirror=false` and `FINISHED`. It never selects by artifact mtime, as `barcheck_verdicts.py`
  does.
- **Takes the bar from `summary.OUTSIDE_THE_BAR`**: the best of the logged baselines not in that
  set. One definition, chosen on the CV mean per run, as ADR 0007 decision 8 chooses it.
  - It does not use the mean-of-three-seeds minimum (ADR 0017's table).
  - It does not use the four-name tuples in the 14 report scripts, plus `calibration_report.py`
    and `retune_report.py`.
- **Picks K or H by the `decode_checkpoint` tag.**
  - On `loss` runs, K is the `induced_checkpoint` family.
  - On `induced` runs, K is the headline.
  - The store's `gap_to_*` keys on E34 and E35 are H-based and are never read as K.
- **Opponents:**
  - the bar of 5;
  - missForest;
  - the best of all six;
  - the per-type bar (2d);
  - later, the linear arms (2b) and the tuned bar (2c).
- **Statistics:**
  - the project's verdict rule, with the 15-pair t-interval and seed agreement;
  - an equivalence call under the SESOI (D3);
  - a seed random-effects interval as a sensitivity column: a one-way model on the seed means,
    in numpy (statsmodels is not a dependency);
  - Wilcoxon signed-rank over tables, with Holm across opponents. With 9 tables the smallest
    attainable two-sided p is 0.0039.
- **Every count is printed with its test named beside it,** for example "better 16 / none 3 /
  worse 2 (fold-pair t, seed agreement)".

**The thesis table, from existing runs only:**

- **101/202/303 (primary, today's default)** through a cell map:
  - E34 K on 16 variants;
  - E38 ALL on pendigits_20nan and letter_20nan;
  - E39 D on the three 40nan variants that lost their files.
- **42/7/13 (E35),** labelled "pre-ADR-0019". It used the old files on three variants and the old
  encoder on two. Those five are never pooled with the other sets.
- **404/505/606 (E40)** on its 12 variants.
- **Development label.** The 21 variants and these three seed sets are the development benchmark,
  and the table says so (threat 2).

**Answers:** RQ1's headline, finding 1, threat 5, and the bar-definition part of finding 6 in the
2026-10-07 code review.

**Depends on:** decisions D2 and D3. ADR 0020 records them.

**Protected:** no. It is read-only on the store.

**Effort:** 4–6 h of code with seam tests. The tests worth having catch silent wrongness:
- the K-or-H choice by tag;
- a bar that excludes `OUTSIDE_THE_BAR`;
- the verdict rule on a known case.

Building the table takes seconds.

### 2b · New baselines: condexp, inductive SoftImpute, ICE (T22)

**Goal.** Measure, on the decoder's own cells, how much of the model's lead on small numerical
tables is linear-Gaussian or low-rank structure (finding 2). Also add the classical ICE row a
reader expects (finding 3).

**Where the code lives.** A new module `src/training/linear_baselines.py` under strict mypy. It
implements ADR 0007's `BaselineImputer` protocol:

- condexp and SoftImpute in numpy;
- ICE as a `MissForestImputer` subclass that overrides `_fit_one` and `impute`. It uses the
  sequential per-round transform, because the inherited replay is invalid for linear learners.

E41 first measures it offline: a driver calls it through `recompute_folds(..., fold_scorer=...)`.
`imputation_baselines.py` and `imputation_metrics.py` are never edited (cache keys).

**Then two tiers of baselines, every one of them logged (the user's design, 2026-10-16).** This
replaces D5's first answer, "external, offline only".

- **The weak tier is today's bar of 5,** unchanged: mean_mode, knn5, knn10, hgb and lgbm.
  `baseline/best`, the tag `best_baseline/<population>` and `gap_to_best_baseline*` keep their
  names and values on every run, so nothing recorded moves.
- **The strong tier:**
  - missForest, which leaves its "external" status;
  - condexp, SoftImpute and ICE, as a fourth cache family (`linear`);
  - later, any stronger baseline that is cheap enough to run live.

  The tuned bar (2c) stays offline while its cost is about 12× a fit.
- **Each tier logs its own best:**
  - `impute/<population>/baseline/best_strong/*`, chosen once per run on the CV mean, as ADR 0007
    decision 8 chooses the weak one;
  - the tag `best_strong_baseline/<population>`;
  - `gap_to_best_strong_baseline` and its `_pct`.

  `gap_to_missforest` stays. The best of all is the lower of the two bests, and the helper derives
  it.
- **Code:** `summary.py`, the module that sets the tiers, plus the new family in `baseline_cache.py`
  (a new family leaves the existing entries valid). No training path changes.
- **Backfill:** in the ADR 0017 pattern, over the roughly 1,000 imputation parents. It is checked
  against E41's offline CSV to the last digit and holds the store for about 2 h, in a morning gap
  between studies.
- **Order:** E41 (offline) → ADR 0021 (the tiers) → code → backfill. E44 does not wait: the
  backfill writes the strong tier onto its runs too, since baselines never see the model.

**Answers:** RQ1's baseline breadth and threat 1, the opponent side.

**Depends on:**
- 2a's helper for the verdicts;
- decisions D5–D7: the rules are fixed before the first score.

**Protected:** no. The weak bar never changes; the strong tier comes through ADR 0021.

**Effort:**
- about 1 day of code, plus stubs for `sklearn.linear_model` (ICE only);
- about 1 h of CPU for 162 (variant, seed) pairs in four processes.

### 2c · Fairness: a baseline tuning budget and a deployable checkpoint (T23)

**Two parts:**

1. **Baseline budget (threat 1).** Each family gets a declared, validation-only selection, chosen
   per fold on the same signal the model's checkpoint uses, with the chosen setting logged. It is
   built as new subclasses and an offline driver (`budgetskeptic_*.py` as templates), off-store.
   The untuned bar stays as the continuity row.
2. **A deployable checkpoint (threat 3).** `DECODE_CHECKPOINT=masked` keeps the epoch with the best
   λ-free `impute_score` on the fixed `hidden_validation` mask:
   - the machinery is at `decoding.py:345-352`;
   - `_ValidationGapScore` at `:491-559` is the template.

   **Diagnostic readouts under the K default:**
   - H as `impute/*/loss_checkpoint/*`, which today is scored only when `by_gaps` is false
     (`:291-296`);
   - M as `impute/*/masked_checkpoint/*`.

   Both come from the same trajectory. They are eval-mode forward passes, so the run draws
   nothing more, and the readouts equal what a run under that criterion would keep. ADR 0015
   verified this identity for K.

**Answers:** threats 1 and 3, and whether the result transfers to RQ2/RQ4.

**Depends on:**
- **budget:** D8 and 2a;
- **checkpoint:** D9 and D10, plus code authorization.

**Protected:**
- **budget:** not protected, but it changes what the bar means, so ADR 0022 amends ADR 0007
  decision 5 if the tuned bar becomes an opponent of record;
- **`DECODE_CHECKPOINT=masked`:** protected (training behaviour; ADR 0024).
  - **Flag:** `--decode_checkpoint masked`.
  - **Tag:** the existing `decode_checkpoint` tag gains a value, so earlier runs keep theirs and
    no backfill is needed.
  - **Default:** K.

**Effort:**
- **budget:** about 1 day of code, then 25–40 CPU-h (estimate);
- **checkpoint:** half a day of code with tests, then E43's 27 cells.

### 2d · Metrics: the per-kind panel and the per-type bar (T24)

**Goal.** Threat 4: the metric pair decides the missForest and categorical verdicts. Heavy tails
decide kc2 and biodeg.

**The per-type bar (7b), as this plan reads it.**

- For each kind, the best baseline is chosen separately on the CV mean of that kind's ratio:
  - numerical cells: RMSE over the mean/mode RMSE;
  - categorical cells: error over the mode's error.
- The two are recombined with `impute_score`'s cell weights. This is what `metric_probeB.parts()`
  computes.
- It never sits above the bar of 5.
- **It differs from the bar only on mixed tables.** Those are the four credit-g levels and
  electricity_20nan in the development set, plus electricity at 40, 60 and 80nan in the
  confirmation set. There, cell-count weighting can hide a categorical loss: electricity's
  categorical ratio is 0.81 for the model against 0.71 for hgb.
- **Computable from stored files:**
  - for the four ADR 0007 baselines on every run;
  - for all six only where lgbm and missForest are in the fold CSV.

**What each source can give:**

| measure | from the fold CSVs and the per-column files (model and every baseline in them) | from the server's fold ledgers (model only) | needs an offline recompute (baselines) | needs a `score_cells` change |
|---|---|---|---|---|
| per-kind RMSE, MAE and error ratios; macro-F1 gap; the per-type bar | yes | yes | no | no |
| per-column ratios to the mean/mode fill; a column-normalised numeric ratio | yes (`per_column_imputation.csv`; the model side is the headline: H on E34/E35, K on E38–E40) | yes | no | no |
| signed bias per column; balanced accuracy; dispersion ratio; 1-D Wasserstein; category-frequency TV; the twin/no-twin split; the share of context-free rows | no | yes | yes, since baseline fills are never persisted | no |
| any of them logged live per run, with a gap key in the store | no | no | no | yes (ADR 0004 amendment; misses all 7,218 cache entries) |

**Recommendation (D16).**

- Compute the panel offline:
  - stored files for the ratios;
  - ledgers for the model;
  - a `recompute_folds` scorer for the baselines' fills.
- Leave `score_cells` alone for now.
- Pre-register the panel as secondary. A claim stands only where the primary and the per-kind
  readings agree; the disagreements are listed.
- Balanced accuracy is preferred over macro-F1, which rare-label noise inflates.

**Answers:** threat 4, plus the signed-bias term the MNAR arm needs.

**Depends on:** D1 and D16; 2a's helper.

**Protected:** no. Evaluation only, and offline.

**Effort:** 2–3 h for the stored part; about 1 day with the ledger and recompute parts.

### 2e · Confirmation on the 15 unused variants (T25)

**Goal.** Threat 2: every default was chosen on the same 21 variants and one MCAR draw. One frozen,
pre-registered read on the 15 variants that no decision has touched, on never-used seeds (E44).

**Scope.** The 15 are new draws and new missing levels of 6 of the 9 tables, not new tables. That
limit is stated in the ticket. Adding 2–3 new OpenML tables is a separate option (D13).

**Inputs fixed before the run:**

- **seeds:** D11;
- **ADR 0018 inheritance:** D12;
- **splits:** D13;
- **readouts:** H, M and KC; D9 and D15;
- **the reading:** D14;
- **context-free rows:** D17. electricity_80nan has 16.8% of its rows fully missing.

**Answers:** threat 2 and the generalisation side of RQ1.

**Depends on:**
- 2a, for the reading;
- 2c's code, if M is wanted in its cells;
- 2b and 2c's offline opponents: their *definitions* (grids, selection signal, rank and clip
  rules) must be frozen in their tickets before E44 launches. The scores themselves can be computed
  after the GPU pass, because baselines never see the model. Only the cache warm-up (the bar of 5
  and missForest) has to come before it, to save GPU time.

**Protected:**
- the runs need no code: a new runner, which prints the resolved configuration, and a queue cloned
  from `run_calibration_queue.sh`;
- `StratifiedGroupKFold` would be protected (split order; ADR 0026), and the recommendation is not
  to use it here.

**Effort:**
- **summed GPU time:** 33 h without inheritance, 46 h with it;
- **wall time on 4 queues:** about 8–9 h, or 11–12 h with inheritance (section 3, E44).

### 2f · MNAR: a smoke family and the t2 readout (T26)

**Goal.** Item 8: does the MCAR lead survive value-dependent missingness?

**Data and arms.**

- A new `datasets/generate_mnar.py` writes `<base>_probit_60nan.csv` for biodeg, kc2 and vehicle.
  It hides each column's MCAR count of cells by weights Φ(λ·z − c). Do not call these cells
  "self-masked"; that term is taken (CONTEXT.md).
- Arms G and Mx on the MNAR and MCAR 60nan variants: 36 cells.

**Item 9 (t2).**

- **The MCAR half needs no compute:**
  - E36 is functionally a tied-token arm;
  - E31–E33 come from the stored CSVs through the helper.
- **The MNAR half** is E45's G−Mx contrast.

**Answers:** RQ1's MNAR arm, RQ4, and todos t2, t5 and t7.

**Depends on:**
- D18: the store or not, and the mechanism parameters, fixed before generation;
- 2c, for an honest headline (K is an oracle under MNAR);
- 2d's ledger code, for signed bias.

**Protected:** no.
- The generator writes new variant files.
- `generate_splits.py` is untouched; it overwrites in place.
- Signed bias comes from the ledgers.

**Effort:** 5–7 h of code; under 1 h of wall time on 4 queues.

### 2g · Later: MaCoDE, TabPFN v2 and classification step zero (T27)

- **MaCoDE (item 10).**
  - Vendored under MIT behind an inductive wrapper.
  - Its bins bug comes first: `CustomDataset` keeps 10 bins while the model is built for 50.
    Its other blockers are in the merge digest's item 11.
  - 10–14 h of code; 3–5 GPU-h off-store.
- **TabPFN v2 per column (item 11).** It needs a separate venv pinned to `ModelVersion.V2`
  (approval: D21), 5–8 h of code and 6–10 GPU-h, uncertain by about 2×.
- **Classification step zero (item 12):**
  - a native-NaN GBDT bar and impute-plus-indicator arms, on CPU, on the sweep's folds;
  - an `EPOCHS_PRE=0` arm (an override, no code);
  - `--cv_folds 10` for 80/10/10.

  Ranking classification trials on validation needs an `opt.py` change: search behaviour, an ADR.

**Protected:**
- MaCoDE and TabPFN: no, as new baseline modules;
- classification's search objective: yes.

**Status:** backlog, in GPU gaps only, after 2a–2e.

## 3. Experiment drafts (one ticket each)

**Rules for every experiment:**

- The E-number is assigned when the ticket and the log entry are committed together, in this
  order unless the user reorders.
- Each draft marks with ⚑ what must be decided **before any score is seen**.
- No optional stopping. A crash is fixed and its cell is rerun from the start.
- Every table is built by script from the store or from the study's per-fold CSVs.

### E41 · Linear baselines and the missing per-fold baselines, on the decoder's cells

- **Hypothesis.**
  - The review's probe found, by seed means, that condexp and inductive SoftImpute fall below the
    bar of 5 on about 7 induced variants, and that SoftImpute is below the model on kc2_20nan
    (`merge_digest.txt`, items 2–3).
  - ICE with the sequential transform never beats the model.
  - Paired fold by fold, the kc2_20nan flip may not survive: its intervals are the widest of the
    21.
- **Arms.**
  - condexp, inductive SoftImpute and ICE, plus their range-clipped forms.
  - lgbm and missForest, recomputed through `BaselineScorer(families=("lgbm", "missforest"))`
    wherever a fold CSV lacks them. These are cache hits on gorgona8, from ADR 0017's backfill.
- **⚑ Rules,** frozen in the ticket before the first fold is scored:
  - **condexp:**
    - pairwise-complete covariance turned into a correlation;
    - shrinkage R ← 0.9R + 0.1I;
    - eigenvalue floor 1e-3;
    - E[x_m | x_o] on numerical cells only;
    - categorical columns filled with the mode;
  - **SoftImpute (D7):**
    - 90% energy share;
    - no p//2 cap, with k at most p−1;
    - hard-impute fit;
    - PPCA posterior, with s² the mean of the discarded eigenvalues;
  - **ICE (D6):**
    - one-hot features with the target's own block excluded;
    - BayesianRidge for numbers and multinomial LR for categories;
    - the sequential per-round transform;
    - the round count from `still_converging`;
  - **clip:** every linear fill clipped to the column's training [min, max]. ICE clips inside
    every round; condexp and SoftImpute clip once at the end;
  - **the hidden-mask rule:** NaN is written at `hidden | isna` before any computation.
- **⚑ Cell map:**

  | seed set | model column | variants |
  |---|---|---|
  | 101/202/303 | E34 `induced_checkpoint` | 16 |
  | 101/202/303 | E38 ALL headline | pendigits_20nan, letter_20nan |
  | 101/202/303 | E39 D headline | credit-g_40nan, kr-vs-kp_40nan, spambase_40nan |
  | 42/7/13 | E35 `induced_checkpoint` | 21; 5 labelled "pre-ADR-0019/0018" and never pooled |
  | 404/505/606 | E40 headline | 12 |

  That makes 162 (variant, seed) pairs, 810 folds.
- **Integrity, checked before reading anything:**
  - mean/mode equals 1 ± 1e-9 on every fold;
  - knn5 and hgb reproduce each cell's logged fold values to 1e-9 (same machine).
- **Primary measure.** Per variant and seed set, arm − bar of 5 and arm − model K on the induced
  `impute_score`, over 15 fold pairs, under the verdict rule.
- **⚑ Reading:**
  1. **material:** an arm is "better than the bar" by verdict on at least 2 variants in *both*
     101/202/303 and 42/7/13. ADR 0021 then weighs bar membership.
  2. **flip:** an arm is "better than K" by verdict on any variant. That is a model loss, reported
     in the best-of-all row.
  3. **otherwise:** the arms are completeness rows.
- **Secondary:** the masked population; the clipped vs unclipped forms; the ranks chosen; and
  ICE's rounds.
- **Where it runs:** the gorgona8 CPU, in 4 processes with 4 threads each, beside any GPU study. It
  is off-store. The per-fold CSV goes to `/scratch2/diogoneiss/trident-<date>-linear-baselines/`.
- **Cost:** about 1–2 CPU-h on cache hits.

### E42 · An equal, validation-only tuning budget for the baselines

- **Hypothesis.** A declared budget lowers the bar by about 0.01–0.035 on a few variants.
  - The probe's alternative hgb recipe was picked after seeing seed-42 test scores, so it proves a
    stronger bar exists but does not measure it.
  - Under ADR 0019 the kr-vs-kp_40nan flip the review cites is likely gone: arm D's gap to lgbm is
    −0.045 [−0.054, −0.037].
- **⚑ Grids,** frozen before scoring:
  - **KNN:** k ∈ {5, 10, 20, 50};
  - **hgb and lgbm:** 12 configurations, from learning rate {0.05, 0.1} × iterations
    {100, 200, 400} × depth or leaves {3 or 7 leaves, the default}, with the defaults included;
  - **missForest:** `max_features` ∈ {1/3, 1/2, sqrt} × trees {100, 300};
  - **all learned families:** a shrink-toward-the-mean α ∈ {0, 0.25, 0.5} for numerical fills at
    60–80nan.
  - **hgb:** the early-stopping split is drawn by original row; `early_stopping` is never turned
    off.
- **⚑ Selection (D8).**
  - Per fold, on the validation rows' induced gaps: the signal K uses.
  - The masked validation cells are the sensitivity reading.
  - Fit on the training fold only, with no refit on train plus validation, because the model does
    not refit either.
  - **The seam has to be built.** `recompute_folds`'s `fold_scorer` factory receives only
    `(train, numerical, categorical)`: no fold index, no validation split and no sibling. The driver
    rebuilds the folds with `build_folds` and slices the `_00nan` sibling itself to score the
    validation gaps. This is inside the 1-day estimate.
  - Only the chosen configuration is scored on test. The validation score of every configuration
    is logged.
- **Cells:**
  - the development set on 101/202/303 and 404/505/606 (today's default);
  - E44's variants on its seeds, computed before E44's GPU run.
- **Primary measure.** Per variant, the tuned bar − the untuned bar, and K − the tuned bar, under
  the verdict rule.
- **⚑ Reading.** Descriptive; no default changes. The thesis's opponent of record follows D4.
- **Cost:** 25–40 CPU-h, estimated from ADR 0007's and ADR 0016's fit times. On 4 processes that
  is about 8–10 h of wall time, and it can run beside the GPU queue.

### E43 · A deployable decode checkpoint

- **Prerequisite:** the 2c code (D9), landed while no study is live.
- **Hypothesis.** K's gain over H at 80nan (E34/E35: 0 of 3 variants beat the mean/mode fill under
  H, 2 of 3 under K) is partly a better stopping rule. K also improves the masked test cells, which
  it never chose on. A criterion that needs no gap truth (M) recovers part of it.
- **Arms,** within each run, on one trajectory:
  - K, the default headline and an oracle;
  - H, the loss criterion;
  - M, the masked `impute_score` criterion;
  - KC, `--score_calibrated`, reported as an oracle.
- **⚑ Variants:** the 9 at ≥60nan (kr-vs-kp, spambase and credit-g at 60 and 80nan, plus
  vehicle_60nan, biodeg_60nan and kc2_60nan). That is where K's verdicts fall.
- **⚑ Seeds:** 111, 222 and 333, never used. 27 cells.
- **Primary measure.** M − H and M − K on the induced `impute_score`, over 15 within-run pairs,
  under the verdict rule.
- **⚑ Reading:**
  1. **M suffices:** K is better than M on none of the 9. Transfer claims then report M.
  2. **M helps:** M is better than H on at least 2 of the 3 80nan variants and worse on none. M
     becomes the deployable headline (ADR 0024); K is reported as the oracle bound.
  3. **otherwise:** H is the deployable headline, and "beats the mean/mode fill at 80nan" is
     labelled oracle-only.
- **Secondary:**
  - each readout against the bar of 5 at 80nan;
  - the chosen epochs;
  - the masked population;
  - KC − K, a third replication of E40's sign at 60 and 80nan.
- **Cost:** about 7.5 summed GPU-h, about 2 h of wall time.

### E44 · Confirmation on the 15 unused variants

- **Hypothesis.** The frozen configuration's lead over the bar on the development set generalises
  to new draws and levels of the same tables. Today's default on 101/202/303 is 16 better, 3 no
  difference and 2 worse by the review's rebuild from fold CSVs (`barcheck_out.txt`); T21
  regenerates it.
- **⚑ Configuration:** frozen, as a default run of the checkout reads it. There are no new files,
  and ADR 0018 is not inherited unless D12 says otherwise. The runner prints the resolved
  configuration of every cell.
- **⚑ Readouts:**
  - K headline;
  - H and M, if E43's code exists;
  - KC through `--score_calibrated`.

  **Without E43's code,** the cells run as
  `--decode_checkpoint loss --score_induced_checkpoint --score_calibrated`. That is the same
  trajectory: K is in `induced_checkpoint`, H is the headline and KC is
  `induced_checkpoint_calibrated`. The helper reads K by tag.
- **⚑ Variants and seeds.** The 15 variants on seeds 707, 808 and 909 (D11): 45 cells.
- **Run first:** one electricity_40nan cell, smoked with `--no-mlflow` before launch. Both of its
  earlier runs failed (on the laptop).
- **⚑ Opponents.** Their definitions are frozen in the tickets before launch; the scores can come
  after the GPU pass, because baselines never see the model:
  - the bar of 5 and missForest, live in each cell, with the cache warmed beforehand;
  - the tuned bar (E42), offline;
  - the linear arms (E41), offline;
  - the per-type bar, from the fold CSVs.
- **Primary measure.** Per variant, K − the opponent of record (D4) on the induced
  `impute_score`, over 15 fold pairs, under the verdict rule. Each variant is labelled better,
  worse, equivalent within the SESOI, or undecided.
- **⚑ Reading (D14), as recommended:**
  - **supported** if K is better on ≥8 of 15 and worse on ≤2;
  - **refuted** if worse on ≥5, or better on ≤4;
  - **mixed** otherwise.

  The six 80nan variants are also reported apart, since on the development set the model sits
  near the mean/mode fill there.

  **The table-level Wilcoxon is secondary here,** although ADR 0020 B makes it the primary method
  claim on the development set. E44 has only 6 tables, so its smallest two-sided p is 0.031, and
  they are the same tables as the development set. What E44 adds is new draws and levels, so the
  per-variant tally carries its reading.
- **Secondary, pre-registered:**
  - the tallies against missForest, the best of all, the per-type bar and the tuned bar;
  - the metric panel (2d), computed offline on E44's ledgers and recomputed baseline fills;
  - K − H and K − M;
  - KC − K, with T20's thresholds 40 and 60 compared;
  - the share of context-free rows, and the scores with those rows excluded (D17);
  - the twin/no-twin split on letter and kc2.
- **Read once.** Nothing is extended or rerun on its result.
- **Cost estimate,** from the cell times in section 0, fact 10:

  | cells | minutes each | total |
  |---|---|---|
  | electricity, 9 cells | about 110 | 990 min |
  | letter, 9 cells | 59 (default) or 111 (ALL) | 531 or 999 min |
  | pendigits, 9 cells | 38 or 73 | 342 or 657 min |
  | vehicle, biodeg, kc2, 18 cells | | about 110 min |

  - **Without inheritance:** about 33 GPU-h, about 8–9 h of wall time.
  - **With inheritance:** about 46 GPU-h, about 11–12 h.

### E45 · MNAR smoke family (and t2's MNAR half)

- **Prerequisite:** the generator, with a row-alignment and rate test.
- **⚑ Mechanism.**
  - Probit weights Φ(0.5·z_j − c), large values hidden, at exactly the MCAR sibling's per-column
    count.
  - c is fixed in the ticket.
  - Weighted sampling at a matched count is not exactly Le Morvan and Varoquaux's probit; the
    ticket says so.
- **Arms:**
  - **G:** today's default with the readouts H, K and M (or the `loss` form above);
  - **Mx:** `--decode_gap_token null --decode_checkpoint loss --score_column_wise
    --score_induced_checkpoint`.
- **Cells.** Both arms on the MNAR and the MCAR 60nan variants of biodeg, kc2 and vehicle, on seeds
  42/7/13: 36 cells.
  - G on MCAR must equal E35 to the last digit; that is the integrity check.
  - The MCAR controls re-run on the server and pair with the MNAR cells.
- **⚑ Headline:** H, or M once E43 exists. K and KC are oracles under MNAR.
- **Measures:**
  - model − bar and model − missForest per mechanism, read only as within-variant ratios or in
    original units;
  - G − Mx under MNAR against MCAR (t2's positive control);
  - K − H under MNAR against MCAR (how optimistic the oracle becomes);
  - signed bias per column: from the ledgers for the model, from a recompute for the baselines.
- **⚑ Reading.** A smoke test: descriptive, plus go or no-go for the full family. The full family
  adds spambase and the 20nan rung, about 48 cells, plus the biodeg tilt dial's 24.
  - **Go:** the generator's tests pass, all 36 cells finish and the integrity check holds.
- **Cost:** under 1 h of wall time.

## 4. ADRs this needs

Numbers are proposed in the order decided.

**The minimum before E44 launches:**

- ADR 0026, or its content written into the E44 ticket;
- ADR 0024, if M is in the cells.

**The others** can follow their studies.

| ADR | question | options | recommendation |
|---|---|---|---|
| 0020 · reporting protocol | How counts and claims are read. | (A) seed-mean counts as today. (B) Per-opponent verdict tallies plus a table-level Wilcoxon with Holm as the primary method claim, the SESOI, and the test named beside every count. (C) B with a Bayesian signed-rank test. | **B.** Names the bar definition in ADR 0017's table header. Labels development vs confirmation. Gives every report one helper. |
| 0021 · baseline tiers | Where do condexp, SoftImpute and ICE (and missForest) sit? | (A) In the bar: retags `baseline/best`, and the headline count moves (the review's probe: 19 to 17 by means). (B) External, like missForest, each with a gap of its own. (C) A "linear" family, external, offline only. (D) **Two tiers, all logged** (the user's design): the bar of 5 is the weak tier, unchanged; missForest and the linear family form the strong tier with its own best, tag and gap. | **D**, decided 2026-10-16. Written after E41; the live family and backfill follow. |
| 0022 · baseline budget (amends 0007 d5) | Do baselines get a tuning budget, and which bar is of record? | (A) Fixed bar, with the limit stated. (B) An offline validation-only budget; the tuned bar co-reported. (C) Live tuned baselines in every run: about 12× the fit cost and new cache families. | **B.** Select on the same signal as K, with no refit on train plus validation. Opponent of record per D4. |
| 0023 · per-type bar | How strong the opponent is on mixed tables. | (A) None. (B) The per-type bar as a reported secondary opponent and the per-kind panel. (C) The per-type bar as the bar of record. | **B.** It moves only 5 development variants and 3 confirmation variants, and as the bar of record it would combine two imputers into one that no reader can run. |
| 0024 · deployable checkpoint (amends 0015) | Which checkpoint carries claims that must transfer? | (A) K stays the headline everywhere, with the limit stated. (B) Code `masked` behind its flag, keep K as the default, add the H and M readouts, report the deployable readout for transfer claims and K as the oracle bound. (C) Make H or M the default. | **B.** Status "proposed" until E43; E43's reading picks M or H. T20 waits on it (D10). |
| 0025 · metric suite (amends 0004 d6) | Which secondary metrics are pre-registered, and where they are computed. | (A) `impute_score` only. (B) The offline panel, with claims only where the per-kind readings agree. (C) A live `score_cells` change, which misses the whole cache. | **B.** Revisit C only if a panel metric becomes primary. Land it between studies with a cache warm pass. |
| 0026 · confirmation protocol and splits | How E44's set is split and what counts as frozen. | (A) The same `StratifiedKFold` as the development set, plus a twin/no-twin readout. (B) `StratifiedGroupKFold` on a feature-vector hash for the confirmation set only: protected, flag and tag, and the two sets stop being comparable. (C) B everywhere. | **A.** Twins favour kNN, so they are conservative for the model. Grouped splits wait for the clinical tables. The ADR also records the seeds, the inheritance rule, the 21 variants as the development set, and retiring a seed set once it has decided something. |

## 5. Schedule (two weeks from Saturday 2026-10-17, GMT-3)

**Rules of the machine:**

- **The store.** One GPU study at a time writes it. CPU passes only read the store and the cache
  and write the cache's JSON files and their own CSVs, so they can run beside any study. A store
  backfill (ADR 0021 A or B only) runs in a morning gap between studies, after a backup through
  SQLite's API (about 2 h hold).
- **Code.** It lands only while no study is live, because every cell is a fresh process that reads
  the checkout (E38's lesson). Then: the unit suite, mypy included, and one `--disable_mlflow`
  smoke.
- **CPU passes.** At most 4 processes of 4 threads each, with `missforest.THREADS` and
  `lightgbm_baseline.THREADS` set to 4. Each driver checks `free -g` first.

**The calendar assumes the decisions arrive on Saturday 10-17 in the morning.** Code is counted
honestly, one item at a time:

| code item | effort |
|---|---|
| T21's helper | 4–6 h |
| E41's module and driver | about 1 day |
| E42's driver | about 1 day |
| E43's protected change with its tests | about half a day |
| E44's runner and queue | about 2 h |
| E45's generator and runner | 5–7 h |

**What must precede E44's launch.** E44 needs none of the offline code. Before launch, only these
must exist:

- its ticket, with E41's and E42's definitions frozen;
- its runner;
- one electricity_40nan smoke;
- the cache warm-up.

So the choice that sets the calendar is whether E44 carries the M readout (D9).

**Path A, a minimal E44.** It launches on Saturday night.

- **Readouts:** H, K and KC through the `loss` form. There is no M.
- **Opponents live:** the bar of 5 and missForest.
- **Opponents computed later:** the tuned bar and the linear arms, offline.
- **The cost:** the confirmation set never gets a deployable readout without rerunning all 45
  cells (about 33 GPU-h).

**Path B, a full E44 (recommended).** It launches one night later.

- Saturday is spent on E43's code, which runs that night.
- E44 then carries M, and threat 3 is answered on the one confirmation read.

| when | GPU queue (4 trainers), path A | GPU queue, path B | CPU, beside it (both paths) |
|---|---|---|---|
| Sat 10-17, day | idle; E44's ticket, runner and smoke | idle; E43's code, tests and smoke, and its ticket | the cache warm-up for E44: the bar of 5 and missForest on 15 variants × 707/808/909, about 4–6 CPU-h, mostly electricity's KNN. E41's and E42's tickets (definitions only). |
| Sat → Sun, night | **E44** (45 cells, ~8–9 h; ~11–12 h with inheritance) | **E43** (27 cells, ~2 h) | the warm-up finishes |
| Sun 10-18 | E44's primary report; then E43's code | E43's report; E44's ticket, runner and smoke | T21's helper and the thesis table |
| Sun → Mon, night | E43 (no longer feeds E44) or E45 | **E44** (to about 06:00–08:00 on Monday) | E41's module, then its pass (~1–2 CPU-h) |
| Mon 10-19 – Tue 10-20 | E45, once its generator is ready | E44's primary report; E45 on Tuesday night | E42's driver, then its pass on the development set and on E44's folds (~8–10 h on 4 processes); the E41 arms on E44's folds |
| Wed 10-21 – Fri 10-23 | a free night: T12 (a lower `LR_PRE`), or the MNAR full family if E45 says go, pre-registered under its own E-number | the same | the panel on the development ledgers and on E44's; the t2 MCAR readout; delta-adjusted hgb on the MNAR variants; ADR drafts 0020–0026 |
| week 2 (10-26 to 10-30) | MaCoDE's off-store grid (3–5 GPU-h) and TabPFN v2 (6–10 GPU-h, if D21), each in a free night; classification step zero (`EPOCHS_PRE=0`) in idle nights | the same | MaCoDE vendoring (1.5–2 days of code); the classification GBDT-MIA and impute-plus-indicator arms (a few CPU-h). A linear-family backfill, only if ADR 0021 says A or B, takes one morning between studies (about 2 h store hold). |

**Who writes the store:**

- each night's GPU study, alone;
- a backfill, only in a morning gap.

**When code lands:** only during the daytime, in the gaps between studies.

**What runs in parallel:**

- every CPU pass with any GPU study;
- E41 with E42;
- the cache warm-up with E43 and E45.

**What never overlaps:**

- two GPU studies;
- a backfill with a study;
- a code landing with a live study.

## 6. Risks and traps

- **Cache-key bytes.** Never edit `imputation_baselines.py` or `imputation_metrics.py`. Doing so
  misses all three families, 7,218 entries, and costs hours of KNN on electricity and minutes of
  missForest per fold. New imputers and metrics go into new modules. A new family added to
  `_FAMILIES` leaves the existing entries valid.
- **The hidden mask is not NaN.** `score_baselines` passes a boolean `hidden` mask, and the frame
  still holds the truth of every masked cell. Every new imputer must write NaN at `hidden | isna`
  before fitting or predicting, or it reads the answer.
- **E34/E35 CSVs lack per-fold lgbm and missForest.** The backfills wrote only the store. E41
  recomputes them through the cache. E38's CSVs share E34's folds on 101/202/303 for 16 variants;
  the early cells logged a bar of 4 on 5 variants.
- **E34/E35's store headline and gap keys are H.** They were `decode_checkpoint=loss` runs and
  `summary.py:174` reads `{population}/impute_score`. A store-native query of E34/E35 returns H
  (16, not 19, by means). Only the helper's tag-aware pick reads K.
- **`metrics/<variant>_metrics.csv` gets overwritten.** It is a fixed path under the checkout;
  45 such files exist, and six sweep copies were already overwritten by imputation runs. Queues
  pass `$STATE/metrics/<cell>`. Every ad-hoc or smoke run must pass a private `--metrics_dir`.
- **CRLF `.sh` files and the sync push path.** On Windows (`autocrlf=true`, no `.gitattributes`)
  the working tree holds CRLF. `sync_remote.py code --apply` ships working-tree bytes and mirrors
  `.git`, so the next code push from Windows would break every queue script here.
  - **Until it is fixed:** code moves only through git (commit here, `git push`, Windows pulls),
    and no `code --apply` or `push --apply` runs from Windows.
  - **The fix:** a `.gitattributes` with `*.sh text eol=lf`. It is a code change (D20).
  - **Also:** `prepare` and `install_store` still read a dangling link as "no store" (review item
    10).
- **The template scripts hard-code Windows and scratchpad paths.** 16 of them name
  `C:\Users\Diogo Neiss\...` or JSON inputs that were not kept. `barcheck_*` selects cells by the
  artifacts' modification-time windows. Port the logic, not the files:
  - select by tags through the helper;
  - make paths relative to the checkout;
  - copy rather than import report modules that open `mlflow.db` at import time
    (`selection_replication_report.py:63`, `:92`).
- **The inherited ICE transform.** `MissForestImputer.impute` (`missforest.py:147-163`) replays the
  kept round's models from the training means. For linear learners that scores 0.6 to 19× the
  mean/mode fill. ICE must override `impute` with the per-round sequence.
- **missForest's own transform.** Three of ADR 0017's near-ties (electricity_20nan,
  pendigits_20nan and vehicle_60nan) could flip under per-round application (finding 4,
  one-seed projection). An optional `missforest_seq` arm in E41 checks it on small tables; it is
  memory-heavy on spambase.
- **Runners that do not pin.** Old runners resolve pendigits_20nan and letter_20nan to ADR 0018's
  encoder, and E36's check compares only baselines. Every new runner prints the resolved
  configuration, and each study has an integrity check against a known cell.
- **Seeds.** 404/505/606 are spent. The confirmation's triple is used nowhere before E44.
- **electricity.** electricity_40nan failed twice on the laptop, so smoke it first. electricity_80nan
  has 7,593 rows with every feature missing, so report the share of context-free rows. Its KNN
  transform is the dearest baseline step; warm the cache first.
- **Generated data.** `processed_datasets/` is gitignored, and `generate_splits.py:202` overwrites
  in place. MNAR variants and any second MCAR draw go to new names; the generator is committed.
- **The masked population across levels.** Its realised rate is not monotone in the level, so it
  is never read across levels. Level effects are read within a table or with table fixed effects.
- **Other sessions' files.** `graphify-out/*` and `scripts/experiments/default_imputation_overview.*`
  are another session's uncommitted work: never stage them, and do not run `graphify update` while
  they are uncommitted.
- **Store hygiene.** `is_mirror = false` in every cross-experiment query. Never `mv`, `cp` or `rm`
  the `mlflow.db` link.

## 7. Decisions only the user can make

Each question comes with the recommended answer.

**Reporting and metrics**

- **D1.** Is item 7b, "the per-type bar", the best baseline chosen separately for numerical and for
  categorical cells, recombined with `impute_score`'s cell weights?
  - **Recommended:** yes. Report it as a secondary opponent with the per-kind panel, not as the
    bar of record (ADR 0023 B). If you meant something else, say what; it was not in the wiki.
- **D2.** Is the primary method claim the table-level Wilcoxon, with Holm across opponents, and are
  the per-variant tallies descriptive, with the test named beside each count?
  - **Recommended:** yes, with a seed random-effects column as the sensitivity check (ADR 0020 B).
- **D3.** What is the smallest effect of interest?
  - **Recommended:** 1% of the opponent's mean, used for equivalence calls in new studies. The
    0.003 tie band stays only to reproduce the probe.
- **D4.** Which opponent is of record for the thesis and for E44's primary?
  - **Recommended:** a win counts only where the model beats both the untuned bar of 5 (what the
    store logs) and the tuned bar (E42). The primary tally is against the tuned bar, the stricter
    and fairer one.
- **D16.** Should the metric panel be computed offline (stored files, ledgers, recompute) with
  `score_cells` left untouched for now, and pre-registered as secondary with claims only where it
  agrees?
  - **Recommended:** yes (ADR 0025 B).

**New baselines and the budget**

- **D5.** Do condexp, SoftImpute and ICE stay outside the bar, read from offline CSVs, until E41
  shows whether they matter?
  - **Recommended:** yes (ADR 0021 C), with no store backfill now.
  - **Superseded on 2026-10-16 by the user's design** (ADR 0021 D, §2b): two tiers, all logged.
    The bar of 5 is the weak tier; missForest and the new arms form the strong tier, with its own
    best and gap. E41 measures first, then the tiers are coded and backfilled.
- **D6.** Which ICE design is the thesis row?
  - **Recommended:** the categorical-correct one: one-hot features without the target's own block,
    multinomial LR, the sequential transform, the `still_converging` round count and a per-round
    clip. It is named `ice`, not MICE.
- **D7.** Which SoftImpute rank rule, fixed before scoring?
  - **Recommended:** a 90% energy share with no p//2 cap (k ≤ p−1), the PPCA posterior and the
    final clip. condexp keeps the review's frozen rules, plus the clip.
- **D8.** How do baselines select their setting?
  - **Recommended:** per fold on the validation rows' induced gaps, the same signal as K, with the
    masked cells as a sensitivity reading. Fit on the training fold with no refit, and log the
    chosen settings. The grids are those of E42. ADR 0022 B.

**Checkpoint and calibration**

- **D9.** Do you authorize the code for `DECODE_CHECKPOINT=masked` and the H and M readouts under
  the default? It is a flag plus a new tag value; K stays the default.
  - **Recommended:** yes, then E43 before E44.
- **D10.** What happens to the calibration under the deployable arm?
  - **Recommended:** KC is reported only as an oracle readout. T20 stays parked until E43, E44's KC
    readout and the panel are in. No calibration fitted on masked cells for now.

**Confirmation**

- **D11.** Which seeds for the confirmation?
  - **Recommended:** 707/808/909, reserved for E44 alone. E43 uses 111/222/333.
- **D12.** Do pendigits and letter at 40–80nan inherit ADR 0018's bigger encoder?
  - **Recommended:** no. ADR 0018 decided per variant, so the frozen configuration is what the
    checkout reads. An ALL arm on those 6 variants would add about 13 GPU-h and is not needed.
- **D13.** Are E44's splits and table set the same as the development set's?
  - **Recommended:** keep `StratifiedKFold`, with the twin/no-twin split as a readout (ADR 0026 A).
    No new OpenML tables in this window; add them later as a second confirmation if the thesis
    needs new tables.
- **D14.** What is E44's reading?
  - **Recommended:** supported if better on ≥8 of 15 and worse on ≤2; refuted if worse on ≥5 or
    better on ≤4; mixed otherwise. The 80nan subset is also reported apart.
- **D15.** Do E44's cells add `--score_calibrated`, with T20's 40 vs 60 threshold pre-registered as
  a secondary check?
  - **Recommended:** yes. It costs nothing, and those 15 variants are T20's out-of-sample test.
- **D17.** How are context-free rows handled?
  - **Recommended:** keep them in the primary score, and report their share and the score without
    them.

**MNAR, operations and later items**

- **D18.** Does E45 write the store, and what is the headline?
  - **Recommended:** yes, tagged. The headline is H, or M after E43; never K. Fix c in the ticket.
- **D19.** Path A (a minimal E44 on Saturday night, with no M readout) or path B (E43 on Saturday
  night, then the full E44 on Sunday night)? See section 5.
  - **Recommended:** B. One night later, and the one confirmation read carries the deployable
    readout.
- **D20.** Do you authorize a `.gitattributes` with `*.sh text eol=lf`?
  - **Recommended:** yes. It is one line and prevents broken queues after a Windows push.
- **D21.** For week 2: a separate venv for TabPFN v2, vendoring MaCoDE, and `--cv_folds 10` for
  classification step zero?
  - **Recommended:** decide after E44. If yes: MaCoDE first, then TabPFN, and `--cv_folds 10`.

## 8. What this plan does not do

- It does not touch T12 (a lower `LR_PRE`), T08, T09 or T10. T12 fits a free night in week 1.
- It does not adopt the calibration (T20 stays parked).
- It does not fix the review's other code findings beyond the ones listed above. Those stay in the
  review digest and in [BACKLOG.md](../BACKLOG.md).
- Its cost estimates are estimates. Every study times one cell before it is launched.
