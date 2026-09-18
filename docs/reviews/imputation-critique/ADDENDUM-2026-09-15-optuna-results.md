# Addendum — the Optuna results, 2026-09-15

_Revisions to `CONSOLIDATED.md` (2026-09-11) in light of the experiments run 2026-09-12
to 2026-09-13. This file supersedes specific findings; it does not replace the report,
and the thirteen critique and verdict files are left as the record of what was known on
2026-09-11._

Evidence base: `mlflow.db` (read-only) and the user's own experiment notes in
`docs/tickets/imputation-optuna-reduced/issues/05-launcher-and-studies.md` and
`06-comparison-and-docs.md`, written as the runs completed.

Every number below was re-derived by three independent adversarial verifiers working
from their own queries; §7 records what they corrected. All timestamps are local
(UTC-3), as MLflow's UI shows them.

---

## 1. What ran

295 new runs, 2026-09-12 00:19 to 2026-09-13 03:18, all `FINISHED`:

| kind | n | what it is |
|---|---|---|
| `optuna_study` | 6 | the **plateau** replication of the six reduced studies, `-NoPromote` |
| `optuna_trial` | 240 | 40 trials per study |
| `parent` | 15 | the tail of the cosine comparison (3) and the whole plateau comparison (12) |
| `best_fold`/`worst_fold` | 34 | diagnostic children |

The window itself is clean, but **the evening before it is not**: two `FAILED`
`kr-vs-kp_40nan` cosine comparison parents sit at 01:39 and 01:55 (commit `ac57272`).
Those are D-3's predicted crash actually happening — see §4 — and they were rerun after
the fix.

**On the codebase.** The five commits the new runs are tagged with are four
`docs(tickets)` commits plus `39b4ba9` (`chore: refresh graphify knowledge graph`, all
21,475 lines under `graphify-out/`); none touches code, all are ancestors of `HEAD`, and
`git diff 39b4ba9297 ecdb162 -- src/ opt.py main.py train.py` is empty. **That covers the
plateau window only.** The six cosine studies ran earlier, at `89bc904` / `fc34e18` /
`51f64ff`, and six of the twelve cosine comparison parents at `51f64ff` / `ac57272` — all
before `1cb0864 fix(decoding)`, which changed `_score_induced_missing`, the function that
produces the §2 target metric. So this is **not** a single frozen codebase, and an earlier
draft of this addendum wrongly said it was.

The cross-schedule comparison survives anyway, for a reason that has to be stated rather
than assumed: the pre-fix defect raised `KeyError` rather than returning a wrong number,
so any run that reached `FINISHED` provably never hit it. The only runs that did hit it
are the two `FAILED` parents above, which were rerun post-fix. No `FINISHED` number in
§2 or §3 is affected.

Comparison inventory: **24 five-fold runs** — 12 cosine and 12 plateau, tuned and
defaults for each of the six pairs.

---

## 2. The search objective, measured properly

The review characterised the search with one number: `rho = +0.124` against a +/-0.31
noise floor, from `credit-g_20nan` under cosine. **That study is the weakest of the twelve
now available.** Recomputed across every study in the store — Spearman rho (average ranks)
between each trial's `optuna/objective_value` and its `test/impute/induced/impute_score`,
the population ADR 0005 decision 6 declares the headline — with the promoted winner's rank
on that metric:

| study | schedule | distinct objectives /40 | tied at best | rho | winner rank |
|---|---|---|---|---|---|
| credit-g_20nan | cosine | 40 | 1 | **0.124** | 31/40 |
| credit-g_20nan | plateau | 40 | 1 | 0.335 | 15/40 |
| credit-g_40nan | cosine | 40 | 1 | 0.568 | 21/40 |
| credit-g_40nan | plateau | 40 | 1 | 0.653 | 18/40 |
| kr-vs-kp_20nan | cosine | **20** | **2** | 0.340 | **1/40** |
| kr-vs-kp_20nan | plateau | **18** | 1 | 0.441 | 12/40 |
| kr-vs-kp_40nan | cosine | **20** | **2** | 0.372 | **33/40** |
| kr-vs-kp_40nan | plateau | **20** | 1 | 0.169 | 21/40 |
| spambase_20nan | cosine | 40 | 1 | 0.659 | 3/40 |
| spambase_20nan | plateau | 40 | 1 | 0.745 | 23/40 |
| spambase_40nan | cosine | 40 | 1 | 0.497 | 15/40 |
| spambase_40nan | plateau | 40 | 1 | 0.640 | 2/40 |

Fisher-z mean rho: **cosine 0.442, plateau 0.524**. Single-study 95% noise floor at
n = 40 is 0.314; **ten of twelve** studies clear it. Mean winner rank: **cosine 17.3,
plateau 15.2**, against 20.5 by chance.

The two bold winner ranks are the two tie-broken cosine studies, and they were verified
by matching each promoted JSON in `datasets/hiperparams/kr-vs-kp/` field-for-field to the
trial that produced it. The tie-break selected the **best of forty** on `kr-vs-kp_20nan`
and the **thirty-third of forty** on `kr-vs-kp_40nan`. That is a far sharper illustration
of arbitrary selection than two mid-table results would have been.

Four conclusions.

**(a) The objective is not noise.** Ten of twelve studies clear the single-study floor and
the schedule means are 0.44 and 0.52. The review's "the search cannot resolve the
configurations it is ranking" generalised from the one cell where it was true. (Under a
Bonferroni correction for twelve comparisons the floor rises to 0.44 and only six of
twelve clear it, so the per-study count should not be leaned on; the Fisher-z means carry
the claim.)

**(b) Selection is near chance, but not for the reason first proposed.** An earlier draft
argued that "rho ~ 0.5 over 40 trials is nowhere near enough" to make `argmin` of the
proxy land near `argmin` of the target. That is wrong, and the data refutes it: under a
bivariate-normal model at rho = 0.442 / 0.524 with n = 40, the expected winner rank is
**10.2 / 8.4** — roughly twice as good as the 17.3 / 15.2 observed. The objective is
underperforming even its own correlation.

The actual mechanism is that **the correlation lives entirely in the bad tail**. Restrict
Spearman to the twenty best-by-objective trials — the half from which a winner is actually
drawn — and it averages **-0.029 (cosine)** and **+0.076 (plateau)**, against a
truncation-corrected expectation of 0.254 / 0.314. The proxy reliably separates bad
configurations from the rest and is uninformative among the good ones. It is an effective
filter and a useless ranker, and promotion uses it as a ranker.

**(c) Do not pool across studies.** Pooling all 240 plateau trials gives rho = 0.801,
which looks decisive and is an artifact: `kr-vs-kp` sits at 0.61-0.81 on the induced score
while `credit-g` and `spambase` sit at 0.88-1.04, non-overlapping. Rank-normalising within
each study and then pooling collapses it to **0.497**, the arithmetic mean of the
per-study values, as it must.

**(d) State the uncertainty.** Under chance the mean of six winner ranks has SD 4.71, so
17.3 and 15.2 are each within one standard deviation of 20.5. The claim these support is
"not distinguishable from chance selection", not "worse than chance".

---

## 3. The decisive evidence: tuning does not beat defaults

The comparison the review said did not exist has now run twice. From ticket 06 and
re-derived from the store, on `cv/test/impute/induced/impute_score/mean` with 95%
intervals, lower is better:

| pair | cosine tuned | cosine defaults | plateau tuned | plateau defaults |
|---|---|---|---|---|
| credit-g_20nan | 0.913 [0.868, 0.958] | 0.895 [0.868, 0.921] | 0.907 [0.859, 0.954] | 0.901 [0.868, 0.934] |
| credit-g_40nan | 0.968 [0.940, 0.995] | 0.962 [0.941, 0.983] | **0.928 [0.915, 0.942]** | 0.949 [0.920, 0.978] |
| kr-vs-kp_20nan | 0.603 [0.559, 0.646] | 0.615 [0.584, 0.646] | 0.619 [0.581, 0.656] | 0.612 [0.587, 0.637] |
| kr-vs-kp_40nan | 0.778 [0.734, 0.822] | 0.756 [0.721, 0.791] | 0.755 [0.712, 0.798] | 0.753 [0.700, 0.805] |
| spambase_20nan | 0.886 [0.859, 0.913] | 0.887 [0.859, 0.914] | 0.890 [0.870, 0.910] | 0.883 [0.861, 0.905] |
| spambase_40nan | 0.930 [0.909, 0.952] | 0.927 [0.897, 0.957] | 0.924 [0.892, 0.955] | 0.930 [0.907, 0.952] |

Across twelve tuned-versus-default comparisons **exactly one separates** at 95%
(credit-g_40nan under plateau, the tuned interval excluding the default mean, the masked
metric agreeing). One in twelve is approximately what chance produces — the user's own
reading in ticket 06, and the correct one. The caveat F-08-4 raised applies to this very
table: two overlapping single-arm Student-t intervals are not a test of a difference and
discard the pairing the shared folds give for free.

The sharpest single result in the effort: under plateau the studies reached a validation
objective **better by 0.023 to 0.061** on four of six pairs, and that gap left **no trace**
in the induced benchmark, where the two schedules land within 0.013 at the defaults and no
pair separates. A large consistent improvement in the quantity being optimised produced no
detectable improvement in the quantity that is reported. That is a more direct indictment
of the proxy than any rank correlation, and it supersedes the rho argument as the primary
evidence.

---

## 4. Finding-by-finding ledger

| finding | was | now | basis |
|---|---|---|---|
| **F-07-1** (critical) — the search cannot identify the configuration it promotes | SOUND/critical, argued from rho = +0.124 and two tie-broken winners | **Conclusion upheld, argument replaced.** The promoted configurations remain not evidence-backed — because tuning does not beat defaults in 11 of 12 comparisons, and because the proxy is uninformative among good configurations (§2b), not because it is noise. Severity stays **critical**. | §2, §3 |
| F-07-1's tie-break claim | both `kr-vs-kp` winners decided by tie-break | **Confirmed and sharpened.** Both cosine `kr-vs-kp` studies have 2 trials tied at the best objective; the rule selected rank 1/40 on one and 33/40 on the other. Does not recur under plateau. | §2 |
| F-07-1's resolution claim | the objective collapses to a lattice on all-categorical tables | **Confirmed, schedule-independent.** `kr-vs-kp` yields 18-20 distinct values from 40 trials under both schedules; `credit-g` and `spambase` yield 40/40. | §2 |
| **D-3** (CONFIRMED/medium) — two shipped variants crash before scoring an induced cell | predicted | **Occurred.** Two `FAILED` `kr-vs-kp_40nan` cosine comparison parents (2026-09-12 01:39, 01:55, commit `ac57272`); fixed by `1cb0864` and rerun. The prediction was correct and is now closed. | §1 |
| **F-07-7** (UNSOUND, corrected form medium) — a held-out number exists on the winning trial's own run and nothing reads it | corrected form retained | **Confirmed by measurement.** §2's rank column is exactly the number the finding said existed and went unread; it would have shown the promoted winner ranking 31st, 21st and 33rd of 40 on three pairs. | §2 |
| **F-08-5** — the search optimises the masked proxy while the comparison is judged on the induced population | SOUND | **Confirmed directly.** The winner is not the best trial on the induced metric in 11 of 12 studies. | §2 |
| F-07-5 — fANOVA drops `HEAD_DIM` | CONFIRMED/low (not SOUND) | **Unchanged, and carries no new evidence.** The reduced profile never samples `HEAD_DIM`, so these twelve studies could not have exposed it either way. Four studies log four importance keys; the two `credit-g` studies log five (`LAMBDA_NUM` is conditional). | study keys |
| F-08-2 — the launcher hardcodes `--search_space reduced` | **UNSOUND, residue low** (not SOUND/high) | **Residue unchanged.** All twelve studies ran `reduced` and no `full` run exists anywhere in the store. The residue is that fANOVA can only expose over-inclusion; "the reduction is untested" is the critique's framing, which its verifier rejected. | study params |
| F-08-4 — the comparison cannot support a generalisation claim | SOUND | **Confirmed empirically.** One separation in twelve at 95%, single seed, unpaired intervals over correlated folds. §3's own table inherits the flaw. | §3 |
| F-08-8 — neither launcher isolates `--metrics_dir` | CONFIRMED | **Unchanged and consequential.** This is why five classification metrics CSVs were overwritten and had to be rebuilt from MLflow on 2026-09-13. | — |
| F-07-3 — no way to run the defaults arm once a configuration is promoted | already recorded as wrong (the critique never opened `imputation_studies.ps1`) | **Closed.** Twelve defaults-arm runs exist. | §1 |
| F-07-2, F-07-6, F-08-3 — the objective is scored on the cells that chose the checkpoint; search rows reappear as test rows | SOUND | **Unchanged.** Structural; no run addresses them. | — |
| Dimension 12 — "none of the twelve decision-6 comparison runs exists yet" | true when written | **Superseded by later runs, not an error.** 24 comparison runs now exist. The consequence matters: F-12-2's "no published number currently rests on that protocol" no longer holds, so its severity basis needs revisiting. | §1 |
| `CONSOLIDATED.md` §4 heading — "the twelve comparison runs (none has been run)" | true when written | **Parenthetical superseded.** §4's six "cannot support" items all stand; item 1 is the caveat §3's own intervals carry, and item 5 (F-08-9) was avoided by staging the plateau winners rather than promoting them. | §3 |

---

## 5. Carried forward and new

**C-1 (methodology, medium) — the reduced search's rate finding attenuates under plateau,
and the promoted rate is schedule-dependent.** Recorded first in ticket 05; verified and
corrected here. The cosine batch's headline had two halves, and **only one moves**. The
half that replicates: Spearman(`LR_DECODE`, objective) is negative in **12 of 12** studies
(cosine -0.273 to -0.524, plateau -0.042 to -0.467), the worst-eight median rate is below
the best-eight median in 5 of 6 plateau studies, and `LR_DECODE` remains the
highest-mean-importance knob under both schedules (0.483 cosine, 0.421 plateau). Low rates
really are worse, under both. The half that attenuates: `LR_DECODE` is the first knob on
3 of 6 pairs under plateau against 5 of 6 under cosine — two pairs lose the top spot
(`kr-vs-kp_20nan`, `credit-g_40nan`) and `spambase_20nan` never held it. The promoted rate
swings 5-25x between schedules and two plateau winners fall at or below the default 1e-3.
The `kr-vs-kp_40nan` "reversal" reported in ticket 05 is **not** load-bearing: its 40
trials take only 20 distinct objective values, both octile boundaries fall inside ties, and
the best-8/worst-8 comparison flips with the tie-break convention (4.50e-3 vs 4.67e-3 under
a trial-number tie-break, not 4.5e-3 vs 5.2e-3). Likewise `kr-vs-kp_20nan`'s winner is in
the bottom quartile of the log range, not "the very bottom" — 16 of 40 trials sampled a
lower rate. The safe statement is that a rate finding from a single schedule must not be
reported as a property of the task; "does not replicate" is too strong.

**N-1 (methodology, medium) — this effort has no measured noise floor, and the experiment
needed is a second seed, not a repeat.** None of the 24 comparison runs replicates another:
the `spambase` pairs that ran on both 09-12 and 09-13 differ by schedule, not by noise.
The differences being interpreted (**0.001 to 0.022** between arms) are smaller than the
interval half-widths (**0.014 to 0.053**, eight of the twenty-four above 0.033).

The prescription an earlier draft gave — repeat a configuration under identical conditions
— would return a number already known: the two 2026-09-09 `vehicle_00nan` runs share
configuration and seed and are **bit-identical on all 68 logged metrics**, despite being
launched from different commits. The pipeline is deterministic at a fixed seed, so a
same-seed repeat measures nothing. The missing measurement is a **second seed** at a fixed
configuration. The nearest bounds available are the nine replicated classification
configurations (repeat spread 0.0001-0.0047 in CV-mean accuracy, i.e. determinism) and the
seed 42/420 classification pairs (0.006-0.025 in CV-mean macro F1) — the latter is the
relevant order of magnitude, and it is comparable to every gap in §3.

---

## 6. What is still unmeasured

- **A second seed for any imputation run.** Four seed-420 imputation runs were attempted on
  2026-09-11 (`electricity_20nan`/`_40nan`, both schedules) and **all four failed**; every
  `FINISHED` imputation run in the store is seed 42. This gates every claim in §3.
- **A `full`-profile study on any pair.** `search_space` is `reduced` on all 492 Optuna runs
  and the string `full` appears nowhere in tags or params.
- **An objective scored on the induced population, or over more folds.** Ticket 05 names
  both. §2(b) sharpens the target: the proxy fails specifically among good configurations,
  so a replacement must discriminate *within* the top half, which more folds would do and a
  different population might not.
- **The structural findings** (F-07-2, F-07-6, F-08-3) remain untouched, because no amount
  of running the existing protocol can address selection bias in that protocol.

---

## 7. Verification record

Three adversarial verifiers (statistics, ledger, new-findings) re-derived this addendum
from their own queries against a read-only `mlflow.db`. They corrected, and this version
incorporates: a false "single frozen codebase" claim in §1; two wrong winner-rank cells in
§2 (`kr-vs-kp` cosine, reported as 20/40 and 21/40, actually 1/40 and 33/40); a
"nine of twelve" miscount (ten); a mechanism in §2(b) refuted by the addendum's own data;
`39b4ba9` mislabelled as a docs commit; "no failures" in §1 concealing D-3's actual
occurrence; two wrong "was" verdicts in §4 (F-07-5, F-08-2); five findings missing from the
ledger (D-3, F-07-7, F-08-5, F-07-3, F-08-8); an over-strong N-1 with three wrong numbers;
and an N-2 whose prescription would have measured determinism rather than noise.

What survived attack unchanged: the census and run breakdown, all twelve distinct-objective
and tie counts, all twelve rho values, the Fisher-z means, the noise floor, the pooling
diagnosis (independently confirmed by within-study rank-normalisation collapsing 0.801 to
0.497), §3's comparison table, and §6's full-profile claim.

---

# §8 — Experiments run 2026-09-17

_Nine runs commissioned to settle §6's open questions: four five-fold comparison runs and
five Optuna studies. `--metrics_dir` and `--output_dir` isolated, no `--promote_best`, so
no committed configuration was touched. Code at `b6ca510` for the 09-17 00:25-05:21 batch
and `8b84c24` (post-ADR-0006, mirroring on) for everything from 09-17 23:17; both descend
from the review commit `ecdb162`, and §8.1 shows the induced metric is bit-identical at each._

> **Mirror-run caveat (ADR 0006).** Every source run now has a copy under
> `TRIDENT/mirror/<task>` carrying `is_mirror = true`. The figures in this addendum were
> computed **before** that backfill landed and match a source-only recount exactly (median
> cost 0.0357, identical winner-rank list), so nothing here is double-counted. But the same
> queries run today return **31 reduced studies instead of 16**. Any re-derivation of these
> numbers must filter `tags.is_mirror = 'false'`; the mirrors are exact copies (15 of 15
> pairs share `best_objective_value` to 1e-12; the empty RUNNING study has no mirror and no
> best value), so they inflate every count by 2x and leave
> every median unchanged — which makes the contamination easy to miss.

## 8.1 The typing refactor preserved imputation behaviour exactly

`df873f8 feat(typing)` changed eleven files between the 09-11 runs and tonight, including
`decoding.py` and `runner.py`. Re-running seed 42 at HEAD reproduces the stored values to
the last digit:

| pair | stored 09-11 | tonight (`b6ca510` and `8b84c24`) | delta |
|---|---|---|---|
| credit-g_20nan | 0.912797 | 0.912797 | **0.000000** |
| credit-g_40nan | 0.967689 | 0.967689 | **0.000000** |

Every run in this addendum is therefore comparable with every run in §2 and §3.

## 8.2 The noise floor N-1 asked for

The tuned cosine configuration, five folds, at seeds 7, 13, 42, 99 and 420 — the seed
changes the CV split and every mask, so this is the noise between two single-seed runs
of the kind §3 compares:

| pair | five induced means | SD | range | tuned-vs-defaults gap (seed 42) | gap in SDs |
|---|---|---|---|---|---|
| credit-g_20nan | 0.9090 0.8969 0.9128 0.9100 0.9125 | **0.0066** | 0.0159 | 0.0181 (defaults better) | **2.8** |
| credit-g_40nan | 0.9710 0.9882 0.9677 0.9505 0.9772 | **0.0138** | 0.0377 | 0.0056 (defaults better) | **0.4** |

So `credit-g_40nan`'s arm difference is inside seed noise and is not interpretable in
either direction, while `credit-g_20nan`'s sits at 2.8 SD — suggestive, and pointing
**against** the promoted configuration — but 2.8 against an SD carrying 4 degrees of freedom
does not clear the t(4) reference of 2.78, and it is one arm's SD under a two-arm difference
(§8.11 does this properly).

The general lesson is the scale: **a single five-fold run carries an envelope of about
+/-0.013 to +/-0.027 at 1.96 SD** on this metric — and since each SD has only 4 degrees of
freedom, the honest t(4) envelope is wider, +/-0.018 to +/-0.038. Every tuned-versus-default
gap measured across the sixteen five-fold comparisons now in the store lies between 0.0005
and 0.0283, i.e. inside those envelopes.

> **Correction.** An earlier draft of this section, written from seeds 42 and 420 alone,
> reported spreads of 0.000270 and 0.009483, called the first gap "67x the noise", and
> concluded that seed sensitivity differed 35x between pairs so that no global noise floor
> could exist. All three claims were artifacts of n = 2: seeds 42 and 420 happened to land
> within 0.0003 of each other on `credit-g_20nan`. With five seeds the SDs differ 2x, which
> is unremarkable, and the "67x" becomes 2.8 SD. A difference of two runs is a scale, not a
> standard deviation, and should not have been used as one.

## 8.3 Mis-selection on the study's own split — and, in §8.14, how little of it survives five-fold

For every study in the store, the induced score of the trial the objective picked, against
the best induced score among that study's own 40 trials:

| profile | studies | median cost | range |
|---|---|---|---|
| `reduced`, single best trial | 13 | **0.0357** | 0.0006 - 0.0535 |
| `full` | 3 | 0.0142 | 0.0000 - 0.0264 |

Two `reduced` studies are excluded because their best objective is **tied** and the cost
therefore depends on which tied trial the tie-break happens to take:

| study | tied | cost range |
|---|---|---|
| kr-vs-kp_20nan seed 42 cosine | 2 trials | **0.0000 - 0.0552** |
| kr-vs-kp_40nan seed 42 cosine | 2 trials | 0.0471 - 0.0749 |

That is the tie-break arbitrariness of §2 given a price: on `kr-vs-kp_20nan`, choosing
between two configurations the objective scores **identically** is worth 0.0552 — more than
any tuned-versus-default difference this effort has argued about. (`opt.py` promoted the
lucky one there and the unlucky one on `kr-vs-kp_40nan`.)

**Every tuned-versus-default gap in the sixteen five-fold comparisons is 0.0005 to 0.0283.**
The median mis-selection cost of 0.0357 exceeds the largest of them, and in nine of the
thirteen untied reduced studies a trial at least 0.025 better than the promoted one exists.

Across all fifteen `reduced` studies the **promoted** trial's rank (min-rank on ties) is
`[1, 2, 2, 3, 10, 15, 15, 18, 18, 21, 21, 23, 24, 31, 33]` — median **18/40**, mean 15.8,
against 20.5 by chance. Occasionally excellent, usually mid-pack, and not distinguishable
from chance as a procedure. (An earlier draft listed the *non-promoted* tied trial for the
two tie-broken cosine studies — 20 and 21 in place of 1 and 33 — contradicting §2; the
median is 18 either way.)

This reframes F-07-1, with one honest limit. Trials are scored on the study's single
predefined split (`build_folds` with `cv_folds=None`), while the comparison runs use
five-fold KFold; so the 0.036 is a **min-of-40 on one split** and carries the selection bias
of a minimum. It is an upper bound on recoverable regret, not a measured gain, until one
best-induced trial is re-run five-fold — the one control neither night ran. **That control has now been run (§8.14):** on `credit-g` about a fifth of the single-split gap
survives five-fold — a realised gain of roughly 0.016 on `credit-g_20nan` and an amount inside
noise on `credit-g_40nan`, not 0.036. The "find but do not promote" reading therefore shrinks
to its measured size. On `credit-g_20nan` the best-induced trial sits at the defaults (+0.0001
over two seeds) while the promoted file is measurably worse (+0.013, p ≈ 0.01, §8.11); on
`credit-g_40nan` the two configurations are not distinguishable from each other and both
trend worse than the defaults.

§8.8 tests the obvious remedy and rules it out.

And the reduced space does not look flat: within-study induced spread across 40 trials is
0.060 to 0.187. That range is not directly comparable to §8.2's SD (a range of 40 draws is
already ~4 SD wide even under pure noise, and single-split trial scores are noisier than
five-fold means), so no multiplier is claimed — but the median cost of 0.0357 is 2.6 to 5.4
times the five-fold seed SD, which is the like-for-like scale. Configuration choice matters
here; the selector is what does not work.

## 8.4 `full` and `reduced` are disjoint profiles, not nested ones

`opt.py`'s own docstring says "``full`` samples every knob the task uses" and "a key the
profile does not return is *held*: the trial runs it at the task default". Both halves are
true and together they are a trap:

| knob | `reduced` holds at | `full` samples from | source |
|---|---|---|---|
| `EPOCHS_PRE` | **300** | **20-60** | `opt.py:124`, `types.py:97` |
| `EPOCHS_DECODE` | **150** | **20-60** | `opt.py:134`, `types.py:112` |
| `EPOCH_FINE` (classification) | **150** | **20-60** | `opt.py:146`, `types.py:98` |

The full profile's entire range lies 5x to 15x (pre-training) and 2.5x to 7.5x (decode)
**below** the value the reduced profile holds fixed, so a `full` study can never evaluate a configuration a `reduced` study
evaluates. Consequences:

- **"What does the reduction cost?" cannot be answered by running both profiles.**
  Comparing their best objectives measures the epoch budget, not the search space. An
  earlier reading of tonight's data as "reduced wins" (full 0.818 against reduced 0.776 on
  `credit-g_20nan`) is confounded and withdrawn.
- The two profiles silently buy different compute budgets: a `full` trial ran in ~21 s
  against ~65 s for `reduced`, purely because it trains far shorter. The 14-minute full
  study against the 43-minute reduced study is the symptom that exposed this.
- Classification is affected identically through `EPOCH_FINE`, which compounds the first
  review's finding that the classification `full` space changed with no legacy profile.

A third `full` study, on `kr-vs-kp_20nan` (seed 42, cosine, 48 min), is a caution rather
than support for F-13-1. Its winner trained **20 pre-training epochs** (also 30 decode
epochs, batch 64, and a different model on every sampled knob) and beat the `reduced`
winner on the **validation objective**, 0.6484 against 0.6703 on the same split. On the
**induced test score** — the population F-13 argued on — the same trial is **0.074 worse**
(0.6787 against 0.6043, on a pair whose seed SD is 0.004), and none of the 40 full trials
reaches the reduced study's best induced score. Within 20-60 the objective shows no
detectable epoch signal (Spearman(`EPOCHS_PRE`, objective) = +0.013), which says nothing
about 300. The objective "gain" from cheap pre-training does not transfer to the task
metric — exactly the failure §8.8 documents for the objective generally.

This was **not** new: the 08 verdict (`08-experimentation-design.verdict.md`, L108-116 and
L320-324) and the 07 verdict (L216) stated the disjoint epoch ranges from the code and
prescribed pinning them before running a matched `full` arm. Tonight's studies are the first
execution of the confound they predicted; an earlier draft of this section wrongly claimed
no dimension had found it.

## 8.5 Rho does not predict selection quality

| study | profile | seed | rho | winner rank |
|---|---|---|---|---|
| credit-g_20nan | reduced | 42 | 0.124 | 31/40 |
| credit-g_20nan | reduced | 420 | 0.610 | 18/40 |
| credit-g_20nan | **full** | 42 | 0.747 | **4/40** |
| credit-g_40nan | reduced | 42 | 0.568 | 21/40 |
| credit-g_40nan | reduced | 420 | 0.515 | 24/40 |
| credit-g_40nan | **full** | 42 | 0.516 | **1/40** |
| kr-vs-kp_20nan | reduced | 42 | 0.340 | 1/40 (promoted) / 20/40 (tie) |
| kr-vs-kp_20nan | reduced | 420 | 0.540 | 2/40 |
| kr-vs-kp_20nan | **full** | 42 | **0.724** | 11/40 |

The `kr-vs-kp_20nan` full study is the sharpest single row: the **highest rho of the
four studies on that pair** and a mid-table pick (11/40, cost 0.026), while the reduced
seed-420 study with rho 0.540 picked 2/40. So "the full profile selects near-optimally"
(the first draft's reading of `credit-g`) does not generalise — it is two of three — and
what does generalise is the decoupling itself. `credit-g_40nan` is the clean case: rho 0.515 against 0.516 — indistinguishable — and winner
rank 24/40 against **1/40**. Correlation and selection quality are decoupled, exactly as
§2(b)'s bad-tail mechanism predicts. This retires rho as the statistic to report: **the
winner's rank is the quantity that corresponds to the decision being made**, and an earlier
draft's proposal of Fisher-z means as "the honest statistic" was itself the wrong summary.

Across all 18 completed studies, Spearman(rho, winner rank) is **−0.23 (p = 0.36)**: the
expected sign, but far too weak to use rho as a predictor of the pick; each study is one draw,
and at rho ~ 0.5 with n = 40 a single draw spans ranks ~1-25 at the 90% level. Rho itself
varies across seeds and schedules on the same pair — `credit-g_20nan` 0.124 / 0.335 / 0.610,
`credit-g_40nan` 0.515 / 0.568 / 0.653 (n = 3 each) — which, by §8.2's own standard, is a
range and not a stability claim in either direction.

## 8.6 Corrections to earlier sections

- §3's "exactly one separates at 95%" is now **replicated and does not hold** (§8.10): on
  seeds 420 and 7 the same pair gives −0.0097 and +0.0005, neither separating. §3's reading
  ("about what chance produces") was the right one.
- §6's "no `full`-profile study exists anywhere in the store" is **superseded**: two now do.
- §6's "a second seed for any imputation run" is **closed** for `credit-g` by §8.2.
- N-1's prescription (a second seed) was right; §8.2 delivers five, and the first draft's
  reading of two of them — a 35x pair-specific floor — is withdrawn as an n = 2 artifact.

## 8.8 Re-selection does not recover the loss — the metric is not the lever

The natural fix for §8.3 is to select on a different validation statistic. Every trial in
the store carries the full validation set (`impute_score`, `rmse_num_z`, `mae_num_z`,
`acc_cat`, `macro_f1_cat`), so this is testable at zero GPU cost by re-ranking trials that
already ran. Any candidate must use **validation** signal only; selecting on the induced
test score would be leakage, since that is the target.

Paired over the six studies where every candidate is computable:

| selector | median cost | mean | studies won |
|---|---|---|---|
| current (`impute_score`) | 0.0386 | 0.0394 | 1 |
| `rmse_num_z` alone | 0.0369 | 0.0384 | 3 |
| `macro_f1_cat` alone | **0.0342** | **0.0322** | 3 |
| rank-average(`rmse`, `-f1`) | 0.0363 | 0.0398 | 1 |
| `mae_num_z` alone | 0.0310 | 0.0291 | — |

Medians lie within 0.008 and means within 0.010 of each other, against a loss of ~0.036 to
recover; `mae_num_z` is the best single candidate on these six and still leaves ~0.03 on
the table. On the nine single-kind studies (where only one of `rmse`/`f1` exists) the
distinct candidate wins on some datasets and loses on others. **No validation-only
statistic wins consistently across datasets.**

This is a negative result with a precise scope: every candidate is a function of the same
masked validation cells on the same split, so the sweep tests **statistic choice only**. It
cannot distinguish "the validation split does not predict the induced population" from "the
target — a min-of-40 on one fixed split — is a selection-biased minimum no selector could
reach" (§8.3's limit). The control has now been run (§8.14), with one candidate: the trial that is best on the
study's single split. Re-run five-fold it recovers about 0.016 on `credit-g_20nan` and an
amount indistinguishable from zero on `credit-g_40nan` (two seeds each), and it is not
distinguishable from the defaults. That is a **realised gain of one trial, not a ceiling**:
the single-split ranking is itself noisy (its 0.036-0.054 advantage collapsed to 0.002-0.018),
so among the 40 trials another, near-tied on the single split, could do better five-fold —
on `credit-g_20nan` the five best single-split induced scores lie within 0.008 of each other.
A true bound over selectors would need all 40 trials scored five-fold, which has not been run.
The prescription (score the induced population on validation, or widen the folds, as ticket
05 proposed) therefore stands as the hypothesis it was, with this much added: on `credit-g`
the one candidate tried lands at the defaults. What *is*
established: an earlier draft of §8.3 recommended "stop selecting on
`validation/impute/masked/impute_score` alone"; that recommendation is **withdrawn** —
swapping the statistic does nothing.

## 8.9 D-1 re-measured at HEAD: still present, to the cell

`CONSOLIDATED.md` D-1 (`decoding.py:262` hands `--score_null_path` a frame-ordered mask
where `embedder.py:189` needs an embedder-ordered one) was CONFIRMED/high on 2026-09-11.
`decoding.py` has changed since (`a395a58`, `df873f8`), so it was re-run: `credit-g_20nan`,
cosine, seed 42, five folds, `--score_null_path`, at `8b84c24` (run `936754ad`).

The ledger records `(row, column)` for every scored cell, so the check is the overlap
between the cells the correct `induced` population scores and the cells the
`induced_null_token` diagnostic scores — the two populations have identical *counts*
(789/794/807/785/825 per fold), which is exactly why the existing test cannot see this
(F-09-1):

| fold | induced | null-token | overlap | scored but never hidden |
|---|---|---|---|---|
| 1 | 789 | 789 | 189 | 76.0% |
| 2 | 794 | 794 | 184 | 76.8% |
| 3 | 807 | 807 | 206 | 74.5% |
| 4 | 785 | 785 | 178 | 77.3% |
| 5 | 825 | 825 | 187 | 77.3% |

**4000 cells scored, 944 truly induced: 76.4% of the diagnostic's cells were observed
values the model was never asked to reconstruct.** Dimension 04 reported 4000 / 944 on
09-11; the figure reproduces exactly. D-1 stands unchanged at HEAD, and ADR 0004 decision
11's recorded numbers for the `[NULL]`-path gate remain measurements of the wrong cells.

## 8.10 The one separation in twelve, replicated: it does not hold

§3 found exactly one tuned-versus-default comparison separating at 95%: `credit-g_40nan`
under plateau, seed 42. Ticket 06 said it "should be replicated on another seed before
being believed". It was replicated on two, with the plateau winner staged into the
task-keyed file for the tuned arm and the file moved aside for the defaults arm (both
reversed by `git checkout` after each run):

| seed | tuned | defaults | tuned − defaults | tuned CI excludes defaults mean |
|---|---|---|---|---|
| 42 | 0.9285 | 0.9491 | **−0.0207** | yes — the original separation |
| 420 | 0.9450 | 0.9547 | −0.0097 | no |
| 7 | 0.9497 | 0.9492 | +0.0005 | no |

Mean difference **−0.0100**, SD of the differences 0.0106, paired t = −1.63 on 2 degrees
of freedom (|t| > 4.30 needed at p < 0.05); tuned better on 2 of 3 seeds; the mean
difference is 0.7 of a single run's SD on this pair (§8.2).

So the separation was, in the expected proportion, a favourable draw: the magnitude
halves at the second seed and vanishes at the third. What survives is a direction — tuned
ahead on two seeds of three, mean −0.010 — that is not distinguishable from zero. **The
effort's only positive tuning result is, on replication, "possibly a small effect, not
established."** Combined with §8.3 (the search leaves ~0.036 on the table by mis-ranking
its own trials) the picture is consistent: whatever the promoted configuration gains over
the defaults is smaller than what the selector loses.

## 8.11 The committed `credit-g_20nan` promoted file loses to the defaults; `credit-g_40nan` leans the same way

The cosine winners are the configurations actually committed in `datasets/hiperparams/`
(`fcaadfb`). §3 had them tuned-versus-defaults at seed 42 only. The defaults arm was
re-run at seed 420 (file moved aside, restored after) to pair with §8.1's tuned seed-420
runs:

The defaults arm was run at three seeds on `credit-g_20nan` (7/42/420: SD 0.0032) and four
on `credit-g_40nan` (7/13/42/420: SD 0.0070) — smaller than the tuned arm's on both pairs —
so the right yardstick for a *difference* of two runs is
σ_diff = sqrt(SD_tuned² + SD_defaults²) = 0.0073 and 0.0155:

| pair | seed | tuned (promoted file) | defaults | tuned − defaults | in σ_diff |
|---|---|---|---|---|---|
| credit-g_20nan | 42 | 0.9128 | 0.8947 | **+0.0181** | **+2.5** |
| credit-g_20nan | 420 | 0.9125 | 0.8992 | **+0.0133** | +1.8 |
| credit-g_40nan | 42 | 0.9677 | 0.9621 | +0.0056 | +0.4 |
| credit-g_40nan | 420 | 0.9772 | 0.9489 | +0.0283 | +1.8 |

Lower is better, so every row favours the **defaults** — and with the defaults arm at three
and four seeds the comparison no longer rests on individual rows:

| pair | tuned (n = 5) | defaults | difference | Welch t | df | p < 0.05? |
|---|---|---|---|---|---|---|
| credit-g_20nan | 0.9082 | 0.8956 (n = 3: 0.8930, 0.8947, 0.8992) | **+0.0126** | **3.64** | ~5.9 | yes (crit 2.45) |
| credit-g_40nan | 0.9709 | 0.9529 (n = 4: 0.9462, 0.9544, 0.9621, 0.9489) | **+0.0180** | **2.53** | ~6.1 | yes, just (crit 2.45, p ≈ 0.045) |

Per-seed gaps in σ_diff: `credit-g_20nan` +2.2 / +2.5 / +1.8 at seeds 7 / 42 / 420;
`credit-g_40nan` +1.6 / +2.2 / +0.4 / +1.8 at 7 / 13 / 42 / 420. Every one of the seven
shared-seed comparisons favours the defaults. `credit-g_40nan` went from p ≈ 0.06 at three
defaults seeds to p ≈ 0.045 at four — it clears the threshold, but this has to be
read with two disclosures. First, the third defaults seed was the pre-stated stopping point
("a fourth would settle it either way") and gave p ≈ 0.058; the fourth was added after that,
and only then did p cross 0.05. Second, the result is one-seed-fragile: dropping any single
seed from either arm leaves p > 0.05 in seven of nine variants. The paired-by-seed test —
which §3 said was the right instrument, since both arms share the CV split — is stronger
(`credit-g_20nan` mean +0.0158, t = 11.4 on 2 df, p ≈ 0.008; `credit-g_40nan` mean +0.0231,
t = 3.78 on 3 df, p ≈ 0.033) but inherits the same fragility for `credit-g_40nan`. Under plateau (§8.10) the same protocol leaned the other
way by a smaller, non-significant margin — but plateau winners were never promoted; the
cosine winners are what a `--task imputation` run on `credit-g` **loads today**.

For `credit-g_20nan` this upgrades F-07-1 from "the promoted configuration is not
evidence-backed" to **"the evidence points against it"**: a run that reads the promoted file
scores measurably worse than one that finds no file, at every seed tried (p ≈ 0.01 Welch,
≈ 0.008 paired). For `credit-g_40nan` the evidence label stays **hold**: consistent sign on
four of four shared seeds, p ≈ 0.044 Welch / 0.033 paired, fragile to any one seed, and
reached after extending past the pre-stated stopping seed. The immediate,
reversible action is to stop reading that file — `git rm` of
`datasets/hiperparams/credit-g/credit-g_20nan.imputation.json`, or a deliberate defaults run —
until a selector that survives §8.8 exists. Removing `credit-g_40nan.imputation.json` as well
is defensible, but on a different ground: not "shown worse" but **"no evidence supports the
file, and removal is reversible"** — every comparison leans against it and nothing in its
study's forty trials has been shown better than the defaults (§8.14). Mis-promotion is thus
not only a missed gain (§8.3's bound) but, on at least one pair, an active loss of ~0.013 to
~0.018.

Caveat carried honestly: two pairs of one dataset, seven shared-seed comparisons, Welch
tests on ~6 df each. The consistent sign across all seven and the two Welch tests carry
the conclusion; no single row does, and nothing here says anything about `kr-vs-kp` or
`spambase`, where the sign runs the other way on at least one pair (§8.12).

## 8.12 The sign of tuned-versus-defaults follows the winner's rank

`kr-vs-kp_20nan`, tuned cosine, five folds, seeds 7 / 42 / 420: 0.6110 / 0.6027 / 0.6085 —
**SD 0.0043 on 2 degrees of freedom**, range 0.0083. On this pair the promoted file *beats*
the defaults at seed 42 by −0.0119: 2.8 times that SD, but with 2 df the reference is
t(2) = 4.30 — the threshold §8.10 applies to itself — and the defaults arm exists at one
seed only, so the √2 bound gives 2.0. Suggestive, in the tuning direction, not established.

*2026-09-18, second defaults seed (420):* defaults 0.6146 / 0.6156 against tuned 0.6074
(n = 3, SD 0.0043); difference −0.0077. The Welch test gives t = −3.07 on ~2.2 df (p ≈ 0.08),
but that rests on a defaults SD of 0.0007 estimated on **one degree of freedom** — the same
n = 2 artifact §8.2 corrected — whose 95% interval runs from 0.0003 to 0.024. With any
plausible defaults SD the test is weaker: 0.0032 (the `credit-g_20nan` defaults) gives
p ≈ 0.11, the tuned arm's own 0.0043 gives p ≈ 0.17; paired on the two shared seeds,
p ≈ 0.16. The per-seed gaps are −0.0119 (42) and −0.0071 (420): same direction, smaller on
replication, as in §8.10 (§8.11's `credit-g` rows show no such shrinkage — `credit-g_40nan`
grew from +0.006 to +0.028). Standing, stated plainly: the tuning-wins pair is
sign-consistent and **not significant** (p ≈ 0.08-0.17); the defaults-win `credit-g_20nan` is
significant (p ≈ 0.01); `credit-g_40nan` is in between (p ≈ 0.04, fragile).

Put this beside what §2 established about which trial each promotion actually took — and,
rather than three pairs, all twelve seed-42 (rank, gap) pairs from §2 and §3:

| pair (cosine) | promoted trial's induced rank | tuned − defaults (seed 42) |
|---|---|---|
| kr-vs-kp_20nan | **1/40** (lucky tie-break) | −0.0119 (tuning wins) |
| spambase_20nan | 3/40 | −0.0010 |
| spambase_40nan | 15/40 | +0.0030 |
| credit-g_40nan | 21/40 | +0.0056 |
| credit-g_20nan | 31/40 | +0.0181 |
| kr-vs-kp_40nan | **33/40** (unlucky tie-break) | +0.0219 (defaults win) |

Under **cosine** the relationship is perfectly monotone: Spearman(rank, gap) = **+1.00** over
six pairs (permutation p = 0.003) — the pair whose pick landed at the bottom is the largest
defaults win. Under **plateau** it is not: rho = +0.31 (p = 0.56) over the six plateau pairs.
So the sign of tuned-versus-defaults tracks where the selector's pick landed — which §8.3
shows is a lottery with median rank 18/40 — on the schedule whose winners were actually
promoted, and the claim does not extend to "not a property of the schedule" (an earlier
draft said so from three pairs; the plateau pairs do not support it). Within that scope it
still ties §8.3, §8.8, §8.10 and §8.11 together: promotion delivered a gain where the
tie-break got lucky and a loss where the pick was near the bottom, and which datasets gain
was decided by chance.

## 8.13 Verification record for §8

Three adversarial verifiers (statistics; profiles and ranks; arguments) re-derived §8 from
their own mirror-excluded queries against a read-only `mlflow.db`, the five D-1 ledgers,
`opt.py` and `types.py`. Reproduced to the digit: §8.1's bit-identity at both commits, §8.2's
five-seed SDs and ranges, both cost tables and the tied ranges in §8.3, every §8.4 range,
every row of §8.5's table including the promoted-file-to-trial match, §8.8's six-study
table, all five §8.9 overlap rows, §8.10 in full (t = −1.63, df 2, two-sided p 0.24), and
§8.11/§8.12's raw values.

Corrected in this version: the preamble's "code at `ecdb162`" (runs are at `b6ca510` and
`8b84c24`); "16 of 16" mirror pairs (15); "stored 09-13" (09-11); §8.2's envelope stated
without its 4-df caveat and a gap range that was seed-42 only; §8.3's rank list, which used
the non-promoted tied trial in two studies, its "0.0010 to 0.0219" and "roughly twice", and a
spread-to-SD multiplier that was not like-for-like; §8.3's "already sitting in the trials,
paid for", which ignored that trials score on one predefined split while comparisons are
five-fold; §8.4's "no dimension found this" (the 07 and 08 verdicts had) and a 2.5-5x
multiplier that understated the 5-15x gap; §8.4's F-13 paragraph, which reported the
objective and omitted that the same trial is 0.074 worse on the induced score; §8.5's
"prove it" and a stable/unstable reading of n = 2; §8.8's "within 0.004" (medians only) and
an omitted `mae_num_z` candidate, plus a prescription presented as shown when the experiment
tests statistic choice only; §8.11's single-arm SD applied to two-arm differences (three
rows "at two SD" became one) and a false claim that the defaults spread was unmeasured;
§8.12's 2-df SD used as if 4-df and a "not a property of the schedule" extrapolated from
three pairs that the six plateau pairs contradict.

**Second pass (2026-09-18, after batches 4 and 5; two verifiers, statistics and arguments).**
Every number in §8.11, §8.12's second-seed paragraph and §8.14 re-derived to the digit,
including the trial identities and the 17-parameter match of the staged runs to their source
trials. Corrected in this version: §8.11's escalation of `credit-g_40nan` from "hold" to
`git rm` on a p ≈ 0.044 reached after a fourth defaults seed was added past the pre-stated
third (p ≈ 0.058), and fragile to any one seed — restored to "hold", with removal grounded
as "no evidence supports the file" rather than "shown worse"; §8.12's Welch t = −3.07 built
on a one-degree-of-freedom defaults SD (the n = 2 artifact again), a "magnitude halved"
pattern that `credit-g_40nan` contradicts, and a stale "same standing" sentence
contradicting §8.11; §8.14(a)'s "does beat, four of four" where the `credit-g_40nan` rows
are 0.1-0.3 of the two-run sigma; §8.14(b)'s "even a perfect selector would not beat the
defaults" (paired p ≈ 0.41, detectable effect ≈ 0.03 — absence of evidence); §8.3's "both
roughly as good as the defaults", which contradicted §8.11 in the same document; and §8.8's
"ceiling", which is the wrong direction — one trial's realised gain is a lower bound on what
a five-fold selector could recover. The verifiers also noted, and this version records, that
the paired-by-seed test is *stronger* than the Welch test used (p ≈ 0.008 / 0.033), so the
Welch choice was conservative, and that the staged runs share `config_source` with the
promoted file.

## 8.14 The control: what the best trial is worth once it is scored the way the comparison is

§8.3's 0.036 is a min-of-40 on each study's single predefined split; §8.8's prescription
assumed most of it could be recovered by a better selector. The test is direct: stage each
`credit-g` cosine study's best-**induced** trial (the trial a perfect selector would have
promoted) into the task-keyed file, run it five-fold at seeds 42 and 420, and compare it
with the promoted file and the defaults under the identical protocol. Staged configs:
`credit-g_20nan` trial `52176d70` (LR_DECODE 3.18e-3, single-split induced 0.8950 against the
promoted pick's 0.9485), `credit-g_40nan` trial `df2b6776` (LR_DECODE 5.16e-3, 0.9206 against
0.9562). Files restored by `git checkout` after each pair. Provenance note for anyone
re-deriving from the store: the four staged runs carry `config_source` = the promoted file's
path (that is where the config was staged), so only `LR_DECODE` (0.003181 / 0.005160 against
the promoted 0.002539 / 0.008706) separates them from promoted-file runs; a query filtering on
`config_source` alone will mix them into §8.11's tuned arm.

| pair | seed | defaults | best-induced trial | promoted file | best − promoted | single-split gap | survives | best − defaults |
|---|---|---|---|---|---|---|---|---|
| credit-g_20nan | 42 | 0.8947 | 0.8999 | 0.9128 | −0.0129 | 0.0535 | 24% | +0.0053 |
| credit-g_20nan | 420 | 0.8992 | 0.8942 | 0.9125 | −0.0184 | 0.0535 | 34% | −0.0050 |
| credit-g_40nan | 42 | 0.9621 | 0.9617 | 0.9677 | −0.0060 | 0.0357 | 17% | −0.0004 |
| credit-g_40nan | 420 | 0.9489 | 0.9748 | 0.9772 | −0.0024 | 0.0357 | 7% | +0.0259 |

Two results, both against earlier drafts of this addendum.

**(a) The best trial lands below the promoted one on all four runs — decisively on one pair,
inside noise on the other.** On `credit-g_20nan` by 0.013-0.018 (1.8-2.5 σ_diff at both
seeds); on `credit-g_40nan` by 0.002-0.006, which is 0.1-0.3 of that pair's two-run sigma,
so its 17% and 7% "survives" figures divide noise by a real number and carry no precision.
Pooled, four of four in direction, paired t = −2.78 on 3 df, p ≈ 0.07. On the two studies
controlled, roughly a fifth of the single-split gap survives (mean 20.5%), and four fifths
of "the 0.036 left on the table" was the selection bias of a minimum over forty noisy
single-split scores. The realised gain of this one trial is about **0.016 on `credit-g_20nan`**
and unresolved on `credit-g_40nan` — the same size as the tuned-versus-defaults gaps, not
twice them.

**(b) No evidence, at two seeds, that the best-induced trial beats the defaults.** Best-induced
minus defaults is +0.0053, −0.0050, −0.0004, +0.0259: two of four worse, mean +0.006, paired
t = 0.95 on 3 df, p ≈ 0.41, 95% interval −0.015 to +0.028; with n = 4 and this noise the
smallest detectable gain is about 0.03, so a real 0.01 advantage either way would be
invisible here. The mean is also carried by one row (`credit-g_40nan` seed 420, where the
defaults arm posted its best draw); without it the mean is 0.000. This is absence of
evidence, not evidence of absence: on the two studies controlled, the reduced space has not
*shown* anything better than the defaults — it has not been shown to lack it either. Per
pair: `credit-g_20nan`'s best-induced trial sits at the defaults (+0.0001) while its promoted
file is significantly worse (§8.11); `credit-g_40nan`'s two configurations are not
distinguishable from each other.

What this settles: §8.3's reframing survives in direction and shrinks in size; §8.8's
prescription gets a measured lower bound of one candidate (≈0.016 on `credit-g_20nan`), not
a ceiling; §8.11's `git rm` of the `credit-g_20nan` file stands and is the whole of the
established actionable content for this dataset. On `kr-vs-kp_20nan` the same control turns out to be already in the store, for a reason
worth stating: the study's best-induced trial (`8ff4c4e4`, induced 0.6043) **is** the
promoted trial — the lucky half of the tie at objective 0.6703 (§8.3) — so "what a perfect
selector would have promoted" and "what was promoted" coincide, and their five-fold runs are
§8.12's tuned arm: 0.6027 / 0.6085 against defaults 0.6146 / 0.6156. The single-split induced
argmin *is* the promoted trial, so the §8.14 control for this pair is already §8.12's tuned
arm, and no further gain over the promotion is available from re-selecting among these 40
trials on the single split (the next-best single-split trials sit 0.019 behind). Whether a
five-fold ranking of the 40 would pick a different trial is the same open question as on
`credit-g`. The 0.0552 single-split gap on this pair is between the two tied trials, not
between the best and the objective's choice in general.

## 8.7 What is still open

- `r420_kk20` (`kr-vs-kp_20nan`, reduced, seed 420) **completed**: 40 trials, 24 distinct
  objectives (still resolution-limited, up from 20 at seed 42), winner rank 2/40, cost
  0.0096. So the resolution-limited pair selected *well* at this seed — further evidence
  that selection quality is a lottery rather than a property of the pair.
- **The next experiment is a validation protocol, not a selector.** §8.8 rules out the
  cheap fix. What has not been run is a study whose objective scores the *induced*
  population on the validation split, or one with enough folds that the ranking signal
  clears the noise. Both need a code change and so were out of scope tonight.
- `full42_kk20` **completed** (48 min): 40 trials, 25 distinct objectives, rho 0.724,
  winner rank 11/40, cost 0.0264; winner at `EPOCHS_PRE = 20` with objective 0.6484
  against the reduced winner's 0.6703 at 300 (§8.4, §8.5).
- **Run on 2026-09-18 (batch 4):** the five-fold re-run of each `credit-g` study's
  best-induced trial at two seeds (§8.14); a third defaults seed on `credit-g_40nan`
  (§8.11); a second defaults seed on `kr-vs-kp_20nan` (§8.12). A fourth defaults seed on
  `credit-g_40nan` and a third on `credit-g_20nan` (batch 5) are in §8.11.
- **Still unmeasured:** any plateau comparison beyond `credit-g_40nan`; the `credit-g`
  controls of §8.14 on a third dataset (`spambase`); and everything that needs a code change
  (§6). The `kr-vs-kp_20nan` best-induced control is already answered by the store (§8.14).
- A `full` profile whose epoch ranges include the task defaults, without which the
  reduction remains untested (this is F-08-2's residue, sharpened).
- One empty `credit-g_40nan` study parent exists in the store from a duplicate process
  killed at 01:31; filter studies on trial count when querying.
