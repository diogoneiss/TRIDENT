# Re-tuning the promoted imputation files under today's defaults (pre-registered 2026-10-15)

This ticket was written and committed before the first study (`00766f4`, 23:34 GMT-3 on
2026-10-15). The workers started at 23:34:31 GMT-3, so about 07:00 GMT-3 on 2026-10-16 was the
expected end.
The user chose this study ("vamos para o 2", 2026-10-15) from the follow-ups left open after ADR
0018, and asked for it to run overnight.

## Why

Five variants read a promoted file (ADR 0005): credit-g_40nan, kr-vs-kp_20nan, kr-vs-kp_40nan,
spambase_20nan and spambase_40nan.

- **The sampled values in those files are stale.** ADR 0005's reduced studies chose them on
  2026-09-11 under the old configuration: `embedding` pre-training, decode 150, gaps shown as
  `[NULL]`, the loss checkpoint, and trials ranked by the masked validation cells. ADRs 0013 to 0015
  rewrote each file's held keys to today's values. The sampled ones (`PROB_MASCARA`, `LR_DECODE`,
  `WEIGHT_DECAY_DECODE`, `DROPOUT`, and `LAMBDA_NUM` on credit-g) stayed as they were, so nobody
  chose them for today's regime.
- **Tuning had not paid even then.** ADR 0005's comparison found the tuned files no better than
  the defaults: no interval separated. credit-g_20nan's file was removed when it proved worse
  across seeds.
- **The old studies ranked trials on the wrong population.** ADR 0008 found the masked objective a
  poor guide to the induced benchmark: on credit-g_80nan its winner ranked 40 of 40 by the induced
  cells.

The question: under today's defaults, does a fresh reduced search ranked by the validation gaps
find configurations that beat the current files? And do the current files still beat having no
file at all?

## Design (fixed before launch)

### Studies

- **Five studies,** one per promoted file. Each runs:

  ```
  opt.py --dataset_name <variant> --task imputation --n_trials 40 --seed 42 \
    --search_space reduced --search_objective induced
  ```

  TPE is seeded 42, and each trial trains once on seed 42's predefined split.
- **Sampled,** with ADR 0005's ranges:

  | knob | range |
  |---|---|
  | `PROB_MASCARA` | 0.2–0.6 |
  | `LR_DECODE` | 1e-4–1e-2, log |
  | `WEIGHT_DECAY_DECODE` | 1e-5–1e-2, log |
  | `DROPOUT` | 0.1–0.5 |
  | `LAMBDA_NUM` | 0.1–10, log; credit-g only, the one mixed table |

- **Held at today's defaults** through the task layer (ADR 0013): normalised pre-training,
  decode 450, `cosine`, gaps shown as `[MASK]` and the checkpoint on the validation gaps. The
  encoder keeps 2 layers and a feed-forward width of 32.
- **No `--promote_best` and no `--retrain_best`:** nothing under `datasets/hiperparams` changes
  tonight.

### Arms

Three arms, on seeds 101, 202 and 303 (which no study saw), five folds each:

| arm | configuration | cells |
|---|---|---|
| **T** | the study's winner. Its `best_hyperparameters.json` is the override, so every other key takes today's defaults. This is how the trials trained, and how a promoted file of the winner would resolve. | 15 new |
| **D** | today's defaults with no file (an empty override) | 15 new |
| **F** | the current file, read from E34's cells as their K readout (`impute/*/induced_checkpoint/*`), which equals a run reading the file to the last digit (check 1 below) | none new |

- **Runner:** `scripts/experiments/retune_run.py`.
- **Tags:** `experiment=retune-2026-10-15`, plus `arm=T` or `arm=D`.
- **Pairing:** each pair shares folds and masks. The integrity check holds each seed's baseline
  scores equal across the three arms.

### Out of scope

- **pendigits_20nan and letter_20nan (ADR 0018).** A reduced trial runs every held knob at the task
  defaults. Their studies would therefore train the 2-layer encoder, discarding ADR 0018, and cost
  about 5 to 7 h each.
- **Variants with no promoted file,** credit-g_20nan included (its file was removed on 2026-09-30).

### Running it

- **Four workers share one job list** (`scripts/experiments/run_retune_queue.sh 1..4`; the list is
  `retune_report.py --jobs`). Each worker runs one trainer at a time, with CPU threads capped at 4.
- **Priority:**
  1. the five studies, longest first, so both spambase studies start at once;
  2. each variant's T cells, as soon as its study closes;
  3. the D cells. They depend on nothing, so they fill any idle worker.
- **Estimated duration.** A trial costs about one fold of E35's cells with four trainers sharing
  the GPU:

  | variant | a trial | a 40-trial study | a 5-fold cell |
  |---|---|---|---|
  | credit-g | ~1 min | ~0.7 h | ~5 min |
  | kr-vs-kp | ~4 min | ~2.4–2.7 h | ~19 min |
  | spambase | ~7 min | ~4.8 h | ~33 min |

  That totals about 26 worker-hours, so about 6.5 to 7 h of wall time.
- **Cutoff:** no job starts after 10:00 GMT-3 on 2026-10-16. A job not started by then is reported
  missing. A variant missing a cell reads "incomplete", and its recommendation waits.
- **No reruns or extensions on results.** A crash is fixed, its job rerun from the start, and the
  rerun noted here. A killed study restarts from its first trial, because `opt.py` opens a new
  study directory on every launch.

### Checks before launch

1. **The T path reproduces a file run.** The runner took the current credit-g_40nan file's five
   sampled values as `--config` (seed 101, `--no-mlflow`). It equalled E34's K readout fold by fold
   on all 14 model metrics and all 40 baseline columns. So T's path is a run reading the file, and
   nothing has moved since E34. The resolved configuration it printed was normalised pre-training,
   decode 450, `cosine`, `mask`, `induced`, 2 layers, `DIM_FEED` 32 and no final norm.
2. **A one-trial smoke of the study command** (`--disable_mlflow`) passed. It ranked by
   `validation/impute/induced/impute_score`, trained decode 450 and wrote
   `best_hyperparameters.json`. Its score was not read.
3. **The store held no earlier study of this kind.** It had no study of the reduced profile ranked
   by the induced objective, so the profile, the objective and the start time identify tonight's
   five.

## Measures

- **Primary.** Per variant, three contrasts on the induced `impute_score`: T − F, D − F and T − D.
  Each is paired by seed and fold, so 15 pairs. A verdict needs the t-interval to exclude zero and
  all three seeds' mean differences to agree in sign.
- **Recommendation, fixed now.** An arm qualifies when it beats at least one other arm by verdict and
  loses to none. For each variant, the first rule that applies:
  1. **T qualifies:** recommend promoting the study's winner as the variant's file. It would be a
     complete file, as `--promote_best` writes it.
  2. **D qualifies and beats F:** recommend removing the file (ADR 0005's credit-g_20nan
     precedent).
  3. **Otherwise:** keep F.

  The decision is the user's, by an ADR amending ADR 0005.
- **Secondary, descriptive:**
  - the masked contrasts;
  - each study's winner beside the file's current values, with the fANOVA importances;
  - the best objective (a validation score, never compared with a test number);
  - each arm's mean against the best baseline;
  - minutes a cell;
  - the integrity check.
- **Analysis:** `scripts/experiments/retune_report.py`. `--tally` gives the verdicts, the
  recommendations and the integrity check; `--studies` gives the winners and the importances.

## Known limits, stated in advance

- **One study per variant:** one seed and one split, so the winner carries the noise of its
  selection. A 40-trial TPE over 4 or 5 dimensions explores coarsely. ADR 0005 found the decode
  learning rate on a plateau from about 1e-3 to 1e-2, so a small gain at most is the prior.
- **The objective and the checkpoint share cells.** Under `DECODE_CHECKPOINT induced`, the induced
  objective scores the same validation gaps that chose each trial's checkpoint (ADR 0008's
  amendment of 2026-10-03). The bias is common to every trial, so the ranking survives, but the
  best objective is optimistic.
- **F comes from other runs.** Its cells are E34's (2026-10-03), not tonight's. Check 1 and the
  integrity check guard against drift.
- **Multiplicity:** 15 primary verdicts (5 variants × 3 contrasts).

## Outcome (2026-10-16, 07:15 GMT-3)

**Run.** All 35 jobs ran from 23:34 GMT-3 on 2026-10-15 to 06:56 GMT-3 on 2026-10-16: 7 h 22 min
against the 6.5 to 7 h planned. None failed and none is missing.

- **Studies:** each of the five reached 40 trials, none pruned.
- **Cells:** 15 T and 15 D.
- **Integrity:** all 30 pairs against F have equal baseline scores.

**Tables.** Numbers come from `scripts/experiments/retune_report.py`; the tables were built from it
by script. Lower is better; a negative difference favours the first arm. The columns are:

- the induced `impute_score` of each arm, over 15 fold pairs;
- each contrast as mean [95% interval], then the three seeds' means in parentheses, then the
  verdict;
- "rule": the recommendation fixed in advance.

| variant | F | D | T | best baseline | T − F | D − F | T − D | rule |
|---|---|---|---|---|---|---|---|---|
| `spambase_40nan` | 0.9068 | 0.9023 | 0.9019 | 0.9345 | −0.0048 [−0.0106, +0.0009] (+0.0008, −0.0069, −0.0084) none | −0.0044 [−0.0081, −0.0007] (−0.0027, −0.0061, −0.0045) D better | −0.0004 [−0.0054, +0.0046] (+0.0035, −0.0007, −0.0040) none | remove the file |
| `spambase_20nan` | 0.8529 | 0.8439 | 0.8535 | 0.8597 | +0.0006 [−0.0070, +0.0081] (−0.0093, +0.0078, +0.0032) none | −0.0091 [−0.0186, +0.0005] (−0.0225, +0.0028, −0.0074) none | +0.0096 [+0.0015, +0.0177] (+0.0132, +0.0050, +0.0106) D better | keep F |
| `kr-vs-kp_40nan` | 0.7626 | 0.7258 | 0.7234 | 0.7721 | −0.0392 [−0.0467, −0.0317] (−0.0389, −0.0352, −0.0435) T better | −0.0368 [−0.0449, −0.0286] (−0.0346, −0.0356, −0.0402) D better | −0.0024 [−0.0117, +0.0068] (−0.0043, +0.0004, −0.0033) none | promote T |
| `kr-vs-kp_20nan` | 0.5866 | 0.5829 | 0.5830 | 0.6118 | −0.0035 [−0.0213, +0.0142] (+0.0055, −0.0091, −0.0070) none | −0.0037 [−0.0142, +0.0069] (−0.0112, +0.0021, −0.0020) none | +0.0001 [−0.0129, +0.0132] (+0.0166, −0.0112, −0.0050) none | keep F |
| `credit-g_40nan` | 0.9662 | 0.9208 | 0.9162 | 0.9645 | −0.0501 [−0.0603, −0.0398] (−0.0494, −0.0434, −0.0575) T better | −0.0454 [−0.0570, −0.0339] (−0.0393, −0.0447, −0.0522) D better | −0.0046 [−0.0146, +0.0054] (−0.0100, +0.0014, −0.0053) none | promote T |

**Reading, by the rule fixed in advance:**

| recommendation | variants |
|---|---|
| promote the study's winner | credit-g_40nan, kr-vs-kp_40nan |
| remove the file | spambase_40nan |
| keep the file | spambase_20nan, kr-vs-kp_20nan |

**The current files hurt where they differ from the defaults.** D beats F on three variants:

| variant | D against F |
|---|---|
| kr-vs-kp_40nan | −4.8% |
| credit-g_40nan | −4.7% |
| spambase_40nan | −0.5% |

D is worse than F on none. With its file, credit-g_40nan scored worse than its best baseline
(0.9662 against 0.9645); without it, 0.9208. These sampled values have been in effect since the files were
promoted (`fcaadfb`, 2026-09-11, ADR 0005), so every headline on these five variants since then,
E30 to E38 included, used them. Tonight's D cells log `config_source = override` (an empty
override), not `defaults`.

**The re-tuning does not beat the defaults.** T − D has no verdict on four variants and is "D
better" on spambase_20nan (+1.1%). Where the rule promotes T, D is as good:

| variant | T − D |
|---|---|
| kr-vs-kp_40nan | −0.3% |
| credit-g_40nan | −0.5% |

Neither interval excludes zero. The gain over F on those two variants is the stale values going
away, not anything the search found. That is ADR 0005's finding, replicated under today's regime.
So a decision-equivalent alternative to the rule's "promote T" there is "remove the file". The
pre-stated rule ranks T first and its reading stands; the choice is the user's.

**Secondary measures:**

- **Masked population:**

  | variant | T − F | D − F | T − D | minutes a cell, T / D / F |
  |---|---|---|---|---|
  | `spambase_40nan` | −0.0049 [−0.0074, −0.0024] (−0.0045, −0.0060, −0.0042) T better | +0.0032 [−0.0024, +0.0088] (+0.0065, −0.0014, +0.0044) none | −0.0081 [−0.0135, −0.0027] (−0.0110, −0.0046, −0.0085) T better | 34.8 / 35.4 / 35.7 |
  | `spambase_20nan` | +0.0059 [−0.0011, +0.0130] (+0.0085, +0.0033, +0.0061) none | −0.0035 [−0.0089, +0.0019] (−0.0057, +0.0004, −0.0052) none | +0.0094 [+0.0046, +0.0143] (+0.0142, +0.0029, +0.0113) D better | 36.0 / 35.7 / 34.0 |
  | `kr-vs-kp_40nan` | −0.0321 [−0.0546, −0.0097] (−0.0434, −0.0496, −0.0035) T better | −0.0372 [−0.0532, −0.0211] (−0.0463, −0.0389, −0.0262) D better | +0.0050 [−0.0145, +0.0245] (+0.0030, −0.0107, +0.0228) none | 18.7 / 19.0 / 18.6 |
  | `kr-vs-kp_20nan` | −0.0182 [−0.0394, +0.0030] (−0.0274, −0.0018, −0.0253) none | −0.0194 [−0.0304, −0.0084] (−0.0296, −0.0058, −0.0230) D better | +0.0013 [−0.0126, +0.0151] (+0.0022, +0.0040, −0.0023) none | 20.6 / 15.6 / 17.0 |
  | `credit-g_40nan` | −0.0385 [−0.0682, −0.0089] (−0.0255, −0.0336, −0.0565) T better | −0.0402 [−0.0702, −0.0103] (−0.0282, −0.0171, −0.0754) D better | +0.0017 [−0.0196, +0.0230] (+0.0027, −0.0165, +0.0189) none | 7.9 / 6.9 / 4.5 |

  The masked reading agrees with the induced one, with two exceptions:
  - spambase_40nan's winner is better than both F and D on the masked cells (−0.008 against D),
    though not on the induced ones;
  - D beats F on kr-vs-kp_20nan's masked cells.
- **Each study's winner beside the file.** Values are winner → file, with the fANOVA importance in
  parentheses. The objective is a validation score.

  | variant | study h | best trial | objective | `PROB_MASCARA` | `LR_DECODE` | `WEIGHT_DECAY_DECODE` | `DROPOUT` | `LAMBDA_NUM` |
  |---|---|---|---|---|---|---|---|---|
  | `spambase_40nan` | 4.94 | 37 | 0.9214 | 0.5 → 0.5 (0.39) | 2.28e-3 → 1.08e-3 (0.52) | 9.98e-3 → 2.97e-4 (0.06) | 0.3 → 0.2 (0.03) | — |
  | `spambase_20nan` | 4.94 | 32 | 0.8321 | 0.2 → 0.4 (0.04) | 1.36e-3 → 1.33e-3 (0.34) | 3.65e-4 → 4.21e-5 (0.13) | 0.1 → 0.3 (0.49) | — |
  | `kr-vs-kp_40nan` | 3.10 | 6 | 0.6699 | 0.4 → 0.3 (0.36) | 3.72e-3 → 1.70e-3 (0.25) | 3.97e-5 → 2.15e-4 (0.24) | 0.3 → 0.1 (0.15) | — |
  | `kr-vs-kp_20nan` | 3.12 | 25 | 0.5298 | 0.5 → 0.2 (0.02) | 5.60e-3 → 1.74e-3 (0.73) | 1.10e-5 → 3.90e-3 (0.00) | 0.3 → 0.2 (0.26) | — |
  | `credit-g_40nan` | 1.31 | 32 | 0.8561 | 0.6 → 0.2 (0.08) | 1.66e-3 → 8.71e-3 (0.16) | 8.48e-5 → 3.14e-3 (0.04) | 0.5 → 0.2 (0.51) | 0.948 → 0.231 (0.22) |

  `LR_DECODE` leads the importances on three studies of five and `DROPOUT` on two. The defaults
  are `PROB_MASCARA` 0.5, `LR_DECODE` 1e-3, `WEIGHT_DECAY_DECODE` 1.9e-3, `DROPOUT` 0.2 and
  `LAMBDA_NUM` 1.0. credit-g_40nan's file had the largest departure from them, a decode learning
  rate of 8.7e-3 and a loss balance of 0.23, and its winner moved back toward them.
- **Cost:** T and D cost what F costs on spambase and kr-vs-kp. The credit-g F cells come from
  E34 under another load, so their 4.5 minutes do not compare.

**Decided 2026-10-16:** the user chose to remove the three files that lose to the defaults
(credit-g_40nan, kr-vs-kp_40nan, spambase_40nan), rather than promote T on two of them, since T
equals D there. spambase_20nan and kr-vs-kp_20nan keep theirs. See
[ADR 0019](../../adr/0019-remove-the-stale-promoted-imputation-files.md).
