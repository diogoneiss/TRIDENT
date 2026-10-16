# Re-tuning the promoted imputation files under today's defaults (pre-registered 2026-10-15)

This ticket was written and committed before the first study, on the night of 2026-10-15 GMT-3.
The workers start right after the commit, so about 07:00 GMT-3 on 2026-10-16 is the expected end.
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
