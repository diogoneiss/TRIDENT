# Does the candidate configuration hold on all 21 imputation variants? (pre-registered 2026-10-01)

Written and committed before the first run, at 00:22 GMT-3. Settles task T00 (the imputation
default) with the variants E24–E29 never used.

## Why

Every pre-training and decode result so far rests on four variants. On them the normalised
pre-training objective was never worse than the current one (E25), adding a 450-epoch decode
stage was never worse than either half (E26), and a decode patience of 50 kept those scores on
three of four tables at a fraction of the time (E27). One seed of a five-fold run carries about
±0.02 of noise (E12), so four variants cannot license a default for 21.

## Arms (fixed before launch)

| arm | pre-training | decode | what it is |
|---|---|---|---|
| **A** | `embedding`, configured epochs (300) | configured (150) | the current default |
| **M** | `embedding_normalized`, 300 | configured + pre-training (450) | candidate (E26) |
| **P** | `embedding_normalized`, 300 | up to 450, `DECODE_PATIENCE` 50 | candidate, faster (E27) |

Each variant uses the configuration a run of it loads today (the promoted `*.imputation.json`
of credit-g_40nan, kr-vs-kp_20nan and _40nan, spambase_20nan and _40nan; the defaults
elsewhere), cosine, five folds, the default batched decoder heads (ADR 0012). Every cell runs now;
none is reused, so every pair is one machine and one head mode.

- **Variants (21):** credit-g, kr-vs-kp, spambase at 20/40/60/80nan; vehicle, biodeg, kc2 at
  20/60nan; pendigits, letter, electricity at 20nan, the set of ADR 0007.
- **Seeds 42, 7, 13** for every arm and variant; 189 cells tagged
  `experiment=confirmatory-2026-10-01`, `arm=A|M|P`.
- **Running it:** four concurrent queues (`scripts/experiments/run_confirmatory_queue.sh 1..4`),
  one trainer each, CPU threads capped at 4. The (variant, seed) trios are assigned longest first
  to the least-loaded queue (about 360 single-trainer minutes each), and each trio runs A, M, P in
  turn. No cell starts after 14:00 GMT-3 on 2026-10-01; a cell not started by then is reported
  missing. Nothing is rerun or extended on its result; a crash is fixed and its cell rerun from
  the start, noted here.

## Measures

- **Primary:** `impute/induced/impute_score` per fold; per variant M − A and P − A paired by seed
  and fold (15 pairs), t-interval, a verdict only when the interval excludes zero and all three
  seeds agree in sign; and the tally of verdicts over the 21 variants.
- **Secondary:** P − M the same way; masked cells; each arm against the run's best baseline;
  decode epochs trained by P; fold time per cell (descriptive, four trainers share the GPU).
- **Reading for the default, fixed now:**
  1. A candidate qualifies if no variant shows "A better than" it.
  2. If both qualify, P is recommended (it is the cheaper one) unless "M better than P" holds on
     three or more variants, in which case M is.
  3. If neither qualifies, the default stays A and the report lists where each candidate wins
     and loses.
  The user takes the decision; this is the recommendation the result will carry.
- Analysis: `scripts/experiments/confirmatory_report.py` (`--tally` for the counts).

## Known limits, stated in advance

One configuration per variant (the promoted files were tuned for A's pipeline), three seeds, the
default `LR_PRE`. 21 variants and two primary comparisons are 42 tests; with the seed-agreement
rule a false verdict is rarer than 1 in 20 per test, but a lone verdict among many nulls deserves
less weight than a pattern.
