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

## Running notes

- **2026-10-01 12:36 GMT-3, queue 2 split (scheduling only).** Queues 4 and 3 had finished
  (11:46 and 12:32 GMT-3), and queue 2's 24 remaining cells were projected to end
  near the 14:00 GMT-3 cutoff. Queue 2 was stopped between cells (STOP file, after
  `kr-vs-kp_40nan_s7_P`) and its remaining cells, read from the store, were split across two
  launchers that run the same cell runner, thread cap, cutoff and log format: the kr-vs-kp and
  kc2 trios in `q2/`, the biodeg, credit-g and vehicle trios in `q2b/`. At most three trainers
  ran at once, against four before. No cell, arm, seed, configuration or cutoff changed. The
  split launcher and its two cell lists are kept in the state directory,
  `/scratch2/diogoneiss/trident-2026-10-01-confirmatory/split/`.

## Outcome (2026-10-01, 13:20 GMT-3)

All 189 cells ran (00:22–13:19 GMT-3), none failed, none missing; the report found one finished
run per cell. Numbers from `scripts/experiments/confirmatory_report.py`, kept with `--tally` and
the secondary measures in the state directory. Lower `impute_score` is better, so a negative
M − A favours M. Induced cells, 15 fold pairs per variant (seeds 42, 7, 13):

| variant | M − A | verdict | P − A | verdict | P − M | verdict | fold time per cell, A / M / P |
|---|---|---|---|---|---|---|---|
| `credit-g_20nan` | −0.0042 [−0.0222, +0.0137] | none | +0.0022 [−0.0110, +0.0153] | none | +0.0064 [−0.0068, +0.0196] | none | 2.9 / 5.3 / 2.2 min |
| `credit-g_40nan` | −0.0025 [−0.0201, +0.0151] | none | −0.0068 [−0.0247, +0.0111] | none | −0.0043 [−0.0258, +0.0171] | none | 3.6 / 4.8 / 2.1 min |
| `credit-g_60nan` | +0.0013 [−0.0165, +0.0191] | none | +0.0041 [−0.0109, +0.0191] | none | +0.0028 [−0.0108, +0.0163] | none | 2.7 / 3.8 / 1.7 min |
| `credit-g_80nan` | −0.0091 [−0.0267, +0.0084] | none | −0.0123 [−0.0292, +0.0046] | none | −0.0032 [−0.0160, +0.0096] | none | 2.6 / 4.6 / 1.9 min |
| `kr-vs-kp_20nan` | +0.0069 [−0.0088, +0.0227] | none | +0.0055 [−0.0082, +0.0191] | none | −0.0015 [−0.0115, +0.0085] | none | 8.4 / 12.6 / 6.7 min |
| `kr-vs-kp_40nan` | +0.0061 [−0.0037, +0.0159] | none | +0.0140 [+0.0027, +0.0254] | A better than P | +0.0079 [−0.0011, +0.0169] | none | 8.3 / 15.2 / 7.4 min |
| `kr-vs-kp_60nan` | −0.0117 [−0.0248, +0.0015] | none | −0.0088 [−0.0209, +0.0033] | none | +0.0029 [−0.0101, +0.0159] | none | 9.0 / 14.5 / 7.0 min |
| `kr-vs-kp_80nan` | −0.0243 [−0.0453, −0.0032] | M better than A | −0.0340 [−0.0580, −0.0100] | P better than A | −0.0098 [−0.0197, +0.0002] | none | 8.1 / 11.8 / 5.5 min |
| `spambase_20nan` | −0.0146 [−0.0247, −0.0045] | M better than A | −0.0049 [−0.0116, +0.0018] | none | +0.0097 [−0.0005, +0.0198] | none | 16.8 / 28.9 / 17.7 min |
| `spambase_40nan` | +0.0241 [−0.0089, +0.0571] | none | +0.0140 [+0.0009, +0.0270] | A better than P | −0.0102 [−0.0406, +0.0202] | none | 17.9 / 29.2 / 17.1 min |
| `spambase_60nan` | +0.0010 [−0.0354, +0.0373] | none | +0.0700 [−0.0098, +0.1498] | none | +0.0691 [−0.0076, +0.1457] | none | 18.8 / 31.7 / 16.1 min |
| `spambase_80nan` | +0.0072 [−0.0274, +0.0418] | none | −0.0008 [−0.0254, +0.0238] | none | −0.0080 [−0.0254, +0.0094] | none | 19.0 / 30.2 / 14.9 min |
| `vehicle_20nan` | −0.0287 [−0.0466, −0.0108] | M better than A | −0.0253 [−0.0408, −0.0099] | P better than A | +0.0034 [−0.0138, +0.0206] | none | 2.6 / 4.5 / 2.5 min |
| `vehicle_60nan` | −0.0117 [−0.0232, −0.0001] | M better than A | −0.0029 [−0.0158, +0.0100] | none | +0.0088 [−0.0029, +0.0205] | none | 2.2 / 3.5 / 2.1 min |
| `biodeg_20nan` | −0.0460 [−0.0670, −0.0251] | M better than A | −0.0320 [−0.0499, −0.0141] | P better than A | +0.0141 [−0.0046, +0.0327] | none | 4.2 / 6.3 / 4.1 min |
| `biodeg_60nan` | −0.0577 [−0.0771, −0.0383] | M better than A | −0.0397 [−0.0591, −0.0203] | P better than A | +0.0180 [+0.0032, +0.0329] | M better than P | 4.0 / 6.2 / 3.8 min |
| `kc2_20nan` | −0.0250 [−0.0674, +0.0175] | none | +0.0133 [−0.0227, +0.0493] | none | +0.0383 [−0.0143, +0.0908] | none | 1.8 / 2.6 / 1.5 min |
| `kc2_60nan` | −0.0417 [−0.0835, +0.0002] | none | +0.0096 [−0.0402, +0.0593] | none | +0.0512 [+0.0093, +0.0931] | M better than P | 1.8 / 2.9 / 1.6 min |
| `pendigits_20nan` | −0.0682 [−0.0719, −0.0645] | M better than A | −0.0680 [−0.0710, −0.0651] | P better than A | +0.0002 [−0.0022, +0.0025] | none | 21.1 / 41.9 / 42.0 min |
| `letter_20nan` | −0.0538 [−0.0562, −0.0513] | M better than A | −0.0520 [−0.0551, −0.0489] | P better than A | +0.0018 [+0.0002, +0.0034] | M better than P | 32.6 / 57.4 / 57.4 min |
| `electricity_20nan` | −0.0343 [−0.0422, −0.0263] | M better than A | −0.0317 [−0.0418, −0.0217] | P better than A | +0.0025 [−0.0060, +0.0111] | none | 64.5 / 100.3 / 94.4 min |

Fold time per cell is a cell's five folds, descriptive: up to four trainers shared the GPU.

**Tally.** M − A: M better on 9 variants, A better on none, no detectable difference on 12.
P − A: P better on 7, A better on 2 (`kr-vs-kp_40nan` and `spambase_40nan`, both +0.014), none on
12. P − M: M better on 3 (`biodeg_60nan`, `kc2_60nan`, `letter_20nan`), none on 18.

**Reading, by the rule fixed in advance.**
1. M qualifies: no variant shows "A better than M".
2. P does not: two variants show "A better than P". Rule 2 compares P with M only when both
   qualify, so with one qualifying the recommendation is **M**: the normalised pre-training
   objective (`embedding_normalized`) with a 450-epoch decode stage. The user takes the decision
   (T00).

**Secondary measures and caveats, stated rather than ruled on.**
- **P on `spambase_60nan`:** worse than A on all three seeds (+0.061, +0.039, +0.110; mean 1.044
  against 0.974), yet the interval is too wide for a verdict. A failure of patience 50 that the
  rule cannot see, on top of its two verdicts.
- **Where P lost, it stopped early.** P trained a mean of 100 decode epochs on `kr-vs-kp_40nan`
  and 151 on `spambase_40nan`. Outside pendigits, letter and electricity it stopped after
  51–358 epochs (means 63–196), never at 450. Only pendigits, letter and electricity reached the 450 ceiling, on 4, 8
  and 3 of 15 folds. A longer patience is T11's question, not this study's.
- **Masked cells:** kr-vs-kp_20nan gives "A better than M" (+0.021) and "A better than P"
  (+0.026). Elsewhere the masked verdicts favour the candidates: M and P each better on biodeg_60nan,
  pendigits and letter. M better than P on kc2_20nan. The rule is on induced cells only.
- **Against the best baseline:** the arm's mean induced score is below the best baseline's on 7
  variants for A, 11 for M, 9 for P (masked: 12, 14, 13). M moves below it where A was above
  on six variants (pendigits, letter, both biodeg, kc2_60nan, kr-vs-kp_60nan) and above it where
  A was below on two (kr-vs-kp_20nan, spambase_40nan).
- **Price.** Summed over the 21 variants, a cell of M takes 1.65x A's time (418 against 253
  minutes per seed), P 1.22x (310). On electricity, 64.5 → 100.3 min a cell.
- **Time.** Planned about 8.5 h wall (end about 09:15 GMT-3); it took 12 h 57 min, 1.52x the
  plan. Small tables took 3–5x their per-cell estimate (a fixed cost per cell that does not
  shrink with the table), kr-vs-kp 3.5x, and four trainers shared the GPU.
