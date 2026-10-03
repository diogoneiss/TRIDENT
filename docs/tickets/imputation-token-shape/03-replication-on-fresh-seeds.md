# Does the gaps-as-mask default hold on seeds it was not chosen on? (pre-registered 2026-10-03)

This ticket was written and committed before the first run, at 00:44 GMT-3 on 2026-10-03 (commit
`e2f4e47`; the queues started 00:44:31 GMT-3). The user adopted
`DECODE_GAP_TOKEN mask` as the imputation default (ADR 0014) and asked for an overnight study to
validate it.

## Why

E32 chose the new default on seeds 42, 7 and 13, the same seeds every study since E24 used, so
the evidence for it and the decision to adopt it rest on the same three draws. A replication on
seeds no imputation study has used tests whether "better on 11 of 21, worse on none" was a
property of the change or of those draws.

## Design (fixed before launch)

- **Arms, both run fresh:**
  - **G:** a plain imputation run under today's defaults (ADR 0014), naming nothing.
  - **M:** the same run with `--decode_gap_token null`, the default before ADR 0014 (ADR 0013's
    configuration, E30's and E32's arm M).
  - Each variant reads its promoted file or the defaults exactly as any run does; five folds.
- **Variants:** all 21 of ADR 0007. **Seeds:** 101, 202, 303, none used by an earlier imputation
  study (the store holds 1, 7, 13, 42, 99 and 420). **Size:** 126 cells, tagged
  `experiment=gap-token-replication-2026-10-03`, `arm=G|M`.
- **Running it:** four concurrent queues (`scripts/experiments/run_gap_token_replication_queue.sh
  1..4`), one trainer each, CPU threads capped at 4. (variant, seed) pairs are assigned longest
  first by twice E30's arm-M cell time, about 627 minutes of those per queue; each pair runs G
  then M. E32's cells ran at about 1.08x those times, so about 11.3 h of wall time is expected,
  ending near 12:05 GMT-3. No cell starts after 18:00 GMT-3 on 2026-10-03 (21:00 UTC); a cell not
  started by then is reported missing. Nothing is rerun or extended on its result; a crash is
  fixed, its cell rerun from the start, and the rerun noted here.
- **Checks before launch:** default runs of credit-g_40nan (promoted) and vehicle_20nan (no
  promoted file) at seed 42, after ADR 0014, must equal E32's G cells on every `impute/*` metric;
  both regression fixtures must pass; the kit's check cell must hold its reference.

## Measures

- **Primary:** per variant, D = G − M on the induced `impute_score`, paired by seed and fold (15
  pairs on the fresh seeds), t-interval; a verdict only when the interval excludes zero and all
  three seeds' mean D agree in sign. A negative D favours G.
- **Reading, fixed now:**
  1. **Replicated:** no variant among the 21 shows "M better", and G is better on at least 5 of
     the 9 variants at 60nan or more (credit-g, kr-vs-kp and spambase at 60 and 80; vehicle,
     biodeg and kc2 at 60). This is E32's criterion, on new draws.
  2. **Safe, weaker:** no variant shows "M better", but G is better on fewer than 5 of the 9. The
     default stands, since it was never worse, and the effect is reported as smaller than E32's.
  3. **Not replicated:** any variant shows "M better". The report lists those variants and the
     user reconsiders the default; nothing is reverted automatically.

  E32 landed exactly at the heavy threshold (5 of 9, two of them close to zero), so outcome 2 is
  a likely result on new draws even if the change works as E32 measured: a regression toward the
  mean of a result chosen at its threshold, not a failure. The safety clause, no variant worse,
  is the one the adoption rests on.
- **Secondary, descriptive:**
  - D by missing level (20, 40, 60, 80nan), against E32's −0.011, −0.026, −0.036, −0.012;
  - the pooled view, E32 and this study together: six seeds, 30 pairs per variant, the same
    verdict rule with all six seeds agreeing;
  - the masked population, G − M;
  - means over the 21 variants, and how many sit below the best baseline under each arm (E32:
    G 16, M 11);
  - fold time per cell, and estimated against real wall time.
- **Analysis:** `scripts/experiments/gap_token_replication_report.py` (`--tally` for the counts).

## Known limits, stated in advance

- **Same code, same machine, same variants.** This replicates over seeds, not over datasets or
  hardware; a fresh seed changes the folds, the masks and the initialisation together.
- **Paired streams.** As in E32, a G cell and its M cell consume the same random stream, so the
  pairs are tighter than independent draws.
- **Multiplicity:** 21 tests with the seed-agreement rule, as in E30 to E32.

## Running notes

- **Checks before launch, done:** the default runs of credit-g_40nan and vehicle_20nan at seed 42
  equal E32's G cells on every `impute/*` metric (55 and 35); both regression fixtures pass; the
  kit's check cell holds its reference with both pins; the unit suite passes (284 tests, mypy
  included).
