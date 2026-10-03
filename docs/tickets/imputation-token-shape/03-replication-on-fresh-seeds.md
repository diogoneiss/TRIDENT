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

## Outcome (2026-10-03, 12:15 GMT-3)

All 126 cells ran (00:44 to 12:05 GMT-3), none failed, none missing. Numbers from
`scripts/experiments/gap_token_replication_report.py` (kept with `--tally` in the state
directory). Lower is better; a negative difference favours G. ★ marks the nine variants at
60nan or more. Induced cells, 15 fold pairs per variant on the fresh seeds.

| variant | G | M | best baseline | G − M, fresh seeds (101, 202, 303) | verdict | pooled with E32, six seeds | verdict |
|---|---|---|---|---|---|---|---|
| `credit-g_20nan` | 0.8817 | 0.9018 | 0.9138 | −0.0200 [−0.0392, −0.0009] (−0.0135, −0.0116, −0.0350) | G better | −0.0117 [−0.0232, −0.0001] | none |
| `credit-g_40nan` | 0.9734 | 0.9784 | 0.9645 | −0.0050 [−0.0203, +0.0103] (−0.0177, +0.0009, +0.0018) | none | −0.0105 [−0.0205, −0.0006] | none |
| `credit-g_60nan` ★ | 1.0003 | 1.0133 | 1.0000 | −0.0130 [−0.0299, +0.0039] (−0.0103, −0.0101, −0.0186) | none | −0.0110 [−0.0218, −0.0002] | none |
| `credit-g_80nan` ★ | 1.0305 | 1.0562 | 1.0000 | −0.0256 [−0.0477, −0.0036] (−0.0222, −0.0486, −0.0061) | G better | −0.0207 [−0.0331, −0.0082] | none |
| `kr-vs-kp_20nan` | 0.5947 | 0.6205 | 0.6118 | −0.0258 [−0.0349, −0.0167] (−0.0253, −0.0236, −0.0286) | G better | −0.0239 [−0.0318, −0.0161] | G better |
| `kr-vs-kp_40nan` | 0.7600 | 0.7760 | 0.7721 | −0.0161 [−0.0290, −0.0031] (−0.0224, −0.0072, −0.0186) | G better | −0.0128 [−0.0206, −0.0050] | G better |
| `kr-vs-kp_60nan` ★ | 0.8716 | 0.8964 | 0.9090 | −0.0249 [−0.0330, −0.0167] (−0.0180, −0.0289, −0.0276) | G better | −0.0253 [−0.0310, −0.0196] | G better |
| `kr-vs-kp_80nan` ★ | 1.0104 | 1.0087 | 1.0000 | +0.0017 [−0.0150, +0.0185] (+0.0024, −0.0007, +0.0035) | none | +0.0053 [−0.0079, +0.0185] | none |
| `spambase_20nan` | 0.8495 | 0.8764 | 0.8597 | −0.0270 [−0.0367, −0.0173] (−0.0379, −0.0197, −0.0233) | G better | −0.0223 [−0.0292, −0.0154] | G better |
| `spambase_40nan` | 0.9064 | 0.9421 | 0.9345 | −0.0357 [−0.0508, −0.0206] (−0.0179, −0.0443, −0.0449) | G better | −0.0434 [−0.0600, −0.0269] | G better |
| `spambase_60nan` ★ | 0.9438 | 1.0195 | 0.9814 | −0.0757 [−0.1401, −0.0113] (−0.0718, −0.0642, −0.0912) | G better | −0.0529 [−0.0871, −0.0187] | G better |
| `spambase_80nan` ★ | 0.9946 | 1.0272 | 1.0000 | −0.0326 [−0.0659, +0.0007] (−0.0331, −0.0615, −0.0032) | none | −0.0310 [−0.0529, −0.0091] | G better |
| `vehicle_20nan` | 0.4148 | 0.4336 | 0.4643 | −0.0188 [−0.0238, −0.0138] (−0.0215, −0.0119, −0.0230) | G better | −0.0177 [−0.0248, −0.0107] | G better |
| `vehicle_60nan` ★ | 0.6586 | 0.6806 | 0.7034 | −0.0220 [−0.0352, −0.0088] (−0.0283, −0.0152, −0.0225) | G better | −0.0227 [−0.0339, −0.0116] | G better |
| `biodeg_20nan` | 0.5991 | 0.6325 | 0.6508 | −0.0334 [−0.0436, −0.0233] (−0.0274, −0.0386, −0.0343) | G better | −0.0295 [−0.0388, −0.0203] | G better |
| `biodeg_60nan` ★ | 0.7795 | 0.8355 | 0.8622 | −0.0559 [−0.0735, −0.0384] (−0.0620, −0.0457, −0.0601) | G better | −0.0571 [−0.0683, −0.0459] | G better |
| `kc2_20nan` | 0.5402 | 0.5683 | 0.5216 | −0.0281 [−0.0718, +0.0155] (−0.0553, +0.0144, −0.0435) | none | −0.0175 [−0.0411, +0.0061] | none |
| `kc2_60nan` ★ | 0.5699 | 0.6632 | 0.7242 | −0.0933 [−0.1508, −0.0358] (−0.1515, −0.0763, −0.0521) | G better | −0.0800 [−0.1194, −0.0405] | G better |
| `pendigits_20nan` | 0.3335 | 0.3345 | 0.3510 | −0.0010 [−0.0027, +0.0007] (−0.0002, −0.0011, −0.0017) | none | −0.0013 [−0.0024, −0.0001] | none |
| `letter_20nan` | 0.4595 | 0.4607 | 0.4722 | −0.0012 [−0.0024, −0.0001] (−0.0021, −0.0017, +0.0001) | none | −0.0013 [−0.0021, −0.0006] | none |
| `electricity_20nan` | 0.6177 | 0.6261 | 0.6022 | −0.0084 [−0.0173, +0.0004] (−0.0049, −0.0127, −0.0077) | none | −0.0065 [−0.0148, +0.0018] | none |

**Tally.** On the fresh seeds: G better on 13 of 21 variants, M better on none, no detectable
difference on 8. On the nine ★ variants: G better on 6 (credit-g_80nan, kr-vs-kp_60nan,
spambase_60nan, vehicle_60nan, biodeg_60nan, kc2_60nan), M better on none, no detectable
difference on 3.

**Reading, by the rule fixed in advance: replicated.** No variant shows "M better", and G is
better on 6 of the 9 heavy variants, one more than the 5 E32 found and the rule required. The
regression toward the mean that the ticket warned of did not happen; on these seeds the effect is
larger, not smaller. ADR 0014's default stands on evidence from six seeds.

**Secondary measures.**
- **Gains at every missing level, larger than E32's:** mean G − M −0.018 at 20nan (9 variants),
  −0.019 at 40nan (3), −0.048 at 60nan (6), −0.019 at 80nan (3); E32 had −0.011, −0.026, −0.036,
  −0.012.
- **Pooled with E32 (six seeds, 30 pairs, all six seeds agreeing):** G better on 12, M better on
  none; on the heavy nine, 6 and 0. The pooled rule is the stricter one, since one seed of six
  leaning the other way vetoes a verdict: credit-g_20nan and _80nan, for example, are "G better"
  on the fresh seeds and short of a pooled verdict because E32 seeds lean the other way (two for
  credit-g_20nan, one for credit-g_80nan); spambase_80nan goes the other way, no verdict on the
  fresh seeds and "G better" pooled.
- **Means over the 21 variants:** G 0.7519, M 0.7786. Below the best baseline: G on 15 variants,
  M on 9 (E32: 16 and 11).
- **At or above 1.0, the mean/mode fill:** under G, credit-g_60nan (1.0003), credit-g_80nan
  (1.0305) and kr-vs-kp_80nan (1.0104), the same three as in E32; under M, those three plus
  spambase_60nan (1.0195) and spambase_80nan (1.0272).
- **The one variant leaning M:** kr-vs-kp_80nan, +0.0017 on the fresh seeds and +0.0053 pooled,
  no verdict either way, as in E32 (+0.0088).
- **Masked population (secondary):** G better on 8 variants, M better on none.
- **Time.** Planned about 11.3 h of wall time, ending near 12:05 GMT-3; it took 11 h 21 min and
  ended at 12:05 GMT-3, 1.00x the plan. A G cell and an M cell cost the same (441.5 and 430.6
  minutes summed over the variants per seed).

**What follows.** Nothing to decide: the default ADR 0014 adopted is confirmed. The open threads
are the ones the evidence leaves:
- **The three variants at the mean/mode fill** (credit-g_60nan and _80nan, kr-vs-kp_80nan) are
  where the model still adds nothing over the simplest fill; the token shape was not their
  problem.
- **Pre-training still shows gaps as `[NULL]`**; aligning it too is the natural next study in
  this line.
- **The calibration and checkpoint ideas** from 2026-10-02 would now be studied on top of G.
