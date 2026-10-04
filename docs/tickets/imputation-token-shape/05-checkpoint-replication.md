# Does the gap-chosen checkpoint hold on E32's seeds? (pre-registered 2026-10-04)

This ticket was written and committed before the first run (the commit and the queues' start
time are in the Running notes). The user adopted E34's K as the imputation default (ADR 0015)
and chose to replicate it on other seeds before anything else.

## Why

E34 found the checkpoint chosen on the validation rows' own gaps (K) better than the one chosen
on the validation-mask loss (H) on 3 of 21 variants, all at 80nan, and worse on none, on seeds
101, 202 and 303. ADR 0015 adopted it on those three seeds alone. Seeds 42, 7 and 13 have the
same configuration's cells from E32 (arm G), which K's runs must reproduce at H, so the
replication costs one study and carries its own integrity check.

## Design (fixed before launch)

- **E34's cell on other seeds.** One run per cell under today's defaults with the checkpoint
  pinned to the loss (`--decode_checkpoint loss`), plus `--score_induced_checkpoint` and
  `--score_calibrated`: the same trajectory scored as H (the loss's epoch), K (the gaps' epoch,
  ADR 0015's default), C and KC, exactly as E34 read it.
- **Variants:** all 21 of ADR 0007. **Seeds:** 42, 7, 13. **Size:** 63 cells, tagged
  `experiment=selection-replication-2026-10-04`.
- **Integrity check:** H must equal E32's arm-G cells (`experiment=gap-token-2026-10-02`) to
  the last digit on every cell; one differing cell means the diagnostics leaked into training.
  A smoke run of credit-g_20nan at seed 42 matched on all 55 `impute/*` metrics; its diagnostic
  values were not read.
- **Running it:** four concurrent queues (`scripts/experiments/run_selection_replication_queue.sh
  1..4`), one trainer each, CPU threads capped at 4, cells assigned as in E34. E34 took 5 h
  51 min, so about 5.8 h is expected. No cell starts after 14:00 GMT-3 on 2026-10-04 (17:00 UTC);
  a cell not started by then is reported missing. Nothing is rerun or extended on its result; a
  crash is fixed, its cell rerun from the start, and the rerun noted here.

## Measures

- **Primary:** per variant, K − H on the induced `impute_score`, 15 within-run fold pairs,
  t-interval; a verdict only when the interval excludes zero and all three seeds agree in sign.
- **Reading, fixed now:**
  1. **Replicated:** no variant among the 21 shows "H better", and K is better on at least 2 of
     the 3 variants where E34 found it (credit-g_80nan, kr-vs-kp_80nan, spambase_80nan).
  2. **Safe, weaker:** no variant shows "H better", but K is better on fewer than 2 of those 3.
     ADR 0015 stands, since K was never worse; the effect is reported as smaller than E34's.
  3. **Not replicated:** any variant shows "H better". The report lists those variants and the
     user reconsiders ADR 0015; nothing is reverted automatically.
- **Secondary, descriptive (no decision rests on them):**
  - K − H pooled with E34 (six seeds, 30 pairs, all six agreeing);
  - C − H, KC − H, KC − K, KC − C on these seeds and pooled with E34: they inform whether a more
    conservative calibration is worth its own study, E34's having failed on spambase_20nan;
  - the two criteria's epochs; differences by missing level; means; best-baseline counts; time.
- **Analysis:** `scripts/experiments/selection_replication_report.py` (`--tally` prints the
  reading, the pooled counts and the integrity check).

## Known limits, stated in advance

- **Within-run pairs,** as in E34: narrow intervals, so a small effect can earn a verdict.
- **The same three variants are named before the data:** the 2-of-3 criterion was chosen from
  E34's result, which is the point of a replication.
- **Multiplicity:** 21 primary tests with the seed-agreement rule.
