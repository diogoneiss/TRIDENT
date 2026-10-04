# Does the gap-chosen checkpoint hold on E32's seeds? (pre-registered 2026-10-04)

This ticket was written and committed before the first run, at 00:23 GMT-3 on 2026-10-04 (commit
`64f6d54`; the queues started 00:23:12 GMT-3, so about 06:15 GMT-3 is the expected end). The user adopted E34's K as the imputation default (ADR 0015)
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

## Outcome (2026-10-04, 06:30 GMT-3)

All 63 cells ran (00:23 to 06:14 GMT-3), none failed, none missing. The integrity check passed:
every cell's H, induced and masked, equals E32's arm-G cell to the last digit. Numbers from
`scripts/experiments/selection_replication_report.py` (kept with `--tally` in the state
directory). Lower is better; a negative difference favours K. ★ marks E34's three variants, on
which the reading is fixed. Induced cells, 15 within-run fold pairs per variant; the pooled
column adds E34's seeds (30 pairs, all six seeds agreeing); the epochs are fold means.

| variant | H | K | C | KC | best baseline | K − H (seeds 42, 7, 13) | verdict | pooled with E34 | epoch, loss → gaps |
|---|---|---|---|---|---|---|---|---|---|
| `credit-g_20nan` | 0.8873 | 0.8818 | 0.8865 | 0.8793 | 0.9180 | −0.0055 [−0.0174, +0.0063] (+0.0034, −0.0079, −0.0121) | none | −0.0061, none | 73 → 136 |
| `credit-g_40nan` | 0.9632 | 0.9556 | 0.9526 | 0.9501 | 0.9638 | −0.0076 [−0.0210, +0.0058] (+0.0005, −0.0186, −0.0047) | none | −0.0074, none | 42 → 63 |
| `credit-g_60nan` | 1.0030 | 0.9922 | 0.9917 | 0.9808 | 1.0000 | −0.0108 [−0.0241, +0.0025] (−0.0183, −0.0092, −0.0048) | none | −0.0090, none | 31 → 43 |
| `credit-g_80nan` ★ | 1.0157 | 1.0079 | 1.0070 | 1.0023 | 1.0000 | −0.0078 [−0.0172, +0.0016] (−0.0036, −0.0161, −0.0036) | none | −0.0139, K better | 11 → 14 |
| `kr-vs-kp_20nan` | 0.5934 | 0.5863 | 0.5961 | 0.5977 | 0.6119 | −0.0071 [−0.0171, +0.0029] (−0.0049, −0.0039, −0.0126) | none | −0.0076, none | 263 → 249 |
| `kr-vs-kp_40nan` | 0.7608 | 0.7582 | 0.7521 | 0.7533 | 0.7725 | −0.0026 [−0.0097, +0.0045] (−0.0009, −0.0010, −0.0059) | none | +0.0000, none | 104 → 130 |
| `kr-vs-kp_60nan` | 0.8684 | 0.8649 | 0.8587 | 0.8617 | 0.9030 | −0.0035 [−0.0120, +0.0051] (−0.0026, −0.0034, −0.0044) | none | −0.0020, none | 90 → 84 |
| `kr-vs-kp_80nan` ★ | 1.0100 | 0.9759 | 0.9751 | 0.9692 | 1.0000 | −0.0341 [−0.0530, −0.0151] (−0.0406, −0.0496, −0.0120) | K better | −0.0315, K better | 36 → 22 |
| `spambase_20nan` | 0.8504 | 0.8481 | 0.8559 | 0.8514 | 0.8664 | −0.0023 [−0.0094, +0.0047] (+0.0002, −0.0015, −0.0057) | none | +0.0006, none | 289 → 295 |
| `spambase_40nan` | 0.9065 | 0.9030 | 0.9048 | 0.9029 | 0.9371 | −0.0036 [−0.0057, −0.0014] (−0.0038, −0.0047, −0.0021) | K better | −0.0016, none | 241 → 260 |
| `spambase_60nan` | 0.9449 | 0.9422 | 0.9413 | 0.9393 | 0.9845 | −0.0027 [−0.0045, −0.0008] (−0.0009, −0.0030, −0.0041) | K better | −0.0030, K better | 100 → 184 |
| `spambase_80nan` ★ | 0.9971 | 0.9871 | 0.9840 | 0.9813 | 1.0000 | −0.0100 [−0.0157, −0.0043] (−0.0069, −0.0136, −0.0095) | K better | −0.0100, K better | 75 → 55 |
| `vehicle_20nan` | 0.4218 | 0.4178 | 0.4216 | 0.4175 | 0.4731 | −0.0040 [−0.0120, +0.0041] (−0.0011, −0.0084, −0.0023) | none | −0.0037, none | 275 → 309 |
| `vehicle_60nan` | 0.6458 | 0.6449 | 0.6347 | 0.6353 | 0.7006 | −0.0009 [−0.0066, +0.0047] (−0.0043, −0.0009, +0.0024) | none | −0.0099, none | 231 → 225 |
| `biodeg_20nan` | 0.5954 | 0.5865 | 0.5962 | 0.5871 | 0.6356 | −0.0089 [−0.0203, +0.0025] (−0.0083, −0.0036, −0.0148) | none | −0.0091, none | 256 → 285 |
| `biodeg_60nan` | 0.7684 | 0.7605 | 0.7622 | 0.7563 | 0.8658 | −0.0079 [−0.0141, −0.0016] (−0.0123, −0.0087, −0.0027) | K better | −0.0115, K better | 239 → 299 |
| `kc2_20nan` | 0.5775 | 0.5414 | 0.5811 | 0.5446 | 0.5665 | −0.0361 [−0.1065, +0.0343] (−0.0031, −0.0999, −0.0052) | none | −0.0199, none | 161 → 195 |
| `kc2_60nan` | 0.6168 | 0.5723 | 0.5965 | 0.5690 | 0.7129 | −0.0445 [−0.0793, −0.0096] (−0.0506, −0.0306, −0.0523) | K better | −0.0307, K better | 184 → 227 |
| `pendigits_20nan` | 0.3328 | 0.3324 | 0.3312 | 0.3309 | 0.3500 | −0.0003 [−0.0012, +0.0005] (−0.0007, +0.0001, −0.0004) | none | −0.0004, none | 401 → 406 |
| `letter_20nan` | 0.4598 | 0.4593 | 0.4580 | 0.4575 | 0.4729 | −0.0006 [−0.0015, +0.0003] (−0.0005, −0.0011, −0.0001) | none | −0.0005, none | 406 → 421 |
| `electricity_20nan` | 0.6201 | 0.6179 | 0.6199 | 0.6182 | 0.6058 | −0.0021 [−0.0100, +0.0057] (+0.0013, −0.0015, −0.0062) | none | −0.0008, none | 379 → 345 |

**Tally.** K − H on these seeds: K better on 6 variants (kr-vs-kp_80nan −0.034, spambase_40nan
−0.004, spambase_60nan −0.003, spambase_80nan −0.010, biodeg_60nan −0.008, kc2_60nan −0.045), H
better on none, no detectable difference on 15. On E34's three: K better on kr-vs-kp_80nan and
spambase_80nan; credit-g_80nan leans K on every seed (−0.008) without a verdict.

**Reading, by the rule fixed in advance: replicated.** No variant shows "H better", and K is
better on 2 of the 3 variants where E34 found it. ADR 0015 stands on six seeds.

**Secondary measures.**
- **Pooled with E34 (six seeds):** K better on 6 variants (credit-g_80nan, kr-vs-kp_80nan,
  spambase_60nan, spambase_80nan, biodeg_60nan, kc2_60nan), H better on none. The six-seed mean
  difference is negative on 19 of the 21 variants, zero on kr-vs-kp_40nan (+0.0000) and positive
  only on spambase_20nan (+0.0006, no verdict).
- **By missing level:** K − H −0.007 (20nan), −0.005 (40), −0.012 (60), −0.017 (80); E34 had
  −0.003, −0.001, −0.010, −0.020.
- **Means over the 21 variants:** H 0.7542, K 0.7446, C 0.7480, KC 0.7422 (E34: H 0.7519, K
  0.7446, C 0.7453, KC 0.7412). Below the best baseline: H on 16 variants, K on 19, C on 18, KC on
  19.
- **The calibration fails again on the same variant:** C − H is "H better" on spambase_20nan
  (+0.0055), as in E34 (+0.0053), and "C better" on 13 variants on both seed sets. KC − K is "K
  better" on kr-vs-kp_20nan and spambase_20nan. A calibration that leaves the 20nan tables alone,
  where the model is already well calibrated (mean α 0.85 to 0.98 there), is the study this
  points to.
- **Epochs:** as in E34, at 80nan the gap criterion keeps an earlier epoch on kr-vs-kp (36 → 22)
  and spambase (75 → 55), and on most other variants a later one.
- **Time.** Planned about 5.8 h of wall time, ending near 06:15 GMT-3; it took 5 h 51 min and
  ended at 06:14, 1.01x the plan.
