# H for claims that must transfer; K stays the default as the oracle bound

## Status

Accepted. The plan of 2026-10-17 proposed it, and the user accepted every recommendation on
2026-10-16 at about 22:45 GMT-3: option B of this ADR, D9 and D10. It was to stay "proposed" until
E43 chose between M and H. E43 (ticket
[imputation-protocol/03](../tickets/imputation-protocol/03-deployable-checkpoint.md)) chose H on
2026-10-17, and E44 confirmed the readouts.

It amends ADR [0015](0015-decode-checkpoint-chosen-on-the-validation-gaps.md): K stays the
default, and this ADR adds the readout that claims needing transfer are read on. ADR
[0027](0027-calibrated-imputation-output-at-40-percent-missing.md) later made KC the default output
at 40% missing and above, and inherits the same limit.

## Context

K, the checkpoint ADR 0015 keeps, is chosen on the validation rows' own gaps scored against the
complete table. On the benchmark that table exists, because every `_XXnan` variant was cut from a
`_00nan` one. On real data it does not: the gaps' true values are unknown. So K, and every number
read on K, is an oracle bound. The review of 2026-10-16 named this threat 3.

Two checkpoints need no complete table:

- **H:** the epoch of the lowest loss on the fixed validation mask, every run's criterion before
  ADR 0015;
- **M:** the epoch whose fixed validation mask scores best by `impute_score` (E43,
  `--decode_checkpoint masked`).

**What the studies found:**

- **E43** (9 variants at 60nan and above, seeds 111/222/333):
  - M equals H on numerical tables by construction, and H is the simpler criterion;
  - K beats M on 4 of 9;
  - the 80nan wins over the mean/mode fill are K's only.
  - Reading 3: H is the deployable headline.
- **E44** (the confirmation set, 15 variants): H against the bar is better on 11, no difference on
  2, equivalent on 1 and worse on 1. K − H is better on 5 and worse on none. The oracle's advantage
  on new data is small.
- **E45 and E46** (MNAR): K − H does not grow under value-dependent missingness on any pair or
  slope.

## Decision

1. **K stays the default checkpoint** (ADR 0015). Benchmark tables and the headline of every default
   run are on K, read through `scripts/experiments/report_common.py`.
2. **Claims that must transfer are read on H,** with K beside it as the oracle bound. They include:
   - statements about data without a complete table;
   - deployment;
   - MNAR.
   - A study that makes such a claim logs H on the same trajectory (`--score_checkpoints`, or
     `--decode_checkpoint loss` with `--score_induced_checkpoint`).
3. **`--decode_checkpoint masked` stays available** (E43), off by default. It is not the deployable
   headline: on numerical tables it equals H.
4. **`report_common` reads each readout by the run's `decode_checkpoint` tag.** It refuses rather
   than guess. Since ADR 0027 it reads K, H and M uncalibrated, and KC separately.
5. **Calibration (D10, revised):**
   - KC, fitted on the validation gaps, is an oracle readout like K.
   - Its deployable analogue, a calibration fitted on the validation mask's cells and read with H,
     is tested by E47 (ticket
     [imputation-calibration/02](../tickets/imputation-calibration/02-deployable-calibration.md)).
     Until E47 is read, no deployable calibration is claimed.

## Consequences

- **The thesis states both numbers where they differ.** On the confirmation set they mostly agree:
  K is better than H on 5 of 15, never worse.
- **The 80nan results are oracle-only** on the development set (E43). They are reported as such.
- **A future default change toward H** (option C, rejected for now) would need its own study, an
  ADR and the usual gate. It would move every benchmark headline.
