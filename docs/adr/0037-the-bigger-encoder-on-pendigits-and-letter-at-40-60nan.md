# ADR 0037: The bigger encoder on pendigits and letter at 40 and 60nan

- **Status:** **accepted 2026-11-03** by the user ("sim, de acordo"), from E113 and E116
  ([imputation-protocol/46](../tickets/imputation-protocol/46-bigger-encoder-large-tables.md),
  [47](../tickets/imputation-protocol/47-bigger-encoder-confirmation.md)); verified 10:43 GMT-3.
- **Builds on:** ADR 0018 (the bigger encoder on pendigits_20nan and letter_20nan), ADR 0036 (P600 on
  pendigits at 20–60nan and letter at 20–40nan), ADR 0005 (promoted per-variant files).

## Context

- **ADR 0018's encoder** (LAYERS 4, DIM_FEED 256, ENCODER_FINAL_NORM layer, against 2 / 32 / none) runs only
  on pendigits_20nan and letter_20nan. E59 saw it help those tables at 40nan too (about −4% on K), less at
  60nan, nothing at 80nan, and the user judged it not worth the cost then (2026-10-22).
- **On today's defaults** (ADR 0036's P600 included where it applies), the induced error, BIG − default,
  15 fold pairs a variant, the project's rule:

| variant | E113 (42/7/13) | E116 (6161/7171/8181) |
|---|---|---|
| pendigits_40nan | −6.04% better | −6.26% better |
| pendigits_60nan | −3.14% better | −3.09% better |
| letter_40nan | −5.47% better | −5.45% better |
| letter_60nan | −1.12% better | −1.23% better |

- **Cost (gorgona8, E113, two queues sharing the GPU):** pendigits 32–33 against 19–22 minutes a cell,
  letter 54–63 against 32–50 (about 1.3–1.7x).

## Decision

1. **pendigits_40nan, pendigits_60nan, letter_40nan and letter_60nan train with ADR 0018's encoder:**
   LAYERS 4, DIM_FEED 256, ENCODER_FINAL_NORM layer.
   - Through their promoted files: pendigits_40nan, pendigits_60nan and letter_40nan already have one
     (ADR 0036), which gains the three keys; letter_60nan gets a new file with today's complete
     configuration and the three keys. No code change.
2. **No other variant changes.** At 80nan E117 ([48](../tickets/imputation-protocol/48-bigger-encoder-80nan.md))
   was mixed and under the SESOI (pendigits −0.39%, letter +0.37%); electricity gained nothing from the
   bigger encoder in E38.

## Verification (2026-11-03 10:43 GMT-3)

- **The files** (c349aa3): `datasets/hiperparams/pendigits/pendigits_{40,60}nan.imputation.json` and
  `datasets/hiperparams/letter/letter_{40,60}nan.imputation.json` resolve to ADR 0018's encoder over today's
  default (ADR 0036's P600 included on pendigits_40/60nan and letter_40nan).
- **Default runs match E116's BIG:** a plain imputation run of each of the four variants at seed 7171 on
  medusa1, with no override and MLflow off, equals E116's BIG cell of that seed and machine on all 75
  numeric fold-metric columns, timings aside, none differing.
- **Default run matches the pre-computed thesis cell:** a plain run of pendigits_40nan at seed 707 on
  gorgona8 equals its cell under tag `adr37-thesis-pre-2026-11-03` on all 75 columns, none differing.
- **Tests:** the unit suite with the typing gate (461) and both regression fixtures pass.
- **The thesis tables** now read the 9 pre-computed cells (`report_common.ADR37`, `ADR37_EXPERIMENT`).
  Against TabPFN on the confirmation set (`tabpfn_report.py`, before → after), K − TabPFN:
  pendigits_40nan −0.0279 → −0.0489, pendigits_60nan −0.0244 → −0.0356, letter_40nan −0.0431 → −0.0661;
  no other variant moved and no call changed.

## Consequences

- **Comparisons across time:** runs of these four variants after the decision train a bigger encoder;
  filter on `LAYERS`, `DIM_FEED`, the tag `encoder_final_norm` and `config_source`. Older arms re-run on
  them must pin LAYERS 2, DIM_FEED 32, ENCODER_FINAL_NORM none.
- **The thesis tables** read pendigits_40nan, pendigits_60nan and letter_40nan in the confirmation set
  (E44, 707/808/909). Those 9 cells were pre-computed on gorgona8 before the decision (2026-11-03, tag
  `adr37-thesis-pre-2026-11-03`, ADR 0018's encoder as an override over today's default); a default run was
  checked identical to one of them (Verification), and the tables read them, as E108 did for ADR 0036.
- **Cost:** about 1.3–1.7x a cell on these variants.
