# ADR 0036: A large batch and a longer decode on pendigits and letter at 20 and 40nan

- **Status:** **accepted 2026-10-31** by the user ("aprovo, Extenda"), as proposed: the four
  variants, pendigits_60nan left out. Proposed the same day from E104's Outcome
  ([imputation-protocol/41](../tickets/imputation-protocol/41-large-batch-rule.md)), whose ticket
  fixed this proposal as the step after a replication. **Amended 2026-11-02 00:09 GMT-3: pendigits_60nan joins** (E109, [44](../tickets/imputation-protocol/44-large-batch-pendigits-60nan.md), supported; the user: "sim, incluir"). Implemented and verified 2026-10-31 18:23 GMT-3 (below).
- **Builds on:** ADR 0005 (promoted per-variant files), ADR 0013 (the imputation defaults), ADR 0018
  (the bigger encoder on pendigits_20nan and letter_20nan, kept here).

## Context

- **Every table trains at one batch, 256,** with LR_PRE 0.00034 and LR_DECODE 0.001 and a 450-epoch
  decode (ADR 0013). E28 found batch 1024 with the learning rates x4 helping pendigits only, where an
  epoch kept enough steps.
- **Four studies on today's defaults** compared P600 (BATCH 1024, LR_PRE and LR_DECODE x4,
  EPOCHS_DECODE 600, over each variant's resolved configuration) with the default, on the induced
  `impute_score`, relative, 15 fold pairs a variant, with the project's rule and the 1% SESOI:

| variant | E101 (42/7/13) | E102 (7878/8989/9090) | E103 (42/7/13) | E104 (1357/2468/3579) |
|---|---|---|---|---|
| pendigits_20nan | −5.37% better | −5.84% better | | −5.93% better |
| pendigits_40nan | −4.27% better | −4.21% better | | −4.10% better |
| pendigits_60nan | −1.19% better | −1.18% better | | (E109, 2121/3232/4343: −1.34% better) |
| pendigits_80nan | +0.16% equivalent | +0.42% worse | | |
| letter_20nan | | | −2.90% better | −3.25% better |
| letter_40nan | | | −1.46% better | −1.54% better |
| letter_60nan | | | −0.13% equivalent | |
| letter_80nan | | | +0.21% worse | |
| electricity, 20–80nan | | | better on none | |

- **The scope "pendigits and letter at 20 and 40nan" was chosen after E101–E103,** so E104 tested
  it on seeds none of them used, with its rule fixed first: replicated, better on 4 of 4, every
  seed below zero ([41](../tickets/imputation-protocol/41-large-batch-rule.md)).
- **Cost (gorgona8, two queues sharing the GPU, E101/E103/E104 logs):** no dearer, and cheaper at
  40nan. pendigits_40nan 20–23 against 28–30 minutes a cell, letter_40nan 30–39 against 52–54. At
  20nan, with ADR 0018's bigger encoder, about the same: pendigits 29 against 29–33, letter 36–55
  against 51–63.

## Decision

1. **pendigits_20nan, pendigits_40nan, letter_20nan and letter_40nan train with BATCH 1024,
   LR_PRE and LR_DECODE four times their resolved values (0.00136 and 0.004), and EPOCHS_DECODE
   600.**
   - Through promoted per-variant files, as ADR 0018: the two 20nan files gain the four keys and
     keep the bigger encoder; the two 40nan variants get new
     `datasets/hiperparams/<base>/<variant>.imputation.json` files holding today's complete
     imputation configuration with the four keys. No code changes and no global default moves.
2. **pendigits_60nan too** (amended 2026-11-02 00:09 GMT-3): better in E101 and E102 (−1.2%), then on fresh seeds in
   E109 (−1.34% [−1.66%, −1.02%], every seed below zero). Its new promoted file carries the same four
   keys.
3. **No other variant changes.** letter_60nan was equivalent; both 80nan variants are a little
   worse; electricity gains nothing; the other tables leave an epoch too few steps at 1024.

## Verification (2026-10-31 18:23 GMT-3)

- **The files:** `datasets/hiperparams/pendigits/pendigits_{20,40}nan.imputation.json` and
  `datasets/hiperparams/letter/letter_{20,40}nan.imputation.json`. Each variant's resolved
  configuration now equals E104's P600 override and differs from the old default only in BATCH,
  LR_PRE, LR_DECODE and EPOCHS_DECODE.
- **Default runs match E104's P600:** a plain imputation run of each variant at seed 1357 on
  gorgona8, with no override and MLflow off, equals E104's P600 cell of that seed and machine on
  every numeric column of its fold metrics, timings aside (68 columns on the 20nan variants, 75 on
  the 40nan ones, none differing).
- **Tests:** the unit suite with the typing gate (459) and both regression fixtures pass.
- **pendigits_60nan (amendment):** a plain default run at seed 2121 on medusa1 equals E109's P600
  cell of that seed and machine on all 75 numeric fold-metric columns, timings aside.

## Considered options

- **Leave every table at one batch.** The gain is replicated on fresh seeds at 1.5–5.9% and the
  cells cost no more, so only simplicity argues for keeping these four as they are.
- **Include pendigits_60nan.** Better on 6 seeds over two studies (−1.2% both), outside E104's
  confirmation; checked on fresh seeds by E109 and added by the amendment.
- **A rule by steps an epoch or by table size.** electricity, the table with the most steps at 1024,
  gained nothing, so no such rule is supported.

## Consequences

- **Comparisons across time:** a run of the four variants after this decision trains with a larger
  batch, higher rates and a longer decode than before. Filter on the logged params `BATCH`,
  `LR_PRE`, `LR_DECODE`, `EPOCHS_DECODE` and `config_source`.
- **Re-running an older arm** on them must pin the old values (BATCH 256, LR_PRE 0.00034, LR_DECODE
  0.001, EPOCHS_DECODE 450).
- **Reports:** the thesis tables that include these variants change for runs after the decision;
  older runs keep their numbers.
