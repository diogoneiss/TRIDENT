# ADR 0036: A large batch and a longer decode on pendigits and letter at 20 and 40nan

- **Status:** **proposed 2026-10-31** for the user's decision, from E104's Outcome
  ([imputation-protocol/41](../tickets/imputation-protocol/41-large-batch-rule.md)), whose ticket
  fixed this proposal as the step after a replication. Nothing is implemented until it is accepted.
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
| pendigits_60nan | −1.19% better | −1.18% better | | |
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

## Decision (proposed)

1. **pendigits_20nan, pendigits_40nan, letter_20nan and letter_40nan train with BATCH 1024,
   LR_PRE and LR_DECODE four times their resolved values (0.00136 and 0.004), and EPOCHS_DECODE
   600.**
   - Through promoted per-variant files, as ADR 0018: the two 20nan files gain the four keys and
     keep the bigger encoder; the two 40nan variants get new
     `datasets/hiperparams/<base>/<variant>.imputation.json` files holding today's complete
     imputation configuration with the four keys. No code changes and no global default moves.
2. **No other variant changes.** pendigits_60nan was better twice (−1.2%, E101 and E102) but sits
   outside E104's confirmed scope; letter_60nan was equivalent; both 80nan variants are a little
   worse; electricity gains nothing; the other tables leave an epoch too few steps at 1024.

## Verification (on acceptance)

- A default run of each of the four variants at one of E104's seeds equals E104's P600 cell of that
  seed on every model metric, and records the four keys and the promoted file as `config_source`.
- Both regression fixtures and the unit suite pass.

## Considered options

- **Leave every table at one batch.** The gain is replicated on fresh seeds at 1.5–5.9% and the
  cells cost no more, so only simplicity argues for keeping these four as they are.
- **Include pendigits_60nan.** Better on 6 seeds over two studies (−1.2% both), but its scope was
  not part of E104's confirmation; adding it is the user's call, with or without another check.
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
