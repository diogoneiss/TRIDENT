# The bigger encoder at 80nan on pendigits and letter (E117; pre-registered 2026-11-03 04:38 GMT-3)

This ticket was written and committed before the study's first cell. No cell has run.

- **Why:** ADR 0037 (proposed) stops at 60nan because 80nan was not tested on today's defaults (E59
  saw nothing at 80nan on the older ones). This settles the boundary before the user decides.
- **The user's go:** the night's mandate ("o que mais agendar para rodar na madrugada? voce tem ate as
  10h da manha para executar coisas", 2026-11-03, before 00:38 GMT-3), with both machines free at 04:37.
- **No code change** beyond `large_batch.py --study e117`.

## Design, measures and rule (fixed before the first cell)

- pendigits_80nan and letter_80nan × seeds 42, 7, 13 × arms M (today's default) and BIG (ADR 0018's
  encoder), five folds, MLflow off; seed 42 on gorgona8, 7 and 13 on medusa1. 12 cells.
- BIG − M on the induced `impute_score`, relative, 15 fold pairs, 1% SESOI. **Supported** if better on both;
  **refuted** if better on neither; **mixed** otherwise.
- **What follows:** supported → the user may extend ADR 0037 to 80nan (a fresh-seed check first, as for
  every scope added after a result).

## Outcome

(open)
