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

**Written 2026-11-03 06:09 GMT-3. Mixed: BIG is better on pendigits_80nan (−0.39%) and worse on letter_80nan (+0.37%)**,
both under the 1% SESOI, every seed agreeing in each. At 80nan the bigger encoder neither helps nor
costs anything that matters; ADR 0037's proposal rightly stops at 60nan.

- **Complete:** all 12 cells, 04:38–06:09 GMT-3, no failure.

### The tally, unedited (`large_batch.py --study e117 --tally`, 2026-11-03 06:09 GMT-3)

```
## BIG − M on impute/induced/impute_score, relative to M; 15 fold pairs a variant (fold-pair t, seed agreement, SESOI 0.01)
- pendigits_80nan: -0.39% [-0.62%, -0.15%] seeds -0.22%, -0.54%, -0.39% -> better; masked +0.26%
- letter_80nan: +0.37% [+0.26%, +0.48%] seeds +0.41%, +0.30%, +0.40% -> worse; masked +0.31%

Primary (E117, BIG − M at 80nan): better 1 / no difference 0 / worse 1 (fold-pair t, seed agreement) of 2 -> mixed
```
