# The bigger encoder on pendigits and letter at 40 and 60nan, on today's defaults (E113, T32; pre-registered 2026-11-02 17:22 GMT-3)

This ticket was written and committed before the study's first cell. No cell has run.

- **Why:** T32. E59 found ADR 0018's bigger encoder (LAYERS 4, DIM_FEED 256, ENCODER_FINAL_NORM layer)
  about −4% on K for pendigits and letter at 40nan, less at 60nan, nothing at 80nan; the user then
  judged it not worth the cost ("por enquanto nao vale a pena", 2026-10-22). Since then ADR 0036 gave
  pendigits_40/60nan and letter_40nan a large batch and a longer decode, which also made their cells
  cheaper. Does the bigger encoder still add on top of today's defaults?
- **The user's go:** "rode o t32 na gorgona entao" (2026-11-02, after 16:32 and before this ticket).
- **No code change** beyond `large_batch.py --study e113` (arm BIG: the three encoder keys over each
  variant's resolved configuration; checked to change only those keys).

## Design (fixed before the first cell)

- **Cells:** pendigits and letter at 40 and 60nan × seeds 42, 7, 13 × arms M (today's default) and BIG,
  five folds, MLflow off, all on gorgona8 (two queues). 24 cells. Outputs in
  `trident-2026-10-30-large-batch-e113/`.

## Measures and rule, fixed now (`large_batch.py --study e113 --tally`)

- BIG − M on the induced `impute_score`, relative, 15 fold pairs a variant, the project's rule, 1%
  SESOI.
- **Supported** if BIG is better on 2 or more of the 4 and worse on none; **refuted** if better on
  none; **mixed** otherwise.
- **What follows:** supported → extending ADR 0018 to those variants is the user's decision (with a
  fresh-seed check if the scope is chosen after the result).

## Outcome

**Written 2026-11-03 01:44 GMT-3. Supported: BIG is better on 4 of 4, every seed below zero** — pendigits_40nan −6.0%,
pendigits_60nan −3.1%, letter_40nan −5.5%, letter_60nan −1.1% on the induced error. On today's defaults
(ADR 0036's P600 included) the bigger encoder helps more than E59 saw over the older ones.

- **Complete:** all 24 cells on gorgona8, 2026-11-02 17:22 to 2026-11-03 01:43 GMT-3, no failure.
- **Descriptive:** the masked score moves the same way, by less (−4.0%, −0.8%, −3.7%, −0.1%).
- **What follows:** the scope was fixed before the result, but as for ADR 0036 a fresh-seed
  confirmation (E116) comes before an ADR for the user.

### The tally, unedited (`large_batch.py --study e113 --tally`, 2026-11-03 01:44 GMT-3)

```
## BIG − M on impute/induced/impute_score, relative to M; 15 fold pairs a variant (fold-pair t, seed agreement, SESOI 0.01)
- pendigits_40nan: -6.04% [-6.80%, -5.27%] seeds -5.88%, -6.49%, -5.74% -> better; masked -4.00%
- pendigits_60nan: -3.14% [-3.55%, -2.72%] seeds -3.27%, -3.11%, -3.04% -> better; masked -0.77%
- letter_40nan: -5.47% [-5.85%, -5.10%] seeds -5.64%, -5.41%, -5.36% -> better; masked -3.71%
- letter_60nan: -1.12% [-1.35%, -0.89%] seeds -0.99%, -1.07%, -1.30% -> better; masked -0.07%

Primary (E113, BIG − M): better 4 / no difference 0 / worse 0 (fold-pair t, seed agreement) of 4 -> supported
```
