# ReMasker on letter, the paper's rule (E159, T53; pre-registered 2026-11-07 21:43 GMT-3)

This ticket was written and committed before the study's first scored cell.

- **Why:** E157 and E158 left letter out for cost (about 25 process-hours at the paper's batch of 64); E158's accepted
  sentence covers 9 numerical confirmation variants. letter, the largest numerical table of the confirmation set, is
  where a masked autoencoder has the most rows to learn from (ReMasker beat the strong tier only on pendigits, the
  other large table).
- **The user's go:** "De acordo, va" (2026-11-07), on the proposal "A: ReMasker no letter 40/60/80, regra do artigo",
  launched together with B (E160) to keep both machines' GPUs busy.
- **Disclosure:** E44's letter cells (K, the strong tier, the bar) are on the record; ReMasker has no score on letter.

## Design (E158's in all else)

- **Arm:** ReMasker, `remask="paper"`, untuned, E158's settings (ticket 56).
- **Scope:** `letter_{40,60,80}nan` at 707/808/909, 9 pairs, 45 fits; K the confirmation cells' own.
- **Machines (ADR 0035):** medusa1 runs `letter_60nan` and `letter_80nan` (6 pairs, six lanes, three on each GPU,
  beside another user's job that holds 27 GB of each); gorgona8 runs `letter_40nan` (3 pairs, three lanes, beside
  E160). Outs: `/scratch/diogoneiss/trident-2026-11-07-remasker-letter` (medusa1) and
  `/scratch2/diogoneiss/trident-2026-11-07-remasker-letter` (gorgona8); medusa1's CSVs are copied over with their
  SHA-256 checked before the read.
- **Read:** once, with all 9 pairs in:
  `remasker_report.py --paper --variants letter_40nan,letter_60nan,letter_80nan --remasker <letter csv> --study E159`,
  then the 12 variants together (E158's nine and these three), from both CSV directories.
  Expected end: about 01:00–03:00 GMT-3 on 2026-11-08.

## Measures and reading (fixed now)

- E158's measures on the three letter variants: K − ReMasker (primary), ReMasker − strong / bar / tuned bar, K − best
  of all, fit time.
- **Strong-tier rule over the 12 numerical variants** (E158's nine plus letter's three): better than the strong
  tier's best on at least 4 of 12 (a third, E48's) → an ADR amending 0021 is proposed for the user's decision;
  otherwise an external deep baseline only. E158 alone gave 2 of 9.
- **The accepted sentence** ("TRIDENT supera o ReMasker nas 9 variantes numéricas..."): if K is better on all three
  letter variants, the sentence's count becomes 12; otherwise the letter variants are stated beside it. The wording
  stays the user's.
- No default changes, no bar change.
