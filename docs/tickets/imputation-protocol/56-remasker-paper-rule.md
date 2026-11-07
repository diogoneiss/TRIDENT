# ReMasker with the paper's re-mask count (E158, T53; pre-registered 2026-11-07 08:15 GMT-3)

This ticket was written and committed before the study's first scored cell.

- **Why:** [E157](55-remasker.md) found at its read that the reference code's re-mask count, `int(L · 0.5)` tokens kept
  per row with gaps sorted last, re-masks almost no observed cell once a row has fewer observed cells than half its
  columns: 0.0–0.1% of observed cells at 80% missing, 1–4% at 60%, about 20% at 40%. Its 60–80nan readings measure an arm
  with almost no reconstruction signal. The paper counts the ratio over observed values (§4.4: "The masking ratio
  controls the number of re-masked values (after excluding missing values)").
- **The user's go:** "Rode de novo com a regra do artigo para vermos" (2026-11-07), with "Você pode usar um pouco da
  medusa1 para acelerar".
- **Disclosure, known before the first cell:** E157's whole printout (ticket 55), including the strong tier's, the
  bar's and K's values on these folds, which this study shares. The paper-rule arm itself has no score on any study
  table.

## The change (`src/training/remasker.py`, `remask="paper"`; the default stays E157's `reference`)

- A row with `n_obs` observed cells keeps `int(n_obs · 0.5)` of them visible, drawn at random, and re-masks the rest.
  On a complete row this equals the reference's count. Everything else is E157's arm unchanged: settings, departures,
  seed, scaling, fill.
- The driver selects it with `REMASKER_REMASK=paper` and records it in every CSV row (`baseline/remasker/remask_paper`).
  The report refuses a CSV whose rule differs from the one it reads (`remasker_report.py --paper`).
- Unit tests: the paper rule never shows a gap, keeps `n_obs // 2` per row, and matches the reference's count on
  complete rows (`tests/unit/test_remasker.py`; the unit suite passed, 509).
- **Pre-launch toy check** (synthetic, not a study table: 2,000 rows, six columns on one latent and two independent, MCAR,
  60 epochs, RMSE over the gaps):

| missing | mean fill | reference rule | paper rule |
|---|---|---|---|
| 0.4 | 0.743 | 0.442 | 0.431 |
| 0.8 | 0.750 | 0.896 | 0.648 |

  At 80% the reference rule is worse than the mean fill, as in E157; the paper rule is not.

## Design (fixed now, E157's in all else)

- **Scope:** E157's 9 variants (`pendigits_{40,60,80}nan`, `vehicle_`, `biodeg_`, `kc2_{40,80}nan`) at seeds
  707/808/909, 27 pairs, 135 fits; K the confirmation cells' own.
- **Machines (ADR 0035, every arm of a pair on one machine):** medusa1 runs `pendigits_80nan` at all three seeds and
  `pendigits_60nan` s909, four lanes over its two GPUs (they are shared with another user's jobs; about 1.6 GB a
  lane); gorgona8 runs the other 23 pairs in five lanes. The CSVs are pulled to gorgona8 for the read; nothing is written
  to any store. A pair's machine is in its lane log and in this ticket.
  - gorgona8 out: `/scratch2/diogoneiss/trident-2026-11-07-remasker-paper`; medusa1 out:
    `/scratch/diogoneiss/trident-2026-11-07-remasker-paper`.
- **Read:** once, with all 27 pairs in: `remasker_report.py --paper --remasker <csv> --against
  /scratch2/diogoneiss/trident-2026-11-06-remasker/csv`. Expected end about 13:00–15:00 GMT-3 on 2026-11-07. A failed
  pair is rerun on the same machine before the read.

## Measures and reading (E157's, fixed now)

1. **Primary:** K − ReMasker (paper rule) per variant, tallied.
2. ReMasker − the strong tier's best, − the bar, − the tuned bar; K − the best of all (ReMasker paper rule included,
   E157's arm not).
3. **Strong-tier rule:** better than the strong tier's best on at least 3 of 9 → an ADR amending 0021 is proposed for
   the user's decision; otherwise an external deep baseline only.
4. **Descriptive, decides nothing:** paper rule − reference rule per variant (E158 − E157); fit time.

**What it leads to:** E158 replaces E157 as the thesis's ReMasker comparison at every level, since it is the paper's
rule; E157 stays on the record as the reference code's. The thesis sentence is the user's call (asked again later, as
the user requested). No default changes, no bar change, no ADR unless the strong-tier rule fires.

## Limits, stated now

- medusa1's GPUs are RTX 5090s, gorgona8's an RTX 3090 Ti; fills differ per fold between GPUs (E86), so a pair's
  machine is fixed and recorded, and the four pendigits pairs on medusa1 are not compared with gorgona8's seed by seed.
- Untuned, as E157; TRIDENT's configuration was tuned on the development set.
