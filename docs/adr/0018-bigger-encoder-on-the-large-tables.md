# Use a bigger encoder for imputation on pendigits_20nan and letter_20nan

## Status

Accepted (2026-10-07, 09:15 GMT-3). The user decided this after studies
[imputation-architecture/01](../tickets/imputation-architecture/01-architecture-screening.md) (E37)
and [imputation-architecture/02](../tickets/imputation-architecture/02-bigger-encoder-by-table-size.md)
(E38), which close task T03. Before E38 ran, the user chose a size-dependent default as the outcome
of a confirmed hypothesis. It changes no code and no global default: two promoted per-variant files
(ADR 0005) carry the bigger encoder.

## Context

The imputation encoder has 2 layers, a feed-forward width of 32 against a model width of 128, and
no LayerNorm after its last layer (the option `ENCODER_FINAL_NORM`, default `none`, came with
E37). E37
screened a deeper, a wider and a normed encoder, and the three together (ALL: `LAYERS` 4,
`DIM_FEED` 256, `ENCODER_FINAL_NORM layer`) on five variants. ALL was better on 3 and worse on none,
but its gain sat in the largest table: pendigits −4.0%, while the smallest, credit-g_20nan, was
+1.5%.

E38 tested ALL on all 21 variants on fresh seeds (101, 202, 303), grouped by the rows of their
table:

| group | variants | ALL better | ALL worse | mean change [95% interval] | cost a cell |
|---|---|---|---|---|---|
| large, ≥10,000 rows | 3 | 2 | 0 | −3.0% [−5.1%, −1.0%] | 1.84x |
| medium, 3,196–4,601 rows | 8 | 1 | 0 | −0.2% [−0.6%, +0.2%] | 1.95x |
| small, ~520–1,055 rows | 10 | 0 | 4 | +1.9% [+0.5%, +3.2%] | 1.77x |

Among the large tables, ALL is better on letter_20nan (−5.6%, 0.4590 → 0.4331) and
pendigits_20nan (−3.8%, 0.3331 → 0.3203, replicating E37's −4.0%), and no different on
electricity_20nan (+0.3%), the largest table but one with 8 columns.

## Decision

1. **pendigits_20nan and letter_20nan get the bigger encoder.** Each has a promoted
   `datasets/hiperparams/<base>/<variant>.imputation.json` holding today's imputation configuration
   (ADRs 0013 to 0015) with `LAYERS` 4, `DIM_FEED` 256 and `ENCODER_FINAL_NORM layer`. Every key is
   written out, as a promoted file must (ADR 0005, decision 5).
2. **No other variant changes.** electricity_20nan, though large, keeps today's encoder: the
   pre-stated rule would have included it, but it gained nothing at 1.77x the cost, and the user
   chose to leave it out. The medium and small tables keep today's encoder, since E38 found the
   bigger one neutral on the former and worse on the latter.
3. **Only the two studied variants.** pendigits and letter also exist at 40/60/80nan, outside
   ADR 0007's 21; those read no promoted file and keep today's encoder until checked.

## Verification

- **Default runs match E38's ALL.** Imputation runs of pendigits_20nan and letter_20nan at seed 101,
  through `main.py` with no flag, equal E38's ALL cells of the same seed to the last digit on every
  model metric (11 each) and on the four ADR 0007 baselines; both record `LAYERS` 4, `DIM_FEED` 256
  and `ENCODER_FINAL_NORM layer`.
- **Tests:** both regression fixtures and the unit suite (346 tests, mypy included) pass; no code
  changed.

## Consequences

- **Cost:** a run of either variant costs about 1.9x what it did (pendigits about 73 against 38
  minutes a cell, letter 111 against 59, with four trainers sharing the GPU).
- **Comparisons across time:** a pendigits_20nan or letter_20nan run after this decision trains a
  bigger encoder than one before it. Filter on the logged params `LAYERS` and `DIM_FEED`, the tag
  `encoder_final_norm` and the param `config_source` (the promoted file's path).
- **Re-running an older arm** on either variant must now pin the old encoder too (`LAYERS` 2,
  `DIM_FEED` 32, `ENCODER_FINAL_NORM none`), beside ADR 0013 to 0015's list; the commit before this
  decision has no promoted file for either.
- **An Optuna `--promote_best` on either variant** would overwrite its file with the study's
  winner, as for any promoted variant.
- **Where depth pays is not only a matter of rows:** electricity, the largest table, did not gain.
  A rule by size beyond these two variants would need its own study.
