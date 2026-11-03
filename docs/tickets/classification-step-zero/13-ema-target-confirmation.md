# The EMA target at 40 and 60nan on fresh seeds (E114; pre-registered 2026-11-02 22:37 GMT-3)

This ticket was written and committed before the study's first cell. No cell has run.

- **Why:** E112 ([12](12-ema-target.md)) was mixed: the EMA target (`PRETRAIN_OBJECTIVE embedding_ema`)
  was ahead of no pre-training at 40nan (+0.005) and 60nan (+0.009), every seed above zero, and
  equivalent at 00, 20 and 80nan. Those two levels were picked after the result, so they are checked
  here on seeds nothing has used.
- **The user's go:** "faça o 1" (2026-11-02, after 22:34 GMT-3 and before this ticket), on the option of
  a fresh-seed check at 40–60nan.
- **No model code change:** `cls_ema.py --study e114` and `STUDY=e114 run_cls_ema_queue.sh`.

## Design (fixed before the first cell)

- **Cells:** the 18 variants at 40 and 60nan × fresh seeds 3131, 4141, 5252 × arms E (EMA target) and B
  (no pre-training), E87's protocol otherwise (the old sweep's configuration, `cosine_legacy`, ten
  folds). Both arms on medusa1, eight queues; MLflow tag `experiment=cls-ema-e114-2026-11-02`. 108 cells.

## Measures and rule, fixed now (`cls_ema.py --study e114 --tally`, on medusa1)

- E − B per level, test macro-F1, paired by fold and seed, pooled over the 9 tables (270 pairs), the
  project's rule, SESOI 0.01.
- **Replicated** if E is higher at both 40 and 60nan; **not replicated** if higher at neither or B is
  higher at either; **partial** otherwise.
- **What follows:** replicated → an ADR proposal for the user on the EMA target as classification's
  pre-training, stating that the gain is below the SESOI. Otherwise T13 closes for classification.

## Outcome

**Written 2026-11-03 02:14 GMT-3. Partial: E is higher than B at 60nan (+0.008 [+0.003, +0.013]) and has no verdict at
40nan (+0.009 [+0.005, +0.013], one seed at −0.0005).** The rule asked for both levels.

- **Complete:** all 108 cells on medusa1 (2026-11-02 22:37 to 2026-11-03 02:13 GMT-3), no failure.
- **Reading:** the direction repeats at both levels (+0.009 and +0.008, close to E112's +0.005 and
  +0.009), but the gain stays under the 0.01 SESOI and misses seed agreement at 40nan by one seed.
- **What follows, as fixed above:** not replicated in full, so T13 closes for classification: the EMA
  target moves classification pre-training a little at 40–60nan, below what the project counts as
  relevant. No ADR. The flag stays, off by default.

### The tally, unedited (`cls_ema.py --study e114 --tally`, 2026-11-03 02:14 GMT-3)

```
- 40nan: E − B +0.0086 [+0.0045, +0.0127] seeds +0.0154, +0.0109, -0.0005 -> no verdict
- 60nan: E − B +0.0083 [+0.0033, +0.0134] seeds +0.0117, +0.0013, +0.0121 -> E higher

Primary (E114, E − B at 40/60nan, fresh seeds): E higher at ['60'], B higher at no level -> partial (fold-pair t, seed agreement, SESOI 0.01)
```
