# The quarter-label channel at 20, 40 and 80% missingness (E143; pre-registered 2026-11-05)

This ticket was written and committed before the cut's first cell (00:30 GMT-3).

- **Why:** E136/E138 read the channel at 60nan only. The thesis's premise is pre-training on gappy tables, so the
  channel's dependence on the missingness level is the natural axis. Night run on the user's standing request.
- **Code:** none new. Cut `12` in `c1_run.py` (seed 1515 on gorgona8); reader `c1_report.py --levels`.

## Design and rule (fixed before the first cell)

- **Cells:** `{biodeg,kc2,vehicle,spambase}_{20,40,80}nan` (MCAR) × seeds 1515/2626/3737 × S25V / B25V = 72.
- **Per level, three tests as one family:** S25V − B25V pooled over the four tables, 15 replicates, plain 95% t
  interval, Holm over the three levels; **open** / **closed** as E140. E138's 60nan MCAR half printed beside.
- **What follows:** descriptive for the thesis; no ADR.
- **Read only with all 72 cells done.**

## Outcome (2026-11-05, 02:43 GMT-3; 72 of 72 cells, gorgona8 24 and medusa1 48, cut 12 done on medusa1 02:24 and on gorgona8 02:38 GMT-3, 0 failed)

Read with `c1_report.py --levels`. Unedited tally, S25V − B25V in test macro-F1, seeds 1515/2626/3737 × 5 folds,
pooled over the four tables per level, Holm over the three levels:

| level | pooled S − B | NB | p | Holm | channel | tables better / worse |
|---|---|---|---|---|---|---|
| 20nan | **+0.0246 [+0.0048, +0.0445]** | [−0.0210, +0.0702] | 0.0185 | 0.0392 | **OPEN** | none / none |
| 40nan | **+0.0234 [+0.0057, +0.0410]** | [−0.0172, +0.0640] | 0.0131 | 0.0392 | **OPEN** | kc2_40nan (+0.066) / none |
| 60nan (E138 MCAR half, beside) | +0.0258 [+0.0083, +0.0432] | | | | open | vehicle_60nan (+0.050), spambase_60nan (+0.012) / none |
| 80nan | +0.0040 [−0.0164, +0.0245] | [−0.0430, +0.0511] | 0.6786 | 0.6786 | closed | spambase_80nan (+0.018) / none |

- **Flat from 20 to 60% missingness (+0.023 to +0.026), closed at 80%.** The channel does not grow with the missingness
  level; it holds steady until the table is mostly gaps, where only spambase (the largest of the four) still gains.
  No table is worse at any level.
- **Why 80nan closes, two readings not tested here:** with four fifths of the cells missing, the pre-training signal
  (`embedding_normalized`, an embedding-level objective over masked cells in a mostly empty context) has less
  structure to learn and hand over; and the
  small tables' per-seed means swing both ways at 80nan (vehicle −0.022, kc2 +0.020), so the pooled interval
  widens.
- **Reading, by the rule:** open at 20 and 40 (Holm 0.039), closed at 80. Descriptive for the thesis: on these four
  tables with a quarter of the labels, pre-training pays about +0.025 macro-F1 at 20–60% missingness and nothing
  measurable at 80%. No ADR.
