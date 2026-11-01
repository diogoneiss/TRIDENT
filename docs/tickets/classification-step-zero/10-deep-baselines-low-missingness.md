# The deep baselines at 00 and 20nan, descriptive (E107; pre-registered 2026-11-01 09:22 GMT-3)

This ticket was written and committed before the study's first cell. No cell has run.

- **Why:** E99 ([08](08-deep-baselines.md)) supported TRIDENT leading MLP, SAINT and TabNet at
  40–80nan. Its scope left out 00 and 20nan. Does TRIDENT also lead with few gaps? Descriptive: no
  claim depends on it.
- **The user's go:** "3 e 4 agora então" (2026-11-01, after 05:58 and before 09:22 GMT-3); this is
  item 4.
- **No new code:** `deep_baselines.py --study e107` is E99's driver, recipe and spaces at
  `LEVELS = ("00", "20")`.

## Design (fixed before the first cell)

- **Units:** MLP, SAINT and TabNet × the 18 variants at 00 and 20nan = 54, each E99's (20 TPE trials
  on seed 42's validation macro-F1, evaluations at seeds 42/7/13, ten folds). TRIDENT's folds are
  E98's at those levels.
- **Running:** `STUDY=e107 run_deep_baselines_queue.sh`, the units dealt by cost (41 on medusa1, 13 on
  gorgona8), beside E106. Outputs in `trident-2026-11-01-deep-e107/`; MLflow off.
- **The cap:** no unit starts after 2026-11-02 08:00 GMT-3 (a STOP file on both machines).

## Measures (`deep_baselines.py --study e107 --tally`)

- **Per level and baseline:** TRIDENT − baseline, test macro-F1, 270 fold pairs, the project's rule:
  "TRIDENT higher", "baseline higher" or no verdict. Printed; no primary call.

## Outcome

**Written 2026-11-01 16:15 GMT-3 (descriptive).** With few gaps TRIDENT still leads SAINT and TabNet, and MLP
from 20nan; on the complete tables (00nan) MLP is ahead.

| TRIDENT − baseline, test macro-F1 (270 fold pairs) | 00nan | 20nan |
|---|---|---|
| MLP | **−0.012** (MLP higher) | +0.009 (TRIDENT higher) |
| SAINT | +0.006 (TRIDENT higher) | +0.029 (TRIDENT higher) |
| TabNet | +0.055 (TRIDENT higher) | +0.108 (TRIDENT higher) |

- **Complete:** all 54 units, 09:22–16:15 GMT-3, no failure.
- **Read with E99:** against MLP, TRIDENT is behind at 00nan, ahead at 20, 40 and 60nan, tied at
  80nan; against SAINT and TabNet it is ahead at every level from 00 to 80nan. A thesis statement
  about deep baselines can name the range where it holds (20–60nan against all three) and the MLP
  lead on complete tables.

### The tally, unedited (`deep_baselines.py --study e107 --tally`, 2026-11-01 16:15 GMT-3)

```
## 00nan
- TRIDENT − mlp: -0.0120 [-0.0164, -0.0075] seeds -0.0122, -0.0132, -0.0105 -> baseline higher
- TRIDENT − saint: +0.0056 [+0.0002, +0.0109] seeds +0.0045, +0.0041, +0.0082 -> TRIDENT higher
- TRIDENT − tabnet: +0.0546 [+0.0464, +0.0628] seeds +0.0496, +0.0614, +0.0527 -> TRIDENT higher

## 20nan
- TRIDENT − mlp: +0.0086 [+0.0050, +0.0122] seeds +0.0098, +0.0080, +0.0080 -> TRIDENT higher
- TRIDENT − saint: +0.0287 [+0.0226, +0.0348] seeds +0.0328, +0.0305, +0.0228 -> TRIDENT higher
- TRIDENT − tabnet: +0.1082 [+0.0981, +0.1182] seeds +0.1120, +0.1043, +0.1082 -> TRIDENT higher

Descriptive (E107, 00/20nan): mlp 00nan baseline higher, 20nan TRIDENT higher; saint 00nan TRIDENT higher, 20nan TRIDENT higher; tabnet 00nan TRIDENT higher, 20nan TRIDENT higher
```
