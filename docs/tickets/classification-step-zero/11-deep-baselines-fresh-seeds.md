# E99 on fresh seeds: TRIDENT against MLP and SAINT at 40–80nan (E111; pre-registered 2026-11-02 00:01 GMT-3)

This ticket was written and committed before the study's first cell. No cell has run.

- **Why:** E99 ([08](08-deep-baselines.md)) found TRIDENT leading MLP, SAINT and TabNet at 40–80nan
  on seeds 42/7/13; the MLP margin is small (+0.013, +0.008, a tie at 80nan). A replication on seeds
  E99 never used, without re-tuning either side, tests whether the reading depends on those seeds.
  TabNet is left out: its margin is large (+0.09 to +0.11) and E106 already checked it.
- **The user's go:** "voce pode colocar mais coisas pra rodar nsa duas maquinas" (2026-11-01, before
  23:59 GMT-3; back at 09:00).
- **Code:** `stage4_trident.py eval <variant> <seed> <experiment>` (an optional experiment tag) and
  `run_e111_trident_queue.sh`; `deep_baselines.py --study e111` (MLP and SAINT, E99's configurations
  from `e99_json/`, copied unread, E99's recipe, no search).

## Design (fixed before the first cell)

- **Variants:** the 27 at 40, 60 and 80nan. **Seeds:** 5151, 6262, 7373 (never used), ten folds each
  (`build_folds(frame, 'class', 10, seed)`).
- **TRIDENT:** each variant's E98 best configuration, evaluated at the three seeds, MLflow tag
  `experiment=step-zero-e111-2026-11-02`; on the machine holding its E98 study (electricity_40nan on
  gorgona8, the rest on medusa1).
- **MLP and SAINT:** E99's best configuration per variant, evaluated at the three seeds; 54 units
  dealt by cost between the machines (gorgona8's start when E108 ends).
- **The cap:** no item starts after 2026-11-02 08:30 GMT-3 (STOP files on both machines). Missing items
  are reported missing and their levels incomplete.
- **Amendment (2026-11-02 00:04 GMT-3, before any result):** the cap is removed on the user's word ("nao precisa
  interromper"); the study runs until every item has ended.

## Measures and rule, fixed now (`deep_baselines.py --study e111 --tally`)

- E99's: per level and baseline, TRIDENT − baseline test macro-F1 over 270 fold pairs, the project's
  rule. TRIDENT leads a baseline if higher at 2 or more of the 3 levels and lower at none.
- **Replicated** if TRIDENT leads both MLP and SAINT; **not replicated** otherwise.

## Outcome

**Written 2026-11-02 05:04 GMT-3. Replicated: TRIDENT leads MLP and SAINT on fresh seeds**, by E99's rule.

| TRIDENT − baseline, test macro-F1 (270 fold pairs) | 40nan | 60nan | 80nan |
|---|---|---|---|
| MLP | +0.009 (E99 +0.013) | +0.007 (E99 +0.008) | −0.000, no verdict (E99 −0.001) |
| SAINT | +0.035 (E99 +0.036) | +0.035 (E99 +0.036) | +0.029 (E99 +0.029) |

- **Complete:** 81 TRIDENT evaluations and 54 baseline units, 2026-11-02 00:01–05:03 GMT-3, no
  failure. The end chain's rsync error is harmless: E111 tunes nothing, so no `json/` exists.
- **The same picture as E99:** about a point over MLP at 40–60nan, a tie at 80nan, three to four
  points over SAINT everywhere.

### The tally, unedited (`deep_baselines.py --study e111 --tally`, 2026-11-02 05:04 GMT-3)

```
## 40nan
- TRIDENT − mlp: +0.0087 [+0.0042, +0.0133] seeds +0.0073, +0.0074, +0.0115 -> TRIDENT higher
- TRIDENT − saint: +0.0353 [+0.0285, +0.0421] seeds +0.0311, +0.0349, +0.0398 -> TRIDENT higher

## 60nan
- TRIDENT − mlp: +0.0069 [+0.0013, +0.0125] seeds +0.0089, +0.0046, +0.0072 -> TRIDENT higher
- TRIDENT − saint: +0.0347 [+0.0275, +0.0420] seeds +0.0395, +0.0283, +0.0364 -> TRIDENT higher

## 80nan
- TRIDENT − mlp: -0.0003 [-0.0067, +0.0060] seeds -0.0009, -0.0043, +0.0041 -> no verdict
- TRIDENT − saint: +0.0293 [+0.0216, +0.0370] seeds +0.0304, +0.0252, +0.0324 -> TRIDENT higher

- leads: mlp yes, saint yes

Primary (E111, E99 on seeds 5151/6262/7373, MLP and SAINT): replicated (fold-pair t, seed agreement)
```
