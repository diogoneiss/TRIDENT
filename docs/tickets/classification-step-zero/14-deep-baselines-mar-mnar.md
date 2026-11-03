# TRIDENT against MLP and SAINT under MAR and MNAR at 60nan (E115, descriptive; pre-registered 2026-11-03 00:53 GMT-3)

This ticket was written and committed before the study's first cell. No cell has run.

- **Why:** E99, E107 and E111 compare TRIDENT with the deep baselines under MCAR only. The thesis makes
  MNAR claims for imputation; does the classification picture hold when the gaps are not random?
  Descriptive: no claim depends on it.
- **The user's go:** "gosto da ideia de rodar o 3, com a outra maquina rodar 1 ou 2. vá" (2026-11-03,
  after 00:38 GMT-3 and before this ticket).
- **Code:** `stage4_trident.py eval <variant> <seed> <experiment> <config_variant>` (a variant can borrow
  its MCAR sibling's E98 configuration) and `run_e115_trident_queue.sh`; `deep_baselines.py --study e115`
  (MLP and SAINT with the sibling's E99 configuration, E99's recipe, no search).

## Design (fixed before the first cell)

- **Variants (existing data):** MNAR (self-masking probit, slope 0.5) on biodeg, kc2, vehicle and spambase;
  MAR (`marf`) on biodeg, kc2, vehicle, spambase, credit-g and kr-vs-kp; all at 60nan. 10 variants.
- **Each side keeps its MCAR tuning:** TRIDENT takes `<base>_60nan`'s E98 configuration, MLP and SAINT
  `<base>_60nan`'s E99 one; nothing is re-tuned. Seeds 42/7/13, ten folds
  (`build_folds(frame, 'class', 10, seed)`). Everything on medusa1, where the E98 studies live. TRIDENT's
  evaluations carry MLflow tag `experiment=step-zero-e115-2026-11-03`; the baselines log nothing.
- 30 TRIDENT evaluations and 20 baseline units.

## Measures (`deep_baselines.py --study e115 --tally`, on medusa1)

- Per mechanism (MNAR's 4 variants, MAR's 6), TRIDENT − baseline test macro-F1, paired by fold and seed,
  pooled, one array per seed, the project's rule: "TRIDENT higher", "baseline higher" or no verdict.
  Read beside E99's MCAR result at 60nan (+0.008 over MLP, +0.036 over SAINT). No primary call.

## Outcome

**Written 2026-11-03 01:19 GMT-3 (descriptive).** The MCAR picture holds off MCAR: TRIDENT stays well ahead of SAINT
under both mechanisms, and ahead of MLP under MAR; under MNAR its lead over MLP has no verdict.

| TRIDENT − baseline, test macro-F1 | MNAR (4 variants, 120 pairs) | MAR (6 variants, 180 pairs) | MCAR at 60nan (E99) |
|---|---|---|---|
| MLP | +0.006 [−0.005, +0.016], no verdict (seeds +0.007, +0.011, −0.001) | **+0.011 TRIDENT higher** | +0.008 TRIDENT higher |
| SAINT | **+0.045 TRIDENT higher** | **+0.043 TRIDENT higher** | +0.036 TRIDENT higher |

- **Complete:** 30 TRIDENT evaluations and 20 baseline units on medusa1, 00:53–01:19 GMT-3, no failure.
- Each side used its MCAR sibling's tuning; a variant-specific tuning could move either side.

### The tally, unedited (`deep_baselines.py --study e115 --tally`, 2026-11-03 01:19 GMT-3)

```
## MNAR (probit): biodeg, kc2, vehicle, spambase
- TRIDENT − mlp: +0.0056 [-0.0046, +0.0157] seeds +0.0074, +0.0106, -0.0012, 120 fold pairs -> no verdict
- TRIDENT − saint: +0.0449 [+0.0313, +0.0585] seeds +0.0425, +0.0447, +0.0475, 120 fold pairs -> TRIDENT higher

## MAR (marf): biodeg, kc2, vehicle, spambase, credit-g, kr-vs-kp
- TRIDENT − mlp: +0.0112 [+0.0048, +0.0176] seeds +0.0114, +0.0131, +0.0092, 180 fold pairs -> TRIDENT higher
- TRIDENT − saint: +0.0433 [+0.0322, +0.0544] seeds +0.0492, +0.0412, +0.0394, 180 fold pairs -> TRIDENT higher
```
