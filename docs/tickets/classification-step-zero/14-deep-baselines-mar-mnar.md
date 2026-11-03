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

(open)
