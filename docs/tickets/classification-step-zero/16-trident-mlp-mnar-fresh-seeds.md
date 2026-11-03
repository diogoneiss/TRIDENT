# TRIDENT against MLP under MNAR, on fresh seeds (E120; pre-registered 2026-11-03)

This ticket was written and committed before the study's first cell (15:20 GMT-3).

- **Why:** E115 ([14](14-deep-baselines-mar-mnar.md), descriptive) left TRIDENT − MLP under MNAR without a
  verdict (+0.006 [−0.005, +0.016], one seed below zero). Extending E115's seeds would be optional stopping,
  so this is a new study on fresh seeds, with a primary call. It decides whether "TRIDENT leads similar deep
  models" can be said under MNAR too.
- **The user's go:** "1 e 2 agora" (2026-11-03).
- **Code:** `deep_baselines.py --study e120` and `STUDY=e120` for `run_e115_trident_queue.sh`; no change to
  `src/`.

## Design (fixed before the first cell)

- **E115's design, verbatim, at seeds 5151, 6262, 7373** (E111's fresh seeds, so MCAR numbers exist at the
  same seeds): the same 10 variants at 60nan (MNAR probit slope 0.5 on biodeg, kc2, vehicle, spambase; MAR
  `marf` on those and credit-g, kr-vs-kp), each side with its MCAR sibling's tuning (TRIDENT E98's, MLP and
  SAINT E99's), nothing re-tuned, ten folds (`build_folds(frame, 'class', 10, seed)`).
- All on medusa1, where the E98 studies live. TRIDENT's evaluations carry MLflow tag
  `experiment=step-zero-e120-2026-11-03`; the baselines log nothing. 30 TRIDENT evaluations, 20 baseline
  units; about 30 minutes, as E115.

## Measures and rule, fixed now (`deep_baselines.py --study e120 --tally`)

- **Primary:** TRIDENT − MLP test macro-F1 under MNAR, pooled over the 4 variants, paired by fold and seed,
  one array per seed (120 pairs), the project's rule with SESOI 0.01:
  - **supported** (TRIDENT leads MLP under MNAR) if TRIDENT is higher;
  - **refuted** if MLP is higher;
  - **tie** if equivalent (inside ±0.01): no lead to claim under MNAR, but no deficit either;
  - **no verdict** otherwise.
- **Descriptive:** TRIDENT − SAINT under MNAR, and both under MAR, read beside E115.
- **Read only with every unit done.**

## Outcome

**Written 2026-11-03 16:03 GMT-3. Primary: no verdict.** Under MNAR, TRIDENT − MLP is +0.0025 [−0.0090, +0.0141] over 120
fold pairs: every seed above zero (+0.0028, +0.0009, +0.0040), but the interval crosses both zero and the
±0.01 margin, so it shows neither a lead nor a tie. With E115 (+0.006, no verdict), two independent seed sets
agree: under MNAR TRIDENT and MLP are close, and no lead over MLP can be claimed there.

- **Complete:** 30 TRIDENT evaluations and 20 baseline units on medusa1, 15:21–15:54 GMT-3, none failed.
- **Descriptive, beside E115:** TRIDENT stays ahead of SAINT under MNAR (+0.035) and MAR (+0.040), and of MLP
  under MAR (+0.012), all "TRIDENT higher", as in E115.
- **For the thesis:** "TRIDENT leads similar deep models" holds under MCAR and MAR against MLP and SAINT, and
  under MNAR against SAINT; against MLP under MNAR the two are not separable.

### The tally, unedited (`deep_baselines.py --study e120 --tally`, medusa1, 2026-11-03 16:03 GMT-3)

```
## MNAR (probit): biodeg, kc2, vehicle, spambase
- TRIDENT − mlp: +0.0025 [-0.0090, +0.0141] seeds +0.0028, +0.0009, +0.0040, 120 fold pairs -> no verdict
- TRIDENT − saint: +0.0349 [+0.0187, +0.0510] seeds +0.0366, +0.0342, +0.0337, 120 fold pairs -> TRIDENT higher

## MAR (marf): biodeg, kc2, vehicle, spambase, credit-g, kr-vs-kp
- TRIDENT − mlp: +0.0119 [+0.0058, +0.0180] seeds +0.0202, +0.0055, +0.0100, 180 fold pairs -> TRIDENT higher
- TRIDENT − saint: +0.0398 [+0.0295, +0.0502] seeds +0.0442, +0.0341, +0.0411, 180 fold pairs -> TRIDENT higher

Primary (E120, TRIDENT − MLP under MNAR, SESOI 0.01): no verdict
```
