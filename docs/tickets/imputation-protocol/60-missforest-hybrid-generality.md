# The missForest-side hybrid on every variant with T39's cell tables (E164, T54; pre-registered 2026-11-08 21:14 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** E162 ([59](59-missforest-hybrid.md)) found ADR 0031 decision 6's `table` rule, with missForest as the
  other side, better than H and than missForest on `vehicle_80nan` and `biodeg_80nan`. Those two were chosen because
  H trails missForest there, so extending ADR 0031 to missForest needs the rule read where nothing was chosen.
- **The user's go:** "faça 1 2 e 3" (2026-11-08), on "3. Estender o E162 a todas as variantes a 80% em que TRIDENT
  perde para o missForest".
- **The literal scope was empty (disclosed):** at 80nan, the runs' own H trails their logged missForest only on
  vehicle and biodeg, which E162 already read. It leads on credit-g, electricity, kr-vs-kp, kc2 and pendigits
  (E90/E91), and on letter and spambase (E44's and E43's runs: H 0.89 against 0.97, and 0.98–0.99 against 1.03).
  This study is the extension an ADR would need instead: every variant with T39's cell tables, none chosen by score.
- **Disclosure:** the runs' H and missForest scores on these variants are on the record. They name the secondary's
  scope, `TRAILS` in the runner: the 7 variants where H trails missForest by more than 1% of its score. These are
  `biodeg_80nan`, `vehicle_80nan`, `credit-g_40nan`, `biodeg_probit0p25_60nan`, `biodeg_probit_60nan`,
  `vehicle_probit_60nan` and `electricity_20nan`. The two 80nan ones are E162's on E90's seeds.

## Design

- **Cells:** E90's 10, E91's 7, E93's 12 and E95's 9 variants (`hybrid_confirmation.STUDIES`), seeds
  4545/5656/6767, five folds: 38 variants, 114 pairs. H's and TabPFN's fills are the studies' own cell tables.
  missForest (ADR 0016) is refit on each training fold. It fills the test induced cells, and the hidden observed
  validation cells group by group, as TabPFN did. Its test score is asserted equal to the run's logged missForest
  score, fold by fold.
- **Machine:** medusa1, CPU only, nothing in the store. gorgona8's CPU is saturated by the user's other project.
  E90/E91/E93's tables were copied from gorgona8, and all 1,740 files' SHA-256 matched. E95's are medusa1's own. The
  assert also checks that medusa1's missForest reproduces gorgona8's.
- **Runner:** `scripts/experiments/missforest_generality.py` (`missforest_hybrid.add_missforest`, then
  `missforest_hybrid.both_sides`). State `/scratch/diogoneiss/trident-2026-11-08-missforest-generality`.
- **Cost:** a `vehicle_80nan` pair took 187 s on gorgona8 (E162's refactor check). electricity's nine pairs,
  45,312 rows, are unmeasured and dominate. The estimate is several hours over 10 lanes.

## Measures and rule (fixed now)

- **Verdict:** `common.verdict` on 15 (seed, fold) differences of `impute_score`, SESOI 1% of the opponent's mean.
- **Primary,** `table_missforest` on the 38 variants:
  - **supported** if it is worse than H on none and worse than missForest on at most 2;
  - **refuted** if it is worse than H on 2 or more, or worse than missForest on 5 or more;
  - **mixed** otherwise.
- **Secondary:** how many variants it is better than H on, overall and on `TRAILS`.
- **Read:** once, with all 114 pairs in, `missforest_generality.py --tally e164` on medusa1.
- **What it leads to:** extending ADR 0031 decision 6 to missForest is the user's call. E165 reads a rule that picks
  the side on the same cells.
