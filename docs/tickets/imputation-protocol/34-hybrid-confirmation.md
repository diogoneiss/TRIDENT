# Does the deployable hybrid's `table` rule hold on fresh seeds, and where it is not needed? (E90, E91; pre-registered 2026-10-28)

This ticket was written and committed before the studies' queues started.

- **Why:** E89 ([33](33-hybrid-selection-rules.md)) found two rules that close most of the
  deployable hybrid's gap to TabPFN. Both are better than H on 10 of 10. `table` is worse than
  TabPFN on none, and the primary `shrunk` is worse on 2. That was a development read on E82's own
  seeds, and E89 fixed beforehand that a confirmation on fresh seeds must follow, with one rule.
- **The user's go:** on the confirmation, and to keep gorgona8 busy until about 09:00 ("voce pode
  rodar ate mais coisa, quero ocupar a gorgona8 até umas 9 da manha", 2026-10-28, about 22:50
  GMT-3).
- **The rule, fixed now: `table`.** It is the simpler rule, and it did better than `shrunk` in
  E89. `shrunk`, `column` and `pooled` are printed as secondary and decide nothing.
  - `table` is one numerical weight per table, from E69's grid, with each column's residuals
    scaled by its training split's sd, plus one categorical pick per table. Both are chosen on fold
    k's hidden observed validation cells.
- **Code:** no change to `src/`. `scripts/experiments/hybrid_confirmation.py` points E82's
  `train` and E89's `cells` at each study's tag and state. `run_hybrid_confirmation_queue.sh` runs
  four training queues and two cells lanes. A lane takes each cell as soon as its run is in the
  store copy.
- **Smoke:** E90's vehicle_80nan s4545 was trained before this commit (CELL_DONE; five selection
  files). Only its run's headline was printed (H 0.814), and no rule was scored. It counts.

## Design (fixed before the queues)

- **Seeds 4545, 5656, 6767,** used by no earlier run in the store.
- **E90, the confirmation:** E82's 10 hard variants (`blend_weights.HARD`), so 30 cells. These are
  electricity at 40 and 80nan; vehicle and biodeg at 80nan; kr-vs-kp and credit-g at 40, 60 and
  80nan. Each cell is E82's exactly:
  - H (`--decode_checkpoint loss --decode_calibration none`) with the hidden observed validation
    cells read in place;
  - then TabPFN v2 refit on each training fold.
- **E91, where the hybrid is not needed:** 7 of E44's confirmation variants outside the hard set,
  where E60's equal blend already beat TabPFN, so 21 cells. They are biodeg_40nan, kc2_40nan, kc2_80nan, pendigits at 40, 60 and
  80nan, and vehicle_40nan.
  - The question: does the deployable hybrid cost anything there?
  - letter and electricity_60nan are left out to fit the GPU window.
- **Running:** gorgona8. Four training queues deal both studies' cells by measured minutes,
  longest first; two cells lanes follow the training.
  - MLflow tags `experiment=hybrid-confirmation-2026-10-28` (E90) and
    `hybrid-generality-2026-10-28` (E91), logged through the leader's server.
  - About 4.7 h of training on four queues, and the refits overlap it; done by about 06:00–07:30.
  - **Deadlines, by the clock alone:** no training cell starts after 2026-10-29 07:00 GMT-3, and no
    cells job after 08:40 GMT-3. A study with cells missing at the deadline is reported partial,
    with no call.

## Measures and rule, fixed now (`hybrid_confirmation.py --study <e90|e91> --tally`)

- **Per variant:** `table` − H, `table` − TabPFN and `table` − the equal blend on `impute_score`,
  over 15 fold pairs. The project's rule applies, with a SESOI of 1% of the opponent's mean (E82's).
- **E90, in E82's form:**
  - **supported** if better than H on 8 or more of 10, worse on none, and worse than TabPFN on 2 or
    fewer;
  - **refuted** if better than H on 5 or fewer, or worse on 2 or more, or worse than TabPFN on 5 or
    more;
  - **mixed** otherwise.
- **E91:**
  - **supported** (no cost) if worse than H on none and worse than TabPFN on 1 or fewer of 7;
  - **refuted** if worse than H on 2 or more, or worse than TabPFN on 3 or more;
  - **mixed** otherwise.
- **What follows:** if E90 is supported, the deployable hybrid with `table` has a fresh-seed
  result. Whether ADR 0031 gains a deployable output is then the user's decision, with E91 saying
  whether it costs anything elsewhere.

## Outcome

(open)
