# The per-column KC + TabPFN hybrid as an output: does E69's result hold on fresh seeds, over today's default? (E74; pre-registered 2026-10-24)

This ticket was written and committed before the study's first cell.

- **Why:** E69 ([24](24-blend-weights.md)) was mixed and descriptive. A per-column, cross-fitted
  blend of K's and TabPFN's fills beat both on 13 of 21 variants, beat TabPFN on its 80nan ties, and
  reached TabPFN's level on the categorical tables. E69's Outcome said that making it an output
  needs its own pre-registered rule on a fresh seed set.
- **The user's go:** "2 gosto da ideia" and "sim, mas lance o 2 imediatamente em seguida"
  (2026-10-24, 11:49 and 11:56 GMT-3), to running it on the 10 hard variants.
- **Code, nothing in `src/` and nothing in the store beyond the training runs:**
  - `scripts/experiments/hybrid_blend.py`: `train` runs today's default; `blend` is E69's procedure
    (`blend_weights.collect` and `score_pair`, refactored out of E69's runner; E69's
    credit-g_80nan pair at 404 reproduces to 0.0 after the refactor);
  - `scripts/experiments/run_hybrid_queue.sh`: four training queues, then two blend lanes.

## Design (fixed before the first cell)

- **The model side is today's default output.** Each cell is a default imputation run of the
  checkout, five folds. At these rates ADR 0027 calibrates the induced output, so the headline and
  the fold ledger's fills are **KC**, not E69's K. `collect` asserts the ledger's score equals the
  run's KC fold by fold (checked on E64's calibrated biodeg_marf_60nan at 404 before this ticket).
- **Fresh seeds:** 1212, 2323 and 3434; no run in the store uses them.
- **Variants:** E69's 10 hard ones: electricity_40nan and _80nan, vehicle_80nan, biodeg_80nan,
  kr-vs-kp and credit-g at 40/60/80nan. 30 training cells, then 30 blend pairs.
- **The hybrid, E69's exactly:** for fold k and each column, the model's weight from
  {0, 0.25, 0.5, 0.75, 1} (numerical, least squared error) or a pick of either category
  (categorical, more right), chosen on the other four folds' induced cells; ties go to the equal
  blend. TabPFN v2 is refitted on each training fold.
- **Running:** four GPU training queues, then two GPU blend lanes, chained; MLflow tag
  `experiment=hybrid-blend-2026-10-24`; state in `/scratch2/diogoneiss/trident-2026-10-24-hybrid-blend/`.
  About 3.5–4.5 h, with no cutoff.

## Measures and reading, fixed now (`hybrid_blend.py --tally`)

On the induced `impute_score` over 15 fold pairs, with the verdict rule and the 1% SESOI:

- **Per variant:** hybrid − KC, hybrid − TabPFN, hybrid − equal blend, and TabPFN − KC.
- **Primary, the output question,** over the 10 variants:
  - **supported** (the hybrid is worth proposing as an output): better than KC on 8 or more, and
    worse than TabPFN on at most 1;
  - **refuted:** better than KC on at most 5, or worse than TabPFN on 3 or more;
  - **mixed:** otherwise.
- **Secondary:** how many variants the hybrid beats both KC and TabPFN on (E69: 3 of these 10
  against TabPFN with K as the model side).
- **What each reading leads to:**
  - **supported:** a proposed ADR for an opt-in hybrid output (TabPFN is an offline optional
    extra, ADR 0028), for the user to decide; nothing changes by default.
  - **refuted:** the hybrid stays a descriptive result (E60, E66, E69).
  - **mixed:** reported, and the user chooses.
