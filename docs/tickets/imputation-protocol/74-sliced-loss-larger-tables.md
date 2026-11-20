# E184's sliced Wasserstein loss term on the eight larger MCAR variants (E185, T62; pre-registered 2026-11-20 19:10 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** E184 (ticket 73) was mixed on the 18 small and medium MCAR variants: the term lowered the output's
  `sw_score` on 5 (with the `sw_score` selector) and 4 (with the default), the same few under both, and by much only
  on `spambase_20nan` (−0.23 / −0.20), the largest table in that set. This study asks whether the term does more on
  the larger tables E184 left out.
- **The user's go:** "teste em mais tabelas!" (2026-11-20), answering my offer after E184 of the larger or more-missing
  tables (`spambase` 40–80nan, `pendigits`, `letter`) or a weight sweep. The design below is mine.
- **Disclosed:**
  - E184's results were read before this design, including the `spambase_20nan` gain that motivates it.
  - The variants are every MCAR variant of E182/E183's set that E184 did not run (no choice by result):
    `pendigits_20/40/60/80nan`, `electricity_20/40/80nan`, `letter_20nan`. `spambase` 40–80nan are not in that set
    and have no E182 baseline files, so they are left out.
  - No new code in `src/`; the weight is E184's (0.8), with no new smoke.
- **Runner (this commit):** `scripts/experiments/sliced_loss_large_run.py` runs `sliced_loss_run.py`'s functions with
  this study's variants, state `/scratch/diogoneiss/trident-2026-11-20-sliced-loss-large`, tag
  `experiment = sliced-loss-large-2026-11-20`, and the rule below. `pendigits` and `letter` use their promoted
  files (ADR 0018 / 0037 bigger encoder where they apply), as in E183.
- **Scenarios:** the 8 variants × seeds 101/202/303 × five folds, all on gorgona8's RTX 3090 Ti, four queues
  (`c15_queue.sh`, two threads each), a pair's two arms side by side.
  - Core (the primary): **S** (`--decode_metric sw_score`) and **LS** (S plus `--decode_sw_weight 0.8`). 48 runs.
  - Second batch, after the core: **K** (the default run) and **L** (K plus the term). 48 runs.
  - Cost: about 14–18 h a batch (E183's gorgona8 clocks: `letter_20nan` about 115 min a run, `electricity_20nan`
    about 70; the term adds about 20% to decode).
- **Checks before the read:** `--gate` and `--pairing`, as in E184. A failing check is reported, not dropped.
- **Measures and decision rule:** as E184: each arm's test-fold ledgers scored by E182's code (`sw_score`,
  `dist_score`, `impute_score`) on the calibrated output, fold-pair rule with seed agreement, margin 0.01 of the
  control's mean.
  - Hypothesis: on the larger tables the term lowers the output's `sw_score`.
  - Primary, LS − S on `sw_score` over the variants whose shuffled control reaches 0.10: supported if better on half
    or more and worse on at most 1 (E184's 2 of 18 scaled to 8); refuted if better on a sixth or fewer (at most 1 of
    8); mixed otherwise. The same rule is printed for L − K once the second batch is in.
  - Secondary, no rule: `impute_score` and `dist_score` for both pairs; kept epochs; each arm against E182's best
    strong-tier baseline on `sw_score` and `impute_score`.
  - Read once: `--tally core` when the core is in, `--tally` when all four arms are. No default changes whatever the
    result; pooling with E184 is not planned.
- **Not in this study:** other weights; `spambase` 40–80nan; pre-training; calibration off.

## Outcome

Not read yet.
