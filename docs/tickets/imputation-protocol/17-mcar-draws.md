# Fresh MCAR draws: is a result tied to one missingness pattern? (E62; pre-registered 2026-10-22)

This ticket was written and committed before the study's first cell. The gate cell runs without
MLflow and is reported under "Launch".

- **Why:** this is T31. Each MCAR rate is a single draw from `_00nan`: `inject_nans` with seed
  42 + the rate.
  - The seeds vary the folds and the model, never which cells are missing.
  - The critique asked about it (`docs/reviews/imputation-critique/08-experimentation-design.md`),
    and the review of 2026-10-16 named it ("one MCAR draw each").
- **The user's go:** "Faça a t31 agora então: gere os geradores, runners e execute logo depois"
  (2026-10-22, about 06:40 GMT-3).
- **The code, with nothing in `src/`:**
  - `datasets/generate_mcar_draws.py`, the generator;
  - `scripts/experiments/mcar_draws_run.py`, the runner;
  - `scripts/experiments/mcar_draws_report.py`, the reader;
  - `scripts/experiments/run_mcar_draws_queue.sh`, the queues.

## Design (fixed before the first cell)

- **Variants:** pendigits_20nan, kr-vs-kp_40nan, credit-g_60nan and biodeg_80nan. These span the
  sizes, both column kinds and every rate.
- **Draws:**
  - **Draw 1 is the shipped file.** The generator regenerates it with seed 42 + pp and refuses to
    go on unless the two are equal cell for cell; all four are.
  - **Draws 2 and 3** use seeds 42 + pp + 1000 and 42 + pp + 2000. They are written as
    `<base>_d2_<pp>nan` and `<base>_d3_<pp>nan`, with the rate last, as the MNAR variants have it.
  - Every draw has the shipped per-column count, and its gaps overlap the shipped draw's at about
    the rate, as independent draws do. The data directory is gitignored; the generator is the
    record.
- **Cells:** 4 variants × 3 draws × seeds 42/7/13 make 36 cells of today's default.
  - A seed builds the same folds on all three draws, since folds depend on the labels only.
  - **Each draw trains with the configuration the shipped variant resolves,** passed as an override.
    So ADR 0018's file for pendigits_20nan reaches its draws too.
  - `--score_checkpoints` and `--score_calibrated` are on, as in E44 and E51.
  - ADR 0027's `auto` calibration resolves on each draw's measured rate, which equals the
    original's.
  - The strong tier is scored live in the run (ADR 0021). TabPFN comes from an offline pass
    (`tabpfn_run.py`) into the state directory's `tabpfn/`, one lane.
- **Draw 1 is rerun rather than reused.**
  - E35's runs at 42/7/13 predate ADR 0018 on pendigits_20nan and ADR 0019's removal of
    kr-vs-kp_40nan's file.
  - biodeg_80nan never ran at 42/7/13.
  - So the three draws come from one study on one commit.
- **The gate,** before launch, with `--no-mlflow`: credit-g_60nan, draw 1, seed 42, against E35's
  run, fold by fold on K. Any difference stops the launch, since it would mean the override path
  does not reproduce a default run.
- **Running:**
  - four queues, longest cell first, about 4 h;
  - one TabPFN lane beside them;
  - no cutoff;
  - state in `/scratch2/diogoneiss/trident-2026-10-22-mcar-draws/`.

## Measures and reading, fixed now (`mcar_draws_report.py --tally`)

On the induced `impute_score` over 15 fold pairs, with the verdict rule and the 1% SESOI:

- **Per draw:**
  - K − bar;
  - K − strong (ADR 0028's tier, TabPFN included);
  - K − TabPFN;
  - KC − strong (KC is the default output at 30% and above, ADR 0027).
- **Across a variant's three draws, per contrast:**
  - **same:** one call;
  - **a reversal:** better in one draw and worse in another;
  - **a drift:** any other change.
  - **Descriptive:** the span of the three draws' means against the mean interval half-width.
- **Primary:** K − bar and K − strong on all four variants.
  - **holds:** no reversal;
  - **tied to the draw:** any reversal, each named.
- **Secondary:** the patterns on K − TabPFN and KC − strong, and every drift, listed.
- **What each reading leads to (no default changes from this study):**
  - **holds:** the thesis states that the calls do not depend on the MCAR draw on these four
    tables.
  - **tied to the draw:** each reversal is named as a limit in the thesis, its table's claims are
    qualified, and it gets a ticket.
- **An incomplete variant** (any draw without all three seeds) is reported as incomplete and not
  read.

## Launch

- **The gate passed (2026-10-22, 06:44 GMT-3).** credit-g_60nan, draw 1, seed 42, with
  `--no-mlflow`, equals E35's run fold by fold on K, with a largest difference of 0.
  - The configuration came from the task defaults through the override.
  - The measured rate, 0.6000, resolved `auto` to `gaps`.
  - The run scored the strong tier live.
- **Launched** on the commit of this ticket: four queues (`run_mcar_draws_queue.sh`) and
  `tabpfn_lane.sh`, both in the state directory.

## Outcome (2026-10-22, 11:39 GMT-3)

**The run:** 36 of 36 cells and 36 of 36 TabPFN pairs, 06:44–11:39 GMT-3, with none failed.

- A pendigits cell took about 85 minutes, against the 65 estimated. Five processes shared the GPU.

**The printout** (`scripts/experiments/mcar_draws_report.py --tally`, unedited):

```text
## `pendigits_20nan`
- draw 1 (shipped): K − bar (knn10) -0.0309 [-0.0337, -0.0280] better; K − strong (tabpfn) +0.0112 [+0.0070, +0.0154] worse; K − tabpfn (tabpfn) +0.0112 [+0.0070, +0.0154] worse; KC − strong (tabpfn) +0.0023 [-0.0020, +0.0066] none
- draw 2 (seed 1062): K − bar (knn10) -0.0274 [-0.0310, -0.0239] better; K − strong (tabpfn) +0.0158 [+0.0125, +0.0192] worse; K − tabpfn (tabpfn) +0.0158 [+0.0125, +0.0192] worse; KC − strong (tabpfn) +0.0067 [+0.0036, +0.0098] worse
- draw 3 (seed 2062): K − bar (knn10) -0.0322 [-0.0354, -0.0291] better; K − strong (tabpfn) +0.0118 [+0.0088, +0.0147] worse; K − tabpfn (tabpfn) +0.0118 [+0.0088, +0.0147] worse; KC − strong (tabpfn) +0.0029 [+0.0001, +0.0056] worse
  - K − bar (primary): better / better / better -> same; the draws' means span 0.0048 against a mean interval half-width of 0.0032
  - K − strong (primary): worse / worse / worse -> same; the draws' means span 0.0047 against a mean interval half-width of 0.0035
  - K − tabpfn: worse / worse / worse -> same; the draws' means span 0.0047 against a mean interval half-width of 0.0035
  - KC − strong: none / worse / worse -> drift; the draws' means span 0.0044 against a mean interval half-width of 0.0034

## `kr-vs-kp_40nan`
- draw 1 (shipped): K − bar (lgbm) -0.0392 [-0.0477, -0.0307] better; K − strong (tabpfn) +0.0130 [+0.0060, +0.0200] worse; K − tabpfn (tabpfn) +0.0130 [+0.0060, +0.0200] worse; KC − strong (tabpfn) +0.0134 [+0.0078, +0.0190] worse
- draw 2 (seed 1082): K − bar (hgb/lgbm) -0.0414 [-0.0517, -0.0311] better; K − strong (tabpfn) +0.0142 [+0.0067, +0.0217] worse; K − tabpfn (tabpfn) +0.0142 [+0.0067, +0.0217] worse; KC − strong (tabpfn) +0.0136 [+0.0049, +0.0223] worse
- draw 3 (seed 2082): K − bar (hgb/lgbm) -0.0393 [-0.0494, -0.0292] better; K − strong (tabpfn) +0.0160 [+0.0116, +0.0204] worse; K − tabpfn (tabpfn) +0.0160 [+0.0116, +0.0204] worse; KC − strong (tabpfn) +0.0198 [+0.0167, +0.0229] worse
  - K − bar (primary): better / better / better -> same; the draws' means span 0.0022 against a mean interval half-width of 0.0097
  - K − strong (primary): worse / worse / worse -> same; the draws' means span 0.0030 against a mean interval half-width of 0.0063
  - K − tabpfn: worse / worse / worse -> same; the draws' means span 0.0030 against a mean interval half-width of 0.0063
  - KC − strong: worse / worse / worse -> same; the draws' means span 0.0064 against a mean interval half-width of 0.0058

## `credit-g_60nan`
- draw 1 (shipped): K − bar (mean_mode) -0.0078 [-0.0162, +0.0006] none; K − strong (tabpfn) +0.0387 [+0.0300, +0.0475] worse; K − tabpfn (tabpfn) +0.0387 [+0.0300, +0.0475] worse; KC − strong (tabpfn) +0.0274 [+0.0196, +0.0352] worse
- draw 2 (seed 1102): K − bar (mean_mode) -0.0129 [-0.0235, -0.0023] better; K − strong (tabpfn) +0.0322 [+0.0222, +0.0422] worse; K − tabpfn (tabpfn) +0.0322 [+0.0222, +0.0422] worse; KC − strong (tabpfn) +0.0228 [+0.0137, +0.0319] worse
- draw 3 (seed 2102): K − bar (mean_mode) -0.0115 [-0.0196, -0.0035] better; K − strong (tabpfn) +0.0304 [+0.0227, +0.0382] worse; K − tabpfn (tabpfn) +0.0304 [+0.0227, +0.0382] worse; KC − strong (tabpfn) +0.0246 [+0.0183, +0.0309] worse
  - K − bar (primary): none / better / better -> drift; the draws' means span 0.0051 against a mean interval half-width of 0.0090
  - K − strong (primary): worse / worse / worse -> same; the draws' means span 0.0083 against a mean interval half-width of 0.0088
  - K − tabpfn: worse / worse / worse -> same; the draws' means span 0.0083 against a mean interval half-width of 0.0088
  - KC − strong: worse / worse / worse -> same; the draws' means span 0.0046 against a mean interval half-width of 0.0077

## `biodeg_80nan`
- draw 1 (shipped): K − bar (mean_mode) -0.0298 [-0.0398, -0.0199] better; K − strong (missforest/tabpfn) +0.0515 [+0.0415, +0.0614] worse; K − tabpfn (tabpfn) +0.0512 [+0.0413, +0.0612] worse; KC − strong (missforest/tabpfn) +0.0211 [+0.0136, +0.0286] worse
- draw 2 (seed 1122): K − bar (mean_mode) -0.0197 [-0.0282, -0.0113] better; K − strong (tabpfn) +0.0645 [+0.0573, +0.0716] worse; K − tabpfn (tabpfn) +0.0645 [+0.0573, +0.0716] worse; KC − strong (tabpfn) +0.0355 [+0.0267, +0.0442] worse
- draw 3 (seed 2122): K − bar (mean_mode) -0.0367 [-0.0453, -0.0282] better; K − strong (tabpfn) +0.0510 [+0.0413, +0.0606] worse; K − tabpfn (tabpfn) +0.0510 [+0.0413, +0.0606] worse; KC − strong (tabpfn) +0.0207 [+0.0139, +0.0276] worse
  - K − bar (primary): better / better / better -> same; the draws' means span 0.0170 against a mean interval half-width of 0.0090
  - K − strong (primary): worse / worse / worse -> same; the draws' means span 0.0135 against a mean interval half-width of 0.0089
  - K − tabpfn: worse / worse / worse -> same; the draws' means span 0.0135 against a mean interval half-width of 0.0089
  - KC − strong: worse / worse / worse -> same; the draws' means span 0.0147 against a mean interval half-width of 0.0077

Primary: reversals on K − bar and K − strong: none -> holds (fold-pair t, seed agreement)
```

**Verdict: holds.** K − bar and K − strong show no reversal across the three draws on any of the
four variants.

**Reading:**

- **The primary calls:**
  - **K − bar:** better on every draw of pendigits_20nan, kr-vs-kp_40nan and biodeg_80nan.
    - On credit-g_60nan it drifts: no difference on the shipped draw, better on draws 2 and 3.
  - **K − strong:** worse on every draw of all four variants, TabPFN being the strong tier's best
    each time.
- **The secondary calls:**
  - **K − TabPFN:** the same on every variant.
  - **KC − strong:** worse throughout, except on pendigits_20nan, where it drifts from no
    difference on the shipped draw to worse on the fresh two.
    - At 20nan KC is not the default output (ADR 0027), so this drift does not touch a default's
      claim.
- **The shipped draw is not a favourable one.**
  - It is the most favourable to K against the strong tier on pendigits_20nan and kr-vs-kp_40nan.
  - It is the least favourable on credit-g_60nan, on both primary contrasts.
  - On K − bar it sits between the other two on pendigits_20nan, kr-vs-kp_40nan and biodeg_80nan.
- **How much the draw moves a contrast (descriptive):**
  - On pendigits_20nan, kr-vs-kp_40nan and credit-g_60nan, the span of the three draws' means is
    0.2 to 1.5 times a single draw's interval half-width.
  - On biodeg_80nan it is 1.5 to 1.9 times. There, draw 2 puts K − strong at +0.0645, against
    about +0.051 on the other two.
  - **For the thesis:** an interval over folds and seeds is conditional on the draw. The draw
    adds variation of the same order, about twice as much on the small table at 80nan. No call
    depends on it.
- **biodeg_80nan at 42/7/13, for the first time:** worse than TabPFN on every draw, as E55 found
  on the confirmation seeds.

**What it leads to (fixed in the ticket):**

- The thesis can state that the calls do not depend on the MCAR draw on these four tables. They
  span the sizes, both column kinds and the four rates.
- It adds a caveat: the per-variant intervals are conditional on the draw.
- No default changes.
