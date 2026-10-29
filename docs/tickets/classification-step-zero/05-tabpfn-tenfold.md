# TabPFN v2's classifier on E87's ten folds, plain and class-balanced (E92; pre-registered 2026-10-28)

This ticket was written and committed before the study's first fit.

- **Why (T40):** E83 put untuned TabPFN v2 about 2 macro-F1 points ahead of TRIDENT at every level,
  on E71's 3 folds and 2 seeds. TRIDENT led only on credit-g (from 20nan) and electricity (from
  60nan), where TabPFN predicted the majority class and TRIDENT's class-weighted loss traded
  accuracy for macro-F1. E83 left one thing open: a fair macro-F1 comparison should class-balance
  TabPFN too.
  - E87 is training TRIDENT on ten folds and five seeds. Its arm A gives TabPFN a partner with
    E87's power.
- **The user's go:** "ok, encadeie esse trabalho pra ca, assim como o desenvolvimento do 2"
  (2026-10-28, about 23:15 GMT-3): chain the plain arm (item 1) and the class-balanced arm (item 2)
  after E87 on medusa1.
- **Code:** no change to `src/`, no training, nothing in the store.
  - `scripts/experiments/tabpfn_classification.py --study e92` has `run`, `--todo <n>` and
    `--tally`. E83's path (no `--study`) is unchanged.
  - `scripts/experiments/run_tabpfn_tenfold_queue.sh <n>`: two GPU queues that wait until no
    `run_cls_tenfold_queue.sh` is left, then run their pairs.
- **Smoke, outside the study (seed 1):** credit-g_60nan and electricity_40nan, 10 folds each,
  about 2 min in all, with E87's eight queues holding both GPUs.
  - Shapes: both arms' columns are written and finite. The CSVs were deleted.
  - The numbers were seen: on credit-g_60nan the balanced arm's macro-F1 was above the plain arm's
    on every fold shown. Seed 1 is not part of the study.

## Design (fixed before the first fit)

- **Cells:** E87's 8 tables with at most 10 classes (pendigits, electricity, vehicle, kr-vs-kp,
  spambase, credit-g, biodeg, kc2; letter is out) × 5 levels (0/20/40/60/80nan, MCAR) × seeds 42,
  7, 13, 101 and 202 = 200 pairs, 10 folds each.
- **Folds:** `build_folds(frame, label, 10, seed)`, the split E87's runs use.
- **Settings:** E83's, from `tabpfn_baseline`: v2's checkpoints, 8 estimators, `random_state` 0,
  the GPU, the training split as context capped at 10,000 rows by a stratified fixed-seed subsample.
- **Arms**, from one fit per fold:
  - **plain:** `predict`, as in E83;
  - **balanced:** the same fitted model with `balance_probabilities = True`, predicted again.
    TabPFN applies it only as a post-processing step, so the two arms differ in nothing else.
- **Running:** medusa1, so every pair shares a machine with E87 (E86). Two queues, one per GPU,
  launched now and waiting for E87's queues to end. State in
  `/scratch/diogoneiss/trident-2026-10-28-tabpfn-tenfold/`.
  - **Cost:** electricity takes about 100 s for 10 folds with the GPUs contended. Estimated 1–2 h
    for the 200 pairs on two queues.

## Measures and rule, fixed now (`tabpfn_classification.py --study e92 --tally`)

- **Per level and per arm:** arm − TRIDENT (E87 arm A) test macro-F1, paired fold by fold, pooled
  over the 8 tables with one array per seed (400 fold pairs). The project's rule applies
  (95% t-interval, seed agreement), with SESOI 0.01: TabPFN ahead, TRIDENT ahead, equivalent, or no
  verdict.
- **The call, per arm:**
  - **supported** (E83's lead holds) if TabPFN is ahead at 4 or more of the 5 levels and TRIDENT
    at none;
  - **refuted** if TabPFN is ahead at 1 level or none;
  - **mixed** otherwise.
- **Descriptive:** balanced − plain per level by the same rule; each table's mean macro-F1 for both
  arms and for TRIDENT, per level. That answers whether balancing closes credit-g and electricity.
- **Read only with every pair and E87 done.** The tally refuses a partial read.

## Limits fixed now

- TRIDENT's side is E87's arm A, the fork's untuned defaults, not step zero's tuned arms.
- letter is not covered. MCAR only.
- `balance_probabilities` is TabPFN's own reweighting by the context's class frequencies, not a
  class-weighted fit.

## Outcome

(open)
