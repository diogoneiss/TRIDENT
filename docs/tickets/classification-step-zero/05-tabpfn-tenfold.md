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

## Amendments

- **Amendment 2026-10-29 (about 10:35 GMT-3): the GPU was lost mid-run.** medusa1's GPU0
  (`41:00.0`) failed at 04:52 GMT-3, and since then CUDA does not start on either GPU. E87 had ended
  at 03:35 GMT-3 and is not affected.
  - 85 pairs had finished on the GPU before 04:52; they stand.
  - The script fell back to the CPU: 54 pairs failed (TabPFN refuses over 1,000 rows on the CPU),
    and 61 small-table pairs were written from the CPU. A CPU fit differs fold by fold from a GPU one
    (E86), so the 61 were moved, not deleted, to `quarantine-cpu/` in the state directory and are
    not read. pendigits_60nan_s13, in flight at 04:52, is among them.
  - `run_pair` now exits without a CUDA device. The 115 missing pairs run when the GPU is back; the
    design and the rule are unchanged.

## Outcome

**Done 2026-10-29, 21:50 GMT-3: supported for both arms.** All 200 pairs ran on the GPU; the 61
quarantined CPU pairs were rerun and stay unread. `tabpfn_classification.py --study e92 --tally`:

| level | plain − TRIDENT | balanced − TRIDENT | balanced − plain |
|---|---|---|---|
| 0nan | +0.0166 [+0.0118, +0.0215] ahead | +0.0182 [+0.0145, +0.0219] ahead | +0.0016 [−0.0021, +0.0053] equivalent |
| 20nan | +0.0098 [+0.0058, +0.0139] ahead | +0.0210 [+0.0178, +0.0242] ahead | +0.0111 [+0.0075, +0.0148] ahead |
| 40nan | +0.0111 [+0.0050, +0.0171] ahead | +0.0204 [+0.0164, +0.0245] ahead | +0.0094 [+0.0047, +0.0140] ahead |
| 60nan | +0.0123 [+0.0058, +0.0188] ahead | +0.0282 [+0.0237, +0.0327] ahead | +0.0159 [+0.0108, +0.0209] ahead |
| 80nan | +0.0114 [+0.0045, +0.0182] ahead | +0.0369 [+0.0311, +0.0427] ahead | +0.0255 [+0.0203, +0.0307] ahead |

Every seed agrees in every "ahead" cell. TabPFN ahead at 5 of 5 levels for both arms, TRIDENT at
none.

- **credit-g:** balanced 0.693, 0.669, 0.635, 0.592, 0.573 against TRIDENT 0.673, 0.641, 0.637,
  0.569, 0.520 (plain falls to 0.454 at 80nan). Balancing turns E83's TRIDENT lead into a TabPFN
  lead, except a tie at 40nan.
- **electricity:** balanced 0.736, 0.678, 0.604 against TRIDENT 0.734, 0.678, 0.602 at 40–80nan:
  a tie where E83 had TRIDENT ahead.
- **Reading:** TRIDENT's E83 wins came from its class-weighted loss against an unbalanced TabPFN.
  The balanced arm is the fairer strong tier for ADR 0032; switching it is the user's decision.
