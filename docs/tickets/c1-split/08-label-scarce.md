# Does pre-training open a channel when labels are scarce? S, T and B on a tenth and a quarter of the training labels (E136; pre-registered 2026-11-04)

This ticket was written and committed before the cut's first cell (22:33 GMT-3).

- **Why:** T46 closed on 2026-11-04 as *Inconclusive for want of a channel*: pre-training leaves about +0.01 to a
  frozen head (E132) and fine-tuning rebuilds +0.10 from any start, so S ≈ T ≈ B (E131) and the split's claim could
  not be tested. Every C1 cell fine-tuned on every training label. The regime where tabular pre-training is
  expected to matter is the one with few labels: the encoder sees every row, the head few. The user reopened T46
  there ("vamos no 1", 2026-11-04, about 21:45 GMT-3).
- **Code:** `FINETUNE_LABEL_FRACTION` (`src/training/types.py`, `finetuning.py`): the share of the fold's
  training rows whose labels fine-tuning sees; 1.0 (every run before it) keeps them all. A stratified draw fixed by
  the rows (`random_state` 0, sorted), so S, T and B of one fold see the *same* labelled rows and the run's seed
  enters through the fold split alone. Sliced after `preprocess_table` (which fits nothing: it replaces nulls by the
  token) and before the class weights, which come from the kept labels. Pre-training (all training rows, no
  labels), validation and test are untouched. The readout records `labelled_rows`. Unit tests
  (`tests/unit/test_c1_scarce.py`): fraction 1.0 is the identity, the draw is fixed, every class survives a tenth of a
  kc2-like table, the flag validates. Arms and cut `8` in `c1_run.py`; reader `c1_report.py --scarce`.
- **Smoke, outside the study (seed 42, biodeg_probit_60nan, `C1_DETERMINISTIC=1`):** B10 twice, bit-identical test
  probabilities on 5 of 5 folds; 73 labelled rows on both B10 and S10 (a tenth of 738). Deleted.

## Design (fixed before the first cell)

- **Arms, on the four MNAR files and their MCAR siblings, decision mode (seeds 42/7/13 × 5 folds):** **S10**, **T10**,
  **B10** (fraction 0.1) and **S25**, **T25**, **B25** (0.25); otherwise E131's configuration (`embedding_normalized`,
  150 fine-tuning epochs, `cosine_legacy`). 144 cells. S, T and B at fraction 1.0 are E124's and E131's cells.
- **Labelled rows a fold (seed 42, fold 1; per class):** biodeg 73 [48, 25] / 184 [122, 62]; kc2 36 [29, 7] /
  91 [72, 19]; vehicle 59 [15, 15, 15, 14] / 147 [38, 37, 37, 35]; spambase 322 [195, 127] / 805 [488, 317].
- **Machines (ADR 0035):** by `c15_run.machine`, as every C1 cut (gorgona8 seed 42 of kc2 and spambase).
- **A fact fixed now:** at a tenth, every file but spambase has one batch an epoch, so `cosine_legacy`'s sawtooth
  (period `EPOCH_FINE` steps) becomes one 150-epoch annealing; shared by the three arms, so not a confound for
  S − B or S − T, but a different regime from E131's, whose kept epochs do not carry over.

## Rule, fixed now

- **Gate G2(f), two tests as one family:** S(f) − B(f) in test macro-F1, 15 paired replicates a file, pooled over
  the eight files per replicate, plain 95% t interval (14 df), Nadeau–Bengio beside; p from the paired t test, Holm
  over the two fractions. The channel is **open** at f if Holm p < 0.05 with a positive mean; otherwise **closed**.
- **Primary, read only at a fraction where the gate is open:** S(f) − T(f) on the four MNAR files, pooled, with C1's
  verdicts (ticket 03: Supported / Reversed / Null / Inconclusive on the plain interval, ±0.010 as the band); the
  MCAR files as the fixed-sequence second step; the interaction (S − T) MNAR − MCAR beside. Per-file lines and log
  loss are flags. T's cells run now so no round trip is needed; the gate is a reading rule, declared before the first
  cell.
- **Beside, descriptive:** S(f) − S(1.0), the cost of scarcity; `labelled_rows` per file, to audit the fraction.
- **What follows:** a channel open at f with S − T Supported reopens C1's claim in the label-scarce regime, to be
  confirmed on fresh seeds in its own ticket; a channel open with S − T Null or Reversed closes the claim where it
  could be tested; no channel at either fraction closes C1 in this regime too, and the lever is pre-training itself.
- **Threats stated now:** the validation split keeps every label and selects the kept epoch for all three arms (an
  oracle selector shared by the arms: this reads pre-training's content under scarce labels, not a deployable
  semi-supervised protocol, whose selector would be scarce too); kc2 at a tenth has 7 positives, so its lines are
  noisy; 0.1 and 0.25 are the first fractions tried; one base rate and budget for every arm.
- **Read only with all 144 cells done.**
