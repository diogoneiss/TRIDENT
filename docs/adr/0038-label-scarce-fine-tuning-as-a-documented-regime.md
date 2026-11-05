# ADR 0038: Label-scarce fine-tuning as a documented regime for classification

- **Status:** **accepted 2026-11-05** by the user ("aceito a adr"), recorded 16:58 GMT-3; proposed the same day
  at 16:02 GMT-3 from E138, E140, E141, E143 and E144.
- **Builds on:** ADR 0029 (classification step zero), ADR 0032 (TabPFN v2 as a separate strong tier), ADR 0035
  (studies split between servers by seed). It amends none of them and changes no default.

## Context

- **With every label, pre-training buys classification nothing detectable:** E71 (the fork's sweep), E118 (the
  upstream defaults), E131 (the C1 files, S − B +0.004 [−0.008, +0.016]).
- **Two flags, both off by default,** make fine-tuning label-scarce: `FINETUNE_LABEL_FRACTION` (c997932) and
  `FINETUNE_SCARCE_VALIDATION` (5ffc28d). With them off a run is bit-identical to before (the `vehicle_00nan`
  regression fixture passes).
- **With them on, pre-training on every row pays.** S pre-trains and fine-tunes; B is the same model with
  `EPOCHS_PRE 0`. The readers' summary lines, unedited (`results_digest.py E138 E140 E141 E143 E144`; per-variant
  lines in the tickets [c1-split/09](../tickets/c1-split/09-scarce-confirmation.md) to
  [13](../tickets/c1-split/13-selector-squares.md)):

```text
## Scarce-label confirmation (E138): S25V − B25V (a quarter of the training AND validation labels), seeds 1515/2626/3737 × 5 folds, pooled over the eight files; holds if the plain 95% interval is above 0
Pooled S25V − B25V: +0.0218 [+0.0090, +0.0346] (NB [-0.0077, +0.0512]); p 0.0026 -> HOLDS; MNAR half +0.0178, MCAR half +0.0258
## Label-fraction curve (E140): S(f)V − B(f)V, validation scarce too, seeds 1515/2626/3737 × 5 folds, pooled over the eight files; Holm over the two new fractions; E138's quarter beside
Gate 10% pooled: +0.0290 [+0.0165, +0.0416] (NB [+0.0002, +0.0579]); p 0.0002, Holm 0.0004 -> channel OPEN
Gate 50% pooled: +0.0157 [+0.0075, +0.0239] (NB [-0.0031, +0.0345]); p 0.0010, Holm 0.0010 -> channel OPEN
Beside, 25% (E138): +0.0218 [+0.0090, +0.0346]
## Stage 2 (E141): S25V − B25V on the 21 confirmatory variants, seeds 42/7/13/101/202 × 5 folds (25 replicates a variant), pooled over the 21; holds if the plain 95% interval is above 0; groups beside
Pooled over 21: +0.0101 [+0.0045, +0.0157] (NB [-0.0060, +0.0262]) -> HOLDS
  group categorical (credit-g, kr-vs-kp): +0.0087 [+0.0003, +0.0171]
  group small numeric (vehicle, biodeg, kc2): +0.0080 [-0.0040, +0.0201]
  group spambase: +0.0118 [+0.0061, +0.0175]
  group large (pendigits, letter, electricity): +0.0159 [+0.0127, +0.0190]
## Missingness levels (E143): S25V − B25V at 20/40/80nan (MCAR files of the four tables), seeds 1515/2626/3737 × 5 folds, pooled over the four tables per level; Holm over the three levels; E138's 60nan MCAR half beside
Level 20nan pooled: +0.0246 [+0.0048, +0.0445] (NB [-0.0210, +0.0702]); p 0.0185, Holm 0.0392 -> channel OPEN
Level 40nan pooled: +0.0234 [+0.0057, +0.0410] (NB [-0.0172, +0.0640]); p 0.0131, Holm 0.0392 -> channel OPEN
Level 80nan pooled: +0.0040 [-0.0164, +0.0245] (NB [-0.0430, +0.0511]); p 0.6786, Holm 0.6786 -> channel closed
Beside, 60nan MCAR (E138): +0.0258 [+0.0083, +0.0432]
## Selector squares (E144): (S(f)V − B(f)V) − (S(f) − B(f)) paired per (file, seed, fold), seeds 42/7/13 and 1515/2626/3737, pooled over the eight files; Holm over the two fractions; the four cell means beside
  10% cell means, S − B pooled over files and six seeds: scarce +0.0172, oracle +0.0228
  25% cell means, S − B pooled over files and six seeds: scarce +0.0142, oracle +0.0198
Selector effect 10% pooled: -0.0056 [-0.0194, +0.0081] (NB [-0.0482, +0.0369]); p 0.4098, Holm 0.5737 -> not supported
Selector effect 25% pooled: -0.0056 [-0.0161, +0.0050] (NB [-0.0382, +0.0270]); p 0.2868, Holm 0.5737 -> not supported
```

- **On the 21 variants** 8 are better and none is worse (spambase at four levels, pendigits, letter, electricity,
  kr-vs-kp_40nan; ticket 11).
- **Every cell overrode `PRETRAIN_OBJECTIVE` to `embedding_normalized`** (the user's C1 decision of 2026-11-04);
  the classification default objective was not run at scarce labels.
- **No baseline was run at scarce labels.** The contrast is internal, S against B.

## Decision

1. **The regime's definition.** Pre-training sees every row of the fold's training split, labels unused.
   Fine-tuning sees a stratified fraction f of that split's labels (`FINETUNE_LABEL_FRACTION f`), drawn once per
   fold by `train_test_split(..., train_size=f, stratify=labels, random_state=0)` after preprocessing and before
   the class weights. The draw depends on the rows alone, so two arms of a pair see the same labels.
2. **The deployable form draws the validation split the same way** (`FINETUNE_SCARCE_VALIDATION on`). A thesis
   claim about the regime uses this form. The whole-validation form is a diagnostic, since it chooses the kept epoch
   with labels a label-scarce user would not have.
3. **No default changes.** The flags stay at `1.0` and `off`; no promoted file changes; classification stays
   bit-identical.
4. **Tags.** `finetune_label_fraction` and `finetune_scarce_validation` are run tags on every classification parent
   and trial (`runner.py`), by the standing rule for training-behaviour flags. Runs recorded before them are stamped
   `1.0` and `off` and marked `finetune_label_backfilled=true` by `scripts/backfill_finetune_label_tags.py`: applied
   2026-11-05 17:34 GMT-3 through the leader's server on 2906 runs (1986 parents, 920 Optuna trials) and their 2888
   active mirrors. The 18 without an active mirror are 16 runs a kill left `RUNNING` and 2 failed runs whose mirrors
   were deleted. No run in the store had used another fraction. E136–E144 ran with MLflow off, from readout files,
   and carry no tag.
5. **The claim's scope.** Fractions 0.1 to 0.5; 20 to 60nan (closed at 80nan); MCAR on the 21 confirmatory variants
   and probit MNAR on four tables; `embedding_normalized` pre-training.

## What this does not decide

- **The objective.** Whether the classification default objective opens the same channel is untested.
- **The split.** S against a tied `[NULL]`/`[MASK]` token (T46) stays closed: Inconclusive under MNAR, behind under
  MCAR at a quarter of the labels (E136).
- **Any standing against a baseline or TabPFN at scarce labels.** None was measured.

## Consequences

- The thesis may state the scarce-label sentence (consolidated report, section 3d) with its caveats: the
  Nadeau–Bengio interval on the 21 variants crosses 0, and the small numeric tables are undecided.
- A later study at scarce labels pairs with these cells by using the same flags, fraction and seeds.
