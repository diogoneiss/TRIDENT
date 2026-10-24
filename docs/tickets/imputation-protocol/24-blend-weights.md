# Per-column weights for the K + TabPFN blend, cross-fitted: do they reach past TabPFN where the equal blend could not? (E69; pre-registered 2026-10-23)

This ticket was written and committed before the study's first cell. Two smoke pairs ran before
it, written to a scratch directory and not counted (see "Launch").

- **Why:** this is T37, accepted by the user on 2026-10-23 ("Aprovo T36 e T37").
  - E60 found an equal-weight blend of K's and TabPFN's fills beat both on 10 of 15 confirmation
    variants ([15](15-tabpfn-blend.md)), and E66 found it at the rule's floor on E40's set, 3 of 9
    ([21](21-tabpfn-blend-replication.md)).
  - Where one method dominates, an equal weight lands near the better one at best. On kr-vs-kp
    and credit-g, where TabPFN leads from 40nan and E61 put 71–100% of K's gap in the categorical
    cells, the equal blend keeps the model's category and cannot reach TabPFN.
  - A weight per column is the untested lever.
- **The user's go:** "vamos com t37" (2026-10-23, about 22:50 GMT-3). It runs ahead of T35 by that
  go; TASKS had T37 after T33–T35.
- **Route (a) of T37, offline:** `scripts/experiments/blend_weights.py`. It reuses E60's fills
  (`tabpfn_blend.py`'s ledger reader and TabPFN refit) and writes nothing to the store or to `src/`.
  Route (b), weights from the validation rows' own gaps, needs fills the ledgers do not hold and is
  not run.
- **Descriptive, like E60:** no output or default changes on any reading. A hybrid output would
  need its own pre-registered rule on a fresh seed set.

## Design (fixed before the first cell)

- **Cells, 21 variant sets, 63 pairs:**
  - **confirmation:** E44's 15 confirmation variants at 707/808/909, E60's set;
  - **e40:** E40's kr-vs-kp and credit-g at 40, 60 and 80nan, at 404/505/606, E61's set (a
    subset of E66's).
- **The fills, E60's:**
  - **the model's:** K, from the run's fold ledger, asserted equal to the run's K fold by fold;
  - **TabPFN's:** v2 refitted on the same training fold and asked for the ledger's induced cells.
    The smoke pairs reproduced E60's and E66's TabPFN fold scores to 1e-16.
- **The weighted blend, cross-fitted.** All five folds' fills are collected first. For fold k and
  each column, the weight comes from the other four folds' induced cells of that column:
  - **numerical:** the model's weight w from {0, 0.25, 0.5, 0.75, 1}, the blend being
    w·model + (1 − w)·TabPFN, that minimises the summed squared error in the ledger's z units;
  - **categorical:** the model's or TabPFN's category, whichever is right on more of those cells;
  - **ties, and a column with no cells on the other folds:** 0.5 for a numerical column and the
    model for a categorical one, which is E60's equal blend.
- **Why this is the grid's best and why it does not leak:**
  - `score_cells` pools numerical cells into one RMSE and categorical ones into one accuracy, so
    squared errors and correct counts add across columns. A per-column choice is the grid's best
    on the other folds.
  - Fold k's truth never enters fold k's weights; the other folds' cells were scored by their own
    models. Because 0.5 is in the grid, the weighted blend can lose to the equal one only on the
    held-out fold.
- **Outputs, in `/scratch2/diogoneiss/trident-2026-10-23-blend-weights/`:**
  - `csv/`: per fold, `impute_score` for the model, TabPFN, the equal blend and the weighted blend;
  - `weights/`: the chosen weight per fold and column;
  - `cells/`: both fills per induced cell, so the reading can be redone offline.
- **Running:** two GPU lanes, large tables first, resumable, about 2 h (E60 and E66's per-pair
  times). Threads capped at 4, beside E68's four CPU lanes.

## Measures and reading, fixed now (`blend_weights.py --tally`)

On the induced `impute_score` over 15 fold pairs, with the verdict rule and the 1% SESOI of the
opponent's value:

- **Per variant:** weighted − equal, weighted − TabPFN, weighted − model, and the equal blend's
  own equal − TabPFN and equal − model as this run recomputes them.
- **Primary, weighted − TabPFN on the 10 hard variants,** the ones where E60's or E66's equal
  blend was not better than TabPFN: electricity_40nan (no difference), electricity_80nan,
  vehicle_80nan and biodeg_80nan (equivalent), and kr-vs-kp and credit-g at 40/60/80nan (worse).
  - **supported** (per-column weights reach past TabPFN where the equal blend could not): better on
    4 or more;
  - **refuted:** better on at most 1;
  - **mixed:** otherwise.
- **Secondary:**
  - weighted − equal over all 21: do the weights add anything at all;
  - how many variants the weighted and the equal blend each beat both the model and TabPFN on.
- **Descriptive (`--weights`):** TabPFN's mean weight per variant and kind, the spread of chosen
  weights, and how often a column's modal choice holds across its 15 folds. Which columns prefer
  TabPFN, and how stable that is.
- **What each reading leads to (no default changes):**
  - **supported:** a per-column hybrid output is worth a pre-registered test on a fresh seed set,
    for the user to decide.
  - **refuted:** E60's complementarity is the whole of it; per-column weights do not overturn
    TabPFN's lead where it holds.
  - **mixed:** reported, and the user chooses.

## Launch

- **Smoke pairs (not counted), to a scratch directory:** kc2_80nan at 707 and kr-vs-kp_40nan at
  404, run after the script and its rule were written and before this ticket. Both matched the
  run's K on every fold and reproduced E60's and E66's TabPFN and equal-blend fold scores (largest
  difference 1.1e-16). Their fold scores were seen; neither is a hard variant's only seed, and the
  rule above was in the script before they ran. Their pairs run again in the study.
- **Launched** 2026-10-23, 22:58 GMT-3, on the pre-registration commit (5b80142): two GPU lanes.

## Outcome (2026-10-24, 00:37 GMT-3)

**The run:** 63 of 63 pairs, 22:58 GMT-3 on 2026-10-23 to 00:36 GMT-3 on 2026-10-24, with none
failed. On every fold the model's score equals the run's K, and TabPFN's and the equal blend's fold
scores equal E60's and E66's within 2.2e-16.

**The printout** (`scripts/experiments/blend_weights.py --tally`, unedited):

```text
# confirmation (seeds 707/808/909)

- `pendigits_40nan` (numeric share 1.00): weighted − equal +0.0003 [+0.0000, +0.0006] equivalent; weighted − tabpfn -0.0242 [-0.0260, -0.0223] better; weighted − model -0.0134 [-0.0147, -0.0122] better; equal − tabpfn -0.0245 [-0.0261, -0.0228] better; equal − model -0.0138 [-0.0151, -0.0124] better
- `pendigits_60nan` (numeric share 1.00): weighted − equal -0.0005 [-0.0007, -0.0003] better; weighted − tabpfn -0.0307 [-0.0321, -0.0293] better; weighted − model -0.0127 [-0.0136, -0.0118] better; equal − tabpfn -0.0302 [-0.0316, -0.0289] better; equal − model -0.0122 [-0.0132, -0.0112] better
- `pendigits_80nan` (numeric share 1.00): weighted − equal -0.0012 [-0.0015, -0.0009] better; weighted − tabpfn -0.0057 [-0.0065, -0.0048] better; weighted − model -0.0219 [-0.0230, -0.0207] better; equal − tabpfn -0.0045 [-0.0055, -0.0034] better; equal − model -0.0207 [-0.0218, -0.0196] better
- `letter_40nan` (numeric share 1.00): weighted − equal -0.0038 [-0.0044, -0.0032] better; weighted − tabpfn -0.0404 [-0.0424, -0.0385] better; weighted − model -0.0060 [-0.0067, -0.0054] better; equal − tabpfn -0.0366 [-0.0380, -0.0353] better; equal − model -0.0022 [-0.0035, -0.0010] better
- `letter_60nan` (numeric share 1.00): weighted − equal -0.0026 [-0.0029, -0.0024] better; weighted − tabpfn -0.0381 [-0.0394, -0.0369] better; weighted − model -0.0081 [-0.0087, -0.0075] better; equal − tabpfn -0.0355 [-0.0366, -0.0345] better; equal − model -0.0055 [-0.0063, -0.0046] better
- `letter_80nan` (numeric share 1.00): weighted − equal -0.0002 [-0.0004, +0.0000] equivalent; weighted − tabpfn -0.0085 [-0.0093, -0.0076] better; weighted − model -0.0105 [-0.0113, -0.0096] better; equal − tabpfn -0.0083 [-0.0092, -0.0074] better; equal − model -0.0103 [-0.0111, -0.0095] better
- `electricity_40nan` (hard) (numeric share 0.88): weighted − equal +0.0033 [-0.0109, +0.0176] none; weighted − tabpfn +0.0000 [-0.0252, +0.0252] none; weighted − model -0.0304 [-0.0588, -0.0019] better; equal − tabpfn -0.0033 [-0.0172, +0.0106] none; equal − model -0.0337 [-0.0669, -0.0005] better
- `electricity_60nan` (numeric share 0.87): weighted − equal +0.0008 [-0.0007, +0.0022] equivalent; weighted − tabpfn -0.0054 [-0.0070, -0.0038] better; weighted − model -0.0091 [-0.0146, -0.0037] better; equal − tabpfn -0.0062 [-0.0084, -0.0039] better; equal − model -0.0099 [-0.0151, -0.0047] better
- `electricity_80nan` (hard) (numeric share 0.87): weighted − equal -0.0021 [-0.0039, -0.0003] better; weighted − tabpfn -0.0019 [-0.0027, -0.0012] better; weighted − model -0.0123 [-0.0192, -0.0053] better; equal − tabpfn +0.0001 [-0.0019, +0.0021] equivalent; equal − model -0.0102 [-0.0154, -0.0050] better
- `vehicle_40nan` (numeric share 1.00): weighted − equal +0.0029 [+0.0002, +0.0056] worse; weighted − tabpfn -0.0139 [-0.0212, -0.0067] better; weighted − model -0.0136 [-0.0206, -0.0066] better; equal − tabpfn -0.0168 [-0.0247, -0.0090] better; equal − model -0.0165 [-0.0226, -0.0104] better
- `vehicle_80nan` (hard) (numeric share 1.00): weighted − equal -0.0065 [-0.0094, -0.0036] better; weighted − tabpfn -0.0034 [-0.0050, -0.0017] better; weighted − model -0.0564 [-0.0640, -0.0488] better; equal − tabpfn +0.0031 [-0.0011, +0.0073] equivalent; equal − model -0.0499 [-0.0556, -0.0443] better
- `biodeg_40nan` (numeric share 1.00): weighted − equal -0.0164 [-0.0227, -0.0101] better; weighted − tabpfn -0.0732 [-0.0958, -0.0506] better; weighted − model -0.0318 [-0.0412, -0.0225] better; equal − tabpfn -0.0568 [-0.0766, -0.0371] better; equal − model -0.0154 [-0.0253, -0.0056] better
- `biodeg_80nan` (hard) (numeric share 1.00): weighted − equal -0.0108 [-0.0130, -0.0085] better; weighted − tabpfn -0.0113 [-0.0169, -0.0058] better; weighted − model -0.0638 [-0.0713, -0.0563] better; equal − tabpfn -0.0005 [-0.0067, +0.0056] equivalent; equal − model -0.0530 [-0.0602, -0.0459] better
- `kc2_40nan` (numeric share 1.00): weighted − equal +0.0003 [-0.0268, +0.0274] none; weighted − tabpfn -0.0493 [-0.0850, -0.0135] none; weighted − model -0.0476 [-0.0778, -0.0174] better; equal − tabpfn -0.0496 [-0.0827, -0.0164] better; equal − model -0.0479 [-0.0821, -0.0137] better
- `kc2_80nan` (numeric share 1.00): weighted − equal -0.0131 [-0.0300, +0.0037] none; weighted − tabpfn -0.0820 [-0.1194, -0.0447] better; weighted − model -0.0243 [-0.0391, -0.0095] better; equal − tabpfn -0.0689 [-0.0925, -0.0453] better; equal − model -0.0112 [-0.0405, +0.0182] none

# e40 (seeds 404/505/606)

- `kr-vs-kp_40nan` (hard) (numeric share 0.00): weighted − equal -0.0087 [-0.0136, -0.0037] better; weighted − tabpfn +0.0024 [-0.0008, +0.0055] equivalent; weighted − model -0.0087 [-0.0136, -0.0037] better; equal − tabpfn +0.0110 [+0.0054, +0.0167] worse; equal − model +0.0000 [+0.0000, +0.0000] equivalent
- `kr-vs-kp_60nan` (hard) (numeric share 0.00): weighted − equal -0.0418 [-0.0484, -0.0351] better; weighted − tabpfn +0.0016 [+0.0005, +0.0027] worse; weighted − model -0.0418 [-0.0484, -0.0351] better; equal − tabpfn +0.0434 [+0.0366, +0.0501] worse; equal − model +0.0000 [+0.0000, +0.0000] equivalent
- `kr-vs-kp_80nan` (hard) (numeric share 0.00): weighted − equal -0.0364 [-0.0459, -0.0270] better; weighted − tabpfn +0.0000 [-0.0018, +0.0019] equivalent; weighted − model -0.0364 [-0.0459, -0.0270] better; equal − tabpfn +0.0365 [+0.0271, +0.0458] worse; equal − model +0.0000 [+0.0000, +0.0000] equivalent
- `credit-g_40nan` (hard) (numeric share 0.35): weighted − equal -0.0121 [-0.0182, -0.0061] better; weighted − tabpfn +0.0018 [-0.0060, +0.0096] none; weighted − model -0.0202 [-0.0270, -0.0133] better; equal − tabpfn +0.0139 [+0.0063, +0.0215] worse; equal − model -0.0080 [-0.0101, -0.0059] better
- `credit-g_60nan` (hard) (numeric share 0.35): weighted − equal -0.0313 [-0.0408, -0.0218] better; weighted − tabpfn -0.0011 [-0.0035, +0.0013] equivalent; weighted − model -0.0402 [-0.0504, -0.0299] better; equal − tabpfn +0.0302 [+0.0200, +0.0405] worse; equal − model -0.0089 [-0.0105, -0.0072] better
- `credit-g_80nan` (hard) (numeric share 0.35): weighted − equal -0.0094 [-0.0144, -0.0045] better; weighted − tabpfn +0.0001 [-0.0028, +0.0030] equivalent; weighted − model -0.0138 [-0.0184, -0.0091] better; equal − tabpfn +0.0095 [+0.0041, +0.0150] worse; equal − model -0.0043 [-0.0056, -0.0031] better

- weighted − equal: better 14 / no difference 3 / equivalent 3 / worse 1 (fold-pair t, seed agreement)
- weighted − tabpfn: better 13 / no difference 3 / equivalent 4 / worse 1 (fold-pair t, seed agreement)
- weighted − model: better 21 / no difference 0 / worse 0 (fold-pair t, seed agreement)
- equal − tabpfn: better 11 / no difference 1 / equivalent 3 / worse 6 (fold-pair t, seed agreement)
- equal − model: better 17 / no difference 1 / equivalent 3 / worse 0 (fold-pair t, seed agreement)
- weighted beats both the model and TabPFN on 13: ['pendigits_40nan', 'pendigits_60nan', 'pendigits_80nan', 'letter_40nan', 'letter_60nan', 'letter_80nan', 'electricity_60nan', 'electricity_80nan', 'vehicle_40nan', 'vehicle_80nan', 'biodeg_40nan', 'biodeg_80nan', 'kc2_80nan']
- equal beats both the model and TabPFN on 10: ['pendigits_40nan', 'pendigits_60nan', 'pendigits_80nan', 'letter_40nan', 'letter_60nan', 'letter_80nan', 'electricity_60nan', 'vehicle_40nan', 'biodeg_40nan', 'kc2_40nan']

Primary, weighted − TabPFN on the 10 hard variants: better on ['electricity_80nan', 'vehicle_80nan', 'biodeg_80nan'], worse on ['kr-vs-kp_60nan'] -> mixed (fold-pair t, seed agreement)
```

**The weights** (`--weights`, unedited):

```text
- `biodeg_40nan` numerical (41 columns): TabPFN's mean weight 0.57; model weights {0: 98, 0.25: 198, 0.5: 170, 0.75: 81, 1: 68}; a column's modal choice holds on 71% of its folds
- `biodeg_80nan` numerical (41 columns): TabPFN's mean weight 0.76; model weights {0: 239, 0.25: 238, 0.5: 86, 0.75: 22, 1: 30}; a column's modal choice holds on 71% of its folds
- `credit-g_40nan` numerical (7 columns): TabPFN's mean weight 0.68; model weights {0: 17, 0.25: 45, 0.5: 39, 0.75: 4}; a column's modal choice holds on 57% of its folds
- `credit-g_40nan` categorical (13 columns): TabPFN's mean weight 0.68; model weights {0: 133, 1: 62}; a column's modal choice holds on 80% of its folds
- `credit-g_60nan` numerical (7 columns): TabPFN's mean weight 0.64; model weights {0: 17, 0.25: 37, 0.5: 39, 0.75: 12}; a column's modal choice holds on 68% of its folds
- `credit-g_60nan` categorical (13 columns): TabPFN's mean weight 0.67; model weights {0: 130, 1: 65}; a column's modal choice holds on 88% of its folds
- `credit-g_80nan` numerical (7 columns): TabPFN's mean weight 0.65; model weights {0: 29, 0.25: 26, 0.5: 32, 0.75: 13, 1: 5}; a column's modal choice holds on 58% of its folds
- `credit-g_80nan` categorical (13 columns): TabPFN's mean weight 0.46; model weights {0: 90, 1: 105}; a column's modal choice holds on 91% of its folds
- `electricity_40nan` numerical (7 columns): TabPFN's mean weight 0.51; model weights {0: 16, 0.25: 12, 0.5: 40, 0.75: 36, 1: 1}; a column's modal choice holds on 84% of its folds
- `electricity_40nan` categorical (1 columns): TabPFN's mean weight 0.00; model weights {1: 15}; a column's modal choice holds on 100% of its folds
- `electricity_60nan` numerical (7 columns): TabPFN's mean weight 0.59; model weights {0: 4, 0.25: 38, 0.5: 54, 0.75: 9}; a column's modal choice holds on 79% of its folds
- `electricity_60nan` categorical (1 columns): TabPFN's mean weight 0.00; model weights {1: 15}; a column's modal choice holds on 100% of its folds
- `electricity_80nan` numerical (7 columns): TabPFN's mean weight 0.73; model weights {0: 30, 0.25: 40, 0.5: 33, 0.75: 2}; a column's modal choice holds on 68% of its folds
- `electricity_80nan` categorical (1 columns): TabPFN's mean weight 0.00; model weights {1: 15}; a column's modal choice holds on 100% of its folds
- `kc2_40nan` numerical (21 columns): TabPFN's mean weight 0.48; model weights {0: 77, 0.25: 54, 0.5: 53, 0.75: 29, 1: 102}; a column's modal choice holds on 76% of its folds
- `kc2_80nan` numerical (21 columns): TabPFN's mean weight 0.23; model weights {0: 12, 0.25: 26, 0.5: 61, 0.75: 39, 1: 177}; a column's modal choice holds on 77% of its folds
- `kr-vs-kp_40nan` categorical (36 columns): TabPFN's mean weight 0.58; model weights {0: 313, 1: 227}; a column's modal choice holds on 78% of its folds
- `kr-vs-kp_60nan` categorical (36 columns): TabPFN's mean weight 0.79; model weights {0: 429, 1: 111}; a column's modal choice holds on 88% of its folds
- `kr-vs-kp_80nan` categorical (36 columns): TabPFN's mean weight 0.57; model weights {0: 310, 1: 230}; a column's modal choice holds on 92% of its folds
- `letter_40nan` numerical (16 columns): TabPFN's mean weight 0.25; model weights {0.5: 4, 0.75: 232, 1: 4}; a column's modal choice holds on 97% of its folds
- `letter_60nan` numerical (16 columns): TabPFN's mean weight 0.29; model weights {0.5: 42, 0.75: 198}; a column's modal choice holds on 93% of its folds
- `letter_80nan` numerical (16 columns): TabPFN's mean weight 0.54; model weights {0.25: 58, 0.5: 159, 0.75: 23}; a column's modal choice holds on 86% of its folds
- `pendigits_40nan` numerical (16 columns): TabPFN's mean weight 0.46; model weights {0.25: 1, 0.5: 199, 0.75: 40}; a column's modal choice holds on 88% of its folds
- `pendigits_60nan` numerical (16 columns): TabPFN's mean weight 0.40; model weights {0.5: 142, 0.75: 98}; a column's modal choice holds on 90% of its folds
- `pendigits_80nan` numerical (16 columns): TabPFN's mean weight 0.69; model weights {0.25: 187, 0.5: 53}; a column's modal choice holds on 85% of its folds
- `vehicle_40nan` numerical (18 columns): TabPFN's mean weight 0.50; model weights {0: 30, 0.25: 35, 0.5: 136, 0.75: 48, 1: 21}; a column's modal choice holds on 67% of its folds
- `vehicle_80nan` numerical (18 columns): TabPFN's mean weight 0.76; model weights {0: 62, 0.25: 158, 0.5: 50}; a column's modal choice holds on 73% of its folds
```

**Verdict: mixed.** Weighted − TabPFN on the 10 hard variants is better on 3 (electricity_80nan,
vehicle_80nan, biodeg_80nan) and worse on 1 (kr-vs-kp_60nan, +0.0016). Supported needed 4; refuted
needed at most 1 better. By the ticket, the user chooses.

**Reading:**

- **On the numerical tables the weights turn TabPFN's 80nan ties into wins.** The three 80nan
  variants where the equal blend was equivalent to TabPFN are better than it with the weights, by
  0.002–0.011. The weighted blend beats both the model and TabPFN on 13 of 21 variants, against
  the equal blend's 10 (kc2_40nan drops out on seed agreement; electricity_80nan, vehicle_80nan,
  biodeg_80nan and kc2_80nan come in).
- **On the categorical tables a per-column pick reaches TabPFN, not past it.** On kr-vs-kp and
  credit-g the equal blend was worse than TabPFN on all six; the weighted blend is equivalent on
  four, no different on credit-g_40nan, and worse by 0.0016 on kr-vs-kp_60nan. It takes TabPFN's
  category on 58–79% of kr-vs-kp's columns, and the columns it keeps from K do not add a margin.
  E61's categorical deficit is closed by choosing TabPFN, not overturned.
- **The weights add almost everywhere and never cost the model:** weighted − equal is better on 14
  of 21 and worse on 1 (vehicle_40nan, +0.003); weighted − model is better on all 21.
- **The weights are stable and follow the missingness:** a column's modal choice holds on 57–100%
  of its 15 folds (86–97% on pendigits and letter). TabPFN's mean weight rises with missingness on
  pendigits (0.46 to 0.69), letter (0.25 to 0.54), vehicle (0.50 to 0.76) and biodeg (0.57 to
  0.76), and falls on kc2 (0.48 to 0.23), where K leads.
- **For the user (mixed):** a per-column K + TabPFN hybrid is the strongest imputer measured here
  (better than both on 13 of 21), but it does not beat TabPFN where TabPFN leads on categorical
  cells. Making it an output would need its own pre-registered rule on a fresh seed set (T37's
  note). No default changes.

