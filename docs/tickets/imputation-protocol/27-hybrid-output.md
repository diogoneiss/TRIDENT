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
- **Launched** 2026-10-24, 12:10 GMT-3, on the pre-registration commit (6af05e3).

## Outcome (2026-10-24, 17:15 GMT-3)

**The run:** 30 of 30 training cells (12:10–16:49 GMT-3) and 30 of 30 blend pairs (16:49–17:13),
none failed; every fold's ledger score equals the run's KC.

**The printout** (`scripts/experiments/hybrid_blend.py --tally`, unedited):

```text
- `electricity_40nan` (numeric share 0.88; KC 0.7494): weighted − model -0.0153 [-0.0352, +0.0047] none; weighted − tabpfn -0.0094 [-0.0192, +0.0003] none; weighted − equal +0.0005 [-0.0093, +0.0103] none; tabpfn − model -0.0058 [-0.0282, +0.0165] none
- `electricity_80nan` (numeric share 0.88; KC 0.8970): weighted − model -0.0087 [-0.0124, -0.0051] better; weighted − tabpfn -0.0022 [-0.0028, -0.0015] better; weighted − equal -0.0012 [-0.0023, -0.0002] better; tabpfn − model -0.0066 [-0.0101, -0.0030] better
- `vehicle_80nan` (numeric share 1.00; KC 0.7873): weighted − model -0.0374 [-0.0423, -0.0324] better; weighted − tabpfn -0.0070 [-0.0094, -0.0046] better; weighted − equal -0.0045 [-0.0068, -0.0021] better; tabpfn − model -0.0304 [-0.0365, -0.0243] better
- `biodeg_80nan` (numeric share 1.00; KC 0.9409): weighted − model -0.0342 [-0.0380, -0.0305] better; weighted − tabpfn -0.0128 [-0.0160, -0.0096] better; weighted − equal -0.0059 [-0.0079, -0.0039] better; tabpfn − model -0.0214 [-0.0270, -0.0159] better
- `kr-vs-kp_40nan` (numeric share 0.00; KC 0.7243): weighted − model -0.0051 [-0.0106, +0.0004] none; weighted − tabpfn +0.0053 [+0.0006, +0.0100] worse; weighted − equal -0.0051 [-0.0106, +0.0004] none; tabpfn − model -0.0104 [-0.0190, -0.0017] better
- `kr-vs-kp_60nan` (numeric share 0.00; KC 0.8572): weighted − model -0.0275 [-0.0331, -0.0218] better; weighted − tabpfn +0.0036 [+0.0002, +0.0070] worse; weighted − equal -0.0275 [-0.0331, -0.0218] better; tabpfn − model -0.0311 [-0.0377, -0.0244] better
- `kr-vs-kp_80nan` (numeric share 0.00; KC 0.9700): weighted − model -0.0263 [-0.0321, -0.0206] better; weighted − tabpfn +0.0007 [-0.0012, +0.0025] equivalent; weighted − equal -0.0263 [-0.0321, -0.0206] better; tabpfn − model -0.0270 [-0.0330, -0.0210] better
- `credit-g_40nan` (numeric share 0.35; KC 0.9235): weighted − model -0.0183 [-0.0255, -0.0112] better; weighted − tabpfn +0.0010 [-0.0042, +0.0061] equivalent; weighted − equal -0.0120 [-0.0180, -0.0059] better; tabpfn − model -0.0193 [-0.0287, -0.0098] better
- `credit-g_60nan` (numeric share 0.35; KC 0.9813): weighted − model -0.0306 [-0.0383, -0.0229] better; weighted − tabpfn -0.0017 [-0.0052, +0.0018] equivalent; weighted − equal -0.0262 [-0.0335, -0.0188] better; tabpfn − model -0.0289 [-0.0380, -0.0199] better
- `credit-g_80nan` (numeric share 0.35; KC 1.0054): weighted − model -0.0062 [-0.0107, -0.0016] none; weighted − tabpfn -0.0009 [-0.0057, +0.0039] equivalent; weighted − equal -0.0048 [-0.0091, -0.0005] equivalent; tabpfn − model -0.0053 [-0.0114, +0.0009] none

- weighted − model: better 7 / no difference 3 / worse 0 (fold-pair t, seed agreement)
- weighted − tabpfn: better 3 / no difference 1 / equivalent 4 / worse 2 (fold-pair t, seed agreement)
- weighted − equal: better 7 / no difference 2 / equivalent 1 / worse 0 (fold-pair t, seed agreement)
- tabpfn − model: better 8 / no difference 2 / worse 0 (fold-pair t, seed agreement)
- the hybrid beats both KC and TabPFN on 3: ['electricity_80nan', 'vehicle_80nan', 'biodeg_80nan']

Primary: the hybrid is better than KC on 7 of 10 and worse than TabPFN on 2 (['kr-vs-kp_40nan', 'kr-vs-kp_60nan']) -> mixed (fold-pair t, seed agreement)
```

**The weights** (`blend_weights.py --weights` on the blend directory, unedited):

```text
- `biodeg_80nan` numerical (41 columns): TabPFN's mean weight 0.65; model weights {0: 171, 0.25: 219, 0.5: 111, 0.75: 47, 1: 67}; a column's modal choice holds on 69% of its folds
- `credit-g_40nan` numerical (7 columns): TabPFN's mean weight 0.64; model weights {0: 19, 0.25: 29, 0.5: 50, 0.75: 6, 1: 1}; a column's modal choice holds on 59% of its folds
- `credit-g_40nan` categorical (13 columns): TabPFN's mean weight 0.65; model weights {0: 127, 1: 68}; a column's modal choice holds on 81% of its folds
- `credit-g_60nan` numerical (7 columns): TabPFN's mean weight 0.57; model weights {0: 24, 0.25: 29, 0.5: 16, 0.75: 26, 1: 10}; a column's modal choice holds on 57% of its folds
- `credit-g_60nan` categorical (13 columns): TabPFN's mean weight 0.69; model weights {0: 134, 1: 61}; a column's modal choice holds on 88% of its folds
- `credit-g_80nan` numerical (7 columns): TabPFN's mean weight 0.42; model weights {0: 14, 0.25: 21, 0.5: 20, 0.75: 19, 1: 31}; a column's modal choice holds on 54% of its folds
- `credit-g_80nan` categorical (13 columns): TabPFN's mean weight 0.42; model weights {0: 82, 1: 113}; a column's modal choice holds on 83% of its folds
- `electricity_40nan` numerical (7 columns): TabPFN's mean weight 0.49; model weights {0: 13, 0.25: 16, 0.5: 36, 0.75: 33, 1: 7}; a column's modal choice holds on 78% of its folds
- `electricity_40nan` categorical (1 columns): TabPFN's mean weight 0.00; model weights {1: 15}; a column's modal choice holds on 100% of its folds
- `electricity_80nan` numerical (7 columns): TabPFN's mean weight 0.68; model weights {0: 17, 0.25: 43, 0.5: 44, 0.75: 1}; a column's modal choice holds on 89% of its folds
- `electricity_80nan` categorical (1 columns): TabPFN's mean weight 0.00; model weights {1: 15}; a column's modal choice holds on 100% of its folds
- `kr-vs-kp_40nan` categorical (36 columns): TabPFN's mean weight 0.53; model weights {0: 288, 1: 252}; a column's modal choice holds on 76% of its folds
- `kr-vs-kp_60nan` categorical (36 columns): TabPFN's mean weight 0.72; model weights {0: 390, 1: 150}; a column's modal choice holds on 86% of its folds
- `kr-vs-kp_80nan` categorical (36 columns): TabPFN's mean weight 0.56; model weights {0: 300, 1: 240}; a column's modal choice holds on 91% of its folds
- `vehicle_80nan` numerical (18 columns): TabPFN's mean weight 0.67; model weights {0: 49, 0.25: 106, 0.5: 103, 0.75: 5, 1: 7}; a column's modal choice holds on 73% of its folds
```

**Verdict: mixed.** The hybrid is better than KC on 7 of 10 (supported needed 8) and worse than
TabPFN on 2, kr-vs-kp_40nan (+0.005) and kr-vs-kp_60nan (+0.004) (supported allowed 1). By the
ticket, the user chooses.

**Reading:**

- **On these 10 variants TabPFN alone already beats today's default:** TabPFN − KC is better on 8
  and no different on 2. The hybrid is never worse than KC (better on 7, no difference on 3).
- **E69's numerical result replicates exactly:** the hybrid beats both KC and TabPFN on the same
  three variants as E69 (electricity_80nan, vehicle_80nan, biodeg_80nan), by 0.002–0.013 over
  TabPFN.
- **On the categorical tables it tracks TabPFN, slightly behind on kr-vs-kp:** equivalent on
  kr-vs-kp_80nan and credit-g at all three rates, worse by 0.004–0.005 on kr-vs-kp at 40 and
  60nan, where it takes TabPFN's category on 53–72% of column-folds and K's picks cost a little.
- **The weights replicate E69's pattern:** TabPFN's weight is highest at 80nan on the numerical
  tables (0.65–0.68) and a column's choice holds on 54–100% of its folds.
- **For the user (mixed):** as an output the hybrid is a modest gain over TabPFN on numerical cells
  at 80nan, and TabPFN-level elsewhere; most of its gain over today's default is TabPFN's own. An
  opt-in ADR would rest on "never worse than KC, best of the three on numerical 80nan", not on the
  rule's supported reading. No default changes.

