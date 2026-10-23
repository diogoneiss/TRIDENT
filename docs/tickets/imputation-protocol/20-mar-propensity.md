# Do decode masks that follow the missingness help under MAR? A propensity-mask arm (E65; pre-registered 2026-10-22)

This ticket was written and committed before the study's first cell. Two looks came first, and
they are disclosed under "Before the ticket".

- **Why:** E63 found the model losing ground against TabPFN under MAR, and E64 asks whether that
  holds.
  - E63 named an untested hypothesis. The decode stage hides training cells uniformly over the
    observed ones, while under MAR the test gaps sit in rows the other columns pick out.
  - This arm makes the training masks follow the fold's own missingness.
- **The user's go:** "vamos fazer o 1 e 2 agora" (2026-10-22, about 18:55 GMT-3). This is item 2;
  E64 is item 1.
- **The code,** behind a flag, with the old behaviour as the default:
  - **`--decode_mask_weighting propensity`** (`DECODE_MASK_WEIGHTING`), on branch
    `feat/decode-mask-propensity` (commit dd33b81, pushed);
  - **`src/training/mask_propensity.py`:**
    - per feature column with at least 20 gaps and 20 observed cells in the training fold, a
      logistic model (C = 0.1) of whether a cell is missing, on the row's other columns;
    - a numerical column enters standardised with a gap at 0, plus its missing indicator; a
      categorical one as indicators of up to 10 levels, plus its missing indicator;
    - a cell's weight is its propensity over the column's mean on observed cells. The decode stage
      hides it with the uniform probability times that weight, clipped at 1.
    - The unmasked-row fallback stays uniform. With unit weights, the draw equals today's draw
      for draw.
  - **The tag** `decode_mask_weighting`. A backfill script stamps `uniform` on earlier runs; its
    `--apply` waits on the user's go and a store backup.
  - **Checks:**
    - the unit suite (427 tests with the typing gate) passes;
    - the new tests pass (weights, draw, wiring, backfill);
    - both integration fixtures pass unchanged: vehicle_00nan and credit-g_20nan, on the CPU.
- **It runs beside E64, from a worktree** (`/scratch2/diogoneiss/TRIDENT-propensity`, the branch
  above).
  - The worktree has its own virtual environment. Its `mlflow.db` and `results/baseline_cache` are
    links to the checkout's store and cache, and `datasets/processed_datasets` is a link to the
    data.
  - So `src/` in the checkout E64 imports is never touched.
  - The branch is merged into `feat/imputation-task` after both studies end.

## Design (fixed before the first cell)

- **Cells:** E64's 26 variants × seeds 404/505/606 make 78 cells of arm P.
  - Each is today's default plus `DECODE_MASK_WEIGHTING propensity`, with `<base>_<pp>nan`'s
    resolved configuration as an override, `--score_checkpoints` and `--score_calibrated`.
  - The tag is experiment=mar-propensity-2026-10-22, arm P.
- **The comparison is E64's G** on the same variant, seed and folds. TabPFN and the strong tier
  come from E64's runs and CSVs.
- **Running:** two queues (`run_mar_propensity_queue.sh`) beside E64's four, with no cutoff. State
  in `/scratch2/diogoneiss/trident-2026-10-22-mar-propensity/`.

## Measures and reading, fixed now (`mar_propensity_report.py --tally`)

On the induced `impute_score` over 15 fold pairs, with the verdict rule:

- **Per table-rate and mechanism, with the 1% SESOI of G's value:** P − G on H, K and KC.
- **The interaction, fold-paired:** (P − G on H), MAR minus control.
- **Each arm's** (H − strong), MAR minus control: P's beside G's (E64's).
- **Primary,** over the 12 table-rates at slope 1.5:
  - **supported:** P − G on H under MAR is better on 4 or more and worse on at most 1, and under
    the control it is worse on at most 1;
  - **refuted:** under MAR it is better on at most 1;
  - **mixed:** otherwise.
- **The dial,** descriptive: P − G on H under MAR on biodeg_60nan at slopes 0.75, 1.5 and 3.
- **What each reading leads to (no default changes; MAR is never known in practice):**
  - **supported:** the hypothesis stands. Before any ADR, a pre-registered check that P is neutral
    on the shipped MCAR variants, and a fresh-seed MAR confirmation.
  - **refuted:** the mask mismatch is not the mechanism. MAR against TabPFN stays a limit, with its
    cause open.
  - **mixed:** reported, and the user chooses.

## Before the ticket (disclosed)

- **The regularisation C = 0.1 was chosen on the weights alone,** before any score.
  - The measure was their spread on one training fold (seed 404, fold 1) of E64's biodeg and
    kr-vs-kp at 60nan, MAR against control.
  - At 0.1 the MAR spread is 0.44 and 0.41, against 0.22 and 0.11 on the controls. At 1.0 it is
    0.50 and 0.47, against 0.27 and 0.12.
- **One short training smoke ran** to show the path works end to end.
  - biodeg_marf_60nan, seed 404, with EPOCHS_PRE 10 and EPOCHS_DECODE 30, so not E65's
    configuration.
  - K: uniform 0.9219, propensity 0.9086. It is one seed at a tenth of the epochs, so not a
    result, but I have seen it.

## Outcome (2026-10-23, 05:43 GMT-3)

**The run:** 78 of 78 cells, 19:18 GMT-3 on 2026-10-22 to 05:42 GMT-3 on 2026-10-23, with none
failed.

- The runs carry `decode_mask_weighting=propensity`, arm P, and the branch's commit 2cfc54b.
- They ran from the worktree, beside E64, which the OOM described in E64's Outcome did not touch.

**The printout** (`scripts/experiments/mar_propensity_report.py --tally`, unedited):

```text
## `biodeg_20nan`
- MAR: P − G, H +0.0195 [-0.0080, +0.0470] none; P − G, K +0.0198 [-0.0120, +0.0517] none; P − G, KC +0.0148 [-0.0140, +0.0436] none
- control: P − G, H +0.0143 [-0.0081, +0.0367] none; P − G, K +0.0255 [-0.0002, +0.0511] none; P − G, KC +0.0295 [+0.0067, +0.0523] none
  - (P − G on H), MAR minus control: +0.0052 [-0.0340, +0.0444] none
  - G's (H − strong), MAR minus control: -0.0012 none
  - P's (H − strong), MAR minus control: +0.0040 none

## `biodeg_60nan`
- MAR: P − G, H +0.0011 [-0.0186, +0.0209] none; P − G, K +0.0107 [-0.0003, +0.0217] none; P − G, KC +0.0116 [+0.0001, +0.0230] worse
- control: P − G, H +0.0074 [-0.0066, +0.0214] none; P − G, K +0.0104 [+0.0011, +0.0196] worse; P − G, KC +0.0085 [+0.0009, +0.0161] worse
  - (P − G on H), MAR minus control: -0.0063 [-0.0338, +0.0212] none
  - G's (H − strong), MAR minus control: +0.0291 worse
  - P's (H − strong), MAR minus control: +0.0228 none

## `kc2_20nan`
- MAR: P − G, H -0.0084 [-0.0792, +0.0625] none; P − G, K +0.0370 [+0.0110, +0.0629] worse; P − G, KC +0.0366 [+0.0101, +0.0632] worse
- control: P − G, H +0.0053 [-0.0652, +0.0758] none; P − G, K -0.0091 [-0.0699, +0.0516] none; P − G, KC -0.0061 [-0.0593, +0.0471] none
  - (P − G on H), MAR minus control: -0.0137 [-0.1001, +0.0726] none
  - G's (H − strong), MAR minus control: -0.0327 none
  - P's (H − strong), MAR minus control: -0.0464 none

## `kc2_60nan`
- MAR: P − G, H -0.0112 [-0.0442, +0.0218] none; P − G, K -0.0176 [-0.0492, +0.0139] none; P − G, KC -0.0150 [-0.0458, +0.0158] none
- control: P − G, H +0.0225 [+0.0001, +0.0450] worse; P − G, K +0.0270 [+0.0059, +0.0480] worse; P − G, KC +0.0254 [+0.0083, +0.0426] worse
  - (P − G on H), MAR minus control: -0.0337 [-0.0811, +0.0136] none
  - G's (H − strong), MAR minus control: -0.0079 none
  - P's (H − strong), MAR minus control: -0.0417 none

## `vehicle_20nan`
- MAR: P − G, H +0.0155 [+0.0028, +0.0282] worse; P − G, K +0.0076 [-0.0069, +0.0222] none; P − G, KC +0.0078 [-0.0073, +0.0229] none
- control: P − G, H +0.0234 [-0.0089, +0.0558] none; P − G, K +0.0253 [-0.0048, +0.0554] none; P − G, KC +0.0253 [-0.0047, +0.0553] none
  - (P − G on H), MAR minus control: -0.0079 [-0.0417, +0.0259] none
  - G's (H − strong), MAR minus control: -0.0025 none
  - P's (H − strong), MAR minus control: -0.0105 none

## `vehicle_60nan`
- MAR: P − G, H +0.0065 [-0.0033, +0.0164] none; P − G, K -0.0007 [-0.0112, +0.0099] none; P − G, KC -0.0012 [-0.0112, +0.0087] none
- control: P − G, H +0.0200 [+0.0027, +0.0374] worse; P − G, K +0.0071 [-0.0028, +0.0170] none; P − G, KC +0.0064 [-0.0038, +0.0166] none
  - (P − G on H), MAR minus control: -0.0135 [-0.0362, +0.0092] none
  - G's (H − strong), MAR minus control: -0.0026 none
  - P's (H − strong), MAR minus control: -0.0161 none

## `kr-vs-kp_20nan`
- MAR: P − G, H +0.0076 [-0.0066, +0.0217] none; P − G, K +0.0119 [+0.0007, +0.0230] worse; P − G, KC +0.0079 [-0.0023, +0.0181] none
- control: P − G, H +0.0087 [-0.0027, +0.0201] none; P − G, K +0.0008 [-0.0120, +0.0137] none; P − G, KC +0.0062 [-0.0037, +0.0161] none
  - (P − G on H), MAR minus control: -0.0012 [-0.0166, +0.0143] none
  - G's (H − strong), MAR minus control: -0.0070 none
  - P's (H − strong), MAR minus control: -0.0081 none

## `kr-vs-kp_60nan`
- MAR: P − G, H +0.0092 [+0.0008, +0.0175] worse; P − G, K +0.0066 [-0.0022, +0.0153] none; P − G, KC +0.0097 [+0.0001, +0.0192] none
- control: P − G, H +0.0064 [-0.0040, +0.0168] none; P − G, K +0.0086 [+0.0001, +0.0171] worse; P − G, KC +0.0068 [-0.0022, +0.0158] none
  - (P − G on H), MAR minus control: +0.0028 [-0.0119, +0.0174] none
  - G's (H − strong), MAR minus control: +0.0020 none
  - P's (H − strong), MAR minus control: +0.0047 none

## `spambase_20nan`
- MAR: P − G, H +0.0306 [+0.0199, +0.0414] worse; P − G, K +0.0262 [+0.0171, +0.0354] worse; P − G, KC +0.0216 [+0.0135, +0.0298] worse
- control: P − G, H +0.0391 [+0.0271, +0.0510] worse; P − G, K +0.0449 [+0.0337, +0.0561] worse; P − G, KC +0.0325 [+0.0211, +0.0440] worse
  - (P − G on H), MAR minus control: -0.0085 [-0.0232, +0.0063] none
  - G's (H − strong), MAR minus control: +0.0004 none
  - P's (H − strong), MAR minus control: -0.0081 none

## `spambase_60nan`
- MAR: P − G, H +0.0239 [+0.0184, +0.0294] worse; P − G, K +0.0180 [+0.0132, +0.0229] worse; P − G, KC +0.0113 [+0.0071, +0.0156] worse
- control: P − G, H +0.0095 [+0.0024, +0.0166] worse; P − G, K +0.0085 [+0.0011, +0.0158] worse; P − G, KC +0.0077 [+0.0026, +0.0128] worse
  - (P − G on H), MAR minus control: +0.0145 [+0.0041, +0.0248] worse
  - G's (H − strong), MAR minus control: -0.0018 none
  - P's (H − strong), MAR minus control: +0.0127 worse

## `credit-g_20nan`
- MAR: P − G, H +0.0255 [+0.0087, +0.0424] worse; P − G, K +0.0426 [+0.0243, +0.0610] worse; P − G, KC +0.0386 [+0.0200, +0.0573] worse
- control: P − G, H +0.0232 [+0.0077, +0.0387] worse; P − G, K +0.0023 [-0.0179, +0.0225] none; P − G, KC +0.0014 [-0.0186, +0.0214] none
  - (P − G on H), MAR minus control: +0.0023 [-0.0161, +0.0207] none
  - G's (H − strong), MAR minus control: +0.0160 none
  - P's (H − strong), MAR minus control: +0.0183 none

## `credit-g_60nan`
- MAR: P − G, H +0.0122 [+0.0011, +0.0234] worse; P − G, K +0.0163 [+0.0096, +0.0229] worse; P − G, KC +0.0128 [+0.0058, +0.0198] worse
- control: P − G, H +0.0089 [-0.0060, +0.0237] none; P − G, K +0.0016 [-0.0085, +0.0117] none; P − G, KC -0.0027 [-0.0119, +0.0065] none
  - (P − G on H), MAR minus control: +0.0034 [-0.0149, +0.0216] none
  - G's (H − strong), MAR minus control: +0.0036 none
  - P's (H − strong), MAR minus control: +0.0070 none

## The dial: `biodeg_60nan`, P − G on H under MAR by slope
- slope 0p75: +0.0134 [-0.0012, +0.0279] none
- slope 1p5: +0.0011 [-0.0186, +0.0209] none
- slope 3: +0.0063 [-0.0068, +0.0194] none

- P − G on H under MAR: better 0 / no difference 6 / worse 6 (fold-pair t, seed agreement)
- P − G on H under the control: better 0 / no difference 7 / worse 5 (fold-pair t, seed agreement)

Primary, P − G on H: under MAR better on none, worse on ['vehicle_20nan', 'kr-vs-kp_60nan', 'spambase_20nan', 'spambase_60nan', 'credit-g_20nan', 'credit-g_60nan']; under the control worse on ['kc2_60nan', 'vehicle_60nan', 'spambase_20nan', 'spambase_60nan', 'credit-g_20nan'] -> refuted (fold-pair t, seed agreement)
```

**Verdict: refuted.** Under MAR, P − G on H is better on none of the 12 table-rates (the rule
needed 4) and worse on 6. Under the control it is worse on 5.

**Reading:**

- **The propensity masks do not help, and they cost accuracy under both mechanisms.** No readout
  on any table-rate is better.
  - **Worse on H under MAR:** vehicle_20nan, kr-vs-kp_60nan, spambase at both rates, credit-g at
    both rates.
  - **Worse on H under the control:** kc2_60nan, vehicle_60nan, spambase at both rates,
    credit-g_20nan.
  - spambase loses most, +0.0095 to +0.0391.
- **The mask-mismatch hypothesis is not supported.** The interaction, (P − G on H) MAR minus
  control, is no difference on 11 table-rates, and worse on spambase_60nan.
  - If uniform masks were what cost the model under MAR, P would gain under MAR relative to the
    control. It does not.
- **On biodeg_60nan,** E64's one MAR exception: G's (H − strong) MAR minus control is +0.0291,
  worse; P's is +0.0228, no difference. P − G under MAR there is +0.0011, no difference. So P does
  not close the exception either. P's smaller MAR-minus-control gap comes from its control being a
  little weaker (P − G under the control +0.0074, no difference), not from a gain under MAR.
- **The dial:** no difference at any slope.
- **Why P costs under MCAR too (a reading, not tested):**
  - Even under MCAR the fitted weights spread (E65's smoke of the weights: 0.11 to 0.22 on the
    controls).
  - So P concentrates the training masks on some rows for no reason, and trains on a less varied
    set of hidden cells.
- **With E64,** the cause of E63's smoke result needs no mechanism. E64 refuted the loss of ground
  itself.

**What it leads to (fixed in the ticket):**

- **refuted:** the mask mismatch is not the mechanism. And since E64 refuted the MAR loss of
  ground, there is no MAR limit against the strong tier to explain, beyond biodeg_60nan.
- **The flag stays,** off by default, as a documented negative result. Its tag's backfill still
  needs the user's go.
- No default changes.
