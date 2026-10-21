# TabPFN v2 as an external foundation-model baseline (E55; pre-registered 2026-10-20)

This ticket was written and committed before the first scored cell of the study.

- **Plan:** [2026-10-17-experiment-improvements](../../plans/2026-10-17-experiment-improvements.md)
  (§2g, item 11; D21).
- **The user's go** came in two steps:
  - 2026-10-20 afternoon: add TabPFN through uv instead of a separate venv, after T28 and T29;
  - 2026-10-20 about 23:10 GMT-3: "comece a executar experimentos que não precisam da minha
    supervisão direta, como aquilo do tab", with the machine until 08:30.
- **Why:** MaCoDE (E48) is the nearest deep prior art. TabPFN v2 (Hollmann et al., *Nature* 2025) is
  the tabular foundation model a reader will ask about.

## The code (landed with this ticket)

- **The dependency:** an optional extra, `tabpfn==9.2.0` (Apache-2.0), in `pyproject.toml`.
  - The lock gains packages and changes none: torch 2.5.1+cu121, scikit-learn 1.9.0, numpy 2.4.6,
    pandas 2.3.3, scipy 1.17.1 and lightgbm 4.7.0 are as before.
  - Without `--extra tabpfn` nothing imports it, so the default environment and Windows are
    unaffected.
- **`src/training/tabpfn_baseline.py`:** a column-wise, inductive wrapper. Its docstring has every
  choice; the main ones:
  - TabPFN v2's checkpoints, through `create_default_for_version(ModelVersion.V2)`. The package
    defaults to the gated v3.5.
  - One regressor or classifier a column, fitted on the training fold's rows where the column is
    observed, with the other columns as features. Gaps and hidden cells enter as NaN, and
    categories as training-fold indices declared categorical.
  - A context of at most 10,000 rows (v2's limit), subsampled with a fixed seed on letter and
    electricity.
  - 8 estimators (v2's default, named explicitly), `random_state` 0, the GPU.
  - Its own random streams; constant fills for a column with nothing to learn.
  - **Only the asked-for cells are predicted,** and a fold's masked call also predicts the frame's
    own gaps in the same pass, so its induced call reuses them.
    - TabPFN's fills depend slightly on which rows share a `predict`. On electricity_20nan at
      seed 7, a fold's induced `impute_score` moved by at most 1.6e-4 against separate calls,
      and the masked one by 0.
- **Weights:** downloaded once to `/scratch2/diogoneiss/tabpfn-weights` (`TABPFN_MODEL_CACHE_DIR`).
  They are the package's v2 defaults, Prior Labs License 1.1 (Apache 2.0 with an attribution
  requirement), not gated:
  - `tabpfn-v2-regressor.ckpt` (Hugging Face `Prior-Labs/TabPFN-v2-reg`, revision `4972a65`);
  - `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` (`Prior-Labs/TabPFN-v2-clf`, revision
    `f851f2a`).
  - The thesis cites Hollmann et al. 2025 wherever it reports TabPFN.
- **Checks:**
  - unit tests with a fake estimator (`tests/unit/test_tabpfn_baseline.py`, 5 tests);
  - the unit suite (419 passed, the typing gate included);
  - both regression fixtures, unchanged.
  - No training path of TRIDENT changes.
- **Driver and reading:**
  - `scripts/experiments/tabpfn_run.py` (E48's driver, through `recompute_folds`);
  - `tabpfn_report.py` (E48's report, with MaCoDE in the best of all where its CSVs exist);
  - `run_tabpfn_lanes.py`;
  - `report_common.DEEP` gains `tabpfn`: outside the bar and the strong tier, inside the best of
    all.

## Design (fixed before the first scored cell)

- **The arm:** TabPFN v2 as above, untuned. TRIDENT's configuration was tuned on the development
  set, so this is stated beside every comparison.
- **The cells,** as in E48:
  - **E44's confirmation set** at seeds 707/808/909: 15 variants, 45 pairs, which run first in
    every lane;
  - **the development set** at seeds 101/202/303: 21 variants, 63 pairs. K comes from
    `report_common.development_source`.
- **The smoke (2026-10-20, seed 7, not a study seed), timing only:**
  - electricity_20nan: 121 s for five folds with the one-pass fill;
  - before it: spambase_20nan 1,129 s and credit-g_60nan 40 s.
  - The column count drives the cost: spambase's 57 columns are the slowest pairs.
  - The smoke's CSVs were read only for the old-against-new difference above, then deleted.
- **Running it:** two GPU lanes (`run_tabpfn_lanes.py`), off-store, from 01:00 GMT-3 on
  2026-10-21.
  - About 6.4 h estimated in all; the development pairs may not all finish.
  - **The deadline:** a pair starts only if now + 1.3 × its estimate is before 08:00 GMT-3. A
    skipped pair is logged.
  - **Incomplete variants:** a variant without all three seeds is reported as incomplete and not
    read. The rest may run in a later GPU gap; a relaunch skips finished files.

## Measures

All on the induced `impute_score` over 15 fold pairs, with the verdict rule and the 1% SESOI (ADR
0020), through `scripts/experiments/tabpfn_report.py`:

1. **Primary:** K − TabPFN per variant, tallied on each set.
2. TabPFN − the strong tier's best (missForest and E41's arms), − the bar, and − the tuned bar
   (E42).
3. K − the best of all, TabPFN and MaCoDE included.
4. TabPFN's fit time per fold.

## Reading, fixed now (E48's)

**Descriptive on the development set,** where TRIDENT was tuned and TabPFN was not.

**On the confirmation set:**

- **K − TabPFN is the thesis's foundation-model statement,** reported as a tally beside E44's and
  E48's.
- **Whether TabPFN joins the strong tier:**
  - If it is better than the strong tier's best on at least 5 of the 15 confirmation variants,
    propose (by an ADR amending 0021) that it join the strong tier as an offline member.
  - Otherwise it is reported as an external baseline only.
- **No default changes either way.**

## Launch

- **Started at 01:00 GMT-3 on 2026-10-21** on cb17295, by `start_at_0100.sh` in
  `/scratch2/diogoneiss/trident-2026-10-21-tabpfn/`, in two lanes.
- **Ended at 03:48:** all 108 pairs, none failed, none skipped.
  - Pairs ran at about half the estimate: letter_40nan took 8.4 minutes against 18.

## Outcome (2026-10-21, 03:51 GMT-3)

**The printout** (`scripts/experiments/tabpfn_report.py`, unedited). Each reading is a mean
difference over 15 fold pairs of the induced `impute_score` (lower is better), with the 95% interval
and the call.

```text
## Development set, seeds 101/202/303 (labelled variants: none on this set)
- `kr-vs-kp_20nan`: K − tabpfn +0.0059 [-0.0053, +0.0172] none; tabpfn − strong -0.0809 [-0.0911, -0.0707] better; tabpfn − bar -0.0253 [-0.0338, -0.0167] better; tabpfn − tuned -0.0191 [-0.0307, -0.0075] better; K − best_of_all +0.0059 [-0.0053, +0.0172] none; best of all tabpfn; fit 7 s a fold
- `kr-vs-kp_40nan`: K − tabpfn +0.0102 [+0.0013, +0.0191] worse; tabpfn − strong -0.0750 [-0.0850, -0.0650] better; tabpfn − bar -0.0547 [-0.0587, -0.0508] better; tabpfn − tuned -0.0327 [-0.0389, -0.0264] better; K − best_of_all +0.0102 [+0.0013, +0.0191] worse; best of all tabpfn; fit 7 s a fold
- `kr-vs-kp_60nan`: K − tabpfn +0.0432 [+0.0352, +0.0512] worse; tabpfn − strong -0.0785 [-0.0862, -0.0707] better; tabpfn − bar -0.0811 [-0.0861, -0.0762] better; tabpfn − tuned -0.0295 [-0.0351, -0.0238] better; K − best_of_all +0.0432 [+0.0352, +0.0512] worse; best of all tabpfn; fit 6 s a fold
- `kr-vs-kp_80nan`: K − tabpfn +0.0404 [+0.0330, +0.0479] worse; tabpfn − strong -0.0589 [-0.0630, -0.0548] better; tabpfn − bar -0.0589 [-0.0630, -0.0548] better; tabpfn − tuned -0.0364 [-0.0408, -0.0321] better; K − best_of_all +0.0404 [+0.0330, +0.0479] worse; best of all tabpfn; fit 5 s a fold
- `spambase_20nan`: K − tabpfn -0.0843 [-0.0987, -0.0698] better; tabpfn − strong +0.0790 [+0.0662, +0.0918] worse; tabpfn − bar +0.0782 [+0.0640, +0.0924] worse; tabpfn − tuned +0.0851 [+0.0725, +0.0978] worse; K − best_of_all -0.0047 [-0.0162, +0.0067] none; best of all hgb/missforest; fit 53 s a fold
- `spambase_40nan`: K − tabpfn -0.0550 [-0.0672, -0.0428] better; tabpfn − strong +0.0192 [+0.0090, +0.0294] worse; tabpfn − bar +0.0233 [+0.0135, +0.0331] worse; tabpfn − tuned +0.0421 [+0.0337, +0.0506] worse; K − best_of_all -0.0223 [-0.0288, -0.0158] better; best of all macode; fit 40 s a fold
- `spambase_60nan`: K − tabpfn -0.0337 [-0.0412, -0.0262] better; tabpfn − strong +0.0174 [+0.0094, +0.0253] worse; tabpfn − bar -0.0073 [-0.0109, -0.0037] better; tabpfn − tuned +0.0197 [+0.0166, +0.0229] worse; K − best_of_all -0.0163 [-0.0200, -0.0126] better; best of all condexp; fit 27 s a fold
- `spambase_80nan`: K − tabpfn -0.0074 [-0.0115, -0.0033] better; tabpfn − strong -0.0070 [-0.0112, -0.0028] better; tabpfn − bar -0.0079 [-0.0090, -0.0069] better; tabpfn − tuned +0.0015 [+0.0004, +0.0026] worse; K − best_of_all -0.0074 [-0.0115, -0.0033] better; best of all tabpfn; fit 21 s a fold
- `credit-g_20nan`: K − tabpfn -0.0011 [-0.0140, +0.0117] none; tabpfn − strong -0.0145 [-0.0365, +0.0076] none; tabpfn − bar -0.0375 [-0.0567, -0.0184] better; tabpfn − tuned -0.0186 [-0.0361, -0.0012] better; K − best_of_all -0.0011 [-0.0140, +0.0117] none; best of all tabpfn; fit 3 s a fold
- `credit-g_40nan`: K − tabpfn +0.0169 [+0.0075, +0.0263] worse; tabpfn − strong -0.0274 [-0.0377, -0.0170] better; tabpfn − bar -0.0605 [-0.0725, -0.0486] better; tabpfn − tuned -0.0344 [-0.0477, -0.0210] better; K − best_of_all +0.0169 [+0.0075, +0.0263] worse; best of all tabpfn; fit 3 s a fold
- `credit-g_60nan`: K − tabpfn +0.0422 [+0.0284, +0.0560] worse; tabpfn − strong -0.0373 [-0.0441, -0.0305] better; tabpfn − bar -0.0490 [-0.0563, -0.0417] better; tabpfn − tuned -0.0293 [-0.0364, -0.0222] better; K − best_of_all +0.0422 [+0.0284, +0.0560] worse; best of all tabpfn; fit 2 s a fold
- `credit-g_80nan`: K − tabpfn +0.0141 [+0.0065, +0.0217] worse; tabpfn − strong -0.0017 [-0.0067, +0.0033] equivalent; tabpfn − bar -0.0036 [-0.0082, +0.0010] equivalent; tabpfn − tuned -0.0036 [-0.0082, +0.0010] equivalent; K − best_of_all +0.0150 [+0.0080, +0.0219] worse; best of all condexp/tabpfn; fit 2 s a fold
- `vehicle_20nan`: K − tabpfn -0.0077 [-0.0207, +0.0053] none; tabpfn − strong -0.0271 [-0.0449, -0.0092] better; tabpfn − bar -0.0448 [-0.0576, -0.0321] better; tabpfn − tuned -0.0400 [-0.0523, -0.0278] better; K − best_of_all -0.0077 [-0.0207, +0.0053] none; best of all tabpfn; fit 3 s a fold
- `vehicle_60nan`: K − tabpfn +0.0052 [-0.0049, +0.0152] none; tabpfn − strong -0.0162 [-0.0279, -0.0046] better; tabpfn − bar -0.0683 [-0.0809, -0.0557] better; tabpfn − tuned -0.0619 [-0.0727, -0.0511] better; K − best_of_all +0.0052 [-0.0049, +0.0152] none; best of all tabpfn; fit 3 s a fold
- `biodeg_20nan`: K − tabpfn -0.0322 [-0.0616, -0.0027] better; tabpfn − strong -0.0049 [-0.0320, +0.0221] none; tabpfn − bar -0.0267 [-0.0525, -0.0009] better; tabpfn − tuned -0.0201 [-0.0425, +0.0022] none; K − best_of_all -0.0302 [-0.0609, +0.0005] none; best of all missforest/tabpfn; fit 13 s a fold
- `biodeg_60nan`: K − tabpfn -0.0200 [-0.0322, -0.0077] better; tabpfn − strong +0.0315 [+0.0183, +0.0448] worse; tabpfn − bar -0.0778 [-0.0899, -0.0657] better; tabpfn − tuned -0.0613 [-0.0697, -0.0529] better; K − best_of_all +0.0115 [-0.0024, +0.0255] none; best of all missforest; fit 10 s a fold
- `kc2_20nan`: K − tabpfn +0.0434 [-0.0570, +0.1437] none; tabpfn − strong -0.0438 [-0.1170, +0.0293] none; tabpfn − bar -0.0286 [-0.0770, +0.0198] none; tabpfn − tuned -0.0612 [-0.1014, -0.0211] better; K − best_of_all +0.0434 [-0.0570, +0.1437] none; best of all tabpfn; fit 4 s a fold
- `kc2_60nan`: K − tabpfn -0.0684 [-0.1178, -0.0189] better; tabpfn − strong +0.0305 [-0.0266, +0.0877] none; tabpfn − bar -0.1029 [-0.1406, -0.0653] better; tabpfn − tuned -0.1238 [-0.1583, -0.0893] better; K − best_of_all -0.0378 [-0.0613, -0.0144] better; best of all softimpute; fit 3 s a fold
- `pendigits_20nan`: K − tabpfn +0.0123 [+0.0080, +0.0165] worse; tabpfn − strong -0.0260 [-0.0303, -0.0218] better; tabpfn − bar -0.0430 [-0.0481, -0.0378] better; tabpfn − tuned -0.0359 [-0.0390, -0.0329] better; K − best_of_all +0.0123 [+0.0080, +0.0165] worse; best of all tabpfn; fit 13 s a fold
- `letter_20nan`: K − tabpfn -0.0209 [-0.0240, -0.0179] better; tabpfn − strong -0.0089 [-0.0129, -0.0049] better; tabpfn − bar -0.0182 [-0.0214, -0.0150] better; tabpfn − tuned -0.0146 [-0.0163, -0.0128] better; K − best_of_all -0.0041 [-0.0073, -0.0009] better; best of all macode; fit 24 s a fold
- `electricity_20nan`: K − tabpfn -0.0116 [-0.0237, +0.0004] none; tabpfn − strong +0.0107 [-0.0022, +0.0235] none; tabpfn − bar +0.0278 [+0.0205, +0.0350] worse; tabpfn − tuned +0.0504 [+0.0445, +0.0562] worse; K − best_of_all +0.0161 [+0.0056, +0.0267] worse; best of all hgb; fit 6 s a fold
  - K − tabpfn: better 8 / no difference 6 / worse 7 (fold-pair t, seed agreement)
  - tabpfn − strong: better 11 / no difference 5 / equivalent 1 / worse 4 (fold-pair t, seed agreement)
  - tabpfn − bar: better 16 / no difference 1 / equivalent 1 / worse 3 (fold-pair t, seed agreement)
  - tabpfn − tuned: better 14 / no difference 1 / equivalent 1 / worse 5 (fold-pair t, seed agreement)
  - K − best_of_all: better 5 / no difference 8 / worse 8 (fold-pair t, seed agreement)

## Confirmation set (E44), seeds 707/808/909
- `pendigits_40nan`: K − tabpfn -0.0107 [-0.0137, -0.0078] better; tabpfn − strong -0.0599 [-0.0636, -0.0561] better; tabpfn − bar -0.0546 [-0.0563, -0.0530] better; tabpfn − tuned -0.0421 [-0.0434, -0.0407] better; K − best_of_all -0.0107 [-0.0137, -0.0078] better; best of all tabpfn; fit 9 s a fold
- `pendigits_60nan`: K − tabpfn -0.0180 [-0.0202, -0.0158] better; tabpfn − strong -0.1035 [-0.1085, -0.0985] better; tabpfn − bar -0.0388 [-0.0403, -0.0373] better; tabpfn − tuned -0.0357 [-0.0371, -0.0343] better; K − best_of_all -0.0180 [-0.0202, -0.0158] better; best of all tabpfn; fit 5 s a fold
- `pendigits_80nan`: K − tabpfn +0.0162 [+0.0150, +0.0174] worse; tabpfn − strong -0.0341 [-0.0359, -0.0322] better; tabpfn − bar -0.0408 [-0.0420, -0.0396] better; tabpfn − tuned -0.0200 [-0.0208, -0.0192] better; K − best_of_all +0.0162 [+0.0150, +0.0174] worse; best of all tabpfn; fit 3 s a fold
- `letter_40nan`: K − tabpfn -0.0344 [-0.0369, -0.0319] better; tabpfn − strong -0.0210 [-0.0237, -0.0182] better; tabpfn − bar -0.0211 [-0.0227, -0.0196] better; tabpfn − tuned -0.0070 [-0.0089, -0.0050] better; K − best_of_all -0.0246 [-0.0264, -0.0227] better; best of all macode; fit 17 s a fold
- `letter_60nan`: K − tabpfn -0.0301 [-0.0317, -0.0284] better; tabpfn − strong -0.0541 [-0.0564, -0.0518] better; tabpfn − bar -0.0042 [-0.0055, -0.0030] better; tabpfn − tuned -0.0024 [-0.0037, -0.0010] better; K − best_of_all -0.0301 [-0.0317, -0.0284] better; best of all tabpfn; fit 8 s a fold
- `letter_80nan`: K − tabpfn +0.0020 [+0.0006, +0.0034] worse; tabpfn − strong -0.0164 [-0.0173, -0.0155] better; tabpfn − bar -0.0159 [-0.0172, -0.0146] better; tabpfn − tuned -0.0002 [-0.0014, +0.0009] equivalent; K − best_of_all +0.0020 [+0.0006, +0.0034] worse; best of all tabpfn; fit 4 s a fold
- `electricity_40nan`: K − tabpfn +0.0303 [-0.0166, +0.0773] none; tabpfn − strong -0.0467 [-0.0628, -0.0307] better; tabpfn − bar +0.0132 [-0.0025, +0.0289] none; tabpfn − tuned +0.0192 [+0.0131, +0.0254] worse; K − best_of_all +0.0436 [-0.0013, +0.0885] none; best of all hgb/lgbm; fit 5 s a fold
- `electricity_60nan`: K − tabpfn +0.0038 [-0.0022, +0.0098] none; tabpfn − strong -0.0940 [-0.1085, -0.0794] better; tabpfn − bar +0.0074 [+0.0044, +0.0104] worse; tabpfn − tuned +0.0070 [+0.0042, +0.0098] worse; K − best_of_all +0.0112 [+0.0059, +0.0164] worse; best of all hgb/lgbm; fit 4 s a fold
- `electricity_80nan`: K − tabpfn +0.0103 [+0.0032, +0.0175] worse; tabpfn − strong -0.0672 [-0.0708, -0.0637] better; tabpfn − bar -0.0100 [-0.0130, -0.0070] better; tabpfn − tuned -0.0054 [-0.0085, -0.0023] better; K − best_of_all +0.0103 [+0.0032, +0.0175] worse; best of all tabpfn; fit 2 s a fold
- `vehicle_40nan`: K − tabpfn -0.0003 [-0.0134, +0.0127] none; tabpfn − strong -0.0253 [-0.0389, -0.0116] better; tabpfn − bar -0.0731 [-0.0824, -0.0638] better; tabpfn − tuned -0.0735 [-0.0820, -0.0650] better; K − best_of_all -0.0003 [-0.0134, +0.0127] none; best of all tabpfn; fit 3 s a fold
- `vehicle_80nan`: K − tabpfn +0.0530 [+0.0443, +0.0618] worse; tabpfn − strong -0.0106 [-0.0164, -0.0047] better; tabpfn − bar -0.0608 [-0.0668, -0.0548] better; tabpfn − tuned -0.0627 [-0.0692, -0.0563] better; K − best_of_all +0.0530 [+0.0443, +0.0618] worse; best of all tabpfn; fit 3 s a fold
- `biodeg_40nan`: K − tabpfn -0.0414 [-0.0674, -0.0154] better; tabpfn − strong +0.0172 [-0.0015, +0.0358] none; tabpfn − bar -0.0345 [-0.0454, -0.0237] better; tabpfn − tuned -0.0311 [-0.0452, -0.0171] better; K − best_of_all -0.0242 [-0.0390, -0.0094] better; best of all missforest; fit 12 s a fold
- `biodeg_80nan`: K − tabpfn +0.0525 [+0.0404, +0.0646] worse; tabpfn − strong -0.0036 [-0.0129, +0.0057] none; tabpfn − bar -0.0813 [-0.0871, -0.0756] better; tabpfn − tuned -0.0500 [-0.0567, -0.0433] better; K − best_of_all +0.0526 [+0.0410, +0.0641] worse; best of all missforest/tabpfn; fit 9 s a fold
- `kc2_40nan`: K − tabpfn -0.0016 [-0.0493, +0.0461] none; tabpfn − strong -0.0588 [-0.1207, +0.0031] none; tabpfn − bar -0.0840 [-0.1118, -0.0562] better; tabpfn − tuned -0.0966 [-0.1312, -0.0619] better; K − best_of_all -0.0016 [-0.0493, +0.0461] none; best of all tabpfn; fit 4 s a fold
- `kc2_80nan`: K − tabpfn -0.0577 [-0.1038, -0.0117] better; tabpfn − strong -0.0010 [-0.0301, +0.0280] none; tabpfn − bar -0.0989 [-0.1191, -0.0786] better; tabpfn − tuned -0.1081 [-0.1301, -0.0862] better; K − best_of_all -0.0534 [-0.0991, -0.0078] better; best of all missforest/tabpfn; fit 3 s a fold
  - K − tabpfn: better 6 / no difference 4 / worse 5 (fold-pair t, seed agreement)
  - tabpfn − strong: better 11 / no difference 4 / worse 0 (fold-pair t, seed agreement)
  - tabpfn − bar: better 13 / no difference 1 / worse 1 (fold-pair t, seed agreement)
  - tabpfn − tuned: better 12 / no difference 0 / equivalent 1 / worse 2 (fold-pair t, seed agreement)
  - K − best_of_all: better 6 / no difference 3 / worse 6 (fold-pair t, seed agreement)

Decision input: TabPFN better than the strong tier's best on 11 of 15 confirmation variants (at least 5: propose it for the strong tier).
```

**By missing rate** (built by script from the printout above):

| set | rate | K − TabPFN (better / none or equivalent / worse) | TabPFN − strong (better / none or equivalent / worse) |
|---|---|---|---|
| confirmation | 40nan | 3 / 3 / 0 | 4 / 2 / 0 |
| confirmation | 60nan | 2 / 1 / 0 | 3 / 0 / 0 |
| confirmation | 80nan | 1 / 0 / 5 | 4 / 2 / 0 |
| development | 20nan | 3 / 5 / 1 | 4 / 4 / 1 |
| development | 40nan | 1 / 0 / 2 | 2 / 0 / 1 |
| development | 60nan | 3 / 1 / 2 | 3 / 1 / 2 |
| development | 80nan | 1 / 0 / 2 | 2 / 1 / 0 |

**Reading, by the rule fixed in advance:**

- **TabPFN is better than the strong tier's best on 11 of the 15 confirmation variants,** worse on
  none. That clears the threshold of 5: the rule proposes, by an ADR amending 0021, that TabPFN join
  the strong tier as an offline member.
  - **That is the user's decision.** Nothing changes until then.
- **K − TabPFN on confirmation, the thesis's foundation-model statement:** better 6, no difference
  4, worse 5.
  - **Development (descriptive):** better 8, no difference 6, worse 7.

**What the tallies show:**

- **TabPFN is the strongest single baseline so far.** Against the bar it is better on 13 of 15
  confirmation variants and 16 of 21 development ones.
  - It loses to the boosters only on electricity, where the `date` key (E54) favours trees.
  - On spambase it is weaker than the strong tier at 20–60nan.
- **The split is by missing rate.**
  - **At 40 and 60nan on confirmation,** K is better than TabPFN on 5 variants and no different on
    4, never worse.
  - **At 80nan,** TabPFN beats K on 5 of 6: pendigits, letter, electricity, vehicle and biodeg.
    kc2_80nan is the exception, where K is better.
  - The development set agrees at 80nan (worse on 2 of 3).
- **On the tables TRIDENT was built for,** K beats TabPFN: spambase at every rate, letter and
  pendigits at 20–60nan, biodeg at 20–60nan.
  - On kr-vs-kp (all categorical) and credit-g, at 40nan and above, TabPFN wins.
- **K against the best of all** (TabPFN and MaCoDE included):
  - confirmation: better 6, no difference 3, worse 6;
  - development: better 5, no difference 8, worse 8.
- **Cost:** TabPFN's fit and fill take 2–53 s a fold, with no training. That favours it as a
  baseline.

**Limits:**

- TabPFN is untuned, and TRIDENT was tuned on the development set.
- Letter and electricity use a 10,000-row context subsample.
- The one-pass fill moves a fold's score by at most about 1.6e-4.
- The decision to add TabPFN to the strong tier would re-read every K − strong count in the thesis
  table. It is the user's, by an ADR.

