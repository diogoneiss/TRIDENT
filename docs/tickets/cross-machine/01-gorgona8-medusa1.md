# Do gorgona8 and medusa1 give the same fold scores for the same cell? (E86; pre-registered 2026-10-28)

This ticket was written before the comparison could be read. Two gorgona8 cells ran first as a
smoke check of the script (classification vehicle_40nan s42, imputation kc2_60nan s101); they
count, and no medusa1 cell existed when this was committed.

- **Why:** since ADR 0033 medusa1 (two RTX 5090) trains beside gorgona8 (RTX 3090 Ti). The next
  studies (T41's old-sweep half, the occlusion map under MNAR/MAR) would split their cells across
  the two. That is sound only if the machine adds no systematic shift.
- **The user's go:** "vamos testar o 0!" (2026-10-28, about 15:10 GMT-3), with the instruction to
  mind the mirrored runs.
- **The question:** not bit identity. The GPUs are different architectures (Ampere, Blackwell), and
  CUDA dropout masks already differ between GPU models (imputation-decoder ticket 11), so fold
  scores will differ. The question is whether the difference has zero mean within the SESOI.
- **Code:** no change to `src/`. `scripts/experiments/cross_machine.py` has `train`, `--todo` and
  `--tally`.
- **Environment, checked before launch:** both machines at commit of this ticket, torch
  2.11.0+cu128, the same `uv pip freeze` (md5 `d9a77c89…`), the same six processed CSVs (md5 of
  their md5s `e8d22df0…`), and the same old-sweep runs for the four classification variants.

## Design (fixed now)

- **MLflow off on both machines.** Nothing reaches the store, so no run and no mirror is
  duplicated; each cell writes to `<state>/<hostname>/<task>/<variant>_s<seed>/`. State:
  `/scratch2/diogoneiss/trident-2026-10-28-cross-machine/` (gorgona8),
  `/scratch/diogoneiss/trident-2026-10-28-cross-machine/` (medusa1, whose `/scratch2` belongs to
  root).
- **Imputation:** today's default, five folds, as E84's controls ran. credit-g_20nan (both column
  types), kc2_60nan, vehicle_40nan, biodeg_40nan × seeds 101, 202, 303. 12 cells.
- **Classification:** E71's arm A (the old sweep's `hyperparameters.json`, `cosine_legacy`, three
  folds). vehicle_40nan (E71's gate cell), kc2_40nan, credit-g_20nan, biodeg_60nan × seeds 42, 7,
  13. 12 cells.
- **Run-to-run duplicates on each machine:** imputation credit-g_20nan s101 and classification
  vehicle_40nan s42, run a second time.
- **Cells:** 26 per machine, two queues each (medusa1: one per GPU, `CUDA_VISIBLE_DEVICES` 0 and
  1; another user's job holds 7–9 GB on each). About 1 h, at 1–7 min a cell. This is more than the
  "~6 cells" first quoted to the user; the cost stays near 1 h.
- **Expected:** medusa1 recomputes the CPU baselines on its first cells, since
  `baseline_cache.py` keys the cache by hostname. That makes those cells slower and changes no
  fill.

## Measures and rule (`cross_machine.py --tally`)

- **Primary, per task:** per fold, medusa − gorgona8.
  - Imputation: relative difference on the default output's `impute/induced/impute_score`
    (60 fold pairs), SESOI 1%.
  - Classification: absolute difference on test macro-F1 (36 fold pairs), SESOI 0.01.
- **The verdict:** the project's rule (95% t-interval over the pooled fold pairs, seed agreement).
  - **equivalent** (interval inside ±SESOI): a study may split its cells across the machines by
    seed. Both arms of a pair stay on one machine, and each cell records where it ran.
  - **better or worse** (a systematic shift): each study runs whole on one machine.
  - **none** (inconclusive at this size): the tie-break is the size of the per-fold differences.
    If the largest |difference| is within the seed-to-seed spread (imputation: E12's seed SD,
    0.007–0.014; classification: E70's duplicate spread, 0.0002–0.009), the split is allowed,
    as for equivalent. Otherwise each study runs whole on one machine.
- **Descriptive only:**
  - the run-to-run duplicates (gorgona8 measured 0 on E71's gate; medusa1's figure is new);
  - the masked score;
  - the knn5 and hgb baselines. They run on the CPU, and hgb's last digits differ between
    machines by design (`baseline_cache.py`), so a small difference there is expected.

## Launch and tally

```bash
# both machines, at this ticket's commit
ssh medusa1 'git -C /scratch/diogoneiss/TRIDENT pull --ff-only'
# per queue n in 1 2 (medusa1: CUDA_VISIBLE_DEVICES=n-1, TZ=America/Sao_Paulo)
uv run --python 3.11 python -B scripts/experiments/cross_machine.py --todo <n> | while read t v s d; do
  uv run --python 3.11 python -B scripts/experiments/cross_machine.py train $t $v $s $d; done
# on gorgona8, after both finish
rsync -a medusa1:/scratch/diogoneiss/trident-2026-10-28-cross-machine/medusa/ \
  /scratch2/diogoneiss/trident-2026-10-28-cross-machine/medusa/
uv run --python 3.11 python -B scripts/experiments/cross_machine.py --tally
```

## Outcome

**Done 2026-10-28, 15:55 GMT-3.** The cells ran 15:23–15:54 GMT-3, and all 26 finished on each
machine with none failed. medusa1 shared its GPUs with another user's job.

The tally, unedited (`cross_machine.py --tally`):

```
## imputation: medusa − gorgona8 on impute/induced/impute_score (relative)
- credit-g_20nan s101: -0.0046, +0.0147, -0.0073, +0.0562, +0.0230
- credit-g_20nan s202: -0.0258, +0.0494, +0.0212, +0.0265, +0.0120
- credit-g_20nan s303: +0.0131, -0.0019, -0.0025, +0.0365, -0.0038
- kc2_60nan s101: +0.0178, -0.0331, -0.0131, +0.0106, -0.0186
- kc2_60nan s202: -0.0129, -0.1596, +0.0261, -0.0683, +0.0379
- kc2_60nan s303: -0.0198, +0.0856, +0.0315, +0.1440, +0.0003
- vehicle_40nan s101: +0.0007, -0.0133, -0.0326, +0.1641, +0.0673
- vehicle_40nan s202: -0.0298, -0.0096, -0.0131, +0.0230, +0.0846
- vehicle_40nan s303: +0.0078, +0.0023, +0.0016, -0.0032, -0.0215
- biodeg_40nan s101: +0.0480, -0.0470, -0.0168, +0.0056, -0.0444
- biodeg_40nan s202: +0.0266, -0.0136, +0.0077, +0.0030, +0.0002
- biodeg_40nan s303: +0.0227, -0.0051, -0.0102, -0.0307, -0.0556
- largest |difference| per metric: impute/induced/impute_score 0.1641; impute/masked/impute_score 0.3337; impute/induced/baseline/hgb/impute_score 0.0000; impute/induced/baseline/knn5/impute_score 0.0007
- pooled +0.0059 [-0.0061, +0.0178], seeds +0.0089, -0.0007, +0.0096, 60 fold pairs -> none (SESOI 0.01; fold-pair t, seed agreement)

## classification: medusa − gorgona8 on f1_macro
- vehicle_40nan s42: -0.0056, +0.0244, +0.0170
- vehicle_40nan s7: +0.0185, +0.0059, +0.0247
- vehicle_40nan s13: +0.0258, +0.0455, +0.0410
- kc2_40nan s42: -0.0062, -0.0742, +0.0168
- kc2_40nan s7: +0.0167, +0.0481, +0.0019
- kc2_40nan s13: -0.0199, -0.0731, -0.0109
- credit-g_20nan s42: +0.0314, -0.0055, -0.0532
- credit-g_20nan s7: -0.0615, -0.0271, +0.0071
- credit-g_20nan s13: +0.0038, -0.0185, +0.0024
- biodeg_60nan s42: +0.0477, +0.0253, +0.0195
- biodeg_60nan s7: +0.0095, +0.0046, +0.0343
- biodeg_60nan s13: +0.0202, +0.0080, -0.0026
- largest |difference| per metric: f1_macro 0.0742; accuracy 0.0808
- pooled +0.0039 [-0.0066, +0.0145], seeds +0.0031, +0.0069, +0.0018, 36 fold pairs -> none (SESOI 0.01; fold-pair t, seed agreement)

## run to run, same machine
- gorgona8 imputation credit-g_20nan s101: largest |difference| 0.00e+00
- medusa imputation credit-g_20nan s101: largest |difference| 0.00e+00
- gorgona8 classification vehicle_40nan s42: largest |difference| 0.00e+00
- medusa classification vehicle_40nan s42: largest |difference| 0.00e+00
```

- **Primary: no verdict on either task.**
  - Imputation: +0.59% [−0.61%, +1.78%]. Classification: +0.0039 [−0.0066, +0.0145].
  - Neither interval excludes zero, and neither lies inside the SESOI.
- **The pre-registered tie-break: each study runs whole on one machine.** It compares the largest
  per-fold |difference| with the seed spread.
  - Imputation: 0.164 relative, against E12's 0.007–0.014.
  - Classification: 0.074 macro-F1, against E70's 0.0002–0.009.
  - Both lie far outside.
- **Descriptive:**
  - Each machine is deterministic: both duplicates are bit-identical on both machines.
  - The CPU baselines agree (hgb 0, knn5 at most 0.0007). So the data and folds are the same, and
    the differences come from GPU training alone.
  - The machine acts like another random draw of the same seed. The mean shift is small and
    unproven, but one fold can move by as much as a seed change moves it.
- **What follows:**
  - A study keeps every cell on one machine.
  - Two studies can run in parallel, one on each machine.
  - Splitting one study by seed would need a new pre-registered check with more pairs, one that
    shows equivalence. That is the user's call.
