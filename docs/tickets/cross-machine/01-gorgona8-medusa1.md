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

(open)
