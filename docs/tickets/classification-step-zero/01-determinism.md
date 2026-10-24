# Stage 0: does HEAD on gorgona8 reproduce the old classification sweep's folds? (E70; pre-registered 2026-10-23)

This ticket was written and committed before its first run.

- **Why:** Stage 0 of classification step zero
  ([plan](../../plans/2026-10-23-classification-step-zero.md); review item 16, trap 1). Stage 2's
  no-pre-training arm B is to be paired with the old sweep as arm A, which holds only if HEAD
  reproduces the sweep.
- **The user's go:** "sim aprovo" (2026-10-23, 23:44 GMT-3), to "pré-registrar e rodar o Stage 0
  agora", with Stage 2 after E69 and T35's code in parallel.
- **What could break the pairing:**
  - **the environment:** the sweep ran on Python 3.10, sklearn 1.7.2 and numpy 2.2.6; HEAD's venv
    has 3.11, 1.9.0 and 2.4.6 (torch unchanged);
  - **the code:** commits after the sweep's 6d49ed5 and c0a909b;
  - **the machine:** the sweep ran on the Windows laptop (its provenance paths are Windows paths;
    the review names its RTX 3050). CUDA dropout masks differ between GPU models (AGENTS.md, the
    credit-g fixture), so a GPU run on gorgona8's RTX 3090 Ti may differ for that reason alone.
- **Code:** `scripts/experiments/cls_determinism_run.py`. No `src/` change and no store write.

## Design (fixed before the first run)

- **Two runs, on gorgona8's GPU, as Stage 2 would run:** `vehicle_40nan` and `kc2_40nan`. Both are
  small and gapped, and both sweep runs come from 6d49ed5.
  - **The sweep's run:** the latest active, non-mirror classification parent of the variant with
    seed 42, 3 folds and `cosine_legacy` (review item 13's selection).
  - **The rerun:** its `hyperparameters.json` as `hyperparams_override`, `lr_scheduler
    cosine_legacy`, 3 folds, seed 42, MLflow off, a private metrics directory.
- **Compared fold by fold:** test `accuracy`, `f1_micro` and `f1_macro`
  (`cls_determinism_run.py --compare`).
- **A companion check:** the classification regression test
  (`pytest tests/integration/test_vehicle_regression.py`, vehicle_00nan at 2 epochs). It tells
  whether HEAD's classification code path still matches its reviewed baseline, apart from the
  machine.

## Reading, fixed now

- **Reproduces:** every fold's three metrics within 1e-6 on both variants. Stage 2 then pairs its
  arm B with the sweep's runs as arm A.
- **Does not reproduce:** any larger difference. Stage 2 then reruns arm A on gorgona8 beside
  arm B, in Stage 2's own ticket. The size of the difference is reported against the sweep's
  duplicate-run spread (0.0002–0.009 macro-F1), descriptively.
  - With the companion test passing, a difference points at the machine or the environment, not
    the code; this check cannot tell those two apart, and Stage 2 does not need it to.
