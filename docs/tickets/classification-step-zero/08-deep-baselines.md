# Deep baselines for classification: MLP, SAINT and TabNet (E99; DRAFT, not pre-registered)

**Draft written 2026-10-30 16:58 GMT-3.** Nothing has run: no cell, no cost probe, no score read. The design is
to be settled with the user in a separate session before any cell; this file only hands that
session what exists.

- **Why:** the user's claim to test is that TRIDENT "em geral" beats similar algorithms, meaning
  deep tabular models trained per dataset. The user chose MLP, SAINT and TabNet (2026-10-30). TabPFN
  stays in the strong tier (ADR 0032). GBDTs are an internal sanity check only and stay out of the
  thesis (the user, 2026-10-30; E98).
- **What exists (uncommitted until this commit, untested):**
  - `pyproject.toml`: the `deep-baselines` extra (`pytorch-widedeep==1.5.0`, plus `torchvision`
    from the cu128 index, which its PyPI wheel does not match); a mypy override for
    `pytorch_widedeep.*`. `uv.lock` updated; torch, scikit-learn, numpy, pandas and pyarrow are
    unchanged. Installed on gorgona8 (`uv sync --inexact --extra tabpfn --extra deep-baselines`),
    not on medusa1.
  - `scripts/experiments/deep_baselines.py`: `run <model> <variant>` and a cost-only `probe`.
    TRIDENT's recipe: 20 TPE trials on mean validation macro-F1 over seed 42's ten folds, balanced
    class weights, the lowest-validation-loss epoch kept (patience 20, at most 200 epochs),
    evaluation at seeds 42/7/13. Gaps: numerical mean plus indicators, categorical `[NULL]` level.
    pytorch-widedeep 1.5.0 needs `np.Inf` (removed in NumPy 2), patched in the script.
  - The TRIDENT arm would be E98's tuned evaluations, reused on the same folds.
- **To decide before the first cell:** the rule for "em geral" (proposed: TRIDENT leads a baseline
  when higher at 3 or more of 5 levels and lower at none; supported if it leads all three, refuted if
  any baseline is higher at 3 or more levels, mixed otherwise); the search spaces and batch sizes in
  the script; a cost probe, then the compute cap.

## Cost probe (medusa1, 2026-10-30 17:15–17:49 GMT-3; seconds only, no score read)

One mid-space configuration per model (`MIDDLE` in the script), seed 42's ten folds, each table at
40nan, `deep_baselines.py probe`; log in `/scratch/diogoneiss/trident-2026-10-30-deep-probe/`. MLP
and TabNet shared GPU0, SAINT ran on GPU1; from 17:31 E100's queues shared both.

| table | MLP | SAINT | TabNet |
|---|---|---|---|
| kc2 | 4 s | 14 s | 18 s |
| credit-g | 7 s | 12 s | 19 s |
| vehicle | 6 s | 19 s | 21 s |
| biodeg | 6 s | out of memory | 20 s |
| spambase | 12 s | out of memory | 110 s |
| kr-vs-kp | 20 s | 28 s | 70 s |
| pendigits | 63 s | 117 s | 317 s |
| letter | 211 s | 140 s | 717 s |
| electricity | 223 s | 286 s | 683 s |
| **sum** | **9 min** | **10 min + 2** | **33 min** |

- **The whole study** is about 23 ten-fold fits a variant (20 trials and three evaluations) over 5
  levels, so roughly 18, 20+ and 63 GPU-hours: about 100 in all, before parallelism.
- **SAINT ran out of memory** (21 GB) on biodeg and spambase, the tables with the most columns
  once the gap indicators double them. To settle: a smaller SAINT batch, or no indicators for it.
- **Fixed during the probe** (the commit after 6e155e8): the all-numerical tables (kc2) and widedeep's default
  of one loader worker per core, which cost 38 s a kc2 fold against 0.4 s in-process.
- Written 2026-10-30 18:36 GMT-3.
