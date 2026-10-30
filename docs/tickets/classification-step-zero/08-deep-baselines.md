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
