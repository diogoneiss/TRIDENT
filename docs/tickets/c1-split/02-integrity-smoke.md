# C1 cut 2: the integrity smoke (E130; pre-registered 2026-11-04)

This ticket was written and committed before the smoke's first cell (12:48 GMT-3).

- **Why:** the protocol's go/no-go before the primary: under no pre-training, S and T run the same code path and
  must be bit-identical (integrity check 1); checks 2–4 are unit tests (`tests/unit/test_c1_tied.py`).
- **Code:** `c1_run.py` with `C1_CUT=2` and `C1_DETERMINISTIC=1` (`torch.use_deterministic_algorithms(True)`, cuDNN
  deterministic, `CUBLAS_WORKSPACE_CONFIG=:4096:8`; the user's decision 20 A). A 3-epoch pre-check on biodeg's two
  folds gave B and BT identical probabilities and states on the GPU (12:47 GMT-3), so the CPU fallback is not needed.

## Design and rule (fixed before the first cell)

- **Cells:** S, T, B and BT on `biodeg_probit_60nan`, seed 42, five folds, the cut-3 configuration (resolved
  classification configuration, `PRETRAIN_OBJECTIVE embedding_normalized`, `C15_ARM A1`), on gorgona8, under
  `<state>/smoke`.
- **Passes if** B's and BT's test probabilities and kept states are identical on all five folds. S and T run to show
  the arms train end to end; **no S − T contrast is read from the smoke** (short mode sets no status).
- **If it fails,** cut 3 does not start until the difference is explained.

## Outcome (2026-11-04 12:55 GMT-3): passes; cut 3 may start

Four cells on gorgona8, 12:50–12:55 GMT-3, deterministic, none failed. The unedited `c1_report.py --smoke` at 013432a:

```
## Smoke (E130, biodeg_probit_60nan, seed 42, deterministic): integrity check 1, B against BT per fold
- largest |P(B) − P(BT)| over the test rows of 5 folds: 0.00e+00; kept states identical: True
- integrity check 1: passes
- S: macro-F1 per fold [0.7182, 0.7224, 0.754, 0.6865, 0.7616], log loss [0.5686, 0.5315, 0.4707, 0.546, 0.4917] (no contrast is read)
- T: macro-F1 per fold [0.7187, 0.7224, 0.7488, 0.7012, 0.7796], log loss [0.5809, 0.5153, 0.4357, 0.5486, 0.474] (no contrast is read)
```

B and BT are bit-identical on all five folds (probabilities and kept states), so the tied arm's code path changes
nothing outside pre-training. S and T trained end to end; their per-fold numbers are listed only to show the arms
run, and no contrast is read from them (short mode, one seed).
