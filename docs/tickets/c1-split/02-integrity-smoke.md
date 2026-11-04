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
