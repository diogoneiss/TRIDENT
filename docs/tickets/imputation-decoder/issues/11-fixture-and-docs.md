# 11. Regression fixture and documentation

Status: done (this commit) on `feat/imputation-task`
Blocked by: 08, 10
Plan task: 11. ADR 0004 decision 14. Wayfinder ticket 14.

## Goal

Pin the imputation task's behaviour on `credit-g_20nan` in the existing fixture's exact
pattern, and bring the documentation in line with ADR 0004.

## Seams under test

- `tests/integration/test_credit_g_imputation_regression.py` (`@pytest.mark.integration`),
  same shape as the vehicle test, reading
  `tests/fixtures/credit-g_20nan_imputation_regression.json`.

## Acceptance criteria

- [x] Fixture generated from the ADR decision 14 configuration (`DIM` 16, `HIDDEN_DIM` 8,
      `HEADS` 4, `LAYERS` 1, `DIM_FEED` 16, `DROPOUT` 0.1, `EPOCHS_PRE` 2, `EPOCHS_DECODE` 2,
      `BATCH` 64, `LAMBDA_NUM` 1.0, `EVAL_MASK_RATE` 0.2, `cv_folds` 2, seed 42, tracking
      disabled), run three times, `observed_max_deviation` recorded, `tolerance` set well
      above it, environment block recorded, regeneration command in the test docstring.
- [x] Pins `impute_score`, `rmse_num_z`, `acc_cat` for both populations, per fold and mean.
- [x] README: `--task`, `--score_null_path`, the six new keys with the `PROB_MASCARA` /
      `EVAL_MASK_RATE` pairing, `task` and `is_optuna` filters in the MLflow comparison
      section, the artifact inventory, the `--lr_scheduler cosine` recommendation.
- [x] `docs/ARCHITECTURE.md`: a `TridentDecoder` section with theory and implementation
      diagrams; the decode stage in the orchestration diagram.
- [x] `AGENTS.md` names both regression fixtures. `docs/BACKLOG.md` marks B3 fixed.
- [x] Full verification: `pytest -m "not integration"`, `pytest -m integration`,
      `graphify update .`.
- [x] ADR 0004's fourteen decisions each map to a test or a documented manual check.

## Constraints

- The degenerate compositions (all-numerical, all-categorical, zero baseline) are covered
  by ticket 03's unit tests, not by extra integration runs.
- Do not tighten the tolerance below the existing fixture's 0.01.

## Comments

- 2026-09-10 (from ticket 06): gather the rest of ticket 12's pre-registered comparison.
  One fold of `credit-g_20nan` put the `[MASK]` path at 0.965 and the `[NULL]` path at
  1.042; the criterion needs a majority of folds on at least two datasets at both 20% and
  60%. Record the verdict in ADR 0004 decision 11.

- 2026-09-10, ticket complete.

  **Fixture.** `tests/fixtures/credit-g_20nan_imputation_regression.json`, generated from
  three runs of the ADR's configuration. `observed_max_deviation` is **0.0**, so the
  imputation path is bit-reproducible exactly as the classification one is; `tolerance`
  stays at the existing fixture's 0.01 so the test survives a different device or torch
  build. It pins `impute_score`, `rmse_num_z` and `acc_cat` for both populations, per fold
  and mean. The scores it records are poor (around 1.4, worse than mean-and-mode) because
  two decode epochs cannot beat a naive imputer; the fixture pins determinism, not quality.

  Regenerate with the block in this ticket's history, or re-derive it from
  `tests/integration/test_credit_g_imputation_regression.py`'s configuration.

  **Ticket 12's pre-registered comparison, measured.** The criterion is **not met**, so
  ticket 03's decision to substitute `[MASK]` stands and now rests on evidence:

  | dataset | folds | `[MASK]` | `[NULL]` |
  |---|---|---|---|
  | `credit-g_20nan` | 2 | 1.008, 1.004 | 1.155, 1.180 |
  | `credit-g_60nan` | 2 | 1.022, 1.121 | 1.227, 1.422 |
  | `kr-vs-kp_20nan` | 2 | 0.984, 0.969 | 1.920, 1.533 |
  | `kr-vs-kp_60nan` | 2 | 1.012, 0.980 | 1.986, 1.640 |

  `[MASK]` won **8 of 8** folds. The margin is widest on all-categorical `kr-vs-kp`, where
  the `[NULL]` path is about twice as bad as filling the mode -- a head never trained at a
  null position has nothing to transfer from. Recorded in ADR 0004 decision 11.

  **Documentation.** README gains a task-selection section, an "Imputation Task" section
  (what is scored, the ranking score, the artifact table), the six new config keys with the
  nominal-versus-realised warning on `EVAL_MASK_RATE`, and the `task` / `is_optuna` filters
  in the MLflow comparison section. `docs/ARCHITECTURE.md` gains a `TridentDecoder` section
  in both theory and implementation flavours plus the task branch in the orchestration
  diagram. `AGENTS.md` names both fixtures and the two tasks. `docs/BACKLOG.md` marks B3
  fixed. Every relative link in the touched documents resolves.

  Unit suite 128 green; **both** integration regressions pass, the vehicle one unedited.

- 2026-09-29, the credit-g regression now runs on the CPU.

  **What changed.** On gorgona8 (RTX 3090 Ti) the test failed: fold 1 masked
  `impute_score` came out at 1.613 against the fixture's 1.402. Commit `7a8e02d` answered
  that with an absolute tolerance of 0.25 on Linux. That part of the commit is reverted,
  and its `test_sync_remote` fix stays. The test now hides CUDA
  (`monkeypatch.setattr(torch.cuda, "is_available", lambda: False)`), so every fold runs on
  the CPU. The fixture was regenerated there, and `tolerance` stays at 0.01.

  **Why the GPUs disagree.** CUDA dropout masks depend on the GPU model. The libraries match
  on both machines: torch 2.5.1+cu121, CUDA 12.1, cuBLAS 12.1.3, cuBLASLt 120103 and
  cuDNN 9.1.0. Only the driver differs (595.95 against 575.51.03). Each machine repeats its
  own result bit for bit, deterministic mode included. PyTorch's CUDA dropout kernel hands
  random numbers out per thread and caps its grid at SMs x (max threads per SM / 256)
  blocks. Masks therefore agree only up to SMs x 6 x 256 x 4 elements: 98,304 on the
  RTX 3050 Laptop (16 SMs) and 516,096 on the RTX 3090 Ti (84 SMs). Above that limit the
  smaller card draws other masks for the excess elements. It also advances the Philox
  offset by a different amount, so every later dropout call in the run differs too. The
  limit was measured, not read from the source (`aten/src/ATen/native/cuda/Dropout.cu`):
  masks match up to 98,304 elements and differ from 98,305. Sizes that are not a multiple
  of 4 diverge earlier. `HEADS` 4 keeps every tensor here a multiple of 4, but a search
  over other head counts could hit that case.

  This fixture's attention dropout covers 64 x 4 x 21 x 21 = 112,896 elements, over the
  3050's limit. The vehicle fixture's 92,416 elements stay under it, so that test still runs
  on the GPU and passes on both machines. The largest absolute difference between the two
  cards over the 12 pinned fold metrics:

  | configuration | largest difference, 3050 against 3090 Ti |
  |---|---|
  | fixture configuration | 0.21 |
  | `DROPOUT` 0 | 1.3e-8 |
  | `BATCH` 48 (84,672 elements, under the limit) | 3.0e-9 |
  | CPU on both machines | 2e-9 |

  cuBLAS also picks different kernels on the two cards, and matmul outputs differ by about
  1e-6. That does not amplify here: once the dropout masks match, the metrics agree to
  1e-8. The first explanation, rounding noise amplified by training, was wrong for this
  run. So was reading the CPU-against-GPU gap as rounding, because the CPU draws dropout
  with a different generator. All of this was measured on this fixture's two-epoch run
  only.

  **Beyond this test.** A run whose dropout tensors exceed the smaller card's limit does
  not reproduce across the two GPUs. The tuned configurations use `BATCH` 256 and `HEADS`
  16, which puts attention dropout above 1.1M elements, over both limits. Their
  cross-machine differences are different dropout noise, seed-like variance rather than
  numerical error: compare distributions across machines, not digits. Deterministic mode
  and `CUBLAS_WORKSPACE_CONFIG` cannot fix this, because the difference comes from how the
  kernel lays out the random numbers, not from run-to-run nondeterminism.

  **Regenerate** from the checkout root with `uv run --python 3.11 python -`, feeding it
  this block. It forces the CPU, runs three times and rewrites the fixture in place:

  ```python
  import json, platform, sys, tempfile
  from pathlib import Path

  import torch

  torch.cuda.is_available = lambda: False  # the fixture is recorded on the CPU, see above
  from src.training.runner import run_training
  from src.training.types import DatasetSpec, Hyperparameters, RuntimeOptions, TrainingRequest

  path = Path("tests/fixtures/credit-g_20nan_imputation_regression.json")
  fixture = json.loads(path.read_text())
  pinned = list(fixture["mean"])
  runs = []
  for _ in range(3):
      tmp = Path(tempfile.mkdtemp())
      result = run_training(TrainingRequest(
          dataset=DatasetSpec.from_name("credit-g_20nan", "class"),
          hyperparameters=Hyperparameters.from_mapping(fixture["hyperparameters"]),
          runtime=RuntimeOptions(output_dir=tmp / "results", metrics_dir=tmp / "metrics",
                                 tracking_enabled=False),
          seed=fixture["environment"]["seed"], cv_folds=fixture["environment"]["cv_folds"],
          plot_losses=False, save_model=False, task="imputation"))
      runs.append(([{m: f.metrics[m] for m in pinned} for f in result.fold_results],
                   {m: result.mean_metrics[m] for m in pinned}))
  folds, mean = runs[0]
  deviation = max(abs(a[m] - b[m]) for other_folds, other_mean in runs[1:]
                  for a, b in zip(folds + [mean], other_folds + [other_mean]) for m in pinned)
  fixture["environment"].update(python=f"{sys.version_info.major}.{sys.version_info.minor}",
                                platform=platform.platform(), torch=torch.__version__,
                                device="cpu", runs=len(runs))
  fixture.update(folds=folds, mean=mean, observed_max_deviation=deviation)
  path.write_text(json.dumps(fixture, indent=2) + "\n")
  ```

  The diagnostic scripts and their outputs are outside the repository, in
  `gorgona8:/scratch2/diogoneiss/bench/diagnostico/`: `gpu_fingerprint.py` for versions,
  dropout masks and matmul hashes, `regcheck_ab.py` for the table above, and
  `RESULTADO_GPU.md`.
