# C1 cut 3: the primary, S against T on the four MNAR 60nan files and their MCAR siblings; the anchor (E131; pre-registered 2026-11-04)

This ticket was written and committed before the cut's first cell (12:48 GMT-3); the cells start only
after cut 2 ([02](02-integrity-smoke.md)) passes.

- **Why:** the thesis's main claim ([plan](../../plans/2026-11-04-c1-split.md); the protocol's cut 3).
- **S is reused:** E124's A1 cells ([c15-mnar-prediction/03](../c15-mnar-prediction/03-stage1.md)) on the eight files,
  seeds 42/7/13, are C1's S (same configuration, objective, folds, readouts); nothing of S is rerun for the reading.
  `S_GEOM` runs S once more at seed 42 on the four MNAR files only for the token dump the reused cells lack; its
  scores enter nothing.
- **Code:** `c1_run.py` (`C1_CUT=3`), `c1_report.py --primary`, `--anchor`, `--cosine`.

## Design (fixed before the first cell)

- **Files:** MNAR `{biodeg,kc2,vehicle,spambase}_probit_60nan`; MCAR `{biodeg,kc2,vehicle,spambase}_60nan`.
- **Cells, decision mode (seeds 42/7/13 × 5 folds):** T and B on all eight files (48 cells); ANCHOR (S with
  `PRETRAIN_OBJECTIVE embedding`) on the four MNAR files (12); S_GEOM at seed 42 on the four MNAR files (4). 64 new
  cells; S's 24 reused. Configuration as E124's: each variant's resolved classification configuration (300/150
  epochs, batch 256, `cosine_legacy`), `embedding_normalized` by override, `C15_ARM A1`, the shipped checkpoint rule,
  MLflow off, `torch.compile` off.
- **Machines (ADR 0035):** each T, B, ANCHOR and S_GEOM cell on the machine holding its pair's S cell:
  gorgona8 seed 42 of kc2 and spambase (12 cells), medusa1 the rest (52).

## Rules, fixed now

- **Unit:** a replicate is (seed, fold); S − T paired, 15 per file. Read only with every cell in.
- **Per file:** macro-F1(S) − macro-F1(T), plain 95% t interval (14 df): *better* or *worse* if the interval is off 0
  and the three seed means agree in sign; *equivalent* if inside ±0.010; *no difference* otherwise. Flags, not vetoes.
- **Pooled primary (MNAR):** D_r = the mean over the four files per replicate; plain 95% t interval over the 15 D_r.
  **Supported** if above 0 (label *relevant* if Δ ≥ +0.010, *real but small* otherwise); **Reversed** if below 0
  (symmetric); **Null** if it contains 0, lies inside ±0.010, and G1 ([01](01-g1-probe.md)) and G2 pass;
  **Inconclusive** otherwise. The Nadeau–Bengio-corrected interval (variance × (1/15 + 0.286)) is printed beside as a
  sensitivity read; a Supported reading whose corrected interval reaches below 0 is reported as such.
- **G2 (pre-training channel):** S − B pooled over the four MNAR files, passes if its plain interval excludes 0.
- **MCAR, the fixed-sequence second step:** the same rules on the four MCAR files; it decides only whether a
  Supported or Reversed status is scoped to informative missingness. The interaction (S − T)_MNAR − (S − T)_MCAR per
  table, paired by replicate, is descriptive.
- **Log loss:** LL(T) − LL(S), the same per-file rule without a margin; a sign disagreement with macro-F1 whose
  interval excludes 0 is a caveat, not a reversal.
- **Anchor (descriptive):** S(`embedding_normalized`) − S(`embedding`) per file and pooled; decides nothing.
- **Token geometry (diagnostic):** in S_GEOM after pre-training, and in S's reused cells after fine-tuning, the median
  cos([MASK]_j, [NULL]_j) per file, the cross-column cos([NULL]_j, [NULL]_j') beside, reference SD 1/√128 ≈ 0.088.
  A median ≥ 0.9 after pre-training on the MNAR files reads "tied by the model"; token vectors alone, without the
  positional embedding.
- **Threats stated now:** the scaler and vocabularies are fit on the whole table (label-free, equal across arms); the
  configuration descends from S; each MNAR file is one draw (seed 1060); kc2's twin rows; GPU nondeterminism of up to
  0.009 macro-F1 between same-seed runs enters every paired difference.
