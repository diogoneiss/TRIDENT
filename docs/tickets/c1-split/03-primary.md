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

## Outcome (2026-11-04 13:26 GMT-3): Inconclusive; G2 fails (no pre-training channel), so the split has nowhere to act

All 64 new cells done (12:55–13:16 GMT-3 on medusa1, 12:55–13:24 on gorgona8; none failed), S's 24 from E124. The
unedited readers at c9680c3, read 13:25 GMT-3:

```
## C1 primary (E131): S − T on the four MNAR 60nan files, 15 paired replicates each; plain 95% t intervals, Nadeau–Bengio beside; G2 = S − B; the MCAR files as the fixed-sequence second step

### MNAR
- biodeg_probit_60nan: F1 S − T -0.0031 [-0.0228, +0.0167] (NB [-0.0485, +0.0424]; s42 +0.0053, s7 -0.0070, s13 -0.0075) -> no difference; LL(T) − LL(S)  -0.0000 [-0.0252, +0.0252] (NB [-0.0580, +0.0580]; s42 -0.0078, s7 -0.0029, s13 +0.0107) -> no difference; S − B F1 -0.0036
- kc2_probit_60nan: F1 S − T -0.0106 [-0.0331, +0.0120] (NB [-0.0625, +0.0413]; s42 -0.0141, s7 -0.0012, s13 -0.0164) -> no difference; LL(T) − LL(S)  -0.0846 [-0.2045, +0.0353] (NB [-0.3602, +0.1911]; s42 -0.1463, s7 -0.0067, s13 -0.1006) -> no difference; S − B F1 -0.0095
- vehicle_probit_60nan: F1 S − T +0.0005 [-0.0179, +0.0190] (NB [-0.0419, +0.0430]; s42 +0.0127, s7 +0.0008, s13 -0.0118) -> no difference; LL(T) − LL(S)  -0.0029 [-0.0528, +0.0470] (NB [-0.1177, +0.1118]; s42 -0.0151, s7 +0.0030, s13 +0.0034) -> no difference; S − B F1 +0.0169
- spambase_probit_60nan: F1 S − T -0.0005 [-0.0074, +0.0063] (NB [-0.0163, +0.0153]; s42 -0.0057, s7 -0.0033, s13 +0.0075) -> equivalent; LL(T) − LL(S)  +0.0032 [-0.0071, +0.0134] (NB [-0.0204, +0.0267]; s42 -0.0034, s7 +0.0007, s13 +0.0121) -> no difference; S − B F1 +0.0120

Pooled S − T (MNAR): -0.0034 [-0.0110, +0.0042] (NB [-0.0209, +0.0140]) -> Inconclusive (interval reaches beyond the band); flags: none
G2 S − B (MNAR): +0.0040 [-0.0077, +0.0157] -> fails
Pooled LL(T) − LL(S) (MNAR):  -0.0211 [-0.0573, +0.0152] (NB [-0.1044, +0.0623]; s42 -0.0432, s7 -0.0015, s13 -0.0186) -> no difference

Primary (C1, MNAR): Inconclusive (interval reaches beyond the band); G1 passes

### MCAR
- biodeg_60nan: F1 S − T -0.0000 [-0.0123, +0.0123] (NB [-0.0282, +0.0282]; s42 -0.0168, s7 +0.0126, s13 +0.0042) -> no difference; LL(T) − LL(S)  -0.0095 [-0.0230, +0.0039] (NB [-0.0405, +0.0215]; s42 -0.0138, s7 +0.0015, s13 -0.0163) -> no difference; S − B F1 -0.0102
- kc2_60nan: F1 S − T -0.0111 [-0.0346, +0.0124] (NB [-0.0651, +0.0429]; s42 -0.0181, s7 -0.0065, s13 -0.0087) -> no difference; LL(T) − LL(S)  +0.0013 [-0.0328, +0.0354] (NB [-0.0771, +0.0797]; s42 -0.0302, s7 +0.0384, s13 -0.0044) -> no difference; S − B F1 -0.0042
- vehicle_60nan: F1 S − T +0.0007 [-0.0153, +0.0167] (NB [-0.0360, +0.0374]; s42 -0.0097, s7 +0.0123, s13 -0.0005) -> no difference; LL(T) − LL(S)  +0.0072 [-0.0136, +0.0279] (NB [-0.0405, +0.0549]; s42 +0.0030, s7 +0.0224, s13 -0.0038) -> no difference; S − B F1 +0.0206
- spambase_60nan: F1 S − T +0.0010 [-0.0033, +0.0052] (NB [-0.0088, +0.0108]; s42 -0.0025, s7 +0.0041, s13 +0.0014) -> equivalent; LL(T) − LL(S)  -0.0004 [-0.0100, +0.0093] (NB [-0.0226, +0.0219]; s42 -0.0131, s7 +0.0039, s13 +0.0081) -> no difference; S − B F1 +0.0074

Pooled S − T (MCAR): -0.0024 [-0.0107, +0.0060] (NB [-0.0215, +0.0168]) -> Inconclusive (interval reaches beyond the band); flags: none
G2 S − B (MCAR): +0.0034 [-0.0025, +0.0093] -> fails
Pooled LL(T) − LL(S) (MCAR):  -0.0004 [-0.0114, +0.0107] (NB [-0.0258, +0.0251]; s42 -0.0135, s7 +0.0166, s13 -0.0041) -> no difference

### Mechanism interaction, per table: (S − T) MNAR − (S − T) MCAR, paired by replicate
- biodeg: -0.0031 [-0.0280, +0.0218]
- kc2: +0.0005 [-0.0281, +0.0292]
- vehicle: -0.0002 [-0.0287, +0.0284]
- spambase: -0.0015 [-0.0099, +0.0069]
```

```
## Anchor (descriptive): S(embedding_normalized) − S(embedding) on the four MNAR files, 15 replicates
- biodeg_probit_60nan: F1  +0.0021 [-0.0201, +0.0244] (NB [-0.0491, +0.0533]; s42 -0.0143, s7 +0.0189, s13 +0.0018) -> no difference; LL(embedding) − LL(normalized) +0.0108
- kc2_probit_60nan: F1  -0.0034 [-0.0301, +0.0233] (NB [-0.0648, +0.0580]; s42 -0.0214, s7 +0.0035, s13 +0.0077) -> no difference; LL(embedding) − LL(normalized) -0.0120
- vehicle_probit_60nan: F1  +0.0154 [+0.0034, +0.0274] (NB [-0.0122, +0.0430]; s42 +0.0007, s7 +0.0230, s13 +0.0226) -> better; LL(embedding) − LL(normalized) -0.0034
- spambase_probit_60nan: F1  +0.0058 [-0.0011, +0.0128] (NB [-0.0102, +0.0219]; s42 +0.0047, s7 +0.0019, s13 +0.0109) -> no difference; LL(embedding) − LL(normalized) +0.0109
Pooled:  +0.0050 [-0.0032, +0.0132] (NB [-0.0139, +0.0239]; s42 -0.0076, s7 +0.0118, s13 +0.0108) -> no difference
```

```
## Token geometry in S: median cos([MASK]_j, [NULL]_j) per file; cross-column cos([NULL]_j, [NULL]_j') beside; reference SD at initialisation 1/sqrt(128) = 0.088; 'tied by the model' at median >= 0.9
- biodeg_probit_60nan: after pre-training within +0.004, cross +0.003 (5 folds); after fine-tuning within +0.007, cross -0.000 (15 folds)
- kc2_probit_60nan: after pre-training within -0.014, cross +0.005 (5 folds); after fine-tuning within -0.004, cross +0.002 (15 folds)
- vehicle_probit_60nan: after pre-training within -0.019, cross -0.005 (5 folds); after fine-tuning within +0.011, cross -0.003 (15 folds)
- spambase_probit_60nan: after pre-training within +0.014, cross +0.002 (5 folds); after fine-tuning within +0.012, cross +0.000 (15 folds)
- biodeg_60nan: after pre-training missing; after fine-tuning within +0.006, cross -0.000 (15 folds)
- kc2_60nan: after pre-training missing; after fine-tuning within -0.006, cross +0.002 (15 folds)
- vehicle_60nan: after pre-training missing; after fine-tuning within +0.010, cross -0.003 (15 folds)
- spambase_60nan: after pre-training missing; after fine-tuning within +0.013, cross -0.000 (15 folds)
```

- **Primary (MNAR):** pooled S − T = −0.0034 [−0.0110, +0.0042] macro-F1. The interval contains 0 and its lower end
  reaches just past the −0.010 band, so the status is **Inconclusive** by the rule, not Null; per file, no verdict
  (spambase *equivalent*, the other three *no difference*), no flags, the seed means of mixed sign. Log loss agrees
  (pooled LL(T) − LL(S) −0.021 [−0.057, +0.015]).
- **G2 fails:** S − B = +0.0040 [−0.0077, +0.0157] on the MNAR files (+0.0034 [−0.0025, +0.0093] on the MCAR files).
  Pre-training reaches the head by no measurable amount under this configuration, as E71 found on the MCAR sweep. The
  split lives only in pre-training (fine-tuning shows no `[MASK]`), so with S ≈ B it has no channel: even an interval
  inside the band would have read Inconclusive, not Null.
- **MCAR second step:** the same picture (−0.0024 [−0.0107, +0.0060]); the mechanism interaction is 0 on every table.
  Nothing to scope.
- **Anchor:** `embedding_normalized` against the paper's `embedding`: +0.005 [−0.003, +0.013] pooled; *better* only on
  vehicle (+0.015). The objective choice did not drive the result.
- **Token geometry:** median cos([MASK]_j, [NULL]_j) between −0.02 and +0.01 after pre-training and after fine-tuning,
  at the initialisation reference (SD 0.088), cross-column cosines at 0. The model did not tie the tokens itself; the
  two vectors stay as distinct as at initialisation, which with S ≈ T says the head does not use the distinction.
- **What follows:** C1 stays "untested" in the ledger's sense, with the reason "no pre-training channel" (the
  protocol's Inconclusive row). The reserve (`--cv_folds 10` × 3 seeds) addresses power, not the channel, and is not
  proposed; cut 4 (dose) would measure the dose of an effect that has no channel. The next step is the user's.
