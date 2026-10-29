# Does the deployable `table` hybrid hold at 20nan? (E95; pre-registered 2026-10-29)

This ticket was written and committed before the study's first cell.

- **Why:** ADR 0031's decision 6 (accepted 2026-10-29) covers MCAR at 40–80nan (E90, E91), and
  E93 tests MNAR and MAR. 20nan is the last untested MCAR rate. There the model usually leads
  TabPFN, so the question is E91's: does the hybrid cost anything?
- **The user's go:** "sim, vamos" (2026-10-29, about 15:00 GMT-3), on running it whole on medusa1.
- **Code:** no change to `src/`.
  - `hybrid_confirmation.py --study e95` runs E90's pipeline.
  - `--todo <n> e95` and `STUDY=e95` in `run_hybrid_confirmation_queue.sh` limit the queues to this
    study.
  - The TabPFN v2 regressor checkpoint was copied to medusa1 from gorgona8's cache; the md5 of
    both checkpoints matches.
- **Smoke:** kc2_20nan s4545, trained and refit on medusa1 before the queues start. It counts, and
  only its reproduction (the ledger equals the run's H) is read.

## Design (fixed before the first cell)

- **Cells:** the nine tables at 20nan × seeds 4545, 5656 and 6767, so 27 cells. The tables are
  electricity, letter, pendigits, spambase, kr-vs-kp, biodeg, credit-g, vehicle and kc2.
  - Each cell is E90's: H with the hidden observed validation cells read in place, then TabPFN v2
    refit on each training fold, and the `table` rule.
- **Running:** all on medusa1, so every cell's H and TabPFN share a machine (E86).
  - Four training queues and two cells lanes, all on GPU0 (`CUDA_VISIBLE_DEVICES=0`); E94 uses
    GPU1.
  - MLflow tag `experiment=hybrid-20nan-2026-10-29`, logged through the leader's server; state in
    `/scratch/diogoneiss/trident-2026-10-29-hybrid-20nan/`.
  - About 5–7 h: electricity, letter and pendigits dominate both training and refits.
  - **Deadlines:** no training cell starts after 2026-10-30 02:00 GMT-3, and no cells job after
    03:00 GMT-3. Anything missing then is partial.

## Measures and rule, fixed now (`hybrid_confirmation.py --study e95 --tally`)

- **Per variant:** `table` − H and `table` − TabPFN on `impute_score`, over 15 fold pairs, with a
  SESOI of 1% of the opponent's mean.
- **The call, in E91's form** (does it cost anything):
  - **supported** if worse than H on none and worse than TabPFN on 1 or fewer of 9;
  - **refuted** if worse than H on 2 or more, or worse than TabPFN on 3 or more;
  - **mixed** otherwise.
- **Descriptive:** better-than counts, and `table` − the equal blend.
- **What follows:** if supported, extending decision 6's scope to 20nan is the user's decision.

## Outcome

**Done 2026-10-29, 19:36 GMT-3: supported (no cost at 20nan).** All 27 cells were trained and refit
on medusa1 GPU0, 15:03–19:35 GMT-3, with none failed: 26 in the queues plus the smoke cell. The
tally ran on medusa1 (also `tally.txt` in the state directory). Unedited:

```
- `electricity_20nan` (H 0.6233, TabPFN 0.6259): column − tabpfn -0.0190 [-0.0293, -0.0087] better; table − model -0.0101 [-0.0145, -0.0057] better; table − tabpfn -0.0128 [-0.0241, -0.0014] better; table − equal +0.0044 [-0.0014, +0.0103] none; shrunk − tabpfn -0.0190 [-0.0293, -0.0087] better; pooled − tabpfn -0.0188 [-0.0290, -0.0086] better
- `letter_20nan` (H 0.4345, TabPFN 0.4542): column − tabpfn -0.0353 [-0.0374, -0.0332] better; table − model -0.0156 [-0.0161, -0.0151] better; table − tabpfn -0.0353 [-0.0374, -0.0332] better; table − equal +0.0018 [+0.0012, +0.0024] worse; shrunk − tabpfn -0.0356 [-0.0377, -0.0335] better; pooled − tabpfn -0.0353 [-0.0374, -0.0332] better
- `pendigits_20nan` (H 0.3200, TabPFN 0.3090): column − tabpfn -0.0132 [-0.0176, -0.0089] better; table − model -0.0214 [-0.0221, -0.0207] better; table − tabpfn -0.0103 [-0.0149, -0.0057] better; table − equal +0.0095 [+0.0083, +0.0106] worse; shrunk − tabpfn -0.0138 [-0.0182, -0.0095] better; pooled − tabpfn -0.0124 [-0.0170, -0.0078] better
- `spambase_20nan` (H 0.8479, TabPFN 0.9367): column − tabpfn -0.0842 [-0.0991, -0.0692] better; table − model -0.0003 [-0.0014, +0.0007] equivalent; table − tabpfn -0.0891 [-0.1057, -0.0725] better; table − equal -0.0180 [-0.0240, -0.0120] better; shrunk − tabpfn -0.0847 [-0.0993, -0.0701] better; pooled − tabpfn -0.0863 [-0.1008, -0.0718] better
- `kr-vs-kp_20nan` (H 0.5899, TabPFN 0.5847): column − tabpfn +0.0018 [-0.0043, +0.0080] none; table − model -0.0017 [-0.0075, +0.0040] none; table − tabpfn +0.0035 [-0.0034, +0.0103] none; table − equal -0.0017 [-0.0075, +0.0040] none; shrunk − tabpfn +0.0018 [-0.0043, +0.0080] none; pooled − tabpfn +0.0009 [-0.0069, +0.0087] none
- `biodeg_20nan` (H 0.5930, TabPFN 0.6150): column − tabpfn -0.0429 [-0.0596, -0.0261] better; table − model -0.0167 [-0.0256, -0.0079] better; table − tabpfn -0.0387 [-0.0573, -0.0201] better; table − equal +0.0074 [-0.0012, +0.0160] none; shrunk − tabpfn -0.0488 [-0.0659, -0.0318] better; pooled − tabpfn -0.0466 [-0.0639, -0.0294] better
- `credit-g_20nan` (H 0.8858, TabPFN 0.8727): column − tabpfn +0.0012 [-0.0052, +0.0076] equivalent; table − model -0.0171 [-0.0267, -0.0075] better; table − tabpfn -0.0041 [-0.0076, -0.0005] better; table − equal -0.0095 [-0.0186, -0.0005] better; shrunk − tabpfn +0.0007 [-0.0057, +0.0071] equivalent; pooled − tabpfn -0.0027 [-0.0103, +0.0050] none
- `vehicle_20nan` (H 0.4197, TabPFN 0.4170): column − tabpfn -0.0107 [-0.0225, +0.0011] none; table − model -0.0144 [-0.0191, -0.0096] better; table − tabpfn -0.0117 [-0.0277, +0.0043] none; table − equal +0.0037 [-0.0005, +0.0079] none; shrunk − tabpfn -0.0143 [-0.0270, -0.0016] better; pooled − tabpfn -0.0110 [-0.0250, +0.0029] none
- `kc2_20nan` (H 0.5369, TabPFN 0.5029): column − tabpfn -0.0380 [-0.0957, +0.0197] none; table − model -0.0587 [-0.1218, +0.0045] none; table − tabpfn -0.0247 [-0.0646, +0.0152] none; table − equal +0.0094 [-0.0135, +0.0324] none; shrunk − tabpfn -0.0456 [-0.0929, +0.0017] none; pooled − tabpfn -0.0498 [-0.1056, +0.0061] none

Secondary `column`: better than H on 7 of 9, worse on none, worse than TabPFN on 0 (none) -> supported
Primary `table`: better than H on 6 of 9, worse on none, worse than TabPFN on 0 (none) -> supported
Secondary `shrunk`: better than H on 7 of 9, worse on none, worse than TabPFN on 0 (none) -> supported
Secondary `pooled`: better than H on 7 of 9, worse on none, worse than TabPFN on 0 (none) -> supported

(fold-pair t, seed agreement; SESOI 1% of the opponent's mean)
```

- **Primary, `table`: supported.**
  - Worse than H on none of 9, and better on 6.
  - Worse than TabPFN on none, and better on 6 (electricity, letter, pendigits, spambase, biodeg,
    credit-g).
  - The other three have no verdict: kr-vs-kp, vehicle and kc2. kc2 has wide intervals.
- **All four rules are supported** at 20nan. H's own mean is below TabPFN's on only 4 of 9
  (electricity, letter, spambase, biodeg), so the hybrid's lead over TabPFN comes from the blend.
- **Descriptive:** `table` is a little worse than the equal blend on letter (+0.0018) and pendigits
  (+0.0095), where one weight per table is coarse for many numerical columns, as E91 found at 40nan.
- **What follows:** decision 6's MCAR scope can extend to 20nan (E95 with E90 and E91: 26 variants,
  never worse than H or TabPFN). That is the user's decision.
