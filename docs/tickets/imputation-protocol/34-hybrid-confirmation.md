# Does the deployable hybrid's `table` rule hold on fresh seeds, and where it is not needed? (E90, E91; pre-registered 2026-10-28)

This ticket was written and committed before the studies' queues started.

- **Why:** E89 ([33](33-hybrid-selection-rules.md)) found two rules that close most of the
  deployable hybrid's gap to TabPFN. Both are better than H on 10 of 10. `table` is worse than
  TabPFN on none, and the primary `shrunk` is worse on 2. That was a development read on E82's own
  seeds, and E89 fixed beforehand that a confirmation on fresh seeds must follow, with one rule.
- **The user's go:** on the confirmation, and to keep gorgona8 busy until about 09:00 ("voce pode
  rodar ate mais coisa, quero ocupar a gorgona8 até umas 9 da manha", 2026-10-28, about 22:50
  GMT-3).
- **The rule, fixed now: `table`.** It is the simpler rule, and it did better than `shrunk` in
  E89. `shrunk`, `column` and `pooled` are printed as secondary and decide nothing.
  - `table` is one numerical weight per table, from E69's grid, with each column's residuals
    scaled by its training split's sd, plus one categorical pick per table. Both are chosen on fold
    k's hidden observed validation cells.
- **Code:** no change to `src/`. `scripts/experiments/hybrid_confirmation.py` points E82's
  `train` and E89's `cells` at each study's tag and state. `run_hybrid_confirmation_queue.sh` runs
  four training queues and two cells lanes. A lane takes each cell as soon as its run is in the
  store copy.
- **Smoke:** E90's vehicle_80nan s4545 was trained before this commit (CELL_DONE; five selection
  files). Only its run's headline was printed (H 0.814), and no rule was scored. It counts.

## Design (fixed before the queues)

- **Seeds 4545, 5656, 6767,** used by no earlier run in the store.
- **E90, the confirmation:** E82's 10 hard variants (`blend_weights.HARD`), so 30 cells. These are
  electricity at 40 and 80nan; vehicle and biodeg at 80nan; kr-vs-kp and credit-g at 40, 60 and
  80nan. Each cell is E82's exactly:
  - H (`--decode_checkpoint loss --decode_calibration none`) with the hidden observed validation
    cells read in place;
  - then TabPFN v2 refit on each training fold.
- **E91, where the hybrid is not needed:** 7 of E44's confirmation variants outside the hard set,
  where E60's equal blend already beat TabPFN, so 21 cells. They are biodeg_40nan, kc2_40nan, kc2_80nan, pendigits at 40, 60 and
  80nan, and vehicle_40nan.
  - The question: does the deployable hybrid cost anything there?
  - letter and electricity_60nan are left out to fit the GPU window.
- **Running:** gorgona8. Four training queues deal both studies' cells by measured minutes,
  longest first; two cells lanes follow the training.
  - MLflow tags `experiment=hybrid-confirmation-2026-10-28` (E90) and
    `hybrid-generality-2026-10-28` (E91), logged through the leader's server.
  - About 4.7 h of training on four queues, and the refits overlap it; done by about 06:00–07:30.
  - **Deadlines, by the clock alone:** no training cell starts after 2026-10-29 07:00 GMT-3, and no
    cells job after 08:40 GMT-3. A study with cells missing at the deadline is reported partial,
    with no call.

- **Amendment 2026-10-29, 06:45 GMT-3, before any E90 or E91 result was read:** the training
  deadline moves from 07:00 to 08:00 GMT-3.
  - The queues ran slower than planned: electricity took about 2 h 50 min a cell with four sharing
    the GPU, against E82's 1 h 46 min. At 06:45, 40 of 51 cells were trained, and four to six small
    E91 cells (kc2, vehicle) would have missed 07:00.
  - The user's window runs to about 09:00.
  - **How:** the four queues keep their 07:00 cut. A rescue runner starts only after all of them
    have stopped. It trains the cells they left, one at a time, starting none after 08:00. No cell
    can run twice.
  - The cells deadline stays at 08:40. Anything missing then is still reported partial.

## Measures and rule, fixed now (`hybrid_confirmation.py --study <e90|e91> --tally`)

- **Per variant:** `table` − H, `table` − TabPFN and `table` − the equal blend on `impute_score`,
  over 15 fold pairs. The project's rule applies, with a SESOI of 1% of the opponent's mean (E82's).
- **E90, in E82's form:**
  - **supported** if better than H on 8 or more of 10, worse on none, and worse than TabPFN on 2 or
    fewer;
  - **refuted** if better than H on 5 or fewer, or worse on 2 or more, or worse than TabPFN on 5 or
    more;
  - **mixed** otherwise.
- **E91:**
  - **supported** (no cost) if worse than H on none and worse than TabPFN on 1 or fewer of 7;
  - **refuted** if worse than H on 2 or more, or worse than TabPFN on 3 or more;
  - **mixed** otherwise.
- **What follows:** if E90 is supported, the deployable hybrid with `table` has a fresh-seed
  result. Whether ADR 0031 gains a deployable output is then the user's decision, with E91 saying
  whether it costs anything elsewhere.

## Outcome

**Done 2026-10-29, 07:37 GMT-3: E90 supported, E91 supported.** All 51 cells were trained and
refit on gorgona8, 2026-10-28 23:10 to 2026-10-29 07:35 GMT-3, with none failed:
- 46 in the four queues, 4 by the rescue runner after the 07:00 cut (amendment), and the smoke
  cell;
- every refit within the 08:40 deadline.

E90's tally, unedited (`hybrid_confirmation.py --study e90 --tally`):

```
- `electricity_40nan` (H 0.7544, TabPFN 0.7427): column − tabpfn -0.0152 [-0.0215, -0.0088] better; table − model -0.0140 [-0.0350, +0.0070] none; table − tabpfn -0.0023 [-0.0127, +0.0080] none; table − equal +0.0065 [+0.0023, +0.0106] worse; shrunk − tabpfn -0.0152 [-0.0215, -0.0089] better; pooled − tabpfn -0.0072 [-0.0186, +0.0042] none
- `electricity_80nan` (H 0.8981, TabPFN 0.8907): column − tabpfn -0.0009 [-0.0015, -0.0002] better; table − model -0.0084 [-0.0101, -0.0066] better; table − tabpfn -0.0010 [-0.0017, -0.0003] better; table − equal -0.0004 [-0.0012, +0.0003] equivalent; shrunk − tabpfn -0.0009 [-0.0016, -0.0003] better; pooled − tabpfn -0.0014 [-0.0022, -0.0007] better
- `vehicle_80nan` (H 0.8119, TabPFN 0.7536): column − tabpfn +0.0052 [+0.0015, +0.0089] worse; table − model -0.0581 [-0.0633, -0.0528] better; table − tabpfn +0.0002 [-0.0019, +0.0023] equivalent; table − equal -0.0062 [-0.0092, -0.0032] better; shrunk − tabpfn +0.0036 [+0.0007, +0.0066] worse; pooled − tabpfn -0.0018 [-0.0036, -0.0001] better
- `biodeg_80nan` (H 0.9855, TabPFN 0.9188): column − tabpfn +0.0082 [+0.0042, +0.0122] worse; table − model -0.0693 [-0.0775, -0.0611] better; table − tabpfn -0.0027 [-0.0055, +0.0001] equivalent; table − equal -0.0108 [-0.0150, -0.0067] better; shrunk − tabpfn +0.0044 [-0.0004, +0.0091] equivalent; pooled − tabpfn +0.0044 [+0.0010, +0.0078] worse
- `kr-vs-kp_40nan` (H 0.7256, TabPFN 0.7118): column − tabpfn +0.0037 [-0.0012, +0.0086] none; table − model -0.0072 [-0.0132, -0.0012] better; table − tabpfn +0.0065 [-0.0002, +0.0133] none; table − equal -0.0072 [-0.0132, -0.0012] better; shrunk − tabpfn +0.0037 [-0.0012, +0.0086] none; pooled − tabpfn +0.0047 [-0.0006, +0.0099] none
- `kr-vs-kp_60nan` (H 0.8715, TabPFN 0.8301): column − tabpfn +0.0159 [+0.0113, +0.0206] worse; table − model -0.0414 [-0.0474, -0.0355] better; table − tabpfn +0.0000 [+0.0000, +0.0000] equivalent; table − equal -0.0414 [-0.0474, -0.0355] better; shrunk − tabpfn +0.0159 [+0.0113, +0.0206] worse; pooled − tabpfn +0.0045 [+0.0012, +0.0078] worse
- `kr-vs-kp_80nan` (H 1.0096, TabPFN 0.9406): column − tabpfn +0.0229 [+0.0164, +0.0295] worse; table − model -0.0652 [-0.0809, -0.0494] better; table − tabpfn +0.0039 [-0.0044, +0.0121] none; table − equal -0.0652 [-0.0809, -0.0494] better; shrunk − tabpfn +0.0219 [+0.0160, +0.0277] worse; pooled − tabpfn +0.0103 [+0.0055, +0.0150] worse
- `credit-g_40nan` (H 0.9378, TabPFN 0.8999): column − tabpfn +0.0129 [+0.0077, +0.0181] worse; table − model -0.0381 [-0.0500, -0.0261] better; table − tabpfn -0.0003 [-0.0033, +0.0028] equivalent; table − equal -0.0311 [-0.0414, -0.0208] better; shrunk − tabpfn +0.0116 [+0.0064, +0.0169] worse; pooled − tabpfn +0.0075 [+0.0017, +0.0133] none
- `credit-g_60nan` (H 1.0018, TabPFN 0.9510): column − tabpfn +0.0150 [+0.0072, +0.0227] worse; table − model -0.0454 [-0.0590, -0.0317] better; table − tabpfn +0.0054 [-0.0041, +0.0149] none; table − equal -0.0336 [-0.0460, -0.0212] better; shrunk − tabpfn +0.0052 [-0.0046, +0.0150] none; pooled − tabpfn +0.0098 [+0.0013, +0.0184] worse
- `credit-g_80nan` (H 1.0382, TabPFN 1.0028): column − tabpfn +0.0111 [+0.0031, +0.0191] worse; table − model -0.0305 [-0.0489, -0.0121] better; table − tabpfn +0.0049 [-0.0041, +0.0139] none; table − equal -0.0215 [-0.0356, -0.0074] better; shrunk − tabpfn +0.0039 [-0.0043, +0.0121] none; pooled − tabpfn +0.0064 [-0.0012, +0.0141] none

Secondary `column`: better than H on 9 of 10, worse on none, worse than TabPFN on 7 (['vehicle_80nan', 'biodeg_80nan', 'kr-vs-kp_60nan', 'kr-vs-kp_80nan', 'credit-g_40nan', 'credit-g_60nan', 'credit-g_80nan']) -> refuted
Primary `table`: better than H on 9 of 10, worse on none, worse than TabPFN on 0 (none) -> supported
Secondary `shrunk`: better than H on 9 of 10, worse on none, worse than TabPFN on 4 (['vehicle_80nan', 'kr-vs-kp_60nan', 'kr-vs-kp_80nan', 'credit-g_40nan']) -> mixed
Secondary `pooled`: better than H on 9 of 10, worse on none, worse than TabPFN on 4 (['biodeg_80nan', 'kr-vs-kp_60nan', 'kr-vs-kp_80nan', 'credit-g_60nan']) -> mixed

(fold-pair t, seed agreement; SESOI 1% of the opponent's mean)
```

E91's tally, unedited (`--study e91 --tally`):

```
- `biodeg_40nan` (H 0.6832, TabPFN 0.7149): column − tabpfn -0.0469 [-0.0693, -0.0244] better; table − model -0.0216 [-0.0297, -0.0135] better; table − tabpfn -0.0533 [-0.0740, -0.0326] better; table − equal -0.0019 [-0.0068, +0.0029] none; shrunk − tabpfn -0.0536 [-0.0747, -0.0325] better; pooled − tabpfn -0.0620 [-0.0847, -0.0392] better
- `kc2_40nan` (H 0.5057, TabPFN 0.5067): column − tabpfn -0.0353 [-0.0624, -0.0083] better; table − model -0.0174 [-0.0621, +0.0273] none; table − tabpfn -0.0184 [-0.0467, +0.0099] none; table − equal +0.0325 [+0.0058, +0.0592] worse; shrunk − tabpfn -0.0544 [-0.0814, -0.0274] better; pooled − tabpfn -0.0377 [-0.0651, -0.0103] better
- `kc2_80nan` (H 0.6802, TabPFN 0.7396): column − tabpfn -0.0487 [-0.0650, -0.0325] better; table − model +0.0009 [-0.0287, +0.0305] none; table − tabpfn -0.0585 [-0.0780, -0.0390] better; table − equal +0.0084 [-0.0035, +0.0203] none; shrunk − tabpfn -0.0671 [-0.0821, -0.0520] better; pooled − tabpfn -0.0590 [-0.0721, -0.0459] better
- `pendigits_40nan` (H 0.4274, TabPFN 0.4363): column − tabpfn -0.0212 [-0.0229, -0.0195] better; table − model -0.0122 [-0.0128, -0.0117] better; table − tabpfn -0.0212 [-0.0230, -0.0193] better; table − equal +0.0026 [+0.0020, +0.0032] worse; shrunk − tabpfn -0.0217 [-0.0234, -0.0200] better; pooled − tabpfn -0.0215 [-0.0232, -0.0197] better
- `pendigits_60nan` (H 0.5902, TabPFN 0.6038): column − tabpfn -0.0265 [-0.0284, -0.0246] better; table − model -0.0139 [-0.0155, -0.0124] better; table − tabpfn -0.0275 [-0.0293, -0.0258] better; table − equal +0.0004 [-0.0002, +0.0009] equivalent; shrunk − tabpfn -0.0270 [-0.0288, -0.0251] better; pooled − tabpfn -0.0273 [-0.0292, -0.0254] better
- `pendigits_80nan` (H 0.8360, TabPFN 0.8144): column − tabpfn -0.0023 [-0.0042, -0.0004] better; table − model -0.0253 [-0.0295, -0.0210] better; table − tabpfn -0.0036 [-0.0056, -0.0017] better; table − equal -0.0014 [-0.0024, -0.0003] better; shrunk − tabpfn -0.0029 [-0.0048, -0.0010] better; pooled − tabpfn -0.0037 [-0.0054, -0.0021] better
- `vehicle_40nan` (H 0.4964, TabPFN 0.5008): column − tabpfn -0.0119 [-0.0181, -0.0057] better; table − model -0.0135 [-0.0217, -0.0052] better; table − tabpfn -0.0179 [-0.0291, -0.0067] better; table − equal +0.0015 [-0.0001, +0.0032] equivalent; shrunk − tabpfn -0.0172 [-0.0255, -0.0089] better; pooled − tabpfn -0.0138 [-0.0244, -0.0032] better

Secondary `column`: better than H on 4 of 7, worse on none, worse than TabPFN on 0 (none) -> supported
Primary `table`: better than H on 5 of 7, worse on none, worse than TabPFN on 0 (none) -> supported
Secondary `shrunk`: better than H on 6 of 7, worse on none, worse than TabPFN on 0 (none) -> supported
Secondary `pooled`: better than H on 5 of 7, worse on none, worse than TabPFN on 0 (none) -> supported

(fold-pair t, seed agreement; SESOI 1% of the opponent's mean)
```

- **E90, primary `table`: supported on fresh seeds.**
  - Better than H on 9 of 10 and worse on none. electricity_40nan has no verdict: −0.0140, with an
    interval crossing zero.
  - **Worse than TabPFN on none.** It is better than TabPFN on electricity_80nan, equivalent on 4
    (vehicle and biodeg at 80nan, kr-vs-kp_60nan, credit-g_40nan) and without verdict on 5
    (electricity_40nan, kr-vs-kp at 40 and 80nan, credit-g at 60 and 80nan).
  - E89's development result therefore holds on seeds the rule never saw.
- **The secondary rules on the same seeds:**
  - `column` (E82's): worse than TabPFN on 7, refuted again;
  - `shrunk`: on 4, mixed. E89's primary does not replicate;
  - `pooled`: on 4, mixed.
  - So the single categorical pick per table is what holds kr-vs-kp and credit-g. Shrinkage alone
    does not.
- **E91, primary `table`: supported (no cost).**
  - Worse than H on none of 7 (better on 5).
  - Worse than TabPFN on none: better on 6, and kc2_40nan without verdict.
- **Descriptive:**
  - `table` is worse than the equal blend on electricity_40nan (+0.0065), kc2_40nan (+0.0325) and
    pendigits_40nan (+0.0026).
  - Where one weight per table is too coarse for mixed numerical columns, the equal blend can be a
    little better.
- **What follows, as fixed:** the deployable hybrid with `table` has a fresh-seed result. It reads
  no complete table, beats H on 14 of 17, and is never worse than H or TabPFN on the 17 variants.
  Whether ADR 0031 gains it as a deployable output is the user's decision.
