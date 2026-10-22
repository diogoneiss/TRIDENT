# ADR 0018's bigger encoder on the large tables at 40–80nan (E59; pre-registered 2026-10-21)

This ticket was written and committed before the first cell.

- **Why:** ADR 0018 put E37's ALL arm on pendigits_20nan and letter_20nan. ALL is 4 layers,
  feed-forward 256 and a final LayerNorm.
  - E38 tested it only on the 21 development variants. The large tables appear there at 20nan
    alone (pendigits −3.8%, letter −5.6%; electricity +0.3%, n.s.).
  - It was never tried at 40–80nan, where E55 and E57 find TabPFN ahead of the default output at
    80nan on pendigits and letter.
- **The user's go:** "sim, vamos" (2026-10-21, about 19:20 GMT-3), to the proposal of the bigger
  encoder at 40–80nan tonight.
- **No new code in `src/`.** ALL is a configuration (`LAYERS 4`, `DIM_FEED 256`,
  `ENCODER_FINAL_NORM layer`). Runner: `scripts/experiments/large_encoder_run.py`; tag
  `experiment=large-encoder-2026-10-21`, arm `ALL`.

## Design (fixed before the first cell)

- **Cells:** pendigits and letter at 40, 60 and 80nan, seeds 707/808/909, arm ALL: 18 cells.
- **G** is E44's run of today's default on the same seed and folds. No promoted file applies to
  these variants.
  - E44 ran before ADR 0027, and the K path has not moved since. E51 reproduced E35 and E50
    reproduced E38 and E39, each to a difference of 0.
  - KC is E44's `--score_calibrated` readout against ALL's ADR 0027 headline: the same calibration
    on the validation gaps.
- **electricity is left out.** ALL did not help it at 20nan (E38), and a cell would take about 3 h.
- **The seeds are E44's confirmation seeds,** so this is a development reading.
  - Any adoption needs a fresh-seed confirmation, pre-registered first, before an ADR amending
    0018.
- **Running it:** four GPU queues (`run_large_encoder_queue.sh`), longest first. About 6.6 h,
  beside E60's one lane. No cutoff.

## Measures and reading, fixed now

`scripts/experiments/large_encoder_report.py --tally`, on the induced `impute_score` over 15 fold
pairs, with the verdict rule and the 1% SESOI:

- **Primary:** ALL − G on K.
  - **supported:** better on 4 or more of the 6, worse on none;
  - **refuted:** better on 1 or none, or worse on 2 or more;
  - **mixed:** otherwise.
- **Secondary:**
  - ALL − G on KC (the default output) and on H;
  - G's and ALL's KC − TabPFN (E55's CSVs): does the bigger encoder close the 80nan gap?
- **What each reading leads to (no default changes from this study):**
  - **supported:** a fresh-seed confirmation, then an ADR amending 0018 for the user;
  - **refuted:** ADR 0018 stays at 20nan;
  - **mixed:** reported, and the user chooses.

## Outcome (2026-10-22, 06:03 GMT-3)

**The run:** 18 of 18 cells, 19:28 GMT-3 on 2026-10-21 to 04:11 GMT-3 on 2026-10-22, with none
failed. That is 8 h 43 min against the ticket's 6.6 h estimate.

- A letter cell took about 183 min while E60's lane shared the GPU, and about 135 min after it
  ended.
- A pendigits cell took 75 min, falling to about 53 min as the queues drained.

**The printout** (`scripts/experiments/large_encoder_report.py --tally`, unedited):

```text
- `pendigits_40nan`: ALL − G, K -0.0177 (-4.2% of G's K) [-0.0206, -0.0148] better; ALL − G, KC -0.0244 (-5.7% of G's K) [-0.0270, -0.0218] better; ALL − G, H -0.0167 (-3.9% of G's K) [-0.0192, -0.0142] better; G KC − tabpfn -0.0119 better; ALL KC − tabpfn -0.0363 better
- `pendigits_60nan`: ALL − G, K -0.0069 (-1.2% of G's K) [-0.0102, -0.0036] better; ALL − G, KC -0.0144 (-2.5% of G's K) [-0.0174, -0.0114] better; ALL − G, H -0.0043 (-0.7% of G's K) [-0.0105, +0.0018] none; G KC − tabpfn -0.0216 better; ALL KC − tabpfn -0.0360 better
- `pendigits_80nan`: ALL − G, K +0.0013 (+0.2% of G's K) [-0.0008, +0.0034] equivalent; ALL − G, KC -0.0017 (-0.2% of G's K) [-0.0044, +0.0010] equivalent; ALL − G, H +0.0010 (+0.1% of G's K) [-0.0027, +0.0047] equivalent; G KC − tabpfn +0.0121 worse; ALL KC − tabpfn +0.0104 worse
- `letter_40nan`: ALL − G, K -0.0212 (-3.8% of G's K) [-0.0229, -0.0195] better; ALL − G, KC -0.0277 (-4.9% of G's K) [-0.0292, -0.0261] better; ALL − G, H -0.0216 (-3.9% of G's K) [-0.0236, -0.0195] better; G KC − tabpfn -0.0358 better; ALL KC − tabpfn -0.0635 better
- `letter_60nan`: ALL − G, K -0.0005 (-0.1% of G's K) [-0.0023, +0.0012] equivalent; ALL − G, KC -0.0073 (-1.0% of G's K) [-0.0092, -0.0055] better; ALL − G, H -0.0037 (-0.5% of G's K) [-0.0081, +0.0007] none; G KC − tabpfn -0.0331 better; ALL KC − tabpfn -0.0404 better
- `letter_80nan`: ALL − G, K +0.0036 (+0.4% of G's K) [+0.0023, +0.0050] worse; ALL − G, KC +0.0009 (+0.1% of G's K) [-0.0003, +0.0021] equivalent; ALL − G, H +0.0046 (+0.5% of G's K) [+0.0020, +0.0072] worse; G KC − tabpfn -0.0002 equivalent; ALL KC − tabpfn +0.0007 equivalent

Primary, ALL − G on K: better on ['pendigits_40nan', 'pendigits_60nan', 'letter_40nan'], worse on ['letter_80nan'] -> mixed (fold-pair t, seed agreement)
```

**Verdict: mixed.** On K, ALL − G is better on 3 of the 6 variants (pendigits 40 and 60nan,
letter_40nan), equivalent on 2, and worse on 1 (letter_80nan).

- **The reading would be mixed without the worse call too.** "Supported" needs 4 better, and
  there are 3.
- **letter_80nan's worse call is small.** Its mean is +0.4% of G's K, with interval
  [+0.0023, +0.0050], inside the 1% SESOI.
  - The rule checks sign and seed agreement before equivalence, so it counts as worse.

**Reading:**

- **The gain fades with the missing rate.** ALL − G on K, as a share of G's K:

  | table | 40nan | 60nan | 80nan |
  |---|---|---|---|
  | pendigits | −4.2% | −1.2% | +0.2% |
  | letter | −3.8% | −0.1% | +0.4% |

  At 40nan the bigger encoder gains about what E38 found at 20nan (−3.8% and −5.6%). At 80nan it
  gains nothing.
- **On KC, the default output at these rates (secondary):** better 4 (pendigits and letter at 40
  and 60nan), equivalent 2 (both at 80nan), worse 0.
  - Had KC been the primary, this would have met the supported threshold. The verdict stays on K,
    as fixed.
  - On KC, ALL's gain is larger than on K at 40nan (−5.7% and −4.9%) and at 60nan (−2.5% and
    −1.0%): calibration helps ALL more than it helps G.
- **On H (secondary):** better 2, no difference 2, equivalent 1, worse 1 (letter_80nan, +0.5%).
- **The 80nan gap to TabPFN does not close.** KC − TabPFN:
  - pendigits_80nan stays worse (G +0.0121, ALL +0.0104);
  - letter_80nan stays equivalent (G −0.0002, ALL +0.0007).
  - Where K already led TabPFN, the lead widens, e.g. letter_40nan from −0.0358 to −0.0635.
- **Cost:** ALL is E37's 1.82× arm.
- **The seeds are E44's confirmation seeds,** so this is a development reading.

**What it leads to (mixed: the user chooses), with no default change:**

- **ADR 0018 stays at 20nan.** The bigger encoder does not help where TabPFN leads.
- **Or a fresh-seed confirmation** of ALL on pendigits and letter at 40nan, and at 60nan on KC,
  pre-registered first, before any ADR amending 0018.

## Decision (2026-10-22, 06:10 GMT-3)

- **The user's words:** "deixe o encoder maior como possibilidade futura para extrair mais um pouco
  de performance, porem por enquanto nao vale a pena."
- **ADR 0018 stays at 20nan.** No fresh-seed confirmation is run now.
- **The bigger encoder at 40–60nan is recorded as a future possibility** (TASKS T32). It gains
  about 4% on K at 40nan and more on KC, at about twice the training time.
- **The measured cost,** per fold on cells with about three other trainings alongside, the load
  E44's runs had: 1.98× on pendigits (883 s against 446 s) and 2.05× on letter (1,559 s against
  760 s).
  - The Outcome's 1.82× was E37's screening figure.
  - The letter cells that shared the GPU with E60's lane are left out; they reached 2.9×.
