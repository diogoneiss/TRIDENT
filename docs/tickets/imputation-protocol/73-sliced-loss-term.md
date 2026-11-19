# A sliced Wasserstein term in the decode loss: do fills trained on the rows' joint distribution score better on it? (E184, T61; pre-registered 2026-11-19 18:18 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** E183 (ticket 72) showed that choosing the decode checkpoint by `sw_score` (the sliced Wasserstein distance
  between filled and true rows, over the mean/mode fill's) lowers the output's `sw_score` on 15 of 26 MCAR variants.
  Ticket 72 left out the loss, which is per cell (cross-entropy, `lambda_num` × squared error) and has no term on the
  joint distribution of a row's fills. This study adds one.
- **The user's go:** "quero fazer o 1 e 2" (2026-11-19), answering my two offers after E183: 1, offer the `sw_score`
  selector as a reporting option; 2, test a term in the loss on the joint distribution of the rows ("Isso muda o
  algoritmo de treino e precisa do seu sim"). Item 1 is documentation only (README, this commit). The design below is mine.
- **Disclosed:**
  - E183's results were read before this design, and the variants were chosen by cost knowing them; they include
    E183's largest effects (`credit-g_60nan`, `_80nan`, `kr-vs-kp_80nan`).
  - The design was drafted with a multi-agent review (three designs, two judges; the minimal one chosen), then
    implemented and reviewed adversarially; nothing of it trained a study seed.
  - A weight smoke ran on 2026-11-19 at seed 999 (outside the study), MLflow off, first fold's decode only, on
    `credit-g_40nan` and `kr-vs-kp_40nan` at weight 1 and 0. Only the term's and the per-cell loss's training
    magnitudes and the clock were read; no score. A first attempt at 16:54 GMT-3 ran the full two folds with four
    processes oversubscribing the CPU and was killed at 18:05 unread.
  - On this machine's CPU the `vehicle_00nan` classification fixture fails on HEAD as on this change (it was recorded
    on CUDA); it passes on gorgona8's GPU with this change; the `credit-g_20nan` imputation fixture passes on CPU.
- **Code (this commit):** `--decode_sw_weight w` (`DECODE_SW_WEIGHT`, default 0, imputation only, tag
  `decode_sw_weight` with a backfill script, not yet applied); at 0 nothing runs and the run is as before.
  `src/models.py` (`TridentDecoder._sliced_term`, used by both head paths and the short-batch fallback),
  `src/training/decoding.py` (passes it; logs `decode/train_sliced`), plumbing in `types.py`, `config.py`,
  `mlflow_utils.py`, `runner.py`; tests `tests/unit/test_decode_sliced_loss.py`; runner
  `scripts/experiments/sliced_loss_run.py`; README entries for the flag and for `--decode_metric`'s
  `impute_score_classcond` and `sw_score` values (item 1).
- **The term, in training batches only:** the rows holding a hidden cell, as filled (numerical cells by the head's
  value, categorical cells by the head's softmax probabilities at their vocabulary slots) and as true, in `sw_score`'s
  layout (scaled numbers; each categorical column one-hot over its ids plus an out-of-vocabulary slot, /√2); 512 fixed
  unit directions from `default_rng(1)` (the selector uses seed 0); T = √width × mean |sort(filled·A) − sort(true·A)|.
  Loss = per-cell loss + w × T. The validation pass runs in eval mode without it, so `decode/val_loss`, the loss
  checkpoint, the plateau scheduler and patience are as before. Writes go to distinct positions (deterministic on CUDA).
- **The weight, fixed by a rule stated before the smoke:** r = T / per-cell loss, median over decode epochs 0–49 per
  variant; w = 0.25 / (median of the two r), one significant figure; stop if the two r differ by more than 3×. The
  smoke's printout:

  ```
  credit-g_40nan median r ep0-49 0.1672 | T ep0 0.3893 ep49 0.3144 ep449 0.2218 | cell loss ep0 2.4764 ep449 1.1942 | decode s on 51 off 41
  kr-vs-kp_40nan median r ep0-49 0.4835 | T ep0 0.3111 ep49 0.1655 ep449 0.1456 | cell loss ep0 0.6022 ep449 0.2662 | decode s on 64 off 53
  r ratio 2.89 (stop if > 3); median r 0.3254; w = 0.25/median r = 0.7684 -> one significant figure 0.8
  ```

  **w = 0.8.** The term adds about 20% to the decode clock (51 s against 41 s, 64 against 53 s a fold).
- **Scenarios:** 18 MCAR variants (`kc2`, `vehicle`, `biodeg` at 20/40/80nan; `credit-g`, `kr-vs-kp` at 20/40/60/80nan;
  `spambase_20nan`), seeds 101/202/303, five folds. All on gorgona8's RTX 3090 Ti, two queues, both arms of a pair
  side by side, tag `experiment = sliced-loss-2026-11-19`.
  - Core (the primary): **S**, `--decode_metric sw_score` (E183's arm, retrained here), and **LS**, S plus
    `--decode_sw_weight 0.8`. 108 runs. Reading the term at the `sw_score` selector keeps E183's early-epoch effect of
    the default checkpoint out of the contrast.
  - Second batch, after the core: **K**, the default run, and **L**, K plus the term. 108 runs.
  - Cost: about 15 h a batch (E183's and E174's clocks; the term's +20% on decode only).
- **Checks before the read:** `--gate` (every LS/L run carries `decode_sw_weight` 0.8 and a nonzero
  `decode/train_sliced`; S and K carry 0.0); `--pairing` (per pair, scored cell counts and hgb's score the same per
  fold; the loss epoch differs by design). A failing check is reported, not dropped.
- **Measures and decision rule:** each arm's test-fold ledgers scored by E182's code (`sw_score`, `dist_score`,
  `impute_score`) on the calibrated output (ADR 0027's default; no uncalibrated read), 15 (seed, fold) values a
  variant, fold-pair rule with seed agreement, margin 0.01 of the control's mean.
  - Hypothesis: the term lowers the output's `sw_score`.
  - Primary, LS − S on `sw_score` over the variants whose shuffled control (from this study's S files) reaches 0.10:
    supported if better on half or more and worse on at most 2; refuted if better on a sixth or fewer; mixed otherwise.
    The same rule is printed for L − K once the second batch is in.
  - Secondary, no rule: LS − S and L − K on `impute_score` (the cost) and `dist_score`; the mean kept epoch per arm;
    each arm against E182's best strong-tier baseline (medusa1's fits) on `sw_score` and `impute_score`.
  - A null at 60–80nan reads "no gain under this training mask" (about one hidden cell a training row there), not
    "a joint term does not help".
  - Read once: `sliced_loss_run.py --tally core` when the core is in, `--tally` when all four arms are. No default
    changes whatever the result.
- **Not in this study:** pre-training (unchanged); other weights; calibration off.

## Outcome

Not read yet.
