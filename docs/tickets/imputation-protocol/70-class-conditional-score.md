# TRIDENT against its baselines on `impute_score_classcond`, the score measured against the class-conditional fill (E181, T58; pre-registered 2026-11-17 14:35 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** `impute_score` divides a method's error by the error of the naive fill, the training fold's mean or mode of
  the column. The user proposed a stricter naive fill: the mean or mode among the training rows of the scored row's own
  class (class-conditional mean imputation), and asked whether TRIDENT stays competitive on that scale when it is also
  selected by it.
- **The user's go:** "eu tenho um novo experimento para propor: impute_score considerando a imputacao pela media/mode
  calculada onde y=A [...] Gostaria que fosse implementado no codigo essa segunda metrica adicional (no momento nao vamos
  alterar defaults, mas quero que no lugar do impute_score normal voce tente otimizar contra ela) [...] e ao final compare
  contra os baselines, quero saber se ainda assim trident continua competitivo contra baselines" (2026-11-17, about
  13:55 GMT-3). The name `impute_score_classcond` is my proposal, as asked.
- **Code (this commit):** `--decode_metric impute_score_classcond` (`DECODE_METRIC`), a third value beside `impute_score`
  and `dist_score`; no default changes, and a run that does not ask logs nothing new (both regression fixtures pass).
  - `src/training/imputation_metrics.py`: `class_mean_mode_baselines`, `class_naive_errors`, `class_conditional_score`.
  - `src/training/decoding.py`: under that metric the `induced` and `masked` checkpoints keep the epoch the score names,
    a search ranks by `validation/impute/<population>/impute_score_classcond`, and the run logs the score for the model
    and for every in-run baseline, with the class-conditional fill's own two errors (`classcond_naive_rmse_num_z`,
    `classcond_naive_error_cat`), on the masked and induced populations.
  - `scripts/experiments/classcond_run.py`: `train`, `--todo`, `--tally`.
- **What the score is:** the method's own pooled errors (`rmse_num_z`, `1 − acc_cat`) over the class-conditional fill's
  errors on the same cells, weighted by the cells of each kind as `impute_score` weights them. The class-conditional
  fill scores 1. A column a class never shows in the training fold, and a class the fold never shows, keep the
  whole-fold fill. The fill reads the row's class; the model and the baselines never see it. It is a stricter bar, not a
  deployable imputer.
- **Known by construction, before any cell:**
  - Only the denominators differ from `impute_score`. On a table of one kind of column the two scores differ by a
    constant factor within a fold, so they keep the same epoch, and the order of the model and every baseline within a
    fold is the same on both scales. Seven of the nine tables are of one kind (`kr-vs-kp` categorical; `biodeg`, `kc2`,
    `pendigits`, `vehicle`, `letter`, `spambase` numerical).
  - On `credit-g` (7 numerical, 13 categorical columns) and `electricity` (7 and 1) the weight of each kind changes, so
    the kept epoch and the order can move.
  - H (ADR 0024) keeps the epoch of the lowest validation loss and never reads the decode metric, so there is nothing
    to optimise on H. The arm is the default K (KC at 40% missing and above, ADR 0027), whose checkpoint is the score's.
  - So what the study can show that is not known: TRIDENT's level against the class-aware fill (is it below 1, and by
    how much), and the two mixed tables.
- **Disclosure:**
  - The arm (default K), the variants, the seeds and the thresholds below are my choices; the user asked for the metric,
    the optimisation and the comparison.
  - K's `impute_score` against the bar and the strong tier on these variants at seeds 101/202/303 is known from the
    thesis table's studies (gorgona8 runs), and by construction the calls on one-kind tables can change only through the
    relative margin. They are secondary here.
  - Smoke, outside the study: `credit-g_20nan` at seed 999, both arms, MLflow off, and the tally on those two runs. Its
    numbers were seen.
  - TabPFN v2 (ADR 0028's offline member) is not in this study: its files for these seeds are on gorgona8, which is off
    the network. The logged fill errors let a later reading put it on this scale without retraining.

## Design

- **Arms:**
  - **Kc:** the imputation task's default run with `--decode_metric impute_score_classcond`.
  - **K:** the default run, on `credit-g` and `electricity` only (the tables where the kept epoch can differ). It does
    not log the new score; the reader computes it from K's fold errors and the fill errors Kc logged for the same cells.
- **Variants (26), `hybrid_fresh_seeds.MCAR`**, E174's: `biodeg_20/40/80nan`, `vehicle_20/40/80nan`,
  `credit-g_20/40/60/80nan`, `electricity_20/40/80nan`, `letter_20nan`, `spambase_20nan`, `kr-vs-kp_20/40/60/80nan`,
  `kc2_20/40/80nan`, `pendigits_20/40/60/80nan`.
- **Seeds:** 101/202/303, the thesis table's primary set (ADR 0020). 78 Kc runs and 21 K runs, 5 folds each.
- **Machine (ADR 0035):** everything on medusa1 (gorgona8 is off the network since 2026-11-15), eight queues, four a GPU,
  `c15_queue.sh` with `QUEUE_RUNNER=scripts/experiments/classcond_run.py`,
  `QUEUE_STATE=/scratch/diogoneiss/trident-2026-11-17-classcond`; MLflow tag `experiment = classcond-score-2026-11-17`.
- **Cost:** about 50 GPU-hours at gorgona8's speed by the runners' per-table minutes; about 8–12 h on medusa1.

## Measures and rule (fixed now)

Induced population, 15 (seed, fold) values a variant, the fold-pair rule with seed agreement (`report_common.verdict`).

- **Primary, Kc against the class-conditional fill:** Kc's `impute_score_classcond` − 1 per fold, margin 0.01.
  - **supported** (TRIDENT stays competitive) if Kc is better than the fill on 18 or more of the 26 variants and worse
    on at most 2;
  - **refuted** if worse on 6 or more;
  - **mixed** otherwise.
- **Secondary, no rule:**
  - Kc against the bar (the weak baseline with the lowest mean on that scale) and against the best strong-tier
    baseline, on `impute_score_classcond`, margin 1% of the opponent's mean; the same two on `impute_score` in the same
    runs, and the variants whose call differs between the scales;
  - Kc − K on both scales, on `credit-g` and `electricity`: what choosing the checkpoint by the new score changes;
  - the whole-fold mean/mode fill's score on the new scale: how much stricter the bar is, per variant.
- Three seeds, fixed; no extension.
- **Read** once with the 99 runs in:
  `QUEUE_STATE=/scratch/diogoneiss/trident-2026-11-17-classcond uv run --python 3.11 python -B scripts/experiments/classcond_run.py --tally`.
- **What it leads to:** a sentence for the thesis on whether the lead over the naive fill survives a class-aware one.
  No default changes. Whether the score joins the protocol (an ADR) is the user's call after the reading.

## Amendments

- **Amendment (2026-11-17 17:05 GMT-3):** medusa1 was restarted at about 15:38 GMT-3. Eight cells in flight were
  interrupted at 15:27 (`KeyboardInterrupt`, the shutdown) and the queues died with the machine; 3 of 99 runs were in
  (`letter_20nan`, the three seeds). The eight queues were relaunched at 17:02 and retrain the 96 others whole; no
  change to design, seeds or rule, nothing read.
- **Amendment (2026-11-17 20:00 GMT-3):** between 19:23 and 19:25 GMT-3 every training process of the study was
  killed (logs cut mid-epoch with no traceback; the machine did not restart; the cause was not found), then the
  eight queues; 8 of 99 runs were in. Relaunched at 19:57 as twelve queues (eight on GPU 1, four on GPU 0), with
  `--todo` dealing the shortest cells first; the other 91 retrain whole. No change to design, seeds or rule; nothing read.
- **Amendment (2026-11-17 20:50 GMT-3), partial look at the user's request, not the read:** with 20 of 99 cells holding
  fold metrics (18 Kc, 2 K; eight of them were killed at about 20:07 during the final MLflow logging, after writing
  the five folds, and count as in), Kc's fold-mean score on the class scale is below 1 in all 18 Kc cells (0.43 to 0.74).
  Fold means over the seeds in, impute_score / impute_score_classcond, Kc; best weak; best strong; TabPFN v2 (E175's
  cells at the same seeds, same scored cells, rescaled by the fold's constant; none local for `letter`, `electricity`):
  `letter_20nan` (3 seeds) 0.421/0.513; knn10 0.472/0.577; missforest 0.463/0.565.
  `pendigits_20nan` (2) 0.302/0.431; knn10 0.351/0.501; missforest 0.334/0.477; TabPFN 0.307/0.439.
  `vehicle_20nan` (2) 0.416/0.446; lgbm 0.466/0.499; missforest 0.449/0.481; TabPFN 0.420/0.450.
  `kc2_40nan` (3) 0.484/0.524; knn5 0.590/0.640; softimpute 0.550/0.597; TabPFN 0.488/0.529.
  `kc2_80nan` (3) 0.673/0.711; knn5 0.837/0.886; missforest 0.737/0.779; TabPFN 0.721/0.762.
  `kc2_20nan` (3) 0.544/0.586; knn5 0.522/0.557; softimpute 0.538/0.577; TabPFN 0.493/0.528.
  `electricity_20nan` (2) 0.618/0.634; hgb 0.602/0.618; missforest 0.619/0.635.
  No call is made from these, no Kc − K pair is in yet, and design, seeds and rule stand; the read stays the one with
  the 99 runs in. Ten more cells failed between 20:06 and 20:09 (eight killed without a traceback, one on a CUDA
  "device busy" error, one not inspected), cause not found; the two without fold metrics are retrained by a last pass of the queues.

## Outcome

(open)
