# Experimentation log

One entry per hypothesis-driven experiment on this project (Optuna studies, ablations, sweeps,
cross-machine checks), oldest first. Each entry summarises its ticket, ADR or review section and
links to it; that source stays the contract and holds the full numbers. Scores are
`impute_score` unless stated (lower is better, 1.0 = the mean/mode fill); intervals are 95%.

Entries E01–E23 were written on 2026-09-29 (GMT-3) from the existing records; from E24 on, each entry is
opened when the experiment is pre-registered and closed when it ends (see `CLAUDE.md`).

## Template

```
### Enn · <short title> (<start date>[ – <end date>])

- **Status:** in progress | done | stopped early
- **Hypothesis:** what was expected, or the question asked, and why
- **Scenarios:** arms, variants, seeds, budgets, what was held fixed
- **Measures and decision rule:** primary and secondary measures; the verdict rule, if pre-registered
- **Amendment <date>:** (only if the design changed mid-run) what changed and why
- **Results:** headline numbers with intervals
- **Conclusion:** what the source concluded, and what was decided or changed because of it
- **Sources:** repo paths with sections, commits, MLflow tags or run names
```

## Index

| ID | experiment | dates | status | headline |
|---|---|---|---|---|
| E01 | `[MASK]` vs `[NULL]` token path for induced cells | 2026-09-10; re-checked 2026-09-11 | done | `[MASK]` won 8 of 8 folds; credit-g rows scored the wrong cells, fixed and re-measured 2026-09-30: 8 of 8 again |
| E02 | Reduced Optuna search: six studies under cosine | 2026-09-11 | done | `LR_DECODE` leads the importances in 5 of 6; both kr-vs-kp winners tie-broken |
| E03 | Promoted configuration vs defaults, cosine, seed 42 | 2026-09-11 – 2026-09-12 | done | the reduced search did not beat the defaults; no interval separates |
| E04 | Does the pre-training loss drop mean learning? | critique probe, 2026-09-11 | done | the loss drop is norm-matching; the encoder is worse than a per-column constant |
| E05 | Pre-training epochs 300 vs 2 | critique ablation, 2026-09-11 | done | no detectable induced difference between 300 and 2 epochs |
| E06 | Decode checkpoint criterion vs ranking metric | critique probe, 2026-09-11 | done | the two criteria disagree (ρ 0.605) but cost ~0.005 on test |
| E07 | Plateau halvings driven by pre-training's re-rolled validation mask | critique read, 2026-09-11 | done | `plateau` froze spambase pre-training on mask noise |
| E08 | Resolution and fidelity of the masked search objective | critique reads, 2026-09-11 | done | masked objective coarse (ties on kr-vs-kp) and loosely tied to the induced headline |
| E09 | Plateau replication of the six reduced studies | 2026-09-12 | done | the learning-rate finding half-replicates under plateau |
| E10 | Tuned vs defaults under plateau, and plateau vs cosine | 2026-09-12 – 2026-09-13 | done | tuning helps no more under plateau; the objective is a poor proxy for the benchmark |
| E11 | Does the search objective rank trials? Twelve-study re-analysis | 2026-09-15 | done | the objective is "an effective filter and a useless ranker" |
| E12 | Seed noise floor of a five-fold imputation run | 2026-09-17 – 2026-09-18 | done | seed SD 0.007–0.014; every tuned-vs-default gap sits inside it |
| E13 | Full-profile vs reduced studies | 2026-09-17 – 2026-09-18 | done | `full` and `reduced` are disjoint; the comparison measures the epoch budget |
| E14 | Second-seed reduced studies and the cost of mis-selection | 2026-09-17 – 2026-09-18 | done | selection quality is a lottery; median cost 0.036, an upper bound |
| E15 | Re-selecting trials on other validation statistics | 2026-09-17 – 2026-09-18 | done | no validation-only statistic selects better |
| E16 | Replicating the one separation: credit-g_40nan under plateau | 2026-09-17 – 2026-09-18 | done | the one plateau separation did not replicate |
| E17 | Committed promoted files vs defaults across seeds | 2026-09-17 – 2026-09-18 | done | credit-g_20nan's promoted file is worse than the defaults (paired p ≈ 0.008) |
| E18 | Control: the best-induced trial scored five-fold | 2026-09-18 | done | about four fifths of the mis-selection gap was selection bias |
| E19 | Decoder vs baseline imputers on the decoder's own cells | 2026-09-24 – 2026-09-28 | done | below the best baseline on 6 of 21 induced variants, never clearly |
| E20 | Sensitivity of the KNN bar to k | 2026-09-24/25 – 2026-09-27 | done | k = 5 is the weaker KNN bar at high missingness |
| E21 | Overnight full-profile Optuna studies against the baseline bar | 2026-09-28 | done | the tuned model beats the bar nowhere on induced cells |
| E22 | Induced validation objective on credit-g_80nan | 2026-09-28 | done | induced objective beats the masked winner on 5 of 5 folds; level with the defaults |
| E23 | Induced validation objective on the other six variants | 2026-09-28 – 2026-09-29 | stopped early | better on 2 of 6, level on 2, worse on 2; never beats the best baseline |
| E24 | What pre-training contributes to imputation | 2026-09-29 – 2026-09-30 | done | never helps vs none; hurts on pendigits, where C beats the best baseline; A better than C on kr-vs-kp |
| E25 | Value and normalised-embedding pre-training objectives | 2026-09-30 | done | N never worse than A, better on 2/4 and better than no pre-training on 3/4; V helps only on pendigits |
| E26 | Normalised-target pre-training with a long decode stage | 2026-09-30 | done | never worse than C or N; beats C on kr-vs-kp, matches C on pendigits (both clear knn10) |
| E27 | Stopping the decode stage on a validation plateau | 2026-09-30 | done | same scores with 61–107 of 450 epochs on three tables (kr-vs-kp 18.5 → 4.9 min); pendigits loses 0.006 |
| E28 | Batch 1024 with a scaled learning rate | 2026-09-30 | done | helps only pendigits (x4: −0.011, best yet); hurts credit-g_20nan (x4) and kr-vs-kp (x2) |
| E29 | Batched decoder heads against per-column | 2026-09-30 – 2026-10-01 | done | no detectable difference on all four (pendigits +0.0008, lower bound at zero) |
| E30 | The candidate configuration on all 21 imputation variants | 2026-10-01 | done | M better than A on 9 of 21, worse on none; P loses on 2 (kr-vs-kp_40nan, spambase_40nan); M recommended, 1.65x A's time |
| E31 | The induced gaps asked one column at a time, the rest as `[NULL]` | 2026-10-01 – 2026-10-02 | done | supported: column-wise better on 5 of the 9 variants at ≥60nan, mask on none; the effect grows with the missing level (+0.004 at 20nan, −0.022 at 60nan); mask wins on 6 light variants by ≤0.013 |
| E32 | Training the decode stage with the gaps shown as `[MASK]` | 2026-10-02 | done | supported: G better than M on 11 of 21 (5 of the 9 heavy), worse on none; gains at every missing level; beats column-wise scoring on 12; G qualifies as the default |
| E33 | The gaps-as-mask default on fresh seeds | 2026-10-03 | done | replicated: G better on 13 of 21 (6 of 9 heavy), worse on none, on seeds 101/202/303; larger gains than E32 at every level; ADR 0014 stands |
| E34 | Checkpoint by the validation gaps, and calibration on them | 2026-10-03 – | in progress | pending |

## Entries

### E01 · `[MASK]` vs `[NULL]` token path for induced cells (2026-09-10; re-checked 2026-09-11)

- **Status:** done; the original verdict was later found to rest on one dataset.
- **Hypothesis:** ADR 0004 decision 5 scores induced-missing cells by showing the model
  `[MASK]`. Decision 11 predicted that the `[NULL]` path would not transfer: the head is never
  trained at a null position.
- **Scenarios:** `--score_null_path` diagnostic on `credit-g_20nan`, `credit-g_60nan`,
  `kr-vs-kp_20nan`, `kr-vs-kp_60nan`, two folds each.
- **Measures and decision rule (pre-registered):** `impute_score`; decision 5 is overturned
  only if the null path beats `[MASK]` across a majority of folds on at least two datasets at
  both the 20% and 60% variants.
- **Results:** `[MASK]` won on all eight folds (credit-g_20nan 1.008, 1.004 vs 1.155, 1.180;
  credit-g_60nan 1.022, 1.121 vs 1.227, 1.422; kr-vs-kp_20nan 0.984, 0.969 vs 1.920, 1.533;
  kr-vs-kp_60nan 1.012, 0.980 vs 1.986, 1.640). **Re-check (critique D-1):** on credit-g,
  which interleaves column kinds, the null path scored the wrong cells: only 944 of 4000
  (23.6%) of the cells it scored are actually missing (verifier: 478 of 1976, 24.2%), because
  `_select` passes the mask in CSV column order and the decoder reads categoricals first.
  kr-vs-kp is all-categorical, so its rows are aligned.
- **Conclusion:** ADR 0004: "the criterion is not met and decision 5 stands". The critique
  (CONFIRMED, high, by three verifiers): the credit-g numbers "are not biased, they are
  meaningless", so the gate's "at least two datasets" rests on kr-vs-kp alone. Proposed:
  build the selection in the embedder's column order and re-measure or strike the credit-g
  rows. Fixed 2026-09-30 (`918bcd6`) and re-measured on the same four variants (two folds,
  seed 42, defaults, cosine): `[MASK]` wins on all 8 folds again (credit-g_20nan 0.924, 0.902
  against 0.985, 1.023), so decision 5 stands on valid credit-g rows (ADR 0004 decision 11).
- **Sources:** `docs/adr/0004-imputation-decoder-task.md` decision 11 (verdict `74229fd`);
  `docs/reviews/imputation-critique/CONSOLIDATED.md` D-1; `04-ground-truth-protocol`,
  `05-embedder-tokens`, `09-test-adequacy` (`.md` and `.verdict.md`), F-04-1, F-05-1, F-09-1.

### E02 · Reduced Optuna search: six studies under cosine (2026-09-11)

- **Status:** done.
- **Hypothesis:** a knob earns a search dimension only if it governs the decode stage or the
  corruption the decoder learns from (ADR 0005 decision 2); fANOVA importances check that
  reduction (decision 6); a full-profile pilot returns "if the reduced studies' importances
  look flat".
- **Scenarios:** credit-g, kr-vs-kp, spambase at `_20nan` and `_40nan`, one study each.
  `--search_space reduced`: `PROB_MASCARA` 0.2..0.6, `LR_DECODE` 1e-4..1e-2 (log),
  `WEIGHT_DECAY_DECODE` 1e-5..1e-2 (log), `DROPOUT` 0.1..0.5, `LAMBDA_NUM` 0.1..10 (log, credit-g
  only); architecture, `BATCH` 256, `EPOCHS_PRE` 300, `LR_PRE` 0.00034, `EPOCHS_DECODE` 150 held.
  40 trials, cosine, seed 42, TPE seeded with the run seed, predefined single split, no pruner,
  `--promote_best`. Launcher `imputation_studies.ps1`.
- **Measures:** `validation/impute/masked/impute_score` (not comparable with any `test/`
  number); `optuna/importance/<knob>`; wall clock against the ADR estimate. No rule set.
- **Results:** 40 of 40 trials everywhere; 20.5 h in total (estimate ~19 h).

  | pair | best objective (trial) | winner `LR_DECODE` | leading importances |
  |---|---|---|---|
  | credit-g_20nan | 0.7763 (30) | 2.54e-3 | LR_DECODE 0.49, LAMBDA_NUM 0.33 |
  | credit-g_40nan | 0.8051 (2) | 8.71e-3 | LR_DECODE 0.52, WD 0.22 |
  | kr-vs-kp_20nan | 0.6703 (17) | 1.74e-3 | LR_DECODE 0.39, WD 0.30 |
  | kr-vs-kp_40nan | 0.6480 (32) | 1.70e-3 | LR_DECODE 0.49, DROPOUT 0.22 |
  | spambase_20nan | 0.8993 (16) | 1.33e-3 | DROPOUT 0.53, WD 0.25 |
  | spambase_40nan | 0.8996 (28) | 1.08e-3 | LR_DECODE 0.80, DROPOUT 0.10 |

- **Conclusion:** the decode learning rate leads in five of six (0.39–0.80); the worst trials
  sit at its low end, the objective is flat from ~1e-3 to 1e-2, winners 2–9x above the default;
  the mask rate barely matters; `LAMBDA_NUM` points both ways. Promoted files committed.
  Qualified later by the plateau replication (addendum C-1) and by addendum § 2: both
  kr-vs-kp winners were picked by a tie-break.
- **Sources:** `docs/tickets/imputation-optuna-reduced/issues/05-launcher-and-studies.md`
  § Comments; ADR 0005 decisions 2, 4, 6. Commits `d039c47` (launcher), `fcaadfb` (promoted
  files). Tags `is_optuna=true`, `search_space=reduced`, `lr_scheduler=cosine`.

### E03 · Promoted configuration vs defaults, cosine, seed 42 (2026-09-11 – 2026-09-12)

- **Status:** done.
- **Hypothesis:** each study's promoted configuration beats the defaults on the induced
  benchmark ("proving the tuning helped", ADR 0005 decision 6).
- **Scenarios:** six pairs, two arms: promoted file in place vs moved aside (defaults); both
  `--task imputation --cv_folds 5 --lr_scheduler cosine --seed 42`
  (`imputation_studies.ps1 -Compare`), the configuration the only difference.
- **Measures:** `cv/test/impute/induced/impute_score/mean` with its 95% interval (masked as
  companion); per pair, whether the promoted interval excludes the default mean. No adoption
  threshold stated.
- **Results:**

  | pair | promoted, induced | defaults, induced | masked, promoted / defaults |
  |---|---|---|---|
  | credit-g_20nan | 0.913 [0.868, 0.958] | 0.895 [0.868, 0.921] | 0.926 / 0.896 |
  | credit-g_40nan | 0.968 [0.940, 0.995] | 0.962 [0.941, 0.983] | 0.965 / 0.974 |
  | kr-vs-kp_20nan | 0.603 [0.559, 0.646] | 0.615 [0.584, 0.646] | 0.660 / 0.673 |
  | kr-vs-kp_40nan | 0.778 [0.734, 0.822] | 0.756 [0.721, 0.791] | 0.802 / 0.775 |
  | spambase_20nan | 0.886 [0.859, 0.913] | 0.887 [0.859, 0.914] | 0.879 / 0.882 |
  | spambase_40nan | 0.930 [0.909, 0.952] | 0.927 [0.897, 0.957] | 0.931 / 0.921 |

  No promoted interval excludes the default mean. kr-vs-kp_40nan first crashed in fold 4
  (`KeyError: 't'`, fixed in `1cb0864`, both runs repeated; two `FAILED` parents remain).
- **Conclusion:** "The reduced search did not beat the defaults on the induced benchmark":
  worse on four pairs, better by at most 0.012 on two. Reasons given: the default 1e-3 already
  sat in the flat region; on credit-g the validation objective chose configurations that
  generalise worse, "which is what a thousand-row validation split affords". One seed. Whether
  runs keep reading the promoted files "is a separate decision" (see the multi-seed entry
  below). Addendum § 3 re-derived the table unchanged and notes that two overlapping
  single-arm intervals are not a test of a difference (F-08-4).
- **Sources:** ADR 0005 § Status › Outcome;
  `docs/tickets/imputation-optuna-reduced/issues/06-comparison-and-docs.md` § Comments;
  addendum § 3. Commits `1cb0864`, `4cec3bd`, `ecec5e2`. `params.config_source` tells the
  arms apart.
### E04 · Does the pre-training loss drop mean learning? (critique probe, 2026-09-11)

- **Status:** done (critique measurement, not pre-registered).
- **Hypothesis:** the 95–99.7% pre-training loss drop in the store is the encoder's output norm
  shrinking onto its target, not learning to predict hidden cells: the loss is an
  un-normalised MSE against a target that trainable parameters produce, with no predictor
  head, EMA target or variance term.
- **Scenarios:** `credit-g_20nan`, full 300-epoch `train_pretrainer` at defaults on CPU, scored
  on the hidden cells of one fixed validation mask: (a) the critic's probe; (b) the same at
  `BATCH=64` (3300 steps instead of 900); (c) the verifier's independent probe (fold 0 of
  2, `cosine_legacy`, 623 cells).
- **Measures:** NMSE = model error / error of a per-column mean target embedding (1.0 = parity
  with a constant per column); centred cosine; output and target norms.
- **Results:** (a) loss −98.5%, NMSE 187.61 → 4.21, centred cosine −0.0109 → 0.0001, output
  norm 14.530 → 3.204 against target 2.622. (b) NMSE 1.51. (c) loss −97.4%, NMSE 242.7 → 7.70,
  centred cosine +0.0068 (1.9 SE); the trained model beats the zero vector by only 19%.
- **Conclusion:** F-13-1, verdict SOUND, severity cut from critical to high: the encoder learns
  the column-constant part of the target, not the part that identifies the value; the loss
  cannot show learning. Not shown: that the objective cannot work (the `BATCH=64` run moves
  the right way). One dataset, one fold. The verifier found the embedder also shrinks its own
  discriminative variance, which a value-space anchor would stop and normalising the loss would
  not. With the next entry, it led to the pre-training ablation.
- **Sources:** `docs/reviews/imputation-critique/13-pretraining-transfer.md` and `.verdict.md`
  F-13-1, F-13-3; `CONSOLIDATED.md` § 3.6. Probe scripts lived outside the repo.

### E05 · Pre-training epochs 300 vs 2 (critique ablation, 2026-09-11)

- **Status:** done (critique measurement, not pre-registered; not re-run by the verifier).
- **Hypothesis:** if pre-training teaches nothing about values (F-13-1), cutting `EPOCHS_PRE`
  from 300 to 2 at a fixed decode budget leaves the induced score inside noise. ADR 0004/0005
  held 300 epochs as "shared and protected" with no measurement behind it.
- **Scenarios:** `EPOCHS_PRE` 300 vs 2, `EPOCHS_DECODE` 150, defaults otherwise;
  `credit-g_20nan`, `cv_folds=2`, cosine, seeds 42 and 43, CPU, MLflow off. Confound: the arms
  consume the RNG streams differently, so the decode masks differ between arms.
- **Measures:** `impute/induced/impute_score`; no rule stated; the arm gap is read against the
  fold and seed spread.
- **Results:** induced 0.9218 / 0.9095 (seed 42, 300 / 2) and 0.9173 / 0.9311 (seed 43); arm
  means 0.9196 vs 0.9203, the sign flipping between seeds. Masked: 300 epochs ahead on both
  seeds (by 0.0096 and 0.0183). Noise: folds of one run differ by 0.042 and 0.047. Time 305.8 /
  71.4 s and 149.7 / 80.8 s; pre-training takes a median 0.514 of stage time over 185 trials.
- **Conclusion:** F-13-2, verdict SOUND, high: no detectable benefit from 298 of 300 epochs on
  induced cells, "suggestive, not conclusive"; the design cannot separate "no benefit" from
  "a benefit smaller than this design can see". The masked-only gap fits pre-training
  teaching familiarity with `[MASK]` rather than values. Counterweight: a confounded
  classification contrast on `vehicle_00nan` points the other way. Led to the pre-registered
  ablation (`7af595b`).
- **Sources:** `13-pretraining-transfer.md` and `.verdict.md` F-13-2, F-13-4;
  `CONSOLIDATED.md` § 3.6.

### E06 · Decode checkpoint criterion vs ranking metric (critique probe, 2026-09-11)

- **Status:** done (one fold, not replicated).
- **Hypothesis:** the decode stage keeps the epoch with the lowest λ-weighted validation loss,
  but folds are ranked on `impute_score`, so the two pick different epochs and `LAMBDA_NUM`
  also decides which epoch survives.
- **Scenarios:** `credit-g_20nan`, fold 0 of 2, promoted configuration, 40 pre-training + 80
  decode epochs, both criteria scored every epoch.
- **Results:** loss picks epoch 32 (test 0.9604), validation `impute_score` picks epoch 50
  (test 0.9658); best achievable 0.9088 at epoch 38; rank correlation of the criteria 0.605.
- **Conclusion:** F-02-1, verdict SOUND, cut to low: structural, not a demonstrated loss (the
  criteria land 0.005 apart on test, well below epoch-to-epoch noise). Measure over folds and
  seeds before switching. Handoff item 6 (2026-09-29) still lists picking the checkpoint by
  validation `impute_score`.
- **Sources:** `02-decoder-model.md` and `.verdict.md` F-02-1; `CONSOLIDATED.md` § 3.4.

### E07 · Plateau halvings driven by pre-training's re-rolled validation mask (critique read, 2026-09-11)

- **Status:** done (read of logged runs).
- **Hypothesis:** pre-training re-rolls its validation mask every epoch and feeds that loss to
  `ReduceLROnPlateau` (factor 0.5, patience 10), so mask noise, not progress, halves the rate.
- **Scenarios:** `pretrain/learning_rate` and `pretrain/val_loss` per fold of the four
  imputation parents tagged `plateau`.
- **Results:** credit-g_20nan: no halvings. spambase_20nan: 11 halvings on one fold (3.40e-04 →
  1.66e-07, every 11 epochs from epoch 238), with `val_loss` still spanning 1.39x (CV ≈ 9%)
  while frozen; 7 halvings on another.
- **Conclusion:** CONFIRMED, medium (D-4): a `plateau` pre-training run is not the configured
  run on some datasets. Capped: `plateau` is not the default, and ADR 0005 refuses those runs
  as a baseline. Proposed: draw the validation mask once per fold, behind a flag. No fix
  recorded.
- **Sources:** `13-pretraining-transfer.md` and `.verdict.md` F-13-5; `CONSOLIDATED.md` D-4.
### E08 · Resolution and fidelity of the masked search objective (critique reads, 2026-09-11)

- **Status:** done (reads of the reduced studies; revised by the 2026-09-15 re-analysis below).
- **Hypothesis:** `validation/impute/masked/impute_score`, scored on the masked cells of a 10%
  validation split, is too coarse to separate configurations (F-07-1), may not track the
  induced headline (F-07-7, F-08-5), and reads optimistic because the same cells chose the
  checkpoint (F-07-2).
- **Scenarios:** the 2026-09-11 reduced cosine studies (four finished, spambase_20nan in
  flight), study databases and `mlflow.db` read-only; 180 trial runs.
- **Results:** scored validation cells 283, 179, 1511, 910, 3390 (credit-g_20/40, kr-vs-kp_20/40,
  spambase_20); both kr-vs-kp studies have only 20 distinct values in 40 trials, best == 2nd
  bit-for-bit, and the promoted files hold tie-break winners (trials 17 and 32); the tie cost
  on kr-vs-kp_40nan was 0.028 on induced test cells. Spearman of the objective vs test induced:
  +0.124, +0.568, +0.340, +0.372, +0.812; winners ranked 31/40, 21/40, 1/40, 33/40 on induced.
  Validation reads better than test masked by 0.012 to 0.064 on average (25/40 to 38/40
  trials).
- **Conclusion:** F-07-1 SOUND, critical ("if anything the finding understates itself";
  direction, not a proven loss); F-07-2 SOUND, high (suggestive; the test split stays clean);
  F-07-7 and F-08-5 corrected to medium and low. Proposed: score the induced population on
  the validation rows (1.5x–5x more cells), later offered as `--search_objective induced`
  (ADR 0008).
- **Sources:** `docs/reviews/imputation-critique/07-optuna-imputation.md` and `.verdict.md`
  F-07-1, F-07-2, F-07-7; `08-experimentation-design.verdict.md` F-08-5; `CONSOLIDATED.md`
  § 1, § 3.3; `AUDIT.md` change 1.

### E09 · Plateau replication of the six reduced studies (2026-09-12)

- **Status:** done (exploration outside ADR 0005's plan, at the user's request).
- **Hypothesis:** the question whether the cosine studies' learning-rate finding depends on
  the schedule.
- **Scenarios:** the same six studies (reduced, 40 trials, seed 42) under `plateau`,
  `imputation_studies.ps1 -Scheduler plateau -NoPromote`; the cosine winners stay promoted.
- **Measures:** objective against the cosine twin, winner `LR_DECODE`, fANOVA top knob. No rule.
- **Results:** objective cosine / plateau: credit-g_20nan 0.7763 / 0.7503; credit-g_40nan
  0.8051 / 0.7820; kr-vs-kp_20nan 0.6703 / 0.6374; kr-vs-kp_40nan 0.6480 / 0.5866;
  spambase_20nan 0.8993 / 0.8995; spambase_40nan 0.8996 / 0.8999. Plateau winners' `LR_DECODE`
  scatter from 3.24e-4 to 8.26e-3; the rate loses the top spot on three pairs.
- **Conclusion:** ticket 05: "the cosine finding does **not** replicate". Addendum § 5 C-1
  corrects it to half: Spearman(`LR_DECODE`, objective) is negative in 12 of 12 studies and
  `LR_DECODE` has the highest mean importance under both schedules (0.483 cosine, 0.421
  plateau), but it leads on only 3 of 6 plateau pairs. Safe statement: a rate finding from a
  single schedule must not be reported as a property of the task.
- **Sources:** ticket 05 § Comments (2026-09-12 22:52); ADR 0005 § Outcome "Replicated under
  another schedule"; addendum § 5 C-1. Commits `d69e344`, `6e1f763`, `a86610e`.

### E10 · Tuned vs defaults under plateau, and plateau vs cosine (2026-09-12 – 2026-09-13)

- **Status:** done.
- **Hypothesis:** the plateau studies' larger objective gains might carry over to the induced
  benchmark where the cosine ones did not; does tuning help under plateau, and does plateau
  beat cosine?
- **Scenarios:** twelve five-fold runs, ADR 0005's protocol with `--lr_scheduler plateau`,
  seed 42; the plateau winner staged into the task-keyed path per pair and the committed cosine
  files restored after (byte-identical at the end). Plateau vs cosine at the defaults reuses
  the cosine defaults runs.
- **Measures:** as in the cosine comparison; "separates" when the tuned interval excludes the
  default mean.
- **Results (induced; plateau tuned / plateau defaults / cosine defaults):** credit-g_20nan
  0.907 / 0.901 / 0.895; credit-g_40nan **0.928 [0.915, 0.942]** / 0.949 / 0.962;
  kr-vs-kp_20nan 0.619 / 0.612 / 0.615; kr-vs-kp_40nan 0.755 / 0.753 / 0.756; spambase_20nan
  0.890 / 0.883 / 0.887; spambase_40nan 0.924 / 0.930 / 0.927. Tuning better on two, worse on
  three, tied on one; only credit-g_40nan separates. At the defaults the schedules land within
  0.013 everywhere; no pair separates.
- **Conclusion:** under plateau tuning helps "no more than under cosine"; one separation in
  twelve "is about what chance alone produces" (a later replication confirmed it did not hold).
  Plateau beat cosine on the objective but not on the benchmark, so "the search objective
  (masked cells, validation split) is a poor proxy for the induced benchmark": evidence against
  ADR 0005 decision 3 itself. Recorded as fog, not actioned; ADR 0008 later made the objective
  selectable (default unchanged).
- **Sources:** ticket 05 § Comments (2026-09-13 03:19); ADR 0005 § Outcome; addendum § 3,
  § 4. Commit `20bd371`.

### E11 · Does the search objective rank trials? Twelve-study re-analysis (2026-09-15)

- **Status:** done (no new training).
- **Hypothesis:** tests the argument behind F-07-1, that the search "cannot resolve the
  configurations it is ranking", argued from one ρ = +0.124 and two tie-broken winners.
- **Scenarios:** the twelve stored studies (six pairs × cosine, plateau; 40 trials), read-only;
  Spearman ρ between each trial's objective and its `test/impute/induced/impute_score`, and the
  promoted winner's induced rank. Three adversarial verifiers.
- **Measures:** single-study 95% noise floor 0.314 at n = 40; Fisher-z mean ρ per schedule; mean
  winner rank against 20.5 by chance (SD of a six-rank mean 4.71).
- **Results:** mean ρ 0.442 (cosine), 0.524 (plateau); ten of twelve clear 0.314 (six of twelve
  after Bonferroni). Mean winner rank 17.3 and 15.2 (a bivariate-normal model predicts 10.2 and
  8.4). Among the twenty best-by-objective trials, ρ is −0.029 and +0.076. In 11 of 12 studies
  the winner is not the best trial on the induced metric.
- **Conclusion:** "The objective is not noise", but selection is "not distinguishable from
  chance selection": the correlation "lives entirely in the bad tail", "an effective filter and
  a useless ranker, and promotion uses it as a ranker". F-07-1 stands at critical with its
  argument replaced; do not pool across studies.
- **Sources:** `docs/reviews/imputation-critique/ADDENDUM-2026-09-15-optuna-results.md` § 2–4,
  § 7; `CONSOLIDATED.md` F-07-1. Commit `bc369be`.

### E12 · Seed noise floor of a five-fold imputation run (2026-09-17 – 2026-09-18)

- **Status:** done.
- **Hypothesis:** the effort had no measured noise floor; the arm differences being read (0.001
  to 0.022) are smaller than the interval half-widths, and same-seed repeats are bit-identical,
  so the missing measurement is a second seed at a fixed configuration (addendum N-1).
- **Scenarios:** the committed cosine promoted configuration, five folds, seeds 7, 13, 42, 99,
  420, on credit-g_20nan and credit-g_40nan (the seed changes the split and every mask).
- **Measures:** SD and range of the five induced means; the seed-42 tuned-vs-defaults gap in SDs
  against t(4) = 2.78.
- **Results:** seed 42 reproduces the stored values exactly (0.912797, 0.967689). credit-g_20nan
  SD 0.0066 (range 0.0159), gap 0.0181 = 2.8 SD, defaults better; credit-g_40nan SD 0.0138
  (range 0.0377), gap 0.0056 = 0.4 SD. A single run carries about ±0.018 to ±0.038 under t(4);
  every tuned-vs-default gap across the sixteen five-fold comparisons (0.0005 to 0.0283) lies
  inside it.
- **Conclusion:** on credit-g_40nan the arms differ inside seed noise; on credit-g_20nan the gap
  is "suggestive, and pointing against the promoted configuration" (tested properly below). An
  earlier two-seed draft ("67x the noise") is withdrawn as an n = 2 artifact.
- **Sources:** addendum § 5 N-1, § 8.1, § 8.2, § 8.6. Commit `bc369be`.

### E13 · Full-profile vs reduced studies (2026-09-17 – 2026-09-18)

- **Status:** done.
- **Hypothesis:** "What does the reduction cost?" (addendum § 8.4); the kr-vs-kp_20nan study
  also bears on F-13-1.
- **Scenarios:** `--search_space full`, seed 42, cosine, 40 trials, on credit-g_20nan,
  credit-g_40nan, kr-vs-kp_20nan; `full` samples both epoch counts in 20–60 where `reduced`
  holds 300 and 150.
- **Measures:** best objective against the reduced study's; the winner's induced test score;
  winner rank and mis-selection cost.
- **Results:** credit-g_20nan: objective 0.818 (full) vs 0.776 (reduced), 14 vs 43 minutes.
  kr-vs-kp_20nan: the full winner (20 pre-training, 30 decode epochs, batch 64) beats the
  reduced winner on the objective (0.6484 vs 0.6703) but is 0.074 worse on induced test cells
  (0.6787 vs 0.6043; seed SD 0.004); none of its 40 trials reaches the reduced study's best
  induced score. Winner ranks on credit-g 4/40 and 1/40.
- **Conclusion:** `full` and `reduced` are "disjoint profiles, not nested ones": their epoch
  ranges lie far below the fixed values, so comparing best objectives measures the epoch budget,
  not the search space; the "reduced wins" reading is withdrawn. kr-vs-kp_20nan is "a caution
  rather than support for F-13-1". Open: a `full` profile whose epoch ranges include the
  defaults.
- **Sources:** addendum § 6, § 8.3–8.7. Commit `bc369be`.

### E14 · Second-seed reduced studies and the cost of mis-selection (2026-09-17 – 2026-09-18)

- **Status:** done.
- **Hypothesis:** is selection quality a property of the pair, does ρ predict it, and how much
  induced score does the objective's pick leave on the table?
- **Scenarios:** three more reduced cosine studies at seed 420 (credit-g_20nan, credit-g_40nan,
  kr-vs-kp_20nan), analysed with the twelve seed-42 and three `full` studies (18 in all). Cost =
  induced score of the pick minus the best induced score among the study's own 40 trials.
- **Results:** median cost 0.0357 (0.0006–0.0535) over the 13 untied reduced studies; the two
  tied cosine kr-vs-kp studies cost 0.0000–0.0552 and 0.0471–0.0749 depending on the tie-break;
  promoted ranks median 18/40 (chance 20.5); Spearman(ρ, winner rank) over 18 studies −0.23
  (p = 0.36).
- **Conclusion:** selection quality is "a lottery rather than a property of the pair"; ρ is
  dropped as the statistic to report in favour of the winner's rank. "Configuration choice
  matters here; the selector is what does not work." The 0.0357 is "an upper bound on
  recoverable regret, not a measured gain" (tested by the control below).
- **Sources:** addendum § 8.3, § 8.5, § 8.7. Commit `bc369be`.

### E15 · Re-selecting trials on other validation statistics (2026-09-17 – 2026-09-18)

- **Status:** done (no new training).
- **Hypothesis:** selecting on a different validation statistic would recover the ~0.036 lost
  to mis-selection.
- **Scenarios:** existing trials re-ranked on validation-only statistics: `impute_score`
  (current), `rmse_num_z`, `macro_f1_cat`, rank-average(`rmse`, `-f1`), `mae_num_z`.
- **Results:** median cost 0.0386 (current), 0.0369, 0.0342, 0.0363, 0.0310; medians within
  0.008 and means within 0.010 of each other; single-kind studies split both ways.
- **Conclusion:** "No validation-only statistic wins consistently across datasets"; "swapping
  the statistic does nothing". Next step named: "a validation protocol, not a selector"
  (induced cells on the validation split, or more folds).
- **Sources:** addendum § 8.7, § 8.8. Commit `bc369be`.

### E16 · Replicating the one separation: credit-g_40nan under plateau (2026-09-17 – 2026-09-18)

- **Status:** done.
- **Hypothesis:** the plateau comparison's only separating cell "should be replicated on another
  seed before being believed".
- **Scenarios:** credit-g_40nan, plateau, five folds, seeds 420 and 7 added to 42; tuned
  staged, defaults with the file moved aside, both reversed by `git checkout`.
- **Results:** tuned − defaults: −0.0207 (seed 42), −0.0097 (420), +0.0005 (7); mean −0.0100,
  SD 0.0106, paired t = −1.63 on 2 df (p 0.24).
- **Conclusion:** the separation was "a favourable draw"; the only positive tuning result is
  "possibly a small effect, not established".
- **Sources:** addendum § 8.10. Commit `bc369be`.

### E17 · Committed promoted files vs defaults across seeds (2026-09-17 – 2026-09-18)

- **Status:** done.
- **Hypothesis:** with both arms at several seeds, is the promoted file better or worse than the
  defaults a `--task imputation` run would otherwise use?
- **Scenarios:** cosine, five folds, file in place vs moved aside. credit-g_20nan: tuned at five
  seeds, defaults at 7, 42, 420. credit-g_40nan: tuned at five, defaults at 7, 13, 42, 420 (the
  fourth added after the pre-set stopping point). kr-vs-kp_20nan: tuned at 7, 42, 420, defaults
  at 42, 420.
- **Measures:** Welch t per pair; paired-by-seed t as the stronger check.
- **Results:** credit-g_20nan tuned 0.9082 vs defaults 0.8956, +0.0126, Welch t = 3.64, paired
  +0.0158 (t = 11.4, 2 df, p ≈ 0.008). credit-g_40nan +0.0180 (Welch p ≈ 0.045), paired +0.0231
  (p ≈ 0.033), but p ≈ 0.058 at the stopping seed. kr-vs-kp_20nan −0.0077 (p ≈ 0.16 paired).
  All seven credit-g comparisons at shared seeds favour the defaults.
- **Conclusion:** credit-g_20nan: "the evidence points against" the promoted file;
  credit-g_40nan stays at "hold" (the extension past the stopping seed does not count);
  kr-vs-kp_20nan "sign-consistent and **not significant**". Under cosine the sign follows where
  the selector's pick landed, which is a lottery. Recommended (reversible): stop reading
  `datasets/hiperparams/credit-g/credit-g_20nan.imputation.json`. Done 2026-09-30 at the
  user's request (ADR 0005 § Outcome); every run of E24–E26 on credit-g_20nan still read it.
- **Sources:** addendum § 8.11–8.13. Commit `05c6595`.

### E18 · Control: the best-induced trial scored five-fold (2026-09-18)

- **Status:** done.
- **Hypothesis:** if the ~0.036 mis-selection loss were recoverable, the trial a perfect
  selector would promote should beat the promoted file and the defaults under the comparison's
  own protocol.
- **Scenarios:** each credit-g cosine study's best-induced trial (`52176d70`, `df2b6776`) staged
  and run five-fold at seeds 42 and 420; kr-vs-kp_20nan's best-induced trial is its promoted one.
- **Results:** best − promoted −0.0129, −0.0184 (credit-g_20nan), −0.0060, −0.0024
  (credit-g_40nan), paired t = −2.78, 3 df, p ≈ 0.07; on average 20.5% of the single-split gap
  survives. Best − defaults mean +0.006 (p ≈ 0.41, 95% interval −0.015 to +0.028).
- **Conclusion:** about four fifths of "the 0.036 left on the table" was the selection bias of
  taking a minimum; no evidence the best-induced trial beats the defaults ("absence of evidence,
  not evidence of absence"). The `git rm` of the credit-g_20nan file "stands and is the whole of
  the established actionable content for this dataset".
- **Sources:** addendum § 8.14. Commit `05c6595`. Staged runs carry the promoted file's
  `config_source`; only `LR_DECODE` tells them apart.
### E19 · Decoder vs baseline imputers on the decoder's own cells (2026-09-24 – 2026-09-28)

- **Status:** done (not pre-registered).
- **Hypothesis:** no direction stated. `impute_score` only compares the decoder with a
  mean/mode fill; the question is how low that bar is per variant and "whether a plain tabular
  imputer would have cleared it by more", which "is what a reader of the thesis will ask".
- **Scenarios:** the decoder against `mean_mode`, `knn5`, `knn10` (`KNNImputer`, uniform) and
  `hgb` (one `HistGradientBoosting` per column, `random_state = 0`), every baseline scored on
  the model's own folds and cells with the same naive denominators. 21 variants: credit-g,
  kr-vs-kp, spambase at 20/40/60/80nan; vehicle, biodeg, kc2 at 20/60nan; pendigits, letter,
  electricity at 20nan. Five folds, seed 42, cosine; promoted configuration for credit-g,
  kr-vs-kp, spambase at 20/40nan, defaults elsewhere. The 2026-09-24/25 runs logged `knn`
  (k = 5) live; `knn10` and `hgb` were added 2026-09-27 and backfilled 2026-09-28.
- **Measures and decision rule:** `cv/test/impute/induced/impute_score/mean` (lower is better;
  1.0 = mean/mode parity) against the **baseline bar**, the lowest of the four baselines per
  run and population, with the summary's 95% intervals. The model "does something no baseline
  does only where its number is below it"; disjoint intervals reported separately.
- **Results (induced, after the backfill):** the model is below the best baseline on **6 of 21**
  (`credit-g_20nan`, `kc2_20nan`, `kr-vs-kp_20nan`, `spambase_40nan`, `spambase_60nan`,
  `vehicle_60nan`), never with disjoint intervals; a baseline is below it with disjoint
  intervals on 3. `hgb` sets the bar on 13 of 21; at 80% missingness and on `credit-g_60nan`
  nothing beats mean/mode; KNN holds the bar on kc2, pendigits, letter.

  | variant | model | best baseline |
  |---|---|---|
  | `credit-g_20nan` | 0.913 [0.868, 0.958] | `hgb` 0.936 [0.912, 0.959] |
  | `credit-g_80nan` | 1.024 [1.008, 1.040] | `mean_mode` 1.000 [1.000, 1.000], disjoint |
  | `pendigits_20nan` | 0.399 [0.384, 0.414] | `knn10` 0.348 [0.339, 0.357], disjoint |
  | `letter_20nan` | 0.515 [0.507, 0.522] | `knn10` 0.473 [0.468, 0.478], disjoint |

  Superseded first reading (2026-09-27, KNN-only bar): below the bar on 18 of 21, disjoint on
  6. Electricity's row is the 2026-09-27 rerun `impute_electricity_20nan_20260927_195444`; the
  2026-09-25 run scored every induced `day` cell as a miss (a dtype-spelling defect, fixed).
- **Conclusion:** "The stronger bar changes the verdict": below it on 6 of 21, never clearly;
  `knn10` beats the model clearly on pendigits and letter, mean/mode on credit-g_80nan.
  Decided: log both KNN sizes and `hgb`, read the baseline bar; every CV parent logs
  `baseline/best` and tag `best_baseline/<population>` (decision 8) and
  `gap_to_best_baseline/impute_score` (+ `_pct`), backfilled onto 24 runs (decision 9). The
  6-of-21 result motivated the full-profile studies below.
- **Sources:** `docs/adr/0007-baseline-imputers-scored-on-the-decoders-cells.md` (§ Status
  amendments, § Decision 8–9, § How to compare, § Outcome). Commits `7b6dbb9`, `42edbf2`
  (KNN-bar reading, 2026-09-27), `6cc8e19`, decision 9 in `4368d74`, `338ab5b`, `e79217a`
  (2026-09-28). Backfill tags `baselines_backfilled=true`, `best_baseline_gap_backfilled=true`.

### E20 · Sensitivity of the KNN bar to k (2026-09-24/25 – 2026-09-27)

- **Status:** done (scoring only, not pre-registered).
- **Hypothesis:** no direction stated: "How much the KNN bar depends on `k`", since the runs
  had logged only k = 5.
- **Scenarios:** no training; "the runs' own folds and masks, seed 42, 27 variants,
  `impute_score` averaged over 5 folds, induced population"; k = 5, 10, 20 uniform, and
  distance weighting at k = 5.
- **Measures and decision rule:** mean induced `impute_score` per variant across k; no rule.
- **Results:** k = 10 below k = 5 on 20 of 27, by 0.01 to 0.08; k = 20 "lower still wherever
  missingness is 40% or more" (`credit-g_20nan` 1.014 → 0.955 → 0.938; `spambase_20nan`
  0.995 → 0.939 → 0.925; `kr-vs-kp_40nan` 0.927 → 0.883 → 0.850; `pendigits_20nan`
  0.358 → 0.348); distance weighting worse than uniform at k = 5 on 21 of 27.
- **Conclusion:** "`k = 5` is the *weaker* of the reasonable bars at high missingness." Decided
  2026-09-27: log `knn5` and `knn10` and read the better; a single k = 10 was rejected because
  k = 5 is the better bar on small low-missingness tables (biodeg, kc2 at 20–40%). Replaced
  the same day by the four-baseline bar (entry above).
- **Sources:** ADR 0007 § Outcome "How much the KNN bar depends on `k`", § Status (first
  amendment), § Considered options. Commit `7b6dbb9`.

### E21 · Overnight full-profile Optuna studies against the baseline bar (2026-09-28)

- **Status:** done.
- **Hypothesis:** a pre-registered question: "Does a `full`-profile search find a configuration
  whose five-fold induced `impute_score` is below the baseline bar of the same run, on the
  tables where the decoder loses?" Motivation: below the bar on only 6 of 21; twelve
  `reduced` studies did not beat the defaults; the `full` profile never validated out of
  sample nor where a baseline wins clearly.
- **Scenarios:** `full` profile (architecture, pre-training and decode knobs; both stages 20
  to 60 epochs), 40 trials, `TPESampler(seed=42)`, cosine, seed 42, predefined single split,
  objective `validation/impute/masked/impute_score`, no `--promote_best`; best trial retrained
  five-fold at seed 42 with every baseline. Queue, once: pendigits_20nan, letter_20nan,
  electricity_20nan, kr-vs-kp_40nan, biodeg_20nan, spambase_20nan, credit-g_80nan.
- **Measures and decision rule:** primary induced CV mean vs the run's own `baseline/best`;
  "beats the bar" only where its upper bound is below the bar's lower bound. Secondary:
  fold-paired against the 2026-09-24/25 defaults run (2026-09-27 rerun for electricity); the
  masked population the same way. Stated limits: one seed, one trial per study, masked
  selection, a budget different from the defaults' 300 + 150 epochs.
- **Results (induced):**

  | variant | tuned | best baseline | verdict | tuned − defaults (folds lower) |
  |---|---|---|---|---|
  | `pendigits_20nan` | 0.345 [0.335, 0.355] | `knn10` 0.348 [0.339, 0.357] | overlaps | −0.054 [−0.060, −0.048], 5/5 |
  | `letter_20nan` | 0.469 [0.462, 0.475] | `knn10` 0.473 [0.468, 0.478] | overlaps | −0.046 [−0.053, −0.039], 5/5 |
  | `electricity_20nan` | 0.653 [0.613, 0.693] | `hgb` 0.593 [0.547, 0.640] | overlaps | −0.016 [−0.047, +0.015], 4/5 |
  | `kr-vs-kp_40nan` | 0.776 [0.748, 0.805] | `hgb` 0.772 [0.746, 0.797] | overlaps | −0.002 [−0.026, +0.022], 3/5 |
  | `biodeg_20nan` | 0.686 [0.520, 0.852] | `hgb` 0.644 [0.452, 0.837] | overlaps | +0.012 [−0.008, +0.032], 2/5 |
  | `spambase_20nan` | 0.891 [0.856, 0.926] | `hgb` 0.865 [0.834, 0.897] | overlaps | +0.005 [−0.022, +0.032], 2/5 |
  | `credit-g_80nan` | 1.160 [1.083, 1.237] | `mean_mode` 1.000 [1.000, 1.000] | loses to the bar | +0.137 [+0.045, +0.229], 0/5 |

  Only masked win: `letter_20nan` 0.505 [0.500, 0.510] vs `knn10` 0.544 [0.541, 0.547].
- **Conclusion:** of 14 comparisons the tuned model beats the bar once (letter, masked), loses
  once (credit-g_80nan, induced), overlaps on 12; "On the induced population, the headline, it
  beats the bar nowhere." Against the defaults, "the search helped where KNN is the bar and
  nowhere else" (pendigits, letter: 0.04–0.05 on every fold). On credit-g_80nan the masked
  objective's choice does not carry over to induced cells, which led to ADR 0008 and the two
  induced-objective studies. Every winner has 4–6 layers (default 2) and `LR_PRE` 2 to 34
  times below 3.4e-4, cited by the pre-training ablation.
- **Sources:** `docs/tickets/imputation-optuna-full/01-overnight-full-profile-studies.md`
  (§ Design, § Measures, § Outcome). Commits `8716eb8` (pre-registration), `69c9868`
  (interim, superseded), `3757565` (outcome). Retrains `impute_<variant>_20260928_*`, tags
  `optuna_study_run_id`, `retrained_from=best_trial`.
### E22 · Induced validation objective on credit-g_80nan (2026-09-28)

- **Status:** done
- **Hypothesis:** a pre-registered question, not a prediction: on `credit-g_80nan`, does
  ranking trials by `validation/impute/induced/impute_score` pick a configuration that does
  better on the induced test cells than the masked objective's pick? Motivation: study 01's
  masked winner was worse than the defaults on every fold (1.160 [1.083, 1.237] vs 1.024,
  fold-paired +0.137) and lost to mean/mode with disjoint intervals. ADR 0008 made an induced
  ranking possible.
- **Scenarios:** one arm, study 01's `credit-g_80nan` study with only the objective changed:
  `--search_space full`, 40 trials, `TPESampler(seed=42)`, `--lr_scheduler cosine`, seed 42,
  predefined single split, `--search_objective induced`, no `--promote_best`. The same seed
  repeats study 01's first draws. Winner retrained five-fold at seed 42. One study, one
  retrain, nothing rerun or extended.
- **Measures and decision rule:** primary `cv/test/impute/induced/impute_score/mean` with its
  95% interval, fold-paired against study 01's masked winner and the defaults ("better" only
  where the paired interval excludes zero), and against the run's best baseline ("beats the
  bar" only with disjoint intervals). Secondary: the same on masked cells. Exploratory:
  masked-vs-induced validation Spearman over the 40 trials; study 01's winner's induced rank.
- **Results:** study `optuna_credit-g_80nan_20260928_221137` picked trial 15 (validation
  induced 0.985). Induced test cells (negative difference favours the induced winner):

  | run | score | induced winner − run, fold-paired |
  |---|---|---|
  | induced winner (this study) | 1.017 [0.999, 1.035] | |
  | masked winner (study 01) | 1.160 [1.083, 1.237] | −0.143 [−0.225, −0.062], 5/5 folds |
  | defaults | 1.024 [1.008, 1.040] | −0.007 [−0.030, +0.017], 3/5 folds |
  | best baseline (`mean_mode`) | 1.000 [1.000, 1.000] | overlaps the bar |

  Masked cells: 1.004 [0.977, 1.031], level with the masked winner (−0.034 [−0.138, +0.069])
  and the defaults (+0.009 [−0.043, +0.060]). Validation objectives unrelated across the 40
  trials (Spearman −0.145, p = 0.37); study 01's winner (trial 9, a repeated draw) ranks
  **40th of 40** by induced validation (1.389). Winner: `DIM` 136, `HEADS` 8, `LAYERS` 3,
  `HIDDEN_DIM` 40, `DIM_FEED` 96, `DROPOUT` 0.4, `BATCH` 512, `PROB_MASCARA` 0.4, `LR_PRE`
  4.3e-4, `LR_DECODE` 5.3e-4, `LAMBDA_NUM` 2.55, `EPOCHS_PRE` 50, `EPOCHS_DECODE` 60.
- **Conclusion:** the induced objective chose better than the masked one (every fold, interval
  excludes zero), level with the defaults, and does not beat mean/mode, "which at 80%
  missingness nothing tried so far has". The masked objective "did not merely fail to find
  the best induced configuration; it picked the worst." Stated limit: one seed, one variant,
  one selected trial. Next: ticket 03 repeats the design on the other six variants.
- **Sources:** `docs/tickets/imputation-optuna-full/02-credit-g-80nan-induced-objective.md`;
  ADR 0008. Commits `b020653` (pre-registration), `a098e61` (outcome), `54741cc` (retrain name
  corrected to `impute_credit-g_80nan_20260928_223114`). Retrain tags `retrained_from=best_trial`,
  `selected_by=validation/impute/induced/impute_score`; masked winner
  `impute_credit-g_80nan_20260928_200740`; defaults `impute_credit-g_80nan_20260924_225231`.

### E23 · Induced validation objective on the other six variants (2026-09-28 – 2026-09-29)

- **Status:** stopped early (`electricity_20nan` stopped in its first trial at the user's
  request, before any result).
- **Hypothesis:** the same pre-registered question as ticket 02, on each of the six other
  variants study 01 ran under the masked objective. Motivation: on `credit-g_80nan` the
  induced objective won on every fold and the two validation objectives were unrelated.
- **Scenarios:** study 01's design with `--search_objective induced` (`--search_space full`,
  40 trials, `TPESampler(seed=42)`, `--lr_scheduler cosine`, seed 42, predefined split, no
  `--promote_best`); winner retrained five-fold at seed 42. Queue, cheapest first:
  `biodeg_20nan`, `kr-vs-kp_40nan`, `pendigits_20nan`, `spambase_20nan`, `letter_20nan`,
  `electricity_20nan`. One pass. Stated limit: on `_20nan` variants the induced validation
  population is smaller than the masked one, so selection may be noisier.
- **Measures and decision rule:** as in ticket 02, per variant; across variants, how many of
  the six are "better". No claim beyond these seven variants.
- **Results:** induced test cells; negative differences favour the induced winner; the fold
  count is folds on which the induced winner was lower.

  | variant | induced winner | masked winner | induced − masked | verdict | defaults | induced − defaults | best baseline | bar |
  |---|---|---|---|---|---|---|---|---|
  | `biodeg_20nan` | 0.655 [0.483, 0.827] | 0.686 | −0.031 [−0.060, −0.002], 5/5 | **better** | 0.674 | −0.019 [−0.042, +0.003], 4/5 | `hgb` 0.644 | overlaps |
  | `kr-vs-kp_40nan` | 0.784 [0.746, 0.821] | 0.776 | +0.008 [−0.008, +0.023], 1/5 | level | 0.778 | +0.006 [−0.008, +0.020], 2/5 | `hgb` 0.772 | overlaps |
  | `pendigits_20nan` | 0.354 [0.339, 0.369] | 0.345 | +0.009 [+0.004, +0.015], 0/5 | **worse** | 0.399 | −0.045 [−0.046, −0.044], 5/5 | `knn10` 0.348 | overlaps |
  | `spambase_20nan` | 0.900 [0.878, 0.923] | 0.891 | +0.009 [−0.012, +0.031], 1/5 | level | 0.886 | +0.015 [−0.007, +0.036], 1/5 | `hgb` 0.865 | overlaps |
  | `letter_20nan` | 0.497 [0.491, 0.503] | 0.469 | +0.028 [+0.023, +0.034], 0/5 | **worse** | 0.515 | −0.018 [−0.022, −0.014], 5/5 | `knn10` 0.473 | **loses** |
  | `electricity_20nan` | unfinished | | | | | | | |

  Masked cells (induced − masked winner): biodeg −0.037 [−0.077, +0.004]; kr-vs-kp −0.007
  [−0.058, +0.044]; pendigits +0.007 [−0.001, +0.015]; spambase +0.013 [−0.024, +0.050];
  letter +0.037 [+0.030, +0.044]. With `credit-g_80nan` from ticket 02: better on 2 of 6,
  level on 2, worse on 2.
- **Conclusion:** the induced objective is "not a uniform improvement", and "on no variant did
  it produce a model that beats its best baseline." Exploratory, as corrected by `8d7c01f`
  (the outcome `8072d01` first said it "orders the outcomes"): the masked–induced validation
  rank correlation "roughly tracks the outcomes, with one exception": −0.15 on credit-g_80nan
  (better), 0.80 on biodeg (better), 0.92–0.99 on spambase, pendigits, letter (level or
  worse); kr-vs-kp at 0.57 came out level. Read by the source as "a hypothesis for a
  pre-registered test, not a finding". No follow-up decision recorded; `masked` stays the
  default objective.
- **Sources:** `docs/tickets/imputation-optuna-full/03-induced-objective-on-the-other-variants.md`;
  `docs/tickets/imputation-optuna-full/01-overnight-full-profile-studies.md` § Outcome (masked
  winners, defaults, their intervals); ADR 0008. Commits `37b996f` (pre-registration),
  `8072d01` (outcome), `8d7c01f` (correction). Retrain tags as in ticket 02.
### E24 · What pre-training contributes to imputation (2026-09-29 – 2026-09-30)

- **Status:** done. 23 of the 32 cells ran on the Windows RTX 3050 on 2026-09-29; the other
  nine, plus pendigits' fresh A/42, on gorgona8 (RTX 3090 Ti) that night, finishing
  2026-09-30 01:10 GMT-3.
- **Hypothesis:** the pre-training stage adds little or nothing to imputation. It regresses
  onto the detached clean *embedding* of a masked cell, not its value; the critique found the
  resulting encoder worse than a per-column constant at that target and no detectable
  difference on induced cells between 300 and 2 pre-training epochs (F-13-1, F-13-2); every
  full-profile winner chose a pre-training learning rate 2 to 34 times below the default.
- **Scenarios:** three arms, only the epoch counts change; each variant's promoted
  `*.imputation.json` (or the defaults), `--lr_scheduler cosine`, five folds.
  A = `EPOCHS_PRE` 300 + `EPOCHS_DECODE` 150 (the current pipeline); B = 0 + 150 (no
  pre-training); C = 0 + 450 (the same epochs, all on the value loss). Variants
  credit-g_20nan, credit-g_80nan, kr-vs-kp_40nan, pendigits_20nan; seeds 42, 7, 13, fixed in
  advance. A at seed 42 reuses the runs of 2026-09-24/25 (reproduced to the last digit),
  except pendigits (see amendment). MLflow tag `ablation=pretraining-2026-09-29`, `arm`.
- **Measures and decision rule:** primary `impute/induced/impute_score` (lower is better),
  B − A and C − A paired by seed and fold (15 pairs), t-interval over the pairs. "Pre-training
  helps" only when the interval excludes zero in A's favour **and** all three seeds agree in
  sign; the mirror case "hurts"; otherwise "no detectable difference". Secondary: masked
  cells the same way, each arm against the run's best baseline, wall time per arm (per
  machine). One pass, nothing rerun or extended on its result.
- **Amendment 2026-09-29, 21:24 GMT-3 (before the remaining cells):** a check cell (credit-g_20nan s42 B,
  MLflow off) on the server gave induced 0.9165 vs Windows 0.9186 (per-fold sd of the gap
  0.044, mean −0.002); mean_mode/knn5/knn10 baselines identical, so folds and masks are
  shared across machines, and the model is another training draw on a different GPU.
  pendigits A/42 runs fresh on the server so its pairs stay on one machine; kr-vs-kp s13 is
  the one declared cross-machine pair (B, C Windows; A server).
- **Amendment 2026-09-29, 23:08 GMT-3 (paused at 22:13 GMT-3 mid pendigits s42 C, which is not
  counted and runs again from the start):** the last eight cells write to the SSD store in WAL
  mode with batched metric writes (ADR 0010, ticket 0005); metric values are unchanged
  (ticket 0005's exactness check), so scores stay comparable. Wall time does not: the cost
  measure becomes the sum of fold `total_seconds` (taken before the store write), with run
  wall clock reported per machine and per store.
- **Results (induced, B − A and C − A over 15 fold pairs; positive favours A):**

  | variant | B − A | C − A | best baseline |
  |---|---|---|---|
  | credit-g_20nan | −0.0015 [−0.0173, +0.0142], no detectable difference | +0.0092 [−0.0047, +0.0230], no detectable difference | hgb 0.918; every arm overlaps it |
  | credit-g_80nan | +0.0218 [−0.0065, +0.0502], no detectable difference | +0.0141 [−0.0121, +0.0404], no detectable difference | mean_mode 1.000; every arm loses to it |
  | kr-vs-kp_40nan | +0.0105 [−0.0049, +0.0258], no detectable difference | +0.0184 [+0.0046, +0.0323], **A better than C** | hgb 0.773; every arm overlaps it |
  | pendigits_20nan | −0.0216 [−0.0250, −0.0181], **pre-training hurts** | −0.0664 [−0.0701, −0.0627], **C better than A** | knn10 0.350; **C 0.335 [0.331, 0.340] beats it**, A and B lose |

  Masked cells agree on every variant. kr-vs-kp's C − A rests on the declared cross-machine
  seed 13: over its 10 same-machine pairs it is +0.0179 [−0.0034, +0.0393]. Machine gap on
  pendigits A/42 (server − Windows): +0.0036, per-fold sd 0.0050, against 0.044 on the
  credit-g check cell. Cost (fold time per cell): C about 1.2x–1.8x A; server pendigits A 13.5,
  B 7.6, C 21.4 min.
- **Conclusion:** no variant shows "pre-training helps" against none. The stage's worth depends
  on the table: on the all-numerical pendigits the embedding objective costs 0.022 against
  none, and the same epochs on the value loss gain 0.066 and clear the best baseline for the
  first time in the effort (the decode stage is under-trained at 150 epochs there); on the
  all-categorical kr-vs-kp that move loses 0.018 (resting on the cross-machine pair); nothing
  is detectable on credit-g. Next: the value and normalised-embedding objectives (E25).
- **Sources:** `docs/tickets/imputation-pretraining/01-pretraining-ablation.md`
  (pre-registration 7af595b, amendment ebf5e86); analysis script
  `~/trident-handoff-2026-09-29/kit/pretrain_ablation_report.py` (outside the repo, on
  gorgona8); `docs/reviews/imputation-critique/13-pretraining-transfer.md`.

### E25 · Value and normalised-embedding pre-training objectives (2026-09-30)

- **Status:** done, 01:18–04:28 GMT-3, all 50 cells, none missing (relaunched once at 01:38
  with a CPU thread cap after an oversubscription stall; scores unaffected).
- **Hypothesis:** the current pre-training target (the detached clean embedding) lets the
  loss fall by rescaling rather than predicting (F-13-1), which would explain why the stage
  adds little (E04, E05, E24). A target aligned with the values (`value`) or one that cannot
  be met by rescaling (`embedding_normalized`, layer-normalised target through a predictor
  head) might make pre-training useful (ADR 0011).
- **Scenarios:** arms A (`embedding`, 300 + 150 epochs), B (no pre-training, 0 + 150), C (0 +
  450, added after E24's outcome, before any cell ran), V
  (`value`, 300 + 150), N (`embedding_normalized`, 300 + 150); promoted configuration or
  defaults, cosine, five folds; variants credit-g_20nan, credit-g_80nan, kr-vs-kp_40nan,
  pendigits_20nan; seeds 42, 7, 13. Every A, B and C reference is a server cell: the ablation's
  server cells are reused (pendigits A, B and C, kr-vs-kp s13 A), the other 26 run now. V differs
  from the ablation's C by being its own stage (pre-training's rate, weight decay and cycle,
  heads discarded) before a fresh decode stage. Tag `experiment=pretrain-objective-2026-09-30`.
- **Measures and decision rule:** primary V − A and N − A on `impute/induced/impute_score`,
  paired by seed and fold (15 pairs), t-interval; a verdict only when the interval excludes
  zero and all three seeds agree in sign. Secondary: V − C and N − C (reading fixed in advance:
  V ≈ C → drop pre-training and lengthen decode; V better → keep a value stage; C better →
  train the decoder longer), V − B, N − B, masked cells, each arm against the best baseline. Cost descriptive only (two trainers share the GPU).
- **Results (induced, 15 fold pairs; negative favours the new arm):**

  | variant | V − A | N − A | N − B | N − C |
  |---|---|---|---|---|
  | credit-g_20nan | −0.0002, none | +0.0060, none | −0.0054, none | −0.0044, none |
  | credit-g_80nan | +0.0267, **A better** | −0.0147, **N better** | −0.0290, **N better** | −0.0099, none |
  | kr-vs-kp_40nan | +0.0233, **A better** | +0.0001, none | −0.0191, **N better** | −0.0288, **N better** |
  | pendigits_20nan | −0.0421, **V better** | −0.0373, **N better** | −0.0157, **N better** | +0.0291, **C better** |

  V − C: C better on credit-g_80nan and pendigits, none elsewhere. Masked cells agree. Only C
  on pendigits clears a best baseline.
- **Conclusion:** the value target is not what pre-training lacks (better than the current
  objective only on the all-numerical table, worse on two, never better than C). The
  normalised target is: never worse than the current objective, better on two variants, and
  better than no pre-training on three of four, which the current objective is on none (E24).
  The decode budget that wins on pendigits loses on kr-vs-kp. Candidate default for
  imputation: `embedding_normalized` (a decision for the user, ADR 0011); E26 tests it with
  the long decode stage.
- **Sources:** `docs/tickets/imputation-pretraining/02-pretraining-objectives.md`;
  `docs/adr/0011-selectable-pretraining-objective.md`;
  `scripts/experiments/pretrain_objective_report.py`.

### E26 · Normalised-target pre-training with a long decode stage (2026-09-30)

- **Status:** done, 03:00–05:00 GMT-3, all 12 cells.
- **Hypothesis:** the two partial wins of the night might add up: the long decode stage (arm C,
  450 epochs, best on pendigits in E24) and the normalised embedding objective (N, better than
  the current objective on pendigits and credit-g_80nan in E25's first cells). Motivation post
  hoc, design fixed before launch.
- **Scenarios:** arm M = `embedding_normalized` pre-training 300 epochs + decode 450, against C
  (0 + 450) and N (300 + 150), all on gorgona8; four variants, seeds 42, 7, 13; 12 new cells
  tagged `experiment=pretrain-decode-budget-2026-09-30`.
- **Measures and decision rule:** primary M − C on `impute/induced/impute_score`, 15 fold pairs,
  verdict only when the interval excludes zero and all three seeds agree in sign; secondary
  M − N, masked cells, best baseline. M trains 750 epochs against C's 450, so "M better"
  confounds objective and budget.
- **Results (induced, 15 fold pairs; negative favours M):** M − C: credit-g_20nan −0.0065,
  none; credit-g_80nan −0.0042, none; kr-vs-kp −0.0205 [−0.0373, −0.0037], **M better**;
  pendigits −0.0013, none. M − N: pendigits −0.0305, **M better**; none elsewhere. Pendigits M
  0.334 beats `knn10` 0.350, as C does. Masked: no M − C verdict; M better than N on pendigits.
- **Conclusion:** normalised pre-training plus the 450-epoch decode stage is never worse than
  either half: it keeps the long decode's gain on pendigits and pre-training's gain on
  kr-vs-kp. Against A, B and V it was not compared formally; descriptively it sits within
  noise of the best arm on each variant. The extra budget (750
  against 450 epochs) is a confound, but N − C on kr-vs-kp in E25 (450 epochs in total) points
  the same way. Candidate imputation default for the user to decide: `embedding_normalized`
  with `EPOCHS_DECODE` 450.
- **Sources:** `docs/tickets/imputation-pretraining/03-normalised-pretraining-with-a-long-decode.md`;
  `scripts/experiments/pretrain_decode_budget_report.py`.

### E27 · Stopping the decode stage on a validation plateau (2026-09-30)

- **Status:** done, 13:16–16:00 GMT-3, all 12 cells (two rerun from the start after a
  finalisation crash, `a1d54a9`).
- **Hypothesis:** the 450-epoch decode stage's gain lives where the validation loss is still
  falling (pendigits, best epoch ~390), and elsewhere it only costs time (best epochs 9–87), so a
  patience of 50 should keep M's scores while training far fewer epochs.
- **Scenarios:** arm P = M plus `DECODE_PATIENCE` 50, against M (E26) and N (E25); four variants,
  seeds 42, 7, 13; credit-g_20nan pinned to the configuration M read (`fcaadfb`). Tag
  `experiment=decode-early-stopping-2026-09-30`.
- **Measures and decision rule:** primary P − M on `impute/induced/impute_score`, 15 fold pairs,
  verdict only when the interval excludes zero and all seeds agree; decode epochs trained.
  Secondary P − N, masked, best baseline. Limits: the schedule still spans 450 epochs; a stop
  shifts later folds' training draws.
- **Results (induced, 15 fold pairs):** P − M: credit-g_20nan +0.0001, credit-g_80nan −0.0031,
  kr-vs-kp −0.0018, all no detectable difference; pendigits +0.0057 [+0.0006, +0.0108], **M
  better**. Decode epochs trained (median of 450): 91, 61, 107, 388. Fold time per cell: 4.1 →
  1.5, 5.9 → 1.4, 18.5 → 4.9, 35.4 → 30.5 min (descriptive). Fold 1 equals M to the last digit in
  9 of 12 cells, those where M's best epoch preceded P's stop.
- **Conclusion:** a patience of 50 is a safe speed-up where the long decode buys nothing (three
  of four tables) and a small loss on pendigits, where the validation loss still falls late.
- **Sources:** `docs/tickets/imputation-pretraining/04-decode-early-stopping.md`;
  `scripts/experiments/decode_patience_batch_report.py`.

### E28 · Batch 1024 with a scaled learning rate (2026-09-30)

- **Status:** done, 13:16–15:27 GMT-3, all 24 cells.
- **Hypothesis:** a four times larger batch runs an epoch 1.55x–2.1x faster; with both learning
  rates scaled by the square-root (x2) or linear (x4) rule it may keep M's scores.
- **Scenarios:** arms Q2 and Q4 = M with `BATCH` 1024 and `LR_PRE`, `LR_DECODE` x2 or x4, against
  M (E26); four variants, seeds 42, 7, 13. Tag `experiment=larger-batch-2026-09-30`.
- **Measures and decision rule:** primary Q2 − M and Q4 − M, as in E27; secondary Q4 − Q2,
  masked, best baseline, fold time per cell.
- **Results (induced, 15 fold pairs; negative favours the larger batch):** Q2 − M:
  credit-g_20nan −0.0033, none; credit-g_80nan +0.0146, none; kr-vs-kp +0.0115, **M better**;
  pendigits −0.0035, **Q2 better**. Q4 − M: credit-g_20nan +0.0190, **M better**;
  credit-g_80nan +0.0095, none; kr-vs-kp +0.0012, none; pendigits −0.0107, **Q4 better** (0.323,
  the best pendigits score yet). Fold time per cell falls by about 40% on pendigits and 25% on
  kr-vs-kp (descriptive).
- **Conclusion:** the gain depends on how many steps an epoch keeps: batch 1024 leaves one step
  an epoch on credit-g, three on kr-vs-kp, eight on pendigits, and only pendigits gains. No
  single rule is safe across tables; a per-variant batch or a steps-aware rule would be the
  next question.
- **Sources:** `docs/tickets/imputation-pretraining/05-larger-batch.md`;
  `scripts/experiments/decode_patience_batch_report.py`.

### E29 · Batched decoder heads against per-column (2026-09-30 – 2026-10-01)

- **Status:** done, 23:19–00:08 GMT-3, all 12 cells.
- **Hypothesis:** the batched heads (`c9be71f`) regroup the same arithmetic, so beyond giving
  another training draw they change nothing; a pairing over 15 folds per variant would show no
  detectable difference.
- **Scenarios:** arm H = arm A with `DECODER_HEADS` batched, against the per-column A cells on
  the server; four variants, seeds 42, 7, 13; credit-g_20nan pinned to `fcaadfb`. Tag
  `experiment=batched-heads-2026-10-01`.
- **Measures and decision rule:** primary H − A on `impute/induced/impute_score`, 15 fold pairs,
  verdict only when the interval excludes zero and all seeds agree. "No detectable difference"
  on all four licenses the default switch; any verdict keeps per-column.
- **Results (induced, 15 fold pairs):** H − A: credit-g_20nan +0.0026, credit-g_80nan +0.0001,
  kr-vs-kp −0.0006, pendigits +0.0008 [−0.0000, +0.0016] (all three seeds +0.0006 to +0.0009); no
  detectable difference on any variant, induced or masked.
- **Conclusion:** the rule licenses the batched heads as the default. Caveat: pendigits leans the
  same way on every seed by 0.2% of the score, an order below its machine gap.
- **Sources:** `docs/tickets/imputation-pretraining/06-batched-decoder-heads.md`;
  `scripts/experiments/batched_heads_report.py`.

### E30 · The candidate configuration on all 21 imputation variants (2026-10-01)

- **Status:** done, 00:22–13:19 GMT-3 on gorgona8, all 189 cells, none failed or missing.
- **Hypothesis:** the normalised pre-training objective with a 450-epoch decode stage (M), and its
  early-stopped form (P), hold their E25–E27 gains, or at least lose nowhere, beyond the four
  variants they were found on.
- **Scenarios:** arms A (current default), M, P; the 21 variants of ADR 0007; seeds 42, 7, 13; each
  variant's configuration as loaded today; batched decoder heads; all 189 cells fresh. Tag
  `experiment=confirmatory-2026-10-01`.
- **Measures and decision rule:** primary M − A and P − A on `impute/induced/impute_score`, 15 fold
  pairs per variant, verdict only when the interval excludes zero and all seeds agree; tally over
  21 variants. Default reading fixed in advance: a candidate qualifies with no "A better" verdict;
  P is recommended over M unless "M better than P" on three or more variants; neither qualifying
  keeps A.
- **Note, 2026-10-01 12:36 GMT-3:** queue 2, projected to end near the cutoff, split across two
  launchers once queues 3 and 4 had finished (scheduling only; no cell or cutoff changed; ticket
  07, Running notes).
- **Results (induced, 15 fold pairs per variant):** M − A: M better on 9 variants (kr-vs-kp_80nan
  −0.024, spambase_20nan −0.015, vehicle_20nan −0.029, vehicle_60nan −0.012, biodeg_20nan −0.046,
  biodeg_60nan −0.058, pendigits −0.068, letter −0.054, electricity −0.034), A better on none.
  P − A: P better on 7, A better on 2 (kr-vs-kp_40nan +0.014, spambase_40nan +0.014). P − M: M
  better on 3 (biodeg_60nan, kc2_60nan, letter). P is also worse than A on all three seeds of
  spambase_60nan (+0.070, interval too wide for a verdict). Masked cells: A better than both
  candidates on kr-vs-kp_20nan; both better than A on biodeg_60nan, pendigits and letter. Mean
  induced score below the best baseline on 7 variants for A, 11 for M, 9 for P. A cell of M costs
  1.65x A's time summed over the variants, P 1.22x. Wall time 12 h 57 min against about 8.5 h
  planned (1.52x).
- **Conclusion:** by the rule fixed in advance, M qualifies and P does not, so the recommendation is
  M (`embedding_normalized` pre-training with a 450-epoch decode stage) as the imputation
  default; the user decides (T00). Its verdicts fall on pendigits, letter, electricity, both
  biodeg and both vehicle variants, spambase_20nan and kr-vs-kp_80nan; no credit-g or kc2
  variant shows a detectable difference. P stopped after a mean of 100 and 151 decode epochs
  where it lost (kr-vs-kp_40nan, spambase_40nan); whether a longer patience recovers those is
  T11's question. **Decided 2026-10-01:** M is the imputation default, with the `cosine`
  schedule it ran under ([ADR 0013](adr/0013-imputation-defaults-from-the-confirmatory-study.md)).
- **Sources:** `docs/tickets/imputation-pretraining/07-confirmatory-21-variants.md` (Outcome);
  `scripts/experiments/confirmatory_report.py`; report, tally and secondary measures in
  `/scratch2/diogoneiss/trident-2026-10-01-confirmatory/`.

### E31 · The induced gaps asked one column at a time, the rest as `[NULL]` (2026-10-01 – 2026-10-02)

- **Status:** done, 23:36 GMT-3 to 05:35 GMT-3 on gorgona8, all 63 cells; the mask path matched
  E30's arm M to the last digit on every cell.
- **Hypothesis:** the decode stage trains on rows whose real gaps are `[NULL]` and only a share of
  the other cells is `[MASK]` (about 2% of an 80nan row), but the induced headline shows every gap
  as `[MASK]` at once. Scoring the same cells in the trained shape, one gap column at a time with
  the other gaps as `[NULL]`, should score better, and more so where gaps are many (T02 step 1).
- **Scenarios:** one arm, today's defaults (ADR 0013, E30's M), with `--score_column_wise`; the
  same model scores the induced cells both ways. All 21 variants, seeds 42, 7, 13, five folds; 63
  cells tagged `experiment=column-wise-2026-10-02`.
- **Measures and decision rule:**
  - **Primary:** D = column-wise − mask on `impute/induced/impute_score`, 15 fold pairs per
    variant from one model each. A verdict needs the interval to exclude zero and all three seeds
    to agree in sign.
  - **Reading on the 9 variants at 60nan or more:**
    - supported if column-wise is better on at least 5 and mask better on none;
    - refuted if mask is better on at least 5, or if at least 7 have no verdict;
    - mixed otherwise.
  - **Secondary:** D by missing level; variants below the best baseline under each path; the mask
    path must equal E30's M cells to the last digit.
- **Results (column-wise − mask, induced, 15 fold pairs per variant):** on the 9 variants at
  60nan or more, column-wise better on 5 (credit-g_80nan −0.015, kr-vs-kp_60nan −0.016,
  spambase_80nan −0.032, biodeg_60nan −0.038, kc2_60nan −0.042), mask better on none, no verdict
  on 4. On the 12 light variants, mask better on 6 (pendigits +0.001, letter +0.001, kr-vs-kp_20nan
  +0.012, kr-vs-kp_40nan +0.009, spambase_20nan +0.013, vehicle_20nan +0.009), column-wise on
  none. Mean difference by level: +0.0043 (20nan), −0.0072 (40), −0.0222 (60), −0.0120 (80).
  Means over 21: mask 0.7745, column-wise 0.7673, best baseline 0.7781; below the best baseline
  on 11 and 12 variants. Wall time 5 h 59 min against about 5.2 h planned (1.15x).
- **Conclusion:** supported by the rule fixed in advance, at its threshold (5 of 9, two of them
  with an upper bound just under zero). The train/score token-shape mismatch
  costs real score where gaps are many and nothing where they are few, so it is a cause of the
  model's losses at high missingness, not the only one (credit-g_60nan and _80nan and
  kr-vs-kp_80nan stay at or above the mean/mode fill under both paths). The two pre-stated
  follow-ups, step 2 of T02 (train on the induced row shape) and an ADR making column-wise the
  headline, wait on the user's decision.
- **Sources:** `docs/tickets/imputation-token-shape/01-column-wise-scoring.md` (Outcome);
  `scripts/experiments/column_wise_report.py`.

### E32 · Training the decode stage with the gaps shown as `[MASK]` (2026-10-02)

- **Status:** done, 10:44 to 16:22 GMT-3 on gorgona8, all 63 cells, none failed or missing.
- **Hypothesis:** E31 showed the train/score token-shape mismatch costs score where gaps are
  many. Showing every real gap as `[MASK]` in the decode stage (`DECODE_GAP_TOKEN mask`), the
  shape the induced headline scores in, should recover those gains from the training side
  without the price column-wise scoring paid at 20nan (T02 step 2).
- **Scenarios:** arm G = today's defaults (ADR 0013) with `DECODE_GAP_TOKEN mask`, against E30's
  arm-M cells (read through E31's). All 21 variants, seeds 42, 7, 13, five folds; 63 new cells
  tagged `experiment=gap-token-2026-10-02`. Two trained models per pair, as in E30.
- **Measures and decision rule:**
  - **Primary:** D = G − M on `impute/induced/impute_score`, 15 fold pairs per variant; a
    verdict needs the interval to exclude zero and all three seeds to agree in sign.
  - **Reading on the 9 variants at 60nan or more:** supported if G is better on at least 5 and
    M better on none; refuted if M is better on at least 5, or if at least 7 have no verdict;
    mixed otherwise. G is recommended as the default only if supported and no variant among the
    21 shows M better.
  - **Secondary:** D by missing level; G against M scored column-wise (E31); the masked
    population; variants below the best baseline and at or above 1.0; time.
- **Results (G − M, induced, 15 fold pairs per variant):** G better on 11 of 21 variants
  (kr-vs-kp_20nan −0.022, kr-vs-kp_60nan −0.026, spambase_20nan −0.018, spambase_40nan −0.051,
  spambase_60nan −0.030, vehicle_20nan −0.017, vehicle_60nan −0.024, biodeg_20nan −0.026,
  biodeg_60nan −0.058, kc2_60nan −0.067, letter −0.001), M better on none; on the 9 heavy
  variants, 5 and 0. Mean by level: −0.011 (20nan), −0.026 (40), −0.036 (60), −0.012 (80).
  Against M scored column-wise (E31): G better on 12, worse on none. Means over 21: G 0.7542,
  M 0.7745, column-wise 0.7673, best baseline 0.7781; below the best baseline on 16 variants
  (M: 11). Masked population: G better on 10, worse on none. Wall time 5 h 38 min against about
  5.8 h planned (0.97x); a cell costs the same as M.
- **Conclusion:** supported by the rule fixed in advance, and G qualifies as the default (no
  variant shows M better). Showing the gaps as `[MASK]` in the decode stage, the shape the
  headline scores in, gains at every missing level, where column-wise scoring (E31) gained only
  where gaps were many and cost a little elsewhere: aligning the training beats aligning the
  scoring. The recommendation is `DECODE_GAP_TOKEN mask` as the imputation default, by an ADR;
  the user decides. credit-g_60nan and _80nan and kr-vs-kp_80nan stay at the mean/mode fill
  under both arms, so the token shape was not the only cause there. **Decided 2026-10-03:** the
  user adopted it ([ADR 0014](adr/0014-gaps-shown-as-mask-in-the-decode-stage.md)) and asked for
  a replication on fresh seeds (E33).
- **Sources:** `docs/tickets/imputation-token-shape/02-gaps-as-mask-in-training.md` (Outcome);
  `scripts/experiments/gap_token_report.py`.

### E33 · The gaps-as-mask default on fresh seeds (2026-10-03)

- **Status:** done, 00:44 to 12:05 GMT-3 on gorgona8, all 126 cells, none failed or missing.
- **Hypothesis:** E32's "better on 11 of 21, worse on none" is a property of showing the gaps as
  `[MASK]` in the decode stage, not of the three seeds it was chosen on.
- **Scenarios:** arms G (ADR 0014's defaults, naming nothing) and M (`--decode_gap_token null`),
  both fresh; all 21 variants; seeds 101, 202, 303, unused by any earlier imputation study; five
  folds; 126 cells tagged `experiment=gap-token-replication-2026-10-03`.
- **Measures and decision rule:**
  - **Primary:** D = G − M on `impute/induced/impute_score`, 15 fold pairs per variant; a verdict
    needs the interval to exclude zero and all three seeds to agree in sign.
  - **Reading:** replicated if no variant shows M better and G is better on at least 5 of the 9
    heavy variants; safe but weaker if no variant shows M better and G wins fewer than 5;
    not replicated if any variant shows M better, which goes back to the user.
  - **Secondary:** D by level; E32 and E33 pooled (six seeds); masked; means and the best-baseline
    count; time.
- **Results (G − M, induced, 15 fold pairs per variant, fresh seeds):** G better on 13 of 21
  variants, M better on none; on the 9 heavy variants, 6 and 0 (credit-g_80nan −0.026,
  kr-vs-kp_60nan −0.025, spambase_60nan −0.076, vehicle_60nan −0.022, biodeg_60nan −0.056,
  kc2_60nan −0.093). Mean by level: −0.018 (20nan), −0.019 (40), −0.048 (60), −0.019 (80).
  Pooled with E32 (six seeds): G better on 12, M on none. Means over 21: G 0.7519, M 0.7786;
  below the best baseline on 15 and 9 variants. Masked: G better on 8, M on none. Wall time 11 h
  21 min against about 11.3 h planned (1.00x).
- **Conclusion:** replicated by the rule fixed in advance; the effect is larger on the fresh seeds
  than on E32's, not smaller. ADR 0014 stands on six seeds. credit-g_60nan and _80nan and
  kr-vs-kp_80nan remain at the mean/mode fill under both arms; kr-vs-kp_80nan is the only
  variant leaning M (+0.002, no verdict, as in E32).
- **Sources:** `docs/tickets/imputation-token-shape/03-replication-on-fresh-seeds.md` (Outcome);
  `scripts/experiments/gap_token_replication_report.py`.

### E34 · Checkpoint by the validation gaps, and calibration on them (2026-10-03 – )

- **Status:** in progress (gorgona8, four concurrent queues; no cell starts after 02:00 GMT-3 on
  2026-10-04).
- **Hypothesis:** the decode stage chooses its checkpoint on a few noisy masked cells instead of
  the validation split's own gaps, the population it is scored on (T07), and its guesses are
  overconfident (median slope of truth on guess 0.85 in E30's ledgers). Choosing the epoch by the
  validation gaps' `impute_score` (K), or calibrating the guesses on those gaps (C), should score
  better.
- **Scenarios:** one run per cell under ADR 0014's defaults with `--score_induced_checkpoint` and
  `--score_calibrated`; four readouts of one trajectory: H (headline), C, K, KC. All 21 variants,
  seeds 101, 202, 303 (H must equal E33's G cells), five folds; 63 cells tagged
  `experiment=selection-2026-10-03`. Calibration constants fixed in the ticket.
- **Measures and decision rule:**
  - **Primary:** C − H and K − H on `impute/induced/impute_score`, 15 within-run fold pairs per
    variant; a verdict needs the interval to exclude zero and all three seeds to agree in sign.
  - **Reading on all 21:** a candidate qualifies if no variant shows H better; each qualifying
    candidate is recommended on its own; KC over them only if it qualifies and loses to neither;
    if neither qualifies, nothing changes. A default switch needs its own ADR.
  - **Secondary:** KC against H, K and C; K's masked score; the two criteria's epochs; the
    calibration's reach; by level; means; best-baseline counts; time.
- **Results:** pending.
- **Conclusion:** pending.
- **Sources:** `docs/tickets/imputation-token-shape/04-checkpoint-and-calibration.md`;
  `scripts/experiments/selection_report.py`.
