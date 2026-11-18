# Graph Report - TRIDENT  (2026-11-18)

## Corpus Check
- 772 files · ~1,630,081 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 7777 nodes · 15124 edges · 533 communities (476 shown, 41 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 451 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e92dbc55`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- MultiHeadAttention
- test_opt_search_space.py
- 0005-reduced-optuna-search-for-imputation.md
- Hyperparameters
- Answer
- test_mirroring.py
- TRIDENT Experiment Tracking Context (CONTEXT.md)
- 3. Design and methodology critiques judged sound
- TRIDENT Architecture
- Answer
- generate_splits.py
- StoreError
- Answer
- selection_replication_report.py
- trident
- tabpfn_seed_sets_report.py
- Which columns does the decoder fill each column from? (E78; pre-registered 2026-10-26)
- imputation_metrics.py
- test_training_schedulers.py
- TRIDENT Backlog: Bugs, Pendencies and Improvements
- mean_mode_baselines
- test_training_tracking.py
- test_c15_inputs.py
- decoding.py
- linear_baselines.py
- backfill
- Findings
- backfill_lr_scheduler_tag.py
- Reduced Optuna Search for Imputation: Implementation Plan
- preprocess_table
- test_training_decoding.py
- mirror_run_tree
- interval
- test_prune_cache.py
- spread_mechanisms.py
- run_hyperparameter_optimization
- backfill_baseline_imputers.py
- Imputation Decoder Task Implementation Plan
- Resolution
- imputation_baselines.py
- Answer
- 05. Third-party stubs for sklearn and scipy
- complete_configuration
- test_lightgbm_baseline.py
- Findings
- 02. Request, hyperparameters and command line
- Reduce the imputation search to the knobs that move it, score it on the validation split, and promote its result explicitly
- column_wise_report.py
- 03. Imputation metrics as pure functions
- Add an imputation task with a value decoder, keep classification the default, and tag it in MLflow
- ArtifactWriter
- 06. The decode stage
- 07. Imputation artifacts
- 08. Runner, MLflow identity, batch summary
- 09. Tag backfill
- 10. Optuna for the imputation task
- gap_reliance.py
- Map: Imputation decoder task
- summarize_cross_validation
- conformal_row_gaps.py
- occlusion_map.py
- DatasetSpec
- 10. Record the design: ADR and implementation plan
- model
- test_backfill_decode_checkpoint_tag.py
- PreparedDataset
- test_backfill_decode_calibration_tag.py
- T29's test: is the boosters' lead on electricity carried by `date` as a key? (E54; pre-registered 2026-10-20)
- leadership.py
- architecture_report.py
- CLAUDE.md
- backfill_encoder_final_norm_tag.py
- confirmatory_report.py
- pretrain_gap_token_report.py
- 01. Search-space profiles
- Findings
- Verdicts
- 02. The search objective on the validation split
- 04. The importance artifact
- 05. The launcher and the six studies
- 06. The comparison and the documentation
- Verdicts
- Verdicts
- as_category_strings
- test_imputation_artifacts.py
- MissForestImputer
- run_confirmatory_queue.sh
- sync_remote.py
- gap_token_replication_report.py
- Make the learning-rate schedule selectable, keep the legacy one as default, and tag it in MLflow
- impprobe_arms.py
- pretrain_probe.py
- mimic_report.py
- test_sync_remote.py
- types.py
- 03. Task-keyed lookup and explicit promotion
- Research: third-party stub strategy for sklearn and scipy
- Keep missForest outside the baseline bar, and put a single-pass LightGBM in it
- train_and_evaluate_decoder
- 06. The hyperparameter JSON schema and the three naming layers
- 07. Enforcing the constants convention beyond what the checker catches
- 09. Record the design: ADR 0006 and the implementation plan
- _ValidationGapScore
- Map: Typing and naming standard for TRIDENT
- Addendum — the Optuna results, 2026-09-15
- 2. Confirmed defects
- Findings
- test_typing_gate.py
- Findings
- Findings
- Verdicts
- Verdicts
- Findings
- Verdicts
- Findings
- Findings
- Findings
- Findings
- Verdicts
- Verdicts
- Findings
- Verdicts
- Verdicts
- Findings
- Verdicts
- Verdicts
- Verdicts
- backfill_decode_gap_token_tag.py
- Fidelity audit of `CONSOLIDATED.md`
- test_backfill_baseline_imputers.py
- §8 — Experiments run 2026-09-17
- Training Refactor Implementation Plan
- Consolidated report — the imputation work on `feat/imputation-task`
- 4. What the experiments can and cannot support
- Appendix: every reader's summary lines (`results_digest.py`, unedited)
- 0014-gaps-shown-as-mask-in-the-decode-stage.md
- test_backfill_pretrain_gap_token_tag.py
- Entries
- RsyncTransport
- build_fold_result
- DirectoryServer
- batched_heads_report.py
- Transport
- gap_token_report.py
- summary.py
- pretrain_decode_budget_report.py
- sliced_run.py
- Counter
- 2026-10-23-classification-step-zero.md
- ADR 0037: The bigger encoder on pendigits and letter at 40 and 60nan
- not_miwae.py
- Overnight on gorgona8, 2026-09-29/30
- 8. The configuration for ticket 08
- tally
- test_embedder.py
- _MetricClock
- 0004-imputation-decoder-task.md
- backfill_search_objective_tag.py
- What does pre-training contribute to imputation? (pre-registered 2026-09-29)
- decode_patience_batch_report.py
- test_leadership.py
- Mirror every run into one derived MLflow experiment per task
- backfill_lgbm_baseline.py
- pretrain_objective_report.py
- start_mlflow.sh script
- Do batched decoder heads change anything but the training draw? (pre-registered 2026-09-30)
- cls_stage1_bar.py
- TASKS.md
- main
- The deployable hybrid on `mimic-wide`: safe and still a gain where TabPFN alone is weak? (E178, T57; pre-registered 2026-11-13 08:28 GMT-3)
- setup_mlflow
- baseline_cache.py
- joint_draws.py
- mnar_family.py
- mlflow_utils.py
- test_training_tasks.py
- TabPFNImputer
- _DisabledMlflow
- test_training_pretraining.py
- run_pretrain_decode_budget_queue.sh
- Next up, in the recommended order
- TRIDENT README
- run_decode_patience_batch_queue.sh
- backfill_decode_patience_tag.py
- backfill_decoder_heads_tag.py
- remasker.py
- run_pretrain_objective_queue.sh
- calibration-as-imputation-output.md
- propensity_weights
- Do decode masks that follow the missingness help under MAR? A propensity-mask arm (E65; pre-registered 2026-10-22)
- _cells
- ADR 0038: Label-scarce fine-tuning as a documented regime for classification
- TabularEmbedder
- run_gap_token_queue.sh
- run_column_wise_queue.sh
- 0013-imputation-defaults-from-the-confirmatory-study.md
- Rascunho: TRIDENT como imputador — fundamentação, metodologia, resultados e direções
- BaselineImputer
- stage4_gbdt.py
- run_selection_queue.sh
- run_batched_heads_queue.sh
- large_batch.py
- run_pretrain_gap_token_queue.sh
- Does a bigger encoder help the large tables? (pre-registered 2026-10-06)
- 04. The decoder model
- Apontamentos por slide
- Does ADR 0037's bigger encoder help off MCAR? (E119; pre-registered 2026-11-03)
- deployable_calibration_report.py
- mnar_family_report.py
- parents
- remasker_offmcar_report.py
- impprobe_arms_v1_preregistered.py
- fold_metrics
- run_gap_token_replication_queue.sh
- confirmation_report.py
- backfill_strong_tier.py
- meeting_figures.py
- store_merge.py
- test_backfill_lgbm_baseline.py
- test_missforest.py
- run_selection_replication_queue.sh
- generate_mar_family.py
- ADR 0036: A large batch and a longer decode on pendigits and letter at 20 and 40nan
- backfill_decode_calibration_tag.py
- retune_report.py
- run_architecture_queue.sh
- model_for
- mar_family_report.py
- The decode stage's levers against washing out the encoder (E50; pre-registered 2026-10-19)
- build_mimic.py
- 02-equivalence.md
- run_from_namespace
- deep_baselines.py
- report_common.py
- 08-deep-baselines.md
- architecture_size_report.py
- decode_metric_run.py
- mnar_bias_report.py
- Does ADR 0018's bigger encoder help off MCAR at 20nan? (E121; pre-registered 2026-11-03)
- Choosing the decode checkpoint by a distributional score (E145; pre-registered 2026-11-05)
- 62-hybrid-fresh-seeds.md
- c15_stage0.py
- checkpoint_report.py
- 3. Hypotheses
- classcond_run.py
- test_generate_mnar.py
- Proposal: multiple imputation with draws joint across a row's gaps (would be E179, T56; drafted 2026-11-13 20:24 GMT-3)
- mfreplay_arms.py
- occlusion_spread.py
- mcar_draws_report.py
- mimic_mortality.py
- distribution_scores
- tabpfn_report.py
- electricity_report.py
- make_mimic_variants.py
- cross_machine.py
- stage4_trident.py
- backfill_host_tag.py
- Plan: improve the imputation experiment protocol and baselines (2026-10-17)
- calibration_report.py
- mar_smoke_report.py
- test_backfill_strong_tier.py
- test_report_common.py
- categorical_deficit.py
- mnar_smoke_report.py
- backfill_decode_mask_weighting_tag.py
- backfill_embedding_lookup_tag.py
- Does pre-training help classification under the upstream code's defaults? (E118; pre-registered 2026-11-03)
- TRIDENT on 10,000 rows of `mimic-wide` at the full table's optimizer steps: is the capped arm's loss rows or steps? (E180, T57; pre-registered 2026-11-14 19:39 GMT-3)
- mar_tuned_bar_report.py
- BACKLOG.md
- TRIDENT against MLP and SAINT under MAR and MNAR at 60nan (E115, descriptive; pre-registered 2026-11-03 00:53 GMT-3)
- An equal, validation-only tuning budget for the baselines (E42; pre-registered 2026-10-16)
- Confirmation on the 15 unused variants (E44; pre-registered 2026-10-16)
- stage4_tally.py
- backfill_decode_checkpoint_tag.py
- backfill_pretrain_gap_token_tag.py
- C15 Stage 2: MICE without an indicator (A4), the complete and MAR tables, H2-c2, H2.2 and H2b on development (E125; pre-registered 2026-11-04)
- Does pre-training help classification with ten folds and five seeds? (E87; pre-registered 2026-10-28)
- Fills and gap patterns judged by a mortality classifier (E150; pre-registered 2026-11-05 as E152)
- electricity_split.py
- barcheck_verdicts.py
- Linear baselines and the missing per-fold baselines, on the decoder's cells (E41; pre-registered 2026-10-16)
- The full MNAR family: spambase, the 20nan rung and biodeg's tilt dial (E46; pre-registered 2026-10-17)
- Design (fixed before launch)
- Stages
- What the imputation studies support, as of 2026-11-05 (consolidated)
- collapse_probes_report.py
- Services
- 0028-tabpfn-joins-the-strong-tier-as-an-offline-member.md
- test_backfill_decode_encoder_tags.py
- A deployable calibration: fitted on the validation mask, read with H (E47; pre-registered 2026-10-18)
- An MNAR smoke family: value-dependent missingness at the MCAR counts (E45; pre-registered 2026-10-17)
- test_backfill_decode_mask_weighting_tag.py
- run_retune_queue.sh
- steps_rule.py
- test_backfill_embedding_lookup_tag.py
- Path
- Let an imputation study rank its trials by the validation split's induced gaps
- test_backfill_encoder_final_norm_tag.py
- A deployable decode checkpoint (E43; pre-registered 2026-10-16)
- Does the re-masking spread still rank errors under MNAR and MAR? (E84; pre-registered 2026-10-27)
- ReMasker as an external deep baseline (E157, T53; pre-registered 2026-11-06)
- ADR 0031 decision 6's hybrid on the MIMIC-IV tables with natural gaps (E173, T55; pre-registered 2026-11-10 16:45 GMT-3)
- decode_curves_report.py
- 05. Data support for evaluation
- 03-primary.md
- barcheck_tally.py
- H2 behind an external gate: A1 against A2 on fresh seeds, on P's fresh draws and on MIMIC-IV's natural tables (E155, T51; pre-registered 2026-11-06)
- Does A1 close on A2 where the gap depends on the outcome, not on the value? MIMIC-IV complete tables with known mechanisms (E156, T52; pre-registered 2026-11-06)
- TabPFN v2's classifier on E87's ten folds, plain and class-balanced (E92; pre-registered 2026-10-28)
- Deep baselines for classification: does TRIDENT generally beat MLP, SAINT and TabNet? (E99; pre-registered 2026-10-31 21:34 GMT-3)
- Does an EMA target make classification pre-training help? (E112, T13; pre-registered 2026-11-02 10:17 GMT-3)
- backfill_pretrain_objective_tag.py
- 2. Workstreams
- ReMasker on letter, the paper's rule (E159, T53; pre-registered 2026-11-07 21:43 GMT-3)
- ReMasker off MCAR: the MAR and MNAR families (E160, T53; pre-registered 2026-11-07 21:46 GMT-3)
- ADR 0031 decision 6's hybrid at the thesis seeds, on the 26 MCAR variants (E174, T54; pre-registered 2026-11-10 16:40 GMT-3)
- A MIMIC-IV table past TabPFN's context: does the decoder hold against TabPFN v2 there? (E176, T57; pre-registered 2026-11-12 23:49 GMT-3)
- run_adr36_thesis_queue.sh
- run_blend_weights_queue.sh
- run_calibration_queue.sh
- run_checkpoint_queue.sh
- run_cls_ema_queue.sh
- run_cls_tenfold_queue.sh
- run_confirmation_queue.sh
- run_default_refresh_queue.sh
- run_deployable_calibration_queue.sh
- run_deployable_hybrid_queue.sh
- run_e111_trident_queue.sh
- run_e115_trident_queue.sh
- run_encoder_levers_queue.sh
- run_frozen_encoder_queue.sh
- run_hybrid_confirmation_queue.sh
- run_hybrid_queue.sh
- run_imputer_utility_chain_gorgona8.sh
- run_large_encoder_queue.sh
- run_mar_family_queue.sh
- run_mar_propensity_queue.sh
- run_mar_smoke_queue.sh
- run_mcar_draws_queue.sh
- run_mnar_family_queue.sh
- run_mnar_smoke_queue.sh
- run_notmiwae_fill_queue.sh
- run_null_gaps_queue.sh
- run_occlusion_queue.sh
- run_stage4_helper.sh
- run_stage4_queue.sh
- run_stage4_sweep.sh
- run_stage4_variants.sh
- run_vae_mnar_queue.sh
- make_mimic_complete.py
- make_mimic_numeric.py
- Remove the promoted imputation files that lose to the defaults
- Hand the tracking store to a remote server whole, one writer at a time
- An offline secondary metric panel; impute_score stays the only primary
- C15: `[NULL]` tokens against imputation + indicator under informative missingness (plan, 2026-11-03)
- Quick wins and methodology review, 2026-10-16
- stats_reanalyse.py
- C15 Stage 1: gate G1 on the positive controls, and A1 against A2 on the development set (E124; pre-registered 2026-11-04)
- Thesis table with TabPFN v2 in the strong tier (built by script, 2026-10-21, 17:18 GMT-3)
- Does a value target or a normalised embedding target make pre-training useful? (pre-registered 2026-09-30)
- A test-time re-masking ensemble: does averaging passes help, and does their spread rank the errors? (E75; pre-registered 2026-10-24)
- A steps-aware training batch on the small tables (E100, T14; pre-registered 2026-10-30 17:31 GMT-3)
- The thesis tables' cells that ADR 0036 changed, rerun on today's default (2026-11-01 16:44 GMT-3)
- Intervals calibrated per bin of a row's gaps (E151, T43; pre-registered 2026-11-06)
- test_backfill_decode_metric_tag.py
- Datasets read from outside the checkout (T50, stage 0; written 2026-11-05)
- Two tables from the kg-impute cohort (T50, stage 0; written 2026-11-05)
- ReMasker on the MIMIC-IV tables with natural gaps (E161, T53; pre-registered 2026-11-08 10:50 GMT-3)
- ADR 0031: An opt-in offline hybrid output: KC and TabPFN, per column
- c15_queue.sh
- metric_panel_report.py
- run_cls_stage2_queue.sh
- run_cross_machine_e94_queue.sh
- run_cross_machine_queue.sh
- run_deep_baselines_queue.sh
- run_gap_reliance_queue.sh
- run_hybrid_selection_rules_queue.sh
- run_large_batch_queue.sh
- run_remask_queue.sh
- run_spread_40nan_queue.sh
- run_spread_mechanisms_queue.sh
- run_steps_rule_queue.sh
- run_tabpfn_tenfold_queue.sh
- ADR 0030: An opt-in per-cell uncertainty signal from re-masking passes
- Score baseline imputers on exactly the cells the decoder is scored on
- build.py
- fair_probe_verdicts.py
- Ticket 0005: Batched MLflow Metric Writes (Behavior-Preserving)
- meansonlyskeptic_probe.py
- C15 Stage 0: the driving columns and gate G2, which variants carry informative missingness? (E122; pre-registered 2026-11-03)
- Does the occlusion map hold at 20nan and 80nan? (E85; pre-registered 2026-10-27)
- 04-channel-probe.md
- The EMA target at 40 and 60nan on fresh seeds (E114; pre-registered 2026-11-02 22:37 GMT-3)
- Does stopping the decode stage on a validation plateau lose anything? (pre-registered 2026-09-30)
- Does the occlusion map hold under MNAR and MAR? (E88; pre-registered 2026-10-28)
- Does a larger batch with a scaled learning rate keep M's scores? (pre-registered 2026-09-30)
- Does showing the gaps as [MASK] to pre-training help too? (pre-registered 2026-10-05)
- Does the re-masking spread make intervals narrower at the same coverage? (E77; pre-registered 2026-10-25)
- null_gaps_report.py
- evaluation_mask
- Does the re-masking spread still rank errors under MNAR and MAR at 40nan? (E105; pre-registered 2026-10-31 10:16 GMT-3)
- Does the re-masking spread still rank errors under MNAR and MAR at 20nan? (E110; pre-registered 2026-11-01 23:59 GMT-3)
- TRIDENT against its baselines on `impute_score_classcond`, the score measured against the class-conditional fill (E181, T58; pre-registered 2026-11-17 14:35 GMT-3)
- Real gap patterns transplanted onto complete rows (E149; pre-registered 2026-11-05 as E151)
- C15's positive-control gate on the SOFA table's natural gaps (E152; pre-registered 2026-11-06)
- credit-g_80nan ranked by the induced validation gaps (pre-registered 2026-09-28)
- Overnight full-profile studies against the baseline bar (pre-registered 2026-09-28)
- The induced validation objective on the other six variants (pre-registered 2026-09-28)
- Does normalised-target pre-training still help once the decoder gets 450 epochs? (pre-registered 2026-09-30)
- run_e155_chain.sh
- run_e156_chain.sh
- run_imputer_utility_chain.sh
- budgetskeptic_ci.py
- dupskeptic_twinsplit.py
- metricskeptic_e35.py
- C15 Stage 0, second gate: G2 read out of fold over every training and validation row (E123; pre-registered 2026-11-03)
- TabPFN v2 as an external foundation-model baseline (E55; pre-registered 2026-10-20)
- C15 Stage 3: confirmation, read once: A1 against A2 on fresh draws of P and of its MCAR siblings (E126; pre-registered 2026-11-04)
- C15's gate G1 again: A2 against A3 in decision mode (E128; pre-registered 2026-11-04)
- plan_run
- Let pre-training regress a masked cell onto its value, or onto a normalised embedding
- Two tiers of baseline imputers, every one logged, each tier with its own best
- 01. Per-task fold-ranking contract
- 11. Regression fixture and documentation
- Does a decoder that can tell a real gap from its own masks read the gap pattern under MNAR? (E81; pre-registered 2026-10-27)
- Today's default on fresh seeds at 20nan (E51; pre-registered 2026-10-19)
- The strong tier with TabPFN on the seed sets E55 did not cover (E58; pre-registered 2026-10-21)
- frozen_encoder_run.py
- Read every imputation comparison one way: verdict tallies per opponent, the test named
- A validation-only tuning budget for the baselines, co-reported offline
- Thesis table on the development set (built by script, 2026-10-17; regenerated 2026-10-18 23:09 GMT-3)
- Observed cells hidden on tables with natural gaps (E148; pre-registered 2026-11-05 as E150)
- C15 Stage 4 (optional, secondary): A4 on the dial, the sign arm, A3 and the null model A5 (E127; pre-registered 2026-11-04)
- ADR 0030's binned interval under real gap patterns (E153; pre-registered 2026-11-06)
- ADR 0030's binned interval on the natural gaps' rows (E154; pre-registered 2026-11-06)
- ADR 0018's bigger encoder on the large tables at 40–80nan (E59; pre-registered 2026-10-21)
- Map: Reduced Optuna search for the imputation task
- imputer_utility_levels.py
- run_mimic_chain.sh
- run_sofa_g1_chain.sh
- run_tabpfn_lanes.py
- test_experimentation_log_index.py
- Experimentation log
- v1.build.py
- cvdepskeptic_mixed.py
- mfreplay_report.py
- Where does K lose to TabPFN on kr-vs-kp and credit-g? A per-column read (E61; pre-registered 2026-10-22)
- c15_chain.sh
- results_digest.py
- run_macode_lanes.py
- run_remasker_lanes.py
- tables_manifest.sh
- barcheck_e39mf.py
- barcheck_inventory.py
- c15_chain4.sh
- c1_chain.sh
- c1_diag_extract.py
- c1_diag_plot.py
- h2_chain.sh
- h2c_chain.sh
- experiments/__init__.py
- run_architecture_size_queue.sh
- mlflow_backend
- Fresh MCAR draws: is a result tied to one missingness pattern? (E62; pre-registered 2026-10-22)
- The gaps as `[NULL]` in the decode stage, alone: does the token help under MNAR? (E67; pre-registered 2026-10-23)
- The tuned bar on the MAR folds: D4 under MAR (E68; pre-registered 2026-10-23)
- 02. Research: how do tabular masked-modeling transformers decode a masked cell back to a value?
- Per-column weights for the K + TabPFN blend, cross-fitted: do they reach past TabPFN where the equal blend could not? (E69; pre-registered 2026-10-23)
- Does the spread grow when a cell's top sources are missing? (E80; pre-registered 2026-10-27)
- Reduced Optuna search for imputation: implementation tickets
- An imputer that models the mechanism: does not-MIWAE shrink the MNAR bias? (E72; pre-registered 2026-10-24)
- not-MIWAE with a mean-based fill: does modelling the mechanism shrink the MNAR bias on biodeg and spambase? (E73; pre-registered 2026-10-24)
- Does the re-masking spread rank cells on the development variants? (E76; pre-registered 2026-10-25)
- Can the hybrid's weights be chosen without the complete table? (E82; pre-registered 2026-10-27)
- 36-hybrid-20nan.md
- Does the cross-fit pick remove the hybrid's weakness off MCAR? (E170, T54; pre-registered 2026-11-09 19:59 GMT-3)
- 04. Decoder architecture and the stage that trains it
- The deployable hybrid on the MIMIC-IV transplant variants (E163, T55; pre-registered 2026-11-08 21:08 GMT-3)
- 06. How are imputation runs identified in MLflow?
- 07. Prototype: the human-friendly imputed-vs-actual preview
- 08. CLI flag and hyperparameter schema for the imputation task
- 09. How does the training package branch per task without touching classification?
- TRIDENT on 10,000 rows of `mimic-wide`: does the lead over TabPFN v2 need the other rows? (E177, T57; pre-registered 2026-11-13 08:28 GMT-3)
- T29 · Why the boosters win on electricity
- TabPFN v2 under informative missingness: native and explicit-mask arms beside A1 (E137, T44; pre-registered 2026-11-04)
- H2 under outcome-dependent MNAR: TabPFN and A1 on F3 against same-seed MCAR siblings (E139; pre-registered 2026-11-04)
- 01. The held set and the ranges
- 02. The search objective is scored on the validation split
- 03. Study protocol: datasets, variants, budget, schedule, seed
- 04. Where a promoted configuration lives and how a run finds it
- 05. The search-space profile flag and its record
- 06. Checking the reduction and proving the tuning helped
- 07. Task: repair the imputation Optuna path and prove it with a two-trial study
- Does pre-training open a channel when labels are scarce? S, T and B on a tenth and a quarter of the training labels (E136; pre-registered 2026-11-04)
- The quarter-label channel on fresh seeds, with the validation split scarce too (E138; pre-registered 2026-11-04)
- log_param_importances
- The selector × seeds squares: is the scarce-validation selector what opens a tenth? (E144; pre-registered 2026-11-05)
- Imputation decoder task: implementation tickets
- The default output against TabPFN: KC and H (E57; pre-registered 2026-10-21)
- Are TRIDENT's and TabPFN's errors complementary? A blend of their fills (E60; pre-registered 2026-10-21)
- A MAR smoke against matched MCAR controls (E63; pre-registered 2026-10-22)
- The MAR family: does MAR cost the model ground against the strong tier? (E64; pre-registered 2026-10-22)
- Does E60's complementarity hold on other tables and seeds? The blend on E40's set (E66; pre-registered 2026-10-23)
- Can fewer or better-fed choices close the deployable hybrid's gap to TabPFN? (E89; pre-registered 2026-10-28)
- Does the deployable hybrid's `table` rule hold on fresh seeds, and where it is not needed? (E90, E91; pre-registered 2026-10-28)
- Does the deployable `table` hybrid hold under MNAR and MAR? (E93; pre-registered 2026-10-29)
- Which score orders imputers as a classifier does, across TRIDENT, the strong baselines and three mechanisms? (E147, a pilot; pre-registered 2026-11-05 as E149)
- The cross-fit side pick on E93's off-MCAR variants, fresh seeds (E172, T54; pre-registered 2026-11-09 21:18 GMT-3)
- task_defaults
- The label-fraction curve under a scarce validation split: a tenth and a half, fresh seeds (E140; pre-registered 2026-11-05)
- Stage 2: the quarter-label channel on the 21 confirmatory variants, five seeds (E141; pre-registered 2026-11-05)
- The quarter-label channel at 20, 40 and 80% missingness (E143; pre-registered 2026-11-05)
- Which fills serve a downstream classifier: the checkpoint by `impute_score` or by `dist_score`? (E146, a pilot; pre-registered 2026-11-05)
- _BudgetSpent
- _matrix

## God Nodes (most connected - your core abstractions)
1. `Entries` - 184 edges
2. `main()` - 111 edges
3. `DatasetSpec` - 74 edges
4. `build_folds()` - 70 edges
5. `mirror_run_tree()` - 70 edges
6. `Hyperparameters` - 70 edges
7. `mean_mode_baselines()` - 66 edges
8. `prepare_dataset()` - 64 edges
9. `resolve_training_request()` - 63 edges
10. `PreparedDataset` - 60 edges

## Surprising Connections (you probably didn't know these)
- `TRIDENT README` --conceptually_related_to--> `src/training Package`  [AMBIGUOUS]
  README.md → AGENTS.md
- `compute_cv_summary()` --semantically_similar_to--> `CrossValidationSummary`  [INFERRED] [semantically similar]
  README.md → docs/superpowers/plans/2026-08-03-curated-mlflow-cross-validation.md
- `TRIDENT README` --conceptually_related_to--> `kr-vs-kp Categorical Columns`  [AMBIGUOUS]
  README.md → datasets/categorical_columns/kr-vs-kp.txt
- `hourly()` --calls--> `events()`  [INFERRED]
  datasets/build_mimic.py → tests/unit/test_leadership.py
- `_NumericFill` --uses--> `MeanModeImputer`  [INFERRED]
  docs/reviews/2026-10-16-quick-wins/scripts/impprobe_arms.py → src/training/imputation_baselines.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Curated MLflow Cross-Validation Documentation Trail** — docs_adr_0002_curated_cross_validation_mlflow_runs, docs_superpowers_specs_2026_08_03_curated_mlflow_cross_validation_design, docs_superpowers_plans_2026_08_03_curated_mlflow_cross_validation, docs_tickets_0002_mlflow_logged_model_lifecycle [INFERRED 0.85]
- **Training Refactor Documentation Trail** — docs_tickets_0001_training_refactor, docs_superpowers_specs_2026_08_02_training_refactor_design, docs_superpowers_plans_2026_08_02_training_refactor, docs_adr_0001_training_package_with_compatibility_facade, _superpowers_sdd_2026_08_02_training_refactor_task_3_report [INFERRED 0.85]
- **TRIDENT Model Architecture Pipeline** — readme_tabularembedder, readme_tabulartransformerencoder, readme_tridentpretrainer, readme_tridentclassifier [INFERRED 0.85]

## Communities (533 total, 41 thin omitted)

### Community 0 - "MultiHeadAttention"
Cohesion: 0.17
Nodes (8): EncoderLayer, FeedForwardNetwork, initialize_weight(), MultiHeadAttention, Simple feed-forward network with two linear layers and ReLU activation, Multi-head attention layer for transformers, Initializes the weight of a layer using Xavier uniform for weights and zeros…, Transformer encoder layer with self-attention and feed-forward

### Community 1 - "test_opt_search_space.py"
Cohesion: 0.08
Nodes (30): build_parser(), define_search_space(), ObjectiveFunctionWrapper, ArgumentParser, Path, Wrapper class for the Optuna objective function to maintain state, Save the best parameters found so far, inside the study's own directory. The…, The training parser, so a study launched here accepts every flag main.py… (+22 more)

### Community 2 - "0005-reduced-optuna-search-for-imputation.md"
Cohesion: 0.23
Nodes (4): 08. Record the design: ADR 0005 and the implementation plan, Answer, Comments, Question

### Community 3 - "Hyperparameters"
Cohesion: 0.03
Nodes (146): ADR 0036's thesis cells: today's default on the four variants it changed, at…, train(), C15 Stage 1's readers (docs/tickets/c15-mnar-prediction/03-stage1.md): gate G1…, cell_dir(), done(), machine(), Path, C15 Stage 1 cells (docs/tickets/c15-mnar-prediction/03-stage1.md), MLflow off,… (+138 more)

### Community 4 - "Answer"
Cohesion: 0.22
Nodes (8): 02. The checker, the strictness ramp and the gate, Answer, Checker: mypy, Question, Scope: all of `src/` plus the entry points, Strictness: `strict = true` with per-module overrides, The baseline this rests on, The gate: a unit test that shells out to mypy

### Community 5 - "test_mirroring.py"
Cohesion: 0.14
Nodes (34): _artifact_paths(), _bare_run(), _dataset(), _history(), fixture, MlflowClient, Path, Run (+26 more)

### Community 6 - "TRIDENT Experiment Tracking Context (CONTEXT.md)"
Cohesion: 0.26
Nodes (14): TRIDENT Experiment Tracking Context (CONTEXT.md), Component-Ablation Benchmark, CV Summary, Dataset Provenance, Diagnostic Fold Run, Experiment Family, Fold-Ranking Metric, Full-Factorial Ablation (+6 more)

### Community 7 - "3. Design and methodology critiques judged sound"
Cohesion: 0.17
Nodes (12): 3.10 The documentation contract, 3.11 Residual leaks and units, 3.1 The evaluation mask is the training corruption helper, and one ADR sentence rests on it, 3.2 The two scored populations are not a pair, and the induced view is out of distribution, 3.3 The search cannot identify a winner, and the winner it publishes is selection-biased, 3.4 Checkpoint selection is a different objective from the one that ranks the fold, 3.5 What `impute_score` normalises, and what the record keeps of it, 3.6 Pre-training is not measurably contributing, and nothing in the record could say so (+4 more)

### Community 8 - "TRIDENT Architecture"
Cohesion: 0.08
Nodes (26): Building folds — `build_folds` (`src/training/data.py:69`), Contents, Data engineering pipeline, End-to-end training orchestration, Hyperparameter → code wiring reference, Implementation, Implementation, Implementation (+18 more)

### Community 9 - "Answer"
Cohesion: 0.25
Nodes (8): 01. The Python floor moves to 3.11, Acceptance for the bump, Answer, Question, What 3.11 buys, concretely, What was verified, and what was not, Why not further, Why the floor mattered more than it looks

### Community 10 - "generate_splits.py"
Cohesion: 0.12
Nodes (23): main(), DataFrame, Fresh MCAR draws of a variant, from its complete table, with the shipped…, The frame as a reader of its CSV sees it, so it compares with a shipped file., _roundtrip(), _same(), seed_of(), _create_split_dict() (+15 more)

### Community 11 - "StoreError"
Cohesion: 0.11
Nodes (45): Run a step of this script in the peer's checkout: yield, backup, rejoin., adopt(), copy_store(), default_master(), find_rsync(), journal_mode(), main(), open_store() (+37 more)

### Community 12 - "Answer"
Cohesion: 0.25
Nodes (8): 03. Five StrEnums, and where constants live, Answer, Decision: homes split keys from values, Decision: `StrEnum`, with the tuples derived, Domain-modeling note: no CONTEXT.md changes, Implementation cautions for the execution tickets, Question, What the diagnosis actually was

### Community 13 - "selection_replication_report.py"
Cohesion: 0.27
Nodes (14): _best_baseline(), folds(), integrity(), interval(), _path(), DataFrame, ndarray, Pre-registered analysis of the checkpoint replication on E32's seeds… (+6 more)

### Community 15 - "tabpfn_seed_sets_report.py"
Cohesion: 0.33
Nodes (11): cells(), main(), _merge(), mnar(), per_seed(), DataFrame, ndarray, Path (+3 more)

### Community 17 - "Which columns does the decoder fill each column from? (E78; pre-registered 2026-10-26)"
Cohesion: 0.17
Nodes (10): Design (fixed before the first cell), Limits fixed now, Measures and reading, fixed now (`occlusion_map.py --tally`), Outcome (2026-10-27, 03:14 GMT-3): supported, The method, Which columns does the decoder fill each column from? (E78; pre-registered 2026-10-26), Design (fixed before the first cell), Does the occlusion map hold at 60nan? (E79; pre-registered 2026-10-26) (+2 more)

### Community 18 - "imputation_metrics.py"
Cohesion: 0.09
Nodes (36): _class_conditional_scores(), ``<owner>/impute_score_classcond`` for the model scored under ``prefix`` and…, class_conditional_score(), class_mean_mode_baselines(), class_naive_errors(), _error_metrics(), DataFrame, Imputation error, and the score that ranks folds by it. Pure functions over a… (+28 more)

### Community 19 - "test_training_schedulers.py"
Cohesion: 0.09
Nodes (25): Optimizer, The rows a stage's model trains on: the fold's training rows, or a fixed-seed…, Linear warmup to the base rate, then cosine decay to zero, measured in steps.…, One training stage's learning-rate schedule behind a uniform hook interface., StageScheduler, training_rows(), validate_lr_scheduler_name(), WarmupCosineMultiplier (+17 more)

### Community 20 - "TRIDENT Backlog: Bugs, Pendencies and Improvements"
Cohesion: 0.07
Nodes (27): B1. The learning-rate schedule completes a full cosine cycle — protected, B2. `--cv_folds 1` crashes, B3. Optuna proposes head counts that crash, and scores them as 0.0, B4. Optuna with MLflow enabled scored every trial 0.0, Bugs, C1. Scaler and encoders are fit before splitting — protected, C2. `"nan"` is a real category next to `[NULL]` — protected, C3. `LABELS` is logged but never used (+19 more)

### Community 21 - "mean_mode_baselines"
Cohesion: 0.08
Nodes (55): KnnImputer, The stronger baseline: each hidden cell filled from its nearest training rows.…, Each baseline imputer's error on the scored cells, keyed…, score_baselines(), mean_mode_baselines(), What a naive imputer would fill each column's cells with. The mean of a…, _cells(), _dependent_frame() (+47 more)

### Community 22 - "test_training_tracking.py"
Cohesion: 0.12
Nodes (50): get_or_create_experiment(), mirror_experiment_name(), Return the experiment_id for the given dataset, creating it if absent.…, The mirror experiment for a task, e.g. ``TRIDENT/mirror/imputation``., create_tracker(), Return an MLflow-backed tracker only when tracking has been requested., Disabled tracking must keep every MLflow API completely untouched., test_disabled_tracker_never_calls_mlflow_apis() (+42 more)

### Community 23 - "test_c15_inputs.py"
Cohesion: 0.27
Nodes (14): blocked_position(), fold_dataset(), has_indicator(), C15's arms (docs/plans/2026-11-03-c15-mnar-prediction.md): what a…, A5's blocked sequence position: 1 for [CLS] plus the column's token index; None…, ``dataset`` as this fold's arm shows it: A1 unchanged, the others filled from…, _dataset(), _fold() (+6 more)

### Community 24 - "decoding.py"
Cohesion: 0.13
Nodes (28): apply_calibration(), Calibration, fit_calibration(), DataFrame, Calibrating the decoder's guesses on the validation split's own gaps. A…, Per column: the shrinking factor of a numerical one, the confidence threshold…, The same cells with calibrated guesses; a column the calibration never saw is…, _threshold() (+20 more)

### Community 25 - "linear_baselines.py"
Cohesion: 0.06
Nodes (49): _factory(), main(), E41's offline pass: every baseline, the linear arms included, on one variant's…, ConditionalExpectationImputer, fit_linear_imputers(), IceImputer, _Labels, _NumericalFill (+41 more)

### Community 26 - "backfill"
Cohesion: 0.12
Nodes (23): backfill(), BackfillReport, iter_all_runs(), main(), MlflowClient, Run, Stamp a tag onto MLflow runs recorded before that tag existed. A tag that only…, Every run in every experiment, including deleted ones, as (experiment name,… (+15 more)

### Community 27 - "Findings"
Cohesion: 0.13
Nodes (14): 07 - The Optuna search for imputation, Checked and cleared, F-07-10 - The study's running-best file uses classification's promoted filename and holds only the sampled subset, F-07-1 - The search objective cannot resolve the configurations it is ranking; two of the four promoted files were decided by a tie-break, not by the objective, F-07-2 - The objective is measured on exactly the cells that chose the checkpoint, so `optuna/best_objective_value` is a minimum-of-minima, not a validation score, F-07-3 - There is no way to run the defaults arm of ADR 0005's own comparison protocol once a configuration has been promoted, F-07-4 - An imputation run with no promoted file of its own silently loads classification's tuned configuration, F-07-5 - fANOVA silently drops `HEAD_DIM` from the importance ranking, so the check on the reduction cannot see the model width (+6 more)

### Community 28 - "backfill_lr_scheduler_tag.py"
Cohesion: 0.16
Nodes (17): backfill(), BackfillReport, iter_all_runs(), main(), MlflowClient, Run, Backfill the ``lr_scheduler`` tag on MLflow runs recorded before ADR 0003.…, Every run in every experiment, including deleted ones, as (experiment name,… (+9 more)

### Community 29 - "Reduced Optuna Search for Imputation: Implementation Plan"
Cohesion: 0.22
Nodes (9): File structure, Global Constraints, Reduced Optuna Search for Imputation: Implementation Plan, Task 1: Search-space profiles, Task 2: The search objective on the validation split, Task 3: Task-keyed lookup and explicit promotion, Task 4: The importance artifact, Task 5: The launcher and the six studies (+1 more)

### Community 30 - "preprocess_table"
Cohesion: 0.08
Nodes (32): EpochMasker, ndarray, The frame with these cells hidden, as ``draw`` hides the cells it draws.…, Reduce a column name to a valid ModuleDict/ParameterDict key., An epoch's fresh mask over a frame encoded once, without going back through…, _sanitize(), create_pretrain_datasets(), draw_dynamic_mask() (+24 more)

### Community 31 - "test_training_decoding.py"
Cohesion: 0.06
Nodes (81): _complete_frame(), _dataset(), _encoder_state(), _fold(), _hyperparameters(), _pretrained(), DataFrame, parametrize (+73 more)

### Community 32 - "mirror_run_tree"
Cohesion: 0.14
Nodes (26): describe(), main(), Mirror every run in every experiment family into ``TRIDENT/mirror/<task>`` (ADR…, is_mirror_experiment(), Whether an experiment holds mirror runs, so it is never mirrored itself., children_of(), iter_runs(), metric() (+18 more)

### Community 33 - "interval"
Cohesion: 0.06
Nodes (71): interval(), ndarray, Per fold, (macro-F1, unweighted log loss) of the test rows., scores(), tally(), anchor(), arms_against_s(), budget() (+63 more)

### Community 34 - "test_prune_cache.py"
Cohesion: 0.19
Nodes (12): main(), prune(), PruneReport, Path, Keep a result cache under a size limit, dropping the entries used longest ago.…, _entry(), Path, Keeping a result cache under its size limit by dropping what was used longest… (+4 more)

### Community 35 - "spread_mechanisms.py"
Cohesion: 0.14
Nodes (20): _ledger_path(), _per_seed(), ndarray, Path, E105: does the re-masking spread still rank errors under MNAR and MAR at 40nan?…, Largest tables first, each cell to the least-loaded queue, as E84 dealt them., tally(), todo() (+12 more)

### Community 36 - "run_hyperparameter_optimization"
Cohesion: 0.12
Nodes (47): Namespace, Run hyperparameter optimization with Optuna Annotated ahead of the rest of this…, run_hyperparameter_optimization(), _optuna_args(), MlflowClient, Namespace, Run, MLflow structure of an Optuna study: one study run, one light run per trial. (+39 more)

### Community 37 - "backfill_baseline_imputers.py"
Cohesion: 0.13
Nodes (29): FoldMetrics, apply_plan(), _arguments(), candidate_runs(), _children(), _dataset_bytes(), _fold_rows(), _fold_value() (+21 more)

### Community 38 - "Imputation Decoder Task Implementation Plan"
Cohesion: 0.14
Nodes (14): File structure, Global Constraints, Imputation Decoder Task Implementation Plan, Task 10: Optuna, Task 11: Regression fixture, documentation, complete verification, Task 1: Per-task ranking contract, classification byte-identical, Task 2: Request, hyperparameters and command line, Task 3: Imputation metrics as pure functions (+6 more)

### Community 39 - "Resolution"
Cohesion: 0.14
Nodes (14): 08. Configure the checker and make it green, Acceptance, Acceptance, Left for ticket 09, Notes, Question, Resolution, The bump: verified, not assumed (+6 more)

### Community 41 - "imputation_baselines.py"
Cohesion: 0.07
Nodes (45): make(), _Constant, GradientBoostingImputer, Baseline imputers, scored on exactly the cells the decoder is scored on. Four…, A column with nothing to learn from: every fill is the same value., A learned baseline: one gradient-boosting model per column, in a single pass.…, LightGbmImputer, bool_ (+37 more)

### Community 42 - "Answer"
Cohesion: 0.29
Nodes (7): 04. The task-branch union and exhaustiveness, Answer, Question, Related, and deliberately not bundled, The fix, Why the `Protocol` is out of scope, Why this one error earns its own ticket

### Community 43 - "05. Third-party stubs for sklearn and scipy"
Cohesion: 0.29
Nodes (7): 05. Third-party stubs for sklearn and scipy, Notes, Question, Resolution, The three candidate answers, Three things ticket 08 must not rediscover the hard way, What to find out

### Community 44 - "complete_configuration"
Cohesion: 0.07
Nodes (40): promote_best_configuration(), Publish a study's winning configuration where the task's runs will find it. The…, main(), One cell of the calibration confirmation at 40% missing and above (imputation-…, main(), One cell of the deployable-checkpoint study (imputation-protocol/03, E43). uv…, T06 feasibility: a short real run, eager against ``torch.compile``, timed…, main() (+32 more)

### Community 45 - "test_lightgbm_baseline.py"
Cohesion: 0.24
Nodes (14): _imputer(), DataFrame, The single-pass LightGBM baseline, ``lgbm``: the hgb recipe with LightGBM (ADR…, ``level`` jumps from -3 to 3 where ``x`` passes 10, and ``side`` follows the…, The hgb test, with LightGBM. Worked by hand: at ``x = 2`` the rule says ``level…, The cache keys an entry on the machine, not its cores: two fits of one fold…, The check missForest's threaded forests failed (ADR 0017): on a noisy target…, _step_frame() (+6 more)

### Community 46 - "Findings"
Cohesion: 0.13
Nodes (14): 08 - Experimental design of the launchers, Checked and cleared, F-08-10 - `experiment_imputation.ps1` takes its backups and writes the config outside the `try`, so a mid-setup failure leaves a clobbered file with no restore, F-08-1 - The committed `experiment_imputation.ps1` is a byte-for-byte copy of `experiment.ps1` and runs classification, not imputation, F-08-2 - `imputation_studies.ps1` hardcodes `--search_space reduced`, so the experiment cannot test the decision it exists to test, F-08-3 - The studies never cross-validate: hyperparameters are chosen on one fixed validation split whose rows then make up 8.5-13% of every CV test fold they are evaluated on, F-08-4 - The twelve comparison runs cannot support a generalisation claim: one seed, one missingness draw, unpaired t-intervals over five correlated folds, six of them read at once, F-08-5 - The search optimises the masked proxy; the comparison is judged on the induced population, and the study's winner is not the best trial on it (+6 more)

### Community 48 - "02. Request, hyperparameters and command line"
Cohesion: 0.25
Nodes (7): 02. Request, hyperparameters and command line, Acceptance criteria, Comments, Constraints, Files, Goal, Seams under test

### Community 49 - "Reduce the imputation search to the knobs that move it, score it on the validation split, and promote its result explicitly"
Cohesion: 0.25
Nodes (8): Consequences, Considered options, Context, Decision, How to compare, Outcome (2026-09-12), Reduce the imputation search to the knobs that move it, score it on the validation split, and promote its result explicitly, Status

### Community 50 - "column_wise_report.py"
Cohesion: 0.27
Nodes (14): _best_baseline(), folds(), _folds(), integrity(), interval(), _path(), DataFrame, ndarray (+6 more)

### Community 51 - "03. Imputation metrics as pure functions"
Cohesion: 0.29
Nodes (6): 03. Imputation metrics as pure functions, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 52 - "Add an imputation task with a value decoder, keep classification the default, and tag it in MLflow"
Cohesion: 0.29
Nodes (7): Add an imputation task with a value decoder, keep classification the default, and tag it in MLflow, Consequences, Considered options, Context, Decision, How to compare, Status

### Community 53 - "ArtifactWriter"
Cohesion: 0.10
Nodes (25): ArtifactWriter, _partition(), Any, DataFrame, Path, StandardScaler, Filesystem artifacts emitted by a training runtime., Write prepared-dataset lineage for a parent training run. (+17 more)

### Community 54 - "06. The decode stage"
Cohesion: 0.29
Nodes (6): 06. The decode stage, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 55 - "07. Imputation artifacts"
Cohesion: 0.29
Nodes (6): 07. Imputation artifacts, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 56 - "08. Runner, MLflow identity, batch summary"
Cohesion: 0.29
Nodes (6): 08. Runner, MLflow identity, batch summary, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 57 - "09. Tag backfill"
Cohesion: 0.29
Nodes (6): 09. Tag backfill, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 58 - "10. Optuna for the imputation task"
Cohesion: 0.29
Nodes (6): 10. Optuna for the imputation task, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 59 - "gap_reliance.py"
Cohesion: 0.23
Nodes (11): _fills(), Any, DataFrame, device, ndarray, Path, E81: does a decoder that can tell a real gap from its own masks read the gap…, _read_in_place() (+3 more)

### Community 60 - "Map: Imputation decoder task"
Cohesion: 0.33
Nodes (6): Decisions so far, Destination, Map: Imputation decoder task, Not yet specified, Notes, Out of scope

### Community 61 - "summarize_cross_validation"
Cohesion: 0.14
Nodes (34): Build the per-fold timing events from the two measured stage durations., Aggregate completed CV folds into comparable final and loss statistics.…, stage_timing_metrics(), summarize_cross_validation(), FoldResult, LoggedMetric, A run may skip pre-training (zero epochs) to measure what it contributes. No…, Folds that disagree about whether a stage ran are not comparable, as before. (+26 more)

### Community 62 - "conformal_row_gaps.py"
Cohesion: 0.08
Nodes (40): Ledger, cells(), main(), DataFrame, ndarray, E154 (docs/tickets/mimic/08-conformal-natural-gaps.md): ADR 0030 decision 8's…, A fold's masked numerical cells with error and b, the row's natural gaps plus…, The bin (0 the fewest gaps) with the largest under-coverage of 0.90. (+32 more)

### Community 63 - "occlusion_map.py"
Cohesion: 0.10
Nodes (30): _attention(), _auroc(), _block_of(), _change(), _deletion(), _faithfulness(), _occlusion(), Any (+22 more)

### Community 64 - "DatasetSpec"
Cohesion: 0.04
Nodes (98): main(), Scoring-only: induced impute_score of the hgb recipe at a few booster settings,…, Read-only: induced impute_score of KNN baselines at k in {5,10,20,50} on each…, Read-only: model vs knn5/knn10 on twin / no-twin induced cells, from current-…, MiceImputer, main(), h2b_fold(), differences() (+90 more)

### Community 65 - "10. Record the design: ADR and implementation plan"
Cohesion: 0.40
Nodes (4): 10. Record the design: ADR and implementation plan, Answer, Comments, Question

### Community 66 - "model"
Cohesion: 0.07
Nodes (50): cells(), contrast(), _line(), Run, Pre-registered analysis of E51, today's imputation default on fresh seeds at…, ``readout`` (K, H, M or KC) minus ``opponent`` (an opponent of…, report(), _runs() (+42 more)

### Community 67 - "test_backfill_decode_checkpoint_tag.py"
Cohesion: 0.30
Nodes (11): client(), fixture, MlflowClient, Stamping the decode checkpoint criterion on imputation runs recorded before it…, Every imputation run before the flag kept the epoch of the lowest validation…, The fast path: a stamped run that already has a mirror gets the two tags on…, _seed_runs(), test_a_dry_run_writes_nothing_and_a_second_apply_finds_nothing() (+3 more)

### Community 68 - "PreparedDataset"
Cohesion: 0.06
Nodes (38): _Entry, build_fold_tags(), build_run_tags(), parse_missingness_percent(), Return the normalized percentage encoded by a ``_<n>nan`` suffix., Build a flat dict of MLflow tags for a training run. Parameters ----------…, Build tags for a per-fold child run. Parameters ---------- fold_idx: Zero-based…, chunks() (+30 more)

### Community 69 - "test_backfill_decode_calibration_tag.py"
Cohesion: 0.30
Nodes (11): client(), fixture, MlflowClient, Stamping the decode calibration on imputation runs recorded before ADR 0027…, Every imputation run before ADR 0027 scored its output uncalibrated. A parent…, The fast path: a stamped run that already has a mirror gets the two tags on…, _seed_runs(), test_a_dry_run_writes_nothing_and_a_second_apply_finds_nothing() (+3 more)

### Community 70 - "T29's test: is the boosters' lead on electricity carried by `date` as a key? (E54; pre-registered 2026-10-20)"
Cohesion: 0.40
Nodes (5): Design (fixed before the first subset), Launch, Outcome (2026-10-20, 23:16 GMT-3), Reading, fixed now, T29's test: is the boosters' lead on electricity carried by `date` as a key? (E54; pre-registered 2026-10-20)

### Community 71 - "leadership.py"
Cohesion: 0.14
Nodes (43): compiled_init(), checked_backup(), deposed_tick(), follower_tick(), guard_roots(), init(), leader_tick(), live_store() (+35 more)

### Community 72 - "architecture_report.py"
Cohesion: 0.25
Nodes (13): _best_baseline(), folds(), integrity(), interval(), _path(), DataFrame, ndarray, Pre-registered analysis of the architecture screening (imputation-… (+5 more)

### Community 73 - "CLAUDE.md"
Cohesion: 0.07
Nodes (27): TRIDENT Agent Guide (AGENTS.md), PyArrow <24 Pin Constraint, Background & full reference, Experimentation log, gorgona8: the store lives on the SSD, graphify, MLflow run analysis, Times (+19 more)

### Community 74 - "backfill_encoder_final_norm_tag.py"
Cohesion: 0.27
Nodes (10): backfill(), BackfillReport, main(), _mirrors_by_source(), _pretraining_runs(), MlflowClient, Run, Every active mirror run, by the run it mirrors. (+2 more)

### Community 75 - "confirmatory_report.py"
Cohesion: 0.24
Nodes (9): folds(), interval(), _path(), DataFrame, ndarray, Pre-registered analysis of the confirmatory study on 21 variants (imputation-…, How many variants each primary comparison calls each way., report() (+1 more)

### Community 76 - "pretrain_gap_token_report.py"
Cohesion: 0.28
Nodes (13): _best_baseline(), folds(), integrity(), interval(), _path(), DataFrame, ndarray, Pre-registered analysis of the pre-training gap-token study (imputation-token-… (+5 more)

### Community 78 - "01. Search-space profiles"
Cohesion: 0.29
Nodes (6): 01. Search-space profiles, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 79 - "Findings"
Cohesion: 0.13
Nodes (14): 09 - Test adequacy for the imputation task, Checked and cleared, F-09-10 - Tests that restate the implementation and could go, F-09-1 - The `[NULL]`-path diagnostic scores a different set of cells than it claims, and the only test that guards it asserts a permutation-invariant count, F-09-2 - No test pins `realised_rate`, and the metric it would have pinned shows `EVAL_MASK_RATE` is nearly inert above 40% missingness, F-09-3 - Nothing asserts the decoder ever beats the naive baseline, so a decoder whose outputs are unrelated to the truth passes every test including the fixture, F-09-4 - The fixture's own justification for choosing `credit-g_20nan` is false, and the dataset it wrongly excluded is the one that would have caught a shipped crash, F-09-5 - The fixture pins six pooled averages and no structural invariant, so the cheapest and most decisive checks are absent (+6 more)

### Community 80 - "Verdicts"
Cohesion: 0.14
Nodes (13): 07 - Optuna search for imputation: verdicts, F-07-10 - The study's running-best file carries classification's promoted filename and holds only the sampled subset, F-07-1 - The validation objective is too coarse to separate the configurations it ranks, and two of the four promoted files were decided by trial order rather than by the objective, F-07-2 - The checkpoint is selected on exactly the cells the objective is then scored on, so `optuna/best_objective_value` is a best-of-`EPOCHS_DECODE` minimum, not a plain validation score, F-07-3 - Once a configuration is promoted there is no way to run the defaults arm of ADR 0005 decision 6's own comparison, F-07-4 - An imputation run with no promoted file of its own loads classification's tuned configuration wholesale, F-07-5 - fANOVA silently drops `HEAD_DIM`, so the importance ranking cannot see the model width, F-07-6 - The predefined validation rows the search selects on are scored again as test rows in the 5-fold comparison (+5 more)

### Community 81 - "02. The search objective on the validation split"
Cohesion: 0.29
Nodes (6): 02. The search objective on the validation split, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 82 - "04. The importance artifact"
Cohesion: 0.29
Nodes (6): 04. The importance artifact, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 83 - "05. The launcher and the six studies"
Cohesion: 0.29
Nodes (6): 05. The launcher and the six studies, Acceptance criteria, Comments, Constraints, Goal, What the script runs

### Community 84 - "06. The comparison and the documentation"
Cohesion: 0.29
Nodes (6): 06. The comparison and the documentation, Acceptance criteria, Comments, Constraints, Goal, What runs

### Community 85 - "Verdicts"
Cohesion: 0.14
Nodes (13): 08 - Experimental design of the launchers: verdicts, F-08-10 - Backups and the config write sit outside the `try`, so a mid-setup failure leaves a clobbered file with no restore, F-08-1 - The committed `experiment_imputation.ps1` is byte-identical to `experiment.ps1`; only an uncommitted edit makes it an imputation launcher, F-08-2 - `imputation_studies.ps1` hardcodes `--search_space reduced`, so the twelve comparison runs cannot test reduced-vs-full, F-08-3 - Optuna studies never cross-validate; the config is chosen on one fixed validation split whose rows then form 8.5-13% of every CV test fold, F-08-4 - Two overlapping unpaired single-arm t-intervals over five correlated folds, one seed, one missingness draw, six pairs read at once, F-08-5 - The search ranks on the masked proxy while the comparison headlines the induced population; the search winner is not the best trial on the headline, F-08-6 - The sweep writes `<dataset>.json`, which `--task imputation` reads only as a fallback, so on nan variants its config is silently shadowed while `[setup]` claims it was applied (+5 more)

### Community 86 - "Verdicts"
Cohesion: 0.14
Nodes (13): 09 - Test adequacy for the imputation task: verdicts, F-09-10 - Tests that restate the implementation and could go, F-09-1 - The `[NULL]`-path diagnostic builds its cell selection in frame order and it is read in embedder order, and the only guard asserts a count, F-09-2 - No test pins `realised_rate`, and the number it would pin shows `EVAL_MASK_RATE` is nearly inert at high missingness, F-09-3 - Nothing asserts the decoder ever beats the naive baseline, so a decoder wired to the wrong truth passes the whole suite, F-09-4 - The fixture's stated reason for choosing `credit-g_20nan` is false, and the variant it wrongly excluded is the one whose induced path crashes, F-09-5 - The fixture pins six pooled averages and no structural invariant, F-09-6 - `test_imputation_summary_carries_the_decode_stage_timing` fabricates a metric key no producer writes (+5 more)

### Community 88 - "as_category_strings"
Cohesion: 0.06
Nodes (43): values(), The stock impute's column assembly, verbatim., QuantileTransformer, as_category_strings(), Stringify a categorical column with every kind of missing value collapsed to…, _position(), bool_, _ColumnModel (+35 more)

### Community 89 - "test_imputation_artifacts.py"
Cohesion: 0.12
Nodes (27): _identity_scaler(), DataFrame, StandardScaler, The files an imputation run leaves behind (ADR 0004, decision 8)., A row can have a dozen cells filled in, and one table that wide reads as noise.…, Fitted so a scaled value and its original unit are the same number. Keeps a…, Two rows' worth of scored cells, one of each kind and each population., Ticket 0004: `1.144e-09` reads as a real measurement when it means zero.… (+19 more)

### Community 90 - "MissForestImputer"
Cohesion: 0.10
Nodes (21): _count(), PassCounting, The stock impute's loop, verbatim, from a given starting fill., MissForestImputer, bool_, _ColumnModel, DataFrame, float64 (+13 more)

### Community 92 - "sync_remote.py"
Cohesion: 0.08
Nodes (46): artifact_prefix(), _backup(), branch_tips(), code_files(), _columns_holding(), copy_replica(), delete_pushed_files(), _foreign_roots() (+38 more)

### Community 93 - "gap_token_replication_report.py"
Cohesion: 0.29
Nodes (11): _best_baseline(), folds(), interval(), _path(), DataFrame, ndarray, Pre-registered analysis of the gap-token replication on fresh seeds…, _read() (+3 more)

### Community 94 - "Make the learning-rate schedule selectable, keep the legacy one as default, and tag it in MLflow"
Cohesion: 0.29
Nodes (7): Consequences, Considered options, Context, Decision, How to compare, Make the learning-rate schedule selectable, keep the legacy one as default, and tag it in MLflow, Status

### Community 95 - "impprobe_arms.py"
Cohesion: 0.06
Nodes (25): _ClipEnd, _Clipped, CondExpImputer, main(), make_factory(), MiceImputer, MiceSeqClipImputer, MiceSeqImputer (+17 more)

### Community 96 - "pretrain_probe.py"
Cohesion: 0.12
Nodes (38): Float, _cells(), _centred(), _cka(), _collapse(), _config(), _diagnostics(), drift() (+30 more)

### Community 97 - "mimic_report.py"
Cohesion: 0.07
Nodes (38): by_row_gaps(), folds(), line(), loss_belongs_to(), main(), DataFrame, Floats, Path (+30 more)

### Community 98 - "test_sync_remote.py"
Cohesion: 0.12
Nodes (64): _branches(), _local_time(), _mode(), _names(), pull(), push(), push_code(), Rewrite every artifact root under ``source_prefix`` to ``target_prefix``, in… (+56 more)

### Community 99 - "types.py"
Cohesion: 0.05
Nodes (79): _Fold, _request(), parse_args(), Compatibility entry points for command-line and programmatic training., labelled_rows(), device, Classifier fine-tuning and fold metric calculation., Fine-tune and evaluate the classifier using the legacy optimization loop. (+71 more)

### Community 100 - "03. Task-keyed lookup and explicit promotion"
Cohesion: 0.29
Nodes (6): 03. Task-keyed lookup and explicit promotion, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 101 - "Research: third-party stub strategy for sklearn and scipy"
Cohesion: 0.08
Nodes (24): 10. Out of scope, but found, 11. What could not be verified, 12. Sources, 1. Answer, 2.1 typeshed — no, for either library, 2.2 scipy — `scipy-stubs` is official and current, 2.3 sklearn — nothing credible exists, 2. Does a maintained stub package exist? (+16 more)

### Community 102 - "Keep missForest outside the baseline bar, and put a single-pass LightGBM in it"
Cohesion: 0.25
Nodes (7): Consequences, Considered options, Context, Decision, Keep missForest outside the baseline bar, and put a single-pass LightGBM in it, Outcome, Status

### Community 103 - "train_and_evaluate_decoder"
Cohesion: 0.10
Nodes (29): train(), DecodedCells, What the decoder would fill every cell of a batch with. ``categorical_ids`` are…, Fill every cell of the batch, in the column's own vocabulary., _already_missing(), _clean(), _decode_parameters(), _exact() (+21 more)

### Community 104 - "06. The hyperparameter JSON schema and the three naming layers"
Cohesion: 0.25
Nodes (7): 06. The hyperparameter JSON schema and the three naming layers, Answer, For the execution ticket, Notes, Question, Scope, already fixed, What to decide

### Community 105 - "07. Enforcing the constants convention beyond what the checker catches"
Cohesion: 0.22
Nodes (8): 07. Enforcing the constants convention beyond what the checker catches, A prior worth stating, Answer, Metric keys: same treatment where it is free, no new machinery, Notes, Question, The measurement this answer depends on, What to decide

### Community 106 - "09. Record the design: ADR 0006 and the implementation plan"
Cohesion: 0.33
Nodes (5): 09. Record the design: ADR 0006 and the implementation plan, Acceptance, Notes, Question, The work

### Community 107 - "_ValidationGapScore"
Cohesion: 0.11
Nodes (19): SimpleNamespace, The fixed validation mask's ``impute_score`` of the model's current weights,…, The rows ``sw_score`` compares, one vector each: the numbers as they are…, The validation rows' induced ``impute_score`` of the model's current weights,…, What ``dist_score`` needs beside the cells, fixed for the fold: each numerical…, What ``sw_score`` needs, fixed for the fold (imputation-protocol/72): the rows…, ``impute_score_classcond``: the same errors over the class-conditional fill's,…, _sliced_rows() (+11 more)

### Community 108 - "Map: Typing and naming standard for TRIDENT"
Cohesion: 0.33
Nodes (6): Decisions so far, Destination, Map: Typing and naming standard for TRIDENT, Not yet specified, Notes, Out of scope

### Community 109 - "Addendum — the Optuna results, 2026-09-15"
Cohesion: 0.20
Nodes (8): 1. What ran, 2. The search objective, measured properly, 3. The decisive evidence: tuning does not beat defaults, 4. Finding-by-finding ledger, 5. Carried forward and new, 6. What is still unmeasured, 7. Verification record, Addendum — the Optuna results, 2026-09-15

### Community 110 - "2. Confirmed defects"
Cohesion: 0.20
Nodes (10): 2. Confirmed defects, D-1 — `_select` hands the decoder a frame-ordered mask where a token-ordered one is required (**high**), D-2 — `experiment_imputation.ps1` at HEAD is byte-identical to `experiment.ps1` and runs classification (**medium**), D-3 — Two shipped variants crash inside the shipped configuration space before scoring an induced cell (**medium**), D-4 — Pre-training's re-rolled validation mask drives real `plateau` learning-rate decisions (**medium**), D-5 — The flat per-dataset metrics file is task-blind and last-writer-wins (**medium**), D-6 — `--score_null_path` writes every induced cell twice and mislabels it in the preview (**low**), D-7 — Extra evaluation rates collide silently on a rounded percent key (**low**) (+2 more)

### Community 111 - "Findings"
Cohesion: 0.14
Nodes (13): 11 - The documentation contract: CONTEXT.md, ARCHITECTURE.md and README against the code, Checked and cleared, F-11-1 - CONTEXT's "Search objective" promises an untouched test split; every trial scores it, logs it, and parks it one column from the objective, F-11-2 - ARCHITECTURE states a false universal about categorical vocabularies and the wrong head width; the code is right and the ADR is right, F-11-3 - The "Hyperparameter → code wiring reference" was updated for the schedule and not for the task, and three of its surviving rows are now wrong for imputation, F-11-4 - `--lr_scheduler cosine` is prescribed in two documents, enforced in none, and the only pinned decode artifact is frozen under the schedule the documents warn against, F-11-5 - CONTEXT defines the `full` search-space profile as sampling everything; it holds the evaluation rate and the schedule, by design, F-11-6 - The stage-timing contract is false for the imputation task: the window is training plus the entire scoring pass, and it moves with flags that change no training (+5 more)

### Community 114 - "Findings"
Cohesion: 0.15
Nodes (12): 04 - Ground truth and the evaluation protocol, Checked and cleared, F-04-1 - `--score_null_path` scores the wrong cells on any table whose CSV interleaves categorical and numerical columns, F-04-2 - the self-masked population's sampling design changes character across the ladder, and the ADR asserts it is comparable anyway, F-04-3 - a growing share of the headline induced population has no observed cell to condition on, bounding `impute_score` toward 1.0, F-04-4 - the row-alignment guard passes on the one sibling incompatibility present in the shipped corpus, F-04-5 - the README's `EVAL_MASK_RATE` explanation is wrong in direction and points at a metric with a different denominator, F-04-6 - the validation and test evaluation masks are the same draw, not two independent ones (+4 more)

### Community 115 - "Findings"
Cohesion: 0.15
Nodes (12): 06 - Artifacts, metric keys and MLflow identity, Checked and cleared, F-06-1 - The shared per-dataset metrics file is task-blind, and imputation runs have already destroyed five classification baselines in the working tree, F-06-2 - The denominator of the headline `impute_score` is never recorded anywhere, and the per-column artifact reports raw error with no baseline beside it, F-06-3 - `--score_null_path` writes every induced cell into the ledger twice, and the preview prints the column twice in one block with a wrong cell count, F-06-4 - The repo's own MLflow filtering rule was edited by this branch and still omits `tags.task`, on the only two metric families both tasks share, F-06-5 - An imputation fold logs no step-less test loss, so the decoder's own objective is never scored on held-out data, F-06-6 - The imputation fold metric key set is data-dependent, and a fold that disagrees aborts the whole CV run after every fold has trained (+4 more)

### Community 116 - "Verdicts"
Cohesion: 0.15
Nodes (12): 11 - The documentation contract: CONTEXT.md, ARCHITECTURE.md and README against the code: verdicts, F-11-1 - CONTEXT's "Search objective" promises the test split "stays untouched"; every trial scores it and logs it, F-11-2 - ARCHITECTURE asserts every categorical vocabulary holds `"nan"` and the head is `Linear(d, V_col − 3)`; on `_00nan` variants it is `V_col − 2`, F-11-3 - The hyperparameter wiring table gained an `lr_scheduler` row, no decode rows, and three surviving rows are now task-inaccurate, F-11-4 - `--lr_scheduler cosine` is prescribed in two documents, defaulted in none, and the imputation fixture is pinned under `cosine_legacy`, F-11-5 - CONTEXT defines `full` as sampling every hyperparameter the task uses; it permanently holds three, F-11-6 - The stage-timing contract: the imputation window wraps the entire scoring pass and moves with flags that change no training, F-11-7 - "On the test fold only" in two documents, while a trial scores a validation population and `--score_null_path` adds a fourth (+4 more)

### Community 117 - "Verdicts"
Cohesion: 0.17
Nodes (11): 06 - Imputation artifacts, metric keys and MLflow identity: verdicts, F-06-1 - The root `metrics/<dataset>_metrics.csv` is keyed by dataset alone, so an imputation run overwrites a classification run's fold table with a different column set, F-06-2 - The naive denominator of `impute_score` is never persisted, and the per-column artifact reports raw error with no baseline beside it, F-06-3 - With `--score_null_path` the same cell is in the ledger under two populations, and the preview prints it twice in one block with a row-count header, F-06-4 - `CLAUDE.md`'s comparison rule was edited on this branch and still omits `tags.task`, on the only metric families both tasks share, F-06-5 - No step-less `test/loss` on an imputation fold, so the decoder's objective is allegedly never scored on held-out data, F-06-6 - The imputation fold metric key set is data-dependent, so a disagreeing fold aborts the CV run after every fold has trained, F-06-7 - Extra evaluation rates are keyed by a rounded integer percent, so nearby rates collide silently and a sub-0.5% rate becomes `rate_0` (+3 more)

### Community 118 - "Findings"
Cohesion: 0.18
Nodes (10): 03 - Imputation metrics and the composite score, Checked and cleared, F-03-1 - The masked `impute_score` is not comparable across the missingness ladder: the exam's realised mask rate is non-monotone, 0.209 → 0.155 → 0.250, F-03-2 - "1.0 is baseline parity" is false for the composite: a model that has collapsed to mode imputation on every categorical column scores 0.75, F-03-3 - The Optuna search objective is scored on the exact cells that selected the checkpoint, F-03-4 - The baseline normalisation is a per-fold scalar rescale, not a per-column normalisation: it removes no column heterogeneity and adds sampling noise to the CI, F-03-5 - `_ratio`'s zero guard protects the pooled error, not the constant column ADR 0004 says it protects, and inverts the ranking when it does fire, F-03-6 - The per-column artifact carries errors without their baselines, so the one artifact meant to localise a bad fold cannot (+2 more)

### Community 119 - "Verdicts"
Cohesion: 0.18
Nodes (10): 04 - Ground truth and the evaluation protocol: verdicts, F-04-1 - `--score_null_path` marks the selection tensor in CSV column order while the model's mask is in categorical-then-numerical order, so the diagnostic scores transposed cells, F-04-2 - the self-masked population's realised rate is non-monotone and floor-dominated across the ladder, contradicting the ADR's comparability claim, F-04-3 - a growing share of the induced population sits in rows with no observed cell, bounding those cells' contribution at baseline parity, F-04-4 - `_assert_row_aligned` checks only value equality through an object-dtype comparison, and certifies the corpus's one sibling incompatibility, F-04-5 - the README's `EVAL_MASK_RATE` paragraph omits the per-row floor and quotes a figure against a different denominator than the metric it names, F-04-6 - validation and test evaluation masks are drawn from one stream reseeded identically, so the two questions are not independent, F-04-7 - `rmse_num_z` / `mae_num_z` are in units of a sigma fit over all rows including the test fold, re-estimated at each rung (+2 more)

### Community 120 - "Findings"
Cohesion: 0.18
Nodes (10): 10 - Configuration plumbing and classification bit-identity, Checked and cleared, F-10-1 - The flat metrics aggregate is not task-keyed, so the two tasks overwrite each other's record for the same dataset variant, F-10-2 - The JSON/override loader's epoch fallbacks moved from 40 to 300/150, changing classification training length for any partial configuration, F-10-3 - The imputation launcher configures the decode stage through classification's keys, edits classification's config file, and tells the operator to compare a metric imputation never emits, F-10-4 - `config_source` over-claims: an imputation run reports a file that supplied none of its decode parameters, and silently inherits that file's schedule, F-10-5 - `EVAL_MASK_RATES_EXTRA` is in no run record and is silently dropped by promotion, contradicting the documented "a promoted file is complete", F-10-6 - The agent-facing MLflow comparison rule was updated for the schedule but not for the task (+2 more)

### Community 121 - "Findings"
Cohesion: 0.18
Nodes (10): 12 - Reproducibility and provenance of an imputation run end to end, Checked and cleared, F-12-1 - The `_XXnan` missingness draw, which *is* the induced benchmark, is recorded nowhere and no check in the pipeline can detect its substitution, F-12-2 - Training corruption is drawn from one global numpy stream whose position at fold *k* is a function of every earlier fold's draws, so ADR 0005 decision 6's "the configuration is the only difference" is false from fold 2 and no single fold is reproducible on its own, F-12-3 - `config_source` is a mutable path into an untracked directory; the commit the run records does not contain the file, and re-running that commit silently trains the defaults instead, F-12-4 - The branch-new environment record names torch and CUDA only, omitting exactly the libraries whose behaviour the imputation path depends on, F-12-5 - A run's on-disk artifact tree and its MLflow run are joined only by a timestamp drawn twice, and both successful imputation parents in the store are off by one second, Findings (+2 more)

### Community 122 - "Findings"
Cohesion: 0.20
Nodes (9): 01 - The decode stage, Checked and cleared, F-01-1 - The masked population's realised mask rate is neither the configured rate nor comparable across the missingness ladder, and the code's stated explanation of the gap has the sign backwards, F-01-2 - The masked and induced populations are drawn from systematically different rows and shown different context, yet are reported side by side as the same measurement at different difficulties, F-01-3 - The validation and test masks are drawn from one RNG stream, so the checkpoint is selected under the same masked-column configurations the test score is measured under, F-01-4 - The checkpoint criterion and the fold-ranking metric weight the two column kinds differently, so the epoch chosen is not the epoch that best serves the reported score, F-01-5 - The extra-rate diagnostic ladder collapses at the low end and silently overwrites itself when two rates round to the same percent, Findings (+1 more)

### Community 123 - "Findings"
Cohesion: 0.20
Nodes (9): 02 - The decoder model and its heads, Checked and cleared, F-02-1 - Checkpoint selection runs on a λ-weighted composite, so `LAMBDA_NUM` decides which epoch is kept as well as how the gradient is mixed, F-02-2 - The per-kind loss terms are computed and detached specifically for reporting, and both callers throw them away, F-02-3 - Per-cell pooling inside the categorical term: a column that cannot be got wrong still takes its share of the denominator, F-02-4 - A categorical column with no real categories builds a zero-width head and kills `predict` after the whole stage has run, F-02-5 - Nothing in the suite asserts that decode gradients reach the embedder and the transformer, Findings (+1 more)

### Community 124 - "Verdicts"
Cohesion: 0.20
Nodes (9): 03 - Imputation metrics and the composite score: verdicts, F-03-1 - The masked exam is a different exam at each rung of the ladder, so the ADR's cross-ladder comparability claim is unsupported, F-03-2 - "1.0 is baseline parity" allegedly false for the composite, F-03-3 - The Optuna trial score is read off the same cells that selected the checkpoint, F-03-4 - The baseline normalisation is a per-fold scalar rescale that allegedly injects noise into the promotion CI, F-03-5 - `_ratio`'s zero guard allegedly inverts the ranking and guards the wrong thing, F-03-6 - The per-column artifact carries errors without their baselines, Verdicts (+1 more)

### Community 125 - "Verdicts"
Cohesion: 0.20
Nodes (9): 10 - Configuration plumbing and classification bit-identity: verdicts, F-10-1 - The flat `metrics/<dataset>_metrics.csv` carries no task key, so the two tasks silently overwrite each other's record for one dataset variant, F-10-2 - `from_mapping`'s epoch fallbacks moved 40/40 → 300/150, changing classification training length for partial configs, F-10-3 - `experiment_imputation.ps1` drives the decode stage through classification's keys, writes the shared classification file, and closes with an f1_macro query, F-10-4 - An imputation run falling back to the shared file logs a `config_source` naming a file that supplied none of its decode parameters, and silently takes its `LR_SCHEDULER`, F-10-5 - `EVAL_MASK_RATES_EXTRA` is in no run record and is deleted by `--promote_best`, against README's "a promoted file is complete", F-10-6 - `CLAUDE.md`'s comparison rule gained `tags.lr_scheduler` but not `tags.task`, Verdicts (+1 more)

### Community 126 - "Findings"
Cohesion: 0.20
Nodes (9): 13 - What the shared pre-training stage contributes to imputation, Checked and cleared, F-13-1 - 300 epochs of pre-training end 4.2x worse than a 20-vector constant lookup on pre-training's own objective; the 98.5% loss drop is the encoder's output norm shrinking, not learning, F-13-2 - A paired ablation cannot detect any benefit from 298 of the 300 pre-training epochs: the arms sit inside fold-to-fold noise with the sign flipping between seeds, and nothing in the branch has ever run this comparison, F-13-3 - Nothing in the branch logs, tests or asserts any scale-free property of the pretrained representation, so "pre-training worked" is unfalsifiable from a run's record, F-13-4 - The reduced profile spends two thirds of every trial's epoch budget on a held stage whose contribution is unmeasured, and its one cross-stage knob makes the two stages' corruption inseparable, F-13-5 - Pre-training's re-rolled validation mask drives real `plateau` learning-rate decisions: measured 11 halvings to 1.66e-07 on spambase, from a signal with a 9% noise floor, Findings (+1 more)

### Community 127 - "Verdicts"
Cohesion: 0.22
Nodes (8): 01 - The decode stage: verdicts, F-01-1 - The realised eval-mask rate is neither the configured rate nor comparable across the ladder, and `decoding.py:152-153` names a mechanism that is already netted out, F-01-2 - The masked and induced populations are drawn from systematically different rows and shown different context, yet reported as a pair, F-01-3 - Validation and test masks come from one RNG stream, so the checkpoint is selected under the configurations the test score is measured under, F-01-4 - The checkpoint criterion and the fold-ranking metric weight the two column kinds differently, F-01-5 - The `rate_*` family collides when two configured rates round to the same percent and overwrites silently; the low end of the ladder is degenerate, Verdicts, What this critique missed

### Community 128 - "Verdicts"
Cohesion: 0.22
Nodes (8): 02 - The decoder model and its heads: verdicts, F-02-1 - The checkpoint is chosen by minimising the same λ-weighted composite the decoder trains on, so `LAMBDA_NUM` also decides which epoch survives, and that criterion is only loosely aligned with the `impute_score` that ranks the fold, F-02-2 - The per-kind loss terms are built and detached for reporting and every caller discards them, so no run can decompose the decode loss curve, F-02-3 - The categorical term pools per cell across columns, so a column that cannot be got wrong contributes zero loss while still inflating the denominator, F-02-4 - An all-missing categorical column builds `nn.Linear(d, 0)` and crashes `predict` after the whole decode stage has run; `local_of` maps excluded ids to `-1` with no `ignore_index` guard, F-02-5 - No test asserts that decode-stage gradients reach the embedder and transformer, so the ADR's "the encoder is fine-tuned, not frozen" is unguarded, Verdicts, What this critique missed

### Community 129 - "Findings"
Cohesion: 0.22
Nodes (8): 05 - The embedder, [MASK] and [NULL] semantics, Checked and cleared, F-05-1 - `_select` hands the decoder a frame-ordered mask where a token-ordered one is required, so the `[NULL]`-path diagnostic scores cells that were never hidden, F-05-2 - `encode` has a third representation of a missing cell - literal `NaN` and the `'nan'` category - and pre-training feeds exactly that to its clean targets, F-05-3 - the induced population is scored on a row shape the decoder never trained on: every gap becomes `[MASK]` and no `[NULL]` survives, F-05-4 - `rmse_num_z`'s unit is fit on the whole table, evaluation rows included, Findings, Open questions

### Community 130 - "Verdicts"
Cohesion: 0.22
Nodes (8): 12 - Reproducibility and provenance of an imputation run end to end: verdicts, F-12-1 - The induced-missing benchmark rests on an unrecorded, ungoverned missingness draw that nothing can detect the substitution of, F-12-2 - One global numpy stream means folds >= 2 enter at an offset set by earlier folds, so decision 6's "the configuration is the only difference" is not what it sounds like, F-12-3 - `config_source` is a mutable path into an untracked directory, and the fallback is silent, F-12-4 - The environment record omits the scikit-learn version, F-12-5 - The results directory and the MLflow run are joined only by a timestamp drawn twice, and both finished parents are off by a second, Verdicts, What this critique missed

### Community 131 - "Verdicts"
Cohesion: 0.22
Nodes (8): 13 - What the shared pre-training stage contributes to imputation: verdicts, F-13-1 - Pre-training's own loss falls 97-99% while the encoder ends worse than a constant-per-column lookup, and carries no per-cell information about the hidden value, F-13-2 - The "shared and protected" `EPOCHS_PRE = 300` has no measurement behind it anywhere in the branch, and the critic's first ablation cannot detect a benefit, F-13-3 - The stage logs three scalars and nothing scale-free, and no unit test ever executes it, F-13-4 - The reduced profile holds two thirds of each trial's epoch budget on a stage the search cannot touch, and no sampled dimension moves pre-training alone, F-13-5 - Pre-training's re-rolled validation mask drives real `plateau` learning-rate decisions, Verdicts, What this critique missed

### Community 132 - "Verdicts"
Cohesion: 0.25
Nodes (7): 05 - The embedder, [MASK] and [NULL] semantics: verdicts, F-05-1 - `_score_induced_missing` overrides `masked_positions` with a DataFrame-ordered `isna()` matrix, while every reader of that field indexes it categorical-then-numerical, so the `[NULL]` diagnostic scores the wrong cells, F-05-2 - `encode` on a frame that never went through `preprocess_table` represents a missing cell as raw `NaN` (numerical) or the `'nan'` category, and pre-training's clean targets are built that way, F-05-3 - the induced population is scored on rows where every gap has become `[MASK]` and no `[NULL]` survives, a missingness pattern the decoder essentially never trained on, F-05-4 - the `StandardScaler` that defines `rmse_num_z`'s unit is fitted on the whole table, including the rows the metric scores, Verdicts, What this critique missed

### Community 133 - "backfill_decode_gap_token_tag.py"
Cohesion: 0.16
Nodes (17): backfill(), BackfillReport, main(), MlflowClient, Run, Stamp the ``decode_gap_token`` tag on imputation runs recorded before it…, Active parent and trial runs in the experiment families, never in the mirrors., _training_runs() (+9 more)

### Community 134 - "Fidelity audit of `CONSOLIDATED.md`"
Cohesion: 0.29
Nodes (6): Fidelity audit of `CONSOLIDATED.md`, Judged faithful, recorded rather than changed, Per-source confirmation, What this audit did not check, What was checked, and how, What was found and changed

### Community 135 - "test_backfill_baseline_imputers.py"
Cohesion: 0.11
Nodes (21): client(), _folds(), _gap_rows(), fixture, MlflowClient, Backfilling the baseline imputers of ADR 0007 onto runs recorded before the…, A run recorded before ``baseline/best`` existed gets it from its own logged…, Two folds as ``raw_fold_metrics.csv`` holds them for a run of 2026-09-24: the… (+13 more)

### Community 136 - "§8 — Experiments run 2026-09-17"
Cohesion: 0.13
Nodes (15): 8.10 The one separation in twelve, replicated: it does not hold, 8.11 The committed `credit-g_20nan` promoted file loses to the defaults; `credit-g_40nan` leans the same way, 8.12 The sign of tuned-versus-defaults follows the winner's rank, 8.13 Verification record for §8, 8.14 The control: what the best trial is worth once it is scored the way the comparison is, 8.1 The typing refactor preserved imputation behaviour exactly, 8.2 The noise floor N-1 asked for, 8.3 Mis-selection on the study's own split — and, in §8.14, how little of it survives five-fold (+7 more)

### Community 137 - "Training Refactor Implementation Plan"
Cohesion: 0.14
Nodes (21): Task 3 Report: Runtime Adapters and Regression Baseline, src/training Package, train.main(args, return_metrics=False) Compatibility Facade, vehicle_00nan Regression Fixture, ADR 0001: Training Package with Compatibility Facade, Compatibility Facade Pattern, Plan: Add --all CLI Flag for Batch Dataset Training, discover_datasets() (+13 more)

### Community 138 - "Consolidated report — the imputation work on `feat/imputation-task`"
Cohesion: 0.29
Nodes (7): 1. Verdict, 5. Checked and cleared, 6. Coverage map, Consolidated report — the imputation work on `feat/imputation-task`, Refuted, and severity-none, Source files the branch touched that **no dimension opened**, Suspected and disproved — do not spend time here

### Community 139 - "4. What the experiments can and cannot support"
Cohesion: 0.33
Nodes (6): 4. What the experiments can and cannot support, Cross-ladder reads (`_20nan` → `_80nan`), `experiment_imputation.ps1` — do not run it as it stands, `imputation_studies.ps1 -Compare` — the twelve comparison runs (none has been run), `imputation_studies.ps1` — the study branch (five studies run, one in flight, one not started), Two crashes to expect

### Community 140 - "Appendix: every reader's summary lines (`results_digest.py`, unedited)"
Cohesion: 0.05
Nodes (44): Appendix: every reader's summary lines (`results_digest.py`, unedited), E118: classification, the paper's pre-training gain under the upstream defaults (classification-step-zero/15), E119: ADR 0037's encoder under MAR and MNAR (imputation-protocol/49), E120: classification, TRIDENT against MLP under MNAR on fresh seeds (classification-step-zero/16), E121: ADR 0018's encoder under MAR and MNAR at 20nan (imputation-protocol/50), E136: classification, S/T/B at a tenth and a quarter of the labels (c1-split/08), E137: TabPFN v2 under informative missingness (c15-mnar-prediction/08), E138: the quarter-label channel on fresh seeds, validation scarce (c1-split/09) (+36 more)

### Community 141 - "0014-gaps-shown-as-mask-in-the-decode-stage.md"
Cohesion: 0.07
Nodes (26): Consequences, Context, Decision, Show the real gaps as [MASK] in the imputation decode stage by default, Status, Verification, Design (fixed before launch), Does asking the gaps one column at a time, the rest as [NULL], score better? (pre-registered 2026-10-01) (+18 more)

### Community 142 - "test_backfill_pretrain_gap_token_tag.py"
Cohesion: 0.35
Nodes (10): client(), fixture, MlflowClient, Stamping the pre-training gap token on runs recorded before it could be chosen., Every run before the choice showed its gaps to pre-training as [NULL],…, _seed_runs(), test_a_dry_run_writes_nothing_and_a_second_apply_finds_nothing(), test_an_existing_mirror_is_stamped_in_place_and_never_duplicated() (+2 more)

### Community 143 - "Entries"
Cohesion: 0.01
Nodes (184): E01 · `[MASK]` vs `[NULL]` token path for induced cells (2026-09-10; re-checked 2026-09-11), E02 · Reduced Optuna search: six studies under cosine (2026-09-11), E03 · Promoted configuration vs defaults, cosine, seed 42 (2026-09-11 – 2026-09-12), E04 · Does the pre-training loss drop mean learning? (critique probe, 2026-09-11), E05 · Pre-training epochs 300 vs 2 (critique ablation, 2026-09-11), E06 · Decode checkpoint criterion vs ranking metric (critique probe, 2026-09-11), E07 · Plateau halvings driven by pre-training's re-rolled validation mask (critique read, 2026-09-11), E08 · Resolution and fidelity of the masked search objective (critique reads, 2026-09-11) (+176 more)

### Community 144 - "RsyncTransport"
Cohesion: 0.37
Nodes (3): CompletedProcess, rsync and ssh; through WSL on Windows, where their keys and known_hosts live., RsyncTransport

### Community 145 - "build_fold_result"
Cohesion: 0.22
Nodes (12): build_fold_result(), ndarray, Build the legacy per-fold classification metric set., compute_cv_summary(), DataFrame, Path, Return the legacy mean, sample standard deviation, and display string., test_build_fold_result_keeps_binary_confusion_fields_for_single_class_fold() (+4 more)

### Community 146 - "DirectoryServer"
Cohesion: 0.24
Nodes (4): _copy(), _copy_tree(), DirectoryServer, The server as a directory on this machine: copies stand in for rsync and the…

### Community 147 - "batched_heads_report.py"
Cohesion: 0.27
Nodes (7): folds(), interval(), _path(), DataFrame, ndarray, Pre-registered analysis of the batched-heads check (imputation-pretraining/06),…, report()

### Community 148 - "Transport"
Cohesion: 0.22
Nodes (3): Protocol, How files and commands reach the server. Paths are relative to each side's root., Transport

### Community 149 - "gap_token_report.py"
Cohesion: 0.29
Nodes (10): _best_baseline(), folds(), interval(), _path(), DataFrame, ndarray, Pre-registered analysis of the gaps-as-mask training study (imputation-token-…, report() (+2 more)

### Community 150 - "summary.py"
Cohesion: 0.12
Nodes (29): _best_baseline_copies(), best_baseline_gaps(), _best_baselines(), best_strong_baseline_gaps(), _best_strong_baselines(), _diagnostic_roles(), external_baseline_gaps(), fold_timings_for_tracking() (+21 more)

### Community 151 - "pretrain_decode_budget_report.py"
Cohesion: 0.25
Nodes (8): folds(), interval(), _path(), DataFrame, ndarray, Pre-registered analysis of the decode-budget study, read from the store read-…, report(), source_of()

### Community 152 - "sliced_run.py"
Cohesion: 0.08
Nodes (49): _fitter(), _pooled(), DataFrame, A per-(variant, seed, fold) quantity averaged over the variants, one value a…, directions(), _fitter(), jobs(), _ledger() (+41 more)

### Community 153 - "Counter"
Cohesion: 0.24
Nodes (15): Counter, _best_baseline(), folds(), integrity(), interval(), _path(), DataFrame, ndarray (+7 more)

### Community 154 - "2026-10-23-classification-step-zero.md"
Cohesion: 0.05
Nodes (33): ADR 0029: Classification search ranked on validation macro-F1, Consequences, Context, Decision, Open for the user, ADR 0032: TabPFN v2 as a separate strong tier for classification, Consequences, Context (+25 more)

### Community 155 - "ADR 0037: The bigger encoder on pendigits and letter at 40 and 60nan"
Cohesion: 0.09
Nodes (20): ADR 0037: The bigger encoder on pendigits and letter at 40 and 60nan, Consequences, Context, Decision, Verification (2026-11-03 10:43 GMT-3), Verification of decision 3 (2026-11-03 20:46 GMT-3), Design (fixed before the first cell), Measures and rule, fixed now (`large_batch.py --study e113 --tally`) (+12 more)

### Community 156 - "not_miwae.py"
Cohesion: 0.11
Nodes (19): _FillView, fit_notmiwae_fills(), fit_vae_imputers(), _mlp(), _Model, _normal_log(), bool_, DataFrame (+11 more)

### Community 157 - "Overnight on gorgona8, 2026-09-29/30"
Cohesion: 0.18
Nodes (11): 1. Pre-training ablation (E24), 2. What changed in the code, 3. Pre-training objectives (E25), 4. Normalised pre-training with a long decode stage (E26), 5. Where the CPU time goes, 6. Decisions for you, 7. State, Afternoon follow-up (2026-09-30) (+3 more)

### Community 158 - "8. The configuration for ticket 08"
Cohesion: 0.40
Nodes (5): 8.1 `pyproject.toml`, 8.2 Where the stubs live, 8.3 Resolver impact — verified against copies of the real `pyproject.toml` + `uv.lock`, 8.4 Honest cost to ticket 08, 8. The configuration for ticket 08

### Community 159 - "tally"
Cohesion: 0.05
Nodes (82): RuntimeError, apply(), choose(), collect(), DataFrame, ndarray, Path, E69: per-column weights for the K + TabPFN blend, cross-fitted, offline… (+74 more)

### Community 160 - "test_embedder.py"
Cohesion: 0.31
Nodes (8): _embedder(), _mixed_frame(), DataFrame, Behaviour of the shared TabularEmbedder, used by both tasks., ``None`` and ``NaN`` are the same absence, so they share one vocabulary entry.…, T06. ``EMBEDDING_LOOKUP batched`` reads the categorical tables in one gather.…, test_a_missing_category_is_one_category_however_it_was_written(), test_the_batched_lookup_gives_the_per_column_values_and_gradients_to_rounding()

### Community 161 - "_MetricClock"
Cohesion: 0.40
Nodes (3): _MetricClock, Hand out metric timestamps that only ever increase within this process. A flush…, Claim ``count`` consecutive milliseconds; return the first.

### Community 162 - "0004-imputation-decoder-task.md"
Cohesion: 0.06
Nodes (28): 01. Research: how do imputation papers score mixed-type tabular imputation?, Answer, Comments, Question, 03. What is the imputation ground truth, and on which cells is error scored?, Answer, Comments, Question (+20 more)

### Community 163 - "backfill_search_objective_tag.py"
Cohesion: 0.14
Nodes (20): backfill(), BackfillReport, default_objective(), main(), _optuna_runs(), MlflowClient, Run, Stamp the ``search_objective`` tag on Optuna runs recorded before ADR 0008.… (+12 more)

### Community 164 - "What does pre-training contribute to imputation? (pre-registered 2026-09-29)"
Cohesion: 0.22
Nodes (9): Amendment 2026-09-29 (21:24 GMT-3): the remaining cells run on another machine, Amendment 2026-09-29 (23:08 GMT-3): the last eight cells write to the SSD store, Arms (fixed before launch), Known limits, stated in advance, Measures, Outcome (2026-09-30, 01:10 GMT-3), Question, What does pre-training contribute to imputation? (pre-registered 2026-09-29) (+1 more)

### Community 165 - "decode_patience_batch_report.py"
Cohesion: 0.27
Nodes (7): folds(), interval(), _path(), DataFrame, ndarray, Pre-registered analysis of the early-stopping (04) and larger-batch (05)…, report()

### Community 166 - "test_leadership.py"
Cohesion: 0.17
Nodes (29): load_role(), client_for(), events(), FakeServices, follow(), LocalPeer, pair(), Any (+21 more)

### Community 167 - "Mirror every run into one derived MLflow experiment per task"
Cohesion: 0.25
Nodes (8): Consequences, Considered options, Context, Decision, How to compare, Mirror every run into one derived MLflow experiment per task, Outcome (2026-09-17), Status

### Community 168 - "backfill_lgbm_baseline.py"
Cohesion: 0.11
Nodes (28): Pattern, _baseline_populations(), plan_gap(), The gap-to-best-baseline statistics a run lacks, or why it will not get them.…, apply_lgbm_plan(), _as_planned_from(), _kept(), main() (+20 more)

### Community 169 - "pretrain_objective_report.py"
Cohesion: 0.22
Nodes (11): folds(), interval(), _path(), DataFrame, ndarray, queue(), Pre-registered analysis of the pre-training objective study, read from the…, The pre-registered order of each of the two concurrent queues. (+3 more)

### Community 171 - "Do batched decoder heads change anything but the training draw? (pre-registered 2026-09-30)"
Cohesion: 0.14
Nodes (12): Apply the decoder's per-column heads in batched operations by default, Consequences, Context, Decision, Status, Arms (fixed before launch), Do batched decoder heads change anything but the training draw? (pre-registered 2026-09-30), Known limits, stated in advance (+4 more)

### Community 173 - "cls_stage1_bar.py"
Cohesion: 0.16
Nodes (22): _balanced(), _codes(), _impute(), _interval(), _learn(), _out(), Any, bool_ (+14 more)

### Community 174 - "TASKS.md"
Cohesion: 0.05
Nodes (10): H2's fresh-draw confirmation: odds 3 at seed 4848 against its MCAR siblings, confirmation seeds (E142; pre-registered 2026-11-05), Outcome (2026-11-05, 01:24 GMT-3; condition met by E139 at 00:34; 18 of 18 A1 cells and 18 TabPFN pairs on medusa1, done by 01:20 GMT-3, 0 failed), Design (fixed before the first cell), Measures and reading, fixed now (`hybrid_blend.py --tally`), Outcome (2026-10-24, 17:15 GMT-3), The per-column KC + TabPFN hybrid as an output: does E69's result hold on fresh seeds, over today's default? (E74; pre-registered 2026-10-24), Outcome, TRIDENT against its baselines on a distributional score that is not marginal: the sliced Wasserstein distance between filled and true rows (E182, T59; pre-registered 2026-11-18 13:00 GMT-3) (+2 more)

### Community 175 - "main"
Cohesion: 0.06
Nodes (43): main(), One cell of the architecture screening (imputation-architecture/01, task T03).…, configuration(), main(), One cell of the batched-heads check (imputation-pretraining/06). uv run…, E132's diagnostic: S and B on the four MNAR files, seed 42, MLflow ON (leader),…, main(), One cell of the column-wise scoring study (imputation-token-shape/01, task T02… (+35 more)

### Community 176 - "The deployable hybrid on `mimic-wide`: safe and still a gain where TabPFN alone is weak? (E178, T57; pre-registered 2026-11-13 08:28 GMT-3)"
Cohesion: 0.29
Nodes (7): Amendment (2026-11-13 11:33 GMT-3): medusa1's GPU 0 fell off the bus, the rest runs on GPU 1, Amendment (2026-11-13 19:09 GMT-3): GPU 1 was not usable either; the study is paused, nothing read, Amendment (2026-11-13 20:17 GMT-3): the rest runs on gorgona8, on the user's call; a departure from ADR 0035, Design, Measures and rule (fixed now), Outcome (2026-11-14 14:32 GMT-3; read once 2026-11-14 14:30 GMT-3, on gorgona8 at 48b6cbe, all 9 runs and their cells in), The deployable hybrid on `mimic-wide`: safe and still a gain where TabPFN alone is weak? (E178, T57; pre-registered 2026-11-13 08:28 GMT-3)

### Community 177 - "setup_mlflow"
Cohesion: 0.15
Nodes (25): backfill(), BackfillReport, main(), _mirrors_by_source(), MlflowClient, Run, Every active mirror run, by the run it mirrors., Stamp the ``decode_freeze_epochs`` and ``decode_encoder_lr_scale`` tags on… (+17 more)

### Community 178 - "baseline_cache.py"
Cohesion: 0.14
Nodes (20): main(), DataFrame, FoldScorer, Fill the baseline cache for cells a study will run, before its GPU pass (plan…, _scorer(), BaselineScorer, _environment_digest(), _Family (+12 more)

### Community 179 - "joint_draws.py"
Cohesion: 0.09
Nodes (30): between(), check(), correlation(), joint_residual_draws(), measures(), model_draws(), Any, bool_ (+22 more)

### Community 180 - "mnar_family.py"
Cohesion: 0.11
Nodes (32): dial(), hide(), labels(), main(), mcar(), noise(), outcome(), probit() (+24 more)

### Community 181 - "mlflow_utils.py"
Cohesion: 0.10
Nodes (22): build_hyperparams_dict(), _experiment_id(), get_or_create_mirror_experiment(), src/mlflow_utils.py ------------------- Centralised MLflow utilities for…, The id of the experiment of this name, created if absent., Return the experiment_id of the task's mirror experiment, creating it if…, Build a flat dict of all TRIDENT hyperparameters suitable for…, device (+14 more)

### Community 182 - "test_training_tasks.py"
Cohesion: 0.12
Nodes (26): Return the ranking contract for a task name, rejecting unknown names., task_spec(), The per-task fold-ranking contract (ADR 0004, decisions 6 and 12)., One metric to read beside the model's: per population, the baseline whose mean…, How far the model is from the bar, as a number to sort and minimise: per fold,…, A search ranks its trials by its own objective, which need not be the fold-…, missForest is an external comparison (ADR 0017): it would set the bar on most…, ADR 0021: missForest and the linear arms form the strong tier. The bar (the… (+18 more)

### Community 183 - "TabPFNImputer"
Cohesion: 0.10
Nodes (27): Factory, _Column, Any, bool_, DataFrame, float64, int64, NDArray (+19 more)

### Community 184 - "_DisabledMlflow"
Cohesion: 0.18
Nodes (4): _DisabledMlflow, info, Drop-in no-op used when an Optuna invocation disables tracking., _Run

### Community 185 - "test_training_pretraining.py"
Cohesion: 0.12
Nodes (28): device, Train the masked reconstruction model using the legacy optimization loop., train_pretrainer(), The training batch: ``batch_size``, shrunk so an epoch over ``row_count`` rows…, training_batch_size(), BufferedFoldTracker, Collect one fold's MLflow events until its diagnostic role is known., T14: off keeps BATCH; on, a small table gets floor(rows / N) and so at least N… (+20 more)

### Community 187 - "Next up, in the recommended order"
Cohesion: 0.12
Nodes (16): Backlog, Done (2026-09-30), Later baselines and classification (plan 2026-10-17), Model selection, Next up, in the recommended order, Process, Specific follow-ups, Speed (+8 more)

### Community 188 - "TRIDENT README"
Cohesion: 0.13
Nodes (21): credit-g Categorical Columns, electricity Categorical Columns, kr-vs-kp Categorical Columns, ADR 0002: Curated Cross-Validation MLflow Runs, ADR 0034: A host tag on every run, Consequences, Context, Decision (+13 more)

### Community 190 - "backfill_decode_patience_tag.py"
Cohesion: 0.16
Nodes (17): backfill(), BackfillReport, main(), MlflowClient, Run, Stamp the ``decode_patience`` tag on imputation runs recorded before it…, Active parent and trial runs in the experiment families, never in the mirrors., _training_runs() (+9 more)

### Community 191 - "backfill_decoder_heads_tag.py"
Cohesion: 0.16
Nodes (17): backfill(), BackfillReport, main(), MlflowClient, Run, Stamp the ``decoder_heads`` tag on imputation runs recorded before it existed.…, Active parent and trial runs in the experiment families, never in the mirrors., _training_runs() (+9 more)

### Community 192 - "remasker.py"
Cohesion: 0.10
Nodes (26): ModuleList, _blocks(), _learning_rate(), _Model, _positions(), bool_, DataFrame, device (+18 more)

### Community 194 - "calibration-as-imputation-output.md"
Cohesion: 0.08
Nodes (24): Evidence, Proposal: the calibration as the imputation output at 40% missing and above, Sources, The idea, What adopting it takes, Design (fixed before launch), Does the calibration help at 40% missing and above? (pre-registered 2026-10-16), Known limits, stated in advance (+16 more)

### Community 195 - "propensity_weights"
Cohesion: 0.16
Nodes (20): _blocks(), propensity_weights(), DataFrame, ndarray, Where a training fold's own gaps fall, so the decode stage's masks can follow…, Each column's features for the other columns' models, in frame order., (rows, columns) weights in ``frame``'s column order; 1 where a column has no…, draw_weighted_mask() (+12 more)

### Community 196 - "Do decode masks that follow the missingness help under MAR? A propensity-mask arm (E65; pre-registered 2026-10-22)"
Cohesion: 0.33
Nodes (6): Before the ticket (disclosed), Design (fixed before the first cell), Do decode masks that follow the missingness help under MAR? A propensity-mask arm (E65; pre-registered 2026-10-22), Measures and reading, fixed now (`mar_propensity_report.py --tally`), Outcome (2026-10-23, 05:43 GMT-3), The tag backfill (2026-10-23)

### Community 197 - "_cells"
Cohesion: 0.07
Nodes (48): contrast(), _defaults_kept(), main(), _masked_selection_bar(), missforest(), Cells, DataFrame, ndarray (+40 more)

### Community 198 - "ADR 0038: Label-scarce fine-tuning as a documented regime for classification"
Cohesion: 0.40
Nodes (5): ADR 0038: Label-scarce fine-tuning as a documented regime for classification, Consequences, Context, Decision, What this does not decide

### Community 199 - "TabularEmbedder"
Cohesion: 0.04
Nodes (63): dtype, EncodedTable, Tensor, Class that encapsulates the creation of embeddings for tabular data: -…, Per device, the (``[NULL]``, ``[MASK]``) ids of every categorical column, as…, The same table with every real gap shown as its column's ``[MASK]``. ADR 0004…, C1's arm T: the same table with every pre-training hole shown as its column's…, C15's A2: the feature columns with each filled cell back to NaN, the gap… (+55 more)

### Community 202 - "0013-imputation-defaults-from-the-confirmatory-study.md"
Cohesion: 0.06
Nodes (30): Consequences, Context, Decision, Give the imputation task the configuration the confirmatory study chose, Status, Verification, Choose the imputation decode checkpoint on the validation rows' own gaps by default, Consequences (+22 more)

### Community 203 - "Rascunho: TRIDENT como imputador — fundamentação, metodologia, resultados e direções"
Cohesion: 0.07
Nodes (28): 0. Abertura `[entra]`, 10. Trabalho futuro e decisões `[entra]`, com opções neutras, 1.1 Motivação, 1.2 RQs e cobertura, 1.3 Posicionamento: os parentes e o que cada um faz `[entra]`, 1. A tese `[entra]`, 2.1 O que o fork mudou em relação ao TRIDENT do artigo `[entra]`, 2.2 Dados e variantes (+20 more)

### Community 204 - "BaselineImputer"
Cohesion: 0.05
Nodes (54): main(), make_factory(), Add-on to mfreplay_arms.py: stock test-pass counts per fold, and arm C.…, Population, _cells(), _frame_values(), DataFrame, FoldScorerFactory (+46 more)

### Community 205 - "stage4_gbdt.py"
Cohesion: 0.21
Nodes (19): HistGradientBoostingClassifier, _atomic(), _csv(), _fit(), Fold, _folds(), _json(), Any (+11 more)

### Community 208 - "large_batch.py"
Cohesion: 0.08
Nodes (27): cell_dir(), folds(), DataFrame, Path, E101: pendigits at batch 1024 with the learning rates x4, and a longer decode…, E117: the bigger encoder at 80nan, where ADR 0037's proposal stops, split by…, E119: ADR 0037's encoder off MCAR, on new MAR and MNAR variants, split by seed…, E121: ADR 0018's encoder off MCAR at 20nan, as E119 at 40-60nan, split by seed… (+19 more)

### Community 210 - "Does a bigger encoder help the large tables? (pre-registered 2026-10-06)"
Cohesion: 0.08
Nodes (22): Consequences, Context, Decision, Status, Use a bigger encoder for imputation on pendigits_20nan and letter_20nan, Verification, Verification of decision 4 (2026-11-03 23:05 GMT-3), Design (fixed before launch) (+14 more)

### Community 211 - "04. The decoder model"
Cohesion: 0.29
Nodes (6): 04. The decoder model, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 212 - "Apontamentos por slide"
Cohesion: 0.07
Nodes (27): Apontamentos por slide, Conferências feitas depois do juiz, Lint dos slides da reunião com pesquisadores (2026-11-15), Recomendações gerais, Slide 1, Slide 10, Slide 11, Slide 12 (+19 more)

### Community 213 - "Does ADR 0037's bigger encoder help off MCAR? (E119; pre-registered 2026-11-03)"
Cohesion: 0.33
Nodes (6): Amendment 2026-11-03 18:56 GMT-3 (the user's decision), Data (generated before the first cell; the data directory is gitignored, the scripts are the record), Design (fixed before the first cell), Does ADR 0037's bigger encoder help off MCAR? (E119; pre-registered 2026-11-03), Measures and rule, fixed now (`large_batch.py --study e119 --tally`), Outcome (2026-11-03 19:54 GMT-3): holds under MAR and under MNAR

### Community 214 - "deployable_calibration_report.py"
Cohesion: 0.23
Nodes (12): cells(), contrast(), integrity(), line(), DataFrame, ndarray, Run, Pre-registered analysis of the deployable calibration study (imputation-… (+4 more)

### Community 215 - "mnar_family_report.py"
Cohesion: 0.17
Nodes (26): cells(), contrasts(), design(), dial(), gaps(), integrity(), line(), mcar() (+18 more)

### Community 216 - "parents"
Cohesion: 0.11
Nodes (31): todo(), conformal(), development_primary(), _dispersion(), _interval(), _intervals(), _ledger(), _numerical() (+23 more)

### Community 217 - "remasker_offmcar_report.py"
Cohesion: 0.13
Nodes (25): gaps(), main(), mar_cells(), mar_pairs(), mnar_cells(), mnar_pairs(), Cells, DataFrame (+17 more)

### Community 218 - "impprobe_arms_v1_preregistered.py"
Cohesion: 0.12
Nodes (15): CondExpImputer, main(), make_factory(), MiceImputer, _NumericFill, _pairwise_cov(), DataFrame, ndarray (+7 more)

### Community 219 - "fold_metrics"
Cohesion: 0.09
Nodes (41): compare(), Path, Classification step zero, Stage 0: reproduce an old sweep run's fold macro-F1…, The sweep's run of ``variant``: its id and its artifact directory., run(), sweep_run(), _cell(), _contrast() (+33 more)

### Community 221 - "confirmation_report.py"
Cohesion: 0.13
Nodes (23): cell(), cells(), contrast(), DataFrame, Run, Pre-registered analysis of the confirmation on the 15 unused variants…, ``readout`` (K, H, M, KC) minus ``opponent`` (an opponent of ``report_common``,…, report() (+15 more)

### Community 222 - "backfill_strong_tier.py"
Cohesion: 0.16
Nodes (19): apply_strong_tier_plan(), _as_planned_from(), main(), _offline_mismatch(), _pending(), Plan, plan_strong_tier(), FoldScorerFactory (+11 more)

### Community 223 - "meeting_figures.py"
Cohesion: 0.20
Nodes (25): contrast(), find_section(), main(), occlusion_synthetic(), parse_bias(), parse_cls(), parse_confirmation(), parse_frozen() (+17 more)

### Community 224 - "store_merge.py"
Cohesion: 0.24
Nodes (24): Say, _change_experiment(), _change_run(), columns(), connect(), _ids(), _insert(), _insert_experiment() (+16 more)

### Community 225 - "test_backfill_lgbm_baseline.py"
Cohesion: 0.13
Nodes (24): client(), _logged(), _plan(), fixture, MlflowClient, Backfilling ADR 0017 onto runs recorded before it: ``lgbm`` joins the bar,…, ADR 0016 logged missForest from forests that predicted on several threads,…, Runs recorded on the Windows PC carry KNN and hgb numbers this machine… (+16 more)

### Community 226 - "test_missforest.py"
Cohesion: 0.15
Nodes (24): _gappy(), _imputer(), DataFrame, The missForest baseline imputer, an external comparison outside the bar (ADRs…, Where a third of every training column is missing, the rounds have gaps to fill…, The scored frame still holds the truth under every hidden cell. A fill that…, The cache keys an entry on the machine, not on its cores, and gorgona8 and the…, A forest that predicts on several threads sums its trees in whatever order they… (+16 more)

### Community 228 - "generate_mar_family.py"
Cohesion: 0.16
Nodes (22): _categorical(), family(), main(), DataFrame, ndarray, Path, The MAR family: E63's mechanism on more tables and rates, with a slope dial…, _roundtrip() (+14 more)

### Community 229 - "ADR 0036: A large batch and a longer decode on pendigits and letter at 20 and 40nan"
Cohesion: 0.06
Nodes (31): ADR 0036: A large batch and a longer decode on pendigits and letter at 20 and 40nan, Consequences, Considered options, Context, Decision, Verification (2026-10-31 18:23 GMT-3), Design (fixed before the first cell), Measures and rule, fixed now (`large_batch.py --tally`) (+23 more)

### Community 230 - "backfill_decode_calibration_tag.py"
Cohesion: 0.27
Nodes (10): backfill(), BackfillReport, main(), _mirrors_by_source(), MlflowClient, Run, Every active mirror run, by the run it mirrors., Stamp the ``decode_calibration`` tag on imputation runs recorded before it… (+2 more)

### Community 231 - "retune_report.py"
Cohesion: 0.17
Nodes (21): _best_baseline(), _cells(), folds(), integrity(), interval(), _minutes(), _path(), _promoted() (+13 more)

### Community 233 - "model_for"
Cohesion: 0.09
Nodes (35): column_contrasts(), _draws(), h22(), h2b(), Any, DataFrame, ndarray, random_effects() (+27 more)

### Community 234 - "mar_family_report.py"
Cohesion: 0.18
Nodes (20): cells(), effect(), gaps(), _merge(), name(), DataFrame, ndarray, Path (+12 more)

### Community 235 - "The decode stage's levers against washing out the encoder (E50; pre-registered 2026-10-19)"
Cohesion: 0.07
Nodes (24): D1: the decode curves, pre-trained against not, from the store (E49; pre-registered 2026-10-19), Data, fixed now, Measures, fixed now, Outcome (2026-10-19, 19:39 GMT-3), Question, Reading, fixed now, Design (fixed before the first cell), Launch (+16 more)

### Community 236 - "build_mimic.py"
Cohesion: 0.21
Nodes (20): cardio(), compare_with_cardio(), describe(), hourly(), main(), DataFrame, ndarray, Path (+12 more)

### Community 237 - "02-equivalence.md"
Cohesion: 0.10
Nodes (17): ADR 0035: A study may split its cells between the servers by seed, Consequences, Context, Decision, Design (fixed now), Do gorgona8 and medusa1 give the same fold scores for the same cell? (E86; pre-registered 2026-10-28), Launch and tally, Measures and rule (`cross_machine.py --tally`) (+9 more)

### Community 238 - "run_from_namespace"
Cohesion: 0.17
Nodes (14): _format_duration(), main(), Namespace, Return a human-readable duration string., Command-line entry point for TRIDENT., Run training sequentially on every discovered dataset., run_all(), Namespace (+6 more)

### Community 239 - "deep_baselines.py"
Cohesion: 0.17
Nodes (20): _atomic(), _dataset(), deal(), _fit(), _folds(), probe(), Any, DataFrame (+12 more)

### Community 240 - "report_common.py"
Cohesion: 0.25
Nodes (16): baselines_in(), _best(), _excluded(), kind_ratios(), opponent(), per_type_bar(), DataFrame, float64 (+8 more)

### Community 241 - "08-deep-baselines.md"
Cohesion: 0.10
Nodes (16): Design (fixed before the first cell), Measures and rule, fixed now (`deep_baselines.py --study e106 --tally`), Outcome, TabNet-B − TabNet (E99), same folds (descriptive, scripted, 2026-11-01 14:39 GMT-3), The tally, unedited (`deep_baselines.py --study e106 --tally`, 2026-11-01 14:39 GMT-3), Was E99's TabNet undertrained? Its tuned configuration with a bigger budget (E106; pre-registered 2026-11-01 11:26 GMT-3), Design (fixed before the first cell), Measures (`deep_baselines.py --study e107 --tally`) (+8 more)

### Community 242 - "architecture_size_report.py"
Cohesion: 0.23
Nodes (17): _best_baseline(), folds(), integrity(), interval(), _minutes(), _pairs(), _path(), DataFrame (+9 more)

### Community 243 - "decode_metric_run.py"
Cohesion: 0.31
Nodes (7): cell_dir(), done(), Path, E145's cells: the decode checkpoint chosen by ``impute_score`` (A) or by the…, todo(), train(), E147's GPU tier (pre-registered and run as E149): E145's two arms on MAR and…

### Community 244 - "mnar_bias_report.py"
Cohesion: 0.18
Nodes (17): _load(), main(), _per_seed(), DataFrame, ndarray, Path, E45's and E46's measure 5: the signed bias per column under MNAR against MCAR…, _load() (+9 more)

### Community 245 - "Does ADR 0018's bigger encoder help off MCAR at 20nan? (E121; pre-registered 2026-11-03)"
Cohesion: 0.33
Nodes (6): Amendment 2026-11-03 22:03 GMT-3 (the user's decision, "sim"), Data (generated before the first cell; the data directory is gitignored, the scripts are the record), Design (fixed before the first cell), Does ADR 0018's bigger encoder help off MCAR at 20nan? (E121; pre-registered 2026-11-03), Measures and rule, fixed now (`large_batch.py --study e121 --tally`), Outcome (2026-11-03 22:47 GMT-3): holds under MAR and under MNAR

### Community 246 - "Choosing the decode checkpoint by a distributional score (E145; pre-registered 2026-11-05)"
Cohesion: 0.40
Nodes (5): Amendment (2026-11-05, 19:29 GMT-3), Choosing the decode checkpoint by a distributional score (E145; pre-registered 2026-11-05), Design and rule (fixed before the first cell), Outcome (2026-11-05, 20:22 GMT-3; 72 of 72 cells on medusa1, 0 failed, the last queue done 20:12 GMT-3), The tag's backfill (2026-11-05, applied 18:53 to 19:54 GMT-3, verified 20:03 GMT-3)

### Community 247 - "62-hybrid-fresh-seeds.md"
Cohesion: 0.05
Nodes (37): Design, Measures and rule (fixed now), Outcome (2026-11-08 11:46 GMT-3), Printout (unedited), The deployable hybrid with missForest where TRIDENT trails it (E162, T54; pre-registered 2026-11-08 11:01 GMT-3), Design, Measures and rule (fixed now), Outcome (2026-11-08 22:05 GMT-3) (+29 more)

### Community 248 - "c15_stage0.py"
Cohesion: 0.21
Nodes (14): drivers(), g2_cell(), g2_oof_cell(), gbdt(), interval(), load(), DataFrame, ndarray (+6 more)

### Community 249 - "checkpoint_report.py"
Cohesion: 0.22
Nodes (16): bar(), _cells(), folds(), interval(), _path(), DataFrame, ndarray, Pre-registered analysis of the deployable-checkpoint study (imputation-… (+8 more)

### Community 250 - "3. Hypotheses"
Cohesion: 0.12
Nodes (17): 1. The evidence, 2. What pre-training optimises now, 3. Hypotheses, 4. What the literature offers, 5. Recommended order, H1 · The objective collapses onto the column-constant part of its target, H2 · The target is in the wrong space, so what the encoder learns does not transfer, H3 · The decode stage repeats the pretext and washes the encoder out (+9 more)

### Community 251 - "classcond_run.py"
Cohesion: 0.13
Nodes (25): _best(), cell_dir(), cells(), folds(), on_class_scale(), DataFrame, ndarray, Path (+17 more)

### Community 252 - "test_generate_mnar.py"
Cohesion: 0.17
Nodes (13): main(), DataFrame, Value-dependent (MNAR) variants of the complete tables, at the MCAR siblings'…, The frame with round(n * pct) cells of every feature column hidden, larger…, ``<base>_probit_<pp>nan`` at the default slope; any other slope is spelt into…, tilted(), variant_name(), _complete() (+5 more)

### Community 253 - "Proposal: multiple imputation with draws joint across a row's gaps (would be E179, T56; drafted 2026-11-13 20:24 GMT-3)"
Cohesion: 0.13
Nodes (13): Design proposed, Proposal: multiple imputation with draws joint across a row's gaps (would be E179, T56; drafted 2026-11-13 20:24 GMT-3), Risks said now, The user's calls, Why, Design (fixed now), Measures and rule (fixed now), Multiple imputation from H's fills: do Rubin's intervals cover? (E175, T56; pre-registered 2026-11-12 23:13 GMT-3) (+5 more)

### Community 254 - "mfreplay_arms.py"
Cohesion: 0.09
Nodes (13): main(), make_factory(), MissForestSeqImputer, _missing_of(), DataFrame, ndarray, Probe: replay vs sequential test-time fill for the fork's real missForest…, A frame to be filled sequentially during the fit. (+5 more)

### Community 255 - "occlusion_spread.py"
Cohesion: 0.15
Nodes (18): _files(), Path, E88's cells: E84's 12 shifted variants at 60nan, at E84's seeds; 10 and 5 of 12…, E85's cells: the ten tables at 20nan and 80nan, each at the seed set its…, E79's cells: the ten tables at 60nan, each at the seed set whose earlier runs K…, _use_grid(), _use_mechanisms(), _use_sixty() (+10 more)

### Community 256 - "mcar_draws_report.py"
Cohesion: 0.19
Nodes (17): cells(), contrast(), draw_name(), gate(), _merge(), pattern(), DataFrame, ndarray (+9 more)

### Community 257 - "mimic_mortality.py"
Cohesion: 0.20
Nodes (14): inputs(), DataFrame, ndarray, E150: fills and gap patterns judged by a mortality classifier…, The four tables a fold's classifier reads: numerical columns standardised,…, The ticket's classifier, fitted on the training rows and scored on the test…, run(), score() (+6 more)

### Community 258 - "distribution_scores"
Cohesion: 0.22
Nodes (14): distribution_scores(), naive_distances(), DataFrame, A distributional score for imputed cells, read beside ``impute_score``…, The mean/mode fill's own distances, averaged over the columns of each kind: a…, ``wd_num_z``, ``tv_cat`` and ``dist_score`` of scored cells carrying ``actual``…, _cells(), DataFrame (+6 more)

### Community 259 - "tabpfn_report.py"
Cohesion: 0.23
Nodes (15): contrast(), _has_kc(), main(), Cells, DataFrame, ndarray, Path, Run (+7 more)

### Community 260 - "electricity_report.py"
Cohesion: 0.22
Nodes (14): common_cells, _cells(), columns_table(), main(), _per_column(), profile(), DataFrame, Series (+6 more)

### Community 261 - "make_mimic_variants.py"
Cohesion: 0.18
Nodes (11): cut(), main(), nearest_incomplete(), DataFrame, ndarray, Path, Cut E149's variants: real gap patterns transplanted onto complete rows…, The positions of the complete rows, and for each the position of its nearest… (+3 more)

### Community 262 - "cross_machine.py"
Cohesion: 0.19
Nodes (14): cell_dir(), _diff(), folds(), DataFrame, ndarray, Path, E86: do gorgona8 and medusa1 give the same fold scores for the same cell?…, E94's cells: seven imputation and five classification variants on small tables,… (+6 more)

### Community 263 - "stage4_trident.py"
Cohesion: 0.26
Nodes (14): E115 (descriptive): per mechanism, TRIDENT − baseline pooled over its variants,…, tally_e115(), best_config(), claim(), eval_dir(), evaluate(), Path, E98: step zero Stage 4, the TRIDENT arm tuned on validation (classification-… (+6 more)

### Community 264 - "backfill_host_tag.py"
Cohesion: 0.21
Nodes (11): RunTag, backfill(), main(), MlflowClient, Stamp the ``host`` tag on runs recorded before it, from their ``gpu_name`` (ADR…, _tag(), client(), fixture (+3 more)

### Community 265 - "Plan: improve the imputation experiment protocol and baselines (2026-10-17)"
Cohesion: 0.14
Nodes (14): 0. What this plan rests on, 1. Where E40 changes or overlaps the review, 3. Experiment drafts (one ticket each), 4. ADRs this needs, 5. Schedule (two weeks from Saturday 2026-10-17, GMT-3), 6. Risks and traps, 7. Decisions only the user can make, 8. What this plan does not do (+6 more)

### Community 266 - "calibration_report.py"
Cohesion: 0.29
Nodes (11): _best_baseline(), _cells(), folds(), interval(), _path(), DataFrame, ndarray, Pre-registered analysis of the calibration at 40% missing and above… (+3 more)

### Community 267 - "mar_smoke_report.py"
Cohesion: 0.26
Nodes (13): cells(), gaps(), _merge(), DataFrame, ndarray, Path, Run, queue() (+5 more)

### Community 268 - "test_backfill_strong_tier.py"
Cohesion: 0.23
Nodes (13): client(), _logged(), _plan(), fixture, Backfilling ADR 0021 onto runs recorded before it: the linear arms and the…, A run of two folds as ADR 0017 left it: model 0.9 and 0.7, mean/mode at parity,…, condexp 0.5 and 0.6 (mean 0.55) beats missForest (0.75): it is the strong…, _recomputed() (+5 more)

### Community 269 - "test_report_common.py"
Cohesion: 0.21
Nodes (10): _frame(), DataFrame, The shared report reading (T21): which column is K, what the bar is, and the…, E34/E35 kept the loss's epoch, so their headline is H and K sits in…, ADR 0027 makes the calibrated output the headline at 40% missing and above.…, _run(), test_k_is_the_induced_checkpoint_family_on_a_loss_run_and_the_headline_on_an_induced_one(), test_on_a_calibrated_run_k_is_the_uncalibrated_family_and_kc_the_headline() (+2 more)

### Community 270 - "categorical_deficit.py"
Cohesion: 0.28
Nodes (12): _entropy(), _errors(), _ledger(), _parts(), DataFrame, ndarray, Path, E61: where does K lose to TabPFN on kr-vs-kp and credit-g? A per-column read,… (+4 more)

### Community 271 - "mnar_smoke_report.py"
Cohesion: 0.29
Nodes (11): cells(), gap(), integrity(), DataFrame, ndarray, Run, Pre-registered analysis of the MNAR smoke family (imputation-protocol/05, E45).…, G on the MCAR variants must reproduce E35's cells: its K is E35's… (+3 more)

### Community 272 - "backfill_decode_mask_weighting_tag.py"
Cohesion: 0.27
Nodes (11): backfill(), BackfillReport, main(), _mirrors_by_source(), MlflowClient, Run, Every active mirror run, by the run it mirrors., Stamp the ``decode_mask_weighting`` tag on imputation runs recorded before it… (+3 more)

### Community 273 - "backfill_embedding_lookup_tag.py"
Cohesion: 0.27
Nodes (11): backfill(), BackfillReport, main(), _mirrors_by_source(), MlflowClient, Run, Every active mirror run, by the run it mirrors., Stamp the ``embedding_lookup`` tag on imputation runs recorded before it (T06).… (+3 more)

### Community 274 - "Does pre-training help classification under the upstream code's defaults? (E118; pre-registered 2026-11-03)"
Cohesion: 0.40
Nodes (5): Design (fixed before the first cell), Does pre-training help classification under the upstream code's defaults? (E118; pre-registered 2026-11-03), Measures and rule, fixed now (`cls_tenfold.py --study e118 --tally`), Outcome, The tally, unedited (`cls_tenfold.py --study e118 --tally`, medusa1, 2026-11-03 14:26 GMT-3)

### Community 275 - "TRIDENT on 10,000 rows of `mimic-wide` at the full table's optimizer steps: is the capped arm's loss rows or steps? (E180, T57; pre-registered 2026-11-14 19:39 GMT-3)"
Cohesion: 0.40
Nodes (5): Amendment (2026-11-15 21:20 GMT-3): the leader left the network, four cells failed, the queues wait for it; nothing read, Amendment (2026-11-18 11:27 GMT-3): the resume script died with gorgona8's reboots; restarted, still 11 of 27, nothing read, Design, Measures and rule (fixed now), TRIDENT on 10,000 rows of `mimic-wide` at the full table's optimizer steps: is the capped arm's loss rows or steps? (E180, T57; pre-registered 2026-11-14 19:39 GMT-3)

### Community 276 - "mar_tuned_bar_report.py"
Cohesion: 0.22
Nodes (12): cells(), gaps(), _merge(), pairs(), DataFrame, ndarray, Path, Run (+4 more)

### Community 277 - "BACKLOG.md"
Cohesion: 0.05
Nodes (36): Add missForest, with random forests and with LightGBM, to the baseline imputers, Bias, Consequences, Considered options, Context, Decision, Outcome, Status (+28 more)

### Community 278 - "TRIDENT against MLP and SAINT under MAR and MNAR at 60nan (E115, descriptive; pre-registered 2026-11-03 00:53 GMT-3)"
Cohesion: 0.17
Nodes (10): Design (fixed before the first cell), Measures (`deep_baselines.py --study e115 --tally`, on medusa1), Outcome, The tally, unedited (`deep_baselines.py --study e115 --tally`, 2026-11-03 01:19 GMT-3), TRIDENT against MLP and SAINT under MAR and MNAR at 60nan (E115, descriptive; pre-registered 2026-11-03 00:53 GMT-3), Design (fixed before the first cell), Measures and rule, fixed now (`deep_baselines.py --study e120 --tally`), Outcome (+2 more)

### Community 279 - "An equal, validation-only tuning budget for the baselines (E42; pre-registered 2026-10-16)"
Cohesion: 0.17
Nodes (12): An equal, validation-only tuning budget for the baselines (E42; pre-registered 2026-10-16), Cells, Design (fixed before the first score), Grids, Known limits, stated in advance, Measures, Note 2026-10-16, about 23:15 GMT-3: the output contract E44 reads, Note 2026-10-16, about 23:45 GMT-3: the driver, and two passes instead of one (+4 more)

### Community 280 - "Confirmation on the 15 unused variants (E44; pre-registered 2026-10-16)"
Cohesion: 0.17
Nodes (12): Cells, splits and runs, Configuration, Confirmation on the 15 unused variants (E44; pre-registered 2026-10-16), Design (fixed before launch), Known limits, stated in advance, Launch (2026-10-17, 01:51 GMT-3), Measures, Offline readings (added 2026-10-17, 15:45 GMT-3) (+4 more)

### Community 281 - "stage4_tally.py"
Cohesion: 0.30
Nodes (11): E99's rule: per level and baseline, TRIDENT − baseline test macro-F1 over 270…, tally(), _f1(), fork(), gbdt(), main(), DataFrame, ndarray (+3 more)

### Community 282 - "backfill_decode_checkpoint_tag.py"
Cohesion: 0.27
Nodes (10): backfill(), BackfillReport, main(), _mirrors_by_source(), MlflowClient, Run, Every active mirror run, by the run it mirrors., Stamp the ``decode_checkpoint`` tag on imputation runs recorded before it… (+2 more)

### Community 283 - "backfill_pretrain_gap_token_tag.py"
Cohesion: 0.27
Nodes (10): backfill(), BackfillReport, main(), _mirrors_by_source(), _pretraining_runs(), MlflowClient, Run, Every active mirror run, by the run it mirrors. (+2 more)

### Community 284 - "C15 Stage 2: MICE without an indicator (A4), the complete and MAR tables, H2-c2, H2.2 and H2b on development (E125; pre-registered 2026-11-04)"
Cohesion: 0.50
Nodes (4): C15 Stage 2: MICE without an indicator (A4), the complete and MAR tables, H2-c2, H2.2 and H2b on development (E125; pre-registered 2026-11-04), Design (fixed before the first cell), Measures and rules, fixed now (the protocol's, on development; all descriptive except as stated), Withdrawn unrun (2026-11-04 01:16 GMT-3)

### Community 285 - "Does pre-training help classification with ten folds and five seeds? (E87; pre-registered 2026-10-28)"
Cohesion: 0.50
Nodes (4): Design (fixed before the first cell), Does pre-training help classification with ten folds and five seeds? (E87; pre-registered 2026-10-28), Measures and rule, fixed now (`cls_tenfold.py --tally`), Outcome

### Community 286 - "Fills and gap patterns judged by a mortality classifier (E150; pre-registered 2026-11-05 as E152)"
Cohesion: 0.50
Nodes (4): Design and rule (fixed before the first cell), Fills and gap patterns judged by a mortality classifier (E150; pre-registered 2026-11-05 as E152), Implementation (2026-11-06, before any cell), Outcome (2026-11-06 10:27 GMT-3)

### Community 287 - "electricity_split.py"
Cohesion: 0.30
Nodes (11): _bar(), _contrast(), _ledger(), DataFrame, Path, E54, T29's test: is the boosters' lead on electricity carried by ``date`` as a…, _score(), split() (+3 more)

### Community 288 - "barcheck_verdicts.py"
Cohesion: 0.20
Nodes (5): pick(), Read-only: paired verdicts of the model vs the baseline bar and missForest,…, model/base: {seed: {fold: value}} -> mean diff, CI, per-seed means, verdict., ts(), verdict()

### Community 289 - "Linear baselines and the missing per-fold baselines, on the decoder's cells (E41; pre-registered 2026-10-16)"
Cohesion: 0.18
Nodes (11): Amendment 2026-10-16, about 23:10 GMT-3 (before any arm's score was read), Design (fixed before the first score), Integrity, before anything is read, Known limits, stated in advance, Linear baselines and the missing per-fold baselines, on the decoder's cells (E41; pre-registered 2026-10-16), Measures, Outcome (2026-10-17, 01:30 GMT-3), Running it (+3 more)

### Community 290 - "The full MNAR family: spambase, the 20nan rung and biodeg's tilt dial (E46; pre-registered 2026-10-17)"
Cohesion: 0.18
Nodes (11): Arms (runner `scripts/experiments/mnar_family_run.py`, tag `experiment=mnar-family-2026-10-17` plus `arm`), Cells, seeds and runs, Data (`datasets/generate_mnar.py`, seed 1060, as in E45), Design (fixed before launch), Launch, Measures, Outcome (2026-10-17, 18:15 GMT-3), Reading, fixed now (+3 more)

### Community 291 - "Design (fixed before launch)"
Cohesion: 0.18
Nodes (11): Arms, Checks before launch, Design (fixed before launch), Known limits, stated in advance, Measures, Out of scope, Outcome (2026-10-16, 07:15 GMT-3), Re-tuning the promoted imputation files under today's defaults (pre-registered 2026-10-15) (+3 more)

### Community 293 - "Stages"
Cohesion: 0.20
Nodes (10): Classification step zero: a plan for the user to approve (T27; 2026-10-23), Decisions (what "approve" means), Later, not step zero, Stage 0 · A determinism spot check (review #16, trap 1), Stage 1 · The CPU bar on the sweep's folds (review #14, #15), Stage 2 · The no-pre-training arm (review #16), Stage 3 · Validation-ranked classification search (ADR 0029, proposed), Stage 4 · Step zero proper (H1c) (+2 more)

### Community 294 - "What the imputation studies support, as of 2026-11-05 (consolidated)"
Cohesion: 0.20
Nodes (10): 1. Against the baselines, 2. Robustness, 3. TRIDENT and TabPFN, closer, 3b. Since 2026-10-23: E67–E85, 3c. 2026-11-03: E118–E121, 3d. 2026-11-05: classification at scarce labels and under informative missingness (E136–E144), 4. Limits the thesis states, 5. Open (+2 more)

### Community 295 - "collapse_probes_report.py"
Cohesion: 0.40
Nodes (9): _collapse_call(), main(), _probe(), DataFrame, ndarray, Path, Pre-registered analysis of E53, T28's D2: collapse and linear probes on the…, _read() (+1 more)

### Community 296 - "Services"
Cohesion: 0.30
Nodes (3): The MLflow server and the tunnel, both bound to 127.0.0.1:<port>; never both on…, Stop the whole process group: uv, mlflow and its uvicorn workers (ADR 0009)., Services

### Community 297 - "0028-tabpfn-joins-the-strong-tier-as-an-offline-member.md"
Cohesion: 0.17
Nodes (10): Consequences, Considered options, Context, Decision, Status, TabPFN v2 joins the strong tier as an offline member, ADR 0025's panel on K against TabPFN, confirmation set (E56; pre-registered 2026-10-21), Design (fixed before the first cell) (+2 more)

### Community 298 - "test_backfill_decode_encoder_tags.py"
Cohesion: 0.30
Nodes (11): client(), fixture, MlflowClient, Stamping the decode stage's encoder levers on imputation runs recorded before…, Every imputation run before E50 trained every parameter from the start. A…, The fast path: a stamped run that already has a mirror gets the two tags on…, _seed_runs(), test_a_dry_run_writes_nothing_and_a_second_apply_finds_nothing() (+3 more)

### Community 299 - "A deployable calibration: fitted on the validation mask, read with H (E47; pre-registered 2026-10-18)"
Cohesion: 0.20
Nodes (9): A deployable calibration: fitted on the validation mask, read with H (E47; pre-registered 2026-10-18), Cells, Design (fixed before launch), Known limits (note added 2026-10-18, about 23:20 GMT-3, before any score was read), Launch, Measures, Outcome (2026-10-19, 01:39 GMT-3), Reading, fixed now (+1 more)

### Community 300 - "An MNAR smoke family: value-dependent missingness at the MCAR counts (E45; pre-registered 2026-10-17)"
Cohesion: 0.22
Nodes (9): An MNAR smoke family: value-dependent missingness at the MCAR counts (E45; pre-registered 2026-10-17), Arms (runner `scripts/experiments/mnar_smoke_run.py`, tag `experiment=mnar-smoke-2026-10-17` plus `arm`), Cells, seeds and runs, Data (`datasets/generate_mnar.py`, seed 1060), Design (fixed before launch), Launch, Measures, Outcome (2026-10-17, 12:24 GMT-3) (+1 more)

### Community 301 - "test_backfill_decode_mask_weighting_tag.py"
Cohesion: 0.30
Nodes (11): client(), fixture, MlflowClient, Stamping the decode mask weighting on imputation runs recorded before E65 (none…, Every imputation run before E65 hid its decode training cells uniformly. A…, The fast path: a stamped run that already has a mirror gets the tags on that…, _seed_runs(), test_a_dry_run_writes_nothing_and_a_second_apply_finds_nothing() (+3 more)

### Community 302 - "run_retune_queue.sh"
Cohesion: 0.28
Nodes (7): best_of(), MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, pick(), run_retune_queue.sh script, TZ

### Community 303 - "steps_rule.py"
Cohesion: 0.36
Nodes (8): cell_dir(), folds(), DataFrame, Path, E100: does a steps-aware training batch help the small tables? (imputation-…, tally(), todo(), train()

### Community 304 - "test_backfill_embedding_lookup_tag.py"
Cohesion: 0.30
Nodes (11): client(), fixture, MlflowClient, Stamping the embedding lookup on imputation runs recorded before T06 (none…, Every imputation run before T06 read its categorical tables one column at a…, The fast path: a stamped run that already has a mirror gets the tags on that…, _seed_runs(), test_a_dry_run_writes_nothing_and_a_second_apply_finds_nothing() (+3 more)

### Community 305 - "Path"
Cohesion: 0.22
Nodes (9): Path, Both tasks read datasets/hiperparams/<base>/<dataset>.json, and it names no…, Backlog I2: the study used to overwrite…, Classification's promoted file would be the shared ``<dataset>.json``…, A variant with gaps beside the fixture's complete table, so an induced study…, test_a_classification_study_no_longer_writes_the_shared_file_on_its_own(), test_a_finished_imputation_study_leaves_the_shared_config_alone(), test_promotion_is_refused_for_classification_before_any_trial() (+1 more)

### Community 306 - "Let an imputation study rank its trials by the validation split's induced gaps"
Cohesion: 0.20
Nodes (7): Consequences, Considered options, Context, Decision, Let an imputation study rank its trials by the validation split's induced gaps, Status, Verification

### Community 307 - "test_backfill_encoder_final_norm_tag.py"
Cohesion: 0.35
Nodes (10): client(), fixture, MlflowClient, Stamping the encoder's final norm on runs recorded before it could be chosen., Every run before the choice ended its encoder without a final norm, whichever…, _seed_runs(), test_a_dry_run_writes_nothing_and_a_second_apply_finds_nothing(), test_an_existing_mirror_is_stamped_in_place_and_never_duplicated() (+2 more)

### Community 308 - "A deployable decode checkpoint (E43; pre-registered 2026-10-16)"
Cohesion: 0.22
Nodes (8): A deployable decode checkpoint (E43; pre-registered 2026-10-16), Checks before launch, on credit-g_60nan at seed 404 (an E40 cell), with `--no-mlflow`, Design (fixed before launch), Known limits, stated in advance, Measures, Outcome (2026-10-17, 02:00 GMT-3), The code (landed with this ticket), Why

### Community 309 - "Does the re-masking spread still rank errors under MNAR and MAR? (E84; pre-registered 2026-10-27)"
Cohesion: 0.25
Nodes (8): Amendment (2026-10-27, 20:44 GMT-3, before any reading): cut to end before a store migration, Design (fixed before the first cell), Does the re-masking spread still rank errors under MNAR and MAR? (E84; pre-registered 2026-10-27), Limits fixed now, Measures and rule, fixed now (`spread_mechanisms.py --tally`), Outcome (2026-10-27, 22:34 GMT-3): partial, not complete, Outcome on the full scope (2026-10-28, 01:42 GMT-3): supported, Second amendment (2026-10-28, 00:30 GMT-3): the full scope is restored, on torch 2.11.0

### Community 310 - "ReMasker as an external deep baseline (E157, T53; pre-registered 2026-11-06)"
Cohesion: 0.25
Nodes (8): Design (fixed before the first scored cell), Measures (E48's, through `scripts/experiments/remasker_report.py`), Outcome (2026-11-07 06:12 GMT-3): K better than ReMasker on 9 of 9; ReMasker better than the strong tier's best on 1 of 9, so an external deep baseline only. At 60–80% missing the arm barely trained, so only the 40nan readings describe ReMasker, Pre-launch timing smoke (2026-11-06, timing only, no score computed), Reading, fixed now, ReMasker as an external deep baseline (E157, T53; pre-registered 2026-11-06), What it leads to, Why 60–80nan do not describe ReMasker (found at the read; the reading above is not changed by it)

### Community 311 - "ADR 0031 decision 6's hybrid on the MIMIC-IV tables with natural gaps (E173, T55; pre-registered 2026-11-10 16:45 GMT-3)"
Cohesion: 0.25
Nodes (8): ADR 0031 decision 6's hybrid on the MIMIC-IV tables with natural gaps (E173, T55; pre-registered 2026-11-10 16:45 GMT-3), Amendment, 2026-11-11 23:25 GMT-3: first runs lost to the leader's outage (nothing read), Amendment, 2026-11-12 16:03 GMT-3: seed set 4545/5656/6767 moves to medusa1 (nothing read, no E173 run exists), Amendment, 2026-11-12 16:46 GMT-3: the lost runs' result directories moved aside on gorgona8 (nothing read), Amendment, 2026-11-12 21:08 GMT-3: medusa1 rebooted at 17:38, one `e173b` cell taken again (nothing read), Design, Measures and rule (fixed now), Outcome (read once 2026-11-12 21:22 GMT-3: supported; written 2026-11-12 21:23 GMT-3)

### Community 312 - "decode_curves_report.py"
Cohesion: 0.43
Nodes (6): Curve, _curves(), main(), D1 of T28 (E49): the decode curves, pre-trained against not, read from the…, (study, arm, variant, seed) -> the decode validation loss per epoch., _reference()

### Community 313 - "05. Data support for evaluation"
Cohesion: 0.29
Nodes (6): 05. Data support for evaluation, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 314 - "03-primary.md"
Cohesion: 0.10
Nodes (17): Arms (the code), C1: does the `[NULL]`/`[MASK]` split earn its place? (plan, 2026-11-04), Constraints, Cuts, Decisions, The claim, What is reused (the store checked first), C1 cut 1: gate G1, the indicators-only probe (E129; pre-registered 2026-11-04) (+9 more)

### Community 315 - "barcheck_tally.py"
Cohesion: 0.38
Nodes (5): lvl(), Tally verdicts by missing level and by table from barcheck_out.txt (stdlib…, recs: list of (variant, pop, outcome)., show(), tbl()

### Community 316 - "H2 behind an external gate: A1 against A2 on fresh seeds, on P's fresh draws and on MIMIC-IV's natural tables (E155, T51; pre-registered 2026-11-06)"
Cohesion: 0.29
Nodes (7): A deviation, named, Design (fixed now), Disclosure (what is known before the first cell), H2 behind an external gate: A1 against A2 on fresh seeds, on P's fresh draws and on MIMIC-IV's natural tables (E155, T51; pre-registered 2026-11-06), Launch note (2026-11-06 18:21 GMT-3, before any cell ended), Outcome (2026-11-06 21:48 GMT-3): inconclusive on both primaries; the secondary reads *gains*, Rule, fixed now

### Community 317 - "Does A1 close on A2 where the gap depends on the outcome, not on the value? MIMIC-IV complete tables with known mechanisms (E156, T52; pre-registered 2026-11-06)"
Cohesion: 0.29
Nodes (7): Data (written 2026-11-06, before this ticket's commit; the scripts are the record), Design (fixed now), Disclosure (what is known before the first cell), Does A1 close on A2 where the gap depends on the outcome, not on the value? MIMIC-IV complete tables with known mechanisms (E156, T52; pre-registered 2026-11-06), Limits, stated now, Outcome (2026-11-07 03:53 GMT-3): not confirmed; every call is *no verdict*, and A1 does not trail A2 under MCAR on these tables, Rule, fixed now

### Community 318 - "TabPFN v2's classifier on E87's ten folds, plain and class-balanced (E92; pre-registered 2026-10-28)"
Cohesion: 0.33
Nodes (6): Amendments, Design (fixed before the first fit), Limits fixed now, Measures and rule, fixed now (`tabpfn_classification.py --study e92 --tally`), Outcome, TabPFN v2's classifier on E87's ten folds, plain and class-balanced (E92; pre-registered 2026-10-28)

### Community 319 - "Deep baselines for classification: does TRIDENT generally beat MLP, SAINT and TabNet? (E99; pre-registered 2026-10-31 21:34 GMT-3)"
Cohesion: 0.29
Nodes (7): Before the ticket (disclosed), Cost probe (medusa1, 2026-10-30 17:15–17:49 GMT-3; seconds only, no score read), Deep baselines for classification: does TRIDENT generally beat MLP, SAINT and TabNet? (E99; pre-registered 2026-10-31 21:34 GMT-3), Design (fixed before the first cell), Measures and rule, fixed now (`deep_baselines.py --tally`), Outcome, The tally, unedited (`deep_baselines.py --tally`, 2026-11-01 05:57 GMT-3)

### Community 320 - "Does an EMA target make classification pre-training help? (E112, T13; pre-registered 2026-11-02 10:17 GMT-3)"
Cohesion: 0.29
Nodes (7): Amendment (2026-11-02 11:28 GMT-3): a NaN bug, fixed, and a relaunch, Design (fixed before the first cell), Does an EMA target make classification pre-training help? (E112, T13; pre-registered 2026-11-02 10:17 GMT-3), Measures and rule, fixed now (`cls_ema.py --tally`), Outcome, Second amendment (2026-11-02 14:31 GMT-3, before any reading): seed 202 moves to medusa1, The tally, unedited (`cls_ema.py --tally`, 2026-11-02 22:34 GMT-3)

### Community 322 - "backfill_pretrain_objective_tag.py"
Cohesion: 0.16
Nodes (17): backfill(), BackfillReport, main(), _pretraining_runs(), MlflowClient, Run, Stamp the ``pretrain_objective`` tag on runs recorded before ADR 0011. Every…, Active runs that pre-trained, in the experiment families, never in the mirrors. (+9 more)

### Community 323 - "2. Workstreams"
Cohesion: 0.22
Nodes (9): 2. Workstreams, 2a · Reporting: one shared helper and the thesis table (T21), 2b · New baselines: condexp, inductive SoftImpute, ICE (T22), 2c · Fairness: a baseline tuning budget and a deployable checkpoint (T23), 2d · Metrics: the per-kind panel and the per-type bar (T24), 2e · Confirmation on the 15 unused variants (T25), 2f · MNAR: a smoke family and the t2 readout (T26), 2g · Later: MaCoDE, TabPFN v2 and classification step zero (T27) (+1 more)

### Community 324 - "ReMasker on letter, the paper's rule (E159, T53; pre-registered 2026-11-07 21:43 GMT-3)"
Cohesion: 0.33
Nodes (6): Design (E158's in all else), Measures and reading (fixed now), Outcome (2026-11-08 02:38 GMT-3), Printout, letter only (unedited), Printout, the 12 numerical variants (E158's CSVs and these; unedited), ReMasker on letter, the paper's rule (E159, T53; pre-registered 2026-11-07 21:43 GMT-3)

### Community 325 - "ReMasker off MCAR: the MAR and MNAR families (E160, T53; pre-registered 2026-11-07 21:46 GMT-3)"
Cohesion: 0.29
Nodes (7): Amendment (2026-11-08 10:50 GMT-3), Design, Limits, stated now, Measures and rule (fixed now; H is the headline those studies read, K beside), Outcome (2026-11-08 02:38 GMT-3), Printout (unedited), ReMasker off MCAR: the MAR and MNAR families (E160, T53; pre-registered 2026-11-07 21:46 GMT-3)

### Community 326 - "ADR 0031 decision 6's hybrid at the thesis seeds, on the 26 MCAR variants (E174, T54; pre-registered 2026-11-10 16:40 GMT-3)"
Cohesion: 0.29
Nodes (7): ADR 0031 decision 6's hybrid at the thesis seeds, on the 26 MCAR variants (E174, T54; pre-registered 2026-11-10 16:40 GMT-3), Amendment, 2026-11-11 02:11 GMT-3: paused while the leader is down (nothing read), Amendment, 2026-11-11 23:25 GMT-3: relaunched at 14:06, leader down again (nothing read), Amendment, 2026-11-12 16:46 GMT-3: `pendigits_20nan` rerun, its offline cells taken twice (nothing read), Design, Measures and rule (fixed now), Outcome (read once 2026-11-12 18:05 GMT-3: supported; written 2026-11-12 18:09 GMT-3)

### Community 327 - "A MIMIC-IV table past TabPFN's context: does the decoder hold against TabPFN v2 there? (E176, T57; pre-registered 2026-11-12 23:49 GMT-3)"
Cohesion: 0.40
Nodes (5): A MIMIC-IV table past TabPFN's context: does the decoder hold against TabPFN v2 there? (E176, T57; pre-registered 2026-11-12 23:49 GMT-3), Design, Measures and rule (fixed now), Outcome (2026-11-13 07:23 GMT-3): supported, The table (built 2026-11-12 23:30 GMT-3, commit 8566da7)

### Community 328 - "run_adr36_thesis_queue.sh"
Cohesion: 0.29
Nodes (6): MKL_NUM_THREADS, MLFLOW_TRACKING_URI, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_adr36_thesis_queue.sh script, TZ

### Community 329 - "run_blend_weights_queue.sh"
Cohesion: 0.29
Nodes (6): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_blend_weights_queue.sh script, TABPFN_MODEL_CACHE_DIR, TZ

### Community 330 - "run_calibration_queue.sh"
Cohesion: 0.29
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_calibration_queue.sh script, TZ

### Community 331 - "run_checkpoint_queue.sh"
Cohesion: 0.29
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_checkpoint_queue.sh script, TZ

### Community 332 - "run_cls_ema_queue.sh"
Cohesion: 0.29
Nodes (6): MKL_NUM_THREADS, MLFLOW_TRACKING_URI, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_cls_ema_queue.sh script, TZ

### Community 333 - "run_cls_tenfold_queue.sh"
Cohesion: 0.29
Nodes (6): MKL_NUM_THREADS, MLFLOW_TRACKING_URI, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_cls_tenfold_queue.sh script, TZ

### Community 334 - "run_confirmation_queue.sh"
Cohesion: 0.29
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_confirmation_queue.sh script, TZ

### Community 335 - "run_default_refresh_queue.sh"
Cohesion: 0.29
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_default_refresh_queue.sh script, TZ

### Community 336 - "run_deployable_calibration_queue.sh"
Cohesion: 0.29
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_deployable_calibration_queue.sh script, TZ

### Community 337 - "run_deployable_hybrid_queue.sh"
Cohesion: 0.29
Nodes (6): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_deployable_hybrid_queue.sh script, TABPFN_MODEL_CACHE_DIR, TZ

### Community 338 - "run_e111_trident_queue.sh"
Cohesion: 0.29
Nodes (6): MKL_NUM_THREADS, MLFLOW_TRACKING_URI, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_e111_trident_queue.sh script, TZ

### Community 339 - "run_e115_trident_queue.sh"
Cohesion: 0.29
Nodes (6): MKL_NUM_THREADS, MLFLOW_TRACKING_URI, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_e115_trident_queue.sh script, TZ

### Community 340 - "run_encoder_levers_queue.sh"
Cohesion: 0.29
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_encoder_levers_queue.sh script, TZ

### Community 341 - "run_frozen_encoder_queue.sh"
Cohesion: 0.29
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_frozen_encoder_queue.sh script, TZ

### Community 342 - "run_hybrid_confirmation_queue.sh"
Cohesion: 0.29
Nodes (6): MKL_NUM_THREADS, MLFLOW_TRACKING_URI, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_hybrid_confirmation_queue.sh script, TZ

### Community 343 - "run_hybrid_queue.sh"
Cohesion: 0.29
Nodes (6): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_hybrid_queue.sh script, TABPFN_MODEL_CACHE_DIR, TZ

### Community 344 - "run_imputer_utility_chain_gorgona8.sh"
Cohesion: 0.33
Nodes (6): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, say(), run_imputer_utility_chain_gorgona8.sh script, TZ

### Community 345 - "run_large_encoder_queue.sh"
Cohesion: 0.29
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_large_encoder_queue.sh script, TZ

### Community 346 - "run_mar_family_queue.sh"
Cohesion: 0.29
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_mar_family_queue.sh script, TZ

### Community 347 - "run_mar_propensity_queue.sh"
Cohesion: 0.29
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_mar_propensity_queue.sh script, TZ

### Community 348 - "run_mar_smoke_queue.sh"
Cohesion: 0.29
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_mar_smoke_queue.sh script, TZ

### Community 349 - "run_mcar_draws_queue.sh"
Cohesion: 0.29
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_mcar_draws_queue.sh script, TZ

### Community 350 - "run_mnar_family_queue.sh"
Cohesion: 0.29
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_mnar_family_queue.sh script, TZ

### Community 351 - "run_mnar_smoke_queue.sh"
Cohesion: 0.29
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_mnar_smoke_queue.sh script, TZ

### Community 352 - "run_notmiwae_fill_queue.sh"
Cohesion: 0.29
Nodes (6): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_notmiwae_fill_queue.sh script, STATE, TZ

### Community 353 - "run_null_gaps_queue.sh"
Cohesion: 0.29
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_null_gaps_queue.sh script, TZ

### Community 354 - "run_occlusion_queue.sh"
Cohesion: 0.29
Nodes (6): MKL_NUM_THREADS, MLFLOW_TRACKING_URI, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_occlusion_queue.sh script, TZ

### Community 355 - "run_stage4_helper.sh"
Cohesion: 0.29
Nodes (6): MKL_NUM_THREADS, MLFLOW_TRACKING_URI, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_stage4_helper.sh script, TZ

### Community 356 - "run_stage4_queue.sh"
Cohesion: 0.29
Nodes (6): MKL_NUM_THREADS, MLFLOW_TRACKING_URI, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_stage4_queue.sh script, TZ

### Community 357 - "run_stage4_sweep.sh"
Cohesion: 0.29
Nodes (6): MKL_NUM_THREADS, MLFLOW_TRACKING_URI, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_stage4_sweep.sh script, TZ

### Community 358 - "run_stage4_variants.sh"
Cohesion: 0.29
Nodes (6): MKL_NUM_THREADS, MLFLOW_TRACKING_URI, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_stage4_variants.sh script, TZ

### Community 359 - "run_vae_mnar_queue.sh"
Cohesion: 0.29
Nodes (6): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_vae_mnar_queue.sh script, STATE, TZ

### Community 360 - "make_mimic_complete.py"
Cohesion: 0.40
Nodes (5): complete(), main(), DataFrame, Path, A complete numerical table cut from a natural MIMIC-IV table, to inject known…

### Community 361 - "make_mimic_numeric.py"
Cohesion: 0.40
Nodes (5): main(), numeric(), DataFrame, Path, A natural MIMIC-IV table with its categorical columns made numerical, since…

### Community 362 - "Remove the promoted imputation files that lose to the defaults"
Cohesion: 0.33
Nodes (6): Consequences, Context, Decision, Remove the promoted imputation files that lose to the defaults, Status, Verification

### Community 363 - "Hand the tracking store to a remote server whole, one writer at a time"
Cohesion: 0.25
Nodes (8): Consequences, Context, Decision, Hand the tracking store to a remote server whole, one writer at a time, How to use, Outcome (2026-09-29, rehearsal), Outcome (2026-10-16, decision 8), Status

### Community 364 - "An offline secondary metric panel; impute_score stays the only primary"
Cohesion: 0.33
Nodes (5): An offline secondary metric panel; impute_score stays the only primary, Consequences, Context, Decision, Status

### Community 365 - "C15: `[NULL]` tokens against imputation + indicator under informative missingness (plan, 2026-11-03)"
Cohesion: 0.33
Nodes (6): C15: `[NULL]` tokens against imputation + indicator under informative missingness (plan, 2026-11-03), Constraints, Decisions, Fork changes, in the order the stages need them, Stages, The claim

### Community 366 - "Quick wins and methodology review, 2026-10-16"
Cohesion: 0.33
Nodes (5): Caveats, Digests, Quick wins and methodology review, 2026-10-16, Scripts, Workflow results

### Community 368 - "C15 Stage 1: gate G1 on the positive controls, and A1 against A2 on the development set (E124; pre-registered 2026-11-04)"
Cohesion: 0.33
Nodes (6): Amendment 2026-11-04 01:10 GMT-3 (machines only; the user's "utilize as gpus o melhor possivel"), C15 Stage 1: gate G1 on the positive controls, and A1 against A2 on the development set (E124; pre-registered 2026-11-04), Design (fixed before the first cell), G1's reading (2026-11-04 01:16 GMT-3): fails, 1 of 3; by the rule, C15 stops before Stage 2, Outcome (2026-11-04 01:34 GMT-3): G1 failed (above); Stage 1b, descriptive, Rules, fixed now

### Community 369 - "Thesis table with TabPFN v2 in the strong tier (built by script, 2026-10-21, 17:18 GMT-3)"
Cohesion: 0.22
Nodes (9): Confirmation set (E44, seeds 707/808/909): summary, Development set at 40nan and above, seeds 404/505/606, Development set, seeds 101/202/303, induced impute_score, K readout, Development set, seeds 101/202/303 (primary), Development set, seeds 404/505/606, induced impute_score, K readout, Development set, seeds 42/7/13, Development set, seeds 42/7/13, induced impute_score, K readout, The statement, in five lines (+1 more)

### Community 370 - "Does a value target or a normalised embedding target make pre-training useful? (pre-registered 2026-09-30)"
Cohesion: 0.22
Nodes (9): Arms (fixed before launch), Checked before launch, Does a value target or a normalised embedding target make pre-training useful? (pre-registered 2026-09-30), Known limits, stated in advance, Measures, Operational note, 2026-09-30 01:40 GMT-3: CPU thread limit, Outcome (2026-09-30, 04:28 GMT-3), Question (+1 more)

### Community 371 - "A test-time re-masking ensemble: does averaging passes help, and does their spread rank the errors? (E75; pre-registered 2026-10-24)"
Cohesion: 0.33
Nodes (6): A test-time re-masking ensemble: does averaging passes help, and does their spread rank the errors? (E75; pre-registered 2026-10-24), Design (fixed before the first cell), Launch checks (2026-10-24, before the first cell), Measures and reading, fixed now (`remask_ensemble.py --tally`), Outcome (2026-10-25, 02:38 GMT-3), Post hoc: cells or columns? (2026-10-25, 09:37 GMT-3; not pre-registered)

### Community 372 - "A steps-aware training batch on the small tables (E100, T14; pre-registered 2026-10-30 17:31 GMT-3)"
Cohesion: 0.40
Nodes (5): A steps-aware training batch on the small tables (E100, T14; pre-registered 2026-10-30 17:31 GMT-3), Design (fixed before the first cell), Measures and rule, fixed now (`steps_rule.py --tally`), Outcome, The tally, unedited (`steps_rule.py --tally`, 2026-10-30 18:41 GMT-3)

### Community 373 - "The thesis tables' cells that ADR 0036 changed, rerun on today's default (2026-11-01 16:44 GMT-3)"
Cohesion: 0.40
Nodes (5): Amendment (2026-11-01, 19:09 GMT-3): every cell on gorgona8, Before → after, unedited (`diff` of the scripts' outputs, 2026-11-02 02:49 GMT-3), Outcome, Second amendment (2026-11-01, 23:53 GMT-3): pendigits_60nan joins, The thesis tables' cells that ADR 0036 changed, rerun on today's default (2026-11-01 16:44 GMT-3)

### Community 374 - "Intervals calibrated per bin of a row's gaps (E151, T43; pre-registered 2026-11-06)"
Cohesion: 0.33
Nodes (6): ADR 0030 amended (2026-11-06 16:55 GMT-3), Design (fixed now), Intervals calibrated per bin of a row's gaps (E151, T43; pre-registered 2026-11-06), Measures, Outcome (2026-11-06, 13:05 GMT-3), Reading, fixed now (on the primary 15, under K)

### Community 375 - "test_backfill_decode_metric_tag.py"
Cohesion: 0.36
Nodes (8): client(), fixture, MlflowClient, Stamping ``decode_metric`` on imputation runs recorded before the flag (all…, A parent and an Optuna trial each chose a checkpoint, so they get the stamp,…, _seed_runs(), test_an_existing_mirror_is_stamped_in_place_and_a_missing_one_is_made(), test_every_imputation_run_gets_impute_score_and_nothing_else_is_touched()

### Community 376 - "Datasets read from outside the checkout (T50, stage 0; written 2026-11-05)"
Cohesion: 0.33
Nodes (6): Datasets read from outside the checkout (T50, stage 0; written 2026-11-05), Done when, The change, The toy problem, When, Where things live

### Community 377 - "Two tables from the kg-impute cohort (T50, stage 0; written 2026-11-05)"
Cohesion: 0.33
Nodes (6): Donor tables, for E149, Limits, `mimic-cardio`, `mimic-sofa`, Names and files, Two tables from the kg-impute cohort (T50, stage 0; written 2026-11-05)

### Community 378 - "ReMasker on the MIMIC-IV tables with natural gaps (E161, T53; pre-registered 2026-11-08 10:50 GMT-3)"
Cohesion: 0.33
Nodes (6): Design, Diagnostic after the read (2026-11-08 20:24 GMT-3), Measures and rule (fixed now), Outcome (2026-11-08 13:46 GMT-3), ReMasker on the MIMIC-IV tables with natural gaps (E161, T53; pre-registered 2026-11-08 10:50 GMT-3), The thesis sentence (accepted 2026-11-08 20:24 GMT-3)

### Community 379 - "ADR 0031: An opt-in offline hybrid output: KC and TabPFN, per column"
Cohesion: 0.40
Nodes (5): ADR 0031: An opt-in offline hybrid output: KC and TabPFN, per column, Consequences, Considered options, Context, Decision

### Community 380 - "c15_queue.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, c15_queue.sh script, TZ

### Community 381 - "metric_panel_report.py"
Cohesion: 0.40
Nodes (5): main(), DataFrame, Reading of the secondary metric panel (T24; D16): where the per-kind readings…, Model minus the weak tier's best on each subset of cells, with the subsets'…, split()

### Community 382 - "run_cls_stage2_queue.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_cls_stage2_queue.sh script, TZ

### Community 383 - "run_cross_machine_e94_queue.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_cross_machine_e94_queue.sh script, TZ

### Community 384 - "run_cross_machine_queue.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_cross_machine_queue.sh script, TZ

### Community 385 - "run_deep_baselines_queue.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_deep_baselines_queue.sh script, TZ

### Community 386 - "run_gap_reliance_queue.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_gap_reliance_queue.sh script, TZ

### Community 387 - "run_hybrid_selection_rules_queue.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_hybrid_selection_rules_queue.sh script, TZ

### Community 388 - "run_large_batch_queue.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_large_batch_queue.sh script, TZ

### Community 389 - "run_remask_queue.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_remask_queue.sh script, TZ

### Community 390 - "run_spread_40nan_queue.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_spread_40nan_queue.sh script, TZ

### Community 391 - "run_spread_mechanisms_queue.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_spread_mechanisms_queue.sh script, TZ

### Community 392 - "run_steps_rule_queue.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_steps_rule_queue.sh script, TZ

### Community 393 - "run_tabpfn_tenfold_queue.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_tabpfn_tenfold_queue.sh script, TZ

### Community 394 - "ADR 0030: An opt-in per-cell uncertainty signal from re-masking passes"
Cohesion: 0.40
Nodes (5): ADR 0030: An opt-in per-cell uncertainty signal from re-masking passes, Consequences, Considered options, Context, Decision

### Community 395 - "Score baseline imputers on exactly the cells the decoder is scored on"
Cohesion: 0.25
Nodes (8): Consequences, Considered options, Context, Decision, How to compare, Outcome (2026-09-24/25 and 27: 22 five-fold runs with the change, seed 42, cosine), Score baseline imputers on exactly the cells the decoder is scored on, Status

### Community 396 - "build.py"
Cohesion: 0.50
Nodes (4): main(), Path, Build docs/reports/2026-11-15-reuniao-pesquisadores.html (page) and…, render()

### Community 398 - "Ticket 0005: Batched MLflow Metric Writes (Behavior-Preserving)"
Cohesion: 0.25
Nodes (7): Exactness standard and result, Measured speedup, Problem, Status, Tests, Ticket 0005: Batched MLflow Metric Writes (Behavior-Preserving), What changed

### Community 400 - "C15 Stage 0: the driving columns and gate G2, which variants carry informative missingness? (E122; pre-registered 2026-11-03)"
Cohesion: 0.40
Nodes (5): C15 Stage 0: the driving columns and gate G2, which variants carry informative missingness? (E122; pre-registered 2026-11-03), Data (generated before the first cell; the data directory is gitignored, the scripts are the record), Design (fixed before the first cell), Outcome (2026-11-03 23:45 GMT-3): G2 invalid, two of the three positive controls fail; by the rule, C15 stops here, Rule, fixed now (`c15_stage0.py --tally`)

### Community 401 - "Does the occlusion map hold at 20nan and 80nan? (E85; pre-registered 2026-10-27)"
Cohesion: 0.25
Nodes (8): Amendment (2026-10-27, 20:44 GMT-3): deferred before any cell, Design (fixed before the first cell), Does the occlusion map hold at 20nan and 80nan? (E85; pre-registered 2026-10-27), Measures and reading, Outcome (2026-10-28, 09:14 GMT-3): 80nan refuted; 20nan partial, no verdict, Outcome on all 60 cells (2026-10-28, 11:47 GMT-3): 20nan supported, 80nan refuted, Second amendment (2026-10-28, 00:30 GMT-3): running, 80nan first, on torch 2.11.0, deadline 08:30, Third amendment (2026-10-28, 10:00 GMT-3): the 20nan rate's remaining 20 cells run

### Community 402 - "04-channel-probe.md"
Cohesion: 0.10
Nodes (17): Addendum (2026-11-04 18:52 GMT-3): the loss curves, read before the fix ticket (the user's request), C1's channel probe: a frozen pre-trained encoder against a frozen random one (E132; pre-registered 2026-11-04), Design (fixed before the first cell), Outcome (2026-11-04 18:34 GMT-3): content present, but small; full fine-tuning adds ten times more than pre-training leaves, Rule, fixed now, C1's fine-tuning fix: a gentler fine-tuning against S, and the probe with a converging head (E133; pre-registered 2026-11-04), Design (fixed before the first cell), Outcome (2026-11-04 19:31 GMT-3): neither arm helps; the fine-tuning side is closed (+9 more)

### Community 403 - "The EMA target at 40 and 60nan on fresh seeds (E114; pre-registered 2026-11-02 22:37 GMT-3)"
Cohesion: 0.40
Nodes (5): Design (fixed before the first cell), Measures and rule, fixed now (`cls_ema.py --study e114 --tally`, on medusa1), Outcome, The EMA target at 40 and 60nan on fresh seeds (E114; pre-registered 2026-11-02 22:37 GMT-3), The tally, unedited (`cls_ema.py --study e114 --tally`, 2026-11-03 02:14 GMT-3)

### Community 404 - "Does stopping the decode stage on a validation plateau lose anything? (pre-registered 2026-09-30)"
Cohesion: 0.25
Nodes (8): Arms (fixed before launch), Does stopping the decode stage on a validation plateau lose anything? (pre-registered 2026-09-30), Known limits, stated in advance, Measures, Operational note, 2026-09-30 14:05 GMT-3: a crash at finalisation, fixed, Outcome (2026-09-30, 16:00 GMT-3), Question, Why

### Community 405 - "Does the occlusion map hold under MNAR and MAR? (E88; pre-registered 2026-10-28)"
Cohesion: 0.50
Nodes (4): Design (fixed before the first cell), Does the occlusion map hold under MNAR and MAR? (E88; pre-registered 2026-10-28), Measures and rule, fixed now (`occlusion_map.py --study e88 --tally`), Outcome

### Community 406 - "Does a larger batch with a scaled learning rate keep M's scores? (pre-registered 2026-09-30)"
Cohesion: 0.25
Nodes (7): Arms (fixed before launch), Does a larger batch with a scaled learning rate keep M's scores? (pre-registered 2026-09-30), Known limits, stated in advance, Measures, Outcome (2026-09-30, 15:27 GMT-3), Question, Why

### Community 407 - "Does showing the gaps as [MASK] to pre-training help too? (pre-registered 2026-10-05)"
Cohesion: 0.25
Nodes (7): Design (fixed before launch), Does showing the gaps as [MASK] to pre-training help too? (pre-registered 2026-10-05), Known limits, stated in advance, Measures, Outcome (2026-10-05, 06:20 GMT-3), Running notes, Why

### Community 408 - "Does the re-masking spread make intervals narrower at the same coverage? (E77; pre-registered 2026-10-25)"
Cohesion: 0.40
Nodes (5): Amendment (2026-10-25, about 09:58 GMT-3, before any result was read), Design (fixed now), Does the re-masking spread make intervals narrower at the same coverage? (E77; pre-registered 2026-10-25), Outcome (2026-10-25, 09:59 GMT-3), Reading, fixed now (`remask_ensemble.py --conformal`)

### Community 409 - "null_gaps_report.py"
Cohesion: 0.39
Nodes (7): arm_gap(), cells(), ndarray, queue(), Pre-registered analysis of E67: the gaps as ``[NULL]`` in the decode stage,…, tally(), variants()

### Community 410 - "evaluation_mask"
Cohesion: 0.29
Nodes (8): evaluation_mask(), Hide cells for scoring, the same way every time this fold is scored. Training…, _maskable(), DataFrame, Which cells a fold is scored on is fixed, so its numbers can be compared at…, Evaluation must not consume randomness training was going to use. Every seeded…, test_drawing_an_evaluation_mask_leaves_the_training_draws_alone(), test_scoring_the_same_fold_twice_asks_the_same_question()

### Community 411 - "Does the re-masking spread still rank errors under MNAR and MAR at 40nan? (E105; pre-registered 2026-10-31 10:16 GMT-3)"
Cohesion: 0.40
Nodes (5): Design (fixed before the first cell), Does the re-masking spread still rank errors under MNAR and MAR at 40nan? (E105; pre-registered 2026-10-31 10:16 GMT-3), Measures and rule, fixed now (`spread_40nan.py --tally`), E84's, Outcome, The tally, unedited (`spread_40nan.py --tally`, 2026-10-31 12:11 GMT-3)

### Community 412 - "Does the re-masking spread still rank errors under MNAR and MAR at 20nan? (E110; pre-registered 2026-11-01 23:59 GMT-3)"
Cohesion: 0.50
Nodes (4): Design, measures and rule (fixed now): E105's at 20nan, Does the re-masking spread still rank errors under MNAR and MAR at 20nan? (E110; pre-registered 2026-11-01 23:59 GMT-3), Outcome, The tally, unedited (`spread_40nan.py --level 20 --tally`, 2026-11-02 05:04 GMT-3)

### Community 413 - "TRIDENT against its baselines on `impute_score_classcond`, the score measured against the class-conditional fill (E181, T58; pre-registered 2026-11-17 14:35 GMT-3)"
Cohesion: 0.40
Nodes (5): Amendments, Design, Measures and rule (fixed now), Outcome, TRIDENT against its baselines on `impute_score_classcond`, the score measured against the class-conditional fill (E181, T58; pre-registered 2026-11-17 14:35 GMT-3)

### Community 414 - "Real gap patterns transplanted onto complete rows (E149; pre-registered 2026-11-05 as E151)"
Cohesion: 0.40
Nodes (5): Design and rule (fixed before the first cell), Implementation (2026-11-06, before any cell), Outcome (2026-11-06 15:14 GMT-3), Real gap patterns transplanted onto complete rows (E149; pre-registered 2026-11-05 as E151), Run

### Community 415 - "C15's positive-control gate on the SOFA table's natural gaps (E152; pre-registered 2026-11-06)"
Cohesion: 0.40
Nodes (5): C15's positive-control gate on the SOFA table's natural gaps (E152; pre-registered 2026-11-06), Design (fixed before the first cell), Launch note (2026-11-06 13:19 GMT-3, before the first cell), Outcome (2026-11-06 16:34 GMT-3): G1 fails, Rule, fixed now (E128's, one table)

### Community 416 - "credit-g_80nan ranked by the induced validation gaps (pre-registered 2026-09-28)"
Cohesion: 0.29
Nodes (7): credit-g_80nan ranked by the induced validation gaps (pre-registered 2026-09-28), Design (fixed before launch), Known limits, stated in advance, Measures, Outcome (2026-09-28, study 22:11 to 22:31, retrain to 22:33), Question, Why

### Community 417 - "Overnight full-profile studies against the baseline bar (pre-registered 2026-09-28)"
Cohesion: 0.29
Nodes (7): Design (fixed before launch), Known limits, stated in advance, Measures, Outcome (2026-09-28; the queue completed at 20:10), Overnight full-profile studies against the baseline bar (pre-registered 2026-09-28), Question, Why

### Community 418 - "The induced validation objective on the other six variants (pre-registered 2026-09-28)"
Cohesion: 0.29
Nodes (7): Design (fixed before launch), Known limits, stated in advance, Measures (per variant, as in 02), Outcome (2026-09-29; stopped before the last variant), Question, The induced validation objective on the other six variants (pre-registered 2026-09-28), Why

### Community 419 - "Does normalised-target pre-training still help once the decoder gets 450 epochs? (pre-registered 2026-09-30)"
Cohesion: 0.29
Nodes (7): Arms (fixed before launch), Does normalised-target pre-training still help once the decoder gets 450 epochs? (pre-registered 2026-09-30), Known limits, stated in advance, Measures, Outcome (2026-09-30, 05:00 GMT-3), Question, Why

### Community 420 - "run_e155_chain.sh"
Cohesion: 0.80
Nodes (4): cells(), queue(), say(), run_e155_chain.sh script

### Community 421 - "run_e156_chain.sh"
Cohesion: 0.80
Nodes (4): cells(), queue(), say(), run_e156_chain.sh script

### Community 422 - "run_imputer_utility_chain.sh"
Cohesion: 0.60
Nodes (4): progress(), say(), run_imputer_utility_chain.sh script, TZ

### Community 425 - "metricskeptic_e35.py"
Cohesion: 0.83
Nodes (3): call(), parts(), run()

### Community 427 - "C15 Stage 0, second gate: G2 read out of fold over every training and validation row (E123; pre-registered 2026-11-03)"
Cohesion: 0.50
Nodes (4): C15 Stage 0, second gate: G2 read out of fold over every training and validation row (E123; pre-registered 2026-11-03), Design (fixed before the first cell), Outcome (2026-11-04 00:12 GMT-3): G2 valid; P holds 5 variants, Rule, fixed now (unchanged from E122)

### Community 428 - "TabPFN v2 as an external foundation-model baseline (E55; pre-registered 2026-10-20)"
Cohesion: 0.29
Nodes (7): Design (fixed before the first scored cell), Launch, Measures, Outcome (2026-10-21, 03:51 GMT-3), Reading, fixed now (E48's), TabPFN v2 as an external foundation-model baseline (E55; pre-registered 2026-10-20), The code (landed with this ticket)

### Community 429 - "C15 Stage 3: confirmation, read once: A1 against A2 on fresh draws of P and of its MCAR siblings (E126; pre-registered 2026-11-04)"
Cohesion: 0.50
Nodes (4): C15 Stage 3: confirmation, read once: A1 against A2 on fresh draws of P and of its MCAR siblings (E126; pre-registered 2026-11-04), Data and design (fixed before the first cell), Rule, fixed now (the protocol's H2, pooled tables), Withdrawn unrun (2026-11-04 01:16 GMT-3)

### Community 430 - "C15's gate G1 again: A2 against A3 in decision mode (E128; pre-registered 2026-11-04)"
Cohesion: 0.50
Nodes (4): C15's gate G1 again: A2 against A3 in decision mode (E128; pre-registered 2026-11-04), Design (fixed before the first cell), Outcome (2026-11-04 08:26 GMT-3): G1 fails again, 1 of 3; by the binding clause, C15 stops for good, Rule, fixed now

### Community 431 - "plan_run"
Cohesion: 0.29
Nodes (7): best_statistics(), plan_best(), plan_run(), Collection, Check the recomputation against the run's own baselines, then list what is…, The ``baseline/best`` statistics and tags a run lacks, from what it already…, For each named population, the baseline of the bar with the lowest mean…

### Community 433 - "Let pre-training regress a masked cell onto its value, or onto a normalised embedding"
Cohesion: 0.33
Nodes (6): Consequences, Considered options, Context, Decision, Let pre-training regress a masked cell onto its value, or onto a normalised embedding, Status

### Community 434 - "Two tiers of baseline imputers, every one logged, each tier with its own best"
Cohesion: 0.33
Nodes (6): Consequences, Considered options, Context, Decision, Status, Two tiers of baseline imputers, every one logged, each tier with its own best

### Community 435 - "01. Per-task fold-ranking contract"
Cohesion: 0.33
Nodes (5): 01. Per-task fold-ranking contract, Acceptance criteria, Comments, Goal, Seams under test

### Community 436 - "11. Regression fixture and documentation"
Cohesion: 0.33
Nodes (6): 11. Regression fixture and documentation, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 437 - "Does a decoder that can tell a real gap from its own masks read the gap pattern under MNAR? (E81; pre-registered 2026-10-27)"
Cohesion: 0.33
Nodes (6): Design (fixed before the first cell), Does a decoder that can tell a real gap from its own masks read the gap pattern under MNAR? (E81; pre-registered 2026-10-27), Limits fixed now, Outcome (2026-10-27, 12:29 GMT-3): mixed, Rule, fixed now (`gap_reliance.py --tally`), The measure

### Community 438 - "Today's default on fresh seeds at 20nan (E51; pre-registered 2026-10-19)"
Cohesion: 0.33
Nodes (6): Design (fixed before the first cell), Launch, Measures, Outcome (2026-10-20, 08:13 GMT-3), Reading, fixed now, Today's default on fresh seeds at 20nan (E51; pre-registered 2026-10-19)

### Community 439 - "The strong tier with TabPFN on the seed sets E55 did not cover (E58; pre-registered 2026-10-21)"
Cohesion: 0.33
Nodes (5): Design (fixed before the first cell), Launch, Outcome (2026-10-21, 14:09 GMT-3), Reading, fixed now (descriptive; ADR 0028 stands), The strong tier with TabPFN on the seed sets E55 did not cover (E58; pre-registered 2026-10-21)

### Community 440 - "frozen_encoder_run.py"
Cohesion: 0.47
Nodes (5): changes(), _check_frozen(), main(), One cell of E52, T28's D3: a frozen encoder, pre-trained against random…, Wrap the decode stage so each fold asserts its encoder came out unchanged.

### Community 441 - "Read every imputation comparison one way: verdict tallies per opponent, the test named"
Cohesion: 0.40
Nodes (5): Consequences, Context, Decision, Read every imputation comparison one way: verdict tallies per opponent, the test named, Status

### Community 442 - "A validation-only tuning budget for the baselines, co-reported offline"
Cohesion: 0.40
Nodes (5): A validation-only tuning budget for the baselines, co-reported offline, Consequences, Context, Decision, Status

### Community 443 - "Thesis table on the development set (built by script, 2026-10-17; regenerated 2026-10-18 23:09 GMT-3)"
Cohesion: 0.40
Nodes (4): Development set, seeds 101/202/303, induced impute_score, K readout, Development set, seeds 404/505/606, induced impute_score, K readout, Development set, seeds 42/7/13, induced impute_score, K readout, Thesis table on the development set (built by script, 2026-10-17; regenerated 2026-10-18 23:09 GMT-3)

### Community 444 - "Observed cells hidden on tables with natural gaps (E148; pre-registered 2026-11-05 as E150)"
Cohesion: 0.50
Nodes (4): Design and rule (fixed before the first cell), Observed cells hidden on tables with natural gaps (E148; pre-registered 2026-11-05 as E150), Outcome (2026-11-06 10:27 GMT-3), Run

### Community 445 - "C15 Stage 4 (optional, secondary): A4 on the dial, the sign arm, A3 and the null model A5 (E127; pre-registered 2026-11-04)"
Cohesion: 0.40
Nodes (5): C15 Stage 4 (optional, secondary): A4 on the dial, the sign arm, A3 and the null model A5 (E127; pre-registered 2026-11-04), Data (generated before the first cell), Design (fixed before the first cell), Readings (descriptive, fixed now), Withdrawn unrun (2026-11-04 01:16 GMT-3)

### Community 446 - "ADR 0030's binned interval under real gap patterns (E153; pre-registered 2026-11-06)"
Cohesion: 0.50
Nodes (4): ADR 0030's binned interval under real gap patterns (E153; pre-registered 2026-11-06), Design (fixed now), Outcome (2026-11-06 17:13 GMT-3): refuted, one width per column already covers, Reading, fixed now (on `tmatch`, the two donor tables)

### Community 447 - "ADR 0030's binned interval on the natural gaps' rows (E154; pre-registered 2026-11-06)"
Cohesion: 0.50
Nodes (4): ADR 0030's binned interval on the natural gaps' rows (E154; pre-registered 2026-11-06), Design (fixed now), Outcome (2026-11-06 18:08 GMT-3): refuted, one width per column already covers, Reading, fixed now (the two tables)

### Community 448 - "ADR 0018's bigger encoder on the large tables at 40–80nan (E59; pre-registered 2026-10-21)"
Cohesion: 0.40
Nodes (5): ADR 0018's bigger encoder on the large tables at 40–80nan (E59; pre-registered 2026-10-21), Decision (2026-10-22, 06:10 GMT-3), Design (fixed before the first cell), Measures and reading, fixed now, Outcome (2026-10-22, 06:03 GMT-3)

### Community 449 - "Map: Reduced Optuna search for the imputation task"
Cohesion: 0.33
Nodes (6): Decisions so far, Destination, Map: Reduced Optuna search for the imputation task, Not yet specified, Notes, Out of scope

### Community 450 - "imputer_utility_levels.py"
Cohesion: 0.50
Nodes (3): main(), Path, E147 (ticket imputation-protocol/53; run as E149), post hoc and descriptive…

### Community 451 - "run_mimic_chain.sh"
Cohesion: 0.67
Nodes (3): say(), run_mimic_chain.sh script, TZ

### Community 452 - "run_sofa_g1_chain.sh"
Cohesion: 1.00
Nodes (3): queue(), say(), run_sofa_g1_chain.sh script

### Community 453 - "run_tabpfn_lanes.py"
Cohesion: 0.67
Nodes (3): main(), minutes(), Start E55's offline TabPFN v2 pass in GPU lanes (imputation-protocol/10),…

### Community 455 - "Experimentation log"
Cohesion: 0.67
Nodes (3): Experimentation log, Index, Template

### Community 461 - "Where does K lose to TabPFN on kr-vs-kp and credit-g? A per-column read (E61; pre-registered 2026-10-22)"
Cohesion: 0.40
Nodes (5): Before the ticket (disclosed), Design (fixed before the first cell), Outcome (2026-10-22, 06:33 GMT-3), Reading, fixed now (`categorical_deficit.py --tally`), Where does K lose to TabPFN on kr-vs-kp and credit-g? A per-column read (E61; pre-registered 2026-10-22)

### Community 479 - "Fresh MCAR draws: is a result tied to one missingness pattern? (E62; pre-registered 2026-10-22)"
Cohesion: 0.40
Nodes (5): Design (fixed before the first cell), Fresh MCAR draws: is a result tied to one missingness pattern? (E62; pre-registered 2026-10-22), Launch, Measures and reading, fixed now (`mcar_draws_report.py --tally`), Outcome (2026-10-22, 11:39 GMT-3)

### Community 482 - "The gaps as `[NULL]` in the decode stage, alone: does the token help under MNAR? (E67; pre-registered 2026-10-23)"
Cohesion: 0.40
Nodes (5): Design (fixed before the first cell), Launch, Measures and reading, fixed now (`null_gaps_report.py --tally`), Outcome (2026-10-23, 21:53 GMT-3), The gaps as `[NULL]` in the decode stage, alone: does the token help under MNAR? (E67; pre-registered 2026-10-23)

### Community 483 - "The tuned bar on the MAR folds: D4 under MAR (E68; pre-registered 2026-10-23)"
Cohesion: 0.40
Nodes (5): Design (fixed before the first pair), Measures and reading, fixed now (`mar_tuned_bar_report.py --tally`), Outcome (2026-10-23, 23:54 GMT-3), The tuned bar on the MAR folds: D4 under MAR (E68; pre-registered 2026-10-23), The user's decision (2026-10-24)

### Community 484 - "02. Research: how do tabular masked-modeling transformers decode a masked cell back to a value?"
Cohesion: 0.40
Nodes (4): 02. Research: how do tabular masked-modeling transformers decode a masked cell back to a value?, Answer, Comments, Question

### Community 485 - "Per-column weights for the K + TabPFN blend, cross-fitted: do they reach past TabPFN where the equal blend could not? (E69; pre-registered 2026-10-23)"
Cohesion: 0.40
Nodes (5): Design (fixed before the first cell), Launch, Measures and reading, fixed now (`blend_weights.py --tally`), Outcome (2026-10-24, 00:37 GMT-3), Per-column weights for the K + TabPFN blend, cross-fitted: do they reach past TabPFN where the equal blend could not? (E69; pre-registered 2026-10-23)

### Community 486 - "Does the spread grow when a cell's top sources are missing? (E80; pre-registered 2026-10-27)"
Cohesion: 0.50
Nodes (4): Design (fixed before the reading), Does the spread grow when a cell's top sources are missing? (E80; pre-registered 2026-10-27), Measure and rule, fixed now (`occlusion_spread.py`), Outcome (2026-10-27, 08:33 GMT-3): mixed

### Community 487 - "Reduced Optuna search for imputation: implementation tickets"
Cohesion: 0.50
Nodes (4): Global constraints (apply to every ticket), Reduced Optuna search for imputation: implementation tickets, Status vocabulary, Tickets

### Community 488 - "An imputer that models the mechanism: does not-MIWAE shrink the MNAR bias? (E72; pre-registered 2026-10-24)"
Cohesion: 0.40
Nodes (5): An imputer that models the mechanism: does not-MIWAE shrink the MNAR bias? (E72; pre-registered 2026-10-24), Design (fixed before the first cell), Launch, Measures and reading, fixed now (`vae_mnar_report.py --tally`), Outcome (2026-10-24, 05:33 GMT-3)

### Community 489 - "not-MIWAE with a mean-based fill: does modelling the mechanism shrink the MNAR bias on biodeg and spambase? (E73; pre-registered 2026-10-24)"
Cohesion: 0.40
Nodes (5): Design (fixed before the first cell), Launch, Measures and reading, fixed now (`notmiwae_fill_report.py --tally`), not-MIWAE with a mean-based fill: does modelling the mechanism shrink the MNAR bias on biodeg and spambase? (E73; pre-registered 2026-10-24), Outcome (2026-10-24, 13:53 GMT-3)

### Community 490 - "Does the re-masking spread rank cells on the development variants? (E76; pre-registered 2026-10-25)"
Cohesion: 0.40
Nodes (5): Amendment (2026-10-26, 16:51 GMT-3): the machine went down mid-run, Design (fixed before the first cell), Does the re-masking spread rank cells on the development variants? (E76; pre-registered 2026-10-25), Measures and reading, fixed now (`remask_ensemble.py --study e76 --tally`), Outcome (2026-10-26, 22:15 GMT-3)

### Community 491 - "Can the hybrid's weights be chosen without the complete table? (E82; pre-registered 2026-10-27)"
Cohesion: 0.40
Nodes (5): Can the hybrid's weights be chosen without the complete table? (E82; pre-registered 2026-10-27), Design (fixed before the first cell), Limits fixed now, Measures and reading, fixed now (`deployable_hybrid.py --tally`), Outcome (2026-10-27, 18:42 GMT-3): refuted

### Community 492 - "36-hybrid-20nan.md"
Cohesion: 0.40
Nodes (4): Design (fixed before the first cell), Does the deployable `table` hybrid hold at 20nan? (E95; pre-registered 2026-10-29), Measures and rule, fixed now (`hybrid_confirmation.py --study e95 --tally`), Outcome

### Community 493 - "Does the cross-fit pick remove the hybrid's weakness off MCAR? (E170, T54; pre-registered 2026-11-09 19:59 GMT-3)"
Cohesion: 0.40
Nodes (5): Amendment, Design, Does the cross-fit pick remove the hybrid's weakness off MCAR? (E170, T54; pre-registered 2026-11-09 19:59 GMT-3), Measures and rule (fixed now), Outcome (read once 2026-11-10 12:41 GMT-3: mixed; written 2026-11-10 12:44 GMT-3)

### Community 494 - "04. Decoder architecture and the stage that trains it"
Cohesion: 0.50
Nodes (4): 04. Decoder architecture and the stage that trains it, Answer, Comments, Question

### Community 495 - "The deployable hybrid on the MIMIC-IV transplant variants (E163, T55; pre-registered 2026-11-08 21:08 GMT-3)"
Cohesion: 0.40
Nodes (5): Decision, Design, Measures and rule (fixed now), Outcome (read once 2026-11-09 00:09 GMT-3): supported, The deployable hybrid on the MIMIC-IV transplant variants (E163, T55; pre-registered 2026-11-08 21:08 GMT-3)

### Community 496 - "06. How are imputation runs identified in MLflow?"
Cohesion: 0.50
Nodes (4): 06. How are imputation runs identified in MLflow?, Answer, Comments, Question

### Community 497 - "07. Prototype: the human-friendly imputed-vs-actual preview"
Cohesion: 0.50
Nodes (4): 07. Prototype: the human-friendly imputed-vs-actual preview, Answer, Comments, Question

### Community 498 - "08. CLI flag and hyperparameter schema for the imputation task"
Cohesion: 0.50
Nodes (4): 08. CLI flag and hyperparameter schema for the imputation task, Answer, Comments, Question

### Community 499 - "09. How does the training package branch per task without touching classification?"
Cohesion: 0.50
Nodes (4): 09. How does the training package branch per task without touching classification?, Answer, Comments, Question

### Community 500 - "TRIDENT on 10,000 rows of `mimic-wide`: does the lead over TabPFN v2 need the other rows? (E177, T57; pre-registered 2026-11-13 08:28 GMT-3)"
Cohesion: 0.40
Nodes (5): Design, Measures and rule (fixed now), Outcome (2026-11-13 11:29 GMT-3): supported, The flag (commit 9e780d3), TRIDENT on 10,000 rows of `mimic-wide`: does the lead over TabPFN v2 need the other rows? (E177, T57; pre-registered 2026-11-13 08:28 GMT-3)

### Community 501 - "T29 · Why the boosters win on electricity"
Cohesion: 0.50
Nodes (4): In five lines, Printout, Reading the per-column tables, T29 · Why the boosters win on electricity

### Community 502 - "TabPFN v2 under informative missingness: native and explicit-mask arms beside A1 (E137, T44; pre-registered 2026-11-04)"
Cohesion: 0.50
Nodes (4): Design (fixed before the first fit), Outcome (2026-11-04, 23:22 GMT-3; 69 of 69 pairs, gorgona8 10 done 23:09 GMT-3, medusa1 59 done before 23:15 GMT-3), Rule, fixed now, TabPFN v2 under informative missingness: native and explicit-mask arms beside A1 (E137, T44; pre-registered 2026-11-04)

### Community 503 - "H2 under outcome-dependent MNAR: TabPFN and A1 on F3 against same-seed MCAR siblings (E139; pre-registered 2026-11-04)"
Cohesion: 0.50
Nodes (4): Design (fixed before the first cell), H2 under outcome-dependent MNAR: TabPFN and A1 on F3 against same-seed MCAR siblings (E139; pre-registered 2026-11-04), Outcome (2026-11-05, 00:34 GMT-3; 9 of 9 A1 cells and 9 of 9 TabPFN pairs; gorgona8 done 00:28 GMT-3, medusa1 by 00:33 GMT-3 after its GPUs freed at 00:21), Rule, fixed now

### Community 504 - "01. The held set and the ranges"
Cohesion: 0.50
Nodes (4): 01. The held set and the ranges, Answer, Comments, Question

### Community 505 - "02. The search objective is scored on the validation split"
Cohesion: 0.50
Nodes (4): 02. The search objective is scored on the validation split, Answer, Comments, Question

### Community 506 - "03. Study protocol: datasets, variants, budget, schedule, seed"
Cohesion: 0.50
Nodes (4): 03. Study protocol: datasets, variants, budget, schedule, seed, Answer, Comments, Question

### Community 507 - "04. Where a promoted configuration lives and how a run finds it"
Cohesion: 0.50
Nodes (4): 04. Where a promoted configuration lives and how a run finds it, Answer, Comments, Question

### Community 508 - "05. The search-space profile flag and its record"
Cohesion: 0.50
Nodes (4): 05. The search-space profile flag and its record, Answer, Comments, Question

### Community 509 - "06. Checking the reduction and proving the tuning helped"
Cohesion: 0.50
Nodes (4): 06. Checking the reduction and proving the tuning helped, Answer, Comments, Question

### Community 510 - "07. Task: repair the imputation Optuna path and prove it with a two-trial study"
Cohesion: 0.50
Nodes (4): 07. Task: repair the imputation Optuna path and prove it with a two-trial study, Answer, Comments, Question

### Community 511 - "Does pre-training open a channel when labels are scarce? S, T and B on a tenth and a quarter of the training labels (E136; pre-registered 2026-11-04)"
Cohesion: 0.50
Nodes (4): Design (fixed before the first cell), Does pre-training open a channel when labels are scarce? S, T and B on a tenth and a quarter of the training labels (E136; pre-registered 2026-11-04), Outcome (2026-11-04, 23:24 GMT-3; medusa1 120 cells done 23:15 GMT-3, gorgona8 24 done 23:20 GMT-3, 0 failed), Rule, fixed now

### Community 512 - "The quarter-label channel on fresh seeds, with the validation split scarce too (E138; pre-registered 2026-11-04)"
Cohesion: 0.50
Nodes (4): Design (fixed before the first cell), Outcome (2026-11-05, 00:30 GMT-3; 48 of 48 cells on gorgona8, done 00:28 GMT-3, 0 failed), Rule, fixed now, The quarter-label channel on fresh seeds, with the validation split scarce too (E138; pre-registered 2026-11-04)

### Community 514 - "The selector × seeds squares: is the scarce-validation selector what opens a tenth? (E144; pre-registered 2026-11-05)"
Cohesion: 0.50
Nodes (4): Amendment (2026-11-05, 08:15 GMT-3), Design and rule (fixed before the first cell), Outcome (2026-11-05, 09:40 GMT-3; 168 of 168 cells, 0 failed; gorgona8's first 76 by 03:56, the moved 92 from 08:30 to 09:34 GMT-3), The selector × seeds squares: is the scarce-validation selector what opens a tenth? (E144; pre-registered 2026-11-05)

### Community 515 - "Imputation decoder task: implementation tickets"
Cohesion: 0.50
Nodes (4): Global constraints (apply to every ticket), Imputation decoder task: implementation tickets, Status vocabulary, Tickets

### Community 516 - "The default output against TabPFN: KC and H (E57; pre-registered 2026-10-21)"
Cohesion: 0.50
Nodes (3): Outcome (2026-10-21, 10:04 GMT-3), Reading, fixed now, The default output against TabPFN: KC and H (E57; pre-registered 2026-10-21)

### Community 517 - "Are TRIDENT's and TabPFN's errors complementary? A blend of their fills (E60; pre-registered 2026-10-21)"
Cohesion: 0.50
Nodes (4): Are TRIDENT's and TabPFN's errors complementary? A blend of their fills (E60; pre-registered 2026-10-21), Design (fixed before the first cell), Outcome (2026-10-22, 00:05 GMT-3), Reading, fixed now

### Community 518 - "A MAR smoke against matched MCAR controls (E63; pre-registered 2026-10-22)"
Cohesion: 0.50
Nodes (4): A MAR smoke against matched MCAR controls (E63; pre-registered 2026-10-22), Design (fixed before the first cell), Measures and reading, fixed now (`mar_smoke_report.py --tally`), Outcome (2026-10-22, 15:12 GMT-3)

### Community 519 - "The MAR family: does MAR cost the model ground against the strong tier? (E64; pre-registered 2026-10-22)"
Cohesion: 0.50
Nodes (4): Design (fixed before the first cell), Measures and reading, fixed now (`mar_family_report.py --tally`), Outcome (2026-10-23, 04:19 GMT-3), The MAR family: does MAR cost the model ground against the strong tier? (E64; pre-registered 2026-10-22)

### Community 520 - "Does E60's complementarity hold on other tables and seeds? The blend on E40's set (E66; pre-registered 2026-10-23)"
Cohesion: 0.50
Nodes (4): Design (fixed before the first cell), Does E60's complementarity hold on other tables and seeds? The blend on E40's set (E66; pre-registered 2026-10-23), Outcome (2026-10-23, 07:58 GMT-3), Reading, fixed now (`tabpfn_blend.py --tally --set e40`)

### Community 521 - "Can fewer or better-fed choices close the deployable hybrid's gap to TabPFN? (E89; pre-registered 2026-10-28)"
Cohesion: 0.50
Nodes (4): Can fewer or better-fed choices close the deployable hybrid's gap to TabPFN? (E89; pre-registered 2026-10-28), Design (fixed before the first lane), Measures and rule, fixed now (`hybrid_selection_rules.py --tally`), Outcome

### Community 522 - "Does the deployable hybrid's `table` rule hold on fresh seeds, and where it is not needed? (E90, E91; pre-registered 2026-10-28)"
Cohesion: 0.50
Nodes (4): Design (fixed before the queues), Does the deployable hybrid's `table` rule hold on fresh seeds, and where it is not needed? (E90, E91; pre-registered 2026-10-28), Measures and rule, fixed now (`hybrid_confirmation.py --study <e90|e91> --tally`), Outcome

### Community 523 - "Does the deployable `table` hybrid hold under MNAR and MAR? (E93; pre-registered 2026-10-29)"
Cohesion: 0.50
Nodes (4): Design (fixed before the queues), Does the deployable `table` hybrid hold under MNAR and MAR? (E93; pre-registered 2026-10-29), Measures and rule, fixed now (`hybrid_confirmation.py --study e93 --tally`), Outcome

### Community 524 - "Which score orders imputers as a classifier does, across TRIDENT, the strong baselines and three mechanisms? (E147, a pilot; pre-registered 2026-11-05 as E149)"
Cohesion: 0.50
Nodes (4): Amendment (2026-11-05, 22:40 GMT-3), Design and rule (fixed before the first fit), Outcome (2026-11-06, 06:27 GMT-3; 60 of 60 cells and 396 of 396 files, none failed), Which score orders imputers as a classifier does, across TRIDENT, the strong baselines and three mechanisms? (E147, a pilot; pre-registered 2026-11-05 as E149)

### Community 525 - "The cross-fit side pick on E93's off-MCAR variants, fresh seeds (E172, T54; pre-registered 2026-11-09 21:18 GMT-3)"
Cohesion: 0.50
Nodes (4): Design, Measures and rule (fixed now), Outcome (read once 2026-11-10 04:01 GMT-3: mixed; written 2026-11-10 04:02 GMT-3), The cross-fit side pick on E93's off-MCAR variants, fresh seeds (E172, T54; pre-registered 2026-11-09 21:18 GMT-3)

### Community 526 - "task_defaults"
Cohesion: 0.50
Nodes (4): default_lr_scheduler(), The values a task's runs take where their configuration names none., The schedule a task's runs follow when neither a flag nor a configuration names…, task_defaults()

### Community 527 - "The label-fraction curve under a scarce validation split: a tenth and a half, fresh seeds (E140; pre-registered 2026-11-05)"
Cohesion: 0.67
Nodes (3): Design and rule (fixed before the first cell), Outcome (2026-11-05, 01:25 GMT-3; 96 of 96 cells, gorgona8 32 and medusa1 64, cut 10 done on both by 01:08 GMT-3, 0 failed), The label-fraction curve under a scarce validation split: a tenth and a half, fresh seeds (E140; pre-registered 2026-11-05)

### Community 528 - "Stage 2: the quarter-label channel on the 21 confirmatory variants, five seeds (E141; pre-registered 2026-11-05)"
Cohesion: 0.67
Nodes (3): Design and rule (fixed before the first cell), Outcome (2026-11-05, 02:43 GMT-3; 210 of 210 cells, gorgona8 42 and medusa1 168, cut 11 done on medusa1 02:13 and on gorgona8 02:20 GMT-3, 0 failed), Stage 2: the quarter-label channel on the 21 confirmatory variants, five seeds (E141; pre-registered 2026-11-05)

### Community 529 - "The quarter-label channel at 20, 40 and 80% missingness (E143; pre-registered 2026-11-05)"
Cohesion: 0.67
Nodes (3): Design and rule (fixed before the first cell), Outcome (2026-11-05, 02:43 GMT-3; 72 of 72 cells, gorgona8 24 and medusa1 48, cut 12 done on medusa1 02:24 and on gorgona8 02:38 GMT-3, 0 failed), The quarter-label channel at 20, 40 and 80% missingness (E143; pre-registered 2026-11-05)

### Community 530 - "Which fills serve a downstream classifier: the checkpoint by `impute_score` or by `dist_score`? (E146, a pilot; pre-registered 2026-11-05)"
Cohesion: 0.67
Nodes (3): Design and rule (fixed before the first fit), Outcome (2026-11-05, 21:12 GMT-3; 36 of 36 (variant, seed) files, run on medusa1's CPUs at 21:11 GMT-3), Which fills serve a downstream classifier: the checkpoint by `impute_score` or by `dist_score`? (E146, a pilot; pre-registered 2026-11-05)

### Community 531 - "_BudgetSpent"
Cohesion: 0.67
Nodes (3): Exception, _BudgetSpent, The pass has recomputed as many folds as it was allowed to; the cache keeps…

### Community 532 - "_matrix"
Cohesion: 0.67
Nodes (3): _matrix(), DataFrame, ndarray

## Ambiguous Edges - Review These
- `src/training Package` → `TRIDENT README`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to
- `kr-vs-kp Categorical Columns` → `TRIDENT README`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to

## Knowledge Gaps
- **2133 isolated node(s):** `tables_manifest.sh script`, `info`, `trident`, `c15_chain4.sh script`, `c15_queue.sh script` (+2128 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 3903 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **41 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `src/training Package` and `TRIDENT README`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `kr-vs-kp Categorical Columns` and `TRIDENT README`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `SyncError` connect `StoreError` to `test_sync_remote.py`, `test_leadership.py`, `leadership.py`, `RsyncTransport`, `sync_remote.py`, `tally`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Why does `build_folds()` connect `DatasetSpec` to `pretrain_probe.py`, `interval`, `Hyperparameters`, `electricity_split.py`, `backfill_baseline_imputers.py`, `types.py`, `model_for`, `BaselineImputer`, `cls_stage1_bar.py`, `categorical_deficit.py`, `deep_baselines.py`, `stage4_gbdt.py`, `joint_draws.py`, `c15_stage0.py`, `Counter`, `fold_metrics`, `sliced_run.py`, `tally`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Why does `DatasetSpec` connect `DatasetSpec` to `interval`, `Hyperparameters`, `run_hyperparameter_optimization`, `backfill_baseline_imputers.py`, `electricity_split.py`, `types.py`, `model_for`, `BaselineImputer`, `cls_stage1_bar.py`, `categorical_deficit.py`, `deep_baselines.py`, `stage4_gbdt.py`, `complete_configuration`, `joint_draws.py`, `sliced_run.py`, `fold_metrics`, `tally`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Are the 38 inferred relationships involving `DatasetSpec` (e.g. with `main()` and `run_hyperparameter_optimization()`) actually correct?**
  _`DatasetSpec` has 38 INFERRED edges - model-reasoned connections that need verification._
- **What connects `tables_manifest.sh script`, `info`, `trident` to the rest of the system?**
  _2133 weakly-connected nodes found - possible documentation gaps or missing edges._