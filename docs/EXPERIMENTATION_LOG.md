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
| E34 | Checkpoint by the validation gaps, and calibration on them | 2026-10-03 | done | K (checkpoint by validation gaps) qualifies: better on 3 (all 80nan), worse on none; C better on 13 but worse on spambase_20nan, so it fails; KC loses to K there too |
| E35 | The gap-chosen checkpoint on E32's seeds | 2026-10-04 | done | replicated: K better on 6 of 21 (2 of E34's 3), worse on none, on seeds 42/7/13; pooled six seeds 6/0; ADR 0015 stands; calibration fails on spambase_20nan again |
| E36 | Showing the gaps as `[MASK]` to pre-training too | 2026-10-05 | done | no gain: PG better on none, worse on electricity_20nan (+0.011); means 0.7450 vs 0.7446; keep `null` |
| E37 | Architecture screening: deeper, wider or normed encoder | 2026-10-05 – 2026-10-06 | done | ALL (4 layers + FF 256 + final LN) better on 3 of 5, worse on none (credit-g_20nan a near miss); F256 better on 1; L4 and LN worse on 1 each; ALL advances, 1.82x the time |
| E38 | The bigger encoder by table size, on fresh seeds | 2026-10-06 – 2026-10-07 | done | supported: large −3.0% (pendigits −3.8%, letter −5.6%, electricity +0.3% n.s.); medium −0.2% (1 better); small +1.9% (4 worse); paused for ADR 0017, then completed |
| E39 | Re-tuning the promoted files under today's defaults | 2026-10-15 – 2026-10-16 | done | stale files hurt: no file beats them on 3 of 5 (kr-vs-kp_40nan −4.8%, credit-g_40nan −4.7%, spambase_40nan −0.5%); the re-tuned winners beat the defaults nowhere; rule: promote T on 2, remove the file on 1, keep 2 |
| E40 | Calibration at 40% missing and above, on fresh seeds | 2026-10-16 | done | supported: KC better on 6 of 6 named (7 of 12), K better on none; gain grows with missingness (40nan −0.0009, 60nan −0.0037, 80nan −0.0059), none at 40nan earns a verdict |
| E41 | Linear baselines (condexp, SoftImpute, ICE) and the missing per-fold baselines, on the decoder's cells | 2026-10-16 – 2026-10-17 | done | condexp beats the weak bar on 2–4 variants per seed set (spambase_60nan and credit-g_60nan in all); an arm beats K only on credit-g_80nan; the kc2_20nan flip does not hold; K vs the strong tier 13/7/1 on 101/202/303 (16/3/2 vs the weak bar) |
| E42 | An equal, validation-only tuning budget for the baselines | 2026-10-16 – 2026-10-18 | done | the tuned bar beats the untuned one on 17 of 21 and 9 of 12 (1–3%, at most 5.7%); K vs tuned bar 13/5/3 and 7/2/1/2; verdicts change on kr-vs-kp 20/60/80 and spambase_80; a tuned missForest moves K vs the strong tier on 2 of 48 readings |
| E43 | A deployable decode checkpoint (M, the validation mask's score) against K and H | 2026-10-16 – 2026-10-17 | done | reading 3: H is the deployable headline; M equals H on numerical tables by construction and is worse on kr-vs-kp_60nan; K better than M on 4 of 9; the 80nan wins over the mean/mode fill are oracle-only; KC better on E40's six again |
| E44 | Confirmation of the frozen configuration on the 15 unused variants | 2026-10-17 | done | mixed: better than both bars on 9 of 15, worse on 3 (electricity_40/60nan, letter_80nan by 0.2%); large tables and 40nan generalise, 80nan mostly ties; H nearly matches K; KC better on 8, threshold 40 supported |
| E45 | MNAR smoke family: value-dependent missingness at the MCAR counts | 2026-10-17 | done | go for the full family: 36 of 36 cells, integrity exact; the lead over the bar survives MNAR on all 3 tables, shrinks only on biodeg; K − H unchanged; t2's control weak (Mx gains only on biodeg) |
| E46 | The full MNAR family: spambase, the 20nan rung and biodeg's tilt dial | 2026-10-17 | done | supported: H beats the weak bar under MNAR on 4 of 4 new pairs; dial dose-response no (flat 0.25–0.5, then rising; H loses to the bar at slope 2.0); Mx beats G only at slope 2.0 |
| E47 | A deployable calibration: fitted on the validation mask, read with H | 2026-10-18 – 2026-10-19 | done | supported by the rule (HC_m better than H on 4 of 6 named, worse on none) but small: at most 0.4% of H, keeping 3–29% of the oracle calibration's gain; kr-vs-kp unchanged |
| E48 | MaCoDE as an external deep baseline (vendored, inductive, the paper's settings) | 2026-10-19 | done | TRIDENT beats MaCoDE on 20/21 (development) and 14/15 (confirmation), never worse; MaCoDE beats the strong tier on 4 of 15 confirmation variants, so an external deep baseline only; strong on large or categorical tables, weak on small ones |
| E49 | D1 of T28: the decode curves, pre-trained against not, from the store | 2026-10-19 | done | washout on pendigits (+58% head start erased by epoch ~185 of 450), real transfer on kr-vs-kp (M − C +2.5% at the minima), no clear effect on credit-g |
| E50 | The decode stage's levers against washing out the encoder (freeze, encoder LR) | 2026-10-19 – 10-20 | done | refuted: F100 +3.4% and L10 +11.6% worse on pendigits; L10 also worse on kr-vs-kp; no pre-training is 1.2% worse on pendigits, equivalent on kr-vs-kp |
| E51 | Today's default on fresh seeds at 20nan, the stalest variants | 2026-10-20 | done | holds: K − bar better 7 / none 1 / worse 1 (electricity), no reversal; K − strong 6 / 3 / 0; KC better than K on letter and pendigits at 20nan (−2 to −3%) |
| E52 | T28's D3: a frozen encoder, pre-trained against random | 2026-10-20 | done | mixed: frozen FN beats random on pendigits (−53%) and kr-vs-kp (−16%), not on credit-g; the full decode overtakes it (+10 to +21%) and C0 ≈ G |
| E53 | T28's D2: collapse and linear probes on pre-training, and the encoder's drift | 2026-10-20 | done | no collapse under N (0 of 3); value information present (pendigits +36%, kr-vs-kp +7.5%, credit-g none); BATCH 64 moves credit-g's probe; the decode keeps 40–60% CKA |
| E54 | T29's test: is the boosters' lead on electricity carried by `date` as a key? | 2026-10-20 | done | supported: with the row's date K − bar is worse (+0.025, +0.033 at 20/60nan); without it no worse (−0.034 better, −0.002); `day`'s +0.13 lead vanishes without the date |
| E55 | TabPFN v2 as an external foundation-model baseline | 2026-10-21 | done | TabPFN beats the strong tier on 11/15 confirmation variants (rule: propose it for the strong tier); K − TabPFN 6/4/5 (confirmation), 8/6/7 (development); TabPFN wins at 80nan |
| E56 | ADR 0025's panel on K against TabPFN, confirmation set | 2026-10-21 | done | RMSE agrees everywhere; MAE disagrees on biodeg_40nan and pendigits_40nan (K's wins); K keeps the spread better on 14 of 15 (TabPFN's 80nan wins come with shrunk fills) |
| E57 | The default output against TabPFN: KC and H | 2026-10-21 | done | KC − TabPFN 6/4/1 equivalent/4 on confirmation (all 4 losses at 80nan); only letter_80nan changes (worse → equivalent); H − TabPFN 5/5/5 |
| E58 | The strong tier with TabPFN on the seed sets E55 did not cover | 2026-10-21 | done | K − strong: 42/7/13 8/7/1 → 3/9/4, 404/505/606 6/4/2 → 3/2/7, E51 6/3/0 → 1/7/1; MNAR: H − TabPFN 6/4/0, TabPFN gains no ground under MNAR |
| E59 | ADR 0018's bigger encoder on the large tables at 40–80nan | 2026-10-21 – 2026-10-22 | done | mixed: ALL − G on K better 3/6 (both at 40nan, pendigits_60nan), worse on letter_80nan (+0.4%, inside the SESOI); on KC 4/0; gain fades with the rate; the 80nan gap to TabPFN does not close |
| E60 | Are TRIDENT's and TabPFN's errors complementary? A blend of their fills | 2026-10-21 | done | complementary: the blend beats both on 10/15 (8/9 at 40–60nan; at 80nan only pendigits and letter); blend − K 14/1/0, blend − TabPFN 11/1/3 equivalent/0 |
| E61 | Where does K lose to TabPFN on kr-vs-kp and credit-g? A per-column read | 2026-10-22 | done | no target: no column stuck at the marginal on any of 6; K learns every column TabPFN learns, a few points less (gain correlation +0.76 to +0.94); T10's weighting half closes |
| E62 | Fresh MCAR draws: is a result tied to one missingness pattern? | 2026-10-22 | done | holds: no reversal on K − bar or K − strong across 3 draws of 4 variants; drifts on credit-g_60nan K − bar and pendigits_20nan KC − strong; the draw moves a contrast about as much as the fold-and-seed interval (biodeg_80nan about twice) |
| E63 | A MAR smoke against matched MCAR controls | 2026-10-22 | done | loses ground: (H − strong) MAR minus control worse on biodeg and kr-vs-kp, better on kc2; (H − bar) never worse; unlike MNAR, TabPFN gains under MAR on 2 of 4 |
| E64 | The MAR family: does MAR cost the model ground against the strong tier? | 2026-10-22 – 2026-10-23 | done | refuted: (H − strong) MAR minus control worse on 1 of 12 (biodeg_60nan, replicating the smoke); (H − bar) worse on 3 (the small numerical tables at 60nan); the smoke's "loses ground" does not survive |
| E65 | Do decode masks that follow the missingness help under MAR? | 2026-10-22 – 2026-10-23 | done | refuted: P − G on H better on 0 of 12 under MAR, worse on 6; worse on 5 under the control; the interaction is null on 11; the flag stays off |
| E66 | Does E60's complementarity hold on other tables and seeds? | 2026-10-23 | done | complementary at the floor: the blend beats both on 3 of 9 (spambase_80nan, vehicle_60nan, biodeg_60nan); where one method dominates it lands near the better one; on credit-g it cannot reach TabPFN (categorical cells) |
| E67 | The gaps as `[NULL]` in the decode stage, alone: does the token help under MNAR? | 2026-10-23 | refuted | the interaction on H is better on 1 of 8 pairs and worse on 2; on biodeg's dial `[NULL]` alone costs more as the tilt steepens, so Mx's signal came from its other two changes |
| E68 | The tuned bar on the MAR folds: D4 under MAR | 2026-10-23 | mixed | (H − tuned bar) MAR minus control worse on 3 of 12 (biodeg/kc2/vehicle at 60nan), better on 2; the model is never behind the tuned bar on E64's family; D4 not lower under MAR |
| E69 | Per-column weights for the K + TabPFN blend, cross-fitted | 2026-10-23 – 2026-10-24 | mixed | weighted − TabPFN better on 3 of 10 hard variants (the 80nan ties), worse on 1 (kr-vs-kp_60nan, +0.0016); beats both on 13 of 21 (equal blend 10); categorical tables reach TabPFN, not past it |
| E70 | Classification step zero, Stage 0: does HEAD on gorgona8 reproduce the old sweep's folds? | 2026-10-23 | does not reproduce | fold differences up to 0.026 (vehicle) and 0.076 (kc2) macro-F1, against a 0.009 duplicate spread; the regression test passes; Stage 2 reruns arm A on gorgona8 |
| E71 | Classification step zero, Stage 2: what does pre-training buy classification under the fork's defaults? | 2026-10-23 – 2026-10-24 | no gain detected | B − A and C − A no verdict at every level (means −0.005 to +0.004 macro-F1), seeds 42 and 7; the paper's +1.7 points is outside the intervals at 0–60nan |
| E72 | An imputer that models the mechanism: does not-MIWAE shrink the MNAR bias? | 2026-10-24 | mixed | interaction better on 3 of 10 (vehicle at 60 and 20nan, biodeg slope 2.0), worse on none; not-MIWAE's fills fail on biodeg and spambase (impute_score 1.7–9.4); on vehicle it matches the model and its bias moves least of 12 |
| E73 | not-MIWAE with a mean-based fill on biodeg and spambase | 2026-10-24 | supported by the count, one informative pair | interaction better on 3 of 6, worse on none; 2 of the 3 are flagged (fill not below the mean fill); the mean fill removes E72's failure but both VAEs sit near the mean fill on these tables |
| E74 | The per-column KC + TabPFN hybrid as an output, on fresh seeds | 2026-10-24 | mixed | better than KC on 7 of 10 (rule needed 8), worse than TabPFN on 2 (kr-vs-kp 40/60nan, ≤0.005); beats both on E69's same three numerical 80nan variants; TabPFN alone beats KC on 8 |
| E75 | A test-time re-masking ensemble: averaged passes and their spread | 2026-10-24 – 2026-10-25 | mixed (accuracy); spread meets the uncertainty bar | remask − single better 4 / worse 4 (K), 6 / 4 (H), worse on pendigits and letter 40/60nan; gains at 80nan are shrinkage (dispersion lower 15/15; calibrated worse than KC on 8); numerical spread ranks errors on 12/15 under K and H, coverage90 0.17–0.66 |
| E76 | Does the re-masking spread rank cells on the development variants? | 2026-10-25 – 2026-10-26 | supported | within-column signal on 14 of 15 up to 60nan under K and H, none against (credit-g_60nan the exception); categorical: single-pass confidence beats the vote share 9/9; K reproduced the thesis-table runs exactly |
| E77 | Does the re-masking spread make intervals narrower at the same coverage? | 2026-10-25 | refuted | spread interval wider than a per-column constant one on 15 of 15 (K) at the same 90% coverage; +9% to +144% at 40nan; the spread ranks but cannot adapt a width |
| E78 | Which columns does the decoder fill each column from? An occlusion map | 2026-10-26 – 2026-10-27 | supported | faithful on 9 of 9 real tables at 40nan (D +0.04 to +0.41); synthetic blocks recovered, AUROC 0.979 (|Spearman| 0.847) |
| E79 | Does the occlusion map hold at 60nan? | 2026-10-27 | supported (at the bar) | faithful on 7 of 9 at 60nan (kc2, credit-g inconclusive), none against; synthetic AUROC 0.914; every D 29–74% smaller than at 40nan |
| E80 | Does the spread grow when a cell's top sources are missing? | 2026-10-27 | mixed | specific on 7 of 16, against on 6; the spread falls as a row's gaps grow (16 of 16) while error rises with missing top sources (16 of 16) |
| E81 | Does a decoder that can tell a real gap from its own masks read the gap pattern under MNAR? | 2026-10-27 | mixed | null arm reads the pattern at slopes 0.25–1.0 (S +0.02 to +0.07) and on vehicle/kc2 MNAR, not at MCAR, but only +0.005 at slope 2.0; less accurate than mask on 11 of 11 |
| E82 | Can the hybrid's weights be chosen without the complete table? | 2026-10-27 | refuted | better than H on 10/10 and than KC on 7/10 (never worse), but worse than TabPFN on 5 and than the oracle weights on 6; the loss is the selection (small and categorical tables) |
| E83 | TabPFN v2's classifier against TRIDENT's classification, on E71's folds | 2026-10-27 | done | TabPFN ahead at every level by +0.017 to +0.024 macro-F1; TRIDENT leads only on credit-g and electricity, through its class-weighted loss (post hoc) |
| E84 | Does the re-masking spread still rank errors under MNAR and MAR? | 2026-10-27 – 2026-10-28 | supported (full scope) | signal on 12 of 12 under K and H, none against (lowest bound +0.084); on biodeg the ranking weakens steadily with the MNAR tilt; 11 cells under torch 2.11.0 |
| E85 | Does the occlusion map hold at 20nan and 80nan? | 2026-10-27 – 2026-10-28 | 20nan supported, 80nan refuted | 20nan: faithful on 8 of 9 (kc2 inconclusive), synthetic AUROC 0.994; 80nan: faithful on 4 of 9, not faithful on biodeg/kc2/spambase/credit-g; the map holds from 20 to 60nan |
| E86 | Do gorgona8 and medusa1 give the same fold scores for the same cell? | 2026-10-28 | done | no verdict on either task; per-fold differences up to 0.164 (imputation) and 0.074 (classification): a study stays on one machine |
| E87 | Does pre-training help classification with ten folds and five seeds? | 2026-10-28 – 2026-10-29 | refuted | B − A equivalent (inside ±0.01) at all five levels, −0.0035 to +0.0022; the paper's +1.7-point gain is excluded |
| E88 | Does the occlusion map hold under MNAR and MAR? | 2026-10-28 | supported | faithful on 11 of 12 (kc2_probit inconclusive); self-masking weakens D with the slope, MAR strengthens it |
| E89 | Can fewer or better-fed choices close the deployable hybrid's gap to TabPFN? | 2026-10-28 | supported (development) | `shrunk` better than H 10/10, behind TabPFN on 2 (kr-vs-kp); `table` behind TabPFN on none; fresh seeds to confirm |
| E90 | Does the deployable hybrid's `table` rule hold on fresh seeds? | 2026-10-28 – 2026-10-29 | supported | better than H on 9/10, worse than TabPFN on none (fresh seeds); `shrunk` mixed (4 behind) |
| E91 | Does the deployable hybrid cost anything where it is not needed? | 2026-10-28 – 2026-10-29 | supported | worse than H on none of 7, worse than TabPFN on none (better on 6) |
| E92 | Does TabPFN v2's classification lead hold on ten folds, and does class balancing close credit-g and electricity? | 2026-10-28 – 2026-10-29 | supported (both arms) | balanced TabPFN − TRIDENT +0.018 to +0.037 macro-F1, ahead at 5 of 5 levels; balancing closes credit-g and electricity |
| E93 | Does the deployable `table` hybrid hold under MNAR and MAR? | 2026-10-29 | mixed | better than H on 6/12 (rule 8), worse on spambase_probit; worse than TabPFN on credit-g_marf only; H already leads TabPFN off MCAR |
| E94 | Is the machine's shift inside the SESOI? | 2026-10-29 | classification equivalent; imputation no verdict | macro-F1 +0.0009 [−0.0056, +0.0073]; imputation +0.54% [−0.06%, +1.13%]: split by seed allowed for classification only |
| E95 | Does the deployable `table` hybrid hold at 20nan? | 2026-10-29 | supported | worse than H or TabPFN on none of 9; better than both on 6 |
| E96 | Is the imputation shift inside the SESOI once kc2 is left out? | 2026-10-29 | equivalent | +0.04% [−0.30%, +0.38%]: imputation on credit-g/vehicle/biodeg may split by seed |
| E97 | Step zero Stage 1: the CPU bar on the sweep's folds | 2026-10-29 | done (descriptive) | untuned native HGB leads TRIDENT by +0.015 to +0.033 macro-F1 at 20–80nan; best arm missForest+ind+HGB at 20–60nan |
| E98 | Step zero proper (H1c): does TRIDENT's classification lead survive a fair rerun? | 2026-10-29 – 2026-10-30 | refuted | the tuned GBDT bar beats tuned TRIDENT at every level, by 0.024–0.032 macro-F1 at 40–80nan; tuning does not move TRIDENT off the fork budget |
| E99 | Does TRIDENT generally beat MLP, SAINT and TabNet on classification at 40–80nan? | 2026-10-31 – 2026-11-01 | supported | leads all three; over MLP +0.013/+0.008 at 40/60nan and a tie at 80nan; SAINT +0.03–0.04; TabNet +0.09–0.11 (a weak TabNet) |
| E100 | Does a steps-aware training batch help the small tables? (T14) | 2026-10-30 | refuted | S8 better on 0 of 16, worse on 4 (credit-g 40/60nan, biodeg 20/60nan); T14 closed |
| E101 | Does pendigits still gain from batch 1024 with the learning rates x4, and from a longer decode? (T11) | 2026-10-30 | supported | P600 better on 3 of 4 (−5.4%, −4.3%, −1.2% at 20–60nan), equivalent at 80nan; a fresh-seed confirmation next |
| E102 | Does E101's pendigits gain replicate on fresh seeds? | 2026-10-30 – 2026-10-31 | not replicated | better on 3 (−5.8%, −4.2%, −1.2% at 20–60nan, as in E101), worse at 80nan (+0.42%, under the SESOI); the rule asked for worse on none |
| E103 | Does the large batch and longer decode help letter and electricity? | 2026-10-31 | letter mixed; electricity refuted | letter −2.9%/−1.5% at 20/40nan, worse +0.2% at 80nan; electricity better on none |
| E104 | Does the large batch hold on pendigits and letter at 20–40nan, on fresh seeds? | 2026-10-31 | replicated | better on 4 of 4: pendigits −5.9%/−4.1%, letter −3.3%/−1.5%; an ADR proposal for the user |
| E105 | Does the re-masking spread still rank errors under MNAR and MAR at 40nan? | 2026-10-31 | supported | a signal on 12 of 12 under K and H, none against; weakens on biodeg with the tilt, as at 60nan |
| E106 | Was E99's TabNet undertrained? Its configuration with a bigger budget | 2026-11-01 | holds | TRIDENT still leads TabNet-B by +0.08 to +0.10; the budget adds +0.008 to +0.016 to TabNet |
| E107 | The deep baselines at 00 and 20nan (descriptive) | 2026-11-01 | done (descriptive) | TRIDENT ahead of SAINT and TabNet at both; MLP ahead at 00nan (−0.012), TRIDENT ahead at 20nan (+0.009) |
| E108 | The thesis tables' cells ADR 0036 changed, rerun on today's default | 2026-11-01 – 2026-11-02 | done | only the five ADR 0036 variants moved, all toward TRIDENT; pendigits_20nan now beats TabPFN on both development sets |
| E109 | Does the large batch help pendigits_60nan on fresh seeds? | 2026-11-01 | supported | −1.34%, every seed below zero; the third time (E101, E102); adding it to ADR 0036 is the user's call |
| E110 | Does the re-masking spread still rank errors under MNAR and MAR at 20nan? | 2026-11-01 – 2026-11-02 | supported | a signal on 12 of 12 under K and H, none against; no weakening with the tilt at 20nan |
| E111 | E99 on fresh seeds: does TRIDENT still lead MLP and SAINT at 40–80nan? | 2026-11-02 | replicated | MLP +0.009/+0.007/tie, SAINT +0.035/+0.035/+0.029, as in E99 |
| E112 | Does an EMA target make classification pre-training help? (T13) | 2026-11-02 | mixed | E − B higher at 40nan (+0.005) and 60nan (+0.009), equivalent elsewhere; below the SESOI |
| E113 | Does the bigger encoder still help pendigits and letter at 40–60nan on today's defaults? (T32) | 2026-11-02 – 2026-11-03 | supported | better on 4 of 4: pendigits −6.0%/−3.1%, letter −5.5%/−1.1% |
| E114 | Does the EMA target's gain at 40–60nan replicate on fresh seeds? | 2026-11-02 – 2026-11-03 | partial | E higher at 60nan (+0.008), no verdict at 40nan (+0.009, one seed below zero); below the SESOI; T13 closes |
| E115 | TRIDENT against MLP and SAINT under MAR and MNAR at 60nan (descriptive) | 2026-11-03 | done (descriptive) | SAINT +0.045/+0.043 (MNAR/MAR); MLP +0.011 under MAR, no verdict under MNAR (+0.006) |
| E116 | Does the bigger encoder's gain on pendigits and letter at 40–60nan replicate on fresh seeds? | 2026-11-03 | replicated | better on 4 of 4 at E113's sizes (−6.3%, −3.1%, −5.5%, −1.2%); ADR 0037 proposed |
| E117 | Does the bigger encoder help pendigits and letter at 80nan? | 2026-11-03 | mixed | pendigits −0.39% better, letter +0.37% worse, both under the SESOI; ADR 0037 stops at 60nan |

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

### E34 · Checkpoint by the validation gaps, and calibration on them (2026-10-03)

- **Status:** done, 12:52 to 18:43 GMT-3 on gorgona8, all 63 cells; H equals E33's G cells to
  the last digit on every cell.
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
- **Results (within-run, 15 fold pairs per variant):** K − H: K better on 3 (credit-g_80nan
  −0.020, kr-vs-kp_80nan −0.029, spambase_80nan −0.010), H better on none. C − H: C better on 13,
  H better on 1 (spambase_20nan +0.005). KC − H: KC better on 9, H on none; KC − K: KC better on
  12, K on 1 (spambase_20nan). Means over 21: H 0.7519, C 0.7453, K 0.7446, KC 0.7412; below the
  best baseline on 15, 18, 17, 18 variants. Both mechanisms help most at 80nan (K −0.020, C
  −0.022, KC −0.026), where the gap criterion stops earlier than the loss. Wall time 5 h 51 min
  against 5.5 to 6 h planned.
- **Conclusion:** by the rule fixed in advance, K qualifies and is the recommendation; C and KC
  fail on one variant each (spambase_20nan, by about 0.005), while gaining on most others. The
  user decides whether K becomes the default (ADR, flag, tag); a calibration that leaves
  well-calibrated columns alone would be a new pre-registered study. **Decided 2026-10-03:** the
  user adopted K ([ADR 0015](adr/0015-decode-checkpoint-chosen-on-the-validation-gaps.md)).
- **Sources:** `docs/tickets/imputation-token-shape/04-checkpoint-and-calibration.md` (Outcome);
  `scripts/experiments/selection_report.py`.

### E35 · The gap-chosen checkpoint on E32's seeds (2026-10-04)

- **Status:** done, 00:23 to 06:14 GMT-3 on gorgona8, all 63 cells; H equals E32's G cells to the
  last digit on every cell.
- **Hypothesis:** E34's result, the checkpoint chosen on the validation gaps (K) better than the
  loss's (H) on 3 variants at 80nan and worse on none, is a property of the criterion, not of
  seeds 101, 202 and 303. ADR 0015 adopted it on those seeds alone.
- **Scenarios:** E34's cell on seeds 42, 7, 13: `--decode_checkpoint loss` with
  `--score_induced_checkpoint` and `--score_calibrated`, so one trajectory gives H, K, C and KC.
  All 21 variants, five folds; 63 cells tagged `experiment=selection-replication-2026-10-04`;
  H must equal E32's G cells.
- **Measures and decision rule:**
  - **Primary:** K − H on `impute/induced/impute_score`, 15 within-run fold pairs per variant;
    a verdict needs the interval to exclude zero and all three seeds to agree in sign.
  - **Reading:** replicated if no variant shows H better and K is better on at least 2 of
    credit-g_80nan, kr-vs-kp_80nan and spambase_80nan; safe but weaker if no variant shows H
    better and K wins fewer than 2 of them; not replicated if any variant shows H better, which
    goes back to the user.
  - **Secondary:** K − H pooled with E34; C, KC on these seeds and pooled; epochs; by level;
    means; best-baseline counts; time.
- **Results (within-run, 15 fold pairs per variant, seeds 42, 7, 13):** K − H: K better on 6
  (kr-vs-kp_80nan −0.034, spambase_40nan −0.004, spambase_60nan −0.003, spambase_80nan −0.010,
  biodeg_60nan −0.008, kc2_60nan −0.045), H better on none; on E34's three, K better on
  kr-vs-kp_80nan and spambase_80nan, credit-g_80nan leaning (−0.008) without a verdict. Pooled with
  E34 (six seeds): K better on 6, H on none; the mean difference is negative on 19 variants. By
  level −0.007, −0.005, −0.012, −0.017. Means over 21: H 0.7542, K 0.7446, C 0.7480, KC 0.7422;
  below the best baseline on 16, 19, 18, 19 variants. C − H again "H better" on spambase_20nan
  (+0.0055) and "C better" on 13. Wall time 5 h 51 min against about 5.8 h planned (1.01x).
- **Conclusion:** replicated by the rule fixed in advance; ADR 0015 stands on six seeds. The
  calibration fails the same way on both seed sets (only spambase_20nan, where the model is
  already well calibrated), which points to a calibration that leaves the 20nan tables alone as
  a possible next study.
- **Sources:** `docs/tickets/imputation-token-shape/05-checkpoint-replication.md` (Outcome);
  `scripts/experiments/selection_replication_report.py`.

### E36 · Showing the gaps as `[MASK]` to pre-training too (2026-10-05)

- **Status:** done, 00:16 to 06:08 GMT-3 on gorgona8, all 63 cells; every pair's baselines equal.
- **Hypothesis:** the pre-training stage still shows the real gaps as `[NULL]` while the decode
  stage and the scoring show `[MASK]` (ADR 0014); aligning pre-training too might score better.
  The prior is weak: the decode stage retrains the encoder for 450 epochs (E05: 300 pre-training
  epochs no better than 2).
- **Scenarios:** arm PG = today's defaults with `--pretrain_gap_token mask`; reference R = E35's
  K readout of the same cells, equal to a default run to the last digit. All 21 variants, seeds
  42, 7, 13, five folds; 63 new cells tagged `experiment=pretrain-gap-token-2026-10-05`. Pairs
  share their random stream.
- **Measures and decision rule:**
  - **Primary:** PG − R on `impute/induced/impute_score`, 15 fold pairs per variant; a verdict
    needs the interval to exclude zero and all three seeds to agree in sign.
  - **Reading on all 21:** any R better → keep `null`; no R better and PG better on at least 3 →
    recommend PG as the default (ADR, user decides, then replicate); otherwise keep `null`.
  - **Secondary:** by level; masked; decode epochs; means and best-baseline counts; time.
- **Results (PG − R, induced, 15 fold pairs per variant):** R better on 1 (electricity_20nan
  +0.0105 [+0.0020, +0.0190]), PG better on none, no detectable difference on 20. Mean by level
  +0.0003 (20nan), −0.0001 (40), +0.0016 (60), −0.0012 (80). Means over 21: PG 0.7450, R 0.7446;
  below the best baseline on 19 under both. Masked: no detectable difference on all 21. Wall time
  5 h 52 min against about 5.8 h planned (1.01x).
- **Conclusion:** by the rule fixed in advance, keep `null`. The pre-training gap token does not
  matter once the decode stage shows `[MASK]` and retrains the encoder for 450 epochs, as the
  ticket's weak prior and E05 suggested. The flag stays, default `null`.
- **Sources:** `docs/tickets/imputation-token-shape/06-pretraining-gaps-as-mask.md` (Outcome);
  `scripts/experiments/pretrain_gap_token_report.py`.

### E37 · Architecture screening: deeper, wider or normed encoder (2026-10-05 – 2026-10-06)

- **Status:** done, 19:20 GMT-3 to 00:39 GMT-3 on gorgona8, all 60 cells; every pair's baselines
  equal.
- **Hypothesis:** the encoder is small for the task (2 layers, feed-forward 32 against width 128,
  no final LayerNorm on a pre-norm stack; the full-profile winners used 4 to 6 layers). A deeper,
  wider or normed encoder should impute better (T03).
- **Scenarios:** arms L4 (`LAYERS` 4), F256 (`DIM_FEED` 256), LN (`ENCODER_FINAL_NORM layer`),
  ALL (the three), each today's configuration with one change; reference R = E35's K readout.
  Variants credit-g_20nan, credit-g_80nan, kr-vs-kp_40nan, biodeg_60nan, pendigits_20nan; seeds
  42, 7, 13; five folds; 60 cells tagged `experiment=architecture-2026-10-05`.
- **Measures and decision rule:**
  - **Primary:** arm − R on `impute/induced/impute_score`, 15 fold pairs per variant; a verdict
    needs the interval to exclude zero and all three seeds to agree in sign.
  - **Reading:** an arm advances if no variant shows R better and at least one shows the arm
    better; the advancing arm with the most "better" verdicts (tie: most negative mean) goes to a
    21-variant confirmation, pre-registered separately with the user's go-ahead; none advancing
    closes T03.
  - **Secondary:** masked; minutes a cell per arm; means; time.
- **Results (arm − R, induced, 15 fold pairs per variant):** ALL better on 3 (credit-g_80nan
  −0.0057, kr-vs-kp_40nan −0.0120, pendigits −0.0133), worse on none, but credit-g_20nan +0.0133
  [−0.0008, +0.0274] with every seed positive (masked there: R better). F256 better on 1
  (pendigits −0.0116). L4 better on pendigits (−0.0174), worse on credit-g_20nan (+0.0197). LN
  worse on pendigits (+0.0142). Mean over the five: L4 +0.0036, F256 −0.0007, LN +0.0022, ALL
  −0.0020. Cost a cell against R: 1.57x, 1.17x, 0.99x, 1.82x. Wall time 5 h 19 min against about
  5.4 h planned (0.98x).
- **Conclusion:** by the rule fixed in advance, F256 and ALL advance and ALL goes to a 21-variant
  confirmation, pending the user's go-ahead. The gain is concentrated in the largest table
  (pendigits); depth hurts the smallest (credit-g_20nan), where ALL is a near miss, so a
  confirmation under the "no variant worse" rule is at risk on the small tables.
- **Sources:** `docs/tickets/imputation-architecture/01-architecture-screening.md` (Outcome);
  `scripts/experiments/architecture_report.py`.

### E38 · The bigger encoder by table size, on fresh seeds (2026-10-06 – 2026-10-07)

- **Status:** done, all 63 cells: 15 from 14:48 to 21:12 GMT-3 on 2026-10-06, paused at the user's
  request so that ADR 0017 could land, the other 48 from 02:32 to 07:15 GMT-3 on 2026-10-07.
- **Hypothesis:** the bigger encoder (4 layers, feed-forward 256, final LayerNorm), whose E37 gain
  concentrated in pendigits (−4.0%), helps the large tables (10,000 rows or more). The user chose
  in advance a size-dependent default if it holds, so the cost on smaller tables is described.
- **Scenarios:** arm ALL on all 21 variants, seeds 101, 202, 303 (unused by E37); reference R =
  E34's K readout on the same seeds. Groups by rows: large (pendigits, letter, electricity),
  medium (kr-vs-kp, spambase), small (credit-g, vehicle, biodeg, kc2). 63 cells tagged
  `experiment=architecture-large-2026-10-06`.
- **Measures and decision rule:**
  - **Primary:** ALL − R on `impute/induced/impute_score` on the three large variants, 15 fold
    pairs each; a verdict needs the interval to exclude zero and all three seeds to agree in sign.
  - **Reading:** supported if ALL better on at least 2 of the 3 and worse on none (then a
    size-dependent default by ADR, the user decides); refuted if worse on any; mixed otherwise.
  - **Secondary:** per group, verdict counts, mean relative change with an interval, cost; masked;
    time.
- **Results (primary, ALL − R, induced, 15 fold pairs):** pendigits −0.0128 (−3.8%), letter
  −0.0259 (−5.6%), both "ALL better"; electricity +0.0021 (+0.3%), no verdict. Mean relative change
  over the large tables −3.0% [−5.1%, −1.0%]. Cost 1.84x a cell. Medium group (8): −0.2%
  [−0.6%, +0.2%], better on kr-vs-kp_40nan only, worse on none, 1.95x. Small group (10): +1.9%
  [+0.5%, +3.2%], worse on credit-g_20nan, vehicle_20nan, vehicle_60nan and kc2_60nan, better on
  none, 1.77x. Integrity: 63 of 63 pairs. The resumed 48 cells took 4 h 43 min against ~4 h.
- **Conclusion (primary):** supported by the rule fixed in advance; the recommendation is the
  size-dependent default the user chose, bigger encoder on the large tables, with the caveat that
  electricity, the largest, gains nothing at 1.77x the cost. The described cost confirms the size
  dependence: neutral on the medium tables, worse on the small ones, so the size-dependent choice
  keeps today's encoder where the bigger one does not pay. **Decided 2026-10-07:** the user adopted the bigger encoder on
  pendigits_20nan and letter_20nan only, leaving electricity out
  ([ADR 0018](adr/0018-bigger-encoder-on-the-large-tables.md)).
- **Sources:** `docs/tickets/imputation-architecture/02-bigger-encoder-by-table-size.md`;
  `scripts/experiments/architecture_size_report.py`.

### E39 · Re-tuning the promoted files under today's defaults (2026-10-15 – 2026-10-16)

- **Status:** done. Pre-registered in `00766f4` at 23:34 GMT-3 on 2026-10-15. All 35 jobs ran
  from 23:34 GMT-3 on 2026-10-15 to 06:56 GMT-3 on 2026-10-16 (7 h 22 min against 6.5 to 7 h
  planned); none failed.
- **Hypothesis:** the five promoted files (credit-g_40nan, kr-vs-kp_20nan, kr-vs-kp_40nan,
  spambase_20nan, spambase_40nan) carry sampled values chosen on 2026-09-11 under the old
  configuration and the masked objective. The question is whether a fresh reduced search under
  today's defaults (ADRs 0013 to 0015), ranked by the validation gaps (ADR 0008), beats them, and
  whether the files still beat having no file. The prior is a small gain at most: ADR 0005's
  tuning did not beat the defaults.
- **Scenarios:**
  - **Studies:** five, one per file. Each has 40 trials, seed 42, the `reduced` profile and the
    `induced` objective. No promotion.
  - **Arms,** on seeds 101/202/303 with five folds each:
    - **T:** the study's winner, 15 cells;
    - **D:** today's defaults with no file, 15 cells;
    - **F:** the current file, read from E34's K readout.
  - **Tags:** `experiment=retune-2026-10-15`, plus `arm`.
  - **Out of scope:** pendigits_20nan and letter_20nan (a reduced trial would drop ADR 0018's
    encoder).
- **Measures and decision rule:**
  - **Primary:** T − F, D − F and T − D on the induced `impute_score`, 15 fold pairs each. A
    verdict needs the interval to exclude zero and all three seeds to agree in sign.
  - **Recommendation:** an arm qualifies when it beats another arm by verdict and loses to none.
    Applied in order, first match wins:
    1. T qualifies: promote the winner.
    2. D qualifies and beats F: remove the file.
    3. Otherwise: keep F.

    The user decides, by an ADR.
  - **Secondary:** masked contrasts, winners beside the files' values, importances, cost.
- **Results (induced, 15 fold pairs):**

  | variant | F | D | T | T − F | D − F | T − D |
  |---|---|---|---|---|---|---|
  | credit-g_40nan | 0.9662 | 0.9208 | 0.9162 | −0.0501, T better | −0.0454, D better | −0.0046, none |
  | kr-vs-kp_40nan | 0.7626 | 0.7258 | 0.7234 | −0.0392, T better | −0.0368, D better | −0.0024, none |
  | spambase_40nan | 0.9068 | 0.9023 | 0.9019 | −0.0048, none | −0.0044, D better | none |
  | spambase_20nan | 0.8529 | 0.8439 | 0.8535 | none | −0.0091, none | +0.0096, D better |
  | kr-vs-kp_20nan | 0.5866 | 0.5829 | 0.5830 | none | none | none |

  The integrity check passed on 30 of 30 pairs. In the studies, `LR_DECODE` or `DROPOUT` leads the
  importances, and credit-g_40nan's winner moved back toward the defaults (decode learning rate
  1.7e-3 against the file's 8.7e-3).
- **Conclusion:**
  - **By the rule fixed in advance:** promote the winner on credit-g_40nan and kr-vs-kp_40nan,
    remove spambase_40nan's file, and keep the other two.
  - **The broader reading:** the stale files hurt on three variants, and the re-tuning beats the
    defaults nowhere. Where T is promoted, D is equivalent (−0.3% and −0.5%, no verdict), so
    "remove the file" is the decision-equivalent alternative there. ADR 0005's finding (the reduced
    search does not beat the defaults) replicates under today's regime.
  - **Decided 2026-10-16:** the user removed the three files that lose to the defaults
    (credit-g_40nan, kr-vs-kp_40nan, spambase_40nan) and kept the other two
    ([ADR 0019](adr/0019-remove-the-stale-promoted-imputation-files.md)).
- **Sources:** `docs/tickets/imputation-retune/01-retune-promoted-files.md` (Outcome);
  `scripts/experiments/retune_report.py`, `retune_run.py`, `run_retune_queue.sh`.

### E40 · Calibration at 40% missing and above, on fresh seeds (2026-10-16)

- **Status:** done. Pre-registered in `5634858` at 10:55 GMT-3 on 2026-10-16. All 36 cells ran
  from 10:55 to 13:54 GMT-3 (2 h 58 min against 2.5 h planned); none failed.
- **Hypothesis:** the calibration fitted on the validation gaps (KC) helps at 40% missing and
  above, where E34 and E35 never found it worse than K and found it better on 6 variants in both
  seed sets. At 20nan it failed (spambase_20nan, kr-vs-kp_20nan), so the rule leaves those alone.
- **Scenarios:**
  - **Cells:** one run per (variant, seed) of today's default with `--score_calibrated`, so K is
    the headline and KC the calibrated readout, within-run.
  - **Variants:** the 12 at ≥40nan.
  - **Seeds:** 404, 505 and 606, unused by any earlier study; 36 cells.
  - **Tag:** `experiment=calibration-2026-10-16`.
- **Measures and decision rule:**
  - **Primary:** KC − K on the induced `impute_score`, 15 fold pairs per variant. A verdict needs
    the interval to exclude zero and all three seeds to agree in sign.
  - **Reading:**
    - **supported** if KC is better on ≥4 of the 6 named variants (kr-vs-kp_80nan,
      spambase_60nan, spambase_80nan, credit-g_60nan, credit-g_80nan, vehicle_60nan) and K is
      better on none of the 12. Adoption would need a code change and an ADR.
    - **refuted** if K is better on any of the 12;
    - **mixed** otherwise.
  - **Secondary:** counts over all 12, the mean per level, α, how many variants each arm puts
    below the best baseline, and cost.
- **Results (KC − K, induced, 15 within-run fold pairs):**
  - **KC better** on the six named variants:

    | variant | KC − K |
    |---|---|
    | kr-vs-kp_80nan | −0.85% |
    | spambase_60nan | −0.21% |
    | spambase_80nan | −0.55% |
    | credit-g_60nan | −0.62% |
    | credit-g_80nan | −0.39% |
    | vehicle_60nan | −1.27% |

    and also on biodeg_60nan (−0.61%).
  - **No detectable difference** on kr-vs-kp_40nan, kr-vs-kp_60nan, spambase_40nan, credit-g_40nan
    and kc2_60nan.
  - **K better** on none.
  - **By level:** mean KC − K is −0.0009 at 40nan, −0.0037 at 60nan and −0.0059 at 80nan.
  - **Below the best baseline:** 11 of 12 under both arms; credit-g_80nan goes from 1.0112 to
    1.0073.
- **Conclusion:** supported by the rule fixed in advance. The recommendation is the calibration as
  the imputation output at 40% missing or more, by an ADR.
  - Adopting it needs a code change, authorized separately. Today the calibration is only a
    diagnostic readout.
  - **Caveat:** no 40nan variant earns a verdict, so the threshold is safe but idle at 40. A
    threshold of 60, suggested by these data, would need its own check.
- **Sources:** `docs/tickets/imputation-calibration/01-calibration-at-higher-missingness.md`
  (Outcome); `scripts/experiments/calibration_report.py`, `calibration_run.py`,
  `run_calibration_queue.sh`.

### E41 · Linear baselines and the missing per-fold baselines, on the decoder's cells (2026-10-16 – 2026-10-17)

- **Status:** done. Pre-registered at 22:47 GMT-3 on 2026-10-16 (`5278337`), before any arm was
  scored on gorgona8.
- **Hypothesis:** the review's laptop probe (indicative, not pre-registered) found condexp and
  inductive SoftImpute setting the bar on about 7 induced variants by seed means, and SoftImpute
  below the model on kc2_20nan. ICE with the sequential transform never beat the model. Paired fold
  by fold, the kc2_20nan flip may not survive.
- **Scenarios:**
  - **Arms:** condexp, SoftImpute and ICE, with the rules frozen in the ticket (clip to the training
    range; 90% energy share with no p//2 cap; one-hot ICE with the sequential transform); an
    optional `missforest_seq`.
  - **Missing baselines:** lgbm and missForest, recomputed where the fold CSVs lack them.
  - **Cells:** the cell map of the ticket, offline through `recompute_folds`:
    - 101/202/303: E34 K, with E38 ALL on pendigits and letter_20nan and E39 D on the three 40nan
      variants;
    - 42/7/13: E35 K, five variants labelled;
    - 404/505/606: E40.

    162 (variant, seed) pairs.
  - **No store writes.**
- **Measures and decision rule:**
  - **Primary:** each arm − the bar of 5, each arm − K, and K − the strong tier's best (missForest
    and the new arms), on the induced `impute_score`. 15 fold pairs each, with the verdict rule,
    plus equivalence within ±1%.
  - **Reading:** descriptive. The two-tier design (weak = the bar of 5; strong = missForest and the
    new arms) was decided by the user on 2026-10-16 and is written as ADR 0021 after this study.
- **Results:** all 162 pairs ran from 23:06 to 01:26 GMT-3; integrity held to 1e-9.
  - condexp is better than the weak bar on 4 / 4 / 2 variants (seed sets 101/202/303, 42/7/13,
    404/505/606), with spambase_60nan and credit-g_60nan in every set.
  - SoftImpute is better on 1 / 0 / 2 and ICE on 1 / 0 / 1, both on kc2_60nan.
  - An arm is better than K only on credit-g_80nan (condexp, every set), where K is above 1.0.
  - The kc2_20nan flip does not hold paired: SoftImpute − K is +0.0013, no difference.
  - Without its cap, SoftImpute fails worse on spambase (+0.54 at 20nan).
  - K against the strong tier's best: 13/7/1, 8/7/1 and 6/4/2, against 16/3/2, 12/3/1 and 10/1/1
    against the weak bar. The model stays below it on 9 of 9 tables on 101/202/303 (Wilcoxon on
    per-table means, p = 0.0039).
  - missForest's sequential fill flips no model win.
- **Conclusion:** the linear arms matter on a few small numerical or mixed tables at 60nan and
  above, and beat the model only where it already loses to the mean/mode fill. Next is ADR 0021
  (two tiers), then the live family and its backfill.
- **Sources:** `docs/tickets/imputation-protocol/01-linear-baselines.md` (Outcome);
  `docs/plans/2026-10-17-experiment-improvements.md` §2b; `docs/reports/2026-10-17-thesis-table-development.md`.

### E42 · An equal, validation-only tuning budget for the baselines (2026-10-16 – 2026-10-18)

- **Status:** done. Both passes are read: the weak tier on 2026-10-17, missForest on 2026-10-18.
  Pre-registered at 22:47 GMT-3 on 2026-10-16 (`5278337`), before any tuned baseline was scored.
- **Hypothesis:** a declared validation-only budget lowers the bar by about 0.01–0.035 on a few
  variants (review threat 1). The probe's recipe was chosen on test, so the size is unmeasured.
- **Scenarios:**
  - **Grids:** knn k ∈ {5, 10, 20, 50}; 12 configurations each for hgb and lgbm; 4 for missForest;
    a mean-shrinkage α ∈ {0, 0.25, 0.5} at 60nan and 80nan.
  - **Selection:** per fold on the validation rows' own gaps (the signal K uses), with no refit;
    only the chosen configuration is scored on test. Masked-cell selection is the sensitivity
    reading.
  - **Cells:** E41's cell map on 101/202/303 and 404/505/606, plus E44's variants and seeds.
  - **No store writes.**
- **Measures and decision rule:**
  - **Primary:** the tuned bar − the untuned bar, and K − the tuned bar, on the induced
    `impute_score`, 15 fold pairs, with the verdict rule.
  - **Reading:** descriptive. A thesis win needs K to beat both bars (D4); variants whose verdict
    changes are listed.
- **Results (weak tier):** 207 files, from 2026-10-16 23:39 to 2026-10-17 15:43 GMT-3.
  - **The budget:** the tuned bar is better than the untuned bar of 5 on 17 of 21 (101/202/303) and
    9 of 12 (404/505/606), typically by 1–3%, at most 5.7% (kr-vs-kp_60nan). It is worse once
    (kc2_60nan, +2.9%).
  - **K against the tuned bar:** better 13 / none 5 / worse 3, and 7 / 2 / equivalent 1 / 2,
    against 16/3/2 and 10/1/1 for the untuned bar.
  - **Verdict changes:** kr-vs-kp_60nan (better → worse) and kr-vs-kp_80nan (better → none) on
    both sets; kr-vs-kp_20nan and spambase_80nan on one.
  - **Masked selection:** K is better on 14 of 21 and 9 of 12. hgb and lgbm almost never keep
    their current settings.
- **Conclusion:** the budget matters on most variants but moves few verdicts. The all-categorical
  kr-vs-kp is where it catches up with the model. Development wins under D4: 13 of 21 and 7 of 12.
- **The missForest pass** ran from 2026-10-17 22:52 to 2026-10-18 22:21 GMT-3.
  - The tuned missForest beats the live one on 7 of 21, 4 of 12 and 9 of 15 (E44), never worse.
    The gains are at 60–80nan, up to 5.3%, where shrinkage wins.
  - K against the strong tier with missForest tuned: 13/7/1, 5/5/1/1 and 12/1/2, against 13/7/1,
    6/4/2 and 12/1/2 as logged. The only two changes are on 404/505/606.
- **Sources:** `docs/tickets/imputation-protocol/02-baseline-budget.md` (Outcome);
  `scripts/experiments/baseline_budget_run.py`, `baseline_budget_report.py`; plan §2c.

### E43 · A deployable decode checkpoint (2026-10-16 – 2026-10-17)

- **Status:** done. Pre-registered with its code on 2026-10-16 at 22:58 GMT-3 (`9a7e249`). All 27
  cells ran from 22:58 GMT-3 on 2026-10-16 to 01:50 GMT-3 on 2026-10-17; none failed.
- **Hypothesis:** K (ADR 0015) uses the validation gaps' truth, which a deployed model never has.
  Its 80nan wins over the mean/mode fill exist only under it (E34/E35: 0 of 3 under H, 2 of 3
  under K). A criterion that needs no gap truth, M (the fixed validation mask's `impute_score`),
  may recover part of the gain, since K also improves masked cells it never chose on.
- **Scenarios:**
  - **Runs:** one run per cell of today's default with `--score_checkpoints --score_calibrated`.
    K (the headline), H (`loss_checkpoint`), M (`masked_checkpoint`) and KC (`calibrated`) all come
    from one trajectory.
  - **Variants:** the 9 at ≥60nan.
  - **Seeds:** 111/222/333, unused.
  - **Size:** 27 cells, tag `experiment=deployable-checkpoint-2026-10-17`.
  - **Checks before launch:** the runner equals an E40 cell, and a `masked` run equals the M
    readout.
- **Measures and decision rule:**
  - **Primary:** M − H and M − K on the induced `impute_score`, 15 within-run fold pairs, with the
    verdict rule.
  - **Reading, first match wins:**
    1. K better than M on none of the 9: M suffices;
    2. M better than H on ≥2 of the 3 80nan variants and worse on none: M is the deployable
       headline (ADR amending 0015);
    3. otherwise H, and the 80nan wins are labelled oracle-only.

    The default stays K.
- **Results:**
  - M − H: H better on kr-vs-kp_60nan, none on 8.
  - M − K: K better on 4 (kr-vs-kp_60nan, kr-vs-kp_80nan, credit-g_60nan, biodeg_60nan).
  - On the five numerical tables M equals H exactly. The mean squared error on the validation
    mask and M's RMSE ratio rank epochs the same way.
  - At 80nan, K beats the bar on kr-vs-kp and spambase; H and M on none.
  - K − H: K better on 3, worse on none (ADR 0015, a third seed set).
  - KC − K: KC better on 6 of 9, E40's named six (T20, a fourth seed set).
- **Conclusion:** reading 3. H is the deployable headline for claims that must transfer, and the
  80nan wins over the mean/mode fill are oracle-only. The default stays K.
- **Sources:** `docs/tickets/imputation-protocol/03-deployable-checkpoint.md` (Outcome);
  `scripts/experiments/checkpoint_run.py`, `checkpoint_report.py`, `run_checkpoint_queue.sh`.

### E44 · Confirmation on the 15 unused variants (2026-10-17)

- **Status:** done. Pre-registered before the first cell and before E43's outcome was read; read
  once on 2026-10-17.
- **Hypothesis:** the frozen configuration's lead over the bar on the development set (today's
  default on 101/202/303: 16 better, 3 no difference, 2 worse, by the review's rebuild)
  generalises to new draws and levels of the same tables.
- **Scenarios:**
  - **Configuration:** frozen post-E40 (ADRs 0013–0015, 0018 at 20nan only, 0019), with no
    inheritance at 40–80nan.
  - **Readouts:** K headline, H and M (`--score_checkpoints`), KC (`--score_calibrated`), on one
    trajectory per run.
  - **Variants:** pendigits, letter and electricity at 40/60/80nan; vehicle, biodeg and kc2 at
    40/80nan.
  - **Seeds and splits:** 707/808/909; `StratifiedKFold`, 5 folds; 45 cells, tag
    `experiment=confirmation-2026-10-17`.
  - **Opponents:** the bar of 5 and missForest live; E42's tuned bar and E41's strong tier
    offline on the same folds.
- **Measures and decision rule:**
  - **Primary:** K − the opponents, 15 fold pairs, with the verdict rule and ±1% equivalence.
  - **Win and loss (D4):** a win is better than both the bar of 5 and the tuned bar; a loss is
    worse than either.
  - **Reading:** supported if wins ≥ 8 and losses ≤ 2; refuted if losses ≥ 5 or wins ≤ 4; mixed
    otherwise.
  - **Secondary:** every opponent's tally; 80nan apart; the Wilcoxon over tables (secondary here);
    K/H/M; T20's threshold; the metric panel; context-free rows; twins.
- **Results:** 45 of 45 cells ran from 01:51 to 10:24 GMT-3.
  - **Reading: mixed.** 9 wins (pendigits and letter at 40/60nan, pendigits_80nan, vehicle_40nan,
    biodeg_40nan, kc2_40nan, kc2_80nan) and 3 losses: electricity_40nan, electricity_60nan, and
    letter_80nan against the tuned bar by +0.0018.
  - **Tallies of K:** against the bar of 5, 11/3/1; against the tuned bar, 9/3/3; against
    missForest and the strong tier, 12/1/2. Across tables, 5 of 6 are below the bar (p = 0.062).
  - **At 80nan:** 4/2/0 against the bar, 2/3/1 against the tuned bar.
  - **Deployability:** H against the bar is 11 better, 2 no difference, 1 equivalent, 1 worse; K
    beats H on 5, never worse.
  - **T20:** KC − K is better on 8, worse on none, including pendigits_40nan and letter_40nan, so
    threshold 40 is supported.
- **Conclusion:** the lead generalises on the large tables and at 40nan. electricity is the table
  where the model loses, now at three levels. At 80nan the lead mostly ties the tuned bar.
  - **Offline readings (added 15:45 GMT-3):** the ratio metrics never disagree with
    `impute_score`.
  - The only disagreements are distributional: abs_bias at 80nan (and letter_60nan), and on
    letter_80nan and pendigits_80nan fills more shrunk than the boosters' (dispersion,
    Wasserstein).
  - kc2's wins come from the non-twin cells; the model loses to knn on the twins.
  - Dropping the context-free cells changes no verdict.
- **Decided by the user (2026-10-17, about 22:45 GMT-3):**
  - the thesis's confirmation statement, as in the ticket;
  - electricity is a stated limit, with a bounded investigation later (T29);
  - T20 is revived: the calibration becomes the default at 40% missing and above (ADR 0027).
- **Sources:** `docs/tickets/imputation-protocol/04-confirmation.md` (Outcome);
  `scripts/experiments/confirmation_run.py`, `confirmation_report.py`, `run_confirmation_queue.sh`.

### E45 · MNAR smoke family (2026-10-17)

- **Status:** done. Pre-registered before the first cell; it ran after E44 on the same queues,
  from 10:24 to 11:29 GMT-3, before the 12:00 GMT-3 cutoff.
- **Hypothesis:** this is a smoke test of the MNAR machinery, read descriptively. Does the model's
  lead over the bar survive value-dependent missingness? And does the untied `[NULL]`/`[MASK]` arm
  gain under MNAR (t2's positive control)?
- **Scenarios:**
  - **Data:** `datasets/generate_mnar.py` (seed 1060) hides every column's MCAR count of cells
    with weights Phi(0.5 z), on biodeg, kc2 and vehicle at 60nan.
  - **Arms:** G (default, `--score_checkpoints`; H read) and Mx (`[NULL]` gaps, loss checkpoint,
    column-wise scoring).
  - **Cells:** MNAR and MCAR variants, seeds 42/7/13: 36 cells, tag
    `experiment=mnar-smoke-2026-10-17`.
  - **Integrity:** G on MCAR must equal E35. Its smoke cell did, exactly.
- **Measures and decision rule:**
  - **Readings:** G's H against the bar, missForest and the strong tier, per mechanism; the
    mechanism effect (MNAR − MCAR); K − H under MNAR against MCAR; t2 (Mx column-wise − G's H).
    All fold-paired, with the verdict rule. Signed bias offline.
  - **Rule:** descriptive. Go for the full family if all cells finish and the integrity check
    holds.
- **Results:** 36 of 36 cells; G on MCAR equals E35 on all 9 cells (largest difference 0).
  - **Reading: go for the full family.**
  - **G's H against the bar under MNAR:** better on all three tables (biodeg −0.017, kc2 −0.116,
    vehicle −0.040). Against missForest: worse on biodeg (+0.020), no difference on kc2 and
    vehicle.
  - **The mechanism effect** ((H − bar), MNAR minus MCAR): worse on biodeg (+0.081), no verdict on
    kc2 and vehicle.
  - **The oracle's optimism** (K − H, MNAR minus MCAR): no verdict on any table.
  - **t2:** Mx column-wise is worse than G's H under both mechanisms on biodeg and vehicle; its
    deficit shrinks under MNAR on biodeg only (−0.011).
- **Conclusion:** the machinery works and the lead survives value-dependent missingness on these
  three tables; biodeg is where MNAR costs the model, so it is the table for the tilt dial.
  - **The signed bias per column (read 2026-10-19):** every imputer falls further below the
    hidden values under MNAR. The model's bias moves least of 10 methods on all three pairs.
- **Sources:** `docs/tickets/imputation-protocol/05-mnar-smoke.md` (Outcome);
  `scripts/experiments/mnar_smoke_run.py`, `mnar_smoke_report.py`, `run_mnar_smoke_queue.sh`;
  `datasets/generate_mnar.py`.

### E46 · The full MNAR family (2026-10-17)

- **Status:** done. Pre-registered before the first cell, after E45's rule read "go for the full
  family" and on the user's request; it ran from 14:38 to 18:11 GMT-3.
- **Hypothesis:** the model's lead over the weak bar survives value-dependent missingness on a
  larger table (spambase at 60nan) and at a lighter rung (biodeg, kc2 and vehicle at 20nan); on
  biodeg at 60nan, its cost grows with the tilt's strength.
- **Scenarios:**
  - **Data:** `datasets/generate_mnar.py` (seed 1060), now with `--slope`. Seven new files: the four
    `_probit_` pairs at slope 0.5, and biodeg_60nan at slopes 0.25, 1.0 and 2.0. All checks passed,
    except that 3 heavy-tailed kc2 columns barely tilt at 20nan, so kc2_20nan is a weak tilt.
  - **Arms:** E45's G (H read) and Mx (column-wise), tag `experiment=mnar-family-2026-10-17`.
  - **Cells:** 48 on the new pairs (both mechanisms) and 18 on the dial; seeds 42/7/13; 66 cells in
    all, about 3.7 h on 4 queues. The dial's slope 0.5 and its MCAR control are E45's cells.
  - **Integrity:** G on the four MCAR controls must equal E35 fold by fold.
- **Measures and decision rule:**
  - **Primary:** G's H against the weak bar under MNAR on the 4 new pairs. Supported if better on
    ≥3 and worse on none; refuted if worse on ≥2 or better on ≤1; mixed otherwise.
  - **Dial:** dose-response if the mechanism effect's mean rises strictly over slopes
    0.25/0.5/1.0/2.0 and is "worse" at 2.0; separately, whether H still beats the bar at 2.0.
  - **Descriptive:** missForest (the strong tier is missForest alone, before ADR 0021's merge), the
    oracle's optimism, t2, and the rung (20nan beside 60nan).
- **Results:** 66 of 66 cells. G on the MCAR controls equals E35 on all 12 (largest difference 0).
  - **Primary: supported.** Under MNAR, H is better than the weak bar on all 4 new pairs
    (spambase_60 −0.014, biodeg_20 −0.035, kc2_20 −0.094, vehicle_20 −0.077).
  - **Mechanism effect:** worse on spambase (+0.026), none on biodeg_20 and vehicle_20, better on
    kc2_20 (the weak-tilt pair). On biodeg it grows from 20nan to 60nan (+0.078).
  - **Dial: dose-response "no".** The effect is +0.083, +0.081, +0.096 and +0.103 at slopes 0.25,
    0.5, 1.0 and 2.0: flat, then rising. H against the bar: better, better, none, worse.
    Against missForest the model improves with the slope (worse to better).
  - **The oracle's optimism:** no verdict anywhere.
  - **t2:** Mx's deficit shrinks with the tilt on biodeg, and Mx beats G at slope 2.0 (−0.006).
    No verdict on the new pairs, except kc2_20 (worse).
- **Conclusion:** the lead survives MNAR on 7 of 7 pairs (E45 and E46). It shrinks on spambase and
  on biodeg at 60nan, and is lost only at biodeg's steepest tilt.
  - **Signed bias (read 2026-10-19):** under MNAR every imputer under-predicts, and the model's
    bias moves least of 10 methods on 9 of 10 pairs (2nd at slope 0.25). Its fills are K's.
- **Sources:** `docs/tickets/imputation-protocol/06-mnar-family.md` (Outcome);
  `scripts/experiments/mnar_family_run.py`, `mnar_family_report.py`, `run_mnar_family_queue.sh`;
  `datasets/generate_mnar.py`.

### E47 · A deployable calibration (2026-10-18 – 2026-10-19)

- **Status:** done. Pre-registered before the first cell, with the user's agreement (2026-10-18),
  as T20's follow-up after ADR 0027; it ran from 23:07 to 01:37 GMT-3.
- **Hypothesis:** a calibration fitted on the fixed validation mask's cells (observed, so a
  deployed model can fit it), applied to H, keeps part of KC's gain without needing the validation
  gaps' truth.
- **Scenarios:**
  - **Arm:** one per cell, on one trajectory: H headline (`--decode_checkpoint loss`,
    `--decode_calibration none`), plus the readouts HC_m (`--score_masked_calibration`, new and
    off by default), HC_g, K and KC.
  - **Cells:** E40's 12 variants at 40nan and above, seeds 404/505/606; 36 cells, tag
    `experiment=deployable-calibration-2026-10-18`.
  - **Integrity:** K and KC must equal E40's cells.
- **Measures and decision rule:**
  - **Primary:** HC_m − H on the induced `impute_score`, with the verdict rule and the 1% SESOI.
    Supported if better on at least 4 of E40's 6 named variants and worse on none of the 12;
    refuted if worse on at least 2 or better on at most 1 named; mixed otherwise.
  - **Secondary (descriptive):** HC_g − H and the share of its gain HC_m keeps; the masked
    population; HC_m − K; the bars; α and the fit's cell counts.
  - **Supported** leads to a proposed ADR amending 0027; otherwise 0027 stands.
- **Results:** 36 of 36 cells; K and KC equal E40's on all 12 variants.
  - **Primary: supported.** HC_m is better than H on 4 of the 6 named variants (spambase_80nan,
    credit-g_60nan, credit-g_80nan, vehicle_60nan) and on biodeg_60nan. It is worse on none and
    equivalent on 7.
  - **Small:** −0.0004 to −0.0032 (at most 0.4% of H). It keeps 3–29% of HC_g's gain on the
    named variants.
  - **kr-vs-kp at 60/80nan is unchanged:** no categorical column has the 30 fit cells.
  - **α:** 0.84–0.99 for the masked fit, against 0.53–0.93 for the gaps' fit. HC_m stays below K
    on four variants.
- **Conclusion:** a deployable calibration helps, but little. The validation mask has too few
  cells for the fit. The seeds decided E40, so any ADR would need fresh seeds first.
  - **Decided by the user (2026-10-19):** stop. KC is reported as an oracle readout, with HC_m's
    small deployable gain stated.
- **Sources:** `docs/tickets/imputation-calibration/02-deployable-calibration.md` (Outcome);
  `scripts/experiments/deployable_calibration_run.py`, `deployable_calibration_report.py`,
  `run_deployable_calibration_queue.sh`.

### E48 · MaCoDE as an external deep baseline (2026-10-19)

- **Status:** done. Pre-registered before the first scored cell, on the user's go (2026-10-19); it
  ran from 07:57 to 18:58 GMT-3, off-store.
- **Hypothesis:** TRIDENT's lead over the classical baselines holds against the nearest deep prior
  art, MaCoDE: a masked transformer over quantile bins, with no `[NULL]`/`[MASK]` split and no
  pre-training stage.
- **Scenarios:**
  - **Code:** `src/training/macode.py`, vendored from commit 8867879 (MIT) with an inductive
    wrapper, the bins fix, one-type tables, the paper's settings, its own random streams and a
    posterior-mean fill. It is untuned, and TRIDENT was tuned on the development set.
  - **Cells:** the development set at 101/202/303 (63 pairs) and E44's 15 variants at 707/808/909
    (45 pairs), 540 fits, off-store (`scripts/experiments/macode_run.py`).
- **Measures and decision rule:**
  - **Primary:** K − MaCoDE per variant, tallied per set.
  - **Secondary:** MaCoDE against the strong tier's best, the bar and the tuned bar; K against the
    best of all; the fit time.
  - **Descriptive on development.** If MaCoDE beats the strong tier's best on at least 5 of 15
    confirmation variants, propose by ADR that it join the strong tier offline; otherwise it is an
    external deep baseline only. No default changes.
- **Results:** 108 of 108 pairs.
  - **K − MaCoDE:** better 20 / none 1 on the development set and 14 / 1 on the confirmation set;
    worse on none.
  - **MaCoDE − strong:** better 5 / worse 16 on the development set and 4 / 1 / 10 on the
    confirmation set. It is strong on kr-vs-kp, letter, electricity and spambase_40nan, and weak
    (below the weak bar) on credit-g, vehicle, biodeg, kc2 and pendigits.
  - **K − best of all,** MaCoDE included: unchanged, 13/6/2 and 10/2/3.
- **Conclusion:** the nearest deep prior art does not change any of TRIDENT's verdicts. By the rule
  it stays an external deep baseline (4 of 15 confirmation variants, fewer than 5). MaCoDE ran
  untuned at the paper's settings.
- **Sources:** `docs/tickets/imputation-protocol/07-macode.md` (Outcome); `src/training/macode.py`;
  `scripts/experiments/macode_run.py`, `macode_report.py`.

### E49 · D1 of T28: the decode curves, pre-trained against not (2026-10-19)

- **Status:** done. Pre-registered before any curve was read, on the user's request (2026-10-19).
- **Hypothesis (H3 of T28's report):** the decode stage erases pre-training's head start
  (washout).
- **Scenarios:**
  - **Data:** each parent's `cv/decode/val_loss/mean` per epoch, from E24, E25 and E26. No GPU and
    no new run.
  - **Pairs:** A, N and V against B (150 decode epochs), and M against C (450), by variant and
    seed.
- **Measures and decision rule:**
  - **Measures:** the relative gap at epochs 0, 10 and 50, the catch-up epoch and the gap at the
    minima.
  - **Reading per variant and arm:** real transfer if the gap at the minima is at least +1% on
    every seed; washout if not, and the head start at epoch 0 is at least +1% on every seed;
    nothing usable if the head start is below +1% on every seed; mixed otherwise.
  - **Limit:** it is the decode loss, not `impute_score`.
- **Results:** 48 pairs.
  - **pendigits_20nan:** normalised pre-training gives a +58% head start at epoch 0. It lasts
    through 150 decode epochs (N and V: real transfer). At 450, the arm without pre-training
    catches up by about epoch 185 (M − C: washout, −0.5% at the minima).
  - **kr-vs-kp_40nan:** M − C is real transfer (+2.5% at the minima).
  - **credit-g:** mixed.
  - **The shipped objective (A)** hurts pendigits from epoch 0.
- **Conclusion:** H3 holds on pendigits, and kr-vs-kp keeps a real gain. The next lever is a
  frozen-encoder start or a lower encoder learning rate in the decode stage (T28 §5), before T13.
  D3 would locate kr-vs-kp's gain. This is the decode loss, not `impute_score`.
- **Sources:** `docs/tickets/imputation-pretraining/08-decode-curves.md` (Outcome);
  `scripts/experiments/decode_curves_report.py`.

### E50 · The decode stage's levers against washing out the encoder (2026-10-19 – 10-20)

- **Status:** done 2026-10-20. Pre-registered before the first cell, after E49 and with the
  user's agreement (2026-10-19).
- **Hypothesis (T28, H3):** holding the pre-trained encoder fixed for the first decode epochs, or
  giving it a lower learning rate, keeps what pre-training gave pendigits and improves its
  `impute_score`, without costing kr-vs-kp.
- **Scenarios:**
  - **Code:** `--decode_freeze_epochs` and `--decode_encoder_lr_scale`, with defaults 0 and 1.0,
    their tags, and a dry-run tag backfill. The fixtures are unchanged.
  - **Arms:** G (today's default), F100 (heads only for 100 of 450 decode epochs), L10 (encoder
    LR ×0.1) and C0 (no pre-training), all with `--score_checkpoints`.
  - **Cells:** pendigits_20nan and kr-vs-kp_40nan, seeds 101/202/303; 24 cells, tag
    `experiment=encoder-levers-2026-10-19`.
  - **Integrity:** G equals E38 ALL and E39 D.
- **Measures and decision rule:**
  - **Primary:** F100 − G and L10 − G on K, induced.
  - **Rule:** supported if a lever is better on pendigits and not worse on kr-vs-kp; refuted if
    neither is better on pendigits; mixed otherwise.
  - **Secondary:** H, masked, C0 − G, and the decode-loss minima.
  - **Supported** leads to a 21-variant screening; no default changes.
- **Results:** 24 of 24 cells; integrity passed (G equals E38 ALL and E39 D, difference 0).
  - **F100 − G:** pendigits +0.0109 (+3.4%) worse; kr-vs-kp +0.0016, no difference.
  - **L10 − G:** pendigits +0.0371 (+11.6%) worse; kr-vs-kp +0.0322 (+4.4%) worse.
  - **C0 − G:** pendigits +0.0038 (+1.2%) worse on K and H; kr-vs-kp equivalent on K.
  - **Decode-loss minima:** both levers above G on both variants.
- **Conclusion:** refuted. The decode stage needs the encoder to move at full rate from the
  start. The washout levers are closed, and their flags stay at 0 and 1.0. T28 moves to D2 and
  D3. Under today's default, pre-training still leaves pendigits 1.2% better at K and gives
  kr-vs-kp_40nan nothing. Development seeds.
- **Sources:** `docs/tickets/imputation-pretraining/09-encoder-levers.md` (Outcome);
  `scripts/experiments/encoder_levers_run.py`, `encoder_levers_report.py`,
  `run_encoder_levers_queue.sh`.

### E51 · Today's default on fresh seeds at 20nan (2026-10-20)

- **Status:** done 2026-10-20. Pre-registered before the first GPU cell, on the user's request to use
  the idle GPU until 08:00 GMT-3 on 2026-10-20 for the variants not run under today's defaults for
  longest (2026-10-19).
- **Hypothesis:** today's default keeps its earlier calls against the bar on the nine `_20nan`
  variants on a seed set it has never seen. Their latest default runs are from 10-03 to 10-06.
- **Scenarios:**
  - **Cells:** electricity, letter, pendigits, spambase, kr-vs-kp, biodeg, credit-g, vehicle and kc2
    at 20nan × seeds 1001/2002/3003 (unused). 27 cells, tag `experiment=default-refresh-2026-10-20`.
  - **The configuration:** today's default as a default run reads it, with `--score_checkpoints`
    and `--score_calibrated`; the strong tier live.
  - **Running it:** the baseline cache warmed on the CPU first. Four GPU queues start as E50's end.
    No cell starts unless it can end by 08:00 GMT-3.
  - **Integrity:** kc2_20nan s42 with `--no-mlflow` equals E35 fold by fold.
- **Measures and decision rule:**
  - **Primary:** K − bar, induced, 15 fold pairs, compared with the calls on 101/202/303 and
    42/7/13.
  - **Rule:** holds if no variant reverses (better ↔ worse); drifts otherwise, each reversal
    listed and ticketed. No default changes.
  - **Secondary:** K − strong (live), missForest, best of all, per-type bar; H − bar, M − bar,
    K − H, K − M, KC − K.
- **Results:** 27 of 27 cells, 01:14 to 06:35 GMT-3.
  - **K − bar:** better 7, no difference 1 (kc2), worse 1 (electricity, −2.3% of K), with no
    reversal.
    - Two changes short of a reversal, each matching the other earlier set: electricity none →
      worse, spambase none → better.
  - **K − strong (live):** better 6, no difference 3, worse 0. credit-g moves from no difference to
    better.
  - **H − bar:** the same calls as K − bar.
  - **KC − K:** better on letter (−2.1%) and pendigits (−2.8%), against about −0.4% on E34 and E35's
    old encoder; worse on spambase (+0.6%).
- **Conclusion:** holds. Today's default keeps its 20nan calls on an unseen seed set. Whether the
  threshold for the two bigger-encoder variants should move is the user's call (T20); no default
  changes.
- **Sources:** `docs/tickets/imputation-protocol/08-default-refresh.md` (Outcome);
  `scripts/experiments/default_refresh_run.py`, `default_refresh_report.py`,
  `run_default_refresh_queue.sh`.

### E52 · T28's D3: a frozen encoder, pre-trained against random (2026-10-20)

- **Status:** done 2026-10-20. Pre-registered before the first cell, on the user's go for T28's
  D2 and D3 (2026-10-20).
- **Hypothesis (T28, H1 against H3):** today's normalised pre-training leaves a representation that
  heads alone can decode better than they decode a never-trained encoder.
- **Scenarios:**
  - **Arms:** FN (pre-trained, encoder frozen for the whole decode budget), FA (the same after the
    shipped objective), FR (`EPOCHS_PRE 0`, frozen) and C0 (a full decode without pre-training, on
    credit-g; E50's on the other two). G is read from E50 and the development runs.
  - **Cells:** pendigits_20nan, kr-vs-kp_40nan, credit-g_20nan and credit-g_80nan, seeds
    101/202/303; 42 cells, tag `experiment=frozen-encoder-2026-10-20`.
  - **Code:** none in `src/`; E50's freeze flag at the decode budget. A `--check-frozen` smoke shows
    the encoder bit-identical after decoding.
- **Measures and decision rule:**
  - **Primary:** FN − FR on K, induced, per variant.
  - **Rule:** usable if FN is better on 3 or more variants; nothing usable if on none; mixed
    otherwise.
  - **Secondary:** FA − FR, FN − FA, FN − G, FR − C0 and C0 − G, also on H and masked.
  - **With D2 (E53):** the decision table in the ticket, fixed now. No default changes.
- **Results:** 42 of 42 cells, 19:21 to 22:19 GMT-3.
  - **FN − FR on K:** pendigits −0.4392 (−53%), kr-vs-kp −0.1574 (−16%), credit-g_20nan −0.8%
    (no difference), credit-g_80nan −0.2% (equivalent).
  - **FA:** below FN everywhere it differs, and worse than random on kr-vs-kp (+4.1%) and
    credit-g_20nan (+2.6%).
  - **FN − G:** +21% on pendigits, +10% on kr-vs-kp and on credit-g_20nan.
  - **C0 − G:** +1.2%, equivalent, +1.7% and +0.6%.
- **Conclusion:** mixed (2 of 4). The representation is real on pendigits and kr-vs-kp, and the
  full decode learns one as good from scratch. With E53 (no collapse, value information present,
  `BATCH 64` moves credit-g's probe), the decision table's last row applies: T12 first, with a
  batch arm, for the user to decide. No default changes. Development seeds.
- **Sources:** `docs/tickets/imputation-pretraining/10-frozen-encoder.md` (Outcome);
  `scripts/experiments/frozen_encoder_run.py`, `frozen_encoder_report.py`,
  `run_frozen_encoder_queue.sh`.

### E53 · T28's D2: collapse and linear probes on pre-training, and the encoder's drift (2026-10-20)

- **Status:** done 2026-10-20. Pre-registered before the first study cell, with E52 (2026-10-20).
- **Hypotheses (T28):**
  - **H1:** under N the stage collapses onto the column-constant part of its target.
  - **H2:** the encoder learns nothing linearly decodable about the hidden value.
  - **H5:** a lower `LR_PRE` or a smaller batch changes either.
- **Scenarios:**
  - **Cells:** probe cells on the first fold: N and A on credit-g_20nan, kr-vs-kp_40nan and
    pendigits_20nan; N_lr4 and N_b64 on credit-g_20nan; seeds 101/202/303 (24, CPU). Drift cells:
    N and the Xavier start, then today's decode, on E52's four variants at seed 101 (8, GPU).
  - **Code:** `scripts/experiments/pretrain_probe.py` watches `train_pretrainer` through its
    tracker with every random stream restored. `fidelity`: identical validation losses. Offline;
    nothing is written to the store.
- **Measures and decision rule:**
  - **Measures:** E04's NMSE, centred cosine and within-column share (also of the raw target);
    effective ranks; linear probes from the transformer's output to the hidden value (1.0 = no
    information); CKA and relative weight change across the decode stage.
  - **Rule:** N collapsing if collapse (share falls, |cos| < 2 SE, NMSE ≥ 1, all seeds) on 2 or
    more of 3 tables. Value information if the probe gains more than 1% with seed agreement on 2
    or more of 3. H5 if N_lr4 or N_b64 moves a call or the probe by more than 1%.
  - **Then:** E52's decision table. No default changes.
- **Results:** 24 probe cells and 8 drift cells.
  - **Collapse under N:** none. pendigits' within-column share rises from 0.10 to 0.30 (NMSE 0.29).
    credit-g's share falls 14% with a negative centred cosine (−4.6 to −16.7 SE).
  - **Probes, start to end:** pendigits 0.787 → 0.506 (+35.8%), kr-vs-kp 0.954 → 0.882 (+7.5%),
    credit-g +1.0% with one seed worse (absent).
  - **H5:** `LR_PRE` ÷ 4 does not move; `BATCH 64` moves credit-g's probe by −0.016.
  - **A:** erodes its target 40–55% (E04) and its probes gain less than N's.
  - **Drift:** the decode keeps a CKA of 0.40 (pendigits) and 0.60 (kr-vs-kp) with N's start,
    0.03 and 0.53 with random. On those two tables both starts end at about the same score.
- **Conclusion:** no collapse, value information present; washout (H3) is the mechanism. With
  E52, the decision table's last row applies: T12 first with a `BATCH 64` arm, for the user to
  decide. No default changes. Development seeds.
- **Sources:** `docs/tickets/imputation-pretraining/11-collapse-probes.md` (Outcome);
  `scripts/experiments/pretrain_probe.py`.

### E54 · T29's test: is the boosters' lead on electricity carried by `date` as a key? (2026-10-20)

- **Status:** done 2026-10-20. Pre-registered before any subset was computed, after T29's
  report (2026-10-20).
- **Hypothesis:** hgb and lgbm beat the model on electricity by looking values up by `date`, a
  933-level key (48 rows per date, `day` constant within it). The model reads `date` as one scalar.
- **Scenarios:**
  - **Runs:** E51's electricity_20nan (seeds 1001/2002/3003) and E44's 40/60/80nan (707/808/909).
  - **Fills:** the model's from the fold ledgers; hgb and lgbm refitted on the same folds.
    Integrity: the full-cell scores equal the runs' to 1e-9.
  - **Subsets:** the seven non-date columns' cells, split by whether the row's `date` is observed;
    secondary splits for `day`, `nswdemand` and the constant regime.
- **Measures and decision rule:**
  - **Primary:** K − bar (the run's own booster) on each side, 15 fold pairs, 1% SESOI, at 20 and
    60nan.
  - **Rule:** supported if the date-observed side is worse and the date-missing side not worse at
    both; refuted if the date-missing side is also worse at both; mixed otherwise.
  - **Then:** supported → the limit is stated with its mechanism, and any model-side fix is an ADR
    for the user. Otherwise the limit is stated as is. No default changes.
- **Results:** 12 of 12 cells; integrity passed on all 60 folds.
  - **Date observed:** K − bar worse at every rate: +0.0252, +0.0253, +0.0334 and +0.0546 at 20,
    40, 60 and 80nan.
  - **Date missing:** never worse: −0.0341 (better), +0.0542 (no difference), −0.0016 (no
    difference), −0.0127 (better).
  - **`day`:** +0.13 to +0.05 with the date, no difference or better without it.
  - **The regime split** does not separate the two sides consistently.
- **Conclusion:** supported. The boosters look values up by `date`, a 933-level key; without it,
  the model is the better imputer. The thesis states the limit with this mechanism. A model-side
  response is an ADR for the user. T29 closes; no default changes.
- **Sources:** `docs/tickets/imputation-protocol/09-electricity.md` (Outcome);
  `docs/reports/2026-10-20-t29-electricity.md`; `scripts/experiments/electricity_report.py`,
  `electricity_split.py`.

### E55 · TabPFN v2 as an external foundation-model baseline (2026-10-21)

- **Status:** done 2026-10-21 (01:00–03:48 GMT-3, 108 of 108 pairs). Pre-registered before the
  first scored cell, on the user's go (2026-10-20).
- **Hypothesis:** none directional. TabPFN v2 is the foundation-model baseline a reader will ask
  about. The question is how K compares, and whether it beats the strong tier often enough to
  join it.
- **Scenarios:**
  - **Code:** the optional extra `tabpfn==9.2.0` (lock: additions only) and
    `src/training/tabpfn_baseline.py`, a column-wise wrapper over v2's checkpoints (10,000-row
    context, 8 estimators).
  - **Cells:** E48's: E44's confirmation set (15 variants × 707/808/909) first, then the
    development set (21 × 101/202/303); 108 pairs.
  - **Running it:** offline, two GPU lanes from 01:00 GMT-3. No pair starts unless it can end by
    08:00; incomplete variants are not read.
- **Measures and decision rule:**
  - **Primary:** K − TabPFN per variant, 15 fold pairs, 1% SESOI.
  - **Secondary:** TabPFN − strong, − bar, − tuned bar; K − best of all; fit time.
  - **Rule (E48's):** TabPFN joins the strong tier (an ADR for the user) only if it is better than
    the strong tier's best on 5 or more of 15 confirmation variants; otherwise an external
    baseline. No default changes.
- **Results:**
  - **TabPFN − strong:** better on 11 of 15 confirmation variants, worse on none (development 11,
    5 and 1 equivalent, 4).
  - **TabPFN − bar:** better on 13 of 15.
  - **K − TabPFN:**
    - confirmation: better 6, no difference 4, worse 5;
    - development: better 8, no difference 6, worse 7;
    - confirmation 40/60nan: never worse;
    - confirmation 80nan: worse on 5 of 6.
  - **K beats TabPFN** on spambase, letter, pendigits and biodeg below 80nan. TabPFN wins on
    kr-vs-kp and credit-g at 40nan and above.
- **Conclusion:** TabPFN is the strongest baseline so far. By the rule, propose it for the strong
  tier (an ADR amending 0021, the user's decision). K's lead holds at 20–60nan on most tables and
  fails at 80nan. No default changes.
- **Sources:** `docs/tickets/imputation-protocol/10-tabpfn.md` (Outcome); `src/training/tabpfn_baseline.py`;
  `scripts/experiments/tabpfn_run.py`, `tabpfn_report.py`, `run_tabpfn_lanes.py`.

### E56 · ADR 0025's panel on K against TabPFN, confirmation set (2026-10-21)

- **Status:** done 2026-10-21 (03:53–04:53 GMT-3, 45 of 45). Pre-registered before the first
  cell, the panel check ADR 0025 asks for behind E55's new claim.
- **Hypothesis:** none. The check is whether K − TabPFN's `impute_score` calls hold in the
  per-kind ratios.
- **Scenarios:**
  - **Cells:** E44's 15 confirmation variants × 707/808/909. The model's fills come from the
    ledgers; TabPFN is refitted on the same folds.
  - **Code:** `metric_panel.py --fitters tabpfn` and `metric_panel_report.py --opponent tabpfn`,
    with the defaults unchanged.
  - **Running it:** two GPU lanes; no pair starts after 07:30 GMT-3.
- **Measures and decision rule:** ADR 0025's reading.
  - A call stands where the per-kind ratios agree, and disagreements are listed.
  - Distributional metrics are caveats.
  - No decision follows here.
- **Results:**
  - **Integrity:** the model's fold values equal the runs' K; TabPFN's equal E55's to 3.6e-4.
  - **RMSE ratio:** agrees with `impute_score` on all 15 variants.
  - **MAE:** reverses K's wins on biodeg_40nan and pendigits_40nan.
  - **electricity_80nan:** worse on the numbers, better on `day`.
  - **Dispersion:** K better on 14 of 15; Wasserstein better on 11.
- **Conclusion:** E55's calls stand for point accuracy. TabPFN's 80nan wins come with fills more
  shrunk than K's, a caveat under ADR 0025 decision 4. It is reported with E55's proposal; no
  decision here.
- **Sources:** `docs/tickets/imputation-protocol/11-tabpfn-panel.md` (Outcome);
  `scripts/experiments/metric_panel.py`, `metric_panel_report.py`.

### E57 · The default output against TabPFN: KC and H (2026-10-21)

- **Status:** done 2026-10-21. Pre-registered before the reading ran.
- **Hypothesis:** none directional. KC is the default output at 40nan and above (ADR 0027), and
  E55 read K.
- **Scenarios:** E44's 15 confirmation variants (707/808/909) with their logged KC and H. The
  development variants at 40nan and above are read where their runs have KC.
- **Measures and decision rule:** KC − TabPFN and H − TabPFN beside K − TabPFN, 15 fold pairs,
  1% SESOI; the calls that change from K to KC are listed. Descriptive; ADR 0028 stands.
- **Results:**
  - **KC − TabPFN (confirmation):** better 6, no difference 4, equivalent 1, worse 4. The losses
    are pendigits, electricity, vehicle and biodeg at 80nan.
  - **The one change from K:** letter_80nan (worse → equivalent).
  - **Calibration narrows the 80nan gaps** by up to half (vehicle +0.053 → +0.032).
  - **H − TabPFN:** better 5, no difference 5, worse 5.
- **Conclusion:** the thesis states KC − TabPFN: the default output never loses to TabPFN at 40–60nan
  and loses on 4 of 6 at 80nan.
- **Sources:** `docs/tickets/imputation-protocol/12-default-output-vs-tabpfn.md` (Outcome);
  `scripts/experiments/tabpfn_report.py`.

### E58 · The strong tier with TabPFN on the seed sets E55 did not cover (2026-10-21)

- **Status:** done 2026-10-21 (10:06–14:07 GMT-3, 156 of 156). Pre-registered before the first
  cell, on the user's go.
- **Hypothesis:** none. ADR 0028's strong tier is read on every seed set the thesis uses.
- **Scenarios:** TabPFN v2 offline, as in E55, on 156 pairs:
  - 21 development variants × 42/7/13;
  - E40's 12 × 404/505/606;
  - E51's nine × 1001/2002/3003;
  - ten MNAR variants × 42/7/13.
  - Two GPU lanes, MNAR first; nothing to the store.
- **Measures:**
  - K − strong_live, K − strong and K − TabPFN on each seed set;
  - for MNAR, H − the same, and (H − TabPFN) MNAR minus MCAR.
  - Descriptive; no default changes.
- **Results:**
  - **K − strong (with TabPFN):**
    - 42/7/13: 3 better, 9 no difference, 4 worse (live four: 8/7/1);
    - 404/505/606: 3/2/7 (live four: 6/4/2);
    - E51's fresh seeds: 1/7/1 (live four: 6/3/0).
  - **TabPFN's wins repeat** on credit-g and kr-vs-kp from 40nan and on pendigits_20nan; K's on
    spambase, letter_20nan, kc2_60nan and biodeg_60nan.
  - **MNAR:** H − TabPFN better 6, no difference 4, worse 0. (H − TabPFN) MNAR minus MCAR is never
    worse.
  - **The reader:** a correction for the dial pairs' control source, made before their lines were
    read.
- **Conclusion:** ADR 0028's tier reads the same way on every seed set. Under MNAR the model keeps
  its standing against TabPFN. No default changes.
- **Sources:** `docs/tickets/imputation-protocol/13-tabpfn-seed-sets.md` (Outcome);
  `scripts/experiments/run_tabpfn_lanes.py`, `tabpfn_seed_sets_report.py`.

### E59 · ADR 0018's bigger encoder on the large tables at 40–80nan (2026-10-21 – 2026-10-22)

- **Status:** done 2026-10-22, 06:03 GMT-3. Pre-registered before the first cell, on the user's go
  (2026-10-21).
- **Hypothesis:** the bigger encoder (ALL), which helped the large tables at 20nan (E38), also helps
  them at 40–80nan, where TabPFN leads at 80nan.
- **Scenarios:** ALL on pendigits and letter at 40/60/80nan, seeds 707/808/909: 18 cells. G is E44's
  default runs on the same folds. A development reading.
- **Measures and decision rule:**
  - **Primary:** ALL − G on K. Supported if better on 4 or more of 6 and worse on none; refuted if
    better on 1 or none, or worse on 2 or more.
  - **Secondary:** KC and H, and KC − TabPFN per arm.
  - **Supported** leads to a fresh-seed confirmation before any ADR amending 0018.
- **Results:**
  - 18 of 18 cells, 19:28 GMT-3 on 10-21 to 04:11 GMT-3 on 10-22 (8 h 43 min).
  - **ALL − G on K, as a share of G's K:** pendigits −4.2% / −1.2% / +0.2% and letter −3.8% /
    −0.1% / +0.4% at 40/60/80nan. Better 3, equivalent 2, worse 1 (letter_80nan, interval inside
    the 1% SESOI).
  - **On KC:** better 4 (both tables at 40 and 60nan), equivalent 2, worse 0. **On H:** better 2,
    no difference 2, equivalent 1, worse 1.
  - **KC − TabPFN at 80nan:** pendigits worse with both encoders (+0.0121 → +0.0104); letter
    equivalent with both.
- **Conclusion:** mixed, and mixed without the worse call too (3 better, 4 needed). The bigger
  encoder's gain fades with the missing rate and is gone at 80nan, so it does not close the gap to
  TabPFN there. No default changes.
- **Decision (2026-10-22, 06:10 GMT-3):** the user keeps ADR 0018 at 20nan and records the bigger
  encoder at 40–60nan as a future possibility (T32). The measured cost is about 2× the training
  time (1.98× pendigits, 2.05× letter).
- **Sources:** `docs/tickets/imputation-protocol/14-large-encoder.md` (Outcome);
  `scripts/experiments/large_encoder_run.py`, `large_encoder_report.py`,
  `run_large_encoder_queue.sh`.

### E60 · Are TRIDENT's and TabPFN's errors complementary? A blend of their fills (2026-10-21)

- **Status:** done 2026-10-22, 00:05 GMT-3. Pre-registered before the first cell, on the user's go
  (2026-10-21).
- **Hypothesis:** none directional. If the two err on different cells, the average of their fills
  beats both.
- **Scenarios:** E44's 15 confirmation variants × 707/808/909. The model's K fills come from the
  ledgers; TabPFN is refitted on the same folds. The blend averages numerical fills and keeps the
  model's categories. Offline, one GPU lane.
- **Measures and decision rule:**
  - Blend − K and blend − TabPFN, 15 fold pairs.
  - Complementary if the blend beats both on 5 or more of 15; not complementary on 1 or none;
    mixed otherwise.
  - Descriptive.
- **Results:**
  - 45 of 45 pairs; the model's ledger score equals the run's K on every fold.
  - **The blend beats both on 10 of 15:** pendigits and letter at 40/60/80nan, electricity_60nan,
    and vehicle, biodeg and kc2 at 40nan.
  - **Blend − K:** better 14, no difference 1 (kc2_80nan), worse 0.
  - **Blend − TabPFN:** better 11, no difference 1 (electricity_40nan), equivalent 3 (electricity,
    vehicle and biodeg at 80nan), worse 0.
  - **The zero "worse" calls are arithmetic:** an averaged numerical fill's RMSE is never above
    the mean of the two RMSEs. Only "better than the better one" is evidence.
- **Conclusion:** complementary. The two fills' errors are not parallel: the blend gains on both at
  40–60nan and on the large tables at 80nan. Elsewhere at 80nan it lands on TabPFN's level. The
  model side is K, not KC. Descriptive; no default changes.
- **Sources:** `docs/tickets/imputation-protocol/15-tabpfn-blend.md` (Outcome);
  `scripts/experiments/tabpfn_blend.py`.

### E61 · Where does K lose to TabPFN on kr-vs-kp and credit-g? A per-column read (2026-10-22)

- **Status:** done 2026-10-22, 06:33 GMT-3. Pre-registered before the first cell, on the user's go
  (2026-10-22).
  T10's first step.
- **Hypothesis:** none directional. A per-column loss weighting can only help if K's deficit to
  TabPFN sits in columns where K learned just the marginal. Under MCAR, a plain column mean equals
  the pooled term the loss already uses.
- **Scenarios:**
  - E40's runs of kr-vs-kp and credit-g at 40/60/80nan, seeds 404/505/606, 18 pairs;
  - TabPFN v2 refitted per fold, as E60 did;
  - errors per column from the ledgers; offline, two GPU lanes.
- **Measures and decision rule:**
  - K − TabPFN split exactly into its categorical and numerical terms. A variant counts if the
    categorical term carries at least half of the gap.
  - A column is "stuck" if K's gain over the mode is at most 5% while TabPFN's is at least 20%.
  - A target if stuck columns hold at least 50% of the net excess errors on 3 or more counted
    variants; no target on 1 or none; mixed otherwise.
- **Results:**
  - 18 of 18 pairs, 06:20–06:32 GMT-3. The model's ledger score equals K on every fold; TabPFN is
    within 1.0e-3 of E58's.
  - All 6 variants are counted (the categorical term carries 71–100% of the gap). No column is
    stuck at the marginal on any of them.
  - At 40–60nan, K's gain over the mode trails TabPFN's by about 3–10 points column by column. The
    per-column gains correlate at +0.76 to +0.94.
  - The top 20% of columns hold 53–89% of the excess (TabPFN's own errors: 38–47%), but they are
    columns K has learned.
  - credit-g_80nan: both are at the marginal, and K is a little below the mode, which KC's
    fallback addresses.
- **Conclusion:** no target. K's deficit is a uniform shortfall in conditional structure, not
  columns it missed, so no per-column weighting can move it. T10's weighting half closes, and its
  evaluation-mask half stays in the backlog. No default changes.
- **Sources:** `docs/tickets/imputation-protocol/16-categorical-deficit.md` (Outcome);
  `scripts/experiments/categorical_deficit.py`.

### E62 · Fresh MCAR draws: is a result tied to one missingness pattern? (2026-10-22)

- **Status:** done 2026-10-22, 11:39 GMT-3. Pre-registered before the first cell, on the user's go
  (2026-10-22). T31.
- **Hypothesis:** none directional. Each MCAR rate is one draw, so a call could be tied to which
  cells happen to be missing.
- **Scenarios:**
  - pendigits_20nan, kr-vs-kp_40nan, credit-g_60nan and biodeg_80nan;
  - three draws each: the shipped file, reproduced by the generator cell for cell, and two fresh
    ones at the same per-column counts;
  - seeds 42/7/13, 36 cells of today's default, each draw with its shipped variant's resolved
    configuration;
  - TabPFN as an offline pass.
- **Measures and decision rule:**
  - Per draw: K − bar, K − strong, K − TabPFN and KC − strong.
  - Across draws: same, reversal or drift.
  - **Primary:** holds if K − bar and K − strong show no reversal on any variant; tied to the draw
    otherwise, each reversal named.
- **Results:**
  - 36 of 36 cells and TabPFN pairs, 06:44–11:39 GMT-3. The gate (credit-g_60nan s42 against
    E35) matched at 0.
  - **K − bar:** better on every draw of pendigits_20nan, kr-vs-kp_40nan and biodeg_80nan. On
    credit-g_60nan it drifts (shipped: no difference; fresh: better).
  - **K − strong:** worse on every draw of all four, TabPFN the best each time.
  - **KC − strong:** worse throughout, except on pendigits_20nan (shipped: no difference; fresh:
    worse), where KC is not the default.
  - The shipped draw is not systematically favourable to K.
  - The span of the draws' means is 0.2–1.5× a draw's interval half-width, and 1.5–1.9× on
    biodeg_80nan.
- **Conclusion:** holds. No call depends on the MCAR draw on these four tables. The thesis's
  intervals are conditional on the draw, which adds variation of the same order. No default
  changes.
- **Sources:** `docs/tickets/imputation-protocol/17-mcar-draws.md` (Outcome);
  `datasets/generate_mcar_draws.py`; `scripts/experiments/mcar_draws_run.py`,
  `mcar_draws_report.py`, `run_mcar_draws_queue.sh`.

### E63 · A MAR smoke against matched MCAR controls (2026-10-22)

- **Status:** done 2026-10-22, 15:12 GMT-3. Pre-registered before the first cell, on the user's go
  (2026-10-22). T30.
- **Hypothesis:** none directional. MAR, missingness that depends on other observed columns, is the
  one classical mechanism not yet tested.
- **Scenarios:**
  - biodeg, kc2, vehicle and kr-vs-kp at 60nan;
  - a MAR variant (weights sigmoid(1.5 s) on 30% fully observed drivers) and a matched MCAR control
    (the same drivers and per-column counts, drawn uniformly);
  - seeds 42/7/13, 24 cells of today's default;
  - TabPFN as an offline pass.
- **Measures and decision rule:**
  - Per mechanism: H and K against the bar, the strong tier and TabPFN.
  - **Primary:** (H − bar) and (H − strong), MAR minus control. It holds if neither is worse on
    any table, loses ground if worse on 2 or more, and is mixed otherwise.
- **Results:**
  - 24 of 24 cells and TabPFN pairs, 13:44–15:10 GMT-3.
  - **(H − bar), MAR minus control:** better 1 (kc2), no difference 3, worse 0.
  - **(H − strong), MAR minus control:** worse on biodeg (+0.0410) and kr-vs-kp (+0.0103), better
    on kc2 (−0.1124), no difference on vehicle. TabPFN was the strong tier's best each time.
  - **(K − H), MAR minus control:** no difference on all four.
- **Conclusion:** loses ground. Under MAR, TRIDENT keeps its standing against the bar but loses
  ground against TabPFN on two of four tables, unlike under MNAR. One untested hypothesis is that
  the decode masks are uniform over observed cells while the test gaps follow the drivers. A
  fuller MAR family is proposed for the user. No default changes.
- **Sources:** `docs/tickets/imputation-protocol/18-mar-smoke.md` (Outcome);
  `datasets/generate_mar.py`; `scripts/experiments/mar_smoke_run.py`, `mar_smoke_report.py`,
  `run_mar_smoke_queue.sh`.

### E64 · The MAR family: does MAR cost the model ground against the strong tier? (2026-10-22 – 2026-10-23)

- **Status:** done 2026-10-23, 04:19 GMT-3. Pre-registered before the first cell, on the user's go
  (2026-10-22). T30.
- **Hypothesis:** E63's smoke found MAR costing the model ground against TabPFN on biodeg and
  kr-vs-kp (and the opposite on kc2). Does that hold more broadly?
- **Scenarios:**
  - biodeg, kc2, vehicle, kr-vs-kp, spambase and credit-g at 20 and 60nan;
  - a new MAR draw (generator seed 4060) and matched controls;
  - a slope dial (0.75, 1.5, 3) on biodeg_60nan;
  - seeds 404/505/606, 78 cells of today's default;
  - TabPFN as an offline pass.
- **Measures and decision rule:**
  - (H − bar), (H − strong) and (K − H), MAR minus control, fold-paired.
  - **Primary,** (H − strong) over the 12 table-rates: supported if worse on 4 or more and better
    on at most 1; refuted if worse on at most 1; mixed otherwise.
- **Results:**
  - 78 of 78 cells. Two spambase cells died of CUDA out of memory, from my concurrent scheduling,
    and were rerun before the read.
  - **(H − strong), MAR minus control:** worse on biodeg_60nan only (+0.0291), no difference on 11.
  - **(H − bar), MAR minus control:** worse on biodeg_60nan, kc2_60nan and vehicle_60nan, no
    difference on 9.
  - **(K − H):** no difference on 12.
  - **The dial:** not monotone.
- **Conclusion:** refuted. Against the strong tier, MAR does not broadly cost TRIDENT ground.
  biodeg_60nan is a replicated exception, and the smoke's kr-vs-kp is not. The bar gains at 60nan
  on the small numerical tables. E65 is read as exploratory, and no default changes.
- **Sources:** `docs/tickets/imputation-protocol/19-mar-family.md` (Outcome);
  `datasets/generate_mar_family.py`; `scripts/experiments/mar_family_run.py`,
  `mar_family_report.py`, `run_mar_family_queue.sh`.

### E65 · Do decode masks that follow the missingness help under MAR? (2026-10-22 – 2026-10-23)

- **Status:** done 2026-10-23, 05:43 GMT-3. Pre-registered before the first cell, on the user's go
  (2026-10-22).
- **Hypothesis:** E63's. The model loses ground under MAR because its decode masks are uniform over
  observed cells while the test gaps follow the drivers. Masks weighted by a fitted missingness
  propensity should help under MAR and be neutral under the control.
- **Scenarios:**
  - arm P, today's default plus `--decode_mask_weighting propensity` (branch
    `feat/decode-mask-propensity`, dd33b81);
  - E64's 26 variants × 404/505/606, 78 cells, against E64's G on the same folds;
  - run from a worktree, so E64's checkout is untouched.
- **Measures and decision rule:**
  - P − G on H, K and KC per mechanism; (P − G), MAR minus control.
  - **Primary:** supported if P − G on H under MAR is better on 4 or more of 12 and worse on at
    most 1, and under the control worse on at most 1; refuted if better on at most 1 under MAR;
    mixed otherwise.
- **Results:**
  - 78 of 78 cells, none failed.
  - **P − G on H under MAR:** better 0, no difference 6, worse 6.
  - **Under the control:** better 0, no difference 7, worse 5. spambase loses most.
  - **(P − G), MAR minus control:** no difference on 11, worse on spambase_60nan.
  - **The dial:** no difference at any slope.
- **Conclusion:** refuted. Propensity-weighted decode masks cost accuracy under both mechanisms,
  and the mask-mismatch hypothesis finds no support. With E64 having refuted the MAR loss of ground
  itself, there is little left to explain. The flag stays, off by default; no default changes.
- **Sources:** `docs/tickets/imputation-protocol/20-mar-propensity.md` (Outcome);
  `src/training/mask_propensity.py`; `scripts/experiments/mar_propensity_run.py`,
  `mar_propensity_report.py`, `run_mar_propensity_queue.sh`.

### E66 · Does E60's complementarity hold on other tables and seeds? (2026-10-23)

- **Status:** done 2026-10-23, 07:58 GMT-3. Pre-registered before the first cell, on the user's go
  (2026-10-23).
- **Hypothesis:** E60's. TRIDENT's and TabPFN's errors are complementary, so an equal-weight blend
  of their numerical fills beats both.
- **Scenarios:** E40's runs at 404/505/606: kr-vs-kp, spambase and credit-g at 40/60/80nan, and
  vehicle, biodeg and kc2 at 60nan. That is 36 pairs; TabPFN is refitted per fold. Offline, two
  GPU lanes.
- **Measures and decision rule:**
  - Blend − K and blend − TabPFN.
  - Over the 9 variants with numerical cells: complementary if the blend beats both on 3 or more;
    not complementary on 1 or none; mixed otherwise.
  - kr-vs-kp is printed, not counted.
- **Results:**
  - 36 of 36 pairs; the ledgers equal K on every fold.
  - **The blend beats both on 3 of 9:** spambase_80nan, vehicle_60nan and biodeg_60nan.
  - **Blend − K:** better 6, no difference 1, equivalent 5, worse 0. The equivalents are spambase
    at 40 and 60nan, where K leads, and kr-vs-kp by construction.
  - **Blend − TabPFN:** better 6, worse 6. The worse are credit-g's three, where TabPFN leads on
    categorical cells the blend does not touch, and kr-vs-kp's three.
- **Conclusion:** complementary, at the floor. The effect is real but narrower than E60's 10 of 15:
  it shows on numerical tables where the two methods are close, and an equal-weight blend cannot
  move categorical cells. Descriptive; no default changes.
- **Sources:** `docs/tickets/imputation-protocol/21-tabpfn-blend-replication.md` (Outcome);
  `scripts/experiments/tabpfn_blend.py` (`--set e40`).

### E67 · The gaps as `[NULL]` in the decode stage, alone: does the token help under MNAR? (2026-10-23)

- **Status:** done, refuted (2026-10-23, 21:53 GMT-3). Pre-registered before the first cell, on the user's go (2026-10-23). T33.
- **Hypothesis:** the `[NULL]`/`[MASK]` split helps more under value-dependent missingness. E45/E46's
  Mx hinted at it on biodeg's steep tilt but changed three things at once.
- **Scenarios:**
  - arm N, today's default with `DECODE_GAP_TOKEN null` only;
  - 8 MNAR pairs: biodeg's dial (slopes 0.25, 0.5, 1.0, 2.0) against biodeg_60nan, and E46's four
    pairs;
  - seeds 42/7/13, 39 cells, against E45/E46's G on the same folds.
- **Measures and decision rule:**
  - N − G on H and K per mechanism, and the interaction (N − G), MNAR minus MCAR.
  - **Primary,** the interaction on H over 8 pairs: supported if better on 3 or more and worse on
    none; refuted if better on at most 1; mixed otherwise.
- **Sources:** `docs/tickets/imputation-protocol/22-null-gaps.md`;
  `scripts/experiments/null_gaps_run.py`, `null_gaps_report.py`, `run_null_gaps_queue.sh`.
- **Results:** 39 of 39 cells, 17:55–20:01 GMT-3, none failed.
  - The interaction (N − G on H), MNAR minus MCAR: better 1 / no difference 5 / worse 2. Better
    only at biodeg's slope 0.25; worse at slopes 1.0 and 2.0. On K: better 1 / no difference 4 /
    worse 3.
  - N − G on H is worse on 7 of 8 MNAR variants and 4 of 5 MCAR controls (kc2_20nan: no
    difference).
  - The dial, N − G on H at biodeg_60nan: +0.030 / +0.066 / +0.076 / +0.099 at slopes 0.25 / 0.5 /
    1.0 / 2.0, against +0.058 under MCAR.
- **Conclusion:** refuted. `[NULL]` gaps alone cost more as the tilt steepens, so E45/E46's Mx
  signal came from the loss checkpoint and column-wise scoring. No MNAR-specific gain on the
  imputation side; ADR 0014 stands; the split's test moves to classification (wiki t2).

### E68 · The tuned bar on the MAR folds: D4 under MAR (2026-10-23)

- **Status:** done, mixed (2026-10-23, 23:54 GMT-3). Pre-registered before the first pair, on the user's go (2026-10-23). T34.
- **Hypothesis:** none directional. ADR 0022 decision 5 asks every claim against the bar to face the
  tuned bar too, and the MAR studies E63 and E64 did not.
- **Scenarios:** E42's driver (knn, hgb and lgbm tuned on validation gaps) on E64's 24 variants at
  404/505/606 and E63's 8 at 42/7/13. That is 96 pairs, offline, on four CPU lanes.
- **Measures and decision rule:**
  - H and K against the bar and the tuned bar per mechanism, and D4.
  - **Primary,** over E64's 12 table-rates, (H − tuned bar) MAR minus control: supported if worse on
    4 or more and better on at most 1; refuted if worse on at most 1; mixed otherwise.
- **Sources:** `docs/tickets/imputation-protocol/23-mar-tuned-bar.md`;
  `scripts/experiments/baseline_budget_run.py`, `mar_tuned_bar_report.py`.
- **Results:** 96 of 96 pairs, 17:55–23:53 GMT-3, none failed.
  - Primary, (H − tuned bar) MAR minus control: better 2 / no difference 7 / worse 3 (worse:
    biodeg_60nan, kc2_60nan, vehicle_60nan; better: kc2_20nan, spambase_60nan). On K: 3 / 7 / 2.
  - H − tuned bar under MAR: better on 6 of 12, worse on none (control: 5 better, none worse).
  - D4 on H: 6 of 12 under MAR, 5 under the control; on K 8 and 4. E63's smoke: better 1 / none 3.
- **Conclusion:** mixed, for the user. MAR costs the model ground only at 60nan on the numerical
  tables, against the plain and the tuned bar alike, and it never puts the model behind the tuned
  bar on E64's family.

### E69 · Per-column weights for the K + TabPFN blend, cross-fitted (2026-10-23 – 2026-10-24)

- **Status:** done, mixed (2026-10-24, 00:37 GMT-3). Pre-registered before the first cell, on the user's go ("vamos com t37",
  2026-10-23, about 22:50 GMT-3). T37.
- **Hypothesis:** a weight per column, chosen on the other folds' cells, lets the blend beat TabPFN
  where E60/E66's equal blend only matched it or lost (TabPFN's 80nan ties; kr-vs-kp and credit-g's
  categorical cells).
- **Scenarios:** E60's fills (K from the ledgers, TabPFN v2 refitted) on E44's 15 confirmation
  variants at 707/808/909 and E40's kr-vs-kp and credit-g at 40/60/80nan at 404/505/606: 63 pairs.
  Numerical weights from {0, 0.25, 0.5, 0.75, 1}, categorical a pick of either category; fold k's
  weights from the other four folds' induced cells; ties to the equal blend.
- **Measures and decision rule:**
  - weighted − equal, weighted − TabPFN and weighted − model per variant, with the 1% SESOI;
  - **primary,** weighted − TabPFN on the 10 hard variants: supported if better on 4 or more;
    refuted if better on at most 1; mixed otherwise. Descriptive, no default changes.
- **Sources:** `docs/tickets/imputation-protocol/24-blend-weights.md`;
  `scripts/experiments/blend_weights.py`, `run_blend_weights_queue.sh`.
- **Results:** 63 of 63 pairs, 22:58–00:36 GMT-3, none failed; fills reproduce E60/E66 within 2.2e-16.
  - Primary, weighted − TabPFN on the 10 hard variants: better on electricity_80nan, vehicle_80nan
    and biodeg_80nan; worse on kr-vs-kp_60nan (+0.0016); equivalent or no difference on the rest.
  - Weighted − equal: better 14 / none 3 / equivalent 3 / worse 1. Weighted − model: better 21 of 21.
    Beats both: 13 of 21 (equal blend 10).
  - Weights: a column's modal choice holds on 57–100% of folds; TabPFN's weight rises with
    missingness except on kc2.
- **Conclusion:** mixed, for the user. Per-column weights win where the two are close on numerical
  cells and reach TabPFN's level on categorical ones, without passing it.

### E70 · Classification step zero, Stage 0: does HEAD on gorgona8 reproduce the old sweep's folds? (2026-10-23)

- **Status:** done, does not reproduce (2026-10-23, 23:49 GMT-3). Pre-registered before the first run, on the user's go ("sim aprovo",
  2026-10-23, 23:44 GMT-3). T27, classification step zero.
- **Hypothesis:** none directional; a gate for Stage 2's pairing with the old sweep (arm A).
- **Scenarios:** reruns of the sweep's vehicle_40nan and kc2_40nan runs at HEAD on gorgona8's GPU
  (their hyperparameters as an override, `cosine_legacy`, 3 folds, seed 42, MLflow off), and the
  classification regression test.
- **Measures and decision rule:** test accuracy, micro- and macro-F1 fold by fold. Reproduces if
  every difference is within 1e-6, and Stage 2 reuses the sweep as arm A; otherwise Stage 2 reruns
  arm A on gorgona8, and the difference is reported against the 0.0002–0.009 duplicate spread.
- **Sources:** `docs/tickets/classification-step-zero/01-determinism.md`;
  `scripts/experiments/cls_determinism_run.py`.
- **Results:** largest fold difference 0.028 (vehicle_40nan) and 0.086 (kc2_40nan); macro-F1 0.026
  and 0.076. Means move by −0.006 and +0.012, folds in both directions. The vehicle regression test
  passes.
- **Conclusion:** does not reproduce. Stage 2 reruns arm A on gorgona8 beside arm B; the old sweep is
  a descriptive reference only. Stage 2 checks gorgona8's run-to-run determinism first.

### E71 · Classification step zero, Stage 2: what does pre-training buy classification under the fork's defaults? (2026-10-23 – 2026-10-24)

- **Status:** done, no gain detected (2026-10-24, 09:36 GMT-3). Pre-registered before the first cell, on the user's go ("sim aprovo",
  2026-10-23, 23:44 GMT-3). T27, classification step zero.
- **Hypothesis:** none directional. The paper's Table 4 reports about +1.7 macro-F1 from
  pre-training; the fork's default sweep already sits within 0.017 of the no-pre-training arm.
- **Scenarios:** the old sweep's configuration as an override, `cosine_legacy`, 3 folds, on all 45
  variants. A as is; B with `EPOCHS_PRE 0`; C with `EPOCHS_PRE 0` and `EPOCH_FINE 450`. Seeds 42,
  then 7, then 13, on four GPU queues, with a clock stop at 09:30 GMT-3 on 2026-10-24; only complete
  seeds are read. A rerun on gorgona8, since E70 found the old sweep does not reproduce.
- **Measures and decision rule:** per level, B − A and C − A on test macro-F1, fold-paired, pooled
  over 9 tables, with the verdict rule; descriptive (folds within a table are not independent).
- **Sources:** `docs/tickets/classification-step-zero/02-no-pretraining.md`;
  `scripts/experiments/cls_stage2_run.py`, `run_cls_stage2_queue.sh`.
- **Amendment (2026-10-24, 00:38 GMT-3):** the sweep-run lookup matched E71's own runs; seven cells
  failed before training, the lookup was fixed, and the seven were rerun (ticket 02).
- **Results:** seeds 42 and 7 complete, seed 13 at 89 of 135 (not read); stop at 09:34 GMT-3.
  - B − A: +0.003 / +0.003 / +0.004 / −0.005 / −0.005 at 0/20/40/60/80nan, no verdict at any.
  - C − A: +0.003 / −0.004 / +0.001 / −0.004 / −0.004, no verdict at any.
  - A's means within 0.011 of the old sweep's at every level.
- **Conclusion:** pre-training gives no detectable gain on classification under the fork's
  defaults, as E24 found for imputation; a 1.7-point gain is outside the intervals at 0–60nan.

### E72 · An imputer that models the mechanism: does not-MIWAE shrink the MNAR bias? (2026-10-24)

- **Status:** done, mixed (2026-10-24, 05:33 GMT-3). Pre-registered before the first cell, on the user's go ("sim aprovo",
  2026-10-23, 23:44 GMT-3). T35.
- **Hypothesis:** modelling the missingness mechanism shrinks the bias MNAR adds (the wiki's H1d).
- **Scenarios:** clean-room MIWAE and not-MIWAE (self-masking logistic, sign learnt), identical but
  for the mechanism term; K 20, batch 16, Adam 1e-3, 100,000 steps; refitted per training fold on
  E45/E46's 17 variants at 42/7/13 and asked for the ledgers' induced cells (51 pairs).
- **Measures and decision rule:**
  - notmiwae − miwae on `abs_bias`, `signed_bias` and `impute_score`, and notmiwae − model (K);
  - **primary,** (notmiwae − miwae on `abs_bias`), MNAR minus MCAR, over 10 pairs: supported if
    better on 4 or more and worse on at most 1; refuted if better on at most 1; mixed otherwise.
- **Sources:** `docs/tickets/imputation-protocol/25-not-miwae.md`; `scripts/experiments/not_miwae.py`,
  `metric_panel.py --fitters vae`, `vae_mnar_report.py`, `run_vae_mnar_queue.sh`.
- **Results:** 51 of 51 pairs, 00:01–05:30 GMT-3, none failed.
  - Primary interaction: better on vehicle_probit_60nan, vehicle_probit_20nan and
    biodeg_probit2p0_60nan; worse on none; no difference on seven.
  - not-MIWAE's `impute_score`: 1.7–4.5 on biodeg and 8.3–9.4 on spambase (worse than the mean
    fill, fill dispersion ≥ the truth's), 0.82–0.91 on kc2, 0.61 on vehicle (the model 0.62).
  - notmiwae − model on `impute_score` under MNAR: worse 9 / no difference 1.
- **Conclusion:** mixed, for the user. H1d holds on vehicle, where the implementation imputes
  competently; on biodeg and spambase the fill fails, so those pairs do not test it.

### E73 · not-MIWAE with a mean-based fill on biodeg and spambase (2026-10-24)

- **Status:** done, supported by the count with two of three pairs flagged (2026-10-24, 13:53 GMT-3). Pre-registered before the first cell, on the user's go ("sim, mas lance o
  2 imediatamente em seguida", 2026-10-24, 11:56 GMT-3). T35.
- **Hypothesis:** with a fill that works, modelling the mechanism shrinks the MNAR bias on biodeg
  and spambase too (H1d). A pilot outside the study found E72's draw-averaged fill produced extreme
  values; the decoder's weighted means did not.
- **Scenarios:** E72's not-MIWAE fit, read as `notmiwae_mean` (primary) and `notmiwae_draws` (10,000
  draws); MIWAE from E72; biodeg's five MNAR pairs and spambase's, 9 variants at 42/7/13, 27 pairs.
- **Measures and decision rule:** (notmiwae_mean − miwae on `abs_bias`), MNAR minus MCAR, over 6
  pairs: supported if better on 3 or more and worse on at most 1; refuted if better on at most 1;
  mixed otherwise. Pairs where notmiwae_mean is not below the mean fill are flagged.
- **Sources:** `docs/tickets/imputation-protocol/26-notmiwae-fill.md`; `scripts/experiments/not_miwae.py`,
  `metric_panel.py --fitters notmiwae_fills`, `notmiwae_fill_report.py`, `run_notmiwae_fill_queue.sh`.
- **Results:** 27 of 27 pairs, 12:04–13:52 GMT-3, none failed.
  - Interaction better on biodeg_probit_60nan, biodeg_probit1p0_60nan and biodeg_probit2p0_60nan,
    worse on none; flagged (notmiwae_mean not below 1.0): slopes 1.0 and 2.0, and spambase.
  - notmiwae_mean's `impute_score` 0.84–1.03, level with MIWAE; E72's fill 1.7–9.4; 10,000 draws
    1.5–8.3. The model 0.59–0.95.
  - On biodeg's dial the interaction grows with the tilt: none, −0.036, −0.095, −0.160.
- **Conclusion:** the mean fill fixes E72's failure, and the mechanism term shrinks the bias more
  as the tilt steepens, but on these tables both VAEs impute near the mean fill. H1d's support is
  where the VAE is competent: vehicle (E72) and biodeg_probit_60nan.

### E74 · The per-column KC + TabPFN hybrid as an output, on fresh seeds (2026-10-24)

- **Status:** done, mixed (2026-10-24, 17:15 GMT-3). Pre-registered before the first cell, on the user's go ("2 gosto da
  ideia", "sim, mas lance o 2 imediatamente em seguida", 2026-10-24, 11:49 and 11:56 GMT-3). T37.
- **Hypothesis:** E69's per-column hybrid improves today's default output (KC) and is never
  meaningfully behind TabPFN.
- **Scenarios:** today's default trained at 1212/2323/3434 on E69's 10 hard variants (30 cells, KC
  headline by ADR 0027), then E69's cross-fitted blend against a refitted TabPFN v2 (30 pairs).
- **Measures and decision rule:** hybrid − KC and hybrid − TabPFN per variant, with the 1% SESOI.
  Supported if better than KC on 8 or more and worse than TabPFN on at most 1; refuted if better
  than KC on at most 5 or worse than TabPFN on 3 or more; mixed otherwise.
- **Sources:** `docs/tickets/imputation-protocol/27-hybrid-output.md`;
  `scripts/experiments/hybrid_blend.py`, `blend_weights.py`, `run_hybrid_queue.sh`.
- **Results:** 30 + 30 of 30, none failed; ledgers equal the runs' KC.
  - Hybrid − KC: better 7 / no difference 3 / worse 0. Hybrid − TabPFN: better 3 / no difference 1
    / equivalent 4 / worse 2 (kr-vs-kp 40 and 60nan). TabPFN − KC: better 8 / no difference 2.
  - Beats both on electricity_80nan, vehicle_80nan and biodeg_80nan, E69's same three.
- **Conclusion:** mixed, for the user. The hybrid is never worse than today's default and best of
  the three on numerical 80nan; elsewhere its gain over KC is TabPFN's own.

### E75 · A test-time re-masking ensemble: averaged passes and their spread (2026-10-24)

- **Status:** done, mixed (2026-10-25, 02:38 GMT-3). Pre-registered before the first cell, on the user's go ("Aprovo T36 e
  T37", 2026-10-23; "autorizo o 1, vá", 2026-10-24, about 18:15 GMT-3, the `src/` authorization). T36.
- **Hypothesis:** averaging 20 passes that each also hide a fresh draw of the row's observed cells
  improves the induced fill under K and H, and the passes' spread ranks the per-cell errors.
- **Scenarios:** today's default with `--score_checkpoints --score_remask` on E44's 15 confirmation
  variants at 707/808/909 (45 cells); passes at K and H, uncalibrated, and calibrated against KC.
- **Measures and decision rule:** remask − single pass on induced `impute_score`, 1% SESOI. Supported
  if better on 8 or more and worse on at most 1 under both K and H; refuted if better on at most 3 or
  worse on 5 or more under K; mixed otherwise. Descriptive: Spearman of spread against error,
  90% coverage, and the panel's dispersion.
- **Sources:** `docs/tickets/imputation-protocol/28-remask-ensemble.md`;
  `scripts/experiments/remask_ensemble.py`, `run_remask_queue.sh`; `src/training/decoding.py`
  (`_remask_induced`).
- **Results:** 45 of 45, none failed; K equals E44's runs to the last digit.
  - Accuracy, remask − single: better 4 / no difference 5 / equivalent 2 / worse 4 (K); 6 / 3 / 2 / 4
    (H); calibrated against KC, 1 / 2 / 4 / 8. Worse on pendigits and letter at 40/60nan, better at
    80nan on numerical tables.
  - Dispersion lower on 15 of 15. `num_rank` signals on 12 of 15 (K and H), none at pendigits,
    letter and vehicle 80nan; `coverage90` 0.17–0.66. Categorical: the single pass's confidence ranks
    misses better than the vote share on 3 of 3.
- **Conclusion:** mixed on accuracy, and the gains at 80nan are shrinkage that calibration already
  takes, so averaging is not proposed. The numerical spread meets the pre-stated bar for an opt-in
  uncertainty output, a ranking signal and not an interval; the user decides on an ADR.

### E76 · Does the re-masking spread rank cells on the development variants? (2026-10-25)

- **Status:** done, supported (2026-10-26, 22:15 GMT-3). Pre-registered before the first cell, on the user's go (2026-10-25, about
  09:45 GMT-3). Machine down mid-run; four cells rerun (amendment). T36; tests ADR 0030's (proposed) scope.
- **Hypothesis:** the numerical spread ranks the errors within a column up to 60nan (E75's post-hoc
  reading), on other tables and seeds.
- **Scenarios:** today's default with `--score_checkpoints --score_remask` on the 21 development
  variants at 101/202/303 (63 cells).
- **Measures and decision rule:** within-column Spearman of spread against absolute error, a signal
  where its 95% interval over 15 folds lies above 0, on the 15 variants up to 60nan with numerical
  gaps. Supported if a signal on 13 or more under K and H and none against under K; refuted if 10 or
  fewer under K or any against; mixed otherwise. E75's other measures descriptive.
- **Sources:** `docs/tickets/imputation-protocol/29-remask-development.md`;
  `scripts/experiments/remask_ensemble.py --study e76`, `run_remask_queue.sh` (`STUDY=e76`).
- **Results:** 63 of 63. Within-column `num_rank` signals on 14 of 15 up to 60nan (K and H), none
  against; ρ 0.11–0.50; credit-g_60nan +0.005. At 80nan, spambase signals, credit-g does not. Vote
  share signals on 9 of 9 categorical variants, below the single pass's confidence on all 9.
  Remask − single better 6 / worse 5 (K; all losses at 20nan); dispersion lower 17 of 17.
- **Conclusion:** supported. ADR 0030's scope (a numerical rank up to 60nan) replicates on other
  tables and seeds, pre-registered.

### E77 · Does the re-masking spread make intervals narrower at the same coverage? (2026-10-25)

- **Status:** done, refuted (2026-10-25, 09:59 GMT-3). Pre-registered before the reading ran (2026-10-25, on item 4 of the plan
  the user accepted). Offline, on E75's ledgers. T36; ADR 0030 (proposed).
- **Hypothesis:** scaled to a 90% target, the spread gives narrower intervals than a constant width
  at the same coverage.
- **Scenarios:** E75's 15 variants × 3 seeds × 5 folds; per numerical column, a spread interval and a
  constant one, each fitted on the other four folds; centred on the headline fill (K) and on H's
  passes (descriptive).
- **Measures and decision rule:** (spread − constant) / constant mean width, 1% SESOI. Supported if
  narrower on 10 or more under K and wider on at most 1; refuted if narrower on 5 or fewer or wider
  on 4 or more; mixed otherwise. Disclosure: the self-check saw two folds of one E75 cell before this
  entry (ticket).
- **Sources:** `docs/tickets/imputation-protocol/30-remask-conformal.md`;
  `scripts/experiments/remask_ensemble.py --conformal`.
- **Amendment (before the reading):** zero spreads (passes that agree; 10–32% of cells at 80nan)
  floored at the column's smallest positive spread.
- **Results:** narrower on 0 / wider on 15 under K; H wider on 14, no difference on 1; coverage
  0.84–0.90 for both. At 40nan the spread interval is 9–144% wider; the 60–80nan magnitudes reflect
  the floor.
- **Conclusion:** refuted. The spread ranks errors within a column but too weakly to adapt an
  interval; ADR 0030 stays a rank.

### E78 · Which columns does the decoder fill each column from? An occlusion map (2026-10-26)

- **Status:** done, supported (2026-10-27, 03:14 GMT-3). Pre-registered before the first cell, on the user's go ("sim, vamos!",
  2026-10-26, before 22:45 GMT-3). T38. No `src/` change; the model (K) is read in place after each
  fold, with the random states restored. A smoke run outside the study reproduced K to 0.0.
- **Hypothesis:** hiding one column's observed cells as `[MASK]` and measuring the change in each
  gap's fill gives a column-dependency map. The map is faithful to the decoder: hiding a target's
  top sources hurts more than hiding random ones. On a table with known structure, it recovers that
  structure.
- **Scenarios:** today's default with `--score_remask`, 40nan, 30 cells:
  - `synthblock_40nan` (three independent nonlinear blocks of four columns plus three noise
    columns; `datasets/generate_synthblock.py`) at 707/808/909;
  - E44's six tables at 707/808/909;
  - kr-vs-kp, spambase and credit-g at 101/202/303.
- **Measures and decision rule:**
  - **B, primary:** sources are ranked on the validation gaps. On test, D is
    (top-3 hidden error − random-3 hidden error, 20 draws) / unhidden error, averaged over targets.
    A variant is faithful if D's 95% interval over 15 folds lies above the 1% SESOI with every seed
    above 0, and not faithful if it lies below. **Supported** if faithful on 7 or more of 9 and not
    faithful on none; **refuted** if faithful on 4 or fewer; **mixed** otherwise.
  - **C, gate:** the AUROC with which the test map separates in-block sources, averaged over the 12
    block columns. It recovers the blocks if the interval lies above 0.8 with every seed above 0.8.
    It is inconclusive (floor) if K is not below `mean_mode` on synthblock.
  - **Descriptive:** the map's top sources; its seed stability; its agreement with |Spearman|;
    attention against occlusion; D with the top 1; K against the earlier runs.
- **Sources:** `docs/tickets/imputation-interpretability/01-occlusion-map.md`;
  `scripts/experiments/occlusion_map.py`, `run_occlusion_queue.sh`; MLflow tag
  `experiment=occlusion-map-2026-10-26`.
- **Results:**
  - 30 of 30 cells; K reproduced the earlier runs to 0.0 on 27.
  - **B:** faithful on 9 of 9.
    - D (top 3) is +0.21 to +0.41 on pendigits, letter, electricity, vehicle, biodeg and kr-vs-kp.
    - It is +0.07 to +0.08 on spambase and credit-g, and +0.036 [+0.015, +0.058] on kc2.
  - **C:** AUROC 0.979 [0.972, 0.987], K 0.724 against `mean_mode` 1.0. Pairs with Pearson ≈ 0
    are found (a2 ← a4, a1). The same reading from |Spearman| gives 0.847.
  - **Descriptive:** seed stability 0.87–0.997 (kc2 0.80); attention agrees with occlusion
    0.75–0.95 (kc2 0.60).
- **Conclusion:** supported. At 40nan under MCAR, the occlusion map is faithful to K and recovers
  known nonlinear structure. It is reported descriptively, with no default changed. E79 tests it at
  60nan.

### E79 · Does the occlusion map hold at 60nan? (2026-10-27)

- **Status:** done, supported at the bar (2026-10-27, 07:00 GMT-3). Pre-registered before its first cell and before any E78 result was read,
  on the user's go to use the GPU until 07:30 (2026-10-26, before 23:22 GMT-3). T38.
- **Hypothesis:** E78's map stays faithful, and recovers the synthetic blocks, at 60nan, where a
  target has fewer observed sources a row.
- **Scenarios:** E78's ten tables at 60nan, 30 cells:
  - `synthblock_60nan`, pendigits, letter and electricity at 707/808/909;
  - vehicle, biodeg, kc2, kr-vs-kp, spambase and credit-g at 101/202/303.
  - Each queue starts when E78's queue of the same number ends. No cell starts after 2026-10-27,
    07:00 GMT-3; the reading waits for all 30.
- **Measures and decision rule:** E78's, unchanged. B is primary: supported if faithful on 7 or more
  of 9 and not faithful on none; refuted if faithful on 4 or fewer. C, the AUROC above 0.8, is the
  gate.
- **Sources:** `docs/tickets/imputation-interpretability/02-occlusion-map-60nan.md`;
  `occlusion_map.py --study e79`, `run_occlusion_queue.sh` (`STUDY=e79`); MLflow tag
  `experiment=occlusion-map-60nan-2026-10-27`.
- **Results:**
  - 30 of 30 cells by 06:57 GMT-3; K reproduced to 0.0 on 27.
  - **B:** faithful on 7 of 9, with D (top 3) +0.04 to +0.24. kc2 (+0.012) and credit-g (+0.019)
    are inconclusive; none is against.
  - **C:** AUROC 0.914 [0.901, 0.926], against 0.864 from |Spearman|.
  - Every D is smaller than at 40nan: by 29–43% on pendigits, letter, electricity, vehicle and
    biodeg, and by 67–74% on kc2, kr-vs-kp and credit-g.
- **Conclusion:** supported, exactly at the bar. With E78, the map is faithful at 40 and 60nan
  under MCAR. At 60nan it is not shown on kc2 and credit-g, and the effect shrinks with
  missingness. No default changed.

### E80 · Does the spread grow when a cell's top sources are missing? (2026-10-27)

- **Status:** done, mixed (2026-10-27, 08:33 GMT-3). Pre-registered before the reading, on the user's go ("otimo, quero fazer
  E e F agora, vá", 2026-10-27, before 08:29 GMT-3). Offline on E78/E79's cells. T38.
  - Disclosure: E78/E79's maps had been read; no spread measure had been computed.
- **Hypothesis:** a numerical cell's re-masking spread (ADR 0030) is higher when its top 3 sources
  (the validation map's) are missing in its row, beyond the row missing more.
- **Scenarios:** the 16 real variants with numerical gaps at 40nan and 60nan (kr-vs-kp out), 15
  folds each, K.
- **Measures and decision rule:** within-column Spearman(spread, top-3 missing count) minus the same
  for 3 random sources (20 draws), averaged over targets.
  - A variant is specific if the 95% interval lies above 0 with every seed above 0, and against if
    the interval lies below 0.
  - **Supported** if specific on 12 or more of 16 and against on none; **refuted** if specific on
    6 or fewer; **mixed** otherwise.
  - The total-gap and error correlations, and synthblock with the true blocks, are descriptive.
- **Sources:** `docs/tickets/imputation-interpretability/03-spread-and-missing-sources.md`;
  `scripts/experiments/occlusion_spread.py`.
- **Results:**
  - **Specific on 7:** pendigits, letter and vehicle at both rates, and biodeg_40nan.
  - **Against on 6:** kc2 and credit-g at both rates, electricity_60nan and spambase_60nan.
  - **Descriptive:** the error's ρ with the top-3 missing count is positive on all 16 (+0.02 to
    +0.19). The spread's ρ with the row's gaps is negative on all 16 (−0.04 to −0.47).
- **Conclusion:** mixed. The two outputs are separate signals.
  - The spread runs against the error along a row's missingness: the passes agree most where the
    model knows least (post-hoc reading: there are fewer observed cells to re-mask).
  - This is a limit on ADR 0030, for the user to record or not.

### E81 · Does a decoder that can tell a real gap from its own masks read the gap pattern under MNAR? (2026-10-27)

- **Status:** done, mixed (2026-10-27, 12:29 GMT-3). Pre-registered before the first cell, on the same go. T38.
- **Hypothesis:** under the default (`DECODE_GAP_TOKEN mask`) the question cannot be asked: a gap
  and a hidden cell are one token. A decoder trained with `--decode_gap_token null` reads `[NULL]`
  at a correlated column as "that column was large" under self-masking MNAR, and not under MCAR.
- **Scenarios:**
  - biodeg at 60nan, at MCAR and at probit slopes 0.25, 0.5, 1.0 and 2.0;
  - vehicle, kc2 and spambase probit_60nan with their MCAR twins;
  - arms null and mask; seeds 101/202/303; 66 cells.
- **Measures and decision rule:** S, the signed fill shift when a correlated gap's `[NULL]`
  (|ρ| ≥ 0.2) is swapped for `[MASK]`, aligned with sign(ρ), on arm null.
  - **Supported** if S at slope 2.0 lies above +0.01 with every seed above 0 and S at MCAR does not
    lie above +0.01; **refuted** if S at slope 2.0 does not lie above 0; **mixed** otherwise.
  - The dial shape, the pairs and the accuracy of null − mask (K, KC) are descriptive.
- **Sources:** `docs/tickets/imputation-interpretability/04-gap-reliance.md`;
  `scripts/experiments/gap_reliance.py`, `run_gap_reliance_queue.sh`; MLflow tag
  `experiment=gap-reliance-2026-10-27`.
- **Results:**
  - 66 of 66; mask-arm K reproduced E79 to 0.0 on 12.
  - **S at slope 2.0:** +0.0049 [+0.0015, +0.0083], not above the SESOI. At MCAR: −0.0022.
  - **The dial:** −0.002, +0.020, +0.035, +0.070, +0.005: rising to slope 1.0, then collapsing.
  - **The pairs:** above under MNAR on vehicle and kc2, and on neither MCAR twin; spambase below on
    both.
  - **Accuracy:** null worse than mask on 11 of 11, under K and KC.
- **Conclusion:** mixed. The default cannot read the gap pattern by construction. A null decoder
  reads it under moderate self-masking, but not measurably at the strongest tilt (post-hoc
  candidates: a truncated observed range, fewer correlated pairs). Reading it costs accuracy
  everywhere, so ADR 0014 stands under MNAR.

### E82 · Can the hybrid's weights be chosen without the complete table? (2026-10-27)

- **Status:** done, refuted (2026-10-27, 18:42 GMT-3). Pre-registered before the first cell, on the user's go ("vamos no 1
  agora então", 2026-10-27, before 13:30 GMT-3). T39; tests ADR 0031's deployable route.
- **Hypothesis:** per-column weights chosen on observed validation cells hidden on purpose (no
  complete table), with H as the model side, keep the hybrid's gain.
- **Scenarios:** E74's 10 hard variants at 1212/2323/3434, 30 cells.
  - Training: `--decode_checkpoint loss --decode_calibration none`.
  - The validation observed cells are dealt into 5 groups and hidden in turn; TabPFN is refitted
    per fold.
  - Five fills: H, TabPFN, equal, deployable (selection-cell weights) and oracle (other folds'
    induced weights, H side).
- **Measures and decision rule:** `impute_score`, 15 fold pairs, the 1% SESOI.
  - **Supported** if the deployable hybrid is better than H on 8 or more of 10, worse than H on
    none, and worse than TabPFN on 2 or fewer.
  - **Refuted** if it is better than H on 5 or fewer, worse than H on 2 or more, or worse than
    TabPFN on 5 or more.
  - **Mixed** otherwise.
  - The price of deployability (deployable − oracle, weight agreement) is secondary. The
    comparisons with KC and E74's hybrid are descriptive.
- **Sources:** `docs/tickets/imputation-protocol/31-deployable-hybrid.md`;
  `scripts/experiments/deployable_hybrid.py`, `run_deployable_hybrid_queue.sh`; MLflow tag
  `experiment=deployable-hybrid-2026-10-27`.
- **Results:**
  - 30 of 30.
  - **Against H:** better on 10 and worse on none.
  - **Against TabPFN:** worse on 5 (vehicle_80, biodeg_80, kr-vs-kp_60/80, credit-g_60) and better
    on 2 (electricity).
  - **Against the oracle:** worse on 6. The deployable and oracle weights agree on 27–62%.
  - **Against KC (descriptive):** better on 7, worse on none.
  - **Post hoc:** the oracle H hybrid is at TabPFN's level (worse on 2 only), so the loss is the
    selection.
- **Conclusion:** refuted. Weights chosen on the validation split's observed cells beat H
  everywhere and KC on 7, without the complete table. They fall short of TabPFN where a column
  has few selection cells or is categorical. ADR 0031 stays a benchmark output.

### E83 · TabPFN v2's classifier against TRIDENT's classification, on E71's folds (2026-10-27)

- **Status:** done (2026-10-27, 20:05 GMT-3). Pre-registered before the first fit, on the user's go ("vamos para o
  T40", 2026-10-27, before 19:40 GMT-3). T40.
- **Hypothesis:** none directional; a first strong-baseline reading for classification.
- **Scenarios:** TabPFN v2's classifier, untuned (E55's settings), on E71's 3-fold `build_folds`
  at seeds 42 and 7. That is 8 tables × 5 MCAR levels, 80 pairs.
  - Context: the training split, capped at 10,000 rows.
  - letter is out, since v2 allows 10 classes.
- **Measures and decision rule:** per level, TabPFN − TRIDENT (E71 arm A) test macro-F1, fold-paired
  and pooled over tables (48 pairs), with the verdict rule and a SESOI of 0.01. It is descriptive.
  It informs T40's choice of step zero's bar or a separate tier.
- **Sources:** `docs/tickets/classification-step-zero/03-tabpfn-classifier.md`;
  `scripts/experiments/tabpfn_classification.py`.
- **Results:**
  - 80 of 80.
  - **TabPFN − TRIDENT macro-F1:** +0.021, +0.019, +0.017, +0.024 and +0.024 at 0–80nan, "TabPFN
    ahead" at every level with both seeds agreeing.
  - **TRIDENT leads** on credit-g from 20nan and electricity from 60nan. Post hoc, there TabPFN
    predicts the majority class (credit-g accuracy 0.70 at 60–80nan), and TRIDENT's class-weighted
    loss trades accuracy for macro-F1.
- **Conclusion:** untuned TabPFN v2 is the stronger classifier on 8 of 9 tables. It qualifies as
  T40's strong baseline; where it joins is the user's call. A fair macro-F1 comparison should
  class-balance TabPFN too (step zero's design).

### E84 · Does the re-masking spread still rank errors under MNAR and MAR? (2026-10-27)

- **Status:** done, supported on the full pre-registered scope (2026-10-28, 01:42 GMT-3); it was partial from 2026-10-27, 22:56 GMT-3, until its remaining cells ran. The amended 8 variants read supported at 22:34 GMT-3; 4 of the 12 pre-registered variants did not run. Pre-registered before the first cell, on the user's go (2026-10-27,
  before 20:40 GMT-3). ADR 0030's follow-up.
- **Hypothesis:** the spread's within-column ranking of errors (E76) holds under self-masking MNAR
  and MAR.
- **Scenarios:**
  - **12 shifted variants at 60nan:** biodeg probit at slopes 0.25/0.5/1/2; vehicle, kc2 and
    spambase probit; `marf` for biodeg, kc2, vehicle, spambase and credit-g.
  - **Controls:** E79's MCAR twins, and the `marfctl` controls run here.
  - Seeds 101/202/303; 51 new cells with `--score_checkpoints --score_remask`.
- **Measures and decision rule:** within-column `num_rank`, a signal where the 95% interval over 15
  folds lies above 0 with seed agreement.
  - **Supported** with a signal on 10 or more of 12 under K and H and none against under K.
  - **Refuted** with 7 or fewer under K or any against.
  - **Mixed** otherwise.
  - Shifted minus control is descriptive.
- **Sources:** `docs/tickets/imputation-protocol/32-spread-mechanisms.md`;
  `scripts/experiments/spread_mechanisms.py`, `run_spread_mechanisms_queue.sh`; MLflow tag
  `experiment=spread-mechanisms-2026-10-27`.
- **Amendment (2026-10-27, 20:44 GMT-3, before any reading):** the study was cut so it ends before a
  store migration at about 23:00.
  - **Scope:** 8 shifted variants (MNAR: biodeg at slopes 0.5 and 2.0, vehicle, kc2; MAR: biodeg,
    kc2, vehicle, credit-g), 36 cells.
  - **The rule, scaled:** supported with a signal on 7 or more of 8 under K and H and none against;
    refuted with 4 or fewer or any against.
  - **Running:** four spambase cells already running are let finish, unread. No cell starts after
    22:45 GMT-3.
- **Results:** 36 of 36 in scope.
  - The within-column `num_rank` signals on 8 of 8 under K and H, ρ +0.10 to +0.52; none against.
  - **Against the control:** weaker on biodeg MNAR (−0.05 at slope 0.5, −0.11 at 2.0) and kc2 MAR
    (K); stronger on kc2 MNAR and vehicle MAR.
- **Conclusion at the partial stage (2026-10-27, 22:56 GMT-3, superseded below):** not complete.
  - On the 8 variants that ran, the ranking holds under MAR and self-masking MNAR at 60nan
    (biodeg, kc2, vehicle and credit-g).
  - The pre-registered study had 12 variants. spambase (MNAR and MAR) and biodeg's slopes 0.25 and
    1.0 did not run, so the study carries no verdict on its own scope; 11 cells remain.
  - ADR 0030 is not amended from it.
  - The user's instruction (2026-10-27): a study is not marked complete unless the evidence is
    undeniable.
- **Second amendment (2026-10-28, 00:30 GMT-3):** the full 12-variant scope is restored on the user's go, and the
  original rule will be read. The 11 remaining cells run under torch 2.11.0 (the first 40 ran
  under 2.5.1); no cell starts after 08:30 GMT-3 on 2026-10-28.
- **Results on the full scope:** 51 of 51 cells.
  - A signal on 12 of 12 under K and 12 of 12 under H, with none against; ρ +0.10 to +0.52 and
    the lowest lower bound +0.084.
  - **On biodeg the ranking weakens with the tilt:** shifted minus MCAR is −0.023, −0.049, −0.077
    and −0.111 at slopes 0.25 to 2.0.
  - **Conclusion:** supported. The spread's within-column ranking holds under one MAR draw and
    self-masking MNAR at 60nan on five tables, weakening as the tilt grows. Extending ADR 0030 is
    the user's decision. **Extended 2026-10-31 on the user's decision (ADR 0030, decision 7).**


### E85 · Does the occlusion map hold at 20nan and 80nan? (2026-10-27)

- **Status:** done (2026-10-28, 11:47 GMT-3): 20nan supported, 80nan refuted; all 60 cells ran. It was partial from 09:14 GMT-3, with 20nan at 10 of 30. Pre-registered before the first cell, on the same go. T38.
- **Hypothesis:** E78's map stays faithful, and recovers the synthetic blocks, at 20nan and 80nan.
- **Scenarios:** the ten tables at 20nan (real tables at 101/202/303) and at 80nan (E44's six at
  707/808/909; kr-vs-kp, spambase and credit-g at 101/202/303); synthblock at 707/808/909; 60
  cells. Each queue starts when E84's same-numbered queue ends.
- **Measures and decision rule:** E78's, applied to each rate separately. B is primary:
  **supported** if faithful on 7 or more of 9 and not faithful on none; **refuted** if faithful on
  4 or fewer. C, the AUROC above 0.8, is the gate. Thin support at 80nan reads inconclusive, as
  fixed in the ticket.
- **Sources:** `docs/tickets/imputation-interpretability/05-occlusion-map-grid.md`;
  `occlusion_map.py --study e85`, `run_occlusion_queue.sh` (`STUDY=e85`); MLflow tag
  `experiment=occlusion-map-grid-2026-10-27`.
- **Amendment (2026-10-27, 20:44 GMT-3):** not run tonight. Its chained waiters were stopped before
  any cell started, for the user's store migration. It stays pre-registered and waits for the
  user's go.
- **Second amendment (2026-10-28, 00:30 GMT-3):** running on the user's go, under torch 2.11.0, with each queue's
  80nan cells first and no cell after 08:30 GMT-3 on 2026-10-28. A rate that does not finish stays
  partial, with no verdict.
- **Results:** 40 of 60 cells, torch 2.11.0, the last one ending at 09:11 GMT-3.
  - **80nan (complete): refuted.** Faithful on 4 of 9 (pendigits, letter, electricity, vehicle;
    D +0.03 to +0.08). Not faithful on biodeg, kc2, spambase and credit-g; kr-vs-kp inconclusive.
  - **C at 80nan:** AUROC 0.853, above the bar, but below |Spearman|'s 0.878.
  - **20nan:** 10 of 30 cells, not read.
- **Conclusion:** the map's faithfulness holds up to 60nan (E78, E79). At 80nan it holds only on
  pendigits, letter, electricity and vehicle. The 20nan rate waits for its remaining 20 cells on the user's go.
- **Third amendment (2026-10-28, 10:00 GMT-3):** the 20nan rate's remaining 20 cells run on the
  user's go, under torch 2.11.0, with no deadline. The 20nan rule is read when all 30 have run.
- **Results on all 60 cells:**
  - **20nan: supported.** Faithful on 8 of 9 (kc2 inconclusive), none against; synthetic AUROC
    0.994.
  - **80nan: refuted,** as above.
  - **Faithful of 9 by rate:** 8, 9, 7 and 4 at 20, 40, 60 and 80nan (E85, E78, E79, E85).
- **Conclusion:** the occlusion map is faithful from 20 to 60nan (MCAR, K) and not at 80nan.

### E86 · Do gorgona8 and medusa1 give the same fold scores for the same cell? (2026-10-28)

- **Status:** done
- **Hypothesis:** the machine (RTX 3090 Ti against RTX 5090, same commit, torch 2.11.0+cu128 and
  packages) adds no systematic shift to a cell's fold scores. Bit identity is not expected across
  GPU architectures; the question is a zero-mean difference within the SESOI. If it holds, later
  studies may split their cells across the two by seed.
- **Scenarios:** MLflow off on both machines, so no run or mirror reaches the store.
  - Imputation, today's default, five folds: credit-g_20nan, kc2_60nan, vehicle_40nan,
    biodeg_40nan × seeds 101/202/303.
  - Classification, E71's arm A, three folds: vehicle_40nan, kc2_40nan, credit-g_20nan,
    biodeg_60nan × seeds 42/7/13.
  - One duplicate per task on each machine (credit-g_20nan s101, vehicle_40nan s42).
  - 26 cells a machine, two queues each, about 1 h.
- **Measures and decision rule:** per fold, medusa − gorgona8.
  - Imputation: relative, on the default output's induced `impute_score`, 60 pairs, SESOI 1%.
  - Classification: macro-F1, 36 pairs, SESOI 0.01.
  - The project's rule. *Equivalent* allows splitting by seed (both arms of a pair on one machine).
    *Better or worse* keeps each study on one machine. *None* is broken by the largest per-fold
    |difference| against the seed spread (E12's 0.007–0.014; E70's 0.0002–0.009).
  - Descriptive: the duplicates, the masked score, and the knn5 and hgb baselines (hgb's last
    digits differ between machines by design).
- **Results (2026-10-28, 15:55 GMT-3):** no verdict on either task.
  - Imputation: medusa − gorgona8 +0.59% [−0.61%, +1.78%] over 60 fold pairs.
    Classification: +0.0039 [−0.0066, +0.0145] over 36.
  - The largest per-fold |difference| is 0.164 (imputation) and 0.074 (classification), far
    outside the seed spread.
  - Both machines are bit-identical run to run, and the CPU baselines agree (hgb 0, knn5
    ≤0.0007).
- **Conclusion:** by the pre-registered tie-break, each study runs whole on one machine. The
  machine acts like another random draw of the same seed, so two studies can run in parallel,
  one per machine, but one study is not split across them. Splitting by seed would need a larger
  check that shows equivalence; the user decides.
- **Sources:** [cross-machine/01](tickets/cross-machine/01-gorgona8-medusa1.md);
  `scripts/experiments/cross_machine.py`, `run_cross_machine_queue.sh`; state in
  `/scratch2/diogoneiss/trident-2026-10-28-cross-machine/` and medusa1's
  `/scratch/diogoneiss/trident-2026-10-28-cross-machine/`.

### E87 · Does pre-training help classification with ten folds and five seeds? (2026-10-28 – 2026-10-29)

- **Status:** done
- **Hypothesis:** the paper's Table 4 pre-training gain (about +1.7 macro-F1 points) shows once the
  power is there. E71 found none with three folds and two seeds. This is T41's old-sweep half.
- **Scenarios:** E71's arms A (the old sweep's hyperparameters, `cosine_legacy`) and B
  (`EPOCHS_PRE 0`).
  - 9 tables × 5 levels × seeds 42/7/13/101/202, `--cv_folds 10`: 450 cells.
  - All on medusa1, eight queues; about 16–24 h.
- **Measures and decision rule:** per level, B − A on test macro-F1, pooled over the tables with
  one array per seed (450 fold pairs), the project's rule with SESOI 0.01.
  - **Supported** if A is higher at 3 or more of the 5 levels and B at none.
  - **Refuted** if A is higher at 1 level or none.
  - **Mixed** otherwise.
  - Read only with every cell done.
- **Results (2026-10-29, 03:37 GMT-3):** equivalent at every level.
  - B − A runs from −0.0035 [−0.0070, +0.0000] (60nan) to +0.0022 [−0.0006, +0.0049] (20nan), 450
    fold pairs a level. Every interval lies inside ±0.01.
  - A is higher at no level and B at no level: refuted.
- **Conclusion:** no pre-training gain on classification under the old sweep configuration. A
  gain of the paper's size (about +1.7 points) is excluded at every rate. T41's paper-parameters
  half is the open route.
- **Sources:** [classification-step-zero/04](tickets/classification-step-zero/04-tenfold-pretraining.md);
  `scripts/experiments/cls_tenfold.py`, `run_cls_tenfold_queue.sh`; MLflow
  `experiment=cls-tenfold-2026-10-28`.

### E88 · Does the occlusion map hold under MNAR and MAR? (2026-10-28)

- **Status:** done
- **Hypothesis:** the map, faithful from 20 to 60nan under MCAR (E78, E79, E85), stays faithful on
  E84's 12 MNAR and MAR variants at 60nan.
- **Scenarios:** 7 MNAR (biodeg at four slopes; vehicle, kc2 and spambase) and 5 MAR (`marf`)
  variants × seeds 101/202/303: 36 cells.
  - Today's default, read in place as E78.
  - No synthetic table, since C is MCAR-only.
  - All on gorgona8, four queues; about 3–4 h.
- **Measures and decision rule:** B, D on top 3, per variant over 15 folds (SESOI 0.01).
  - **Supported** if faithful on 10 or more of 12 and not faithful on none.
  - **Refuted** if faithful on 5 or fewer.
  - **Mixed** otherwise.
  - Descriptive: D against the MCAR twins (E79).
- **Results (19:48 GMT-3):** faithful on 11 of 12, not faithful on none (kc2_probit inconclusive)
  → supported.
  - Against the MCAR twins: self-masking weakens D on biodeg as the slope grows (−0.029 to −0.094),
    and MAR strengthens it on biodeg, vehicle and kc2 (+0.086 to +0.194).
- **Conclusion:** the occlusion map is faithful at 60nan under MNAR and MAR as well as MCAR. Its
  claim is "20–60nan MCAR, and 60nan MNAR/MAR, at K"; the 80nan limit stands.
- **Sources:** [imputation-interpretability/06](tickets/imputation-interpretability/06-occlusion-map-mechanisms.md);
  `occlusion_map.py --study e88`; MLflow `experiment=occlusion-mechanisms-2026-10-28`.

### E89 · Can fewer or better-fed choices close the deployable hybrid's gap to TabPFN? (2026-10-28)

- **Status:** done
- **Hypothesis:** E82's deployable hybrid lost to TabPFN on 5 of 10 because its per-column choices
  were noisy. Choosing less, or on more cells, closes that gap. This is a development read: the
  rules were written after E82, and E89 scores them on E82's seeds and H runs.
- **Scenarios:** E82's 30 cells, with TabPFN refit and no training. Four rules, none reading the
  complete table:
  - `column` (E82's own; the reproduction check);
  - `table` (one weight and one categorical pick per table);
  - `shrunk` (the primary: toward the equal blend with N0 = 50);
  - `pooled` (per column, on all five folds' validation cells).
- **Measures and decision rule:** E82's rule per rule: supported if better than H on 8 or more of
  10, worse on none, and worse than TabPFN on 2 or fewer; refuted if better than H on 5 or fewer,
  worse on 2 or more, or worse than TabPFN on 5 or more. If supported, the next step is a
  fresh-seed confirmation in E82's shape, with the user's go.
- **Results (22:44 GMT-3):** supported (development).
  - `column` reproduces E82: worse than TabPFN on 5, refuted.
  - **`shrunk` (primary):** better than H on 10 of 10, and worse than TabPFN on 2 (kr-vs-kp at 60
    and 80nan).
  - **`table`:** better than H on 10, and worse than TabPFN on none.
  - **`pooled`:** mixed, worse than TabPFN on 3.
- **Conclusion:** fewer choices close most of the deployable hybrid's gap to TabPFN. Shrinking
  closes the thin tables; one pick per table also closes kr-vs-kp. On E82's own seeds this is a
  development result, so ADR 0031 is unchanged. A fresh-seed confirmation with one pre-registered
  rule (`table` or `shrunk`) waits on the user.
- **Sources:** [imputation-protocol/33](tickets/imputation-protocol/33-hybrid-selection-rules.md);
  `scripts/experiments/hybrid_selection_rules.py`, `run_hybrid_selection_rules_queue.sh`.

### E90 · Does the deployable hybrid's `table` rule hold on fresh seeds? (2026-10-28 – 2026-10-29)

- **Status:** done
- **Hypothesis:** E89's `table` rule (one weight and one categorical pick per table, chosen on
  hidden observed validation cells) keeps the deployable hybrid at TabPFN's level on fresh seeds.
- **Scenarios:** E82's exact shape (H with the selection read, then TabPFN refit) on E82's 10 hard
  variants × seeds 4545/5656/6767: 30 cells, gorgona8.
- **Measures and decision rule:** `table` − H and `table` − TabPFN over 15 fold pairs (SESOI 1% of
  the opponent's mean).
  - **Supported** if better than H on 8 or more, worse on none, and worse than TabPFN on 2 or
    fewer.
  - **Refuted** if better than H on 5 or fewer, worse on 2 or more, or worse than TabPFN on 5 or
    more.
  - **Mixed** otherwise.
  - The other three rules are secondary. No cell starts after 07:00 (training) or 08:40 (refits)
    GMT-3 on 10-29, and a study cut there is partial.
- **Amendment 2026-10-29, 06:45 GMT-3 (E90 and E91, before any result was read):** the training
  deadline moves from 07:00 to 08:00. The queues ran slower than planned, and small E91 cells would
  have missed 07:00. A rescue runner trains the leftover cells once the four queues stop.
- **Results (2026-10-29, 07:37 GMT-3):** `table`: better than H on 9 of 10 (electricity_40nan no
  verdict), worse on none, and worse than TabPFN on none. Supported.
  - It is better than TabPFN on electricity_80nan, equivalent on 4, and without verdict on 5.
  - Secondary: `column` worse than TabPFN on 7 (refuted), `shrunk` on 4 (mixed), `pooled` on 4
    (mixed).
- **Conclusion:** the deployable hybrid with one weight and one categorical pick per table holds
  on fresh seeds: never worse than H or TabPFN on the 10 hard variants, with no complete table.
  E89's primary (`shrunk`) does not replicate. Adding a deployable output to ADR 0031 is the
  user's decision.
- **Sources:** [imputation-protocol/34](tickets/imputation-protocol/34-hybrid-confirmation.md);
  `scripts/experiments/hybrid_confirmation.py --study e90`, `run_hybrid_confirmation_queue.sh`.

### E91 · Does the deployable hybrid cost anything where it is not needed? (2026-10-28 – 2026-10-29)

- **Status:** done
- **Hypothesis:** on confirmation variants where E60's equal blend already beat TabPFN, the
  `table` hybrid costs nothing against H or TabPFN.
- **Scenarios:** E90's shape on biodeg_40nan, kc2_40nan, kc2_80nan, pendigits at 40/60/80nan and
  vehicle_40nan × seeds 4545/5656/6767: 21 cells, gorgona8, sharing E90's queues.
- **Measures and decision rule:**
  - **Supported** if worse than H on none and worse than TabPFN on 1 or fewer of 7.
  - **Refuted** if worse than H on 2 or more, or worse than TabPFN on 3 or more.
  - **Mixed** otherwise.
- **Results (2026-10-29, 07:37 GMT-3):** `table`: worse than H on none of 7 (better on 5),
  and worse than TabPFN on none (better on 6). Supported.
- **Conclusion:** the hybrid costs nothing where it is not needed. With E90 it is never worse
  than H or TabPFN on 17 variants.
- **Sources:** as E90, `--study e91`.

### E92 · Does TabPFN v2's classification lead hold on ten folds, and does class balancing close credit-g and electricity? (2026-10-28)

- **Status:** done, supported for both arms (2026-10-29, 21:50 GMT-3).
- **Hypothesis:** E83's lead (about +2 macro-F1 points at every level, 3 folds, 2 seeds) holds on
  E87's ten folds and five seeds. A class-balanced TabPFN also wins on credit-g and electricity,
  where E83's TabPFN predicted the majority class. T40.
- **Scenarios:** untuned TabPFN v2 (E83's settings), one fit per fold, predicted twice: plain and
  with `balance_probabilities`. It runs on E87's folds (`build_folds`, 10 folds, seeds
  42/7/13/101/202).
  - 8 tables (letter is out) × 5 levels × 5 seeds: 200 pairs, 2,000 fits.
  - medusa1, two GPU queues that start when E87's queues end.
- **Measures and decision rule:** per level, each arm − E87 arm A on test macro-F1, fold-paired,
  pooled over the 8 tables with one array per seed (400 fold pairs), the project's rule with SESOI 0.01.
  - Per arm: **supported** if TabPFN is ahead at 4 or more of 5 levels and TRIDENT at none;
    **refuted** if TabPFN is ahead at 1 level or none; **mixed** otherwise.
  - Balanced − plain per level, and per-table means: descriptive.
  - Read only with all 200 pairs and E87 done.
- **Amendment 2026-10-29:** GPU0 failed at 04:52 GMT-3 and CUDA is down on both GPUs. 85 pairs
  finished on the GPU and stand; 61 written from the CPU fallback are quarantined, unread (E86); the
  script now refuses to run without CUDA. The 115 missing pairs wait for the GPU; design unchanged.
- **Results:** 200 of 200 pairs, finished 2026-10-29 at 21:41 GMT-3, all on the GPU (115 rerun on
  GPU1 after the reboot). TabPFN − E87 arm A, test macro-F1, 400 fold pairs a level:
  - **plain:** +0.017, +0.010, +0.011, +0.012, +0.011 at 0–80nan, TabPFN ahead at 5 of 5, every seed
    agreeing: **supported**.
  - **balanced:** +0.018, +0.021, +0.020, +0.028, +0.037, ahead at 5 of 5: **supported**.
  - **balanced − plain:** +0.002 (equivalent) at 0nan, then +0.011, +0.009, +0.016, +0.026, ahead at
    every level with gaps.
  - **credit-g and electricity:** balancing removes TRIDENT's edge. credit-g balanced 0.669, 0.635,
    0.592, 0.573 against TRIDENT 0.641, 0.637, 0.569, 0.520 at 20–80nan (plain 0.619 to 0.454).
    electricity balanced 0.736, 0.678, 0.604 against TRIDENT 0.734, 0.678, 0.602 at 40–80nan, a tie.
  - TRIDENT is ahead on no level; per table it leads only kr-vs-kp at 0nan (0.995 against 0.994) and
    pendigits at 80nan (0.534 against 0.530).
- **Conclusion:** E83's lead holds with E87's power, and class balancing widens it as gaps grow:
  about +2 macro-F1 points at 20–40nan, +3.7 at 80nan. TRIDENT's only wins in E83 came from its
  class-weighted loss against an unbalanced TabPFN; balanced, TabPFN matches or beats it on every
  table. ADR 0032's strong tier for classification should use the balanced arm; that is the user's
  call.
- **Sources:** [classification-step-zero/05](tickets/classification-step-zero/05-tabpfn-tenfold.md);
  `tabpfn_classification.py --study e92`, `run_tabpfn_tenfold_queue.sh`; no MLflow runs.

### E93 · Does the deployable `table` hybrid hold under MNAR and MAR? (2026-10-29)

- **Status:** done
- **Hypothesis:** ADR 0031 decision 6's hybrid (H + TabPFN, one weight and one categorical pick per
  table, chosen on hidden observed validation cells) holds beyond MCAR.
- **Scenarios:** E84's 12 variants at 60nan (7 self-masking MNAR, 5 MAR) × seeds 4545/5656/6767,
  so 36 cells. This is E90's pipeline, on gorgona8.
- **Measures and decision rule:** E90's.
  - **Supported** if better than H on 8 or more of 12, worse on none, and worse than TabPFN on 2 or
    fewer.
  - **Refuted** if better than H on 5 or fewer, worse on 2 or more, or worse than TabPFN on 5 or
    more.
  - **Mixed** otherwise.
  - Descriptive: the MNAR and MAR splits.
  - Deadlines: training 18:00, refits 19:00 GMT-3.
- **Amendment 2026-10-29, 14:49 GMT-3 (before any result was read):** four cells failed on the
  leader's MLflow API while medusa1 rebooted (14:22–14:32). They are retrained once with the same
  seeds by a rescue runner, under the same deadline.
- **Results (16:03 GMT-3):** mixed.
  - `table` is better than H on 6 of 12 (the rule needed 8).
  - It is worse than H on spambase_probit (+0.0090) and worse than TabPFN on credit-g_marf
    (+0.0184).
  - It is better than TabPFN on 10 of 12. Under MNAR and MAR, H already leads TabPFN on most
    variants.
- **Conclusion:** decision 6's scope stays MCAR. Off MCAR the hybrid stays within 1% of H and beats
  TabPFN on 10 of 12, but it adds little over H.
- **Sources:** [imputation-protocol/35](tickets/imputation-protocol/35-hybrid-mechanisms.md);
  `hybrid_confirmation.py --study e93`, `run_hybrid_confirmation_queue.sh`.

### E94 · Is the machine's shift inside the SESOI? (2026-10-29)

- **Status:** done
- **Hypothesis:** the gorgona8 − medusa1 shift that E86 could not bound (+0.59%, +0.0039) lies
  inside the SESOI once there are enough fold pairs. If it does, a study's cells may be split
  between the machines by seed.
- **Scenarios:** MLflow off.
  - Imputation (default, five folds): 7 small-table variants × seeds 1–10, so 350 fold pairs.
  - Classification (E71's arm A, three folds): 5 variants × seeds 1–10, so 150 fold pairs.
  - 120 cells on each machine.
- **Measures and decision rule:** per fold, medusa − gorgona8, relative on imputation (SESOI 1%)
  and absolute macro-F1 (SESOI 0.01).
  - **Equivalent:** the split is allowed for that task.
  - **Anything else:** each study stays on one machine.
- **Results (17:53 GMT-3):**
  - **Classification:** +0.0009 [−0.0056, +0.0073], 150 pairs: equivalent.
  - **Imputation:** +0.54% [−0.06%, +1.13%], 350 pairs: no verdict. The upper bound passes 1%, and
    kc2_20nan carries the heavy tail.
- **Conclusion:** classification studies may be split between the servers by seed. Imputation
  studies still run whole on one machine.
- **Sources:** [cross-machine/02](tickets/cross-machine/02-equivalence.md);
  `cross_machine.py --study e94`, `run_cross_machine_e94_queue.sh`.

### E95 · Does the deployable `table` hybrid hold at 20nan? (2026-10-29)

- **Status:** done
- **Hypothesis:** at 20nan, where the model usually leads TabPFN, decision 6's hybrid costs
  nothing.
- **Scenarios:** the nine tables at 20nan × seeds 4545/5656/6767, so 27 cells. This is E90's
  pipeline, all on medusa1 GPU0.
- **Measures and decision rule:** E91's.
  - **Supported** if worse than H on none and worse than TabPFN on 1 or fewer of 9.
  - **Refuted** if worse than H on 2 or more, or worse than TabPFN on 3 or more.
  - **Mixed** otherwise.
  - Deadlines: training 02:00, refits 03:00 GMT-3 on 10-30.
- **Results (19:36 GMT-3):** supported. `table` is worse than H on none of 9 (better on 6) and
  worse than TabPFN on none (better on 6). All four rules are supported.
- **Conclusion:** the deployable hybrid costs nothing at 20nan. Extending decision 6 to 20nan is
  the user's decision. **Extended 2026-10-31 on the user's decision (ADR 0031).**
- **Sources:** [imputation-protocol/36](tickets/imputation-protocol/36-hybrid-20nan.md);
  `hybrid_confirmation.py --study e95`.

### E96 · Is the imputation shift inside the SESOI once kc2 is left out? (2026-10-29)

- **Status:** done
- **Hypothesis:** without kc2, whose folds made E94's imputation tail, the machine shift lies
  inside ±1%. A post hoc read of E94 gives −0.02% [−0.30%, +0.26%] on these five variants; this
  study re-runs them on fresh seeds.
- **Scenarios:** MLflow off. credit-g at 20/60nan, vehicle at 40/80nan and biodeg_40nan × seeds
  11–20, so 50 cells on each machine and 250 fold pairs.
- **Measures and decision rule:** medusa − gorgona8, relative, SESOI 1%. If equivalent, imputation
  studies on these tables may be split by seed; kc2 stays excluded.
- **Results (19:36 GMT-3):** equivalent, +0.04% [−0.30%, +0.38%] over 250 fold pairs.
- **Conclusion:** imputation studies on credit-g, vehicle and biodeg may split between the servers
  by seed. kc2 and the unchecked tables stay on one machine.
- **Sources:** [cross-machine/03](tickets/cross-machine/03-equivalence-without-kc2.md);
  `cross_machine.py --study e96`.

### E97 · Step zero Stage 1: the CPU bar on the sweep's folds (2026-10-29)

- **Status:** done
- **Hypothesis:** descriptive. How far untuned GBDTs with native NaN, and impute-then-learn arms,
  sit from TRIDENT's classification on the same folds, before Stage 4 tunes any of them.
- **Scenarios:** the sweep's 3 folds at seed 42 on all 45 variants. 20 arms a fold:
  - native NaN with HGB and LightGBM;
  - three imputers (mode+LightGBM, mean/mode, missForest), each with and without indicators, each
    followed by HGB, LightGBM and RandomForest.
  - All balanced and untuned, on gorgona8's CPU.
- **Measures:** (arm − TRIDENT) macro-F1 per level, against E71's arm A at seed 42 (the headline
  reference) and the old sweep (descriptive). Descriptive only, with no call.
- **Results (22:14 GMT-3, descriptive):** native/hgb − TRIDENT (E71 A, s42) is +0.007 (00nan), +0.015,
  +0.019, +0.033 and +0.032 (20–80nan), every interval with gaps above zero. The best arm is
  missForest + indicators + HGB at 20–60nan (up to +0.037), and native HGB at 80nan. The
  indicator contrast is a null-to-small MCAR control.
- **Conclusion:** untuned GBDTs lead the fork's TRIDENT classification by 1.5–3.3 points wherever
  there are gaps. Stage 4 tunes native HGB/LightGBM and missForest+ind+HGB as the bar.
- **Sources:** [classification-step-zero/06](tickets/classification-step-zero/06-cpu-bar.md);
  `scripts/experiments/cls_stage1_bar.py`.

### E98 · Step zero proper (H1c): does TRIDENT's classification lead survive a fair rerun? (2026-10-29 – 2026-10-30)

- **Status:** refuted
- **Hypothesis:** H1c. Tuned on validation only, TRIDENT leads the best tuned native-NaN GBDT at
  40, 60 and 80nan.
- **Scenarios:** all 45 variants, ten folds.
  - **TRIDENT:** an ADR 0029 study per variant, 20 trials of the `full` profile (20–60 epochs),
    ranked on mean validation macro-F1 at seed 42, then evaluated at seeds 42/7/13.
  - **The bar:** HGB and LightGBM, tuned the same way (20 trials, spaces in the ticket). Per
    variant, the one with the better validation score is used.
  - **References:** E87's arm A, the untuned fork budget; E92's TabPFN, the strong tier.
  - **Running:** six GPU queues over gorgona8 and medusa1; the cap is 2026-10-31 08:00.
- **Measures and decision rule:** tuned TRIDENT − bar, per level over 270 fold pairs, with the
  project's rule.
  - **Refuted** if the bar is higher at any of 40/60/80nan.
  - **Supported** if TRIDENT is higher at all three.
  - **Unconfirmed** otherwise.
- **Sources:** [classification-step-zero/07](tickets/classification-step-zero/07-step-zero.md);
  `scripts/experiments/stage4_trident.py`, `run_stage4_queue.sh`, `stage4_gbdt.py`, `stage4_tally.py`.
- **Results (2026-10-30 15:59 GMT-3):** all 45 variants, 270 fold pairs a level. Tuned TRIDENT − the bar:
  00nan −0.017 [−0.023, −0.011], 20nan −0.025 [−0.029, −0.020], 40nan −0.024 [−0.029, −0.019],
  60nan −0.032 [−0.039, −0.026], 80nan −0.030 [−0.035, −0.025]; every seed below zero at every level.
  - Against the untuned fork budget (E87 A): no verdict at 20–80nan, −0.004 at 00nan.
  - Against TabPFN: behind at every level (plain −0.010 to −0.020, balanced −0.020 to −0.035).
  - TRIDENT's mean is above the bar only on kc2 00/20nan, credit-g 40nan and kr-vs-kp 60nan.
  - 15 of 1350 folds collapsed.
- **Conclusion:** refuted. With validation-only tuning and the same budget, the GBDT bar leads at
  40, 60 and 80nan, and tuning leaves TRIDENT where the fork budget had it. What H1c becomes is the
  user's decision; no ADR changed.

### E100 · Does a steps-aware training batch help the small tables? (T14) (2026-10-30)

- **Status:** refuted
- **Hypothesis:** at BATCH 256 an epoch takes 2–3 steps on kc2, credit-g, vehicle and biodeg; a
  batch shrunk to give at least 8 steps (`--min_steps_per_epoch 8`) imputes them better.
- **Scenarios:** 16 variants (those four tables at 20–80nan) × seeds 42/7/13 × arms M (default)
  and S8, five folds, MLflow off; seed 42 on gorgona8, 7 and 13 on medusa1.
- **Measures and decision rule:** S8 − M on the induced `impute_score`, relative, 15 fold pairs a
  variant, 1% SESOI. Supported if better on 4 or more and worse on at most 1; refuted if better on
  at most 1; mixed otherwise.
- **Sources:** [imputation-protocol/37](tickets/imputation-protocol/37-steps-rule.md);
  `scripts/experiments/steps_rule.py`, `run_steps_rule_queue.sh`.
- **Results (2026-10-30 18:41 GMT-3):** all 96 cells. S8 − M: better on 0, equivalent on 1 (credit-g_80nan), no
  verdict on 11, worse on 4: credit-g_40nan +1.99%, credit-g_60nan +1.17%, biodeg_20nan +5.32%,
  biodeg_60nan +1.85%. The masked score is higher under S8 on 14 of 16.
- **Conclusion:** refuted. More steps at a smaller batch, with the learning rates unchanged, do not
  help the small tables. T14 closes; BATCH stays one value for every table, and the flag stays off.

### E101 · Does pendigits still gain from batch 1024 with the learning rates x4, and from a longer decode? (T11) (2026-10-30)

- **Status:** supported
- **Hypothesis:** E28's Q4 still helps pendigits on today's defaults, and a 600-epoch decode adds
  to it.
- **Scenarios:** pendigits 20–80nan × seeds 42/7/13 × arms M, P (BATCH 1024, LR x4) and P600 (P with
  EPOCHS_DECODE 600), five folds, MLflow off; seed 42 on gorgona8, 7 and 13 on medusa1.
- **Measures and decision rule:** P600 − M on the induced `impute_score`, relative, 15 fold pairs,
  1% SESOI. Supported if better on 2 or more of 4 and worse on none; refuted if better on none.
- **Sources:** [imputation-protocol/38](tickets/imputation-protocol/38-large-batch.md);
  `scripts/experiments/large_batch.py`, `run_large_batch_queue.sh`.
- **Results (2026-10-30 23:13 GMT-3):** all 36 cells. P600 − M: 20nan −5.37%, 40nan −4.27%, 60nan −1.19% (better),
  80nan +0.16% (equivalent). P − M: better at 20–60nan (−3.39%, −2.53%, −0.88%), worse at 80nan
  (+0.37%).
- **Conclusion:** supported. On today's defaults pendigits still gains from batch 1024 with the
  learning rates x4, and more with a 600-epoch decode. A fresh-seed confirmation comes before any
  ADR; no default changed.

### E102 · Does E101's pendigits gain replicate on fresh seeds? (2026-10-30 – 2026-10-31)

- **Status:** not replicated
- **Hypothesis:** P600 (BATCH 1024, LR x4, EPOCHS_DECODE 600) stays better than the default on
  pendigits on seeds E101 never saw.
- **Scenarios:** pendigits 20–80nan × seeds 7878/8989/9090 × arms M and P600, five folds, MLflow off;
  7878 on gorgona8, 8989 and 9090 on medusa1.
- **Measures and decision rule:** P600 − M on the induced `impute_score`, relative, 15 fold pairs,
  1% SESOI. Replicated if better on 2 or more of 4 and worse on none; not replicated if better on
  none or worse on any; partial otherwise.
- **Sources:** [imputation-protocol/39](tickets/imputation-protocol/39-large-batch-confirmation.md);
  `scripts/experiments/large_batch.py --study e102`.
- **Results (2026-10-31 01:09 GMT-3):** all 24 cells. P600 − M: 20nan −5.84%, 40nan −4.21%, 60nan −1.18% (better), 80nan
  +0.42% (worse, every seed above zero, under the 1% SESOI).
- **Conclusion:** not replicated by the pre-registered rule (worse on one). The 20–60nan gains
  replicate at E101's sizes; at 80nan the large batch costs a little. No ADR proposal follows from
  this rule.

### E103 · Does the large batch and longer decode help letter and electricity? (2026-10-31)

- **Status:** letter mixed; electricity refuted
- **Hypothesis:** E101's P600 (BATCH 1024, LR x4, EPOCHS_DECODE 600) beats the default on the other
  two tables where batch 1024 keeps enough steps an epoch.
- **Scenarios:** letter and electricity 20–80nan × seeds 42/7/13 × arms M and P600, five folds,
  MLflow off; 42 on gorgona8, 7 and 13 on medusa1; no cell starts after 2026-10-31 09:00 GMT-3.
- **Measures and decision rule:** P600 − M on the induced `impute_score`, relative, 15 fold pairs,
  1% SESOI; per table, supported if better on 2 or more of 4 and worse on none, refuted if better
  on none, mixed otherwise.
- **Sources:** [imputation-protocol/40](tickets/imputation-protocol/40-large-batch-large-tables.md);
  `scripts/experiments/large_batch.py --study e103`.
- **Results (2026-10-31 07:36 GMT-3):** all 48 cells, inside the cap. letter: 20nan −2.90%, 40nan −1.46% (better),
  60nan −0.13% (equivalent), 80nan +0.21% (worse). electricity: 20nan +0.09%, 40nan +0.80% (no
  verdict), 60nan −0.09%, 80nan −0.02% (equivalent).
- **Conclusion:** letter mixed, electricity refuted. With E101 and E102: the large batch helps
  pendigits and letter at 20–40nan and fades or costs a little by 60–80nan; it does nothing on
  electricity. No default change follows; the user decides whether anything does.

### E104 · Does the large batch hold on pendigits and letter at 20–40nan, on fresh seeds? (2026-10-31)

- **Status:** replicated
- **Hypothesis:** the scope E101–E103 suggest, chosen after those results, holds on unseen seeds:
  P600 beats the default on pendigits and letter at 20 and 40nan.
- **Scenarios:** those 4 variants × seeds 1357/2468/3579 × arms M and P600, five folds, MLflow off;
  1357 on gorgona8, 2468 and 3579 on medusa1.
- **Measures and decision rule:** P600 − M on the induced `impute_score`, relative, 15 fold pairs,
  1% SESOI. Replicated if better on 3 or more of 4 and worse on none; not replicated if better on 1
  or fewer or worse on any; partial otherwise.
- **Sources:** [imputation-protocol/41](tickets/imputation-protocol/41-large-batch-rule.md);
  `scripts/experiments/large_batch.py --study e104`.
- **Results (2026-10-31 12:58 GMT-3):** all 24 cells. P600 − M: pendigits_20nan −5.93%, pendigits_40nan −4.10%,
  letter_20nan −3.25%, letter_40nan −1.54%, all better with every seed below zero.
- **Conclusion:** replicated. The rule chosen from E101–E103 holds on unseen seeds; an ADR proposal
  for these four variants goes to the user. **ADR 0036 accepted 2026-10-31; the four promoted files carry P600.**

### E105 · Does the re-masking spread still rank errors under MNAR and MAR at 40nan? (2026-10-31)

- **Status:** supported
- **Hypothesis:** E84's result (the spread's within-column ranking holds under MAR and self-masking
  MNAR) holds at 40nan too.
- **Scenarios:** E84's 12 shifted variants at 40nan (new data files), their MCAR twins and `marfctl`
  controls, seeds 101/202/303, 63 cells, MLflow off, medusa1.
- **Measures and decision rule:** E84's: a signal on 10 or more of 12 under K and H and none against
  → supported; 7 or fewer under K or any against → refuted; mixed otherwise.
- **Sources:** [imputation-protocol/42](tickets/imputation-protocol/42-spread-40nan.md);
  `scripts/experiments/spread_40nan.py`, `run_spread_40nan_queue.sh`.
- **Results (2026-10-31 12:11 GMT-3):** all 63 cells. Signal on 12 of 12 under K and under H, none against; ρ
  +0.11 to +0.54 under K, +0.10 to +0.53 under H. On biodeg, shifted minus MCAR is +0.002, +0.001, −0.021, −0.080 at slopes 0.25–2.0.
- **Conclusion:** supported. The spread's ranking holds under MAR and self-masking MNAR at 40nan as
  at 60nan, weakening with the tilt. Extending ADR 0030's decision 7 to 40nan is the user's decision. **Extended 2026-10-31 on the user's decision (ADR 0030, decision 7).**

### E99 · Does TRIDENT generally beat MLP, SAINT and TabNet on classification at 40–80nan? (2026-10-31 – 2026-11-01)

- **Status:** supported (numbered when drafted on 2026-10-30; pre-registered 2026-10-31).
- **Hypothesis:** the user's claim: TRIDENT, tuned on validation (E98), beats deep tabular models
  trained per dataset (MLP, SAINT, TabNet), tuned the same way, at 40, 60 and 80nan.
- **Scenarios:** 27 variants × 3 models, each a 20-trial TPE study on seed 42's validation macro-F1
  and evaluations at seeds 42/7/13 on ten folds; TRIDENT's folds from E98. pytorch-widedeep 1.5.0;
  medusa1 and gorgona8 by unit; no unit starts after 2026-11-01 08:00 GMT-3.
- **Measures and decision rule:** TRIDENT − baseline test macro-F1, 270 fold pairs a level, the
  project's rule. TRIDENT leads a baseline if higher at 2 or more of 3 levels and lower at none.
  Supported if it leads all three; refuted if any baseline is higher at 2 or more levels; mixed
  otherwise.
- **Sources:** [classification-step-zero/08](tickets/classification-step-zero/08-deep-baselines.md);
  `scripts/experiments/deep_baselines.py`, `run_deep_baselines_queue.sh`.
- **Results (2026-11-01 05:57 GMT-3):** all 81 units. TRIDENT − MLP +0.013, +0.008, −0.001 (no verdict) at 40/60/80nan;
  − SAINT +0.036, +0.036, +0.029; − TabNet +0.111, +0.106, +0.091. TabNet is the weakest baseline by
  far (mean 0.47–0.64), though not collapsed.
- **Conclusion:** supported: TRIDENT leads all three by the majority rule. The margin over MLP is about
  a point and vanishes at 80nan; a report states it and does not lean on the TabNet margin.

### E107 · The deep baselines at 00 and 20nan (descriptive) (2026-11-01)

- **Status:** done (descriptive)
- **Question:** does TRIDENT also lead MLP, SAINT and TabNet with few gaps (E99 covered 40–80nan)?
- **Scenarios:** E99's 3 models × the 18 variants at 00 and 20nan, E99's recipe; TRIDENT from E98;
  both machines; no unit starts after 2026-11-02 08:00 GMT-3.
- **Measures:** TRIDENT − baseline test macro-F1, 270 fold pairs a level, the project's rule;
  descriptive, no primary call.
- **Sources:** [classification-step-zero/10](tickets/classification-step-zero/10-deep-baselines-low-missingness.md);
  `deep_baselines.py --study e107`.
- **Results (2026-11-01 16:15 GMT-3):** all 54 units. TRIDENT − MLP −0.012 (00nan, MLP higher), +0.009 (20nan); − SAINT
  +0.006, +0.029; − TabNet +0.055, +0.108; every call with seed agreement.
- **Conclusion (descriptive):** with E99, TRIDENT leads all three from 20 to 60nan, ties MLP at
  80nan, and trails MLP on the complete tables.

### E106 · Was E99's TabNet undertrained? Its configuration with a bigger budget (2026-11-01)

- **Status:** holds (E99's TabNet reading)
- **Hypothesis to rule out:** E99's TabNet margin (+0.09 to +0.11) is a budget artefact.
- **Scenarios:** E99's best TabNet configuration per variant (27 at 40–80nan), retrained with batch
  256, up to 400 epochs and patience 50, seeds 42/7/13, ten folds; no new search (the user's
  choice, on a probe pricing a re-tune at ~190 GPU-hours).
- **Measures and decision rule:** TRIDENT − TabNet-B, 270 fold pairs a level; E99's TabNet reading
  holds if TRIDENT leads by E99's rule. TabNet-B − TabNet is descriptive.
- **Sources:** [classification-step-zero/09](tickets/classification-step-zero/09-tabnet-budget.md);
  `deep_baselines.py --study e106`.
- **Results (2026-11-01 14:39 GMT-3):** all 27 units. TRIDENT − TabNet-B +0.096, +0.098, +0.076 at 40/60/80nan, every
  seed above zero. TabNet-B − TabNet +0.015, +0.008, +0.016.
- **Conclusion:** E99's TabNet margin is not a budget artefact; TRIDENT still leads by about 0.08–0.10.

### E108 · The thesis tables' cells ADR 0036 changed, rerun on today's default (2026-11-01 – 2026-11-02)

- **Status:** done (a rerun for reporting, not a hypothesis test).
- **Scenarios:** pendigits_20nan and letter_20nan at 101/202/303 and 42/7/13; pendigits_40nan and
  letter_40nan at 707/808/909; today's default, five folds, `--score_checkpoints
  --score_calibrated`, MLflow tag `experiment=adr36-thesis-2026-11-01`.
- **Measures:** the thesis tables (`thesis_table.py`, `tabpfn_report.py`) rebuilt with these runs,
  before and after.
- **Sources:** [imputation-protocol/43](tickets/imputation-protocol/43-adr36-thesis-cells.md).
- **Results (2026-11-02 02:49 GMT-3):** 21 cells on gorgona8 (medusa1's first 18 set aside: its kNN baselines differ
  from E41's). K − TabPFN on pendigits_20nan +0.012 → −0.008 (101/202/303) and +0.025 → −0.007 (42/7/13);
  letter_20nan −0.021 → −0.033 and +0.005 → −0.032. Development K vs TabPFN 8/6/7 → 9/6/6; the 42/7/13 set
  grows to 18 variants. No other variant moved.

### E109 · Does the large batch help pendigits_60nan on fresh seeds? (2026-11-01)

- **Status:** supported
- **Hypothesis:** E101's and E102's −1.2% on pendigits_60nan under P600 replicates on unseen seeds.
- **Scenarios:** pendigits_60nan × seeds 2121/3232/4343 × arms M and P600, five folds, MLflow off,
  medusa1.
- **Measures and decision rule:** P600 − M on the induced `impute_score`, relative, 15 fold pairs, 1%
  SESOI; supported if better.
- **Sources:** [imputation-protocol/44](tickets/imputation-protocol/44-large-batch-pendigits-60nan.md);
  `large_batch.py --study e109`.
- **Results (2026-11-01 22:42 GMT-3):** P600 − M −1.34% [−1.66%, −1.02%], seeds −0.99%, −1.49%, −1.54%: better.
- **Conclusion:** supported; adding pendigits_60nan to ADR 0036 is the user's decision. **Added 2026-11-02 on the user's decision (ADR 0036 amended; verified identical to E109's P600 cell).**

### E110 · Does the re-masking spread still rank errors under MNAR and MAR at 20nan? (2026-11-01 – 2026-11-02)

- **Status:** supported
- **Hypothesis:** E84's and E105's result holds at 20nan.
- **Scenarios, measures and rule:** E105's at 20nan (63 cells, medusa1, MLflow off); no cell starts after
  2026-11-02 08:30 GMT-3. **Cap removed 2026-11-02 00:04 GMT-3 on the user's word, before any result.**
- **Sources:** [imputation-protocol/45](tickets/imputation-protocol/45-spread-20nan.md);
  `spread_40nan.py --level 20`.
- **Results (2026-11-02 05:04 GMT-3):** 63 cells; a signal on 12 of 12 under K and H, none against; ρ +0.14 to +0.55.
  On biodeg the ranking does not weaken with the tilt at 20nan.
- **Conclusion:** supported; extending ADR 0030's decision 7 to 20nan is the user's decision. **Extended 2026-11-03 on the user's decision (ADR 0030, decision 7).**

### E111 · E99 on fresh seeds: does TRIDENT still lead MLP and SAINT at 40–80nan? (2026-11-02)

- **Status:** replicated
- **Hypothesis:** E99's reading holds on seeds 5151/6262/7373, with no re-tuning of either side.
- **Scenarios:** 27 variants at 40–80nan; TRIDENT (E98's configurations) and MLP and SAINT (E99's)
  evaluated at the three seeds, ten folds; both machines; no item starts after 2026-11-02 08:30 GMT-3. **Cap removed 2026-11-02 00:04 GMT-3 on the user's word, before any result.**
- **Measures and decision rule:** E99's; replicated if TRIDENT leads both MLP and SAINT.
- **Sources:** [classification-step-zero/11](tickets/classification-step-zero/11-deep-baselines-fresh-seeds.md).
- **Results (2026-11-02 05:04 GMT-3):** TRIDENT − MLP +0.009, +0.007, −0.000 (no verdict) at 40/60/80nan; − SAINT +0.035,
  +0.035, +0.029.
- **Conclusion:** replicated; E99's reading holds on seeds it never used.

### E112 · Does an EMA target make classification pre-training help? (T13) (2026-11-02)

- **Status:** mixed
- **Hypothesis:** with an EMA teacher as the target (`PRETRAIN_OBJECTIVE embedding_ema`), pre-training
  beats no pre-training on classification, where E87 found the legacy objective gave nothing.
- **Scenarios:** E87's protocol (old sweep configuration, ten folds, 45 variants, seeds
  42/7/13/101/202); arm E new, arms A and B from E87 (B rerun on gorgona8 at seed 202, beside its E).
- **Measures and decision rule:** E − B per level, 450 fold pairs, SESOI 0.01; supported if E is
  higher at 3 or more of 5 levels and lower at none, refuted if higher at 1 or fewer. E − A descriptive.
- **Sources:** [classification-step-zero/12](tickets/classification-step-zero/12-ema-target.md). **Amended 2026-11-02 11:28 GMT-3:** a NaN bug (the teacher read gaps as NaN) fixed and the study relaunched from scratch under tag `cls-ema-v2-2026-11-02`; design and rule unchanged. **Second amendment 2026-11-02 14:31 GMT-3:** seed 202 moved whole to medusa1 (gorgona8 about 3x slower an E cell), against E87's B there; gorgona8's cells unread.
- **Results (2026-11-02 22:34 GMT-3):** 225 E cells. E − B: 00nan +0.0012, 20nan +0.0014 (equivalent), 40nan +0.0050, 60nan
  +0.0085 (E higher), 80nan +0.0036 (equivalent). E − A: E higher at 40nan (+0.004), equivalent elsewhere.
- **Conclusion:** mixed. The EMA target is the first pre-training to move classification (40–60nan), by
  less than the SESOI; the user decides between a fresh-seed check and closing T13.

### E113 · Does the bigger encoder still help pendigits and letter at 40–60nan on today's defaults? (T32) (2026-11-02 – 2026-11-03)

- **Status:** supported
- **Hypothesis:** ADR 0018's bigger encoder improves pendigits and letter at 40 and 60nan over today's
  defaults (ADR 0036's P600 included), as E59 saw over the older ones.
- **Scenarios:** those 4 variants × seeds 42/7/13 × arms M and BIG, five folds, MLflow off, gorgona8.
- **Measures and decision rule:** BIG − M on the induced `impute_score`, relative, 15 fold pairs, 1%
  SESOI; supported if better on 2 or more of 4 and worse on none, refuted if better on none.
- **Sources:** [imputation-protocol/46](tickets/imputation-protocol/46-bigger-encoder-large-tables.md);
  `large_batch.py --study e113`.
- **Results (2026-11-03 01:44 GMT-3):** BIG − M: pendigits_40nan −6.04%, pendigits_60nan −3.14%, letter_40nan −5.47%,
  letter_60nan −1.12%, all better with every seed below zero.
- **Conclusion:** supported; a fresh-seed confirmation (E116) before an ADR for the user.

### E114 · Does the EMA target's gain at 40–60nan replicate on fresh seeds? (2026-11-02 – 2026-11-03)

- **Status:** partial
- **Hypothesis:** E112's E − B gains at 40 and 60nan hold on seeds 3131/4141/5252.
- **Scenarios:** 18 variants at 40/60nan × 3 seeds × arms E and B, E87's protocol, medusa1.
- **Measures and decision rule:** E − B per level, 270 fold pairs, SESOI 0.01; replicated if E higher at
  both levels, not replicated if at neither or B higher at either, partial otherwise.
- **Sources:** [classification-step-zero/13](tickets/classification-step-zero/13-ema-target-confirmation.md).
- **Results (2026-11-03 02:14 GMT-3):** E − B 40nan +0.0086 (no verdict, seed 5252 −0.0005), 60nan +0.0083 (E higher).
- **Conclusion:** partial; T13 closes for classification (a small, sub-SESOI gain at 40–60nan), no ADR.

### E115 · TRIDENT against MLP and SAINT under MAR and MNAR at 60nan (descriptive) (2026-11-03)

- **Status:** done (descriptive)
- **Question:** does E99's MCAR picture (TRIDENT ahead of MLP and SAINT at 60nan) hold under MAR and
  self-masking MNAR?
- **Scenarios:** 4 MNAR and 6 MAR variants at 60nan, each side with its MCAR sibling's tuned
  configuration (E98, E99), seeds 42/7/13, ten folds, medusa1.
- **Measures:** per mechanism, TRIDENT − baseline test macro-F1, the project's rule; descriptive.
- **Sources:** [classification-step-zero/14](tickets/classification-step-zero/14-deep-baselines-mar-mnar.md).
- **Results (2026-11-03 01:19 GMT-3):** TRIDENT − SAINT +0.045 (MNAR), +0.043 (MAR), TRIDENT higher; − MLP +0.011 (MAR, TRIDENT
  higher), +0.006 (MNAR, no verdict: one seed below zero).

### E116 · Does the bigger encoder's gain on pendigits and letter at 40–60nan replicate on fresh seeds? (2026-11-03)

- **Status:** replicated
- **Hypothesis:** E113's BIG − M gains hold on seeds 6161/7171/8181.
- **Scenarios:** the 4 variants × 3 seeds × arms M and BIG, five folds, MLflow off; 6161 on gorgona8,
  7171/8181 on medusa1.
- **Measures and decision rule:** BIG − M, relative, 15 fold pairs, 1% SESOI; replicated if better on 3 or
  more of 4 and worse on none.
- **Sources:** [imputation-protocol/47](tickets/imputation-protocol/47-bigger-encoder-confirmation.md).
- **Results (2026-11-03 04:37 GMT-3):** BIG − M pendigits_40nan −6.26%, pendigits_60nan −3.09%, letter_40nan −5.45%, letter_60nan
  −1.23%, all better, every seed below zero.
- **Conclusion:** replicated; ADR 0037 is proposed for the user.

### E117 · Does the bigger encoder help pendigits and letter at 80nan? (2026-11-03)

- **Status:** mixed
- **Hypothesis:** BIG beats today's default at 80nan too (ADR 0037's proposal stops at 60nan).
- **Scenarios:** pendigits_80nan and letter_80nan × seeds 42/7/13 × M and BIG, five folds; 42 on gorgona8,
  7/13 on medusa1.
- **Measures and decision rule:** BIG − M, relative, 15 fold pairs, 1% SESOI; supported if better on both,
  refuted if on neither.
- **Sources:** [imputation-protocol/48](tickets/imputation-protocol/48-bigger-encoder-80nan.md).
- **Results (2026-11-03 06:09 GMT-3):** BIG − M pendigits_80nan −0.39% (better), letter_80nan +0.37% (worse).
- **Conclusion:** mixed, both under the SESOI; ADR 0037 keeps its 40–60nan scope.

### E118 · Does pre-training help classification under the upstream code's defaults? (2026-11-03)

- **Status:** refuted
- **Hypothesis:** the paper's Table 4 pre-training gain (about +1.7 macro-F1) shows under the upstream
  code's built-in defaults (40/40 epochs), where E87's old sweep (300/150) showed none. The paper's tuned
  per-dataset values are not in the repository and stay untested.
- **Scenarios:** arms A (`EPOCHS_PRE 40`) and B (`EPOCHS_PRE 0`), both `EPOCH_FINE 40`, `cosine_legacy`;
  9 tables × 5 levels × seeds 42/7/13/101 (medusa1) and 202 (gorgona8), ten folds; 450 cells.
- **Measures and decision rule:** E87's: B − A on test macro-F1, 450 fold pairs a level, SESOI 0.01;
  supported if A is higher at 3 or more of 5 levels and B at none, refuted if at 1 or none.
- **Sources:** [classification-step-zero/15](tickets/classification-step-zero/15-upstream-defaults-pretraining.md), T41.
- **Results (2026-11-03 14:26 GMT-3):** B − A equivalent at every level (inside ±0.01), from −0.0009 (60nan) to +0.0023
  (00nan); A higher at no level, B at none. 450 cells, none failed.
- **Conclusion:** refuted; no pre-training gain under the upstream defaults, as under E87's old sweep. The
  paper's tuned per-pair values stay untested.

### E119 · Does ADR 0037's bigger encoder help pendigits and letter under MAR and MNAR? (2026-11-03)

- **Status:** holds (MAR and MNAR)
- **Hypothesis:** BIG's MCAR gain (E113, E116) holds on MAR (`marf`) and MNAR (`probit`) variants at 40–60nan.
- **Scenarios:** 8 new variants × arms M (sibling's configuration, encoder undone) and BIG × seeds 42
  (gorgona8), 7/13 (medusa1), five folds, MLflow off.
- **Measures and decision rule:** BIG − M, relative, 15 fold pairs, 1% SESOI; per mechanism, holds if better on
  3 or more of 4 and worse on none, does not hold if better on 1 or fewer or worse on any.
- **Sources:** [imputation-protocol/49](tickets/imputation-protocol/49-bigger-encoder-mar-mnar.md).
- **Amendment (2026-11-03 18:56 GMT-3):** the pair `letter_probit_60nan` seed 42 moved from gorgona8 to medusa1
  (both arms there) to finish sooner; same cells and rule.
- **Results (2026-11-03 19:54 GMT-3):** BIG − M better on 4 of 4 MAR variants (−3.6% to −5.9%) and 4 of 4 MNAR (−1.8% to
  −6.9%), worse on none; every seed mean negative.
- **Conclusion:** ADR 0037's encoder helps pendigits and letter at 40–60nan under MAR and MNAR as under MCAR;
  the ADR's scope is the user's decision.

### E120 · Does TRIDENT lead MLP under MNAR on fresh seeds? (2026-11-03)

- **Status:** no verdict
- **Hypothesis:** TRIDENT − MLP test macro-F1 under MNAR is positive, as under MCAR and MAR (E115 left it
  without a verdict).
- **Scenarios:** E115's 10 variants at 60nan, seeds 5151/6262/7373, ten folds, MCAR siblings' tuning, medusa1.
- **Measures and decision rule:** TRIDENT − MLP under MNAR, 120 fold pairs, SESOI 0.01: supported if TRIDENT
  higher, refuted if MLP higher, tie if equivalent.
- **Sources:** [classification-step-zero/16](tickets/classification-step-zero/16-trident-mlp-mnar-fresh-seeds.md).
- **Results (2026-11-03 16:03 GMT-3):** TRIDENT − MLP under MNAR +0.0025 [−0.0090, +0.0141], seeds all positive, no
  verdict; SAINT under MNAR, and both under MAR, TRIDENT higher as in E115.
- **Conclusion:** no verdict; with E115, no lead over MLP can be claimed under MNAR.

### E121 · Does ADR 0018's bigger encoder help pendigits and letter at 20nan under MAR and MNAR? (2026-11-03)

- **Status:** holds (MAR and MNAR)
- **Hypothesis:** the bigger encoder's MCAR gain at 20nan (E38) holds on MAR (`marf`) and MNAR (`probit`), as
  E119 found at 40–60nan.
- **Scenarios:** 4 variants × arms M (sibling's configuration, encoder undone) and BIG × seeds 42 (gorgona8),
  7/13 (medusa1), five folds, MLflow off.
- **Measures and decision rule:** BIG − M, relative, 15 fold pairs, 1% SESOI; per mechanism, holds if better on
  both and worse on none, does not hold if better on neither or worse on any.
- **Sources:** [imputation-protocol/50](tickets/imputation-protocol/50-bigger-encoder-20nan-mar-mnar.md).
- **Amendment (2026-11-03 22:03 GMT-3):** the pair `letter_probit_20nan` seed 42 moved from gorgona8 to medusa1
  (both arms there) to finish sooner; same cells and rule.
- **Results (2026-11-03 22:47 GMT-3):** BIG − M better on 2 of 2 MAR variants (−4.6%, −5.8%) and 2 of 2 MNAR (−5.0%, −6.5%),
  worse on none; every seed mean negative.
- **Conclusion:** ADR 0018's encoder helps pendigits and letter at 20nan under MAR and MNAR as under MCAR; the
  ADR's scope is the user's decision.

### E122 · C15 Stage 0: which MNAR variants carry informative missingness (gate G2)? (2026-11-03)

- **Status:** invalid (G2's positive controls fail on 2 of 3)
- **Hypothesis:** the indicators lower a GBDT's validation log loss by at least 2% on the positive controls (F5)
  and on some self-masking (F1) and outcome-dependent (F3) variants, which then form C15's primary set P.
- **Scenarios:** 19 candidates (F1 7, F2 3, F3 6, F5 3) × 15 cells (seeds 42/7/13 × 5 folds, the decision mode's
  folds), missForest fills fitted on training rows, CPU on gorgona8; plus the driving-column table.
- **Measures and decision rule:** relative drop in validation log loss with the indicators; informative if the
  mean is ≥ 0.02 with the 95% t interval above 0; every F5 must pass or C15 stops.
- **Sources:** [c15-mnar-prediction/01](tickets/c15-mnar-prediction/01-stage0.md),
  [plan](plans/2026-11-03-c15-mnar-prediction.md).
- **Results (2026-11-03 23:45 GMT-3):** F5 controls: spambase passes, biodeg (+0.038 [−0.022, +0.097]) and kc2 (+0.102
  [−0.020, +0.224]) do not; F1 passes on biodeg_probit_60nan only; F3 and F2 all pass (descriptive, since G2 is invalid).
- **Conclusion:** by the pre-registered rule C15 stops at Stage 0; a continuation needs a new pre-registration.

### E123 · C15's gate G2 again, read out of fold over every training and validation row (2026-11-03)

- **Status:** valid; P = 5 variants
- **Hypothesis:** with about 8 times the rows per cell, the positive controls (F5) clear G2's rule, and the gate
  defines C15's primary set P.
- **Scenarios:** E122's 19 candidates × 15 cells; per cell, missForest fitted on the training and validation rows'
  features, the GBDT pair and the probe read on 5 inner folds, test rows unread. A deviation chosen after E122.
- **Measures and decision rule:** as E122 (mean drop ≥ 0.02, 95% t interval above 0; every F5 must pass); if the
  controls fail again, C15 stops at Stage 0 for good.
- **Sources:** [c15-mnar-prediction/02](tickets/c15-mnar-prediction/02-g2-out-of-fold.md).
- **Results (2026-11-04 00:12 GMT-3):** F5 controls pass 3 of 3 (+0.043, +0.055, +0.085); P = biodeg_probit_60nan,
  vehicle_probit_60nan and the odds-3 outcome variants of biodeg, kc2, spambase; F2 passes at slopes 1.0 and 2.0.
- **Conclusion:** G2 holds; C15's primary set is fixed. Five of seven self-masking variants carry too little
  signal for prediction; the outcome-dependent family carries a lot (+0.76 to +0.85 at odds 3).

### E124 · C15 Stage 1: gate G1 on the positive controls, and A1 against A2 on the development set (2026-11-04)

- **Status:** done: G1 failed (1 of 3); Stage 1b descriptive
- **Hypothesis:** on F5 the indicator arm beats MICE without one and A1's direct contrast on the noise column is
  positive (G1); the development set gives A1 − A2 estimates before confirmation.
- **Scenarios:** 1a: A1, A2, A4:0–4 on F5's three controls, seed 42 (21 cells); 1b: A1, A2 on 23 development
  variants × seeds 42/7/13 (138 cells); five folds, `embedding_normalized`, gorgona8 seed 42, medusa1 the rest.
- **Measures and decision rule:** G1 passes if, on at least 2 of 3 controls, A2 − A4 relative log loss is below 0
  and A1's noise-column contrast above 0 (Nadeau–Bengio corrected intervals); 1b descriptive.
- **Sources:** [c15-mnar-prediction/03](tickets/c15-mnar-prediction/03-stage1.md).
- **Amendment (2026-11-04 01:10 GMT-3):** seed 42 of the development set moves to medusa1 except spambase and kc2 (whole pairs, every stage), to balance the machines.
- **Results, G1 (2026-11-04 01:16 GMT-3):** A1's noise-column contrast positive on 3 of 3 (+1.78, +3.80, +1.75 logit; interval above 0 on 2); A2 beats A4 on log loss only on spambase (−0.33); biodeg and kc2 intervals wide over 5 cells. G1 fails, C15 stops before Stage 2.
- **Results, Stage 1b (2026-11-04 01:34 GMT-3):** A1 − A2 inferior on 5 of 23 development variants (vehicle 60nan, both vehicle
  self-masking levels, biodeg_60nan, biodeg's dial at 0.25), non-inferior on none; the deficit shows under MCAR too.
- **Conclusion:** G1 fails, so C15 stops before Stage 2; on development, `[NULL]` trails missForest + indicator in
  the same encoder, MCAR included. The next step is the user's.

### E125 · C15 Stage 2: A4 (MICE, no indicator), the complete and MAR tables; H2-c2, H2.2 and H2b on development (2026-11-04)

- **Status:** withdrawn unrun (G1 failed)
- **Hypothesis:** A1 beats A4 on P's development counterparts (H2-c2); the direct contrast grows on driving columns
  under MNAR (H2.2 dry run, E63's MAR the negative control); mask-reading arms lose more than A4 under shifted test
  patterns (H2b).
- **Scenarios:** A4:0–4 on F0, F1, F3 × seeds 42/7/13 (300 cells); A1 on F7 (12) and F6 (18).
- **Measures and decision rule:** the protocol's, in [c15-mnar-prediction/04](tickets/c15-mnar-prediction/04-stage2.md).
- **Sources:** [c15-mnar-prediction/04](tickets/c15-mnar-prediction/04-stage2.md).

### E126 · C15 Stage 3: H2's confirmation, A1 against A2 on fresh draws of P (2026-11-04)

- **Status:** withdrawn unrun (G1 failed)
- **Hypothesis (H2):** `[NULL]` tokens are non-inferior to missForest + indicator, pooled over P (−0.01 macro-F1,
  +2% relative log loss), and inferior on no variant.
- **Scenarios:** A1, A2 on P's fresh draws (5) and their MCAR siblings (4), data seed 4848, model seeds 1515/2626/3737,
  54 cells on medusa1.
- **Measures and decision rule:** pooled D_c over P, Nadeau–Bengio corrected; supported / reversed / inconclusive as
  in [c15-mnar-prediction/05](tickets/c15-mnar-prediction/05-stage3.md).
- **Sources:** [c15-mnar-prediction/05](tickets/c15-mnar-prediction/05-stage3.md).

### E127 · C15 Stage 4 (optional): A4 on the dial, the sign arm, A3 and the null model A5 (2026-11-04)

- **Status:** withdrawn unrun (G1 failed)
- **Hypothesis:** none decides H2; H2-dial (A4's loss against A1's grows along the tilt), the sign arm's contrast
  signs, what the indicator adds over the same fill (A2 − A3), and the null model's exact-zero contrast.
- **Scenarios:** A4 on F2 (45 cells), A1/A2 on F4 (36), A3 on the 60nan self-masking and MCAR pairs (24), A5 on the
  four self-masking 60nan variants (12); seeds 42/7/13.
- **Measures and decision rule:** descriptive, in [c15-mnar-prediction/06](tickets/c15-mnar-prediction/06-stage4.md).
- **Sources:** [c15-mnar-prediction/06](tickets/c15-mnar-prediction/06-stage4.md).

### E128 · C15's gate G1 again: A2 against A3 in decision mode (2026-11-04)

- **Status:** done: G1 failed again (1 of 3); C15 stopped for good
- **Hypothesis:** on F5 the indicator arm beats the same missForest fill without one (A2 − A3 below 0) and A1's direct
  contrast on the noise column is positive, now over 15 cells. A deviation after E124's failure (the user's choice);
  if it fails, C15 stops for good.
- **Scenarios:** A3 at seeds 42/7/13 and A1, A2 at seeds 7/13 on F5's three controls (21 cells; E124's seed-42 A1 and
  A2 reused), five folds, `embedding_normalized`; gorgona8 biodeg's and kc2's seed 13, medusa1 the rest.
- **Measures and decision rule:** passes if, on at least 2 of 3 controls, A2 − A3 relative log loss is below 0 and
  A1's contrast above 0 (Nadeau–Bengio corrected intervals over 15 cells).
- **Sources:** [c15-mnar-prediction/07](tickets/c15-mnar-prediction/07-g1-a3-decision-mode.md).
- **Results (2026-11-04 08:26 GMT-3):** A1's noise-column contrast above 0 on 3 of 3 (+1.78, +2.89, +1.68 logit, every
  interval above 0); A2 beats A3 on log loss only on spambase (−0.098 [−0.153, −0.044]); biodeg (−0.045) and kc2
  (−0.021) below 0 in point but their corrected intervals (about ±0.14) include 0. G1 passes on 1 of 3: fails.
- **Conclusion:** by the binding clause C15 stops for good before Stage 2 (E125–E127 stay withdrawn, T45 closed), H2
  undecided. Descriptively, `[NULL]` reads an informative gap on every control; the indicator's gain over the same fill
  is measurable only on the largest table.

### E129 · C1 cut 1: gate G1, the indicators-only probe on the primary MNAR files and their MCAR siblings (2026-11-04)

- **Status:** done: G1 passes (4 of 4)
- **Hypothesis:** the gaps of the four MNAR 60nan files carry outcome information their MCAR siblings' gaps do not
  (the manipulation check; it conditions only C1's Null row).
- **Scenarios:** logistic regression on the 0/1 gap matrix, fitted on the training rows and scored on the validation
  rows of the decision mode's 15 folds (seeds 42/7/13 × 5), on `{biodeg,kc2,vehicle,spambase}_probit_60nan` and
  `_60nan`; CPU only.
- **Measures and decision rule:** MNAR − MCAR validation AUROC, pooled over the four tables per replicate; passes if
  the plain 95% t interval is above 0 and no table's mean is negative.
- **Sources:** [c1-split/01](tickets/c1-split/01-g1-probe.md); [plan](plans/2026-11-04-c1-split.md).
- **Results (2026-11-04 12:50 GMT-3):** MNAR − MCAR validation AUROC above 0 on every table (biodeg +0.16, kc2 +0.11,
  vehicle +0.10, spambase +0.13; MCAR probes at chance); pooled +0.126 [+0.107, +0.144].
- **Conclusion:** G1 passes; the primary files' gap patterns carry outcome information their MCAR siblings' do not,
  kc2 and spambase included (E123's GBDT gain measured something else: the indicators' value beyond a fill).

### E130 · C1 cut 2: the integrity smoke, S, T, B and BT on biodeg_probit_60nan, deterministic (2026-11-04)

- **Status:** done: passes
- **Hypothesis:** under no pre-training, S's and T's code paths give bit-identical results (integrity check 1), so
  the tied arm differs from S only through pre-training.
- **Scenarios:** S, T, B and BT (B through T's path) on `biodeg_probit_60nan`, seed 42, five folds, deterministic
  algorithms forced; gorgona8.
- **Measures and decision rule:** B's and BT's test probabilities and kept states identical on all five folds; no S − T
  contrast is read.
- **Sources:** [c1-split/02](tickets/c1-split/02-integrity-smoke.md).
- **Results (2026-11-04 12:55 GMT-3):** B and BT identical on all five folds (largest probability difference 0, kept states
  equal); S and T ran end to end.
- **Conclusion:** integrity check 1 holds; cut 3 starts.

### E131 · C1 cut 3: S against T on the four MNAR 60nan files and their MCAR siblings; G2; the anchor (2026-11-04)

- **Status:** done: Inconclusive (G2 fails, no pre-training channel)
- **Hypothesis (C1):** per-column `[NULL]` apart from `[MASK]` (S) beats one tied token per column (T) on macro-F1,
  pooled over the four MNAR files; the MCAR files, read second, say whether the gain is specific to informative
  missingness.
- **Scenarios:** S reused from E124's A1 cells; T (`PRETRAIN_HOLE_TOKEN null`) and B (`EPOCHS_PRE 0`) on
  `{biodeg,kc2,vehicle,spambase}_probit_60nan` and `_60nan`, seeds 42/7/13 × 5 folds (48 cells); ANCHOR (S with
  `embedding`) on the four MNAR files (12); S_GEOM at seed 42 for the token dump (4). `embedding_normalized` by
  override; gorgona8 seed 42 of kc2 and spambase, medusa1 the rest (ADR 0035).
- **Measures and decision rule:** pooled S − T macro-F1 over the 15 replicates, plain 95% t interval: Supported above
  0 (relevant at ≥ +0.010, real but small below), Reversed below 0, Null inside ±0.010 with G1 and G2 passing,
  Inconclusive otherwise; per-file verdicts are flags; G2 = S − B interval off 0; Nadeau–Bengio beside; log loss the
  same rule without a margin; anchor and token cosines descriptive.
- **Sources:** [c1-split/03](tickets/c1-split/03-primary.md); [plan](plans/2026-11-04-c1-split.md).
- **Results (2026-11-04 13:26 GMT-3):** pooled S − T on the MNAR files −0.0034 [−0.0110, +0.0042] macro-F1 (no per-file verdict,
  no flags); G2 S − B +0.0040 [−0.0077, +0.0157] fails; MCAR −0.0024 [−0.0107, +0.0060], interaction 0 on every table;
  anchor +0.005 [−0.003, +0.013]; token cosines at the initialisation reference (|median| ≤ 0.02).
- **Conclusion:** Inconclusive, reason "no pre-training channel": with S ≈ B (E71 again), the split, which acts only in
  pre-training, has nowhere to act; S and T tie within about ±0.01 on every file and the model does not tie the
  tokens itself. The reserve and cut 4 are not proposed; the next step is the user's.

### E132 · C1's channel probe: a frozen pre-trained encoder against a frozen random one, head only (2026-11-04)

- **Status:** done: content present, small (washout)
- **Hypothesis:** none directional. If pre-training leaves label-relevant structure (FP − FR above 0), E131's S ≈ B is
  washout and the fix is on the fine-tuning side; if a random frozen encoder serves the head as well (inside ±0.010),
  the lever is pre-training itself.
- **Scenarios:** FP (S's pre-training, encoder frozen in fine-tuning, head alone trains) and FR (`EPOCHS_PRE 0`, same
  freeze) on `{biodeg,kc2,vehicle,spambase}_probit_60nan` and `_60nan`, seeds 42/7/13 × 5 folds (48 cells); S and B
  from E124/E131.
- **Measures and decision rule:** FP − FR macro-F1 pooled over the four MNAR files, plain 95% t interval: content
  present above 0, absent inside ±0.010, no verdict otherwise; MCAR second step, S − FP and B − FR descriptive.
- **Sources:** [c1-split/04](tickets/c1-split/04-channel-probe.md).
- **Results (2026-11-04 18:34 GMT-3):** FP − FR pooled over the MNAR files +0.0092 [+0.0013, +0.0171] (NB reaches −0.009); MCAR
  +0.0160 [+0.0043, +0.0277], *better* on kc2 and spambase; S − FP +0.098 and B − FR +0.103 pooled.
- **Conclusion:** pre-training leaves label-relevant structure worth about one macro-F1 point to a frozen head, and
  full fine-tuning adds ten times that from either start: washout, the imputation side's diagnosis. A fine-tuning-side
  fix is bounded near +0.01; the user decides whether to run it.

### E133 · C1's fine-tuning fix: encoder at a tenth of the rate, or frozen for 50 epochs, against S; the probe with a converging head (2026-11-04)

- **Status:** done: neither arm helps (no verdict, means negative)
- **Hypothesis:** a gentler fine-tuning keeps what pre-training left (E132: about +0.01 to a frozen head) without losing
  what full fine-tuning adds, so LRS or FRZ beats S on macro-F1; the ceiling is about +0.01.
- **Scenarios:** LRS (`FINETUNE_ENCODER_LR_SCALE 0.1`) and FRZ (`FINETUNE_FREEZE_EPOCHS 50`) on
  `{biodeg,kc2,vehicle,spambase}_probit_60nan` and `_60nan`, seeds 42/7/13 × 5 folds (48 cells); FP10 and FR10 (E132's
  frozen arms with the head at ten times the rate) on the four MNAR files (24 cells).
- **Measures and decision rule:** X − S macro-F1 pooled over the four MNAR files, plain 95% t interval, Holm over the
  two arms: helps / hurts / no verdict; MCAR second step, X − B and FP10 − FR10 descriptive.
- **Sources:** [c1-split/05](tickets/c1-split/05-finetuning-fix.md).
- **Results (2026-11-04 19:31 GMT-3):** LRS − S −0.0013 [−0.0067, +0.0041], FRZ − S −0.0027 [−0.0142, +0.0088] on the MNAR files
  (Holm p 1.0); MCAR +0.003 / +0.000; X − B within +0.013 everywhere. Kept epochs move from 14–25 (S) to 52–137 (LRS)
  and 74–109 (FRZ) at the same scores. FP10/FR10 diverged at the higher head rate (spambase at chance): no reading.
- **Conclusion:** a gentler fine-tuning keeps nothing measurable of pre-training; the fine-tuning side of C1's channel is
  closed, and the +0.01 ceiling from E132 was not reached. E131 stays Inconclusive.

### E134 · The fine-tuning budget: 30 and 50 epochs against 150 under cosine_legacy (2026-11-04)

- **Status:** done 2026-11-04 (96 of 96 cells, 0 failed; read 21:21 GMT-3)
- **Hypothesis:** none directional; fine-tuning's validation minimum falls by epoch 13–24 (E132's curves), so a shorter
  budget may match or beat 150 epochs at a third of the cost.
- **Results:** S30 − S −0.0081 [−0.0153, −0.0010], p 0.029, Holm 0.058 → no verdict, vehicle worse on both files
  (−0.041, −0.047; its validation minimum sits at epoch 30–35, so 30 truncates it), kc2 better on both (+0.021,
  +0.019); S50 − S −0.0009 [−0.0049, +0.0031], p 0.64 → no verdict, no file better or worse. S(k) − B(k) small and of
  both signs on every file.
- **Conclusion:** no budget is better; S30 is not a cost argument (two files worse); S50 is a cost argument only (a
  third of fine-tuning's time, within ±0.005 pooled of 150), the user's call, with a caveat: vehicle's kept epoch
  still reaches the 50 cap on some replicates and both vehicle files lean −0.013/−0.016. No ADR change.
- **Scenarios:** S30, S50, B30, B50 (`EPOCH_FINE` 30/50, with and without pre-training) on
  `{biodeg,kc2,vehicle,spambase}_probit_60nan` and `_60nan`, seeds 42/7/13 × 5 folds (96 cells); S and B at 150 reused.
- **Measures and decision rule:** S(k) − S(150) macro-F1 pooled over the eight files, plain 95% t interval, Holm over
  the two budgets: better / worse / no verdict; S(k) − B(k) and log loss descriptive. A default change needs an ADR and
  a 21-variant confirmation, not done here.
- **Sources:** [c1-split/06](tickets/c1-split/06-finetuning-budget.md).

### E135 · Fine-tuning's schedule: cosine, warmup_cosine, plateau and constant against cosine_legacy (2026-11-04)

- **Status:** done 2026-11-04 (144 of 144 cells, 0 failed; read 21:21 GMT-3)
- **Hypothesis:** none directional; `cosine_legacy` is a per-batch sawtooth (ADR 0003) never measured on
  classification, and the user asked for the best schedule to be studied apart from the budget.
- **Results:** every arm no verdict (Holm p ≥ 0.85): COS150 −0.0023 [−0.0054, +0.0009], COS50 −0.0017 [−0.0064,
  +0.0030], WUC150 −0.0007 [−0.0078, +0.0064], WUC50 +0.0014 [−0.0043, +0.0072], PLAT150 −0.0025 [−0.0059, +0.0009],
  CONST150 −0.0026 [−0.0062, +0.0011]; no file better anywhere, COS150 worse on vehicle_probit, PLAT150/CONST150
  worse on spambase_60nan. Single annealings keep earlier epochs (medians 8–25 vs 11–32); COS50 keeps the same epochs
  as COS150. PLAT150 equals CONST150 on five of eight files (the kept state precedes the first halving).
- **Conclusion:** `cosine_legacy` stays fine-tuning's default; the schedule question is closed at this base rate and
  budget; no ADR proposal. `FINETUNE_LR_SCHEDULER` stays, default "".
- **Scenarios:** `FINETUNE_LR_SCHEDULER` (new, "" = shared schedule) at cosine 150/50, warmup_cosine 150/50, plateau
  150, constant 150, S's pre-training, on the eight files, seeds 42/7/13 × 5 folds (144 cells); S under cosine_legacy
  reused.
- **Measures and decision rule:** X − S macro-F1 pooled over the eight files, plain 95% t interval, Holm over the six
  arms: better / worse / no verdict; log loss and kept epochs descriptive; a winner goes to an ADR proposal with a
  21-variant confirmation.
- **Sources:** [c1-split/07](tickets/c1-split/07-finetuning-scheduler.md).

### E136 · Label-scarce fine-tuning: does pre-training open a channel at a tenth or a quarter of the labels? (2026-11-04)

- **Status:** done 2026-11-04 (144 of 144 cells, 0 failed; read 23:23 GMT-3)
- **Results:** gate S(f) − B(f): 0.10 +0.011 [−0.005, +0.026] (Holm 0.15, closed); 0.25 +0.016 [+0.007, +0.024]
  (Holm 0.003, OPEN; better on 4 files, worse on none; every seed positive). Primary at 0.25, S25 − T25 MNAR −0.002
  [−0.012, +0.007] → Inconclusive; MCAR −0.012 [−0.023, −0.001] descriptive (T ahead, vehicle worse −0.020);
  interaction +0.010 [−0.008, +0.027]. Cost of scarcity S25 − S −0.094, S10 − S −0.154. labelled_rows audited equal
  across arms.
- **Conclusion:** pre-training reaches the head at a quarter of the labels (the first positive S − B under full
  fine-tuning on classification here; E132's frozen-head +0.009 was the hint); the split S vs T still does not separate (Inconclusive under MNAR, T ahead under MCAR): no evidence for
  C1's claim where it could be tested. T is never worse than S. Next steps are the user's.
- **Hypothesis:** with few labels the encoder (pre-trained on every row) carries content the head cannot rebuild, so
  S − B > 0; if so, the `[NULL]`/`[MASK]` split (S − T) becomes testable where E131 could not.
- **Scenarios:** S10/T10/B10 (`FINETUNE_LABEL_FRACTION` 0.1) and S25/T25/B25 (0.25) on
  `{biodeg,kc2,vehicle,spambase}_probit_60nan` and `_60nan`, seeds 42/7/13 × 5 folds (144 cells); the full-label
  arms are E124's and E131's cells. Validation and test keep every label (a shared oracle selector, stated).
- **Measures and decision rule:** gate S(f) − B(f) macro-F1 pooled over the eight files, plain 95% t interval, Holm
  over the two fractions (open if Holm p < 0.05 with a positive mean); past the gate, S(f) − T(f) on the MNAR files
  with C1's verdicts (±0.010 band), MCAR second, interaction beside; S(f) − S(1.0) descriptive.
- **Sources:** [c1-split/08](tickets/c1-split/08-label-scarce.md).

### E137 · TabPFN v2 under informative missingness: native and explicit-mask arms beside A1 (2026-11-04)

- **Status:** done 2026-11-04 (69 of 69 pairs; read 23:21 GMT-3)
- **Results:** balanced, arm − A1 macro-F1: native +0.028 on MCAR controls, +0.036 on probit MNAR, +0.048 on the probit
  scales, −0.003 [−0.009, +0.003] on outcome-dependent MNAR; mask within 0.003 of native everywhere (mask − native
  +0.0003 [−0.002, +0.003]). Interaction MNAR − MCAR: native +0.0076 [−0.0014, +0.0166] (Holm 0.18), mask +0.0038
  [−0.0034, +0.0110] (Holm 0.27): no verdict for both.
- **Conclusion:** TabPFN's lead is robust to value-dependent MNAR (it leans larger, not smaller) and the explicit
  mask adds nothing; the lead disappears only where the gap carries the label directly (outcome-dependent F3, A1 ahead
  on kc2), the regime where H2 could be tested with a sibling design. ADR 0032 unchanged. T44 done.
- **Hypothesis:** none directional (T44): TabPFN's MCAR lead over TRIDENT (E83, E92) may shrink, hold or grow when
  missingness carries the label, which no published study tests; an explicit 0/1 mask may recover what its
  MCAR-trained indicator leaves unused.
- **Scenarios:** two offline arms (native NaNs; mask columns appended), balanced probabilities primary and plain
  beside, one fit a fold, context = training split, on E124's 69 A1 pairs (F0–F3 × seeds 42/7/13, 5 folds).
- **Measures and decision rule:** interaction (arm − A1) MNAR − MCAR macro-F1 over the seven probit/MCAR sibling
  pairs, pooled per replicate, plain 95% t interval, Holm over the two arms: shrinks / grows / holds (±0.01) / no
  verdict; leads per group, F2/F3 per file and mask − native descriptive.
- **Sources:** [c15-mnar-prediction/08](tickets/c15-mnar-prediction/08-tabpfn-arm.md).

### E138 · The quarter-label pre-training channel on fresh seeds, validation scarce too (2026-11-04)

- **Status:** done 2026-11-05 (48 of 48 cells, 0 failed; read 00:29 GMT-3)
- **Results:** S25V − B25V pooled +0.0218 [+0.0090, +0.0346] (NB [−0.0077, +0.0512]), p 0.0026 → HOLDS; better on
  spambase ×2 and vehicle_60nan, worse on none; MNAR half +0.018, MCAR half +0.026; log loss favours S on 8 of 8; row
  counts identical between arms.
- **Conclusion:** the quarter-label channel is a deployable result on the four 60nan tables; stage 2 (E141, 21
  variants) follows by the rule.
- **Hypothesis:** E136's S25 − B25 +0.016 holds when the kept epoch is chosen on a quarter of the validation labels
  and the seeds are fresh (1515/2626/3737).
- **Scenarios:** S25V vs B25V (`FINETUNE_LABEL_FRACTION` 0.25, `FINETUNE_SCARCE_VALIDATION` on, new, default off) on
  the eight C1 files, 3 seeds × 5 folds, 48 cells on medusa1.
- **Measures and decision rule:** S25V − B25V macro-F1 pooled over the eight files, plain 95% t interval: holds if
  above 0, else does not hold; halves and log loss flags; rows audited. Holds → stage 2 proposal (21 variants, 630
  cells) for the user's go.
- **Sources:** [c1-split/09](tickets/c1-split/09-scarce-confirmation.md).

### E139 · H2 under outcome-dependent MNAR: F3 against same-seed MCAR siblings, TabPFN beside A1 (2026-11-04)

- **Status:** done 2026-11-05 (9 A1 cells, 9 TabPFN pairs; read 00:34 GMT-3)
- **Results:** interaction (arm − A1) F3 − sibling, balanced: native −0.0451 [−0.0555, −0.0347], mask −0.0364
  [−0.0472, −0.0256], Holm < 0.0001, all twelve per-file intervals below zero → the lead SHRINKS for both arms. On the
  siblings TabPFN leads by +0.03 to +0.05; on the outcome files it is within ±0.013 and behind A1 on kc2. Plain
  probabilities: native shrinks (−0.020), mask no verdict. Both models gain from outcome-dependent gaps; A1 gains more.
- **Conclusion:** the first positive H2 reading: TRIDENT's per-column `[NULL]` gains ground where the gap carries the
  label, and an explicit mask does not give TabPFN the same. E142 (fresh draw s4848, confirmation seeds) runs now.
- **Hypothesis:** TabPFN's lead over A1 shrinks where the gap carries the label (E137's F3 reading, −0.003, made
  paired by MCAR siblings of the same seed).
- **Scenarios:** new files `{biodeg,kc2,spambase}_mcar_s2060_60nan`; A1 on them (9 cells, seeds 42/7/13); TabPFN native
  and mask arms offline on the 9 pairs; F3's cells reused.
- **Measures and decision rule:** interaction (arm − A1) F3 − sibling macro-F1 pooled over the six F3 files, plain 95%
  t interval, Holm over the two arms: shrinks / grows / holds (±0.01) / no verdict; each model's own F3 − sibling
  descriptive.
- **Sources:** [c15-mnar-prediction/09](tickets/c15-mnar-prediction/09-h2-outcome-siblings.md).

### E140 · The label-fraction curve under a scarce validation split: a tenth and a half (2026-11-05)

- **Status:** done 2026-11-05 (96 of 96 cells, 0 failed; read 01:25 GMT-3)
- **Results:** 10% +0.0290 [+0.0165, +0.0416] (Holm 0.0004, OPEN; kc2_60nan and vehicle_60nan better, none worse);
  50% +0.0157 [+0.0075, +0.0239] (Holm 0.0010, OPEN; no file better or worse); 25% from E138 +0.022; 100% from E131
  +0.004 closed.
- **Conclusion:** the curve is monotone: pre-training is worth more the scarcer the labels (+0.03 at a tenth, +0.004 at
  every label). A tenth is open under a scarce selector where E136's oracle selector read it closed (hypothesis: B's
  epoch selection suffers more from few validation rows). Descriptive for the thesis; no ADR.
- **Hypothesis:** the channel E138 confirmed at a quarter opens somewhere between a tenth (closed in E136) and a half.
- **Scenarios:** S10V/B10V and S50V/B50V on the eight C1 files, seeds 1515/2626/3737 × 5 folds, 96 cells.
- **Measures and decision rule:** S(f)V − B(f)V pooled over the eight files, plain 95% t interval, Holm over the two
  fractions: open / closed; E138's quarter beside.
- **Sources:** [c1-split/10](tickets/c1-split/10-label-fraction-curve.md).

### E141 · Stage 2: the quarter-label channel on the 21 confirmatory variants, five seeds (2026-11-05)

- **Status:** done 2026-11-05 (210 of 210 cells, 0 failed; read 02:43 GMT-3)
- **Results:** pooled over 21: +0.0101 [+0.0045, +0.0157] (NB [−0.0060, +0.0262]) → HOLDS; 8 variants better
  (spambase ×4, pendigits, letter, electricity, kr-vs-kp_40nan), none worse. Groups: categorical +0.009, small numeric
  +0.008 [−0.004, +0.020], spambase +0.012, large +0.016 [+0.013, +0.019].
- **Conclusion:** the quarter-label channel holds on the confirmatory set: smaller than on the four 60nan tables
  (+0.010 against +0.022, the 21 being mostly 20–40nan and large), tightest on the large tables, NB interval crossing 0
  (caveat for the sentence). Thesis-claim proposal and optional ADR for the label-scarce path put to the user; no
  default change.
- **Hypothesis:** S25V − B25V > 0 pooled over E30's 21 variants (E138 held on the four 60nan tables).
- **Scenarios:** 21 variants × seeds 42/7/13/101/202 × S25V/B25V = 210 cells (gorgona8 seed 42, medusa1 the rest).
- **Measures and decision rule:** pooled over the 21 per replicate (25 replicates), plain 95% t interval: holds if
  above 0; per-variant flags, four groups descriptive. Holds → thesis claim proposal; no default change.
- **Sources:** [c1-split/11](tickets/c1-split/11-stage2-21-variants.md).

### E142 · H2's fresh-draw confirmation (conditional on E139 reading "shrinks") (2026-11-05)

- **Status:** done 2026-11-05 (18 A1 cells, 18 TabPFN pairs; read 01:24 GMT-3)
- **Results:** interaction (arm − A1) odds-3 − sibling, balanced: native −0.0394 [−0.0513, −0.0275], mask −0.0278
  [−0.0398, −0.0159], Holm ≤ 0.0002 → SHRINKS for both; plain also shrinks (−0.027, −0.022). A1 gains +0.15 to
  +0.27 from the label-bearing gaps, TabPFN +0.12 to +0.24.
- **Conclusion:** E139 confirmed on a fresh draw and fresh seeds: TRIDENT's `[NULL]` gains 0.03–0.05 macro-F1 of ground
  on TabPFN v2 under outcome-dependent MNAR relative to MCAR; the explicit mask does not close it. Thesis sentence the
  user's call; ADR 0032 unchanged.
- **Scenarios:** A1 on `{biodeg,kc2,spambase}_outcome3p0_s4848_60nan` and `_mcar_s4848_60nan`, seeds 1515/2626/3737 (18
  cells); TabPFN arms offline on the same pairs.
- **Measures and decision rule:** E139's interaction rule on the three odds-3 files, Holm over the two arms.
- **Sources:** [c15-mnar-prediction/10](tickets/c15-mnar-prediction/10-h2-fresh-draws.md).

### E143 · The quarter-label channel at 20, 40 and 80% missingness (2026-11-05)

- **Status:** done 2026-11-05 (72 of 72 cells, 0 failed; read 02:43 GMT-3)
- **Results:** 20nan +0.0246 [+0.0048, +0.0445] (Holm 0.039, OPEN); 40nan +0.0234 [+0.0057, +0.0410] (Holm 0.039,
  OPEN; kc2_40nan better); 80nan +0.0040 [−0.0164, +0.0245] (Holm 0.68, closed; spambase_80nan better); E138's 60nan
  MCAR half beside +0.0258 [+0.0083, +0.0432]. No table worse at any level.
- **Conclusion:** the channel is flat at +0.023…+0.026 from 20 to 60% missingness and closes at 80%, where only
  spambase still gains; it does not grow with the missingness level. Descriptive for the thesis; no ADR.
- **Hypothesis:** none directional; the channel's size may grow or shrink with the missingness level.
- **Scenarios:** `{biodeg,kc2,vehicle,spambase}_{20,40,80}nan` × seeds 1515/2626/3737 × S25V/B25V = 72 cells.
- **Measures and decision rule:** per level pooled over the four tables, plain 95% t interval, Holm over three levels:
  open / closed; E138's 60nan MCAR half beside.
- **Sources:** [c1-split/12](tickets/c1-split/12-missingness-levels.md).

### E144 · The selector × seeds squares at a tenth and a quarter of the labels (2026-11-05)

- **Status:** done 2026-11-05 (168 of 168 cells, 0 failed; read 09:40 GMT-3)
- **Results:** selector effect 10% −0.0056 [−0.0194, +0.0081] (Holm 0.57), 25% −0.0056 [−0.0161, +0.0050] (Holm 0.57)
  → not supported at both; 16 of 16 file lines no difference. Over six seeds S − B is +0.017 scarce / +0.023 oracle at
  a tenth, +0.014 / +0.020 at a quarter.
- **Conclusion:** the scarce selector does not widen S − B; E140's hypothesis is withdrawn and the E136 / E140
  difference at a tenth is the seeds (kc2's 36 labelled rows swing ±0.1 by seed). The deployable readings
  E138/E140/E141/E143 need no selector caveat. No ADR.
- **Hypothesis:** the scarce-validation selector widens S − B relative to the whole-validation selector (E140's
  reading of why a tenth opened where E136 read it closed).
- **Scenarios:** cut 13: S10/B10 × seeds 1515/2626/3737 and S10V/B10V × seeds 42/7/13 on the eight C1 files (96 cells);
  cut 14: S25/B25 × 1515/2626/3737 and S25V/B25V on the four probit files × 42/7/13 (72 cells; MCAR reused from
  E141). Pairs sit with their earlier arms' machine (two reused MCAR pairs cross machines, stated).
- **Measures and decision rule:** per fraction (S(f)V − B(f)V) − (S(f) − B(f)) paired per (file, seed, fold), pooled
  over the eight files (30 replicates), plain 95% t interval, Holm over the two fractions: supported / not supported.
  E136's and E140's verdicts stand as written. No ADR.
- **Amendment 2026-11-05 08:15 GMT-3:** medusa1's GPUs held by another user's job since ~02:40; its 92 cells moved to
  gorgona8 on the user's decision. Within-pair differences stay on one machine; the second-order contrast crosses
  machines for every medusa1-sibling pair (stated threat).
- **Sources:** [c1-split/13](tickets/c1-split/13-selector-squares.md).

### E145 · Choosing the decode checkpoint by a distributional score (2026-11-05)

- **Status:** done 2026-11-05 (72 of 72 cells on medusa1, 0 failed; read 20:22 GMT-3)
- **Results:** B − A pooled over the 12 variants: `dist_score` −0.1875 [−0.1941, −0.1808] (better on 12 of 12; Wasserstein
  −0.096, total variation −0.107, dispersion +0.148 toward 1); `impute_score` +0.0397 [+0.0374, +0.0421], worse on 7
  (kr-vs-kp and spambase at 60/80nan, credit-g at 40/60/80nan; up to +0.162), no different on 5 → improves the
  distribution at a cost in point accuracy.
- **Conclusion:** the selector does what it is asked, at four times the margin in point accuracy. At 80nan
  `impute_score` keeps an epoch near the start (10 to 56 of 450) whose fills are almost the mean/mode fill (credit-g:
  dispersion 0.07, `impute_score` 1.007), and `dist_score` a late one whose fills spread like the truths but lose to
  the mean/mode fill cell by cell (about 1.17): ADR 0025's caveat on the 80nan wins, measured. A trade-off, recorded;
  the flag stays off by default, no ADR. Open: a criterion between the two, KC, the large tables, fresh seeds.
- **Hypothesis:** keeping the decode epoch whose validation gaps score best by `dist_score` (per-column Wasserstein
  distance and category total variation, as a ratio to the mean/mode fill) improves the test fills' distribution
  without a relevant cost in `impute_score`.
- **Scenarios:** E40's 12 development variants at 40nan+ × seeds 42/7/13 × arms A (`impute_score`, today's K) and B
  (`--decode_metric dist_score`), `--decode_calibration none` in both, 72 cells (gorgona8 seed 42, medusa1 seeds 7/13), MLflow off; K's induced
  fills from the five fold ledgers.
- **Measures and decision rule:** B − A pooled over the 12 variants per replicate (15), plain 95% t interval:
  supported if `dist_score` is below 0 and `impute_score`'s upper bound is below +0.01; "at a cost" if only the
  first; else not supported. No default change, no ADR; ADR 0025 stands (`score_cells` untouched).
- **Amendment 2026-11-05 19:29 GMT-3:** gorgona8's queues stopped on the user's decision (30 minutes a cell there);
  all 72 cells on medusa1, gorgona8's three finished arm-A cells discarded and their pairs trained whole on medusa1.
- **Sources:** [imputation-protocol/51](tickets/imputation-protocol/51-decode-metric.md).

### E146 · Which fills serve a downstream classifier: the checkpoint by `impute_score` or by `dist_score`? (pilot, 2026-11-05)

- **Status:** done 2026-11-05 (36 of 36 files; read 21:12 GMT-3); a pilot, pre-registered and read on a local branch
  on the user's instruction and kept on the record on their decision of 2026-11-06
- **Results:** classifier fitted on complete rows: F1(B) − F1(A) +0.0141 [+0.0107, +0.0175], B better on 5 variants,
  A on none; `dist_score` on the right side on 5 of 5 verdicts, `impute_score` on 0; Spearman +0.42 against −0.23.
  Fitted on each fill's own table: +0.0029 [−0.0011, +0.0069], no difference.
- **Conclusion:** the checkpoint chosen by `dist_score` gives fills a classifier trained on real rows reads better,
  most where `impute_score` charged the most; under impute-then-learn the two are not separable. Exploratory; a claim
  needs a pre-registered rerun on fresh seeds.
- **Hypothesis:** none directional; the vault's review expects imputation quality to couple weakly to downstream
  accuracy.
- **Scenarios:** E145's 12 variants × seeds 42/7/13 × 5 folds, K's fills under both selectors; logistic regression and
  a small MLP fitted on the truth-filled rows outside the test fold, scored on the test fold filled with the truth,
  arm A's fills, arm B's and the mean/mode; an impute-then-learn protocol beside. CPU on medusa1, no store write.
- **Measures and decision rule:** F1(B) − F1(A), mean of the two learners, pooled over the 12 variants per replicate,
  plain 95% t interval; per variant, which score favoured the more useful arm; replicate-level Spearman correlations.
- **Sources:** [imputation-protocol/52](tickets/imputation-protocol/52-downstream-utility.md).

### E147 · Which score orders imputers as a classifier does, across TRIDENT, the strong baselines and three mechanisms? (pilot, 2026-11-05 – 2026-11-06)

- **Status:** done 2026-11-06 06:27 GMT-3, mixed; a pilot, pre-registered, run and read as E149 on a local branch, kept on
  the record and renumbered on the user's decision of 2026-11-06 (E148 and E149 went to T50's experiments the same
  day, so "E149" in this pilot's tallies and logs is this entry)
- **Hypothesis:** from E146's sign, `dist_score` orders imputers more like a downstream classifier than `impute_score`.
- **Scenarios:** MCAR (E145's 12 variants), MAR (`marf_60nan`, six tables) and MNAR (`probit_60nan`, four tables),
  seeds 42/7/13 × 5 folds; 60 new TRIDENT cells for MAR and MNAR under E145's two arms; per fold TRIDENT A and B, the
  mean/mode, knn5, knn10, hgb, lgbm, missForest, condexp, SoftImpute, ICE, the sequential missForest and TabPFN v2,
  each with `impute_score`, `dist_score` and the macro-F1 of logistic regression and an MLP fitted on complete rows.
- **Measures and decision rule:** per mechanism, rho(`dist_score`) − rho(`impute_score`), each the within-fold
  Spearman correlation with the macro-F1 across the imputers, pooled over the variants per replicate, plain 95% t
  interval, Holm over the mechanisms read complete. Secondary: the F1 lost by each score's best imputer; TRIDENT
  against the best strong baseline on `dist_score`; B against A a mechanism.
- **Amendment 2026-11-05 22:39 GMT-3:** on the user's decision the 60 MAR/MNAR cells and everything after them run
  on gorgona8 (four queues), without waiting for medusa1's GPUs; the MCAR CPU imputers had already finished on medusa1.
- **Results:** complete, 60 of 60 cells and 396 of 396 files, none failed; twelve imputers a fold, not thirteen
  (the sequential missForest was listed but no group fitted it, my error). Primary, rho(`dist_score`) −
  rho(`impute_score`): MCAR +0.1543 [+0.0983, +0.2103], Holm 0.0001, `dist_score`; MAR −0.1195 [−0.1830, −0.0559],
  Holm 0.0012, `impute_score`; MNAR −0.2206 [−0.2538, −0.1873], Holm 0.0000, `impute_score`; beside, MCAR at 60nan
  +0.0475 [−0.0208, +0.1157], no difference. Taking each score's best imputer, `impute_score` loses less F1 in every
  mechanism (0.030 against 0.042, 0.016 against 0.025, 0.014 against 0.032). B against A on F1: MCAR +0.0141, MNAR
  +0.0050, MAR +0.0020 (no difference). Against the best strong baseline of each variant on F1, A trails in all
  three mechanisms (−0.015, −0.007, −0.011) and B ties under MCAR (−0.001) and trails under MAR (−0.005) and MNAR
  (−0.006); on `dist_score` B leads under MCAR only (+0.054). Post hoc, descriptive: inside MCAR the primary is −0.110
  at 40nan, +0.048 at 60nan and +0.632 at 80nan, where `impute_score`'s correlation with the F1 is negative (−0.18).
- **Conclusion:** mixed; the hypothesis is not supported as a general statement. The MCAR verdict is a matter of the
  level, not the mechanism: at 60nan there is no difference under MCAR and `impute_score` orders the imputers better
  under MAR and MNAR. `dist_score`'s ordering is steadier (+0.45 to +0.53 everywhere) and `impute_score`'s is higher
  at moderate levels and breaks at 80nan, the regime of E145's near-naive fills. Nothing argues for replacing
  `impute_score` as the reported score (ADR 0025). MAR and MNAR at one level, one seed set. Whether the pilots are
  kept is the user's verdict.
- **Sources:** [imputation-protocol/53](tickets/imputation-protocol/53-imputer-utility.md).

### E148 · Observed cells hidden on MIMIC-IV tables with natural gaps (pre-registered 2026-11-05; 2026-11-06)

- **Status:** done 2026-11-06 10:27 GMT-3, supported on both tables; ran 2026-11-06 07:20 to 10:14 GMT-3; pre-registered
  2026-11-05 as E150 (renumbered 2026-11-06 to
  keep the ids contiguous, the user's decision)
- **Hypothesis:** on a clinical table whose gaps are natural, the decoder's fills of hidden observed cells (the
  masked population; the natural gaps stay `[NULL]` inputs) are better than the best baseline of the bar's.
- **Scenarios:** `mimic-cardio_natural` and `mimic-sofa_natural` (the kg-impute cohort, 11,082 ICU stays; about a
  fifth of the cardio table's measured cells missing by the probe) × seeds 42/7/13, five folds, imputation defaults
  with `--decode_checkpoint masked` (ADR 0015's criterion needs a sibling), 6 runs.
- **Measures and decision rule:** per table, the masked population's `impute_score` minus the best baseline's,
  paired per (seed, fold), 15 replicates, plain 95% t interval, Holm over the two tables: better, worse or no
  different. The hidden cells are draws among measured values; the true gaps are not scored.
- **Amendment 2026-11-06 07:20 GMT-3:** the cells share gorgona8's GPU with kg-impute's training queue (the user:
  "Comece a rodar em paralelo com o kg impute") instead of waiting for a free one; the design is unchanged. They carry
  `--export_gap_fills`, the fills E150 reads, which changes no score. Cost from the store: about 40 minutes a run
  on `pendigits`, a table of this size; one to two hours a cell expected beside the other trainers. Measured on the
  first two cells: 33 and 36 minutes.
- **Results:** complete, 6 of 6 runs, none failed. Decoder minus the bar's best baseline (`hgb` in every run), 15
  replicates: `mimic-cardio_natural` 0.8072 against 0.8199, −0.0127 [−0.0177, −0.0076], Holm p 0.0002, better;
  `mimic-sofa_natural` 0.8848 against 0.8984, −0.0137 [−0.0202, −0.0072], Holm p 0.0005, better. Descriptive: the
  decoder has the lowest mean score of every imputer the runs scored, the strong tier's four included (nearest: ICE
  0.8191 on the cardio table, `hgb` on the SOFA one); on the eight sparse columns it has the lowest RMSE of the
  bar's on seven (not PaO2/FiO2: `lgbm` 0.797, decoder 0.820), and ICE is lower on three of the cardio table's five;
  troponin is near the mean fill for every imputer (best 1.007 z). By a row's count of natural gaps (a form chosen
  after the primary: numerical cells, RMSE over the training-mean fill's) the ratio shows no trend, 0.76 to 0.85 and
  0.87 to 0.89. TabPFN (offline pass, same cells; added 2026-11-06 after it ended at 11:41 GMT-3): the decoder's
  score is lower on both, −0.0387 [−0.0443, −0.0331] and −0.0108 [−0.0165, −0.0052] (TabPFN 0.8459 and 0.8956); the
  lead is in RMSE (0.812 against 0.852, 0.886 against 0.900), not in MAE (0.451 both; 0.493 against TabPFN's 0.467).
- **Conclusion:** supported on both tables, by about 1.5% of the bar's score. The first reading of the decoder on
  clinical data with natural gaps left in the input. It says nothing of the true gaps: the hidden cells are a draw
  among measured values, and E150 is the only reading of what is done with the unmeasured ones. One seed set, two
  views of one cohort; no default change, no ADR.
- **Sources:** [mimic/03](tickets/mimic/03-masked-cells-natural-gaps.md); tables in
  [mimic/02](tickets/mimic/02-tables.md); runner `scripts/experiments/mimic_run.py`, report `mimic_report.py e148`;
  MLflow tag
  `experiment = mimic-natural-gaps-2026-11-06`; state `/scratch2/diogoneiss/trident-2026-11-06-mimic-e148`.

### E149 · Real gap patterns transplanted onto complete rows (pre-registered 2026-11-05)

- **Status:** pre-registered 2026-11-05 as E151 (renumbered 2026-11-06 to keep the ids contiguous, the user's
  decision); the variants were cut on 2026-11-06 07:27 GMT-3 (30 and 29 distinct row patterns, 12.9% and 10.5% of
  feature cells, named `_20nan`); started 2026-11-06 10:15 GMT-3 by `run_mimic_chain.sh` when E148's queues ended;
  **done 2026-11-06 15:14 GMT-3, supported on both donor tables**
- **Hypothesis:** under gap patterns copied from real ICU stays onto rows that have every value, the decoder's fills
  are better than the best baseline of the bar's.
- **Scenarios:** donor tables `mimic-cardio-donor` (1,292 complete rows) and `mimic-sofa-donor` (2,143), run only
  at 1,000 rows or more × variants `tmatch` (the pattern of the nearest incomplete stay), `trand`
  (the same patterns permuted) and `mcar` (the same count per column, uniform) × seeds 42/7/13, five folds,
  imputation defaults, 18 runs.
- **Measures and decision rule:** per donor table, on `tmatch`, the induced cells' `impute_score` minus the best
  baseline's, paired per (seed, fold), 15 replicates, plain 95% t interval, Holm over the two tables. Descriptive:
  `tmatch − trand`, `trand − mcar`, TabPFN on `tmatch`. Threats stated: the donors are the sicker patients, and a
  transplant carries where gaps fall, not the missing values.
- **Amendment 2026-11-06 10:26 GMT-3:** one cell (`mimic-cardio-donor_tmatch_20nan`, seed 42) failed at its start,
  before any training: both queues asked the tracking server to create the table's experiment in the same second.
  The queues are relaunched when the first pass ends and run it again; nothing in the design changes.
- **Results:** complete, 18 of 18 runs (17 on the first attempt, the failed cell on its second, 14:37 to 15:06
  GMT-3). Decoder minus the bar's best baseline on `tmatch`'s induced cells, 15 replicates: `mimic-cardio-donor`
  0.8606 against 0.9061, −0.0455 [−0.0604, −0.0307], Holm p below 0.0001, better; `mimic-sofa-donor` 0.9209 against
  0.9481, −0.0272 [−0.0423, −0.0122], Holm p 0.0017, better. Descriptive: on `tmatch` the decoder has the lowest mean
  of every imputer the runs scored (nearest: the strong tier's `condexp`, 0.8616 on the cardio table, not told
  apart by levels, and 0.9305 on the SOFA one); on `mimic-sofa-donor_mcar` missForest's mean is lower (0.9151
  against 0.9237). `trand − mcar`: the decoder's score is higher under the real blocks on the cardio table (+0.0660
  [+0.0465, +0.0854]) and its gap to the bar is not (−0.0161 [−0.0374, +0.0052]); both hold zero on the SOFA table.
  `tmatch − trand`: every interval holds zero, with opposite signs for the decoder's score on the two tables.
  TabPFN on `tmatch` (offline pass, same cells; added 2026-11-06 15:44 GMT-3): the two tables disagree, the decoder
  lower on the cardio table (−0.0600 [−0.0698, −0.0503], TabPFN 0.9206) and TabPFN lower on the SOFA one (+0.0265
  [+0.0155, +0.0375], TabPFN 0.8944), in RMSE and in MAE alike. Cost: 19 to 33 minutes for 16
  cells and 57 and 59 for two, against an estimate of 5 to 15; 4 h 51 min of wall time.
- **Conclusion:** supported on both donor tables, by about 5% and 3% of the bar's score: under gap patterns copied
  from real stays, with a ground truth, the decoder is better than the bar's best baseline. The stated threats
  stand (sicker donors; a transplant carries where gaps fall, not the missing values), and the two descriptive
  contrasts support no reading of a mechanism. Against TabPFN the decoder leads on one donor table and trails on
  the other, so the claim is against the bar alone. One seed set, two views of one cohort; no default change, no
  ADR.
- **Sources:** [mimic/04](tickets/mimic/04-pattern-transplant.md); runner `mimic_transplant_run.py`, report
  `mimic_report.py e149`; MLflow tag `experiment = mimic-pattern-transplant-2026-11-06`; state
  `/scratch2/diogoneiss/trident-2026-11-06-mimic-e149`.

### E150 · Fills and gap patterns judged by a mortality classifier (pre-registered 2026-11-05; 2026-11-06)

- **Status:** pre-registered 2026-11-05 as E152 (renumbered 2026-11-06 to keep the ids contiguous, the user's
  decision); done 2026-11-06 10:27 GMT-3: (i) not supported, (ii) holds on the SOFA table only
- **Hypothesis:** (i) a logistic regression fed the decoder's fills of the natural gaps predicts death in hospital
  better than one fed the mean/mode fill; (ii) the mean/mode fill with one indicator per column does better than
  without them (the pattern carries signal: deaths 19.9% with a lactate measured, 7.9% without, by the probe).
- **Scenarios:** the two natural tables × seeds 42/7/13 × five folds × inputs `N`, `NI`, `T`, `TI`; one fixed
  logistic regression; fills from E148's runs, fold k from fold k's decoder alone; no new training.
- **Measures and decision rule:** `T − N` and `NI − N` in macro-F1 per table, paired per (seed, fold), 15
  replicates, Holm over the four tests; each holds with Holm p < 0.05 and a mean above 0. Descriptive: `TI − NI`,
  AUROC, AUPRC, and as levels TRIDENT's classifier and TabPFN on the natural table.
- **Results:** complete, 120 fits from E148's six runs. Macro-F1 levels `N`, `NI`, `T`, `TI`: cardio 0.6566, 0.6581,
  0.6580, 0.6588; SOFA 0.6629, 0.6716, 0.6649, 0.6701. `T − N`: +0.0014 [−0.0012, +0.0040] and +0.0020 [−0.0006,
  +0.0046], Holm p 0.35 both, does not hold. `NI − N`: cardio +0.0014 [−0.0005, +0.0034], Holm p 0.35, does not
  hold; SOFA +0.0087 [+0.0047, +0.0126], Holm p 0.0012, holds. Descriptive: `TI − NI` +0.0008 [−0.0007, +0.0023]
  and −0.0014 [−0.0032, +0.0003]; AUROC 0.802 to 0.812 (cardio) and 0.815 to 0.825 (SOFA), ordered `N`, `T`, `NI`,
  `TI` on both. The two levels (TRIDENT's classifier, TabPFN) were not run: new GPU work beside E149.
- **Conclusion:** the decoder's fills of the natural gaps do not help a logistic regression predict death over the
  mean/mode fill, and add nothing once the pattern is given; the pattern itself helps on the SOFA table only, by
  0.009 of macro-F1. Every effect is under 0.01. One linear reader, one seed set; a better or equal classifier says
  nothing of how close the fills are to the unmeasured values. No default change, no ADR.
- **Sources:** [mimic/05](tickets/mimic/05-downstream-mortality.md); reader `scripts/experiments/mimic_mortality.py`;
  per fold in `/scratch2/diogoneiss/trident-2026-11-06-mimic-e148/mortality.csv`.

### E151 · Intervals calibrated per bin of a row's gaps (pre-registered 2026-11-06)

- **Status:** done 2026-11-06 13:05 GMT-3: Q1 supported, Q2 refuted. Pre-registered 2026-11-06 on the user's go ("vou na sua sugestão, vá"). T43. Offline on
  E75's, E76's and E84's fold ledgers; no training. Disclosure: E77's and E80's results are known and motivate it.
- **Hypothesis:** (Q1) E77's per-column constant 90% interval under-covers the rows that miss the most, and
  calibrating per column and per tertile of the row's gap count (Mondrian conformal, Zaffran et al. 2023) fixes it;
  (Q2) inside a bin, where E80's confound is held fixed, the re-masking spread adapts a width better than a constant.
- **Scenarios:** primary E75's 15 confirmation variants (MCAR, 707/808/909, five folds), numerical gaps, K's fill
  with K's spread; E76's 17 development variants with numerical gaps by the same rules, reported beside; E84's MNAR
  and MAR variants and H descriptive. Four arms cross-fitted on the same seed's other folds at the split-conformal
  level: C and S per column (E77's constant and spread-scaled), MC and MS per column and bin (groups under 30 fit
  cells fall back to the column's).
- **Measures and decision rule:** Q1, per fold, worst-bin under-coverage u(MC) − u(C), SESOI 0.01; Q2, per fold,
  (width MS − width MC) / width MC, SESOI 0.01; lower is better; per variant the project's verdict rule over 15 fold
  differences. Each question: supported if better on 10 or more of 15 and worse on at most 1, refuted if better on 5
  or fewer or worse on 4 or more, mixed otherwise. Oracle calibration (other test folds' gaps), as E77.
- **Results:** Q1 better on 11 of 15 under K, worse on none (H 11); Q2 better on 0, worse on 15 (H worse on 13).
  Development: Q1 7 of 17 better, worse on credit-g_80nan (+0.007), mixed; Q2 0 better, 15 worse. The per-column
  constant covers 0.81 to 0.87 in the top tertile of the row's gaps on pendigits, letter, electricity and vehicle
  (0.92 to 0.94 in the bottom one); binned, 0.900 to 0.905. On biodeg, kc2, spambase and credit-g under MCAR it already
  covers (0.88 to 0.91). E84, descriptive: better on all 7 self-masking variants (kc2_probit_60nan 0.776 against
  0.907), C already covers under MAR. Post hoc width: binning costs −0.022 to +0.001 on the large tables, 1 to 12% on
  the small. Inside a bin the spread interval is still wider than a constant everywhere (+0.06 to +1.57 at 40nan).
- **Conclusion:** a 90% interval around the fill, calibrated per column and per tertile of the row's gaps, restores
  coverage on the rows that miss the most, at no width cost on the large tables; the spread still does not adapt a
  width. Oracle calibration as E77. ADR 0030 stays a rank for the spread. **ADR 0030 amended 2026-11-06 16:55
  GMT-3 on the user's decision (decision 8): the binned interval, without the spread, is the fill's interval.**
- **Sources:** [imputation-protocol/54](tickets/imputation-protocol/54-conformal-by-row-gaps.md); reader
  `scripts/experiments/conformal_row_gaps.py`.

### E152 · C15's positive-control gate on the SOFA table's natural gaps (pre-registered 2026-11-06)

- **Status:** done 2026-11-06 16:34 GMT-3: G1 fails (condition 2 holds, condition 1 does not). Pre-registered
  2026-11-06 on the user's go ("vou na sua sugestão, vá"); ran on gorgona8 after E149's cells, 9 of 9 cells. T50, and
  RQ2 after T45. Disclosure: E150's result and the cohort's death rates by gap are known.
- **Hypothesis:** on `mimic-sofa_natural`, where E150 found the gap pattern informative for death in hospital, the
  pipeline sees it: the indicator arm beats the plain fill, and the `[NULL]` arm reads the bilirubin gap with the
  cohort's sign.
- **Scenarios:** `mimic-sofa-num_natural` (the three categorical columns made numerical, as the C15 arms need); C15's
  A1 (`[NULL]`), A2 (missForest fill + indicator), A3 (missForest fill) × seeds 42/7/13 × five folds, 9 cells; the
  table's classification defaults with `embedding_normalized`, MLflow off.
- **Measures and decision rule:** E128's G1 on one table. Passes if both hold over 15 cells with Nadeau–Bengio
  corrected 95% t intervals: (1) (LL_A2 − LL_A3) / LL_A3 below 0; (2) A1's direct contrast on `bilirubin_max` (`[NULL]`
  against the mean over 20 MICE draws) below 0. Passing leads to H2's A1-against-A2 test on fresh seeds, for the
  user's go; failing stops RQ2 at the gate on natural gaps too.
- **Sources:** [mimic/06](tickets/mimic/06-sofa-positive-control.md); `datasets/make_mimic_numeric.py`,
  `scripts/experiments/sofa_g1.py`.
- **Results:** (1) A2 − A3 relative test log loss −0.0153, corrected 95% interval [−0.1066, +0.0760] (plain [−0.0550,
  +0.0244]); per seed s42 +0.0311, s7 −0.0399, s13 −0.0371. (2) A1's `bilirubin_max` contrast −0.651 logit, corrected
  [−0.851, −0.452]; per seed −0.644, −0.695, −0.615. Macro-F1, descriptive: A1 0.6568, A2 0.6684, A3 0.6664.
- **Conclusion:** G1 fails, with E128's split: the `[NULL]` arm reads the gap with the cohort's sign, and the
  indicator arm is not shown to beat the plain fill on log loss. By the rule RQ2 stops at the gate on natural gaps too;
  H2's A1-against-A2 test is not pre-registered.

### E153 · ADR 0030's binned interval under real gap patterns (pre-registered 2026-11-06)

- **Status:** done 2026-11-06 17:13 GMT-3: refuted, one width per column already covers. Pre-registered 2026-11-06
  on the user's go ("vamos no 1, faça pull antes"); offline. T43
  after ADR 0030's decision 8, on T50's E149 ledgers. Disclosure: E149's scores and E151's results are known; one
  ledger per donor table was opened to see what it holds, no coverage computed.
- **Hypothesis:** under gap patterns copied from real ICU stays, one width per column under-covers the rows with the
  most gaps, and decision 8's per-tertile width fixes it.
- **Scenarios:** E149's 18 runs (`mimic-cardio-donor`, `mimic-sofa-donor` × `tmatch`, `trand`, `mcar` × seeds
  42/7/13, five folds), the `induced` numerical cells; arms C (per column) and MC (per column and tertile of the row's
  gaps), cross-fitted as E151.
- **Measures and decision rule:** E151's Q1, u(MC) − u(C) per fold, SESOI 0.01, `report_common.verdict` per variant.
  On `tmatch`: supported if better on both tables, mixed if better on one and worse on neither, refuted otherwise.
  Q2 is not read (no `spread` in these ledgers). Supported leads to proposing decision 8's scope include real
  transplanted patterns, for the user's decision.
- **Sources:** [mimic/07](tickets/mimic/07-conformal-real-patterns.md); `scripts/experiments/conformal_real_patterns.py`.
- **Results:** on `tmatch`, u(MC) − u(C) cardio −0.0016 [−0.0075, +0.0044], SOFA +0.0009 [−0.0052, +0.0070], both
  equivalent. C's top tertile 0.900 and 0.909, u(C) 0.027 and 0.018. Width (MC − C) / C +0.030 and +0.006, worse.
  `trand` and `mcar` the same, none better or worse; fallback 0.01 to 0.07.
- **Conclusion:** refuted, on the "u(C) near 0" branch: on these tables near 20nan one width per column covers every
  tertile, and the tertiles only cost width. Proposed: decision 8 states it, for the user's decision; ADR 0030 not
  edited.

### E154 · ADR 0030's binned interval on the natural gaps' rows (pre-registered 2026-11-06)

- **Status:** done 2026-11-06 18:08 GMT-3: refuted, one width per column already covers. Pre-registered 2026-11-06 on the
  user's go ("ok va"); offline. T43 after ADR 0030's
  decision 8, on T50's E148 ledgers. Disclosure: E148's error by the row's natural gaps is known (highest in rows with
  none); E151 and E153 known.
- **Hypothesis:** on the cohort's own rows, whose natural gaps vary widely, one width per column under-covers some
  tertile of the row's gaps, and decision 8's per-tertile width fixes it.
- **Scenarios:** E148's 6 runs (`mimic-cardio_natural`, `mimic-sofa_natural` × seeds 42/7/13, five folds), the
  `masked` numerical cells; b = the row's natural gaps plus its masked cells; arms C and MC as E153.
- **Measures and decision rule:** E151's Q1, u(MC) − u(C) per fold, SESOI 0.01, `report_common.verdict` per table.
  Supported if better on both tables, mixed if better on one and worse on neither, refuted otherwise. Supported leads
  to proposing decision 8's scope include MIMIC-IV's natural gaps, for the user's decision.
- **Sources:** [mimic/08](tickets/mimic/08-conformal-natural-gaps.md); `scripts/experiments/conformal_natural_gaps.py`.
- **Results:** u(MC) − u(C) cardio +0.0006 [−0.0035, +0.0048], SOFA −0.0000 [−0.0036, +0.0036], both equivalent.
  u(C) 0.009 and 0.008; C's worst bin mostly the rows with the fewest gaps (9 and 11 of 15 folds), top tertile 0.898
  and 0.905. Width (MC − C) / C +0.012 and +0.013, worse.
- **Conclusion:** refuted, on the "u(C) near 0" branch. With E153: on MIMIC-IV one width per column already covers;
  proposed that decision 8 states it, for the user's decision; ADR 0030 not edited.

### E155 · H2 behind an external gate: A1 against A2 on fresh seeds (pre-registered 2026-11-06)

- **Status:** done 2026-11-06 21:48 GMT-3: inconclusive on both primaries, secondary *gains*. Pre-registered
  2026-11-06 on the user's go ("vamos no 1"). T51. **A deviation:** E128's
  binding stop (C15 ends if G1 fails again) and E152's leads-to are overridden; the gate moves outside the pipeline
  (E123's G2 for P, E150's logistic regression for the SOFA table, its null on cardio as the negative control).
  Disclosure: E124's Stage 1b (A1 inferior on 5 of 23 development variants, non-inferior on none) and E152's SOFA
  macro-F1 (A1 −0.012 against A2) lean against A1.
- **Hypothesis (H2):** under informative missingness, `[NULL]` tokens (A1) are non-inferior to missForest + indicator
  (A2) in the same encoder.
- **Scenarios:** A1 and A2 at seeds 8282/8383/9292 (new), five folds, E124's configuration. Synthetic: P's fresh draws
  (s4848: biodeg and vehicle probit 0.5, biodeg/kc2/spambase outcome odds 3) and four MCAR siblings, 54 cells. Natural:
  `mimic-sofa-num_natural`, `mimic-cardio-num_natural`, 12 cells. All on gorgona8.
- **Measures and decision rule:** margins −0.01 macro-F1, +2% relative log loss, Nadeau–Bengio corrected. Two
  primaries with Holm (97.5% first): (1) E126's pooled H2 over P, supported / reversed / inconclusive; (2) the SOFA
  table's per-variant call, non-inferior / inferior / inconclusive. Secondary: E139's interaction, (A1 − A2) informative
  minus MCAR sibling, gains / loses / holds / no verdict. Cardio descriptive. Read once with all 66 cells.
- **Sources:** [c15-mnar-prediction/11](tickets/c15-mnar-prediction/11-h2-external-gate.md); `C15_STAGE=5 c15_run.py`,
  `scripts/experiments/h2_natural.py`, `c15_report.py --h2`.
- **Results:** all 66 cells, read once 21:46 GMT-3. Primary 1, pooled over P: F1 −0.0269 [−0.0463, −0.0076]
  (97.5% corrected; plain [−0.0341, −0.0197]), LL ratio noise (spambase odds 3 +20.8); A1 inferior on vehicle probit
  (F1 −0.098, LL +22%), inconclusive on the other four. Primary 2, SOFA: F1 +0.0021 [−0.0285, +0.0326], LL −0.7%
  [−9.3%, +7.9%] (E152 at seeds 42/7/13: −0.012). Secondary: +0.0155 [+0.0018, +0.0291] plain (NB [−0.0160, +0.0469]);
  outcome files +0.041/+0.061/+0.011, probit files −0.015/−0.021. Cardio: F1 +0.0083 [−0.0124, +0.0290], inconclusive.
- **Conclusion:** inconclusive on both primaries (Holm's second step never opened); H2 has no confirmatory answer at
  this power, and no seeds are added. Primary 1 leans against A1: point estimate 2.7 times the margin, *reversed*
  missed by 0.0024. The secondary reads *gains* by the plain interval the rule named (the corrected one crosses 0): A1
  trails A2 under MCAR on all four siblings and reaches parity on the outcome-dependent files, not on the probit ones.
  Thesis sentence the user's call; no ADR. [Outcome](tickets/c15-mnar-prediction/11-h2-external-gate.md).

### E156 · Does A1 close on A2 where the gap depends on the outcome? MIMIC-IV complete tables, known mechanisms (pre-registered 2026-11-06)

- **Status:** done 2026-11-07 03:53 GMT-3, **not confirmed** (every call no verdict). Pre-registered 2026-11-06 on the user's go
  ("Vamos fazer o 1 e 3 até amanhã a tarde"). T52. Confirms on new tables the split E155's secondary found in its own
  data.
- **Hypothesis:** A1 (`[NULL]`) trails A2 (missForest + indicator) under MCAR and closes the gap where missingness
  depends on the outcome, not where it depends on the value (self-masking).
- **Scenarios:** `mimic-sofa-semi` (10,553 rows, 6 columns) and `mimic-cardio-semi` (10,795 rows, 8 columns), cut
  complete from the natural tables by `datasets/make_mimic_complete.py`; gaps injected at 60% by `mnar_family.py`
  (data seed 5858): outcome odds 3, probit slope 0.5, MCAR. A1 and A2 at seeds 2525/4747/6969, five folds, 36 cells on
  gorgona8; E123's G2 on the six files (90 CPU cells).
- **Measures and decision rule:** macro-F1 differences of differences per (seed, fold), Nadeau–Bengio corrected 95%;
  gains / loses / holds (±0.01) / no verdict. Primary: (A1 − A2) outcome minus MCAR, pooled over the tables whose
  outcome file passes G2, predicted gains. Secondary 1: probit minus MCAR, predicted not gains. Secondary 2: outcome
  minus probit, predicted gains. Read once with all cells in.
- **Sources:** [c15-mnar-prediction/12](tickets/c15-mnar-prediction/12-mechanism-split.md);
  `scripts/experiments/mechanism_split.py`, `run_e156_chain.sh`.
- **Results:** 36 cells and 90 G2 cells. G2: both outcome files informative (+7.7%, +21.9%), probit and MCAR files
  not (+0.2% to +0.8%). Primary outcome − MCAR pooled +0.0069 [−0.0212, +0.0350], no verdict (sofa +0.0061, cardio
  +0.0078). Secondary 1 probit − MCAR +0.0059 [−0.0191, +0.0309], no verdict (probit files are non-informative
  controls by G2). Secondary 2 outcome − probit +0.0010 [−0.0283, +0.0303], no verdict. A1 − A2 macro-F1 per file
  +0.0002 / −0.0007 on MCAR, +0.0062 / +0.0070 on outcome, +0.0045 / +0.0067 on probit; log loss lower for A1 on the
  outcome files (−0.012).
- **Conclusion:** not confirmed. E155's split does not replicate here because its premise is absent: A1 does not trail
  A2 under MCAR on these small clinical tables, so there is nothing for the outcome mechanism to close. The H2 sentence
  keeps only E155's inconclusive primaries; no ADR changes. The corrected interval (about ±0.03) could not resolve
  effects of 0.007 either way.

### E157 · ReMasker as an external deep baseline (pre-registered 2026-11-06)

- **Status:** done 2026-11-07 06:12 GMT-3; external deep baseline only (strong-tier rule 1 of 9); 60–80nan readings flagged
  (the arm barely trained there). Pre-registered 2026-11-06 on the user's go ("Vamos fazer o 1 e 3 até amanhã a
  tarde"). T53.
- **Hypothesis:** none directional; E48's measurement for MaCoDE repeated for ReMasker (Du, Melis and Wang, ICLR 2024),
  the nearest masked-autoencoder imputer.
- **Scenarios:** ReMasker at the reference code's settings (600 epochs, batch 64), written in `src/training/remasker.py`
  (the reference has no licence; departures stated: inductive, gaps never visible, batched fill, own streams). E44's
  confirmation set at seeds 707/808/909, its numerical variants that fit a night: pendigits 40/60/80nan, vehicle,
  biodeg, kc2 at 40/80nan; 27 pairs. Letter (about 25 process-hours) and electricity (categorical) out.
- **Measures and decision rule:** E48's: K − ReMasker per variant (primary), ReMasker − strong tier / bar / tuned bar,
  K − best of all, fit time; induced `impute_score`, verdict rule, 1% SESOI. Strong-tier proposal if better than its
  best on at least 3 of 9. Read once with all pairs in, by 14:00 GMT-3 on 2026-11-07, else partial.
- **Sources:** [imputation-protocol/55](tickets/imputation-protocol/55-remasker.md); `remasker_run.py`,
  `run_remasker_lanes.py`, `remasker_report.py`.
- **Results:** 27 pairs. K − ReMasker better on 9/9. ReMasker − the strong tier's best: better 1 (pendigits_40nan
  −0.075), no difference 1, worse 7. ReMasker − the bar: better 1, no difference 2, worse 6; the same against the
  tuned bar. K − best of all: better 7, worse 2 (vehicle_80nan, biodeg_80nan; missForest). At 80nan ReMasker is worse
  than the mean fill by 0.58–1.11. Found at the read: the reference's keep count of `int(L·0.5)` tokens re-masks
  0.0–0.1% of observed cells at 80% missing and 1–4% at 60%, against 19–22% at 40%. The paper defines the ratio over
  observed values (§4.4).
- **Conclusion:** an external deep baseline only; no ADR. Only the 40nan readings describe ReMasker (K better on 4/4,
  ReMasker above the strong tier on pendigits only). The 60–80nan readings measure an arm with almost no
  reconstruction signal. A per-row observed-cell re-mask rerun is proposed for the user's decision.

### E158 · ReMasker with the paper's re-mask count (pre-registered 2026-11-07 08:15 GMT-3)

- **Status:** done 2026-11-07 11:42 GMT-3; external deep baseline only (strong-tier rule 2 of 9). Pre-registered 2026-11-07 08:15
  GMT-3 on the user's go ("Rode de novo com a regra do artigo para vermos"). T53.
- **Hypothesis:** none directional; E157 repeated with the re-mask ratio counted over each row's observed cells, as
  the paper states (§4.4), since E157's reference-code count re-masks almost nothing at 60–80% missing.
- **Scenarios:** E157's 9 variants × 707/808/909, 27 pairs; `remask="paper"`, everything else E157's. medusa1 runs
  `pendigits_80nan` × 3 and `pendigits_60nan` s909, gorgona8 the other 23 (ADR 0035). Toy check before launch:
  RMSE at 80% missing, paper rule 0.648, reference 0.896, mean fill 0.750.
- **Measures and decision rule:** E157's: K − ReMasker (primary), ReMasker − strong / bar / tuned bar, K − best of all,
  strong-tier proposal at 3 of 9; paper − reference descriptive. Read once with all pairs in.
- **Sources:** [imputation-protocol/56](tickets/imputation-protocol/56-remasker-paper-rule.md); `remasker_run.py`
  (`REMASKER_REMASK`), `run_remasker_lanes.py --rule paper --host`, `remasker_report.py --paper --against`.
- **Results:** 27 pairs (4 on medusa1), none failed. K − ReMasker better on 9/9. ReMasker − strong tier: better 2
  (pendigits 40/60nan), no difference 2, worse 5. ReMasker − bar and − tuned bar: better 3, no difference 2, worse 4. K − best
  of all better 7, worse 2 (vehicle_80nan, biodeg_80nan; missForest). Paper − reference rule: better on the four 80nan
  variants (−0.52 to −1.07) and pendigits_60nan (−0.14), worse on pendigits_40nan (+0.006) and biodeg_40nan (+0.066).
- **Conclusion:** an external deep baseline only, no ADR. The paper's rule fixes E157's 80nan collapse; K beats
  ReMasker on every variant under both rules. E158 is the thesis's ReMasker comparison. The thesis sentence was accepted
  2026-11-07 12:00 GMT-3 ("aprovo a formulação"); it is in ticket 56.

### E159 · ReMasker on letter, the paper's rule (pre-registered 2026-11-07 21:43 GMT-3)

- **Status:** done 2026-11-08 02:38 GMT-3 (pre-registered 2026-11-07 21:43 GMT-3 on the user's go, "De acordo, va"). T53.
- **Hypothesis:** none directional; E158 extended to `letter_{40,60,80}nan`, the numerical confirmation table left out
  for cost.
- **Scenarios:** ReMasker paper rule × 707/808/909, 9 pairs; medusa1 letter 60/80nan, gorgona8 letter 40nan (ADR 0035).
- **Measures and decision rule:** E158's; strong-tier proposal if better than its best on at least 4 of the 12 numerical
  variants (E158's nine with these). Read once with all pairs in.
- **Sources:** [imputation-protocol/57](tickets/imputation-protocol/57-remasker-letter.md); `run_remasker_lanes.py
  --variants --gpus`, `remasker_report.py --variants --remasker DIR...`.
- **Results:** K better than ReMasker on 3/3 letter variants (−0.040, −0.012, −0.054), so on 12/12 numerical variants.
  ReMasker better than the strong tier's best on letter 40nan (−0.047) and 60nan (−0.072), worse at 80nan (+0.040):
  4 of 12 overall (pendigits and letter at 40/60nan), the pre-registered threshold. medusa1's CSVs SHA-256 6/6 OK.
- **Conclusion:** an ADR amending 0021 to admit ReMasker to the strong tier is proposed for the user's decision (not
  written); against it, every win is on the two large tables at 40–60nan, and a fold costs 58 min on letter_40nan. The
  accepted sentence's count becomes 12, with letter added beside pendigits; wording the user's.

### E160 · ReMasker off MCAR: the MAR and MNAR families (pre-registered 2026-11-07 21:46 GMT-3)

- **Status:** done 2026-11-08 02:38 GMT-3 (pre-registered 2026-11-07 21:46 GMT-3 on the user's go, "De acordo, va"). T53.
- **Hypothesis:** does ReMasker (paper rule) gain ground on TRIDENT off MCAR, as its paper's robustness claim would
  suggest?
- **Scenarios:** MAR: E64's `{biodeg,kc2,vehicle}_{marf,marfctl}_{20,60}nan` × 404/505/606; MNAR: E45/E46's
  `{biodeg,kc2,vehicle}_probit_{20,60}nan` and MCAR controls × 42/7/13; 72 pairs on gorgona8.
- **Measures and decision rule:** per variant H/K − ReMasker, ReMasker − strong/bar; (H − ReMasker) mechanism minus
  control per table-rate; primary over 12 table-rates, supported if worse on ≥4 and better on ≤1, refuted if worse on
  ≤1, mixed otherwise. Read once with all pairs in.
- **Sources:** [imputation-protocol/58](tickets/imputation-protocol/58-remasker-off-mcar.md);
  `remasker_offmcar_report.py`, `run_remasker_lanes.py --seeds`.
- **Amendment 2026-11-08 10:50 GMT-3:** the pre-launch fold check compared mean/mode's `impute_score`, which is 1 by
  construction, so it proved nothing. Redone on `rmse_num_z` and `mae_num_z` for the same five variants, it holds
  (max |diff| 4e-16): the rebuilt folds are the runs' own.
- **Results:** primary refuted: (H − ReMasker), mechanism minus control, worse on 1 of 12 table-rates (biodeg probit
  60nan, +0.059), better on 5. MAR: TRIDENT's lead grows on 4/6 (−0.07 to −0.22), none worse; MNAR: 4 no difference, 1
  better, 1 worse. H better than ReMasker on 11/12 mechanism variants (kc2_probit_60nan no difference); ReMasker worse
  than the strong tier on 10/12.
- **Conclusion:** ReMasker does not gain ground on TRIDENT off MCAR; a sentence for the user, no default, bar or tier
  change.

### E161 · ReMasker on the MIMIC-IV tables with natural gaps (pre-registered 2026-11-08 10:50 GMT-3)

- **Status:** done 2026-11-08 13:46 GMT-3, primary better on both tables but the arm is in doubt (pre-registered 2026-11-08 10:50 GMT-3 on the user's go, "Execute d e c"). T53.
- **Hypothesis:** none directional. E148's comparison gets a deep baseline on gaps a clinician left.
- **Scenarios:** ReMasker (paper rule) on E148's masked cells, `mimic-{cardio,sofa}_natural` × 42/7/13, 6 pairs.
  ReMasker sees the numerical columns only and mode-fills the categorical ones. Cardio runs on medusa1, SOFA on
  gorgona8 (ADR 0035).
- **Measures and decision rule:** per table, decoder − ReMasker `rmse_num_z`, 15 fold pairs, t interval, Holm over two
  tables. Better or worse at adjusted p < 0.05 by sign. Descriptive: `mae_num_z`, `impute_score` (with the mode
  caveat), ReMasker − the best fixed baseline and − TabPFN on `rmse_num_z`. Read once with all pairs in.
- **Sources:** [mimic/09](tickets/mimic/09-remasker-natural-gaps.md); `remasker_run.py` (categorical mode fill),
  `remasker_mimic_report.py`.
- **Results:** decoder − ReMasker `rmse_num_z` −0.809 [−0.970, −0.648] on cardio and −0.875 [−1.106, −0.644] on SOFA,
  Holm p < 0.0001, both better. ReMasker scores 1.620 and 1.761, against mean fill's 1.01–1.04 and 1.00–1.02 on the same
  cells: +0.80 and +0.86 above the best fixed baseline, +0.77 and +0.86 above TabPFN. All 600 epochs ran.
- **Conclusion:** better by the rule, but ReMasker is worse than the mean on these tables, so the comparison says
  nothing about a competent deep baseline and no sentence is drawn. A diagnostic follows off the record; a changed arm
  would be a new E-id.
- **Diagnostic and sentence (2026-11-08 20:24 GMT-3):** a fold-1 diagnostic per column, explanatory, put ReMasker's
  error in the columns whose values pile at an end of the min-max scale: vasopressor doses, `gcs_min`, troponin and
  lactate, 1.3–4.3 against the mean's ~1. On correlated columns it scores 0.39–0.71 against the mean's ~1. The error
  is not over-fitting. Thesis sentence accepted: "Nas tabelas da MIMIC-IV, o ReMasker como publicado fica abaixo até
  do preenchimento pela média; o erro se concentra nas colunas cujos valores se acumulam no limite da escala, como as
  doses de vasopressor." biodeg and kc2 have such columns too; E157–E160 stand as read.

### E162 · The deployable hybrid with missForest where TRIDENT trails it (pre-registered 2026-11-08 11:01 GMT-3)

- **Status:** done 2026-11-08 11:46 GMT-3, supported (pre-registered 2026-11-08 11:01 GMT-3 on the user's go, "Execute d e c"). T54.
- **Hypothesis:** T39's deployable `table` rule, with missForest in TabPFN's place, closes TRIDENT's 80nan gap to
  missForest on `vehicle_80nan` and `biodeg_80nan`.
- **Scenarios:** E89's (1212/2323/3434) and E90's (4545/5656/6767) cell tables, missForest refit per training fold
  (CPU, the run's logged missForest score asserted fold by fold); arms H, missForest, the `table` hybrid.
- **Measures and decision rule:** fold-pair verdict with 1% SESOI. Primary on E90's seeds: supported if the hybrid is
  better than H and not worse than missForest on both variants, refuted if better than H on neither, mixed otherwise.
  E89's seeds beside. Read once with all 12 pairs in.
- **Results:** missForest's refit equals the run's score on all 60 folds. E90's seeds: `table` hybrid − H −0.064
  (vehicle) and −0.072 (biodeg), hybrid − missForest −0.021 and −0.012, all better; H − missForest +0.044 and +0.060.
  E89's seeds agree on every call. The `column` rule trails `table` on every reading.
- **Conclusion:** supported: the deployable rule with missForest as the other side turns TRIDENT's 80nan loss on these
  two tables into a lead over missForest. Extending ADR 0031 decision 6 is the user's call. A deployable version still
  needs a rule that picks the other side without the complete table; this study did not test one.
- **Sources:** [imputation-protocol/59](tickets/imputation-protocol/59-missforest-hybrid.md); `missforest_hybrid.py`.

