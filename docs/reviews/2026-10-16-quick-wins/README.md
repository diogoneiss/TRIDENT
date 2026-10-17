# Quick wins and methodology review, 2026-10-16

Working material from a Claude Code session started in the wiki vault. It is evidence, not decisions: nothing in `src/`, the store or the baseline cache was changed, and nothing here is pre-registered. The readable summary and the suggested work are on the wiki page `projects/trident/findings-2026-10-16-baselines-methodology.md` (the vault is the sibling directory `../Wiki`).

## Digests

- `digests/merge_digest.txt`: the ranked quick-win list (imputation first), with the skeptics' corrections, the conflicts that were resolved, the schedule and the per-variant table from the condexp/SoftImpute/ICE probe.
- `digests/review_digest.txt`: the code and methodology review (8 dimensions, 60 findings confirmed, 3 refuted): top threats, methodology recommendations, code fixes and what was found sound.
- `digests/wf2_method_cards.json`: one card per candidate method (MICE, HyperImpute, TabPFN/TabICL, TabImpute, MaCoDE, ReMasker/CACTI/DiffPuter, GBDT and deep classification, MNAR-aware imputers), plus the operations and reporting cards.

## Workflow results

The full JSON each workflow returned:

- `wf3-merge-imputation-first.json`: the classification readouts, the ADR 0017 table, the probe and the merged list.
- `wf4-code-methodology-review.json`: every review finding with its skeptics' verdicts.
- `wf5-missforest-replay.json`: the missForest test-time transform check.

## Scripts

Kept as templates for the planned work. The rest of the session's scratch was dropped.

| Script | What it is |
|---|---|
| `impprobe_arms_v1_preregistered.py` | condexp and SoftImpute rules as frozen before the first score |
| `impprobe_arms.py` | the run version, with `ice_seq` and the clip diagnostics |
| `impprobe_seqtest.py` | oracle check of `ice_seq` against sklearn |
| `impprobe_summary.json`, `impprobe_rows.json` | per-variant results, seeds 42/7/13 |
| `mfreplay_arms.py`, `mfreplay_passes.py`, `mfreplay_report.py` | missForest stock vs per-round test-time fill, and the pass counts |
| `mfreplay_report.json` | results of that check (seed 42, 8 variants) |
| `barcheck_inventory.py`, `barcheck_verdicts.py`, `barcheck_tally.py`, `barcheck_e39mf.py` | fold-paired verdicts rebuilt from `mlruns/`, the basis for the thesis table (suggested item 1) |
| `barcheck_out.txt` | output of the verdict rebuild |
| `budgetskeptic_hgb2.py`, `budgetskeptic_knn_k.py`, `budgetskeptic_ci.py` | templates for the baseline tuning budget (item 4) |
| `fair_probe_verdicts.py`, `meansonlyskeptic_probe.py` | verdict tallies per opponent vs counts by means |
| `stats_reanalyse.py`, `cvdepskeptic_mixed.py` | the alternative tests (seed random effects, Nadeau-Bengio) |
| `metric_probeB.py`, `metric_probeB4.py`, `metricskeptic_e35.py` | per-type and MAE/macro-F1 readings (items 7 and 7b) |
| `adr0015skeptic_q.py`, `oracleckpt_skeptic_mm.py` | K vs H vs the bar (item 5) |
| `dupskeptic_ledger.py`, `dupskeptic_twinsplit.py` | the twin / no-twin split |
| `levelconf_fe.py`, `maskpop_realised.py` | level confound and the realised masked rate |

## Caveats

- The scripts hard-code the Windows checkout and the session's scratchpad paths (for example `C:/Users/Diogo Neiss/...`, or JSON inputs that were not kept). Read them as a record, or fix the paths before rerunning.
- Every probe ran on the Windows laptop with no cache dir and no store writes, through `recompute_folds(..., fold_scorer=...)` or read-only reads of `mlruns/` and a store snapshot. The numbers are indicative. Thesis numbers come from gorgona8 under a pre-registered ticket.
