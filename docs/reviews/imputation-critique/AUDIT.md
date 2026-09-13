# Fidelity audit of `CONSOLIDATED.md`

_Fact-check of the consolidated report against its 13 critique files and 13 verdict files.
2026-09-11. Read-only on the repository except this file and `CONSOLIDATED.md`._

**Headline: the synthesis is faithful.** 90 findings were filed across 13 dimensions, all 90
received a verdict, and all 90 appear in the consolidated report. No finding was dropped. No
number, commit hash or run id in the report traces to nothing. Seven defects were found and
fixed, most of them one clause each; none of them overturns a conclusion.

---

## What was checked, and how

**1. Finding census (dropped items).** Extracted every `### F-NN-M` heading from each critique
and each verdict file, and grepped each id in `CONSOLIDATED.md`.

- Critique findings filed vs. verdicts issued, per dimension: 5/5, 5/5, 6/6, 7/7, 4/4, 8/8,
  10/10, 10/10, 10/10, 6/6, 9/9, 5/5, 5/5 — **90 filed, 90 adjudicated, none orphaned**.
- Every one of the 90 ids appears at least once in `CONSOLIDATED.md`. **Zero dropped items.**

**2. Verdict and severity fidelity (laundered uncertainty).** Extracted the
`**Verdict:**` and `**Severity after review:**` line for all 90 findings and compared each
against how the consolidated presents it. There are **no `UNCERTAIN` verdicts** anywhere in the
corpus; the four values in play are CONFIRMED (20), SOUND (55), UNSOUND (14) and REFUTED (1).

- Every UNSOUND and REFUTED finding is labelled as such where it appears. Every one of them is
  presented under its *corrected* claim with the refuted headline named — checked individually
  for F-03-2, F-03-4, F-03-5, F-04-6, F-06-5, F-07-3, F-07-7, F-08-2, F-08-5, F-09-10, F-10-2,
  F-10-4, F-11-8, F-12-1, F-12-4.
- Every verifier severity downgrade is carried. Spot-verified the largest cuts: F-13-1
  critical→high, F-03-1 high→medium, F-09-2 high→low, F-09-3 high→medium, F-06-1 high→medium,
  F-06-2 high→medium, F-08-1 high→medium, F-08-3 high→medium, F-08-4 high→medium, F-11-1
  high→medium, F-12-2 high→medium, F-10-1 high→medium, F-07-5 medium→low, F-07-6 medium→low,
  F-11-2/3/4/5 medium→low, F-09-6/7 medium→low. All present with the cut and its reason.
- §1's "exactly one *bug* survives verification at high severity" is correct: the only
  CONFIRMED-at-high verdicts are F-04-1, F-05-1 and F-09-1, which are the same defect (D-1).

**3. Coverage-map arithmetic.** Re-derived every row of §6's "Findings filed → survived"
column from the severity census. All 13 rows are correct, including the awkward ones: `03`
(6→5, 3 UNSOUND of which 1 at none), `10` (6→5 as defects, 1 REFUTED), `11` (9→8, 1 UNSOUND at
none), `13` (5→5, one of the three mediums a CONFIRMED bug).

**4. Verifier-added items ("What this critique missed").** All 13 verdicts carry such a
section. Enumerated every item in all 13 and located each in the report. All present except
three, now fixed (see changes 4, 5 and 7). Verified present: 01's four (erased-category crash,
F-01-3's twin, the `nan`-loss note, two clean checks); 02's five (λ's third channel, the ADR
does document the checkpoint rule, classification's per-epoch proxy, the constant-column
inversion, `_loss_events_by_key` tolerating extra keys); 03's four; 04's two; 05's three; 06's
five; 07's six; 08's six; 09's one large and one of two small; 10's four; 11's four; 12's five;
13's two (the within-column variance collapse, `schedulers.py` absent from `main`).

One item is deliberately absent and correctly so: 02's verifier closes with an "explicitly
**not** a finding" note that `EPOCHS_DECODE` is held at 150 in every reduced trial and that a
longer budget is where F-02-1 would become measurable. §3.4 makes the equivalent point in its
own words ("Earning a higher severity needs a per-epoch validation `impute_score` trace...").

**5. Invented content.** Mechanical sweep of `CONSOLIDATED.md` against the union of the 26
source files:

- hash-like tokens (commit sha, run id, long decimals): 17 distinct, **0 not found in a
  source** — including all nine commits and run ids (`345574b`, `74229fd`, `87d8c82`,
  `9a38b90`, `fc5666d`, `e764df47`, `10cdd533fee5`, `631c3b64a184`, `7ed05b09f090`);
- decimals with two or more places: 152 distinct, **0 not found in a source**;
- 3-to-6-digit integers: 327 distinct, **0 not found in a source**;
- percentages: 50 distinct, 1 not found verbatim (`9.1%`), which is the correct rounding of
  `03`'s measured `ratio cv=0.0908`.

**Nothing in the report is invented.**

**6. `file:line` accuracy (mangled claims).** Oriented with `graphify query`, then read the
real source for the load-bearing references rather than trusting the graph or the critiques.
Verified against the working tree: `src/training/decoding.py:152-153, 154-157, 169-172, 174,
178, 243, 246, 251-255, 262, 264, 288, 307`; `src/embedder.py:177, 189, 192`;
`src/training/artifacts.py:57-64, 80-88, 327, 359`; `src/utils.py:55, 58, 62, 64-74`;
`src/training/config.py:49-51, 56, 61-69, 97-105`; `src/training/imputation_metrics.py:44-57,
48-56, 88-90, 136`; `experiment_imputation.ps1:60`. **All correct.** (Note for future readers:
`graphify` places `_select` at `decoding.py:268`, which is its `def`; the report's `:262` is
the call site where `masked_positions` is overwritten, and `:262` is the right line for the
claim being made.)

**7. Structural completeness.** All six required sections are present and non-empty: Verdict
(§1), Confirmed defects (§2, D-1..D-9), Design and methodology (§3, 3.1–3.11), Experiments
(§4), Checked and cleared (§5), Coverage map (§6, 13 rows plus the untouched-files list).

**Not run:** no pytest (per the brief), no training run — the empirical questions that arose
were all answerable by reading the real source, so the one sanctioned run was not needed.

---

## Per-source confirmation

One line per critique. "Represented" means every finding with a surviving verdict is in the
report, at the verifier's severity, under the verifier's corrected claim.

| Source | Findings | Represented |
|---|---|---|
| `01-decode-stage.md` + verdict | F-01-1..5 (4 SOUND, 1 CONFIRMED) | Yes — 1 in §3.1, 2 in §3.2, 3 in §3.11, 4 in §3.4, 5 in D-7. Verdict's missed items in D-3 and §6. |
| `02-decoder-model.md` + verdict | F-02-1..5 (4 SOUND, 1 CONFIRMED) | Yes — 1 and 2 in §3.4, 3 in §3.5, 4 in D-9, 5 in §3.9. The verifier's third λ channel is in §3.4. |
| `03-imputation-metrics.md` + verdict | F-03-1..6 (3 SOUND, 3 UNSOUND incl. one at none) | Yes — 1 in §3.1, 2/4/6 in §3.5, 3 in §3.3, 5 in §5. All three UNSOUND headlines named as refuted. |
| `04-ground-truth-protocol.md` + verdict | F-04-1..7 (1 CONFIRMED, 5 SOUND, 1 UNSOUND) | Yes — 1 in D-1, 2 in §3.1, 3 in §3.2, 4 in D-3 and §3.11, 5 in §3.1 and §3.10, 6 and 7 in §3.11. |
| `05-embedder-tokens.md` + verdict | F-05-1..4 (2 CONFIRMED, 2 SOUND) | Yes — 1 in D-1, 2 in D-9, 3 in §3.2, 4 in §3.11. All three verdict "missed" items present (preview label in D-6, vocabulary whole-table fit in §3.11, `electricity` crash in D-3). |
| `06-artifacts-tracking.md` + verdict | F-06-1..8 (4 CONFIRMED, 3 SOUND, 1 UNSOUND) | Yes — 1 in D-5, 2 in §3.5, 3 in D-6, 4 in §3.10, 5 in §3.4, 6 in §3.9, 7 in D-7, 8 in D-8. All five verdict "missed" items present. |
| `07-optuna-imputation.md` + verdict | F-07-1..10 (2 CONFIRMED, 6 SOUND, 2 UNSOUND) | Yes — 1/2/6/7 in §3.3, 3/4/9/10 in §3.8, 5 and 8 in D-9. The verifier's Spearman table, induced-on-validation lever and launcher correction are all carried. |
| `08-experimentation-design.md` + verdict | F-08-1..10 (4 CONFIRMED, 4 SOUND, 2 UNSOUND) | Yes — 1/6/7/10 merged into D-2 as the verifier asked, 2/4/9 in §3.7, 3 and 5 in §3.3, 8 in D-5. The `-DryRun` blind spot is in §4. |
| `09-test-adequacy.md` + verdict | F-09-1..10 (2 CONFIRMED, 7 SOUND, 1 UNSOUND) | Yes — 1 in D-1, 2 in §3.1, 3/4/5/6/7/8 and 10 in §3.9, 9 in D-9. Two-aggregator gap present; the `metrics.csv` gap was missing and is now added. |
| `10-config-plumbing.md` + verdict | F-10-1..6 (4 SOUND, 1 UNSOUND, 1 REFUTED) | Yes — 1 in D-5, 2 in §5, 3 in D-2, 4 in §3.8, 5 in D-8, 6 in §3.10. CWD-relative fallback was coverage-map-only and is now in §3.8. |
| `11-documentation-contract.md` + verdict | F-11-1..9 (1 CONFIRMED, 7 SOUND, 1 UNSOUND at none) | Yes — 1/3/4/5/6/7/9 in §3.10, 2 in D-9, 8 in §5. All four verdict "missed" items present. |
| `12-reproducibility-provenance.md` + verdict | F-12-1..5 (1 CONFIRMED, 2 SOUND, 2 UNSOUND) | Yes — 1 and 4 in §5, 2 in §3.7, 3 in §3.8, 5 in D-9. The standalone ledger-`row` section is in §3.11. Four of five verdict "missed" items were present; the dead `splits_path` pointer was not and is now in §5. |
| `13-pretraining-transfer.md` + verdict | F-13-1..5 (1 CONFIRMED, 4 SOUND) | Yes — 1/2/3/4 in §3.6, 5 in D-4. The verifier's within-column variance collapse and `schedulers.py`-on-`main` check are carried. |

---

## What was found and changed

Seven edits, all surgical. Nothing was rewritten and no finding of my own was added.

1. **Laundered uncertainty in §1 — a ρ attached to the wrong population.** The Verdict said
   the objective-vs-headline rank correlation "gives ρ = +0.124 on `credit-g_20nan` and +0.071
   on `kr-vs-kp_40nan`" against `test/impute/induced/impute_score`. The 07 verdict's table
   (`07-optuna-imputation.verdict.md:379-384`) gives `kr-vs-kp_40nan` **+0.372** against
   induced; **+0.071** is its objective-vs-test-*masked* figure. §3.3 reproduces the table
   correctly; only §1 mismatched. *Fixed:* §1 now cites +0.124 as the induced reading, names
   +0.071 as the masked one, and gives kr-vs-kp_40nan's induced ρ of +0.372 as borderline.
   This is the one correction that changes what a reader takes away — the headline population
   is *not* uninformative on both tables, only on `credit-g_20nan`.

2. **Laundered uncertainty in §1 — an unverified ablation stated as fact.** "a paired ablation
   cannot detect any benefit from 298 of the 300 default epochs" was presented as established.
   F-13-2's verdict is `SOUND (premise verified; the ablation's numbers are the critic's, not
   re-run)`, and §3.6 says so plainly. *Fixed:* §1 now attributes the ablation to the critic
   and states that the verifier did not re-run it.

3. **Conservative numbers in §1 where a verified pair exists.** "4.2x worse ... logged loss
   falls 98.5%" are the critic's fold; the verifier's independent reproduction gave NMSE
   **7.70** and a **97.4%** drop. *Fixed:* §1 gives both, attributed.

4. **A missing verifier item — 09's `metrics.csv` coverage gap.** The 09 verdict's "Two
   smaller ones" (a) — no test opens an imputation run's `metrics.csv`, the artifact AGENTS.md
   treats as the deterministic record — appeared nowhere. *Fixed:* one paragraph added at the
   end of §3.9, tied to F-09-8, which is what lets `cv/test/validation/...` keys become columns
   of that file unnoticed. Its sibling item (b), a float `!=` standing in for "a different
   split", is test wording and falls under the brief's exclusions; it is named in §3.9 as
   out of scope rather than filed.

5. **A missing verifier item — 10's CWD-relative silent fallback.** Named only in the §6
   coverage-map cell, with its consequence nowhere in the body. *Fixed:* two sentences in §3.8
   beside F-12-3, carrying the verifier's own scoping (pre-existing on `main`, outside `10`'s
   declared scope, widened by the branch). The verifier names only
   `experiment_imputation.ps1:22` for the `Set-Location $PSScriptRoot` that masks it; I checked
   the tree and `imputation_studies.ps1:34` does it too, so the plural in the added sentence is
   verified rather than assumed.

6. **Two mangled quantities.**
   - §3.6 read "within-column variance shrank 26% while the column-constant part grew 16%".
     In the source table the **+16%** is the target's *total second moment*; the
     column-constant *share* moved 83.3% → 89.5%. *Fixed* to match the table.
   - §4 read "The `rate_N` segments are mislabeled by up to 6x". The 6x is from F-01-5's
     4-feature synthetic frame (`rate_5` realising 0.3086); the largest shipped-table figure
     is `rate_10` realising 0.2548 on `credit-g_80nan`, 2.5x. *Fixed* to give both with their
     bases, since the sentence sits in a paragraph about the shipped ladder.

7. **A missing verifier item — 12's dead `splits_path` pointer.** The 12 verdict's missed item
   1: `prepare_dataset` always sets `splits_path` and `write_tracking_provenance` always writes
   it, but `build_folds` reads it only when `cv_folds is None`, so a CV run's `provenance.json`
   names a split file that had no bearing on any fold *while* omitting the `_00nan` sibling it
   actually read. The report carried the omission half ("the `_00nan` sibling is in no record")
   and not the dead-pointer half. *Fixed:* appended to the F-12-1 residue in §5, where the
   verifier put it. Both line references (`src/training/data.py:48` and `:155-172`) were read
   against the working tree before being written.

**Two structural annotations added, not content changes.**

- **D-3 carries a severity no verifier issued.** It sits in §2 ("Every finding whose verdict
  was CONFIRMED") at **medium**, but no critique filed it — it comes from two verdicts' "What
  this critique missed" sections, where the 01 verifier wrote "I would file it low-to-medium"
  for the `kr-vs-kp` half and the 05 verifier reproduced the `electricity` half without
  assigning a severity at all. The entry is well-evidenced and belongs where it is; what was
  missing was the disclosure. *Added:* a severity note on D-3 naming it a synthesis judgement
  and citing what the two verifiers actually said, plus the two adjacent formal verdicts that
  bracket it (F-09-4 SOUND/medium on the crash, F-04-4 SOUND/low on the guard), and a clause in
  §2's preamble marking D-3 as the one entry there without a CONFIRMED verdict line.
- **§5's "Suspected and disproved — do not spend time here" opens with three entries that are
  not all disproved.** F-12-1 and F-12-4 are UNSOUND *with residue kept at low*, and each
  states its residue in bold inside the entry. The heading could steer a reader past it.
  *Added:* one sentence under the heading flagging that the first three entries carry real
  residue.

---

## Judged faithful, recorded rather than changed

- **F-07-2's severity.** The report's §3.3 heading keeps "high (F-07-2)" — the 07 verdict's
  value — and the body then argues to **medium**. That is not laundering: the 03 verifier put
  the same mechanism at low, the consolidated states the disagreement explicitly rather than
  resolving it silently, and gives its reasons on both sides. It is a **synthesis judgement**,
  and it reads as one. Left as written.
- **§1's "`--score_null_path` scored 76.4% observed cells on `credit-g`"** compresses `04`'s
  frame-level measurement (4000 selected, 944 real gaps) rather than `05`'s end-to-end run
  (1976 scored, 478 real gaps, 75.8% observed). Both are in the sources, both round to ~76%,
  and D-1 gives both in full. Left as written.
- **§3.1's realised-rate ladder** uses `01`'s verdict numbers (0.2085/0.1685/0.1548/0.1572/
  0.2500) where `03`'s verdict reports 0.2085/0.1775/0.1507/0.1598/0.2600 from a different
  probe. The report states the measurement path it used, and it is `01`'s. Not a conflict.
- **§5's subheading "Refuted, and severity-none"** covers F-10-2, which is REFUTED with residue
  at low. The section's own stated rule ("REFUTED findings *and* UNSOUND findings whose
  severity is none") admits it, and the entry names its residue. Imprecise label, correct
  content. Left as written.

---

## What this audit did not check

- The **critique files' underlying claims against the repository** beyond the `file:line`
  sample in check 6. This audit verifies the consolidated report against its sources; the
  verdicts are what verify the critiques against the code.
- **`mlflow.db`.** Every store figure in the report (452 runs, 186 trials, the Spearman table,
  the `plateau` halvings, the `config_source` dangling pointers) was checked for internal
  consistency across the sources that report it, and is consistent, but no query was re-run.
- The **seven source files §6 lists as opened by no dimension** — `scripts/backfill_*.py`,
  `src/training/schedulers.py`, `src/mlflow_utils.py`, `src/training/finetuning.py`,
  `main.py`. The report is honest that they are a gap; this audit does not close it.
