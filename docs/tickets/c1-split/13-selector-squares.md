# The selector × seeds squares: is the scarce-validation selector what opens a tenth? (E144; pre-registered 2026-11-05)

This ticket was written and committed before the cut's first cell (about 03:00 GMT-3).

- **Why:** E140 ([10](10-label-fraction-curve.md)) read a tenth of the labels *open* (+0.029, validation scarce, seeds
  1515/2626/3737) where E136 ([08](08-label-scarce.md)) read it *closed* (+0.011, validation whole, seeds 42/7/13), and
  put a hypothesis on record: B's epoch selection suffers more than S's from few validation rows, so the deployable
  selector favours pre-training more. Two things differed between the readings, the selector and the seeds; three
  of the four cells of each 2 × 2 already exist. This study completes the squares and reads the selector effect as a
  paired contrast. Night run on the user's standing request ("quero te manter ocupado no mínimo até as 7").
- **Code:** cuts `13` and `14` in `c1_run.py` (arms exist; `home()` routes each new pair to the machine its earlier
  arms sit on); reader `c1_report.py --selector` (new).

## Design and rule (fixed before the first cell)

- **Cells, a tenth (cut 13, 96):** S10/B10 (validation whole) on the eight C1 files × seeds 1515/2626/3737 (48, the
  scarce arms are E140's) and S10V/B10V (validation scarce) × seeds 42/7/13 (48, the whole-validation arms are
  E136's). **A quarter (cut 14, 72):** S25/B25 × 1515/2626/3737 (48, scarce arms E138's) and S25V/B25V on the four
  probit files × 42/7/13 (24; the four MCAR files' S25V/B25V at those seeds are E141's cells, reused). 168 cells.
- **Primary, two tests as one family:** per fraction, the selector effect (S(f)V − B(f)V) − (S(f) − B(f)) paired per
  (file, seed, fold), pooled over the eight files per replicate (6 seeds × 5 folds = 30 replicates), plain 95% t
  interval, Holm over the two fractions. **Supported** if Holm p < 0.05 and the mean is above 0 (the scarce selector
  widens S − B); otherwise **not supported**, and the seeds carry the E136/E140 difference. The four cell means
  (S − B by selector, pooled over six seeds) are printed beside, descriptive.
- **What this does not do:** E136's "closed at a tenth" and E140's "open at a tenth" stand as written, each on its
  pre-stated seeds; nothing here extends either study's seed count (no optional stopping). The new cells are read
  only through the paired contrast above.
- **Machines (ADR 0035):** every new pair sits where its earlier arms ran (E136's by `machine()`, E140's seed 1515 on
  gorgona8, E138's all on gorgona8), so each S − B difference is within one machine. Two reused pairs of the
  quarter's MCAR half (biodeg_60nan and vehicle_60nan at seed 42) have E136's whole-validation arms on medusa1 and
  E141's scarce arms on gorgona8: their difference of differences crosses machines. Stated as a threat; 2 of 48
  pairs.
- **Cost, estimated now:** cut 13 about 45 min (gorgona8 24 cells, medusa1 72; E140's 96 tenth-label cells took about
  40 min), cut 14 about 50 min (gorgona8 52, medusa1 20); done by about 04:40 GMT-3. The GPUs are idle after that
  unless the user adds work.
- **What follows:** *supported* → the deployable (scarce-selector) readings E138/E140/E141/E143 stand, and the thesis
  text says that part of the gain at scarce labels is B's noisier epoch selection, not only pre-training's content.
  *Not supported* → the E136/E140 difference is attributed to seeds; the hypothesis in E140's Outcome is withdrawn.
  No ADR either way.
- **Read only with all 168 cells done.**

## Amendment (2026-11-05, 08:15 GMT-3)

Another user's job (31.7 GB on each of medusa1's two GPUs) held medusa1 from about 02:40 and was still there at 08:12;
medusa1 ran none of its 92 cells (gorgona8's 76 were done by 03:56). On the user's decision ("passe as coisas da
medusa para cá") the 92 run on gorgona8 (`home()` returns gorgona8 for cuts 13/14). Each new S − B pair is still
within one machine; the second-order contrast (scarce − oracle) now crosses machines for every pair whose earlier
arms ran on medusa1, not only the two stated above. Machines differ per fold (E86), so this is a stated threat to the
contrast, not to the within-pair differences. The rule and the cell count are unchanged.

## Outcome (2026-11-05, 09:40 GMT-3; 168 of 168 cells, 0 failed; gorgona8's first 76 by 03:56, the moved 92 from 08:30 to 09:34 GMT-3)

Read with `c1_report.py --selector`. Unedited tally, (S(f)V − B(f)V) − (S(f) − B(f)) paired per (file, seed, fold),
seeds 42/7/13 and 1515/2626/3737, pooled over the eight files (30 replicates), Holm over the two fractions:

| labels | selector effect | NB | p | Holm | verdict | S − B scarce / oracle (six seeds) |
|---|---|---|---|---|---|---|
| 10% | −0.0056 [−0.0194, +0.0081] | [−0.0482, +0.0369] | 0.41 | 0.57 | **not supported** | +0.0172 / +0.0228 |
| 25% | −0.0056 [−0.0161, +0.0050] | [−0.0382, +0.0270] | 0.29 | 0.57 | **not supported** | +0.0142 / +0.0198 |

- **The scarce selector does not widen S − B.** The point estimate is slightly negative at both fractions (the
  whole-validation selector gives a marginally larger S − B), with no file in either direction (16 of 16 "no
  difference"). E140's hypothesis, that B's noisier epoch selection under few validation rows is what opened a
  tenth, is withdrawn.
- **So the E136 / E140 difference at a tenth was the seeds.** Descriptively, over six seeds the tenth-label channel
  is about +0.02 under either selector (scarce +0.017, oracle +0.023); E136's closed reading (+0.011 on 42/7/13) and
  E140's open one (+0.029 on 1515/2626/3737) stand as written on their own seeds, and the seed-to-seed swing (kc2
  s3737 +0.15 against s7 −0.08 in the contrast) says a tenth of kc2's labels, 36 rows, is where three seeds do not
  settle a verdict.
- **Reading for the thesis:** the deployable (scarce-validation) readings E138/E140/E141/E143 need no selector caveat:
  the gain at scarce labels is pre-training's, not an artefact of B's epoch selection. The 92 moved cells make the
  contrast cross machines for medusa1-sibling pairs (amendment above); the result is a null well inside the
  interval, so the threat does not change the reading. No ADR.
