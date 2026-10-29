# Does the deployable `table` hybrid hold under MNAR and MAR? (E93; pre-registered 2026-10-29)

This ticket was written and committed before the study's queues started.

- **Why:** ADR 0031's decision 6 (accepted 2026-10-29) makes the H + TabPFN hybrid with the `table`
  rule a deployable output. Its scope is MCAR (E90, E91); MAR and MNAR are listed as untested.
- **The user's go:** "sim" (2026-10-29, about 11:00 GMT-3), on the proposal to run it, after
  checking the MLflow sync: the follower's copy equals the leader's (21,647 runs, the same tag and
  metric counts) at 11:04 GMT-3.
- **Code:** no change to `src/`.
  - `hybrid_confirmation.py --study e93` runs E90's pipeline on E84's variants.
  - `run_hybrid_confirmation_queue.sh` now takes the state directory and both deadlines from the
    environment. E90 and E91's defaults are unchanged.
- **Smoke:** kc2_probit_60nan s4545 was trained before this commit (CELL_DONE, five selection
  files). Only its run's headline was printed (H 0.730), and no rule was scored. It counts.

## Design (fixed before the queues)

- **Cells:** E84's 12 shifted variants at 60nan × seeds 4545, 5656 and 6767 (E90's), so 36 cells.
  - **MNAR (self-masking probit):** biodeg at slopes 0.25, 0.5, 1.0 and 2.0; vehicle, kc2 and
    spambase at 0.5.
  - **MAR (`marf`):** biodeg, kc2, vehicle, spambase and credit-g.
  - Each cell is E90's: H with the hidden observed validation cells read in place, then TabPFN v2
    refit on each training fold.
  - The rule is `table`, unchanged.
- **Running:** gorgona8, four training queues and two cells lanes, MLflow tag
  `experiment=hybrid-mechanisms-2026-10-29` through the leader's server; state in
  `/scratch2/diogoneiss/trident-2026-10-29-hybrid-mechanisms/`.
  - About 3–4 h. spambase dominates both the training (about 37 min a cell in E84) and the refits.
  - **Deadlines, by the clock alone:** no training cell starts after 2026-10-29 18:00 GMT-3, and no
    cells job after 19:00 GMT-3. Anything missing then is reported partial, with no call.

- **Amendment 2026-10-29, 14:49 GMT-3, before any E93 result was read:** four cells are retrained.
  - medusa1 rebooted at about 14:30 GMT-3, taking the leader's MLflow server down. Four training
    cells failed between 14:22 and 14:32 on the server's API (`Max retries exceeded`):
    kc2_probit_60nan s5656, vehicle_marf_60nan s5656, credit-g_marf_60nan s4545 and
    vehicle_probit_60nan s6767. Three had finished training and failed when logging; one failed
    at its start.
  - Their outputs were moved to `failed-2026-10-29/` in the state directory, so the queue sees them
    as not done.
  - Each is retrained once with the same seed, by a rescue runner that starts after the four
    queues stop, under the same 18:00 deadline.
  - The failure is infrastructure. The same cells run again, no result was read, and no other cell
    reruns.
  - Their unfinished runs stay in the store (not FINISHED), so `report_common.parents` does not
    count them.

## Measures and rule, fixed now (`hybrid_confirmation.py --study e93 --tally`)

- **Per variant:** `table` − H and `table` − TabPFN on `impute_score`, over 15 fold pairs. The
  project's rule applies, with a SESOI of 1% of the opponent's mean.
- **The call, in E90's form:**
  - **supported** if better than H on 8 or more of 12, worse on none, and worse than TabPFN on 2 or
    fewer;
  - **refuted** if better than H on 5 or fewer, or worse on 2 or more, or worse than TabPFN on 5 or
    more;
  - **mixed** otherwise.
  - The thresholds are E90's (8, 5, 2 and 5), kept as they were rather than rescaled to 12 cells.
- **Descriptive:** the same counts split by MNAR (7) and MAR (5), biodeg's four slopes in order,
  and the other three rules.
- **What follows:** if supported, extending decision 6's scope to MNAR and MAR at 60nan is the
  user's decision. If refuted or mixed, the scope stays MCAR and the Outcome names where it breaks.

## Outcome

(open)
