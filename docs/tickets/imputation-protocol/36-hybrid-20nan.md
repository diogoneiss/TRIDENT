# Does the deployable `table` hybrid hold at 20nan? (E95; pre-registered 2026-10-29)

This ticket was written and committed before the study's first cell.

- **Why:** ADR 0031's decision 6 (accepted 2026-10-29) covers MCAR at 40–80nan (E90, E91), and
  E93 tests MNAR and MAR. 20nan is the last untested MCAR rate. There the model usually leads
  TabPFN, so the question is E91's: does the hybrid cost anything?
- **The user's go:** "sim, vamos" (2026-10-29, about 15:00 GMT-3), on running it whole on medusa1.
- **Code:** no change to `src/`.
  - `hybrid_confirmation.py --study e95` runs E90's pipeline.
  - `--todo <n> e95` and `STUDY=e95` in `run_hybrid_confirmation_queue.sh` limit the queues to this
    study.
  - The TabPFN v2 regressor checkpoint was copied to medusa1 from gorgona8's cache; the md5 of
    both checkpoints matches.
- **Smoke:** kc2_20nan s4545, trained and refit on medusa1 before the queues start. It counts, and
  only its reproduction (the ledger equals the run's H) is read.

## Design (fixed before the first cell)

- **Cells:** the nine tables at 20nan × seeds 4545, 5656 and 6767, so 27 cells. The tables are
  electricity, letter, pendigits, spambase, kr-vs-kp, biodeg, credit-g, vehicle and kc2.
  - Each cell is E90's: H with the hidden observed validation cells read in place, then TabPFN v2
    refit on each training fold, and the `table` rule.
- **Running:** all on medusa1, so every cell's H and TabPFN share a machine (E86).
  - Four training queues and two cells lanes, all on GPU0 (`CUDA_VISIBLE_DEVICES=0`); E94 uses
    GPU1.
  - MLflow tag `experiment=hybrid-20nan-2026-10-29`, logged through the leader's server; state in
    `/scratch/diogoneiss/trident-2026-10-29-hybrid-20nan/`.
  - About 5–7 h: electricity, letter and pendigits dominate both training and refits.
  - **Deadlines:** no training cell starts after 2026-10-30 02:00 GMT-3, and no cells job after
    03:00 GMT-3. Anything missing then is partial.

## Measures and rule, fixed now (`hybrid_confirmation.py --study e95 --tally`)

- **Per variant:** `table` − H and `table` − TabPFN on `impute_score`, over 15 fold pairs, with a
  SESOI of 1% of the opponent's mean.
- **The call, in E91's form** (does it cost anything):
  - **supported** if worse than H on none and worse than TabPFN on 1 or fewer of 9;
  - **refuted** if worse than H on 2 or more, or worse than TabPFN on 3 or more;
  - **mixed** otherwise.
- **Descriptive:** better-than counts, and `table` − the equal blend.
- **What follows:** if supported, extending decision 6's scope to 20nan is the user's decision.

## Outcome

(open)
