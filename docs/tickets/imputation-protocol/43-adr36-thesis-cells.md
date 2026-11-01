# The thesis tables' cells that ADR 0036 changed, rerun on today's default (2026-11-01 16:44 GMT-3)

Not a hypothesis test: a rerun so the thesis tables report today's default. Written before the
first cell.

- **Why:** ADR 0036 (accepted 2026-10-31) changed the default of pendigits and letter at 20 and
  40nan (BATCH 1024, LR_PRE and LR_DECODE x4, EPOCHS_DECODE 600). The tables still read those
  variants from runs on the old configuration.
- **The user's go:** "ok, va e faça o item 2 então (isso da adr 36)" (2026-11-01, after 16:20 GMT-3).
- **Which cells:** the development tables (`thesis_table.py`) read pendigits_20nan and letter_20nan
  at 101/202/303 and 42/7/13 (404/505/606 does not include them); the confirmation summary
  (`tabpfn_report.py`, `confirmation_report.py`) reads pendigits_40nan and letter_40nan at
  707/808/909. 18 cells.
- **Each cell:** today's default through the promoted files, five folds, `--score_checkpoints
  --score_calibrated` (every readout the tables use), MLflow tag
  `experiment=adr36-thesis-2026-11-01` (`scripts/experiments/adr36_thesis_runs.py`,
  `run_adr36_thesis_queue.sh`; 6 cells on gorgona8, 12 on medusa1).
- **Then:** `report_common.development_source` and `confirmation_report` read these runs for
  those variants, the tables are rebuilt, and the before/after counts are written here. The offline
  baselines (E41's, E42's, E55's TabPFN) depend only on the folds and are reused.

## Amendment (2026-11-01, 19:09 GMT-3): every cell on gorgona8

- **What happened:** the 18 runs finished (16:44–19:05, none failed, all with ADR 0036's BATCH 1024,
  LR_PRE 0.00136, LR_DECODE 0.004, EPOCHS_DECODE 600). Rebuilding the tables stopped at their check
  that a run's logged baselines equal E41's offline recomputation: the 6 gorgona8 runs match to 0,
  the 12 medusa1 runs differ on knn5 and knn10 only, by 4e-5 to 2.3e-3. A CPU difference in kNN,
  not a fold one; but knn10 is pendigits' bar, and the check is exact by design.
- **The fix:** all 18 cells rerun on gorgona8 under `experiment=adr36-thesis-g8-2026-11-01`, where E41's
  baselines were computed; the tables read that tag. The first tag's runs stay in the store,
  unread. No table number was read from the first runs.

## Outcome

(open)
