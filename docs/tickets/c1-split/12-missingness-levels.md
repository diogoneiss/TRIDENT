# The quarter-label channel at 20, 40 and 80% missingness (E143; pre-registered 2026-11-05)

This ticket was written and committed before the cut's first cell (00:30 GMT-3).

- **Why:** E136/E138 read the channel at 60nan only. The thesis's premise is pre-training on gappy tables, so the
  channel's dependence on the missingness level is the natural axis. Night run on the user's standing request.
- **Code:** none new. Cut `12` in `c1_run.py` (seed 1515 on gorgona8); reader `c1_report.py --levels`.

## Design and rule (fixed before the first cell)

- **Cells:** `{biodeg,kc2,vehicle,spambase}_{20,40,80}nan` (MCAR) × seeds 1515/2626/3737 × S25V / B25V = 72.
- **Per level, three tests as one family:** S25V − B25V pooled over the four tables, 15 replicates, plain 95% t
  interval, Holm over the three levels; **open** / **closed** as E140. E138's 60nan MCAR half printed beside.
- **What follows:** descriptive for the thesis; no ADR.
- **Read only with all 72 cells done.**
