# The decode checkpoint chosen by the sliced Wasserstein score of the validation rows (E183, T60; pre-registered 2026-11-18 13:35 GMT-3)

This ticket was written and committed before the study's first cell. E182 (ticket 71), which measures the same score
on the default run and its baselines, had its 390 files in and had not been read; only its gates had been printed. One smoke was seen (`kc2_20nan`, seed
101, MLflow off): its duration, its GPU memory and the quantities that do not depend on the arm, not its scores.

- **Why:** E182 measures `sw_score`, a distributional score that sees the rows, on fills the default run chose by
  point error. The user asked whether the score should enter training. The first step, as E145 did for `dist_score`
  and E181 for the class-conditional score, is to let it choose the decode checkpoint and train nothing differently.
- **The user's go:** "sim, pre registre!" (2026-11-18, about 13:08 GMT-3), to my offer to register the choice of
  checkpoint by `sw_score` and run it on GPUs. The design is my proposal. The user also asked whether the loss should
  count the total decode error rather than the per-cell one: the loss is not changed here (see "Not in this study").
- **Code (this commit):** `--decode_metric sw_score` (`DECODE_METRIC`), a fourth value; no default changes, and a run
  that does not ask is as it was (both regression fixtures pass).
  - `src/training/decoding.py`: the validation scorer's `sw_score`, for the `induced` and `masked` checkpoints.
  - `src/training/types.py`, `config.py`: the value and its help; a search cannot rank by it (the run does not log it).
  - `scripts/experiments/sliced_select_run.py`: `train`, `--todo`, `--tally`.
  - `tests/unit/test_sliced_checkpoint_score.py`.
- **What the selector computes, every epoch:** the validation rows that hold a scored cell, as the current weights
  fill them and as they truly are; numbers in the scaled space, each categorical column one-hot over the embedder's
  categories plus one slot for an id outside them, divided by √2; 512 fixed unit directions; the mean gap between the
  sorted projections of the two; that distance over the mean/mode fill's. The epoch with the lowest value is kept.
  - It differs from E182's offline score in the directions (512 fixed against 2048 per fold) and in the one-hot's
    vocabulary. The read uses E182's offline score on the test folds for both arms, never the selector's own value.
- **Scenarios:** S, the default run with `--decode_metric sw_score`, on E181's 26 MCAR variants at seeds 101/202/303
  (78 runs), on medusa1.
  - Control: E181's `Kc` run of the same variant and seed, which kept the default's epoch (K and Kc differed by 0.0000
    on six of the seven mixed variants and +0.0012 on `credit-g_20nan`). No new control arm is trained.
  - Pairing, checked in the smoke: on `kc2_20nan` seed 101 the epoch the loss criterion names is the same in all five
    folds as in E181's run, and so are the cell counts and hgb's score. The trajectory is the same; only the kept
    epoch moves (S 212/161/220/293/251 against 50/16/370/321/253).
  - medusa1 only. The arms pair because they train on the same GPU model: CUDA dropout masks differ between GPU
    models, so a cell trained on gorgona8 would not pair with E181's. This reverses my offer of both machines.
  - Calibration stays the default (`DECODE_CALIBRATION gaps`, ADR 0027): from 40% missing the output is the kept
    checkpoint's fills after a calibration fitted for point error. So the study measures the deployable output, and
    the calibration can mute what the selection changes. E145 ran with the calibration off; this does not.
- **Measures and decision rule:** both arms' test-fold ledgers scored by E182's code (`sw_score`, `dist_score`,
  `impute_score`), 15 (seed, fold) values a variant, fold-pair rule with seed agreement, margin 0.01 of the control's
  mean.
  - Hypothesis: choosing the checkpoint by `sw_score` lowers the output's `sw_score`.
  - Primary, over the variants E182's control passes: S − K on `sw_score`. Supported if better on half or more of
    them and worse on at most 2; refuted if better on a sixth or fewer; mixed otherwise.
  - Secondary, no rule: S − K on `impute_score` (the cost) and on `dist_score`; S and K against E182's best
    strong-tier baseline on `sw_score` and `impute_score`, and the calls that move; the mean kept epoch.
  - Read once, with the 78 runs and E182's 390 files in. No default changes whatever the result.
- **Disclosed:** the selector's construction, the variants, seeds and thresholds are my choices. E145's precedent:
  selecting by `dist_score` improved it on 12 of 12 variants and cost `impute_score` on 7.
- **Not in this study:** the loss. Today it is per cell: cross-entropy for a categorical cell and `lambda_num` × squared
  error for a numerical one, averaged over the masked cells (`src/models.py`). For point error the sum over a row's
  cells already is the row's error; what the loss lacks is a term on the joint distribution of a row's fills. Adding
  one changes the training algorithm and needs the user's explicit go.
- **Amendment (2026-11-18 16:10 GMT-3):** at 14:34:29 GMT-3 every process of the study was stopped with 57 of 78
  runs in (all of `kc2`, `vehicle`, `credit-g`, `biodeg`, `kr-vs-kp`, and `pendigits` at 20 and 40). Cause, from the
  user's journal: the user's last SSH session on medusa1 closed at 14:34:01, lingering is off for the account, and
  systemd's user manager exited 28 s later, stopping the tmux scopes the queues ran under (`setsid nohup` does not
  leave that scope). The machine did not restart. Not relaunched, on the user's order ("nao reinicie as filas mortas
  aqui, rodaremos apenas na gorgona por enquanto"). The 21 runs left (`pendigits` 60 and 80, `spambase_20nan`,
  `letter_20nan`, `electricity`) cannot move to gorgona8 as designed: its GPU is another model (RTX 3090 Ti), so they
  would not pair with E181's runs. No change to design, seeds or rule; nothing read.

## Outcome

Not read yet.
