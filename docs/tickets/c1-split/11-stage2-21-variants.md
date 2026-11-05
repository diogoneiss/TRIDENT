# Stage 2: the quarter-label channel on the 21 confirmatory variants, five seeds (E141; pre-registered 2026-11-05)

This ticket was written and committed before the cut's first cell (00:30 GMT-3).

- **Why:** E138 ([09](09-scarce-confirmation.md)) held (+0.022 [+0.009, +0.035] on fresh seeds, validation scarce), so
  its stage 2 follows: the result on the project's confirmatory set, the one a thesis sentence can rest on. Launched
  tonight on the user's standing request (2026-11-05, 00:25 GMT-3) and the ticket-09 rule.
- **Code:** none new. Cut `11` in `c1_run.py` (`CONFIRMATORY_21`, `FIVE_SEEDS`; seed 42 on gorgona8, the other
  four on medusa1); reader `c1_report.py --stage2`.

## Design and rule (fixed before the first cell)

- **Cells:** E30's 21 variants (credit-g, kr-vs-kp, spambase at 20/40/60/80nan; vehicle, biodeg, kc2 at 20 and 60nan;
  pendigits, letter, electricity at 20nan) × seeds 42/7/13/101/202 × S25V / B25V = 210 cells. Each variant's resolved
  classification configuration (ADR 0036/0037 encoders where promoted) with `FINETUNE_LABEL_FRACTION 0.25`,
  `FINETUNE_SCARCE_VALIDATION on`, `PRETRAIN_OBJECTIVE embedding_normalized`; B25V has `EPOCHS_PRE 0`.
- **Primary, one test:** S25V − B25V in test macro-F1, 25 paired replicates a variant, pooled over the 21 variants
  per replicate, plain 95% t interval (24 df), Nadeau–Bengio beside. **Holds** if the plain interval is above 0.
  Per-variant lines are flags; the four groups (categorical, small numeric, spambase, large) are descriptive.
- **Cost, estimated now:** large cells 10–12 min (electricity 20nan arm A took 12 min on medusa1 on 2026-11-03), medium
  3–5, small 1–2: about 13 GPU-queue hours; medusa1 (168 cells, 4 queues) about 3 h, gorgona8 (42 cells, 3 queues)
  about 1.5 h, after cut 10.
- **What follows:** *holds* → a proposal to the user: a thesis claim ("pre-training on all rows pays when a quarter of
  the labels are available: +x macro-F1 over no pre-training on 21 variants") and, if wanted, an ADR making the
  label-scarce fine-tuning path a documented option; no default changes (the default regime has every label). *Does
  not hold* → the channel is a property of the four 60nan tables, recorded as such.
- **Threats stated now:** letter's 26 classes at a quarter of the labels; electricity's drift; one fraction; the
  machines differ per fold (E86), handled by keeping every pair's arms on one machine.
- **Read only with all 210 cells done.**
