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

## Outcome (2026-11-05, 02:43 GMT-3; 210 of 210 cells, gorgona8 42 and medusa1 168, cut 11 done on medusa1 02:13 and on gorgona8 02:20 GMT-3, 0 failed)

Read with `c1_report.py --stage2` after syncing medusa1's S25V/B25V readouts. Unedited tally, S25V − B25V in test
macro-F1, seeds 42/7/13/101/202 × 5 folds, pooled over the 21 variants per replicate:

**Pooled over 21: +0.0101 [+0.0045, +0.0157] (NB [−0.0060, +0.0262]) → HOLDS.**

| group | pooled | variants better / worse |
|---|---|---|
| categorical (credit-g, kr-vs-kp) | +0.0087 [+0.0003, +0.0171] | kr-vs-kp_40nan (+0.012) / none |
| small numeric (vehicle, biodeg, kc2) | +0.0080 [−0.0040, +0.0201] | none / none (biodeg_60nan −0.036 [−0.093, +0.020], no verdict) |
| spambase | +0.0118 [+0.0061, +0.0175] | all four levels (+0.011 to +0.015) / none |
| large (pendigits, letter, electricity) | +0.0159 [+0.0127, +0.0190] | all three (+0.010, +0.017, +0.021) / none |

- **8 of 21 variants better, none worse.** Every seed's pooled mean is positive on the large tables and on spambase;
  the small tables are noisy per variant (biodeg_60nan's s13 at −0.10 is one 25-row-labelled fold set) and net to
  about +0.008.
- **The channel is smaller on the 21 than on the four 60nan tables** (+0.010 against E138's +0.022): the 21 are mostly
  20–40nan and large, where a quarter of the labels is still thousands of rows. The large tables nevertheless give
  the tightest and most consistent gain (electricity +0.021 [+0.019, +0.023]).
- **The Nadeau–Bengio interval crosses 0** (fold-overlap correction, printed beside by design); the pre-registered rule
  is the plain interval, which holds. Said here so the thesis sentence carries the caveat.
- **Reading, by the rule:** HOLDS. Proposal for the user (no default change; the default regime has every label): a
  thesis claim of the form "with a quarter of the training labels, pre-training on all rows adds +0.010 [+0.005,
  +0.016] macro-F1 over no pre-training, pooled over the 21 confirmatory variants and five seeds, with no variant
  worse", and, if wanted, an ADR documenting `FINETUNE_LABEL_FRACTION` / `FINETUNE_SCARCE_VALIDATION` as the
  label-scarce fine-tuning path. The user decides both.
