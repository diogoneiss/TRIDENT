# Remove the promoted imputation files that lose to the defaults

## Status

Accepted on 2026-10-16. The user decided after study
[imputation-retune/01](../tickets/imputation-retune/01-retune-promoted-files.md) (E39), which
closes task T19. It amends [ADR 0005](0005-reduced-optuna-search-for-imputation.md): three of its
promoted files are removed. No code and no global default change.

## Context

Five variants read a promoted imputation file (ADR 0005, decision 5): credit-g_40nan,
kr-vs-kp_20nan, kr-vs-kp_40nan, spambase_20nan and spambase_40nan.

The sampled values in those files are `PROB_MASCARA`, `LR_DECODE`, `WEIGHT_DECAY_DECODE`,
`DROPOUT`, and `LAMBDA_NUM` on credit-g. The reduced studies of 2026-09-11 chose them under the old
configuration (commit `fcaadfb`): `embedding` pre-training, decode 150, gaps as `[NULL]`, the loss
checkpoint and the masked objective. ADRs 0013 to 0015 later rewrote the held keys and left the
sampled ones.

E39 compared three arms on seeds 101, 202 and 303, over 15 fold pairs:

- **F:** each current file;
- **D:** no file, so today's defaults;
- **T:** the winner of a fresh reduced study under today's defaults, ranked by the validation gaps.

The table shows the induced `impute_score`; lower is better.

| variant | F (file) | D (no file) | D against F | T against D |
|---|---|---|---|---|
| credit-g_40nan | 0.9662 | 0.9208 | −4.7%, D better | −0.5%, no verdict |
| kr-vs-kp_40nan | 0.7626 | 0.7258 | −4.8%, D better | −0.3%, no verdict |
| spambase_40nan | 0.9068 | 0.9023 | −0.5%, D better | −0.0%, no verdict |
| spambase_20nan | 0.8529 | 0.8439 | −1.1%, no verdict | +1.1%, D better |
| kr-vs-kp_20nan | 0.5866 | 0.5829 | −0.6%, no verdict | +0.0%, no verdict |

The current files are worse than having no file on three variants and better on none. The
re-tuned winners beat the defaults on none.

E39's rule, fixed in advance, recommended:

- promoting T on credit-g_40nan and kr-vs-kp_40nan;
- removing spambase_40nan's file;
- keeping the other two.

On the two "promote T" variants, though, T is equivalent to D. The gain there comes from the stale
values going away, not from anything the search found.

## Decision

1. **Remove three files:** `credit-g_40nan.imputation.json`, `kr-vs-kp_40nan.imputation.json` and
   `spambase_40nan.imputation.json`. Their imputation runs load today's defaults again. The user
   chose this over promoting T on two of them, since T does not beat D there and a file that
   matches the defaults adds nothing.
2. **Keep two files:** spambase_20nan and kr-vs-kp_20nan keep theirs, as the rule says. On those
   two, no arm separates from F.
3. **Keep the two ADR 0018 files** (pendigits_20nan, letter_20nan). E39 did not study them.

## Verification

- **Default runs match E39's D cells.** Runs of the three variants at seed 101 went through
  `main.py` with no flag and no file (`--cv_folds 5 --disable_mlflow`). Each resolves its
  configuration from `defaults`. Each equals E39's D cell of the same seed fold by fold:
  credit-g_40nan on 15 model metrics and 60 baseline columns, kr-vs-kp_40nan and spambase_40nan on
  11 and 36 each.
- **Tests:** the unit suite passes (355 tests, mypy included). No regression fixture reads these
  variants.

## Consequences

- **The 21-variant picture changes on three variants.** Every headline on credit-g_40nan,
  kr-vs-kp_40nan and spambase_40nan since 2026-09-11, E30 to E38 included, used the stale files.
  A run after this decision scores better than one before it there: about −4.7% on credit-g_40nan,
  −4.8% on kr-vs-kp_40nan and −0.5% on spambase_40nan.
- **Comparing runs across time:** filter on the param `config_source`. Runs before this decision
  log the file's path; runs after it log `defaults`. E39's D cells log `override` (an empty
  override) and equal a `defaults` run.
- **Re-running an older arm** on these variants must pass the removed file's values as an
  override. They are in git at the commit before this one.
- **Only two files remain from ADR 0005's studies.** Promoted values chosen under an older
  configuration should be re-checked when the defaults change, because E39 shows they can turn
  harmful.
