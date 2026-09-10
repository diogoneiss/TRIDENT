# 12. Should the [NULL] path be scored as a diagnostic?

Type: grilling
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-10
Blocked by: none

## Question

Ticket 03 fixed the primary token path: induced-missing cells are presented as `[MASK]`
when scored. The paper's claim is that the `[NULL]` embedding carries signal, so a
secondary number, the same cells scored with the model seeing `[NULL]` as the variant
presents them, would test whether the null representation is reconstructable at all.

- Is one extra forward pass per fold worth a second metric family?
- If yes, where does it live (metric names, preview) and is it ever used for ranking?
- If no, close this ticket as out of scope.

Recommended starting answer: log it as a diagnostic only, never for ranking, if it fits
in one forward pass; otherwise rule it out of scope.

## Answer

Resolved 2026-09-10 over two grilling rounds. Decision 1 is the user's option (d), added
to the three offered; decision 3 is the user's amendment to the recommended answer.

**What the diagnostic actually measures** (established while resolving, and it narrows the
claim). `preprocess_table` sets `dynamic_mask[null_matrix.values] = False`
(`src/utils.py:61`), so an already-null cell is never masked, never enters the decode
loss, and the head at a `[NULL]` position is **entirely untrained**. The score therefore
does *not* test whether the `[NULL]` embedding is informative. It tests whether a readout
trained on `[MASK]` positions transfers to `[NULL]` ones.

That distinction matters because the paper's claim is about **classification** -- knowing a
value is missing helps predict the label -- not about reconstructing the value at a null
position. The one thing this number is genuinely good for is evidence for or against
ticket 03's decision to substitute `[MASK]` when imputing real missing data, which is
currently asserted rather than measured.

**Decisions:**

1. **Shipped behind a feature flag, off by default** (option d). Not a permanent
   always-on metric family, which would double the induced-missing metric surface for a
   number whose meaning does not change run to run; not a throwaway script, which would
   leave the capability unrepeatable.
2. **Pre-registered overturn criterion**, fixed before any number is seen: the primary
   path switches to the null token only if it beats the mask path on `impute_score` across
   a **majority of folds on at least two datasets, at both the 20% and 60% variants**.
   Anything narrower is single-fold noise. If the mask path wins or they tie, ticket 03
   stands and this is recorded as its evidence.
3. **A command-line switch, `--score_null_path`,** joining `--plot_losses` and
   `--save_model` as a `store_true` field on `TrainingRequest` defaulting to `False`. It is
   **not** a hyperparameter and **not** a config key: it changes no training, it only asks
   for an extra scoring pass, exactly like plotting or saving. Ticket 08's ban on new
   command-line flags covered hyperparameters such as `LR_DECODE`, which describe the
   experiment and belong per dataset; this is a per-invocation choice.

   Validation splits by what is knowable when:

   | Condition | Behaviour | Why |
   |---|---|---|
   | Flag passed without `--task imputation` | **Rejected at parse time** | Knowable from the arguments alone; there is no decoder under classification, so nothing could be scored |
   | Flag passed on a dataset with no induced-missing cells (a `_00nan` variant, or no sibling) | **Runtime warning**, run continues | Under `--all` the dataset list is not known at parse time |

   **This amends ticket 08**, whose decision 4 stated that no new rejection would be added
   to `validate_parsed_args`. The exception is deliberate and recorded there.
4. **Metrics take a path segment beside the primary induced-missing ones**, following the
   shape ticket 11 established (primary unprefixed, variants segmented):
   `impute/induced/null_token/rmse_num_z` and siblings. They **never** enter fold ranking
   or the Optuna objective. The flag is logged as a run parameter, so runs carrying the
   diagnostic are filterable without adding another tag to the store.
5. **The cell ledger gains one extra column** holding the null-path imputation when the
   flag is on; **the preview is unchanged**. A conditional column is easy to read and easy
   to omit, whereas a conditional fourth row in the triptych would change the shape of the
   main human artifact depending on a diagnostic that is off by default.

**Correction to ticket 09.** It recorded that `TrainingRequest.task` would be "last in
field order". The real constraint is that defaulted fields must follow undefaulted ones;
with `score_null_path` added there are two defaulted fields at the end and their order
between themselves is free.

**Adjacent experiment, deliberately not this ticket.** Whether the *classifier* scores
differently when missing cells arrive as `[MASK]` rather than `[NULL]` would test the
paper's actual claim about null embeddings carrying predictive signal. It needs no decoder
and is a separate experiment; it is recorded in the map's out-of-scope section.

## Comments

- 2026-09-10 (ticket 04): unaffected by ticket 04's decision to exclude `[NULL]` from the
  categorical head's output space. That governs what the decoder may emit; this ticket
  asks what it is fed. Both remain coherent.

- 2026-09-10, **verdict measured**: the pre-registered criterion is **not met**, so
  ticket 03's `[MASK]` substitution stands and now has evidence rather than only an
  argument. The `[MASK]` path won on all 8 folds across `credit-g` and `kr-vs-kp` at both
  20% and 60% missingness. The margin is widest on all-categorical `kr-vs-kp`
  (0.98 against 1.92), where a head never trained at a null position has nothing to
  transfer from. Full table in ADR 0004 decision 11.
