# Assignment 1 Report

- **Name**: Diana de Lorenzo
- **Student ID**: 20119
- **Email**: ddelorenzo.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

**Rent the Runway clothing-fit data** (UCSD McAuley Lab; Misra et al., RecSys
2018). Raw: 192,544 rentals, 15 columns, 2010–2018. One row = one customer's
rental of one item, with the customer's self-reported body measurements, the size
rented, the occasion, and the post-rental feedback (`fit` =
`fit`/`small`/`large`, a 1–10 rating, review text).

Committed sample (`data/rtr_fit_sample.csv`, built by `data/make_sample.py`):
19,867 rows, 70 items, 2015-01-01 to 2017-12-31, 1.7 MB. Sampling rule: keep
2015–2017, then the most-rented items until ~20k rows; drop the review text.

**Why this dataset.** I recently came across this business model in a shop I
visit, and I find it interesting: it addresses a real problem for a real
company, since returns caused by the wrong size are expensive. The columns
also come "dirty" (height as `5' 8"`, weight as `137lbs`, bust as `34d`), so
they have to be parsed and engineered before a model can use them.

**Why sample by item popularity.** The raw data has 5,850 different items. A
random sample of 20,000 rows would give each item about 3 rentals, and one of
the most useful features is "does this item tend to run small or large?".
With only 3 rentals per item that question can't be answered. Keeping the 70
most popular items gives each item between 187 and 828 rentals.

**What the model is valid for.** For a new item there is no rental history, so
the model doesn't know whether it runs small or large: it falls back to the
average misfit rate of all items and treats it as a "normal" item. If the new
item actually behaves abnormally (e.g. runs very small), the model will be
less accurate for it. Likewise, the model has only seen popular dresses and
gowns, so I would not trust it for a different type of garment or a different
situation. For the estimates to be more valid, the model needs to be trained
on many more items.

## Business / real-life framing

Decision supported: **at checkout, should we ship a free backup size with this rental?**

| | Outcome | Assumed cost |
|---|---|---|
| False negative | No backup, item doesn't fit → customer has nothing to wear for the event | €40 |
| False positive | Backup shipped but not needed → extra shipping + dry-cleaning | €8 |

How this framing drives the pipeline:

- **Target**: `misfit = 1` if `fit ∈ {small, large}` (22.3% of training rows).
  To decide *whether* to send a backup size, the only thing that matters is
  that the item doesn't fit: a return happens in both cases (too small or too
  large) and both count as the same cost. If the decision were instead *which*
  backup size to send (bigger or smaller), small and large would have to stay
  separate. Otherwise we would risk sending the wrong backup (a larger size
  when the customer needed a smaller one, or the other way round), or we would
  have to send both, which is more expensive.
- **Leakage**: `rating`, `review_text`, `review_summary` are excluded. A customer
  cannot review a rental they have not received yet, so at checkout (when the
  decision is made) these columns don't exist. Customers whose dress didn't
  fit tend to leave bad reviews, so the rating is strongly tied to the target.
  If the model had these columns it would rely on them heavily, because they
  are the easiest signal (low rating → didn't fit, high rating → fit). But in
  the shop, before the customer has tried the item on, the rating doesn't
  exist yet, so the model's predictions would be much worse. Its test results
  would be misleading: we would trust it, and it would then fail in practice.
  Any column only known *after* the decision has to be removed (data leakage).
- **Split**: chronological 70/15/15 by rental date — train 2015-01-01→2016-11-15 (13,909),
  val 2016-11-16→2017-06-06 (2,999), test 2017-06-07→2017-12-31 (2,959).
  When a customer rents a dress in 2017, we can only use information from
  before that date; 2018 hasn't happened yet. Shuffling the data at random
  isn't realistic: the model could learn from the future and use information
  it wouldn't have had at that moment. The same dresses and customers could
  also appear in both training and test, which makes the results less
  reliable. The test results would look better than they really are, while in
  real life the model only has past data and would probably perform worse.
  So I split by date: train on the past, test on the future. This is closer to
  real life and avoids data leakage. For the same reason, the history features
  are computed using only information available before each rental.
- **Threshold metric**: expected € cost per rental, minimised on the **validation** set, reported on test.
  Theory: with calibrated probabilities the optimal threshold is FP/(FP+FN) = 8/48 = 0.167;
  the validation-chosen one is 0.18.
  - *Why 17% and not 50%:* not all errors cost the same. If we don't send a
    backup when it was needed we lose €40, while sending one unnecessarily
    costs only €8. So it is better to send the backup even when the
    probability of a misfit is low. The 17% threshold reflects this cost
    difference, and it is very close to the 18% chosen on the validation data.
  - *Why not accuracy:* accuracy counts all errors equally, regardless of how
    much they cost. Since 78% of dresses fit, a model that always says "fits"
    would have 78% accuracy but would never avoid the expensive errors (that
    policy costs €8.90 per rental, the worst option). The average cost in euros
    per rental is a better metric because it shows directly what each decision
    costs the company.
  - *Are €40 / €8 reasonable:* €8 is an extra shipment plus dry-cleaning. €40
    covers compensating the customer and the risk of losing them. The €40 is an
    estimate, since it is hard to calculate exactly what losing a customer
    costs. What matters is that as long as a failure costs much more than an
    extra backup, a low threshold still makes sense (the dashboard lets you
    change both costs). With the model the cost is €5.74 per rental, versus
    €6.22 for the best option without a model: a saving of about €480 per
    1,000 rentals.

## Data preparation & feature engineering

All in `features.py`. One rule throughout: *a rental may only use information
that existed before its date*; anything fitted is fitted on train only.

| Group | Features | What it does |
|---|---|---|
| Parsing | `height_in`, `weight_lb`, `bust_band`, `cup`, `bmi` | `5' 8"`→68, `137lbs`→137, `34dd`→34 + ordinal cup |
| Cleaning | `age` outside 15–80 → missing; `*_missing` flags; `body_type` "unknown" | ages 0 and 116 exist in raw data |
| Size deviation | `size_dev_up`, `size_dev_down`, `size_dev_unknown` | `size − expected size`, where expected size = linear regression of size on height/weight/bust fitted on *train rentals that fit*. Split into two hinges |
| Item history | `item_small_rate`, `item_large_rate`, `item_log_past_n` | share of earlier rentals of this item that ran small/large, smoothed toward the prior with m=20 |
| User history | `user_misfit_rate`, `user_log_past_n` | same idea per customer |
| Interactions | `risk_small = item_small_rate × size_dev_down`, `risk_large = item_large_rate × size_dev_up` | "item runs small AND the customer sized down" |
| Categorical | `body_type`, `rented_for`, `category` | one-hot, rare levels (<50) grouped |
| Preprocessing | median imputation + standardisation (fit on train) | |

Ablation (sklearn, C=1, **validation** set, groups added cumulatively):

| added group | val ROC AUC | val log loss | val best cost €/rental |
|---|---|---|---|
| raw_numeric (+ categoricals) | 0.622 | 0.514 | 5.81 |
| parsed_body | 0.624 | 0.514 | 5.76 |
| size_deviation | 0.638 | 0.510 | 5.79 |
| item_history | 0.684 | 0.482 | 5.47 |
| user_history | 0.684 | 0.483 | 5.44 |
| interactions | 0.687 | 0.481 | 5.43 |

**Why two size-deviation features.** A dress can fit badly both when the
customer picks a bigger size than they need and when she picks a smaller one.
Logistic regression is linear, so it cannot represent this "V-shaped" risk
with a single variable; splitting it in two (`size_dev_up`, `size_dev_down`)
measures each case separately. That way the model can learn that moving away
from the right size in either direction increases the risk (both coefficients
come out positive: 0.29 and 0.25).

**Why the item rates are smoothed.** If an item has few rentals, its results
may be pure chance and not represent how it really fits. So the item's rate is
combined with the overall average: with few rentals the average weighs more,
and with many rentals we trust that item's own results more.

**Why expected size uses only rentals that fit, and only training data.** We
use only rentals that fit because we want to learn which size works for each
body type, without including cases where the size was wrong. And we use only
training data to avoid data leakage, so the model never uses information from
the test set that it shouldn't know.

### What didn't work

**Body measurements on their own (AUC 0.622 → 0.624).** Height and weight
alone are not enough to know whether a dress will fit. What matters is
comparing the measurements with the size the customer chose. That is why
computing the difference between the rented and the expected size improves
the model (0.624 → 0.638).

**Customer history (0.684 → 0.684).** Most customers rent only once: there
are 18,247 different customers in 19,867 rentals, and only 6.5% of rentals
come from a customer who had rented before. For almost everyone there is no
previous information, so this variable carries very little information and
doesn't help the model make better predictions.

**Interactions (negative coefficients, 0.684 → 0.687).** The interaction
features use information already included in other variables of the model
(the item rate and the size deviation). This makes the variables highly
correlated (collinearity), and the model splits the weight between them in a
non-intuitive way, so the interactions come out negative. The AUC only
improves from 0.684 to 0.687, so in practice this interaction adds very
little.

## Modeling: three implementations, one model

Logistic regression (binary target). Identical design matrix (42 columns) and
identical objective for all three: mean BCE + (λ/2)‖w‖², bias not penalised,
with λ = 1/(C·n) to match scikit-learn's `C·Σloss + ½‖w‖²`.
C = 0.01 chosen on val log loss (grid 0.001–10); lr = 1.0 for both PyTorch
versions (grid 0.1/0.5/1.0); 3,000 full-batch epochs; zero initialisation.

Test set (thresholds chosen on validation):

| model | threshold | cost €/rental | ROC AUC | PR AUC | log loss | Brier | precision | recall | % flagged |
|---|---|---|---|---|---|---|---|---|---|
| scikit-learn | 0.18 | 5.740 | 0.679 | 0.416 | 0.490 | 0.158 | 0.296 | 0.676 | 50.8% |
| manual PyTorch | 0.18 | 5.737 | 0.679 | 0.416 | 0.490 | 0.158 | 0.296 | 0.676 | 50.8% |
| nn.Module + optim | 0.18 | 5.737 | 0.679 | 0.416 | 0.490 | 0.158 | 0.296 | 0.676 | 50.8% |
| naive: training misfit rate | 0.01 | 6.221 | 0.500 | 0.222 | 0.530 | 0.173 | 0.222 | 1.000 | 100% |
| policy: never backup | – | 8.895 | – | – | – | – | – | 0 | 0% |
| policy: always backup | – | 6.221 | – | – | – | – | 0.222 | 1.000 | 100% |

Agreement:
- max |coef diff|: manual vs sklearn 0.0054, nn.Module vs sklearn 0.0054, manual vs nn.Module 1.2e-7
- max |P(misfit) diff| on test: 0.0035; same backup decision on 99.9% of test rentals
- final training objective: sklearn 0.4929596, manual 0.4929591, nn.Module 0.4929592

**Why logistic regression.** We want to predict whether a dress will fit or
not, so there are two possible outcomes. Logistic regression gives a
probability between 0 and 1, which we can compare with the 17% threshold to
decide whether to send a backup. Linear regression is for predicting numeric
values, such as a price.

**Why the two PyTorch versions are identical.** Both perform the same
calculations, even though the code is written differently: they start from
the same weights (zeros), use the same error function and apply the same
regularisation (`weight_decay` adds the same L2 penalty to the gradient that
the manual loop writes into the loss, and the bias is left unpenalised in
both, as in scikit-learn). The 1e-7 difference is just the computer's
rounding.

**Why scikit-learn is slightly different.** Scikit-learn uses a different
algorithm (L-BFGS) and can stop before reaching exactly the same point. Also,
some variables contain very similar information, so different combinations of
weights produce almost the same error (PyTorch actually reached a very
slightly lower objective). Despite these small differences, the three
versions make the same decision on 99.9% of rentals, so they work practically
the same.

**Is it a good result?** An AUC of 0.68 is acceptable, but it shows the model
still has difficulty predicting every case correctly. However, it is useful
for the business: it lowers the average cost from €6.22 to €5.74 per rental,
a saving of about €480 per 1,000 rentals. A model doesn't need to be perfect
to add value, although more information, such as the exact measurements of
each dress, could improve it.

## Limitations & next steps

1. **Only the 70 most popular items.** The model only knows the history of the
   most rented items, so it works worse with new or rarely rented ones.
   *Next step:* use the whole catalogue and borrow information from similar
   items, such as those of the same brand or type.
2. **Subjective and incomplete labels.** Whether an item runs small depends on
   what the customer says, and we only have data from customers who leave a
   review. *Next step:* use actual return data and collect information from
   all customers.
3. **Almost no repeat customers.** Only 6.5% of rentals come from customers
   who had rented before, so customer history adds little information.
   *Next step:* more years of data, or the shop's purchase history, would give
   more information about each customer.
4. **The costs are estimates.** €40 and €8 are assumptions and may not reflect
   what the company really spends. *Next step:* use the real costs of
   shipping, dry-cleaning and losing a customer to get more precise results.
5. **Threshold chosen on a single period.** The threshold was computed on one
   specific period, so it could behave differently in other seasons, such as
   wedding season. *Next step:* test it on data from different periods and
   update it regularly.
6. **Missing item data.** The model doesn't know each item's exact
   measurements, fabric or stretch, all of which affect fit. *Next step:*
   adding this data could help the model make better predictions and improve
   its AUC.

## Generative AI use disclosure

Claude Code (Anthropic) was used to write the code (`data/make_sample.py`,
`features.py`, `train.py`, `app.py`) and the factual tables in this report.
For the written analysis, Claude explained each concept; I then wrote my own
answers in Spanish, and Claude translated and formatted them into English. I reviewed the code and the final text and
can explain every decision.
