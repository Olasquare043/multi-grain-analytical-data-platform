# Modelling Notes: the Feature Engineering Ladder (v3)

This document is the v3 counterpart to `docs/methodology_notes.md`: every
decision below is recorded because the obvious approach would have produced
a number that looked fine and was wrong, or because a genuine engineering
obstacle changed what was actually run. It covers the modelling layer added
on top of the already-built, already-verified v2 pipeline
(`src/modelling/`, `notebooks/01-03`), which does not modify anything under
`src/extract/`, `src/transform/`, `src/model/`, `src/quality/`, or the
existing `sql/analysis/` outputs.

## 0. Why this exists, and the one rule that governs everything in it

The course's original brief is already satisfied by the v2 pipeline. A later
instruction, sent to the whole class, added a further requirement: the paper
must propose a data engineering approach and provide **empirical evidence**
that it improves machine learning model performance, not merely describe a
well-built pipeline. This document, and the three notebooks it summarises,
are that evidence.

The method is a **feature engineering ladder**: a sequence of model
training runs, each adding one named data engineering technique on top of
the last, with the **model held fixed** (a single `LGBMRegressor`, identical
hyperparameters, `src/modelling/ladder.LGBM_PARAMS`) at every rung of both
tasks except V4's two encoding fits. Because the model never changes, any
change in a rung's test-set error is attributable to the one data
engineering decision that rung added, and to nothing else. Both tasks also
run this way under a **second, capacity-controlled hyperparameter set**,
selected once via time-respecting cross-validation on each task's own
bare, no-engineering baseline features and then frozen across every rung
exactly like the first set is (`src/modelling/tuning.py`) -- so a rung is
never re-tuned against its own features, only the *sample-size-appropriate*
model capacity changes, selected before any rung is fit. Both variants are
kept and reported side by side, never one in place of the other.

**Point-in-time correctness is the single most important technical
requirement in this layer.** Any feature computed from historical data uses
only data that existed strictly before the row it is attached to. Both
tasks split train/test by time, never by random shuffling
(`tests/test_time_split.py` asserts this for every split used), and both
build the leaky (V3a) and point-in-time-correct (V3b) version of the same
historical feature through two separately named functions in
`src/modelling/features.py`, never a single function behind a flag
(`tests/test_point_in_time.py` proves the two disagree whenever the future
actually differs from the past).

**Every rung is run five times, and no difference is claimed that does not
survive the repetition.** A ladder reporting one number per rung cannot
support the claim that one rung beat another, because it never measures how
far a rung's number moves when nothing meaningful changes. Every rung of
both variants of both tasks is therefore run at seeds 1-5
(`splits.REPEAT_SEEDS`), varying the model's `random_state` and — for Task
B, which samples — the sample drawn, together. Results carry a mean, a
standard deviation and a range; rung-to-rung differences are taken **within
seed** and reported with the number of repeats agreeing on the direction
(`src/modelling/repeats.py`, `outputs/tables/ladder_paired_comparisons.csv`).
With five repeats these are descriptive indications of stability, not
inferential tests, and no p-value is reported anywhere in this layer. **A
difference that does not hold its sign across all five repeats is not
treated as an established effect** — one previously reported finding was
withdrawn under exactly that rule, and is documented as withdrawn rather
than quietly dropped.

**Two reference predictors per task anchor the absolute level.** A ladder
shows relative movement and can look orderly while the whole structure sits
above a trivial benchmark. Each task therefore reports a no-model training-mean
predictor and a no-model domain heuristic alongside its rungs
(`src/modelling/baselines.py`). For Task A this turned out to matter more
than any rung-to-rung comparison in it. No published literature figure is
used as a comparison point anywhere in this layer: a published error is not
comparable to one of these unless the window, filtering, sampling and target
definition all match, and none of them do.

**Every number below comes from a notebook cell that was actually executed
and could be re-executed.** Nothing here was written before its
notebook cell ran; several numbers below are not the ones the v3 prompt's
authors likely expected, and they are reported exactly as measured.

---

## Task A: Nigerian petrol price forecasting (`notebooks/01_nigeria_petrol_forecasting.ipynb`)

### The panel, the target, and an honest wrinkle in the split boundary

`fact_fuel_price_monthly`: 1,147 rows, 37 states, 31 months (2023-11 to
2026-05), zero rejected rows at the quality gate, reconciled to ₦0.00
against NBS's own published national means
(`outputs/quality/quality_report.md`).

The target is **next month's price for the same state**, built with a
per-state `shift(-1)` after sorting by `(state, month_key)`. The panel's
final month, 2026-05, has no "next month" to supply a target for, so every
state's 2026-05 row is dropped before modelling (37 rows), leaving a
**1,110-row, 30-feature-month modelling frame**. The v3 prompt states the
test window as "2025-12 through 2026-05 (6 months)"; `nigeria_time_split`
applies that boundary literally to each row's own `month_key` (train ≤
2025-11, test 2025-12..2026-05), which is the same convention as the
training boundary. Because the last month was already dropped, the test
side of the *modelling* frame ends up covering **5** feature months
(2025-12 to 2026-04), predicting target months 2026-01 through 2026-05 --
the last month the panel can possibly predict. This is a mechanical
consequence of framing one-month-ahead prediction as a lag on a
fixed-length panel, not a deviation from the specified boundary, and it is
stated here rather than silently forcing a sixth row with a target that
does not exist.

### Two model-capacity variants, reported side by side

The results below come in two variants, both kept in
`outputs/tables/model_ladder_nigeria.csv` (`variant` column) and both
plotted in `outputs/figures/fig_model_ladder_nigeria.png`:

- **`original (untuned, 300 trees)`** -- `src/modelling/ladder.LGBM_PARAMS`,
  the same fixed hyperparameters used everywhere else in this layer (300
  trees, `num_leaves=31`, no L1/L2), chosen once for both a 925-row and a
  2,000,000-row task.
- **`capacity-controlled (CV-selected on V0)`** -- a second, still-fixed
  hyperparameter set, but this time sized to Task A's own sample. Selected
  **once**, on V0's baseline feature (this month's price predicting next
  month's) and V0's data alone, via `src/modelling/tuning.py`, then frozen
  and reused unchanged at every rung below -- never re-tuned per rung,
  for the identical reason `LGBM_PARAMS` itself is never re-tuned per rung
  (§0). Concretely: `expanding_window_folds` builds 5 forward-chaining
  folds over the 25 training months (minimum 15 months' training data,
  2-month validation blocks, never touching a test-period row); 8
  candidate hyperparameter combinations (`CANDIDATE_PARAMS`, spanning
  `num_leaves` 7-31 and light-to-moderate L1/L2) are each fit per fold with
  early stopping (30 rounds, `n_estimators` capped at 1,000) against that
  fold's validation MAE; the candidate with the lowest **mean** validation
  MAE across all 5 folds wins, and its frozen `n_estimators` is the mean of
  its best iteration on the two folds with the *largest* training windows
  (21 and 23 of the 25 training months) -- deliberately not a median across
  all 5 folds, since the two smallest folds train on far fewer months
  (15, 17) than the full ladder's 25-month fit ever will, and would pull a
  median-based tree count down to a size suited to a training set the
  ladder itself never uses.

  The winner: `num_leaves=7, learning_rate=0.10, min_child_samples=5, reg_alpha=0.0, reg_lambda=0.0`,
  frozen at **`n_estimators=12`** -- a much smaller model than the original
  300-tree default, and the outcome the capacity-control procedure was
  designed to test for.

### Every rung is run five times, and reported with its spread

A single number per rung cannot support a claim that one rung beat another,
because it never measures how far a rung's number moves when nothing
meaningful changes. Every rung of both variants is therefore run **five
times**, at seeds 1-5 (`splits.REPEAT_SEEDS`), varying LightGBM's
`random_state` — its bagging and column-sampling randomness — and nothing
else. Task A does no sampling: the panel is complete and every rung uses all
of it, so the data is byte-identical on every repeat and the spread below is
purely the model's own variability. (Task B, which samples, varies the
sample too.)

Below, **± is one standard deviation across those five repeats**. The `mae_ngn`
and `mape_pct` columns in the CSV retain their original meaning — the first
repeat — but every claim in this document is made against the mean and
qualified by the spread.

| Rung | Original MAE (₦) | Original MAPE | Capacity-controlled MAE (₦) | Capacity-controlled MAPE |
| --- | --- | --- | --- | --- |
| V0 -- naive baseline | 159.53 ± 0.68 | 11.47 ± 0.05% | 207.47 ± 1.93 | 14.41 ± 0.12% |
| V1 -- quality-gated | 159.53 ± 0.68 | 11.47 ± 0.05% | 207.47 ± 1.93 | 14.41 ± 0.12% |
| V2 -- + region_group, month_of_year | 200.88 ± 14.44 | 14.02 ± 0.74% | 244.12 ± 26.18 | 16.84 ± 1.76% |
| **V3a -- LEAKY, DO NOT TRUST** | 217.52 ± 2.71 | 15.13 ± 0.22% | 249.36 ± 13.59 | 17.17 ± 0.90% |
| V3b -- + point-in-time state average | 206.78 ± 1.29 | 14.31 ± 0.12% | 232.78 ± 1.70 | 15.95 ± 0.10% |
| V4a -- + state, one-hot | 212.43 ± 3.36 | 14.54 ± 0.21% | 237.11 ± 5.13 | 16.24 ± 0.35% |
| V4b -- + state, point-in-time target encoding | 212.10 ± 0.85 | 14.71 ± 0.06% | 236.29 ± 7.59 | 16.18 ± 0.57% |
| *REF_mean* (no model: training mean) | *327.34* | *22.54%* | — | — |
| **REF_heuristic** (no model: random walk) | **129.30** | **9.57%** | — | — |

(`outputs/tables/model_ladder_nigeria.csv`,
`outputs/tables/ladder_paired_comparisons_nigeria.csv`,
`outputs/figures/fig_model_ladder_nigeria.png`. Notebook runtime 14.1s for
70 fits plus the CV search.)

**The spread is not uniform, and that matters.** V0's MAE moves by less than
₦1 across repeats; V2's moves by ₦14 in the original variant and ₦26 in the
capacity-controlled one. V2 is the least stable rung in either ladder, which
means the single-run V2 figure this document previously reported (₦185.48)
was a comparatively lucky draw from a distribution whose mean is ₦200.88.
That is precisely the kind of number this pass existed to stop reporting.

### The reference predictors, and the finding that dominates everything else

Two predictors that fit no model at all
(`src/modelling/baselines.py`), scored by the identical code as every rung:

- **REF_mean** — predict the training mean price for every test row.
  MAE ₦327.34.
- **REF_heuristic** — a **random walk**: next month's price = this month's,
  carried forward unchanged. The standard naive benchmark for a price
  series. MAE **₦129.30**, MAPE **9.57%**.

**No rung of either variant beats the random walk, at any seed.** Counted
directly in the notebook: every rung, both variants, **0 of 5 repeats**
below ₦129.30. The best rung in the entire ladder is V0 itself, at
₦159.53 — **23% worse** than carrying last month's price forward unchanged.

Every rung does beat the training mean, in 5 of 5 repeats, so the models are
learning something real from their features; they are simply learning less
than the single fact that petrol prices are highly persistent month to
month. This is the most important sentence in Task A, and it was invisible
until a reference predictor was added: a ladder reports relative movement,
and relative movement can look orderly while the whole structure sits above
a one-line benchmark.

It also sharpens what V0 is. V0 is a gradient booster handed exactly one
column — this month's price — and asked to learn the mapping to next
month's. The random walk is the *identity* mapping on that same column. The
₦30 gap between them is what the model's learned adjustment costs on a
series still trending upward through the test window: the booster's
predictions are bounded by the leaf values it saw in training, while the
identity mapping carries the trend forward for free.

### The capacity-controlled result: worse everywhere, including V0 -- and why that is itself informative

The capacity-control hypothesis was that the original ladder's "every rung
past V0 is worse" pattern was an overfitting artefact of a 300-tree model
on 925 rows, and that a model sized to the sample would show the ladder's
*true* data-engineering effect more clearly. **That hypothesis is not
supported.** The capacity-controlled variant is worse than the original at
every single rung, including V0 itself (MAE 207.47 ± 1.93 vs. 159.53 ± 0.68)
-- a rung with exactly one feature, where "overfitting to extra columns"
cannot be the explanation, since there are no extra columns yet. The gap is
₦48, against repeat spreads of ₦2 and ₦1: this is one of the few
differences in Task A large enough that no amount of repetition would
dissolve it.

The mechanism, confirmed directly rather than merely inferred: Task A's
test period (2025-12 to 2026-04) sits inside the continuation of the same
price rise the training period (2023-11 to 2025-11) only partly captures --
training-period target values average ₦974.94/litre; test-period target
values average ₦1,300.79/litre, a level the training data barely touches.
Gradient-boosted trees cannot predict outside the range of leaf values they
learned from training data; they can only get *closer* to a test period's
higher values by taking more boosting rounds, each one shifting predictions
a little further from the training mean. In a single diagnostic fit of both
variants on the identical single-feature V0 data, comparing predictions
directly: the
**original** model (300 rounds) predicts as high as ₦1,439 on the test set
and undershoots the true test mean by ₦132.93 on average; the
**capacity-controlled** model (12 rounds) tops out at ₦1,301 -- barely past
its own training ceiling -- and undershoots by ₦195.86, roughly half again
as much. Twelve rounds is simply too few to walk predictions as far above
the training-period level as the test period has already moved. The CV
selection that chose 12 rounds is not wrong on its own terms: it correctly
found the tree count that minimises validation error *within* the training
period, where prices are comparatively calmer. It has no way to anticipate
that the held-out test period keeps climbing past anything the training
folds ever validated against -- that is a distribution-shift blind spot in
time-respecting CV itself, not a bug in how the search was run.

This changes what "the headline finding" (below) should be read as evidence
for. It is not that a 300-tree model is too large for 925 rows in the
simple, textbook overfitting sense -- a *smaller* model performs worse, not
better, at the one rung (V0) where overfitting-to-features cannot be the
explanation. The more defensible reading is: **Task A's series has moved
into a price regime the training period never fully reached, and the
original, nominally "overfit-prone" 300-tree model happens to extrapolate
toward that regime better than a leaner model selected to fit the calmer
training period.** Both the original ladder's monotonic V0-is-best pattern
and the capacity-controlled ladder's uniform underperformance are kept and
reported here, in full, as two honest and mutually clarifying findings --
neither is discarded in favour of a tidier story.

### V0 vs. V1: the ladder proving its own honesty, and now a harness control

`fact_fuel_price_monthly` has **zero** rejected rows
(`outputs/quality/quality_report.md`, rules Q002/Q015: 1,147 checked, 0
failed), so V0 and V1 read the byte-identical validated rows and produce
the byte-identical metric. This rung's job was never to guarantee an
improvement story; it was to prove that when a source is already clean,
the ladder says so rather than manufacturing a difference. It does.

With repeats, this rung acquires a second, useful job: it is a **control on
the harness itself**. V0 and V1 are the same fit, on the same rows, at the
same seed, so their paired difference must be exactly zero at every seed.
It is: `mean_diff_mae = 0.0000`, `sd = 0.0000`, in both variants. Had the
repeat machinery been mixing seeds, reusing a stale result, or comparing
across rather than within seed, this is the row where it would have shown
up as a non-zero number.

### Which differences survive repetition, and which do not

This is the question the ladder actually asks, and with five repeats it can
finally be answered rather than asserted. Differences are taken **within
seed** — the same model randomness for both rungs — and summarised across
the five (`outputs/tables/ladder_paired_comparisons_nigeria.csv`;
`tests/test_repeats.py` asserts the pairing arithmetic). Nothing below is a
significance test and none is claimed: with five repeats these are
descriptive indications of stability. **A difference that does not hold its
sign across all five repeats is not treated as an established effect.**

| Comparison | Variant | Mean Δ MAE (₦) | SD | Repeats agreeing | Established? |
| --- | --- | --- | --- | --- | --- |
| V0 → V1 | both | 0.00 | 0.00 | — (exact tie, by construction) | control passes |
| V1 → V2 | original | +41.36 | 13.83 | 5/5 | **yes** — V2 is worse |
| V1 → V2 | capacity-controlled | +36.64 | 27.20 | 5/5 | **yes** — V2 is worse |
| V2 → V3a | original | +16.64 | 14.93 | 5/5 | yes |
| V2 → V3a | capacity-controlled | +5.25 | 30.25 | 4/5 | **no** |
| V3a → V3b | original | −10.74 | 2.41 | 5/5 | **yes** — V3b better |
| V3a → V3b | capacity-controlled | −16.58 | 14.62 | 5/5 | **yes** — V3b better |
| V3b → V4a | original | +5.65 | 2.72 | 5/5 | yes |
| V3b → V4a | capacity-controlled | +4.33 | 4.60 | 4/5 | **no** |
| V4a → V4b | original | −0.33 | 3.78 | 3/5 | **no** |
| V4a → V4b | capacity-controlled | −0.83 | 10.92 | 3/5 | **no** |

Six of twelve comparisons held their direction across all five repeats. The
two that matter most to the paper's argument land on opposite sides of that
line, and both are discussed below: **V3a vs. V3b survives** (5/5, in both
variants), and **V4a vs. V4b does not** (3/5, in both).

### The headline finding among the rungs: every subsequent rung is *worse* than V0

This is the honest result, reported without adjustment. Every rung from V2
onward has a higher MAPE than the one-feature naive baseline, **in both the
original and the capacity-controlled variant** (previous section) -- so
this is not an artefact of one particular model size. Three things explain
it, none of which is "feature engineering is a bad idea":

1. **Untested at first, then directly tested and not confirmed: model
   capacity was not the driver.** The original hypothesis here was that
   300 fixed-depth trees on a 925-row panel simply had room to overfit
   whatever extra columns each rung added. The capacity-controlled variant
   (previous section) tests this directly, by selecting a much smaller
   model (12 trees, `num_leaves=7`) via CV and rerunning the identical
   ladder -- and that smaller model is worse at *every* rung, including V0,
   where there are no extra columns to overfit to. The real driver, argued
   in full in the previous section, is that Task A's test period sits in a
   price regime the training period only partly reaches, and a model's
   *capacity to extrapolate toward it* matters more here than its capacity
   to overfit a small training set does. Whichever model size is used, the
   later rungs' added features do not supply enough signal to outweigh that
   extrapolation gap.
2. **A near-random-walk, strongly autocorrelated target.** Petrol prices
   rose 146% over the series but move smoothly month to month
   (`docs/methodology_notes.md` §14); "last month's price" is already a
   strong predictor of "next month's price," which is precisely what makes
   a naive lag-1 baseline hard to beat without a genuinely informative new
   signal, and *region_group* / *month_of_year* are not that.
3. **This is itself the paper's argument, read the other way round.** A
   ladder that only ever went up would be the suspicious result; a step
   that does not help is a finding, not a failure (v3 prompt §2). Read
   correctly, Task A's ladder says: on a small, strongly-trending panel
   whose test period keeps moving past its training period, these standard
   techniques -- regardless of the model capacity applied to them -- do not
   supply enough new signal to close an extrapolation gap the naive lag-1
   baseline is, by construction, already about as well placed to handle as
   any of them. That is a real, useful, and non-obvious result about *when*
   data engineering pays for itself -- see the cross-task synthesis, §4
   below, for the direct
   contrast with Task B, where the same techniques mostly do pay for
   themselves.

### V3a (leaky) vs. V3b (correct): leaky is *worse*, not better — and this one survives repetition

**Established across all five repeats, in both variants** (5/5 agreeing;
mean Δ MAE −10.74 ± 2.41 original, −16.58 ± 14.62 capacity-controlled).
Alongside V1→V2, this is the most dependable result in Task A.

The expected direction (leaky looks artificially good) does not hold here:
V3a's MAPE (15.13 ± 0.22%) is *higher* than V3b's (14.31 ± 0.12%). The reason is the
series' strong trend: "this state's average price over the *whole*
series" is a poor summary of "the current price level" precisely because
prices roughly quadrupled across the window -- a whole-series average sits
close to a mid-series value, not a late-series one, and using it as a
feature actively misleads the model about where prices currently are. The
point-in-time expanding mean, by only ever looking backward, tracks the
trend instead of averaging across it, and is measurably the better
feature even though it is built from less data early in the series. This
is the most important single number this notebook produces:
**leakage's symptom is not universal, and on a trending series it can look
like the leaky feature is *worse*, not suspiciously better** -- see §4.

### V4: engaging Ayinla (2023) — and a claim this pass had to withdraw

**This comparison does not survive repetition, and the finding this document
previously reported here has been withdrawn.**

An earlier version of this section stated that "one-hot (V4a, MAPE 14.59%)
beats the point-in-time target encoding (V4b, MAPE 14.91%)", and read that
as disagreeing with the direction Ayinla (2023) argues for. Both numbers
were real; the conclusion drawn from them was not supportable. Across five
repeats the difference between the two encodings is **−0.33 ± 3.78 ₦ MAE
(original)** and **−0.83 ± 10.92 (capacity-controlled)** — a mean difference
an order of magnitude smaller than the spread of the differences it
averages, with only **3 of 5 repeats agreeing on the sign** in both
variants. The mean now leans very slightly the *other* way, toward V4b,
which is itself the point: with a difference this small relative to its own
variability, which encoding "wins" is decided by the seed.

What Ayinla (2023) argues is unaffected either way. Ayinla makes a case for
compact, index-mapped ordinal encoding over one-hot's high-dimensional
sparse expansion, on the grounds that it lets a tree-based learner split on
one informative axis instead of fragmenting the signal across many
near-duplicate binary columns. **Task A cannot speak to that claim**: on 37
categories and 925 training rows, the two encodings are indistinguishable
under this model at this sample size. That is a statement about the
resolving power of this experiment, not evidence for or against Ayinla. See
the cross-task synthesis for whether NYC's 265 categories and two million
rows — a far better-resourced test of the same question — can separate them.

### V5: not available

The only externally-sourced, verified covariate this platform carries at
daily grain is `fact_weather_daily` (Open-Meteo's NYC archive), and it
holds no Nigerian observations -- there is no honest key to join it to a
Nigerian state panel by. No other external covariate exists in this
project's verified sources for Nigeria, and none was manufactured to fill
the slot. **V4b is the final rung of Task A.**

---

## Task B: NYC trip duration (`notebooks/02_nyc_trip_duration.ipynb`)

### Sampling: deterministic, content-addressed, and previously not reproducible

Training eight rungs, under two hyperparameter variants, five times each on
all 40,421,155 gold rows is unnecessary -- this notebook compares rungs
*relative to each other* -- so each repeat works on one **bucket** of the
corpus.

An earlier version of this notebook drew that sample with DuckDB's
`USING SAMPLE n ROWS (reservoir, 796)`. That seeds the sampler but not the
order rows arrive from a `read_parquet(glob)` scan, and reservoir sampling
depends on arrival order too, so two runs of identical code drew different
rows and reported different metrics. Sampling is now **content-addressed**:
a trip's bucket is `('0x' || substr(trip_id, 1, 8))::UBIGINT % 17`, and
`trip_id` is already a deterministic MD5 over the trip's business
attributes, so the selected rows depend on nothing but the data.
`tests/test_sampling_determinism.py` proves the selection is unchanged when
the rows are deliberately reordered and identical across two separate OS
processes.

**That fixed the sample and was still not enough.** Re-running the notebook
to verify showed it *still* did not reproduce: `REF_mean`, which depends only
on which rows were selected, came back bit-identical, while the fitted models
did not, and one rung-to-rung conclusion moved with them. A deterministic row
*set* is not a deterministic row *order* -- DuckDB's parallel scan returns
rows in no guaranteed order, and gradient boosting is order-sensitive
(bagging selects by position, split ties break by arrival order). Both pulls
now carry an explicit `ORDER BY`. Measured at full scale, two processes on
the identical bucket produced prediction sums of 202,240,914 and 203,933,890
without ordering, and exactly 208,095,622.694010 each with it; LightGBM's own
`deterministic=True` did **not** fix it, the cause being upstream of the
learner. With both fixes in place, **notebook 02 was executed four times end to
end and its published tables are byte-identical across all four**. The full account is
`docs/methodology_notes.md` §12.

Realised bucket sizes, reported rather than assumed, since a bucket's size
is a property of the data:

| Seed | Bucket | Training rows | Test rows |
| --- | --- | --- | --- |
| 1 | 0 | 1,955,364 | 420,169 |
| 2 | 1 | 1,957,342 | 420,267 |
| 3 | 2 | 1,955,147 | 421,283 |
| 4 | 3 | 1,959,559 | 421,535 |
| 5 | 4 | 1,957,859 | 421,747 |

(Against the 2,000,000 / 400,000 the superseded draw targeted, which is
what 17 buckets was chosen to approximate.)

V0's un-gated draw recomputes the identical MD5 inline from the raw TLC
columns (`splits.RAW_TRIP_ID_SQL`), because the quality gate is what assigns
`trip_id` and V0 reads the source before the gate. Same bucketing rule, both
sides -- but note this still means **V0 and V1 are scored on different test
sets** (~429,000 un-gated rows against ~420,000 gated ones), so the V0→V1
difference measures gating's effect and the change of test-set composition
together. That is inherent to asking "what does the gate buy?", not an
artefact of the bucketing.

### Two model-capacity variants here too, reported side by side

As in Task A (previous section), and for the identical reason -- keeping
both tasks' methodology consistent, and pre-empting the question of why
only one task's model capacity would have been checked -- Task B's ladder
also runs under a second, **capacity-controlled** hyperparameter set,
selected once via expanding-window CV and then frozen for every rung.

One adaptation, stated plainly: Task B's V0 is deliberately sourced from
the **raw, un-gated** Parquet files and carries no `month` column, which
the CV folds need to stay time-ordered, so the search uses **`FEATURES_V1`**
(pickup hour, `pickup_geo_key`, trip distance, passenger count) -- the
ladder's own bare, no-engineering starting point once the gate and `month`
grain exist -- as its baseline feature set, rather than V0's literal raw
columns. This is the same idea as Task A's "V0's own baseline feature,"
adapted to the one place Task B's V0/V1 split (unlike Task A's) actually
differs in shape. The selected hyperparameters are still applied to V0's
own (raw-sourced) features when V0 is fit under this variant. To keep the
search's runtime reasonable it runs on **the first repeat's already-loaded
training bucket (1,955,364 rows)**, not the 40.4M-row full corpus: 4
expanding-window folds over the 10 training months (minimum 6 months'
training data, 1-month validation blocks), 8 candidate hyperparameter
combinations, each fit with early stopping.

**It runs once, not once per repeat.** Re-selecting hyperparameters per seed
would make the capacity-controlled variant a different model on every
repeat, and the spread across repeats would then confound model randomness
with hyperparameter churn rather than measuring it.

The winner: `num_leaves=31, learning_rate=0.10, min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0`,
frozen at **`n_estimators=182`** -- smaller than the original's 300, but
much closer to it than Task A's collapse to 12 (previous section). The
search took 709.9s (11.8 minutes) of this notebook's 3,907.8s (**65.1-minute**)
total runtime, which also covers 80 fits and five independent data loads.
That is well past the v3 prompt §9's 15-25-minute estimate, which was
written for a single pass of a single variant; five repeats of two variants
is roughly ten times that work, and the time is reported as measured rather
than the estimate defended.

Below, **± is one standard deviation across five repeats** (seeds 1-5, each
drawing its own deterministic bucket *and* using its own model
`random_state`, so the spread covers sampling variation and model
randomness together).

| Rung | Original MAE (s) | Original MAPE | Capacity-controlled MAE (s) | Capacity-controlled MAPE |
| --- | --- | --- | --- | --- |
| V0 -- raw baseline, un-gated source | 337.64 ± 2.08 | 78.09 ± 1.71% | 337.32 ± 2.19 | 77.56 ± 2.42% |
| V1 -- quality-gated | 336.36 ± 2.20 | 74.53 ± 2.50% | 335.99 ± 2.26 | 73.58 ± 2.48% |
| V2 -- + borough, service_zone, day_of_week, is_weekend | 330.48 ± 2.27 | 72.40 ± 2.13% | 330.77 ± 2.26 | 72.00 ± 2.47% |
| **V3a -- LEAKY, DO NOT TRUST** | 325.77 ± 2.23 | 70.76 ± 1.79% | 325.86 ± 2.24 | 70.32 ± 2.05% |
| V3b -- + point-in-time zone/hour average | 327.80 ± 2.22 | 74.22 ± 1.89% | 328.10 ± 2.25 | 73.84 ± 1.89% |
| **V4a -- + zone, one-hot (scipy.sparse)** | **324.10 ± 2.24** | 75.28 ± 1.77% | **323.46 ± 2.14** | 74.36 ± 2.01% |
| V4b -- + zone, point-in-time target encoding | 328.67 ± 3.34 | 76.75 ± 3.35% | 329.36 ± 3.83 | 76.43 ± 3.35% |
| V5 -- + weather | 331.96 ± 3.15 | 77.34 ± 3.36% | 332.49 ± 2.68 | 77.67 ± 3.26% |
| *REF_mean* (no model: training mean) | *646.96 ± 3.05* | *210.02 ± 2.27%* | — | — |
| *REF_heuristic* (no model: distance / mean speed) | *548.61 ± 2.65* | *46.74 ± 1.17%* | — | — |

(`outputs/tables/model_ladder_nyc.csv`,
`outputs/tables/ladder_paired_comparisons_nyc.csv`,
`outputs/figures/fig_model_ladder_nyc.png`.)

Unlike Task A, the two variants track each other closely at every rung --
under 1 percentage point of MAPE apart everywhere, and under 1 second of
MAE at most rungs -- rather than the roughly ₦48 gap Task A shows at V0
alone. **Task B's ladder shape is not sensitive to which of these two model
capacities is used; Task A's is.** That is itself a cross-task finding, not
only a methodology footnote.

### Fifteen of sixteen comparisons survive repetition -- and why the pairing is what makes that visible

**Fifteen of the sixteen** rung-to-rung comparisons -- eight per
hyperparameter variant -- held their direction across all five repeats. Task
A managed six of twelve. The single exception is V4b→V5 under the
capacity-controlled variant (4/5, discussed under V5 below); its
counterpart in the original variant does hold at 5/5.

The reason is worth stating, because it is the strongest practical argument
in this document for comparing **within seed** rather than between error
bars. Look at the table above: every rung's MAE has a spread of about
±2.2s, and the rungs sit within ~13s of each other, so several pairs of
error bars overlap substantially. A reader comparing marginal distributions
would conclude that most of this ladder is noise. But the repeats are
*paired* -- the same bucket and the same model seed for both rungs -- and
almost all of that ±2.2s is variation the two rungs experience **together**,
driven by which bucket was drawn. Subtracting within seed cancels it:

| Comparison (original variant) | Mean Δ MAE (s) | SD of the paired differences | Mean ÷ SD |
| --- | --- | --- | --- |
| V2 → V3a | −4.70 | **0.19** | −25.3 |
| V1 → V2 | −5.88 | 0.35 | −16.6 |
| V0 → V4a (best rung) | −13.53 | 0.82 | −16.5 |
| V3b → V4a | −3.69 | 0.31 | −11.7 |
| V3a → V3b | +2.02 | 0.26 | +7.9 |
| V4b → V5 | +3.29 | 1.20 | +2.8 |
| V4a → V4b | +4.56 | 1.91 | +2.4 |
| V0 → V1 | −1.28 | 0.82 | −1.6 |

The paired differences have standard deviations of 0.19 to 1.9 seconds
against marginal spreads of about 2.2 -- up to twelve times tighter.
**The same data supports either "mostly noise" or "nearly every step is
consistent", depending only on whether the comparison respects the
pairing.** It does here, and `tests/test_repeats.py` asserts that it does.

One caveat on that list: V0 → V1 is the weakest of the surviving effects.
Its mean difference (−1.28s) is only 1.6 times the spread of the differences
it averages, and it is the one comparison where the five repeats agreeing on
direction is doing most of the work. It should be read as "gating helped,
slightly and consistently", not as a large effect.

### Does the ladder beat the reference predictors? Yes on MAE -- and the metric matters

Every rung, both variants, beats both references in **5 of 5** repeats on
MAE. The best rung (V4a, 324.10s) beats the distance-over-average-speed
heuristic (548.61s) by **40.9%** and the training-mean predictor (646.96s)
by **49.9%**. Unlike Task A, this ladder is comfortably worth building.

But the two metrics disagree about the heuristic, and the disagreement is
instructive rather than a nuisance. **REF_heuristic's MAPE is 46.74%, lower
-- better -- than every rung in the ladder** (70-78%), while its MAE is 69%
worse than the best rung's. Distance over average speed predicts proportionally: a short trip
gets a small prediction, so its *percentage* error stays moderate. The
models minimise squared error, which is indifferent to relative error, and
they pay for that on the large population of very short trips where a
modest absolute miss is an enormous percentage one. Neither predictor is
simply "better": the heuristic wins on relative error, the models win
decisively on absolute error. Reported here rather than resolved, because
which one matters depends on what the prediction is for -- and because a
ladder that quoted only MAPE would have made the models look worse than a
one-line formula.

A caveat that belongs beside every MAPE figure regardless:
NYC trip durations are heavily right-skewed with a large population of very
short trips, so a MAPE in the 70-80% range is a property of the *target's
distribution*, not evidence the model is unusually poor -- a fixed absolute
error of even one minute is a large percentage error on a three-minute
trip. MAE in seconds is the more stable number to read rung-to-rung; MAPE
is retained because the v3 prompt requires it and because its *direction*
of movement between rungs is still informative, even where its absolute
level is not directly comparable to Task A's.

### V0 vs. V1: a real MAPE improvement, and an MAE/MAPE split worth explaining

Quality gating removed 748,565 of 41,169,720 raw rows (1.82%,
`docs/methodology_notes.md` §7). Across five repeats, V1 improves on V0 on
**both** metrics -- MAE 337.64 → 336.36s, MAPE 78.09% → 74.53% (original
variant) -- and the MAE improvement holds its direction in 5 of 5 repeats.
But it is the **weakest surviving effect in the task**: the paired mean
difference is −1.28s against a paired spread of 0.82s, a ratio of 1.6 where
every other surviving comparison in Task B exceeds 2.3 and most exceed 11.
It is a real effect, and a small one.

That it is small is the point, and the reason is specific: `negative_money`
accounts for 733,787 of the 748,565
rejected rows -- a fare-integrity defect that has nothing to do with
`trip_duration_seconds`, the column this task predicts -- while only
`non_positive_duration` (13,510) and `duration_exceeds_ceiling` (230)
actually corrupt the target itself (13,740 rows, 0.033% of the raw corpus).
Removing `negative_money` rows changes *which* trips are scored without
removing much of what actually corrupts this task's target, so the gate
buys a real but modest improvement, and it buys proportionally more on MAPE
than on MAE because the rows it removes are disproportionately ones whose
duration was a poor *percentage* match to begin with. This is the direct,
quantified counterpart to Task A's V0→V1 (§ above, "the ladder proving its
own honesty"): both tasks show that V1's payoff depends on what the gate
actually caught relative to what the task actually predicts, not on the raw
rejection rate alone.

One confound to state, since this pass is about not overclaiming: V0 and V1
are scored on **different test sets** (~429,000 un-gated rows against
~420,000 gated ones -- the gate removes rows from the test side too). The
V0→V1 difference is therefore the combined effect of training on cleaner
data *and* being scored on a cleaner test set, which cannot be separated
without scoring both models on a common set. That is inherent to the
question "what does the gate buy?" and applies equally to the pre-repeat
version of this result.

### V2: dimensional features help here, where they hurt in Task A

MAE improves from 336.36s to 330.48s (original variant), a paired mean
difference of **−5.88 ± 0.35s, consistent in 5 of 5 repeats**, and the
capacity-controlled variant agrees (−5.22 ± 0.39, also 5/5). At NYC's scale
(~1,956,000 training rows against 6 `service_zone` categories, ~7
`borough`/`day_of_week` combinations), there is enough repetition for
LightGBM to find real, generalisable structure in `borough`,
`service_zone`, `day_of_week` and `is_weekend` without overfitting to it --
the opposite of what happened to the same class of feature in Task A's
925-row panel, and true under *both* model capacities here, not just the
original one. Same idea, same fixed model, opposite verdict from Task A:
see the cross-task synthesis (§4).

### V3a (leaky) vs. V3b (correct): leaky looks *better* here -- the more dangerous direction

Unlike Task A, the expected textbook symptom holds, in both variants, and
it **survives repetition**: V3a looks *better* than the honest V3b on both
metrics -- original MAE 325.77s vs. 327.80s, MAPE 70.76% vs. 74.22%;
capacity-controlled MAE 325.86s vs. 328.10s, MAPE 70.32% vs. 73.84%. The
paired difference is **+2.02 ± 0.26s (5/5 agreeing)** in the original
variant and **+2.24 ± 0.34s (5/5)** in the capacity-controlled one: small
in absolute terms, but seven to eight times its own spread, so it is not
something a rerun would reverse -- and, this having been checked, a rerun
did not. NYC zone/hour traffic patterns are
comparatively stable across 2024's twelve months, so folding a few future
months into a "zone and hour's average duration" barely distorts it -- the
leak buys a small, spurious edge instead of an obvious, dramatic one.
**This is the more dangerous failure mode in practice**, because a leaky
feature that looks *worse* (Task A) is far more likely to be caught by a
practitioner running exactly this kind of before/after comparison; a leaky
feature that looks slightly *better* (Task B) is the one that quietly
ships. See §4 for the direct two-task contrast this enables, and the
mechanistic reason the two tasks point in opposite directions.

### V4: engaging Ayinla (2023), at 265 categories

**This is where the encoding question is actually decided.** Task A could
not resolve it -- on 37 categories and 925 rows the two encodings differ by
less than the seed does, and the finding previously claimed there has been
withdrawn (Task A, "V4: engaging Ayinla (2023)"). Task B, with up to 265
categories and ~1.96M training rows, resolves it cleanly and in both
hyperparameter variants:

**One-hot (V4a) beats the point-in-time target encoding (V4b)**, by a
paired **+4.56 ± 1.91s (5 of 5 repeats agreeing)** in the original variant
and **+5.90 ± 2.55s (5/5)** in the capacity-controlled one -- roughly 2.4
and 2.3 times its own spread. V4a is also the **best rung in the entire
Task B ladder** (324.10 ± 2.24s). Means: original MAE 324.10s vs. 328.67s;
capacity-controlled 323.46s vs. 329.36s. The spread here is wider than for
most Task B comparisons, so the effect is dependable in direction rather
than tightly pinned in size.

This is the opposite of the direction Ayinla (2023) argues for, at a
category count where a sparse expansion is supposed to be starting to show
its disadvantage. Two candidate explanations, offered as candidates: (1)
LightGBM's histogram splitter handles a 265-column sparse one-hot
efficiently, which is precisely the regime Ayinla's argument is weakest
against -- the case for compact encoding is strongest for learners without
native sparse/categorical handling, or at cardinalities far above this; (2)
`target_encode_pit` deliberately gives up signal for leakage safety,
smoothing toward a contemporaneous prior mean, a cost an encoder without
that discipline does not pay. **The honest scope of this finding is: on
LightGBM, at 265 categories, against a target encoding held to point-in-time
correctness, one-hot wins reproducibly.** It is not a general refutation of
Ayinla, and the one task that could not test the question properly is
reported as having failed to test it rather than folded in as agreement.

**An engineering note that belongs here, not only in
`src/modelling/encoders.py`.** V4a was, through two earlier
implementations, the one rung of either ladder that would not run at all: a
dense one-hot expansion (2,000,000 rows × up to 266 columns, >99% zeros by
construction) pushed this container's ~7.7GB memory ceiling past its limit
during LightGBM's own `Dataset` construction -- first via
`category_encoders.OneHotEncoder`, then via a hand-rolled
`pandas.get_dummies(dtype="int8")` DataFrame, then via
`pandas.get_dummies(sparse=True)` (which reports a tiny `memory_usage`,
but LightGBM's sklearn wrapper silently densifies a sparse-dtype
DataFrame before the booster ever sees it -- the crash was unchanged).
The fix was to stop building a DataFrame at all: `one_hot_encode` now
constructs a genuine `scipy.sparse.csr_matrix` and passes that directly to
`LGBMRegressor.fit`, which accepts sparse matrices as a first-class input
and keeps them sparse internally. Measured peak container memory for this
rung dropped from over 7GB (crashing) to under 1GB. This is recorded here
in the same spirit as every other engineering correction in
`docs/methodology_notes.md`: the failed attempts are part of the record,
not edited out once the working version existed.

**A second engineering correction, found the same way.** An earlier
version of both raw-Parquet and gated-`fact_trip` sampling queries applied
`USING SAMPLE n ROWS` to an *unfiltered* scan and only then applied the
`WHERE year/month` predicate, which silently returned far fewer rows than
requested (a 2,000,000-row request for 10 months' worth of data returned
roughly 1.66M rows once the 12-month sample was filtered down). Caught only
by checking `len(result)` against the requested sample size -- now asserted
explicitly in the notebook, not just eyeballed. The fix wraps the filter in
its own subquery so DuckDB samples *after* filtering, not before
(`src/modelling/`'s SQL is inline in the notebook, not a shared module,
since it is data-access glue rather than modelling logic).

### V5: weather does not help

In both variants, weather makes things worse on average: the paired
difference from V4b to V5 is **+3.29 ± 1.20s** in the original variant,
consistent in **5/5** repeats, and **+3.13 ± 2.65s** in the
capacity-controlled one, consistent in only **4 of 5**. Means: original MAE
328.67 → 331.96s, MAPE 76.75% → 77.34%; capacity-controlled 329.36 →
332.49s, 76.43% → 77.67%. V5 is still better than the raw V0 baseline in
absolute terms (both variants), but worse than the rung immediately before
it.

**This is the one Task B comparison that does not fully survive**, and it is
the only one in either variant where the direction flipped in a repeat. The
honest reading: weather did not help, the original variant says so
dependably, and the capacity-controlled variant says so in four repeats out
of five with a spread (2.65s) nearly as large as the effect (3.13s). Do not
report "weather hurts" as established under both variants; report it as
established under one and directionally consistent but not established under
the other.
Daily, city-wide precipitation and maximum temperature are too coarse a
signal to move an individual trip's duration once pickup hour, zone,
day-of-week and the zone/hour historical average are already in the model
-- whatever traffic effect weather has is plausibly already absorbed by
those finer-grained features. Reported exactly as measured, under both
model capacities: the v3 prompt (§4.4, §11) requires reporting whether
weather helps, not requiring that it does.

---

## Cross-task synthesis (`notebooks/03_synthesis.ipynb`, `outputs/figures/fig_ladder_comparison.png`)

### The headline: the two tasks differ in whether their ladders are *reportable at all*

This is the central result of the modelling layer, and it is a statement
about evidence before it is a statement about feature engineering.

| | Task A (Nigeria) | Task B (NYC) |
| --- | --- | --- |
| Comparisons that held direction in all 5 repeats | **6 of 12** | **15 of 16** |
| Best rung | V0 — *no engineered rung beat the baseline* | V4a (one-hot) |
| Best rung vs. the no-model heuristic | **loses by 23.4%** | **beats it by 40.9%** |
| Best rung vs. the training-mean predictor | beats it by 51.3% | beats it by 49.9% |
| Typical paired difference ÷ its own spread | 0.1 – 4.5 | 1.2 – 25.3 |

**Task B's ladder is a result. Task A's mostly is not.** Half of Task A's
comparisons are decided by the seed, its best rung is the one that does no
feature engineering at all, and the entire ladder loses to carrying last
month's price forward unchanged. Task B's ladder improves in a way that
holds its direction in every repeat at fifteen of sixteen comparisons, and
comfortably beats both no-model references.

Reporting a single number per rung would have hidden that difference
entirely: both tasks would have produced an orderly-looking table.

### Five contrasts, each stated with whether it survived repetition

Percent changes below are in mean MAPE relative to each task's own V0, using
the `original` hyperparameter variant; the direction-consistency counts come
from paired MAE differences (`ladder_paired_comparisons.csv`).

1. **V0→V1 depends on what the gate actually caught relative to what the
   task predicts, not on the headline rejection rate.** Nigeria: 0.0 pp,
   exactly, at every seed -- zero rejected rows, so V0 and V1 are the same
   fit on the same data (and that exact zero doubles as a control on the
   repeat harness). NYC: +4.6% in mean MAPE, and consistent in 5/5 repeats
   on MAE, but the *weakest* surviving effect in that task (paired mean
   1.6× its own spread) -- because 98% of NYC's rejected rows are excluded
   for a reason (`negative_money`) that does not touch the column this task
   predicts. **Survived in both tasks, and is small in both.**
2. **V2 (dimensional features) helps where sample size supports it and
   hurts where it does not.** Nigeria (925 training rows): **worse**, and
   the V1→V2 degradation is consistent in 5/5 repeats under both variants
   -- one of Task A's most dependable results, just not a favourable one.
   NYC (~1.96M training rows): **better**, −5.88 ± 0.35s, 5/5. Same class
   of feature, same fixed model, opposite verdict, **and both directions
   survive repetition.** A direct illustration of *n* mattering to whether
   a feature-engineering step pays for itself.
3. **V3a vs. V3b flips direction between a strongly-trending target and a
   comparatively stable one — and the flip is real in both directions.**
   Nigeria (prices +146% over the series): leaky is *worse* than honest,
   −10.74 ± 2.41 ₦, **5/5**. NYC (comparatively stable month-to-month
   traffic): leaky is *better*, +2.02 ± 0.26s, **5/5**. Both tasks, both
   variants, all repeats. This is the most robust cross-task finding in the
   layer: **leakage has no universal symptom**, and the NYC direction is
   the dangerous one precisely because it is the one a practitioner is
   primed to miss.
4. **V4's encoding comparison resolves in exactly one of the two tasks.**
   NYC (265 categories, ~1.96M rows): one-hot beats point-in-time target
   encoding by +4.56 ± 1.91s, **5/5**, in both variants, and V4a is the
   best rung in the task. Nigeria (37 categories, 925 rows): the two
   encodings differ by **less than the seed does** (3/5 agreeing, mean an
   order of magnitude below its own spread), so that task cannot speak to
   the question and the claim previously made there has been withdrawn.
   The result therefore rests on NYC alone, is scoped to LightGBM and to a
   target encoding held to point-in-time correctness, and is **not** offered
   as a general refutation of Ayinla (2023).
5. **Whether the ladder's shape depends on model capacity is itself
   task-specific.** In Nigeria the two hyperparameter variants disagree
   substantially (₦48 at V0 alone, against repeat spreads of ₦1-2) and
   disagree on whether a smaller model helps. In NYC they agree closely
   (under 1 pp of MAPE everywhere, under 1s of MAE at most rungs), and
   fifteen of sixteen comparisons hold under both. The mechanism (Task A, "the
   capacity-controlled result") is Task A's test period sitting in a price
   regime its training period only partly reaches -- a distribution-shift
   effect that model capacity is sensitive to -- against Task B's stable
   month-to-month demand pattern, which no capacity tried here struggles to
   generalise across. **A ladder's sensitivity to model capacity is not a
   fixed property of the method; it is a property of whether the target
   series keeps moving between training and test.**

### Why the pairing, not the error bars, is what made Task B reportable

Worth isolating, because it is a methodological result in its own right and
it generalises beyond this project. In Task B every rung's MAE carries a
spread of about ±2.2s across repeats, and the rungs span only ~13s in
total, so plotted as error bars the ladder looks largely inconclusive. But
almost all of that ±2.2s is variation the rungs experience **together** --
it is driven by which bucket of trips a repeat happened to draw. Comparing
within seed cancels it, and the paired differences have spreads of 0.19 to
1.9 seconds instead: up to twelve times tighter.

**The same five repeats support "mostly noise" or "nearly every step
consistent" depending only on whether the comparison respects the
pairing.** A ladder
that reports error bars without paired comparison understates what it
knows; one that reports neither, as this layer originally did, states more
than it knows. `tests/test_repeats.py` asserts the pairing arithmetic
specifically because everything above depends on it.

**What this comparison does not support.** Neither ladder tuned its
model's hyperparameters per rung, deliberately, since doing so would have
re-mixed "a better model" back into "better data," exactly the confound
fixing the model exists to remove (v3 prompt §3.3). Task A's mostly-negative
bars are therefore not evidence that dimensional modelling, historical
aggregates, or encoding choice are bad ideas in general. They are also,
having now tested it directly (Task A, "the capacity-controlled result"
above), *not* simply evidence that a 925-row training set and an untuned
300-tree gradient booster invites overfitting -- a deliberately
smaller, CV-selected model performs worse still, at every rung including
V0. The better-supported reading is that Task A's test period sits in a
price regime its training period only partly reaches, which limits how
much any of these features -- at any model capacity tried here -- can
improve on a naive lag-1 baseline. A production system would retune
alongside every new feature and might manage the extrapolation gap
directly (e.g. with a trend term); this ladder deliberately did neither,
because isolating the data engineering effect, not building the
best-possible forecaster, is what the exercise exists to measure.

---

## Summary of every engineering correction this layer's build required

Recorded here, in the same spirit as `docs/methodology_notes.md` §11's
"arithmetic errors" and §1's "correction recorded for the examiner,"
because the paper's modelling numbers depend on each of these having been
caught before publication, not after.

| Found | Symptom | Fix |
| --- | --- | --- |
| `category_encoders.OneHotEncoder` at NYC scale | Kernel killed (OOM) during V4a | Replaced with a hand-rolled encoder (below) |
| Dense `pandas.get_dummies(dtype="int8")` one-hot | Kernel killed (OOM) during LightGBM's `Dataset` build | Replaced with `scipy.sparse.csr_matrix` input to `LGBMRegressor.fit` |
| `pandas.get_dummies(sparse=True)` one-hot | Still killed -- LightGBM silently densifies a sparse-dtype DataFrame | Same `scipy.sparse` fix; peak memory 7GB+ → <1GB |
| DuckDB `... WHERE ... USING SAMPLE n ROWS` | Silently returned far fewer rows than requested (sample drawn before the filter) | Wrapped the filter in a subquery so `USING SAMPLE` draws from the already-filtered result; asserted row counts explicitly |
| DuckDB opened with the pipeline's 6GB default | Kernel killed (OOM) once pandas/LightGBM needed the rest of the container | Notebook 02 opens DuckDB with a 2GB ceiling and closes the connection once all data pulling is done, before the memory-heavier rungs run |
| `LGBMRegressor(n_jobs=-1)` | Over-subscribes threads against the container's cgroup CPU limit (4), not the host's core count | Fixed `n_jobs=4` in `LGBM_PARAMS`, identical across every rung of both ladders |
| `select_hyperparameters_cv`'s frozen `n_estimators` taken as the **median** best-iteration across all 5 CV folds | Folds range from 15 to 23 training months; a median is dominated by the smaller, earlier folds, freezing a tree count sized for less data than the ladder's actual 25-month fit uses | Freeze from the **mean of the two largest-training-window folds** instead (closest in size to the full fit); Task A's selected `n_estimators` moved 10 → 12 -- still small, and, once tested, still not the driver of the headline finding (see "the capacity-controlled result", Task A) |
| `USING SAMPLE n ROWS (reservoir, 796)` is **seeded but not reproducible** | The seed fixes the sampler, not the order rows arrive from a `read_parquet(glob)` scan, and reservoir sampling depends on arrival order too. Two runs of identical code drew different rows and reported different metrics (V0 MAE 335.67s vs. 332.88s) | Replaced with **content-addressed bucketing**: `('0x' \|\| substr(trip_id,1,8))::UBIGINT % 17 = k`, where `trip_id` is already a deterministic MD5 of the row's business content. Selection now depends on nothing but the data. `tests/test_sampling_determinism.py` proves order-independence and cross-process stability. Full account in `docs/methodology_notes.md` §12 |
| Notebook 02's pulls had **no `ORDER BY`** | A deterministic row *set* is not a deterministic row *order*: DuckDB's parallel scan returns rows in no guaranteed order, and gradient boosting is order-sensitive (bagging selects by position, split ties break by arrival order). Two processes given provably identical rows -- same count, same target checksum, same distance sum -- fitted different models, and one rung-to-rung conclusion moved between runs. LightGBM's own `deterministic=True` did **not** fix it, the cause being upstream of the learner | Explicit `ORDER BY trip_id` (and the identical inline MD5 on the raw side) on every pull. Verified: two processes then produce byte-identical predictions, and notebook 02 run four times end to end produces byte-identical tables. Guarded by `tests/test_sampling_determinism.py`. Task A never had the defect -- its query was already ordered, which is why it reproduced throughout. Full account in `docs/methodology_notes.md` §12 |
| Every rung reported as a **single point estimate** | Task B's headline was a ~3.4% MAE improvement against a run-to-run variation that had never been measured, so it could not be distinguished from noise -- and Task A's rungs were being compared on differences later shown to be smaller than the seed | Every rung of both variants of both tasks now runs **five times**, with paired within-seed comparison and reported spread (`src/modelling/repeats.py`, `tests/test_repeats.py`). **Outcome: Task B's headline survived** (V0→V4a, −13.53 ± 0.82s, 5/5 repeats, now 4.0%), **Task A's encoding claim did not** (3/5) and was withdrawn. Six of Task A's twelve comparisons are decided by the seed |
| The ladder had **no absolute anchor** | Rung-to-rung movement says nothing about whether the error level is any good; Task A's ladder looked orderly while sitting *above* a one-line benchmark, which no rung-to-rung number could reveal | Two no-model reference predictors per task (`src/modelling/baselines.py`), scored by the same code as every rung. For Task A this produced the single most important result in the task |

The first six fixes are infrastructure -- how the data reaches the model,
not what the model is told -- so none of them touches the "fixed model, only
the data changes" discipline §0 describes; the model class and
hyperparameters in `LGBM_PARAMS` are unchanged by all of them. The last
three are different in kind: they did not change what was computed, they
changed **what could honestly be claimed about it**, and two of them
retired claims this document had already made.

## Limitations

1. **Hyperparameters are fixed within each variant, not tuned per rung, by
   design.** Absolute error levels should not be read as either variant's
   best achievable model; both are held fixed across every rung of both
   tasks specifically so a rung's error change is attributable to its one
   added data engineering technique, not to a retuned model (v3 prompt
   §3.3). Task A's capacity-controlled variant is itself selected once, on
   V0 alone, via CV (`src/modelling/tuning.py`) -- not tuned per rung
   either. See the cross-task synthesis, final paragraph.
2. **Task B is a deterministic sample, not the full corpus.** Each repeat
   works on one content-addressed bucket of the corpus (~1,956,000 training
   and ~421,000 test rows, `src/modelling/splits.trip_bucket_predicate`,
   `docs/methodology_notes.md` §12), not on all 40.4M rows -- a deliberate
   tractability trade-off (v3 prompt §4.3), not a claim that these are the
   full-corpus-optimal error figures. Within a repeat every rung trains on
   identical rows, and across repeats the bucket changes deterministically,
   so the spread reported is sampling variation that has been measured
   rather than assumed away.
3. **Task A's V0-vs-later-rungs result is diagnostic of a test period that
   sits in a price regime its training period only partly reaches, not a
   general claim that feature engineering does not help petrol price
   forecasting.** Tested directly against a plain model-capacity
   explanation (a deliberately smaller, CV-selected model performs worse
   still, at every rung including V0) and not confirmed as the driver. See
   "the headline finding" and "the capacity-controlled result" above.
4. **MAPE on NYC trip duration is inflated by the target's right-skewed
   distribution** (many very short trips), and its absolute level should
   not be compared against Task A's MAPE, only its *direction of movement*
   within Task B's own ladder.
5. **V4's encoding comparison is scoped to LightGBM, to the two category
   counts tested (37, 265), and to a target encoding held to point-in-time
   correctness** -- and, on Task A, it does not resolve at all: the two
   encodings differ by less than the seed does, so nothing about Ayinla
   (2023) can be concluded from that task. Where the comparison does
   resolve it is reported as a result about *this* setup, never as a
   general refutation of that paper.
6. **V5 exists only for Task B.** No verified external covariate for
   Nigeria exists in this project's sources; none was manufactured to fill
   the slot (v3 prompt §3.4).
7. **Five repeats is a small number, and is treated as one.** The spread
   reported per rung is estimated from five observations, so it is itself
   uncertain; the direction-consistency counts are ordinal evidence, not
   probabilities. No p-value or confidence interval is reported anywhere in
   this layer, because neither would mean what a reader would take it to
   mean at n=5. The honest reading of "5/5 agreed" is "this difference was
   robust to everything we varied", not "this difference is significant".
8. **Only two sources of variation were measured.** The repeats vary the
   model's randomness and, for Task B, which rows were sampled. They do not
   vary the train/test boundary, which is fixed by the forecasting question,
   nor the hyperparameters, which are fixed by design. A rung's error could
   move for reasons outside both, and the reported spread does not cover
   those.
9. **Reproducibility is demonstrated within this pinned environment, not
   claimed universally.** Both notebooks reproduce byte-identically when
   re-executed in this container, which is what four independent
   end-to-end runs of notebook 02 established. The bucketing rule is
   deliberately version-independent — it reads the stored `trip_id` string
   rather than calling a library hash function — but the MD5 values
   themselves come from the silver build, and LightGBM's fitted output is
   only guaranteed stable for a given library version on a given row order.
   A different DuckDB or LightGBM release could shift the figures without
   any of the reasoning here changing. The guarantee offered is "re-running
   this reproduces these numbers", not "any environment produces these
   numbers".
