# Methodology Notes

Every decision in this document is one where the data did not permit the obvious
approach, and where taking the obvious approach anyway would have produced a
number that looked fine and was wrong. Each is recorded with what was done, why,
and what it costs.

---

## 1. The unbalanced panel, and why A6 is a chained index rather than a fixed basket

### The problem

The WFP Nigeria price panel is **unbalanced**: the set of markets reporting in a
given month changes from month to month. A national `AVG(price)` over such a
panel moves whenever composition moves, even if no price changed anywhere. If
expensive urban markets enter the panel one month, the naive mean rises — and
that rise is a measurement artefact, not inflation.

### Why a fixed basket was not constructible

The specification asked for a fixed basket or a chained index, with January 2016
as the base. A **true fixed basket anchored to January 2016 is not available on
this panel**: very few (market, commodity, unit, price type) cells report
continuously from 2016 through 2026. A basket restricted to continuously
reporting cells would discard most of the data, and the handful of cells left
would be unrepresentative of the country.

### What was done instead

A **chained matched-model (Jevons) index**. For each pair of consecutive months:

1. identify the cells observed in **both** months — the matched model;
2. take the unweighted **geometric mean** of their price relatives — the Jevons
   formula, standard in official statistics for elementary aggregates without
   quantity weights;
3. chain the resulting monthly links into a continuous series;
4. rebase so that January 2016 = 100.

Composition therefore never enters a single link, while the series still uses
every market in every month it reports.

### Gaps and chain breaks: nothing is carried forward

Two linking rules, both in `config/settings.py`, govern what happens when the
panel thins out:

- **Bridging with observed prices only.** A cell's relative is formed against
  that cell's own most recent *observed* price, provided it is at most **3
  months** old (`PRICE_INDEX_MAX_LINK_GAP_MONTHS`). A cell that skips a month
  therefore still contributes, and the relative it contributes is a real
  observed change, never an estimate. The chain level stays exact; the
  single-month change column becomes "change since prior observation", and
  `longest_bridge_months` shows when that happens.
- **Chain breaks.** A month whose link rests on fewer than **3** matched cells
  (`PRICE_INDEX_MIN_MATCHED_CELLS`) is a break, not a price change. The chain
  restarts there as a new segment, and `chained_index` is published **only for
  the segment that links unbroken to January 2016**. `chain_segment` and
  `is_linked_to_base` are on every row.

**A correction recorded for the examiner.** The first version of this index
treated a month with no matched cells as a relative of 1.0 — "no change" — and
chained straight through it. That is carry-forward interpolation, which
constraint 2.5 forbids, and it was caught at figure review: it drew one
category as a perfectly flat line for eleven years. It was replaced by the rules
above before any figure was published.

On the live panel the effect is concrete, and it narrows what A6 can claim:

| Category | Months linked to Jan 2016 | Last linked month |
| --- | --- | --- |
| cereals and tubers | 127 | 2026-07 (unbroken) |
| pulses and nuts | 85 | 2023-01 |
| oil and fats | 85 | 2023-01 |
| meat, fish and eggs | 1 | 2016-01 |
| milk and dairy | 1 | 2016-01 |
| vegetables and fruits | 1 | 2016-01 |
| miscellaneous food | 1 | 2016-01 |

Only **cereals and tubers** supports a composition-controlled series from the
specified base to the present. Pulses and oils break when the panel contracts to
the North East in early 2023. The four remaining categories do not report
continuously in the months following January 2016, so no honest chain connects
them to that base; their observations remain in the CSV with
`is_linked_to_base = false`. The specified base period therefore suits one
category in seven. A later base would link more categories, but choosing the
base to flatter coverage would be a post-hoc decision, so the specified base is
kept and the limitation reported.

### What it costs, stated plainly

- **The index is unweighted.** WFP publishes no consumption quantities, so there
  are no expenditure weights. A Laspeyres index would require data this study
  does not have. Consequently the index describes *price behaviour in the
  sampled markets*, **not** the cost of a representative household's basket, and
  must never be described as a cost-of-living measure.
- **Chain drift.** Any chained index can drift relative to a direct comparison
  when prices oscillate. Over a strongly trending series such as this one the
  effect is small, but it is a known property, not an absence of one.
- **Link quality varies.** `matched_cells` is published on every row of
  `A6_food_price_index.csv` and plotted in the lower panel of the figure,
  precisely so a month whose link rests on few matched cells is visible as such.

The naive unweighted mean is computed **on the same rows, with the same
rebasing**, and plotted alongside. The gap between the two lines is attributable
solely to the composition correction, and reporting that gap is itself a finding.

---

## 2. Three "missing" NBS months that were never missing

The specification recorded three verified gaps in the NBS fuel panel: 2024-08,
2025-10 and 2025-11.

**None of them is a gap.** All three months are published, and the assembled
panel is complete: **1,147 observations = 31 months × 37 federating units, with
no month absent between 2023-11 and 2026-05.**

The three phantom gaps were an artefact of the parser, not a property of the
source. The specification's parsing strategy scans the first six rows of each
sheet for the `State` header. Several releases — `PMS_March_ 2026.xlsx`,
`PMS_OCT_2025.xlsx`, `Fuel_NOV_2025.xlsx` — place that header on **row 14**,
behind a title banner followed by a block of empty rows. Under a six-row scan
those files yield zero observations silently, and the months they alone publish
as *current* appear to be missing.

Widening the scan to 25 rows recovered them. The pipeline now reports the
difference between the claimed gaps and the observed ones on every run
(`gaps_claimed_but_not_observed`), rather than adopting either figure on trust.

**The methodological point for the paper:** a parser assumption manufactured
three data gaps that a reader would have attributed to the statistical agency.
Nothing in the output would have looked wrong. Only the arithmetic —
1,073 observations not dividing evenly into a rectangular panel — revealed it.

---

## 3. A silent corruption in the NBS workbooks: trailing commentary blocks

Several NBS releases append commentary tables *below* the state table —
"STATES WITH THE HIGHEST AVERAGE PRICES" and its counterpart — which repeat a
subset of states under the same column headers but with different column
meanings.

A header-anchored unpivot reads them as data. In the January 2026 release this
placed six states' **2026-01** prices into the **2025-01** column, producing
duplicate `(month, state)` observations with plausible-looking but wrong values.
The release yielded 117 observations where every other yielded 111.

The parser now terminates the table body at the first **repeated state**, since
the state table carries each federating unit exactly once, and additionally on a
recognised commentary banner. Both terminations are recorded per sheet in the
run manifest (`body_truncated_at`) rather than applied silently.

Without the reconciliation test and the rectangularity check, this corruption
would have reached the paper.

---

## 4. A7 is a within-market analysis, not a national one

The obvious design — regress food prices on petrol prices around the May 2023
subsidy removal — **is not available on this data**, for two independent reasons:

1. **Source D begins 2023-11**, six months *after* the break. It contains no
   pre-break observation at all.
2. **Source C's own fuel commodities** (`Fuel (diesel)`,
   `Fuel (petrol-gasoline)`) carry a verified coverage gap running from early
   2023 into 2025, so they cannot bridge the period either.

Manufacturing a pre-break fuel series by extrapolation or interpolation would be
fabrication, and constraint 2.5 forbids it.

A7 therefore describes the *food* series alone around a known date.

### Why it cannot be a national before-and-after

WFP's retail panel changes composition at almost exactly the break: 14 states
report up to January 2023, and only Borno, Yobe and Adamawa afterwards. A
before-and-after over the whole panel would compare a 14-state panel with a
3-state one. An earlier version of A7 did exactly that; it has been withdrawn.

### The fixed panel

A7 is now restricted to the markets that report **continuously on both sides of
May 2023**, defined as at least one retail staple observation in *every* month of
a symmetric window around the break.

The window cannot be wide. North-eastern reporting has a **panel-wide hole from
June 2022 to January 2023**: only five markets report through it, and four of
those stop in January 2023. No market is continuous over any window that reaches
back before February 2023. The longest symmetric window in which a continuous
panel exists at all is **February to September 2023: four months either side**
(pre-break: February to May 2023, with the announcement month counted as
pre-break; post-break: June to September 2023).

**The fixed panel is 10 markets in 2 states, all in the North East:**

| State | Markets |
| --- | --- |
| Borno (6) | Abba Gamaram, Baga Road, Budum, Bullunkutu, Custom, Monday |
| Yobe (4) | Geidam, Gujba (Buni Yadi), Yunusari, Yusufari |

Adamawa contributes no market: its reporting markets begin only in June 2023.
The panel is computed by the query, not listed by hand, and
`tests/test_a7_scope.py` fails if the counts stated in the SQL header ever
diverge from the counts the query produces.

### Results, and how much weight each can bear

| Measure | Value | Weight it can bear |
| --- | --- | --- |
| **Within-cell level change** — same 96 market-commodity cells, geometric mean of (mean log price after − mean log price before) | **+40.0%** (interquartile range +18.8% to +59.5%) | The robust number: each cell is its own control, and it does not depend on a fitted line |
| Pre-break trend | +4.49 index points / month (R² 0.93, n = 4) | Fragile: four points |
| Post-break trend | +11.92 index points / month (R² 0.89, n = 4) | Fragile: four points |
| Slope change | +7.43 points / month | Fragile |
| Level shift at the break | +1.98 points | Fragile, and sensitive to curvature |

What A7 can say: *within these ten north-eastern markets, staple retail prices
in the four months after the announcement were on average about 40% higher than
in the four months before, and rising faster.* What it cannot say: anything about
Nigeria, and anything about why. The window also contains the June 2023 naira
devaluation, the northern lean season and continuing insecurity in exactly these
markets. **No causal claim is made.**

---

## 5. A9 is a null result, and is not presented as a finding

**A9 (fuel-to-food passthrough) has been demoted from the results to this note
and to the Limitations section below. It is reported as a null result.**

- WFP had stopped reporting **11 of the 14 states it monitored by January 2023**
  (ten of them in January 2023 itself; Sokoto in April 2019).
- The analysis therefore rests on **3 states — Borno, Yobe and Adamawa — all
  located in the North East**. That is 3 of the country's 37 federating units, in
  one of its six geopolitical zones.
- In those conflict-affected north-eastern markets, the correlations between
  monthly changes in state petrol prices and in staple food prices are
  **under ±0.09 at every lag**:

  | Lag (months) | Correlation of log changes | State-month pairs |
  | --- | --- | --- |
  | 0 | −0.067 | 64 |
  | 1 | −0.025 | 64 |
  | 2 | +0.030 | 63 |
  | 3 | +0.086 | 61 |

- **They cannot support any inference.** Even with no allowance for serial
  correlation or for clustering within only three states, nominal 5%
  significance at these sample sizes would need |r| of about 0.25; every
  coefficient falls far inside that. Allowing for the dependence structure would
  raise the bar further. The data show no detectable association, and the design
  could not have detected a modest one.

**What A9 describes.** A9 is a finding, and only a null one, about
**conflict-affected north-eastern markets in Borno, Yobe and Adamawa**, observed
during a period of acute food insecurity under humanitarian monitoring. It is
not a statement about the country as a whole, and it must never be cited as one.

The outputs are **kept but relabelled as exploratory**: the SQL file opens with
an "EXPLORATORY — NULL RESULT — NOT A FINDING" banner, every row of
`A9_fuel_food_passthrough.csv` carries
`analysis_status = 'EXPLORATORY - null result; supports no inference'`, the
figure is titled and captioned as a null result and marks the unadjusted
significance bar, and the run manifest records the analysis's status as
exploratory. The file names are unchanged so that existing references resolve.

The detail below is retained as the record of why coverage is this thin.

**State coverage — far narrower than the source's headline suggests.** Nigeria
has 37 federating units (36 states plus the FCT). Source D covers all 37 from
2023-11. Source C lists 14 states *historically*, but its footprint contracted
sharply: **11 of the 14 stop reporting by January 2023** (Sokoto as early as
April 2019). Across the whole overlap window only **three** states report —
**Borno, Yobe and Adamawa**, all in the North East, where WFP's humanitarian
monitoring continued.

| WFP state | Rows, all time | Last month reported | Retail staple rows since 2023-11 |
| --- | --- | --- | --- |
| Borno | 30,974 | 2026-07 | 4,210 |
| Yobe | 28,502 | 2026-06 | 2,707 |
| Adamawa | 2,715 | 2025-11 | 211 |
| Katsina, Kano, Jigawa, Abia, Oyo, Lagos, Zamfara, Kebbi, Kaduna, Gombe | 2,162 – 4,041 each | 2023-01 | 0 |
| Sokoto | 618 | 2019-04 | 0 |

A9 therefore rests on **3 of 37 federating units (8.1 per cent) in a single
geopolitical zone**. Five of six zones contribute no food data at all, and the
`ALL ZONES` row and the `North East` row are the same three states. An earlier
draft of this note stated "about 14 states, 38 per cent"; that figure was the
source's historical coverage, not its coverage over the window A9 actually
uses, and it was corrected once the per-state reporting dates were queried.

The result describes **conflict-affected north-eastern markets** during a period
of acute food insecurity. It does not describe Nigeria, and the paper must not
present it as a national estimate. To make that impossible to overlook:

- `n_pairs`, `n_states` and `n_months` appear on **every** output row;
- `states_in_overlap`, `states_with_fuel_data`, `states_with_food_data` and
  `pct_of_federating_units` are repeated on every row, so no downstream consumer
  of the CSV can use a correlation without seeing what it rests on;
- cells below the minimum pair threshold are emitted with a **NULL** correlation
  rather than a number computed from too little data;
- the figure carries a dedicated sample-size panel beneath the correlation panel,
  and the coverage statement is printed inside the plot area.

**The result does not generalise to Nigeria as a whole**, and the paper must say
so wherever it is cited.

**Three further caveats:**

- **Independent sources, different methods.** NBS surveys fuel retail outlets and
  publishes a state mean; WFP surveys selected markets and publishes
  per-commodity quotes. No shared sampling frame, reference week or instrument.
- **Logs, not levels.** Both series are differenced in logs before correlating.
  Price *levels* in this window trend strongly, and correlating levels yields a
  large coefficient reflecting nothing but shared trend. The level correlation is
  reported alongside, explicitly labelled `corr_levels_spurious_prone`, so the
  contrast is visible rather than hidden.
- **Association, never causation.** There is no instrument, no control group and
  no exogenous variation. A positive lagged correlation is equally consistent
  with fuel costs passing through to food and with both responding to a common
  macroeconomic shock — the naira devaluation being the obvious candidate.

**The market→state bridge.** `fact_market_price_monthly` sits at the *market*
tier and `fact_fuel_price_monthly` at the *state* tier, so A9 joins them by
hopping `market → parent_geo_name → state` through `dim_geography`'s ragged
hierarchy, with names normalised through `config/nigeria_states.py`. This bridge
is a direct cost of the single-conformed-geography design and is discussed in
[`erd_star_schema.md`](erd_star_schema.md).

---

## 6. Retail only, and why unit is part of the commodity grain

**Retail only** for A7, A8 and A9. WFP publishes both Retail and Wholesale
quotes; Retail is the price a household actually faces, it is far better covered
(66,232 observations against 21,152), and mixing the two would produce a series
that moves whenever the retail/wholesale mix moves — the same composition bias
the index construction exists to eliminate.

**Unit is part of `dim_commodity`'s grain.** WFP prices the same commodity in
different units across markets: `KG`, `100 KG`, `50 KG`, `2.5 KG`, `100 Tubers`.
A price without its unit is meaningless, and collapsing on commodity alone would
average 1 KG of rice with 100 KG of rice. This is why `dim_commodity` holds 67
members across 44 commodity names.

**The staple basket is defined by WFP's published category**
(`cereals and tubers`, `pulses and nuts`) rather than by a list of commodity
names. Verified against the live panel, those two categories cover every
commodity in the intended basket plus its spelling variants — `Cowpeas` for
`Beans`, `Rice (milled, local)` for `Rice (local)`,
`Cassava meal (gari, yellow)` for `Gari (yellow)`, `Yam (Abuja)` for `Yam`. A
hard-coded name list would be silently emptied by a WFP rename; a category
filter would not.

---

## 7. The NYC 2024 corpus: what was rejected, what was kept, and why

Raw input: **41,169,720 rows.** Loaded to `fact_trip`: **40,421,155.**
Rejected: **748,565 (1.82%)**, each attributed to exactly one reason so the
per-reason counts sum to the total:

| Reason | Rows |
| --- | --- |
| `negative_money` (fare or total below zero) | 733,787 |
| `non_positive_duration` (dropoff at or before pickup) | 13,510 |
| `implausible_distance` (over 500 miles) | 1,038 |
| `duration_exceeds_ceiling` (over 24 hours) | 230 |

`negative_money` dominates, and it is not noise: TLC's own dictionary describes
voided and disputed trips, which are published with negative amounts. They are
excluded from the fact because a revenue measure containing refunds is not a
revenue measure — and the count is reported prominently rather than buried,
because 1.78% of a corpus is a material exclusion.

### Out-of-window timestamps are retained, not rejected

The 2024 files genuinely contain pickups stamped outside calendar 2024 — the
observed extremes are **2002-12-31** and **2026-06-26**. This is a defect of the
official data, and constraint 2.5 requires quantifying rather than erasing it.

- **56** raw rows carry an out-of-window pickup.
- **55** survive structural rejection and are **loaded**, landing in partitions
  for 2002, 2008, 2009, 2023, 2025 and 2026. One of the 56 also failed a
  structural rule and was rejected under that reason.
- **38** of those fall outside `dim_date`'s 2016-2026 span and therefore resolve
  to `pickup_date_key = -1`, the Unknown member, which keeps referential
  integrity total while leaving the anomaly countable.
- Quality rule **Q061** reports the count every run as a `warn`, not a `fail`,
  because these are real published rows.

Every analysis restricts to the declared window explicitly, so the defect is
never silently mixed into a seasonal profile.

### Duplicate business keys are reported, not removed

TLC publishes no trip identifier, so `trip_id` is a deterministic MD5 over the
row's business content. **Four** rows collide (0.000010%).

They are **retained**. Rows that are field-for-field identical cannot be
distinguished from two genuinely coincident trips without a vendor-issued
identifier, and deleting them would assert an unverifiable claim about the
world. The rate is reported by rule Q083 as a `warn`.

---

## 8. Fields that fail as a block (analysis A11)

Quality rules Q012, Q013 and Q024 independently reported a null rate of
**9.801145 per cent** on three unrelated fields: `passenger_count`,
`congestion_surcharge` and `RatecodeID`. Three different fields agreeing to six
decimal places is a signature, not a coincidence.

A11 was added to test it, and the result is sharper than the hypothesis. Of the
16 possible combinations of four fields (`passenger_count`,
`congestion_surcharge`, `RatecodeID`, `airport_fee`), **exactly two occur** in
40.4 million trips:

| Pattern | Trips | Share |
| --- | --- | --- |
| all four present (`----`) | 36,459,364 | 90.20% |
| all four absent (`PCRA`) | 3,961,736 | 9.80% |

Not one trip is missing some of the four but not others. The block rate varies
by month from roughly 4 to 14 per cent, across both major vendors, and reaches
100 per cent for vendor 6.

This matters for the engineering argument. If the fields were missing
independently, a per-column imputation strategy could be argued. Because they
fail as a block, the rows are arriving through an upstream path with a reduced
schema, and **no per-column treatment is defensible**. The correct response is to
model them as a distinct record class — which is exactly what routing them to
each dimension's Unknown member does.

This is also why the referential-integrity rule was rewritten to separate
**orphans** (a key absent from the dimension: a genuine integrity breach, always
a failure) from **Unknown-member routing** (a null source field absorbed by
design: a rate to report). Conflating the two either hides real breakage behind a
tolerance or fails the build over data the source never published.

---

## 9. Tips are only interpretable for card payments

The TLC dictionary states the tip field is populated for credit-card tips and
**does not include cash tips**. A cash trip showing a zero tip is evidence about
the recording system, not about the passenger.

A4 handles this openly rather than by quiet filtering: every payment type is
reported so the structural zeros are visible; `dim_payment_type.is_tip_observable`
marks the one type for which a tip statistic means anything; `is_interpretable`
is carried on every output row; and the figure plots only interpretable rows
while captioning the exclusion. **No blended "average tip across all payment
types" is produced anywhere**, because that statistic would be fabricated.

---

## 10. Speed is a congestion proxy, not a congestion measurement

A3 uses mean journey speed as a proxy for road congestion. It is not a
measurement of it. Speed on a taxi trip confounds route choice and trip purpose,
the distance mix (short trips carry proportionally more time stationary at kerbs
and lights), metered distance error, and time stopped with the meter running for
reasons unrelated to traffic.

A fall in mean speed is *consistent with* worsening congestion; it is not
evidence of it. The assumption is stated in the SQL header, in the figure caption
and here. Both a per-trip mean and a distance-weighted fleet speed are reported,
because they answer different questions and they diverge.

Trips with speeds above 100 mph (~0.027%, rule Q045) are excluded from the speed
statistics but counted, since a single bad odometer reading moves a borough-hour
mean noticeably. They are flagged rather than rejected at the silver boundary,
because an implausible speed usually indicates a bad measurement on an otherwise
valid trip.

---

## 11. Two arithmetic errors in the build specification

Recorded because the paper's counts depend on getting them right.

1. **"37 states plus FCT"** implies 38 federating units. Nigeria has **36 states
   plus the FCT = 37**. `config/nigeria_states.py` enforces 37 with an assertion,
   and the geopolitical zone map is derived from a single source so the two
   cannot drift.
2. **"1,092 state-month observations, 28 months"** implies 39 labels per month,
   which matches neither 37 nor 38. The assembled panel is **1,147 = 31 × 37**,
   exactly rectangular. The specification's figure appears to have been a
   pre-filter count taken under the six-row header scan described in note 2.

---

## 12. A sampling defect that made the NYC modelling figures irreproducible

Recorded in full because it invalidated a claim this project had already
written down, and because the way it was found is the reason it was caught at
all.

### The defect

Task B of the modelling layer (`notebooks/02_nyc_trip_duration.ipynb`) works
on a sample of the 40.4M-row NYC corpus rather than the whole of it. That
sample was originally drawn with DuckDB's reservoir sampler:

```sql
SELECT * FROM ( ... WHERE year = 2024 AND month IN (...) )
USING SAMPLE 2000000 ROWS (reservoir, 796)
```

The seed fixes the sampler's own randomness. It does **not** fix which rows
reach the sampler, or in what order — and reservoir sampling's output depends
on arrival order as much as on the seed. The rows arrive from a
`read_parquet('data/raw/nyc/*.parquet')` glob, whose scan order across files
and row groups is not contractually stable between container runs. So the
query was seeded, looked reproducible, and was not.

### How it surfaced

Not by inspection. Notebook 02 was re-executed after an unrelated change, and
its rung V0 reported MAE 332.88s where the previous execution of the same
code had reported 335.67s. Same seed, same SQL, same image. The only
explanation that survived was that the two runs had sampled different rows,
which is what a check of the realised row counts and checksums confirmed.

### Why it mattered enough to stop and fix

The ladder's headline for Task B was an improvement of roughly 3.4% in MAE
between the baseline rung and the best one. Run-to-run variation had never
been measured, so there was no basis on which to claim 3.4% was an effect
rather than the spread. An effect smaller than an unmeasured noise floor is
not a finding, and a reviewer would have said so immediately.

### The fix: select on content, not on position

Rows are now assigned to a fixed bucket by a hash **of the row's own
business content**, so selection cannot depend on scan order, thread count,
file layout or anything else outside the data:

```sql
WHERE ('0x' || substr(trip_id, 1, 8))::UBIGINT % 17 = <bucket>
```

`trip_id` is already a deterministic MD5 over each trip's business attributes
(`sql/ddl/silver_nyc_trip.sql`), so its leading hex digits are a uniformly
distributed number that belongs to the row permanently. Two details are
deliberate:

- **The hex prefix is parsed rather than calling DuckDB's `hash()`.** The
  former depends only on the stored string; the latter depends on a
  library-internal hash that a DuckDB upgrade could legitimately change,
  which would silently reshuffle every sample.
- **The raw, un-gated draw (rung V0) recomputes the identical MD5 inline**
  from the same raw columns (`src/modelling/splits.RAW_TRIP_ID_SQL`), because
  the quality gate is what assigns `trip_id` and V0 reads the source before
  the gate. Same rule, same determinism, both sides.

**17 buckets** was chosen so a bucket lands near the sizes the reservoir draw
had targeted, keeping the new figures comparable to the old ones. The
realised sizes are ~1,956,000 training and ~421,000 test rows per bucket
against targets of 2,000,000 and 400,000; the notebooks **report the realised
counts rather than assuming them**, because a bucket's size is a property of
the data.

### A second defect, found only because the first fix was verified

Content-addressed bucketing made the selected *set* of rows reproducible.
The notebook still did not reproduce.

This was caught by re-running notebook 02 end to end and diffing its outputs
against the previous run — the verification step, not the fix, is what
surfaced it. The diff was diagnostic in a useful way. `REF_mean`, a
predictor that depends on nothing but which rows were selected, was
**bit-identical**. So was the row count, the checksum of the target column
and the sum of trip distances. The rows were provably the same rows. But the
fitted models differed, and one rung-to-rung conclusion moved with them: a
comparison reported as consistent across all five repeats in one run came
back consistent in only four of five in the next.

The cause is that a deterministic row *set* is not a deterministic row
*order*. DuckDB's parallel scan returns rows in no guaranteed order, and
gradient boosting is order-sensitive: bagging (`subsample=0.8`) selects rows
by position, and histogram split ties break by arrival order. Sums are
order-independent, which is exactly why the checksums agreed while the models
did not.

Measured directly at full scale, two processes reading the identical bucket:

| Pull | Process 1 | Process 2 |
| --- | --- | --- |
| No `ORDER BY` | 202,240,914 | 203,933,890 |
| `ORDER BY trip_id` | 208,095,622.694010 | 208,095,622.694010 |

(Sum of test-set predictions from a booster fit on 1,955,364 rows with a
fixed seed.) Note also that ordering changes the *value*, not merely its
stability — the ordered result is not either unordered result, so this was
never a matter of picking whichever run to believe.

LightGBM's own `deterministic=True` / `force_row_wise=True` was tried first
and **did not fix it**, which is consistent with the cause being upstream of
the learner: the booster was faithfully fitting whatever order it was handed.

The fix is an explicit `ORDER BY` on every pull in notebook 02 — `trip_id`
for the gated source, the identical inline MD5 for the raw one. Task A never
had this defect: its query already carried `ORDER BY g.geo_name, f.month_key`,
which is why Task A reproduced bit-identically across runs throughout.

**The general lesson, which is the reason this is recorded at length:**
reproducible sampling is necessary but not sufficient for a reproducible
model. Any pipeline that feeds a database query straight into an
order-sensitive learner needs the query to be ordered, and will otherwise
produce results that look reproducible — same seed, same rows, same
checksums — and are not.

### Proving it, rather than asserting it

`tests/test_sampling_determinism.py` asserts: that the identical rows
presented in a deliberately different physical order select the identical
subset (the property the reservoir draw broke); that the real predicate
against the real `fact_trip` returns the same row count and checksum when run
in **two separate OS processes**, so a cached result cannot satisfy it; that
the buckets partition the data and the five repeat seeds map to five distinct
buckets; and that a booster fit on an **ordered** pull produces identical
predictions across two separate processes, which is the regression guard for
the second defect above.

Beyond the unit tests, notebook 02 was executed four times end to end and
its published tables diffed each time -- byte-identical across all four --
because the first fix passed its unit tests while the notebook still did not
reproduce.

### What it cost, and what it bought

Every Task B figure was regenerated twice over — once for the bucketing fix,
once for the ordering fix — so the numbers in `docs/modelling_notes.md`
moved; nothing that depended on the old draw was retained. In exchange, each
of the five repeats now draws a different, reproducible bucket in a
reproducible order, which makes the variation across repeats a measurement of
sampling variation rather than an untracked confound — and that measurement
is what the ladder's conclusions are now stated against.

The sequence is worth noting for anyone repeating this exercise: the first
fix was verified, the verification failed, and the second defect was only
visible because the first had been eliminated. Had the bucketing not been
fixed first, the ordering defect would have been indistinguishable from
sampling noise and would have stayed hidden.

---

## 13. Summary of every number this study declines to produce

| Not produced | Why |
| --- | --- |
| A pre-2023-11 Nigerian fuel price series | No source covers it; extrapolation would be fabrication. |
| Interpolated values for any absent month | Constraint 2.5. Gaps are reported, never filled. |
| A blended average tip across payment types | Cash tips are not recorded; the blend would be meaningless. |
| A cost-of-living index | No consumption quantities exist to weight one. |
| A causal estimate of the subsidy removal's effect on food prices | No identification strategy is available. |
| A national fuel-food correlation | The overlap is 3 of 37 federating units (8.1%), all in the North East; the figure would not generalise. A9 is reported only as a null result about conflict-affected north-eastern markets. |
| A national before-and-after around the subsidy removal | The WFP panel changes from 14 states to 3 across the break; A7 is restricted to a fixed within-market panel of 10 markets in 2 states. |
| Operator net revenue | Only passenger-paid totals are published. |
| A "bytes read" figure for DuckDB | The engine does not expose one; files opened and rows scanned are reported instead, labelled as such. |

---

## 14. A12: the geography of petrol prices — centrepiece of the Nigerian analysis

A12 (`petrol_price_geography`) uses `fact_fuel_price_monthly` alone. That table
has complete coverage — 37 federating units × 31 months (November 2023 to May
2026), no gaps — and its national mean reconciles exactly with the Bureau's
published figures. Unlike A6–A9 it needs no coverage caveat. It produces four
tables (state-month, national-month, state, zone-month) and four figures.

### Definitions, and why

- **National mean** = the unweighted mean of the 37 state prices. This is the
  Bureau's own definition: it is the figure that reconciles to 0.00 NGN. NBS
  publishes no state fuel volumes or weights, so no weighted mean is possible.
- **Dispersion** uses the **population** standard deviation. The 37 units are a
  census, not a sample. The coefficient of variation (sd / mean) is scale-free,
  so a rise means the states genuinely moved apart rather than prices rising.
- **Premium** = state price / national mean − 1. Because the mean includes the
  state, premiums sum to exactly zero every month (asserted by the test suite).
- **Ranks** are average ranks under ties (11 tie groups, none larger than 3
  states), so the rank correlations are exact Spearman coefficients.
- **Persistence**: a premium (discount) is persistent when the state sits above
  (below) the national mean in at least 80% of months. The longest unbroken run
  is reported too, computed as a gaps-and-islands query, because a share can
  hide whether the months were consecutive.
- **Zones**: a zone's price is the unweighted mean of its states. Each month the
  between-zone share of cross-state variance is reported, which measures how
  much of the dispersion among states the six-zone geography explains.

### Results

**Trajectory.** The national mean rose from **₦648.93 (November 2023) to
₦1,596.25 (May 2026), +146.0%**. The first month is the trough and the last the
peak. The path is not smooth: the largest monthly rises were September 2024
(+24.1%), March 2026 (+22.5%) and April 2026 (+19.0%); the largest fall was May
2025 (−17.1%). Because the series begins six months after the 29 May 2023
subsidy removal, it describes the post-removal market, not the transition.

**Dispersion.** The cross-state CV has a median of **4.8%** (mean 6.1%), ranging
from 2.3% (December 2025) to **17.7% (February 2025)**. It exceeded 10% only in
August 2024 and in four consecutive months, January to April 2025. In February
2025 the dearest state (Jigawa) was 1.81 times the price of the cheapest (Ekiti).
Dispersion is episodic: the states move apart sharply for a few months and then
reconverge.

**Zones explain only part of it.** The between-zone share of cross-state variance
has a median of **34.8%** (range 3.0% to 75.5%). In a typical month about two
thirds of the dispersion lies between states *within* the same zone.

**Persistent positions are rare.** Only **5 of 37 states** hold a persistent
position, and 32 do not:

| Persistent premium | Share of months above mean | Longest run above |
| --- | --- | --- |
| Jigawa (North West) | 87% | 21 months |
| Adamawa (North East) | 84% | 16 months |
| Taraba (North East) | 81% | 9 months |

| Persistent discount | Share of months below mean | Longest run below |
| --- | --- | --- |
| Lagos (South West) | 90% | 18 months |
| Ogun (South West) | 84% | 13 months |

Mean premiums range from **+7.8% (Jigawa) to −6.7% (Lagos)**. Lagos was the
cheapest state in 7 months; Jigawa was the dearest in 5.

**The ranking of states is unstable.** The month-to-month Spearman correlation
has a median of **0.57** and falls as low as −0.11 (May 2025). Measured against
the first month, the correlation fell below zero in 7 months, reaching −0.27
(August 2024), and stood at **0.12 by May 2026**. Within a few months, the
ordering of states in November 2023 tells you almost nothing about their order
later. The median state moved with a standard deviation of 9.3 rank positions.

**Zones.** Averaged over the period: North West +2.54%, South East +2.08%, North
East +1.19%, North Central −0.78%, South South −0.99%, **South West −3.98%**.
Mean within-zone CVs range from 3.4% to 5.5%. The North East and North West were
each the dearest zone in 10 months; the South West and North Central were never
the dearest zone.

### What A12 can and cannot say

It can say where and when petrol was dear or cheap relative to the national mean,
how far apart the states were, and how stable those positions were. The pattern
is consistent with a south-west–north gradient: the two persistent discounts are
the south-western states nearest the Lagos import and refining hub, and the
three persistent premiums are northern. But zones explain only about a third of
the dispersion, and positions reshuffle month to month. **Distance from import
terminals and refineries, transport cost, security and local market structure are
all plausible explanations, and none is tested here.** A12 is descriptive.

---

## 15. Limitations

Each limitation below is also stated where the affected output is produced.

1. **A9 is a null result about conflict-affected north-eastern markets.** WFP had
   stopped reporting 11 of the 14 states it monitored by January 2023, so the
   fuel-to-food analysis rests on 3 states, all in the North East (Borno, Yobe,
   Adamawa). Its correlations describe those markets only, are under ±0.09 at
   every lag, and cannot support any inference. It is retained as an exploratory
   output only (note 5).
2. **A7 is within-market, not national.** It covers a fixed panel of 10 markets
   in 2 states (Borno 6, Yobe 4) that report in every month from February to
   September 2023. It describes those markets only. Its segment trends rest on 4
   points each; the within-cell level change is the number that can bear weight
   (note 4). No causal claim is made.
3. **WFP's geographic footprint contracts sharply in 2023.** Any WFP-based
   statistic after January 2023 describes conflict-affected north-eastern markets
   under humanitarian monitoring, not Nigeria.
4. **A6 links only one category unbroken to its base.** Only cereals and tubers
   chain from January 2016 to the present; pulses and oils break in early 2023,
   and four categories cannot be linked to the base at all (note 1).
5. **The Nigerian price analyses are descriptive.** A12 has complete, reconciled
   coverage and needs no coverage caveat, but it describes where and when petrol
   was dear or cheap; it does not test why.
6. **Speed and tips are proxies with known blind spots.** Journey speed is a
   congestion proxy (note 10); tips are observable only for card payments
   (note 9).
7. **The engineering measurements are environment-bound.** Timings come from one
   laptop under Docker Desktop with a virtualised filesystem and a warm page
   cache. The measured cold-run processing time (35.3 minutes) exceeds the
   specification's 20-minute budget; it is reported as measured, and the budget
   is not met by shrinking the work.
8. **The cloud comparison is illustrative, not controlled.** The BigQuery side
   reads 2019, because the public dataset ends in 2022 and holds no 2024 rows,
   while the local side reads 2024. Hardware, storage layout, network path and
   the definition of a "byte" also differ, and BigQuery's implied costs rest on
   a configured rate that could not be confirmed at runtime. See
   `docs/cloud_architecture.md`.
