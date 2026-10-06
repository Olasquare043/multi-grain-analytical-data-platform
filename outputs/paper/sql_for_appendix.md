# SQL for the appendix

Read-only extraction from the repository at commit `e317b57`. Every code block is copied by line range from the file named above it, not retyped. Paths are relative to the repository root.

## Row count of fact_trip_daily_agg

```sql
SELECT COUNT(*) FROM fact_trip_daily_agg;
```

**Result: 84,910 rows.** Run read-only (`duckdb.connect(..., read_only=True)`) against the existing `data/warehouse.duckdb` on 2026-09-25, where `fact_trip_daily_agg` is a base table. It equals `row_count` for `fact_trip_daily_agg` in `outputs/tables/A10_grain_comparison.csv` (84,910), the figure recorded during the original run.

---

## 1. Statement that builds fact_fuel_price_monthly from silver

`sql/ddl/fact_fuel_price_monthly.sql`, lines 25-40

```sql
CREATE OR REPLACE TABLE fact_fuel_price_monthly AS
SELECT
    CAST(n.month_key AS INTEGER)             AS month_key,
    coalesce(g.geo_key, {unknown_key})       AS geo_key,
    '{fuel_type}'                            AS fuel_type,
    round(n.price_ngn, 4)                    AS price_ngn,
    round(n.mom_pct_change, 6)               AS mom_pct_change,
    round(n.yoy_pct_change, 6)               AS yoy_pct_change,
    n.source_file
FROM read_parquet('{silver_nbs}') n
LEFT JOIN dim_geography g
       ON g.geo_code    = n.state
      AND g.country     = 'Nigeria'
      AND g.admin_level = 'state'
      AND g.is_current
ORDER BY month_key, geo_key;
```

The file is a template: it is rendered with `str.format` before execution. The header comment (lines 1-24 of the same file) states the grain and assumptions.

`src/model/facts.py`, lines 165-173

```python
def build_fact_fuel_price_monthly(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Build the state/month petrol price fact."""
    con.execute(
        _ddl("fact_fuel_price_monthly").format(
            silver_nbs=_sql(settings.SILVER_DIR / silver.SILVER_NBS),
            unknown_key=settings.UNKNOWN_KEY,
            fuel_type=settings.NBS_FUEL_TYPE,
        )
    )
```

Placeholder values: `{silver_nbs}` is the silver NBS Parquet path (`settings.SILVER_DIR / silver.SILVER_NBS`); `{unknown_key}` is `-1` (`config/settings.py:317`, `UNKNOWN_KEY = -1`); `{fuel_type}` is `'PMS'` (`config/settings.py:259`, `NBS_FUEL_TYPE = "PMS"`).

---

## 2. Statement that builds fact_trip_daily_agg

`sql/ddl/fact_trip_daily_agg.sql`, lines 25-39

```sql
CREATE OR REPLACE TABLE fact_trip_daily_agg AS
SELECT
    pickup_date_key                                  AS date_key,
    pickup_geo_key,
    mode_key,
    CAST(count(*) AS BIGINT)                         AS trip_count,
    round(sum(trip_distance_miles), 4)               AS total_distance_miles,
    round(sum(total_amount), 4)                      AS total_revenue,
    round(avg(fare_amount), 4)                       AS avg_fare,
    round(median(fare_amount), 4)                    AS median_fare,
    round(avg(trip_duration_seconds), 2)             AS avg_duration_seconds,
    round(avg(avg_speed_mph), 4)                     AS avg_speed_mph
FROM fact_trip
GROUP BY ALL
ORDER BY date_key, pickup_geo_key;
```

Executed by `src/model/facts.py:202` (`con.execute(_ddl("fact_trip_daily_agg"))`). It takes no placeholders. The header comment (lines 1-24) states the grain and the aggregate-navigation caveats.

---

## 3. Nigerian state mean price feature (V3b point-in-time and V3a leaky)

**These two features are implemented in pandas, not SQL. The repository contains no SQL text for them**, so the code is reproduced as stored.

V3a, leaky:

`src/modelling/features.py`, lines 33-48

```python
def add_leaky_state_avg_nigeria(
    frame: pd.DataFrame,
    state_col: str = "geo_key",
    price_col: str = "price_ngn",
    out_col: str = "state_avg_price_leaky",
) -> pd.DataFrame:
    """LEAKY. "This state's average price," computed over EVERY month in
    `frame`, including months after the row being featurised. A row in
    2023-11 receives a value that was partly computed from 2026-05, a month
    that had not happened yet. Labelled LEAKY everywhere it is used;
    never trusted as a real number. See add_pit_state_avg_nigeria for the
    correct version of the same idea.
    """
    frame = frame.copy()
    frame[out_col] = frame.groupby(state_col)[price_col].transform("mean")
    return frame
```

V3b, point-in-time:

`src/modelling/features.py`, lines 51-68

```python
def add_pit_state_avg_nigeria(
    frame: pd.DataFrame,
    state_col: str = "geo_key",
    month_key_col: str = "month_key",
    price_col: str = "price_ngn",
    out_col: str = "state_avg_price_pit",
) -> pd.DataFrame:
    """Point-in-time-correct. "This state's average price using only months
    strictly before the month being predicted," an expanding mean recomputed
    per row from that state's own prior rows only. A state's first observed
    month has no prior data and is NaN by construction (LightGBM handles
    missing values natively; nothing is imputed here). See
    add_leaky_state_avg_nigeria for the leaky version of the same idea.
    """
    frame = frame.sort_values([state_col, month_key_col]).copy()
    grp = frame.groupby(state_col)[price_col]
    frame[out_col] = grp.transform(lambda s: s.shift(1).expanding().mean())
    return frame
```

Invocation in the ladder notebook (`notebooks/01_nigeria_petrol_forecasting.ipynb`, JSON line 654 and 655; note `state_col="state"`, not the function default `"geo_key"`):

```python
modeling = add_leaky_state_avg_nigeria(modeling, state_col="state")
modeling = add_pit_state_avg_nigeria(modeling, state_col="state")
```

---

## 4. NYC zone-hour point-in-time average duration feature (V3b)

**Also pandas, not SQL; no SQL text exists for it.**

`src/modelling/features.py`, lines 92-128

```python
def add_pit_zone_hour_avg_nyc(
    frame: pd.DataFrame,
    zone_col: str = "pickup_geo_key",
    hour_col: str = "pickup_hour",
    month_col: str = "month",
    duration_col: str = "trip_duration_seconds",
    out_col: str = "zone_hour_avg_duration_pit",
) -> pd.DataFrame:
    """Point-in-time-correct. "This zone and hour's average trip duration
    using only trips from strictly earlier calendar months," computed at
    month granularity (per section 3.4 / 4.4 of the v3 prompt) rather than
    per-row, because that is the natural update cadence of an aggregate a
    production system would actually refresh.

    Implementation: build a small (zone, hour, month) summary table of
    per-cell sum and count, then take each cell's CUMULATIVE sum/count over
    STRICTLY PRIOR months only (current month's own sum/count subtracted back
    out of its cumulative total), and merge that lookup onto every row by
    (zone, hour, month). A (zone, hour) pair's first observed month has no
    prior data and is NaN by construction, exactly as in the Nigeria version.
    See add_leaky_zone_hour_avg_nyc for the leaky version of the same idea.
    """
    monthly = (
        frame.groupby([zone_col, hour_col, month_col])[duration_col]
        .agg(month_sum="sum", month_count="count")
        .reset_index()
        .sort_values([zone_col, hour_col, month_col])
    )
    grp = monthly.groupby([zone_col, hour_col])
    cum_sum = grp["month_sum"].cumsum()
    cum_count = grp["month_count"].cumsum()
    prior_sum = cum_sum - monthly["month_sum"]
    prior_count = cum_count - monthly["month_count"]
    monthly[out_col] = prior_sum / prior_count.replace(0, np.nan)

    lookup = monthly[[zone_col, hour_col, month_col, out_col]]
    return frame.merge(lookup, on=[zone_col, hour_col, month_col], how="left")
```

Invocation (`notebooks/02_nyc_trip_duration.ipynb`, JSON line 364, default arguments):

```python
modeling = add_pit_zone_hour_avg_nyc(modeling)
```

---

## 5. NBS national-mean reconciliation check

The mandatory check is the pytest below. It computes the national mean from the gold `fact_fuel_price_monthly` and asserts it is within tolerance of the published NBS figure.

`tests/test_pms_reconciliation.py`, lines 27-70

```python
TARGETS = settings.NBS_RECONCILIATION_TARGETS
TOLERANCE = settings.NBS_RECONCILIATION_TOLERANCE


def _national_means(con: duckdb.DuckDBPyConnection) -> dict[str, float]:
    """National mean petrol price per month, from the GOLD fact table.

    Computed from ``fact_fuel_price_monthly`` rather than an intermediate, so a
    defect introduced anywhere between the spreadsheet and the star schema will
    surface here.
    """
    rows = con.execute(
        """
        SELECT printf('%04d-%02d', month_key // 10000, (month_key // 100) % 100)
                                             AS year_month,
               round(avg(price_ngn), 2)      AS national_mean
        FROM fact_fuel_price_monthly
        GROUP BY 1
        """
    ).fetchall()
    return {row[0]: float(row[1]) for row in rows}


@pytest.mark.parametrize("month,published", sorted(TARGETS.items()))
def test_national_mean_matches_published_figure(
    con: duckdb.DuckDBPyConnection, month: str, published: float
) -> None:
    """The assembled national mean equals the NBS published figure."""
    require_table(con, "fact_fuel_price_monthly")
    observed = _national_means(con)

    assert month in observed, (
        f"{month} is absent from fact_fuel_price_monthly. The NBS catalogue "
        f"publishes it; a missing month means the workbook parser failed to "
        f"find a header row. Check the per-file diagnostics in "
        f"docs/run_manifest.json under extraction.nbs_pms."
    )
    delta = observed[month] - published
    assert abs(delta) <= TOLERANCE, (
        f"RECONCILIATION FAILED for {month}: assembled {observed[month]:.2f} "
        f"NGN/litre against the published {published:.2f} "
        f"(delta {delta:+.2f}, tolerance {TOLERANCE}). No figure may be "
        f"published from this run until the cause is found."
    )
```

The same comparison is produced for the quality report by `build_nbs_reconciliation`, with its own SQL at lines 137-146:

`src/quality/report.py`, lines 126-170

```python
def build_nbs_reconciliation(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Assembled national mean vs the NBS published figure, per target month.

    This is the pipeline's external correctness proof (section 3, Source D). The
    assembled figure is computed from ``fact_fuel_price_monthly``, i.e. from the
    fully modelled data rather than from an intermediate, so the check covers the
    whole chain from Excel cell to gold fact.
    """
    if not table_exists(con, "fact_fuel_price_monthly"):
        return pd.DataFrame()

    observed = sql_df(
        con,
        """
        SELECT CAST(month_key / 100 AS INTEGER)              AS ym,
               round(avg(price_ngn), 2)                      AS assembled_mean_ngn,
               count(*)                                      AS reporting_states
        FROM fact_fuel_price_monthly
        GROUP BY 1
        """,
    )
    observed["month"] = observed["ym"].apply(
        lambda v: f"{int(v) // 100:04d}-{int(v) % 100:02d}"
    )
    lookup = observed.set_index("month")

    records = []
    for month, published in sorted(settings.NBS_RECONCILIATION_TARGETS.items()):
        if month in lookup.index:
            assembled = float(lookup.loc[month, "assembled_mean_ngn"])
            states = int(lookup.loc[month, "reporting_states"])
            delta = round(assembled - published, 4)
            within = abs(delta) <= settings.NBS_RECONCILIATION_TOLERANCE
        else:
            assembled, states, delta, within = None, 0, None, False
        records.append({
            "month": month,
            "published_national_mean_ngn": published,
            "assembled_national_mean_ngn": assembled,
            "delta": delta,
            "tolerance": settings.NBS_RECONCILIATION_TOLERANCE,
            "within_tolerance": within,
            "reporting_states": states,
        })
    return pd.DataFrame(records)
```

The published figures and tolerance:

`config/settings.py`, lines 271-277

```python
#: Mandatory reconciliation against NBS published national means (NGN/litre).
NBS_RECONCILIATION_TARGETS: dict[str, float] = {
    "2023-11": 648.93,
    "2023-12": 671.86,
    "2024-05": 769.62,
}
NBS_RECONCILIATION_TOLERANCE = 0.05
```

The test extracts year-month with `printf('%04d-%02d', month_key // 10000, (month_key // 100) % 100)`; the report query uses `CAST(month_key / 100 AS INTEGER)` and formats the month in Python. Both aggregate `avg(price_ngn)` over the state rows of each month, rounded to 2 decimals.

---

## 6. National aggregate benchmark query (A10), against fact_trip and fact_fuel_price_monthly

`src/analysis/run_analysis.py`, lines 55-83

```python
#: The "comparable national aggregate" of A10: one monthly national series per
#: fact table. Not the same question, but the same SHAPE of question, which is
#: what makes the timings comparable.
NATIONAL_AGGREGATE_QUERIES: dict[str, str] = {
    "fact_trip": """
        SELECT d.year_month, count(*) AS trips, sum(f.total_amount) AS revenue
        FROM fact_trip f JOIN dim_date d ON d.date_key = f.pickup_date_key
        GROUP BY 1 ORDER BY 1
    """,
    "fact_trip_daily_agg": """
        SELECT d.year_month, sum(a.trip_count) AS trips,
               sum(a.total_revenue) AS revenue
        FROM fact_trip_daily_agg a JOIN dim_date d ON d.date_key = a.date_key
        GROUP BY 1 ORDER BY 1
    """,
    "fact_market_price_monthly": """
        SELECT month_key, count(*) AS observations, avg(price_ngn) AS mean_price
        FROM fact_market_price_monthly GROUP BY 1 ORDER BY 1
    """,
    "fact_fuel_price_monthly": """
        SELECT month_key, count(*) AS observations, avg(price_ngn) AS mean_price
        FROM fact_fuel_price_monthly GROUP BY 1 ORDER BY 1
    """,
    "fact_weather_daily": """
        SELECT date_key / 100 AS year_month, count(*) AS observations,
               avg(temp_max_c) AS mean_value
        FROM fact_weather_daily GROUP BY 1 ORDER BY 1
    """,
}
```

The two requested queries are the `fact_trip` entry (lines 59-63) and the `fact_fuel_price_monthly` entry (lines 74-77). **They are not the same query**: the `fact_trip` query joins `dim_date` and computes `count(*)` and `sum(total_amount)` per year-month, while the `fact_fuel_price_monthly` query has no join and computes `count(*)` and `avg(price_ngn)` per `month_key`. The source comment at lines 55-57 says so: "Not the same question, but the same SHAPE of question".

How each query is timed:

`src/analysis/run_analysis.py`, lines 205-236

```python
def measure_grain_costs(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Measure on-disk size and national-aggregate latency per fact table.

    The aggregate is timed three times and the median reported, matching the
    benchmark protocol in section 8 so the two sets of timings are comparable.
    """
    records = []
    for fact_table, query in NATIONAL_AGGREGATE_QUERIES.items():
        path = _fact_artefact_path(fact_table)
        size = dir_size(path)
        timings: list[float] = []
        for _ in range(settings.BENCHMARK_REPEATS):
            started = time.perf_counter()
            try:
                con.execute(query).fetchall()
            except duckdb.Error as exc:
                LOG.warning("national aggregate for %s failed: %s", fact_table, exc)
                timings = []
                break
            timings.append(time.perf_counter() - started)
        median = round(sorted(timings)[len(timings) // 2], 6) if timings else None
        records.append({
            "fact_table": fact_table,
            "bytes_measured": size,
            "bytes_human_measured": human_bytes(size),
            "national_aggregate_seconds_measured": median,
            "national_aggregate_repeats": len(timings),
        })
        LOG.info("A10 measurement: %-28s %10s, national aggregate %s",
                 fact_table, human_bytes(size),
                 f"{median:.4f}s" if median is not None else "n/a")
    return pd.DataFrame(records)
```

`settings.BENCHMARK_REPEATS` is set at `BENCHMARK_REPEATS = 3` (line 353 of `config/settings.py`); the reported value is the median of the repeats (`sorted(timings)[len(timings) // 2]`, line 225).

---

## 7. Type 2 slowly changing dimension update for dim_geography

The close-out `UPDATE` and the insert of the new versions:

`src/model/dim_geography.py`, lines 272-288

```python
    if len(changed_rows):
        closing_keys = [int(k) for k in changed_rows["geo_key"].tolist()]
        placeholders = ", ".join("?" for _ in closing_keys)
        con.execute(
            f"""
            UPDATE dim_geography
               SET valid_to = ?, is_current = FALSE
             WHERE geo_key IN ({placeholders})
            """,
            [effective - dt.timedelta(days=1), *closing_keys],
        )

    if to_insert:
        frame = pd.DataFrame(to_insert)
        con.register("scd_inserts", frame)
        con.execute("INSERT INTO dim_geography SELECT * FROM scd_inserts")
        con.unregister("scd_inserts")
```

The natural key and tracked attributes that decide which rows are closed and re-inserted (lines 40 and 39):

```python
NATURAL_KEY = ("geo_code", "country", "admin_level")
TRACKED_ATTRIBUTES = ("geo_name", "parent_geo_name", "region_group")
```

The behaviour is documented in the function docstring (`src/model/dim_geography.py`, lines 167-179): a changed tracked attribute closes the current row (`valid_to = effective_date - 1 day`, `is_current = FALSE`) and inserts a new current version; a new natural key is inserted as current; an absent natural key is left current; unchanged rows are untouched.
