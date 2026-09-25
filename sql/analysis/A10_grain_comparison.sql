-- ===========================================================================
-- A10  grain_comparison
--
-- Question   : What does it actually cost, in storage and in query time, to
--              serve fact tables whose grains differ by orders of magnitude
--              from one conformed set of dimensions?
-- Grain      : one row per fact table.
-- Output     : outputs/tables/A10_grain_comparison.csv
-- Figure     : A10_grain_comparison.png
--
-- This is the engineering evidence for the paper's central argument. It is the
-- one analysis whose subject is the platform itself rather than the world.
--
-- Assumptions:
--   1. Row counts and distinct dimension members are measured here in SQL.
--      On-disk bytes, bytes per row and the national-aggregate timing are
--      attached by src/analysis/run_analysis.py, because they are filesystem
--      and wall-clock measurements that SQL cannot make. Every such column is
--      named with a _measured suffix in the output CSV.
--   2. "Distinct dimension members referenced" counts members actually used by
--      the fact, not members available in the dimension. A dimension with 4,019
--      rows of which a fact touches 374 is the interesting number.
--   3. The comparable national aggregate is defined per fact table as a single
--      monthly national series -- for trips, revenue by month; for prices, the
--      mean by month. They are not the same question, but they are the same
--      SHAPE of question, which is what makes the timings comparable.
--   4. fact_trip's byte figure is the whole partitioned dataset including
--      Hive directory overhead, because that is what the platform occupies.
-- ===========================================================================
SELECT
    'fact_trip'                                            AS fact_table,
    'one row per completed taxi trip'                      AS grain_statement,
    'New York City'                                        AS context,
    count(*)                                               AS row_count,
    count(DISTINCT pickup_date_key)                        AS distinct_date_members,
    count(DISTINCT pickup_geo_key)                         AS distinct_geo_members,
    count(DISTINCT mode_key)                               AS distinct_mode_members,
    count(DISTINCT flag_key)                               AS distinct_other_members,
    9                                                      AS foreign_key_count,
    13                                                     AS measure_count
FROM fact_trip

UNION ALL

SELECT
    'fact_trip_daily_agg',
    'one row per date, pickup zone and mode',
    'New York City',
    count(*),
    count(DISTINCT date_key),
    count(DISTINCT pickup_geo_key),
    count(DISTINCT mode_key),
    0,
    3,
    7
FROM fact_trip_daily_agg

UNION ALL

SELECT
    'fact_market_price_monthly',
    'one row per market, commodity, price type and month',
    'Nigeria',
    count(*),
    count(DISTINCT month_key),
    count(DISTINCT geo_key),
    0,
    count(DISTINCT commodity_key),
    3,
    4
FROM fact_market_price_monthly

UNION ALL

SELECT
    'fact_fuel_price_monthly',
    'one row per state and month',
    'Nigeria',
    count(*),
    count(DISTINCT month_key),
    count(DISTINCT geo_key),
    0,
    0,
    2,
    3
FROM fact_fuel_price_monthly

-- The weather fact is OPTIONAL (Source E is non-blocking). This branch is
-- wrapped in optional markers; the analysis runner removes it, and records the
-- omission in the run manifest, whenever fact_weather_daily was not built.
-- optional:fact_weather_daily begin
UNION ALL

SELECT
    'fact_weather_daily',
    'one row per date and geography',
    'New York City',
    count(*),
    count(DISTINCT date_key),
    count(DISTINCT geo_key),
    0,
    0,
    2,
    5
FROM fact_weather_daily
-- optional:fact_weather_daily end

ORDER BY row_count DESC;
