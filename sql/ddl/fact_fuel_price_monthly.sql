-- ===========================================================================
-- fact_fuel_price_monthly
--
-- Purpose : The coarse grain of the platform -- one row per state per month --
--           demonstrating that the same conformed dimensions serve a fact table
--           roughly 37,000x smaller than fact_trip.
-- Grain   : ONE ROW PER (state, month) for Premium Motor Spirit (petrol).
-- Source  : silver_nbs_pms (Source D, NBS PMS Price Watch).
--
-- Assumptions:
--   1. mom_pct_change and yoy_pct_change are COMPUTED HERE from the loaded price
--      series, never read from the published sheets. NBS prints its own change
--      figures; reproducing ours from the underlying observations means every
--      derived number in the paper is traceable to data this pipeline loaded.
--      Where the prior period is absent the change is NULL, never zero and never
--      interpolated.
--   2. geo_key resolves to a Nigerian STATE member of dim_geography. The fuel
--      panel is complete: every federating unit reports in every published
--      month, so unlike the WFP panel this one needs no composition correction.
--   3. Prices are the published state mean retail price in NGN per litre. NBS
--      does not publish the underlying outlet-level observations, so the state
--      mean is the finest grain obtainable, and this fact table cannot be
--      decomposed further however much the taxi fact can.
-- ===========================================================================
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
