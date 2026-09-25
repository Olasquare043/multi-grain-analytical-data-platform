-- ===========================================================================
-- A4  tipping_behaviour
--
-- Question   : How does the tip, as a share of fare, vary by payment type,
--              borough and trip distance?
-- Grain      : one row per (payment type, borough, distance band).
-- Output     : outputs/tables/A4_tipping_behaviour.csv
-- Figure     : A4_tipping_behaviour.png
--
-- THE MEASUREMENT CAVEAT, WHICH GOVERNS EVERY NUMBER BELOW:
--   The TLC data dictionary states that the tip field is populated for credit
--   card tips and DOES NOT INCLUDE CASH TIPS. A cash trip showing a zero tip is
--   therefore evidence about the recording system, not about the passenger.
--
--   This analysis handles that openly rather than quietly filtering:
--     - every payment type is reported, so the structural zeros are visible;
--     - dim_payment_type.is_tip_observable marks the ONE type (credit card) for
--       which a tip statistic is interpretable;
--     - is_interpretable is carried on every output row, and the figure plots
--       only interpretable rows while captioning the exclusion.
--   Reporting a blended "average tip across all payment types" would be a
--   fabricated statistic, and none is produced here.
--
-- Further assumptions:
--   1. Declared window only (calendar 2024).
--   2. Tip percentage is tip_amount / fare_amount, not / total_amount. Tolls,
--      surcharges and taxes are not tipped on, and including them would deflate
--      the ratio by a varying amount per corridor.
--   3. Rows with fare_amount = 0 are excluded from the ratio (division would be
--      undefined) but counted in zero_fare_trips.
--   4. Distance bands are half-open [lower, upper), defined in
--      config/settings.py TRIP_DISTANCE_BANDS so the figure and the query
--      cannot disagree.
--   5. Negotiated and group-ride rate codes are retained but flagged, since a
--      pre-agreed fare changes the tipping decision.
-- ===========================================================================
-- Query shape, which matters at 40.4 M rows:
--   AGGREGATE ON INTEGERS, DECORATE LAST. Dimension joins here yield only small
--   integer codes and booleans; payment and borough LABELS are attached to the
--   ~300 result rows at the end. Carrying three VARCHAR grouping columns
--   through a 40.4 M row aggregation costs more than the aggregation itself.
WITH borough_ids AS (
    SELECT
        coalesce(parent_geo_name, 'Unattributed')       AS borough,
        dense_rank() OVER (ORDER BY coalesce(parent_geo_name, 'Unattributed'))
                                                        AS borough_id
    FROM (
        SELECT DISTINCT parent_geo_name
        FROM dim_geography
        WHERE country = 'United States' AND admin_level = 'zone' AND is_current
    )
),
geo_to_borough AS (
    SELECT g.geo_key, b.borough_id
    FROM dim_geography g
    JOIN borough_ids b
      ON b.borough = coalesce(g.parent_geo_name, 'Unattributed')
    WHERE g.country = 'United States' AND g.admin_level = 'zone' AND g.is_current
),
prearranged_rates AS (
    SELECT rate_key,
           (rate_code_name IN ('Negotiated fare', 'Group ride')) AS is_prearranged
    FROM dim_rate_code
),
eligible AS (
    SELECT
        f.payment_key,
        gb.borough_id,
        CASE
            WHEN f.trip_distance_miles <  1  THEN 1
            WHEN f.trip_distance_miles <  3  THEN 2
            WHEN f.trip_distance_miles <  5  THEN 3
            WHEN f.trip_distance_miles < 10  THEN 4
            ELSE                                  5
        END                                             AS distance_band_order,
        f.fare_amount,
        f.tip_amount,
        f.total_amount,
        pr.is_prearranged                               AS is_prearranged_fare
    FROM fact_trip f
    JOIN geo_to_borough    gb ON gb.geo_key  = f.pickup_geo_key
    JOIN prearranged_rates pr ON pr.rate_key = f.rate_key
    -- Hive partition column, not a dim_date join: the year filter prunes whole
    -- partitions before any row is read, and dim_date contributes nothing else
    -- to this analysis. Equivalent, because the partition value IS the pickup
    -- year that dim_date.year resolves to.
    WHERE f.year = 2024
),
aggregated AS (
SELECT
    payment_key,
    borough_id,
    distance_band_order,
    count(*)                                            AS trip_count,
    count(*) FILTER (WHERE fare_amount = 0)             AS zero_fare_trips,
    count(*) FILTER (WHERE tip_amount > 0)              AS trips_with_recorded_tip,
    round(100.0 * count(*) FILTER (WHERE tip_amount > 0) / count(*), 4)
                                                        AS pct_trips_with_recorded_tip,
    round(avg(fare_amount), 4)                          AS mean_fare,
    round(avg(tip_amount), 4)                           AS mean_tip,
    round(100.0 * avg(tip_amount / nullif(fare_amount, 0)), 4)
                                                        AS mean_tip_pct_of_fare,
    round(100.0 * median(tip_amount / nullif(fare_amount, 0)), 4)
                                                        AS median_tip_pct_of_fare,
    -- Conditional on having tipped at all: separates "how many tip" from
    -- "how much do tippers tip", which the unconditional mean conflates.
    round(100.0 * avg(tip_amount / nullif(fare_amount, 0))
          FILTER (WHERE tip_amount > 0), 4)             AS mean_tip_pct_given_tipped,
    round(100.0 * sum(tip_amount) / nullif(sum(fare_amount), 0), 4)
                                                        AS aggregate_tip_pct_of_fare,
    count(*) FILTER (WHERE is_prearranged_fare)         AS prearranged_fare_trips
FROM eligible
GROUP BY ALL
)
SELECT
    pt.payment_type_name,
    pt.is_tip_observable                                AS is_interpretable,
    b.borough,
    CASE a.distance_band_order
        WHEN 1 THEN '0-1 mi'  WHEN 2 THEN '1-3 mi'  WHEN 3 THEN '3-5 mi'
        WHEN 4 THEN '5-10 mi' ELSE '10+ mi'
    END                                                 AS distance_band,
    a.distance_band_order,
    a.trip_count,
    a.zero_fare_trips,
    a.trips_with_recorded_tip,
    a.pct_trips_with_recorded_tip,
    a.mean_fare,
    a.mean_tip,
    a.mean_tip_pct_of_fare,
    a.median_tip_pct_of_fare,
    a.mean_tip_pct_given_tipped,
    a.aggregate_tip_pct_of_fare,
    a.prearranged_fare_trips
FROM aggregated a
JOIN dim_payment_type pt ON pt.payment_key = a.payment_key
JOIN borough_ids      b  ON b.borough_id   = a.borough_id
ORDER BY is_interpretable DESC, pt.payment_type_name, b.borough,
         a.distance_band_order;
