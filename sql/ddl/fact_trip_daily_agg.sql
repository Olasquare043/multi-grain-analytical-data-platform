-- ===========================================================================
-- fact_trip_daily_agg
--
-- Purpose : Aggregate navigation. A pre-computed summary at a grain most
--           dashboard questions actually ask at, so that benchmark B3 can
--           measure what the aggregate buys against querying 41 M rows.
--           This table is redundant by design; the measurement of that
--           redundancy is the point.
-- Grain   : ONE ROW PER (date, pickup zone, mode).
--           ~366 days x ~262 zones x 1 mode, minus empty combinations.
-- Source  : fact_trip. Derived entirely from the atomic fact, so it can never
--           disagree with it, and can be dropped and rebuilt at any time.
--
-- Assumptions:
--   1. Only additive and semi-additive measures are stored. avg_fare and
--      avg_speed_mph are stored for convenience, but they are NOT re-aggregable:
--      averaging avg_fare across zones weights every zone equally regardless of
--      trip count. Any roll-up must use total_revenue / trip_count. Stated here
--      because this is the classic aggregate-navigation trap.
--   2. median_fare is computed exactly, not approximated, so B3's comparison
--      against the atomic fact is like for like.
--   3. Rows are keyed on PICKUP zone only. A dropoff-keyed aggregate would be a
--      second, different aggregate; conflating them would double-count revenue.
-- ===========================================================================
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
