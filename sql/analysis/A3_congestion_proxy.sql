-- ===========================================================================
-- A3  congestion_proxy
--
-- Question   : How does mean journey speed vary by hour and borough, and does
--              it drift over the year?
-- Grain      : mixed, produced in ONE pass with GROUPING SETS --
--                (borough, pickup_hour)  the diurnal profile
--                (borough, year_month)   the monthly trend
--                (borough)               the borough baseline
--              The grouping_level column names which set a row belongs to.
-- Output     : outputs/tables/A3_congestion_proxy.csv
-- Figure     : A3_congestion_proxy.png
--
-- THE CENTRAL ASSUMPTION, STATED PLAINLY:
--   Mean journey speed is used here as a PROXY for road congestion. It is not a
--   measurement of congestion. Speed on a taxi trip confounds at least:
--     - route choice and trip purpose (a JFK run uses motorways; a crosstown
--       Manhattan hop does not),
--     - the distance mix, since short trips carry proportionally more time
--       stationary at kerbs, junctions and lights,
--     - metered distance error, and
--     - time spent stopped with the meter running for reasons unrelated to
--       traffic.
--   A fall in mean speed is therefore consistent with worsening congestion but
--   is not evidence of it. No causal claim is made anywhere from this table.
--
-- Further assumptions:
--   1. Declared window only (calendar 2024).
--   2. Trips with implausible speeds (> 100 mph, ~0.027 per cent of rows, see
--      quality rule Q045) are EXCLUDED from the speed statistics but counted in
--      excluded_implausible_speed, because a single 900 mph odometer error
--      moves a borough-hour mean noticeably.
--   3. Speeds are trip-level means, equally weighting a 0.5 mile trip and a 20
--      mile one. distance_weighted_speed_mph is reported alongside precisely
--      because the two answer different questions, and they diverge.
--   4. Non-geographic zones excluded: 'Unknown' is not a place with traffic.
-- ===========================================================================
-- Query shape, which matters at 40.4 M rows:
--   AGGREGATE ON INTEGERS, DECORATE LAST. Both grouping attributes -- borough
--   and calendar month -- are VARCHAR in the dimensions. Carrying them through
--   a 40.4 M row aggregation costs more than the aggregation itself, so the
--   joins yield small INTEGER codes here and the labels are attached to the
--   ~220 result rows at the end.
WITH borough_ids AS (
    SELECT
        parent_geo_name                                AS borough,
        dense_rank() OVER (ORDER BY parent_geo_name)   AS borough_id
    FROM (
        SELECT DISTINCT parent_geo_name
        FROM dim_geography
        WHERE country = 'United States' AND admin_level = 'zone'
          AND is_current AND geo_code NOT IN ('264', '265')
          AND parent_geo_name IS NOT NULL
    )
),
geo_to_borough AS (
    SELECT g.geo_key, b.borough_id
    FROM dim_geography g
    JOIN borough_ids b ON b.borough = g.parent_geo_name
    WHERE g.country = 'United States' AND g.admin_level = 'zone'
      AND g.is_current AND g.geo_code NOT IN ('264', '265')
),
date_to_month AS (
    SELECT date_key, month_number AS month_id, year_month
    FROM dim_date WHERE year = 2024
),
eligible AS (
    SELECT
        gb.borough_id,
        dm.month_id,
        f.pickup_hour,
        f.avg_speed_mph,
        f.trip_distance_miles,
        f.trip_duration_seconds,
        (f.avg_speed_mph IS NULL OR f.avg_speed_mph > 100
         OR f.avg_speed_mph <= 0)                      AS is_implausible_speed
    FROM fact_trip f
    JOIN geo_to_borough gb ON gb.geo_key  = f.pickup_geo_key
    JOIN date_to_month  dm ON dm.date_key = f.pickup_date_key
    WHERE f.year = 2024   -- Hive partition column: prunes before reading
),
-- Aggregation note, which matters at 40.4 M rows:
--   The three order statistics are requested as ONE multi-quantile call
--   returning a list, not as three separate aggregates. An exact quantile must
--   buffer every value in its group; three separate calls buffer the same 40.4 M
--   values three times over, across three grouping sets, which is nine copies.
--   quantile_cont(x, [0.25, 0.5, 0.75]) computes all three from a single
--   buffered state. The results are still EXACT -- no approximate digest is
--   used anywhere in this study -- but peak memory falls by roughly two thirds.
aggregated AS (
    SELECT
        CASE
            WHEN grouping(pickup_hour) = 0 THEN 'borough_hour'
            WHEN grouping(month_id)    = 0 THEN 'borough_month'
            ELSE                                'borough_overall'
        END                                            AS grouping_level,
        borough_id,
        pickup_hour,
        month_id,
        count(*)                                       AS trip_count,
        count(*) FILTER (WHERE is_implausible_speed)   AS excluded_implausible_speed,
        avg(avg_speed_mph) FILTER (WHERE NOT is_implausible_speed)
                                                       AS mean_speed_mph_raw,
        quantile_cont(avg_speed_mph, [0.25, 0.5, 0.75])
            FILTER (WHERE NOT is_implausible_speed)    AS speed_quantiles,
        -- Aggregate distance divided by aggregate time: the speed a fleet
        -- operator would observe, as distinct from the mean of per-trip speeds.
        sum(trip_distance_miles) FILTER (WHERE NOT is_implausible_speed)
            / nullif(sum(trip_duration_seconds)
                     FILTER (WHERE NOT is_implausible_speed) / 3600.0, 0)
                                                       AS dw_speed_raw,
        avg(trip_distance_miles) FILTER (WHERE NOT is_implausible_speed)
                                                       AS mean_distance_raw
    FROM eligible
    GROUP BY GROUPING SETS (
        (borough_id, pickup_hour),
        (borough_id, month_id),
        (borough_id)
    )
)
SELECT
    a.grouping_level,
    b.borough,
    a.pickup_hour,
    printf('2024-%02d', a.month_id)                    AS year_month,
    a.trip_count,
    a.excluded_implausible_speed,
    round(a.mean_speed_mph_raw, 4)                     AS mean_speed_mph,
    round(a.speed_quantiles[2], 4)                     AS median_speed_mph,
    round(a.speed_quantiles[1], 4)                     AS p25_speed_mph,
    round(a.speed_quantiles[3], 4)                     AS p75_speed_mph,
    round(a.dw_speed_raw, 4)                           AS distance_weighted_speed_mph,
    round(a.mean_distance_raw, 4)                      AS mean_distance_miles
FROM aggregated a
JOIN borough_ids b ON b.borough_id = a.borough_id
ORDER BY a.grouping_level, b.borough,
         a.pickup_hour NULLS FIRST, a.month_id NULLS FIRST;
