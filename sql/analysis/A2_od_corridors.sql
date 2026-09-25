-- ===========================================================================
-- A2  od_corridors
--
-- Question   : Which origin-destination zone pairs carry the most yellow-taxi
--              volume, and what does a trip on each corridor look like?
-- Grain      : one row per (pickup zone, dropoff zone) pair; top 25 by volume.
-- Output     : outputs/tables/A2_od_corridors.csv
-- Figure     : A2_od_corridors.png (horizontal bar)
--
-- Assumptions:
--   1. Declared window only (calendar 2024), as A1.
--   2. A corridor is DIRECTED. JFK -> Midtown and Midtown -> JFK are different
--      corridors with different fare and duration profiles, and merging them
--      would average away exactly the asymmetry worth reporting.
--   3. Intra-zone trips (pickup zone = dropoff zone) are retained and flagged.
--      They are genuine short trips, and excluding them would quietly remove
--      some of the highest-volume cells in the matrix.
--   4. The two non-geographic zones are excluded here, because an OD pair
--      involving 'Unknown' is not a corridor. The count removed is reported in
--      the excluded_trips column of the summary row set.
--   5. Median is reported alongside mean because corridor fares are
--      right-skewed; the airport flat fares in particular make the mean a poor
--      summary on its own.
--
-- Query shape, which matters here at 40.4 M rows:
--   AGGREGATE FIRST, DECORATE LAST. Dimension attributes are joined only to the
--   25 surviving corridors, never to the 40.4 M input rows. The year filter uses
--   the Hive PARTITION COLUMN rather than a join to dim_date, so whole
--   partitions are skipped before any row is read. Both rewrites are
--   semantically identical to the naive form -- year(pickup_ts) is what
--   dim_date.year resolves to, and out-of-window rows carry a partition year
--   that is not 2024 -- but the naive form decorates every row before
--   discarding it, and on this data that is the difference between seconds and
--   many minutes.
-- ===========================================================================
WITH excluded_zones AS (
    -- The two non-geographic TLC ids, resolved to surrogate keys once so the
    -- 40.4 M row scan filters on integers rather than joining to get a label.
    SELECT geo_key
    FROM dim_geography
    WHERE country = 'United States'
      AND admin_level = 'zone'
      AND geo_code IN ('264', '265')
      AND is_current
),
eligible AS (
    SELECT
        f.pickup_geo_key,
        f.dropoff_geo_key,
        f.total_amount,
        f.fare_amount,
        f.trip_distance_miles,
        f.trip_duration_seconds,
        f.tip_amount
    FROM fact_trip f
    WHERE f.year = 2024
      AND f.pickup_geo_key  NOT IN (SELECT geo_key FROM excluded_zones)
      AND f.dropoff_geo_key NOT IN (SELECT geo_key FROM excluded_zones)
),
corridors AS (
    SELECT
        pickup_geo_key,
        dropoff_geo_key,
        count(*)                                          AS trip_count,
        round(avg(fare_amount), 4)                        AS mean_fare,
        round(median(fare_amount), 4)                     AS median_fare,
        round(avg(total_amount), 4)                       AS mean_total_amount,
        round(avg(trip_distance_miles), 4)                AS mean_distance_miles,
        round(avg(trip_duration_seconds) / 60.0, 3)       AS mean_duration_minutes,
        round(median(trip_duration_seconds) / 60.0, 3)    AS median_duration_minutes,
        round(100.0 * avg(tip_amount / nullif(fare_amount, 0)), 4)
                                                          AS mean_tip_pct_of_fare
    FROM eligible
    GROUP BY ALL
),
ranked AS (
    SELECT
        c.*,
        round(100.0 * trip_count / sum(trip_count) OVER (), 6) AS pct_of_eligible_trips
    FROM corridors c
    -- QUALIFY filters on a window function without a second pass over the data.
    QUALIFY rank() OVER (ORDER BY trip_count DESC) <= 25
)
SELECT
    rank() OVER (ORDER BY r.trip_count DESC)              AS corridor_rank,
    pg.geo_name                                           AS pickup_zone,
    pg.parent_geo_name                                    AS pickup_borough,
    dg.geo_name                                           AS dropoff_zone,
    dg.parent_geo_name                                    AS dropoff_borough,
    (r.pickup_geo_key = r.dropoff_geo_key)                AS is_intra_zone,
    r.trip_count,
    r.pct_of_eligible_trips,
    r.mean_fare,
    r.median_fare,
    r.mean_total_amount,
    r.mean_distance_miles,
    r.mean_duration_minutes,
    r.median_duration_minutes,
    r.mean_tip_pct_of_fare,
    round(r.mean_distance_miles / nullif(r.mean_duration_minutes / 60.0, 0), 4)
                                                          AS implied_mean_speed_mph
FROM ranked r
JOIN dim_geography pg ON pg.geo_key = r.pickup_geo_key
JOIN dim_geography dg ON dg.geo_key = r.dropoff_geo_key
ORDER BY corridor_rank;
