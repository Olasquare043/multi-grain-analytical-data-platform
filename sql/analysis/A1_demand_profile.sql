-- ===========================================================================
-- A1  demand_profile
--
-- Question   : How does yellow-taxi demand vary by hour of day and day of week,
--              and does that shape differ between boroughs?
-- Grain      : one row per (borough, day_of_week, hour_of_day).
-- Output     : outputs/tables/A1_demand_profile.csv
-- Figures    : A1_demand_profile_heatmap.png, A1_demand_profile_lines.png
--
-- Assumptions:
--   1. Restricted to the DECLARED analytical window (calendar 2024) by joining
--      dim_date. The corpus genuinely contains pickups stamped 2002 through
--      2026; they are retained in fact_trip and quantified in the quality
--      report, but including them here would contaminate a seasonal profile
--      with 55 stray rows.
--   2. Borough comes from dim_geography via the PICKUP zone. A trip is
--      attributed to where it began, because this measures demand origination.
--   3. The two non-geographic TLC zones (264 'Unknown', 265 'N/A') are kept in
--      the result as their own borough label rather than dropped, so the
--      reader can see how much demand is unattributable. They are excluded
--      from the figures, which is stated in the figure caption.
--   4. day_of_week is ISO (1 = Monday .. 7 = Sunday).
--   5. Counts are trips, not passengers. passenger_count is NULL for ~9.8 per
--      cent of rows, so a passenger-weighted profile would silently drop them.
-- ===========================================================================
-- Query shape, which matters at 40.4 M rows:
--   AGGREGATE ON INTEGERS, DECORATE LAST. The naive form joins dim_geography
--   and dim_date first and groups by the resulting VARCHAR borough and day
--   name. That materialises 40.4 M string references before a single group is
--   formed, and measured at roughly twice the cost of the whole aggregation.
--   Here the dimension joins yield small INTEGER codes, the grouping is done
--   entirely on integers, and the labels are attached to the ~1,300 result rows
--   at the end. The output is identical.
WITH borough_ids AS (
    SELECT
        coalesce(parent_geo_name, 'Unattributed')        AS borough,
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
date_to_dow AS (
    SELECT date_key, day_of_week FROM dim_date WHERE year = 2024
),
profile AS (
    SELECT
        gb.borough_id,
        dd.day_of_week,
        f.pickup_hour,
        count(*)                                         AS trip_count,
        round(avg(f.total_amount), 4)                    AS avg_total_amount,
        round(avg(f.trip_distance_miles), 4)             AS avg_distance_miles,
        round(avg(f.trip_duration_seconds) / 60.0, 3)    AS avg_duration_minutes
    FROM fact_trip f
    JOIN geo_to_borough gb ON gb.geo_key  = f.pickup_geo_key
    JOIN date_to_dow    dd ON dd.date_key = f.pickup_date_key
    WHERE f.year = 2024   -- Hive partition column: prunes before reading
    GROUP BY ALL
)
SELECT
    b.borough,
    p.day_of_week,
    -- Day name is attached from the calendar at the end rather than carried
    -- through the aggregation as a string.
    (SELECT DISTINCT day_name FROM dim_date d
      WHERE d.day_of_week = p.day_of_week AND d.date_key > 0)  AS day_name,
    pickup_hour,
    trip_count,
    avg_total_amount,
    avg_distance_miles,
    avg_duration_minutes,
    -- Share of the borough's own demand falling in this cell, so boroughs of
    -- very different size remain visually comparable in the heatmap.
    round(100.0 * p.trip_count
          / sum(p.trip_count) OVER (PARTITION BY p.borough_id), 6)
                                                            AS pct_of_borough_trips,
    round(100.0 * p.trip_count / sum(p.trip_count) OVER (), 6)
                                                            AS pct_of_all_trips,
    -- Rank of this hour within its borough-and-weekday, for the written text.
    rank() OVER (PARTITION BY p.borough_id, p.day_of_week
                 ORDER BY p.trip_count DESC)                AS hour_rank_in_day
FROM profile p
JOIN borough_ids b ON b.borough_id = p.borough_id
ORDER BY b.borough, p.day_of_week, p.pickup_hour;
