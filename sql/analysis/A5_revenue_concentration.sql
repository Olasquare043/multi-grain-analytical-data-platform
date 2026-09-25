-- ===========================================================================
-- A5  revenue_concentration
--
-- Question   : How concentrated is yellow-taxi revenue across pickup zones?
-- Grain      : one row per pickup zone, ranked by revenue, carrying the running
--              cumulative share (a Pareto curve).
-- Output     : outputs/tables/A5_revenue_concentration.csv
-- Figure     : A5_revenue_concentration.png (Pareto)
--
-- Assumptions:
--   1. Declared window only (calendar 2024).
--   2. Revenue is sum(total_amount): what the passenger paid, including tolls,
--      surcharges, taxes and recorded tips. It is NOT operator net revenue, and
--      no claim about operator economics is made from it.
--   3. Attributed to the PICKUP zone. A trip's revenue is credited to where it
--      originated, matching how a fleet would think about where to position
--      vehicles. A dropoff-keyed version would be a different analysis.
--   4. The two non-geographic zones are retained here, unlike A2, because
--      excluding them would overstate the concentration of the real zones by
--      redistributing their share. Their rank is visible in the output.
--   5. Cumulative share uses an unbounded preceding window over the revenue
--      ordering, so ties are broken deterministically by zone name and the
--      curve is reproducible run to run.
-- ===========================================================================
WITH zone_revenue AS (
    SELECT
        g.geo_key,
        g.geo_name                                       AS pickup_zone,
        coalesce(g.parent_geo_name, 'Unattributed')      AS borough,
        count(*)                                         AS trip_count,
        round(sum(f.total_amount), 4)                    AS total_revenue,
        round(avg(f.total_amount), 4)                    AS mean_revenue_per_trip,
        round(sum(f.trip_distance_miles), 4)             AS total_distance_miles
    FROM fact_trip f
    JOIN dim_geography g ON g.geo_key  = f.pickup_geo_key
    -- Hive partition column, not a dim_date join: prunes whole partitions
    -- before reading, and dim_date contributes nothing else here.
    WHERE f.year = 2024
    GROUP BY ALL
),
ranked AS (
    SELECT
        zone_revenue.*,
        row_number() OVER w                              AS revenue_rank,
        sum(total_revenue)  OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                                                         AS cumulative_revenue,
        sum(total_revenue)  OVER ()                      AS grand_total_revenue,
        sum(trip_count)     OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                                                         AS cumulative_trips,
        sum(trip_count)     OVER ()                      AS grand_total_trips,
        count(*)            OVER ()                      AS zone_count
    FROM zone_revenue
    WINDOW w AS (ORDER BY total_revenue DESC, pickup_zone)
)
SELECT
    revenue_rank,
    pickup_zone,
    borough,
    trip_count,
    total_revenue,
    mean_revenue_per_trip,
    total_distance_miles,
    round(100.0 * total_revenue / grand_total_revenue, 6)       AS pct_of_revenue,
    round(100.0 * cumulative_revenue / grand_total_revenue, 6)  AS cumulative_pct_revenue,
    round(100.0 * cumulative_trips / grand_total_trips, 6)      AS cumulative_pct_trips,
    round(100.0 * revenue_rank / zone_count, 6)                 AS cumulative_pct_zones,
    -- The headline Pareto statement, computed rather than eyeballed off a chart.
    (round(100.0 * cumulative_revenue / grand_total_revenue, 6) <= 80.0)
                                                                AS within_top_80pct_revenue
FROM ranked
ORDER BY revenue_rank;
