-- ===========================================================================
-- C1  engine_comparison_cloud -- BigQuery variant of A1 (demand_profile)
--
-- Question   : trip counts by borough, ISO day of week and hour of pickup, for
--              one calendar year -- the same question and output shape as the
--              local DuckDB variant (C1_demand_profile_duckdb.sql).
-- Input      : `{public_project}.{dataset}.{trip_table}`, pickups from
--              {start_date} to {end_date} (exclusive), joined to
--              `{public_project}.{dataset}.{zone_table}` for the borough.
--
-- THE YEAR MISMATCH, STATED UP FRONT: the public TLC dataset was not maintained
-- past 2022 and holds no 2024 rows, so this side uses 2019 while the local side
-- uses 2024. The two runs therefore count different trips. The comparison is
-- of architectures, not of results. See docs/cloud_architecture.md.
--
-- Semantic equivalence with the local side:
--   1. The same structural validity rules as the local silver layer are
--      applied, so both sides count "structurally valid trips": positive
--      duration of at most {max_duration_seconds} s, distance 0 to
--      {max_distance_miles} miles, non-negative fare and total, passenger count
--      NULL or 0 to {max_passengers}, pickup zone present.
--   2. Day of week is ISO (1 = Monday): BigQuery's DAYOFWEEK is 1 = Sunday, so
--      MOD(DAYOFWEEK + 5, 7) + 1 converts it.
--   3. Timestamps are cast to DATETIME so the query works whether a vintage
--      stores TIMESTAMP or DATETIME, and ids are compared as STRING.
-- ===========================================================================
WITH trips AS (
    SELECT
        CAST(pickup_datetime  AS DATETIME)  AS pickup_dt,
        CAST(dropoff_datetime AS DATETIME)  AS dropoff_dt,
        CAST(pickup_location_id AS STRING)  AS pickup_zone,
        trip_distance,
        fare_amount,
        total_amount,
        passenger_count
    FROM `{public_project}.{dataset}.{trip_table}`
    WHERE CAST(pickup_datetime AS DATETIME) >= DATETIME '{start_date}'
      AND CAST(pickup_datetime AS DATETIME) <  DATETIME '{end_date}'
),
valid AS (
    SELECT pickup_dt, pickup_zone
    FROM trips
    WHERE dropoff_dt > pickup_dt
      AND DATETIME_DIFF(dropoff_dt, pickup_dt, SECOND) <= {max_duration_seconds}
      AND trip_distance IS NOT NULL
      AND trip_distance BETWEEN 0 AND {max_distance_miles}
      AND fare_amount >= 0
      AND total_amount >= 0
      AND (passenger_count IS NULL OR passenger_count BETWEEN 0 AND {max_passengers})
      AND pickup_zone IS NOT NULL
)
SELECT
    COALESCE(z.borough, 'Unattributed')                AS borough,
    MOD(EXTRACT(DAYOFWEEK FROM v.pickup_dt) + 5, 7) + 1 AS day_of_week,
    EXTRACT(HOUR FROM v.pickup_dt)                     AS pickup_hour,
    COUNT(*)                                           AS trip_count
FROM valid v
LEFT JOIN `{public_project}.{dataset}.{zone_table}` z
       ON CAST(z.zone_id AS STRING) = v.pickup_zone
GROUP BY 1, 2, 3
