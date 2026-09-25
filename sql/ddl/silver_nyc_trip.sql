-- ===========================================================================
-- silver_nyc_trip  (VIEW)
--
-- Purpose : Cleanse, type and derive over the registered NYC TLC 2024 corpus,
--           and attach a single reject reason per row so that the gold build
--           and the quality report cannot disagree about what was excluded.
-- Grain   : one raw trip record from Source A (41,169,720 rows before rejection).
-- Engine  : DuckDB 1.1.x
--
-- Assumptions, each of which is also stated in docs/methodology_notes.md:
--   1. A trip is *structurally* valid when it has both timestamps, a strictly
--      positive duration no greater than 24 hours, a non-negative and plausible
--      distance, non-negative fare and total, a plausible passenger count, and
--      both location ids present. Exactly one reject reason is assigned, in the
--      declared precedence order, so reasons partition the rejected set.
--   2. Out-of-window pickups are NOT a rejection. The January 2024 file
--      genuinely contains a pickup stamped 2002-12-31 22:59:39; that is a defect
--      of the official data. Such rows are retained and flagged
--      (is_outside_declared_window) so the defect can be quantified rather than
--      quietly erased. Analyses restrict to the declared window via dim_date.
--   3. avg_speed_mph is a derived measure, meaningless where duration is zero;
--      duration zero is already a rejection, so the division is safe here.
--   4. TLC publishes no trip identifier. trip_id is a deterministic MD5 over the
--      business content of the row, so it is stable across reruns (idempotence,
--      constraint 2.7) and never random.
--
-- Placeholders: {trip_relation} is the registered raw Parquet scan.
-- ===========================================================================
CREATE OR REPLACE VIEW silver_nyc_trip AS
WITH raw AS (
    SELECT
        regexp_replace(filename, '^.*/', '')                     AS source_file,
        CAST("VendorID"            AS INTEGER)                   AS vendor_id,
        CAST("tpep_pickup_datetime"  AS TIMESTAMP)               AS pickup_ts,
        CAST("tpep_dropoff_datetime" AS TIMESTAMP)               AS dropoff_ts,
        CAST("passenger_count"     AS BIGINT)                    AS passenger_count,
        CAST("trip_distance"       AS DOUBLE)                    AS trip_distance_miles,
        CAST("RatecodeID"          AS BIGINT)                    AS rate_code_id,
        CAST("store_and_fwd_flag"  AS VARCHAR)                   AS store_and_fwd_flag,
        CAST("PULocationID"        AS INTEGER)                   AS pu_location_id,
        CAST("DOLocationID"        AS INTEGER)                   AS do_location_id,
        CAST("payment_type"        AS BIGINT)                    AS payment_type_id,
        CAST("fare_amount"         AS DOUBLE)                    AS fare_amount,
        CAST("extra"               AS DOUBLE)                    AS extra,
        CAST("mta_tax"             AS DOUBLE)                    AS mta_tax,
        CAST("tip_amount"          AS DOUBLE)                    AS tip_amount,
        CAST("tolls_amount"        AS DOUBLE)                    AS tolls_amount,
        CAST("improvement_surcharge" AS DOUBLE)                  AS improvement_surcharge,
        CAST("total_amount"        AS DOUBLE)                    AS total_amount,
        CAST("congestion_surcharge" AS DOUBLE)                   AS congestion_surcharge,
        CAST({airport_fee_column}  AS DOUBLE)                    AS airport_fee
    FROM {trip_relation}
),
derived AS (
    SELECT
        raw.*,
        date_diff('second', pickup_ts, dropoff_ts)               AS trip_duration_seconds,
        (pickup_ts <  TIMESTAMP '{window_start}'
         OR pickup_ts >= TIMESTAMP '{window_end}')               AS is_outside_declared_window
    FROM raw
),
judged AS (
    SELECT
        derived.*,
        -- Precedence order matters: reasons must partition the rejected set so
        -- that the reconciliation table's per-reason counts sum exactly to the
        -- number of rows excluded.
        CASE
            WHEN pickup_ts IS NULL OR dropoff_ts IS NULL
                THEN 'null_timestamp'
            WHEN trip_duration_seconds <= 0
                THEN 'non_positive_duration'
            WHEN trip_duration_seconds > {max_duration_seconds}
                THEN 'duration_exceeds_ceiling'
            WHEN trip_distance_miles IS NULL OR trip_distance_miles < 0
                THEN 'negative_or_null_distance'
            WHEN trip_distance_miles > {max_distance_miles}
                THEN 'implausible_distance'
            WHEN fare_amount < 0 OR total_amount < 0
                THEN 'negative_money'
            WHEN passenger_count < 0 OR passenger_count > {max_passengers}
                THEN 'implausible_passenger_count'
            WHEN pu_location_id IS NULL OR do_location_id IS NULL
                THEN 'null_location_id'
            ELSE NULL
        END                                                      AS reject_reason
    FROM derived
)
SELECT
    md5(concat_ws('|',
        CAST(vendor_id AS VARCHAR),
        CAST(pickup_ts AS VARCHAR),
        CAST(dropoff_ts AS VARCHAR),
        CAST(pu_location_id AS VARCHAR),
        CAST(do_location_id AS VARCHAR),
        CAST(trip_distance_miles AS VARCHAR),
        CAST(fare_amount AS VARCHAR),
        CAST(total_amount AS VARCHAR),
        CAST(payment_type_id AS VARCHAR)
    ))                                                           AS trip_id,
    judged.*,
    CASE
        WHEN trip_duration_seconds > 0
            THEN trip_distance_miles / (trip_duration_seconds / 3600.0)
    END                                                          AS avg_speed_mph,
    (pu_location_id IN {airport_zone_ids}
     OR do_location_id IN {airport_zone_ids})                    AS is_airport_trip,
    (coalesce(tip_amount, 0)   > 0)                              AS has_tip,
    (coalesce(tolls_amount, 0) > 0)                              AS has_toll
FROM judged;
