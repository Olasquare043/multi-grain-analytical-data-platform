-- ===========================================================================
-- fact_trip
--
-- Purpose : The fine-grain end of the platform's grain spectrum, and the reason
--           the architecture has to be measured rather than asserted.
-- Grain   : ONE ROW PER COMPLETED TAXI TRIP.
--           ~41 million rows for calendar 2024.
-- Storage : Parquet, Hive-partitioned year=YYYY/month=M, under data/gold/fact_trip/.
--           Not a table in the DuckDB file: a partitioned Parquet dataset is
--           portable to any engine (including the BigQuery comparison in
--           section 9) and lets the partition-pruning benchmark B2 measure
--           something real.
--
-- Assumptions:
--   1. Only structurally valid trips are loaded. Rejected rows are counted by
--      reason in outputs/quality/quality_report.csv and the reconciliation
--      table; none is dropped silently.
--   2. Out-of-window pickups are LOADED, not rejected -- they are a genuine
--      defect of the official 2024 files. A pickup outside dim_date's span
--      resolves to pickup_date_key = {unknown_key} so it stays countable.
--   3. Every foreign key resolves. Unmatched codes route to the {unknown_key}
--      Unknown member rather than being lost to an inner join; the quality
--      report publishes the unknown rate per key.
--   4. trip_id is a degenerate dimension: deterministic MD5 of the row's
--      business content, stable across reruns, never random.
--   5. mode_key is constant (YELLOW_TAXI) for this load. See dim_transport_mode.
--
-- Written one month per statement so peak memory stays bounded on 8 GB, and so
-- a failed month can be rebuilt alone. {month_predicate} selects the month.
-- ===========================================================================
COPY (
    SELECT
        t.trip_id,
        coalesce(pd.date_key, {unknown_key})              AS pickup_date_key,
        coalesce(dd_.date_key, {unknown_key})             AS dropoff_date_key,
        CAST(hour(t.pickup_ts) AS TINYINT)                AS pickup_hour,
        coalesce(pg.geo_key, {unknown_key})               AS pickup_geo_key,
        coalesce(dg.geo_key, {unknown_key})               AS dropoff_geo_key,
        {mode_key}                                        AS mode_key,
        coalesce(v.vendor_key,  {unknown_key})            AS vendor_key,
        coalesce(pt.payment_key,{unknown_key})            AS payment_key,
        coalesce(rc.rate_key,   {unknown_key})            AS rate_key,
        coalesce(fl.flag_key,   {unknown_key})            AS flag_key,
        CAST(t.passenger_count AS SMALLINT)               AS passenger_count,
        t.trip_distance_miles,
        CAST(t.trip_duration_seconds AS INTEGER)          AS trip_duration_seconds,
        t.avg_speed_mph,
        t.fare_amount,
        t.extra,
        t.mta_tax,
        t.tip_amount,
        t.tolls_amount,
        t.improvement_surcharge,
        t.congestion_surcharge,
        t.airport_fee,
        t.total_amount,
        CAST(year(t.pickup_ts)  AS SMALLINT)              AS year,
        CAST(month(t.pickup_ts) AS TINYINT)               AS month
    FROM silver_nyc_trip t
    LEFT JOIN dim_date pd
           ON pd.full_date = CAST(t.pickup_ts AS DATE)
    LEFT JOIN dim_date dd_
           ON dd_.full_date = CAST(t.dropoff_ts AS DATE)
    LEFT JOIN dim_geography pg
           ON pg.geo_code    = CAST(t.pu_location_id AS VARCHAR)
          AND pg.country     = 'United States'
          AND pg.admin_level = 'zone'
          AND pg.is_current
    LEFT JOIN dim_geography dg
           ON dg.geo_code    = CAST(t.do_location_id AS VARCHAR)
          AND dg.country     = 'United States'
          AND dg.admin_level = 'zone'
          AND dg.is_current
    LEFT JOIN dim_vendor       v  ON v.vendor_id        = t.vendor_id
    LEFT JOIN dim_payment_type pt ON pt.payment_type_id = t.payment_type_id
    LEFT JOIN dim_rate_code    rc ON rc.rate_code_id    = t.rate_code_id
    LEFT JOIN dim_trip_flags   fl
           ON fl.store_and_fwd_flag IS NOT DISTINCT FROM t.store_and_fwd_flag
          AND fl.is_airport_trip    IS NOT DISTINCT FROM t.is_airport_trip
          AND fl.has_tip            IS NOT DISTINCT FROM t.has_tip
          AND fl.has_toll           IS NOT DISTINCT FROM t.has_toll
    WHERE t.reject_reason IS NULL
      AND {month_predicate}
) TO '{output_dir}'
  (FORMAT PARQUET, PARTITION_BY (year, month), OVERWRITE_OR_IGNORE,
   COMPRESSION ZSTD, FILENAME_PATTERN '{filename_pattern}');
