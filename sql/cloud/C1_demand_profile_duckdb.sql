-- ===========================================================================
-- C1  engine_comparison_cloud -- local DuckDB variant of A1 (demand_profile)
--
-- Question   : trip counts by borough, ISO day of week and hour of pickup, for
--              one calendar year -- the same question and output shape as
--              C1_demand_profile_bigquery.sql.
-- Input      : the local gold fact_trip (Hive-partitioned ZSTD Parquet),
--              calendar {local_year}, i.e. NYC TLC trips from {local_year}.
--
-- fact_trip already contains only structurally valid trips (the silver reject
-- rules), which the BigQuery variant reproduces as explicit filters. Pickup
-- year is filtered on the Hive partition column, so only that year's
-- partitions are read. The year differs from the cloud side (2024 here, 2019
-- there): a stated confounder, see docs/cloud_architecture.md.
-- ===========================================================================
SELECT
    coalesce(g.parent_geo_name, 'Unattributed')   AS borough,
    d.day_of_week,
    f.pickup_hour,
    count(*)                                      AS trip_count
FROM fact_trip f
JOIN dim_date      d ON d.date_key = f.pickup_date_key
JOIN dim_geography g ON g.geo_key  = f.pickup_geo_key
WHERE f.year = {local_year}
GROUP BY 1, 2, 3
