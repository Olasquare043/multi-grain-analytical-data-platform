-- ===========================================================================
-- C2  scale_behaviour -- one query, progressively larger inputs
--
-- Question   : trip counts by ISO day of week and hour of pickup. Deliberately
--              simpler than C1: it reads only the pickup timestamp, which every
--              yellow-trip vintage in the public dataset carries, so the same
--              query runs unchanged from one month to the full archive.
-- Input      : {sources_description}
--
-- The runner fills the FROM clause below with one SELECT per table, combined
-- with UNION ALL, each casting pickup_datetime to DATETIME so vintages stored as
-- TIMESTAMP and as DATETIME can be combined. Date filters are written against
-- the column's native type, so a partitioned table can prune; the inventory
-- records whether any table is partitioned, which explains the bytes billed
-- for the one-month step.
-- ===========================================================================
SELECT
    MOD(EXTRACT(DAYOFWEEK FROM pickup_dt) + 5, 7) + 1  AS day_of_week,
    EXTRACT(HOUR FROM pickup_dt)                       AS pickup_hour,
    COUNT(*)                                           AS trip_count
FROM (
{sources}
)
GROUP BY 1, 2
