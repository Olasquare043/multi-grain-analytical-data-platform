-- ===========================================================================
-- dim_trip_flags   (JUNK DIMENSION)
--
-- Purpose : Collapse four low-cardinality trip attributes that have no natural
--           home of their own into one dimension, so fact_trip carries a single
--           4-byte flag_key instead of four separate columns.
--           Across 41,169,720 fact rows that is a real, measurable saving, and
--           it is the textbook case for a junk dimension.
-- Grain   : one row per OBSERVED combination of the four flags.
--           Materialising only observed combinations, rather than the Cartesian
--           product, is the point: the theoretical maximum is 3 x 2 x 2 x 2 = 24
--           (store_and_fwd_flag is 'Y', 'N' or NULL), and the build reports how
--           many actually occur.
-- Source  : derived from silver_nyc_trip.
-- SCD     : Type 1; a combination that stops occurring is not deleted, because
--           facts already reference it.
--
-- Assumptions:
--   1. is_airport_trip is TRUE when EITHER endpoint is a TLC airport zone
--      (Newark 1, JFK 132, LaGuardia 138). Directionality is recoverable from
--      the zone keys on the fact, so it is not duplicated here.
--   2. has_tip / has_toll are existence flags, not amounts. The amounts stay on
--      the fact as additive measures; the flags exist so that a filter on
--      "trips that tipped" does not have to scan a DOUBLE column.
--   3. NULL store_and_fwd_flag is retained as its own member rather than
--      defaulted, because "not recorded" is different evidence from "N".
-- ===========================================================================
CREATE OR REPLACE TABLE dim_trip_flags AS
WITH observed AS (
    SELECT DISTINCT
        store_and_fwd_flag,
        is_airport_trip,
        has_tip,
        has_toll
    FROM silver_nyc_trip
    WHERE reject_reason IS NULL
)
SELECT
    CAST({unknown_key} AS INTEGER) AS flag_key,
    CAST(NULL AS VARCHAR)          AS store_and_fwd_flag,
    CAST(NULL AS BOOLEAN)          AS is_airport_trip,
    CAST(NULL AS BOOLEAN)          AS has_tip,
    CAST(NULL AS BOOLEAN)          AS has_toll

UNION ALL

SELECT
    CAST(row_number() OVER (
        ORDER BY coalesce(store_and_fwd_flag, 'ZZ'),
                 is_airport_trip, has_tip, has_toll
    ) AS INTEGER)                  AS flag_key,
    store_and_fwd_flag,
    is_airport_trip,
    has_tip,
    has_toll
FROM observed
ORDER BY flag_key;
