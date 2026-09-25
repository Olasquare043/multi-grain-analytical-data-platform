-- ===========================================================================
-- A11  missingness_structure          *** SUPPLEMENTARY ANALYSIS, ADDED ***
--
-- This analysis is NOT in the original specification. It was added because the
-- data quality framework surfaced something the specification did not
-- anticipate, and it speaks directly to the paper's subject: not what the
-- movement data says about the world, but what the data says about the systems
-- that produced it.
--
-- Question   : Are the missing fields in the NYC 2024 corpus missing
--              independently, or do they fail as a block? And does the failure
--              rate vary by vendor and over time?
-- Grain      : two record types in one table, distinguished by record_type --
--                'pattern'       one row per observed missingness combination
--                'vendor_month'  one row per (vendor, month)
-- Output     : outputs/tables/A11_missingness_structure.csv
-- Figure     : A11_missingness_structure.png
--
-- WHAT PROMPTED IT
--   Quality rules Q012, Q013 and Q024 each independently reported a null rate
--   of 9.801145 per cent on three unrelated fields -- passenger_count,
--   congestion_surcharge and RatecodeID. Three different fields agreeing to six
--   decimal places is not a coincidence; it is a signature. Either the same
--   rows are missing all three, or the report is wrong.
--
-- WHY IT MATTERS FOR THE PAPER
--   If fields were missing independently, an imputation strategy could be
--   argued for each column separately. If they fail as a block, the rows are
--   arriving through a different upstream path with a reduced schema, and no
--   per-column treatment is defensible -- the correct engineering response is
--   to model them as a distinct record class, which is exactly what routing
--   them to the Unknown dimension member does.
--
-- Assumptions:
--   1. Declared window only (calendar 2024).
--   2. "Missing RatecodeID" is detected as rate_key = -1, i.e. the fact routed
--      the row to the Unknown member. Published code 99 ("Null/unknown") is a
--      REAL value and is deliberately not counted as missing here; conflating
--      an explicit 99 with an absent field would destroy the distinction the
--      analysis exists to test.
--   3. Vendor-month rows are suppressed below 1,000 trips, since a rate over a
--      handful of trips is noise. The suppressed count is reported.
--   4. This analysis describes the RECORDING SYSTEM, not taxi passengers. No
--      inference about trips or demand is drawn from it anywhere.
-- ===========================================================================
WITH flagged AS (
    SELECT
        v.vendor_name,
        d.year_month,
        (f.passenger_count      IS NULL)  AS missing_passenger_count,
        (f.congestion_surcharge IS NULL)  AS missing_congestion_surcharge,
        (f.rate_key             = -1)     AS missing_rate_code,
        (f.airport_fee          IS NULL)  AS missing_airport_fee
    FROM fact_trip f
    JOIN dim_date   d ON d.date_key   = f.pickup_date_key
    JOIN dim_vendor v ON v.vendor_key = f.vendor_key
    WHERE f.year = 2024   -- Hive partition column: prunes before reading
),
patterns AS (
    SELECT
        'pattern'                                              AS record_type,
        CAST(NULL AS VARCHAR)                                  AS vendor_name,
        CAST(NULL AS VARCHAR)                                  AS year_month,
        -- A compact signature such as 'PCRA' -> which fields are absent.
        (CASE WHEN missing_passenger_count      THEN 'P' ELSE '-' END) ||
        (CASE WHEN missing_congestion_surcharge THEN 'C' ELSE '-' END) ||
        (CASE WHEN missing_rate_code            THEN 'R' ELSE '-' END) ||
        (CASE WHEN missing_airport_fee          THEN 'A' ELSE '-' END)
                                                               AS missingness_pattern,
        CAST(missing_passenger_count      AS INTEGER)
      + CAST(missing_congestion_surcharge AS INTEGER)
      + CAST(missing_rate_code            AS INTEGER)
      + CAST(missing_airport_fee          AS INTEGER)          AS fields_missing,
        count(*)                                               AS trip_count,
        round(100.0 * count(*) / sum(count(*)) OVER (), 6)     AS pct_of_trips,
        CAST(NULL AS DOUBLE)                                   AS pct_missing_passenger,
        CAST(NULL AS DOUBLE)                                   AS pct_missing_congestion,
        CAST(NULL AS DOUBLE)                                   AS pct_missing_rate_code,
        CAST(NULL AS DOUBLE)                                   AS pct_all_three_missing
    FROM flagged
    GROUP BY ALL
),
vendor_month AS (
    SELECT
        'vendor_month'                                         AS record_type,
        vendor_name,
        year_month,
        CAST(NULL AS VARCHAR)                                  AS missingness_pattern,
        CAST(NULL AS INTEGER)                                  AS fields_missing,
        count(*)                                               AS trip_count,
        round(100.0 * count(*) / sum(count(*)) OVER (), 6)     AS pct_of_trips,
        round(100.0 * count(*) FILTER (WHERE missing_passenger_count)
              / count(*), 6)                                   AS pct_missing_passenger,
        round(100.0 * count(*) FILTER (WHERE missing_congestion_surcharge)
              / count(*), 6)                                   AS pct_missing_congestion,
        round(100.0 * count(*) FILTER (WHERE missing_rate_code)
              / count(*), 6)                                   AS pct_missing_rate_code,
        -- The test of the hypothesis: if the three fail as a block, this equals
        -- each of the three individual rates exactly.
        round(100.0 * count(*) FILTER (
                  WHERE missing_passenger_count
                    AND missing_congestion_surcharge
                    AND missing_rate_code)
              / count(*), 6)                                   AS pct_all_three_missing
    FROM flagged
    GROUP BY ALL
    HAVING count(*) >= 1000
)
SELECT * FROM patterns
UNION ALL
SELECT * FROM vendor_month
ORDER BY record_type, trip_count DESC, vendor_name, year_month;
