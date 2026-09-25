-- ===========================================================================
-- dim_date
--
-- Purpose : Conformed calendar shared by every fact table in the platform. It
--           is the join that makes a 41-million-row trip fact and a
--           one-row-per-state-per-month price fact addressable by the same
--           time predicate, which is the platform's central design claim.
-- Grain   : one row per calendar day.
-- Span    : {start_date} to {end_date} inclusive, wide enough to cover the WFP
--           panel (from 2002, truncated here to 2016) , the NBS fuel panel
--           (2023-11 onward) and the NYC trip window (2024).
--
-- Assumptions:
--   1. date_key is an INTEGER of the form YYYYMMDD. Integer surrogate keys on a
--      date dimension are a Kimball convention: they sort chronologically, are
--      human-legible in a query result, and cost 4 bytes per fact row instead of
--      the 8 a TIMESTAMP would.
--   2. day_of_week uses ISO numbering (1 = Monday .. 7 = Sunday), stated
--      explicitly because DuckDB's dayofweek() is 0 = Sunday and the two are
--      trivially confused.
--   3. Rows generated, never sourced. Nothing here is derived from input data,
--      so this table cannot be a vector for fabricated numbers.
--   4. An Unknown member at date_key = {unknown_key} carries facts whose date
--      falls outside the generated span. The NYC corpus genuinely contains such
--      rows (a pickup stamped 2002-12-31); routing them to an Unknown member
--      keeps referential integrity total while leaving the anomaly countable.
-- ===========================================================================
CREATE OR REPLACE TABLE dim_date AS
WITH calendar AS (
    SELECT CAST(generated AS DATE) AS full_date
    FROM generate_series(
        DATE '{start_date}',
        DATE '{end_date}',
        INTERVAL 1 DAY
    ) AS t(generated)
)
SELECT
    CAST(strftime(full_date, '%Y%m%d') AS INTEGER)          AS date_key,
    full_date,
    CAST(year(full_date)    AS SMALLINT)                    AS year,
    CAST(quarter(full_date) AS TINYINT)                     AS quarter,
    CAST(month(full_date)   AS TINYINT)                     AS month_number,
    strftime(full_date, '%B')                               AS month_name,
    CAST(day(full_date)     AS TINYINT)                     AS day_of_month,
    CAST(isodow(full_date)  AS TINYINT)                     AS day_of_week,
    strftime(full_date, '%A')                               AS day_name,
    (isodow(full_date) >= 6)                                AS is_weekend,
    CAST(week(full_date)    AS TINYINT)                     AS week_of_year,
    strftime(full_date, '%Y-%m')                            AS year_month,
    (full_date = last_day(full_date))                       AS is_month_end
FROM calendar

UNION ALL

SELECT
    CAST({unknown_key} AS INTEGER)  AS date_key,
    CAST(NULL AS DATE)              AS full_date,
    CAST(NULL AS SMALLINT)          AS year,
    CAST(NULL AS TINYINT)           AS quarter,
    CAST(NULL AS TINYINT)           AS month_number,
    'Unknown'                       AS month_name,
    CAST(NULL AS TINYINT)           AS day_of_month,
    CAST(NULL AS TINYINT)           AS day_of_week,
    'Unknown'                       AS day_name,
    CAST(NULL AS BOOLEAN)           AS is_weekend,
    CAST(NULL AS TINYINT)           AS week_of_year,
    'Unknown'                       AS year_month,
    CAST(NULL AS BOOLEAN)           AS is_month_end

ORDER BY date_key;
