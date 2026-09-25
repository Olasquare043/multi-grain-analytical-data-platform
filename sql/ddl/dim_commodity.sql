-- ===========================================================================
-- dim_commodity
--
-- Purpose : Conform the priced goods in the WFP panel, and mark which of them
--           are fuel. A6 and A7 must be able to exclude fuel from a *food* index
--           without hard-coding commodity names into every query.
-- Grain   : one row per (commodity, unit) as published by WFP.
--           Unit is part of the grain because WFP prices the same commodity in
--           different units across markets (KG, 100 KG, litre, unit), and a
--           price is meaningless without it. Collapsing commodity alone would
--           average 1 KG of rice with 100 KG of rice.
-- Source  : derived from silver_wfp_prices, itself Source C.
-- SCD     : Type 1.
--
-- is_fuel is set from the published category plus the two known fuel commodity
-- names, so the flag survives WFP moving a commodity between categories.
-- ===========================================================================
CREATE OR REPLACE TABLE dim_commodity AS
WITH observed AS (
    SELECT DISTINCT
        commodity,
        coalesce(unit, 'unspecified')                  AS unit,
        category,
        commodity_id
    FROM read_parquet('{silver_wfp}')
    WHERE commodity IS NOT NULL AND commodity <> ''
)
SELECT
    CAST({unknown_key} AS INTEGER)                     AS commodity_key,
    'UNKNOWN'                                          AS commodity_code,
    'Unknown'                                          AS commodity_name,
    'Unknown'                                          AS category,
    'unspecified'                                      AS unit,
    FALSE                                              AS is_fuel

UNION ALL

SELECT
    CAST(row_number() OVER (ORDER BY commodity, unit) AS INTEGER) AS commodity_key,
    upper(regexp_replace(commodity || '_' || unit, '[^A-Za-z0-9]+', '_', 'g'))
                                                       AS commodity_code,
    commodity                                          AS commodity_name,
    category,
    unit,
    (category = '{nonfood_category}' OR commodity IN {fuel_commodities})
                                                       AS is_fuel
FROM observed
ORDER BY commodity_key;
