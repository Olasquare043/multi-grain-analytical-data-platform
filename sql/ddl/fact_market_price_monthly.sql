-- ===========================================================================
-- fact_market_price_monthly
--
-- Purpose : The middle grain of the platform. Nigerian market food prices,
--           sharing dim_date and dim_geography with a fact table 500x its size.
-- Grain   : ONE ROW PER (market, commodity, price type, month).
--           geo_key resolves to a *market*, so this fact sits one tier below
--           fact_fuel_price_monthly in the same ragged hierarchy -- which is
--           exactly why A9 needs an explicit bridge between them.
-- Source  : silver_wfp_prices (Source C, WFP via HDX).
--
-- Assumptions:
--   1. WFP publishes at most one observation per market/commodity/unit/type per
--      month. observation_count records how many source rows actually collapsed
--      into each cell, so a vintage that starts publishing twice a month becomes
--      visible instead of being silently averaged away.
--   2. price_ngn is the published local-currency price; price_usd is WFP's own
--      conversion, carried but never used to construct an index, because the
--      exchange rate would then drive the series.
--   3. Unit is part of dim_commodity's grain, not a measure. A price of 1 KG and
--      a price of 100 KG of the same commodity are different commodity members.
--   4. The panel is UNBALANCED: reporting markets per month vary substantially.
--      Any national aggregate over this table without holding composition fixed
--      measures panel composition as much as price. See A6 and
--      docs/methodology_notes.md.
-- ===========================================================================
CREATE OR REPLACE TABLE fact_market_price_monthly AS
SELECT
    CAST(p.month_key AS INTEGER)                       AS month_key,
    coalesce(g.geo_key, {unknown_key})                 AS geo_key,
    coalesce(c.commodity_key, {unknown_key})           AS commodity_key,
    p.price_type,
    round(avg(p.price_ngn), 4)                         AS price_ngn,
    round(avg(p.price_usd), 6)                         AS price_usd,
    coalesce(p.unit, 'unspecified')                    AS unit,
    CAST(count(*) AS INTEGER)                          AS observation_count
FROM read_parquet('{silver_wfp}') p
LEFT JOIN dim_geography g
       ON g.geo_code    = CAST(coalesce(CAST(p.market_id AS VARCHAR), p.market)
                               AS VARCHAR)
      AND g.country     = 'Nigeria'
      AND g.admin_level = 'market'
      AND g.is_current
LEFT JOIN dim_commodity c
       ON c.commodity_name = p.commodity
      AND c.unit           = coalesce(p.unit, 'unspecified')
GROUP BY ALL
ORDER BY month_key, geo_key, commodity_key, price_type;
