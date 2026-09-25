-- ===========================================================================
-- A8  price_dispersion
--
-- Question   : How much does the price of the same commodity vary across
--              markets within a geopolitical zone, and how has that dispersion
--              moved over time?
-- Grain      : one row per (geopolitical zone, commodity+unit, month).
-- Output     : outputs/tables/A8_price_dispersion.csv
-- Figure     : A8_price_dispersion.png (small multiples by zone)
--
-- Why the coefficient of variation:
--   Nigerian staple prices rose several-fold over the window. A standard
--   deviation measured in naira would rise mechanically with the price level
--   and say nothing about market integration. The coefficient of variation
--   (sd / mean) is scale-free, so a rising CV means prices genuinely diverged
--   ACROSS markets rather than simply rose in all of them.
--
-- Assumptions:
--   1. Dispersion is measured across MARKETS within a zone, for one commodity
--      and unit and price type at a time. Pooling units or commodities would
--      measure product mix, not market integration.
--   2. Cells with fewer than {min_markets} reporting markets are EXCLUDED from
--      the dispersion statistics -- a CV over two markets is not informative --
--      but are counted in cells_below_threshold so the exclusion is visible.
--   3. The sample standard deviation (n-1) is used, being an estimate from a
--      sample of markets rather than a census.
--   4. Zones are assigned via dim_geography's ragged hierarchy: market ->
--      state -> geopolitical zone. Markets whose state could not be resolved
--      are excluded and reported.
--   5. Retail only, for the reason given in config/settings.py.
--   6. A high CV is NOT by itself evidence of market failure. Transport cost,
--      quality differences within a commodity label, and local harvest timing
--      all widen it. The measure describes dispersion, nothing more.
-- ===========================================================================
WITH market_prices AS (
    SELECT
        state_geo.region_group                              AS geopolitical_zone,
        market_geo.parent_geo_name                          AS state,
        market_geo.geo_name                                 AS market,
        c.commodity_name,
        c.unit,
        c.commodity_name || ' (' || c.unit || ')'           AS commodity_label,
        f.price_type,
        printf('%04d-%02d', f.month_key // 10000,
               (f.month_key // 100) % 100)                    AS year_month,
        f.month_key,
        f.price_ngn
    FROM fact_market_price_monthly f
    JOIN dim_commodity c   ON c.commodity_key = f.commodity_key
    JOIN dim_geography market_geo
                           ON market_geo.geo_key = f.geo_key
                          AND market_geo.admin_level = 'market'
    -- Second hop of the ragged hierarchy: market's parent state, to read its
    -- geopolitical zone. This is the same bridge A9 relies on.
    LEFT JOIN dim_geography state_geo
                           ON state_geo.geo_code    = market_geo.parent_geo_name
                          AND state_geo.admin_level = 'state'
                          AND state_geo.country     = 'Nigeria'
                          AND state_geo.is_current
    WHERE NOT c.is_fuel
      AND f.price_type = '{price_type}'
      AND f.price_ngn > 0
),
dispersion AS (
    SELECT
        geopolitical_zone,
        commodity_label,
        commodity_name,
        unit,
        year_month,
        month_key,
        count(DISTINCT market)                              AS reporting_markets,
        round(avg(price_ngn), 4)                            AS mean_price_ngn,
        round(median(price_ngn), 4)                         AS median_price_ngn,
        round(stddev_samp(price_ngn), 6)                    AS sd_price_ngn,
        round(min(price_ngn), 4)                            AS min_price_ngn,
        round(max(price_ngn), 4)                            AS max_price_ngn
    FROM market_prices
    WHERE geopolitical_zone IS NOT NULL
    GROUP BY ALL
)
SELECT
    geopolitical_zone,
    commodity_label,
    commodity_name,
    unit,
    year_month,
    reporting_markets,
    mean_price_ngn,
    median_price_ngn,
    sd_price_ngn,
    min_price_ngn,
    max_price_ngn,
    round(sd_price_ngn / nullif(mean_price_ngn, 0), 6)      AS coefficient_of_variation,
    round(100.0 * sd_price_ngn / nullif(mean_price_ngn, 0), 4)
                                                            AS cv_pct,
    round(max_price_ngn / nullif(min_price_ngn, 0), 4)      AS max_min_price_ratio,
    (reporting_markets >= {min_markets})                    AS meets_market_threshold,
    {min_markets}                                           AS min_markets_required,
    -- Twelve-month change in dispersion, for the written commentary.
    round(
        (sd_price_ngn / nullif(mean_price_ngn, 0))
      - lag(sd_price_ngn / nullif(mean_price_ngn, 0), 12) OVER (
            PARTITION BY geopolitical_zone, commodity_label ORDER BY month_key
        ), 6
    )                                                       AS cv_change_vs_12m_ago
FROM dispersion
ORDER BY geopolitical_zone, commodity_label, year_month;
