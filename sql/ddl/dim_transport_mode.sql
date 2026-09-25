-- ===========================================================================
-- dim_transport_mode
--
-- Purpose : Name the mode of movement a fact row measures. Today the platform
--           loads one mode, so this dimension holds a single row and buys
--           nothing at query time. It exists because the alternative -- assuming
--           "all trips are yellow taxis" -- is the assumption that would have to
--           be unpicked from every fact table and every query the first time
--           green cabs, for-hire vehicles or a Nigerian mode are added.
--           Carrying it now costs one join and one row.
-- Grain   : one row per transport mode.
-- SCD     : Type 1.
--
-- country_scope is deliberately not a foreign key to dim_geography: a mode is
-- scoped to a jurisdiction's regulation, not to a place in the hierarchy.
-- ===========================================================================
CREATE OR REPLACE TABLE dim_transport_mode AS
SELECT * FROM (
    VALUES
        ({unknown_key}, 'UNKNOWN',      'Unknown',               'unknown',
         'Unknown',       FALSE),
        (1,             'YELLOW_TAXI',  'NYC Yellow Medallion Taxi',
         'road_passenger', 'United States', TRUE)
) AS t(mode_key, mode_code, mode_name, mode_category, country_scope, is_metered);
