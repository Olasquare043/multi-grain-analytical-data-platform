-- ===========================================================================
-- dim_geography
--
-- Purpose : The dimension carrying the paper's central design argument. ONE
--           conformed geography dimension serves two countries, three
--           administrative tiers and three fact tables whose grains differ by
--           seven orders of magnitude. A ragged hierarchy is used in preference
--           to separate per-country dimensions, so that a single join path
--           reaches a New York taxi zone and a Nigerian state alike.
-- Grain   : one row per (geo_code, country, admin_level) VERSION.
--           Type 2 slowly changing, so a natural key may hold several rows of
--           which exactly one is current.
--
-- Hierarchy, deliberately ragged:
--   United States : zone   -> parent_geo_name = borough
--   Nigeria       : market -> parent_geo_name = state (admin1)
--   Nigeria       : state  -> parent_geo_name = 'Nigeria'
--   United States : city   -> parent_geo_name = 'United States'
--
--   'city' is a documented extension to the specified zone/market/state
--   domain. The optional fact_weather_daily must carry a geo_key, and
--   Open-Meteo publishes one point series for New York City rather than
--   zone-level weather. Attributing city-wide weather to an arbitrary taxi zone
--   would misrepresent it, so it gets its own member at its true resolution.
--
-- Type 2 mechanics:
--   valid_from / valid_to bound each version; valid_to = '9999-12-31' and
--   is_current = TRUE identify the live row. Tracked attributes are geo_name,
--   parent_geo_name and region_group -- a change to any of them closes the
--   prior version and opens a new one. geo_code, country and admin_level form
--   the natural key and by definition never change.
--   See docs/scd_strategy.md and tests/test_scd2_geography.py.
--
-- Unknown member:
--   geo_key = {unknown_key} guarantees total referential integrity. A trip whose
--   PULocationID is absent from the lookup keeps a valid foreign key and stays
--   countable, instead of vanishing from an inner join.
--
-- This file creates the typed, empty table. Population is a merge, not an
-- INSERT ... SELECT, and lives in src/model/dim_geography.py, because Type 2
-- requires comparing incoming rows against the versions already stored.
-- ===========================================================================
CREATE TABLE IF NOT EXISTS dim_geography (
    geo_key           INTEGER  NOT NULL,   -- surrogate; {unknown_key} = Unknown
    geo_code          VARCHAR  NOT NULL,   -- natural key within country + level
    geo_name          VARCHAR,             -- Type 2 tracked
    country           VARCHAR  NOT NULL,
    admin_level       VARCHAR  NOT NULL,   -- zone | market | state | city
    parent_geo_name   VARCHAR,             -- Type 2 tracked
    region_group      VARCHAR,             -- Type 2 tracked
    valid_from        DATE     NOT NULL,
    valid_to          DATE     NOT NULL,   -- '9999-12-31' while current
    is_current        BOOLEAN  NOT NULL,
    PRIMARY KEY (geo_key)
);
