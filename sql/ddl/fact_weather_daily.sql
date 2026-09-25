-- ===========================================================================
-- fact_weather_daily   (OPTIONAL)
--
-- Purpose : Contextual daily weather for New York City, demonstrating that a
--           REST/JSON source joins into the same conformed dim_date as a bulk
--           Parquet source. The core pipeline must succeed whether or not this
--           table exists (section 3, Source E is non-blocking).
-- Grain   : ONE ROW PER (date, geography).
--           In practice one row per date, because Open-Meteo publishes a single
--           point series for the requested coordinate.
-- Source  : silver_weather_daily (Source E, Open-Meteo archive API).
--
-- Assumptions:
--   1. geo_key resolves to the CITY-level New York City member of
--      dim_geography, not to a taxi zone. The series is a single coordinate in
--      lower Manhattan; attributing it to one of 262 zones would imply a spatial
--      resolution the source does not have.
--   2. Units are as published: degrees Celsius, millimetres, km/h. They are
--      named in the column identifiers so no query has to guess.
--   3. No weather measure is used in any causal claim anywhere in this study.
--      The table exists to demonstrate conformance, not to explain demand.
-- ===========================================================================
CREATE OR REPLACE TABLE fact_weather_daily AS
SELECT
    w.date_key,
    coalesce(g.geo_key, {unknown_key})    AS geo_key,
    w.temp_max_c,
    w.temp_min_c,
    w.precipitation_mm,
    w.snowfall_mm,
    w.wind_speed_max_kmh
FROM read_parquet('{silver_weather}') w
LEFT JOIN dim_geography g
       ON g.geo_code    = '{weather_geo_code}'
      AND g.admin_level = 'city'
      AND g.is_current
ORDER BY w.date_key;
