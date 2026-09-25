-- ===========================================================================
-- dim_rate_code
--
-- Purpose : Decode the TLC RatecodeID so that negotiated, airport and group
--           fares can be separated from standard metered fares. A3 (congestion
--           proxy) and A4 (tipping) both need this separation, because a
--           negotiated JFK flat fare is not evidence about metered pricing.
-- Grain   : one row per rate code.
-- Source  : NYC TLC Yellow Trips data dictionary (authoritative, not inferred):
--           https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf
-- SCD     : Type 1 (publisher-versioned dictionary).
--
-- Note: code 99 is published as "Null/unknown". It is kept as a real member
-- rather than folded into the Unknown surrogate, because a row carrying an
-- explicit 99 is different evidence from a row carrying no rate code at all.
-- ===========================================================================
CREATE OR REPLACE TABLE dim_rate_code AS
SELECT * FROM (
    VALUES
        ({unknown_key}, -1, 'Unknown',                  'Not recorded',      FALSE),
        (1,              1, 'Standard rate',            'Metered city fare', FALSE),
        (2,              2, 'JFK',                      'Flat fare to/from JFK', TRUE),
        (3,              3, 'Newark',                   'Newark airport fare',   TRUE),
        (4,              4, 'Nassau or Westchester',    'Out-of-city fare',  FALSE),
        (5,              5, 'Negotiated fare',          'Fare agreed, not metered', FALSE),
        (6,              6, 'Group ride',               'Shared group ride', FALSE),
        (7,             99, 'Null/unknown',             'Published as 99 by TLC', FALSE)
) AS t(rate_key, rate_code_id, rate_code_name, rate_description, is_airport_rate);
