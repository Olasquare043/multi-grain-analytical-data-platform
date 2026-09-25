-- ===========================================================================
-- dim_payment_type
--
-- Purpose : Decode the TLC payment_type code so that analyses read in business
--           language rather than integers. A4 (tipping behaviour) depends on it.
-- Grain   : one row per payment type code.
-- Source  : NYC TLC Yellow Trips data dictionary (authoritative, not inferred):
--           https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf
-- SCD     : Type 1. The publisher versions this dictionary; a correction here is
--           a correction, not a new historical truth. See docs/scd_strategy.md.
--
-- Assumption carried into A4: the dictionary states that cash tips are NOT
-- recorded by the meter. Tip measures are therefore only interpretable for
-- payment_type = 1 (Credit card); every other type has a structural zero that
-- must not be read as generosity. is_tip_observable encodes that fact once, so
-- no analysis has to remember it.
-- ===========================================================================
CREATE OR REPLACE TABLE dim_payment_type AS
SELECT * FROM (
    VALUES
        ({unknown_key}, -1, 'Unknown',      'Not recorded',                FALSE),
        (1,              0, 'Flex Fare',    'Flex Fare trip',              FALSE),
        (2,              1, 'Credit card',  'Card payment, tip metered',   TRUE),
        (3,              2, 'Cash',         'Cash payment, tip NOT metered', FALSE),
        (4,              3, 'No charge',    'No charge',                   FALSE),
        (5,              4, 'Dispute',      'Disputed fare',               FALSE),
        (6,              5, 'Unknown',      'Unknown to the vendor',       FALSE),
        (7,              6, 'Voided trip',  'Voided trip',                 FALSE)
) AS t(payment_key, payment_type_id, payment_type_name, payment_description,
       is_tip_observable);
