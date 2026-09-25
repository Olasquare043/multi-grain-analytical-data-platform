-- ===========================================================================
-- dim_vendor
--
-- Purpose : Decode VendorID, the technology provider that recorded the trip.
--           Vendor is a plausible confounder for any measurement artefact --
--           odometer precision, timestamp rounding, store-and-forward behaviour
--           -- so it must be joinable rather than buried in an integer.
-- Grain   : one row per licensed technology provider code.
-- Source  : NYC TLC Yellow Trips data dictionary (authoritative, not inferred):
--           https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf
-- SCD     : Type 1 (publisher-versioned dictionary).
--
-- Codes beyond those published are routed to the Unknown member by the fact
-- build rather than being invented here; if the 2024 corpus contains a vendor
-- code absent from this table, the quality report's unknown-rate metric will
-- show it.
-- ===========================================================================
CREATE OR REPLACE TABLE dim_vendor AS
SELECT * FROM (
    VALUES
        ({unknown_key}, -1, 'Unknown',                            'Not recorded or unpublished code'),
        (1,              1, 'Creative Mobile Technologies, LLC',  'Licensed technology provider'),
        (2,              2, 'Curb Mobility, LLC',                 'Licensed technology provider'),
        (3,              6, 'Myle Technologies Inc',              'Licensed technology provider'),
        (4,              7, 'Helix',                              'Licensed technology provider')
) AS t(vendor_key, vendor_id, vendor_name, vendor_description);
