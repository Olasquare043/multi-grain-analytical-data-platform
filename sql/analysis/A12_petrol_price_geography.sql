-- ===========================================================================
-- A12  petrol_price_geography          *** CENTREPIECE OF THE NIGERIAN ANALYSIS
--
-- Question   : Across the post-deregulation period, how did the national petrol
--              price move; how far apart were the states; which states were
--              persistently dearer or cheaper than the nation; how stable was
--              the ordering of states; and how do the six geopolitical zones
--              compare?
-- Source     : fact_fuel_price_monthly ALONE (Source D, NBS PMS Price Watch).
--              Complete coverage: 37 federating units x 31 months (2023-11 to
--              2026-05), no gaps, and its national mean reconciles EXACTLY with
--              the Bureau's published figures (tests/test_pms_reconciliation.py).
--              Unlike A6-A9, no coverage caveat applies.
-- Outputs    : four grains, one statement each (see the @output markers):
--   main      A12_petrol_price_geography.csv           one row per (state, month)
--   national  A12_petrol_price_geography_national.csv  one row per month
--   states    A12_petrol_price_geography_states.csv    one row per state
--   zones     A12_petrol_price_geography_zones.csv     one row per (zone, month)
-- Figures    : A12_petrol_price_national.png, A12_petrol_price_state_premiums.png,
--              A12_petrol_price_rank_stability.png, A12_petrol_price_zones.png
--
-- Definitions and assumptions:
--   1. NATIONAL MEAN = the unweighted mean of the 37 state prices. This is the
--      Bureau's own definition: it is the figure that reconciles to 0.00 NGN
--      against the published national averages. It is not weighted by fuel
--      volume or population, because NBS publishes neither by state.
--   2. "Deregulation period": the petrol subsidy was removed on 29 May 2023;
--      this series begins in November 2023, six months later. Every figure
--      here therefore describes the post-removal market, not the transition.
--   3. DISPERSION uses the POPULATION standard deviation. The 37 federating
--      units are the whole population, not a sample of states, so no n-1
--      correction is appropriate. CV = sd / mean, scale-free, so a rise means
--      the states genuinely moved apart rather than the price level rising.
--   4. PREMIUM = state price / national mean - 1, in per cent. The national
--      mean includes the state itself; premiums therefore sum to exactly zero
--      across states in every month, which the test suite asserts.
--   5. RANKS: 1 = most expensive. Tied prices (11 tie groups in the panel, none
--      larger than 3 states) receive the AVERAGE of the ranks they span, so the
--      rank correlations below are exact Spearman coefficients.
--   6. RANK STABILITY: Spearman's rho between each month's ranking and (a) the
--      previous month's and (b) the first month's. (a) measures churn; (b)
--      measures how quickly the initial geography of prices decays.
--   7. PERSISTENCE: a state holds a persistent premium (discount) when it is
--      above (below) the national mean in at least {persistence_share} of months.
--      The longest unbroken run above and below the mean is also reported --
--      a gaps-and-islands computation -- because a share can hide whether the
--      months were consecutive.
--   8. ZONES: a zone's price is the unweighted mean of its member states. The
--      between-zone share of cross-state variance is reported every month: it
--      says how much of the dispersion among states the six-zone geography
--      explains at all.
--   9. DESCRIPTION, NOT EXPLANATION. Distance from import terminals and
--      refineries, transport cost, security and local market structure are all
--      plausible reasons a state is dear or cheap. None is tested here, and the
--      paper must not present a premium as caused by any of them.
-- ===========================================================================

-- @setup
CREATE OR REPLACE TEMP VIEW a12_panel AS
SELECT
    f.month_key,
    printf('%04d-%02d', f.month_key // 10000, (f.month_key // 100) % 100) AS year_month,
    (f.month_key // 10000) * 12 + ((f.month_key // 100) % 100)           AS month_index,
    g.geo_name                                                            AS state,
    g.region_group                                                        AS zone,
    f.price_ngn
FROM fact_fuel_price_monthly f
JOIN dim_geography g
  ON g.geo_key     = f.geo_key
 AND g.admin_level = 'state'
 AND g.country     = 'Nigeria'
WHERE f.fuel_type = 'PMS';

-- @setup
CREATE OR REPLACE TEMP VIEW a12_state_month AS
WITH ranked AS (
    SELECT
        p.*,
        avg(p.price_ngn) OVER m                                   AS national_mean,
        count(*)         OVER m                                   AS states_reporting,
        avg(p.price_ngn) OVER (PARTITION BY p.month_key, p.zone)  AS zone_mean,
        -- Average rank, 1 = most expensive: ties share the mean of their ranks.
        rank() OVER (PARTITION BY p.month_key ORDER BY p.price_ngn DESC)
          + (count(*) OVER (PARTITION BY p.month_key, p.price_ngn) - 1) / 2.0
                                                                  AS price_rank
    FROM a12_panel p
    WINDOW m AS (PARTITION BY p.month_key)
)
SELECT
    ranked.*,
    100.0 * (price_ngn / national_mean - 1.0)                     AS premium_pct,
    100.0 * (price_ngn / zone_mean - 1.0)                         AS premium_vs_zone_pct,
    (price_ngn > national_mean)                                   AS above_national_mean,
    lag(price_ngn)  OVER s                                        AS price_prev_month,
    lag(price_rank) OVER s                                        AS rank_prev_month,
    first_value(price_rank) OVER s                                AS rank_first_month
FROM ranked
WINDOW s AS (PARTITION BY state ORDER BY month_index);

-- @output: main
SELECT
    year_month,
    state,
    zone,
    round(price_ngn, 2)                                           AS price_ngn,
    round(national_mean, 2)                                       AS national_mean_ngn,
    round(premium_pct, 4)                                         AS premium_vs_national_pct,
    round(zone_mean, 2)                                           AS zone_mean_ngn,
    round(premium_vs_zone_pct, 4)                                 AS premium_vs_zone_pct,
    price_rank                                                    AS rank_most_expensive_first,
    rank_prev_month                                               AS rank_previous_month,
    above_national_mean,
    round(100.0 * (price_ngn / price_prev_month - 1.0), 4)        AS state_mom_pct,
    states_reporting
FROM a12_state_month
ORDER BY year_month, price_rank, state;

-- @output: national
WITH national AS (
    SELECT
        month_key,
        year_month,
        month_index,
        count(*)                                  AS states_reporting,
        avg(price_ngn)                            AS mean_price,
        median(price_ngn)                         AS median_price,
        min(price_ngn)                            AS min_price,
        max(price_ngn)                            AS max_price,
        arg_min(state, price_ngn)                 AS cheapest_state,
        arg_max(state, price_ngn)                 AS dearest_state,
        stddev_pop(price_ngn)                     AS sd_price,
        quantile_cont(price_ngn, [0.25, 0.75])    AS quartiles
    FROM a12_panel
    GROUP BY ALL
),
spearman AS (
    SELECT
        month_key,
        corr(price_rank, rank_prev_month)         AS rho_previous,
        corr(price_rank, rank_first_month)        AS rho_first
    FROM a12_state_month
    GROUP BY month_key
),
variance AS (
    -- Summing (zone mean - national mean)^2 over STATE rows yields
    -- sum over zones of n_zone * (zone mean - national mean)^2: the between-
    -- zone sum of squares, with each zone weighted by its number of states.
    SELECT
        month_key,
        sum(power(price_ngn - national_mean, 2))  AS ss_total,
        sum(power(zone_mean - national_mean, 2))  AS ss_between_zones
    FROM a12_state_month
    GROUP BY month_key
)
SELECT
    n.year_month,
    n.states_reporting,
    round(n.mean_price, 2)                                            AS national_mean_ngn,
    round(n.median_price, 2)                                          AS national_median_ngn,
    round(100.0 * n.mean_price
          / first_value(n.mean_price) OVER (ORDER BY n.month_index), 4)
                                                                      AS index_first_month_100,
    round(100.0 * (n.mean_price
          / lag(n.mean_price) OVER (ORDER BY n.month_index) - 1.0), 4)
                                                                      AS mom_pct,
    round(100.0 * (n.mean_price
          / lag(n.mean_price, 12) OVER (ORDER BY n.month_index) - 1.0), 4)
                                                                      AS yoy_pct,
    round(avg(n.mean_price) OVER (ORDER BY n.month_index
          ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2)               AS rolling_3m_mean_ngn,
    round(n.min_price, 2)                                             AS min_state_price_ngn,
    n.cheapest_state,
    round(n.max_price, 2)                                             AS max_state_price_ngn,
    n.dearest_state,
    round(n.max_price - n.min_price, 2)                               AS range_ngn,
    round(n.max_price / n.min_price, 4)                               AS max_min_ratio,
    round(n.quartiles[1], 2)                                          AS p25_ngn,
    round(n.quartiles[2], 2)                                          AS p75_ngn,
    round(n.quartiles[2] - n.quartiles[1], 2)                         AS iqr_ngn,
    round(n.sd_price, 4)                                              AS sd_ngn_population,
    round(100.0 * n.sd_price / n.mean_price, 4)                       AS cv_pct,
    round(100.0 * v.ss_between_zones / nullif(v.ss_total, 0), 4)
                                                                      AS zone_share_of_variance_pct,
    round(s.rho_previous, 6)                                          AS spearman_vs_previous_month,
    round(s.rho_first, 6)                                             AS spearman_vs_first_month,
    (n.mean_price = max(n.mean_price) OVER ())                        AS is_peak_month,
    (n.mean_price = min(n.mean_price) OVER ())                        AS is_trough_month
FROM national n
JOIN spearman s ON s.month_key = n.month_key
JOIN variance v ON v.month_key = n.month_key
ORDER BY n.month_index;

-- @output: states
WITH islands AS (
    -- Gaps and islands: consecutive months on the same side of the mean share
    -- one value of (month_index - row_number within that side).
    SELECT
        state,
        above_national_mean,
        month_index - row_number() OVER (
            PARTITION BY state, above_national_mean ORDER BY month_index
        )                                                       AS island
    FROM a12_state_month
),
runs AS (
    SELECT
        state,
        max(n) FILTER (WHERE above_national_mean)               AS longest_run_above,
        max(n) FILTER (WHERE NOT above_national_mean)           AS longest_run_below
    FROM (
        SELECT state, above_national_mean, island, count(*) AS n
        FROM islands
        GROUP BY ALL
    )
    GROUP BY state
),
summary AS (
    SELECT
        state,
        any_value(zone)                                         AS zone,
        count(*)                                                AS months,
        avg(price_ngn)                                          AS mean_price,
        avg(premium_pct)                                        AS mean_premium,
        median(premium_pct)                                     AS median_premium,
        stddev_samp(premium_pct)                                AS sd_premium,
        min(premium_pct)                                        AS min_premium,
        max(premium_pct)                                        AS max_premium,
        count(*) FILTER (WHERE above_national_mean)             AS months_above,
        avg(price_rank)                                         AS mean_rank,
        stddev_pop(price_rank)                                  AS sd_rank,
        min(price_rank)                                         AS highest_rank,
        max(price_rank)                                         AS lowest_rank
    FROM a12_state_month
    GROUP BY state
)
SELECT
    rank() OVER (ORDER BY s.mean_premium DESC)                  AS premium_rank,
    s.state,
    s.zone,
    s.months,
    round(s.mean_price, 2)                                      AS mean_price_ngn,
    round(s.mean_premium, 4)                                    AS mean_premium_vs_national_pct,
    round(s.median_premium, 4)                                  AS median_premium_pct,
    round(s.sd_premium, 4)                                      AS sd_premium_pp,
    round(s.min_premium, 4)                                     AS min_premium_pct,
    round(s.max_premium, 4)                                     AS max_premium_pct,
    s.months_above                                              AS months_above_national_mean,
    s.months - s.months_above                                   AS months_at_or_below_mean,
    round(s.months_above / s.months, 4)                         AS share_months_above,
    coalesce(r.longest_run_above, 0)                            AS longest_run_above_months,
    coalesce(r.longest_run_below, 0)                            AS longest_run_below_months,
    round(s.mean_rank, 3)                                       AS mean_rank_most_expensive_first,
    round(s.sd_rank, 3)                                         AS sd_rank,
    s.highest_rank                                              AS highest_rank_reached,
    s.lowest_rank                                               AS lowest_rank_reached,
    CASE
        WHEN s.months_above / s.months >= {persistence_share}
            THEN 'persistent premium'
        WHEN (s.months - s.months_above) / s.months >= {persistence_share}
            THEN 'persistent discount'
        ELSE 'no persistent position'
    END                                                         AS persistence_class,
    {persistence_share}                                         AS persistence_threshold_share
FROM summary s
JOIN runs r ON r.state = s.state
ORDER BY premium_rank;

-- @output: zones
WITH zone_month AS (
    SELECT
        month_key,
        year_month,
        month_index,
        zone,
        count(*)                                                AS states_in_zone,
        avg(price_ngn)                                          AS zone_mean,
        stddev_pop(price_ngn)                                   AS zone_sd,
        min(price_ngn)                                          AS zone_min,
        max(price_ngn)                                          AS zone_max,
        any_value(national_mean)                                AS national_mean
    FROM a12_state_month
    GROUP BY ALL
)
SELECT
    year_month,
    zone,
    states_in_zone,
    round(zone_mean, 2)                                         AS zone_mean_ngn,
    round(national_mean, 2)                                     AS national_mean_ngn,
    round(100.0 * (zone_mean / national_mean - 1.0), 4)         AS zone_premium_vs_national_pct,
    round(100.0 * zone_sd / zone_mean, 4)                       AS within_zone_cv_pct,
    round(zone_min, 2)                                          AS zone_min_state_price_ngn,
    round(zone_max, 2)                                          AS zone_max_state_price_ngn,
    rank() OVER (PARTITION BY month_key ORDER BY zone_mean DESC)
                                                                AS zone_rank_most_expensive_first,
    round(avg(100.0 * (zone_mean / national_mean - 1.0))
          OVER (PARTITION BY zone), 4)                          AS zone_mean_premium_over_period_pct
FROM zone_month
ORDER BY month_index, zone_rank_most_expensive_first;
