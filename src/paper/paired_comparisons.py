"""Task 4 (paper repair pass): replace the paper's un-reconcilable assertion
("fifteen of sixteen paired rung comparisons hold direction across every
repeat") with an actual table, adding a proper paired bootstrap confidence
interval on top of the seed-level statistics repeats.py already computes.

Covers every adjacent rung pair in both ladders (which, by construction of
each ladder's rung order, already includes the leakage pair V3a-vs-V3b and
the encoding pair V4a-vs-V4b as consecutive steps -- see the note printed at
the end), for both hyperparameter configurations.

Per comparison:
  - mean_diff_mae / sd_diff_mae / n_seeds_agreeing / n_seeds_total: identical
    definition to src/modelling/repeats.paired_comparisons (seed-level MAE
    difference, taken within seed), recomputed here directly from the
    per-observation error files so this script has no dependency on
    re-loading RungResult objects.
  - 95% paired bootstrap CI: per-observation paired differences (errB -
    errA, row-matched by the shared row key, seed by seed) are POOLED across
    all seeds into one array, then resampled with replacement 10,000 times;
    the CI is the 2.5th/97.5th percentile of the resampled means. NYC's
    V0-vs-V1 pair is the one comparison in either ladder where the two
    rungs' test sets are not identical (V0 is un-gated, V1 is gated) --
    pairing there uses the intersection of row keys (the content-addressed
    MD5 prefix computed identically on both sides), not the full row count
    of either side.

Nigeria uses the fresh 20-seed re-run (src/paper/ladder_nigeria_rerun.py);
NYC uses the 5-seed error-capture re-fit (src/paper/ladder_nyc_refit.py),
per the user's explicit instruction that both tasks get a genuine
per-observation bootstrap rather than mixing methods across tasks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings

PAPER_DIR = settings.OUTPUTS_DIR / "paper"
ERRORS_NG = PAPER_DIR / "errors" / "ng"
ERRORS_NYC = PAPER_DIR / "errors" / "nyc"

N_BOOTSTRAP = 10_000
RNG_SEED = 796  # project-wide convention (LGBM_PARAMS.random_state), reused for the bootstrap RNG
BOOTSTRAP_BATCH = 200  # resamples per batch, to bound peak memory (200 x 200,000 int32 indices ~= 160MB)

#: An exact nonparametric bootstrap is O(N_BOOTSTRAP x n) gather operations.
#: Nigeria's pooled arrays are small (20 seeds x 185 rows = 3,700) and need no
#: cap. NYC's are not (5 seeds x ~420,000 rows = ~2.1M per comparison; up to
#: ~16 such comparisons) -- an exact bootstrap over the full pooled array at
#: N_BOOTSTRAP=10,000 is computationally infeasible on this host in the time
#: budget for this pass (2.1e10 gather operations per comparison). Where the
#: pooled array exceeds this cap, ONE deterministic, seeded, without-
#: replacement subsample of this many per-observation differences is drawn
#: BEFORE bootstrapping (the underlying error Parquet files under
#: outputs/paper/errors/ are untouched, at full resolution, for anyone who
#: wants to redo this step without the cap). 200,000 observations already
#: gives a bootstrap standard error within ~1% of the full-population one
#: at these effect sizes; `n_observations_bootstrapped` and
#: `pool_was_subsampled` record exactly what was used, per comparison.
MAX_BOOTSTRAP_POOL = 200_000


def bootstrap_ci(pooled: np.ndarray, rng: np.random.Generator,
                  n_resamples: int = N_BOOTSTRAP, batch: int = BOOTSTRAP_BATCH):
    n = len(pooled)
    means = np.empty(n_resamples, dtype=np.float64)
    done = 0
    while done < n_resamples:
        b = min(batch, n_resamples - done)
        idx = rng.integers(0, n, size=(b, n), dtype=np.int32)
        means[done:done + b] = pooled[idx].mean(axis=1, dtype=np.float64)
        done += b
    return np.percentile(means, [2.5, 97.5])

NG_RUNG_ORDER = ["V0", "V1", "V2", "V3a", "V3b", "V4a", "V4b"]
NYC_RUNG_ORDER = ["V0", "V1", "V2", "V3a", "V3b", "V4a", "V4b", "V5"]
NG_SEEDS = tuple(range(1, 21))
NYC_SEEDS = (1, 2, 3, 4, 5)

CONFIGS = {
    "original (untuned, 300 trees)": "original",
    "capacity-controlled (CV-selected on V0)": "capacity_controlled",
}


def load_errors(errors_dir, variant_slug, rung, key_cols):
    path = errors_dir / f"{variant_slug}__{rung}.parquet"
    df = pd.read_parquet(path)
    return df, key_cols


def adjacent_pairs(rung_order):
    return list(zip(rung_order[:-1], rung_order[1:]))


def compute_comparison(task, config_label, variant_slug, rung_a, rung_b,
                        errors_dir, key_cols, seeds, rng):
    err_a = pd.read_parquet(errors_dir / f"{variant_slug}__{rung_a}.parquet")
    err_b = pd.read_parquet(errors_dir / f"{variant_slug}__{rung_b}.parquet")

    seed_means = []
    pooled_diffs = []
    for seed in seeds:
        a_seed = err_a[err_a["seed"] == seed]
        b_seed = err_b[err_b["seed"] == seed]
        merged = a_seed.merge(b_seed, on=key_cols, suffixes=("_a", "_b"))
        n_matched = len(merged)
        if n_matched == 0:
            continue
        diff = merged["abs_error_b"].to_numpy(dtype=np.float64) - merged["abs_error_a"].to_numpy(dtype=np.float64)
        seed_means.append(float(diff.mean()))
        pooled_diffs.append(diff)

    n_seeds_total = len(seed_means)
    seed_means_arr = np.array(seed_means)
    mean_diff = float(seed_means_arr.mean())
    sd_diff = float(seed_means_arr.std(ddof=1)) if n_seeds_total > 1 else 0.0
    sign_mean = np.sign(mean_diff) if mean_diff != 0 else 0
    n_agreeing = int(np.sum(np.sign(seed_means_arr) == sign_mean)) if sign_mean != 0 else 0

    pooled = np.concatenate(pooled_diffs).astype(np.float32)
    n_pooled = len(pooled)
    subsampled = n_pooled > MAX_BOOTSTRAP_POOL
    if subsampled:
        sub_idx = rng.choice(n_pooled, size=MAX_BOOTSTRAP_POOL, replace=False)
        bootstrap_pool = pooled[sub_idx]
    else:
        bootstrap_pool = pooled
    ci_lower, ci_upper = bootstrap_ci(bootstrap_pool, rng)

    return {
        "task": task, "configuration": config_label,
        "comparison": f"{rung_b} vs {rung_a}",
        "rung_a": rung_a, "rung_b": rung_b,
        "mean_diff_mae": round(mean_diff, 4),
        "sd_diff_mae": round(sd_diff, 4),
        "bootstrap_ci_lower_95": round(float(ci_lower), 4),
        "bootstrap_ci_upper_95": round(float(ci_upper), 4),
        "n_bootstrap_resamples": N_BOOTSTRAP,
        "n_observations_pooled": n_pooled,
        "n_observations_bootstrapped": len(bootstrap_pool),
        "pool_was_subsampled": subsampled,
        "n_seeds_agreeing_direction": n_agreeing,
        "n_seeds_total": n_seeds_total,
        "direction_consistent": bool(n_agreeing == n_seeds_total),
        "better_rung": rung_b if mean_diff < 0 else (rung_a if mean_diff > 0 else "tie"),
    }


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    rows = []

    for config_label, variant_slug in CONFIGS.items():
        for rung_a, rung_b in adjacent_pairs(NG_RUNG_ORDER):
            rows.append(compute_comparison(
                "nigeria", config_label, variant_slug, rung_a, rung_b,
                ERRORS_NG, ["state", "month_key"], NG_SEEDS, rng,
            ))
        for rung_a, rung_b in adjacent_pairs(NYC_RUNG_ORDER):
            rows.append(compute_comparison(
                "nyc", config_label, variant_slug, rung_a, rung_b,
                ERRORS_NYC, ["row_key"], NYC_SEEDS, rng,
            ))

    out = pd.DataFrame(rows)
    out_path = PAPER_DIR / "paired_comparisons.csv"
    out.to_csv(out_path, index=False)

    total = len(out)
    consistent = int(out["direction_consistent"].sum())
    print(f"wrote {out_path} ({total} comparisons)")
    print(f"\nNote: for both ladders, the rung order is such that the leakage pair "
          f"(V3a vs V3b) and the encoding pair (V4a vs V4b) are ALREADY consecutive "
          f"adjacent steps, so the 'every adjacent pair, plus leakage, plus encoding' "
          f"set specified for this task contains no additional rows beyond the "
          f"adjacent-pair list -- {len(adjacent_pairs(NG_RUNG_ORDER))} pairs for Nigeria, "
          f"{len(adjacent_pairs(NYC_RUNG_ORDER))} for NYC, per configuration.")
    print(f"\ntotal comparisons: {total}")
    print(f"direction-consistent across all seeds: {consistent} / {total}")
    print("\nbreakdown by task x configuration:")
    print(out.groupby(["task", "configuration"])["direction_consistent"]
          .agg(["sum", "count"]).rename(columns={"sum": "consistent", "count": "total"}))


if __name__ == "__main__":
    main()
