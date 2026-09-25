"""Categorical encoders for the V4 ladder rung (one-hot vs. target encoding).

One-hot encoding went through three implementations before landing on the
one below, and the failures are recorded here rather than erased, in the
same spirit as every other engineering correction in this project (see
``docs/methodology_notes.md``):

1. ``category_encoders.OneHotEncoder``. Handles train/test category
   alignment for free, but at NYC scale (2.4M rows, 265 categories,
   ``notebooks/02_nyc_trip_duration.ipynb`` V4a) its implementation spiked
   container memory badly enough to kill the kernel outright.
2. A hand-rolled dense encoder on ``pandas.get_dummies(..., dtype="int8")``.
   Avoided (1)'s crash, but a dense one-hot expansion is, by construction,
   >99% zeros (exactly one column is 1 per row) -- LightGBM's own internal
   `Dataset` construction over that many dense columns still pushed the
   container over its memory ceiling during the fit itself.
3. The same encoder with ``pandas.get_dummies(..., sparse=True)``. This
   *looked* like the fix -- a pandas `Sparse[int8]`-dtype DataFrame reports
   a tiny `memory_usage(deep=True)` -- but LightGBM's sklearn wrapper
   silently densifies a sparse-dtype DataFrame before it ever reaches the
   C++ booster, so the crash was unchanged: the sparsity was real in
   pandas and imaginary by the time LightGBM saw it.

**What actually works, below**: build a genuine ``scipy.sparse.csr_matrix``
directly (one-hot block via `scipy.sparse`, concatenated with the dense
base columns via `hstack`), and pass that matrix to `LGBMRegressor.fit`
instead of a DataFrame. LightGBM's Python API accepts `scipy.sparse`
matrices as a first-class input and keeps them sparse internally --
verified empirically at the full 2.4M-row scale to hold peak memory under
1GB, against >7GB for the two DataFrame-based attempts. This is also why
``src/modelling/ladder.run_rung`` reads row counts via ``.shape[0]``
rather than ``len(...)``: a `scipy.sparse` matrix does not support `len()`.
``category_encoders`` remains a permitted, pinned dependency
(``requirements.txt``) but is no longer imported here.

Target encoding has no safe off-the-shelf implementation available here:
every target-encoding library on PyPI, including category_encoders' own
``TargetEncoder``, fits one encoding from the whole column it is given and
applies it everywhere, which is exactly the leakage this modelling layer
exists to demonstrate and avoid if the fitted column includes test-period
rows. ``target_encode_pit`` below is hand-rolled for that reason, and reuses
the identical expanding-window discipline as the V3b builders in
``src/modelling/features.py``: every row's encoded value comes only from
rows strictly earlier than it in ``time_col``.

This rung exists specifically to engage Ayinla (2023) on index-mapped
ordinal encoding of categorical attributes; see the discussion in the
notebook markdown at the point this module is used, and in
``docs/modelling_notes.md``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def one_hot_encode(train: pd.DataFrame, test: pd.DataFrame, col: str
                    ) -> tuple["sparse.csr_matrix", "sparse.csr_matrix"]:
    """One-hot encode `col`; fit on train, categories aligned onto test.

    Returns `(X_train, X_test)` as `scipy.sparse.csr_matrix`: every other
    column of `train`/`test` (in their original order) stays dense, packed
    into the leading columns, followed by one sparse binary column per
    category OBSERVED IN TRAIN. A test-only category contributes an
    implicit all-zero column (it simply never appears in the train-fit
    category index), which is itself point-in-time-correct: the model was
    never shown that category, so it cannot condition on it. A train
    category with zero occurrences in this test slice is still an explicit,
    named column of `X_test`, all-zero, not a missing one -- `X_train` and
    `X_test` always have identical column counts and column meaning.

    See the module docstring for why this is a raw `scipy.sparse` matrix
    and not a DataFrame with sparse-dtype columns.
    """
    categories = pd.Index(sorted(train[col].dropna().unique()))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    n_cats = len(categories)

    def _onehot_block(frame: pd.DataFrame) -> sparse.csr_matrix:
        codes = frame[col].map(cat_to_idx)  # NaN for a category train never saw
        known = codes.notna().to_numpy()
        rows = np.nonzero(known)[0]
        cols = codes.to_numpy()[known].astype(np.int64)
        data = np.ones(len(rows), dtype=np.int8)
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(frame), n_cats))

    # A scipy.sparse matrix is numeric-only, so any pandas `category`/object
    # column among the OTHER features (e.g. V3b's `borough`, `service_zone`)
    # is integer-coded here rather than passed through as pandas categorical
    # dtype. LightGBM's own categorical-feature auto-detection (used by
    # every other, DataFrame-based rung) does not apply to a raw sparse
    # matrix; this is a necessary, explicitly-noted side effect of using a
    # sparse matrix for the wide one-hot block, not a change to what
    # information the model receives.
    def _numeric_block(frame: pd.DataFrame) -> np.ndarray:
        columns = []
        for c in other_cols:
            series = frame[c]
            if isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object:
                columns.append(series.cat.codes.to_numpy(dtype=np.float32)
                                if isinstance(series.dtype, pd.CategoricalDtype)
                                else pd.factorize(series)[0].astype(np.float32))
            else:
                columns.append(series.to_numpy(dtype=np.float32))
        return np.column_stack(columns)

    other_cols = [c for c in train.columns if c != col]
    train_dense = sparse.csr_matrix(_numeric_block(train))
    test_dense = sparse.csr_matrix(_numeric_block(test))

    X_train = sparse.hstack([train_dense, _onehot_block(train)], format="csr")
    X_test = sparse.hstack([test_dense, _onehot_block(test)], format="csr")
    return X_train, X_test


def target_encode_pit(
    frame: pd.DataFrame,
    cat_col: str,
    target_col: str,
    time_col: str,
    smoothing: float = 10.0,
    out_col: str | None = None,
) -> pd.DataFrame:
    """Point-in-time-correct smoothed target encoding.

    For every row, `cat_col`'s encoded value is a smoothed historical mean of
    `target_col`, using only rows whose `time_col` is STRICTLY EARLIER than
    that row's own. Smoothing blends the category's own prior mean with the
    contemporaneous population-wide prior mean, weighted by how much prior
    history the category has:

        encoded = (n * cat_prior_mean + smoothing * global_prior_mean)
                  / (n + smoothing)

    where `n` is the count of PRIOR rows in that category. A category with no
    prior observations (n=0) reduces exactly to the global prior mean; the
    handful of very first rows in the whole frame, which have no prior
    observations of ANY kind, fall back to the frame's unconditional target
    mean -- an irreducible edge case for the first instant of a series that
    is noted here rather than hidden.

    `frame` must be the full population (train and test rows together,
    matching features.py's convention): a test row's encoding is allowed to
    see the entire training period, since all of it is strictly earlier than
    any test timestamp, but never a future one.

    Implementation note: this works at (cat_col, time_col) BUCKET
    granularity, not row position, deliberately. A row-position-based
    "expanding" computation (sort by time, cumulative-sum over prior *rows*)
    is only correct when every timestamp is unique. Both tasks here violate
    that constantly -- 37 Nigerian states share every month_key, and
    thousands of NYC trips share every (zone, month) -- so two rows with the
    IDENTICAL timestamp would arbitrarily leak into each other's "prior"
    window depending on incidental sort order. Aggregating to buckets first
    and subtracting each bucket's own contribution back out of its
    cumulative total (the same pattern as add_pit_zone_hour_avg_nyc in
    features.py) makes "strictly earlier" a property of `time_col`'s value,
    not of row order.
    """
    out_col = out_col or f"{cat_col}_target_enc"
    frame = frame.copy()

    # Per-(category, time bucket) prior sum/count, excluding the bucket's own
    # contribution.
    cat_time = (
        frame.groupby([cat_col, time_col])[target_col]
        .agg(bucket_sum="sum", bucket_count="count")
        .reset_index()
        .sort_values([cat_col, time_col])
    )
    cg = cat_time.groupby(cat_col)
    cat_time["cat_prior_sum"] = cg["bucket_sum"].cumsum() - cat_time["bucket_sum"]
    cat_time["cat_prior_count"] = cg["bucket_count"].cumsum() - cat_time["bucket_count"]

    # Per-time-bucket prior sum/count across ALL categories: the
    # contemporaneous fallback for a category with no prior history of its
    # own.
    time_stats = (
        frame.groupby(time_col)[target_col]
        .agg(time_sum="sum", time_count="count")
        .reset_index()
        .sort_values(time_col)
    )
    time_stats["global_prior_sum"] = time_stats["time_sum"].cumsum() - time_stats["time_sum"]
    time_stats["global_prior_count"] = time_stats["time_count"].cumsum() - time_stats["time_count"]
    time_stats["global_prior_mean"] = (
        time_stats["global_prior_sum"] / time_stats["global_prior_count"].replace(0, np.nan)
    )

    cat_time = cat_time.merge(time_stats[[time_col, "global_prior_mean"]],
                               on=time_col, how="left")
    cat_prior_mean = cat_time["cat_prior_sum"] / cat_time["cat_prior_count"].replace(0, np.nan)
    cat_prior_mean_filled = cat_prior_mean.fillna(0.0)
    # The only genuinely irreducible edge case: the very first time bucket in
    # the whole frame has no prior data of ANY kind (no category has reported
    # yet). It falls back to the unconditional target mean, computed once.
    global_prior_mean_filled = cat_time["global_prior_mean"].fillna(frame[target_col].mean())

    numerator = cat_time["cat_prior_count"] * cat_prior_mean_filled + smoothing * global_prior_mean_filled
    denominator = cat_time["cat_prior_count"] + smoothing
    cat_time[out_col] = numerator / denominator

    lookup = cat_time[[cat_col, time_col, out_col]]
    return frame.merge(lookup, on=[cat_col, time_col], how="left")
