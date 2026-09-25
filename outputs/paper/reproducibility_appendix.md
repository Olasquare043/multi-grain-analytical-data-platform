# Reproducibility appendix

Factual record of two reproducibility defects found and fixed in the NYC
trip-duration modelling layer (`notebooks/02_nyc_trip_duration.ipynb`,
`src/modelling/splits.py`). Full account: `docs/methodology_notes.md` §12.
Task A (Nigeria) never exhibited either defect (its query already carried an
explicit `ORDER BY`).

## Defect 1: non-deterministic DuckDB parallel scan ordering affecting row selection

**Observed.** Task B samples ~2,000,000 of 40.4M rows per model-training
repeat. The sample was originally drawn with DuckDB's reservoir sampler,
`USING SAMPLE 2000000 ROWS (reservoir, 796)`, applied to a
`read_parquet('data/raw/nyc/*.parquet')` glob. Notebook 02 was re-executed
after an unrelated change and rung V0 reported MAE 332.88s where the prior
execution of the identical code, same seed, same container image, had
reported 335.67s.

**Diagnosis.** The seed fixes only the reservoir sampler's own internal
randomness, not the order in which rows arrive at it from the glob scan, and
reservoir sampling's output is a function of arrival order as well as the
seed. DuckDB's parallel scan order across files and row groups is not
contractually stable between runs. Checking realised row counts and column
checksums between the two runs confirmed the two executions had sampled
different physical rows.

**Fix.** Row selection was changed from position-based reservoir sampling to
content-addressed bucketing: `('0x' || substr(trip_id, 1, 8))::UBIGINT % 17
= <bucket>`. `trip_id` is a deterministic MD5 over each trip's own business
attributes (`sql/ddl/silver_nyc_trip.sql`), so a trip's bucket membership is
a property of the row's content, independent of scan order, thread count or
file layout. The DuckDB `hash()` function was deliberately not used, since
it depends on a library-internal hash a DuckDB upgrade could legitimately
change; the leading hex digits of the stored MD5 string are parsed directly
instead. The raw, un-gated V0 draw recomputes the identical MD5 inline
(`src/modelling/splits.RAW_TRIP_ID_SQL`) from the same raw columns, since the
quality gate is what assigns `trip_id` and V0 reads the source before the
gate — both sides bucket by the same rule. 17 buckets was chosen so a
bucket's realised size (~1,956,000 train / ~421,000 test rows) lands near
the sizes the superseded reservoir draw had targeted (2,000,000 / 400,000);
the notebook reports realised counts rather than assuming them.

**Verification.** `tests/test_sampling_determinism.py` asserts that
identical rows presented in a deliberately reordered physical layout select
an identical subset, and that the real predicate against the real
`fact_trip` returns an identical row count and column checksum when executed
in two separate OS processes (so a cached result cannot satisfy the
assertion).

**Guard test.** `tests/test_sampling_determinism.py`.

## Defect 2: row-order sensitivity in the gradient-boosting fit, despite deterministic row selection

**Observed.** Content-addressed bucketing (Defect 1's fix) made the
*selected set* of rows reproducible, verified by re-running notebook 02 end
to end and diffing its outputs against the previous run. The notebook still
did not reproduce: one rung-to-rung comparison, reported as direction-
consistent across all five repeats in one run, came back consistent in only
four of five in the next, despite `REF_mean`, the row count, the target
column's checksum and the sum of trip distances all being bit-identical
between runs.

**Diagnosis.** A deterministic row *set* is not a deterministic row *order*.
DuckDB's parallel scan returns rows in no guaranteed order, and gradient
boosting is order-sensitive: bagging (`subsample=0.8`) selects rows by
position, and histogram split-point ties break by arrival order. Sums and
checksums are order-independent, which is why they agreed while the fitted
models did not. Measured directly at full scale, two processes reading the
identical bucket without an `ORDER BY` produced prediction sums of
202,240,914 and 203,933,890 from otherwise-identical fits; with an explicit
`ORDER BY trip_id`, both processes produced 208,095,622.694010 — a third,
different value, confirming this was not a matter of picking which
unordered run to believe. LightGBM's own `deterministic=True` /
`force_row_wise=True` was tried first and did not fix it, consistent with
the cause being upstream of the learner.

**Fix.** An explicit `ORDER BY trip_id` (gated source) or the identical
inline MD5 expression (raw source) was added to every row-selecting pull in
notebook 02.

**Verification.** Beyond the unit test below, notebook 02 was executed four
times end to end and its published tables diffed against each other each
time: byte-identical across all four runs.

**Guard test.** `tests/test_sampling_determinism.py`, which also asserts
that a booster fit on an ordered pull produces identical predictions across
two separate OS processes — the regression guard specifically for this
second defect, in the same file as Defect 1's guard since both concern the
same data-loading path.
