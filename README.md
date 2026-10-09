# CSC 796 Mobility Platform

**Engineering a Multi-Grain Analytical Data Platform for the Economics of
Movement: A Comparative Data Engineering Case Study of New York City Trip Records
and Nigerian Price Statistics**

This repository is the practical component of a term paper for CSC 796: Advanced
Data Engineering (M.Sc. Artificial Intelligence, University of Ibadan). It tests
one argument: **a single conformed dimensional model can serve fact tables whose
grains differ by several orders of magnitude, and the engineering cost of doing
so is measurable rather than theoretical.** The fact tables range from one row
per taxi trip (40.4 million rows) to one row per Nigerian state per month (1,147
rows).

Every number the paper cites comes from real public data, through code an
examiner can rerun. Nothing is synthetic, estimated or interpolated. Where the
data could not support a result, the pipeline reports that.

---

## Run it

The only prerequisite is **Docker Desktop**.

```bash
docker compose up --build
```

That one command downloads all five sources, builds the medallion layers and the
star schema, runs 56 quality rules, executes analyses A1 to A11 and benchmarks
B1 to B4, draws the figures, writes the documentation, and prints a completion
summary listing every artefact with its size. Outputs land in bind-mounted host
folders (`data/`, `outputs/`, `docs/`, `logs/`), so they outlive the container.

| | |
| --- | --- |
| Expected runtime | about 15-20 minutes warm, plus about 7 minutes of downloads on a cold run |
| Disk | about 2.2 GB at rest (budget: 3 GB); benchmark scratch files are deleted after use |
| Resources declared | 4 CPUs, 8 GB RAM (see `docker-compose.yml`) |

On Docker Desktop the Linux VM may have less memory than the declared 8 GB
ceiling. On the development machine it had 7.7 GiB, so the effective ceiling
there was the VM's, not the compose file's. The benchmark report records the
envelope it actually ran in.

### Individual stages

With GNU Make on the host, each target runs inside the container:

```bash
make build           # build the image
make run             # full pipeline
make test            # test suite (106 tests)
make extract-nbs     # any single stage: extract-nyc, extract-hdx, extract-nbs,
make quality         #   extract-weather, bronze, silver, dims, facts, quality,
make analysis        #   analysis, benchmarks, figures, docs, verify
make clean           # remove derived data, keep downloads
make clean-all       # remove everything, forcing a cold run
```

Without Make, call the entrypoint directly:

```bash
docker compose run --rm pipeline python -m src.pipeline --stage quality
docker compose run --rm pipeline python -m pytest tests/ -v
```

Every stage can be rerun on its own, and rerunning never duplicates rows.

---

## What it produces

| Location | Contents |
| --- | --- |
| `outputs/tables/` | `A1_demand_profile.csv` ... `A11_missingness_structure.csv`: one result table per analysis |
| `outputs/figures/` | 12 figures at 300 dpi, colour-blind-safe, with units and source on every chart |
| `outputs/quality/` | `quality_report.{csv,md}`, `reconciliation.csv` (rows in / rejected by reason / loaded, per source), `nbs_reconciliation.csv` |
| `outputs/benchmarks/` | `benchmark_results.{csv,md}` |
| `docs/` | data dictionary (generated from the live schema), ERDs, architecture, SCD strategy, methodology notes, `run_manifest.json` |
| `logs/pipeline.log` | Every stage boundary and timing |

`docs/run_manifest.json` is rewritten on every run. It records the input files
with their row counts and SHA-256 checksums, the rows loaded per table, quality
outcomes, stage timings and library versions. It is the provenance record for
every number in the paper.

---

## Sources and extraction patterns

Four structurally distinct extraction patterns are used, kept parallel so the
ETL chapter can compare them:

| | Source | Pattern | Rows |
| --- | --- | --- | --- |
| A | NYC TLC Yellow Taxi trips, 2024 (12 Parquet files) | bulk binary download | 41,169,720 |
| B | NYC taxi zone lookup | bulk binary download | 265 |
| C | WFP Nigeria market food prices, via HDX | open data API | 87,384 |
| D | NBS Premium Motor Spirit Price Watch (19 workbooks) | HTML scrape + heterogeneous Excel parse | 1,147 |
| E | Open-Meteo NYC daily weather *(optional)* | REST/JSON | 366 |

**The correctness proof.** `tests/test_pms_reconciliation.py` aggregates the
assembled NBS panel to national means and compares them with figures the Bureau
publishes. All three reference months match exactly (a difference of 0.00 NGN
against a 0.05 tolerance). This check tests the whole chain, from spreadsheet
cell to star schema, against numbers nobody on this project chose.

---

## Findings the build surfaced

These came from building and checking the pipeline, not from the brief. Each one
is documented in [`docs/methodology_notes.md`](docs/methodology_notes.md).

- **Three "missing" NBS months were never missing.** They were an artefact of a
  six-row header scan. Some releases put the `State` header on row 14.
- **A silent corruption in the NBS workbooks.** Commentary tables below the state
  table were being read as data, and put January 2026 prices into the January
  2025 column. The parser now stops at the first repeated state.
- **WFP's coverage collapses in January 2023,** from 14 states to 3, all in the
  North East. A9 (fuel-to-food passthrough) is therefore kept only as an
  exploratory null result about conflict-affected north-eastern markets in
  Borno, Yobe and Adamawa: its correlations are under ±0.09 at every lag and
  support no inference. A7 is restricted to a fixed within-market panel of 10
  markets in 2 states and is not a national before-and-after.
- **A12, the centrepiece of the Nigerian analysis,** uses the complete,
  reconciled NBS petrol panel (37 states × 31 months). The national mean rose
  from ₦648.93 to ₦1,596.25 (+146%). Cross-state dispersion is episodic,
  peaking at a CV of 17.7% in February 2025. The six zones explain only about a
  third of it, and just 5 of 37 states hold a persistent premium or discount.
- **Four TLC fields fail together.** Of 16 possible missingness patterns, exactly
  two occur: all four fields present, or all four absent (9.8% of trips). This
  was added as analysis A11.
- **Partition pruning lost to a well-ordered single file** at this scale
  (benchmark B2), because Parquet zone maps did the pruning work.
- **In the cloud, missing partitions cost money.** The public BigQuery TLC tables
  are unpartitioned, so a one-month query billed the same 645 MB as a whole year.
  BigQuery still counted the full 1.56-billion-trip yellow archive (2011–2022) in
  2.6 seconds for 11.7 GB. This comparison is illustrative only: the cloud data
  is 2019, not 2024 (see `docs/cloud_architecture.md`).

---

## Repository layout

```text
config/          settings.py (every tunable), nigeria_states.py, quality_rules.yaml
src/extract/     one module per source, sharing one result contract
src/transform/   bronze.py, silver.py
src/model/       dimensions, Type 2 dim_geography, facts, catalogue
src/quality/     declarative rule engine and reports
src/analysis/    A1-A11 runner
src/benchmark/   subprocess-isolated timing and peak-memory harness, B1-B4
src/viz/         shared figure style and figures
src/docs/        generated data dictionary
src/cloud/       OPTIONAL BigQuery module, isolated from the core run
sql/ddl/         one file per table, executed as written
sql/analysis/    A1-A11, each opening with question, grain and assumptions
sql/cloud/       BigQuery variants
tests/           reconciliation, SCD2, state normaliser, quality engine, dim_date, smoke
```

---

## Dependencies

Every version is pinned in `requirements.txt`. The build brief permitted
`duckdb`, `pandas`, `pyarrow`, `requests`, `matplotlib`, `openpyxl`,
`beautifulsoup4`, `lxml` and `python-dateutil`. Two direct additions are made,
and both are documented in the file itself:

- **PyYAML**, because the brief requires a declarative `config/quality_rules.yaml`
  but permits no YAML parser.
- **pytest**, because the brief requires a test suite but permits no runner.

There is no Java, no Spark, no database server and no paid service. The
warehouse is DuckDB running in-process over Parquet.

---

## Optional cloud module (Google BigQuery sandbox)

The cloud module lives in `src/cloud/` and has its own `requirements-cloud.txt`.
It runs **only** via `make cloud`, and the core pipeline succeeds whether or not
it has ever been run.

1. Create a BigQuery sandbox project. A Google account is enough; no billing
   account or card is needed.
2. Create a service account with the *BigQuery User* and *BigQuery Data Viewer*
   roles, and download its JSON key into `secrets/`.
3. Set both variables in `.env`:

   ```bash
   GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/<your-key>.json
   GCP_PROJECT_ID=<your-project-id>
   ```

4. Run `make cloud`, or `docker compose run --rm pipeline python -m src.cloud.run_cloud`.

`secrets/` and `.env` are excluded from git and from the Docker build context.
Credentials are mounted read-only at runtime and are never printed.

**One confounder is stated up front.** The public BigQuery TLC dataset stops at
2022 and contains no 2024 rows. The cloud comparison therefore runs against
`tlc_yellow_trips_2019`, and every cloud result is labelled with the exact table
and date range it used. The comparison illustrates architectural trade-offs. It
is not a controlled benchmark of one engine against another.
