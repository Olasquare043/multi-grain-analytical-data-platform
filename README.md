# Multi-Grain Analytical Data Platform

**Conformed dimensional integration at extreme grain disparity: integration cost and downstream model effect across two domains**

This repository is the code and reproducibility package for the paper of the same title. It integrates New York City taxi trip records and Nigerian fuel price statistics in one dimensional warehouse, measures what that integration costs, and then measures what each data engineering step contributes to a fixed prediction model.

The central argument is that a single conformed dimensional model can serve fact tables whose grains differ by four orders of magnitude, and that the cost of doing so is measurable rather than assumed. The fact tables range from one row per taxi trip (40.4 million rows after quality gating) to one row per Nigerian state per month (1,147 rows), a ratio of 35,241 to 1.

Every number in the paper comes from real public data, through code that can be rerun. Nothing is synthetic, estimated or interpolated. Where the data could not support a result, the pipeline reports that.

This repository holds code, configuration, tests and documentation only. It holds no source data (see [Data sources, licences and terms of use](#data-sources-licences-and-terms-of-use)).

---

## Run it

The only prerequisite is **Docker Desktop**.

```
docker compose up --build
```

That one command downloads all five sources, builds the medallion layers and the star schema, runs 56 quality rules, executes the descriptive analyses and benchmarks B1 to B4, draws the figures, writes the documentation, and prints a completion summary listing every artefact with its size. Outputs land in bind-mounted host folders (`data/`, `outputs/`, `docs/`, `logs/`), so they outlive the container.

| | |
|---|---|
| Expected runtime | about 15 to 20 minutes warm, plus about 7 minutes of downloads on a cold run |
| Disk | about 2.2 GB at rest (budget: 3 GB); benchmark scratch files are deleted after use |
| Resources declared | 4 CPUs, 8 GB RAM (see `docker-compose.yml`) |

On Docker Desktop the Linux VM may have less memory than the declared 8 GB ceiling. On the development machine it had 7.7 GiB, so the effective ceiling there was the VM's, not the compose file's. The benchmark report records the envelope it actually ran in.

### Individual stages

With GNU Make on the host, each target runs inside the container:

```
make build           # build the image
make run             # full pipeline
make test            # test suite (140 tests)
make extract-nbs     # any single stage: extract-nyc, extract-hdx, extract-nbs,
make quality         #   extract-weather, bronze, silver, dims, facts, quality,
make analysis        #   analysis, benchmarks, figures, docs, verify
make clean           # remove derived data, keep downloads
make clean-all       # remove everything, forcing a cold run
```

Without Make, call the entrypoint directly:

```
docker compose run --rm pipeline python -m src.pipeline --stage quality
docker compose run --rm pipeline python -m pytest tests/ -v
```

Every stage can be rerun on its own, and rerunning never duplicates rows.

<!-- TODO before submission: add the exact command or make target that reruns the feature ladder (code in src/paper/). -->

---

## Reproducibility

- The build runs from an empty data directory with one command inside a Docker container.
- Row selection and row order are deterministic: the modelling samples are drawn by hashing each trip's content-addressed identifier, and queries carry an explicit `ORDER BY`. After this was fixed, four complete executions produced byte-identical outputs, and an independent refit reproduced all eighteen published New York City results to four decimal places.
- `docs/run_manifest.json` is rewritten on every run. It records the input files with their row counts and SHA-256 checksums, the rows loaded per table, quality outcomes, stage timings and library versions.
- The environment reported in the paper is Python 3.11.16, DuckDB 1.1.3 and LightGBM 4.5.0. Every dependency version is pinned in `requirements.txt`.
- The paper's supplementary material (descriptive analyses of both corpora, complete hyperparameter search grids, per-repeat results and the record of the two non-determinism defects) is in `outputs/paper/`. The large per-observation error files are regenerable and are not committed.

---

## What it produces

| Location | Contents |
|---|---|
| `outputs/tables/` | One result table per descriptive analysis, for example `A1_demand_profile.csv` |
| `outputs/figures/` | Figures at 300 dpi, colour-blind-safe, with units and source on every chart |
| `outputs/quality/` | `quality_report.{csv,md}`, `reconciliation.csv` (rows in, rejected by reason, loaded, per source), `nbs_reconciliation.csv` |
| `outputs/benchmarks/` | `benchmark_results.{csv,md}` |
| `outputs/paper/` | Supplementary material for the paper |
| `docs/` | Data dictionary (generated from the live schema), ERDs, architecture, SCD strategy, methodology notes, `run_manifest.json` |
| `logs/pipeline.log` | Every stage boundary and timing |

---

## Sources and extraction patterns

Four structurally distinct extraction patterns are used, kept parallel so that they can be compared:

| | Source | Pattern | Rows |
|---|---|---|---|
| A | NYC TLC yellow taxi trips, 2024 (12 Parquet files) | bulk binary download | 41,169,720 |
| B | NYC taxi zone lookup | bulk binary download | 265 |
| C | WFP Nigeria market food prices, via HDX | open data API | 87,384 |
| D | NBS Premium Motor Spirit Price Watch (19 workbooks) | HTML scrape and heterogeneous Excel parse | 1,147 |
| E | Open-Meteo NYC daily weather (optional) | REST/JSON | 366 |

**The correctness check.** `tests/test_pms_reconciliation.py` aggregates the assembled NBS panel to national means and compares them with figures the Bureau itself publishes. All three reference months match to two decimal places (a difference of 0.00 NGN against a 0.05 tolerance). The check tests the whole chain, from spreadsheet cell to star schema, against numbers the project did not produce.

---

## Headline results

These are the results reported in the paper.

- **Integration cost.** Conformance held at a grain ratio of 35,241 to 1 and added no measurable cost beyond that of grain itself.
- **Physical design.** A pre-aggregated table answered a zone-by-date query 131 times faster than the atomic fact table. Date partitioning ran 1.81 times slower than one file already sorted on the date. On the public BigQuery tables, which are unpartitioned, a one-month query was billed for the same 645 MB as a whole year.
- **Downstream effect.** With the model architecture fixed, successive engineering steps reduced mean absolute error on the New York City task by 3.7% on identical trips (mean absolute percentage error rose by 2.7 points). On the Nigerian task a random walk beat every rung in at least 35 of 37 states.

## Findings the build surfaced

These came from building and checking the pipeline. Each is documented in [`docs/methodology_notes.md`](docs/methodology_notes.md).

- **Three "missing" NBS months were never missing.** They were an artefact of a six-row header scan. Some releases put the `State` header on row 14.
- **A silent corruption in the NBS workbooks.** Commentary tables below the state table were being read as data, which put January 2026 prices into the January 2025 column. The parser now stops at the first repeated state.
- **WFP coverage collapses in January 2023,** from 14 states to 3, all in the North East. The fuel-to-food passthrough analysis is therefore kept only as an exploratory null result about conflict-affected north-eastern markets, with correlations under 0.09 at every lag.
- **Four TLC fields fail together.** Of 16 possible missingness patterns, exactly two occur: all four fields present, or all four absent (9.8% of trips).
- **Two non-determinism defects** in how training rows were drawn were found, fixed and guarded by tests (see the paper, Section on reproducibility).

---

## Repository layout

```
config/          settings.py (every tunable), nigeria_states.py, quality_rules.yaml
src/extract/     one module per source, sharing one result contract
src/transform/   bronze.py, silver.py
src/model/       dimensions, Type 2 dim_geography, facts, catalogue
src/quality/     declarative rule engine and reports
src/analysis/    descriptive analysis runner
src/benchmark/   subprocess-isolated timing and peak-memory harness, B1-B4
src/paper/       feature ladder and evaluation code used in the paper
src/viz/         shared figure style and figures
src/docs/        generated data dictionary
src/cloud/       OPTIONAL BigQuery module, isolated from the core run
sql/ddl/         one file per table, executed as written
sql/analysis/    one query file per analysis, each opening with question, grain and assumptions
sql/cloud/       BigQuery variants
tests/           reconciliation, SCD2, state normaliser, quality engine, dim_date, determinism, smoke
outputs/paper/   supplementary material for the paper
```

## Dependencies

Every version is pinned in `requirements.txt`. The core build uses `duckdb`, `pandas`, `pyarrow`, `requests`, `matplotlib`, `openpyxl`, `beautifulsoup4`, `lxml` and `python-dateutil`, with LightGBM for the modelling experiments, PyYAML for the declarative quality rules and pytest for the test suite. There is no Java, no Spark, no database server and no paid service. The warehouse is DuckDB running in-process over Parquet.

## Optional cloud module (Google BigQuery sandbox)

The cloud module lives in `src/cloud/` and has its own `requirements-cloud.txt`. It runs only via `make cloud`, and the core pipeline succeeds whether or not it has ever been run.

1. Create a BigQuery sandbox project. A Google account is enough; no billing account or card is needed.
2. Create a service account with the *BigQuery User* and *BigQuery Data Viewer* roles, and download its JSON key into `secrets/`.
3. Set both variables in `.env`:

```
GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/<your-key>.json
GCP_PROJECT_ID=<your-project-id>
```

4. Run `make cloud`, or `docker compose run --rm pipeline python -m src.cloud.run_cloud`.

`secrets/` and `.env` are excluded from git and from the Docker build context. Credentials are mounted read-only at runtime and are never printed.

One confounder is stated up front. The public BigQuery taxi dataset stops at 2022 and contains no 2024 rows, so the cloud comparison runs against the 2019 yellow taxi table, and every cloud result is labelled with the exact table and date range it used. The comparison illustrates architectural trade-offs. It is not a controlled benchmark of one engine against another.

---

## Data sources, licences and terms of use

This repository contains no source data. The pipeline downloads each source from its publisher at run time, and everything under `data/` is rebuilt locally (the data lake folders are excluded from version control by `.gitignore`).

Anyone who runs the pipeline is responsible for complying with each publisher's terms. The terms below were read on 6 October 2026. They can change, so check the publisher's pages before reuse.

| Source | Publisher | Terms as published | What this means for users of this repository |
|---|---|---|---|
| Yellow taxi trip records (2024) and taxi zone lookup | New York City Taxi and Limousine Commission (TLC) | Published as New York City public data under the City's terms of use. Under NYC Administrative Code section 23-502(d), public data sets carry no restrictions on use, but the City may require users to identify the source and version of a data set and to describe any modifications. The TLC states that it did not create the trip data and makes no representation about its accuracy. | Cite the source, state the files and access date, and describe modifications. The tables built here are modified (quality gate, hash-based `trip_id`, conformed keys), so do not present them as the TLC's own files. |
| Premium Motor Spirit (PMS) Price Watch | National Bureau of Statistics (NBS), Nigeria | The data are not to be redistributed or sold without the written agreement of the NBS, are for statistical and scientific research only, must be cited to the source, and a copy of any resulting publication is to be sent to the NBS. | The assembled state-by-month panel is not committed to this repository and must not be republished. Rebuild it locally with `make extract-nbs`. See the note below. |
| Market food prices, Nigeria | World Food Programme (WFP), via the Humanitarian Data Exchange (HDX) | Creative Commons Attribution for Intergovernmental Organisations (CC BY-IGO). Confirm on the dataset page. | Credit WFP and HDX, link the licence, and state that the data were modified. Do not imply WFP endorses this work. |
| Daily weather, New York City (optional stage) | Open-Meteo.com | Data under Creative Commons Attribution 4.0 (CC BY 4.0). Free for non-commercial use; commercial use requires a paid licence. | Credit "Weather data by Open-Meteo.com" with a link, state any changes, and cite the sources named in the [Open-Meteo citation guidance](https://open-meteo.com/en/docs/historical-weather-api): Zippenfenig (2023) and, for ERA5 data, Hersbach et al. (2023), with the Copernicus statement given there. |
| Public taxi tables, 2019 (optional cloud module) | Google Cloud Public Dataset Program | Provided through Google BigQuery public datasets, with no service-level agreement. | The cloud module only queries these tables and reports billed bytes. No data are copied into this repository. |

### Note on the NBS data

The NBS publishes the PMS Price Watch as a series of spreadsheet releases. No consolidated series exists, so this project assembles one by reading the catalogue of public releases and parsing each workbook (`src/extract/`). Because the NBS terms require its written agreement for redistribution, this repository ships the extraction and parsing code and the reconciliation test, which checks the assembled national monthly means against the means the NBS itself publishes. It does not ship the assembled panel.

If you use the NBS data in a publication, cite the NBS as the source and send the Bureau a copy of the publication, as its terms ask.

### Modifications

The pipeline modifies all sources: rows are typed, deduplicated and conformed to shared dimensions, state and zone names are normalised, and rows failing the declarative quality rules are rejected and reported by reason (`outputs/quality/`). Describe any figure derived from these tables as derived from the original source, not as the publisher's own statistic.

### Disclaimer

The data are provided by their publishers without warranty, and the authors of this repository accept no responsibility for interpretations or decisions based on them. The reported results describe the data as downloaded at the time of the study.

---

## Licence

The source code, configuration, SQL and tests in this repository are released under the MIT Licence (see [`LICENSE`](LICENSE)). The licence covers the code only. It does not extend to any third-party data downloaded by the pipeline, which remain under the terms in the table above.

## How to cite

If you use this platform or its results, please cite the associated paper:

> Olayinka, S.O. (2026) Conformed dimensional integration at extreme grain disparity: integration cost and downstream model effect across two domains. [Journal, volume, pages and DOI to be added on acceptance.]

and cite the data sources as described above.
