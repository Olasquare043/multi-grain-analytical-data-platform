# Pipeline Architecture

A medallion architecture on a local filesystem, with DuckDB as the only engine.
No JVM, no Spark, no database server, no paid service. The entire warehouse is an
in-process query engine reading Parquet from a bind-mounted volume.

```mermaid
flowchart TB
    subgraph SOURCES["External sources — four structurally distinct extraction patterns"]
        A["<b>A. NYC TLC Yellow Taxi 2024</b><br/>12 monthly Parquet files<br/><i>bulk binary download</i><br/>41,169,720 rows"]
        B["<b>B. NYC Taxi Zone Lookup</b><br/>one CSV<br/><i>bulk binary download</i><br/>265 rows"]
        C["<b>C. WFP Nigeria Market Prices</b><br/>HDX portal CSV + HXL tag row<br/><i>open data API</i><br/>87,384 rows"]
        D["<b>D. NBS PMS Price Watch</b><br/>catalogue page → 19 workbooks<br/><i>HTML scrape + heterogeneous Excel</i><br/>1,147 observations"]
        E["<b>E. Open-Meteo NYC archive</b><br/>JSON parallel arrays<br/><i>REST/JSON — OPTIONAL</i><br/>366 rows"]
    end

    subgraph RAW["data/raw/ — immutable landing, cached by existence"]
        RN["nyc/*.parquet<br/>nyc/taxi_zone_lookup.csv"]
        RH["hdx/*.csv"]
        RB["nbs/*.xlsx, *.zip<br/>nbs/extracted/<br/>nbs_pms_assembled.csv"]
        RW["weather/*.json"]
    end

    subgraph BRONZE["data/bronze/ — source shape, typed, provenance-stamped"]
        BZ["<b>Registered, not copied</b><br/>bronze_nyc_trip_files.parquet<br/><i>the raw Parquet IS the bronze artefact</i>"]
        BM["bronze_nyc_zones<br/>bronze_wfp_prices / _markets<br/>bronze_nbs_pms<br/>bronze_weather_daily"]
    end

    subgraph SILVER["data/silver/ — cleansed, conformed, one reject reason per row"]
        SV["<b>silver_nyc_trip (VIEW)</b><br/>reject predicate + derivations<br/><i>logical, not materialised</i>"]
        SM["silver_nyc_zones<br/>silver_wfp_prices (+ rejects)<br/>silver_nbs_pms (+ rejects)<br/>silver_weather_daily"]
        SR["silver_nyc_trip_reject_summary<br/>silver_nyc_trip_reject_sample"]
    end

    subgraph GOLD["data/gold/ — Kimball dimensional model"]
        DIMS["<b>Conformed dimensions</b><br/>dim_date · dim_geography (Type 2)<br/>dim_transport_mode · dim_commodity<br/>dim_payment_type · dim_rate_code<br/>dim_vendor · dim_trip_flags (junk)"]
        FT["<b>fact_trip</b><br/>Hive year=/month=<br/>40,421,155 rows · 1.4 GB"]
        FM["fact_market_price_monthly<br/>fact_fuel_price_monthly<br/>fact_trip_daily_agg<br/>fact_weather_daily"]
    end

    subgraph OUT["outputs/ — everything the paper cites"]
        TA["tables/A1..A11 .csv"]
        FI["figures/*.png @ 300 dpi"]
        QU["quality/quality_report.{csv,md}<br/>reconciliation.csv<br/>nbs_reconciliation.csv"]
        BE["benchmarks/benchmark_results.{csv,md}"]
        CL["cloud/ — optional, isolated"]
    end

    subgraph DOCS["docs/"]
        DD["data_dictionary.md (generated)<br/>erd_*.md · architecture.md<br/>scd_strategy.md · methodology_notes.md<br/>run_manifest.json (every run)"]
    end

    A --> RN
    B --> RN
    C --> RH
    D --> RB
    E -.optional.-> RW

    RN --> BZ
    RN --> BM
    RH --> BM
    RB --> BM
    RW -.-> BM

    BZ --> SV
    BM --> SM
    BM -.-> SR
    SV --> SR

    SV --> QG1
    SM --> QG1
    QG1{{"<b>QUALITY GATE — silver</b><br/>range · domain · canonical states<br/>fail ⇒ pipeline stops"}}
    QG1 --> DIMS
    QG1 --> FT
    QG1 --> FM

    DIMS --> FT
    DIMS --> FM
    FT --> FM

    FT --> QG2
    FM --> QG2
    DIMS --> QG2
    QG2{{"<b>QUALITY GATE — gold</b><br/>56 declarative rules<br/>row counts · null rates · referential integrity<br/>ranges · temporal sanity · uniqueness<br/>+ NBS external reconciliation<br/>fail ⇒ no figure is published"}}

    QG2 --> TA
    QG2 --> QU
    TA --> FI
    QG2 --> BE
    QG2 -.optional.-> CL
    QG2 --> DD

    classDef gate fill:#FFF4E6,stroke:#D55E00,stroke-width:2px,color:#1A1A1A
    classDef src fill:#E8F1F8,stroke:#0072B2,color:#1A1A1A
    classDef gold fill:#E9F5EF,stroke:#009E73,color:#1A1A1A
    class QG1,QG2 gate
    class A,B,C,D,E src
    class DIMS,FT,FM gold
```

## Why bronze is physical for Nigeria and logical for New York

The most consequential architectural decision in this pipeline is an asymmetry,
and it is driven by a hard constraint: a **3 GB disk budget**.

- The Nigerian sources total a few tens of megabytes. They are materialised into
  Parquet at every layer, giving a genuinely immutable landing record for a
  trivial cost.
- The NYC corpus is 661 MB of already-columnar, already-immutable Parquet.
  Copying 41 million rows into bronze and again into silver would consume roughly
  2 GB of the 3 GB budget to produce two byte-identical restatements of the
  input. So Source A is **registered rather than copied**: bronze is a provenance
  table (`bronze_nyc_trip_files.parquet`, carrying per-file row counts, timestamp
  extents and out-of-window counts) plus a view, and silver is the
  `silver_nyc_trip` view carrying the reject predicate and derived measures.

The same medallion architecture is therefore **physical at one grain and logical
at another**. That is not a compromise of the pattern; it is the pattern meeting
a real constraint, and it is exactly the kind of engineering cost this study set
out to measure.

One consequence matters for correctness: because the silver reject predicate is
a view rather than a materialised table, the gold fact build and the
reconciliation report read *the same* predicate. They cannot drift apart, which a
materialised silver copy would permit.

## Quality gates

Gates sit at the silver and gold boundaries and are declarative
(`config/quality_rules.yaml`, 56 rules). Severity `fail` stops the pipeline
before any figure is produced; severity `warn` is recorded and carried into the
paper. Reports are written *before* the gate is enforced, so a blocked run still
leaves the evidence that explains why it blocked.

## Idempotence and determinism

- `fact_trip`'s partition directory is cleared before the first month is written,
  so a rerun cannot leave orphaned Parquet fragments behind.
- `trip_id` is a deterministic MD5 over the row's business content, never random.
- `dim_geography`'s initial load stamps a fixed epoch rather than the run date,
  so two cold runs on different days produce identical dimensions.
- Extractors cache by existence, so a warm rerun performs no network I/O and the
  20-minute runtime budget (which excludes downloads) stays meaningful.
