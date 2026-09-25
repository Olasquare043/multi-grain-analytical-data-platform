# CSC 796 Advanced Data Engineering -- operational entrypoints.
#
# By default every target executes INSIDE the pipeline container, so a host with
# nothing but Docker Desktop can run the whole study:
#
#     make build && make run
#
# To run a target natively (e.g. from inside an interactive container shell),
# neutralise the runner:  make run RUNNER=
#
RUNNER ?= docker compose run --rm pipeline
PY     := $(RUNNER) python -m src.pipeline --stage

.DEFAULT_GOAL := help
.PHONY: help build run test clean clean-all cloud shell verify \
        extract extract-nyc extract-hdx extract-nbs extract-weather \
        bronze silver dims facts quality analysis benchmarks figures docs

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

build: ## Build the pipeline image
	docker compose build

run: ## Full end-to-end pipeline (sections 1-8, 10-13)
	$(PY) all

# ---------------------------------------------------------------- extraction
extract:         ## All four extraction patterns
	$(PY) extract
extract-nyc:     ## Source A + B: bulk binary download (TLC parquet, zone lookup)
	$(PY) extract-nyc
extract-hdx:     ## Source C: open-data API (WFP Nigeria market prices via HDX)
	$(PY) extract-hdx
extract-nbs:     ## Source D: HTML scrape + heterogeneous Excel parse (NBS PMS)
	$(PY) extract-nbs
extract-weather: ## Source E: REST/JSON (Open-Meteo), optional and non-blocking
	$(PY) extract-weather

# ------------------------------------------------------------------ modelling
bronze:      ## Raw -> bronze (typed, provenance-stamped, still source-shaped)
	$(PY) bronze
silver:      ## Bronze -> silver (cleaned, conformed, quality-gated)
	$(PY) silver
dims:        ## Gold dimensions (incl. Type 2 SCD on dim_geography)
	$(PY) dims
facts:       ## Gold facts (fact_trip partitioned, monthly facts, daily aggregate)
	$(PY) facts

# ------------------------------------------------------------------- outputs
quality:     ## Data quality framework + reconciliation report (section 6)
	$(PY) quality
analysis:    ## Analyses A1-A11 (section 7)
	$(PY) analysis
benchmarks:  ## Benchmarks B1-B4 (section 8)
	$(PY) benchmarks
figures:     ## Publication figures (section 12)
	$(PY) figures
docs:        ## Generated documentation + diagrams (section 10)
	$(PY) docs

# --------------------------------------------------------------------- other
test: ## Run the test suite
	$(RUNNER) python -m pytest tests/ -v

cloud: ## OPTIONAL BigQuery module (section 9) -- isolated from the core run
	$(RUNNER) python -m src.cloud.run_cloud

verify: ## Assert every promised output exists and print a completion summary
	$(PY) verify

shell: ## Interactive shell in the pipeline container
	$(RUNNER) /bin/bash

clean: ## Remove derived data and outputs, keep downloaded raw files
	$(RUNNER) python -m src.pipeline --stage clean

clean-all: ## Remove everything including raw downloads (forces a cold run)
	$(RUNNER) python -m src.pipeline --stage clean --include-raw
