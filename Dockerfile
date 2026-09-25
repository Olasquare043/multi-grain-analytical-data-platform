# CSC 796 Advanced Data Engineering -- reproducible pipeline image.
#
# Deliberately slim: no JVM, no Spark, no database server. The entire warehouse
# is DuckDB running in-process against Parquet on a bind-mounted volume, which is
# the architectural claim the term paper makes about single-node analytics.
# Pinned by DIGEST, not just tag: a tag can be repointed upstream at any time,
# which would silently change the interpreter under a "reproducible" build
# (constraint 2.6). The digest pin also lets BuildKit reuse a locally cached
# base image without querying the registry, so a rebuild survives a DNS outage.
FROM python:3.11-slim-bookworm@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=UTC \
    PYTHONPATH=/app

# curl is used only by the container healthcheck / manual debugging.
# fonts-dejavu gives matplotlib a deterministic font so figures are byte-stable.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl fonts-dejavu-core make \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first so source edits do not trigger a reinstall.
COPY requirements.txt requirements-cloud.txt /app/
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r requirements-cloud.txt

# libgomp1 is the OpenMP runtime lightgbm's compiled booster is linked against
# (v3 modelling layer, added after the base layers above); the slim base image
# does not carry it, and lightgbm fails to import at all without it. Placed
# after the pip install layer, not folded into the apt-get line above, so a
# requirements-only change never invalidates this cheap layer and vice versa.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Source is also bind-mounted at runtime; copying it keeps the image standalone.
COPY . /app

# matplotlib needs a writable config dir when running as an arbitrary uid.
ENV MPLCONFIGDIR=/tmp/mpl

CMD ["python", "-m", "src.pipeline", "--stage", "all"]
