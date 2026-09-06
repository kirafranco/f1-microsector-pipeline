# Spark Connect server for the resampling executor (F013).
#
# Built on the official python image rather than apache/spark, and the reason is
# the project's own pins (D2, amended 2026-09-06). apache/spark:4.1.1 ships
# Python 3.10.12 and no pyspark package; this project pins numpy==2.4.6, which
# requires Python >= 3.11 and has no 3.10 build. Worse, Spark Connect pickles a
# pandas UDF from the client, and bytecode does not travel between Python
# minors -- a 3.12 client driving 3.10 workers is a runtime failure waiting for
# the first UDF that cannot be pickled by reference. python:3.12-slim matches
# the conda env's minor, so client, Airflow and workers share one requirements
# file, one pandas, one numpy, one pyarrow.
#
# Spark itself is the Apache release: the pyspark distribution on PyPI bundles
# the full runtime, including sbin/start-connect-server.sh.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/project \
    SPARK_HOME=/usr/local/lib/python3.12/site-packages/pyspark \
    SPARK_LOCAL_DIRS=/tmp/spark-local \
    HOME=/home/spark

# Spark 4.1 runs on JDK 17 or 21; Debian 13 carries 21.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-21-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, code second: the project arrives by bind mount at runtime.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir pyspark==4.1.1

# Scratch for shuffle spill. Never the bind mount: on Docker Desktop for Windows
# a Windows-filesystem mount makes spill crawl (D2).
# A named account for uid 1000, not just the number. Hadoop's
# UserGroupInformation resolves the effective uid through /etc/passwd on every
# Spark start-up, and a bare `USER 1000:1000` on this base image has no entry:
# the server dies with "Invalid UID, could not determine effective user" before
# it binds a port. The uid is still 1000 so bind-mounted files are not
# root-owned on the host (global 2.1); it now simply has a name and a HOME.
RUN useradd --uid 1000 --create-home --home-dir /home/spark --shell /usr/sbin/nologin spark \
    && mkdir -p /tmp/spark-local \
    && chown -R 1000:1000 /tmp/spark-local /home/spark

# The image runs standalone (global 2.1); the compose mounts the live tree over it.
COPY src/ /opt/project/src/

COPY servicios/spark/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

# Non-root, so files this writes on a bind mount are not root-owned (global 2.1).
USER 1000:1000

EXPOSE 15002
CMD ["/opt/entrypoint.sh"]
