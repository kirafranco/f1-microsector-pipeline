#!/usr/bin/env bash
# Start the Spark Connect server in the foreground, in local mode (D2).
set -euo pipefail

log() { printf '%s spark-entrypoint %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

CORES="${SPARK_LOCAL_CORES:-*}"

# The Connect jar ships inside the pyspark distribution, so there is nothing to
# resolve from Maven at start-up. Passing --packages instead made spark-submit
# run Ivy, which wants a writable HOME the non-root user does not have, and the
# server died on a FileNotFoundException before it ever bound a port.
log "starting Connect server: local[${CORES}], local dir ${SPARK_LOCAL_DIRS}"

# start-connect-server.sh daemonises, so the container would exit immediately.
# Running the class directly keeps PID 1 alive and its logs on stdout, which is
# what PYTHONUNBUFFERED and `docker logs` are for.
exec "${SPARK_HOME}/bin/spark-submit" \
    --class org.apache.spark.sql.connect.service.SparkConnectServer \
    --name f1-spark-connect \
    --master "local[${CORES}]" \
    --conf spark.local.dir="${SPARK_LOCAL_DIRS}" \
    --conf spark.connect.grpc.binding.port=15002 \
    --conf spark.sql.execution.arrow.pyspark.enabled=true \
    --conf spark.sql.session.timeZone=UTC \
    --conf spark.ui.enabled=false \
    ""
