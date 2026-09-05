#!/usr/bin/env bash
# Bring Airflow up against the project's Postgres, idempotently (F006).
#
# Three things have to be true before `airflow standalone` runs, and none of
# them can be done by an initdb script: those run once, when the cluster is
# first created, and data/postgres/ already exists on this machine.
#
#   1. the metadata database exists
#   2. the admin password is the one in .env, not a generated one
#   3. the SQLAlchemy URL is assembled without the password reaching a file
#
# No credential is written into any versioned file; every value arrives in the
# environment from servicios/.env via the compose.
set -euo pipefail

log() { echo "entrypoint: $*" >&2; }

: "${POSTGRES_HOST:?POSTGRES_HOST is not set}"
: "${POSTGRES_PORT:?POSTGRES_PORT is not set}"
: "${POSTGRES_USER:?POSTGRES_USER is not set}"
: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is not set}"
: "${POSTGRES_DB:?POSTGRES_DB is not set}"
: "${AIRFLOW_DB:?AIRFLOW_DB is not set}"
: "${AIRFLOW_ADMIN_USER:?AIRFLOW_ADMIN_USER is not set}"
: "${AIRFLOW_ADMIN_PASSWORD:?AIRFLOW_ADMIN_PASSWORD is not set}"

STATE_DIR="${AIRFLOW_STATE_DIR:-/opt/airflow/state}"
PASSWORD_FILE="${AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE:-${STATE_DIR}/passwords.json}"

mkdir -p "${STATE_DIR}" "${STATE_DIR}/logs"

# --- 1. wait for Postgres ----------------------------------------------------
# The compose already gates on service_healthy; this covers `restart: always`
# restarting the container while the database is still coming back.
export PGPASSWORD="${POSTGRES_PASSWORD}"
for attempt in $(seq 1 60); do
    if psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" \
            -d "${POSTGRES_DB}" -tAc 'select 1' >/dev/null 2>&1; then
        break
    fi
    if [ "${attempt}" -eq 60 ]; then
        log "postgres did not answer after 60 attempts"
        exit 1
    fi
    sleep 2
done
log "postgres is answering"

# --- 2. create the metadata database if it is absent -------------------------
# format() with %I quotes the identifier, so the database name is never
# interpolated into SQL text by the shell. CREATE DATABASE cannot run inside a
# transaction block, hence \gexec on a single statement.
psql -v ON_ERROR_STOP=1 -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" \
     -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -v db="${AIRFLOW_DB}" <<'SQL'
SELECT format('CREATE DATABASE %I', :'db')
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = :'db')
\gexec
SQL
log "metadata database ${AIRFLOW_DB} ready"
unset PGPASSWORD

# --- 3. write the password file before Airflow generates one -----------------
# SimpleAuthManager generates a random password and prints it to stdout for any
# configured user missing from this file. Writing it first means the password
# is the one in .env and nothing is ever echoed into the logs.
python - "$PASSWORD_FILE" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
user = os.environ["AIRFLOW_ADMIN_USER"]
password = os.environ["AIRFLOW_ADMIN_PASSWORD"]

passwords = {}
if path.exists():
    try:
        passwords = json.loads(path.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError:
        passwords = {}

if passwords.get(user) != password:
    passwords[user] = password
    path.write_text(json.dumps(passwords) + "\n", encoding="utf-8")
    path.chmod(0o600)
    print("entrypoint: admin password written from the environment", file=sys.stderr)
else:
    print("entrypoint: admin password already matches the environment", file=sys.stderr)
PY

# --- 4. hand over to Airflow -------------------------------------------------
# The connection comes from the container environment, which the compose sets
# from .env. It is deliberately not exported here: a variable exported by this
# process would not reach `docker compose exec`, and the Airflow CLI would then
# use its default SQLite database and report the schema as unmigrated.
: "${AIRFLOW__DATABASE__SQL_ALCHEMY_CONN:?the compose must provide the metadata connection}"

log "starting airflow $*"
exec airflow "$@"
