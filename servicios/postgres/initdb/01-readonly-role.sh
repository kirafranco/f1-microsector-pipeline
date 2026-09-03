#!/bin/bash
# Create the read-only role the dashboard connects as (global CLAUDE.md 2.2).
#
# Runs once, during first initialisation of an empty data directory. It does
# NOT re-run against an existing cluster, so changing the password afterwards
# means an explicit ALTER ROLE or removing data/postgres/.
#
# The role name and password arrive as psql variables and are quoted by
# format() with %I / %L, so neither is ever interpolated into SQL text by the
# shell. \gexec runs the statement format() produced.
set -euo pipefail

if [ -z "${POSTGRES_READONLY_USER:-}" ] || [ -z "${POSTGRES_READONLY_PASSWORD:-}" ]; then
    echo "01-readonly-role: POSTGRES_READONLY_USER/PASSWORD not set; no read-only role created" >&2
    exit 1
fi

psql -v ON_ERROR_STOP=1 \
     --username "$POSTGRES_USER" \
     --dbname "$POSTGRES_DB" \
     -v role="$POSTGRES_READONLY_USER" \
     -v pass="$POSTGRES_READONLY_PASSWORD" <<'SQL'
SELECT format('CREATE ROLE %I LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT PASSWORD %L',
              :'role', :'pass')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'role')
\gexec

SELECT format('GRANT CONNECT ON DATABASE %I TO %I', current_database(), :'role') \gexec
SELECT format('GRANT USAGE ON SCHEMA public TO %I', :'role') \gexec
SELECT format('GRANT SELECT ON ALL TABLES IN SCHEMA public TO %I', :'role') \gexec
SELECT format('GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO %I', :'role') \gexec

-- Every table F005 later creates as the admin role is readable without a
-- further grant. Without this the dashboard would break on each migration.
SELECT format('ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO %I', :'role') \gexec
SELECT format('ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON SEQUENCES TO %I', :'role') \gexec
SQL

echo "01-readonly-role: read-only role ${POSTGRES_READONLY_USER} ready"
