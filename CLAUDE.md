# CLAUDE.md — f1-microsector-pipeline

Project-specific rules. Global standards in `C:\Users\Kira\.claude\CLAUDE.md` apply; this file only adds or overrides.

## Project

Post-race F1 telemetry pipeline: FastF1 ingestion → 10 m spatial resampling → delta/micro-sector metrics → Postgres star schema → pit-wall dashboard, orchestrated by Airflow. Founding brief: `docs/project-brief.md` (versioned). Decisions: `docs/decisions.md` — **local-only, gitignored (D11)**; D-numbers throughout this file point into it and will not resolve from a fresh clone.

## Overrides and declared exceptions

- **Local dev environment (decision D3):** a per-project conda env (`f1-microsector`), created by Kira from the Anaconda Prompt (miniconda), is the declared exception to global §2 for local development, notebooks, and harness tooling. Services and pipeline runs remain containerized under `servicios/`. Dependencies pinned in `requirements.txt`, shared by the conda env and service images.
- **SDD state is local-only (decision D11):** `harness/` and `docs/decisions.md` are gitignored deliberately. Global §1 requires this state in physical files, not in git, so the standard holds — but there is no history and no remote backup for `feature_list.json`, `specs/`, `progress/`, or the decision log. Anything that must survive a fresh clone goes in `README.md`, `CLAUDE.md`, or `docs/project-brief.md`.
- **Compose profiles are mandatory (decision D9):** the full stack is ~15 GB if every service runs. Profiles: `core` (postgres, grafana), `dev` (jupyter), `pipeline` (spark), `orchestration` (airflow). Never bring up more than the task needs. A service belongs to every profile that needs it — postgres is in both `core` and `orchestration` — because Compose does not enable a dependency's profile on its own.
- **The Airflow image runs as the `airflow` user (F006):** the official `apache/airflow` image is built around its own user (uid 50000) and its entrypoint expects it, so global §2.1's `USER 1000:1000` does not apply. The reason for that rule — bind-mounted files ending up root-owned on the host — does not arise on Docker Desktop for Windows, which does not surface container ownership onto the host filesystem. The exception is written next to the `USER` line in the Dockerfile and asserted by a test.
- **The build context is the project root (F006):** the Airflow image carries `src/` and `requirements.txt`, so `.dockerignore` lives at the repository root and covers every image built here. It excludes `data/`, `**/.env`, `harness/`, `docs/`, `notebooks/`, `tests/`, caches and parquet — nothing with data or a credential in it reaches a layer.
- **graphify is developer tooling, not a dependency (decision D12):** the repo knowledge graph tool is installed on the host through `pipx` (`graphifyy==0.9.53`, isolated, pinned) under the global §2 dev-tooling exception, and never enters `requirements.txt` or a service image. Its skill and hooks are project-scoped under `.claude/` and versioned; its output `graphify-out/` is generated and gitignored, rebuilt with `graphify update .`. Code parsing is local; no API key is configured and the hosted platform is not used. The `## graphify` section below is written by the tool itself.

## Project conventions

- **Credentials.** Every credential lives in `servicios/.env`, which is gitignored (`**/.env`) and never committed. `servicios/.env.example` is versioned, carries the same keys with `CHANGE_ME` placeholders only, and is updated in the same commit as `.env`. No credential is ever written into a compose file, Dockerfile, image layer, notebook, source file, or test — configuration is injected at runtime. FastF1 and Jolpica-F1 are unauthenticated by design; a source demanding credentials would breach the free-and-trusted source policy and needs sign-off before use.
- FastF1 cache path: `data/cache/fastf1` — enabled before any session load, no exceptions.
- Raw snapshots: `data/raw/fastf1/<snapshot-date>/`, immutable.
- Discrete telemetry channels (`nGear`, `Brake`, `DRS`) are never linearly interpolated — step/previous only.
- **Star schema grain.** `dim_lap` holds the natural key `(season, event, session, driver, lap)` plus a surrogate `lap_id`. `fact_telemetry_grid` is keyed `(lap_id, grid_index)`; `fact_microsector` is keyed `(lap_id, grain, microsector_id)` — `grain` is in the key because micro-sector ids are per grain (F009). Uniqueness validated before every load; loads are idempotent per session partition, by truncating that session's leaf partition rather than deleting rows.
- **Telemetry column types:** `float4` for continuous channels, `smallint` for gear/DRS, `bool` for brake. `fact_telemetry_grid` is partitioned by season/event.
- **The maths is engine-agnostic (decision D2).** Per-lap resampling lives as a pure NumPy function in `src/`, unit-testable without a JVM. Spark is the distribution layer only — never the place where interpolation logic is written.
- **The dashboard is a read-only consumer (decision D1).** No dashboard-specific columns, views, or naming in the schema; it must serve Grafana, Superset, or FastAPI equally.
- `grid_index` is only meaningful under the distance-reference method chosen in D5 — its semantics ship with the schema documentation, alongside the D6 sampling-resolution limit.
- **Orchestration (F006).** `src/orchestration/` holds every decision the DAGs make — run paths, availability, calendar selection, stage adapters — and is tested in the conda env. `servicios/airflow/dags/` holds wiring only and imports nothing from a pipeline stage directly, so the path convention has exactly one definition. Airflow itself is never installed in the conda env.
- **A run's artefacts are named `<season>_<event>_<session>_<method>`**, built by `SessionRun` and by nothing else. The event is the schedule's `event_name` ("Japanese Grand Prix"), not the country: three 2024 rounds are in the United States and two in Italy.
- Harness state machine per global §5; no code in `src/`, `tests/`, or `servicios/` while the feature is `pending` or `spec_ready`.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
