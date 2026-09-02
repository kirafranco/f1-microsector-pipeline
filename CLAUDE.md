# CLAUDE.md — f1-microsector-pipeline

Project-specific rules. Global standards in `C:\Users\Kira\.claude\CLAUDE.md` apply; this file only adds or overrides.

## Project

Post-race F1 telemetry pipeline: FastF1 ingestion → 10 m spatial resampling → delta/micro-sector metrics → Postgres star schema → pit-wall dashboard, orchestrated by Airflow. Founding brief: `docs/project-brief.md` (versioned). Decisions: `docs/decisions.md` — **local-only, gitignored (D11)**; D-numbers throughout this file point into it and will not resolve from a fresh clone.

## Overrides and declared exceptions

- **Local dev environment (decision D3):** a per-project conda env (`f1-microsector`), created by Kira from the Anaconda Prompt (miniconda), is the declared exception to global §2 for local development, notebooks, and harness tooling. Services and pipeline runs remain containerized under `servicios/`. Dependencies pinned in `requirements.txt`, shared by the conda env and service images.
- **SDD state is local-only (decision D11):** `harness/` and `docs/decisions.md` are gitignored deliberately. Global §1 requires this state in physical files, not in git, so the standard holds — but there is no history and no remote backup for `feature_list.json`, `specs/`, `progress/`, or the decision log. Anything that must survive a fresh clone goes in `README.md`, `CLAUDE.md`, or `docs/project-brief.md`.
- **Compose profiles are mandatory (decision D9):** the full stack is ~15 GB if every service runs. Profiles: `core` (postgres, grafana), `dev` (jupyter), `pipeline` (spark), `orchestration` (airflow). Never bring up more than the task needs.

## Project conventions

- **Credentials.** Every credential lives in `servicios/.env`, which is gitignored (`**/.env`) and never committed. `servicios/.env.example` is versioned, carries the same keys with `CHANGE_ME` placeholders only, and is updated in the same commit as `.env`. No credential is ever written into a compose file, Dockerfile, image layer, notebook, source file, or test — configuration is injected at runtime. FastF1 and Jolpica-F1 are unauthenticated by design; a source demanding credentials would breach the free-and-trusted source policy and needs sign-off before use.
- FastF1 cache path: `data/cache/fastf1` — enabled before any session load, no exceptions.
- Raw snapshots: `data/raw/fastf1/<snapshot-date>/`, immutable.
- Discrete telemetry channels (`nGear`, `Brake`, `DRS`) are never linearly interpolated — step/previous only.
- **Star schema grain.** `dim_lap` holds the natural key `(season, event, session, driver, lap)` plus a surrogate `lap_id`. `fact_telemetry_grid` is keyed `(lap_id, grid_index)`; `fact_microsector` is keyed `(lap_id, microsector_id)`. Uniqueness validated before every load; loads are idempotent per session partition.
- **Telemetry column types:** `float4` for continuous channels, `smallint` for gear/DRS, `bool` for brake. `fact_telemetry_grid` is partitioned by season/event.
- **The maths is engine-agnostic (decision D2).** Per-lap resampling lives as a pure NumPy function in `src/`, unit-testable without a JVM. Spark is the distribution layer only — never the place where interpolation logic is written.
- **The dashboard is a read-only consumer (decision D1).** No dashboard-specific columns, views, or naming in the schema; it must serve Grafana, Superset, or FastAPI equally.
- `grid_index` is only meaningful under the distance-reference method chosen in D5 — its semantics ship with the schema documentation, alongside the D6 sampling-resolution limit.
- Harness state machine per global §5; no code in `src/`, `tests/`, or `servicios/` while the feature is `pending` or `spec_ready`.
