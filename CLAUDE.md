# CLAUDE.md — f1-microsector-pipeline

Project-specific rules. Global standards in `C:\Users\Kira\.claude\CLAUDE.md` apply; this file only adds or overrides.

## Project

Post-race F1 telemetry pipeline: FastF1 ingestion → 10 m spatial resampling → delta/micro-sector metrics → Postgres star schema → pit-wall dashboard, orchestrated by Airflow. Founding brief: `docs/project-brief.md`. Decisions: `docs/decisions.md`.

## Overrides and declared exceptions

- **Local dev environment (decision D3):** a per-project conda env (`f1-microsector`), created by Kira from the Anaconda Prompt (miniconda), is the declared exception to global §2 for local development, notebooks, and harness tooling. Services and pipeline runs remain containerized under `servicios/`. Dependencies pinned in `requirements.txt`, shared by the conda env and service images.

## Project conventions

- FastF1 cache path: `data/cache/fastf1` — enabled before any session load, no exceptions.
- Raw snapshots: `data/raw/fastf1/<snapshot-date>/`, immutable.
- Discrete telemetry channels (`nGear`, `Brake`, `DRS`) are never linearly interpolated — step/previous only.
- Fact-table business key: `(season, event, session, driver, lap, grid_index)` — uniqueness validated before every load; loads are idempotent per session partition.
- Harness state machine per global §5; no code in `src/`, `tests/`, or `servicios/` while the feature is `pending` or `spec_ready`.
