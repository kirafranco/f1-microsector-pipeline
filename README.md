# f1-microsector-pipeline

Post-race Formula 1 telemetry engineering pipeline for micro-sector and spatial performance analysis.

Raw, asynchronous time-series telemetry from FastF1 is resampled onto a uniform 10-meter distance grid so that any two drivers (or two laps) can be compared point-by-point along the track: continuous time deltas, braking points, minimum apex speeds, and traction-limited zones — the kind of overlay analysis a pit wall runs after a session.

> **Status: design phase.** This repo currently contains the SDD harness and the project brief. No application code exists yet — features move `pending → spec_ready → approval → in_progress → done` (see `harness/feature_list.json`).

## Planned architecture

```
FastF1 API ──► Ingestion (raw parquet snapshots + mandatory FastF1 cache)
                  │
                  ▼
          Spatial resampling — per-lap telemetry → 10 m distance grid
          (linear interp for continuous channels, step interp for discrete)
                  │
                  ▼
          Delta & micro-sector metrics (Δt vs distance, V_min, braking points)
                  │
                  ▼
          PostgreSQL star schema (dim_session / dim_driver / dim_lap / fact_telemetry_grid)
                  │
                  ▼
          Pit-wall dashboard (delta curves, aligned telemetry panels)

Orchestration: Apache Airflow (calendar-driven DAG + data-availability sensor)
```

The full original brief lives in [docs/project-brief.md](docs/project-brief.md). Open design decisions (visualization tool, processing engine) are tracked in [docs/decisions.md](docs/decisions.md).

## Repository layout

```
src/         # importable pipeline modules (created per approved spec)
notebooks/   # exploratory analysis only
data/        # raw/interim/processed/cache — gitignored, never versioned
tests/       # tests
servicios/   # Docker Compose orchestration (one subfolder per service)
harness/     # SDD harness: feature_list.json, specs/, progress/
docs/        # project brief and decision log
```

## Environment

- **Local dev:** per-project conda environment created from the Anaconda Prompt (miniconda), e.g. `conda create -n f1-microsector python=3.12`. Dependencies are pinned in `requirements.txt` once the first feature is spec'd.
- **Services:** Postgres, Airflow, and the dashboard run as Docker containers under `servicios/` (bind mounts only, custom bridge network, config via `.env`).

## Data sources

- [FastF1](https://docs.fastf1.dev/) — primary source: telemetry, laps, tyre stints, weather (2018+). Cache enabled at `data/cache/fastf1` before any session load.
- [Jolpica-F1](https://api.jolpi.ca/ergast/f1/) — historical results and schedules (Ergast successor), cached.

All sources are free and official/well-established; raw downloads are immutable snapshots under `data/raw/<source>/<date>/`.
