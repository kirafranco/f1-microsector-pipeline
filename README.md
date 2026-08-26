# f1-microsector-pipeline

Post-race Formula 1 telemetry engineering pipeline for micro-sector and spatial performance analysis.

Raw, asynchronous time-series telemetry from FastF1 is resampled onto a uniform 10-meter distance grid so that any two drivers (or two laps) can be compared point-by-point along the track: continuous time deltas, braking points, minimum apex speeds, and traction-limited zones — the kind of overlay analysis a pit wall runs after a session.

> **Status: design phase.** This repo currently contains the project brief, the decision log, and the SDD harness. No application code exists yet — features move `pending → spec_ready → approval → in_progress → done`.

## Planned architecture

```
FastF1 API ──► Ingestion (raw parquet snapshots + mandatory FastF1 cache)
                  │
                  ▼
          Track reference — corner-anchored distance coordinate so that a given
          grid index means the same physical place across laps and drivers
                  │
                  ▼
          Spatial resampling — per-lap telemetry → 10 m distance grid
          (NumPy per-lap function; linear interp for continuous channels,
           step interp for discrete; distributed across laps by Spark)
                  │
                  ▼
          Corner segmentation + delta & micro-sector metrics
          (Δt vs distance, V_min, braking points, per-phase time loss)
                  │
                  ▼
          Validation gate — Δt integrated over a lap must equal the actual
          lap-time difference; sector times must reconstruct
                  │
                  ▼
          PostgreSQL star schema
          (dim_session / dim_driver / dim_lap / fact_telemetry_grid / fact_microsector)
                  │
                  ▼
          Grafana pit-wall dashboard (delta curves, shared-crosshair telemetry panels)

Orchestration: Apache Airflow (calendar-driven DAG + data-availability sensor)
```

The build is sliced so that the analysis is proven before any infrastructure is written: slice 1 runs entirely in a conda environment on parquet files and ends when the Δt closure test passes; slice 2 adds containers, Spark, the star schema and the dashboard; slice 3 adds orchestration and generalises beyond the first session.

## Decisions

The original brief is kept verbatim in [docs/project-brief.md](docs/project-brief.md). The full decision log and the feature harness are kept locally and are not versioned; what follows is the summary that matters to a reader.

**Settled:**

- **Grafana** for the dashboard. Dashboard-wide shared crosshair — hover at 1,240 m and read Speed, Throttle, Brake and Δt at that same distance across stacked panels — is the pit-wall interaction, and the Trend panel handles a distance x-axis natively. The schema stays dashboard-agnostic so a Superset or FastAPI front end can be added later without migrating anything.
- **PySpark in local mode** as the distribution layer, chosen deliberately for the engineering experience rather than the data volume, which is modest. The interpolation maths stays a pure NumPy per-lap function, unit-testable without a JVM; Spark distributes it across laps with a grouped-map pandas UDF and batch-writes to Postgres over JDBC.
- **Per-project conda environment** for local development; every service containerised, with compose profiles because the full stack does not fit in laptop RAM at once.
- **Two corrections to the brief, on the data itself:** FastF1's `Brake` channel is boolean, not pressure, so metrics are defined on brake application; and there is no push event at session end, so orchestration polls for availability with backoff.

**Still open, and blocking the specs they touch:** the distance-reference method that makes a grid index mean the same physical place across laps and drivers; grid spacing and the sampling-resolution limit that bounds braking-point precision; the delta baseline; the definition of a micro-sector; the Airflow deployment shape; and ingestion scope.

## Repository layout

```
src/         # importable pipeline modules (created per approved spec)
notebooks/   # exploratory analysis only
data/        # raw/interim/processed/cache — gitignored, never versioned
tests/       # tests
servicios/   # Docker Compose orchestration (one subfolder per service)
docs/        # project brief
```

Spec-driven development state — the feature list, per-feature specs, progress records, and the decision log — is kept on disk and gitignored by design, so it will not appear in a clone. GitHub issues and milestones are the public view of the same plan.

## Environment

- **Local dev:** per-project conda environment created from the Anaconda Prompt (miniconda), e.g. `conda create -n f1-microsector python=3.12`. Dependencies are pinned in `requirements.txt` once the first feature is spec'd.
- **Services:** Postgres, Spark, Airflow, and Grafana run as Docker containers under `servicios/` (bind mounts only, custom bridge network, config via `.env`). Compose profiles are mandatory — the full stack does not fit in laptop RAM at once.

## Data sources

- [FastF1](https://docs.fastf1.dev/) — primary source: telemetry, laps, tyre stints, weather, circuit info (2018+). Cache enabled at `data/cache/fastf1` before any session load.
- [Jolpica-F1](https://api.jolpi.ca/ergast/f1/) — historical results and schedules (Ergast successor), cached.

All sources are free and official/well-established; raw downloads are immutable snapshots under `data/raw/<source>/<date>/`.

## Data use

Formula 1 timing and telemetry data accessed through FastF1 remains the property of its rights holders. This repository is a personal, non-commercial analysis and engineering project; no F1 data is redistributed here (`data/` is gitignored in full). It is not associated with or endorsed by Formula 1, the FIA, or any team.

## License

Code in this repository is MIT licensed — see [LICENSE](LICENSE). The license covers the pipeline code only, not any data it retrieves.
