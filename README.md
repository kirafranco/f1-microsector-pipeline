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

- **Corner-anchored distance alignment.** FastF1's distance channel is the time-integral of speed within a lap, so it carries the driver's line and the sensor's error — two drivers' lap totals differ by tens of metres, and a 30 m registration error is ~0.36 s at 300 km/h. Circuit corner positions are used as anchors and each lap's distance axis is rubber-banded between them, so a given grid index means the same physical point on track for every driver. Every downstream metric depends on this holding, so it is tested rather than assumed.
- **A 10 m grid, with the resolution limit stated.** Car data samples at roughly 4 Hz — about 20–25 m between real samples at racing speed — so the grid deliberately oversamples the source. Interpolation does legitimate work for smooth channels, but braking-point precision is bounded at roughly ±20 m regardless, and `Brake` is boolean, so braking points are reported as a window rather than a number to the metre.
  Measured on Suzuka 2024 Q: source samples are 7.2 m apart at the median but up to 83 m apart in dropouts, so 22% of 10 m bins hold no real sample and are pure interpolation. Every grid point therefore carries `source_gap_m`, the spacing of the two source samples around it, so downstream metrics can tell a measured point from an interpolated one. Time is resampled as a channel (`elapsed_time`) rather than re-integrated from speed, because integrating `ds/v` over a reference-line axis was measured to carry a −0.87 s per-lap bias.
- **Micro-sectors are corner phases** — braking, entry, apex, exit — derived from corner positions and the speed and brake traces, with fixed 100 m bins alongside as a secondary grain.
  The unit of segmentation is a speed trough of the session-median trace with at least 8 km/h of prominence, not the numbered corner: chained corners such as Suzuka's esses or the Casio chicane share one braking event, and five Suzuka corners cost too little speed at qualifying pace to be events at all. Circuit-info corners label events rather than define them. Suzuka 2024 Q yields 8 events and 35 corner-phase micro-sectors including straights; boundaries are stable to one 10 m bin on random half-session jackknifes, and per-lap apex and braking-point positions sit within 20 m of the shared boundaries at the 95th percentile.
- **The delta baseline is a parameter**, defaulting to the session fastest lap.
  Time curves are re-zeroed where each lap reaches the shared distance axis, not where FastF1 opens its telemetry window: 46 of 74 Suzuka laps open up to 0.36 s early, and using the window origin biased every delta by +0.15 s. The interior of the chain is verified against official timing: grid time between the S1 and S2 marshalling lines matches the official sector-2 time with a median error of 0.002 s and a spread of 0.057 s across 74 laps, with no scale error. Both lap ends carry a known offset of about 0.28 s against the official timing line, which the validation suite must reconcile. A synthetic ideal lap built from best micro-sectors is deliberately not offered yet: one-bin sector times are noisy enough that the construction comes out 4.6 s faster than pole.
- **Validated against official timing.** The pipeline reconstructs each lap from its own 10 m grid, between the start/finish line positions it locates on the aligned axis, and compares the result with FastF1's official lap and sector times. On Suzuka 2024 Qualifying the reconstructed lap time matches to a median of -0.004 s with a spread of 0.071 s across 72 laps, delta-t closure against pole sits at 0.054 s median and 0.147 s at the 95th percentile, and all three sector times agree to within 0.014 s. Two laps are excluded and named because their telemetry is genuinely missing at the line, not misaligned. The residual is the source's own floor, roughly 4 m of timing-versus-telemetry registration at each crossing, so any comparison between a grid time and an official time carries about 0.1 s.
- **First target: Suzuka 2024 Qualifying.** One event, one session, one driver pair. Generalisation across the 2024 season comes only after the validation suite passes on this one.

All design decisions are currently settled; nothing in the plan is blocked on an open question.

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

## Running the stack

Services live under `servicios/` and run with Docker Compose, one profile at a time (the full stack does not fit in RAM alongside Windows and Docker Desktop):

```
cd servicios
cp .env.example .env          # then replace every CHANGE_ME; generate passwords, do not invent them
docker compose --profile core up -d
docker compose --profile core down
```

| Profile | Services | Feature |
|---|---|---|
| `core` | Postgres 18, Grafana 12 | F001 |
| `dev` | Jupyter | F014 |
| `pipeline` | Spark | F013 |
| `orchestration` | Airflow | F006 |

Everything persists in bind mounts under `data/` (gitignored), so copying the project folder carries the databases with it. Postgres is reachable from the conda environment at `localhost:55432` and Grafana opens at `http://localhost:3000`. `localhost` works on any network or none, because that traffic never leaves the machine. The interface each port is published on is a per-machine setting in `.env` (`*_BIND_ADDRESS`, default `127.0.0.1`, which is `localhost`); set it to `0.0.0.0` only to reach a service from another device on the same Wi-Fi. Grafana connects through a read-only role and its datasource and dashboards are provisioned from files in `servicios/grafana/provisioning/`. The standards the compose file must obey are asserted by tests that run without Docker; `pytest -m docker` brings the stack up and checks health, permissions, persistence and limits on the real daemon.

## Data sources

- [FastF1](https://docs.fastf1.dev/) — primary source: telemetry, laps, tyre stints, weather, circuit info (2018+). Cache enabled at `data/cache/fastf1` before any session load.
- [Jolpica-F1](https://api.jolpi.ca/ergast/f1/) — historical results and schedules (Ergast successor), cached.

All sources are free and official/well-established; raw downloads are immutable snapshots under `data/raw/<source>/<date>/`.

## Data use

Formula 1 timing and telemetry data accessed through FastF1 remains the property of its rights holders. This repository is a personal, non-commercial analysis and engineering project; no F1 data is redistributed here (`data/` is gitignored in full). It is not associated with or endorsed by Formula 1, the FIA, or any team.

## License

Code in this repository is MIT licensed — see [LICENSE](LICENSE). The license covers the pipeline code only, not any data it retrieves.
