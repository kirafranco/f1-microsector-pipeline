# Project brief (original, as written 2026-08-21)

> Kept verbatim as the founding document. Technical corrections and open decisions are tracked in [decisions.md](decisions.md); the buildable breakdown lives in `harness/feature_list.json`.

# Formula 1 Post-Race Telemetry & Performance Engineering Pipeline

## Executive Summary
Designed and deployed an end-to-end telemetry engineering pipeline to conduct micro-sector and spatial performance analysis on Formula 1 Grand Prix sessions. Built within a fully containerized Docker architecture, the system orchestrates automated post-race data extraction via Apache Airflow, executes large-scale spatial-domain interpolations using PySpark, stores transformed datasets in PostgreSQL, and delivers telemetry analytics through an Apache Superset pit-wall dashboard.

---

## Technical Architecture & Workflow

### 1. Data Ingestion & Orchestration (Apache Airflow)
* Automated post-session data extraction pipelines triggered upon race completion via the `FastF1` API.
* Managed job dependencies and task failure retries within an isolated Docker environment.
* Parametrized execution workflows across Grand Prix weekends, sessions (Qualifying/Race), and targeted driver pairings.

### 2. Distributed Data Transformation & Spatial Interpolation (PySpark)
* Converted raw, asynchronous time-series telemetry ($t_i, d_i, v_i$) into uniform 10-meter spatial grids to enable direct driver-to-driver telemetry correlation.
* Implemented mathematical interpolation algorithms across throttle application, braking pressure, RPM, gear selection, and speed.
* Derived continuous spatial time deltas ($\Delta t$) and micro-sector variance metrics to isolate precise braking points, minimum apex cornering speeds ($V_{min}$), and power delivery phases.

### 3. Storage & Relational Modeling (PostgreSQL)
* Structured high-frequency spatial telemetry tables normalized against session metadata and driver dimensions.
* Integrated PySpark DataFrame writes directly to the PostgreSQL backend using optimized JDBC batch connections.

### 4. Pit-Wall Analytics Dashboard (Apache Superset)
* Visualized continuous driver delta ($\Delta t$) curves mapped over track distance to isolate points of cumulative time loss.
* Designed multi-axis telemetry comparative panels (Speed, Throttle %, Brake %) aligned by track coordinates.
* Isolated micro-sector handling deficiencies and traction-limited zones across varying tire compounds and stint progressions.

---

## Tech Stack
* **Orchestration:** Apache Airflow
* **Data Processing & Distributed Computing:** PySpark, Pandas, NumPy
* **Data Storage:** PostgreSQL
* **Data Visualization:** Apache Superset
* **Containerization & Deployment:** Docker, Docker Compose
* **Domain Data Source:** FastF1
