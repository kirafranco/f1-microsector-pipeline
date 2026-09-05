-- F005 star schema: six dimensions, three facts, one audit trail.
--
-- Nothing here exists for one consumer (decision D1): no view, no function,
-- no denormalised convenience column. Grafana, Superset and a FastAPI service
-- read the same tables with the same SQL. Column names match the parquet
-- contracts, so a reader of the files and a reader of the database share one
-- vocabulary.
--
-- Continuous telemetry is `real` (float4) throughout: measured at 119 B/row
-- all-in, 1.41 GiB for the whole 2024 qualifying-and-race scope against a
-- 20 GiB budget. `double precision` would add half a heap for precision the
-- source does not have.

-- ---------------------------------------------------------------- dimensions

CREATE TABLE dim_session (
    session_id               serial PRIMARY KEY,
    season                   smallint     NOT NULL,
    round                    smallint     NOT NULL,
    session_code             text         NOT NULL,
    session_name             text         NOT NULL,
    event_name               text         NOT NULL,
    circuit_id               text,
    circuit_name             text,
    locality                 text,
    country                  text,
    session_start_utc        timestamptz,
    official_lap_length_m    real,
    reference_line_length_m  real,
    alignment_method         text,
    snapshot_date            date,
    contract_version         text,
    loaded_at                timestamptz  NOT NULL DEFAULT now(),
    UNIQUE (season, round, session_code)
);

COMMENT ON TABLE dim_session IS
    'One row per ingested session. official_lap_length_m comes from the FIA circuit specification (F008), not from the API.';

CREATE TABLE dim_constructor (
    season          smallint NOT NULL,
    constructor_id  text     NOT NULL,
    name            text     NOT NULL,
    nationality     text,
    PRIMARY KEY (season, constructor_id)
);

CREATE TABLE dim_driver (
    driver_key        serial PRIMARY KEY,
    season            smallint NOT NULL,
    code              text     NOT NULL,
    driver_id         text     NOT NULL,
    permanent_number  smallint,
    given_name        text,
    family_name       text,
    full_name         text,
    nationality       text,
    date_of_birth     date,
    UNIQUE (season, code)
);

COMMENT ON TABLE dim_driver IS
    'Driver identity only. A driver''s constructor belongs to a round, not to a season, so it lives on dim_lap.';

CREATE TABLE dim_lap (
    lap_id                  serial PRIMARY KEY,
    session_id              integer  NOT NULL REFERENCES dim_session (session_id) ON DELETE CASCADE,
    season                  smallint NOT NULL,
    round                   smallint NOT NULL,
    session_code            text     NOT NULL,
    code                    text     NOT NULL,
    lap_number              smallint NOT NULL,
    constructor_id          text,
    team_alias              text,
    compound                text,
    stint                   smallint,
    tyre_life               smallint,
    is_reference            boolean  NOT NULL DEFAULT false,
    reference_lap           text,
    n_points                integer,
    lap_time_s              real,
    grid_time_s             real,
    lap_grid_s              real,
    lap_residual_s          real,
    closure_residual_s      real,
    s1_official_s           real,
    s2_official_s           real,
    s3_official_s           real,
    s1_grid_s               real,
    s2_grid_s               real,
    s3_grid_s               real,
    driven_m                real,
    line_start_m            real,
    line_end_m              real,
    window_open_s           real,
    window_close_s          real,
    start_coverage_poor     boolean,
    end_coverage_poor       boolean,
    UNIQUE (session_id, code, lap_number),
    FOREIGN KEY (season, constructor_id) REFERENCES dim_constructor (season, constructor_id)
);

COMMENT ON COLUMN dim_lap.constructor_id IS
    'Resolved from the round''s own driver entry (F012), never by matching FastF1 team strings: the two agree only 6 times in 10.';
COMMENT ON COLUMN dim_lap.team_alias IS
    'FastF1''s team string, kept for display. Not a key and not joinable to dim_constructor.';

CREATE TABLE dim_microsector (
    session_id      integer  NOT NULL REFERENCES dim_session (session_id) ON DELETE CASCADE,
    grain           text     NOT NULL,
    microsector_id  smallint NOT NULL,
    phase           text     NOT NULL,
    event_id        smallint,
    corners         text,
    marginal        boolean,
    start_m         real     NOT NULL,
    end_m           real     NOT NULL,
    start_index     integer  NOT NULL,
    end_index       integer  NOT NULL,
    length_m        real     NOT NULL,
    PRIMARY KEY (session_id, grain, microsector_id)
);

COMMENT ON TABLE dim_microsector IS
    'Boundaries are session medians (F009), so this is a per-session dimension. length_m below ~30 m is individually noisy: filter on it before ranking.';

CREATE TABLE dim_corner_event (
    session_id       integer  NOT NULL REFERENCES dim_session (session_id) ON DELETE CASCADE,
    event_id         smallint NOT NULL,
    corners          text,
    marginal         boolean,
    has_braking      boolean,
    apex_m           real,
    v_min_kmh        real,
    prominence_kmh   real,
    lift_m           real,
    brake_on_m       real,
    brake_off_m      real,
    apex_start_m     real,
    apex_end_m       real,
    exit_end_m       real,
    PRIMARY KEY (session_id, event_id)
);

-- --------------------------------------------------------------------- facts
--
-- Partition columns must appear in the primary key, so every fact carries
-- (season, round, session_code) alongside lap_id. The event level is what
-- decision D10 asks for and what pruning uses; the session leaf is what makes
-- a reload a TRUNCATE rather than a DELETE, so a duplicate row is impossible
-- by construction.

CREATE TABLE fact_telemetry_grid (
    season        smallint NOT NULL,
    round         smallint NOT NULL,
    session_code  text     NOT NULL,
    lap_id        integer  NOT NULL REFERENCES dim_lap (lap_id) ON DELETE CASCADE,
    grid_index    integer  NOT NULL,
    distance_m    real     NOT NULL,
    t_s           real     NOT NULL,
    delta_t_s     real,
    speed         real     NOT NULL,
    throttle      real     NOT NULL,
    rpm           real     NOT NULL,
    x             real     NOT NULL,
    y             real     NOT NULL,
    n_gear        smallint NOT NULL,
    drs           smallint NOT NULL,
    brake         boolean  NOT NULL,
    source_gap_m  real,
    PRIMARY KEY (season, round, session_code, lap_id, grid_index)
) PARTITION BY RANGE (season, round);

COMMENT ON COLUMN fact_telemetry_grid.delta_t_s IS
    'Against the session default reference only (F004). Any other reference is a subtraction of two t_s curves, done on read.';
COMMENT ON COLUMN fact_telemetry_grid.source_gap_m IS
    'Spacing of the two source samples bracketing this grid point. NULL at grid 0. About one point in five is pure interpolation (F003).';

CREATE TABLE fact_microsector (
    season          smallint NOT NULL,
    round           smallint NOT NULL,
    session_code    text     NOT NULL,
    lap_id          integer  NOT NULL REFERENCES dim_lap (lap_id) ON DELETE CASCADE,
    grain           text     NOT NULL,
    microsector_id  smallint NOT NULL,
    time_s          real,
    delta_s         real,
    partial         boolean  NOT NULL,
    PRIMARY KEY (season, round, session_code, lap_id, grain, microsector_id)
) PARTITION BY RANGE (season, round);

COMMENT ON COLUMN fact_microsector.partial IS
    'The lap ended inside this micro-sector, so time_s is NULL or covers only part of it.';

CREATE TABLE fact_corner_metric (
    lap_id        integer  NOT NULL REFERENCES dim_lap (lap_id) ON DELETE CASCADE,
    event_id      smallint NOT NULL,
    v_min_kmh     real,
    v_min_m       real,
    apex_dev_m    real,
    brake_on_m    real,
    brake_dev_m   real,
    brake_gap_m   real,
    PRIMARY KEY (lap_id, event_id)
);

COMMENT ON TABLE fact_corner_metric IS
    'About 180k rows for a whole season, so it is not partitioned; a reload deletes the session''s laps by lap_id.';
COMMENT ON COLUMN fact_corner_metric.brake_gap_m IS
    'Source sample spacing at the braking point: this braking point''s own uncertainty (decision D6).';

-- --------------------------------------------------------------------- audit

CREATE TABLE load_audit (
    load_id             serial PRIMARY KEY,
    session_id          integer     NOT NULL REFERENCES dim_session (session_id) ON DELETE CASCADE,
    loaded_at           timestamptz NOT NULL DEFAULT now(),
    contract_version    text,
    quality_ok          boolean     NOT NULL,
    quality_warnings    integer     NOT NULL DEFAULT 0,
    rows_grid           integer     NOT NULL,
    rows_microsector    integer     NOT NULL,
    rows_corner         integer     NOT NULL,
    rows_lap            integer     NOT NULL,
    snapshot_paths      jsonb,
    elapsed_s           real
);

COMMENT ON TABLE load_audit IS
    'One row per load. quality_ok is always true for a committed load: the F011 gate refuses to write otherwise.';
