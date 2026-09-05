"""One contract per pipeline artefact.

Ranges are physical envelopes rather than the values Suzuka happened to
produce: they exist to catch a unit mistake or a corrupted row, not to encode
one circuit. Permitted nulls are expressed as ``unless`` predicates on strict
rules, so a null anywhere it has no reason to be is still an error.
"""

from __future__ import annotations

from src.quality import predicates as p
from src.quality.engine import TableContract
from src.quality.rules import ERROR, WARNING, AllowedValues, ForeignKey, Invariant, NotNull, Range, Unique
from src.reference.tables import SESSION_CODES

#: FastF1's DRS code set. Anything else is undocumented and worth a look.
DRS_CODES = (0, 1, 8, 9, 10, 12, 14)
#: F012 session codes (FP1..R) come from the reference tables module, so the
#: allowed-value rule and the builder can never drift apart.
#: Corner phases F009 emits, plus the fixed-bin label.
PHASES = ("braking", "entry", "apex", "exit", "straight", "bin")
GRAINS = ("corner_phase", "fixed_100m")
REFERENCE_KINDS = ("session_fastest", "lap", "driver_best")

_RAW_TELEMETRY_KEY = ("driver", "lap_number", "session_time")
_LAP_KEY = ("driver", "lap_number")
_GRID_KEY = ("driver", "lap_number", "grid_index")


def _contract(name, key, rules, parents=()) -> TableContract:
    return TableContract(name=name, key=tuple(key), rules=tuple(rules), parents=tuple(parents))


CONTRACTS: dict[str, TableContract] = {
    # --- F002 raw snapshot ----------------------------------------------------
    "laps": _contract(
        "laps", _LAP_KEY,
        [
            NotNull(check_columns=("driver", "lap_number", "team", "stint", "compound", "is_accurate")),
            NotNull(check_columns=("lap_time", "sector1_time", "sector2_time", "sector3_time", "lap_start_time"),
                    unless=p.lap_not_accurate),
            Unique(key=_LAP_KEY),
            Range(column="lap_number", low=1, high=200),
            # 20-600 s is the unit envelope: it catches a lap time in minutes
            # (1.47) or milliseconds (88197), without asserting how long a
            # circuit is. Suzuka runs 88 s, Monaco 70; a test fixture may be shorter.
            Range(column="lap_time", low=20.0, high=600.0),
            Range(column="sector1_time", low=10.0, high=200.0),
            Range(column="sector2_time", low=10.0, high=200.0),
            Range(column="sector3_time", low=10.0, high=200.0),
            Range(column="stint", low=1, high=50),
            Range(column="tyre_life", low=0, high=200),
        ],
    ),
    "car_telemetry": _contract(
        "car_telemetry", _RAW_TELEMETRY_KEY,
        [
            NotNull(check_columns=("driver", "lap_number", "session_time", "speed", "throttle", "brake", "rpm", "n_gear", "drs")),
            Unique(key=_RAW_TELEMETRY_KEY),
            ForeignKey(key=_LAP_KEY, parent="laps"),
            Range(column="speed", low=0.0, high=400.0),
            # FastF1 reports slightly over 100 % on out-laps; the grid, built from
            # flying laps only, stays at 100. The envelope tolerates both.
            Range(column="throttle", low=0.0, high=110.0),
            Range(column="rpm", low=0.0, high=16000.0),
            Range(column="n_gear", low=0, high=8),
            AllowedValues(column="drs", values=DRS_CODES),
        ],
        parents=("laps",),
    ),
    "pos_data": _contract(
        "pos_data", _RAW_TELEMETRY_KEY,
        [
            NotNull(check_columns=("driver", "lap_number", "session_time", "x", "y", "z")),
            Unique(key=_RAW_TELEMETRY_KEY),
            ForeignKey(key=_LAP_KEY, parent="laps"),
            # Position arrives in 1/10 m (F008 finding), so the envelope is ten
            # times the metre one: no circuit spans 5 km from its origin.
            Range(column="x", low=-50000.0, high=50000.0),
            Range(column="y", low=-50000.0, high=50000.0),
        ],
        parents=("laps",),
    ),
    "weather": _contract(
        "weather", ("session_time",),
        [
            NotNull(check_columns=("session_time", "air_temp", "track_temp", "humidity", "pressure", "rainfall")),
            Unique(key=("session_time",)),
            Range(column="air_temp", low=-20.0, high=60.0),
            Range(column="track_temp", low=-20.0, high=80.0),
            Range(column="humidity", low=0.0, high=100.0),
            Range(column="pressure", low=800.0, high=1100.0),
            Range(column="wind_speed", low=0.0, high=60.0),
            Range(column="wind_direction", low=0, high=360),
        ],
    ),
    "circuit_corners": _contract(
        "circuit_corners", ("number",),
        [
            NotNull(check_columns=("number", "x", "y", "distance")),
            Unique(key=("number", "letter")),
            Range(column="number", low=1, high=40),
            Range(column="distance", low=0.0, high=10000.0),
        ],
    ),
    # --- F008 aligned ---------------------------------------------------------
    "telemetry_aligned": _contract(
        "telemetry_aligned", _RAW_TELEMETRY_KEY,
        [
            NotNull(check_columns=("driver", "lap_number", "session_time", "speed", "throttle", "rpm",
                                   "n_gear", "brake", "drs", "x", "y", "distance_raw", "distance_aligned",
                                   "line_offset_m")),
            Unique(key=_RAW_TELEMETRY_KEY),
            ForeignKey(key=_LAP_KEY, parent="laps"),
            Invariant(name="aligned_laps_are_accurate", check=p.aligned_laps_are_accurate, about=_LAP_KEY),
            Invariant(name="aligned_distance_non_decreasing", check=p.aligned_distance_non_decreasing,
                      about=("distance_aligned",)),
            Range(column="speed", low=0.0, high=400.0),
            Range(column="throttle", low=0.0, high=110.0),
            Range(column="rpm", low=0.0, high=16000.0),
            Range(column="n_gear", low=0, high=8),
            AllowedValues(column="drs", values=DRS_CODES),
            # Metres now: positions were converted out of FastF1's 1/10 m units.
            Range(column="x", low=-5000.0, high=5000.0),
            Range(column="y", low=-5000.0, high=5000.0),
            Range(column="distance_raw", low=0.0, high=10000.0),
            # The axis may start slightly before the reference line (F010).
            Range(column="distance_aligned", low=-200.0, high=10000.0),
            Range(column="line_offset_m", low=0.0, high=100.0),
        ],
        parents=("laps",),
    ),
    # --- F003 grid ------------------------------------------------------------
    "grid": _contract(
        "grid", _GRID_KEY,
        [
            NotNull(check_columns=("driver", "lap_number", "grid_index", "distance_m", "elapsed_time",
                                   "speed", "throttle", "rpm", "x", "y", "n_gear", "brake", "drs")),
            NotNull(check_columns=("source_gap_m",), unless=p.at_grid_zero),
            Unique(key=_GRID_KEY),
            ForeignKey(key=_LAP_KEY, parent="telemetry_aligned"),
            Invariant(name="grid_distance_matches_index", check=p.grid_distance_matches_index,
                      about=("grid_index", "distance_m")),
            Invariant(name="grid_index_contiguous_per_lap", check=p.grid_index_contiguous_per_lap,
                      about=("grid_index",)),
            Invariant(name="elapsed_time_increasing_per_lap", check=p.elapsed_time_increasing_per_lap,
                      about=("elapsed_time",)),
            Range(column="grid_index", low=0, high=10000),
            Range(column="distance_m", low=0.0, high=100000.0),
            Range(column="elapsed_time", low=0.0, high=300.0),
            Range(column="speed", low=0.0, high=400.0),
            Range(column="throttle", low=0.0, high=100.0),
            Range(column="rpm", low=0.0, high=16000.0),
            Range(column="n_gear", low=0, high=8),
            AllowedValues(column="drs", values=DRS_CODES),
            Range(column="x", low=-5000.0, high=5000.0),
            Range(column="y", low=-5000.0, high=5000.0),
            Range(column="source_gap_m", low=0.0, high=500.0),
        ],
        parents=("telemetry_aligned",),
    ),
    # --- F009 segmentation ----------------------------------------------------
    "events": _contract(
        "events", ("event_id",),
        [
            NotNull(check_columns=("event_id", "apex_m", "v_min_kmh", "prominence_kmh", "lift_m",
                                   "apex_start_m", "apex_end_m", "exit_end_m", "has_braking", "marginal")),
            NotNull(check_columns=("brake_on_m", "brake_off_m"), unless=p.event_has_no_braking),
            Unique(key=("event_id",)),
            Invariant(name="event_boundaries_are_ordered", check=p.event_boundaries_are_ordered,
                      about=("lift_m", "apex_start_m", "apex_end_m", "exit_end_m")),
            Range(column="v_min_kmh", low=0.0, high=400.0),
            Range(column="prominence_kmh", low=0.0, high=400.0),
            Range(column="apex_m", low=0.0, high=100000.0),
        ],
    ),
    "microsectors": _contract(
        "microsectors", ("grain", "microsector_id"),
        [
            NotNull(check_columns=("grain", "microsector_id", "phase", "start_m", "end_m",
                                   "start_index", "end_index", "marginal")),
            NotNull(check_columns=("event_id", "corners"), unless=p.sector_is_not_an_event),
            Unique(key=("grain", "microsector_id")),
            ForeignKey(key=("event_id",), parent="events", nullable=True),
            AllowedValues(column="grain", values=GRAINS),
            AllowedValues(column="phase", values=PHASES),
            Invariant(name="microsectors_partition_the_lap", check=p.microsectors_partition_the_lap,
                      about=("start_m", "end_m")),
            Invariant(name="sector_indices_match_distances", check=p.sector_indices_match_distances,
                      about=("start_index", "end_index")),
            Range(column="start_m", low=0.0, high=100000.0),
            Range(column="end_m", low=0.0, high=100000.0),
        ],
        parents=("events",),
    ),
    "corners_aligned": _contract(
        "corners_aligned", ("number",),
        [
            NotNull(check_columns=("number", "distance_m", "line_offset_m", "raw_distance_m")),
            Unique(key=("number", "letter")),
            ForeignKey(key=("event_id",), parent="events", nullable=True),
            Range(column="distance_m", low=0.0, high=100000.0),
            Range(column="line_offset_m", low=0.0, high=100.0),
        ],
        parents=("events",),
    ),
    # --- F004 metrics ---------------------------------------------------------
    "delta_t": _contract(
        "delta_t", _GRID_KEY,
        [
            NotNull(check_columns=("driver", "lap_number", "grid_index", "t_s", "reference", "reference_kind")),
            # Null beyond the reference lap's last point: the rows cannot be
            # identified from this frame alone, so a documented sliver is allowed.
            NotNull(check_columns=("delta_t_s",), max_fraction=0.005),
            Unique(key=_GRID_KEY),
            ForeignKey(key=_GRID_KEY, parent="grid"),
            AllowedValues(column="reference_kind", values=REFERENCE_KINDS),
            Invariant(name="delta_is_zero_at_the_line", check=p.delta_is_zero_at_the_line,
                      about=("grid_index", "delta_t_s")),
            Range(column="t_s", low=0.0, high=300.0),
            Range(column="delta_t_s", low=-120.0, high=120.0),
        ],
        parents=("grid",),
    ),
    "microsector_times": _contract(
        "microsector_times", ("driver", "lap_number", "grain", "microsector_id"),
        [
            NotNull(check_columns=("driver", "lap_number", "grain", "microsector_id", "partial", "length_m")),
            NotNull(check_columns=("time_s", "delta_s"), unless=p.row_is_partial),
            Unique(key=("driver", "lap_number", "grain", "microsector_id")),
            ForeignKey(key=("grain", "microsector_id"), parent="microsectors"),
            ForeignKey(key=_LAP_KEY, parent="grid"),
            AllowedValues(column="grain", values=GRAINS),
            Range(column="time_s", low=0.0, high=300.0),
            Range(column="delta_s", low=-120.0, high=120.0),
            Range(column="length_m", low=0.0, high=100000.0),
        ],
        parents=("microsectors", "grid"),
    ),
    "microsector_summary": _contract(
        "microsector_summary", ("grain", "microsector_id"),
        [
            NotNull(check_columns=("grain", "microsector_id", "phase", "length_m", "n_laps")),
            NotNull(check_columns=("event_id", "corners"), unless=p.sector_is_not_an_event),
            NotNull(check_columns=("mean_s", "std_s", "min_s", "p10_s"), unless=p.sector_has_no_laps),
            Unique(key=("grain", "microsector_id")),
            ForeignKey(key=("grain", "microsector_id"), parent="microsectors"),
            AllowedValues(column="grain", values=GRAINS),
            AllowedValues(column="phase", values=PHASES),
            Range(column="n_laps", low=0, high=10000),
            Range(column="mean_s", low=0.0, high=300.0),
            Range(column="std_s", low=0.0, high=60.0),
        ],
        parents=("microsectors",),
    ),
    "corner_metrics": _contract(
        "corner_metrics", ("driver", "lap_number", "event_id"),
        [
            NotNull(check_columns=("driver", "lap_number", "event_id", "v_min_kmh", "v_min_m", "apex_dev_m")),
            # A braked corner where this lap has no brake sample in the shared
            # window: real, rare (7 of 444 at Suzuka), and reported, not fatal.
            NotNull(check_columns=("brake_on_m", "brake_dev_m"), unless=p.corner_event_has_no_braking,
                    severity=WARNING),
            Unique(key=("driver", "lap_number", "event_id")),
            ForeignKey(key=("event_id",), parent="events"),
            ForeignKey(key=_LAP_KEY, parent="grid"),
            Range(column="v_min_kmh", low=0.0, high=400.0),
            Range(column="v_min_m", low=0.0, high=100000.0),
            Range(column="apex_dev_m", low=-500.0, high=500.0),
            Range(column="brake_dev_m", low=-1000.0, high=1000.0),
            Range(column="brake_gap_m", low=0.0, high=500.0),
        ],
        parents=("events", "grid"),
    ),
    "lap_summary": _contract(
        "lap_summary", _LAP_KEY,
        [
            NotNull(check_columns=("driver", "lap_number", "lap_time_s", "grid_time_s", "n_points",
                                   "delta_t_end_s", "official_delta_s", "residual_s", "reference", "is_reference")),
            Unique(key=_LAP_KEY),
            ForeignKey(key=_LAP_KEY, parent="grid"),
            Range(column="lap_time_s", low=20.0, high=600.0),
            Range(column="grid_time_s", low=20.0, high=600.0),
            Range(column="n_points", low=2, high=10000),
            Range(column="residual_s", low=-10.0, high=10.0),
        ],
        parents=("grid",),
    ),
    "ground_truth": _contract(
        "ground_truth", _LAP_KEY,
        [
            NotNull(check_columns=("driver", "lap_number", "lap_time_s", "lap_grid_s", "lap_residual_s",
                                   "driven_m", "line_start_m", "line_end_m", "window_open_s", "window_close_s",
                                   "start_coverage_poor", "end_coverage_poor", "is_reference")),
            NotNull(check_columns=("closure_residual_s",), unless=p.is_the_reference_lap),
            Unique(key=_LAP_KEY),
            ForeignKey(key=_LAP_KEY, parent="grid"),
            Range(column="lap_time_s", low=20.0, high=600.0),
            Range(column="lap_grid_s", low=20.0, high=600.0),
            Range(column="lap_residual_s", low=-10.0, high=10.0),
            Range(column="closure_residual_s", low=-10.0, high=10.0),
            Range(column="driven_m", low=0.0, high=100000.0),
            Range(column="driven_pct_of_official", low=-10.0, high=10.0),
        ],
        parents=("grid",),
    ),
    "v_min_stability": _contract(
        "v_min_stability", ("driver", "compound", "event_id"),
        [
            NotNull(check_columns=("driver", "compound", "event_id", "n_laps", "v_min_mean_kmh", "v_min_std_kmh")),
            Unique(key=("driver", "compound", "event_id")),
            ForeignKey(key=("event_id",), parent="events"),
            Range(column="n_laps", low=1, high=10000),
            Range(column="v_min_mean_kmh", low=0.0, high=400.0),
            Range(column="v_min_std_kmh", low=0.0, high=200.0),
        ],
        parents=("events",),
    ),
}

# --- F012 reference data (per season, not per session) ------------------------
REFERENCE_CONTRACTS: dict[str, TableContract] = {
    "dim_event": _contract(
        "dim_event", ("season", "round"),
        [
            NotNull(check_columns=("season", "round", "event_name", "circuit_id", "circuit_name",
                                   "country", "lat", "lon", "race_date")),
            Unique(key=("season", "round")),
            Range(column="round", low=1, high=30),
            Range(column="lat", low=-90.0, high=90.0),
            Range(column="lon", low=-180.0, high=180.0),
        ],
    ),
    "dim_session_schedule": _contract(
        "dim_session_schedule", ("season", "round", "session"),
        [
            NotNull(check_columns=("season", "round", "session", "session_start_utc", "has_time")),
            Unique(key=("season", "round", "session")),
            ForeignKey(key=("season", "round"), parent="dim_event"),
            AllowedValues(column="session", values=SESSION_CODES),
            Range(column="round", low=1, high=30),
        ],
        parents=("dim_event",),
    ),
    "dim_driver": _contract(
        "dim_driver", ("season", "code"),
        [
            NotNull(check_columns=("season", "code", "driver_id", "given_name", "family_name", "full_name")),
            Unique(key=("season", "code")),
            Range(column="permanent_number", low=0, high=99),
        ],
    ),
    "dim_constructor": _contract(
        "dim_constructor", ("season", "constructor_id"),
        [
            NotNull(check_columns=("season", "constructor_id", "name")),
            Unique(key=("season", "constructor_id")),
        ],
    ),
    "driver_entry": _contract(
        "driver_entry", ("season", "round", "code"),
        [
            NotNull(check_columns=("season", "round", "code", "driver_id", "constructor_id", "car_number")),
            Unique(key=("season", "round", "code")),
            ForeignKey(key=("season", "round"), parent="dim_event"),
            ForeignKey(key=("season", "code"), parent="dim_driver"),
            ForeignKey(key=("season", "constructor_id"), parent="dim_constructor"),
            Range(column="car_number", low=0, high=99),
        ],
        parents=("dim_event", "dim_driver", "dim_constructor"),
    ),
}

CONTRACTS.update(REFERENCE_CONTRACTS)

#: Reference data is ingested per season, so a per-session check will not see it.
SESSION_TABLES = frozenset(CONTRACTS) - frozenset(REFERENCE_CONTRACTS)

#: Artefacts a session may legitimately not have produced.
OPTIONAL_TABLES = frozenset({"weather", "corners_aligned", "v_min_stability", "ground_truth"})

#: Columns whose observed range is reported next to the contract bound, so
#: drift is visible before it becomes a failure (F015).
REPORTED_RANGES = {
    "car_telemetry": ("speed", "throttle", "rpm"),
    "grid": ("speed", "elapsed_time", "source_gap_m"),
    "telemetry_aligned": ("distance_aligned", "line_offset_m"),
    "lap_summary": ("lap_time_s", "residual_s"),
    "ground_truth": ("lap_residual_s", "closure_residual_s", "driven_pct_of_official"),
    "corner_metrics": ("v_min_kmh", "brake_gap_m"),
}
