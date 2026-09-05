"""A designed session for the F009 tests: every boundary is known in advance.

Lap length 3000 m on a 10 m grid. Three speed dips:

* **A at 800 m** -- braked. Speed falls 300 -> 100 between 640 and 800 m,
  floor to 830 m, recovers to 300 by 1000 m. Brake on 640..780 m, throttle
  closed 640..839 m. Expected: braking [640, 790), entry [790, 800),
  apex [800, 840), exit [840, 920); prominence 200.
* **B at 1800 m** -- lift only, marginal. Speed 300 -> 286 over 1700..1800 m,
  back to 300 by 1930 m. Throttle 60 % on 1720..1869 m, never braked.
  Expected: entry [1720, 1800), apex [1800, 1810), exit [1810, 1870);
  prominence 14.
* **C at 2400 m** -- 5 km/h dip, below the 8 km/h threshold. Not an event.

Corners T1, T2, T3 sit at 800, 1800, 2400 m.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.grid.resample import GRID_SCHEMA

GRID_M = 10.0
LAP_LENGTH_M = 3000.0
N_POINTS = int(LAP_LENGTH_M / GRID_M)
CORNER_DISTANCES = {1: 800.0, 2: 1800.0, 3: 2400.0}
FRAME_ROTATION_DEG = 0.01
FRAME_TRANSLATION_M = (8.7, -29.5)
REFERENCE_LAP = ("AAA", 1)

EXPECTED_PHASES = [
    ("straight", 0.0, 640.0, None),
    ("braking", 640.0, 790.0, 0),
    ("entry", 790.0, 800.0, 0),
    ("apex", 800.0, 840.0, 0),
    ("exit", 840.0, 920.0, 0),
    ("straight", 920.0, 1720.0, None),
    ("entry", 1720.0, 1800.0, 1),
    ("apex", 1800.0, 1810.0, 1),
    ("exit", 1810.0, 1870.0, 1),
    ("straight", 1870.0, 3000.0, None),
]


def distance() -> np.ndarray:
    return np.arange(N_POINTS, dtype=float) * GRID_M


def speed_profile(d: np.ndarray | None = None) -> np.ndarray:
    d = distance() if d is None else d
    v = np.full(len(d), 300.0)
    a_fall = (d >= 640) & (d <= 800)
    v[a_fall] = 300.0 - 200.0 * (d[a_fall] - 640.0) / 160.0
    v[(d > 800) & (d <= 830)] = 100.0
    a_rise = (d > 830) & (d <= 1000)
    v[a_rise] = 100.0 + 200.0 * (d[a_rise] - 830.0) / 170.0
    b_fall = (d >= 1700) & (d <= 1800)
    v[b_fall] = 300.0 - 14.0 * (d[b_fall] - 1700.0) / 100.0
    b_rise = (d > 1800) & (d <= 1930)
    v[b_rise] = 286.0 + 14.0 * (d[b_rise] - 1800.0) / 130.0
    c_fall = (d >= 2350) & (d <= 2400)
    v[c_fall] = 300.0 - 5.0 * (d[c_fall] - 2350.0) / 50.0
    c_rise = (d > 2400) & (d <= 2450)
    v[c_rise] = 295.0 + 5.0 * (d[c_rise] - 2400.0) / 50.0
    return v


def brake_profile(d: np.ndarray | None = None, delay_m: float = 0.0) -> np.ndarray:
    d = distance() if d is None else d
    return (d >= 640 + delay_m) & (d <= 780)


def throttle_profile(d: np.ndarray | None = None) -> np.ndarray:
    d = distance() if d is None else d
    t = np.full(len(d), 100.0)
    t[(d >= 640) & (d < 840)] = 0.0
    t[(d >= 1720) & (d < 1870)] = 60.0
    return t


def traces() -> pd.DataFrame:
    """Median traces exactly as ``median_traces`` would return for a clean session."""
    d = distance()
    return pd.DataFrame(
        {
            "distance_m": d,
            "speed": speed_profile(d),
            "brake": brake_profile(d).astype(float),
            "throttle": throttle_profile(d),
            "n_laps": 4,
        },
        index=pd.Index(np.arange(N_POINTS), name="grid_index"),
    )


def rolled_traces(shift_points: int) -> pd.DataFrame:
    """The same session with the start/finish line moved by ``shift_points`` bins."""
    t = traces()
    out = t.copy()
    for column in ("speed", "brake", "throttle"):
        out[column] = np.roll(t[column].to_numpy(), shift_points)
    return out


def _circle_xy(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = LAP_LENGTH_M / (2 * np.pi)
    angle = 2 * np.pi * d / LAP_LENGTH_M
    return radius * np.cos(angle), radius * np.sin(angle)


def grid(drivers: tuple[str, ...] = ("AAA", "BBB"), laps_per_driver: int = 2, late_braker: str = "BBB") -> pd.DataFrame:
    """A grid frame per F003's contract. ``late_braker`` brakes one bin later."""
    d = distance()
    x, y = _circle_xy(d)
    frames = []
    for driver in drivers:
        for lap in range(1, laps_per_driver + 1):
            v = speed_profile(d) + 0.5 * np.sin(d / 97.0 + lap)  # tiny, trough positions unchanged
            dt = GRID_M / (v / 3.6)
            frames.append(
                pd.DataFrame(
                    {
                        "driver": driver,
                        "lap_number": lap,
                        "grid_index": np.arange(N_POINTS),
                        "distance_m": d,
                        "elapsed_time": np.concatenate([[0.0], np.cumsum(dt[:-1])]),
                        "speed": v,
                        "throttle": throttle_profile(d),
                        "rpm": 10000.0,
                        "x": x,
                        "y": y,
                        "n_gear": 7,
                        "drs": 0,
                        "brake": brake_profile(d, delay_m=GRID_M if driver == late_braker else 0.0),
                        "source_gap_m": 8.0,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True).astype(GRID_SCHEMA)


#: Marshalling-sector lines on the synthetic axis, used by the F004 checks.
S1_M = 1000.0
S2_M = 2000.0
#: Official lap time is grid time plus this constant; S1 carries an extra offset.
LAP_TIME_OFFSET_S = 0.2
S1_OFFSET_S = 0.1
#: Session clock at each lap's first sample.
SESSION_T0 = 1000.0
#: Identity of the designed session, shaped like a real F002 snapshot's
#: session_meta.json. Round 1 so it never collides with Suzuka (round 4).
SESSION_META = {
    "season": 2024,
    "event_requested": "Synthetic",
    "session_requested": "Q",
    "session_name": "Qualifying",
    "session_date": "2024-03-02 06:00:00",
    "event_name": "Synthetic Grand Prix",
    "country": "Nowhere",
    "location": "Synthetica",
    "round_number": 1,
}
#: Official circuit length: a little longer than the driven path, as in reality.
OFFICIAL_LENGTH_M = 3020.0


def aligned_telemetry(grid_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """F008-shaped aligned telemetry for the designed session.

    The synthetic path *is* the reference line, so `distance_raw` and
    `distance_aligned` coincide; `session_time` runs from `SESSION_T0`.
    """
    g = grid() if grid_frame is None else grid_frame
    out = g[["driver", "lap_number", "speed", "throttle", "rpm", "n_gear", "brake", "drs", "x", "y"]].copy()
    out["session_time"] = SESSION_T0 + g["elapsed_time"].astype(float)
    out["distance_raw"] = g["distance_m"].astype(float)
    out["distance_aligned"] = g["distance_m"].astype(float)
    out["line_offset_m"] = 0.0
    out["z"] = 0.0
    return out


def frame_meta() -> dict:
    return {
        "reference_line_lap": f"{REFERENCE_LAP[0]} L{REFERENCE_LAP[1]}",
        "frame": {
            "rotation_deg": FRAME_ROTATION_DEG,
            "translation_m": list(FRAME_TRANSLATION_M),
            "median_residual_m": 0.5,
            "iterations": 5,
        },
        "reference_line_length_m": LAP_LENGTH_M,
        "official_lap_length_m": OFFICIAL_LENGTH_M,
        "sector_consistency": [
            {"boundary": "S1", "n_laps": 4, "median_m": S1_M, "p95_dev_m": 0.0, "max_dev_m": 0.0, "std_m": 0.0},
            {"boundary": "S2", "n_laps": 4, "median_m": S2_M, "p95_dev_m": 0.0, "max_dev_m": 0.0, "std_m": 0.0},
        ],
    }


def official_laps(grid_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """A ``laps.parquet``-shaped table whose official times derive from the grid.

    ``lap_time = grid_time + LAP_TIME_OFFSET_S`` (a constant window offset, as
    in the real data), ``sector2_time`` is exactly the grid time between the
    S1 and S2 lines, and ``sector1_time`` carries an extra ``S1_OFFSET_S``.

    ``lap_start_time`` puts the timing line half the offset before the first
    sample, so the line sits symmetrically outside the axis at both ends: F004,
    whose curves start at grid 0, sees the S1 offset; F010, which starts at the
    line itself, sees no residual at all.
    """
    g = grid() if grid_frame is None else grid_frame
    i1, i2 = int(S1_M / GRID_M), int(S2_M / GRID_M)
    rows = []
    for (driver, lap_number), lap in g.groupby(["driver", "lap_number"], observed=True):
        t = lap.sort_values("grid_index")["elapsed_time"].to_numpy(dtype=float)
        t = t - t[0]
        lap_time = float(t[-1]) + LAP_TIME_OFFSET_S
        s1 = float(t[i1]) + S1_OFFSET_S
        s2 = float(t[i2] - t[i1])
        rows.append(
            dict(driver=str(driver), driver_number="1", team="Synthetic", lap_number=int(lap_number),
                 lap_time=lap_time, sector1_time=s1, sector2_time=s2, sector3_time=lap_time - s1 - s2,
                 lap_start_time=SESSION_T0 - LAP_TIME_OFFSET_S / 2, pit_in_time=np.nan, pit_out_time=np.nan,
                 stint=1, compound="SOFT",
                 tyre_life=int(lap_number), fresh_tyre=True, track_status="1", is_accurate=True)
        )
    frame = pd.DataFrame(rows)
    return frame.astype(
        {"driver": "string", "driver_number": "string", "team": "string", "lap_number": "Int16",
         "stint": "Int16", "compound": "string", "tyre_life": "Int16", "fresh_tyre": "boolean",
         "track_status": "string", "is_accurate": "boolean"}
    )


def raw_channels(grid_frame: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Raw car telemetry, position data and weather, consistent with the grid.

    The synthetic session is built grid-first, so these are derived back out of
    it: positions return to FastF1's 1/10 m units, and every lap key matches
    `official_laps` so the referential rules have something true to check.
    """
    g = grid() if grid_frame is None else grid_frame
    time_s = SESSION_T0 + g["elapsed_time"].astype(float)

    car = pd.DataFrame(
        {
            "driver": g["driver"].astype("string"),
            "lap_number": g["lap_number"].astype("Int16"),
            "session_time": time_s,
            "speed": g["speed"].astype("float32"),
            "throttle": g["throttle"].astype("float32"),
            "brake": g["brake"].astype("boolean"),
            "rpm": g["rpm"].astype("float32"),
            "n_gear": g["n_gear"].astype("Int8"),
            "drs": g["drs"].astype("Int8"),
            "source": pd.array(["car"] * len(g), dtype="string"),
        }
    )
    pos = pd.DataFrame(
        {
            "driver": g["driver"].astype("string"),
            "lap_number": g["lap_number"].astype("Int16"),
            "session_time": time_s,
            "x": (g["x"].astype(float) * 10).astype("float32"),
            "y": (g["y"].astype(float) * 10).astype("float32"),
            "z": pd.array([0.0] * len(g), dtype="float32"),
            "status": pd.array(["OnTrack"] * len(g), dtype="string"),
            "source": pd.array(["pos"] * len(g), dtype="string"),
        }
    )
    weather = pd.DataFrame(
        {
            "session_time": np.arange(5, dtype=float) * 60.0 + SESSION_T0,
            "air_temp": np.float32(20.0),
            "track_temp": np.float32(30.0),
            "humidity": np.float32(50.0),
            "pressure": np.float32(1013.0),
            "wind_speed": np.float32(1.0),
            "wind_direction": pd.array([180] * 5, dtype="Int16"),
            "rainfall": pd.array([False] * 5, dtype="boolean"),
        }
    )
    return car, pos, weather


def write_full_session(root: Path) -> dict[str, Path]:
    """Grid, snapshot (corners + laps), aligned meta and F009 micro-sectors under ``root``."""
    from src.segment.session import segment_session

    grid_root, snapshot_root, aligned_root = write_session(root)
    official_laps().to_parquet(snapshot_root / "laps.parquet", index=False)
    car, pos, weather = raw_channels()
    car.to_parquet(snapshot_root / "car_telemetry.parquet", index=False)
    pos.to_parquet(snapshot_root / "pos_data.parquet", index=False)
    weather.to_parquet(snapshot_root / "weather.parquet", index=False)
    microsector_root = root / "microsectors" / "synthetic"
    segment_session(grid_root, snapshot_root, aligned_root, out_root=microsector_root)
    return {
        "grid_root": grid_root,
        "snapshot_root": snapshot_root,
        "aligned_root": aligned_root,
        "microsector_root": microsector_root,
    }


def raw_corners() -> pd.DataFrame:
    """Corners as F002 stores them: 1/10 m units, in the un-calibrated frame."""
    numbers = np.array(list(CORNER_DISTANCES), dtype=int)
    d = np.array(list(CORNER_DISTANCES.values()), dtype=float)
    x, y = _circle_xy(d)
    xy = np.column_stack([x, y]) - np.array(FRAME_TRANSLATION_M)
    a = -np.deg2rad(FRAME_ROTATION_DEG)
    inverse = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    xy = xy @ inverse.T
    return pd.DataFrame(
        {
            "number": pd.array(numbers, dtype="Int16"),
            "letter": pd.array(["", "", ""], dtype="string"),
            "x": (xy[:, 0] * 10).astype(np.float32),
            "y": (xy[:, 1] * 10).astype(np.float32),
            "angle": np.float32(0.0),
            "distance": (d - 30.0).astype(np.float32),
        }
    )


def write_session(root: Path) -> tuple[Path, Path, Path]:
    """Write grid, snapshot and aligned roots under ``root``; return the three paths."""
    grid_root = root / "grid" / "synthetic"
    snapshot_root = root / "raw" / "synthetic"
    aligned_root = root / "aligned" / "synthetic"
    for path in (grid_root, snapshot_root, aligned_root):
        path.mkdir(parents=True, exist_ok=True)
    grid().to_parquet(grid_root / "grid.parquet", index=False)
    aligned_telemetry().to_parquet(aligned_root / "telemetry_aligned.parquet", index=False)
    (snapshot_root / "session_meta.json").write_text(json.dumps(SESSION_META, indent=2), encoding="utf-8")
    raw_corners().to_parquet(snapshot_root / "circuit_corners.parquet", index=False)
    (aligned_root / "alignment_meta.json").write_text(json.dumps(frame_meta()), encoding="utf-8")
    return grid_root, snapshot_root, aligned_root
