"""Ingestion behaviour, exercised without touching the network.

FastF1 itself is replaced by a stand-in; what is under test is our contract:
cache-before-load ordering, backoff, per-driver fault tolerance, schema and
uniqueness gates, manifest content, and snapshot immutability.
"""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from src.ingest import fastf1_source as mod
from src.ingest.retry import PermanentIngestError

DRIVERS = ["VER", "NOR", "LEC"]
LAPS_PER_DRIVER = 2
SAMPLES_PER_LAP = 5


# ---------------------------------------------------------------- fake data


def _laps_frame(drivers=DRIVERS) -> pd.DataFrame:
    rows = []
    for number, driver in enumerate(drivers, start=1):
        for lap in range(1, LAPS_PER_DRIVER + 1):
            rows.append(
                {
                    "Driver": driver,
                    "DriverNumber": str(number),
                    "Team": f"Team {driver}",
                    "LapNumber": float(lap),
                    "LapTime": pd.Timedelta(seconds=89.0 + lap),
                    "Sector1Time": pd.Timedelta(seconds=30.0),
                    "Sector2Time": pd.Timedelta(seconds=29.0),
                    "Sector3Time": pd.Timedelta(seconds=30.0 + lap),
                    "LapStartTime": pd.Timedelta(seconds=100.0 * lap),
                    "PitInTime": pd.NaT,
                    "PitOutTime": pd.NaT,
                    "Stint": 1.0,
                    "Compound": "SOFT",
                    "TyreLife": float(lap),
                    "FreshTyre": True,
                    "TrackStatus": "1",
                    "IsAccurate": True,
                }
            )
    return pd.DataFrame(rows)


def _car_frame(driver: str, lap: int) -> pd.DataFrame:
    base = 100.0 * lap
    return pd.DataFrame(
        {
            "SessionTime": pd.to_timedelta(
                [base + i for i in range(SAMPLES_PER_LAP)], unit="s"
            ),
            "Speed": [200.0 + i for i in range(SAMPLES_PER_LAP)],
            "Throttle": [100.0] * SAMPLES_PER_LAP,
            "Brake": [False, False, True, True, False],
            "RPM": [11000.0] * SAMPLES_PER_LAP,
            "nGear": [7] * SAMPLES_PER_LAP,
            "DRS": [0] * SAMPLES_PER_LAP,
            "Source": ["car"] * SAMPLES_PER_LAP,
        }
    )


def _pos_frame(driver: str, lap: int) -> pd.DataFrame:
    base = 100.0 * lap
    return pd.DataFrame(
        {
            "SessionTime": pd.to_timedelta(
                [base + i for i in range(SAMPLES_PER_LAP)], unit="s"
            ),
            "X": [float(i) for i in range(SAMPLES_PER_LAP)],
            "Y": [float(i * 2) for i in range(SAMPLES_PER_LAP)],
            "Z": [0.0] * SAMPLES_PER_LAP,
            "Status": ["OnTrack"] * SAMPLES_PER_LAP,
            "Source": ["pos"] * SAMPLES_PER_LAP,
        }
    )


def _weather_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": pd.to_timedelta([0.0, 60.0], unit="s"),
            "AirTemp": [22.0, 22.5],
            "TrackTemp": [35.0, 35.5],
            "Humidity": [40.0, 41.0],
            "Pressure": [1010.0, 1010.5],
            "WindSpeed": [1.2, 1.4],
            "WindDirection": [180, 190],
            "Rainfall": [False, False],
        }
    )


def _corners_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Number": [1, 2, 3],
            "Letter": ["", "", ""],
            "X": [0.0, 100.0, 200.0],
            "Y": [0.0, 50.0, 0.0],
            "Angle": [90.0, 45.0, 120.0],
            "Distance": [300.0, 900.0, 1500.0],
        }
    )


class _FakeLap:
    """Minimal stand-in for a fastf1 Lap."""

    def __init__(self, driver: str, lap_number: int, failures: set) -> None:
        self._driver = driver
        self._lap_number = lap_number
        self._failures = failures

    def __getitem__(self, key: str):
        if key == "LapNumber":
            return self._lap_number
        if key == "Driver":
            return self._driver
        raise KeyError(key)

    def get_car_data(self) -> pd.DataFrame:
        if self._driver in self._failures or (self._driver, self._lap_number) in self._failures:
            raise ValueError(f"no telemetry for {self._driver} lap {self._lap_number}")
        return _car_frame(self._driver, self._lap_number)

    def get_pos_data(self) -> pd.DataFrame:
        return _pos_frame(self._driver, self._lap_number)


class _FakeSession:
    def __init__(self, laps: pd.DataFrame) -> None:
        self.laps = laps
        self.weather_data = _weather_frame()
        self.name = "Qualifying"
        self.date = "2024-04-06"
        self.event = {
            "EventName": "Japanese Grand Prix",
            "Country": "Japan",
            "Location": "Suzuka",
            "RoundNumber": 4,
        }
        self.load_calls: list[dict] = []

    def load(self, **kwargs) -> None:
        self.load_calls.append(kwargs)

    def get_circuit_info(self):
        return SimpleNamespace(corners=_corners_frame())


# ----------------------------------------------------------------- fixtures


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Wire the module to a fake FastF1 and a temporary data root."""
    calls: list[str] = []
    failures: set = set()
    session = _FakeSession(_laps_frame())

    def fake_enable_cache(path):
        calls.append("enable_cache")

    def fake_get_session(season, event, session_id):
        calls.append("get_session")
        return session

    fake_fastf1 = SimpleNamespace(
        Cache=SimpleNamespace(enable_cache=fake_enable_cache),
        get_session=fake_get_session,
        __version__="3.8.3-fake",
    )

    def fake_iter(driver_laps):
        for _, row in driver_laps.iterrows():
            yield _FakeLap(row["Driver"], int(row["LapNumber"]), failures)

    monkeypatch.setattr(mod, "fastf1", fake_fastf1)
    monkeypatch.setattr(mod, "FASTF1_RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(mod, "FASTF1_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(mod, "_iter_driver_laps", fake_iter)

    return SimpleNamespace(
        calls=calls,
        failures=failures,
        session=session,
        raw_root=tmp_path / "raw",
        fastf1=fake_fastf1,
        monkeypatch=monkeypatch,
    )


def _ingest(**kwargs):
    params = dict(season=2024, event="Japan", session="Q", snapshot_date=date(2026, 9, 1))
    params.update(kwargs)
    return mod.ingest_session(**params)


# -------------------------------------------------------------------- tests


class TestCacheDiscipline:
    def test_cache_is_enabled_before_the_session_is_fetched(self, env) -> None:
        """Project CLAUDE.md: cache enabled before any session load, no exceptions."""
        _ingest()
        assert env.calls[:2] == ["enable_cache", "get_session"]

    def test_session_is_loaded_with_the_channels_we_need(self, env) -> None:
        _ingest()
        assert env.session.load_calls == [
            {"laps": True, "telemetry": True, "weather": True, "messages": False}
        ]


class TestArtefacts:
    def test_writes_every_declared_artefact(self, env) -> None:
        snapshot = _ingest()
        expected = {
            "laps.parquet",
            "car_telemetry.parquet",
            "pos_data.parquet",
            "weather.parquet",
            "circuit_corners.parquet",
            "session_meta.json",
            "manifest.json",
        }
        assert {p.name for p in snapshot.root.iterdir()} == expected

    def test_laps_business_key_is_unique(self, env) -> None:
        snapshot = _ingest()
        laps = pd.read_parquet(snapshot.path("laps"))
        assert not laps.duplicated(subset=["driver", "lap_number"]).any()
        assert len(laps) == len(DRIVERS) * LAPS_PER_DRIVER

    def test_telemetry_references_only_real_laps(self, env) -> None:
        snapshot = _ingest()
        laps = pd.read_parquet(snapshot.path("laps"))
        car = pd.read_parquet(snapshot.path("car_telemetry"))

        lap_keys = set(zip(laps["driver"], laps["lap_number"]))
        tel_keys = set(zip(car["driver"], car["lap_number"]))
        assert tel_keys <= lap_keys

    def test_brake_is_stored_as_boolean(self, env) -> None:
        snapshot = _ingest()
        car = pd.read_parquet(snapshot.path("car_telemetry"))
        assert car["brake"].dtype == "boolean"

    def test_no_critical_column_is_entirely_null(self, env) -> None:
        snapshot = _ingest()
        for artefact, columns in (
            ("laps", ["driver", "lap_number"]),
            ("car_telemetry", ["driver", "lap_number", "session_time", "speed"]),
            ("pos_data", ["driver", "lap_number", "x", "y"]),
            ("circuit_corners", ["number", "x", "y", "distance"]),
        ):
            frame = pd.read_parquet(snapshot.root / f"{artefact}.parquet")
            for column in columns:
                assert not frame[column].isna().all(), f"{artefact}.{column} all null"


class TestFaultTolerance:
    def test_one_bad_driver_is_skipped_and_the_rest_land(self, env) -> None:
        env.failures.add("NOR")
        snapshot = _ingest()

        assert snapshot.drivers_ingested == ["LEC", "VER"]
        assert "NOR" in snapshot.drivers_skipped
        assert "no lap yielded usable telemetry" in snapshot.drivers_skipped["NOR"]

        car = pd.read_parquet(snapshot.path("car_telemetry"))
        assert set(car["driver"]) == {"LEC", "VER"}

    def test_one_bad_lap_does_not_lose_the_driver(self, env) -> None:
        env.failures.add(("VER", 1))
        snapshot = _ingest()

        assert "VER" in snapshot.drivers_ingested
        car = pd.read_parquet(snapshot.path("car_telemetry"))
        ver_laps = set(car.loc[car["driver"] == "VER", "lap_number"])
        assert ver_laps == {2}

    def test_all_drivers_failing_is_an_error(self, env) -> None:
        env.failures.update(DRIVERS)
        with pytest.raises(PermanentIngestError, match="every requested driver failed"):
            _ingest()

    def test_failed_ingest_leaves_no_partial_snapshot(self, env) -> None:
        env.failures.update(DRIVERS)
        with pytest.raises(PermanentIngestError):
            _ingest()
        leftovers = list(env.raw_root.rglob("*.partial")) if env.raw_root.exists() else []
        assert leftovers == []


class TestIdempotency:
    def test_second_call_makes_no_network_request(self, env) -> None:
        first = _ingest()

        def explode(*args, **kwargs):
            raise AssertionError("network was hit on a re-ingest")

        env.monkeypatch.setattr(env.fastf1, "get_session", explode)

        second = _ingest()
        assert second.root == first.root
        assert second.drivers_ingested == first.drivers_ingested

    def test_reingest_does_not_rewrite_files(self, env) -> None:
        first = _ingest()
        before = {p.name: p.stat().st_mtime_ns for p in first.root.iterdir()}
        _ingest()
        after = {p.name: p.stat().st_mtime_ns for p in first.root.iterdir()}
        assert before == after

    def test_tampered_snapshot_is_refused(self, env) -> None:
        """Raw snapshots are immutable; silent repair would hide corruption."""
        snapshot = _ingest()
        target = snapshot.path("laps")
        target.write_bytes(target.read_bytes() + b"tampered")

        with pytest.raises(mod.SnapshotCorruptError, match="checksum mismatch"):
            _ingest()

    def test_missing_file_is_refused(self, env) -> None:
        snapshot = _ingest()
        snapshot.path("weather").unlink()
        with pytest.raises(mod.SnapshotCorruptError, match="which is missing"):
            _ingest()


class TestManifest:
    def _manifest(self, snapshot) -> dict:
        return json.loads((snapshot.root / "manifest.json").read_text(encoding="utf-8"))

    def test_records_pinned_versions(self, env) -> None:
        manifest = self._manifest(_ingest())
        assert manifest["versions"]["fastf1"] == "3.8.3-fake"
        assert manifest["versions"]["pandas"] == pd.__version__

    def test_records_a_checksum_and_size_per_file(self, env) -> None:
        snapshot = _ingest()
        manifest = self._manifest(snapshot)

        assert set(manifest["files"]) == {
            "laps.parquet",
            "car_telemetry.parquet",
            "pos_data.parquet",
            "weather.parquet",
            "circuit_corners.parquet",
            "session_meta.json",
        }
        for name, entry in manifest["files"].items():
            assert len(entry["sha256"]) == 64
            assert entry["bytes"] == (snapshot.root / name).stat().st_size

    def test_records_skips_with_reasons(self, env) -> None:
        env.failures.add("LEC")
        manifest = self._manifest(_ingest())

        # Discovered drivers are sorted for a deterministic manifest.
        assert manifest["drivers_requested"] == sorted(DRIVERS)
        assert manifest["drivers_ingested"] == ["NOR", "VER"]
        assert "LEC" in manifest["drivers_skipped"]

    def test_records_rows_per_driver(self, env) -> None:
        manifest = self._manifest(_ingest())
        assert manifest["telemetry_rows_per_driver"] == {
            driver: LAPS_PER_DRIVER * SAMPLES_PER_LAP for driver in sorted(DRIVERS)
        }


class TestDriverSelection:
    def test_explicit_driver_subset_is_honoured(self, env) -> None:
        snapshot = _ingest(drivers=["VER", "NOR"])
        assert snapshot.drivers_ingested == ["VER", "NOR"]

        laps = pd.read_parquet(snapshot.path("laps"))
        assert set(laps["driver"]) == {"VER", "NOR"}

    def test_unknown_driver_is_skipped_not_fatal(self, env) -> None:
        snapshot = _ingest(drivers=["VER", "XXX"])
        assert snapshot.drivers_ingested == ["VER"]
        assert "no laps in session" in snapshot.drivers_skipped["XXX"]


class TestEmptySession:
    def test_session_without_laps_is_an_error(self, env, monkeypatch) -> None:
        monkeypatch.setattr(env.session, "laps", _laps_frame().iloc[0:0])
        with pytest.raises(PermanentIngestError, match="no laps"):
            _ingest()


# --------------------------------------------------------- network (opt-in)


@pytest.mark.network
def test_real_suzuka_2024_qualifying(tmp_path, monkeypatch):
    """Acceptance criterion 1, against the live backend. Run with -m network."""
    monkeypatch.setattr(mod, "FASTF1_RAW_ROOT", tmp_path / "raw")
    snapshot = mod.ingest_session(2024, "Japan", "Q", snapshot_date=date(2026, 9, 1))

    assert len(snapshot.drivers_ingested) >= 15

    laps = pd.read_parquet(snapshot.path("laps"))
    car = pd.read_parquet(snapshot.path("car_telemetry"))
    corners = pd.read_parquet(snapshot.path("circuit_corners"))

    assert not laps.duplicated(subset=["driver", "lap_number"]).any()
    assert car["brake"].dtype == "boolean"
    assert len(corners) > 10
    assert set(zip(car["driver"], car["lap_number"])) <= set(
        zip(laps["driver"], laps["lap_number"])
    )
