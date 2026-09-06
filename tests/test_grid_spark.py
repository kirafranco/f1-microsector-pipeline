"""F013: the Spark executor distributes F003's maths without changing it.

Everything here but the `spark`-marked class runs with no JVM and no container:
the schema transcription, the UDF body and the path translation are ordinary
Python. The marked class needs the Connect server from the `pipeline` profile.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.grid import spark as mod
from src.grid.resample import GRID_SCHEMA, ResampleError, resample_lap
from src.grid.session import resample_session
from tests import synthetic_session as syn

KEY = ["driver", "lap_number", "grid_index"]


class TestSparkSchemaMatchesTheContract:
    """A column added to one and not the other must fail here, not at runtime."""

    def parsed(self) -> dict[str, str]:
        return {part.rsplit(" ", 1)[0]: part.rsplit(" ", 1)[1]
                for part in (piece.strip() for piece in mod.GRID_SPARK_SCHEMA.split(","))}

    def test_every_column_is_declared_in_order(self) -> None:
        assert list(self.parsed()) == list(GRID_SCHEMA)

    def test_the_types_are_the_pandas_types(self) -> None:
        expected = {
            "driver": "string", "lap_number": "smallint", "grid_index": "int",
            "distance_m": "float", "elapsed_time": "float", "speed": "float",
            "throttle": "float", "rpm": "float", "x": "float", "y": "float",
            "n_gear": "smallint", "drs": "tinyint", "brake": "boolean", "source_gap_m": "float",
        }
        assert self.parsed() == expected

    def test_gear_is_wide_enough_for_the_glitch_that_broke_ingest(self) -> None:
        """F015: one sample of the 2024 Japanese GP reads gear 128, past int8."""
        assert self.parsed()["n_gear"] == "smallint"

    def test_nothing_is_inferred(self) -> None:
        """Global CLAUDE.md 3.3: explicit schemas, never inference."""
        assert mod.GRID_SPARK_SCHEMA and len(self.parsed()) == len(GRID_SCHEMA)


class TestTheUdfIsTheSameMaths:
    def lap(self) -> pd.DataFrame:
        telemetry = syn.aligned_telemetry()
        first = telemetry.groupby(["driver", "lap_number"], observed=True).size().index[0]
        mask = (telemetry["driver"] == first[0]) & (telemetry["lap_number"] == first[1])
        return telemetry[mask].reset_index(drop=True)

    def test_it_equals_a_direct_call(self) -> None:
        lap = self.lap()
        direct = resample_lap(lap.sort_values("session_time").reset_index(drop=True))
        pd.testing.assert_frame_equal(mod._resample_group(lap), direct, check_exact=True)

    def test_it_sorts_by_time_first(self) -> None:
        """Spark hands a group in whatever order the partition had."""
        lap = self.lap()
        shuffled = lap.sample(frac=1.0, random_state=0).reset_index(drop=True)
        pd.testing.assert_frame_equal(mod._resample_group(shuffled), mod._resample_group(lap), check_exact=True)

    def test_a_lap_the_maths_refuses_yields_no_rows(self) -> None:
        """One bad lap must not fail the session (global CLAUDE.md 3.1)."""
        lap = self.lap().head(1)
        with pytest.raises(ResampleError):
            resample_lap(lap)
        out = mod._resample_group(lap)
        assert out.empty and list(out.columns) == list(GRID_SCHEMA)


class TestContainerPath:
    def test_a_project_path_is_rewritten_onto_the_container_root(self, tmp_path: Path) -> None:
        root = tmp_path / "project"
        (root / "data" / "interim").mkdir(parents=True)
        target = root / "data" / "interim" / "telemetry_aligned.parquet"
        target.touch()
        assert mod.container_path(target, project_root=root) == "/opt/project/data/interim/telemetry_aligned.parquet"

    def test_a_path_outside_the_project_is_an_error(self, tmp_path: Path) -> None:
        root = tmp_path / "project"
        root.mkdir()
        outside = tmp_path / "elsewhere.parquet"
        outside.touch()
        with pytest.raises(ValueError, match="outside the project root"):
            mod.container_path(outside, project_root=root)

    def test_it_uses_posix_separators(self, tmp_path: Path) -> None:
        root = tmp_path / "project"
        (root / "a" / "b").mkdir(parents=True)
        assert "\\" not in mod.container_path(root / "a" / "b", project_root=root)


class TestExecutorSelection:
    def setup_method(self) -> None:
        self.saved = os.environ.get("F1_GRID_EXECUTOR")

    def teardown_method(self) -> None:
        os.environ.pop("F1_GRID_EXECUTOR", None)
        if self.saved is not None:
            os.environ["F1_GRID_EXECUTOR"] = self.saved

    def test_pandas_is_the_default(self) -> None:
        from src.orchestration.stages import grid_executor

        os.environ.pop("F1_GRID_EXECUTOR", None)
        assert grid_executor() == "pandas"

    @pytest.mark.parametrize("value,expected", [("spark", "spark"), ("PANDAS", "pandas"), (" spark ", "spark"), ("", "pandas")])
    def test_it_reads_the_environment(self, value: str, expected: str) -> None:
        from src.orchestration.stages import grid_executor

        os.environ["F1_GRID_EXECUTOR"] = value
        assert grid_executor() == expected

    def test_an_unknown_executor_is_an_error_not_a_silent_fallback(self) -> None:
        from src.orchestration.stages import grid_executor

        os.environ["F1_GRID_EXECUTOR"] = "dask"
        with pytest.raises(ValueError, match="dask"):
            grid_executor()


@pytest.mark.spark
class TestAgainstTheConnectServer:
    """Needs `docker compose --profile pipeline up -d spark`.

    The session has to live under `data/`, because that is what the Spark
    container can see: `container_path` refuses anything outside the project,
    which is the guard working, not a limitation to route around.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def designed():
        from src.config import DATA_ROOT

        root = DATA_ROOT / "tmp" / "spark_equivalence_test"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        try:
            # write_session returns (grid_root, snapshot_root, aligned_root).
            _, _, aligned_root = syn.write_session(root)
            yield aligned_root
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_it_equals_the_pandas_executor_on_the_designed_session(self, designed) -> None:
        out = designed.parent
        pandas_result = resample_session(designed, out_root=out / "grid_pandas")
        spark_result = mod.resample_session_spark(designed, out_root=out / "grid_spark")
        assert spark_result.rows == pandas_result.rows
        for name in ("grid.parquet", "rejected_laps.parquet"):
            a = pd.read_parquet(out / "grid_pandas" / name)
            b = pd.read_parquet(out / "grid_spark" / name)
            sort = KEY if name == "grid.parquet" else ["driver", "lap_number"]
            pd.testing.assert_frame_equal(a.sort_values(sort).reset_index(drop=True),
                                          b.sort_values(sort).reset_index(drop=True), check_exact=True)

    def test_the_meta_records_which_executor_ran(self, designed) -> None:
        import json

        out = designed.parent / "grid_meta_check"
        mod.resample_session_spark(designed, out_root=out)
        meta = json.loads((out / "grid_meta.json").read_text(encoding="utf-8"))
        assert meta["executor"] == "spark"

    def test_a_real_session_matches_what_is_on_disk(self, tmp_path: Path) -> None:
        """F013's criterion 1, on whichever session is present."""
        from src.config import INTERIM_ROOT

        candidates = sorted((INTERIM_ROOT / "aligned").glob("2024_*_projection"))
        candidates = [c for c in candidates if (INTERIM_ROOT / "grid" / c.name / "grid.parquet").exists()]
        if not candidates:
            pytest.skip("no ingested session under data/interim/")
        aligned = candidates[0]
        out = Path(mod.PROJECT_TMP) / "spark_real_session"
        shutil.rmtree(out, ignore_errors=True)
        try:
            mod.resample_session_spark(aligned, out_root=out)
            a = pd.read_parquet(INTERIM_ROOT / "grid" / aligned.name / "grid.parquet")
            b = pd.read_parquet(out / "grid.parquet")
            pd.testing.assert_frame_equal(a.sort_values(KEY).reset_index(drop=True),
                                          b.sort_values(KEY).reset_index(drop=True), check_exact=True)
        finally:
            shutil.rmtree(out, ignore_errors=True)
