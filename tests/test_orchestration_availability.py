"""F006 availability probe: telling "not published yet" from "never going to work".

The classification here is the reason the sensor exists. FastF1 reports an
unpublished session in a way that this project's retry helper reads as
permanent, so without this translation the pipeline would give up on a race
that finished an hour ago.
"""

from __future__ import annotations

import pytest

from src.ingest.fastf1_source import PermanentIngestError
from src.orchestration.availability import (
    BASE_INTERVAL_S,
    MAX_INTERVAL_S,
    SENSOR_TIMEOUT_S,
    Availability,
    backoff_interval_s,
    classify,
    probe,
)


# FastF1's real exception classes, imported by name so the test breaks if the
# library renames one -- which is exactly when the sensor would start
# misclassifying and nobody would notice.
def fastf1_exceptions() -> dict[str, type[BaseException]]:
    import fastf1.exceptions as exceptions
    from fastf1._api import SessionNotAvailableError

    return {
        "SessionNotAvailableError": SessionNotAvailableError,
        "DataNotLoadedError": exceptions.DataNotLoadedError,
        "NoLapDataError": exceptions.NoLapDataError,
        "InvalidSessionError": exceptions.InvalidSessionError,
        "RateLimitExceededError": exceptions.RateLimitExceededError,
    }


class TestClassify:
    @pytest.mark.parametrize("name", ["SessionNotAvailableError", "DataNotLoadedError", "NoLapDataError"])
    def test_fastf1_says_not_yet(self, name: str) -> None:
        """All three mean the data has not appeared, and all three would be
        read as permanent by the retry helper on their own."""
        exception = fastf1_exceptions()[name]("No data for this session!")
        assert classify(exception) is Availability.NOT_READY

    def test_an_invalid_session_will_never_work(self, name: str = "InvalidSessionError") -> None:
        assert classify(fastf1_exceptions()[name]("no such session")) is Availability.PERMANENT

    def test_the_projects_own_permanent_error_is_permanent(self) -> None:
        assert classify(PermanentIngestError("no laps")) is Availability.PERMANENT

    def test_a_dropped_connection_is_worth_retrying(self) -> None:
        assert classify(ConnectionError("connection reset")) is Availability.TRANSIENT

    def test_a_timeout_is_worth_retrying(self) -> None:
        assert classify(TimeoutError("read timed out")) is Availability.TRANSIENT

    def test_being_rate_limited_is_worth_waiting_out(self) -> None:
        """A season backfill is the first time this project asks the backend for
        more than one session in an hour (F015). The exception carries no HTTP
        status and no timeout in its name, so without this it would read as
        permanent and the pipeline would abandon a session that only needed to
        wait -- which is what the sensor's backoff exists for."""
        limited = fastf1_exceptions()["RateLimitExceededError"]("rate limit exceeded")
        assert classify(limited) is Availability.TRANSIENT
        assert Availability.TRANSIENT.should_wait is True

    def test_the_rate_limit_beats_the_critical_family_it_belongs_to(self) -> None:
        """It is a FastF1CriticalError, which is exactly why it needs naming:
        the family says "do not swallow this", not "never try again"."""
        import fastf1.exceptions as exceptions
        limited = fastf1_exceptions()["RateLimitExceededError"]("slow down")
        assert isinstance(limited, exceptions.FastF1CriticalError)
        assert classify(limited) is not Availability.PERMANENT

    def test_an_unrecognised_failure_is_not_waited_on(self) -> None:
        """A ValueError in the pipeline is a bug, not a publishing delay."""
        assert classify(ValueError("something else entirely")) is Availability.PERMANENT

    def test_a_subclass_is_classified_by_its_ancestry(self) -> None:
        class Derived(fastf1_exceptions()["NoLapDataError"]):  # type: ignore[misc]
            pass

        assert classify(Derived()) is Availability.NOT_READY


class TestShouldWait:
    def test_only_the_two_that_can_change_are_waited_on(self) -> None:
        assert Availability.NOT_READY.should_wait is True
        assert Availability.TRANSIENT.should_wait is True
        assert Availability.PERMANENT.should_wait is False
        assert Availability.READY.should_wait is False


class TestBackoff:
    def test_the_schedule_is_the_one_the_spec_states(self) -> None:
        assert [backoff_interval_s(n) for n in range(1, 9)] == [
            300, 600, 1200, 2400, 3600, 3600, 3600, 3600]

    def test_it_starts_at_the_base_and_never_exceeds_the_cap(self) -> None:
        assert backoff_interval_s(1) == BASE_INTERVAL_S
        assert backoff_interval_s(50) == MAX_INTERVAL_S

    def test_attempts_are_one_based(self) -> None:
        with pytest.raises(ValueError, match="1-based"):
            backoff_interval_s(0)

    def test_it_gives_up_after_half_a_day(self) -> None:
        assert SENSOR_TIMEOUT_S == 12 * 60 * 60


class TestProbe:
    def test_laps_published_is_ready(self) -> None:
        result = probe(2024, "Japan", "Q", loader=lambda *_: 74)
        assert result.ready and result.laps == 74
        assert result.verdict is Availability.READY

    def test_a_session_that_loads_but_holds_no_laps_is_not_ready(self) -> None:
        """It happens between the session ending and the timing feed filling."""
        result = probe(2024, "Japan", "Q", loader=lambda *_: 0)
        assert result.verdict is Availability.NOT_READY
        assert "no laps" in result.detail

    def test_the_verdict_carries_the_reason(self) -> None:
        def unavailable(*_):
            raise fastf1_exceptions()["SessionNotAvailableError"](
                "No data for this session! If this session only finished recently...")

        result = probe(2024, "Japan", "R", loader=unavailable)
        assert result.verdict is Availability.NOT_READY
        assert "SessionNotAvailableError" in result.detail
        assert result.to_dict()["verdict"] == "not_ready"

    def test_a_permanent_failure_is_reported_as_such(self) -> None:
        def invalid(*_):
            raise fastf1_exceptions()["InvalidSessionError"]("no such session")

        assert probe(2024, "Nowhere", "Q", loader=invalid).verdict is Availability.PERMANENT


class TestDebugModeIsRestored:
    """The probe turns FastF1's catch-all off so the real error surfaces. Left
    off, a missing weather feed during ingest -- which the pipeline tolerates --
    would become a hard failure."""

    def test_the_flag_goes_back_to_what_it_was(self) -> None:
        from fastf1.logger import LoggingManager

        from src.orchestration.availability import _debug_mode, _restore_debug_mode

        original = LoggingManager.debug
        try:
            previous = _debug_mode(True)
            assert LoggingManager.debug is True
            _restore_debug_mode(previous)
            assert LoggingManager.debug is original
        finally:
            LoggingManager.debug = original

    def test_it_is_restored_even_when_the_load_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The real path, with only the network call and the cache replaced."""
        import fastf1
        from fastf1.logger import LoggingManager

        import src.ingest.fastf1_source as source

        original = LoggingManager.debug
        seen: list[bool] = []

        def explode(*_args, **_kwargs):
            seen.append(LoggingManager.debug)
            raise fastf1_exceptions()["SessionNotAvailableError"]("No data for this session!")

        monkeypatch.setattr(source, "enable_cache", lambda *_a, **_k: None)
        monkeypatch.setattr(fastf1, "get_session", explode)

        result = probe(2024, "Japan", "Q")

        assert seen == [True], "the catch-all must be off while probing"
        assert result.verdict is Availability.NOT_READY
        assert LoggingManager.debug is original
