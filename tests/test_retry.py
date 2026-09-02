"""Backoff behaviour: transient failures retry, permanent ones do not."""

from __future__ import annotations

import pytest

from src.ingest.retry import (
    PermanentIngestError,
    is_transient,
    retry_with_backoff,
)


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _HttpError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = _Response(status_code)


def _recorder():
    """A sleep stand-in that records delays instead of waiting."""
    delays: list[float] = []
    return delays, delays.append


class TestIsTransient:
    @pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
    def test_server_side_and_rate_limit_are_transient(self, status: int) -> None:
        assert is_transient(_HttpError(status)) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 422])
    def test_client_errors_are_permanent(self, status: int) -> None:
        assert is_transient(_HttpError(status)) is False

    def test_connection_and_timeout_are_transient(self) -> None:
        assert is_transient(ConnectionError("reset")) is True
        assert is_transient(TimeoutError("timed out")) is True

    def test_explicit_permanent_error_is_never_retried(self) -> None:
        assert is_transient(PermanentIngestError("no such driver")) is False


class TestRetryWithBackoff:
    def test_returns_immediately_on_success(self) -> None:
        delays, sleep = _recorder()
        calls = []

        def fn() -> str:
            calls.append(1)
            return "ok"

        assert retry_with_backoff(fn, description="t", sleep=sleep, rng=lambda: 0.0) == "ok"
        assert len(calls) == 1
        assert delays == []

    def test_retries_transient_then_succeeds(self) -> None:
        delays, sleep = _recorder()
        attempts = {"n": 0}

        def fn() -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ConnectionError("flaky")
            return "ok"

        result = retry_with_backoff(
            fn, description="t", base_delay_s=2.0, sleep=sleep, rng=lambda: 0.0
        )
        assert result == "ok"
        assert attempts["n"] == 3
        # Exponential on the base unit: 2s then 4s, jitter pinned to zero.
        assert delays == [2.0, 4.0]

    def test_delay_is_capped(self) -> None:
        delays, sleep = _recorder()

        def fn() -> None:
            raise ConnectionError("always")

        with pytest.raises(ConnectionError):
            retry_with_backoff(
                fn,
                description="t",
                attempts=6,
                base_delay_s=2.0,
                max_delay_s=5.0,
                sleep=sleep,
                rng=lambda: 0.0,
            )
        assert delays == [2.0, 4.0, 5.0, 5.0, 5.0]
        assert max(delays) <= 5.0

    def test_jitter_is_added_on_top_of_the_schedule(self) -> None:
        delays, sleep = _recorder()

        def fn() -> None:
            raise ConnectionError("always")

        with pytest.raises(ConnectionError):
            retry_with_backoff(
                fn, description="t", attempts=2, base_delay_s=2.0, sleep=sleep, rng=lambda: 0.5
            )
        assert delays == [2.0 + 1.0]

    def test_permanent_failure_does_not_retry(self) -> None:
        """A 404 is a real answer: five attempts would waste ~30s to learn nothing."""
        delays, sleep = _recorder()
        attempts = {"n": 0}

        def fn() -> None:
            attempts["n"] += 1
            raise _HttpError(404)

        with pytest.raises(_HttpError):
            retry_with_backoff(fn, description="t", sleep=sleep, rng=lambda: 0.0)

        assert attempts["n"] == 1
        assert delays == []

    def test_raises_last_error_after_exhausting_attempts(self) -> None:
        delays, sleep = _recorder()
        attempts = {"n": 0}

        def fn() -> None:
            attempts["n"] += 1
            raise ConnectionError(f"attempt {attempts['n']}")

        with pytest.raises(ConnectionError, match="attempt 4"):
            retry_with_backoff(
                fn, description="t", attempts=4, sleep=sleep, rng=lambda: 0.0
            )
        assert attempts["n"] == 4
        assert len(delays) == 3  # no sleep after the final attempt

    def test_rejects_nonsense_attempt_count(self) -> None:
        with pytest.raises(ValueError):
            retry_with_backoff(lambda: None, description="t", attempts=0)
