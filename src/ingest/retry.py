"""Exponential backoff for external fetches (global CLAUDE.md 3.1).

Retries only on transient failures. A 404 for a session that does not exist is
a permanent answer and must fail immediately rather than burning five attempts
over roughly half a minute.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_ATTEMPTS = 5
DEFAULT_BASE_DELAY_S = 2.0
DEFAULT_MAX_DELAY_S = 60.0

#: HTTP statuses worth retrying: server-side faults and explicit rate limiting.
TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class PermanentIngestError(Exception):
    """Failure that will not be fixed by trying again."""


def _status_code(exc: BaseException) -> int | None:
    """Best-effort HTTP status extraction, without hard-depending on requests."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    return getattr(response, "status_code", None)


def is_transient(exc: BaseException) -> bool:
    """True when retrying `exc` could plausibly succeed."""
    if isinstance(exc, PermanentIngestError):
        return False

    status = _status_code(exc)
    if status is not None:
        return status in TRANSIENT_STATUS

    # No status attached: fall back on the exception family. Connection and
    # timeout errors are transient by nature.
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    name = type(exc).__name__.lower()
    return any(token in name for token in ("timeout", "connection", "chunked"))


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    description: str,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay_s: float = DEFAULT_BASE_DELAY_S,
    max_delay_s: float = DEFAULT_MAX_DELAY_S,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
) -> T:
    """Call `fn`, retrying transient failures with jittered exponential backoff.

    `sleep` and `rng` are injectable so that tests can assert the delay
    schedule without actually waiting.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    last_exc: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classified immediately below
            last_exc = exc

            if not is_transient(exc):
                logger.error(
                    "fetch_failed_permanent description=%r attempt=%d error=%s",
                    description,
                    attempt,
                    exc,
                )
                raise

            if attempt == attempts:
                logger.error(
                    "fetch_failed_exhausted description=%r attempts=%d error=%s",
                    description,
                    attempts,
                    exc,
                )
                break

            delay = min(base_delay_s * (2 ** (attempt - 1)), max_delay_s)
            delay += rng() * base_delay_s  # full-width jitter on the base unit
            logger.warning(
                "fetch_retry description=%r attempt=%d/%d delay_s=%.2f error=%s",
                description,
                attempt,
                attempts,
                delay,
                exc,
            )
            sleep(delay)

    assert last_exc is not None  # unreachable: loop always sets it before break
    raise last_exc
