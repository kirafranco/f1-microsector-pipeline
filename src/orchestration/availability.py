"""Has this session's data been published yet?

Decision D4: there is no push event at the end of a session. Data appears on
the timing backend some tens of minutes to a few hours later, so orchestration
polls.

What makes polling more than a retry count is FastF1's error handling. Its
loaders are wrapped in a catch-all (`soft_exceptions`) that turns any failure
into a log warning unless debug mode is on. For an unpublished session the
underlying API raises

    SessionNotAvailableError("No data for this session! If this session only
    finished recently, please try again in a few minutes.")

which is precisely the signal wanted -- and it is swallowed. The session object
is then left without laps, and the first access raises DataNotLoadedError,
which carries no HTTP status and no timeout in its name, so `is_transient`
classes it as permanent. Left alone, the pipeline would treat "the race
finished an hour ago" as "this session will never work" and stop.

So the probe turns FastF1's debug mode on for the duration of the probe only.
Process-wide it would be wrong: during a real ingest it would promote a missing
weather feed, which the pipeline tolerates, into a hard failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from src.ingest.retry import is_transient

logger = logging.getLogger(__name__)

#: First wait after a session is expected but absent, and the ceiling. A
#: session publishes tens of minutes to a few hours late (D4), so five minutes
#: is a sensible first look and an hour is a sensible steady state.
BASE_INTERVAL_S = 300
MAX_INTERVAL_S = 3600

#: Give up after half a day. A session that has not published by then has
#: something wrong with it that waiting will not fix.
SENSOR_TIMEOUT_S = 12 * 60 * 60


class Availability(str, Enum):
    """What the probe learned."""

    READY = "ready"
    NOT_READY = "not_ready"
    TRANSIENT = "transient"
    PERMANENT = "permanent"

    @property
    def should_wait(self) -> bool:
        """Whether asking again later could plausibly succeed."""
        return self in (Availability.NOT_READY, Availability.TRANSIENT)


@dataclass(frozen=True)
class ProbeResult:
    verdict: Availability
    detail: str
    laps: int = 0

    @property
    def ready(self) -> bool:
        return self.verdict is Availability.READY

    def to_dict(self) -> dict:
        return {"verdict": self.verdict.value, "detail": self.detail, "laps": self.laps}


#: FastF1 exception names meaning "the data is not there yet". Matched by name
#: rather than by class so this module does not import fastf1 at module scope:
#: the DAG's sensor imports it in a worker that does have fastf1, but the unit
#: tests must run without provoking a network-capable import.
NOT_READY_NAMES = frozenset({
    "SessionNotAvailableError",
    "DataNotLoadedError",
    "NoLapDataError",
})

#: The session identifier itself is wrong -- a round that does not exist, or a
#: session a weekend never held. Waiting cannot fix it.
PERMANENT_NAMES = frozenset({
    "InvalidSessionError",
    "PermanentIngestError",
    "SnapshotCorruptError",
})

#: FastF1 rate-limits its backend, and a season backfill is the first time this
#: project asks for more than one session in an hour (F015). The exception
#: carries no HTTP status and no timeout in its name, so `is_transient` would
#: call it permanent and the pipeline would abandon a session that only needed
#: to wait -- which is exactly what the sensor's backoff is for.
RATE_LIMITED_NAMES = frozenset({"RateLimitExceededError"})


def classify(exc: BaseException) -> Availability:
    """Turn an exception raised while probing into a verdict."""
    names = {type(base).__name__ for base in (exc,)} | {
        cls.__name__ for cls in type(exc).__mro__
    }
    if names & RATE_LIMITED_NAMES:
        return Availability.TRANSIENT
    if names & PERMANENT_NAMES:
        return Availability.PERMANENT
    if names & NOT_READY_NAMES:
        return Availability.NOT_READY
    if is_transient(exc):
        return Availability.TRANSIENT
    return Availability.PERMANENT


def backoff_interval_s(attempt: int, base_s: int = BASE_INTERVAL_S,
                       cap_s: int = MAX_INTERVAL_S) -> int:
    """Seconds to wait before probe number `attempt` + 1.

    Doubling from `base_s`, capped. Attempt is 1-based, so the first wait is
    `base_s` exactly. No jitter: a single scheduler polling a single backend is
    not a thundering herd, and a predictable schedule is easier to read in the
    UI than a random one.
    """
    if attempt < 1:
        raise ValueError(f"attempt is 1-based, got {attempt}")
    return int(min(cap_s, base_s * (2 ** (attempt - 1))))


def _debug_mode(enabled: bool) -> Any:
    """FastF1's catch-all handler, on or off, restoring what it found.

    Imported here rather than at module scope so that this module is importable
    without fastf1 present.
    """
    from fastf1.logger import LoggingManager

    previous = LoggingManager.debug
    LoggingManager.debug = enabled
    return previous


def _restore_debug_mode(previous: Any) -> None:
    from fastf1.logger import LoggingManager

    LoggingManager.debug = previous


def probe(season: int, event: str, session: str,
          loader: Callable[[int, str, str], Any] | None = None) -> ProbeResult:
    """Ask whether the session's timing data has been published.

    Loads laps only -- no telemetry, no weather -- because the question is
    whether anything exists, and the answer arrives an order of magnitude
    faster without the telemetry payload. The real ingest loads everything.
    """
    loader = loader or _load_laps
    try:
        laps = loader(season, event, session)
    except Exception as exc:  # noqa: BLE001 - every failure is a verdict here
        verdict = classify(exc)
        detail = f"{type(exc).__name__}: {exc}"
        logger.info("availability_probe season=%s event=%s session=%s verdict=%s detail=%s",
                    season, event, session, verdict.value, detail)
        return ProbeResult(verdict=verdict, detail=detail)

    if not laps:
        logger.info("availability_probe season=%s event=%s session=%s verdict=not_ready detail=no laps",
                    season, event, session)
        return ProbeResult(verdict=Availability.NOT_READY, detail="session loaded but holds no laps")

    logger.info("availability_probe season=%s event=%s session=%s verdict=ready laps=%d",
                season, event, session, laps)
    return ProbeResult(verdict=Availability.READY, detail=f"{laps} laps published", laps=laps)


def _load_laps(season: int, event: str, session: str) -> int:
    """Lap count for a session, with FastF1's catch-all disabled.

    With the catch-all on, an unpublished session comes back as an object with
    no laps and a warning in the log; with it off, the API's own
    SessionNotAvailableError propagates and says so.
    """
    import fastf1

    from src.ingest.fastf1_source import enable_cache

    enable_cache()
    previous = _debug_mode(True)
    try:
        sess = fastf1.get_session(season, event, session)
        sess.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = sess.laps
        return 0 if laps is None else int(len(laps))
    finally:
        _restore_debug_mode(previous)
